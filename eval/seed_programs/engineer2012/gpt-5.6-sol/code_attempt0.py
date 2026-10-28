import argparse
import json
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def node_string(port, period):
    return f"({port}, {period})"


def virtual_node_string(kind, vessel_id):
    return f"({kind}, {vessel_id})"


def clean_number(value, tolerance=1e-8):
    value = float(value)
    if abs(value) <= tolerance:
        return 0.0
    nearest = round(value)
    if abs(value - nearest) <= tolerance:
        return float(nearest)
    return value


def get_travel_time(data, origin, destination):
    key = f"{origin}->{destination}"
    if key in data["travel_times"]:
        return int(data["travel_times"][key])
    return None


def build_fallback_solution(data):
    """Return a structurally valid no-transfer solution if no MIP incumbent exists."""
    horizon = int(data["planning_horizon"])
    supply_ports = data["supply_ports"]
    demand_ports = data["demand_ports"]
    all_port_data = supply_ports + demand_ports
    all_ports = [p["port_id"] for p in all_port_data]

    inventories = {}
    for port in supply_ports:
        inv = float(port["initial_inventory"])
        entries = {}
        for t in range(1, horizon + 1):
            inv += float(port["production_rate"][t - 1])
            entries[str(t)] = clean_number(inv)
        inventories[port["port_id"]] = entries

    for port in demand_ports:
        inv = float(port["initial_inventory"])
        entries = {}
        for t in range(1, horizon + 1):
            inv -= float(port["consumption_rate"][t - 1])
            entries[str(t)] = clean_number(inv)
        inventories[port["port_id"]] = entries

    vessel_routes = {}
    vessel_operations = {}
    demurrage_cost = 0.0

    for vessel in data["vessels"]:
        vessel_id = vessel["id"]
        start = max(0, min(horizon, int(vessel["availability_start"])))
        end = max(start, min(horizon, int(vessel["availability_end"])))
        initial_load = float(vessel["initial_load"])

        if all_ports:
            def draft_value(port):
                return float(
                    data.get("draft_limits", {}).get(
                        f"{vessel_id}->{port}", vessel["capacity"]
                    )
                )

            port = max(all_ports, key=draft_value)
        else:
            port = "UNSPECIFIED"

        route = [{
            "tail": virtual_node_string("src", vessel_id),
            "head": node_string(port, start),
            "type": "source",
            "cost": 0.0,
            "product_flow": clean_number(initial_load),
        }]

        for t in range(start, end):
            cost = float(vessel["demurrage_cost_per_day"])
            demurrage_cost += cost
            route.append({
                "tail": node_string(port, t),
                "head": node_string(port, t + 1),
                "type": "demurrage",
                "cost": cost,
                "product_flow": clean_number(initial_load),
            })

        route.append({
            "tail": node_string(port, end),
            "head": virtual_node_string("snk", vessel_id),
            "type": "sink",
            "cost": 0.0,
            "product_flow": clean_number(initial_load),
        })

        vessel_routes[vessel_id] = route
        vessel_operations[vessel_id] = []

    breakdown = {
        "transport_cost": 0.0,
        "demurrage_cost": clean_number(demurrage_cost),
        "procurement_cost": 0.0,
        "supply_revenue": 0.0,
        "total_cost": clean_number(demurrage_cost),
    }

    return {
        "objective_value": clean_number(demurrage_cost),
        "vessel_routes": vessel_routes,
        "port_inventories": inventories,
        "vessel_operations": vessel_operations,
        "cost_breakdown": breakdown,
    }


def solve(data, time_limit, logger):
    start_clock = time.monotonic()

    horizon = int(data["planning_horizon"])
    vessels = data["vessels"]
    supply_data = data["supply_ports"]
    demand_data = data["demand_ports"]

    supply_ports = [p["port_id"] for p in supply_data]
    demand_ports = [p["port_id"] for p in demand_data]
    all_ports = supply_ports + demand_ports

    supply_by_id = {p["port_id"]: p for p in supply_data}
    demand_by_id = {p["port_id"]: p for p in demand_data}
    vessel_by_id = {v["id"]: v for v in vessels}
    vessel_ids = [v["id"] for v in vessels]

    model = gp.Model("bulk_shipping")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    draft_limits = data.get("draft_limits", {})

    def cargo_limit(vessel_id, port):
        vessel = vessel_by_id[vessel_id]
        return min(
            float(vessel["capacity"]),
            float(draft_limits.get(
                f"{vessel_id}->{port}", vessel["capacity"]
            )),
        )

    # Each arc dictionary contains its network and output attributes.
    arcs = {}
    incoming = {}
    outgoing = {}

    for vessel in vessels:
        vessel_id = vessel["id"]
        start = max(0, min(horizon, int(vessel["availability_start"])))
        end = max(start, min(horizon, int(vessel["availability_end"])))

        vessel_arcs = []
        arc_in = defaultdict(list)
        arc_out = defaultdict(list)

        def add_arc(tail, head, arc_type, cost, cargo_ub):
            arc_id = len(vessel_arcs)
            item = {
                "id": arc_id,
                "tail": tail,
                "head": head,
                "type": arc_type,
                "cost": float(cost),
                "cargo_ub": max(0.0, float(cargo_ub)),
            }
            vessel_arcs.append(item)
            if isinstance(tail, tuple) and tail[0] == "port":
                arc_out[(tail[1], tail[2])].append(arc_id)
            if isinstance(head, tuple) and head[0] == "port":
                arc_in[(head[1], head[2])].append(arc_id)
            return arc_id

        # Source arcs.
        for port in all_ports:
            add_arc(
                ("src", vessel_id),
                ("port", port, start),
                "source",
                0.0,
                cargo_limit(vessel_id, port),
            )

        # Time-expanded waiting and sailing arcs.
        for t in range(start, end + 1):
            if t < end:
                for port in all_ports:
                    add_arc(
                        ("port", port, t),
                        ("port", port, t + 1),
                        "demurrage",
                        float(vessel["demurrage_cost_per_day"]),
                        cargo_limit(vessel_id, port),
                    )

            for origin in all_ports:
                for destination in all_ports:
                    if origin == destination:
                        continue
                    travel_time = get_travel_time(data, origin, destination)
                    if travel_time is None:
                        continue
                    travel_time = max(1, int(travel_time))
                    arrival = t + travel_time
                    if arrival > end:
                        continue
                    limit = min(
                        cargo_limit(vessel_id, origin),
                        cargo_limit(vessel_id, destination),
                    )
                    add_arc(
                        ("port", origin, t),
                        ("port", destination, arrival),
                        "travel",
                        travel_time * float(vessel["transport_cost_per_day"]),
                        limit,
                    )

        # Sink arcs.
        for port in all_ports:
            add_arc(
                ("port", port, end),
                ("snk", vessel_id),
                "sink",
                0.0,
                cargo_limit(vessel_id, port),
            )

        arcs[vessel_id] = vessel_arcs
        incoming[vessel_id] = arc_in
        outgoing[vessel_id] = arc_out

    # Route and onboard-flow variables.
    x = {}
    z = {}
    for vessel_id in vessel_ids:
        for arc in arcs[vessel_id]:
            key = (vessel_id, arc["id"])
            x[key] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x[{vessel_id},{arc['id']}]",
            )
            z[key] = model.addVar(
                lb=0.0,
                ub=arc["cargo_ub"],
                vtype=GRB.CONTINUOUS,
                name=f"z[{vessel_id},{arc['id']}]",
            )
            model.addConstr(
                z[key] <= arc["cargo_ub"] * x[key],
                name=f"arc_cargo_link[{vessel_id},{arc['id']}]",
            )

    # Transfer variables. Quantities are nonnegative internally.
    operation = {}
    quantity = {}

    for vessel in vessels:
        vessel_id = vessel["id"]
        start = max(0, min(horizon, int(vessel["availability_start"])))
        end = max(start, min(horizon, int(vessel["availability_end"])))

        for t in range(max(1, start), min(horizon, end) + 1):
            for port in all_ports:
                port_data = (
                    supply_by_id[port] if port in supply_by_id
                    else demand_by_id[port]
                )
                key = (vessel_id, port, t)
                operation[key] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"op[{vessel_id},{port},{t}]",
                )
                quantity[key] = model.addVar(
                    lb=0.0,
                    ub=float(port_data["F_max"]),
                    vtype=GRB.CONTINUOUS,
                    name=f"qty[{vessel_id},{port},{t}]",
                )

                model.addConstr(
                    quantity[key] >= float(port_data["F_min"]) * operation[key],
                    name=f"transfer_min[{vessel_id},{port},{t}]",
                )
                model.addConstr(
                    quantity[key] <= float(port_data["F_max"]) * operation[key],
                    name=f"transfer_max[{vessel_id},{port},{t}]",
                )

                visit_expr = gp.quicksum(
                    x[(vessel_id, a)]
                    for a in incoming[vessel_id].get((port, t), [])
                )
                model.addConstr(
                    operation[key] <= visit_expr,
                    name=f"operation_visit[{vessel_id},{port},{t}]",
                )

    # Route and cargo-flow conservation.
    for vessel in vessels:
        vessel_id = vessel["id"]
        start = max(0, min(horizon, int(vessel["availability_start"])))
        end = max(start, min(horizon, int(vessel["availability_end"])))
        initial_load = float(vessel["initial_load"])

        source_arc_ids = [
            arc["id"] for arc in arcs[vessel_id] if arc["type"] == "source"
        ]
        sink_arc_ids = [
            arc["id"] for arc in arcs[vessel_id] if arc["type"] == "sink"
        ]

        model.addConstr(
            gp.quicksum(x[(vessel_id, a)] for a in source_arc_ids) == 1,
            name=f"one_source[{vessel_id}]",
        )
        model.addConstr(
            gp.quicksum(x[(vessel_id, a)] for a in sink_arc_ids) == 1,
            name=f"one_sink[{vessel_id}]",
        )

        for a in source_arc_ids:
            model.addConstr(
                z[(vessel_id, a)] == initial_load * x[(vessel_id, a)],
                name=f"initial_cargo[{vessel_id},{a}]",
            )

        for t in range(start, end + 1):
            for port in all_ports:
                in_ids = incoming[vessel_id].get((port, t), [])
                out_ids = outgoing[vessel_id].get((port, t), [])

                model.addConstr(
                    gp.quicksum(x[(vessel_id, a)] for a in in_ids)
                    == gp.quicksum(x[(vessel_id, a)] for a in out_ids),
                    name=f"route_flow[{vessel_id},{port},{t}]",
                )

                transfer = 0.0
                key = (vessel_id, port, t)
                if key in quantity:
                    if port in supply_by_id:
                        transfer = quantity[key]
                    else:
                        transfer = -quantity[key]

                model.addConstr(
                    gp.quicksum(z[(vessel_id, a)] for a in out_ids)
                    - gp.quicksum(z[(vessel_id, a)] for a in in_ids)
                    == transfer,
                    name=f"cargo_flow[{vessel_id},{port},{t}]",
                )

        # If the vessel loads at all, its first loading cannot be after the
        # stated loading-window end.
        loading_cutoff = int(vessel.get("loading_window_end", horizon))
        early_loads = [
            operation[(vessel_id, port, t)]
            for port in supply_ports
            for t in range(1, horizon + 1)
            if (vessel_id, port, t) in operation and t <= loading_cutoff
        ]
        early_expr = gp.quicksum(early_loads)
        for port in supply_ports:
            for t in range(1, horizon + 1):
                key = (vessel_id, port, t)
                if key in operation and t > loading_cutoff:
                    model.addConstr(
                        operation[key] <= early_expr,
                        name=f"loading_window[{vessel_id},{port},{t}]",
                    )

    # Port inventories.
    inventory = {}

    for port in supply_data:
        port_id = port["port_id"]
        previous = float(port["initial_inventory"])
        for t in range(1, horizon + 1):
            inventory[(port_id, t)] = model.addVar(
                lb=0.0,
                ub=float(port["storage_capacity"][t - 1]),
                vtype=GRB.CONTINUOUS,
                name=f"inventory[{port_id},{t}]",
            )
            loaded = gp.quicksum(
                quantity[(vessel_id, port_id, t)]
                for vessel_id in vessel_ids
                if (vessel_id, port_id, t) in quantity
            )
            model.addConstr(
                inventory[(port_id, t)]
                == previous + float(port["production_rate"][t - 1]) - loaded,
                name=f"supply_inventory_balance[{port_id},{t}]",
            )
            previous = inventory[(port_id, t)]

    for port in demand_data:
        port_id = port["port_id"]
        previous = float(port["initial_inventory"])
        for t in range(1, horizon + 1):
            inventory[(port_id, t)] = model.addVar(
                lb=0.0,
                ub=float(port["storage_capacity"][t - 1]),
                vtype=GRB.CONTINUOUS,
                name=f"inventory[{port_id},{t}]",
            )
            delivered = gp.quicksum(
                quantity[(vessel_id, port_id, t)]
                for vessel_id in vessel_ids
                if (vessel_id, port_id, t) in quantity
            )
            model.addConstr(
                inventory[(port_id, t)]
                == previous - float(port["consumption_rate"][t - 1])
                + delivered,
                name=f"demand_inventory_balance[{port_id},{t}]",
            )
            previous = inventory[(port_id, t)]

    route_cost_expr = gp.quicksum(
        arc["cost"] * x[(vessel_id, arc["id"])]
        for vessel_id in vessel_ids
        for arc in arcs[vessel_id]
    )
    procurement_expr = gp.quicksum(
        float(supply_by_id[port]["unit_procurement_cost"])
        * quantity[(vessel_id, port, t)]
        for vessel_id in vessel_ids
        for port in supply_ports
        for t in range(1, horizon + 1)
        if (vessel_id, port, t) in quantity
    )
    revenue_expr = gp.quicksum(
        float(demand_by_id[port]["unit_supply_revenue"])
        * quantity[(vessel_id, port, t)]
        for vessel_id in vessel_ids
        for port in demand_ports
        for t in range(1, horizon + 1)
        if (vessel_id, port, t) in quantity
    )

    model.setObjective(
        route_cost_expr + procurement_expr - revenue_expr,
        GRB.MINIMIZE,
    )

    model.update()

    def make_solution(value_of):
        vessel_routes = {}
        vessel_operations = {}

        transport_cost = 0.0
        demurrage_cost = 0.0
        procurement_cost = 0.0
        supply_revenue = 0.0

        for vessel_id in vessel_ids:
            selected = []
            for arc in arcs[vessel_id]:
                key = (vessel_id, arc["id"])
                if value_of(x[key]) <= 0.5:
                    continue

                tail = arc["tail"]
                head = arc["head"]

                if tail[0] == "src":
                    tail_text = virtual_node_string("src", vessel_id)
                else:
                    tail_text = node_string(tail[1], tail[2])

                if head[0] == "snk":
                    head_text = virtual_node_string("snk", vessel_id)
                else:
                    head_text = node_string(head[1], head[2])

                arc_cost = float(arc["cost"])
                if arc["type"] == "travel":
                    transport_cost += arc_cost
                elif arc["type"] == "demurrage":
                    demurrage_cost += arc_cost

                selected.append({
                    "tail": tail_text,
                    "head": head_text,
                    "type": arc["type"],
                    "cost": clean_number(arc_cost),
                    "product_flow": clean_number(
                        max(0.0, value_of(z[key]))
                    ),
                    "_sort": (
                        -1 if tail[0] == "src" else tail[2],
                        0 if arc["type"] == "source" else
                        2 if arc["type"] == "sink" else 1,
                    ),
                })

            selected.sort(key=lambda item: item["_sort"])
            for item in selected:
                del item["_sort"]
            vessel_routes[vessel_id] = selected

            operations = []
            for t in range(1, horizon + 1):
                for port in all_ports:
                    key = (vessel_id, port, t)
                    if key not in quantity:
                        continue
                    amount = max(0.0, value_of(quantity[key]))
                    if amount <= 1e-7:
                        continue

                    if port in supply_by_id:
                        procurement_cost += (
                            amount
                            * float(supply_by_id[port]["unit_procurement_cost"])
                        )
                        operations.append({
                            "port": port,
                            "time": t,
                            "quantity": clean_number(amount),
                            "type": "load",
                        })
                    else:
                        supply_revenue += (
                            amount
                            * float(demand_by_id[port]["unit_supply_revenue"])
                        )
                        operations.append({
                            "port": port,
                            "time": t,
                            "quantity": clean_number(-amount),
                            "type": "discharge",
                        })
            vessel_operations[vessel_id] = operations

        port_inventories = {}
        for port in all_ports:
            port_inventories[port] = {
                str(t): clean_number(
                    max(0.0, value_of(inventory[(port, t)]))
                )
                for t in range(1, horizon + 1)
            }

        total_cost = (
            transport_cost
            + demurrage_cost
            + procurement_cost
            - supply_revenue
        )

        cost_breakdown = {
            "transport_cost": clean_number(transport_cost),
            "demurrage_cost": clean_number(demurrage_cost),
            "procurement_cost": clean_number(procurement_cost),
            "supply_revenue": clean_number(supply_revenue),
            "total_cost": clean_number(total_cost),
        }

        return {
            "objective_value": clean_number(total_cost),
            "vessel_routes": vessel_routes,
            "port_inventories": port_inventories,
            "vessel_operations": vessel_operations,
            "cost_breakdown": cost_breakdown,
        }

    all_variables = model.getVars()
    logged_best = [float("inf")]

    def callback(cb_model, where):
        if logger is None or where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj >= logged_best[0] - 1e-7:
                return

            values = cb_model.cbGetSolution(all_variables)
            value_map = {
                variable: values[i]
                for i, variable in enumerate(all_variables)
            }
            solution = make_solution(lambda variable: value_map[variable])
            logger.log_solution(solution["objective_value"], solution)
            logged_best[0] = incumbent_obj
        except Exception:
            # Logging must not interrupt optimization.
            pass

    elapsed = time.monotonic() - start_clock
    model.Params.TimeLimit = max(0.01, float(time_limit) - elapsed)

    if logger is not None:
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        solution = make_solution(lambda variable: variable.X)
        if logger is not None and model.ObjVal < logged_best[0] - 1e-7:
            try:
                logger.log_solution(solution["objective_value"], solution)
            except Exception:
                pass
        return solution

    return build_fallback_solution(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    solution = solve(data, max(1, args.time_limit), logger)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()