import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-7


def clean_number(value):
    value = float(value)
    if abs(value) < 1e-8:
        return 0.0
    return round(value, 10)


def nearest_neighbor_route(customers, distance):
    remaining = set(customers)
    route = [0]
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda j: (distance[current][j], j))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    route.append(0)
    return route


def pack_deliveries(deliveries, capacity, num_vehicles, distance):
    items = [(i, q) for i, q in deliveries.items() if q > TOL]
    items.sort(key=lambda z: (-z[1], z[0]))

    bins = []
    loads = []
    for customer, quantity in items:
        best_bin = None
        best_remaining = None
        for k in range(len(bins)):
            if loads[k] + quantity <= capacity + 1e-7:
                remaining = capacity - loads[k] - quantity
                if best_bin is None or remaining < best_remaining:
                    best_bin = k
                    best_remaining = remaining

        if best_bin is None:
            if len(bins) >= num_vehicles:
                return None
            bins.append([customer])
            loads.append(quantity)
        else:
            bins[best_bin].append(customer)
            loads[best_bin] += quantity

    routes = [nearest_neighbor_route(group, distance) for group in bins]
    return routes


def calculate_solution(instance, period_routes, period_deliveries):
    n = instance["num_customers"]
    periods = instance["num_periods"]
    distance = instance["distance_matrix"]
    depot = instance["depot"]
    customer_data = {c["id"]: c for c in instance["customers"]}

    depot_inventory = float(depot["initial_inventory"])
    customer_inventory = {
        i: float(customer_data[i]["initial_inventory"]) for i in range(1, n + 1)
    }

    details = {"periods": {}}
    total_travel = 0.0
    total_holding = 0.0

    for t in range(1, periods + 1):
        routes = period_routes.get(t, [])
        deliveries = period_deliveries.get(t, {})

        shipped = sum(float(q) for q in deliveries.values())
        depot_inventory += float(depot["production_per_period"]) - shipped

        inventories = {"depot": clean_number(depot_inventory)}
        total_holding += depot_inventory * float(depot["holding_cost"])

        for i in range(1, n + 1):
            customer_inventory[i] += (
                float(deliveries.get(i, 0.0))
                - float(customer_data[i]["demand_per_period"])
            )
            inventories[str(i)] = clean_number(customer_inventory[i])
            total_holding += (
                customer_inventory[i] * float(customer_data[i]["holding_cost"])
            )

        normalized_routes = []
        for route in routes:
            normalized_route = [int(v) for v in route]
            normalized_routes.append(normalized_route)
            for a, b in zip(normalized_route[:-1], normalized_route[1:]):
                total_travel += float(distance[a][b])

        output_deliveries = {
            str(i): clean_number(q)
            for i, q in sorted(deliveries.items())
            if q > TOL
        }

        details["periods"][str(t)] = {
            "routes": normalized_routes,
            "deliveries": output_deliveries,
            "inventories": inventories,
        }

    return {
        "objective_value": clean_number(total_travel + total_holding),
        "solution_details": details,
    }


def construct_greedy_solution(instance):
    """Construct a simple feasible inventory plan and pack it into routes."""
    n = instance["num_customers"]
    periods = instance["num_periods"]
    vehicles = instance["num_vehicles"]
    capacity = float(instance["vehicle_capacity"])
    distance = instance["distance_matrix"]
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}

    depot_inventory = float(depot["initial_inventory"])
    inventories = {
        i: float(customers[i]["initial_inventory"]) for i in range(1, n + 1)
    }

    period_deliveries = {}
    period_routes = {}

    for t in range(1, periods + 1):
        available = depot_inventory + float(depot["production_per_period"])
        deliveries = {}

        # First cover current-period demand and minimum ending inventory.
        for i in range(1, n + 1):
            c = customers[i]
            demand = float(c["demand_per_period"])
            minimum = float(c["min_inventory"])
            maximum = float(c["max_inventory"])

            required = max(0.0, demand + minimum - inventories[i])
            headroom = maximum - inventories[i]

            if required > headroom + 1e-7 or required > capacity + 1e-7:
                return None
            deliveries[i] = required

        required_total = sum(deliveries.values())

        # Ship enough to avoid overflowing the depot at the end of the period.
        minimum_shipment = max(
            0.0, available - float(depot["max_inventory"])
        )
        target_shipment = max(required_total, minimum_shipment)

        if target_shipment > available + 1e-7:
            return None
        if target_shipment > vehicles * capacity + 1e-7:
            return None

        extra = target_shipment - required_total

        # Prefer customers already being visited, then low holding-cost customers.
        order = sorted(
            range(1, n + 1),
            key=lambda i: (
                0 if deliveries[i] > TOL else 1,
                float(customers[i]["holding_cost"]),
                i,
            ),
        )

        for i in order:
            if extra <= TOL:
                break
            maximum = float(customers[i]["max_inventory"])
            storage_headroom = maximum - inventories[i] - deliveries[i]
            vehicle_headroom = capacity - deliveries[i]
            add = min(extra, storage_headroom, vehicle_headroom)
            if add > TOL:
                deliveries[i] += add
                extra -= add

        if extra > 1e-6:
            return None

        routes = pack_deliveries(deliveries, capacity, vehicles, distance)
        if routes is None:
            return None

        shipped = sum(deliveries.values())
        depot_inventory = available - shipped
        if depot_inventory < -1e-6 or depot_inventory > depot["max_inventory"] + 1e-6:
            return None

        for i in range(1, n + 1):
            c = customers[i]
            if inventories[i] + deliveries[i] > c["max_inventory"] + 1e-6:
                return None
            inventories[i] += deliveries[i] - float(c["demand_per_period"])
            if inventories[i] < c["min_inventory"] - 1e-6:
                return None
            if inventories[i] > c["max_inventory"] + 1e-6:
                return None

        period_deliveries[t] = {
            i: q for i, q in deliveries.items() if q > TOL
        }
        period_routes[t] = routes

    return calculate_solution(instance, period_routes, period_deliveries)


def build_model(instance):
    n = instance["num_customers"]
    periods = instance["num_periods"]
    vehicles = instance["num_vehicles"]
    capacity = float(instance["vehicle_capacity"])
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}
    distance = instance["distance_matrix"]

    nodes = range(0, n + 1)
    customer_nodes = range(1, n + 1)
    time_periods = range(1, periods + 1)
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    model = gp.Model("inventory_routing")

    q = model.addVars(customer_nodes, time_periods, lb=0.0, name="delivery")
    inv_depot = model.addVars(
        time_periods,
        lb=0.0,
        ub=float(depot["max_inventory"]),
        name="depot_inventory",
    )

    inv_customer = {}
    for i in customer_nodes:
        c = customers[i]
        for t in time_periods:
            inv_customer[i, t] = model.addVar(
                lb=float(c["min_inventory"]),
                ub=float(c["max_inventory"]),
                name=f"customer_inventory[{i},{t}]",
            )

    y = model.addVars(customer_nodes, time_periods, vtype=GRB.BINARY, name="visit")
    x = model.addVars(arcs, time_periods, vtype=GRB.BINARY, name="arc")
    flow = model.addVars(arcs, time_periods, lb=0.0, name="load_flow")
    order = model.addVars(
        customer_nodes, time_periods, lb=0.0, ub=float(n), name="order"
    )

    model.update()

    # Depot inventory balance.
    for t in time_periods:
        previous = (
            float(depot["initial_inventory"])
            if t == 1
            else inv_depot[t - 1]
        )
        model.addConstr(
            inv_depot[t]
            == previous
            + float(depot["production_per_period"])
            - gp.quicksum(q[i, t] for i in customer_nodes),
            name=f"depot_balance[{t}]",
        )

    # Customer inventory balances and delivery/storage limits.
    for i in customer_nodes:
        c = customers[i]
        demand = float(c["demand_per_period"])
        maximum = float(c["max_inventory"])

        for t in time_periods:
            previous = (
                float(c["initial_inventory"])
                if t == 1
                else inv_customer[i, t - 1]
            )

            model.addConstr(
                inv_customer[i, t] == previous + q[i, t] - demand,
                name=f"customer_balance[{i},{t}]",
            )

            # Inventory immediately after the beginning-of-period delivery
            # cannot exceed the customer's storage capacity.
            model.addConstr(
                previous + q[i, t] <= maximum,
                name=f"pre_demand_capacity[{i},{t}]",
            )

            model.addConstr(
                q[i, t] <= capacity * y[i, t],
                name=f"delivery_visit_link[{i},{t}]",
            )

    for t in time_periods:
        # At most one incoming and one outgoing arc for each visited customer.
        for i in customer_nodes:
            model.addConstr(
                gp.quicksum(x[j, i, t] for j in nodes if j != i) == y[i, t],
                name=f"in_degree[{i},{t}]",
            )
            model.addConstr(
                gp.quicksum(x[i, j, t] for j in nodes if j != i) == y[i, t],
                name=f"out_degree[{i},{t}]",
            )

        model.addConstr(
            gp.quicksum(x[0, j, t] for j in customer_nodes)
            == gp.quicksum(x[j, 0, t] for j in customer_nodes),
            name=f"depot_degree_balance[{t}]",
        )
        model.addConstr(
            gp.quicksum(x[0, j, t] for j in customer_nodes) <= vehicles,
            name=f"vehicle_limit[{t}]",
        )

        # Arc capacity and empty returns to the depot.
        for i, j in arcs:
            model.addConstr(
                flow[i, j, t] <= capacity * x[i, j, t],
                name=f"flow_capacity[{i},{j},{t}]",
            )
        for i in customer_nodes:
            model.addConstr(flow[i, 0, t] == 0.0, name=f"empty_return[{i},{t}]")

        # Commodity-flow conservation. This also enforces route capacities.
        for i in customer_nodes:
            model.addConstr(
                gp.quicksum(flow[j, i, t] for j in nodes if j != i)
                - gp.quicksum(flow[i, j, t] for j in nodes if j != i)
                == q[i, t],
                name=f"flow_balance[{i},{t}]",
            )

        # MTZ constraints eliminate disconnected zero-delivery subtours.
        for i in customer_nodes:
            model.addConstr(order[i, t] >= y[i, t])
            model.addConstr(order[i, t] <= n * y[i, t])

        for i in customer_nodes:
            for j in customer_nodes:
                if i != j:
                    model.addConstr(
                        order[i, t] - order[j, t] + n * x[i, j, t] <= n - 1,
                        name=f"mtz[{i},{j},{t}]",
                    )

    travel_cost = gp.quicksum(
        float(distance[i][j]) * x[i, j, t]
        for i, j in arcs
        for t in time_periods
    )

    holding_cost = gp.quicksum(
        float(depot["holding_cost"]) * inv_depot[t] for t in time_periods
    ) + gp.quicksum(
        float(customers[i]["holding_cost"]) * inv_customer[i, t]
        for i in customer_nodes
        for t in time_periods
    )

    model.setObjective(travel_cost + holding_cost, GRB.MINIMIZE)
    model.update()

    variables = {
        "q": q,
        "inv_depot": inv_depot,
        "inv_customer": inv_customer,
        "y": y,
        "x": x,
        "flow": flow,
        "order": order,
        "arcs": arcs,
    }
    return model, variables


def set_mip_start(instance, variables, solution):
    if solution is None:
        return

    n = instance["num_customers"]
    periods = instance["num_periods"]
    capacity = float(instance["vehicle_capacity"])

    q = variables["q"]
    inv_depot = variables["inv_depot"]
    inv_customer = variables["inv_customer"]
    y = variables["y"]
    x = variables["x"]
    flow = variables["flow"]
    order = variables["order"]
    arcs = variables["arcs"]

    for t in range(1, periods + 1):
        period = solution["solution_details"]["periods"][str(t)]
        deliveries = {
            int(i): float(value) for i, value in period["deliveries"].items()
        }

        inv_depot[t].Start = float(period["inventories"]["depot"])

        for i in range(1, n + 1):
            quantity = deliveries.get(i, 0.0)
            q[i, t].Start = quantity
            inv_customer[i, t].Start = float(period["inventories"][str(i)])
            y[i, t].Start = 1.0 if quantity > TOL else 0.0
            order[i, t].Start = 0.0

        for i, j in arcs:
            x[i, j, t].Start = 0.0
            flow[i, j, t].Start = 0.0

        for route in period["routes"]:
            remaining = sum(deliveries.get(i, 0.0) for i in route if i != 0)
            position = 1
            for a, b in zip(route[:-1], route[1:]):
                x[a, b, t].Start = 1.0
                if a == 0:
                    flow[a, b, t].Start = min(capacity, remaining)
                elif b == 0:
                    remaining -= deliveries.get(a, 0.0)
                    flow[a, b, t].Start = 0.0
                else:
                    remaining -= deliveries.get(a, 0.0)
                    flow[a, b, t].Start = max(0.0, remaining)

                if b != 0:
                    order[b, t].Start = float(position)
                    position += 1


def extract_solution(instance, variables, getter):
    n = instance["num_customers"]
    periods = instance["num_periods"]
    x = variables["x"]
    q = variables["q"]

    period_routes = {}
    period_deliveries = {}

    for t in range(1, periods + 1):
        deliveries = {}
        for i in range(1, n + 1):
            value = max(0.0, float(getter(q[i, t])))
            if value > TOL:
                deliveries[i] = value

        successor = {}
        for i in range(0, n + 1):
            selected = []
            for j in range(0, n + 1):
                if i != j and getter(x[i, j, t]) > 0.5:
                    selected.append(j)
            if selected:
                successor[i] = min(selected)

        starts = [
            j for j in range(1, n + 1)
            if getter(x[0, j, t]) > 0.5
        ]
        starts.sort()

        routes = []
        visited = set()
        for start in starts:
            route = [0, start]
            current = start
            visited.add(start)

            for _ in range(n + 1):
                nxt = successor.get(current, 0)
                route.append(nxt)
                if nxt == 0:
                    break
                if nxt in visited:
                    route.append(0)
                    break
                visited.add(nxt)
                current = nxt

            if route[-1] != 0:
                route.append(0)
            routes.append(route)

        period_routes[t] = routes
        period_deliveries[t] = deliveries

    return calculate_solution(instance, period_routes, period_deliveries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    best_solution = construct_greedy_solution(instance)
    best_objective = (
        float(best_solution["objective_value"])
        if best_solution is not None
        else float("inf")
    )

    if logger and best_solution is not None:
        logger.log_solution(best_objective, best_solution)

    model, variables = build_model(instance)
    set_mip_start(instance, variables, best_solution)

    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.0, float(args.time_limit) - elapsed)

    model._all_variables = model.getVars()
    model._solution_variables = variables
    model._instance = instance
    model._logger = logger
    model._best_solution = best_solution
    model._best_objective = best_objective

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = cb_model.cbGetSolution(cb_model._all_variables)

            def getter(var):
                return values[var.index]

            solution = extract_solution(
                cb_model._instance,
                cb_model._solution_variables,
                getter,
            )
            objective = float(solution["objective_value"])

            if objective < cb_model._best_objective - 1e-7:
                cb_model._best_objective = objective
                cb_model._best_solution = solution
                if cb_model._logger:
                    cb_model._logger.log_solution(objective, solution)
        except Exception:
            # Logging or extraction errors must not abort the optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_candidate = extract_solution(
            instance,
            variables,
            lambda var: var.X,
        )
        final_objective = float(final_candidate["objective_value"])

        if final_objective < model._best_objective - 1e-7:
            model._best_objective = final_objective
            model._best_solution = final_candidate
            if logger:
                logger.log_solution(final_objective, final_candidate)

    best_solution = model._best_solution
    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()