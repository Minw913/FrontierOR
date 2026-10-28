import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clean_number(value, tol=1e-6):
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) <= tol:
        return int(rounded)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_retailers"])
    H = int(data["time_horizon"])
    retailers = data["retailers"]
    supplier = data["supplier"]
    capacity = float(data["vehicle_capacity"])
    distance = data["distance_matrix"]

    retailer_ids = [r["id"] for r in retailers]
    supplier_id = supplier["id"]

    demand = [float(r["demand"]) for r in retailers]
    max_inventory = [float(r["max_inventory"]) for r in retailers]
    retailer_start = [float(r["starting_inventory"]) for r in retailers]
    retailer_cost = [float(r["inventory_cost"]) for r in retailers]

    supplier_start = float(supplier["starting_inventory"])
    production = float(supplier["production_rate"])
    supplier_cost = float(supplier["inventory_cost"])

    delivery_periods = range(1, H + 1)
    inventory_periods = range(1, H + 2)
    R = range(n)

    model = gp.Model("inventory_routing_order_up_to")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)

    # Visit and shipment decisions.
    y = {
        (i, t): model.addVar(vtype=GRB.BINARY, name=f"visit_{i}_{t}")
        for i in R for t in delivery_periods
    }
    q = {
        (i, t): model.addVar(
            lb=0.0,
            ub=max_inventory[i],
            vtype=GRB.CONTINUOUS,
            name=f"shipment_{i}_{t}",
        )
        for i in R for t in delivery_periods
    }

    # Inventory variables include the extra terminal period H+1.
    supplier_upper = max(
        0.0, supplier_start + (H + 1) * max(0.0, production)
    )
    supplier_inventory = {
        t: model.addVar(
            lb=0.0,
            ub=supplier_upper,
            vtype=GRB.CONTINUOUS,
            name=f"supplier_inventory_{t}",
        )
        for t in inventory_periods
    }
    retailer_inventory = {
        (i, t): model.addVar(
            lb=0.0,
            ub=max_inventory[i],
            vtype=GRB.CONTINUOUS,
            name=f"retailer_inventory_{i}_{t}",
        )
        for i in R for t in inventory_periods
    }

    # Whether the vehicle is used in a period.
    route_used = {
        t: model.addVar(vtype=GRB.BINARY, name=f"route_used_{t}")
        for t in delivery_periods
    }

    # Depot-retailer edges can be traversed twice for a direct round trip.
    x_depot = {
        (i, t): model.addVar(
            lb=0.0, ub=2.0, vtype=GRB.INTEGER, name=f"edge_depot_{i}_{t}"
        )
        for i in R for t in delivery_periods
    }

    # Retailer-retailer undirected edges.
    x_retail = {
        (i, j, t): model.addVar(
            vtype=GRB.BINARY, name=f"edge_{i}_{j}_{t}"
        )
        for t in delivery_periods
        for i in R
        for j in range(i + 1, n)
    }

    model.update()

    # Initial inventories.
    model.addConstr(supplier_inventory[1] == supplier_start)
    for i in R:
        model.addConstr(retailer_inventory[i, 1] == retailer_start[i])

    # Inventory recurrences.
    for t in range(2, H + 2):
        model.addConstr(
            supplier_inventory[t]
            == supplier_inventory[t - 1]
            + production
            - gp.quicksum(q[i, t - 1] for i in R)
        )
        for i in R:
            model.addConstr(
                retailer_inventory[i, t]
                == retailer_inventory[i, t - 1]
                + q[i, t - 1]
                - demand[i]
            )

    # Order-up-to shipment policy.
    for i in R:
        U = max_inventory[i]
        for t in delivery_periods:
            model.addConstr(q[i, t] <= U * y[i, t])
            model.addConstr(q[i, t] <= U - retailer_inventory[i, t])
            model.addConstr(
                q[i, t]
                >= U - retailer_inventory[i, t] - U * (1 - y[i, t])
            )

    # Vehicle and supplier capacities.
    for t in delivery_periods:
        total_shipped = gp.quicksum(q[i, t] for i in R)
        model.addConstr(total_shipped <= capacity)
        model.addConstr(total_shipped <= supplier_inventory[t])

    # Route degree constraints.
    for t in delivery_periods:
        for i in R:
            incident = x_depot[i, t]
            incident += gp.quicksum(
                x_retail[min(i, j), max(i, j), t]
                for j in R if j != i
            )
            model.addConstr(incident == 2 * y[i, t])
            model.addConstr(x_depot[i, t] <= 2 * y[i, t])
            model.addConstr(y[i, t] <= route_used[t])

        if n > 0:
            model.addConstr(
                gp.quicksum(x_depot[i, t] for i in R) == 2 * route_used[t]
            )
            model.addConstr(route_used[t] <= gp.quicksum(y[i, t] for i in R))
        else:
            model.addConstr(route_used[t] == 0)

    inventory_objective = supplier_cost * gp.quicksum(
        supplier_inventory[t] for t in inventory_periods
    )
    inventory_objective += gp.quicksum(
        retailer_cost[i] * retailer_inventory[i, t]
        for i in R for t in inventory_periods
    )

    transport_objective = gp.quicksum(
        float(distance[0][i + 1]) * x_depot[i, t]
        for i in R for t in delivery_periods
    )
    transport_objective += gp.quicksum(
        float(distance[i + 1][j + 1]) * x_retail[i, j, t]
        for t in delivery_periods
        for i in R
        for j in range(i + 1, n)
    )

    model.setObjective(inventory_objective + transport_objective, GRB.MINIMIZE)

    def build_solution(get_value):
        supplier_vals = {
            t: float(get_value(supplier_inventory[t]))
            for t in inventory_periods
        }
        retailer_vals = {
            (i, t): float(get_value(retailer_inventory[i, t]))
            for i in R for t in inventory_periods
        }
        y_vals = {
            (i, t): float(get_value(y[i, t]))
            for i in R for t in delivery_periods
        }
        q_vals = {
            (i, t): float(get_value(q[i, t]))
            for i in R for t in delivery_periods
        }
        depot_vals = {
            (i, t): float(get_value(x_depot[i, t]))
            for i in R for t in delivery_periods
        }
        retail_edge_vals = {
            (i, j, t): float(get_value(x_retail[i, j, t]))
            for t in delivery_periods
            for i in R
            for j in range(i + 1, n)
        }

        deliveries = {}
        routes = {}

        for t in delivery_periods:
            period_deliveries = {}
            for i in R:
                if y_vals[i, t] > 0.5:
                    period_deliveries[str(retailer_ids[i])] = clean_number(
                        q_vals[i, t]
                    )
            deliveries[str(t)] = period_deliveries

            period_route = []
            for i in R:
                multiplicity = int(round(depot_vals[i, t]))
                if multiplicity > 0:
                    period_route.append(
                        [supplier_id, retailer_ids[i], multiplicity]
                    )

            for i in R:
                for j in range(i + 1, n):
                    multiplicity = int(round(retail_edge_vals[i, j, t]))
                    if multiplicity > 0:
                        period_route.append(
                            [retailer_ids[i], retailer_ids[j], multiplicity]
                        )
            routes[str(t)] = period_route

        supplier_output = {
            str(t): clean_number(supplier_vals[t])
            for t in inventory_periods
        }
        retailer_output = {
            str(retailer_ids[i]): {
                str(t): clean_number(retailer_vals[i, t])
                for t in inventory_periods
            }
            for i in R
        }

        objective = supplier_cost * sum(
            supplier_vals[t] for t in inventory_periods
        )
        objective += sum(
            retailer_cost[i] * retailer_vals[i, t]
            for i in R for t in inventory_periods
        )
        objective += sum(
            float(distance[0][i + 1]) * depot_vals[i, t]
            for i in R for t in delivery_periods
        )
        objective += sum(
            float(distance[i + 1][j + 1]) * retail_edge_vals[i, j, t]
            for t in delivery_periods
            for i in R
            for j in range(i + 1, n)
        )

        return {
            "objective_value": float(objective),
            "deliveries": deliveries,
            "routes": routes,
            "supplier_inventory": supplier_output,
            "retailer_inventory": retailer_output,
        }

    # Construct a simple feasible latest-delivery MIP start when possible.
    def install_greedy_start():
        if n == 0:
            for t in inventory_periods:
                supplier_inventory[t].Start = supplier_start + (t - 1) * production
            for t in delivery_periods:
                route_used[t].Start = 0
            return

        inv = list(retailer_start)
        sup = supplier_start
        schedule_y = {}
        schedule_q = {}
        schedule_inv = {(i, 1): inv[i] for i in R}
        schedule_sup = {1: sup}
        feasible = True

        for t in delivery_periods:
            total = 0.0
            for i in R:
                must_visit = inv[i] + 1e-9 < demand[i]
                amount = max_inventory[i] - inv[i] if must_visit else 0.0
                schedule_y[i, t] = 1 if must_visit else 0
                schedule_q[i, t] = amount
                total += amount

            if total > capacity + 1e-8 or total > sup + 1e-8:
                feasible = False
                break

            next_inv = []
            for i in R:
                value = inv[i] + schedule_q[i, t] - demand[i]
                if value < -1e-8 or value > max_inventory[i] + 1e-8:
                    feasible = False
                    break
                next_inv.append(max(0.0, value))
                schedule_inv[i, t + 1] = max(0.0, value)
            if not feasible:
                break

            sup = sup + production - total
            if sup < -1e-8:
                feasible = False
                break
            schedule_sup[t + 1] = max(0.0, sup)
            inv = next_inv

        if not feasible:
            return

        for i in R:
            for t in delivery_periods:
                y[i, t].Start = schedule_y[i, t]
                q[i, t].Start = schedule_q[i, t]
        for i in R:
            for t in inventory_periods:
                retailer_inventory[i, t].Start = schedule_inv[i, t]
        for t in inventory_periods:
            supplier_inventory[t].Start = schedule_sup[t]

        for t in delivery_periods:
            visited = [i for i in R if schedule_y[i, t] == 1]
            route_used[t].Start = 1 if visited else 0
            for i in R:
                x_depot[i, t].Start = 0
            for i in R:
                for j in range(i + 1, n):
                    x_retail[i, j, t].Start = 0

            if len(visited) == 1:
                x_depot[visited[0], t].Start = 2
            elif len(visited) > 1:
                # Nearest-neighbor tour used only as a feasible MIP start.
                remaining = set(visited)
                sequence = []
                current_matrix_index = 0
                while remaining:
                    nxt = min(
                        remaining,
                        key=lambda i: float(
                            distance[current_matrix_index][i + 1]
                        ),
                    )
                    sequence.append(nxt)
                    remaining.remove(nxt)
                    current_matrix_index = nxt + 1

                x_depot[sequence[0], t].Start = 1
                x_depot[sequence[-1], t].Start = 1
                for a, b in zip(sequence, sequence[1:]):
                    i, j = min(a, b), max(a, b)
                    x_retail[i, j, t].Start = 1

    install_greedy_start()

    callback_state = {"best_logged": float("inf")}

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        # Identify disconnected retailer-only components and add SECs.
        disconnected_components = []

        for t in delivery_periods:
            visited = [
                i for i in R
                if cb_model.cbGetSolution(y[i, t]) > 0.5
            ]
            if not visited:
                continue

            adjacency = {i: set() for i in visited}
            depot_neighbors = set()

            for i in visited:
                if cb_model.cbGetSolution(x_depot[i, t]) > 0.5:
                    depot_neighbors.add(i)

            for idx, i in enumerate(visited):
                for j in visited[idx + 1:]:
                    if cb_model.cbGetSolution(
                        x_retail[min(i, j), max(i, j), t]
                    ) > 0.5:
                        adjacency[i].add(j)
                        adjacency[j].add(i)

            unseen = set(visited)
            while unseen:
                root = next(iter(unseen))
                stack = [root]
                component = set()
                touches_depot = False

                while stack:
                    node = stack.pop()
                    if node not in unseen:
                        continue
                    unseen.remove(node)
                    component.add(node)
                    if node in depot_neighbors:
                        touches_depot = True
                    stack.extend(adjacency[node] & unseen)

                if not touches_depot:
                    disconnected_components.append((t, component))

        if disconnected_components:
            for t, component in disconnected_components:
                component_list = sorted(component)
                chosen = component_list[0]
                lhs = gp.quicksum(
                    x_retail[i, j, t]
                    for pos, i in enumerate(component_list)
                    for j in component_list[pos + 1:]
                )
                rhs = gp.quicksum(
                    y[i, t] for i in component_list if i != chosen
                )
                cb_model.cbLazy(lhs <= rhs)
            return

        if logger:
            try:
                candidate_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if candidate_obj < callback_state["best_logged"] - 1e-7:
                    solution = build_solution(cb_model.cbGetSolution)
                    logger.log_solution(solution["objective_value"], solution)
                    callback_state["best_logged"] = solution["objective_value"]
            except Exception:
                # Logging must not invalidate the optimization run.
                pass

    model.optimize(callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

    final_solution = build_solution(lambda var: var.X)

    if logger and (
        final_solution["objective_value"]
        < callback_state["best_logged"] - 1e-7
    ):
        logger.log_solution(
            final_solution["objective_value"], final_solution
        )

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()