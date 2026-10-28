import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-7


def configure_model(model, time_limit):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.01, float(time_limit))


def clean_number(value):
    value = float(value)
    if abs(value) < 1e-8:
        return 0.0
    nearest = round(value)
    if abs(value - nearest) < 1e-7:
        return float(nearest)
    return value


def route_cost(route, distances):
    return sum(distances[route[i]][route[i + 1]]
               for i in range(len(route) - 1))


def improve_route_2opt(route, distances):
    if len(route) <= 4:
        return route[:]

    best = route[:]
    improved = True
    passes = 0

    while improved and passes < 20:
        improved = False
        passes += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                old_cost = distances[a][b] + distances[c][d]
                new_cost = distances[a][c] + distances[b][d]
                if new_cost + 1e-9 < old_cost:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
        if not improved:
            break
    return best


def nearest_neighbor_route(customers, distances):
    if not customers:
        return [0, 0]

    remaining = set(customers)
    route = [0]
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda j: (distances[current][j], j))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    route.append(0)
    return improve_route_2opt(route, distances)


def compute_objective(solution, data):
    distances = data["distance_matrix"]
    holding = data["holding_costs"]
    num_periods = data["num_periods"]
    num_nodes = data["num_nodes"]

    total = 0.0
    for period_routes in solution["routes"].values():
        for vehicle_data in period_routes.values():
            route = vehicle_data["route"]
            total += route_cost(route, distances)

    for i in range(num_nodes):
        loc_inv = solution["inventories"][str(i)]
        for t in range(num_periods):
            total += holding[i] * float(loc_inv[str(t)])

    return float(total)


def make_solution_from_plan(data, assignment_values, delivery_values,
                            inventory_values):
    n = data["num_customers"]
    T = data["num_periods"]
    K = data["num_vehicles"]
    distances = data["distance_matrix"]

    routes = {str(t): {} for t in range(T)}

    for t in range(T):
        vehicle_out = 0
        for k in range(K):
            customers = [
                i for i in range(1, n + 1)
                if assignment_values.get((t, k, i), 0.0) > 0.5
            ]
            if not customers:
                continue

            route = nearest_neighbor_route(customers, distances)
            deliveries = {}
            for i in route[1:-1]:
                qty = delivery_values.get((t, k, i), 0.0)
                deliveries[str(i)] = clean_number(qty)

            routes[str(t)][str(vehicle_out)] = {
                "route": route,
                "deliveries": deliveries,
            }
            vehicle_out += 1

    inventories = {}
    for i in range(n + 1):
        inventories[str(i)] = {
            str(t): clean_number(inventory_values[(i, t)])
            for t in range(T)
        }

    solution = {
        "objective_value": 0.0,
        "routes": routes,
        "inventories": inventories,
    }
    solution["objective_value"] = compute_objective(solution, data)
    return solution


def build_compact_planning_model(data, time_limit):
    n = data["num_customers"]
    T = data["num_periods"]
    K = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    capacities = data["inventory_capacities"]
    initial = data["initial_inventories"]
    demands = data["customer_demands"]
    production = data["supplier_production"]
    holding = data["holding_costs"]
    distances = data["distance_matrix"]

    model = gp.Model("inventory_routing_planning")
    configure_model(model, time_limit)

    assign = {}
    delivery = {}
    active = {}
    inventory = {}

    for t in range(T):
        for k in range(K):
            active[t, k] = model.addVar(vtype=GRB.BINARY,
                                        name=f"active_{t}_{k}")
            for i in range(1, n + 1):
                assign[t, k, i] = model.addVar(
                    vtype=GRB.BINARY, name=f"a_{t}_{k}_{i}"
                )
                delivery[t, k, i] = model.addVar(
                    lb=0.0,
                    ub=min(Q, capacities[i]),
                    vtype=GRB.CONTINUOUS,
                    name=f"d_{t}_{k}_{i}",
                )

    for i in range(n + 1):
        for t in range(T):
            inventory[i, t] = model.addVar(
                lb=0.0,
                ub=capacities[i],
                vtype=GRB.CONTINUOUS,
                name=f"I_{i}_{t}",
            )

    model.update()

    for t in range(T):
        for k in range(K):
            model.addConstr(
                gp.quicksum(delivery[t, k, i] for i in range(1, n + 1))
                <= Q * active[t, k]
            )
            for i in range(1, n + 1):
                model.addConstr(assign[t, k, i] <= active[t, k])
                model.addConstr(
                    delivery[t, k, i]
                    <= min(Q, capacities[i]) * assign[t, k, i]
                )

        for k in range(K - 1):
            model.addConstr(active[t, k] >= active[t, k + 1])

        for i in range(1, n + 1):
            model.addConstr(
                gp.quicksum(assign[t, k, i] for k in range(K)) <= 1
            )

            shipped = gp.quicksum(delivery[t, k, i] for k in range(K))
            previous = initial[i] if t == 0 else inventory[i, t - 1]

            model.addConstr(
                inventory[i, t] == previous + shipped - demands[i - 1][t]
            )
            model.addConstr(previous + shipped <= capacities[i])

        total_shipped = gp.quicksum(
            delivery[t, k, i]
            for k in range(K)
            for i in range(1, n + 1)
        )
        previous_depot = initial[0] if t == 0 else inventory[0, t - 1]
        model.addConstr(
            inventory[0, t] == previous_depot + production[t] - total_shipped
        )

    holding_obj = gp.quicksum(
        holding[i] * inventory[i, t]
        for i in range(n + 1)
        for t in range(T)
    )
    visit_proxy = gp.quicksum(
        2.0 * distances[0][i] * assign[t, k, i]
        for t in range(T)
        for k in range(K)
        for i in range(1, n + 1)
    )
    model.setObjective(holding_obj + visit_proxy, GRB.MINIMIZE)

    return model, assign, delivery, inventory, active


def extract_plan_values(model, assign, delivery, inventory):
    assignment_values = {key: var.X for key, var in assign.items()}
    delivery_values = {key: var.X for key, var in delivery.items()}
    inventory_values = {key: var.X for key, var in inventory.items()}
    return assignment_values, delivery_values, inventory_values


def build_exact_model(data, time_limit):
    n = data["num_customers"]
    T = data["num_periods"]
    K = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    capacities = data["inventory_capacities"]
    initial = data["initial_inventories"]
    demands = data["customer_demands"]
    production = data["supplier_production"]
    holding = data["holding_costs"]
    distances = data["distance_matrix"]

    model = gp.Model("inventory_routing_exact")
    configure_model(model, time_limit)

    x = {}
    flow = {}
    visit = {}
    delivery = {}
    order = {}
    inventory = {}

    for t in range(T):
        for i in range(n + 1):
            for j in range(n + 1):
                if i == j:
                    continue
                x[t, i, j] = model.addVar(
                    vtype=GRB.BINARY, name=f"x_{t}_{i}_{j}"
                )
                flow[t, i, j] = model.addVar(
                    lb=0.0, ub=Q, vtype=GRB.CONTINUOUS,
                    name=f"f_{t}_{i}_{j}",
                )

        for i in range(1, n + 1):
            visit[t, i] = model.addVar(
                vtype=GRB.BINARY, name=f"y_{t}_{i}"
            )
            delivery[t, i] = model.addVar(
                lb=0.0,
                ub=min(Q, capacities[i]),
                vtype=GRB.CONTINUOUS,
                name=f"q_{t}_{i}",
            )
            order[t, i] = model.addVar(
                lb=0.0, ub=n, vtype=GRB.CONTINUOUS,
                name=f"u_{t}_{i}",
            )

    for i in range(n + 1):
        for t in range(T):
            inventory[i, t] = model.addVar(
                lb=0.0,
                ub=capacities[i],
                vtype=GRB.CONTINUOUS,
                name=f"I_{i}_{t}",
            )

    model.update()

    for t in range(T):
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    model.addConstr(flow[t, i, j] <= Q * x[t, i, j])

        departures = gp.quicksum(x[t, 0, j] for j in range(1, n + 1))
        returns = gp.quicksum(x[t, i, 0] for i in range(1, n + 1))
        model.addConstr(departures == returns)
        model.addConstr(departures <= K)

        for i in range(1, n + 1):
            incoming_x = gp.quicksum(
                x[t, j, i] for j in range(n + 1) if j != i
            )
            outgoing_x = gp.quicksum(
                x[t, i, j] for j in range(n + 1) if j != i
            )
            model.addConstr(incoming_x == visit[t, i])
            model.addConstr(outgoing_x == visit[t, i])
            model.addConstr(
                delivery[t, i]
                <= min(Q, capacities[i]) * visit[t, i]
            )

            incoming_flow = gp.quicksum(
                flow[t, j, i] for j in range(n + 1) if j != i
            )
            outgoing_flow = gp.quicksum(
                flow[t, i, j] for j in range(n + 1) if j != i
            )
            model.addConstr(incoming_flow - outgoing_flow == delivery[t, i])

            model.addConstr(order[t, i] >= visit[t, i])
            model.addConstr(order[t, i] <= n * visit[t, i])

            previous = initial[i] if t == 0 else inventory[i, t - 1]
            model.addConstr(
                inventory[i, t]
                == previous + delivery[t, i] - demands[i - 1][t]
            )
            model.addConstr(previous + delivery[t, i] <= capacities[i])

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                model.addConstr(
                    order[t, i] - order[t, j] + n * x[t, i, j]
                    <= n - 1 + n * (1 - visit[t, j])
                )

        total_shipped = gp.quicksum(delivery[t, i]
                                    for i in range(1, n + 1))
        previous_depot = initial[0] if t == 0 else inventory[0, t - 1]
        model.addConstr(
            inventory[0, t]
            == previous_depot + production[t] - total_shipped
        )

    travel_obj = gp.quicksum(
        distances[i][j] * x[t, i, j]
        for t in range(T)
        for i in range(n + 1)
        for j in range(n + 1)
        if i != j
    )
    holding_obj = gp.quicksum(
        holding[i] * inventory[i, t]
        for i in range(n + 1)
        for t in range(T)
    )
    model.setObjective(travel_obj + holding_obj, GRB.MINIMIZE)

    return model, x, flow, visit, delivery, order, inventory


def reconstruct_routes_from_arc_values(data, arc_values, delivery_values):
    n = data["num_customers"]
    T = data["num_periods"]

    routes = {str(t): {} for t in range(T)}

    for t in range(T):
        successors = {}
        for i in range(n + 1):
            selected = [
                j for j in range(n + 1)
                if i != j and arc_values.get((t, i, j), 0.0) > 0.5
            ]
            if selected:
                successors[i] = selected

        starts = sorted(successors.get(0, []))
        vehicle = 0
        seen_customers = set()

        for first in starts:
            route = [0]
            current = first
            local_seen = set()

            while current != 0:
                if current in local_seen or current in seen_customers:
                    raise ValueError("Invalid route cycle in incumbent")
                local_seen.add(current)
                seen_customers.add(current)
                route.append(current)

                next_nodes = successors.get(current, [])
                if not next_nodes:
                    raise ValueError("Open route in incumbent")
                current = next_nodes[0]

                if len(route) > n + 1:
                    raise ValueError("Route reconstruction exceeded node count")

            route.append(0)
            deliveries = {
                str(i): clean_number(delivery_values.get((t, i), 0.0))
                for i in route[1:-1]
            }
            routes[str(t)][str(vehicle)] = {
                "route": route,
                "deliveries": deliveries,
            }
            vehicle += 1

    return routes


def make_solution_from_exact_values(data, arc_values, delivery_values,
                                    inventory_values):
    n = data["num_customers"]
    T = data["num_periods"]

    routes = reconstruct_routes_from_arc_values(
        data, arc_values, delivery_values
    )
    inventories = {
        str(i): {
            str(t): clean_number(inventory_values[(i, t)])
            for t in range(T)
        }
        for i in range(n + 1)
    }

    solution = {
        "objective_value": 0.0,
        "routes": routes,
        "inventories": inventories,
    }
    solution["objective_value"] = compute_objective(solution, data)
    return solution


def apply_plan_start(data, plan_values, x, flow, visit, delivery, order,
                     inventory):
    if plan_values is None:
        return

    assignment_values, delivery_values, inventory_values = plan_values
    n = data["num_customers"]
    T = data["num_periods"]
    K = data["num_vehicles"]
    distances = data["distance_matrix"]

    for var in x.values():
        var.Start = 0.0
    for var in flow.values():
        var.Start = 0.0
    for var in visit.values():
        var.Start = 0.0
    for var in delivery.values():
        var.Start = 0.0
    for var in order.values():
        var.Start = 0.0

    for key, var in inventory.items():
        var.Start = inventory_values[key]

    for t in range(T):
        for i in range(1, n + 1):
            qty = sum(delivery_values.get((t, k, i), 0.0)
                      for k in range(K))
            used = any(assignment_values.get((t, k, i), 0.0) > 0.5
                       for k in range(K))
            visit[t, i].Start = 1.0 if used else 0.0
            delivery[t, i].Start = qty

        for k in range(K):
            customers = [
                i for i in range(1, n + 1)
                if assignment_values.get((t, k, i), 0.0) > 0.5
            ]
            if not customers:
                continue

            route = nearest_neighbor_route(customers, distances)
            route_quantities = {
                i: delivery_values.get((t, k, i), 0.0)
                for i in customers
            }

            for position, i in enumerate(route[1:-1], start=1):
                order[t, i].Start = float(position)

            remaining = sum(route_quantities.values())
            for p in range(len(route) - 1):
                i, j = route[p], route[p + 1]
                x[t, i, j].Start = 1.0
                if i != 0:
                    remaining -= route_quantities.get(i, 0.0)
                flow[t, i, j].Start = max(0.0, remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") \
        if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    best_solution = None
    best_objective = math.inf
    plan_values = None

    # First solve a much smaller inventory/assignment model. Any incumbent
    # from this model can be converted directly into feasible vehicle tours.
    if args.time_limit > 1:
        planning_budget = min(
            12.0,
            max(0.5, 0.18 * args.time_limit),
            max(0.01, deadline - time.monotonic()),
        )

        if planning_budget > 0.05:
            try:
                pmodel, passign, pdelivery, pinventory, _ = \
                    build_compact_planning_model(data, planning_budget)

                compact_best_true = [best_objective]
                compact_best_solution = [best_solution]

                def planning_callback(model, where):
                    if where != GRB.Callback.MIPSOL:
                        return
                    try:
                        av = {
                            key: model.cbGetSolution(var)
                            for key, var in passign.items()
                        }
                        dv = {
                            key: model.cbGetSolution(var)
                            for key, var in pdelivery.items()
                        }
                        iv = {
                            key: model.cbGetSolution(var)
                            for key, var in pinventory.items()
                        }
                        sol = make_solution_from_plan(data, av, dv, iv)
                        obj = sol["objective_value"]
                        if obj + 1e-7 < compact_best_true[0]:
                            compact_best_true[0] = obj
                            compact_best_solution[0] = sol
                            if logger:
                                logger.log_solution(obj, sol)
                    except Exception:
                        pass

                pmodel.optimize(planning_callback)

                if pmodel.SolCount > 0:
                    plan_values = extract_plan_values(
                        pmodel, passign, pdelivery, pinventory
                    )
                    final_plan_solution = make_solution_from_plan(
                        data, *plan_values
                    )
                    if (final_plan_solution["objective_value"] + 1e-7
                            < compact_best_true[0]):
                        compact_best_true[0] = \
                            final_plan_solution["objective_value"]
                        compact_best_solution[0] = final_plan_solution
                        if logger:
                            logger.log_solution(
                                final_plan_solution["objective_value"],
                                final_plan_solution,
                            )

                best_objective = compact_best_true[0]
                best_solution = compact_best_solution[0]
                pmodel.dispose()
            except gp.GurobiError:
                pass

    remaining = deadline - time.monotonic()

    # Solve the exact arc-flow formulation with the compact solution as a
    # MIP start. If interrupted, retain the best complete feasible incumbent.
    if remaining > 0.02 or best_solution is None:
        try:
            exact_limit = max(0.01, deadline - time.monotonic())
            model, x, flow, visit, delivery, order, inventory = \
                build_exact_model(data, exact_limit)

            if plan_values is not None:
                apply_plan_start(
                    data, plan_values, x, flow, visit, delivery, order,
                    inventory
                )

            incumbent_objective = [best_objective]
            incumbent_solution = [best_solution]

            def exact_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    model_obj = cb_model.cbGet(
                        GRB.Callback.MIPSOL_OBJ
                    )
                    if model_obj >= incumbent_objective[0] - 1e-7:
                        return

                    xv = {
                        key: cb_model.cbGetSolution(var)
                        for key, var in x.items()
                    }
                    qv = {
                        key: cb_model.cbGetSolution(var)
                        for key, var in delivery.items()
                    }
                    iv = {
                        key: cb_model.cbGetSolution(var)
                        for key, var in inventory.items()
                    }
                    sol = make_solution_from_exact_values(
                        data, xv, qv, iv
                    )
                    obj = sol["objective_value"]

                    if obj + 1e-7 < incumbent_objective[0]:
                        incumbent_objective[0] = obj
                        incumbent_solution[0] = sol
                        if logger:
                            logger.log_solution(obj, sol)
                except Exception:
                    pass

            model.optimize(exact_callback)

            if model.SolCount > 0:
                xv = {key: var.X for key, var in x.items()}
                qv = {key: var.X for key, var in delivery.items()}
                iv = {key: var.X for key, var in inventory.items()}
                final_exact = make_solution_from_exact_values(
                    data, xv, qv, iv
                )
                if (final_exact["objective_value"] + 1e-7
                        < incumbent_objective[0]):
                    incumbent_objective[0] = final_exact["objective_value"]
                    incumbent_solution[0] = final_exact
                    if logger:
                        logger.log_solution(
                            final_exact["objective_value"], final_exact
                        )

            best_objective = incumbent_objective[0]
            best_solution = incumbent_solution[0]
            model.dispose()
        except gp.GurobiError:
            pass

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit. "
            "The instance may be infeasible."
        )

    best_solution["objective_value"] = compute_objective(best_solution, data)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()