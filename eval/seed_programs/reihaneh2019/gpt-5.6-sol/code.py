import argparse
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def hungarian_rectangular(cost):
    """Minimum-cost assignment of each row to a distinct column, for rows <= columns."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m:
        return None

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def pack_customers(order, demands, capacity):
    bins = []
    loads = []

    for c in order:
        d = demands[c]
        best_bin = -1
        best_remaining = float("inf")

        for b, load in enumerate(loads):
            if load + d <= capacity:
                remaining = capacity - load - d
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b

        if best_bin < 0:
            bins.append([c])
            loads.append(d)
        else:
            bins[best_bin].append(c)
            loads[best_bin] += d

    return bins, loads


def make_solution_from_routes(
    route_site_lists,
    route_customers,
    customer_site,
    depot_id,
    site_ids,
    customer_ids,
    routing_cost,
    assignment_cost,
):
    routes = {}
    assignments = {}
    customer_tours = {}
    objective = 0.0

    for k, sites in enumerate(route_site_lists):
        if not sites:
            continue

        positional_sequence = [0] + [s + 1 for s in sites] + [0]
        sequence = [depot_id] + [site_ids[s] for s in sites] + [depot_id]
        arcs = []

        for idx in range(len(positional_sequence) - 1):
            i = positional_sequence[idx]
            j = positional_sequence[idx + 1]
            objective += routing_cost[i][j]
            arcs.append([sequence[idx], sequence[idx + 1]])

        routes[str(k)] = {
            "arcs": arcs,
            "sequence": sequence,
        }

        for c in route_customers[k]:
            s = customer_site[c]
            assignments[str(customer_ids[c])] = site_ids[s]
            customer_tours[str(customer_ids[c])] = {
                "vehicle": k,
                "site": site_ids[s],
            }
            objective += assignment_cost[c][s]

    return {
        "objective_value": float(objective),
        "routes": routes,
        "assignments": assignments,
        "customer_tours": customer_tours,
    }


def construct_heuristic(
    demands,
    capacity,
    max_vehicles,
    num_sites,
    routing_cost,
    assignment_cost,
    depot_id,
    site_ids,
    customer_ids,
    heuristic_deadline,
):
    num_customers = len(demands)
    if num_customers == 0:
        return {
            "objective_value": 0.0,
            "routes": {},
            "assignments": {},
            "customer_tours": {},
        }, None

    if max(demands) > capacity or max_vehicles <= 0 or num_sites <= 0:
        return None, None

    rng = random.Random(0)
    base_order = sorted(
        range(num_customers),
        key=lambda c: (-demands[c], c),
    )

    candidate_orders = [base_order]
    while len(candidate_orders) < 100 and time.monotonic() < heuristic_deadline:
        order = base_order[:]
        rng.shuffle(order)
        candidate_orders.append(order)

    best_solution = None
    best_start = None

    for order in candidate_orders:
        if time.monotonic() >= heuristic_deadline and best_solution is not None:
            break

        bins, loads = pack_customers(order, demands, capacity)
        if len(bins) > max_vehicles or len(bins) > num_sites:
            continue

        packed = sorted(
            zip(bins, loads),
            key=lambda item: (-item[1], min(item[0])),
        )
        bins = [item[0] for item in packed]
        loads = [item[1] for item in packed]

        site_match_cost = []
        for customers in bins:
            row = []
            for s in range(num_sites):
                value = routing_cost[0][s + 1] + routing_cost[s + 1][0]
                value += sum(assignment_cost[c][s] for c in customers)
                row.append(value)
            site_match_cost.append(row)

        matched_sites = hungarian_rectangular(site_match_cost)
        if matched_sites is None or any(s < 0 for s in matched_sites):
            continue

        route_site_lists = [[s] for s in matched_sites]
        customer_site = {}
        for k, customers in enumerate(bins):
            for c in customers:
                customer_site[c] = matched_sites[k]

        solution = make_solution_from_routes(
            route_site_lists,
            bins,
            customer_site,
            depot_id,
            site_ids,
            customer_ids,
            routing_cost,
            assignment_cost,
        )

        if best_solution is None or (
            solution["objective_value"] < best_solution["objective_value"] - 1e-9
        ):
            best_solution = solution
            best_start = {
                "bins": bins,
                "loads": loads,
                "sites": matched_sites,
                "customer_site": customer_site,
            }

    return best_solution, best_start


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

    num_sites = int(data["num_delivery_sites"])
    num_customers = int(data["num_customers"])
    capacity = int(data["vehicle_capacity"])
    available_vehicles = int(data["num_vehicles"])

    depot_id = int(data["depot"]["id"])
    site_ids = [int(site["id"]) for site in data["delivery_sites"]]
    customer_ids = [int(customer["id"]) for customer in data["customers"]]
    demands = [int(customer["demand"]) for customer in data["customers"]]

    routing_cost = data["routing_cost_matrix"]
    assignment_cost = data["assignment_cost_matrix"]

    if num_customers == 0:
        solution = {
            "objective_value": 0.0,
            "routes": {},
            "assignments": {},
            "customer_tours": {},
        }
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
        return

    max_vehicles = min(available_vehicles, num_sites, num_customers)
    heuristic_budget = min(
        0.5,
        max(0.02, 0.03 * max(1, args.time_limit)),
    )
    heuristic_deadline = start_time + heuristic_budget

    best_solution, warm_start = construct_heuristic(
        demands,
        capacity,
        max_vehicles,
        num_sites,
        routing_cost,
        assignment_cost,
        depot_id,
        site_ids,
        customer_ids,
        heuristic_deadline,
    )

    best_logged_value = float("inf")
    if best_solution is not None:
        best_logged_value = best_solution["objective_value"]
        if logger:
            logger.log_solution(best_logged_value, best_solution)

    if time.monotonic() - start_time >= args.time_limit and best_solution is not None:
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(best_solution, f, indent=2)
        return

    if max_vehicles <= 0:
        raise RuntimeError("Instance is infeasible: no vehicle/site can be used.")

    model = gp.Model("integrated_delivery_location_routing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    vehicles = range(max_vehicles)
    sites = range(num_sites)
    customers = range(num_customers)
    nodes = range(num_sites + 1)

    arc_keys = [
        (k, i, j)
        for k in vehicles
        for i in nodes
        for j in nodes
        if i != j
    ]
    pairing_keys = [
        (c, s, k)
        for c in customers
        for s in sites
        for k in vehicles
    ]

    x = model.addVars(arc_keys, vtype=GRB.BINARY, name="x")
    y = model.addVars(vehicles, sites, vtype=GRB.BINARY, name="y")
    use = model.addVars(vehicles, vtype=GRB.BINARY, name="use")
    z = model.addVars(pairing_keys, vtype=GRB.BINARY, name="z")
    position = model.addVars(
        vehicles,
        sites,
        lb=0.0,
        ub=float(num_sites),
        vtype=GRB.CONTINUOUS,
        name="position",
    )

    model.setObjective(
        gp.quicksum(
            routing_cost[i][j] * x[k, i, j]
            for k, i, j in arc_keys
        )
        + gp.quicksum(
            assignment_cost[c][s] * z[c, s, k]
            for c, s, k in pairing_keys
        ),
        GRB.MINIMIZE,
    )

    # Every customer is assigned to exactly one site on exactly one tour.
    model.addConstrs(
        (
            gp.quicksum(z[c, s, k] for s in sites for k in vehicles) == 1
            for c in customers
        ),
        name="customer_coverage",
    )

    # Site visitation and route degrees.
    for k in vehicles:
        model.addConstr(
            gp.quicksum(x[k, 0, j] for j in range(1, num_sites + 1)) == use[k],
            name=f"depot_out_{k}",
        )
        model.addConstr(
            gp.quicksum(x[k, i, 0] for i in range(1, num_sites + 1)) == use[k],
            name=f"depot_in_{k}",
        )

        for s in sites:
            node = s + 1
            model.addConstr(
                gp.quicksum(x[k, node, j] for j in nodes if j != node) == y[k, s],
                name=f"site_out_{k}_{s}",
            )
            model.addConstr(
                gp.quicksum(x[k, i, node] for i in nodes if i != node) == y[k, s],
                name=f"site_in_{k}_{s}",
            )

    # A delivery site belongs to at most one vehicle tour.
    model.addConstrs(
        (
            gp.quicksum(y[k, s] for k in vehicles) <= 1
            for s in sites
        ),
        name="site_unique_tour",
    )

    # Customer-site-tour pairings require that the site is on the tour.
    model.addConstrs(
        (
            z[c, s, k] <= y[k, s]
            for c in customers
            for s in sites
            for k in vehicles
        ),
        name="pairing_visit_link",
    )

    # Do not retain unnecessary visited sites.
    model.addConstrs(
        (
            y[k, s] <= gp.quicksum(z[c, s, k] for c in customers)
            for k in vehicles
            for s in sites
        ),
        name="visited_site_serves_customer",
    )

    # Vehicle capacity.
    route_load_expr = {}
    for k in vehicles:
        route_load_expr[k] = gp.quicksum(
            demands[c] * z[c, s, k]
            for c in customers
            for s in sites
        )
        model.addConstr(
            route_load_expr[k] <= capacity * use[k],
            name=f"capacity_{k}",
        )

    # MTZ constraints eliminate disconnected subtours among delivery sites.
    for k in vehicles:
        for s in sites:
            model.addConstr(position[k, s] >= y[k, s])
            model.addConstr(position[k, s] <= num_sites * y[k, s])

        for si in sites:
            for sj in sites:
                if si != sj:
                    model.addConstr(
                        position[k, si]
                        - position[k, sj]
                        + num_sites * x[k, si + 1, sj + 1]
                        <= num_sites - 1,
                        name=f"mtz_{k}_{si}_{sj}",
                    )

    # Vehicle-label symmetry breaking.
    for k in range(max_vehicles - 1):
        model.addConstr(use[k] >= use[k + 1], name=f"use_symmetry_{k}")
        model.addConstr(
            route_load_expr[k] >= route_load_expr[k + 1],
            name=f"load_symmetry_{k}",
        )

    # Supply the constructive incumbent as a MIP start.
    if warm_start is not None:
        for var in x.values():
            var.Start = 0.0
        for var in y.values():
            var.Start = 0.0
        for var in use.values():
            var.Start = 0.0
        for var in z.values():
            var.Start = 0.0
        for var in position.values():
            var.Start = 0.0

        for k, customers_in_route in enumerate(warm_start["bins"]):
            s = warm_start["sites"][k]
            use[k].Start = 1.0
            y[k, s].Start = 1.0
            position[k, s].Start = 1.0
            x[k, 0, s + 1].Start = 1.0
            x[k, s + 1, 0].Start = 1.0
            for c in customers_in_route:
                z[c, s, k].Start = 1.0

    x_keys = list(x.keys())
    x_vars = [x[key] for key in x_keys]
    z_keys = list(z.keys())
    z_vars = [z[key] for key in z_keys]

    def solution_from_values(x_values, z_values):
        successors = [{} for _ in vehicles]
        selected_z = [[] for _ in customers]

        for key, value in zip(x_keys, x_values):
            if value > 0.5:
                k, i, j = key
                successors[k][i] = j

        for key, value in zip(z_keys, z_values):
            if value > 0.5:
                c, s, k = key
                selected_z[c].append((s, k))

        routes = {}
        assignments = {}
        customer_tours = {}
        objective = 0.0

        for k in vehicles:
            if 0 not in successors[k]:
                continue

            positional_sequence = [0]
            current = 0
            visited_steps = 0

            while visited_steps <= num_sites + 1:
                if current not in successors[k]:
                    break
                nxt = successors[k][current]
                positional_sequence.append(nxt)
                visited_steps += 1
                current = nxt
                if current == 0:
                    break

            if positional_sequence[-1] != 0:
                return None

            sequence = [
                depot_id if node == 0 else site_ids[node - 1]
                for node in positional_sequence
            ]
            arcs = []

            for idx in range(len(positional_sequence) - 1):
                i = positional_sequence[idx]
                j = positional_sequence[idx + 1]
                objective += routing_cost[i][j]
                arcs.append([sequence[idx], sequence[idx + 1]])

            routes[str(k)] = {
                "arcs": arcs,
                "sequence": sequence,
            }

        for c in customers:
            if len(selected_z[c]) != 1:
                return None
            s, k = selected_z[c][0]
            assignments[str(customer_ids[c])] = site_ids[s]
            customer_tours[str(customer_ids[c])] = {
                "vehicle": k,
                "site": site_ids[s],
            }
            objective += assignment_cost[c][s]

        return {
            "objective_value": float(objective),
            "routes": routes,
            "assignments": assignments,
            "customer_tours": customer_tours,
        }

    callback_state = {
        "best_logged": best_logged_value,
        "best_solution": best_solution,
    }

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj >= callback_state["best_logged"] - 1e-8:
                return

            x_values = cb_model.cbGetSolution(x_vars)
            z_values = cb_model.cbGetSolution(z_vars)
            solution = solution_from_values(x_values, z_values)
            if solution is None:
                return

            value = solution["objective_value"]
            if value < callback_state["best_logged"] - 1e-8:
                callback_state["best_logged"] = value
                callback_state["best_solution"] = solution
                if logger:
                    logger.log_solution(value, solution)
        except Exception:
            # Logging must never interrupt optimization.
            pass

    remaining = args.time_limit - (time.monotonic() - start_time)
    model.Params.TimeLimit = max(0.01, remaining)
    model.optimize(callback)

    final_solution = callback_state["best_solution"]

    if model.SolCount > 0:
        model_solution = solution_from_values(
            [var.X for var in x_vars],
            [var.X for var in z_vars],
        )
        if model_solution is not None and (
            final_solution is None
            or model_solution["objective_value"]
            < final_solution["objective_value"] - 1e-8
        ):
            final_solution = model_solution
            if logger and (
                model_solution["objective_value"]
                < callback_state["best_logged"] - 1e-8
            ):
                logger.log_solution(
                    model_solution["objective_value"],
                    model_solution,
                )

    if final_solution is None:
        raise RuntimeError("No feasible solution was found within the time limit.")

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()