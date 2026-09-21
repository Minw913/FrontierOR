import argparse
import json
import math
import os
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["num_customers"])
    Q = float(instance["vehicle_capacity"])
    declared_k = int(instance["num_vehicles_upper_bound"])
    R = n + 1

    depot = instance["depot"]
    customers = sorted(instance["customers"], key=lambda c: int(c["id"]))
    customer_by_id = {int(c["id"]): c for c in customers}

    dist = instance["distance_matrix"]

    earliest = [0.0] * (R + 1)
    latest = [0.0] * (R + 1)
    service = [0.0] * (R + 1)
    demand = [0.0] * (R + 1)

    earliest[0] = float(depot["time_window"][0])
    latest[0] = float(depot["time_window"][1])
    service[0] = float(depot.get("service_time", 0))

    earliest[R] = earliest[0]
    latest[R] = latest[0]
    service[R] = 0.0

    for c in customers:
        i = int(c["id"])
        earliest[i] = float(c["time_window"][0])
        latest[i] = float(c["time_window"][1])
        service[i] = float(c["service_time"])
        demand[i] = float(c["demand"])

    total_demand = sum(demand[1:n + 1])
    lower_bound_vehicles = int(math.ceil((total_demand - 1e-9) / Q))
    natural_upper_bound = sum(
        int(math.ceil((demand[i] - 1e-9) / Q)) for i in range(1, n + 1)
    )
    K = max(declared_k, natural_upper_bound)

    def matrix_node(node):
        return 0 if node == R else node

    def distance(i, j):
        return float(dist[matrix_node(i)][matrix_node(j)])

    def route_cost(sequence):
        previous = 0
        value = 0.0
        for node in sequence:
            value += distance(previous, node)
            previous = node
        value += distance(previous, R)
        return value

    def schedule_route(sequence):
        current_node = 0
        current_time = earliest[0]
        times = [current_time]

        for node in sequence:
            current_time = max(
                earliest[node],
                current_time + service[current_node] + distance(current_node, node),
            )
            if current_time > latest[node] + 1e-7:
                return None
            times.append(current_time)
            current_node = node

        return_time = max(
            earliest[R],
            current_time + service[current_node] + distance(current_node, R),
        )
        if return_time > latest[R] + 1e-7:
            return None
        times.append(return_time)
        return times

    def make_solution(route_data):
        route_data = [r for r in route_data if r["seq"]]
        output_routes = []
        totals = {i: 0.0 for i in range(1, n + 1)}
        objective = 0.0

        for vehicle_index, r in enumerate(route_data, start=1):
            seq = [int(x) for x in r["seq"]]
            objective += route_cost(seq)

            delivered = {}
            for i in seq:
                value = float(r["del"].get(i, 0.0))
                if abs(value) < 1e-10:
                    value = 0.0
                delivered[str(i)] = value
                totals[i] += value

            output_routes.append(
                {
                    "vehicle": vehicle_index,
                    "route": [0] + seq + [R],
                    "deliveries": delivered,
                }
            )

        delivery_summary = {}
        for i in range(1, n + 1):
            original = customer_by_id[i]["demand"]
            delivery_summary[str(i)] = {
                "demand": original,
                "total_delivered": float(totals[i]),
            }

        return {
            "objective_value": float(objective),
            "num_vehicles_used": len(output_routes),
            "routes": output_routes,
            "deliveries": delivery_summary,
        }

    def validate_route_data(route_data):
        totals = [0.0] * (n + 1)

        if len(route_data) < lower_bound_vehicles or len(route_data) > K:
            return False

        for r in route_data:
            seq = r["seq"]
            if not seq or len(seq) != len(set(seq)):
                return False
            if schedule_route(seq) is None:
                return False

            load = 0.0
            for i in seq:
                if i < 1 or i > n:
                    return False
                quantity = float(r["del"].get(i, 0.0))
                if quantity < -1e-5 or quantity > min(demand[i], Q) + 1e-5:
                    return False
                totals[i] += quantity
                load += quantity

            if load > Q + 1e-5:
                return False

        for i in range(1, n + 1):
            if totals[i] < demand[i] - 1e-4:
                return False
        return True

    # Always construct a quick feasible incumbent from single-customer routes.
    heuristic_routes = []
    for i in range(1, n + 1):
        remaining = demand[i]
        while remaining > 1e-9:
            quantity = min(Q, remaining)
            if schedule_route([i]) is None:
                raise RuntimeError(
                    f"Customer {i} cannot be served by a direct depot route"
                )
            heuristic_routes.append(
                {"seq": [i], "del": {i: float(quantity)}, "load": float(quantity)}
            )
            remaining -= quantity

    if len(heuristic_routes) > K:
        raise RuntimeError("The available fleet is insufficient for the instance")

    best_routes = heuristic_routes
    best_solution = make_solution(best_routes)
    best_objective = best_solution["objective_value"]
    logged_objective = float("inf")

    if logger:
        logger.log_solution(best_objective, best_solution)
        logged_objective = best_objective

    # Greedy route merging gives Gurobi a significantly stronger warm start.
    heuristic_budget = min(
        4.0,
        max(0.0, args.time_limit * 0.12),
        max(0.0, deadline - time.monotonic() - 0.25),
    )
    heuristic_deadline = time.monotonic() + heuristic_budget

    while len(best_routes) > lower_bound_vehicles and time.monotonic() < heuristic_deadline:
        best_merge = None
        best_saving = 1e-8
        route_count = len(best_routes)

        interrupted = False
        for a in range(route_count):
            if time.monotonic() >= heuristic_deadline:
                interrupted = True
                break
            ra = best_routes[a]
            set_a = set(ra["seq"])

            for b in range(a + 1, route_count):
                if time.monotonic() >= heuristic_deadline:
                    interrupted = True
                    break

                rb = best_routes[b]
                if ra["load"] + rb["load"] > Q + 1e-9:
                    continue
                if set_a.intersection(rb["seq"]):
                    continue

                base_cost = route_cost(ra["seq"]) + route_cost(rb["seq"])
                orientations_a = [ra["seq"]]
                orientations_b = [rb["seq"]]
                if len(ra["seq"]) > 1:
                    orientations_a.append(list(reversed(ra["seq"])))
                if len(rb["seq"]) > 1:
                    orientations_b.append(list(reversed(rb["seq"])))

                for sa in orientations_a:
                    for sb in orientations_b:
                        for candidate in (sa + sb, sb + sa):
                            if schedule_route(candidate) is None:
                                continue
                            candidate_cost = route_cost(candidate)
                            saving = base_cost - candidate_cost
                            if saving > best_saving + 1e-9:
                                best_saving = saving
                                best_merge = (a, b, candidate)

            if interrupted:
                break

        if best_merge is None:
            break

        a, b, merged_sequence = best_merge
        merged_deliveries = dict(best_routes[a]["del"])
        merged_deliveries.update(best_routes[b]["del"])
        merged = {
            "seq": list(merged_sequence),
            "del": merged_deliveries,
            "load": best_routes[a]["load"] + best_routes[b]["load"],
        }

        new_routes = [
            r for idx, r in enumerate(best_routes) if idx != a and idx != b
        ]
        new_routes.append(merged)
        best_routes = new_routes

        candidate_solution = make_solution(best_routes)
        candidate_objective = candidate_solution["objective_value"]
        if candidate_objective < best_objective - 1e-8:
            best_objective = candidate_objective
            best_solution = candidate_solution
            if logger:
                logger.log_solution(best_objective, best_solution)
                logged_objective = best_objective

    # Build and solve a vehicle-indexed MIP when its estimated size is reasonable.
    if gp is not None and time.monotonic() < deadline - 0.2:
        try:
            allowed_arcs = []

            for j in range(1, n + 1):
                if earliest[0] + service[0] + distance(0, j) <= latest[j] + 1e-9:
                    allowed_arcs.append((0, j))

            for i in range(1, n + 1):
                if earliest[i] + service[i] + distance(i, R) <= latest[R] + 1e-9:
                    allowed_arcs.append((i, R))

                for j in range(1, n + 1):
                    if i == j:
                        continue
                    if (
                        earliest[i] + service[i] + distance(i, j)
                        <= latest[j] + 1e-9
                    ):
                        allowed_arcs.append((i, j))

            incoming = {i: [] for i in range(1, n + 1)}
            outgoing = {i: [] for i in range(1, n + 1)}
            depot_outgoing = []
            return_incoming = []

            for i, j in allowed_arcs:
                if i == 0:
                    depot_outgoing.append((i, j))
                else:
                    outgoing[i].append((i, j))
                if j == R:
                    return_incoming.append((i, j))
                else:
                    incoming[j].append((i, j))

            # Full fleet indexing is useful on moderate instances. On very large
            # instances, use the incumbent route count to control memory.
            estimated_full_binaries = K * (len(allowed_arcs) + n + 1)
            slots = K if estimated_full_binaries <= 800000 else len(best_routes)
            slots = max(slots, len(best_routes), lower_bound_vehicles)
            slots = min(slots, K)

            estimated_variables = slots * (len(allowed_arcs) + 3 * n + 3)
            if estimated_variables <= 1400000 and time.monotonic() < deadline - 0.2:
                model = gp.Model("split_delivery_vrptw")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1

                vehicles = range(slots)
                customer_ids = range(1, n + 1)

                y = model.addVars(vehicles, vtype=GRB.BINARY, name="used")
                v = model.addVars(
                    [(k, i) for k in vehicles for i in customer_ids],
                    vtype=GRB.BINARY,
                    name="visit",
                )
                x = model.addVars(
                    [
                        (k, i, j)
                        for k in vehicles
                        for (i, j) in allowed_arcs
                    ],
                    vtype=GRB.BINARY,
                    name="arc",
                )
                q = model.addVars(
                    [(k, i) for k in vehicles for i in customer_ids],
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name="quantity",
                )
                t = model.addVars(
                    [(k, i) for k in vehicles for i in range(R + 1)],
                    lb={
                        (k, i): earliest[i]
                        for k in vehicles
                        for i in range(R + 1)
                    },
                    ub={
                        (k, i): latest[i]
                        for k in vehicles
                        for i in range(R + 1)
                    },
                    vtype=GRB.CONTINUOUS,
                    name="time",
                )
                order = model.addVars(
                    [(k, i) for k in vehicles for i in customer_ids],
                    lb=0.0,
                    ub=float(n),
                    vtype=GRB.CONTINUOUS,
                    name="order",
                )

                for k in vehicles:
                    model.addConstr(
                        gp.quicksum(x[k, i, j] for i, j in depot_outgoing) == y[k]
                    )
                    model.addConstr(
                        gp.quicksum(x[k, i, j] for i, j in return_incoming) == y[k]
                    )

                    for i in customer_ids:
                        model.addConstr(
                            gp.quicksum(x[k, a, b] for a, b in incoming[i])
                            == v[k, i]
                        )
                        model.addConstr(
                            gp.quicksum(x[k, a, b] for a, b in outgoing[i])
                            == v[k, i]
                        )
                        model.addConstr(q[k, i] <= min(demand[i], Q) * v[k, i])
                        model.addConstr(order[k, i] >= v[k, i])
                        model.addConstr(order[k, i] <= n * v[k, i])

                    model.addConstr(
                        gp.quicksum(q[k, i] for i in customer_ids) <= Q * y[k]
                    )

                for i in customer_ids:
                    model.addConstr(
                        gp.quicksum(q[k, i] for k in vehicles) == demand[i]
                    )

                model.addConstr(
                    gp.quicksum(y[k] for k in vehicles) >= lower_bound_vehicles
                )

                for k in range(slots - 1):
                    model.addConstr(y[k] >= y[k + 1])

                for k in vehicles:
                    for i, j in allowed_arcs:
                        tau = service[i] + distance(i, j)
                        big_m = max(0.0, latest[i] + tau - earliest[j])
                        model.addConstr(
                            t[k, j] >= t[k, i] + tau - big_m * (1 - x[k, i, j])
                        )

                        if 1 <= i <= n and 1 <= j <= n:
                            model.addConstr(
                                order[k, j]
                                >= order[k, i] + 1.0 - n * (1 - x[k, i, j])
                            )

                model.setObjective(
                    gp.quicksum(
                        distance(i, j) * x[k, i, j]
                        for k in vehicles
                        for i, j in allowed_arcs
                    ),
                    GRB.MINIMIZE,
                )

                # MIP start from the best greedy solution.
                for k in vehicles:
                    y[k].Start = 0.0
                    for i in customer_ids:
                        v[k, i].Start = 0.0
                        q[k, i].Start = 0.0
                        order[k, i].Start = 0.0
                    for i in range(R + 1):
                        t[k, i].Start = earliest[i]
                    for i, j in allowed_arcs:
                        x[k, i, j].Start = 0.0

                arc_set = set(allowed_arcs)
                for k, route in enumerate(best_routes):
                    y[k].Start = 1.0
                    seq = route["seq"]
                    times = schedule_route(seq)
                    previous = 0

                    for position, node in enumerate(seq, start=1):
                        v[k, node].Start = 1.0
                        q[k, node].Start = float(route["del"].get(node, 0.0))
                        order[k, node].Start = float(position)
                        t[k, node].Start = times[position]
                        if (previous, node) in arc_set:
                            x[k, previous, node].Start = 1.0
                        previous = node

                    if (previous, R) in arc_set:
                        x[k, previous, R].Start = 1.0
                    t[k, 0].Start = times[0]
                    t[k, R].Start = times[-1]

                x_keys = list(x.keys())
                x_vars = [x[key] for key in x_keys]
                q_keys = list(q.keys())
                q_vars = [q[key] for key in q_keys]
                y_vars = [y[k] for k in vehicles]

                callback_logged = [logged_objective]

                def raw_values_to_routes(y_values, x_values, q_values):
                    selected = {
                        key: value for key, value in zip(x_keys, x_values)
                        if value > 0.5
                    }
                    quantities = {
                        key: max(0.0, float(value))
                        for key, value in zip(q_keys, q_values)
                    }

                    routes = []
                    for k in vehicles:
                        if y_values[k] <= 0.5:
                            continue

                        successors = {}
                        for kk, i, j in selected:
                            if kk == k:
                                successors[i] = j

                        seq = []
                        current = 0
                        seen = set()
                        valid = True

                        while current != R:
                            if current not in successors:
                                valid = False
                                break
                            nxt = successors[current]
                            if nxt != R:
                                if nxt in seen:
                                    valid = False
                                    break
                                seen.add(nxt)
                                seq.append(nxt)
                            current = nxt
                            if len(seq) > n:
                                valid = False
                                break

                        if not valid or not seq:
                            return None

                        deliveries = {
                            i: quantities.get((k, i), 0.0) for i in seq
                        }
                        routes.append(
                            {
                                "seq": seq,
                                "del": deliveries,
                                "load": sum(deliveries.values()),
                            }
                        )
                    return routes

                def mip_callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL or logger is None:
                        return
                    try:
                        incumbent_obj = cb_model.cbGet(
                            GRB.Callback.MIPSOL_OBJ
                        )
                        if incumbent_obj >= callback_logged[0] - 1e-7:
                            return

                        y_values = cb_model.cbGetSolution(y_vars)
                        x_values = cb_model.cbGetSolution(x_vars)
                        q_values = cb_model.cbGetSolution(q_vars)
                        routes = raw_values_to_routes(
                            y_values, x_values, q_values
                        )
                        if routes is None or not validate_route_data(routes):
                            return

                        solution = make_solution(routes)
                        objective = solution["objective_value"]
                        if objective < callback_logged[0] - 1e-7:
                            logger.log_solution(objective, solution)
                            callback_logged[0] = objective
                    except Exception:
                        pass

                remaining = max(0.01, deadline - time.monotonic())
                model.Params.TimeLimit = remaining
                model.optimize(mip_callback)

                if model.SolCount > 0:
                    y_values = [y[k].X for k in vehicles]
                    x_values = [var.X for var in x_vars]
                    q_values = [var.X for var in q_vars]
                    mip_routes = raw_values_to_routes(y_values, x_values, q_values)

                    if mip_routes is not None and validate_route_data(mip_routes):
                        mip_solution = make_solution(mip_routes)
                        mip_objective = mip_solution["objective_value"]
                        if mip_objective < best_objective - 1e-7:
                            best_routes = mip_routes
                            best_solution = mip_solution
                            best_objective = mip_objective
                            if logger and mip_objective < callback_logged[0] - 1e-7:
                                logger.log_solution(mip_objective, mip_solution)
                                callback_logged[0] = mip_objective

                model.dispose()

        except Exception:
            # The constructive incumbent remains valid if model creation,
            # licensing, numerical processing, or optimization fails.
            pass

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()