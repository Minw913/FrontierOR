import argparse
import json
import math
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ProblemData:
    def __init__(self, instance):
        self.instance = instance
        self.n = int(instance["num_customers"])
        self.num_vehicles = int(instance["num_vehicles"])
        self.capacity = float(instance["vehicle_capacity"])
        self.distance = instance["distance_matrix"]

        self.start = int(instance["depot"]["id"])
        self.end = int(instance["depot_end"]["id"])

        self.start_earliest = float(instance["depot"]["time_window"][0])
        self.start_latest = float(instance["depot"]["time_window"][1])
        self.end_earliest = float(instance["depot_end"]["time_window"][0])
        self.end_latest = float(instance["depot_end"]["time_window"][1])

        self.customers = sorted(
            [int(c["id"]) for c in instance["customers"]]
        )
        self.customer_data = {
            int(c["id"]): c for c in instance["customers"]
        }

        default_service = float(instance.get("service_time", 0.0))
        self.demand = {}
        self.earliest = {}
        self.latest = {}
        self.service = {}
        self.preferred = {}

        for i in self.customers:
            c = self.customer_data[i]
            self.demand[i] = float(c["demand"])
            self.earliest[i] = float(c["time_window"][0])
            self.latest[i] = float(c["time_window"][1])
            self.service[i] = float(c.get("service_time", default_service))
            self.preferred[i] = float(
                c.get(
                    "preferred_time",
                    0.5 * (
                        float(c["time_window"][0])
                        + float(c["time_window"][1])
                    ),
                )
            )

        self.rho = float(
            instance.get("inconvenience_cost_function", {}).get("rho", 0.0)
        )

    def dist(self, i, j):
        return float(self.distance[i][j])


class RouteEvaluator:
    """Evaluates and approximately optimizes the continuous schedule of a route."""

    def __init__(self, data):
        self.data = data
        self.cache = {}

    def evaluate(self, route):
        key = tuple(route)
        if key in self.cache:
            return self.cache[key]

        d = self.data
        if not route:
            result = (0.0, {}, 0.0)
            self.cache[key] = result
            return result

        load = sum(d.demand[i] for i in route)
        if load > d.capacity + EPS:
            self.cache[key] = None
            return None

        times = []
        previous = d.start
        previous_time = d.start_earliest
        previous_service = 0.0

        for i in route:
            arrival = previous_time + previous_service + d.dist(previous, i)
            ti = max(d.earliest[i], arrival)
            if ti > d.latest[i] + EPS:
                self.cache[key] = None
                return None
            times.append(ti)
            previous = i
            previous_time = ti
            previous_service = d.service[i]

        if (
            times[-1]
            + d.service[route[-1]]
            + d.dist(route[-1], d.end)
            > d.end_latest + EPS
        ):
            self.cache[key] = None
            return None

        # Coordinate descent on the convex fixed-route scheduling problem.
        # Every update preserves all time-window and precedence constraints.
        for _ in range(80):
            max_change = 0.0

            for direction in (range(len(route)), range(len(route) - 1, -1, -1)):
                for k in direction:
                    i = route[k]

                    if k == 0:
                        lower = max(
                            d.earliest[i],
                            d.start_earliest + d.dist(d.start, i),
                        )
                    else:
                        p = route[k - 1]
                        lower = max(
                            d.earliest[i],
                            times[k - 1] + d.service[p] + d.dist(p, i),
                        )

                    if k == len(route) - 1:
                        upper = min(
                            d.latest[i],
                            d.end_latest - d.service[i] - d.dist(i, d.end),
                        )
                    else:
                        j = route[k + 1]
                        upper = min(
                            d.latest[i],
                            times[k + 1] - d.service[i] - d.dist(i, j),
                        )

                    if lower > upper + 1e-7:
                        self.cache[key] = None
                        return None

                    new_time = min(max(d.preferred[i], lower), upper)
                    max_change = max(max_change, abs(new_time - times[k]))
                    times[k] = new_time

            if max_change < 1e-9:
                break

        routing_cost = d.dist(d.start, route[0])
        for a, b in zip(route, route[1:]):
            routing_cost += d.dist(a, b)
        routing_cost += d.dist(route[-1], d.end)

        inconvenience = sum(
            d.rho * (times[k] - d.preferred[i]) ** 2
            for k, i in enumerate(route)
        )
        schedule = {i: times[k] for k, i in enumerate(route)}
        result = (routing_cost + inconvenience, schedule, load)
        self.cache[key] = result
        return result


def routes_to_solution(data, routes, schedule=None):
    evaluator = RouteEvaluator(data)
    full_schedule = {}
    output_routes = []
    objective = 0.0

    for route in routes:
        if not route:
            continue
        output_routes.append([data.start] + list(route) + [data.end])

        routing = data.dist(data.start, route[0])
        for a, b in zip(route, route[1:]):
            routing += data.dist(a, b)
        routing += data.dist(route[-1], data.end)
        objective += routing

        if schedule is None:
            evaluated = evaluator.evaluate(route)
            if evaluated is None:
                raise ValueError("Attempted to output an infeasible route")
            _, route_schedule, _ = evaluated
        else:
            route_schedule = {i: float(schedule[i]) for i in route}

        for i in route:
            ti = float(route_schedule[i])
            full_schedule[str(i)] = ti
            objective += data.rho * (ti - data.preferred[i]) ** 2

    return {
        "objective_value": float(objective),
        "routes": output_routes,
        "schedule": full_schedule,
    }


def construct_heuristic(data, evaluator, deadline):
    rng = random.Random(0)
    customers = data.customers
    best_routes = None
    best_value = math.inf

    deterministic_orders = [
        sorted(customers, key=lambda i: (data.earliest[i], data.latest[i])),
        sorted(customers, key=lambda i: (data.latest[i], data.earliest[i])),
        sorted(
            customers,
            key=lambda i: (
                data.latest[i] - data.earliest[i],
                data.latest[i],
            ),
        ),
        sorted(customers, key=lambda i: (-data.demand[i], data.latest[i])),
        sorted(
            customers,
            key=lambda i: math.atan2(
                data.customer_data[i]["y"] - data.instance["depot"]["y"],
                data.customer_data[i]["x"] - data.instance["depot"]["x"],
            ),
        ),
    ]

    attempts = 0
    while time.monotonic() < deadline and attempts < 40:
        if attempts < len(deterministic_orders):
            order = deterministic_orders[attempts][:]
        else:
            order = customers[:]
            rng.shuffle(order)

        routes = []
        route_values = []
        feasible = True

        for customer in order:
            if time.monotonic() >= deadline:
                feasible = False
                break

            candidates = []

            for r_idx, route in enumerate(routes):
                old_value = route_values[r_idx]
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    result = evaluator.evaluate(new_route)
                    if result is not None:
                        delta = result[0] - old_value
                        noise = 1e-8 * rng.random()
                        candidates.append(
                            (delta + noise, r_idx, pos, result[0], new_route)
                        )

            if len(routes) < data.num_vehicles:
                result = evaluator.evaluate([customer])
                if result is not None:
                    candidates.append(
                        (result[0] + 1e-8 * rng.random(),
                         len(routes), 0, result[0], [customer])
                    )

            if not candidates:
                feasible = False
                break

            _, r_idx, _, new_value, new_route = min(candidates, key=lambda x: x[0])
            if r_idx == len(routes):
                routes.append(new_route)
                route_values.append(new_value)
            else:
                routes[r_idx] = new_route
                route_values[r_idx] = new_value

        if feasible and len(routes) <= data.num_vehicles:
            value = sum(route_values)
            if value < best_value - 1e-8:
                best_value = value
                best_routes = [r[:] for r in routes]

        attempts += 1

    if best_routes is None:
        return None

    # Bounded local search: relocations, swaps, and intra-route reversals.
    routes = [r[:] for r in best_routes]
    values = [evaluator.evaluate(r)[0] for r in routes]
    current_value = sum(values)
    iterations = 0

    while time.monotonic() < deadline and iterations < 5000:
        iterations += 1
        move_type = rng.randrange(3)
        accepted = False

        if move_type == 0 and len(routes) >= 2:
            a, b = rng.sample(range(len(routes)), 2)
            if not routes[a]:
                continue
            pa = rng.randrange(len(routes[a]))
            pb = rng.randrange(len(routes[b]) + 1)
            customer = routes[a][pa]
            new_a = routes[a][:pa] + routes[a][pa + 1:]
            new_b = routes[b][:pb] + [customer] + routes[b][pb:]

            eval_a = evaluator.evaluate(new_a)
            eval_b = evaluator.evaluate(new_b)
            if eval_a is not None and eval_b is not None:
                new_pair = eval_a[0] + eval_b[0]
                old_pair = values[a] + values[b]
                if new_pair < old_pair - 1e-8:
                    routes[a], routes[b] = new_a, new_b
                    values[a], values[b] = eval_a[0], eval_b[0]
                    current_value += new_pair - old_pair
                    accepted = True

        elif move_type == 1 and len(routes) >= 2:
            a, b = rng.sample(range(len(routes)), 2)
            if not routes[a] or not routes[b]:
                continue
            pa = rng.randrange(len(routes[a]))
            pb = rng.randrange(len(routes[b]))
            new_a = routes[a][:]
            new_b = routes[b][:]
            new_a[pa], new_b[pb] = new_b[pb], new_a[pa]

            eval_a = evaluator.evaluate(new_a)
            eval_b = evaluator.evaluate(new_b)
            if eval_a is not None and eval_b is not None:
                new_pair = eval_a[0] + eval_b[0]
                old_pair = values[a] + values[b]
                if new_pair < old_pair - 1e-8:
                    routes[a], routes[b] = new_a, new_b
                    values[a], values[b] = eval_a[0], eval_b[0]
                    current_value += new_pair - old_pair
                    accepted = True

        elif routes:
            a = rng.randrange(len(routes))
            if len(routes[a]) >= 3:
                p = rng.randrange(len(routes[a]) - 1)
                q = rng.randrange(p + 1, len(routes[a]))
                new_a = (
                    routes[a][:p]
                    + list(reversed(routes[a][p:q + 1]))
                    + routes[a][q + 1:]
                )
                eval_a = evaluator.evaluate(new_a)
                if eval_a is not None and eval_a[0] < values[a] - 1e-8:
                    current_value += eval_a[0] - values[a]
                    routes[a] = new_a
                    values[a] = eval_a[0]
                    accepted = True

        if accepted:
            empty = [idx for idx, r in enumerate(routes) if not r]
            for idx in reversed(empty):
                del routes[idx]
                del values[idx]

            if current_value < best_value - 1e-8:
                best_value = current_value
                best_routes = [r[:] for r in routes]

    return best_routes


def build_model(data, warm_routes, remaining_time):
    model = gp.Model("quadratic_vrptw")

    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, remaining_time)

    customers = data.customers
    start = data.start
    end = data.end

    arcs = []

    for j in customers:
        earliest_arrival = data.start_earliest + data.dist(start, j)
        if (
            earliest_arrival <= data.latest[j] + EPS
            and data.demand[j] <= data.capacity + EPS
        ):
            arcs.append((start, j))

    for i in customers:
        for j in customers:
            if i == j:
                continue
            if (
                data.earliest[i]
                + data.service[i]
                + data.dist(i, j)
                <= data.latest[j] + EPS
            ):
                arcs.append((i, j))

        if (
            data.earliest[i]
            + data.service[i]
            + data.dist(i, end)
            <= data.end_latest + EPS
        ):
            arcs.append((i, end))

    x = {
        arc: model.addVar(vtype=GRB.BINARY, name=f"x_{arc[0]}_{arc[1]}")
        for arc in arcs
    }

    t = {
        i: model.addVar(
            lb=data.earliest[i],
            ub=data.latest[i],
            vtype=GRB.CONTINUOUS,
            name=f"t_{i}",
        )
        for i in customers
    }

    load = {
        i: model.addVar(
            lb=data.demand[i],
            ub=data.capacity,
            vtype=GRB.CONTINUOUS,
            name=f"load_{i}",
        )
        for i in customers
    }

    order = {
        i: model.addVar(
            lb=1.0,
            ub=float(max(1, data.n)),
            vtype=GRB.CONTINUOUS,
            name=f"ord_{i}",
        )
        for i in customers
    }

    incoming = {i: [] for i in customers}
    outgoing = {i: [] for i in customers}
    start_arcs = []
    end_arcs = []

    for (i, j), var in x.items():
        if j in incoming:
            incoming[j].append(var)
        if i in outgoing:
            outgoing[i].append(var)
        if i == start:
            start_arcs.append(var)
        if j == end:
            end_arcs.append(var)

    for i in customers:
        model.addConstr(gp.quicksum(incoming[i]) == 1, name=f"in_{i}")
        model.addConstr(gp.quicksum(outgoing[i]) == 1, name=f"out_{i}")

    model.addConstr(
        gp.quicksum(start_arcs) == gp.quicksum(end_arcs),
        name="route_balance",
    )
    model.addConstr(
        gp.quicksum(start_arcs) <= data.num_vehicles,
        name="vehicle_limit",
    )

    total_demand = sum(data.demand[i] for i in customers)
    if data.capacity > EPS:
        demand_lb = int(math.ceil((total_demand - 1e-9) / data.capacity))
        model.addConstr(
            gp.quicksum(start_arcs) >= demand_lb,
            name="capacity_route_lower_bound",
        )

    for (i, j), var in x.items():
        if i == start and j in t:
            rhs = data.start_earliest + data.dist(start, j)
            big_m = max(0.0, rhs - data.earliest[j])
            model.addConstr(
                t[j] >= rhs - big_m * (1 - var),
                name=f"time_start_{j}",
            )

        elif i in t and j in t:
            travel_service = data.service[i] + data.dist(i, j)
            big_m = max(
                0.0,
                data.latest[i] + travel_service - data.earliest[j],
            )
            model.addConstr(
                t[j] >= t[i] + travel_service - big_m * (1 - var),
                name=f"time_{i}_{j}",
            )

            model.addConstr(
                load[j]
                >= load[i] + data.demand[j] - data.capacity * (1 - var),
                name=f"capacity_{i}_{j}",
            )

            model.addConstr(
                order[j] >= order[i] + 1 - data.n * (1 - var),
                name=f"order_{i}_{j}",
            )

        elif i in t and j == end:
            completion = data.service[i] + data.dist(i, end)
            big_m = max(
                0.0,
                data.latest[i] + completion - data.end_latest,
            )
            model.addConstr(
                t[i] + completion
                <= data.end_latest + big_m * (1 - var),
                name=f"time_end_{i}",
            )

    routing_objective = gp.quicksum(
        data.dist(i, j) * var for (i, j), var in x.items()
    )
    inconvenience_objective = gp.quicksum(
        data.rho * (t[i] - data.preferred[i]) * (t[i] - data.preferred[i])
        for i in customers
    )
    model.setObjective(
        routing_objective + inconvenience_objective,
        GRB.MINIMIZE,
    )

    if warm_routes:
        evaluator = RouteEvaluator(data)
        selected_arcs = set()
        warm_schedule = {}

        for route in warm_routes:
            evaluated = evaluator.evaluate(route)
            if evaluated is None:
                continue
            _, route_schedule, _ = evaluated
            warm_schedule.update(route_schedule)

            previous = start
            cumulative_load = 0.0
            for position, i in enumerate(route, start=1):
                selected_arcs.add((previous, i))
                cumulative_load += data.demand[i]
                load[i].Start = cumulative_load
                order[i].Start = position
                t[i].Start = route_schedule[i]
                previous = i
            selected_arcs.add((previous, end))

        for arc, var in x.items():
            var.Start = 1.0 if arc in selected_arcs else 0.0

    model.update()
    return model, x, t


def extract_solution(data, x_values, t_values):
    successor = {}
    for (i, j), value in x_values.items():
        if value > 0.5:
            successor[i] = j

    starts = sorted(
        j for (i, j), value in x_values.items()
        if i == data.start and value > 0.5
    )

    routes = []
    visited = set()

    for first in starts:
        route = []
        current = first
        steps = 0

        while current != data.end and steps <= data.n:
            if current in visited or current not in data.customer_data:
                return None
            visited.add(current)
            route.append(current)

            if current not in successor:
                return None
            current = successor[current]
            steps += 1

        if current != data.end:
            return None
        routes.append(route)

    if visited != set(data.customers):
        return None

    schedule = {i: float(t_values[i]) for i in data.customers}
    return routes_to_solution(data, routes, schedule)


def main():
    args = parse_args()
    start_clock = time.monotonic()

    logger = SolutionLogger(
        args.log_path, sense="minimize"
    ) if args.log_path else None

    instance = load_instance(args.instance_path)
    data = ProblemData(instance)
    evaluator = RouteEvaluator(data)

    total_limit = max(0, args.time_limit)
    heuristic_budget = min(
        5.0,
        max(0.05, 0.18 * total_limit),
    )
    heuristic_deadline = min(
        start_clock + heuristic_budget,
        start_clock + max(0.0, total_limit - 0.05),
    )

    warm_routes = construct_heuristic(data, evaluator, heuristic_deadline)
    best_solution = None
    best_objective = math.inf

    if warm_routes is not None:
        best_solution = routes_to_solution(data, warm_routes)
        best_objective = best_solution["objective_value"]
        if logger:
            logger.log_solution(best_objective, best_solution)

    elapsed = time.monotonic() - start_clock
    remaining = max(0.01, total_limit - elapsed)

    model, x, t = build_model(data, warm_routes, remaining)
    x_items = list(x.items())
    x_vars = [var for _, var in x_items]
    t_items = list(t.items())
    t_vars = [var for _, var in t_items]

    callback_state = {
        "best": best_objective,
        "solution": best_solution,
    }

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            x_vals_raw = cb_model.cbGetSolution(x_vars)
            t_vals_raw = cb_model.cbGetSolution(t_vars)
            x_values = {
                x_items[k][0]: float(x_vals_raw[k])
                for k in range(len(x_items))
            }
            t_values = {
                t_items[k][0]: float(t_vals_raw[k])
                for k in range(len(t_items))
            }

            solution = extract_solution(data, x_values, t_values)
            if solution is None:
                return

            objective = solution["objective_value"]
            if objective < callback_state["best"] - 1e-7:
                callback_state["best"] = objective
                callback_state["solution"] = solution
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            # Logging or incumbent extraction must never interrupt optimization.
            pass

    try:
        model.optimize(callback)
    except gp.GurobiError:
        pass

    if model.SolCount > 0:
        try:
            x_values = {arc: float(var.X) for arc, var in x.items()}
            t_values = {i: float(var.X) for i, var in t.items()}
            final_mip_solution = extract_solution(data, x_values, t_values)
            if (
                final_mip_solution is not None
                and final_mip_solution["objective_value"]
                < callback_state["best"] - 1e-7
            ):
                callback_state["best"] = final_mip_solution["objective_value"]
                callback_state["solution"] = final_mip_solution
                if logger:
                    logger.log_solution(
                        final_mip_solution["objective_value"],
                        final_mip_solution,
                    )
        except (gp.GurobiError, AttributeError):
            pass

    best_solution = callback_state["solution"]

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    # Reassign to ensure exact consistency between the reported objective
    # and the routes/schedule in the output.
    best_solution["objective_value"] = float(best_solution["objective_value"])

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()