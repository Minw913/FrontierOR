import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


EPS = 1e-7


def route_cost(sequence: List[int], distance: List[List[float]], start: int, end: int) -> float:
    if not sequence:
        return 0.0
    total = distance[start][sequence[0]]
    for a, b in zip(sequence, sequence[1:]):
        total += distance[a][b]
    total += distance[sequence[-1]][end]
    return float(total)


def schedule_route(
    sequence: List[int],
    window_starts: Dict[int, float],
    widths: Dict[int, float],
    distance: List[List[float]],
    depot_window: Tuple[float, float],
    start: int,
    end: int,
) -> Optional[Dict[int, float]]:
    """Construct the earliest feasible service schedule for a fixed route."""
    depot_open, depot_close = depot_window
    current_node = start
    current_time = depot_open
    times = {start: float(current_time)}

    for customer in sequence:
        current_time = max(
            current_time + distance[current_node][customer],
            window_starts[customer],
        )
        if current_time > window_starts[customer] + widths[customer] + 1e-6:
            return None
        times[customer] = float(current_time)
        current_node = customer

    current_time += distance[current_node][end]
    if current_time > depot_close + 1e-6:
        return None
    times[end] = float(max(current_time, depot_open))
    return times


def build_solution(
    scenario_sequences: List[List[List[int]]],
    window_starts: Dict[int, float],
    widths: Dict[int, float],
    scenarios: List[dict],
    distance: List[List[float]],
    depot_window: Tuple[float, float],
    probabilities: List[float],
    start: int,
    end: int,
) -> Optional[dict]:
    routes_output = {}
    expected_cost = 0.0

    for s_idx, scenario in enumerate(scenarios):
        scenario_id = str(scenario["scenario_id"])
        output_routes = []
        scenario_cost = 0.0

        for sequence in scenario_sequences[s_idx]:
            times = schedule_route(
                sequence,
                window_starts,
                widths,
                distance,
                depot_window,
                start,
                end,
            )
            if times is None:
                return None

            cost = route_cost(sequence, distance, start, end)
            scenario_cost += cost
            full_route = [start] + sequence + [end]
            output_routes.append(
                {
                    "route": full_route,
                    "arrival_times": {
                        str(node): float(times[node]) for node in full_route
                    },
                    "cost": float(cost),
                }
            )

        routes_output[scenario_id] = output_routes
        expected_cost += probabilities[s_idx] * scenario_cost

    time_windows = {
        str(i): [
            float(window_starts[i]),
            float(window_starts[i] + widths[i]),
        ]
        for i in sorted(window_starts)
    }

    return {
        "objective_value": float(expected_cost),
        "time_windows": time_windows,
        "routes": routes_output,
    }


def singleton_sequences(num_scenarios: int, customers: List[int]) -> List[List[List[int]]]:
    return [[[i] for i in customers] for _ in range(num_scenarios)]


def clarke_wright_routes(
    customers: List[int],
    demands: Dict[int, int],
    capacity: int,
    window_starts: Dict[int, float],
    widths: Dict[int, float],
    distance: List[List[float]],
    depot_window: Tuple[float, float],
    start: int,
    end: int,
    deadline: float,
) -> List[List[int]]:
    """
    Directed Clarke-Wright merging heuristic. A merge is accepted only if the
    resulting route satisfies capacity and the fixed endogenous time windows.
    """
    routes: Dict[int, List[int]] = {i: [i] for i in customers}
    route_demand: Dict[int, int] = {i: demands[i] for i in customers}
    owner: Dict[int, int] = {i: i for i in customers}

    savings = []
    for i in customers:
        for j in customers:
            if i == j:
                continue
            saving = (
                distance[i][end]
                + distance[start][j]
                - distance[i][j]
            )
            if saving >= -EPS:
                savings.append((saving, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    for position, (_, i, j) in enumerate(savings):
        if position % 128 == 0 and time.monotonic() >= deadline:
            break

        ri = owner[i]
        rj = owner[j]
        if ri == rj or ri not in routes or rj not in routes:
            continue
        if routes[ri][-1] != i or routes[rj][0] != j:
            continue
        if route_demand[ri] + route_demand[rj] > capacity:
            continue

        merged = routes[ri] + routes[rj]
        if schedule_route(
            merged,
            window_starts,
            widths,
            distance,
            depot_window,
            start,
            end,
        ) is None:
            continue

        routes[ri] = merged
        route_demand[ri] += route_demand[rj]
        for customer in routes[rj]:
            owner[customer] = ri
        del routes[rj]
        del route_demand[rj]

    return list(routes.values())


def extract_sequences_from_arc_values(
    arc_values: Dict[Tuple[int, int, int], float],
    num_scenarios: int,
    customers: List[int],
    start: int,
    end: int,
) -> Optional[List[List[List[int]]]]:
    result: List[List[List[int]]] = []

    for s in range(num_scenarios):
        successors: Dict[int, List[int]] = {}
        for (ss, i, j), value in arc_values.items():
            if ss == s and value > 0.5:
                successors.setdefault(i, []).append(j)

        route_starts = sorted(successors.get(start, []))
        sequences = []
        visited = set()

        for first in route_starts:
            sequence = []
            node = first
            guard = 0
            while node != end:
                guard += 1
                if guard > len(customers) + 1:
                    return None
                if node in visited or node not in customers:
                    return None
                visited.add(node)
                sequence.append(node)

                next_nodes = successors.get(node, [])
                if len(next_nodes) != 1:
                    return None
                node = next_nodes[0]
            sequences.append(sequence)

        if visited != set(customers):
            return None
        result.append(sequences)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_wall = time.monotonic()
    absolute_deadline = start_wall + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["num_customers"])
    start_depot = 0
    end_depot = n + 1
    customers = sorted(int(c["id"]) for c in instance["customers"])
    customer_set = set(customers)

    if customer_set != set(range(1, n + 1)):
        raise ValueError("Customer IDs must correspond to matrix indices 1..num_customers")

    distance = [[float(v) for v in row] for row in instance["distance_matrix"]]
    capacity = int(instance["vehicle_capacity"])
    depot_window = (
        float(instance["depot"]["exogenous_time_window"][0]),
        float(instance["depot"]["exogenous_time_window"][1]),
    )

    customer_data = {int(c["id"]): c for c in instance["customers"]}
    exogenous = {
        i: (
            float(customer_data[i]["exogenous_time_window"][0]),
            float(customer_data[i]["exogenous_time_window"][1]),
        )
        for i in customers
    }
    global_width = float(instance.get("endogenous_time_window_width", 0.0))
    widths = {
        i: float(customer_data[i].get("endogenous_time_window_width", global_width))
        for i in customers
    }

    scenarios = instance["scenarios"]
    num_scenarios = len(scenarios)
    probabilities = [float(s["probability"]) for s in scenarios]

    scenario_demands: List[Dict[int, int]] = []
    for scenario in scenarios:
        demand_list = scenario["demands"]
        demands = {i: int(demand_list[i - 1]) for i in customers}
        scenario_demands.append(demands)

    # Feasible placement interval for each endogenous window, accounting for
    # direct singleton access from and back to the depot.
    placement_bounds: Dict[int, Tuple[float, float]] = {}
    depot_open, depot_close = depot_window
    for i in customers:
        a, b = exogenous[i]
        w = widths[i]
        earliest_direct_service = depot_open + distance[start_depot][i]
        latest_direct_service = depot_close - distance[i][end_depot]

        lower = max(a, earliest_direct_service - w)
        upper = min(b - w, latest_direct_service)
        if lower > upper + 1e-7:
            raise RuntimeError(
                f"No feasible endogenous window placement for customer {i}"
            )
        placement_bounds[i] = (lower, max(lower, upper))

    best_solution: Optional[dict] = None
    best_objective = math.inf
    best_window_starts: Optional[Dict[int, float]] = None
    best_sequences: Optional[List[List[List[int]]]] = None

    def register_solution(
        solution: Optional[dict],
        window_starts: Optional[Dict[int, float]] = None,
        sequences: Optional[List[List[List[int]]]] = None,
    ) -> None:
        nonlocal best_solution, best_objective, best_window_starts, best_sequences
        if solution is None:
            return
        objective = float(solution["objective_value"])
        if objective < best_objective - 1e-7:
            best_objective = objective
            best_solution = solution
            if window_starts is not None:
                best_window_starts = dict(window_starts)
            if sequences is not None:
                best_sequences = [[list(r) for r in sroutes] for sroutes in sequences]
            if logger:
                logger.log_solution(objective, solution)

    # Guaranteed initial incumbent: one vehicle per customer.
    initial_starts = {
        i: 0.5 * (placement_bounds[i][0] + placement_bounds[i][1])
        for i in customers
    }
    initial_sequences = singleton_sequences(num_scenarios, customers)
    initial_solution = build_solution(
        initial_sequences,
        initial_starts,
        widths,
        scenarios,
        distance,
        depot_window,
        probabilities,
        start_depot,
        end_depot,
    )
    register_solution(initial_solution, initial_starts, initial_sequences)

    # Generate several coordinated fixed-window heuristic solutions.
    heuristic_budget = min(5.0, max(0.2, 0.15 * max(1, args.time_limit)))
    heuristic_deadline = min(absolute_deadline, time.monotonic() + heuristic_budget)
    alphas = [0.0, 0.5, 1.0, 0.25, 0.75]

    for alpha in alphas:
        if time.monotonic() >= heuristic_deadline:
            break

        starts = {
            i: placement_bounds[i][0]
            + alpha * (placement_bounds[i][1] - placement_bounds[i][0])
            for i in customers
        }
        all_sequences: List[List[List[int]]] = []

        for s_idx in range(num_scenarios):
            if time.monotonic() >= heuristic_deadline:
                # Remaining scenarios retain valid singleton routes.
                all_sequences.extend(
                    singleton_sequences(num_scenarios - s_idx, customers)
                )
                break
            routes = clarke_wright_routes(
                customers,
                scenario_demands[s_idx],
                capacity,
                starts,
                widths,
                distance,
                depot_window,
                start_depot,
                end_depot,
                heuristic_deadline,
            )
            all_sequences.append(routes)

        if len(all_sequences) != num_scenarios:
            continue

        solution = build_solution(
            all_sequences,
            starts,
            widths,
            scenarios,
            distance,
            depot_window,
            probabilities,
            start_depot,
            end_depot,
        )
        register_solution(solution, starts, all_sequences)

    # Compact two-stage stochastic arc-flow MIP.
    remaining = absolute_deadline - time.monotonic()
    if gp is not None and remaining > 0.05:
        try:
            model = gp.Model("stochastic_vrptw_endogenous_windows")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining)

            node_lower = {start_depot: depot_open, end_depot: depot_open}
            node_upper = {start_depot: depot_close, end_depot: depot_close}
            for i in customers:
                node_lower[i] = exogenous[i][0]
                node_upper[i] = exogenous[i][1]

            base_arcs: List[Tuple[int, int]] = []
            for i in [start_depot] + customers:
                for j in customers + [end_depot]:
                    if i == j:
                        continue
                    if i == start_depot and j == end_depot:
                        continue
                    # Remove arcs impossible even under the broad exogenous windows.
                    if node_lower[i] + distance[i][j] > node_upper[j] + 1e-7:
                        continue
                    base_arcs.append((i, j))

            outgoing = {i: [] for i in [start_depot] + customers}
            incoming = {j: [] for j in customers + [end_depot]}
            for i, j in base_arcs:
                outgoing[i].append(j)
                incoming[j].append(i)

            z = {
                i: model.addVar(
                    lb=exogenous[i][0],
                    ub=exogenous[i][1] - widths[i],
                    vtype=GRB.CONTINUOUS,
                    name=f"z_{i}",
                )
                for i in customers
            }

            x = {}
            service_time = {}
            load = {}

            for s in range(num_scenarios):
                for i in [start_depot] + customers + [end_depot]:
                    service_time[s, i] = model.addVar(
                        lb=node_lower[i],
                        ub=node_upper[i],
                        vtype=GRB.CONTINUOUS,
                        name=f"t_{s}_{i}",
                    )
                for i in customers:
                    load[s, i] = model.addVar(
                        lb=scenario_demands[s][i],
                        ub=capacity,
                        vtype=GRB.CONTINUOUS,
                        name=f"load_{s}_{i}",
                    )
                for i, j in base_arcs:
                    x[s, i, j] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{s}_{i}_{j}",
                    )

            model.update()

            for s in range(num_scenarios):
                for i in customers:
                    model.addConstr(
                        gp.quicksum(x[s, h, i] for h in incoming[i]) == 1,
                        name=f"in_{s}_{i}",
                    )
                    model.addConstr(
                        gp.quicksum(x[s, i, j] for j in outgoing[i]) == 1,
                        name=f"out_{s}_{i}",
                    )

                    model.addConstr(service_time[s, i] >= z[i])
                    model.addConstr(service_time[s, i] <= z[i] + widths[i])

                model.addConstr(
                    gp.quicksum(x[s, start_depot, j] for j in outgoing[start_depot])
                    == gp.quicksum(x[s, i, end_depot] for i in incoming[end_depot]),
                    name=f"depot_balance_{s}",
                )

                total_demand = sum(scenario_demands[s].values())
                model.addConstr(
                    gp.quicksum(x[s, start_depot, j] for j in outgoing[start_depot])
                    >= total_demand / float(capacity),
                    name=f"vehicle_lb_{s}",
                )

                for i, j in base_arcs:
                    big_m = max(
                        0.0,
                        node_upper[i] + distance[i][j] - node_lower[j],
                    )
                    model.addConstr(
                        service_time[s, j]
                        >= service_time[s, i]
                        + distance[i][j]
                        - big_m * (1 - x[s, i, j]),
                        name=f"time_{s}_{i}_{j}",
                    )

                    if j in customer_set:
                        source_load = 0.0 if i == start_depot else load[s, i]
                        model.addConstr(
                            load[s, j]
                            >= source_load
                            + scenario_demands[s][j]
                            - capacity * (1 - x[s, i, j]),
                            name=f"capacity_{s}_{i}_{j}",
                        )

            model.setObjective(
                gp.quicksum(
                    probabilities[s] * distance[i][j] * x[s, i, j]
                    for s in range(num_scenarios)
                    for i, j in base_arcs
                ),
                GRB.MINIMIZE,
            )

            # Warm start from the best heuristic incumbent.
            if best_window_starts is not None and best_sequences is not None:
                for i in customers:
                    z[i].Start = best_window_starts[i]

                for variable in x.values():
                    variable.Start = 0.0

                for s in range(num_scenarios):
                    service_time[s, start_depot].Start = depot_open
                    maximum_return = depot_open

                    for route in best_sequences[s]:
                        previous = start_depot
                        current_time = depot_open
                        cumulative_load = 0

                        for customer in route:
                            if (s, previous, customer) in x:
                                x[s, previous, customer].Start = 1.0
                            current_time = max(
                                current_time + distance[previous][customer],
                                best_window_starts[customer],
                            )
                            service_time[s, customer].Start = current_time
                            cumulative_load += scenario_demands[s][customer]
                            load[s, customer].Start = cumulative_load
                            previous = customer

                        if (s, previous, end_depot) in x:
                            x[s, previous, end_depot].Start = 1.0
                        maximum_return = max(
                            maximum_return,
                            current_time + distance[previous][end_depot],
                        )

                    service_time[s, end_depot].Start = maximum_return

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    arc_values = cb_model.cbGetSolution(x)
                    sequences = extract_sequences_from_arc_values(
                        arc_values,
                        num_scenarios,
                        customers,
                        start_depot,
                        end_depot,
                    )
                    if sequences is None:
                        return

                    starts = {
                        i: min(
                            max(float(cb_model.cbGetSolution(z[i])), exogenous[i][0]),
                            exogenous[i][1] - widths[i],
                        )
                        for i in customers
                    }
                    solution = build_solution(
                        sequences,
                        starts,
                        widths,
                        scenarios,
                        distance,
                        depot_window,
                        probabilities,
                        start_depot,
                        end_depot,
                    )
                    register_solution(solution, starts, sequences)
                except Exception:
                    # Never interrupt the optimization because of logging or
                    # incumbent reconstruction issues.
                    return

            model.optimize(incumbent_callback)

            # Explicitly inspect the final incumbent in case the callback was
            # not invoked by a particular Gurobi configuration.
            if model.SolCount > 0:
                arc_values = {key: var.X for key, var in x.items()}
                sequences = extract_sequences_from_arc_values(
                    arc_values,
                    num_scenarios,
                    customers,
                    start_depot,
                    end_depot,
                )
                if sequences is not None:
                    starts = {
                        i: min(
                            max(float(z[i].X), exogenous[i][0]),
                            exogenous[i][1] - widths[i],
                        )
                        for i in customers
                    }
                    final_mip_solution = build_solution(
                        sequences,
                        starts,
                        widths,
                        scenarios,
                        distance,
                        depot_window,
                        probabilities,
                        start_depot,
                        end_depot,
                    )
                    register_solution(final_mip_solution, starts, sequences)

        except Exception:
            # The heuristic incumbent remains available if model construction,
            # licensing, or optimization fails.
            pass

    if best_solution is None:
        raise RuntimeError("No feasible solution could be constructed")

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()