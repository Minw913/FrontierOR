import argparse
import json
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def component_count(nodes, adjacency):
    nodes = set(nodes)
    count = 0
    unseen = set(nodes)

    while unseen:
        count += 1
        start = unseen.pop()
        stack = [start]

        while stack:
            u = stack.pop()
            for v in adjacency.get(u, ()):
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)

    return count


def make_solution(assignments, trades):
    output = {}
    for trade in trades:
        trade_id = trade["id"]
        output[str(trade_id)] = {}
        for period in range(trade["start_period"], trade["end_period"] + 1):
            values = assignments.get((trade_id, period), [])
            output[str(trade_id)][str(period)] = sorted(int(c) for c in values)
    return output


def calculate_objective(assignments, trades):
    objective = 0
    for trade in trades:
        trade_id = trade["id"]
        for period in range(trade["start_period"] + 1, trade["end_period"] + 1):
            previous = set(assignments.get((trade_id, period - 1), []))
            current = set(assignments.get((trade_id, period), []))
            if previous != current:
                objective += 1
    return objective


def construct_trade_candidate(
    free_containers,
    demand,
    max_dispersion,
    previous_assignment,
    previous_owner,
    trade_id,
    adjacency,
    rng,
    repetitions=16,
):
    free_containers = set(free_containers)
    previous_assignment = set(previous_assignment) & free_containers

    if demand == 0:
        return set()
    if len(free_containers) < demand or max_dispersion <= 0:
        return None

    best = None
    best_score = None

    for _ in range(repetitions):
        selected = set()

        while len(selected) < demand:
            choices = []

            for c in free_containers - selected:
                trial = selected | {c}
                components = component_count(trial, adjacency)
                if components > max_dispersion:
                    continue

                neighbors_selected = len(adjacency.get(c, set()) & selected)
                score = 0.0

                if c in previous_assignment:
                    score += 1000.0

                owner = previous_owner.get(c)
                if owner is not None and owner != trade_id:
                    score -= 250.0

                score += 25.0 * neighbors_selected

                # Prefer additions that merge existing components.
                if selected:
                    before = component_count(selected, adjacency)
                    score += 35.0 * (before - components)

                score += rng.random() * 30.0
                choices.append((score, c))

            if not choices:
                break

            choices.sort(reverse=True)
            selected.add(choices[0][1])

        if len(selected) != demand:
            continue
        if component_count(selected, adjacency) > max_dispersion:
            continue

        retained = len(selected & previous_assignment)
        foreign = sum(
            1
            for c in selected
            if previous_owner.get(c) is not None
            and previous_owner.get(c) != trade_id
        )
        score = retained * 10000 - foreign * 100 + rng.random()

        if best is None or score > best_score:
            best = set(selected)
            best_score = score

    return best


def greedy_feasible_solution(
    num_periods,
    containers,
    trades,
    adjacency,
    deadline,
):
    rng = random.Random(0)

    available_by_period = {}
    for period in range(1, num_periods + 1):
        available_by_period[period] = {
            c["id"]
            for c in containers
            if c["availability_start"] <= period <= c["availability_end"]
        }

    active_by_period = {}
    for period in range(1, num_periods + 1):
        active_by_period[period] = [
            trade
            for trade in trades
            if trade["start_period"] <= period <= trade["end_period"]
        ]

    best_assignments = None
    best_objective = None

    global_attempt = 0
    while global_attempt < 12 and (time.monotonic() < deadline or global_attempt == 0):
        global_attempt += 1
        assignments = {}
        feasible = True

        for period in range(1, num_periods + 1):
            active_trades = active_by_period[period]
            if not active_trades:
                continue

            previous_owner = {}
            for trade in active_trades:
                trade_id = trade["id"]
                if period > trade["start_period"]:
                    for c in assignments.get((trade_id, period - 1), []):
                        if c in available_by_period[period]:
                            previous_owner[c] = trade_id

            best_period_assignment = None
            best_period_score = None

            period_attempts = 0
            while period_attempts < 24 and (
                time.monotonic() < deadline or period_attempts == 0
            ):
                period_attempts += 1
                free = set(available_by_period[period])
                current = {}

                def order_key(trade):
                    trade_id = trade["id"]
                    previous = set(assignments.get((trade_id, period - 1), []))
                    retained_available = len(previous & free)
                    flexibility = len(free) - trade["container_demand"]
                    return (
                        -retained_available,
                        flexibility,
                        -trade["container_demand"],
                        rng.random(),
                    )

                ordered_trades = sorted(active_trades, key=order_key)
                attempt_feasible = True

                for trade in ordered_trades:
                    trade_id = trade["id"]
                    demand = trade["container_demand"]
                    dispersion = trade["max_dispersion"]
                    previous = (
                        assignments.get((trade_id, period - 1), [])
                        if period > trade["start_period"]
                        else []
                    )

                    candidate = construct_trade_candidate(
                        free_containers=free,
                        demand=demand,
                        max_dispersion=dispersion,
                        previous_assignment=previous,
                        previous_owner=previous_owner,
                        trade_id=trade_id,
                        adjacency=adjacency,
                        rng=rng,
                    )

                    if candidate is None:
                        attempt_feasible = False
                        break

                    current[trade_id] = candidate
                    free -= candidate

                if not attempt_feasible:
                    continue

                score = 0
                for trade in active_trades:
                    trade_id = trade["id"]
                    if period > trade["start_period"]:
                        previous = set(assignments.get((trade_id, period - 1), []))
                        current_set = current[trade_id]
                        if previous == current_set:
                            score += 100000
                        score += len(previous & current_set) * 100

                if best_period_assignment is None or score > best_period_score:
                    best_period_assignment = current
                    best_period_score = score

            if best_period_assignment is None:
                feasible = False
                break

            for trade in active_trades:
                trade_id = trade["id"]
                assignments[(trade_id, period)] = sorted(
                    best_period_assignment[trade_id]
                )

        if feasible:
            objective = calculate_objective(assignments, trades)
            if best_assignments is None or objective < best_objective:
                best_assignments = assignments
                best_objective = objective
                if objective == 0:
                    break

    return best_assignments, best_objective


def solve(instance, time_limit, logger):
    start_time = time.monotonic()

    params = instance["problem_parameters"]
    num_periods = params["num_periods"]
    containers = instance["containers"]
    trades = instance["trades"]

    container_ids = [c["id"] for c in containers]
    container_by_id = {c["id"]: c for c in containers}

    adjacency = {c: set() for c in container_ids}
    for container in containers:
        c = container["id"]
        for neighbor in container.get("adjacent_containers", []):
            if neighbor in adjacency and neighbor != c:
                adjacency[c].add(neighbor)
                adjacency[neighbor].add(c)

    # Spend only a small portion of the budget constructing an initial incumbent.
    heuristic_budget = min(1.5, max(0.05, time_limit * 0.08))
    heuristic_deadline = min(
        start_time + heuristic_budget,
        start_time + max(0.05, time_limit * 0.15),
    )

    best_assignments, best_objective = greedy_feasible_solution(
        num_periods,
        containers,
        trades,
        adjacency,
        heuristic_deadline,
    )

    best_holder = {
        "objective": best_objective,
        "assignments": best_assignments,
    }

    if best_assignments is not None and logger:
        initial_solution = {
            "objective_value": float(best_objective),
            "assignments": make_solution(best_assignments, trades),
        }
        logger.log_solution(float(best_objective), initial_solution)

    model = gp.Model("container_trade_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    x = {}
    z = {}
    roots = {}
    flows = {}

    # Assignment variables exist only when both the trade and container are present.
    for trade in trades:
        trade_id = trade["id"]
        for period in range(trade["start_period"], trade["end_period"] + 1):
            for c in container_ids:
                container = container_by_id[c]
                if (
                    container["availability_start"]
                    <= period
                    <= container["availability_end"]
                ):
                    x[(trade_id, period, c)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{trade_id}_{period}_{c}",
                    )

            if period > trade["start_period"]:
                z[(trade_id, period)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"change_{trade_id}_{period}",
                )

    model.update()

    # Exact demand for every active trade and period.
    for trade in trades:
        trade_id = trade["id"]
        demand = trade["container_demand"]

        for period in range(trade["start_period"], trade["end_period"] + 1):
            vars_for_trade = [
                x[(trade_id, period, c)]
                for c in container_ids
                if (trade_id, period, c) in x
            ]
            model.addConstr(
                gp.quicksum(vars_for_trade) == demand,
                name=f"demand_{trade_id}_{period}",
            )

    # At most one trade per container and period.
    for period in range(1, num_periods + 1):
        for c in container_ids:
            vars_for_container = [
                x[(trade["id"], period, c)]
                for trade in trades
                if (trade["id"], period, c) in x
            ]
            if vars_for_container:
                model.addConstr(
                    gp.quicksum(vars_for_container) <= 1,
                    name=f"capacity_{period}_{c}",
                )

    # A change is triggered if any container is newly assigned.
    for trade in trades:
        trade_id = trade["id"]
        for period in range(trade["start_period"] + 1, trade["end_period"] + 1):
            change_var = z[(trade_id, period)]

            for c in container_ids:
                current = x.get((trade_id, period, c))
                if current is None:
                    continue

                previous = x.get((trade_id, period - 1, c))
                if previous is None:
                    model.addConstr(
                        change_var >= current,
                        name=f"change_new_{trade_id}_{period}_{c}",
                    )
                else:
                    model.addConstr(
                        change_var >= current - previous,
                        name=f"change_diff_{trade_id}_{period}_{c}",
                    )

    # Connectivity/dispersion formulation using rooted single-commodity flows.
    # Every selected connected component must contain at least one root, and the
    # number of roots is bounded by the trade's dispersion limit.
    for trade in trades:
        trade_id = trade["id"]
        demand = trade["container_demand"]
        dispersion = trade["max_dispersion"]

        if demand <= 1 or dispersion >= demand:
            continue

        for period in range(trade["start_period"], trade["end_period"] + 1):
            eligible = [
                c for c in container_ids if (trade_id, period, c) in x
            ]
            eligible_set = set(eligible)

            for c in eligible:
                roots[(trade_id, period, c)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"root_{trade_id}_{period}_{c}",
                )

            directed_edges = []
            for u in eligible:
                for v in adjacency.get(u, set()):
                    if v in eligible_set:
                        key = (trade_id, period, u, v)
                        flows[key] = model.addVar(
                            lb=0.0,
                            ub=float(demand),
                            vtype=GRB.CONTINUOUS,
                            name=f"flow_{trade_id}_{period}_{u}_{v}",
                        )
                        directed_edges.append((u, v))

            model.update()

            root_vars = [roots[(trade_id, period, c)] for c in eligible]
            model.addConstr(
                gp.quicksum(root_vars) >= 1,
                name=f"root_min_{trade_id}_{period}",
            )
            model.addConstr(
                gp.quicksum(root_vars) <= dispersion,
                name=f"root_max_{trade_id}_{period}",
            )

            for c in eligible:
                root_var = roots[(trade_id, period, c)]
                assignment_var = x[(trade_id, period, c)]
                model.addConstr(
                    root_var <= assignment_var,
                    name=f"root_selected_{trade_id}_{period}_{c}",
                )

                incoming = [
                    flows[(trade_id, period, u, c)]
                    for u in adjacency.get(c, set())
                    if (trade_id, period, u, c) in flows
                ]
                outgoing = [
                    flows[(trade_id, period, c, v)]
                    for v in adjacency.get(c, set())
                    if (trade_id, period, c, v) in flows
                ]

                model.addConstr(
                    gp.quicksum(incoming) - gp.quicksum(outgoing)
                    >= assignment_var - demand * root_var,
                    name=f"flow_balance_{trade_id}_{period}_{c}",
                )

            for u, v in directed_edges:
                flow_var = flows[(trade_id, period, u, v)]
                model.addConstr(
                    flow_var <= demand * x[(trade_id, period, u)],
                    name=f"flow_source_{trade_id}_{period}_{u}_{v}",
                )
                model.addConstr(
                    flow_var <= demand * x[(trade_id, period, v)],
                    name=f"flow_target_{trade_id}_{period}_{u}_{v}",
                )

    model.setObjective(gp.quicksum(z.values()), GRB.MINIMIZE)

    # Supply the greedy assignment as a partial MIP start. Gurobi can construct
    # roots and flows for it automatically.
    if best_assignments is not None:
        for (trade_id, period, c), var in x.items():
            assigned = set(best_assignments.get((trade_id, period), []))
            var.Start = 1.0 if c in assigned else 0.0

        for (trade_id, period), var in z.items():
            previous = set(best_assignments.get((trade_id, period - 1), []))
            current = set(best_assignments.get((trade_id, period), []))
            var.Start = 1.0 if previous != current else 0.0

    remaining = max(0.01, time_limit - (time.monotonic() - start_time))
    model.Params.TimeLimit = remaining

    def extract_callback_assignment(cb_model):
        result = {}
        for trade in trades:
            trade_id = trade["id"]
            for period in range(trade["start_period"], trade["end_period"] + 1):
                assigned = []
                for c in container_ids:
                    var = x.get((trade_id, period, c))
                    if var is not None and cb_model.cbGetSolution(var) > 0.5:
                        assigned.append(c)
                result[(trade_id, period)] = assigned
        return result

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            assignments = extract_callback_assignment(cb_model)
            objective = calculate_objective(assignments, trades)

            current_best = best_holder["objective"]
            if current_best is None or objective < current_best:
                best_holder["objective"] = objective
                best_holder["assignments"] = assignments

                if logger:
                    solution = {
                        "objective_value": float(objective),
                        "assignments": make_solution(assignments, trades),
                    }
                    logger.log_solution(float(objective), solution)
        except Exception:
            # Logging or callback extraction must not terminate optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        solver_assignments = {}
        for trade in trades:
            trade_id = trade["id"]
            for period in range(trade["start_period"], trade["end_period"] + 1):
                assigned = [
                    c
                    for c in container_ids
                    if (trade_id, period, c) in x
                    and x[(trade_id, period, c)].X > 0.5
                ]
                solver_assignments[(trade_id, period)] = assigned

        solver_objective = calculate_objective(solver_assignments, trades)
        if (
            best_holder["objective"] is None
            or solver_objective < best_holder["objective"]
        ):
            best_holder["objective"] = solver_objective
            best_holder["assignments"] = solver_assignments

            if logger:
                solution = {
                    "objective_value": float(solver_objective),
                    "assignments": make_solution(solver_assignments, trades),
                }
                logger.log_solution(float(solver_objective), solution)

    if best_holder["assignments"] is None:
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

    return {
        "objective_value": float(best_holder["objective"]),
        "assignments": make_solution(best_holder["assignments"], trades),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    solution = solve(
        instance=instance,
        time_limit=max(1, args.time_limit),
        logger=logger,
    )

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2)


if __name__ == "__main__":
    main()