import argparse
import itertools
import json
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_adjacency(containers):
    ids = {c["id"] for c in containers}
    adjacency = {c: set() for c in ids}

    for container in containers:
        u = container["id"]
        for v in container.get("adjacent_containers", []):
            if v in ids and v != u:
                adjacency[u].add(v)
                adjacency[v].add(u)

    return adjacency


def count_components(nodes, adjacency):
    unseen = set(nodes)
    count = 0

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


def assignment_objective(assignments, trades):
    objective = 0

    for trade in trades:
        trade_id = trade["id"]
        start = trade["start_period"]
        end = trade["end_period"]

        for period in range(start + 1, end + 1):
            previous = set(assignments.get((trade_id, period - 1), ()))
            current = set(assignments.get((trade_id, period), ()))
            if previous != current:
                objective += 1

    return objective


def output_assignments(assignments, trades):
    result = {}

    for trade in trades:
        trade_id = trade["id"]
        trade_result = {}

        for period in range(
            trade["start_period"],
            trade["end_period"] + 1,
        ):
            trade_result[str(period)] = sorted(
                int(c) for c in assignments.get((trade_id, period), ())
            )

        result[str(trade_id)] = trade_result

    return result


def full_solution(assignments, trades):
    objective = assignment_objective(assignments, trades)
    return {
        "objective_value": float(objective),
        "assignments": output_assignments(assignments, trades),
    }


def random_connected_candidate(
    free,
    demand,
    max_dispersion,
    previous,
    previous_owner,
    trade_id,
    adjacency,
    rng,
):
    free = set(free)
    previous = set(previous)

    if demand == 0:
        return set()

    if demand > len(free) or max_dispersion <= 0:
        return None

    # Retaining the exact previous assignment is always preferable.
    if (
        len(previous) == demand
        and previous <= free
        and count_components(previous, adjacency) <= max_dispersion
    ):
        return previous

    best = None
    best_score = None

    repetitions = min(40, max(10, len(free)))

    for _ in range(repetitions):
        selected = set()

        while len(selected) < demand:
            choices = []

            for c in free - selected:
                trial = selected | {c}
                components = count_components(trial, adjacency)

                if components > max_dispersion:
                    continue

                selected_neighbors = len(adjacency.get(c, ()) & selected)
                score = rng.random() * 20.0
                score += 1000.0 if c in previous else 0.0
                score += 30.0 * selected_neighbors

                owner = previous_owner.get(c)
                if owner is not None and owner != trade_id:
                    score -= 100.0

                choices.append((score, c))

            if not choices:
                break

            choices.sort(reverse=True)
            selected.add(choices[0][1])

        if len(selected) != demand:
            continue

        if count_components(selected, adjacency) > max_dispersion:
            continue

        retained = len(selected & previous)
        foreign = sum(
            1
            for c in selected
            if previous_owner.get(c) is not None
            and previous_owner.get(c) != trade_id
        )
        score = retained * 10000 - foreign * 100 + rng.random()

        if best is None or score > best_score:
            best = selected
            best_score = score

    return best


def exact_small_period_assignment(
    active_trades,
    available,
    previous_assignments,
    adjacency,
    deadline,
):
    available = sorted(available)

    if len(available) > 22:
        return None

    candidate_lists = {}

    for trade in active_trades:
        trade_id = trade["id"]
        demand = trade["container_demand"]
        dispersion = trade["max_dispersion"]
        previous = set(previous_assignments.get(trade_id, ()))

        if demand > len(available):
            return None

        candidates = []
        for combination in itertools.combinations(available, demand):
            if time.monotonic() >= deadline:
                return None

            candidate = frozenset(combination)
            if count_components(candidate, adjacency) <= dispersion:
                unchanged = candidate == previous
                retained = len(candidate & previous)
                candidates.append(
                    (
                        0 if unchanged else 1,
                        -retained,
                        tuple(combination),
                        candidate,
                    )
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[:3])
        candidate_lists[trade_id] = [item[3] for item in candidates]

    ordered = sorted(
        active_trades,
        key=lambda trade: len(candidate_lists[trade["id"]]),
    )

    best_assignment = None
    best_score = None

    def search(index, used, current, changes, retained):
        nonlocal best_assignment, best_score

        if time.monotonic() >= deadline:
            return

        if best_score is not None and changes > best_score[0]:
            return

        if index == len(ordered):
            score = (changes, -retained)
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = {
                    trade_id: set(values)
                    for trade_id, values in current.items()
                }
            return

        trade = ordered[index]
        trade_id = trade["id"]
        previous = set(previous_assignments.get(trade_id, ()))

        for candidate in candidate_lists[trade_id]:
            if candidate & used:
                continue

            is_change = 0 if candidate == previous else 1
            current[trade_id] = candidate
            search(
                index + 1,
                used | candidate,
                current,
                changes + is_change,
                retained + len(candidate & previous),
            )
            del current[trade_id]

    search(0, frozenset(), {}, 0, 0)
    return best_assignment


def greedy_solution(
    containers,
    trades,
    adjacency,
    deadline,
):
    rng = random.Random(0)

    relevant_periods = sorted(
        {
            period
            for trade in trades
            for period in range(
                trade["start_period"],
                trade["end_period"] + 1,
            )
        }
    )

    available_by_period = {
        period: {
            container["id"]
            for container in containers
            if container["availability_start"]
            <= period
            <= container["availability_end"]
        }
        for period in relevant_periods
    }

    active_by_period = {
        period: [
            trade
            for trade in trades
            if trade["start_period"] <= period <= trade["end_period"]
        ]
        for period in relevant_periods
    }

    best_assignments = None
    best_objective = None
    attempt = 0

    while attempt < 20 and (
        attempt == 0 or time.monotonic() < deadline
    ):
        attempt += 1
        assignments = {}
        feasible = True

        for period in relevant_periods:
            active = active_by_period[period]
            available = available_by_period[period]

            if sum(t["container_demand"] for t in active) > len(available):
                feasible = False
                break

            previous_assignments = {}
            previous_owner = {}

            for trade in active:
                trade_id = trade["id"]
                if period > trade["start_period"]:
                    previous = set(
                        assignments.get((trade_id, period - 1), ())
                    )
                else:
                    previous = set()

                previous_assignments[trade_id] = previous
                for c in previous:
                    previous_owner[c] = trade_id

            best_period = None
            best_period_score = None

            period_attempt = 0
            while period_attempt < 40 and (
                period_attempt == 0 or time.monotonic() < deadline
            ):
                period_attempt += 1
                free = set(available)
                current = {}

                def ordering_key(trade):
                    trade_id = trade["id"]
                    previous = previous_assignments[trade_id]
                    retainable = len(previous & free)
                    return (
                        -retainable,
                        trade["max_dispersion"],
                        -trade["container_demand"],
                        rng.random(),
                    )

                ordered = sorted(active, key=ordering_key)
                success = True

                for trade in ordered:
                    trade_id = trade["id"]
                    candidate = random_connected_candidate(
                        free=free,
                        demand=trade["container_demand"],
                        max_dispersion=trade["max_dispersion"],
                        previous=previous_assignments[trade_id],
                        previous_owner=previous_owner,
                        trade_id=trade_id,
                        adjacency=adjacency,
                        rng=rng,
                    )

                    if candidate is None:
                        success = False
                        break

                    current[trade_id] = candidate
                    free -= candidate

                if not success:
                    continue

                changes = 0
                retained = 0
                for trade in active:
                    trade_id = trade["id"]
                    previous = previous_assignments[trade_id]
                    candidate = current[trade_id]

                    if period > trade["start_period"] and candidate != previous:
                        changes += 1
                    retained += len(candidate & previous)

                score = (changes, -retained)
                if best_period_score is None or score < best_period_score:
                    best_period_score = score
                    best_period = current

            if best_period is None and time.monotonic() < deadline:
                best_period = exact_small_period_assignment(
                    active_trades=active,
                    available=available,
                    previous_assignments=previous_assignments,
                    adjacency=adjacency,
                    deadline=deadline,
                )

            if best_period is None:
                feasible = False
                break

            for trade in active:
                trade_id = trade["id"]
                assignments[(trade_id, period)] = sorted(
                    best_period[trade_id]
                )

        if feasible:
            objective = assignment_objective(assignments, trades)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_assignments = assignments

            if objective == 0:
                break

    return best_assignments


def solve(instance, time_limit, logger):
    start_time = time.monotonic()
    deadline = start_time + max(1, time_limit)

    containers = instance["containers"]
    trades = instance["trades"]
    container_ids = [container["id"] for container in containers]
    container_by_id = {
        container["id"]: container for container in containers
    }
    adjacency = build_adjacency(containers)

    relevant_periods = sorted(
        {
            period
            for trade in trades
            for period in range(
                trade["start_period"],
                trade["end_period"] + 1,
            )
        }
    )

    heuristic_deadline = min(
        deadline,
        start_time + min(2.0, max(0.1, time_limit * 0.1)),
    )

    try:
        incumbent_assignments = greedy_solution(
            containers,
            trades,
            adjacency,
            heuristic_deadline,
        )
    except Exception:
        incumbent_assignments = None

    incumbent_objective = None
    if incumbent_assignments is not None:
        incumbent_objective = assignment_objective(
            incumbent_assignments,
            trades,
        )
        if logger:
            solution = full_solution(incumbent_assignments, trades)
            logger.log_solution(
                solution["objective_value"],
                solution,
            )

    model = gp.Model("container_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    x = {}
    changed = {}

    for trade in trades:
        trade_id = trade["id"]

        for period in range(
            trade["start_period"],
            trade["end_period"] + 1,
        ):
            for container_id in container_ids:
                container = container_by_id[container_id]
                if (
                    container["availability_start"]
                    <= period
                    <= container["availability_end"]
                ):
                    x[(trade_id, period, container_id)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{trade_id}_{period}_{container_id}",
                    )

            if period > trade["start_period"]:
                changed[(trade_id, period)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"changed_{trade_id}_{period}",
                )

    model.update()

    # Demand constraints.
    for trade in trades:
        trade_id = trade["id"]
        demand = trade["container_demand"]

        for period in range(
            trade["start_period"],
            trade["end_period"] + 1,
        ):
            variables = [
                x[(trade_id, period, container_id)]
                for container_id in container_ids
                if (trade_id, period, container_id) in x
            ]
            model.addConstr(
                gp.quicksum(variables) == demand,
                name=f"demand_{trade_id}_{period}",
            )

    # Container capacity constraints. Use all relevant period labels rather
    # than assuming that periods start at 1.
    for period in relevant_periods:
        for container_id in container_ids:
            variables = [
                x[(trade["id"], period, container_id)]
                for trade in trades
                if (trade["id"], period, container_id) in x
            ]

            if variables:
                model.addConstr(
                    gp.quicksum(variables) <= 1,
                    name=f"capacity_{period}_{container_id}",
                )

    # With constant demand, an assignment differs exactly when at least one
    # currently assigned container was not assigned in the previous period.
    for trade in trades:
        trade_id = trade["id"]

        for period in range(
            trade["start_period"] + 1,
            trade["end_period"] + 1,
        ):
            change_var = changed[(trade_id, period)]

            for container_id in container_ids:
                current = x.get((trade_id, period, container_id))
                if current is None:
                    continue

                previous = x.get(
                    (trade_id, period - 1, container_id)
                )

                if previous is None:
                    model.addConstr(change_var >= current)
                else:
                    model.addConstr(
                        change_var >= current - previous
                    )

    # Rooted flow connectivity formulation. Every selected component must
    # contain a selected root; the number of roots is bounded by dispersion.
    root = {}
    flow = {}

    for trade in trades:
        trade_id = trade["id"]
        demand = trade["container_demand"]
        dispersion = trade["max_dispersion"]

        # Any set of at most dispersion containers automatically satisfies the
        # component limit.
        if demand <= dispersion or demand <= 1:
            continue

        for period in range(
            trade["start_period"],
            trade["end_period"] + 1,
        ):
            eligible = [
                container_id
                for container_id in container_ids
                if (trade_id, period, container_id) in x
            ]
            eligible_set = set(eligible)

            for container_id in eligible:
                root[(trade_id, period, container_id)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"root_{trade_id}_{period}_{container_id}",
                )

            directed_edges = []
            for u in eligible:
                for v in adjacency.get(u, ()):
                    if v in eligible_set:
                        key = (trade_id, period, u, v)
                        flow[key] = model.addVar(
                            lb=0.0,
                            ub=float(demand),
                            vtype=GRB.CONTINUOUS,
                            name=f"flow_{trade_id}_{period}_{u}_{v}",
                        )
                        directed_edges.append((u, v))

            model.update()

            roots = [
                root[(trade_id, period, container_id)]
                for container_id in eligible
            ]

            model.addConstr(
                gp.quicksum(roots) >= 1,
                name=f"root_lower_{trade_id}_{period}",
            )
            model.addConstr(
                gp.quicksum(roots) <= dispersion,
                name=f"root_upper_{trade_id}_{period}",
            )

            for container_id in eligible:
                root_var = root[
                    (trade_id, period, container_id)
                ]
                selected_var = x[
                    (trade_id, period, container_id)
                ]

                model.addConstr(
                    root_var <= selected_var,
                    name=f"root_selected_{trade_id}_{period}_{container_id}",
                )

                incoming = [
                    flow[(trade_id, period, u, container_id)]
                    for u in adjacency.get(container_id, ())
                    if (trade_id, period, u, container_id) in flow
                ]
                outgoing = [
                    flow[(trade_id, period, container_id, v)]
                    for v in adjacency.get(container_id, ())
                    if (trade_id, period, container_id, v) in flow
                ]

                model.addConstr(
                    gp.quicksum(incoming)
                    - gp.quicksum(outgoing)
                    >= selected_var - demand * root_var,
                    name=f"balance_{trade_id}_{period}_{container_id}",
                )

            for u, v in directed_edges:
                flow_var = flow[(trade_id, period, u, v)]
                model.addConstr(
                    flow_var
                    <= demand * x[(trade_id, period, u)],
                    name=f"flow_from_{trade_id}_{period}_{u}_{v}",
                )
                model.addConstr(
                    flow_var
                    <= demand * x[(trade_id, period, v)],
                    name=f"flow_to_{trade_id}_{period}_{u}_{v}",
                )

    model.setObjective(
        gp.quicksum(changed.values()),
        GRB.MINIMIZE,
    )

    # Complete starts for assignment and change variables.
    if incumbent_assignments is not None:
        assignment_sets = {
            key: set(value)
            for key, value in incumbent_assignments.items()
        }

        for (trade_id, period, container_id), variable in x.items():
            variable.Start = (
                1.0
                if container_id
                in assignment_sets.get((trade_id, period), set())
                else 0.0
            )

        for (trade_id, period), variable in changed.items():
            previous = assignment_sets.get(
                (trade_id, period - 1), set()
            )
            current = assignment_sets.get(
                (trade_id, period), set()
            )
            variable.Start = 1.0 if previous != current else 0.0

    remaining = max(0.01, deadline - time.monotonic())
    model.Params.TimeLimit = remaining

    incumbent_holder = {
        "assignments": incumbent_assignments,
        "objective": incumbent_objective,
    }

    def extract_callback_solution(callback_model):
        result = {}

        for trade in trades:
            trade_id = trade["id"]
            for period in range(
                trade["start_period"],
                trade["end_period"] + 1,
            ):
                assigned = []
                for container_id in container_ids:
                    variable = x.get(
                        (trade_id, period, container_id)
                    )
                    if (
                        variable is not None
                        and callback_model.cbGetSolution(variable) > 0.5
                    ):
                        assigned.append(container_id)

                result[(trade_id, period)] = assigned

        return result

    def callback(callback_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            candidate = extract_callback_solution(callback_model)
            objective = assignment_objective(candidate, trades)

            if (
                incumbent_holder["objective"] is None
                or objective < incumbent_holder["objective"]
            ):
                incumbent_holder["objective"] = objective
                incumbent_holder["assignments"] = candidate

                if logger:
                    solution = full_solution(candidate, trades)
                    logger.log_solution(
                        solution["objective_value"],
                        solution,
                    )
        except Exception:
            # Logging must not interrupt optimization.
            pass

    model.optimize(callback)

    if model.SolCount > 0:
        solver_assignments = {}

        for trade in trades:
            trade_id = trade["id"]
            for period in range(
                trade["start_period"],
                trade["end_period"] + 1,
            ):
                solver_assignments[(trade_id, period)] = [
                    container_id
                    for container_id in container_ids
                    if (
                        (trade_id, period, container_id) in x
                        and x[
                            (trade_id, period, container_id)
                        ].X > 0.5
                    )
                ]

        solver_objective = assignment_objective(
            solver_assignments,
            trades,
        )

        if (
            incumbent_holder["objective"] is None
            or solver_objective < incumbent_holder["objective"]
        ):
            incumbent_holder["assignments"] = solver_assignments
            incumbent_holder["objective"] = solver_objective

            if logger:
                solution = full_solution(
                    solver_assignments,
                    trades,
                )
                logger.log_solution(
                    solution["objective_value"],
                    solution,
                )

    if incumbent_holder["assignments"] is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    return full_solution(
        incumbent_holder["assignments"],
        trades,
    )


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