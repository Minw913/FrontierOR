import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-6


def configure_model(model):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0


def calculate_objective(assignment, flow, distance, setup_costs, chi, alpha, delta):
    n = len(assignment)
    hubs = {assignment[i] for i in range(n)}
    value = sum(setup_costs[h] for h in hubs)

    for i in range(n):
        hi = assignment[i]
        for j in range(n):
            wij = flow[i][j]
            if wij == 0:
                continue
            hj = assignment[j]
            value += wij * (
                chi * distance[i][hi]
                + alpha * distance[hi][hj]
                + delta * distance[hj][j]
            )
    return float(value)


def make_solution(assignment, flow, distance, setup_costs, chi, alpha, delta):
    assignment = [int(v) for v in assignment]
    hubs = sorted(set(assignment))
    objective = calculate_objective(
        assignment, flow, distance, setup_costs, chi, alpha, delta
    )
    return {
        "objective_value": float(objective),
        "hubs": [int(h) for h in hubs],
        "assignment": assignment,
    }


def is_feasible_assignment(assignment, demands, capacities, total_flow=None):
    n = len(assignment)
    if len(assignment) != n:
        return False

    hubs = set(assignment)
    for h in hubs:
        if h < 0 or h >= n or assignment[h] != h:
            return False

    loads = [0.0] * n
    for i, h in enumerate(assignment):
        if h < 0 or h >= n:
            return False
        loads[h] += demands[i]

    for h in hubs:
        if loads[h] > capacities[h] + 1e-5 * max(1.0, capacities[h]):
            return False

    if total_flow is not None:
        if sum(capacities[h] for h in hubs) + 1e-5 < total_flow:
            return False

    return True


def log_if_better(solution, incumbent, logger):
    if solution is None:
        return incumbent
    if incumbent is None or solution["objective_value"] < incumbent["objective_value"] - 1e-7:
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        return solution
    return incumbent


def greedy_initial_solution(demands, capacities, access_cost):
    n = len(demands)
    viable = [k for k in range(n) if capacities[k] + TOL >= demands[k]]
    if not viable:
        return None

    # Opening every viable node gives a robust starting point. Every opened
    # node is assigned to itself, as required.
    assignment = [-1] * n
    residual = [0.0] * n
    for k in viable:
        assignment[k] = k
        residual[k] = capacities[k] - demands[k]

    unassigned = [i for i in range(n) if assignment[i] < 0]
    unassigned.sort(key=lambda i: demands[i], reverse=True)

    for i in unassigned:
        choices = [k for k in viable if residual[k] + TOL >= demands[i]]
        if not choices:
            return None
        k = min(
            choices,
            key=lambda h: (
                access_cost[i][h],
                -residual[h],
                h,
            ),
        )
        assignment[i] = k
        residual[k] -= demands[i]

    return assignment


def solve_surrogate(
    n,
    demands,
    capacities,
    total_flow,
    setup_costs,
    access_cost,
    time_limit,
    start_assignment=None,
):
    if time_limit <= 0.02:
        return None

    model = gp.Model("capacitated_hub_surrogate")
    configure_model(model)
    model.Params.TimeLimit = max(0.01, float(time_limit))

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(n)) == 1)

    for i in range(n):
        for k in range(n):
            if i != k:
                model.addConstr(x[i, k] <= x[k, k])

    for k in range(n):
        model.addConstr(
            gp.quicksum(demands[i] * x[i, k] for i in range(n))
            <= capacities[k] * x[k, k]
        )

    model.addConstr(
        gp.quicksum(capacities[k] * x[k, k] for k in range(n)) >= total_flow
    )

    obj = gp.LinExpr()
    for k in range(n):
        obj += setup_costs[k] * x[k, k]
    for i in range(n):
        for k in range(n):
            obj += access_cost[i][k] * x[i, k]
    model.setObjective(obj, GRB.MINIMIZE)

    if start_assignment is not None:
        for i in range(n):
            ai = start_assignment[i]
            for k in range(n):
                x[i, k].Start = 1.0 if k == ai else 0.0

    model.optimize()

    if model.SolCount == 0:
        return None

    assignment = []
    for i in range(n):
        assignment.append(max(range(n), key=lambda k: x[i, k].X))

    if not is_feasible_assignment(assignment, demands, capacities, total_flow):
        return None
    return assignment


def local_search(
    assignment,
    demands,
    capacities,
    flow,
    distance,
    setup_costs,
    access_cost,
    alpha,
    deadline,
):
    n = len(assignment)
    assignment = list(assignment)

    def loads_and_hubs(a):
        loads = [0.0] * n
        for i, h in enumerate(a):
            loads[h] += demands[i]
        return loads, set(a)

    def transfer_move_delta(a, p, old_hub, new_hub):
        change = access_cost[p][new_hub] - access_cost[p][old_hub]
        for j in range(n):
            if j == p:
                continue
            hj = a[j]
            if flow[p][j]:
                change += (
                    alpha
                    * flow[p][j]
                    * (distance[new_hub][hj] - distance[old_hub][hj])
                )
            if flow[j][p]:
                change += (
                    alpha
                    * flow[j][p]
                    * (distance[hj][new_hub] - distance[hj][old_hub])
                )
        return change

    # First try to close expensive or lightly loaded hubs. A closure is built
    # greedily and accepted only after evaluating the complete exact cost.
    changed = True
    while changed and time.monotonic() < deadline:
        changed = False
        current_obj = None
        loads, hubs = loads_and_hubs(assignment)
        if len(hubs) <= 1:
            break

        ordered_hubs = sorted(
            hubs,
            key=lambda h: (
                -(setup_costs[h] / max(loads[h], 1e-9)),
                -setup_costs[h],
            ),
        )

        best_close_assignment = None
        best_close_obj = math.inf

        for h in ordered_hubs:
            if time.monotonic() >= deadline:
                break
            remaining_hubs = [k for k in hubs if k != h]
            members = [i for i in range(n) if assignment[i] == h]
            members.sort(key=lambda i: demands[i], reverse=True)

            trial = list(assignment)
            trial_loads = list(loads)
            trial_loads[h] = 0.0
            possible = True

            for p in members:
                choices = [
                    k
                    for k in remaining_hubs
                    if trial_loads[k] + demands[p] <= capacities[k] + TOL
                ]
                if not choices:
                    possible = False
                    break

                old = trial[p]
                k = min(
                    choices,
                    key=lambda dest: transfer_move_delta(
                        trial, p, old, dest
                    ),
                )
                trial[p] = k
                trial_loads[k] += demands[p]

            if not possible:
                continue

            if any(trial[k] != k for k in set(trial)):
                continue

            trial_obj = calculate_objective(
                trial,
                flow,
                distance,
                setup_costs,
                # access_cost already includes collection/distribution, but
                # calculate_objective needs the original rates. Closure
                # comparison is performed below by an equivalent expression.
                0.0,
                alpha,
                0.0,
            )
            trial_obj += sum(access_cost[i][trial[i]] for i in range(n))
            trial_obj += sum(setup_costs[k] for k in set(trial))

            if current_obj is None:
                current_obj = calculate_objective(
                    assignment,
                    flow,
                    distance,
                    setup_costs,
                    0.0,
                    alpha,
                    0.0,
                )
                current_obj += sum(access_cost[i][assignment[i]] for i in range(n))
                current_obj += sum(setup_costs[k] for k in set(assignment))

            if trial_obj < current_obj - 1e-7 and trial_obj < best_close_obj:
                best_close_obj = trial_obj
                best_close_assignment = trial

        if best_close_assignment is not None:
            assignment = best_close_assignment
            changed = True

    # Single-node reassignment and hub-opening moves.
    while time.monotonic() < deadline:
        loads, hubs = loads_and_hubs(assignment)
        best_delta = -1e-8
        best_move = None

        for p in range(n):
            if time.monotonic() >= deadline:
                break
            old = assignment[p]

            # An open hub cannot move away from itself in a single-node move.
            if p == old:
                destinations = []
                if p not in hubs and capacities[p] + TOL >= demands[p]:
                    destinations.append(p)
            else:
                destinations = list(hubs)
                # Node p can open itself as a new hub.
                if p not in hubs and capacities[p] + TOL >= demands[p]:
                    destinations.append(p)

            for new in destinations:
                if new == old:
                    continue
                if new in hubs:
                    if loads[new] + demands[p] > capacities[new] + TOL:
                        continue
                    fixed_delta = 0.0
                else:
                    if new != p:
                        continue
                    fixed_delta = setup_costs[new]

                delta_value = fixed_delta + transfer_move_delta(
                    assignment, p, old, new
                )
                if delta_value < best_delta:
                    best_delta = delta_value
                    best_move = (p, new)

        if best_move is None:
            break

        p, new = best_move
        assignment[p] = new

    return assignment


def select_candidates(
    n,
    assignment,
    demands,
    capacities,
    setup_costs,
    access_cost,
    max_candidates,
):
    incumbent_hubs = sorted(set(assignment))
    target = max(len(incumbent_hubs), min(n, max_candidates))

    viable = [k for k in range(n) if capacities[k] + TOL >= demands[k]]

    central_score = []
    for k in viable:
        total_access = sum(access_cost[i][k] for i in range(n))
        score = setup_costs[k] + total_access
        central_score.append((score, k))

    capacity_score = sorted(
        viable,
        key=lambda k: (
            setup_costs[k] / max(capacities[k], 1e-9),
            -capacities[k],
            k,
        ),
    )
    central_score.sort()

    chosen = set(incumbent_hubs)
    central_order = [k for _, k in central_score]

    a = b = 0
    while len(chosen) < target and (a < len(central_order) or b < len(capacity_score)):
        if a < len(central_order):
            chosen.add(central_order[a])
            a += 1
            if len(chosen) >= target:
                break
        if b < len(capacity_score):
            chosen.add(capacity_score[b])
            b += 1

    return sorted(chosen)


def solve_exact_restricted(
    candidates,
    n,
    demands,
    incoming,
    capacities,
    total_flow,
    setup_costs,
    flow,
    distance,
    chi,
    alpha,
    delta,
    time_limit,
    incumbent_solution,
    logger,
):
    if time_limit <= 0.02:
        return None

    C = list(candidates)
    model = gp.Model("single_allocation_hub_location")
    configure_model(model)
    model.Params.TimeLimit = max(0.01, float(time_limit))

    x = model.addVars(range(n), C, vtype=GRB.BINARY, name="x")
    q = model.addVars(range(n), C, C, lb=0.0, vtype=GRB.CONTINUOUS, name="q")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in C) == 1)

    for i in range(n):
        for k in C:
            if i != k:
                model.addConstr(x[i, k] <= x[k, k])

    for k in C:
        model.addConstr(
            gp.quicksum(demands[i] * x[i, k] for i in range(n))
            <= capacities[k] * x[k, k]
        )

    model.addConstr(
        gp.quicksum(capacities[k] * x[k, k] for k in C) >= total_flow
    )

    # For every origin i, q[i,k,m] is the amount of i's traffic transported
    # from origin hub k to destination hub m.
    for i in range(n):
        for k in C:
            model.addConstr(
                gp.quicksum(q[i, k, m] for m in C) == demands[i] * x[i, k]
            )
        for m in C:
            model.addConstr(
                gp.quicksum(q[i, k, m] for k in C)
                == gp.quicksum(flow[i][j] * x[j, m] for j in range(n))
            )

    obj = gp.LinExpr()
    for k in C:
        obj += setup_costs[k] * x[k, k]

    for i in range(n):
        for k in C:
            access = (
                chi * demands[i] * distance[i][k]
                + delta * incoming[i] * distance[k][i]
            )
            if access:
                obj += access * x[i, k]

    for i in range(n):
        for k in C:
            for m in C:
                coefficient = alpha * distance[k][m]
                if coefficient:
                    obj += coefficient * q[i, k, m]

    model.setObjective(obj, GRB.MINIMIZE)

    if incumbent_solution is not None:
        start_assignment = incumbent_solution["assignment"]
        if all(h in C for h in start_assignment):
            for i in range(n):
                ai = start_assignment[i]
                for k in C:
                    x[i, k].Start = 1.0 if k == ai else 0.0

    x_keys = [(i, k) for i in range(n) for k in C]
    x_vars = [x[key] for key in x_keys]
    callback_best = [incumbent_solution["objective_value"]
                     if incumbent_solution else math.inf]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = cb_model.cbGetSolution(x_vars)
            by_node = [[] for _ in range(n)]
            for idx, (i, k) in enumerate(x_keys):
                by_node[i].append((values[idx], k))
            assignment = [
                max(by_node[i], key=lambda pair: pair[0])[1] for i in range(n)
            ]
            if not is_feasible_assignment(
                assignment, demands, capacities, total_flow
            ):
                return
            solution = make_solution(
                assignment,
                flow,
                distance,
                setup_costs,
                chi,
                alpha,
                delta,
            )
            if solution["objective_value"] < callback_best[0] - 1e-7:
                callback_best[0] = solution["objective_value"]
                if logger:
                    logger.log_solution(solution["objective_value"], solution)
        except Exception:
            # Exceptions must not terminate Gurobi's search.
            return

    model.optimize(callback)

    if model.SolCount == 0:
        return None

    assignment = []
    for i in range(n):
        assignment.append(max(C, key=lambda k: x[i, k].X))

    if not is_feasible_assignment(assignment, demands, capacities, total_flow):
        return None

    return make_solution(
        assignment,
        flow,
        distance,
        setup_costs,
        chi,
        alpha,
        delta,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    flow = [[float(v) for v in row] for row in data["flow_matrix"]]
    distance = [[float(v) for v in row] for row in data["distance_matrix"]]
    setup_costs = [float(v) for v in data["setup_costs"]]
    capacities = [float(v) for v in data["capacities"]]

    if "outgoing_flow" in data:
        demands = [float(v) for v in data["outgoing_flow"]]
    else:
        demands = [sum(row) for row in flow]

    incoming = [sum(flow[i][j] for i in range(n)) for j in range(n)]
    total_flow = float(data.get("total_flow_D", sum(demands)))

    params = data["cost_parameters"]
    chi = float(params["collection_cost_chi"])
    alpha = float(params["transfer_cost_alpha"])
    delta = float(params["distribution_cost_delta"])

    access_cost = [
        [
            chi * demands[i] * distance[i][k]
            + delta * incoming[i] * distance[k][i]
            for k in range(n)
        ]
        for i in range(n)
    ]

    incumbent = None

    greedy = greedy_initial_solution(demands, capacities, access_cost)
    if greedy is not None and is_feasible_assignment(
        greedy, demands, capacities, total_flow
    ):
        incumbent = log_if_better(
            make_solution(
                greedy, flow, distance, setup_costs, chi, alpha, delta
            ),
            incumbent,
            logger,
        )

    remaining = max(0.0, deadline - time.monotonic())
    if remaining > 0.05:
        if incumbent is None:
            surrogate_budget = min(remaining * 0.45, max(1.0, remaining - 0.1))
        else:
            surrogate_budget = min(15.0, max(0.15, remaining * 0.15))

        surrogate_assignment = solve_surrogate(
            n,
            demands,
            capacities,
            total_flow,
            setup_costs,
            access_cost,
            min(surrogate_budget, max(0.0, deadline - time.monotonic())),
            greedy,
        )
        if surrogate_assignment is not None:
            surrogate_solution = make_solution(
                surrogate_assignment,
                flow,
                distance,
                setup_costs,
                chi,
                alpha,
                delta,
            )
            incumbent = log_if_better(surrogate_solution, incumbent, logger)

    if incumbent is not None:
        remaining = max(0.0, deadline - time.monotonic())
        local_budget = min(3.0, remaining * 0.08)
        if local_budget > 0.02:
            improved_assignment = local_search(
                incumbent["assignment"],
                demands,
                capacities,
                flow,
                distance,
                setup_costs,
                access_cost,
                alpha,
                time.monotonic() + local_budget,
            )
            if is_feasible_assignment(
                improved_assignment, demands, capacities, total_flow
            ):
                improved_solution = make_solution(
                    improved_assignment,
                    flow,
                    distance,
                    setup_costs,
                    chi,
                    alpha,
                    delta,
                )
                incumbent = log_if_better(improved_solution, incumbent, logger)

    remaining = max(0.0, deadline - time.monotonic())
    if incumbent is not None and remaining > 0.05:
        # Keep the exact formulation at a manageable size. It has
        # n * |C|^2 continuous inter-hub flow variables.
        variable_budget = 450000
        max_candidates = max(
            len(incumbent["hubs"]),
            int(math.sqrt(max(1, variable_budget // max(1, n)))),
        )
        max_candidates = min(n, max(8, min(40, max_candidates)))

        candidates = select_candidates(
            n,
            incumbent["assignment"],
            demands,
            capacities,
            setup_costs,
            access_cost,
            max_candidates,
        )

        exact_solution = solve_exact_restricted(
            candidates,
            n,
            demands,
            incoming,
            capacities,
            total_flow,
            setup_costs,
            flow,
            distance,
            chi,
            alpha,
            delta,
            max(0.01, deadline - time.monotonic()),
            incumbent,
            logger,
        )
        incumbent = log_if_better(exact_solution, incumbent, logger)

    if incumbent is None:
        # Last-resort feasibility solve. This path is normally needed only
        # when the greedy bin packing failed and the initial time budget was
        # extremely small.
        remaining = max(0.01, deadline - time.monotonic())
        assignment = solve_surrogate(
            n,
            demands,
            capacities,
            total_flow,
            setup_costs,
            access_cost,
            remaining,
            None,
        )
        if assignment is None:
            raise RuntimeError("No feasible hub assignment could be found.")
        incumbent = make_solution(
            assignment, flow, distance, setup_costs, chi, alpha, delta
        )
        if logger:
            logger.log_solution(incumbent["objective_value"], incumbent)

    # Recompute the objective to ensure the emitted value exactly corresponds
    # to the discrete assignment in the output.
    incumbent = make_solution(
        incumbent["assignment"],
        flow,
        distance,
        setup_costs,
        chi,
        alpha,
        delta,
    )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(incumbent, f, separators=(",", ":"))


if __name__ == "__main__":
    main()