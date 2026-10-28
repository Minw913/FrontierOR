import argparse
import json
import math
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


EPS = 1e-9


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


def preprocess(data):
    n = int(data["n"])
    scenarios = data["scenarios"]
    num_scenarios = len(scenarios)

    probabilities = [float(sc["probability"]) for sc in scenarios]
    demands = [
        [[float(v) for v in row] for row in sc["demands"]]
        for sc in scenarios
    ]

    outgoing = []
    incoming = []
    for s in range(num_scenarios):
        d = demands[s]
        outgoing.append([sum(d[i][j] for j in range(n)) for i in range(n)])
        incoming.append([sum(d[i][j] for i in range(n)) for j in range(n)])

    return {
        "n": n,
        "S": num_scenarios,
        "probabilities": probabilities,
        "demands": demands,
        "outgoing": outgoing,
        "incoming": incoming,
        "distances": [[float(v) for v in row] for row in data["distances"]],
        "fixed_costs": [float(v) for v in data["fixed_costs"]],
        "capacities": (
            None
            if data.get("hub_capacities") is None
            else [float(v) for v in data["hub_capacities"]]
        ),
        "p_hubs": data.get("p_hubs"),
        "chi": float(data["cost_parameters"]["chi"]),
        "alpha": float(data["cost_parameters"]["alpha"]),
        "delta": float(data["cost_parameters"]["delta"]),
    }


def objective_from_allocations(info, hubs, allocations):
    n = info["n"]
    S = info["S"]
    total = 0.0

    # In the p-hub median variant, setup costs are removed.
    if info["p_hubs"] is None:
        total += sum(info["fixed_costs"][k] for k in hubs)

    for s in range(S):
        probability = info["probabilities"][s]
        demand = info["demands"][s]
        outgoing = info["outgoing"][s]
        incoming = info["incoming"][s]
        alloc = allocations[s]

        scenario_cost = 0.0

        for i in range(n):
            k = alloc[i]
            scenario_cost += info["distances"][i][k] * (
                info["chi"] * outgoing[i] + info["delta"] * incoming[i]
            )

        if abs(info["alpha"]) > EPS:
            for i in range(n):
                hi = alloc[i]
                for j in range(n):
                    flow = demand[i][j]
                    if abs(flow) > EPS:
                        hj = alloc[j]
                        scenario_cost += (
                            info["alpha"]
                            * flow
                            * info["distances"][hi][hj]
                        )

        total += probability * scenario_cost

    return float(total)


def make_solution(info, hubs, allocations):
    n = info["n"]
    S = info["S"]
    hubs = sorted(int(k) for k in hubs)
    hub_set = set(hubs)

    objective = objective_from_allocations(info, hubs, allocations)

    allocation_output = {}
    x_output = {}

    for s in range(S):
        allocation_output[str(s)] = {
            str(i): int(allocations[s][i]) for i in range(n)
        }

        scenario_x = {}
        for i in range(n):
            scenario_x[str(i)] = {
                str(k): float(1.0 if allocations[s][i] == k else 0.0)
                for k in range(n)
                if k != i
            }
        x_output[str(s)] = scenario_x

    return {
        "objective_value": float(objective),
        "hubs": hubs,
        "allocations": allocation_output,
        "z_values": [float(1.0 if i in hub_set else 0.0) for i in range(n)],
        "x_values": x_output,
    }


def hub_ranking(info):
    n = info["n"]
    scores = []
    for k in range(n):
        score = 0.0
        if info["p_hubs"] is None:
            score += info["fixed_costs"][k]
        for s in range(info["S"]):
            p = info["probabilities"][s]
            for i in range(n):
                direct_weight = (
                    info["chi"] * info["outgoing"][s][i]
                    + info["delta"] * info["incoming"][s][i]
                )
                score += p * direct_weight * info["distances"][i][k]
        scores.append((score, k))
    scores.sort()
    return [k for _, k in scores]


def greedy_capacity_allocations(info, hubs):
    n = info["n"]
    hub_set = set(hubs)
    capacities = info["capacities"]
    allocations = []

    for s in range(info["S"]):
        loads = info["outgoing"][s]
        remaining = {k: capacities[k] - loads[k] for k in hubs}

        if any(remaining[k] < -1e-7 for k in hubs):
            return None

        alloc = [-1] * n
        for k in hubs:
            alloc[k] = k

        nonhubs = [i for i in range(n) if i not in hub_set]
        nonhubs.sort(key=lambda i: (-loads[i], i))

        for i in nonhubs:
            feasible = [k for k in hubs if remaining[k] + 1e-7 >= loads[i]]
            if not feasible:
                return None

            direct_weight = (
                info["chi"] * info["outgoing"][s][i]
                + info["delta"] * info["incoming"][s][i]
            )

            # Prefer low direct cost, then a tighter capacity fit.
            best = min(
                feasible,
                key=lambda k: (
                    direct_weight * info["distances"][i][k],
                    remaining[k] - loads[i],
                    k,
                ),
            )
            alloc[i] = best
            remaining[best] -= loads[i]

        allocations.append(alloc)

    return allocations


def uncapacitated_initial_allocations(info, hubs):
    n = info["n"]
    hub_set = set(hubs)
    allocations = []

    for s in range(info["S"]):
        alloc = [-1] * n
        for i in range(n):
            if i in hub_set:
                alloc[i] = i
            else:
                weight = (
                    info["chi"] * info["outgoing"][s][i]
                    + info["delta"] * info["incoming"][s][i]
                )
                alloc[i] = min(
                    hubs,
                    key=lambda k: (weight * info["distances"][i][k], k),
                )
        allocations.append(alloc)

    return allocations


def improve_allocations(info, hubs, allocations, max_passes=10):
    """Coordinate descent for fixed hubs, respecting capacities if present."""
    n = info["n"]
    hub_set = set(hubs)
    capacities = info["capacities"]

    for s in range(info["S"]):
        demand = info["demands"][s]
        outgoing = info["outgoing"][s]
        incoming = info["incoming"][s]
        alloc = allocations[s]

        used = None
        if capacities is not None:
            used = {k: 0.0 for k in hubs}
            for i in range(n):
                used[alloc[i]] += outgoing[i]

        for _ in range(max_passes):
            changed = False

            for i in range(n):
                if i in hub_set:
                    continue

                current = alloc[i]
                best_hub = current
                best_cost = math.inf

                for candidate in hubs:
                    if (
                        capacities is not None
                        and candidate != current
                        and used[candidate] + outgoing[i]
                        > capacities[candidate] + 1e-7
                    ):
                        continue

                    local_cost = info["distances"][i][candidate] * (
                        info["chi"] * outgoing[i]
                        + info["delta"] * incoming[i]
                    )

                    if abs(info["alpha"]) > EPS:
                        for j in range(n):
                            if j == i:
                                continue
                            hj = alloc[j]
                            if abs(demand[i][j]) > EPS:
                                local_cost += (
                                    info["alpha"]
                                    * demand[i][j]
                                    * info["distances"][candidate][hj]
                                )
                            if abs(demand[j][i]) > EPS:
                                local_cost += (
                                    info["alpha"]
                                    * demand[j][i]
                                    * info["distances"][hj][candidate]
                                )

                    if local_cost < best_cost - 1e-8 or (
                        abs(local_cost - best_cost) <= 1e-8
                        and candidate < best_hub
                    ):
                        best_cost = local_cost
                        best_hub = candidate

                if best_hub != current:
                    if capacities is not None:
                        used[current] -= outgoing[i]
                        used[best_hub] += outgoing[i]
                    alloc[i] = best_hub
                    changed = True

            if not changed:
                break

    return allocations


def construct_heuristic(info, deadline):
    n = info["n"]
    p_hubs = info["p_hubs"]
    capacities = info["capacities"]
    ranking = hub_ranking(info)

    if capacities is None:
        if p_hubs is None:
            hubs = [ranking[0]]
        else:
            p = int(p_hubs)
            hubs = sorted(ranking[:p])

        allocations = uncapacitated_initial_allocations(info, hubs)
        allocations = improve_allocations(info, hubs, allocations)
        return make_solution(info, hubs, allocations)

    eligible = [
        k
        for k in range(n)
        if all(
            info["outgoing"][s][k] <= capacities[k] + 1e-7
            for s in range(info["S"])
        )
    ]
    eligible_set = set(eligible)
    ranked_eligible = [k for k in ranking if k in eligible_set]

    candidate_sets = []
    if p_hubs is None:
        # Opening all eligible hubs is a strong feasibility choice.
        if eligible:
            candidate_sets.append(tuple(sorted(eligible)))
        for k in ranked_eligible[: min(20, len(ranked_eligible))]:
            candidate_sets.append((k,))
    else:
        p = int(p_hubs)
        if len(ranked_eligible) >= p:
            candidate_sets.append(tuple(sorted(ranked_eligible[:p])))

            # Deterministic alternatives around the ranking.
            for start in range(min(30, max(0, len(ranked_eligible) - p + 1))):
                candidate_sets.append(
                    tuple(sorted(ranked_eligible[start:start + p]))
                )

            rng = random.Random(0)
            trials = min(200, 10 + 5 * n)
            for _ in range(trials):
                if time.monotonic() >= deadline:
                    break
                candidate_sets.append(
                    tuple(sorted(rng.sample(ranked_eligible, p)))
                )

    seen = set()
    best_solution = None
    for hub_tuple in candidate_sets:
        if time.monotonic() >= deadline:
            break
        if hub_tuple in seen or not hub_tuple:
            continue
        seen.add(hub_tuple)

        hubs = list(hub_tuple)
        allocations = greedy_capacity_allocations(info, hubs)
        if allocations is None:
            continue

        allocations = improve_allocations(info, hubs, allocations)
        solution = make_solution(info, hubs, allocations)

        if (
            best_solution is None
            or solution["objective_value"]
            < best_solution["objective_value"] - 1e-7
        ):
            best_solution = solution

    return best_solution


def solution_to_internal_allocations(solution, S, n):
    return [
        [int(solution["allocations"][str(s)][str(i)]) for i in range(n)]
        for s in range(S)
    ]


def build_model(info, initial_solution=None):
    n = info["n"]
    S = info["S"]

    model = gp.Model("stochastic_hub_location")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    assignment = [
        model.addVars(n, n, vtype=GRB.BINARY, name=f"a_{s}")
        for s in range(S)
    ]

    if info["p_hubs"] is None:
        model.addConstr(gp.quicksum(z[k] for k in range(n)) >= 1)
    else:
        model.addConstr(
            gp.quicksum(z[k] for k in range(n)) == int(info["p_hubs"])
        )

    for s in range(S):
        a = assignment[s]
        for i in range(n):
            model.addConstr(gp.quicksum(a[i, k] for k in range(n)) == 1)
            model.addConstr(a[i, i] == z[i])
            for k in range(n):
                model.addConstr(a[i, k] <= z[k])

        if info["capacities"] is not None:
            for k in range(n):
                model.addConstr(
                    gp.quicksum(
                        info["outgoing"][s][i] * a[i, k]
                        for i in range(n)
                    )
                    <= info["capacities"][k]
                )

    objective = gp.LinExpr()

    if info["p_hubs"] is None:
        objective += gp.quicksum(
            info["fixed_costs"][k] * z[k] for k in range(n)
        )

    transfer_vars = []

    for s in range(S):
        probability = info["probabilities"][s]
        a = assignment[s]

        for i in range(n):
            assignment_weight = (
                info["chi"] * info["outgoing"][s][i]
                + info["delta"] * info["incoming"][s][i]
            )
            if abs(assignment_weight) > EPS:
                for k in range(n):
                    coefficient = (
                        probability
                        * assignment_weight
                        * info["distances"][i][k]
                    )
                    if abs(coefficient) > EPS:
                        objective += coefficient * a[i, k]

        positive_origins = [
            i for i in range(n) if info["outgoing"][s][i] > EPS
        ]

        if abs(info["alpha"]) <= EPS or not positive_origins:
            transfer_vars.append(None)
            continue

        q = model.addVars(
            positive_origins,
            range(n),
            range(n),
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"q_{s}",
        )
        transfer_vars.append(q)

        # q[i,k,l] is origin-i flow whose origin hub is k and
        # destination hub is l.
        for i in positive_origins:
            upper = info["outgoing"][s][i]
            for l in range(n):
                model.addConstr(
                    gp.quicksum(q[i, k, l] for k in range(n))
                    == gp.quicksum(
                        info["demands"][s][i][j] * a[j, l]
                        for j in range(n)
                    )
                )
                for k in range(n):
                    model.addConstr(q[i, k, l] <= upper * a[i, k])

                    coefficient = (
                        probability
                        * info["alpha"]
                        * info["distances"][k][l]
                    )
                    if abs(coefficient) > EPS:
                        objective += coefficient * q[i, k, l]

    model.setObjective(objective, GRB.MINIMIZE)

    if initial_solution is not None:
        hub_set = set(int(k) for k in initial_solution["hubs"])
        initial_allocations = solution_to_internal_allocations(
            initial_solution, S, n
        )

        for k in range(n):
            z[k].Start = 1.0 if k in hub_set else 0.0

        for s in range(S):
            for i in range(n):
                assigned_hub = initial_allocations[s][i]
                for k in range(n):
                    assignment[s][i, k].Start = (
                        1.0 if assigned_hub == k else 0.0
                    )

    return model, z, assignment


def extract_solution_from_values(info, z_values, assignment_values):
    n = info["n"]
    S = info["S"]

    hubs = [k for k in range(n) if z_values[k] > 0.5]
    if not hubs:
        hubs = [max(range(n), key=lambda k: z_values[k])]
    hub_set = set(hubs)

    allocations = []
    for s in range(S):
        alloc = []
        for i in range(n):
            if i in hub_set:
                alloc.append(i)
            else:
                alloc.append(max(hubs, key=lambda k: assignment_values[s][i][k]))
        allocations.append(alloc)

    return make_solution(info, hubs, allocations)


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = load_instance(args.instance_path)
    info = preprocess(data)

    best_solution = construct_heuristic(info, deadline)
    if best_solution is not None and logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    # Build and solve the exact MILP. The compact transfer formulation uses
    # O(S*n^3) continuous variables rather than O(S*n^4) routing variables.
    model = None
    try:
        model, z, assignment = build_model(info, best_solution)

        remaining = deadline - time.monotonic()
        if remaining > 0.01:
            model.Params.TimeLimit = max(0.01, remaining)

            callback_best = [(
                best_solution["objective_value"]
                if best_solution is not None
                else math.inf
            )]

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    z_vals = cb_model.cbGetSolution(z)
                    a_vals = [
                        cb_model.cbGetSolution(assignment[s])
                        for s in range(info["S"])
                    ]

                    z_list = [float(z_vals[k]) for k in range(info["n"])]
                    assignment_list = []
                    for s in range(info["S"]):
                        assignment_list.append([
                            [
                                float(a_vals[s][i, k])
                                for k in range(info["n"])
                            ]
                            for i in range(info["n"])
                        ])

                    solution = extract_solution_from_values(
                        info, z_list, assignment_list
                    )
                    value = solution["objective_value"]

                    if value < callback_best[0] - 1e-7:
                        callback_best[0] = value
                        if logger:
                            logger.log_solution(value, solution)
                except Exception:
                    # Logging must never interrupt the optimization.
                    pass

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                z_values = [float(z[k].X) for k in range(info["n"])]
                assignment_values = [
                    [
                        [float(assignment[s][i, k].X) for k in range(info["n"])]
                        for i in range(info["n"])
                    ]
                    for s in range(info["S"])
                ]
                mip_solution = extract_solution_from_values(
                    info, z_values, assignment_values
                )

                if (
                    best_solution is None
                    or mip_solution["objective_value"]
                    < best_solution["objective_value"] - 1e-7
                ):
                    best_solution = mip_solution
                    if logger:
                        logger.log_solution(
                            best_solution["objective_value"], best_solution
                        )

    except gp.GurobiError:
        # Retain the heuristic incumbent if model construction or solving fails.
        pass

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found. The instance may be infeasible "
            "or the time limit may be too small for model construction."
        )

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()