import argparse
import json
import math
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_in_out_flows(demands: List[List[float]]) -> Tuple[List[float], List[float]]:
    n = len(demands)
    out_flow = [0.0] * n
    in_flow = [0.0] * n
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            v = float(demands[i][j])
            row_sum += v
            in_flow[j] += v
        out_flow[i] = row_sum
    return out_flow, in_flow


def evaluate_objective(
    n: int,
    scenarios: List[dict],
    distances: List[List[float]],
    chi: float,
    alpha: float,
    delta: float,
    fixed_costs: List[float],
    p_hubs: Optional[int],
    z_vals: List[int],
    allocations: Dict[int, List[int]],
) -> float:
    fixed_part = 0.0 if p_hubs is not None else sum(fixed_costs[i] * z_vals[i] for i in range(n))

    expected = 0.0
    for s_idx, sc in enumerate(scenarios):
        prob = float(sc["probability"])
        dem = sc["demands"]
        out_f, in_f = compute_in_out_flows(dem)
        alloc = allocations[s_idx]

        cd_cost = 0.0
        for i in range(n):
            h = alloc[i]
            cd_cost += distances[i][h] * (chi * out_f[i] + delta * in_f[i])

        tr_cost = 0.0
        if alpha != 0.0:
            for i in range(n):
                hi = alloc[i]
                row = dem[i]
                for j in range(n):
                    w = float(row[j])
                    if w == 0.0:
                        continue
                    hj = alloc[j]
                    tr_cost += alpha * w * distances[hi][hj]

        expected += prob * (cd_cost + tr_cost)

    return fixed_part + expected


def build_fallback_solution(instance: dict) -> Tuple[List[int], Dict[int, List[int]], List[int]]:
    n = int(instance["n"])
    p_hubs = instance.get("p_hubs", None)
    fixed_costs = [float(v) for v in instance["fixed_costs"]]
    capacities = instance.get("hub_capacities", None)
    scenarios = instance["scenarios"]
    distances = instance["distances"]

    # Choose hubs
    order = sorted(range(n), key=lambda i: fixed_costs[i])
    if p_hubs is not None:
        k = int(p_hubs)
        hubs = order[:k]
    else:
        hubs = list(range(n))  # safest fallback for feasibility in uncapacitated / general case

    z = [0] * n
    for h in hubs:
        z[h] = 1

    # Scenario-specific allocations (greedy nearest with optional capacities)
    allocations = {}
    for s_idx, sc in enumerate(scenarios):
        dem = sc["demands"]
        out_f, _ = compute_in_out_flows(dem)

        alloc = [-1] * n
        for h in hubs:
            alloc[h] = h

        remaining = None
        if capacities is not None:
            remaining = {h: float(capacities[h]) for h in hubs}
            for h in hubs:
                remaining[h] -= out_f[h]

        non_hubs = [i for i in range(n) if z[i] == 0]
        non_hubs.sort(key=lambda i: out_f[i], reverse=True)

        for i in non_hubs:
            candidates = sorted(hubs, key=lambda h: distances[i][h])
            chosen = None
            if remaining is not None:
                for h in candidates:
                    if remaining[h] + 1e-9 >= out_f[i]:
                        chosen = h
                        break
            if chosen is None:
                chosen = candidates[0]
            alloc[i] = chosen
            if remaining is not None:
                remaining[chosen] -= out_f[i]

        allocations[s_idx] = alloc

    return hubs, allocations, z


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger]):
    start_time = time.time()

    n = int(instance["n"])
    scenarios = instance["scenarios"]
    S = len(scenarios)

    chi = float(instance["cost_parameters"]["chi"])
    alpha = float(instance["cost_parameters"]["alpha"])
    delta = float(instance["cost_parameters"]["delta"])

    p_hubs = instance.get("p_hubs", None)
    fixed_costs = [float(v) for v in instance["fixed_costs"]]
    distances = instance["distances"]
    capacities = instance.get("hub_capacities", None)

    # Precompute scenario data
    probs = []
    out_flows = []
    in_flows = []
    od_positive = []  # list per scenario: (i,j,w)
    for sc in scenarios:
        probs.append(float(sc["probability"]))
        dem = sc["demands"]
        out_f, in_f = compute_in_out_flows(dem)
        out_flows.append(out_f)
        in_flows.append(in_f)

        pairs = []
        if alpha != 0.0:
            for i in range(n):
                row = dem[i]
                for j in range(n):
                    w = float(row[j])
                    if w > 0.0:
                        pairs.append((i, j, w))
        od_positive.append(pairs)

    # Build model
    model = gp.Model("stochastic_hub_location")
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    elapsed = time.time() - start_time
    remaining = max(1, int(time_limit - elapsed))
    model.Params.TimeLimit = remaining
    model.Params.MIPFocus = 1

    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    y = model.addVars(S, n, n, vtype=GRB.BINARY, name="y")

    # Hub count constraints
    if p_hubs is not None:
        model.addConstr(gp.quicksum(z[i] for i in range(n)) == int(p_hubs), name="exact_hubs")
    else:
        model.addConstr(gp.quicksum(z[i] for i in range(n)) >= 1, name="at_least_one_hub")

    # Assignment constraints
    for s in range(S):
        for i in range(n):
            model.addConstr(gp.quicksum(y[s, i, h] for h in range(n)) == 1, name=f"assign_{s}_{i}")
            model.addConstr(y[s, i, i] == z[i], name=f"self_if_hub_{s}_{i}")
            for h in range(n):
                model.addConstr(y[s, i, h] <= z[h], name=f"open_only_{s}_{i}_{h}")

    # Capacities (if present)
    if capacities is not None:
        for s in range(S):
            out_f = out_flows[s]
            for h in range(n):
                model.addConstr(
                    gp.quicksum(out_f[i] * y[s, i, h] for i in range(n)) <= float(capacities[h]) * z[h],
                    name=f"cap_{s}_{h}",
                )

    # Objective: fixed + expected collection/distribution + expected transfer
    obj = gp.LinExpr()

    if p_hubs is None:
        obj += gp.quicksum(fixed_costs[i] * z[i] for i in range(n))

    # collection & distribution
    for s in range(S):
        prob = probs[s]
        out_f = out_flows[s]
        in_f = in_flows[s]
        for i in range(n):
            node_weight = prob * (chi * out_f[i] + delta * in_f[i])
            if node_weight == 0.0:
                continue
            for h in range(n):
                d = float(distances[i][h])
                if d != 0.0:
                    obj += node_weight * d * y[s, i, h]

    # transfer via linearization u = y_i_k * y_j_m for positive-demand OD pairs
    u = {}
    if alpha != 0.0:
        for s in range(S):
            prob = probs[s]
            for (i, j, w) in od_positive[s]:
                base = prob * alpha * w
                if base == 0.0:
                    continue
                for k in range(n):
                    yik = y[s, i, k]
                    for m in range(n):
                        dkm = float(distances[k][m])
                        if dkm == 0.0:
                            continue
                        key = (s, i, j, k, m)
                        uv = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"u_{s}_{i}_{j}_{k}_{m}")
                        u[key] = uv
                        model.addConstr(uv <= yik, name=f"u1_{s}_{i}_{j}_{k}_{m}")
                        model.addConstr(uv <= y[s, j, m], name=f"u2_{s}_{i}_{j}_{k}_{m}")
                        model.addConstr(uv >= yik + y[s, j, m] - 1, name=f"u3_{s}_{i}_{j}_{k}_{m}")
                        obj += base * dkm * uv

    model.setObjective(obj, GRB.MINIMIZE)

    # Logging callback
    best_logged = [float("inf")]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            cur = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if cur + 1e-9 < best_logged[0]:
                best_logged[0] = cur
                if logger:
                    logger.log(float(cur))

    model.optimize(cb)

    # If no feasible solution found, use fallback
    if model.SolCount == 0:
        hubs, allocs, z_int = build_fallback_solution(instance)
        obj_val = evaluate_objective(
            n=n,
            scenarios=scenarios,
            distances=distances,
            chi=chi,
            alpha=alpha,
            delta=delta,
            fixed_costs=fixed_costs,
            p_hubs=p_hubs,
            z_vals=z_int,
            allocations=allocs,
        )
        if logger:
            logger.log(float(obj_val))
        return obj_val, hubs, allocs, [float(v) for v in z_int]

    # Extract incumbent
    z_vals = [1.0 if z[i].X > 0.5 else 0.0 for i in range(n)]
    hubs = [i for i in range(n) if z_vals[i] > 0.5]

    allocations = {}
    for s in range(S):
        alloc = []
        for i in range(n):
            best_h = 0
            best_v = -1.0
            for h in range(n):
                v = y[s, i, h].X
                if v > best_v:
                    best_v = v
                    best_h = h
            alloc.append(int(best_h))
        allocations[s] = alloc

    # Re-evaluate objective from extracted solution for robust output consistency
    obj_val = evaluate_objective(
        n=n,
        scenarios=scenarios,
        distances=distances,
        chi=chi,
        alpha=alpha,
        delta=delta,
        fixed_costs=fixed_costs,
        p_hubs=p_hubs,
        z_vals=[int(round(v)) for v in z_vals],
        allocations=allocations,
    )

    if logger and obj_val + 1e-9 < best_logged[0]:
        logger.log(float(obj_val))

    return obj_val, hubs, allocations, z_vals


def to_output_json(
    n: int,
    num_scenarios: int,
    hubs: List[int],
    allocations_raw: Dict[int, List[int]],
    z_values: List[float],
    objective_value: float,
):
    allocations = {}
    x_values = {}

    hub_set = set(hubs)

    for s in range(num_scenarios):
        alloc = allocations_raw[s]
        s_key = str(s)

        allocations[s_key] = {str(i): int(alloc[i]) for i in range(n)}

        x_values[s_key] = {}
        for i in range(n):
            inner = {}
            assigned_h = int(alloc[i])
            for h in range(n):
                if h == i:
                    continue
                val = 1.0 if (i not in hub_set and assigned_h == h) else 0.0
                inner[str(h)] = float(val)
            x_values[s_key][str(i)] = inner

    return {
        "objective_value": float(objective_value),
        "hubs": [int(h) for h in hubs],
        "allocations": allocations,
        "z_values": [float(v) for v in z_values],
        "x_values": x_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    objective_value, hubs, allocations_raw, z_values = solve(instance, args.time_limit, logger)

    sol = to_output_json(
        n=int(instance["n"]),
        num_scenarios=int(instance["num_scenarios"]),
        hubs=hubs,
        allocations_raw=allocations_raw,
        z_values=z_values,
        objective_value=objective_value,
    )

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()