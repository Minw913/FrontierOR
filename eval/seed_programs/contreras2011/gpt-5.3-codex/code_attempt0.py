import argparse
import json
import time
from typing import List, Tuple, Optional

import numpy as np
from gurobipy import Model, GRB, quicksum

from solution_logger import SolutionLogger


def compute_objective(
    assignment: np.ndarray,
    setup_costs: np.ndarray,
    node_weight: np.ndarray,
    dist: np.ndarray,
    flow: np.ndarray,
    alpha: float,
) -> float:
    hubs = np.where(np.arange(len(assignment)) == assignment)[0]
    setup = float(np.sum(setup_costs[hubs]))
    lin = float(np.sum(node_weight * dist[np.arange(len(assignment)), assignment]))
    inter = float(alpha * np.sum(flow * dist[assignment[:, None], assignment[None, :]]))
    return setup + lin + inter


def greedy_initial_solution(
    n: int,
    flow: np.ndarray,
    dist: np.ndarray,
    setup: np.ndarray,
    cap: np.ndarray,
    O: np.ndarray,
    I: np.ndarray,
    chi: float,
    alpha: float,
    delta: float,
    total_D: float,
    time_limit_end: float,
    logger: Optional[SolutionLogger] = None,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    feasible_hubs = [k for k in range(n) if cap[k] >= O[k] - 1e-9]
    if not feasible_hubs:
        return None, None

    node_weight = chi * O + delta * I
    base_cost = node_weight[:, None] * dist

    # Rank hubs by setup-capacity efficiency
    order = sorted(feasible_hubs, key=lambda k: (setup[k] / max(cap[k], 1e-9), setup[k]))

    open_set = []
    cap_sum = 0.0
    for k in order:
        open_set.append(k)
        cap_sum += cap[k]
        if cap_sum + 1e-9 >= total_D:
            break
    if cap_sum + 1e-9 < total_D:
        return None, None

    def try_assign(open_hubs: List[int]) -> Optional[np.ndarray]:
        a = -np.ones(n, dtype=int)
        residual = np.zeros(n, dtype=float)

        for h in open_hubs:
            a[h] = h
            residual[h] = cap[h] - O[h]
            if residual[h] < -1e-9:
                return None

        non_hubs = [i for i in range(n) if i not in set(open_hubs)]
        non_hubs.sort(key=lambda i: O[i], reverse=True)

        for i in non_hubs:
            best_h = -1
            best_c = float("inf")
            for h in open_hubs:
                if residual[h] + 1e-9 >= O[i]:
                    c = base_cost[i, h]
                    if c < best_c:
                        best_c = c
                        best_h = h
            if best_h == -1:
                return None
            a[i] = best_h
            residual[best_h] -= O[i]

        return a

    assignment = None
    open_set_current = open_set[:]
    opened = set(open_set_current)
    for k in order:
        if time.time() > time_limit_end:
            break
        assignment = try_assign(open_set_current)
        if assignment is not None:
            break
        if k not in opened:
            open_set_current.append(k)
            opened.add(k)

    if assignment is None:
        return None, None

    obj = compute_objective(assignment, setup, node_weight, dist, flow, alpha)
    if logger:
        logger.log(float(obj))

    # Local relocate search (non-hub nodes only)
    residual = np.zeros(n, dtype=float)
    hubs = set(np.where(np.arange(n) == assignment)[0].tolist())
    for h in hubs:
        residual[h] = cap[h]
    for i in range(n):
        residual[assignment[i]] -= O[i]

    improved = True
    while improved and time.time() < time_limit_end:
        improved = False
        a = assignment.copy()
        for i in range(n):
            if time.time() > time_limit_end:
                break
            if i in hubs:
                continue
            p = a[i]
            best_q = p
            best_delta = 0.0

            row_term_p = dist[p, a]
            col_term_p = dist[a, p]
            for q in hubs:
                if q == p:
                    continue
                if residual[q] + 1e-9 < O[i]:
                    continue

                delta_lin = node_weight[i] * (dist[i, q] - dist[i, p])
                delta_inter = alpha * (
                    np.dot(flow[i, :], dist[q, a] - row_term_p)
                    + np.dot(flow[:, i], dist[a, q] - col_term_p)
                )
                delta = float(delta_lin + delta_inter)

                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_q = q

            if best_q != p:
                a[i] = best_q
                residual[p] += O[i]
                residual[best_q] -= O[i]
                obj += best_delta
                assignment = a.copy()
                improved = True
                if logger:
                    logger.log(float(obj))

    return assignment, float(obj)


def feasible_mip_fallback(
    n: int,
    dist: np.ndarray,
    setup: np.ndarray,
    cap: np.ndarray,
    O: np.ndarray,
    I: np.ndarray,
    chi: float,
    delta: float,
    total_D: float,
    time_remain: float,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if time_remain <= 1:
        return None, None

    node_weight = chi * O + delta * I
    m = Model("fallback_assign")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1.0, time_remain)
    m.Params.Threads = 1

    feasible_hub = [1 if cap[k] >= O[k] - 1e-9 else 0 for k in range(n)]

    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = m.addVars(n, vtype=GRB.BINARY, name="y")

    for k in range(n):
        if feasible_hub[k] == 0:
            y[k].UB = 0.0
            for i in range(n):
                x[i, k].UB = 0.0

    for i in range(n):
        m.addConstr(quicksum(x[i, k] for k in range(n)) == 1)

    for i in range(n):
        for k in range(n):
            m.addConstr(x[i, k] <= y[k])

    for k in range(n):
        m.addConstr(x[k, k] == y[k])

    for k in range(n):
        m.addConstr(quicksum(O[i] * x[i, k] for i in range(n)) <= cap[k] * y[k])

    m.addConstr(quicksum(cap[k] * y[k] for k in range(n)) >= total_D)

    m.setObjective(
        quicksum(setup[k] * y[k] for k in range(n))
        + quicksum(node_weight[i] * dist[i, k] * x[i, k] for i in range(n) for k in range(n)),
        GRB.MINIMIZE,
    )

    m.optimize()
    if m.SolCount == 0:
        return None, None

    a = np.zeros(n, dtype=int)
    for i in range(n):
        vals = [x[i, k].X for k in range(n)]
        a[i] = int(np.argmax(vals))
    obj_approx = float(m.ObjVal)
    return a, obj_approx


def miqp_solve(
    n: int,
    flow: np.ndarray,
    dist: np.ndarray,
    setup: np.ndarray,
    cap: np.ndarray,
    O: np.ndarray,
    I: np.ndarray,
    chi: float,
    alpha: float,
    delta: float,
    total_D: float,
    start_assignment: Optional[np.ndarray],
    time_limit: float,
    logger: Optional[SolutionLogger],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if time_limit <= 1:
        return None, None

    node_weight = chi * O + delta * I
    m = Model("CSA_HLP")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1.0, time_limit)
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.NonConvex = 2

    feasible_hub = [1 if cap[k] >= O[k] - 1e-9 else 0 for k in range(n)]
    F = [k for k in range(n) if feasible_hub[k] == 1]

    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = m.addVars(n, vtype=GRB.BINARY, name="y")

    for k in range(n):
        if feasible_hub[k] == 0:
            y[k].UB = 0.0
            for i in range(n):
                x[i, k].UB = 0.0

    for i in range(n):
        m.addConstr(quicksum(x[i, k] for k in range(n)) == 1, name=f"assign_{i}")

    for i in range(n):
        for k in range(n):
            m.addConstr(x[i, k] <= y[k])

    for k in range(n):
        m.addConstr(x[k, k] == y[k], name=f"self_{k}")

    for k in range(n):
        m.addConstr(quicksum(O[i] * x[i, k] for i in range(n)) <= cap[k] * y[k], name=f"cap_{k}")

    m.addConstr(quicksum(cap[k] * y[k] for k in range(n)) >= total_D, name="total_cap")

    lin = quicksum(setup[k] * y[k] for k in range(n))
    lin += quicksum(node_weight[i] * dist[i, k] * x[i, k] for i in range(n) for k in range(n))

    quad = 0
    for i in range(n):
        for j in range(n):
            wij = flow[i, j]
            if abs(wij) <= 1e-12:
                continue
            coeff = alpha * wij
            for k in F:
                xik = x[i, k]
                for m2 in F:
                    dkm = dist[k, m2]
                    if abs(dkm) <= 1e-12:
                        continue
                    quad += coeff * dkm * xik * x[j, m2]

    m.setObjective(lin + quad, GRB.MINIMIZE)

    if start_assignment is not None:
        hubs = set()
        for i in range(n):
            k = int(start_assignment[i])
            if 0 <= k < n and feasible_hub[k] == 1:
                x[i, k].Start = 1.0
                hubs.add(k)
        for k in range(n):
            y[k].Start = 1.0 if k in hubs else 0.0

    cb_best = [float("inf")]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < cb_best[0]:
                cb_best[0] = obj
                logger.log(float(obj))

    m.optimize(cb)

    if m.SolCount == 0:
        return None, None

    a = np.zeros(n, dtype=int)
    for i in range(n):
        vals = [x[i, k].X for k in range(n)]
        a[i] = int(np.argmax(vals))
    return a, float(m.ObjVal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    t0 = time.time()
    deadline = t0 + args.time_limit
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    cp = inst["cost_parameters"]
    chi = float(cp["collection_cost_chi"])
    alpha = float(cp["transfer_cost_alpha"])
    delta = float(cp["distribution_cost_delta"])

    flow = np.array(inst["flow_matrix"], dtype=float)
    dist = np.array(inst["distance_matrix"], dtype=float)
    setup = np.array(inst["setup_costs"], dtype=float)
    cap = np.array(inst["capacities"], dtype=float)
    O = np.array(inst["outgoing_flow"], dtype=float)

    # Incoming flow (schema doesn't explicitly include; compute)
    I = np.sum(flow, axis=0)
    total_D = float(inst["total_flow_D"])

    best_assignment = None
    best_obj = float("inf")

    # 1) Greedy + local search
    heuristic_end = min(deadline - 1.0, t0 + max(1.0, 0.35 * args.time_limit))
    a_heur, obj_heur = greedy_initial_solution(
        n, flow, dist, setup, cap, O, I, chi, alpha, delta, total_D, heuristic_end, logger
    )

    if a_heur is None and time.time() < deadline - 1.0:
        a_fb, _ = feasible_mip_fallback(
            n, dist, setup, cap, O, I, chi, delta, total_D, deadline - time.time() - 0.5
        )
        if a_fb is not None:
            node_weight = chi * O + delta * I
            obj_fb = compute_objective(a_fb, setup, node_weight, dist, flow, alpha)
            a_heur, obj_heur = a_fb, obj_fb
            if logger:
                logger.log(float(obj_fb))

    if a_heur is not None and obj_heur is not None and obj_heur < best_obj:
        best_assignment = a_heur.copy()
        best_obj = float(obj_heur)

    # 2) MIQP polishing on small/medium instances
    time_left = deadline - time.time()
    if n <= 25 and time_left > 2.0:
        a_mip, obj_mip = miqp_solve(
            n, flow, dist, setup, cap, O, I, chi, alpha, delta, total_D,
            best_assignment, time_left - 0.2, logger
        )
        if a_mip is not None:
            node_weight = chi * O + delta * I
            true_obj = compute_objective(a_mip, setup, node_weight, dist, flow, alpha)
            if true_obj < best_obj:
                best_obj = float(true_obj)
                best_assignment = a_mip.copy()
                if logger:
                    logger.log(float(best_obj))

    # Final safety fallback
    if best_assignment is None:
        # last chance linear fallback
        a_fb, _ = feasible_mip_fallback(
            n, dist, setup, cap, O, I, chi, delta, total_D, max(1.0, deadline - time.time())
        )
        if a_fb is None:
            # Infeasible instance fallback (shouldn't happen for valid data)
            best_assignment = np.zeros(n, dtype=int)
            best_obj = float("inf")
        else:
            node_weight = chi * O + delta * I
            best_assignment = a_fb
            best_obj = compute_objective(a_fb, setup, node_weight, dist, flow, alpha)

    assignment_list = [int(x) for x in best_assignment.tolist()]
    hubs = sorted([i for i in range(n) if assignment_list[i] == i])

    # recompute objective for consistency
    node_weight = chi * O + delta * I
    objective_value = float(compute_objective(best_assignment, setup, node_weight, dist, flow, alpha))

    out = {
        "objective_value": objective_value,
        "hubs": hubs,
        "assignment": assignment_list,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()