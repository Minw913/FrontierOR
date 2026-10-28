import argparse
import json
import time
import random
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_no_wait_lags(processing_times: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    # processing_times shape: [n_machines][n_jobs]
    m = len(processing_times)
    n = len(processing_times[0]) if m > 0 else 0

    # cum[j][k] = sum_{h<k} p[h][j]
    cum = [[0] * m for _ in range(n)]
    p_tot = [0] * n
    for j in range(n):
        s = 0
        for k in range(m):
            cum[j][k] = s
            s += processing_times[k][j]
        p_tot[j] = s

    # d[i][j] minimum start-time difference on machine 1 if i is immediately before j
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lag = -10**18
            for k in range(m):
                val = cum[i][k] + processing_times[k][i] - cum[j][k]
                if val > lag:
                    lag = val
            d[i][j] = lag
    return d, p_tot


def evaluate_sequence(
    seq: List[int],
    d: List[List[int]],
    p_tot: List[int],
    due_dates: List[int],
) -> Tuple[bool, float, List[float]]:
    n = len(p_tot)
    s = [0.0] * n
    if not seq:
        return True, 0.0, s

    for idx, j in enumerate(seq):
        if idx == 0:
            s[j] = 0.0
        else:
            i = seq[idx - 1]
            s[j] = s[i] + d[i][j]
        if s[j] + p_tot[j] > due_dates[j] + 1e-9:
            return False, float("inf"), s

    cmax = max(s[j] + p_tot[j] for j in seq)
    return True, float(cmax), s


def insertion_construct(
    order: List[int],
    d: List[List[int]],
    p_tot: List[int],
    due_dates: List[int],
    time_deadline: float,
) -> Tuple[Optional[List[int]], float]:
    seq: List[int] = []
    best_obj = float("inf")

    for job in order:
        if time.time() >= time_deadline:
            break
        best_seq = None
        best_val = float("inf")
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            feas, obj, _ = evaluate_sequence(cand, d, p_tot, due_dates)
            if feas and obj < best_val - 1e-9:
                best_val = obj
                best_seq = cand
        if best_seq is None:
            return None, float("inf")
        seq = best_seq
        best_obj = best_val

    if len(seq) != len(order):
        return None, float("inf")
    return seq, best_obj


def local_search(
    seq: List[int],
    d: List[List[int]],
    p_tot: List[int],
    due_dates: List[int],
    time_deadline: float,
) -> Tuple[List[int], float]:
    feas, cur_obj, _ = evaluate_sequence(seq, d, p_tot, due_dates)
    if not feas:
        return seq, float("inf")

    n = len(seq)
    improved = True
    while improved and time.time() < time_deadline:
        improved = False
        for i in range(n):
            if time.time() >= time_deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                cand = seq[:]
                job = cand.pop(i)
                cand.insert(j, job)
                feas2, obj2, _ = evaluate_sequence(cand, d, p_tot, due_dates)
                if feas2 and obj2 < cur_obj - 1e-9:
                    seq = cand
                    cur_obj = obj2
                    improved = True
                    break
            if improved:
                break
    return seq, cur_obj


def extract_sequence_from_arcs(y_vars, n_jobs: int, src: int, sink: int) -> List[int]:
    succ = {}
    for (i, j), var in y_vars.items():
        if var.X > 0.5:
            succ[i] = j
    seq = []
    cur = src
    visited = set()
    while cur in succ and succ[cur] != sink:
        nxt = succ[cur]
        if nxt in visited or nxt < 0 or nxt >= n_jobs:
            break
        seq.append(nxt)
        visited.add(nxt)
        cur = nxt
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    t_limit = max(1, int(args.time_limit))
    t_end = t0 + t_limit

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n_jobs"])
    m = int(inst["n_machines"])
    processing_times = inst["processing_times"]
    due_dates = inst["due_dates"]

    # Basic guards
    if n == 0:
        sol = {"objective_value": 0.0, "sequence": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    d, p_tot = compute_no_wait_lags(processing_times)

    best_seq = None
    best_obj = float("inf")

    def register_solution(seq: List[int], obj: float):
        nonlocal best_seq, best_obj
        if obj < best_obj - 1e-9:
            best_seq = seq[:]
            best_obj = float(obj)
            if logger:
                logger.log(best_obj)

    # -------- Heuristic phase --------
    heuristic_budget_end = min(t_end, t0 + max(1.0, 0.35 * t_limit))

    jobs = list(range(n))
    orders = []

    # deterministic seeds
    orders.append(sorted(jobs, key=lambda j: due_dates[j]))             # EDD
    orders.append(sorted(jobs, key=lambda j: p_tot[j]))                 # SPT by total processing
    orders.append(sorted(jobs, key=lambda j: due_dates[j] - p_tot[j]))  # minimum slack first

    # random seeds
    rnd = random.Random(0)
    while len(orders) < 12:
        arr = jobs[:]
        rnd.shuffle(arr)
        orders.append(arr)

    for order in orders:
        if time.time() >= heuristic_budget_end:
            break
        seq, obj = insertion_construct(order, d, p_tot, due_dates, heuristic_budget_end)
        if seq is None:
            continue
        seq, obj = local_search(seq, d, p_tot, due_dates, heuristic_budget_end)
        register_solution(seq, obj)

    # -------- MIP phase --------
    remaining = t_end - time.time()
    if remaining > 0.5:
        model = gp.Model("nowait_flowshop_due")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.1, remaining)
        model.Params.MIPFocus = 1

        src = n
        sink = n + 1

        # Arc variables for Hamiltonian path src -> jobs -> sink
        y = {}

        # src -> jobs
        for j in range(n):
            y[(src, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{src}_{j}")
        # jobs -> jobs
        for i in range(n):
            for j in range(n):
                if i != j:
                    y[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
        # jobs -> sink
        for i in range(n):
            y[(i, sink)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{sink}")

        s = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
        cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="cmax")
        u = model.addVars(n, lb=1.0, ub=float(n), vtype=GRB.CONTINUOUS, name="u")  # MTZ order

        # Degree constraints
        model.addConstr(gp.quicksum(y[(src, j)] for j in range(n)) == 1, name="src_out")
        model.addConstr(gp.quicksum(y[(i, sink)] for i in range(n)) == 1, name="sink_in")

        for j in range(n):
            model.addConstr(
                y[(src, j)] + gp.quicksum(y[(i, j)] for i in range(n) if i != j) == 1,
                name=f"in_{j}",
            )
            model.addConstr(
                y[(j, sink)] + gp.quicksum(y[(j, k)] for k in range(n) if k != j) == 1,
                name=f"out_{j}",
            )

        # MTZ subtour elimination (on jobs only)
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * y[(i, j)] <= n - 1, name=f"mtz_{i}_{j}")

        # Timing big-M
        max_due = max(due_dates)
        max_d = max(d[i][j] for i in range(n) for j in range(n) if i != j) if n > 1 else 0
        M = float(max_due + max_d + max(p_tot) + 10)

        # If src->j, enforce s[j] = 0
        for j in range(n):
            model.addConstr(s[j] <= M * (1 - y[(src, j)]), name=f"src_zero_ub_{j}")
            # s[j] >= 0 already by bound

        # Successor timing constraints
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(
                        s[j] >= s[i] + d[i][j] - M * (1 - y[(i, j)]),
                        name=f"time_{i}_{j}",
                    )

        # Due dates + cmax
        for j in range(n):
            model.addConstr(s[j] + p_tot[j] <= due_dates[j], name=f"due_{j}")
            model.addConstr(cmax >= s[j] + p_tot[j], name=f"cmax_{j}")

        model.setObjective(cmax, GRB.MINIMIZE)

        # Warm start from heuristic incumbent
        if best_seq is not None:
            warm_s = [0.0] * n
            _, warm_obj, warm_s = evaluate_sequence(best_seq, d, p_tot, due_dates)

            for key, var in y.items():
                var.Start = 0.0
            y[(src, best_seq[0])].Start = 1.0
            for idx in range(len(best_seq) - 1):
                y[(best_seq[idx], best_seq[idx + 1])].Start = 1.0
            y[(best_seq[-1], sink)].Start = 1.0

            for j in range(n):
                s[j].Start = warm_s[j]
                u[j].Start = float(best_seq.index(j) + 1)
            cmax.Start = warm_obj

        cb_best = [best_obj]

        def callback(model_cb, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < cb_best[0] - 1e-9:
                    cb_best[0] = obj_val
                    if logger:
                        logger.log(float(obj_val))

        model.optimize(callback)

        if model.SolCount > 0:
            mip_seq = extract_sequence_from_arcs(y, n, src, sink)
            if len(mip_seq) == n:
                feas, obj, _ = evaluate_sequence(mip_seq, d, p_tot, due_dates)
                if feas:
                    register_solution(mip_seq, obj)

    # Final fallback (if absolutely no feasible found)
    if best_seq is None:
        fallback = list(range(n))
        _, obj_fb, _ = evaluate_sequence(fallback, d, p_tot, due_dates)
        if obj_fb == float("inf"):
            # No feasible found; still output required schema
            best_seq = fallback
            best_obj = 1e18
        else:
            best_seq = fallback
            best_obj = obj_fb
            if logger:
                logger.log(best_obj)

    solution = {
        "objective_value": float(best_obj),
        "sequence": [j + 1 for j in best_seq],  # 1-based indexing
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()