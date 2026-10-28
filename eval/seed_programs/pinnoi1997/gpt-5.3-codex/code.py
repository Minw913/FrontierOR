import argparse
import json
import math
import random
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def task_sort_key(t):
    try:
        return (0, int(t))
    except Exception:
        return (1, str(t))


def normalize_task_id(x, task_keys_set):
    if x in task_keys_set:
        return x
    sx = str(x)
    if sx in task_keys_set:
        return sx
    return sx  # fallback


def build_topo_order(tasks, preds, rng=None):
    indeg = {t: len(preds[t]) for t in tasks}
    succs = {t: [] for t in tasks}
    for t in tasks:
        for p in preds[t]:
            succs[p].append(t)

    avail = [t for t in tasks if indeg[t] == 0]
    avail.sort(key=task_sort_key)
    order = []

    while avail:
        if rng is None:
            t = avail.pop(0)
        else:
            idx = rng.randrange(len(avail))
            t = avail.pop(idx)

        order.append(t)
        for v in succs[t]:
            indeg[v] -= 1
            if indeg[v] == 0:
                avail.append(v)

    if len(order) != len(tasks):
        return None
    return order


def greedy_assignment(
    tasks: List[str],
    preds: Dict[str, List[str]],
    p: Dict[str, int],
    earliest: Dict[str, int],
    latest: Dict[str, int],
    m: int,
    c: int,
    rng: Optional[random.Random] = None,
):
    order = build_topo_order(tasks, preds, rng)
    if order is None:
        return None, None

    rem = {s: c for s in range(1, m + 1)}
    assign = {}

    for t in order:
        pred_station = 1
        if preds[t]:
            pred_station = max(assign[pr] for pr in preds[t])

        lo = max(earliest[t], pred_station, 1)
        hi = min(latest[t], m)
        if lo > hi:
            return None, None

        candidates = [s for s in range(lo, hi + 1) if rem[s] >= p[t]]
        if not candidates:
            return None, None

        # Best-fit: pack station to reduce slack
        candidates.sort(key=lambda s: (rem[s] - p[t], s))
        s_star = candidates[0]
        assign[t] = s_star
        rem[s_star] -= p[t]

    w = max(rem.values())
    return assign, float(w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    m = int(data["num_stations"])
    c = int(data["cycle_time"])
    proc_raw = data["task_processing_times"]
    task_keys = list(proc_raw.keys())
    task_keys.sort(key=task_sort_key)
    task_set = set(task_keys)

    p = {t: int(proc_raw[t]) for t in task_keys}
    earliest_raw = data.get("earliest_station", {})
    latest_raw = data.get("latest_station", {})

    earliest = {}
    latest = {}
    for t in task_keys:
        e = int(earliest_raw.get(t, 1))
        l = int(latest_raw.get(t, m))
        earliest[t] = max(1, e)
        latest[t] = min(m, l)

    arcs_input = data.get("precedence_arcs", [])
    arcs = []
    for a in arcs_input:
        if len(a) != 2:
            continue
        i = normalize_task_id(a[0], task_set)
        j = normalize_task_id(a[1], task_set)
        if i not in task_set or j not in task_set:
            continue
        if i == j:
            continue
        arcs.append((i, j))

    # Build predecessor map
    preds = {t: [] for t in task_keys}
    for i, j in arcs:
        preds[j].append(i)

    # Optional quick heuristic for warm start / fallback
    best_heur_assign = None
    best_heur_obj = float("inf")

    def log_if_improved(val, holder):
        if val + 1e-9 < holder[0]:
            holder[0] = val
            if logger:
                logger.log(float(val))

    best_logged = [float("inf")]

    # Deterministic try
    assign, obj = greedy_assignment(task_keys, preds, p, earliest, latest, m, c, rng=None)
    if assign is not None:
        best_heur_assign, best_heur_obj = assign, obj
        log_if_improved(best_heur_obj, best_logged)

    # Randomized multistart within small budget
    heur_budget = min(2.0, max(0.0, 0.1 * args.time_limit))
    rng = random.Random(0)
    while time.time() - start_time < heur_budget:
        seed = rng.randint(1, 10**9)
        rr = random.Random(seed)
        assign, obj = greedy_assignment(task_keys, preds, p, earliest, latest, m, c, rng=rr)
        if assign is not None and obj < best_heur_obj - 1e-9:
            best_heur_assign, best_heur_obj = assign, obj
            log_if_improved(best_heur_obj, best_logged)

    # Build MIP model
    model = gp.Model("assembly_line_balance")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    elapsed = time.time() - start_time
    remaining = max(0.0, args.time_limit - elapsed)
    model.Params.TimeLimit = remaining

    # Variables
    x = {}
    allowed_stations = {}
    for t in task_keys:
        lo = max(1, earliest[t])
        hi = min(m, latest[t])
        allowed = list(range(lo, hi + 1))
        allowed_stations[t] = allowed
        if not allowed:
            # keep empty; model will be infeasible
            continue
        for s in allowed:
            x[(t, s)] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{s}")

    w = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=float(c), name="w")

    model.update()

    # Each task exactly one station
    for t in task_keys:
        vars_t = [x[(t, s)] for s in allowed_stations[t] if (t, s) in x]
        if vars_t:
            model.addConstr(gp.quicksum(vars_t) == 1, name=f"assign_{t}")
        else:
            model.addConstr(0 == 1, name=f"assign_infeas_{t}")

    # Precedence
    for i, j in arcs:
        lhs = gp.quicksum(s * x[(i, s)] for s in allowed_stations[i] if (i, s) in x)
        rhs = gp.quicksum(s * x[(j, s)] for s in allowed_stations[j] if (j, s) in x)
        model.addConstr(lhs <= rhs, name=f"prec_{i}_{j}")

    # Capacity and idle max constraints
    for s in range(1, m + 1):
        load_s = gp.quicksum(p[t] * x[(t, s)] for t in task_keys if (t, s) in x)
        model.addConstr(load_s <= c, name=f"cap_{s}")
        model.addConstr(load_s + w >= c, name=f"idlemax_{s}")

    model.setObjective(w, GRB.MINIMIZE)

    # Warm start from heuristic
    if best_heur_assign is not None:
        for (t, s), var in x.items():
            var.Start = 1.0 if best_heur_assign.get(t, None) == s else 0.0
        w.Start = best_heur_obj

    # Callback for incumbent logging
    cb_best = [best_logged[0]]

    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < cb_best[0]:
                cb_best[0] = obj_val
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    final_assign = None
    final_obj = None

    if model.SolCount > 0:
        final_obj = float(model.ObjVal)
        final_assign = {}
        for t in task_keys:
            best_s = None
            best_v = -1.0
            for s in allowed_stations[t]:
                if (t, s) in x:
                    v = x[(t, s)].X
                    if v > best_v:
                        best_v = v
                        best_s = s
            if best_s is None:
                best_s = max(1, min(m, earliest[t]))
            final_assign[t] = int(best_s)
    elif best_heur_assign is not None:
        final_obj = float(best_heur_obj)
        final_assign = {t: int(best_heur_assign[t]) for t in task_keys}
    else:
        # Last-resort schema-compliant output
        final_obj = float(c)
        final_assign = {t: int(max(1, min(m, earliest[t]))) for t in task_keys}

    output = {
        "objective_value": float(final_obj),
        "task_assignments": final_assign,
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()