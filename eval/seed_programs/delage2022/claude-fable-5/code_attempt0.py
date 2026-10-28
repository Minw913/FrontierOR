import json
import math
import time
import argparse

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_solution_dict(inst, used, reps, pieces, extras):
    """Build the output solution dict from integer decision values.

    used:   list[bool] per pattern
    reps:   list[int] per pattern
    pieces: dict (product_id, pattern_idx) -> int
    extras: dict product_id -> number
    Returns (objective_value, solution_dict)
    """
    ptype = inst["problem_type"]
    products = inst["products"]
    P = inst["num_patterns"]
    c_setup = inst["cost_setup_pattern"]
    c_rep = inst["cost_repetition"]
    c_knife = inst.get("cost_knife_setup", 0) or 0
    knife_variant = (ptype == "cscp_with_knife_dependent_setups")

    patterns_out = {}
    total_knives = 0
    n_used = 0
    total_reps = 0
    for j in range(P):
        pj = {str(p["id"]): int(pieces.get((p["id"], j), 0)) for p in products}
        kc = sum(pj.values())
        rj = int(reps[j])
        uj = bool(used[j]) or rj > 0 or kc > 0
        if uj:
            n_used += 1
        total_reps += rj
        total_knives += kc
        entry = {
            "used": uj,
            "repetitions": rj,
            "pieces": pj,
        }
        if knife_variant:
            entry["knife_count"] = int(kc)
        patterns_out[str(j)] = entry

    extras_out = {str(p["id"]): float(extras.get(p["id"], 0)) for p in products}

    prod_q = {}
    for p in products:
        for j in range(P):
            prod_q["product_{}_pattern_{}".format(p["id"], j)] = int(
                pieces.get((p["id"], j), 0)
            ) * int(reps[j])

    obj = c_setup * n_used + c_rep * total_reps
    if knife_variant:
        obj += c_knife * total_knives

    sol = {
        "objective_value": float(obj),
        "patterns": patterns_out,
        "extra_pieces": extras_out,
        "production_quantities": prod_q,
    }
    return float(obj), sol


def greedy_solution(inst):
    """Simple constructive heuristic. Returns (used, reps, pieces, extras) or None."""
    products = inst["products"]
    P = inst["num_patterns"]
    L = inst["machine_length"]
    M = inst["M"]
    ptype = inst["problem_type"]
    open_variant = ptype in ("open_ended_cscp", "cscp_with_knife_dependent_setups")

    rem = {p["id"]: p["demand"] for p in products}
    info = {p["id"]: p for p in products}

    built = []  # list of (a_dict, r)
    while any(v > 0 for v in rem.values()) and len(built) < P:
        order = sorted(
            [i for i in rem if rem[i] > 0], key=lambda i: -rem[i]
        )
        cap = L
        a = {}
        for i in order:
            li = info[i]["length"]
            if li <= 0:
                k = min(info[i]["N_i"], rem[i])
            else:
                k = min(info[i]["N_i"], cap // li, rem[i])
            k = int(k)
            if k > 0:
                a[i] = k
                cap -= k * li
        if not a:
            break
        r = 0
        for i in a:
            r = max(r, math.ceil(rem[i] / a[i]))
        r = int(min(r, M))
        if r <= 0:
            r = 1
        for i in a:
            rem[i] = max(0, rem[i] - a[i] * r)
        built.append((a, r))

    extras = {p["id"]: 0 for p in products}
    leftover = {i: v for i, v in rem.items() if v > 0}
    if leftover:
        if not open_variant:
            return None
        total_reps = sum(r for _, r in built)
        if sum(leftover.values()) > total_reps:
            return None
        for i, v in leftover.items():
            extras[i] = v

    # sort patterns by repetitions descending (symmetry ordering)
    built.sort(key=lambda t: -t[1])

    used = [False] * P
    reps = [0] * P
    pieces = {}
    for j, (a, r) in enumerate(built):
        used[j] = True
        reps[j] = r
        for i, k in a.items():
            pieces[(i, j)] = k
    return used, reps, pieces, extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    inst = load_instance(args.instance_path)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    products = inst["products"]
    P = inst["num_patterns"]
    L = inst["machine_length"]
    M = int(inst["M"])
    c_setup = inst["cost_setup_pattern"]
    c_rep = inst["cost_repetition"]
    ptype = inst["problem_type"]
    knife_variant = (ptype == "cscp_with_knife_dependent_setups")
    c_knife = inst.get("cost_knife_setup", 0) or 0
    closed = (ptype == "closed_ended_cscp") or bool(inst.get("x_i_forced_zero", False))

    pids = [p["id"] for p in products]
    length = {p["id"]: p["length"] for p in products}
    demand = {p["id"]: p["demand"] for p in products}
    N = {p["id"]: p["N_i"] for p in products}

    # ---------------- Greedy warm start ----------------
    greedy = greedy_solution(inst)
    best_sol = None
    best_obj = float("inf")
    if greedy is not None:
        g_used, g_reps, g_pieces, g_extras = greedy
        best_obj, best_sol = build_solution_dict(inst, g_used, g_reps, g_pieces, g_extras)
        if logger:
            logger.log_solution(best_obj, best_sol)

    # ---------------- Gurobi model ----------------
    m = gp.Model("cscp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.NonConvex = 2

    elapsed = time.time() - t_start
    tl = max(1.0, args.time_limit - elapsed - 3.0)
    m.Params.TimeLimit = tl

    y = m.addVars(range(P), vtype=GRB.BINARY, name="y")
    r = m.addVars(range(P), vtype=GRB.INTEGER, lb=0, ub=M, name="r")
    a = {}
    for i in pids:
        ub = min(N[i], demand[i]) if demand[i] > 0 else 0
        for j in range(P):
            a[i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"a_{i}_{j}")
    if closed:
        x = {i: m.addVar(lb=0, ub=0, name=f"x_{i}") for i in pids}
    else:
        x = {
            i: m.addVar(vtype=GRB.INTEGER, lb=0, ub=demand[i], name=f"x_{i}")
            for i in pids
        }

    for j in range(P):
        m.addConstr(gp.quicksum(length[i] * a[i, j] for i in pids) <= L)
        m.addConstr(r[j] <= M * y[j])
        m.addConstr(r[j] >= y[j])
        m.addConstr(gp.quicksum(a[i, j] for i in pids) >= y[j])
        for i in pids:
            m.addConstr(a[i, j] <= min(N[i], demand[i]) * y[j])
    for j in range(P - 1):
        m.addConstr(r[j] >= r[j + 1])

    # demand (bilinear)
    for i in pids:
        m.addQConstr(
            gp.quicksum(a[i, j] * r[j] for j in range(P)) + x[i] >= demand[i]
        )

    if not closed:
        m.addConstr(
            gp.quicksum(x[i] for i in pids)
            <= gp.quicksum(r[j] for j in range(P))
        )

    obj = c_setup * gp.quicksum(y[j] for j in range(P)) + c_rep * gp.quicksum(
        r[j] for j in range(P)
    )
    if knife_variant:
        obj = obj + c_knife * gp.quicksum(a[i, j] for i in pids for j in range(P))
    m.setObjective(obj, GRB.MINIMIZE)

    # warm start
    if greedy is not None:
        g_used, g_reps, g_pieces, g_extras = greedy
        for j in range(P):
            y[j].Start = 1 if g_used[j] else 0
            r[j].Start = g_reps[j]
            for i in pids:
                a[i, j].Start = g_pieces.get((i, j), 0)
        for i in pids:
            x[i].Start = g_extras.get(i, 0)

    # callback logging of incumbents
    m._y = y
    m._r = r
    m._a = a
    m._x = x
    state = {"best": best_obj}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                yv = model.cbGetSolution([y[j] for j in range(P)])
                rv = model.cbGetSolution([r[j] for j in range(P)])
                av = model.cbGetSolution([a[i, j] for i in pids for j in range(P)])
                xv = model.cbGetSolution([x[i] for i in pids])
            except gp.GurobiError:
                return
            used_v = [v > 0.5 for v in yv]
            reps_v = [int(round(v)) for v in rv]
            pieces_v = {}
            idx = 0
            for i in pids:
                for j in range(P):
                    val = int(round(av[idx]))
                    idx += 1
                    if val > 0:
                        pieces_v[(i, j)] = val
            extras_v = {i: int(round(v)) for i, v in zip(pids, xv)}
            obj_v, sol_v = build_solution_dict(inst, used_v, reps_v, pieces_v, extras_v)
            if obj_v < state["best"] - 1e-9:
                state["best"] = obj_v
                if logger:
                    logger.log_solution(obj_v, sol_v)

    try:
        m.optimize(cb)
    except gp.GurobiError:
        pass

    if m.SolCount > 0:
        used_v = [y[j].X > 0.5 for j in range(P)]
        reps_v = [int(round(r[j].X)) for j in range(P)]
        pieces_v = {}
        for i in pids:
            for j in range(P):
                val = int(round(a[i, j].X))
                if val > 0:
                    pieces_v[(i, j)] = val
        extras_v = {i: int(round(x[i].X)) for i in pids}
        obj_v, sol_v = build_solution_dict(inst, used_v, reps_v, pieces_v, extras_v)
        if obj_v < best_obj - 1e-9 or best_sol is None:
            best_obj, best_sol = obj_v, sol_v
            if logger:
                logger.log_solution(best_obj, best_sol)

    if best_sol is None:
        # last-resort empty structure (should not normally happen)
        used_v = [False] * P
        reps_v = [0] * P
        best_obj, best_sol = build_solution_dict(inst, used_v, reps_v, {}, {})

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()