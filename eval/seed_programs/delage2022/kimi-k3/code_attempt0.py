import argparse
import json
import math

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.instance_path) as f:
        inst = json.load(f)

    problem_type = inst.get("problem_type", "")
    closed = (problem_type == "closed_ended_cscp") or bool(inst.get("x_i_forced_zero", False))
    knife_dep = problem_type == "cscp_with_knife_dependent_setups"

    L = int(inst["machine_length"])
    P = int(inst["num_patterns"])
    M = int(inst["M"])
    c_setup = float(inst["cost_setup_pattern"])
    c_rep = float(inst["cost_repetition"])
    c_knife = float(inst.get("cost_knife_setup", 0.0)) if knife_dep else 0.0

    products = inst["products"]
    I = len(products)
    ids = [int(pr["id"]) for pr in products]
    lengths = [int(pr["length"]) for pr in products]
    demands = [int(pr["demand"]) for pr in products]
    Ncap = [int(pr["N_i"]) for pr in products]

    # Per-product upper bound on pieces per pattern (producing more than demand is useless).
    A = []
    for i in range(I):
        fit = L // lengths[i] if lengths[i] > 0 else Ncap[i]
        A.append(max(0, min(Ncap[i], demands[i], fit)))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ------------------------------------------------------------------
    # Greedy constructive heuristic (used for a MIP start and as fallback).
    # ------------------------------------------------------------------
    def greedy():
        rem = list(demands)
        pats = []  # list of [pieces_dict, repetitions]
        order = sorted(range(I), key=lambda i: -lengths[i])
        for _ in range(P):
            if all(v <= 0 for v in rem):
                break
            load = 0
            adict = {}
            for i in order:
                if rem[i] <= 0:
                    continue
                room = (L - load) // lengths[i] if lengths[i] > 0 else A[i]
                cap = min(A[i], room)
                if cap <= 0:
                    continue
                adict[i] = cap
                load += lengths[i] * cap
            if not adict:
                break
            rr = max(math.ceil(rem[i] / adict[i]) for i in adict if adict[i] > 0)
            rr = min(rr, M)
            if rr <= 0:
                break
            for i in adict:
                rem[i] = max(0, rem[i] - adict[i] * rr)
            pats.append([adict, rr])
        extras = [0] * I
        if any(v > 0 for v in rem):
            if closed:
                return None
            need = sum(rem)
            total_r = sum(rr for _, rr in pats)
            for j in range(len(pats)):
                if total_r >= need:
                    break
                bump = min(M - pats[j][1], need - total_r)
                if bump > 0:
                    pats[j][1] += bump
                    total_r += bump
            while total_r < need and len(pats) < P:
                add = min(M, need - total_r)
                pats.append([{}, add])
                total_r += add
            if total_r < need:
                return None
            extras = list(rem)
        pats.sort(key=lambda t: -t[1])
        return pats, extras

    # ------------------------------------------------------------------
    # Solution dictionary builder (from raw variable-value lists).
    # ------------------------------------------------------------------
    def make_solution(yv, rv, av, xv):
        yv = [int(round(v)) for v in yv]
        rv = [max(0, int(round(v))) for v in rv]
        av = [[max(0, int(round(av[p][i]))) for i in range(I)] for p in range(P)]
        xv = [max(0, int(round(v))) for v in xv]

        patterns = {}
        for p in range(P):
            pieces = {str(ids[i]): av[p][i] for i in range(I)}
            entry = {
                "used": bool(yv[p] > 0),
                "repetitions": rv[p],
                "pieces": pieces,
            }
            if knife_dep:
                entry["knife_count"] = int(sum(av[p]))
            patterns[str(p)] = entry

        extra = {str(ids[i]): xv[i] for i in range(I)}
        prod = {}
        for i in range(I):
            for p in range(P):
                prod["product_{}_pattern_{}".format(ids[i], p)] = av[p][i] * rv[p]

        obj = c_setup * sum(1 for p in range(P) if yv[p] > 0)
        obj += c_rep * sum(rv)
        if knife_dep:
            obj += c_knife * sum(av[p][i] for p in range(P) for i in range(I))

        sol = {
            "objective_value": float(obj),
            "patterns": patterns,
            "extra_pieces": extra,
            "production_quantities": prod,
        }
        return sol, float(obj)

    # ------------------------------------------------------------------
    # Build the MIP.
    # q[p,i] = a[p,i] * r[p] is bilinear; we linearize exactly by expanding
    # either r (binary bit expansion of the repetition count) or a (bit
    # expansion of the piece count), whichever yields fewer objects.
    # ------------------------------------------------------------------
    bits_r = M.bit_length() if M > 0 else 0
    bits_a = [A[i].bit_length() if A[i] > 0 else 0 for i in range(I)]

    cost_method_a = P * bits_r + P * I * bits_r
    cost_method_b = 2 * P * sum(bits_a)
    use_r_expansion = cost_method_a <= cost_method_b

    m = gp.Model("cscp")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, args.time_limit)

    y = m.addVars(P, vtype=GRB.BINARY, name="y")
    r = m.addVars(P, vtype=GRB.INTEGER, lb=0, ub=M, name="r")
    a = m.addVars(P, I, vtype=GRB.INTEGER, lb=0, name="a")
    for i in range(I):
        for p in range(P):
            a[p, i].UB = A[i]
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name="x")
    for i in range(I):
        x[i].UB = 0 if closed else demands[i]

    q = {}
    if use_r_expansion:
        if bits_r > 0:
            u = m.addVars(P, bits_r, vtype=GRB.BINARY, name="u")
            w = m.addVars(P, I, bits_r, vtype=GRB.CONTINUOUS, lb=0, name="w")
            for p in range(P):
                m.addConstr(r[p] == gp.quicksum((1 << b) * u[p, b] for b in range(bits_r)),
                            name="rdef_%d" % p)
            for p in range(P):
                for i in range(I):
                    Ai = A[i]
                    if Ai <= 0:
                        q[(p, i)] = 0
                        continue
                    for b in range(bits_r):
                        wb = w[p, i, b]
                        ub = u[p, b]
                        m.addConstr(wb <= Ai * ub)
                        m.addConstr(wb <= a[p, i])
                        m.addConstr(wb >= a[p, i] - Ai * (1 - ub))
                    q[(p, i)] = gp.quicksum((1 << b) * w[p, i, b] for b in range(bits_r))
        else:
            for p in range(P):
                for i in range(I):
                    q[(p, i)] = 0
    else:
        for i in range(I):
            Bi = bits_a[i]
            if Bi <= 0:
                for p in range(P):
                    q[(p, i)] = 0
                continue
            for p in range(P):
                v = m.addVars(Bi, vtype=GRB.BINARY, name="v_%d_%d" % (p, i))
                z = m.addVars(Bi, vtype=GRB.CONTINUOUS, lb=0, name="z_%d_%d" % (p, i))
                m.addConstr(a[p, i] == gp.quicksum((1 << b) * v[b] for b in range(Bi)))
                for b in range(Bi):
                    m.addConstr(z[b] <= M * v[b])
                    m.addConstr(z[b] <= r[p])
                    m.addConstr(z[b] >= r[p] - M * (1 - v[b]))
                q[(p, i)] = gp.quicksum((1 << b) * z[b] for b in range(Bi))

    # Linking, capacity, ordering, demand constraints.
    for p in range(P):
        m.addConstr(r[p] <= M * y[p], name="link_%d" % p)
        m.addConstr(gp.quicksum(lengths[i] * a[p, i] for i in range(I)) <= L * y[p],
                    name="len_%d" % p)
    for p in range(P - 1):
        m.addConstr(r[p] >= r[p + 1], name="ord_%d" % p)
    for i in range(I):
        m.addConstr(gp.quicksum(q[(p, i)] for p in range(P)) + x[i] >= demands[i],
                    name="dem_%d" % i)
    if not closed:
        m.addConstr(gp.quicksum(x[i] for i in range(I))
                    <= gp.quicksum(r[p] for p in range(P)), name="extra_cap")

    obj = c_setup * gp.quicksum(y[p] for p in range(P))
    obj += c_rep * gp.quicksum(r[p] for p in range(P))
    if knife_dep:
        obj += c_knife * gp.quicksum(a[p, i] for p in range(P) for i in range(I))
    m.setObjective(obj, GRB.MINIMIZE)

    # MIP start from the greedy heuristic.
    try:
        g = greedy()
        if g is not None:
            pats, extras = g
            for p in range(P):
                if p < len(pats):
                    adict, rr = pats[p]
                    y[p].Start = 1 if rr > 0 else 0
                    r[p].Start = rr
                    for i in range(I):
                        a[p, i].Start = adict.get(i, 0)
                else:
                    y[p].Start = 0
                    r[p].Start = 0
                    for i in range(I):
                        a[p, i].Start = 0
            for i in range(I):
                x[i].Start = extras[i] if not closed else 0
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Incumbent logging callback.
    # ------------------------------------------------------------------
    y_list = [y[p] for p in range(P)]
    r_list = [r[p] for p in range(P)]
    a_flat = [a[p, i] for p in range(P) for i in range(I)]
    x_list = [x[i] for i in range(I)]
    last_logged = {"obj": None}

    def collect(getter):
        yv = getter(y_list)
        rv = getter(r_list)
        av_flat = getter(a_flat)
        xv = getter(x_list)
        av = [av_flat[p * I:(p + 1) * I] for p in range(P)]
        return make_solution(yv, rv, av, xv)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                sol, objv = collect(lambda vs: model.cbGetSolution(vs))
                logger.log_solution(objv, sol)
                last_logged["obj"] = objv
            except Exception:
                pass

    m.optimize(callback)

    # ------------------------------------------------------------------
    # Extract the best solution (or fall back to the heuristic).
    # ------------------------------------------------------------------
    sol = None
    objv = None
    if m.SolCount > 0:
        try:
            sol, objv = collect(lambda vs: [v.X for v in vs])
        except Exception:
            sol = None

    if sol is None:
        g = greedy()
        if g is not None:
            pats, extras = g
            yv = [1 if (p < len(pats) and pats[p][1] > 0) else 0 for p in range(P)]
            rv = [pats[p][1] if p < len(pats) else 0 for p in range(P)]
            av = [[pats[p][0].get(i, 0) if p < len(pats) else 0 for i in range(I)]
                  for p in range(P)]
            xv = extras if not closed else [0] * I
            sol, objv = make_solution(yv, rv, av, xv)
        else:
            yv = [0] * P
            rv = [0] * P
            av = [[0] * I for _ in range(P)]
            xv = [0] * I
            sol, objv = make_solution(yv, rv, av, xv)

    if logger and (last_logged["obj"] is None or objv < last_logged["obj"] - 1e-9):
        logger.log_solution(objv, sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()