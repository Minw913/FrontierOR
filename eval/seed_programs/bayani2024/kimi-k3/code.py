import argparse
import json
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:
    HAVE_GUROBI = False

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    deadline = t_start + float(args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n_clients"])
    m = int(inst["n_servers"])
    demands = [float(v) for v in inst.get("demands", [1.0] * n)]
    if inst.get("linear_costs") is not None:
        C = [[float(v) for v in row] for row in inst["linear_costs"]]
    else:
        rates = [float(v) for v in inst["processing_rates"]]
        C = [[demands[i] * rates[j] for j in range(m)] for i in range(n)]
    raw_q = inst.get("quadratic_costs", []) or []

    # ---------------- Preprocess quadratic interactions ----------------
    # qmap[(i, j)][(i', j')] = total quadratic cost for that unordered pair.
    # lin_extra[(i, j)] collects self-pair terms (x^2 = x for binaries).
    qmap = {}
    lin_extra = {}
    for e in raw_q:
        i1, j1, i2, j2 = int(e[0]), int(e[1]), int(e[2]), int(e[3])
        q = float(e[4])
        if q == 0.0:
            continue
        if i1 == i2:
            if j1 == j2:
                key = (i1, j1)
                lin_extra[key] = lin_extra.get(key, 0.0) + q
            # same client on two servers can never be simultaneously active
            continue
        d = qmap.setdefault((i1, j1), {})
        d[(i2, j2)] = d.get((i2, j2), 0.0) + q
        d = qmap.setdefault((i2, j2), {})
        d[(i1, j1)] = d.get((i1, j1), 0.0) + q

    # Aggregated unordered pairs (canonical order) for evaluation / model build.
    q_entries = []
    for a, d in qmap.items():
        for b, q in d.items():
            if a < b:
                q_entries.append((a[0], a[1], b[0], b[1], q))

    def evaluate(ass):
        obj = 0.0
        for i in range(n):
            a = ass[i]
            obj += C[i][a] + lin_extra.get((i, a), 0.0)
        for (i1, j1, i2, j2, q) in q_entries:
            if ass[i1] == j1 and ass[i2] == j2:
                obj += q
        return obj

    def make_solution(obj, ass):
        return {
            "objective_value": float(obj),
            "assignment": {str(i): int(ass[i]) for i in range(n)},
        }

    # ---------------- Heuristic: greedy + 1-opt local search ----------------
    heur_deadline = t_start + min(0.2 * float(args.time_limit), 30.0)
    if heur_deadline > deadline - 0.5:
        heur_deadline = deadline - 0.5

    ass = [-1] * n
    order = sorted(range(n), key=lambda i: -demands[i])
    assigned = []
    timed_out = False
    for i in order:
        if timed_out or time.time() > heur_deadline:
            timed_out = True
            best_j, best_c = 0, float("inf")
            for j in range(m):
                c = C[i][j] + lin_extra.get((i, j), 0.0)
                if c < best_c:
                    best_c, best_j = c, j
            ass[i] = best_j
            continue
        best_j, best_c = 0, float("inf")
        for j in range(m):
            c = C[i][j] + lin_extra.get((i, j), 0.0)
            row = qmap.get((i, j))
            if row and assigned:
                s = 0.0
                for (ip, jp) in assigned:
                    s += row.get((ip, jp), 0.0)
                c += s
            if c < best_c:
                best_c, best_j = c, j
        ass[i] = best_j
        assigned.append((i, best_j))

    improved = True
    while improved and time.time() < heur_deadline:
        improved = False
        for i in range(n):
            if time.time() > heur_deadline:
                break
            a = ass[i]
            S = [0.0] * m
            for j in range(m):
                row = qmap.get((i, j))
                if row:
                    s = 0.0
                    for ip in range(n):
                        if ip != i:
                            s += row.get((ip, ass[ip]), 0.0)
                    S[j] = s
            cur = C[i][a] + lin_extra.get((i, a), 0.0) + S[a]
            best_j = -1
            best_v = cur - 1e-9
            for j in range(m):
                if j == a:
                    continue
                v = C[i][j] + lin_extra.get((i, j), 0.0) + S[j]
                if v < best_v:
                    best_v, best_j = v, j
            if best_j >= 0:
                ass[i] = best_j
                improved = True

    best_obj = evaluate(ass)
    best_ass = list(ass)

    if logger is not None:
        logger.log_solution(float(best_obj), make_solution(best_obj, best_ass))

    # ---------------- Exact solve with Gurobi MIQP ----------------
    remaining = deadline - time.time() - 0.3
    if HAVE_GUROBI and remaining > 0.5:
        try:
            model = gp.Model("client_server_assignment")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = remaining
            model.Params.Cutoff = best_obj - 1e-9

            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
            model.addConstrs((x.sum(i, "*") == 1 for i in range(n)), name="assign")

            obj = gp.QuadExpr()
            lcoefs, lvars = [], []
            for i in range(n):
                Ci = C[i]
                for j in range(m):
                    coef = Ci[j] + lin_extra.get((i, j), 0.0)
                    if coef != 0.0:
                        lcoefs.append(coef)
                        lvars.append(x[i, j])
            obj.addTerms(lcoefs, lvars)
            if q_entries:
                qcoefs = [e[4] for e in q_entries]
                qv1 = [x[e[0], e[1]] for e in q_entries]
                qv2 = [x[e[2], e[3]] for e in q_entries]
                obj.addTerms(qcoefs, qv1, qv2)
            model.setObjective(obj, GRB.MINIMIZE)

            # MIP start from heuristic
            for i in range(n):
                ai = best_ass[i]
                for j in range(m):
                    x[i, j].Start = 1.0 if j == ai else 0.0

            xvars = [x[i, j] for i in range(n) for j in range(m)]
            state = {"obj": best_obj, "ass": best_ass}

            def decode(vals):
                cur = [0] * n
                idx = 0
                for i in range(n):
                    bj, bv = 0, vals[idx]
                    for j in range(1, m):
                        v = vals[idx + j]
                        if v > bv:
                            bv, bj = v, j
                    cur[i] = bj
                    idx += m
                return cur

            def callback(md, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = md.cbGetSolution(xvars)
                    except gp.GurobiError:
                        return
                    cur = decode(vals)
                    ov = evaluate(cur)
                    if ov < state["obj"] - 1e-6:
                        state["obj"] = ov
                        state["ass"] = cur
                        if logger is not None:
                            logger.log_solution(float(ov), make_solution(ov, cur))

            model.optimize(callback)

            if model.SolCount > 0:
                cur = decode(model.getAttr("X", xvars))
                ov = evaluate(cur)
                if ov < state["obj"]:
                    state["obj"] = ov
                    state["ass"] = cur

            best_obj, best_ass = state["obj"], state["ass"]
        except Exception:
            pass

    sol = make_solution(best_obj, best_ass)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()