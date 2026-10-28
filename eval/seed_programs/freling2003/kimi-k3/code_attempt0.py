import argparse
import json
import math
import time

try:
    import numpy as np
except Exception:
    np = None


def main():
    parser = argparse.ArgumentParser(
        description="Single-sourcing multi-period production/distribution network optimization."
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))
    deadline = t_start + float(time_limit)
    reserve = max(1.0, min(8.0, 0.05 * time_limit))

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # ---------------- read instance ----------------
    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    params = inst["parameters"]
    F = int(params["num_facilities"])
    C = int(params["num_customers"])
    T = int(params["num_periods"])

    seas = [float(v) for v in inst["seasonal_factors"]]
    demands = [float(v) for v in inst["demands"]]
    holding = [[float(v) for v in r] for r in inst["holding_costs"]]
    cap = [[float(v) for v in r] for r in inst["capacities"]]
    tcost = inst["transportation_costs"]

    # cumulative seasonal factors
    Sfac = [0.0] * T
    acc = 0.0
    for t in range(T):
        acc += seas[t]
        Sfac[t] = acc

    # cumulative capacity and effective (bottleneck) capacity per facility.
    # Feasibility for a facility serving total demand D is:
    #   D * S_t <= cumulative_capacity_t  for every prefix t,
    # i.e. D <= min_t cumcap_t / S_t.
    cumcap = []
    effcap = []
    for i in range(F):
        row = cap[i]
        cc = [0.0] * T
        acc = 0.0
        ec = math.inf
        for t in range(T):
            acc += row[t]
            cc[t] = acc
            if Sfac[t] > 1e-12:
                r = acc / Sfac[t]
                if r < ec:
                    ec = r
        cumcap.append(cc)
        effcap.append(ec)

    # supply cost of assigning customer j to facility i for the whole horizon
    cost = []
    for i in range(F):
        ti = tcost[i]
        row = [0.0] * C
        for j in range(C):
            cj = ti[j]
            tot = 0.0
            for t in range(T):
                tot += cj[t]
            row[j] = tot
        cost.append(row)

    # Minimum inventory cost for facility i serving total demand D.
    # With G_t = D*S_t - cumcap_t, the minimal end-of-period-t inventory is
    #   I_t = max(0, max_{s>t} G_s - G_t)   (produce as late as possible).
    def hold_cost(i, D):
        cc = cumcap[i]
        h = holding[i]
        hc = 0.0
        suffix = -math.inf
        for t in range(T - 1, -1, -1):
            G = D * Sfac[t] - cc[t]
            val = suffix - G
            if val > 0.0:
                hc += h[t] * val
            if G > suffix:
                suffix = G
        return hc

    def inventory_and_cost(i, D):
        cc = cumcap[i]
        h = holding[i]
        I = [0.0] * T
        hc = 0.0
        suffix = -math.inf
        for t in range(T - 1, -1, -1):
            G = D * Sfac[t] - cc[t]
            val = suffix - G
            if val > 1e-9:
                rv = round(val, 9)
                I[t] = rv
                hc += h[t] * rv
            if G > suffix:
                suffix = G
        return I, hc

    def build_solution(assign):
        Dl = [0.0] * F
        for j in range(C):
            Dl[assign[j]] += demands[j]
        supply = 0.0
        for j in range(C):
            supply += cost[assign[j]][j]
        hold_total = 0.0
        inv = {}
        for i in range(F):
            Ii, hi = inventory_and_cost(i, Dl[i])
            inv[str(i)] = Ii
            hold_total += hi
        obj = supply + hold_total
        sol = {
            "objective_value": float(obj),
            "assignment": {str(j): int(assign[j]) for j in range(C)},
            "inventory": inv,
        }
        return sol, obj

    # ---------------- greedy construction ----------------
    FC = F * C
    small = FC <= 60000
    assign = [0] * C
    D = [0.0] * F
    hold_cache = [0.0] * F

    orderfac = None
    if not small and FC <= 5_000_000 and np is not None:
        try:
            orderfac = np.ascontiguousarray(
                np.argsort(np.asarray(cost), axis=0).T
            )
        except Exception:
            orderfac = None

    if C > 0 and F > 0:
        order = sorted(range(C), key=lambda j: -demands[j])
        for j in order:
            dj = demands[j]
            bi = -1
            if small:
                bv = math.inf
                for i in range(F):
                    if D[i] + dj <= effcap[i] + 1e-9:
                        v = cost[i][j] + hold_cost(i, D[i] + dj) - hold_cache[i]
                        if v < bv - 1e-12:
                            bv = v
                            bi = i
            elif orderfac is not None:
                for i0 in orderfac[j]:
                    i = int(i0)
                    if D[i] + dj <= effcap[i] + 1e-9:
                        bi = i
                        break
            else:
                bv = math.inf
                for i in range(F):
                    if D[i] + dj <= effcap[i] + 1e-9:
                        v = cost[i][j]
                        if v < bv - 1e-12:
                            bv = v
                            bi = i
            if bi < 0:
                # no facility has enough effective capacity: take max slack
                bi = 0
                bslack = -math.inf
                for i in range(F):
                    sl = effcap[i] - D[i]
                    if sl > bslack:
                        bslack = sl
                        bi = i
            assign[j] = bi
            D[bi] += dj
            hold_cache[bi] = hold_cost(bi, D[bi])

        # ---------------- local search (best relocate moves) ----------------
        cand_ls = None
        do_ls = small
        if not small and FC <= 1_500_000:
            L = min(F, 20)
            cand_ls = []
            if orderfac is not None:
                for j in range(C):
                    cand_ls.append([int(v) for v in orderfac[j][:L]])
            else:
                for j in range(C):
                    cand_ls.append(sorted(range(F), key=lambda i: cost[i][j])[:L])
            do_ls = True

        if do_ls:
            heur_end = min(
                deadline - reserve,
                time.time() + max(1.0, min(0.2 * time_limit, 25.0)),
            )
            improved = True
            while improved and time.time() < heur_end:
                improved = False
                for j in range(C):
                    if (j & 63) == 0 and time.time() > heur_end:
                        break
                    i0 = assign[j]
                    dj = demands[j]
                    h0new = hold_cost(i0, D[i0] - dj)
                    base = h0new - hold_cache[i0] - cost[i0][j]
                    facs = range(F) if cand_ls is None else cand_ls[j]
                    bi = -1
                    bd = -1e-7
                    bh = 0.0
                    for i in facs:
                        if i == i0:
                            continue
                        if D[i] + dj > effcap[i] + 1e-9:
                            continue
                        hn = hold_cost(i, D[i] + dj)
                        delta = base + cost[i][j] + hn - hold_cache[i]
                        if delta < bd:
                            bd = delta
                            bi = i
                            bh = hn
                    if bi >= 0:
                        D[i0] -= dj
                        hold_cache[i0] = h0new
                        D[bi] += dj
                        hold_cache[bi] = bh
                        assign[j] = bi
                        improved = True

    best_sol, best_obj = build_solution(assign)
    if logger:
        logger.log_solution(float(best_obj), best_sol)

    # ---------------- exact MIP with Gurobi ----------------
    VAR_BUDGET = 250000
    remaining = deadline - time.time() - reserve
    can_mip = (
        remaining > 1.5
        and F > 0
        and C > 0
        and (FC <= VAR_BUDGET or orderfac is not None or FC <= 1_500_000)
    )
    if can_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            if FC <= VAR_BUDGET:
                cand = [range(F)] * C
            else:
                K = min(F, max(4, VAR_BUDGET // C))
                cand = []
                for j in range(C):
                    if orderfac is not None:
                        lst = [int(v) for v in orderfac[j][:K]]
                    else:
                        lst = sorted(range(F), key=lambda i: cost[i][j])[:K]
                    if assign[j] not in lst:
                        lst.append(assign[j])
                    cand.append(lst)

            keys = []
            fac_vars = [[] for _ in range(F)]
            for j in range(C):
                for i in cand[j]:
                    keys.append((i, j))
                    fac_vars[i].append(len(keys) - 1)

            m = gp.Model("distnet")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.OutputFlag = 0

            xt = m.addVars(keys, vtype=GRB.BINARY)
            xvars = [xt[k] for k in keys]
            m.setAttr("Obj", xvars, [cost[i][j] for (i, j) in keys])

            Dv = m.addVars(F, lb=0.0)          # total demand load per facility
            Iv = m.addVars(F, T, lb=0.0)       # end-of-period inventory
            ivars = [Iv[i, t] for i in range(F) for t in range(T)]
            m.setAttr(
                "Obj", ivars,
                [holding[i][t] for i in range(F) for t in range(T)],
            )
            m.ModelSense = GRB.MINIMIZE

            # each customer assigned to exactly one facility
            m.addConstrs((xt.sum('*', j) == 1.0 for j in range(C)), name="assign")

            # facility load definition
            for i in range(F):
                fvs = fac_vars[i]
                if fvs:
                    coeffs = [demands[keys[k][1]] for k in fvs]
                    vs = [xvars[k] for k in fvs]
                    m.addConstr(Dv[i] == gp.LinExpr(coeffs, vs))
                else:
                    m.addConstr(Dv[i] == 0.0)

            # demand_t + I_t <= capacity_t + I_{t-1}
            m.addConstrs(
                (
                    seas[t] * Dv[i] + Iv[i, t] - (Iv[i, t - 1] if t > 0 else 0.0)
                    <= cap[i][t]
                    for i in range(F)
                    for t in range(T)
                ),
                name="bal",
            )

            # valid effective-capacity cuts (implied, may help presolve)
            m.addConstrs(
                (Dv[i] <= effcap[i] for i in range(F) if effcap[i] < math.inf),
                name="effcap",
            )

            # MIP start from heuristic
            m.setAttr(
                "Start", xvars,
                [1.0 if assign[j] == i else 0.0 for (i, j) in keys],
            )
            startI = []
            for i in range(F):
                Ii, _ = inventory_and_cost(i, D[i])
                startI.extend(Ii)
            m.setAttr("Start", ivars, startI)

            ki = [k[0] for k in keys]
            kj = [k[1] for k in keys]
            nkeys = len(keys)
            cb_state = {"obj": best_obj}

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        gobj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        tol = 1e-9 * max(1.0, abs(cb_state["obj"]))
                        if gobj >= cb_state["obj"] - tol:
                            return
                        vals = model.cbGetSolution(xvars)
                        bestv = {}
                        for k in range(nkeys):
                            v = vals[k]
                            j = kj[k]
                            if j not in bestv or v > bestv[j][1]:
                                bestv[j] = (ki[k], v)
                        acb = [bestv[j][0] for j in range(C)]
                        sol, obj = build_solution(acb)
                        if obj < cb_state["obj"] - 1e-7:
                            cb_state["obj"] = obj
                            if logger:
                                logger.log_solution(float(obj), sol)
                    except Exception:
                        pass

            m.Params.TimeLimit = max(0.5, deadline - time.time() - reserve)
            m.optimize(callback)

            if m.SolCount > 0:
                vals = m.getAttr("X", xvars)
                bestv = {}
                for k in range(nkeys):
                    v = vals[k]
                    j = kj[k]
                    if j not in bestv or v > bestv[j][1]:
                        bestv[j] = (ki[k], v)
                ag = [bestv[j][0] for j in range(C)]
                sol, obj = build_solution(ag)
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_sol = sol
                    if logger:
                        logger.log_solution(float(best_obj), best_sol)
        except Exception:
            pass

    # ---------------- write final solution ----------------
    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh)


if __name__ == "__main__":
    main()