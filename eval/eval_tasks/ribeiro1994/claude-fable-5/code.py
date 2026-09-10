import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

INF = float("inf")


def set_params(mdl, tl=None):
    mdl.Params.OutputFlag = 0
    mdl.Params.Seed = 0
    mdl.Params.MIPGap = 1e-4
    mdl.Params.NumericFocus = 0
    mdl.Params.Threads = 1
    if tl is not None:
        mdl.Params.TimeLimit = max(1.0, tl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data["parameters"]["n_trips"]
    m = data["parameters"]["m_depots"]
    caps = data["vehicle_capacities"]
    total_cap = sum(caps)

    arcs = [(a["from_trip"], a["to_trip"], a["cost"]) for a in data["compatible_arcs"]]
    nA = len(arcs)

    out_cost = [[INF] * (n + 1) for _ in range(m + 1)]  # out_cost[d][j]
    for r in data["depot_to_trip_costs"]:
        out_cost[r["depot"]][r["trip"]] = r["cost"]
    in_cost = [[INF] * (m + 1) for _ in range(n + 1)]  # in_cost[i][d]
    for r in data["trip_to_depot_costs"]:
        in_cost[r["trip"]][r["depot"]] = r["cost"]

    arc_cost = {}
    arc_idx = {}
    in_arcs = [[] for _ in range(n + 1)]
    out_arcs = [[] for _ in range(n + 1)]
    for idx, (i, j, c) in enumerate(arcs):
        arc_cost[(i, j)] = c
        arc_idx[(i, j)] = idx
        in_arcs[j].append(idx)
        out_arcs[i].append(idx)

    best_out = [0] * (n + 1)
    best_in = [0] * (n + 1)
    for j in range(1, n + 1):
        best_out[j] = min(out_cost[d][j] for d in range(1, m + 1))
        best_in[j] = min(in_cost[j][d] for d in range(1, m + 1))

    def total_cost(routes):
        c = 0
        for d, trips in routes:
            c += out_cost[d][trips[0]] + in_cost[trips[-1]][d]
            for a, b in zip(trips, trips[1:]):
                c += arc_cost[(a, b)]
        return c

    def sol_dict(routes, cost):
        return {
            "objective_value": float(cost),
            "routes": [{"depot": int(d), "trips": [int(t) for t in trips]}
                       for d, trips in routes],
            "num_vehicles": len(routes),
        }

    def write_solution(routes, cost):
        with open(args.solution_path, "w") as f:
            json.dump(sol_dict(routes, cost), f)

    def remaining():
        return args.time_limit - (time.time() - t0)

    # ---------------- Heuristic: assignment LP + transportation -------------
    def solve_assignment(pen, tl):
        mdl = gp.Model()
        set_params(mdl, tl)
        y = mdl.addVars(nA, lb=0.0, ub=1.0)
        s = mdl.addVars(range(1, n + 1), lb=0.0, ub=1.0)
        t = mdl.addVars(range(1, n + 1), lb=0.0, ub=1.0)
        for j in range(1, n + 1):
            mdl.addConstr(s[j] + gp.quicksum(y[a] for a in in_arcs[j]) == 1)
            mdl.addConstr(t[j] + gp.quicksum(y[a] for a in out_arcs[j]) == 1)
        obj = gp.quicksum(arcs[a][2] * y[a] for a in range(nA)) \
            + gp.quicksum((best_out[j] + pen) * s[j] for j in range(1, n + 1)) \
            + gp.quicksum(best_in[j] * t[j] for j in range(1, n + 1))
        mdl.setObjective(obj, GRB.MINIMIZE)
        mdl.optimize()
        if mdl.SolCount == 0:
            return None
        yv = mdl.getAttr("X", [y[a] for a in range(nA)])
        succ = {}
        for a, v in enumerate(yv):
            if v > 0.5:
                succ[arcs[a][0]] = arcs[a][1]
        sv = mdl.getAttr("X", [s[j] for j in range(1, n + 1)])
        chains = []
        for j0 in range(1, n + 1):
            if sv[j0 - 1] > 0.5:
                ch = [j0]
                cur = j0
                while cur in succ:
                    cur = succ[cur]
                    ch.append(cur)
                chains.append(ch)
        return chains

    def assign_depots(chains, tl):
        K = len(chains)
        if K > total_cap:
            return None
        mdl = gp.Model()
        set_params(mdl, tl)
        z = {}
        for k in range(K):
            fst, lst = chains[k][0], chains[k][-1]
            for d in range(1, m + 1):
                c = out_cost[d][fst] + in_cost[lst][d]
                if c < INF:
                    z[(k, d)] = mdl.addVar(lb=0.0, ub=1.0, obj=c)
        for k in range(K):
            mdl.addConstr(gp.quicksum(z[(k, d)] for d in range(1, m + 1)
                                      if (k, d) in z) == 1)
        for d in range(1, m + 1):
            mdl.addConstr(gp.quicksum(z[(k, d)] for k in range(K)
                                      if (k, d) in z) <= caps[d - 1])
        mdl.ModelSense = GRB.MINIMIZE
        mdl.optimize()
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) or mdl.SolCount == 0:
            return None
        dep = []
        for k in range(K):
            chosen = None
            for d in range(1, m + 1):
                if (k, d) in z and z[(k, d)].X > 0.5:
                    chosen = d
                    break
            if chosen is None:
                return None
            dep.append(chosen)
        return dep

    tl_h = max(5.0, remaining() * 0.3)
    chains = solve_assignment(0, tl_h)
    dep = assign_depots(chains, max(5.0, remaining() * 0.2)) if chains else None
    if dep is None:
        chains_p = solve_assignment(10 ** 7, max(5.0, remaining() * 0.3))
        if chains_p:
            dep_p = assign_depots(chains_p, max(5.0, remaining() * 0.2))
            if dep_p is not None:
                chains, dep = chains_p, dep_p
    if dep is None:
        # greedy fallback (last resort)
        dep = []
        rem_cap = list(caps)
        for ch in chains:
            fst, lst = ch[0], ch[-1]
            order = sorted(range(1, m + 1),
                           key=lambda d: out_cost[d][fst] + in_cost[lst][d])
            pick = None
            for d in order:
                if rem_cap[d - 1] > 0 and out_cost[d][fst] + in_cost[lst][d] < INF:
                    pick = d
                    break
            if pick is None:
                pick = order[0]
            rem_cap[pick - 1] -= 1
            dep.append(pick)

    best_routes = [(dep[k], chains[k]) for k in range(len(chains))]
    best_cost = total_cost(best_routes)
    if logger:
        logger.log_solution(float(best_cost), sol_dict(best_routes, best_cost))
    write_solution(best_routes, best_cost)

    # ---------------- Exact multicommodity-flow MIP --------------------------
    nvars = m * (nA + 2 * n)
    if remaining() > 25 and nvars <= 3_000_000:
        try:
            mdl = gp.Model()
            set_params(mdl)
            x = {}
            o = {}
            e = {}
            for d in range(1, m + 1):
                xd = mdl.addVars(nA, vtype=GRB.BINARY)
                mdl.setAttr("Obj", [xd[a] for a in range(nA)],
                            [arcs[a][2] for a in range(nA)])
                od = mdl.addVars(range(1, n + 1), vtype=GRB.BINARY)
                ed = mdl.addVars(range(1, n + 1), vtype=GRB.BINARY)
                for j in range(1, n + 1):
                    if out_cost[d][j] < INF:
                        od[j].Obj = out_cost[d][j]
                    else:
                        od[j].UB = 0.0
                    if in_cost[j][d] < INF:
                        ed[j].Obj = in_cost[j][d]
                    else:
                        ed[j].UB = 0.0
                x[d], o[d], e[d] = xd, od, ed

            for j in range(1, n + 1):
                mdl.addConstr(
                    gp.quicksum(o[d][j] for d in range(1, m + 1))
                    + gp.quicksum(x[d][a] for d in range(1, m + 1)
                                  for a in in_arcs[j]) == 1)
            for d in range(1, m + 1):
                for i in range(1, n + 1):
                    mdl.addConstr(
                        o[d][i] + gp.quicksum(x[d][a] for a in in_arcs[i])
                        - e[d][i] - gp.quicksum(x[d][a] for a in out_arcs[i]) == 0)
                mdl.addConstr(gp.quicksum(o[d][j] for j in range(1, n + 1))
                              <= caps[d - 1])

            # Warm start from heuristic
            for d in range(1, m + 1):
                mdl.setAttr("Start", [x[d][a] for a in range(nA)], [0.0] * nA)
                mdl.setAttr("Start", [o[d][j] for j in range(1, n + 1)], [0.0] * n)
                mdl.setAttr("Start", [e[d][j] for j in range(1, n + 1)], [0.0] * n)
            for d, trips in best_routes:
                o[d][trips[0]].Start = 1.0
                e[d][trips[-1]].Start = 1.0
                for a, b in zip(trips, trips[1:]):
                    x[d][arc_idx[(a, b)]].Start = 1.0

            o_lists = {d: [o[d][j] for j in range(1, n + 1)] for d in range(1, m + 1)}
            x_lists = {d: [x[d][a] for a in range(nA)] for d in range(1, m + 1)}

            mdl._best = best_cost
            mdl._routes = None

            def cb(mdl_, where):
                if where == GRB.Callback.MIPSOL:
                    objv = mdl_.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objv < mdl_._best - 0.5:
                        routes = []
                        for d in range(1, m + 1):
                            ov = mdl_.cbGetSolution(o_lists[d])
                            xv = mdl_.cbGetSolution(x_lists[d])
                            succ = {}
                            for a, v in enumerate(xv):
                                if v > 0.5:
                                    succ[arcs[a][0]] = arcs[a][1]
                            for jj in range(1, n + 1):
                                if ov[jj - 1] > 0.5:
                                    ch = [jj]
                                    cur = jj
                                    while cur in succ:
                                        cur = succ[cur]
                                        ch.append(cur)
                                    routes.append((d, ch))
                        try:
                            c = total_cost(routes)
                        except Exception:
                            return
                        if c < mdl_._best - 0.5:
                            mdl_._best = c
                            mdl_._routes = routes
                            if logger:
                                logger.log_solution(float(c), sol_dict(routes, c))

            tl_mip = remaining() - 8
            if tl_mip > 5:
                mdl.Params.TimeLimit = tl_mip
                mdl.optimize(cb)
                if mdl.SolCount > 0:
                    routes = []
                    for d in range(1, m + 1):
                        ov = mdl.getAttr("X", o_lists[d])
                        xv = mdl.getAttr("X", x_lists[d])
                        succ = {}
                        for a, v in enumerate(xv):
                            if v > 0.5:
                                succ[arcs[a][0]] = arcs[a][1]
                        for jj in range(1, n + 1):
                            if ov[jj - 1] > 0.5:
                                ch = [jj]
                                cur = jj
                                while cur in succ:
                                    cur = succ[cur]
                                    ch.append(cur)
                                routes.append((d, ch))
                    covered = sum(len(t) for _, t in routes)
                    if covered == n:
                        c = total_cost(routes)
                        if c < best_cost - 0.5:
                            best_cost = c
                            best_routes = routes
                            if logger:
                                logger.log_solution(float(c), sol_dict(routes, c))
        except Exception:
            pass

    write_solution(best_routes, best_cost)


if __name__ == "__main__":
    main()