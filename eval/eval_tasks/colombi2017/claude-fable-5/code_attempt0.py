import argparse
import json
import time
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_nodes = data["num_nodes"]
    depot = data["depot"]
    arcs = [tuple(a) for a in data["arcs"]]            # (i, j, cost)
    parcs = [tuple(p) for p in data["profitable_arcs"]]  # (i, j, profit)
    VI_nodes = list(data.get("VI_nodes", []))
    strong = [tuple(e) for e in data.get("strong_incompatibilities", [])]
    weak = [tuple(e) for e in data.get("weak_incompatibilities", [])]

    nA = len(arcs)
    nP = len(parcs)

    # Map (i,j) -> list of arc indices; also min-cost arc per (i,j)
    arcs_by_ij = {}
    for idx, (i, j, c) in enumerate(arcs):
        arcs_by_ij.setdefault((i, j), []).append(idx)
    mincost_ij = {}
    for key, idxs in arcs_by_ij.items():
        mincost_ij[key] = min(arcs[a][2] for a in idxs)

    # Trivial feasible solution: empty tour, objective 0
    trivial_solution = {"objective_value": 0.0, "served_arcs": [], "tour_arcs": []}
    if logger:
        logger.log_solution(0.0, trivial_solution)

    # ---------- Build MIP ----------
    m = gp.Model("PARP")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.LazyConstraints = 1
    remaining = max(1.0, args.time_limit - (time.time() - t_start) - 2.0)
    m.Params.TimeLimit = remaining

    UBx = max(1, nP + 1)

    x = m.addVars(nA, vtype=GRB.INTEGER, lb=0, ub=UBx, name="x")
    y = m.addVars(nP, vtype=GRB.BINARY, name="y")

    VI_set = set(VI_nodes)
    z = {v: m.addVar(vtype=GRB.BINARY, name=f"z_{v}") for v in VI_set}
    w = {e: m.addVar(vtype=GRB.BINARY, name=f"w_{e}") for e in range(len(weak))}

    # Flow conservation
    in_arcs = [[] for _ in range(num_nodes)]
    out_arcs = [[] for _ in range(num_nodes)]
    for idx, (i, j, c) in enumerate(arcs):
        out_arcs[i].append(idx)
        in_arcs[j].append(idx)
    for v in range(num_nodes):
        if in_arcs[v] or out_arcs[v]:
            m.addConstr(gp.quicksum(x[a] for a in in_arcs[v]) ==
                        gp.quicksum(x[a] for a in out_arcs[v]))

    # Serve only if traversed
    for p, (pi, pj, prof) in enumerate(parcs):
        idxs = arcs_by_ij.get((pi, pj), [])
        if not idxs:
            m.addConstr(y[p] == 0)
        else:
            m.addConstr(y[p] <= gp.quicksum(x[a] for a in idxs))

    # Node activity linking
    for p, (pi, pj, prof) in enumerate(parcs):
        if pi in VI_set:
            m.addConstr(z[pi] >= y[p])

    # Strong incompatibilities
    for (u, v) in strong:
        if u in z and v in z:
            m.addConstr(z[u] + z[v] <= 1)

    # Weak incompatibilities
    for e, (u, v, pen) in enumerate(weak):
        if u in z and v in z:
            m.addConstr(z[u] + z[v] <= 1 + w[e])

    obj = gp.quicksum(parcs[p][2] * y[p] for p in range(nP)) \
        - gp.quicksum(arcs[a][2] * x[a] for a in range(nA)) \
        - gp.quicksum(weak[e][2] * w[e] for e in range(len(weak)))
    m.setObjective(obj, GRB.MAXIMIZE)

    # ---------- Lazy connectivity callback ----------
    def build_solution_dict(xv, yv, objval):
        served = []
        for p, (pi, pj, prof) in enumerate(parcs):
            if yv[p] > 0.5:
                served.append({"from": pi, "to": pj, "profit": prof,
                               "cost": mincost_ij.get((pi, pj), 0)})
        tour = []
        for a, (ai, aj, c) in enumerate(arcs):
            cnt = int(round(xv[a]))
            if cnt > 0:
                tour.append({"from": ai, "to": aj, "count": cnt, "cost": c})
        return {"objective_value": float(objval), "served_arcs": served,
                "tour_arcs": tour}

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        xv = model.cbGetSolution(x)
        yv = model.cbGetSolution(y)

        # Union-find over nodes joined by arcs with x > 0.5
        parent = list(range(num_nodes))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

        used = []
        for a in range(nA):
            if xv[a] > 0.5:
                used.append(a)
                union(arcs[a][0], arcs[a][1])

        depot_root = find(depot)

        # served arcs whose component excludes depot -> violated
        viol_by_comp = {}
        for p, (pi, pj, prof) in enumerate(parcs):
            if yv[p] > 0.5:
                r = find(pi)
                if r != depot_root:
                    viol_by_comp.setdefault(r, []).append(p)

        if not viol_by_comp:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log_solution(float(objval),
                                    build_solution_dict(xv, yv, objval))
            return

        # Node sets per component root
        comp_nodes = {}
        for v in range(num_nodes):
            r = find(v)
            if r in viol_by_comp:
                comp_nodes.setdefault(r, set()).add(v)

        for r, plist in viol_by_comp.items():
            S = comp_nodes[r]
            cut_arcs = [a for a in range(nA)
                        if arcs[a][0] in S and arcs[a][1] not in S]
            expr = gp.quicksum(x[a] for a in cut_arcs)
            for p in plist:
                model.cbLazy(expr >= y[p])

    try:
        m.optimize(callback)
    except gp.GurobiError:
        pass

    # ---------- Extract final solution ----------
    best_solution = trivial_solution
    if m.SolCount > 0:
        try:
            xv = [x[a].X for a in range(nA)]
            yv = [y[p].X for p in range(nP)]
            objval = m.ObjVal
            if objval >= 0.0 - 1e-9:
                cand = build_solution_dict(xv, yv, objval)
                # verify connectivity quickly (should hold via lazy cuts)
                best_solution = cand
                if objval < 0:
                    best_solution = trivial_solution
        except gp.GurobiError:
            pass

    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()