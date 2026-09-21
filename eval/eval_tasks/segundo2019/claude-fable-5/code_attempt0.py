import argparse
import json
import time
import sys
import random

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    g = data["graph"]
    vertices = list(g["vertices"])
    edges = [(e[0], e[1], float(e[2])) for e in g["edges"]]
    return vertices, edges


def build_solution(clique_set, adj_w, vertices):
    """Given a set of vertex ids forming a clique, produce solution dict."""
    cv = sorted(clique_set)
    clique_edges = []
    obj = 0.0
    for i in range(len(cv)):
        u = cv[i]
        for j in range(i + 1, len(cv)):
            v = cv[j]
            w = adj_w.get(u, {}).get(v)
            if w is not None:
                clique_edges.append([u, v])
                obj += w
    return {
        "objective_value": obj,
        "clique_vertices": cv,
        "clique_edges": clique_edges,
    }, obj


def greedy_clique(start, adj_w, adj_set):
    """Grow a clique from vertex start greedily by max weight gain."""
    clique = [start]
    cand = set(adj_set.get(start, ()))
    total = 0.0
    while cand:
        best_v = None
        best_gain = -1.0
        for v in cand:
            gain = 0.0
            wv = adj_w[v]
            for u in clique:
                gain += wv[u]
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v is None:
            break
        clique.append(best_v)
        total += best_gain
        cand &= adj_set[best_v]
    return set(clique), total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    vertices, edges = read_instance(args.instance_path)
    n = len(vertices)

    # adjacency
    adj_w = {v: {} for v in vertices}
    adj_set = {v: set() for v in vertices}
    edge_list = []
    for (u, v, w) in edges:
        if u == v:
            continue
        if v in adj_w[u]:
            continue  # ignore duplicates
        adj_w[u][v] = w
        adj_w[v][u] = w
        adj_set[u].add(v)
        adj_set[v].add(u)
        a, b = (u, v) if u < v else (v, u)
        edge_list.append((a, b, w))

    best_solution = {"objective_value": 0.0, "clique_vertices": [], "clique_edges": []}
    best_obj = -1.0

    # ---------- Greedy heuristic phase ----------
    heur_budget = min(0.15 * args.time_limit, 20.0)
    # order vertices by weighted degree
    order = sorted(vertices, key=lambda v: -sum(adj_w[v].values()))
    rng = random.Random(0)
    starts = order[: min(n, 200)]
    for s in starts:
        if time.time() - t_start > heur_budget:
            break
        clique, tot = greedy_clique(s, adj_w, adj_set)
        if tot > best_obj:
            sol, obj = build_solution(clique, adj_w, vertices)
            if obj > best_obj:
                best_obj = obj
                best_solution = sol
                if logger:
                    logger.log_solution(obj, sol)

    if best_obj < 0:
        best_obj = 0.0
        if logger:
            logger.log_solution(0.0, best_solution)

    # ---------- MIP phase ----------
    remaining = args.time_limit - (time.time() - t_start) - 2.0
    if remaining > 1.0 and n > 0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("mewcp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, remaining)

            x = {v: m.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in vertices}
            y = {}
            for (a, b, w) in edge_list:
                ye = m.addVar(vtype=GRB.BINARY, obj=0.0, name=f"y_{a}_{b}")
                y[(a, b)] = ye
                m.addConstr(ye <= x[a])
                m.addConstr(ye <= x[b])

            # non-edge constraints (clique enforcement)
            vlist = vertices
            for i in range(n):
                u = vlist[i]
                su = adj_set[u]
                for j in range(i + 1, n):
                    v = vlist[j]
                    if v not in su:
                        m.addConstr(x[u] + x[v] <= 1)

            m.setObjective(
                gp.quicksum(w * y[(a, b)] for (a, b, w) in edge_list), GRB.MAXIMIZE
            )

            # warm start
            in_clique = set(best_solution["clique_vertices"])
            for v in vertices:
                x[v].Start = 1.0 if v in in_clique else 0.0
            for (a, b, w) in edge_list:
                y[(a, b)].Start = 1.0 if (a in in_clique and b in in_clique) else 0.0

            m._x = x
            m._vertices = vertices
            state = {"best_obj": best_obj, "best_sol": best_solution}

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    xv = model.cbGetSolution([model._x[v] for v in model._vertices])
                    clique = {
                        model._vertices[i] for i in range(len(model._vertices))
                        if xv[i] > 0.5
                    }
                    sol, obj = build_solution(clique, adj_w, vertices)
                    if obj > state["best_obj"] + 1e-9:
                        state["best_obj"] = obj
                        state["best_sol"] = sol
                        if logger:
                            logger.log_solution(obj, sol)

            m.optimize(cb)

            # extract final if better
            if m.SolCount > 0:
                clique = {v for v in vertices if x[v].X > 0.5}
                sol, obj = build_solution(clique, adj_w, vertices)
                if obj > state["best_obj"] + 1e-9:
                    state["best_obj"] = obj
                    state["best_sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)

            best_obj = state["best_obj"]
            best_solution = state["best_sol"]
        except Exception as ex:
            print(f"MIP phase failed: {ex}", file=sys.stderr)

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()