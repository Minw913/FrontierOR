import json
import argparse
import time
import sys

from solution_logger import SolutionLogger


def tour_length(tour, dm):
    if len(tour) <= 1:
        return 0
    total = 0
    for k in range(len(tour)):
        total += dm[tour[k]][tour[(k + 1) % len(tour)]]
    return total


def tour_to_edges(tour):
    if len(tour) <= 1:
        return []
    edges = []
    for k in range(len(tour)):
        a, b = tour[k], tour[(k + 1) % len(tour)]
        edges.append([a, b])
    return edges


def two_opt(tour, dm, deadline):
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d]
                if delta < -1e-9:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    b = tour[i + 1]
            if time.time() > deadline:
                break
    return tour


def greedy_insertion(tour, dm, scores, D, visited_set, deadline):
    """Insert unvisited nodes with best score/length-increase ratio while feasible."""
    n = len(dm)
    cur_len = tour_length(tour, dm)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        best = None  # (ratio, delta, node, pos)
        m = len(tour)
        for j in range(n):
            if j in visited_set:
                continue
            sj = scores[j]
            # find best insertion position
            best_delta = None
            best_pos = None
            if m == 1:
                d = 2 * dm[tour[0]][j]
                best_delta = d
                best_pos = 1
            else:
                for k in range(m):
                    a = tour[k]
                    b = tour[(k + 1) % m]
                    delta = dm[a][j] + dm[j][b] - dm[a][b]
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_pos = k + 1
            if best_delta is None:
                continue
            if cur_len + best_delta <= D:
                ratio = sj / (best_delta + 1e-6)
                if best is None or ratio > best[0]:
                    best = (ratio, best_delta, j, best_pos)
        if best is not None:
            _, delta, j, pos = best
            tour.insert(pos, j)
            visited_set.add(j)
            cur_len += delta
            changed = True
    return tour


def heuristic_solution(depot, dm, scores, D, deadline):
    n = len(dm)
    tour = [depot]
    visited = {depot}
    for _ in range(6):
        if time.time() > deadline:
            break
        tour = greedy_insertion(tour, dm, scores, D, visited, deadline)
        before = tour_length(tour, dm)
        tour = two_opt(tour, dm, deadline)
        after = tour_length(tour, dm)
        if after >= before - 1e-9 and len(visited) == len(set(tour)):
            # try one more insertion pass; if nothing added, stop
            m_before = len(tour)
            tour = greedy_insertion(tour, dm, scores, D, visited, deadline)
            if len(tour) == m_before:
                break
    obj = sum(scores[v] for v in tour)
    return tour, obj


def solve(instance, time_limit, logger):
    start_time = time.time()
    deadline = start_time + time_limit

    n = instance["num_nodes"]
    D = instance["distance_limitation_d0"]
    depot = instance["depot_vertex"]
    scores = instance["scores"]
    dm = instance["distance_matrix"]

    # ---------- heuristic warm start ----------
    heur_deadline = min(deadline - 1.0, start_time + max(2.0, 0.15 * time_limit))
    tour, heur_obj = heuristic_solution(depot, dm, scores, D, heur_deadline)

    best_solution = {
        "objective_value": float(heur_obj),
        "visited_vertices": sorted(set(tour)),
        "tour_edges": tour_to_edges(tour),
    }
    if logger:
        logger.log_solution(float(heur_obj), best_solution)

    remaining = deadline - time.time() - 1.0
    if remaining < 2.0:
        return best_solution

    # ---------- exact MIP with Gurobi ----------
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return best_solution

    try:
        model = gp.Model("orienteering")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = max(1.0, remaining)

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        x = {}
        for (i, j) in edges:
            ub = 2 if (i == depot or j == depot) else 1
            x[(i, j)] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub,
                                     name=f"x_{i}_{j}")
        y = {}
        for i in range(n):
            y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        y[depot].LB = 1.0

        # degree constraints
        inc = {i: [] for i in range(n)}
        for (i, j) in edges:
            inc[i].append((i, j))
            inc[j].append((i, j))
        for i in range(n):
            model.addConstr(gp.quicksum(x[e] for e in inc[i]) == 2 * y[i])

        # distance budget
        model.addConstr(gp.quicksum(dm[i][j] * x[(i, j)] for (i, j) in edges) <= D)

        model.setObjective(gp.quicksum(scores[i] * y[i] for i in range(n)),
                           GRB.MAXIMIZE)

        # warm start
        tour_edge_count = {}
        m = len(tour)
        if m >= 2:
            for k in range(m):
                a, b = tour[k], tour[(k + 1) % m]
                key = (min(a, b), max(a, b))
                tour_edge_count[key] = tour_edge_count.get(key, 0) + 1
        visited_ws = set(tour)
        for i in range(n):
            y[i].Start = 1.0 if i in visited_ws else 0.0
        for e in edges:
            x[e].Start = float(tour_edge_count.get(e, 0))

        state = {"best_obj": float(heur_obj), "best_sol": best_solution}

        def extract_solution(xvals, yvals):
            visited = [i for i in range(n) if yvals[i] > 0.5]
            edge_list = []
            for (i, j) in edges:
                v = int(round(xvals[(i, j)]))
                for _ in range(v):
                    edge_list.append([i, j])
            return {
                "objective_value": float(sum(scores[i] for i in visited)),
                "visited_vertices": sorted(visited),
                "tour_edges": edge_list,
            }

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            xvals = model.cbGetSolution([x[e] for e in edges])
            yvals = model.cbGetSolution([y[i] for i in range(n)])
            xd = {edges[k]: xvals[k] for k in range(len(edges))}
            # build adjacency of active edges
            adj = {i: [] for i in range(n)}
            for (i, j) in edges:
                if xd[(i, j)] > 0.5:
                    adj[i].append(j)
                    adj[j].append(i)
            active = [i for i in range(n) if yvals[i] > 0.5]
            # find components
            seen = set()
            comps = []
            for s in active:
                if s in seen:
                    continue
                comp = []
                stack = [s]
                seen.add(s)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            stack.append(v)
                comps.append(comp)
            violated = False
            for comp in comps:
                if depot in comp:
                    continue
                violated = True
                S = set(comp)
                inner = [x[(i, j)] for (i, j) in edges if i in S and j in S]
                ysum = gp.quicksum(y[i] for i in S)
                # add GSEC for a couple of anchor nodes
                anchors = sorted(S, key=lambda v: -scores[v])[:2]
                for k in anchors:
                    model.cbLazy(gp.quicksum(inner) <= ysum - y[k])
            if not violated:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > state["best_obj"] + 1e-6:
                    sol = extract_solution(xd, yvals)
                    state["best_obj"] = obj
                    state["best_sol"] = sol
                    if logger:
                        logger.log_solution(float(obj), sol)

        model.optimize(callback)

        if model.SolCount > 0:
            xvals = {e: x[e].X for e in edges}
            yvals = {i: y[i].X for i in range(n)}
            # verify connectivity of final incumbent (should be feasible)
            adj = {i: [] for i in range(n)}
            for (i, j) in edges:
                if xvals[(i, j)] > 0.5:
                    adj[i].append(j)
                    adj[j].append(i)
            comp = set()
            stack = [depot]
            comp.add(depot)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in comp:
                        comp.add(v)
                        stack.append(v)
            active = set(i for i in range(n) if yvals[i] > 0.5)
            if active <= comp:
                sol = extract_solution(xvals, yvals)
                if sol["objective_value"] >= state["best_obj"] - 1e-6:
                    state["best_sol"] = sol
                    state["best_obj"] = sol["objective_value"]
                    if logger and sol["objective_value"] > heur_obj + 1e-6:
                        logger.log_solution(sol["objective_value"], sol)
        return state["best_sol"]
    except Exception:
        return best_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()