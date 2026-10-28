#!/usr/bin/env python3
import argparse
import ast
import json
import re
import time
import math

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def parse_pair_key(key):
    if isinstance(key, (tuple, list)) and len(key) >= 2:
        return int(key[0]), int(key[1])
    s = str(key).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            return int(v[0]), int(v[1])
    except Exception:
        pass
    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    raise ValueError(f"Cannot parse travel-time key: {key}")


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    depot = int(data.get("depot", 1))
    t0 = int(data["t0"])

    prizes_raw = data.get("prizes", {}) or {}
    prize = {}
    for v in range(1, n + 1):
        if str(v) in prizes_raw:
            prize[v] = float(prizes_raw[str(v)])
        elif v in prizes_raw:
            prize[v] = float(prizes_raw[v])
        else:
            prize[v] = 0.0

    d = [[0.0] * (n + 1) for _ in range(n + 1)]
    seen = set()
    max_t = 0.0
    tt = data.get("travel_times", {}) or {}
    for key, val in tt.items():
        i, j = parse_pair_key(key)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if i < 1 or j > n:
            continue
        w = float(val)
        d[i][j] = w
        d[j][i] = w
        seen.add((i, j))
        if w > max_t:
            max_t = w

    inf = max(max_t, float(t0), 1.0) * 1.0e6 + 1.0e9
    for i in range(1, n + 1):
        d[i][i] = 0.0
        for j in range(i + 1, n + 1):
            if (i, j) not in seen:
                d[i][j] = inf
                d[j][i] = inf
    return n, depot, t0, prize, d


def norm_edge(i, j):
    return (i, j) if i < j else (j, i)


def route_cost(tour, d):
    return sum(d[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))


def make_solution_from_tour(tour, depot, prize):
    if not tour:
        return None
    tour = list(tour)
    if tour[0] != depot:
        return None
    if tour[-1] != depot:
        tour.append(depot)
    if len(tour) < 2:
        return None
    visited_order = tour[:-1]
    if len(visited_order) < 3:
        return None
    if len(set(visited_order)) != len(visited_order):
        return None
    if visited_order[0] != depot or depot in visited_order[1:]:
        return None
    edges = []
    for u, v in zip(tour[:-1], tour[1:]):
        if u == v:
            return None
        a, b = norm_edge(u, v)
        edges.append([a, b])
    visited = sorted(set(visited_order))
    obj = float(sum(prize.get(v, 0.0) for v in visited))
    return {
        "objective_value": obj,
        "visited_nodes": visited,
        "edges": edges,
        "tour": tour,
    }


def solution_from_edges(nodes, depot, edges, prize):
    adj = {}
    for i, j in edges:
        if i == j:
            return None
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    visited = [v for v in nodes if len(adj.get(v, [])) > 0]
    if depot not in visited or len(visited) < 3:
        return None
    for v in visited:
        if len(adj.get(v, [])) != 2:
            return None

    tour = [depot]
    prev = None
    curr = depot
    for _ in range(len(visited)):
        neigh = adj.get(curr, [])
        if len(neigh) != 2:
            return None
        if prev is None:
            nxt = neigh[0]
        else:
            if neigh[0] == prev and neigh[1] == prev:
                return None
            nxt = neigh[0] if neigh[0] != prev else neigh[1]
        tour.append(nxt)
        prev, curr = curr, nxt
        if nxt == depot:
            break

    if tour[-1] != depot or len(tour) != len(visited) + 1:
        return None
    if len(set(tour[:-1])) != len(visited) or set(tour[:-1]) != set(visited):
        return None
    return make_solution_from_tour(tour, depot, prize)


def best_insertion(tour, v, d, deadline):
    best_delta = float("inf")
    best_pos = None
    # A full tour is [depot, ..., depot]; inserting at pos puts v between tour[pos-1] and tour[pos].
    for pos in range(1, len(tour)):
        if pos & 15 == 0 and time.monotonic() > deadline:
            return None, None
        u = tour[pos - 1]
        w = tour[pos]
        delta = d[u][v] + d[v][w] - d[u][w]
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_delta, best_pos


def insert_all_possible(tour, in_tour, d, prize, t0, deadline):
    nodes = range(1, len(d))
    inserted = 0
    max_price = max([prize.get(v, 0.0) for v in nodes]) if len(d) > 1 else 0.0
    while time.monotonic() < deadline:
        cost = route_cost(tour, d)
        remaining = float(t0) - cost
        best_score = -float("inf")
        best_v = None
        best_pos = None
        for v in nodes:
            if v == 0 or in_tour[v]:
                continue
            delta, pos = best_insertion(tour, v, d, deadline)
            if pos is None or delta > remaining + 1e-9:
                continue
            pv = prize.get(v, 0.0)
            if delta > 1e-9:
                if pv <= 0.0:
                    continue
                score = pv / delta
            else:
                if pv <= 0.0 and delta > -1e-9:
                    continue
                reduction = max(0.0, -delta)
                score = 1.0e18 + pv * 1.0e6 + (1.0 + max_price) * reduction * 1.0e3
            if score > best_score:
                best_score = score
                best_v = v
                best_pos = pos
        if best_v is None:
            break
        tour.insert(best_pos, best_v)
        in_tour[best_v] = True
        inserted += 1
    return inserted


def insert_passes(tour, in_tour, d, prize, t0, deadline, passes=1):
    nodes = [v for v in range(1, len(d)) if v != 0 and not in_tour[v]]
    depot = tour[0]
    nodes.sort(
        key=lambda v: (prize.get(v, 0.0) + 1.0) / max(2.0 * d[depot][v], 1.0e-9),
        reverse=True,
    )
    total_inserted = 0
    for _ in range(max(1, passes)):
        changed = False
        for v in nodes:
            if in_tour[v] or time.monotonic() > deadline:
                continue
            delta, pos = best_insertion(tour, v, d, deadline)
            if pos is None:
                break
            remaining = float(t0) - route_cost(tour, d)
            if delta <= remaining + 1e-9 and (delta < -1e-9 or prize.get(v, 0.0) > 0.0):
                tour.insert(pos, v)
                in_tour[v] = True
                changed = True
                total_inserted += 1
        if not changed:
            break
    return total_inserted


def two_opt_improve(tour, d, deadline):
    improved = False
    while time.monotonic() < deadline:
        s = tour[:-1]  # starts with depot, does not include final depot
        m = len(s)
        if m < 4:  # depot + at least two other cities; nothing useful to reverse
            return improved
        current = route_cost(tour, d)
        best_delta = -1e-9
        bi = bj = -1
        depot = s[0]
        for i in range(1, m - 1):
            if time.monotonic() > deadline:
                return improved
            a = s[i - 1]
            b = s[i]
            dab = d[a][b]
            for j in range(i + 1, m):
                c = s[j]
                e = s[j + 1] if j + 1 < m else depot
                delta = d[a][c] + d[b][e] - dab - d[c][e]
                if delta < best_delta:
                    best_delta = delta
                    bi, bj = i, j
        if bi < 0:
            break
        new_s = s[:bi] + s[bi:bj + 1][::-1] + s[bj + 1:]
        tour[:] = new_s + [s[0]]
        improved = True
    return improved


def heuristic_solution(n, depot, d, prize, t0, deadline):
    nondep = [v for v in range(1, n + 1) if v != depot]
    if len(nondep) < 2:
        return [depot, depot]

    best_pair = None
    best_pair_profit = -float("inf")
    best_pair_cost = float("inf")
    min_pair = None
    min_pair_cost = float("inf")

    for ai in range(len(nondep)):
        a = nondep[ai]
        for bi in range(ai + 1, len(nondep)):
            b = nondep[bi]
            cost = d[depot][a] + d[a][b] + d[b][depot]
            if cost < min_pair_cost:
                min_pair_cost = cost
                min_pair = (a, b)
            if cost <= float(t0) + 1e-9:
                prof = prize.get(a, 0.0) + prize.get(b, 0.0)
                if prof > best_pair_profit + 1e-9 or (
                    abs(prof - best_pair_profit) <= 1e-9 and cost < best_pair_cost
                ):
                    best_pair = (a, b)
                    best_pair_profit = prof
                    best_pair_cost = cost
    if best_pair is not None:
        a, b = best_pair
    elif min_pair is not None:
        a, b = min_pair
    else:
        return [depot, depot]

    tour = [depot, a, b, depot]
    in_tour = [False] * (n + 1)
    in_tour[depot] = True
    in_tour[a] = True
    in_tour[b] = True

    if n <= 260:
        for _ in range(5):
            if time.monotonic() > deadline:
                break
            before_count = len(tour)
            before_cost = route_cost(tour, d)
            insert_all_possible(tour, in_tour, d, prize, t0, deadline)
            if len(tour) <= 180:
                two_opt_improve(tour, d, deadline)
            insert_all_possible(tour, in_tour, d, prize, t0, deadline)
            after_count = len(tour)
            after_cost = route_cost(tour, d)
            if after_count == before_count and after_cost >= before_cost - 1e-9:
                break
    else:
        for _ in range(12):
            if time.monotonic() > deadline:
                break
            changed = insert_passes(tour, in_tour, d, prize, t0, deadline, passes=1)
            improved = False
            if len(tour) <= 130:
                improved = two_opt_improve(tour, d, deadline)
            if changed == 0 and not improved:
                break

    return tour


def solve_with_mip(n, depot, t0, prize, d, time_limit, start_solution, consider_solution):
    if gp is None or n < 3:
        return
    nodes = list(range(1, n + 1))
    edge_pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]

    m = gp.Model("selective_tour")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.LazyConstraints = 1
    m.Params.OutputFlag = 0
    m.Params.LogToConsole = 0
    m.Params.TimeLimit = max(0.0, float(time_limit))

    y = {v: m.addVar(vtype=GRB.BINARY, obj=prize.get(v, 0.0), name=f"y_{v}") for v in nodes}
    x = {
        (i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        for (i, j) in edge_pairs
    }

    m.setObjective(gp.quicksum(prize.get(v, 0.0) * y[v] for v in nodes), GRB.MAXIMIZE)
    m.addConstr(y[depot] == 1.0, name="depot_selected")
    m.addConstr(gp.quicksum(y[v] for v in nodes) >= 3.0, name="at_least_three")
    m.addConstr(
        gp.quicksum(d[i][j] * x[(i, j)] for (i, j) in edge_pairs) <= float(t0),
        name="time_budget",
    )
    for v in nodes:
        incident = []
        for j in range(1, n + 1):
            if j == v:
                continue
            incident.append(x[norm_edge(v, j)])
        m.addConstr(gp.quicksum(incident) == 2.0 * y[v], name=f"degree_{v}")
    m.addConstr(
        gp.quicksum(x[e] for e in edge_pairs) == gp.quicksum(y[v] for v in nodes),
        name="edge_count",
    )

    if start_solution is not None:
        used = set()
        for e in start_solution.get("edges", []):
            if len(e) >= 2:
                used.add(norm_edge(int(e[0]), int(e[1])))
        visited = set(int(v) for v in start_solution.get("visited_nodes", []))
        for v in nodes:
            y[v].Start = 1.0 if v in visited else 0.0
        y[depot].Start = 1.0
        for e in edge_pairs:
            x[e].Start = 1.0 if e in used else 0.0

    m._nodes = nodes
    m._depot = depot
    m._prize = prize
    m._edge_pairs = edge_pairs
    m._x = x
    m._y = y
    m._consider = consider_solution

    def subtour_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        pairs = model._edge_pairs
        xvars = model._x
        yvars = model._y
        vs = model._nodes
        dep = model._depot

        xvals = model.cbGetSolution([xvars[p] for p in pairs])
        yvals = model.cbGetSolution([yvars[v] for v in vs])
        selected = [v for v, val in zip(vs, yvals) if val > 0.5]
        if len(selected) < 3 or dep not in selected:
            return

        adj = {v: [] for v in selected}
        chosen_pairs = []
        for p, val in zip(pairs, xvals):
            if val > 0.5:
                i, j = p
                if i in adj and j in adj:
                    adj[i].append(j)
                    adj[j].append(i)
                    chosen_pairs.append(p)

        seen = set()
        bad = []
        for v in selected:
            if v in seen:
                continue
            stack = [v]
            seen.add(v)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adj.get(u, []):
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            if dep not in comp:
                bad.append(comp)

        if bad:
            for comp_nodes in bad:
                terms = []
                for ai in range(len(comp_nodes)):
                    for bi in range(ai + 1, len(comp_nodes)):
                        i, j = norm_edge(comp_nodes[ai], comp_nodes[bi])
                        var = xvars.get((i, j))
                        if var is not None:
                            terms.append(var)
                if terms:
                    model.cbLazy(
                        gp.quicksum(terms)
                        <= gp.quicksum(yvars[v] for v in comp_nodes) - 1.0
                    )
            return

        sol = solution_from_edges(vs, dep, chosen_pairs, model._prize)
        if sol is not None:
            model._consider(sol, True)

    m.optimize(subtour_callback)

    if m.SolCount > 0:
        xsol = m.getAttr("x", [x[e] for e in edge_pairs])
        edges = [e for e, val in zip(edge_pairs, xsol) if val > 0.5]
        sol = solution_from_edges(nodes, depot, edges, prize)
        if sol is not None:
            cost = sum(d[i][j] for i, j in (norm_edge(a, b) for a, b in [tuple(map(int, e)) for e in sol["edges"]]))
            consider_solution(sol, cost <= float(t0) + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, int(args.time_limit))
    reserve = min(0.5, max(0.05, 0.02 * max(1, int(args.time_limit))))

    best_sol = None
    best_obj = -float("inf")
    fallback_sol = None

    def consider_solution(sol, feasible=True):
        nonlocal best_sol, best_obj
        if sol is None:
            return False
        obj = float(sol.get("objective_value", 0.0))
        if feasible and obj > best_obj + 1e-9:
            best_obj = obj
            best_sol = sol
            if logger:
                logger.log_solution(obj, sol)
            return True
        return False

    try:
        n, depot, t0, prize, d = read_instance(args.instance_path)
        edge_count = n * (n - 1) // 2
        use_mip = gp is not None and n >= 3 and edge_count <= 45000

        now = time.monotonic()
        if use_mip:
            h_budget = min(2.0, max(0.2, 0.20 * max(1, int(args.time_limit))))
            h_deadline = min(deadline - reserve, now + h_budget)
        else:
            h_deadline = deadline - reserve

        tour = heuristic_solution(n, depot, d, prize, t0, h_deadline)
        sol = make_solution_from_tour(tour, depot, prize)
        if sol is not None:
            cost = route_cost(make_solution_from_tour(tour, depot, prize)["tour"], d) if False else route_cost(tour, d)
            if cost <= float(t0) + 1e-6:
                consider_solution(sol, True)
            else:
                fallback_sol = sol

        if use_mip:
            remaining = deadline - reserve - time.monotonic()
            if remaining > 1e-3:
                start_solution = best_sol if best_sol is not None else None
                solve_with_mip(
                    n, depot, t0, prize, d, remaining, start_solution, consider_solution
                )

        if best_sol is None:
            if fallback_sol is not None:
                best_sol = fallback_sol
            else:
                nondep = [v for v in range(1, n + 1) if v != depot]
                if len(nondep) >= 2:
                    a, b = min(
                        ((i, j) for i in nondep for j in nondep if i < j),
                        key=lambda p: d[p[0]][depot] + d[p[0]][p[1]] + d[p[1]][depot],
                    )
                    best_sol = make_solution_from_tour([depot, a, b, depot], depot, prize)
                if best_sol is None:
                    best_sol = {
                        "objective_value": 0.0,
                        "visited_nodes": [depot],
                        "edges": [],
                        "tour": [depot, depot],
                    }

        best_sol["objective_value"] = float(best_sol.get("objective_value", 0.0))
        if logger and best_sol is not None:
            # The incumbent has already been logged by consider_solution whenever it improved.
            pass

    finally:
        if best_sol is None:
            best_sol = {
                "objective_value": 0.0,
                "visited_nodes": [],
                "edges": [],
                "tour": [],
            }
        with open(args.solution_path, "w") as f:
            json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()