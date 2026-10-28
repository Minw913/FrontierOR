"""
Purchasing Tour Problem solver.

Approach:
 1. Heuristic: mandatory-market seeding, greedy feasibility completion,
    cheapest-insertion tour construction, 2-opt, and add/remove local search.
    For a fixed set of visited markets the optimal purchase plan is computed
    per product by a cheapest-price-first greedy allocation.
 2. An exact MIP (Gurobi) with degree equations, lazy subtour-elimination
    cuts and a warm start from the heuristic, used to polish the solution
    within the remaining time limit.
"""

import argparse
import json
import math
import time
from collections import defaultdict

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# --------------------------------------------------------------------------
# Instance loading
# --------------------------------------------------------------------------
class Inst:
    pass


def load_instance(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    d = Inst()
    d.nV = int(raw['num_vertices'])
    d.nM = int(raw.get('num_markets', d.nV - 1))
    d.nP = int(raw.get('num_products', len(raw.get('product_demands', []))))
    d.depot = int(raw['depot'])

    # travel cost matrix
    d.c = [[0.0] * d.nV for _ in range(d.nV)]
    tc = raw.get('travel_costs') or {}
    for k, v in tc.items():
        ks = k.split(',')
        i = int(ks[0]); j = int(ks[1])
        d.c[i][j] = float(v)
        d.c[j][i] = float(v)
    if not tc and raw.get('coordinates'):
        coords = raw['coordinates']
        for i in range(d.nV):
            xi, yi = coords[i][0], coords[i][1]
            for j in range(i + 1, d.nV):
                w = math.hypot(xi - coords[j][0], yi - coords[j][1])
                d.c[i][j] = w
                d.c[j][i] = w

    d.markets = [v for v in range(d.nV) if v != d.depot]
    d.demands = [float(x) for x in raw.get('product_demands', [])]

    pm = raw.get('product_markets') or {}
    pcost = raw.get('purchase_costs') or {}
    sup = raw.get('supplies') or {}

    # resolve market indices (normally they are vertex indices already)
    raw_ids = set()
    for p in range(d.nP):
        for i in pm.get(str(p), []):
            raw_ids.add(int(i))
    if all(0 <= i < d.nV and i != d.depot for i in raw_ids):
        mapping = {i: i for i in raw_ids}
    else:
        nondepot = [v for v in range(d.nV) if v != d.depot]
        mapping = {}
        for k2, rid in enumerate(sorted(raw_ids)):
            if k2 < len(nondepot):
                mapping[rid] = nondepot[k2]

    d.av = [[] for _ in range(d.nP)]          # per product: (market, price, supply)
    d.price = {}
    d.supply = {}
    d.market_products = defaultdict(list)     # market -> [products]
    for p in range(d.nP):
        seen_mk = set()
        for rid0 in pm.get(str(p), []):
            rid = int(rid0)
            if rid in seen_mk:
                continue
            seen_mk.add(rid)
            i = mapping.get(rid, rid)
            pr = float(pcost.get(f'{p},{rid}', pcost.get(f'{p},{i}', 0.0)))
            s = float(sup.get(f'{p},{rid}', sup.get(f'{p},{i}', 0.0)))
            if s <= 1e-12:
                continue
            d.av[p].append((i, pr, s))
            d.price[(p, i)] = pr
            d.supply[(p, i)] = s
            d.market_products[i].append(p)
    d.av_sorted = [sorted(lst, key=lambda t: (t[1], t[0])) for lst in d.av]

    # mandatory markets
    d.mandatory = set()
    for p in range(d.nP):
        T = sum(s for (_, _, s) in d.av[p])
        D = d.demands[p]
        for (i, pr, s) in d.av[p]:
            if T - s < D - 1e-9:
                d.mandatory.add(i)

    max_pr = max((pr for lst in d.av for (_, pr, _) in lst), default=1.0)
    max_c = 0.0
    for i in range(d.nV):
        row = d.c[i]
        for j in range(d.nV):
            if row[j] > max_c:
                max_c = row[j]
    d.BIG = (max_pr * (sum(d.demands) + 1.0) + max_c * (d.nV + 1.0)) * 100.0 + 1.0
    return d


# --------------------------------------------------------------------------
# Purchase allocation (optimal for a fixed visited set, per product)
# --------------------------------------------------------------------------
def product_greedy(p, S, data, extra=None, exclude=None):
    D = data.demands[p]
    rem = D
    cost = 0.0
    for (i, pr, s) in data.av_sorted[p]:
        if exclude is not None and i == exclude:
            continue
        if i in S or i == extra:
            q = s if s < rem else rem
            cost += pr * q
            rem -= q
            if rem <= 1e-9:
                break
    return cost, D - rem


def compute_purchases(S, data):
    total = 0.0
    purchases = {}
    deficit = 0.0
    for p in range(data.nP):
        D = data.demands[p]
        rem = D
        pd = {}
        for (i, pr, s) in data.av_sorted[p]:
            if i in S:
                q = s if s < rem else rem
                if q > 1e-9:
                    pd[i] = q
                    total += pr * q
                    rem -= q
                if rem <= 1e-9:
                    break
        if rem > 1e-9:
            deficit += rem
        purchases[p] = pd
    return total, purchases, deficit


# --------------------------------------------------------------------------
# TSP helpers
# --------------------------------------------------------------------------
def tour_cost(tour, c):
    return sum(c[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))


def nearest_neighbor(S, depot, c):
    if not S:
        return [depot]
    unvisited = set(S)
    tour = [depot]
    cur = depot
    while unvisited:
        nxt = min(unvisited, key=lambda i: c[cur][i])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    tour.append(depot)
    return tour


def two_opt(tour, c, deadline=None):
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            a = tour[i - 1]; b = tour[i]
            for j in range(i + 1, n - 1):
                u = tour[j]; v = tour[j + 1]
                if c[a][u] + c[b][v] < c[a][b] + c[u][v] - 1e-9:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    b = tour[i]
                    improved = True
            if deadline is not None and time.time() > deadline:
                return tour
    return tour


def cheapest_insertion(tour, i, c):
    if len(tour) == 1:
        return 2.0 * c[tour[0]][i], 0
    best = float('inf')
    pos = 0
    for k in range(len(tour) - 1):
        u = tour[k]; v = tour[k + 1]
        d = c[u][i] + c[i][v] - c[u][v]
        if d < best:
            best = d
            pos = k
    return best, pos


def insert_into_tour(tour, i, pos):
    if len(tour) == 1:
        return [tour[0], i, tour[0]]
    return tour[:pos + 1] + [i] + tour[pos + 1:]


def build_tour_from_edges(edge_list, depot):
    if not edge_list:
        return [depot]
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    tour = [depot]
    prev = None
    cur = depot
    while True:
        nbrs = adj[cur]
        if not nbrs:
            break
        nxt = None
        for nb in nbrs:
            if nb != prev:
                nxt = nb
                break
        if nxt is None:
            nxt = nbrs[0]
        adj[cur].remove(nxt)
        adj[nxt].remove(cur)
        prev = cur
        cur = nxt
        tour.append(cur)
        if cur == depot:
            break
    return tour


# --------------------------------------------------------------------------
# Solution builder
# --------------------------------------------------------------------------
def build_solution(tour, purchases, data):
    c = data.c
    depot = data.depot
    travel = 0.0
    edges = []
    for k in range(len(tour) - 1):
        u = tour[k]; v = tour[k + 1]
        edges.append([int(u), int(v)])
        travel += c[u][v]
    pc = 0.0
    pur_out = {}
    for p in range(data.nP):
        pd = purchases.get(p, {}) if purchases else {}
        sub = {}
        for i, qty in pd.items():
            if qty > 1e-9:
                rq = round(qty)
                if rq >= 1 and abs(qty - rq) < 1e-6:
                    qv = int(rq)
                else:
                    qv = float(qty)
                sub[str(i)] = qv
                pc += data.price[(p, i)] * qv
        pur_out[str(p)] = sub
    visited = []
    seen = set()
    for v in tour:
        if v != depot and v not in seen:
            seen.add(v)
            visited.append(int(v))
    return {
        'objective_value': float(travel + pc),
        'visited_markets': visited,
        'tour': [int(v) for v in tour],
        'tour_edges': edges,
        'purchases': pur_out,
    }


# --------------------------------------------------------------------------
# Heuristic
# --------------------------------------------------------------------------
def heuristic(data, deadline):
    depot = data.depot
    c = data.c
    S = set(data.mandatory)
    prod_cost = [0.0] * data.nP
    prod_bought = [0.0] * data.nP
    for p in range(data.nP):
        prod_cost[p], prod_bought[p] = product_greedy(p, S, data)
    tour = nearest_neighbor(S, depot, c)
    tour = two_opt(tour, c, deadline)

    # ---- feasibility completion ----
    guard = 0
    while True:
        deficit = 0.0
        for p in range(data.nP):
            if prod_bought[p] < data.demands[p] - 1e-9:
                deficit += data.demands[p] - prod_bought[p]
        if deficit <= 1e-9:
            break
        guard += 1
        if guard > data.nM + 5 or (deadline is not None and time.time() > deadline):
            # fast completion fallback
            for p in range(data.nP):
                if prod_bought[p] >= data.demands[p] - 1e-9:
                    continue
                covered = prod_bought[p]
                for (i, pr, s) in data.av_sorted[p]:
                    if i not in S:
                        S.add(i)
                        covered += s
                        if covered >= data.demands[p] - 1e-9:
                            break
            for p in range(data.nP):
                prod_cost[p], prod_bought[p] = product_greedy(p, S, data)
            tour = nearest_neighbor(S, depot, c)
            tour = two_opt(tour, c, deadline)
            break
        best_i = None
        best_pos = None
        best_score = float('inf')
        for i in data.markets:
            if i in S:
                continue
            carried = data.market_products.get(i)
            if not carried:
                continue
            dpc = 0.0
            dbg = 0.0
            for p in carried:
                nc, nb = product_greedy(p, S, data, extra=i)
                dpc += nc - prod_cost[p]
                dbg += nb - prod_bought[p]
            if dbg <= 1e-9:
                continue
            dtr, pos = cheapest_insertion(tour, i, c)
            score = dtr + dpc - data.BIG * dbg
            if score < best_score:
                best_score = score
                best_i = i
                best_pos = pos
        if best_i is None:
            break
        S.add(best_i)
        tour = insert_into_tour(tour, best_i, best_pos)
        for p in data.market_products[best_i]:
            prod_cost[p], prod_bought[p] = product_greedy(p, S, data)

    tour = two_opt(tour, c, deadline)

    # ---- add / remove local search ----
    changed = True
    it = 0
    while changed and it < 25:
        changed = False
        it += 1
        if deadline is not None and time.time() > deadline:
            break
        # removal pass
        for i in list(S):
            if deadline is not None and time.time() > deadline:
                break
            if i in data.mandatory:
                continue
            dpc = 0.0
            feas = True
            for p in data.market_products.get(i, []):
                nc, nb = product_greedy(p, S, data, exclude=i)
                if nb < data.demands[p] - 1e-9:
                    feas = False
                    break
                dpc += nc - prod_cost[p]
            if not feas:
                continue
            idx = tour.index(i)
            u = tour[idx - 1]
            v = tour[idx + 1]
            dtr = -(c[u][i] + c[i][v] - c[u][v])
            if dpc + dtr < -1e-6:
                S.discard(i)
                tour.pop(idx)
                for p in data.market_products.get(i, []):
                    prod_cost[p], prod_bought[p] = product_greedy(p, S, data)
                changed = True
        # addition pass
        for i in data.markets:
            if deadline is not None and time.time() > deadline:
                break
            if i in S:
                continue
            carried = data.market_products.get(i)
            if not carried:
                continue
            dpc = 0.0
            for p in carried:
                nc, nb = product_greedy(p, S, data, extra=i)
                dpc += nc - prod_cost[p]
            if dpc >= -1e-9:
                continue
            dtr, pos = cheapest_insertion(tour, i, c)
            if dpc + dtr < -1e-6:
                S.add(i)
                tour = insert_into_tour(tour, i, pos)
                for p in carried:
                    prod_cost[p], prod_bought[p] = product_greedy(p, S, data)
                changed = True
        if changed:
            tour = two_opt(tour, c, deadline)

    if not S:
        tour = [depot]
    _, purchases, _ = compute_purchases(S, data)
    return build_solution(tour, purchases, data)


def fallback_solution(data, deadline=None):
    S = set()
    for p in range(data.nP):
        for (i, pr, s) in data.av[p]:
            S.add(i)
    tour = nearest_neighbor(S, data.depot, data.c)
    tour = two_opt(tour, data.c, deadline)
    _, purchases, _ = compute_purchases(S, data)
    return build_solution(tour, purchases, data)


# --------------------------------------------------------------------------
# Exact MIP polish (Gurobi, lazy subtour elimination)
# --------------------------------------------------------------------------
def solve_mip(data, warm_sol, time_remaining, logger, best_so_far):
    import gurobipy as gp
    from gurobipy import GRB

    depot = data.depot
    c = data.c
    V = list(range(data.nV))
    markets = data.markets
    edges = [(i, j) for i in V for j in range(i + 1, data.nV)]

    model = gp.Model('purchasing_tour')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1
    model.Params.OutputFlag = 0
    if time_remaining and time_remaining > 0:
        model.Params.TimeLimit = max(1.0, float(time_remaining))

    x = model.addVars(edges, vtype=GRB.BINARY, name='x')
    for (i, j) in edges:
        if i == depot or j == depot:
            x[(i, j)].set(GRB.Attr.VType, GRB.INTEGER)
            x[(i, j)].set(GRB.Attr.UB, 2.0)
    y = model.addVars(markets, vtype=GRB.BINARY, name='y')
    qkeys = [(p, i) for p in range(data.nP) for (i, pr, s) in data.av[p]]
    q = model.addVars(qkeys, lb=0.0, name='q')
    model.update()

    obj = gp.quicksum(c[i][j] * x[(i, j)] for (i, j) in edges)
    obj += gp.quicksum(data.price[(p, i)] * q[(p, i)] for (p, i) in qkeys)
    model.setObjective(obj, GRB.MINIMIZE)

    v_edges = defaultdict(list)
    for e in edges:
        v_edges[e[0]].append(e)
        v_edges[e[1]].append(e)

    model.addConstr(gp.quicksum(x[e] for e in v_edges[depot]) == 2.0, name='deg_depot')
    for i in markets:
        model.addConstr(gp.quicksum(x[e] for e in v_edges[i]) == 2.0 * y[i], name=f'deg_{i}')
    for i in data.mandatory:
        model.addConstr(y[i] == 1.0, name=f'mand_{i}')
    for p in range(data.nP):
        avp = data.av[p]
        model.addConstr(gp.quicksum(q[(p, i)] for (i, pr, s) in avp) == data.demands[p],
                        name=f'dem_{p}')
        model.addConstr(gp.quicksum(s * y[i] for (i, pr, s) in avp) >= data.demands[p],
                        name=f'cap_{p}')
        for (i, pr, s) in avp:
            model.addConstr(q[(p, i)] <= s * y[i], name=f'link_{p}_{i}')

    # warm start
    if warm_sol is not None:
        edge_count = defaultdict(int)
        for e in warm_sol['tour_edges']:
            u, v = e[0], e[1]
            if u > v:
                u, v = v, u
            edge_count[(u, v)] += 1
        for e in edges:
            x[e].Start = float(edge_count.get(e, 0))
        visited = set(warm_sol['visited_markets'])
        for i in markets:
            y[i].Start = 1.0 if i in visited else 0.0
        wp = warm_sol.get('purchases', {})
        for (p, i) in qkeys:
            q[(p, i)].Start = float(wp.get(str(p), {}).get(str(i), 0.0))

    state = {'best': best_so_far if best_so_far is not None else float('inf')}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            xval = m.cbGetSolution(x)
            yval = m.cbGetSolution(y)
            selected = [i for i in markets if yval[i] > 0.5]
            nodeset = set(selected)
            nodeset.add(depot)
            adj = defaultdict(list)
            for (i, j) in edges:
                if xval[(i, j)] > 0.5:
                    adj[i].append(j)
                    adj[j].append(i)
            seen = set()
            bad = []
            for v0 in nodeset:
                if v0 in seen:
                    continue
                comp = []
                stack = [v0]
                seen.add(v0)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for w in adj[u]:
                        if w in nodeset and w not in seen:
                            seen.add(w)
                            stack.append(w)
                if depot not in comp:
                    bad.append(comp)
            if bad:
                for comp in bad:
                    m.cbLazy(gp.quicksum(x[(i, j)] for i in comp for j in comp if i < j)
                             <= len(comp) - 1)
                return
            # feasible single-tour incumbent -> log it
            qval = m.cbGetSolution(q)
            edge_list = []
            for (i, j) in edges:
                r = int(round(xval[(i, j)]))
                for _ in range(r):
                    edge_list.append((i, j))
            tour = build_tour_from_edges(edge_list, depot)
            purchases = defaultdict(dict)
            for (p, i) in qkeys:
                val = qval[(p, i)]
                if val > 1e-6:
                    purchases[p][i] = val
            sol = build_solution(tour, purchases, data)
            objv = sol['objective_value']
            if objv < state['best'] - 1e-6:
                state['best'] = objv
                if logger is not None:
                    try:
                        logger.log_solution(objv, sol)
                    except Exception:
                        pass

    model.optimize(cb)

    if model.SolCount > 0:
        xval = model.getAttr('X', x)
        yval = model.getAttr('X', y)
        qval = model.getAttr('X', q)
        selected = [i for i in markets if yval[i] > 0.5]
        edge_list = []
        for (i, j) in edges:
            r = int(round(xval[(i, j)]))
            for _ in range(r):
                edge_list.append((i, j))
        tour = build_tour_from_edges(edge_list, depot)
        purchases = defaultdict(dict)
        for (p, i) in qkeys:
            val = qval[(p, i)]
            if val > 1e-6:
                purchases[p][i] = val
        sol = build_solution(tour, purchases, data)
        if set(sol['visited_markets']) == set(selected):
            return sol
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, required=True)
    ap.add_argument('--log_path', default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense='minimize')
        except Exception:
            logger = None

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit)
    data = load_instance(args.instance_path)

    n_edges = data.nV * (data.nV - 1) // 2
    total_demand = sum(data.demands)
    use_mip = n_edges <= 60000 and total_demand > 1e-9 and data.nV >= 2

    if use_mip:
        heur_time = min(max(1.0, 0.3 * args.time_limit), 60.0)
        heur_deadline = min(deadline, t0 + heur_time)
    else:
        heur_deadline = deadline - 0.3

    try:
        best_sol = heuristic(data, heur_deadline)
    except Exception:
        best_sol = fallback_solution(data, deadline)
    best_obj = best_sol['objective_value']
    if logger is not None:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    remaining = deadline - time.time() - 0.5
    if use_mip and remaining > 1.0:
        try:
            mip_sol = solve_mip(data, best_sol, remaining, logger, best_obj)
        except Exception:
            mip_sol = None
        if mip_sol is not None and mip_sol['objective_value'] < best_obj - 1e-6:
            best_sol = mip_sol
            best_obj = mip_sol['objective_value']
            if logger is not None:
                try:
                    logger.log_solution(best_obj, best_sol)
                except Exception:
                    pass

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == '__main__':
    main()