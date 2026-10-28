import argparse
import json
import math
import random
import time
from collections import defaultdict

INF = float('inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start_time = time.time()
    total_limit = max(5, args.time_limit)
    global_deadline = start_time + total_limit - 1.5

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst["num_vertices"])
    depot = int(inst["depot"])
    m = int(inst["num_products"])

    c = [[0.0] * n for _ in range(n)]
    for key, v in inst["travel_costs"].items():
        i, j = key.split(',')
        i = int(i); j = int(j)
        c[i][j] = float(v)
        c[j][i] = float(v)

    d = [int(x) for x in inst["product_demands"]]
    pm = [[int(k) for k in inst["product_markets"][str(p)]] for p in range(m)]
    price = {}
    sup = {}
    for key, v in inst["purchase_costs"].items():
        p, k = key.split(',')
        price[(int(p), int(k))] = float(v)
    for key, v in inst["supplies"].items():
        p, k = key.split(',')
        sup[(int(p), int(k))] = int(v)

    sorted_prod = []
    for p in range(m):
        lst = sorted(((price[(p, k)], sup[(p, k)], k) for k in pm[p]))
        sorted_prod.append(lst)

    prods_at = defaultdict(list)
    for p in range(m):
        for k in pm[p]:
            prods_at[k].append(p)

    markets = [i for i in range(n) if i != depot]

    mandatory = set()
    for p in range(m):
        tot = sum(sup[(p, k)] for k in pm[p])
        for k in pm[p]:
            if tot - sup[(p, k)] < d[p]:
                mandatory.add(k)

    dl = [global_deadline]  # mutable deadline for heuristic phase

    # ---------- purchasing helpers ----------
    def prod_cost(p, S, exclude=None, include=None):
        need = d[p]
        cost = 0.0
        for pr, s_, k in sorted_prod[p]:
            if k == exclude:
                continue
            if k in S or k == include:
                t = s_ if s_ < need else need
                cost += pr * t
                need -= t
                if need <= 0:
                    return cost
        return INF

    # ---------- tour helpers ----------
    def tour_cost(tour):
        L = len(tour)
        tc = 0.0
        for i in range(L):
            tc += c[tour[i]][tour[(i + 1) % L]]
        return tc

    def build_tour(S):
        tour = [depot]
        cur = depot
        rem = set(S)
        while rem:
            nxt = min(rem, key=lambda j: c[cur][j])
            tour.append(nxt)
            rem.discard(nxt)
            cur = nxt
        return tour

    def two_opt(tour):
        L = len(tour)
        if L < 4:
            return tour[:]
        t = tour[:]
        improved = True
        while improved and time.time() < dl[0]:
            improved = False
            for i in range(L - 1):
                a = t[i]; b = t[i + 1]
                ca = c[a]
                for j in range(i + 2, L):
                    if i == 0 and j == L - 1:
                        continue
                    e = t[j]; f = t[(j + 1) % L]
                    delta = ca[e] + c[b][f] - ca[b] - c[e][f]
                    if delta < -1e-9:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        b = t[i + 1]
                        improved = True
        return t

    def best_insert(tour, k):
        L = len(tour)
        bins = INF
        bpos = 0
        for i in range(L):
            a = tour[i]; b = tour[(i + 1) % L]
            v = c[a][k] + c[k][b] - c[a][b]
            if v < bins:
                bins = v
                bpos = i
        return bins, bpos

    # ---------- state ----------
    def make_state(S):
        pc = []
        total = 0.0
        avail = []
        for p in range(m):
            cst = prod_cost(p, S)
            if cst == INF:
                return None
            pc.append(cst)
            total += cst
            avail.append(sum(sup[(p, k)] for k in pm[p] if k in S))
        tour = build_tour(S)
        tour = two_opt(tour)
        return {'S': set(S), 'tour': tour, 'pc': pc, 'ptotal': total,
                'avail': avail, 'tcost': tour_cost(tour)}

    def clone(st):
        return {'S': set(st['S']), 'tour': st['tour'][:], 'pc': st['pc'][:],
                'ptotal': st['ptotal'], 'avail': st['avail'][:], 'tcost': st['tcost']}

    def obj(st):
        return st['ptotal'] + st['tcost']

    def apply_move(st, move):
        S = st['S']; tour = st['tour']
        if move[0] == 'drop':
            k = move[1]
            i = tour.index(k)
            L = len(tour)
            a = tour[i - 1]; b = tour[(i + 1) % L]
            if a == k or b == k:
                st['tcost'] -= 2 * c[depot][k]
            else:
                st['tcost'] += c[a][b] - c[a][k] - c[k][b]
            del tour[i]
            S.discard(k)
            for p in prods_at[k]:
                nc = prod_cost(p, S)
                st['ptotal'] += nc - st['pc'][p]
                st['pc'][p] = nc
                st['avail'][p] -= sup[(p, k)]
        else:
            _, k, bpos = move
            L = len(tour)
            a = tour[bpos]; b = tour[(bpos + 1) % L]
            st['tcost'] += c[a][k] + c[k][b] - c[a][b]
            tour.insert(bpos + 1, k)
            S.add(k)
            for p in prods_at[k]:
                nc = prod_cost(p, S)
                st['ptotal'] += nc - st['pc'][p]
                st['pc'][p] = nc
                st['avail'][p] += sup[(p, k)]

    def try_improve_once(st):
        if time.time() > dl[0]:
            return None
        S = st['S']; tour = st['tour']
        L = len(tour)
        best_delta = -1e-9
        best_move = None
        posidx = {v: i for i, v in enumerate(tour)}
        # drop moves
        if len(S) > 1:
            for k in S:
                if k in mandatory:
                    continue
                feas = True
                for p in prods_at[k]:
                    if st['avail'][p] - sup[(p, k)] < d[p]:
                        feas = False
                        break
                if not feas:
                    continue
                dp = 0.0
                for p in prods_at[k]:
                    nc = prod_cost(p, S, exclude=k)
                    if nc == INF:
                        dp = INF
                        break
                    dp += nc - st['pc'][p]
                if dp == INF:
                    continue
                i = posidx[k]
                a = tour[i - 1]; b = tour[(i + 1) % L]
                dt = c[a][b] - c[a][k] - c[k][b]
                delta = dp + dt
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('drop', k)
        # add moves
        cnt = 0
        for k in markets:
            if k in S:
                continue
            cnt += 1
            if (cnt & 31) == 0 and time.time() > dl[0]:
                break
            dp = 0.0
            for p in prods_at[k]:
                nc = prod_cost(p, S, include=k)
                dp += nc - st['pc'][p]
            bins, bpos = best_insert(tour, k)
            delta = dp + bins
            if delta < best_delta:
                best_delta = delta
                best_move = ('add', k, bpos)
        return best_move

    def local_search(st):
        while True:
            changed = False
            while time.time() < dl[0]:
                mv = try_improve_once(st)
                if mv is None:
                    break
                apply_move(st, mv)
                changed = True
            old = st['tcost']
            st['tour'] = two_opt(st['tour'])
            st['tcost'] = tour_cost(st['tour'])
            if st['tcost'] < old - 1e-9:
                changed = True
            if not changed or time.time() >= dl[0]:
                break

    def build_solution(st):
        S = st['S']; tour = st['tour']
        purchases = {}
        ptotal = 0.0
        for p in range(m):
            need = d[p]
            alloc = {}
            for pr, s_, k in sorted_prod[p]:
                if need <= 0:
                    break
                if k in S:
                    t = s_ if s_ < need else need
                    alloc[str(k)] = int(t)
                    ptotal += pr * t
                    need -= t
            purchases[str(p)] = alloc
        L = len(tour)
        if L == 2:
            edges = [[int(tour[0]), int(tour[1])], [int(tour[0]), int(tour[1])]]
        else:
            edges = [[int(tour[i]), int(tour[(i + 1) % L])] for i in range(L)]
        tc = tour_cost(tour)
        return {
            "objective_value": float(ptotal + tc),
            "visited_markets": sorted(int(v) for v in S),
            "tour": [int(v) for v in tour],
            "tour_edges": edges,
            "purchases": purchases
        }

    # ---------- initial solution ----------
    S0 = set(mandatory)
    for p in range(m):
        need = d[p]
        for pr, s_, k in sorted_prod[p]:
            if k in S0:
                need -= s_
        if need > 0:
            for pr, s_, k in sorted_prod[p]:
                if need <= 0:
                    break
                if k not in S0:
                    S0.add(k)
                    need -= s_
    st = make_state(S0)
    local_search(st)
    best = clone(st)
    best_obj = obj(best)
    best_sol = build_solution(best)
    best_obj = best_sol["objective_value"]
    if logger:
        logger.log_solution(best_obj, best_sol)

    # ---------- decide phases ----------
    use_mip = (n <= 150)
    if use_mip:
        heur_deadline = min(global_deadline,
                            start_time + max(3.0, 0.2 * total_limit))
    else:
        heur_deadline = global_deadline
    dl[0] = heur_deadline

    # ---------- ILS ----------
    rng = random.Random(0)

    def perturb(state):
        kmoves = rng.randint(2, 5)
        for _ in range(kmoves):
            S = state['S']
            if rng.random() < 0.5 and len(S) > 1:
                cands = [k for k in S if k not in mandatory]
                rng.shuffle(cands)
                for k in cands:
                    ok = True
                    for p in prods_at[k]:
                        if state['avail'][p] - sup[(p, k)] < d[p]:
                            ok = False
                            break
                    if ok:
                        apply_move(state, ('drop', k))
                        break
            else:
                cands = [k for k in markets if k not in S]
                if cands:
                    k = rng.choice(cands)
                    _, bpos = best_insert(state['tour'], k)
                    apply_move(state, ('add', k, bpos))

    while time.time() < dl[0]:
        cur = clone(best)
        perturb(cur)
        local_search(cur)
        o = obj(cur)
        if o < best_obj - 1e-9:
            best = clone(cur)
            sol = build_solution(best)
            best_obj = sol["objective_value"]
            best_sol = sol
            if logger:
                logger.log_solution(best_obj, best_sol)

    # ---------- MIP phase (exact, for moderate sizes) ----------
    holder = {'obj': best_obj, 'sol': best_sol}
    if use_mip and time.time() < global_deadline - 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB, quicksum

            model = gp.Model("tpp")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.LazyConstraints = 1
            model.Params.TimeLimit = max(1.0, global_deadline - time.time())

            y = {}
            for k in markets:
                y[k] = model.addVar(vtype=GRB.BINARY, name="y_%d" % k)
                if k in mandatory:
                    y[k].LB = 1.0
            x = {}
            for i in range(n):
                for j in range(i + 1, n):
                    ub = 2.0 if (i == depot or j == depot) else 1.0
                    x[(i, j)] = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=ub,
                                             obj=c[i][j], name="x_%d_%d" % (i, j))
            qkeys = []
            q = {}
            for p in range(m):
                for k in pm[p]:
                    q[(p, k)] = model.addVar(vtype=GRB.INTEGER, lb=0.0,
                                             ub=sup[(p, k)], obj=price[(p, k)],
                                             name="q_%d_%d" % (p, k))
                    qkeys.append((p, k))

            inc = defaultdict(list)
            for (i, j), var in x.items():
                inc[i].append(var)
                inc[j].append(var)
            model.addConstr(quicksum(inc[depot]) == 2)
            for k in markets:
                model.addConstr(quicksum(inc[k]) == 2 * y[k])
            for p in range(m):
                model.addConstr(quicksum(q[(p, k)] for k in pm[p]) == d[p])
                for k in pm[p]:
                    model.addConstr(q[(p, k)] <= sup[(p, k)] * y[k])
            model.ModelSense = GRB.MINIMIZE

            # warm start
            for k in markets:
                y[k].Start = 1.0 if k in best['S'] else 0.0
            for e in x:
                x[e].Start = 0.0
            btour = best['tour']
            L = len(btour)
            if L == 2:
                e = (min(btour), max(btour))
                x[e].Start = 2.0
            else:
                for i in range(L):
                    a, b = btour[i], btour[(i + 1) % L]
                    e = (min(a, b), max(a, b))
                    x[e].Start = 1.0
            for p in range(m):
                al = best_sol['purchases'][str(p)]
                for k in pm[p]:
                    q[(p, k)].Start = float(al.get(str(k), 0))

            xkeys = list(x.keys())
            xvars = [x[e] for e in xkeys]
            yvars = [y[k] for k in markets]
            qvars = [q[key] for key in qkeys]

            def extract(xvals, qvals):
                adj = defaultdict(dict)
                for e, cnt in xvals.items():
                    if cnt > 0:
                        a, b = e
                        adj[a][b] = adj[a].get(b, 0) + cnt
                        adj[b][a] = adj[b].get(a, 0) + cnt
                tour = [depot]
                cur = depot
                while True:
                    nxt = None
                    for j2, cnt in adj[cur].items():
                        if cnt > 0:
                            nxt = j2
                            break
                    if nxt is None:
                        break
                    adj[cur][nxt] -= 1
                    adj[nxt][cur] -= 1
                    if nxt == depot:
                        break
                    tour.append(nxt)
                    cur = nxt
                purchases = {str(p): {} for p in range(m)}
                ptot = 0.0
                for (p, k), val in qvals.items():
                    v = int(round(val))
                    if v > 0:
                        purchases[str(p)][str(k)] = v
                        ptot += price[(p, k)] * v
                L2 = len(tour)
                if L2 == 2:
                    edges = [[int(tour[0]), int(tour[1])],
                             [int(tour[0]), int(tour[1])]]
                else:
                    edges = [[int(tour[i]), int(tour[(i + 1) % L2])]
                             for i in range(L2)]
                tc = tour_cost(tour)
                return {
                    "objective_value": float(ptot + tc),
                    "visited_markets": sorted(int(v) for v in tour if v != depot),
                    "tour": [int(v) for v in tour],
                    "tour_edges": edges,
                    "purchases": purchases
                }

            def cb(mdl, where):
                if where != GRB.Callback.MIPSOL:
                    return
                xv = mdl.cbGetSolution(xvars)
                yv = mdl.cbGetSolution(yvars)
                yval = {markets[idx]: yv[idx] for idx in range(len(markets))}
                visited = set([depot] + [k for k in markets if yval[k] > 0.5])
                parent = {v: v for v in visited}

                def find(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a

                xvals = {}
                for idx in range(len(xkeys)):
                    cnt = int(round(xv[idx]))
                    if cnt > 0:
                        e = xkeys[idx]
                        xvals[e] = cnt
                        if e[0] in parent and e[1] in parent:
                            ra, rb = find(e[0]), find(e[1])
                            if ra != rb:
                                parent[ra] = rb
                comps = defaultdict(set)
                for v in visited:
                    comps[find(v)].add(v)
                droot = find(depot)
                bad = [s for r, s in comps.items() if r != droot]
                if bad:
                    for Sset in bad:
                        expr = quicksum(x[e] for e in xkeys
                                        if (e[0] in Sset) != (e[1] in Sset))
                        for k in Sset:
                            mdl.cbLazy(expr >= 2 * y[k])
                    return
                objv = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objv < holder['obj'] - 1e-6:
                    qv = mdl.cbGetSolution(qvars)
                    qvals = {qkeys[idx]: qv[idx] for idx in range(len(qkeys))}
                    sol = extract(xvals, qvals)
                    holder['obj'] = sol["objective_value"]
                    holder['sol'] = sol
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)

            model.optimize(cb)

            if model.SolCount > 0 and model.objVal < holder['obj'] - 1e-6:
                xvals = {}
                for e in xkeys:
                    cnt = int(round(x[e].X))
                    if cnt > 0:
                        xvals[e] = cnt
                qvals = {key: q[key].X for key in qkeys}
                sol = extract(xvals, qvals)
                if sol["objective_value"] < holder['obj'] - 1e-9:
                    holder['obj'] = sol["objective_value"]
                    holder['sol'] = sol
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)
        except Exception:
            # fall back: continue ILS with remaining time
            dl[0] = global_deadline
            while time.time() < dl[0]:
                cur = clone(best)
                perturb(cur)
                local_search(cur)
                o = obj(cur)
                if o < best_obj - 1e-9:
                    best = clone(cur)
                    sol = build_solution(best)
                    best_obj = sol["objective_value"]
                    if sol["objective_value"] < holder['obj'] - 1e-9:
                        holder['obj'] = sol["objective_value"]
                        holder['sol'] = sol
                        if logger:
                            logger.log_solution(holder['obj'], holder['sol'])

    with open(args.solution_path, 'w') as f:
        json.dump(holder['sol'], f, indent=2)


if __name__ == "__main__":
    main()