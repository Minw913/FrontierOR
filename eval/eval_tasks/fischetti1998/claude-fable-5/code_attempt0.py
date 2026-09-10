import argparse
import json
import random
import time

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start_time = time.time()
    hard_deadline = start_time + max(5, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    depot = int(inst['depot'])
    t0 = float(inst['t0'])
    prizes_d = inst.get('prizes', {})
    tt = inst.get('travel_times', {})

    cities = set([depot])
    for k in prizes_d:
        cities.add(int(k))
    parsed_tt = []
    for k, v in tt.items():
        s = k.strip().strip('()')
        a, b = s.split(',')
        i, j = int(a.strip()), int(b.strip())
        cities.add(i)
        cities.add(j)
        parsed_tt.append((i, j, float(v)))
    cities = sorted(cities)
    N = max(cities)

    p = [0.0] * (N + 1)
    for k, v in prizes_d.items():
        p[int(k)] = float(v)
    p[depot] = 0.0

    BIG = float('inf')
    D = [[BIG] * (N + 1) for _ in range(N + 1)]
    for c in cities:
        D[c][c] = 0.0
    for i, j, v in parsed_tt:
        D[i][j] = v
        D[j][i] = v

    others = [c for c in cities if c != depot]

    state = {'best_obj': -1.0, 'best_tour': None}

    def tour_prize(tour):
        return sum(p[v] for v in tour)

    def tour_length(tour):
        m = len(tour)
        return sum(D[tour[i]][tour[(i + 1) % m]] for i in range(m))

    def build_solution(tour):
        m = len(tour)
        edges = [[tour[i], tour[(i + 1) % m]] for i in range(m)]
        return {
            "objective_value": float(tour_prize(tour)),
            "visited_nodes": list(tour),
            "edges": edges,
            "tour": list(tour) + [tour[0]],
        }

    def record(tour):
        obj = tour_prize(tour)
        if obj > state['best_obj'] + 1e-9:
            state['best_obj'] = obj
            state['best_tour'] = list(tour)
            if logger:
                logger.log_solution(float(obj), build_solution(tour))
            return True
        return False

    # ----------------- heuristic components -----------------
    def two_opt(tour, length, deadline):
        m = len(tour)
        if m < 4:
            return length
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for i in range(m - 1):
                a = tour[i]
                b = tour[i + 1]
                dab = D[a][b]
                for j in range(i + 2, m):
                    if i == 0 and j == m - 1:
                        continue
                    c = tour[j]
                    e = tour[(j + 1) % m]
                    delta = D[a][c] + D[b][e] - dab - D[c][e]
                    if delta < -1e-9:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        length += delta
                        improved = True
                        b = tour[i + 1]
                        dab = D[a][b]
                if time.time() > deadline:
                    return length
        return length

    def insert_all(tour, length, visited, deadline, rng=None):
        changed = False
        while True:
            if time.time() > deadline:
                break
            m = len(tour)
            best = None
            for v in others:
                if v in visited or p[v] <= 0:
                    continue
                bc = None
                bp = 0
                for pos in range(m):
                    a = tour[pos]
                    b = tour[(pos + 1) % m]
                    c = D[a][v] + D[v][b] - D[a][b]
                    if bc is None or c < bc:
                        bc = c
                        bp = pos
                if bc is not None and length + bc <= t0 + 1e-9:
                    score = p[v] / (bc + 1e-6)
                    if rng is not None:
                        score *= (0.75 + 0.5 * rng.random())
                    if best is None or score > best[0]:
                        best = (score, v, bp, bc)
            if best is None:
                break
            _, v, bp, bc = best
            tour.insert(bp + 1, v)
            visited.add(v)
            length += bc
            changed = True
        return length, changed

    def improve(tour, deadline, rng=None):
        visited = set(tour)
        length = tour_length(tour)
        while True:
            length = two_opt(tour, length, deadline)
            length, changed = insert_all(tour, length, visited, deadline, rng)
            if not changed or time.time() > deadline:
                break
        return tour, length

    def cheapest_feasible_pair():
        # find pair (j,k) forming feasible 3-node tour, prefer max prize then min cost
        best = None
        for a_idx in range(len(others)):
            j = others[a_idx]
            dj = D[depot][j]
            if dj == BIG:
                continue
            for b_idx in range(a_idx + 1, len(others)):
                k = others[b_idx]
                cost = dj + D[j][k] + D[k][depot]
                if cost <= t0 + 1e-9:
                    val = p[j] + p[k]
                    if best is None or val > best[0] + 1e-12 or (abs(val - best[0]) <= 1e-12 and cost < best[1]):
                        best = (val, cost, j, k)
            if time.time() > hard_deadline:
                break
        if best is None:
            return None
        return [depot, best[2], best[3]]

    def construct(deadline, rng=None):
        tour = [depot]
        visited = {depot}
        length = 0.0
        length, _ = insert_all(tour, length, visited, deadline, rng)
        if len(tour) < 3:
            base = cheapest_feasible_pair()
            if base is None:
                return None
            tour = base
            visited = set(tour)
            length = tour_length(tour)
            length, _ = insert_all(tour, length, visited, deadline, rng)
        return tour

    def perturb_and_improve(base_tour, rng, deadline):
        tour = list(base_tour)
        removable = [i for i in range(len(tour)) if tour[i] != depot]
        if removable:
            k = max(1, len(removable) // 4)
            rem_idx = set(rng.sample(removable, min(k, len(removable))))
            tour = [tour[i] for i in range(len(tour)) if i not in rem_idx]
        if len(tour) < 3:
            tour = list(base_tour)
        tour, _ = improve(tour, deadline, rng)
        if len(tour) >= 3 and tour_length(tour) <= t0 + 1e-9:
            record(tour)

    # ----------------- run heuristic -----------------
    rng = random.Random(0)
    use_mip = (len(cities) <= 500) and (len(cities) >= 3)
    if len(cities) < 3:
        use_mip = False

    if use_mip:
        heur_deadline = min(hard_deadline, start_time + max(3.0, 0.15 * args.time_limit))
    else:
        heur_deadline = hard_deadline

    feasible_exists = True
    if len(cities) >= 3:
        tour = construct(heur_deadline)
        if tour is None:
            feasible_exists = False
        else:
            tour, _ = improve(tour, heur_deadline)
            if len(tour) >= 3 and tour_length(tour) <= t0 + 1e-9:
                record(tour)
            # perturbation loop within heuristic budget
            while time.time() < heur_deadline and state['best_tour'] is not None:
                perturb_and_improve(state['best_tour'], rng, heur_deadline)
    else:
        feasible_exists = False

    # ----------------- exact MIP with lazy subtour elimination -----------------
    if use_mip and feasible_exists and time.time() < hard_deadline - 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("op")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.LazyConstraints = 1
            m.Params.TimeLimit = max(1.0, hard_deadline - time.time())

            clist = cities
            pairs = [(clist[a], clist[b]) for a in range(len(clist)) for b in range(a + 1, len(clist))]
            x = m.addVars(pairs, vtype=GRB.BINARY, name="x")
            y = m.addVars(clist, vtype=GRB.BINARY, name="y")
            y[depot].LB = 1.0

            for i in clist:
                m.addConstr(
                    gp.quicksum(x[(i, j)] if i < j else x[(j, i)] for j in clist if j != i) == 2 * y[i]
                )
            m.addConstr(gp.quicksum(D[i][j] * x[(i, j)] for (i, j) in pairs) <= t0)
            m.addConstr(gp.quicksum(y[i] for i in clist) >= 3)
            m.setObjective(gp.quicksum(p[i] * y[i] for i in clist), GRB.MAXIMIZE)

            # warm start
            if state['best_tour'] is not None:
                bt = state['best_tour']
                tedges = set()
                mm = len(bt)
                for idx in range(mm):
                    a, b = bt[idx], bt[(idx + 1) % mm]
                    tedges.add((min(a, b), max(a, b)))
                vset = set(bt)
                for (i, j) in pairs:
                    x[(i, j)].Start = 1.0 if (i, j) in tedges else 0.0
                for i in clist:
                    y[i].Start = 1.0 if i in vset else 0.0

            def extract_tour_from_adj(adj):
                tour_ = [depot]
                prev = None
                cur = depot
                while True:
                    nxts = [w for w in adj[cur] if w != prev]
                    if not nxts:
                        nxt = adj[cur][0]
                    else:
                        nxt = nxts[0]
                    if nxt == depot:
                        break
                    tour_.append(nxt)
                    prev, cur = cur, nxt
                    if len(tour_) > len(clist) + 1:
                        break
                return tour_

            def cb(model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                yv = model.cbGetSolution(y)
                xv = model.cbGetSolution(x)
                sel = [c for c in clist if yv[c] > 0.5]
                adj = {c: [] for c in sel}
                for (i, j), val in xv.items():
                    if val > 0.5:
                        adj[i].append(j)
                        adj[j].append(i)
                seen = set()
                comps = []
                for c in sel:
                    if c in seen:
                        continue
                    stack = [c]
                    seen.add(c)
                    comp = []
                    while stack:
                        u = stack.pop()
                        comp.append(u)
                        for w in adj[u]:
                            if w not in seen:
                                seen.add(w)
                                stack.append(w)
                    comps.append(comp)
                if len(comps) <= 1:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > state['best_obj'] + 1e-9:
                        try:
                            tour_ = extract_tour_from_adj(adj)
                            if len(tour_) >= 3:
                                record(tour_)
                        except Exception:
                            if logger:
                                logger.log(float(obj))
                    return
                for comp in comps:
                    if depot in comp:
                        continue
                    expr = gp.quicksum(
                        x[(i, j)] for ii, i in enumerate(comp) for j in comp[ii + 1:] if i < j
                    ) + gp.quicksum(
                        x[(j, i)] for ii, i in enumerate(comp) for j in comp[ii + 1:] if j < i
                    )
                    ysum = gp.quicksum(y[i] for i in comp)
                    if len(comp) <= 15:
                        ks = comp
                    else:
                        ks = sorted(comp, key=lambda c: -p[c])[:5]
                    for k in ks:
                        model.cbLazy(expr <= ysum - y[k])

            m.optimize(cb)

            if m.SolCount > 0:
                sel = [c for c in clist if y[c].X > 0.5]
                adj = {c: [] for c in sel}
                for (i, j) in pairs:
                    if x[(i, j)].X > 0.5:
                        adj[i].append(j)
                        adj[j].append(i)
                # verify single component containing depot
                seen = set([depot])
                stack = [depot]
                while stack:
                    u = stack.pop()
                    for w in adj.get(u, []):
                        if w not in seen:
                            seen.add(w)
                            stack.append(w)
                if len(seen) == len(sel) and len(sel) >= 3:
                    tour_ = extract_tour_from_adj(adj)
                    if len(tour_) >= 3 and tour_length(tour_) <= t0 + 1e-6:
                        record(tour_)
        except Exception:
            pass

    # keep polishing with heuristic if time remains and no proven optimum needed
    while time.time() < hard_deadline - 0.5 and state['best_tour'] is not None:
        perturb_and_improve(state['best_tour'], rng, hard_deadline - 0.2)
        if time.time() >= hard_deadline - 0.5:
            break

    # ----------------- write final solution -----------------
    if state['best_tour'] is not None:
        sol = build_solution(state['best_tour'])
    else:
        sol = {
            "objective_value": 0.0,
            "visited_nodes": [depot],
            "edges": [],
            "tour": [depot, depot],
        }
        if logger:
            logger.log_solution(0.0, sol)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()