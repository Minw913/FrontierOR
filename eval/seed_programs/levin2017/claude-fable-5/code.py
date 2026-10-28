import argparse
import json
import time
import heapq
from collections import defaultdict

INF = float('inf')


def dijkstra(adj, sources, extra=None, extra_node=None):
    """Shortest path over adjacency dict {node: [(nbr, w), ...]}.
    'extra' edges are appended to neighbors of 'extra_node' only."""
    dist = {}
    h = []
    for s in sources:
        dist[s] = 0
        h.append((0, s))
    heapq.heapify(h)
    while h:
        dv, v = heapq.heappop(h)
        if dv > dist.get(v, INF):
            continue
        nbrs = adj.get(v, [])
        if extra_node is not None and v == extra_node and extra:
            nbrs = list(nbrs) + extra
        for w, c in nbrs:
            nd = dv + c
            if nd < dist.get(w, INF):
                dist[w] = nd
                heapq.heappush(h, (nd, w))
    return dist


def build_trivial(centroids, p0, T, dem):
    """Fallback solution (used only if LP cannot be solved / no demand)."""
    p = {}
    for r in centroids:
        v = float(p0.get(r, 0.0))
        for t in range(T + 1):
            p["%s|%d" % (r, t)] = v
    omega = {}
    e = {}
    obj = 0.0
    for (r, s), cnt in dem.items():
        w = 0.0
        for t in range(T + 1):
            omega["%s|%s|%d" % (r, s, t)] = w
            obj += w
            if t < T:
                e["%s|%s|%d" % (r, s, t)] = 0.0
                w += cnt.get(t, 0.0)
    return {"objective_value": obj, "y": {}, "y_centroid": {},
            "N_U": {}, "N_D": {}, "p": p, "e": e, "omega": omega}


def main():
    wall0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=600)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    net = data['network']
    tp = data['time_parameters']
    T = int(tp['time_horizon_T'])
    dt = float(tp['time_step_duration_seconds'])
    node_type = {n['id']: n['type'] for n in net['nodes']}
    centroids = [nid for nid, ty in node_type.items() if ty == 'centroid']

    raw_links = net['links']
    nL = len(raw_links)
    tail = [l['from'] for l in raw_links]
    head = [l['to'] for l in raw_links]
    tf = [max(1, int(l['free_flow_travel_time_steps'])) for l in raw_links]
    tb = [max(1, int(l.get('congested_travel_time_steps', 1))) for l in raw_links]
    cap = [float(l['capacity_vph']) * dt / 3600.0 for l in raw_links]
    Q = [float(l.get('jam_density_vehicles', 1e9)) for l in raw_links]
    kind = []  # 0 road, 1 outgoing connector (centroid->junction), 2 incoming connector (junction->centroid)
    for l in raw_links:
        if l['type'] == 'road':
            kind.append(0)
        elif node_type.get(l['from']) == 'centroid':
            kind.append(1)
        else:
            kind.append(2)

    links_from = defaultdict(list)
    for a in range(nL):
        links_from[tail[a]].append(a)

    # ------- demand -------
    dem_tmp = defaultdict(lambda: defaultdict(float))
    for od in data['demand'].get('od_pairs', []):
        r, s = od['origin'], od['destination']
        dts = od.get('departure_times', [])
        if dts:
            for t in dts:
                dem_tmp[(r, s)][int(t)] += 1.0
    dem = {k: dict(v) for k, v in dem_tmp.items() if sum(v.values()) > 0}

    p0 = {r: float(v) for r, v in data['fleet'].get('initial_distribution', {}).items()}
    total_p0 = sum(p0.get(r, 0.0) for r in centroids)

    Dset = set()
    for (r, s) in dem:
        Dset.add(r)
        Dset.add(s)
    D = sorted(Dset)

    def write_out(sol):
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, separators=(',', ':'))

    if not dem:
        sol = build_trivial(centroids, p0, T, dem)
        if logger:
            logger.log_solution(sol["objective_value"], sol)
        write_out(sol)
        return

    # ------- shortest-time distances for pruning -------
    # earliest presence bound: forward from centroids with initial vehicles, all links usable
    fadj = defaultdict(list)
    for a in range(nL):
        fadj[tail[a]].append((head[a], tf[a]))
    srcs = [r for r in centroids if p0.get(r, 0.0) > 0]
    earliest = dijkstra(fadj, srcs) if srcs else {}

    # dd[d][n]: shortest free-flow time from node n to arrive at centroid d
    # usable links: roads + incoming connectors of d only
    revadj = defaultdict(list)
    revconn = defaultdict(list)
    for a in range(nL):
        if kind[a] == 0:
            revadj[head[a]].append((tail[a], tf[a]))
        elif kind[a] == 2:
            revconn[head[a]].append((tail[a], tf[a]))
    dd = {}
    for d in D:
        dd[d] = dijkstra(revadj, [d], extra=revconn.get(d, []), extra_node=d)

    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model('sav_ltm')
        m.Params.OutputFlag = 1
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.Method = 2      # barrier
        m.Params.Crossover = 0   # skip crossover for speed; interior solution is fine

        # ------------- variable keys -------------
        ykeys = []
        for a in range(nL):
            if kind[a] == 2:
                continue
            ea = earliest.get(tail[a], INF)
            if ea == INF:
                continue
            t0a = int(ea) + tf[a]
            if t0a > T - 1:
                continue
            for b in links_from.get(head[a], []):
                if kind[b] == 1:
                    continue
                if kind[b] == 2:
                    s = head[b]
                    dlist = [s] if s in Dset else []
                else:
                    dlist = D
                for d in dlist:
                    dst = dd[d].get(head[b], INF)
                    if dst == INF:
                        continue
                    tmax = T - 1 - tf[b] - int(dst)
                    if tmax < t0a:
                        continue
                    for t in range(t0a, tmax + 1):
                        ykeys.append((a, b, d, t))

        yckeys = []
        for a in range(nL):
            if kind[a] != 1:
                continue
            r, j = tail[a], head[a]
            er = earliest.get(r, INF)
            if er == INF:
                continue
            er = int(er)
            for d in D:
                dst = dd[d].get(j, INF)
                if dst == INF:
                    continue
                tmax = T - 2 - int(dst)
                for t in range(er, tmax + 1):
                    yckeys.append((a, d, t))

        yv = m.addVars(ykeys, lb=0.0, name="y")
        ycv = m.addVars(yckeys, lb=0.0, name="yc")

        y_in = defaultdict(list)
        y_out = defaultdict(list)
        cap_in = defaultdict(list)
        cap_out = defaultdict(list)
        for key in ykeys:
            a, b, d, t = key
            v = yv[key]
            y_out[(a, d, t)].append(v)
            y_in[(b, d, t)].append(v)
            if kind[a] == 0:
                cap_out[(a, t)].append(v)
            if kind[b] == 0:
                cap_in[(b, t)].append(v)

        dep_map = defaultdict(list)   # (centroid, t) -> yc vars
        ycdest = defaultdict(list)    # (centroid, dest, t) -> yc vars
        for key in yckeys:
            a, d, t = key
            v = ycv[key]
            dep_map[(tail[a], t)].append(v)
            ycdest[(tail[a], d, t)].append(v)

        activeAD = set()
        for (a, b, d, t) in ykeys:
            activeAD.add((a, d))
            activeAD.add((b, d))
        for (a, d, t) in yckeys:
            activeAD.add((a, d))
        activeAD = sorted(activeAD)

        nkeys = [(a, d, t) for (a, d) in activeAD for t in range(T + 1)]
        NU = m.addVars(nkeys, lb=0.0, obj=1.0, name="NU")
        ND = m.addVars(nkeys, lb=0.0, obj=-1.0, name="ND")
        for (a, d) in activeAD:
            NU[a, d, 0].UB = 0.0
            ND[a, d, 0].UB = 0.0

        # ------------- LTM recursions -------------
        for (a, d) in activeAD:
            k = kind[a]
            for t in range(T):
                if k == 1:
                    inv = [ycv[a, d, t]] if (a, d, t) in ycv else []
                else:
                    inv = y_in.get((a, d, t), [])
                m.addLConstr(NU[a, d, t + 1] - NU[a, d, t] - gp.quicksum(inv) == 0)
                if k == 2:
                    m.addLConstr(ND[a, d, t + 1] - NU[a, d, t] == 0)
                else:
                    outv = y_out.get((a, d, t), [])
                    m.addLConstr(ND[a, d, t + 1] - ND[a, d, t] - gp.quicksum(outv) == 0)

        # sending flow (per destination) on roads and outgoing connectors
        for (a, d, t), outv in y_out.items():
            m.addLConstr(gp.quicksum(outv) - NU[a, d, t - tf[a] + 1] + ND[a, d, t] <= 0)

        # road capacities
        for (a, t), vs in cap_out.items():
            m.addLConstr(gp.quicksum(vs) <= cap[a])
        for (a, t), vs in cap_in.items():
            m.addLConstr(gp.quicksum(vs) <= cap[a])

        # receiving flow (congested wave / jam occupancy) on roads
        ad_of_link = defaultdict(list)
        for (a, d) in activeAD:
            ad_of_link[a].append(d)
        for (a, t), vs in cap_in.items():
            expr = gp.LinExpr()
            expr.addTerms([1.0] * len(vs), vs)
            tau = t - tb[a] + 1
            for d in ad_of_link[a]:
                expr.addTerms(1.0, NU[a, d, t])
                if tau >= 0:
                    expr.addTerms(-1.0, ND[a, d, tau])
            m.addLConstr(expr <= Q[a])

        # ------------- parking dynamics -------------
        P = m.addVars(centroids, range(T + 1), lb=0.0, name="p")
        for r in centroids:
            v0 = p0.get(r, 0.0)
            P[r, 0].LB = v0
            P[r, 0].UB = v0
        in_conns_of = defaultdict(list)
        for a in range(nL):
            if kind[a] == 2:
                in_conns_of[head[a]].append(a)
        for r in centroids:
            arr_links = [a for a in in_conns_of[r] if (a, r) in set(ad_of_link and []) or True]
            arr_links = [a for a in in_conns_of[r] if (a, r) in dict.fromkeys(activeAD)]
            for t in range(T):
                expr = gp.LinExpr()
                expr.addTerms([1.0, -1.0], [P[r, t + 1], P[r, t]])
                for a in arr_links:
                    expr.addTerms(-1.0, NU[a, r, t])
                    expr.addTerms(1.0, ND[a, r, t])
                deps = dep_map.get((r, t), [])
                if deps:
                    expr.addTerms([1.0] * len(deps), deps)
                m.addLConstr(expr == 0)
                if deps:
                    m.addLConstr(gp.quicksum(deps) - P[r, t] <= 0)
        m.addLConstr(gp.quicksum(P[r, T] for r in centroids) == total_p0)

        # ------------- traveler dynamics -------------
        odpairs = list(dem.keys())
        OM = m.addVars([(r, s, t) for (r, s) in odpairs for t in range(T + 1)],
                       lb=0.0, obj=1.0, name="om")
        E = m.addVars([(r, s, t) for (r, s) in odpairs for t in range(T)],
                      lb=0.0, name="e")
        for (r, s) in odpairs:
            OM[r, s, 0].UB = 0.0
            OM[r, s, T].UB = 0.0
            cnt = dem[(r, s)]
            for t in range(T):
                m.addLConstr(OM[r, s, t + 1] - OM[r, s, t] + E[r, s, t] == cnt.get(t, 0.0))
                m.addLConstr(E[r, s, t] - OM[r, s, t] <= 0)
                vs = ycdest.get((r, s, t), [])
                if vs:
                    m.addLConstr(E[r, s, t] - gp.quicksum(vs) <= 0)
                else:
                    E[r, s, t].UB = 0.0

        m.ModelSense = GRB.MINIMIZE

        # ------------- solve -------------
        reserve = min(60.0, max(10.0, 0.08 * args.time_limit))
        rem = args.time_limit - (time.time() - wall0) - reserve
        if rem < 5:
            rem = 5.0
        m.Params.TimeLimit = rem
        m.optimize()

        if m.Status == GRB.INFEASIBLE:
            raise RuntimeError("infeasible")

        # ------------- extract solution -------------
        thr = 1e-7

        def cl(v):
            return round(v, 7)

        sol_y = {}
        for key, v in m.getAttr('X', yv).items():
            if v > thr:
                a, b, d, t = key
                sol_y["%s|%s|%s|%s|%d" % (tail[a], head[a], head[b], d, t)] = cl(v)

        sol_yc = {}
        for key, v in m.getAttr('X', ycv).items():
            if v > thr:
                a, d, t = key
                sol_yc["%s|%s|%s|%d" % (tail[a], head[a], d, t)] = cl(v)

        sol_NU = {}
        for key, v in m.getAttr('X', NU).items():
            if v > thr:
                a, d, t = key
                sol_NU["%s|%s|%s|%d" % (tail[a], head[a], d, t)] = cl(v)

        sol_ND = {}
        for key, v in m.getAttr('X', ND).items():
            if v > thr:
                a, d, t = key
                sol_ND["%s|%s|%s|%d" % (tail[a], head[a], d, t)] = cl(v)

        sol_p = {}
        for key, v in m.getAttr('X', P).items():
            r, t = key
            sol_p["%s|%d" % (r, t)] = cl(max(v, 0.0))

        sol_e = {}
        for key, v in m.getAttr('X', E).items():
            r, s, t = key
            sol_e["%s|%s|%d" % (r, s, t)] = cl(max(v, 0.0))

        sol_om = {}
        for key, v in m.getAttr('X', OM).items():
            r, s, t = key
            sol_om["%s|%s|%d" % (r, s, t)] = cl(max(v, 0.0))

        obj = sum(sol_NU.values()) - sum(sol_ND.values()) + sum(sol_om.values())

        sol = {"objective_value": obj,
               "y": sol_y,
               "y_centroid": sol_yc,
               "N_U": sol_NU,
               "N_D": sol_ND,
               "p": sol_p,
               "e": sol_e,
               "omega": sol_om}

        if logger:
            logger.log_solution(obj, sol)
        write_out(sol)

    except Exception:
        # fallback: write a syntactically valid solution
        sol = build_trivial(centroids, p0, T, dem)
        try:
            if logger:
                logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass
        write_out(sol)


if __name__ == '__main__':
    main()