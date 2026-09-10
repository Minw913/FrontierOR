import argparse
import json
import time
import itertools
import heapq
from collections import defaultdict, deque


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t0 = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    Gp = inst["global_parameters"]
    lam = float(Gp["lambda"])
    F = float(Gp["frequency_upper_bound_F"])
    Lmax = int(Gp["max_line_length_edges"])
    glob_fc = float(Gp.get("fixed_cost_per_line", 0.0))
    glob_oc = float(Gp.get("operating_cost_per_edge", 0.0))

    net = inst["network"]
    node_ids = [n["id"] for n in net["nodes"]]
    edges = {int(e["id"]): e for e in net["edges"]}
    full_adj = defaultdict(list)
    timeof = {}
    for eid, e in edges.items():
        u, v = e["endpoints"]
        full_adj[u].append((v, eid))
        full_adj[v].append((u, eid))
        timeof[eid] = float(e["traveling_time_seconds"])

    modes = inst["modes"]
    edge_mode = {}
    minfo = []
    for mi, md in enumerate(modes):
        adj = defaultdict(list)
        pe = {}
        for eid in md["edge_indices"]:
            eid = int(eid)
            e = edges[eid]
            u, v = e["endpoints"]
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            k = (min(u, v), max(u, v))
            if k not in pe or timeof[pe[k]] > timeof[eid]:
                pe[k] = eid
            edge_mode[eid] = mi
        terms = [t for t in md["terminals"] if t in adj]
        minfo.append({"adj": adj, "pe": pe, "terms": terms,
                      "fc": float(md.get("fixed_cost_per_line", glob_fc)),
                      "cap": float(md["vehicle_capacity"]),
                      "ocpe": float(md.get("operating_cost_per_edge", glob_oc)),
                      "name": md["name"]})

    lines = []
    line_keys = set()

    def make_line(mi, nseq, eseq):
        key = tuple(eseq)
        ck = min(key, tuple(reversed(key)))
        if ck in line_keys:
            return False
        line_keys.add(ck)
        m0 = minfo[mi]
        oc = sum(float(edges[eid].get("operating_cost", m0["ocpe"])) for eid in eseq)
        lines.append({"mode": m0["name"], "mi": mi, "cap": m0["cap"],
                      "fc": m0["fc"], "oc": oc,
                      "nodes": list(nseq), "edges": list(eseq)})
        return True

    # ---------------- line enumeration ----------------
    tl = args.time_limit
    enum_deadline = t0 + min(0.25 * tl, 60.0)

    def bfs_hops(adj, src):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    all_pairs = []
    for mi, m0 in enumerate(minfo):
        for s, t in itertools.combinations(sorted(set(m0["terms"])), 2):
            all_pairs.append((mi, s, t))
    K = max(2, min(15, 1600 // max(1, len(all_pairs))))

    for mi, s, t in all_pairs:
        if time.time() > enum_deadline:
            break
        adj = minfo[mi]["adj"]
        hop = bfs_hops(adj, t)
        if s not in hop or hop[s] > Lmax:
            continue
        res = []
        cnt = [0]
        visited = {s}
        pn = [s]
        pel = []

        def dfs(u):
            cnt[0] += 1
            if cnt[0] > 40000 or len(res) >= K:
                return
            if u == t:
                res.append((list(pn), list(pel)))
                return
            if len(pel) >= Lmax:
                return
            for v, eid in sorted(adj[u], key=lambda z: hop.get(z[0], 1 << 30)):
                if v in visited:
                    continue
                h = hop.get(v)
                if h is None or len(pel) + 1 + h > Lmax:
                    continue
                visited.add(v)
                pn.append(v)
                pel.append(eid)
                dfs(v)
                visited.discard(v)
                pn.pop()
                pel.pop()
                if cnt[0] > 40000 or len(res) >= K:
                    return

        dfs(s)
        for nseq, eseq in res:
            make_line(mi, nseq, eseq)

    # ---------------- coverage / connectivity repair ----------------
    def compute_covered():
        c = set()
        for ln in lines:
            c.update(ln["edges"])
        return c

    def bfs_to_term(adj, start, avoid, terms):
        if start in avoid:
            return None
        if start in terms:
            return [start]
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v in avoid or v in parent:
                    continue
                parent[v] = u
                if v in terms:
                    p = [v]
                    while parent[p[-1]] is not None:
                        p.append(parent[p[-1]])
                    return p[::-1]
                q.append(v)
        return None

    def repair_edge(eid):
        mi = edge_mode.get(eid)
        if mi is None:
            return False
        m0 = minfo[mi]
        adj = m0["adj"]
        terms = set(m0["terms"])
        pe = m0["pe"]
        u0, v0 = edges[eid]["endpoints"]
        for u, v in ((u0, v0), (v0, u0)):
            P1 = bfs_to_term(adj, u, {v}, terms)
            if P1 is None:
                continue
            s = P1[-1]
            P2 = bfs_to_term(adj, v, set(P1), terms - {s})
            if P2 is None:
                continue
            nseq = P1[::-1] + P2
            if len(nseq) - 1 > Lmax:
                continue
            eseq = []
            ok = True
            for a, b in zip(nseq, nseq[1:]):
                if {a, b} == {u, v}:
                    eseq.append(eid)
                else:
                    k = (min(a, b), max(a, b))
                    if k not in pe:
                        ok = False
                        break
                    eseq.append(pe[k])
            if not ok:
                continue
            make_line(mi, nseq, eseq)
            return True
        return False

    covered = compute_covered()
    for eid in list(edge_mode.keys()):
        if eid not in covered:
            repair_edge(eid)
    covered = compute_covered()

    od = defaultdict(dict)
    for r in inst["od_matrix"]:
        o = r["origin"]
        d = r["destination"]
        q = float(r["demand"])
        if q <= 0 or o == d:
            continue
        od[o][d] = od[o].get(d, 0.0) + q

    def build_cov_adj(cov):
        ca = defaultdict(list)
        for eid in cov:
            u, v = edges[eid]["endpoints"]
            ca[u].append((v, eid))
            ca[v].append((u, eid))
        return ca

    for attempt in range(3):
        ca = build_cov_adj(covered)
        missing = []
        for o, dm in od.items():
            seen = {o}
            q = deque([o])
            while q:
                u = q.popleft()
                for v, _ in ca.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            for d in dm:
                if d not in seen:
                    missing.append((o, d))
        if not missing:
            break
        if attempt == 2:
            for o, d in missing:
                od[o].pop(d, None)
            break
        for o, d in missing:
            par = {o: None}
            q = deque([o])
            while q:
                u = q.popleft()
                if u == d:
                    break
                for v, eid in full_adj[u]:
                    if v not in par:
                        par[v] = (u, eid)
                        q.append(v)
            if d not in par:
                continue
            cur = d
            peid = []
            while par[cur] is not None:
                u, eid = par[cur]
                peid.append(eid)
                cur = u
            for eid in peid:
                if eid not in covered:
                    repair_edge(eid)
        covered = compute_covered()
    od = {o: dm for o, dm in od.items() if dm}

    origins = sorted(od.keys())
    nO = len(origins)

    def write_out(obj, sol):
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)

    if nO == 0 or not lines or not covered:
        sol = {"objective_value": 0.0, "active_lines": [], "active_passenger_paths": []}
        if logger:
            logger.log_solution(0.0, sol)
        write_out(0.0, sol)
        return

    # ---------------- arc structures ----------------
    A = []
    aidx = {}
    for eid in sorted(covered):
        u, v = edges[eid]["endpoints"]
        tt = timeof[eid]
        aidx[2 * eid] = len(A)
        A.append((2 * eid, u, v, tt, eid))
        aidx[2 * eid + 1] = len(A)
        A.append((2 * eid + 1, v, u, tt, eid))
    nA = len(A)
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    covd = defaultdict(list)
    for ai, (aid, u, v, tt, eid) in enumerate(A):
        out_arcs[u].append(ai)
        in_arcs[v].append(ai)
        covd[u].append((v, ai, tt))

    lbe = defaultdict(list)
    for li, ln in enumerate(lines):
        for eid in ln["edges"]:
            lbe[eid].append(li)

    def dijkstra(o):
        dist = {o: 0.0}
        par = {}
        pq = [(0.0, o)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist.get(u, 1e30) + 1e-12:
                continue
            for v, ai, tt in covd[u]:
                nd = dd + tt
                if nd < dist.get(v, 1e30) - 1e-12:
                    dist[v] = nd
                    par[v] = (u, ai)
                    heapq.heappush(pq, (nd, v))
        return dist, par

    sp_cache = {}

    def build_solution(active_freqs, path_recs):
        merged = {}
        for o, d, arcs, fl in path_recs:
            if fl <= 1e-9:
                continue
            k = (o, d, tuple(arcs))
            merged[k] = merged.get(k, 0.0) + fl
        line_cost = 0.0
        alines = []
        for li, fq in sorted(active_freqs.items()):
            if fq <= 1e-7:
                continue
            ln = lines[li]
            line_cost += ln["fc"] + ln["oc"] * fq
            alines.append({"line_index": li, "mode": ln["mode"], "nodes": ln["nodes"],
                           "edges": ln["edges"], "frequency": fq})
        travel = 0.0
        apaths = []
        for (o, d, arcs), fl in merged.items():
            pt = sum(timeof[a // 2] for a in arcs)
            travel += fl * pt
            apaths.append({"origin": o, "destination": d, "arcs": list(arcs), "flow": fl})
        obj = lam * line_cost + (1.0 - lam) * travel
        return obj, {"objective_value": obj, "active_lines": alines,
                     "active_passenger_paths": apaths}

    # ---------------- greedy heuristic ----------------
    heur_lfreq = None
    opaths = {}
    best_obj = None
    best_sol = None
    try:
        arcflow = [0.0] * nA
        ok = True
        for o in origins:
            dist, par = dijkstra(o)
            sp_cache[o] = (dist, par)
            for d, q in od[o].items():
                if d not in dist:
                    ok = False
                    break
                path = []
                cur = d
                while cur != o:
                    u, ai = par[cur]
                    path.append(ai)
                    cur = u
                path.reverse()
                opaths[(o, d)] = path
                for ai in path:
                    arcflow[ai] += q
            if not ok:
                break
        if ok:
            need = {}
            for eid in covered:
                cp = minfo[edge_mode[eid]]["cap"]
                fl = max(arcflow[aidx[2 * eid]], arcflow[aidx[2 * eid + 1]])
                if fl > 1e-9:
                    need[eid] = fl / cp
            resid = {eid: float(edges[eid]["edge_capacity"]) for eid in covered}
            lfreq = defaultdict(float)
            hd = t0 + min(0.45 * tl, 90.0)
            for _ in range(6000):
                if not need or time.time() > hd:
                    break
                bl = None
                bs = 0.0
                for li, ln in enumerate(lines):
                    rem = F - lfreq[li]
                    if rem <= 1e-9:
                        continue
                    es = ln["edges"]
                    hit = [need[e] for e in es if e in need]
                    if not hit:
                        continue
                    mn = min(resid[e] for e in es)
                    if mn <= 1e-9:
                        continue
                    sc = sum(min(v, mn, rem) for v in hit)
                    if sc > bs + 1e-12:
                        bs = sc
                        bl = li
                if bl is None:
                    break
                ln = lines[bl]
                mn = min(resid[e] for e in ln["edges"])
                fq = min(F - lfreq[bl], mn,
                         max(need[e] for e in ln["edges"] if e in need))
                if fq <= 1e-9:
                    break
                lfreq[bl] += fq
                for e in ln["edges"]:
                    resid[e] -= fq
                    if e in need:
                        need[e] -= fq
                        if need[e] <= 1e-9:
                            del need[e]
            if not need:
                heur_lfreq = dict(lfreq)
                heur_paths = [(o, d, [A[ai][0] for ai in opaths[(o, d)]], od[o][d])
                              for o in origins for d in od[o]]
                best_obj, best_sol = build_solution(heur_lfreq, heur_paths)
                if logger:
                    logger.log_solution(best_obj, best_sol)
    except Exception:
        heur_lfreq = None

    # ---------------- flow decomposition ----------------
    def decompose(xv):
        recs = []
        for oi, o in enumerate(origins):
            fl = {}
            for ai in range(nA):
                v = xv[oi, ai]
                if v > 1e-9:
                    fl[ai] = v
            dm = dict(od[o])
            got = defaultdict(list)
            guard = 0
            maxit = 4 * nA + 4 * len(dm) + 50
            while guard < maxit:
                guard += 1
                tgt = None
                for d, q in dm.items():
                    if q > 1e-7:
                        tgt = d
                        break
                if tgt is None:
                    break
                par = {o: None}
                qq = deque([o])
                found = False
                while qq:
                    u = qq.popleft()
                    stop = False
                    for v2, ai, tt in covd[u]:
                        if fl.get(ai, 0.0) > 1e-9 and v2 not in par:
                            par[v2] = (u, ai)
                            if v2 == tgt:
                                stop = True
                                break
                            qq.append(v2)
                    if stop:
                        found = True
                        break
                if found:
                    path = []
                    cur = tgt
                    while par[cur] is not None:
                        u, ai = par[cur]
                        path.append(ai)
                        cur = u
                    path.reverse()
                    amt = min([dm[tgt]] + [fl[ai] for ai in path])
                    for ai in path:
                        fl[ai] -= amt
                        if fl[ai] <= 1e-9:
                            fl.pop(ai, None)
                    dm[tgt] -= amt
                    got[tgt].append([path, amt])
                else:
                    if o not in sp_cache:
                        sp_cache[o] = dijkstra(o)
                    dist, spar = sp_cache[o]
                    if tgt not in dist:
                        dm[tgt] = 0.0
                        continue
                    path = []
                    cur = tgt
                    while cur != o:
                        u, ai = spar[cur]
                        path.append(ai)
                        cur = u
                    path.reverse()
                    got[tgt].append([path, dm[tgt]])
                    dm[tgt] = 0.0
            for d, q0 in od[o].items():
                plist = got.get(d)
                if not plist:
                    continue
                s = sum(a for _, a in plist)
                diff = q0 - s
                if abs(diff) > 1e-12:
                    plist.sort(key=lambda z: -z[1])
                    plist[0][1] += diff
                for path, amt in plist:
                    if amt > 1e-9:
                        recs.append((o, d, [A[ai][0] for ai in path], amt))
        return recs

    # ---------------- MIP ----------------
    try:
        import gurobipy as gp
        from gurobipy import GRB
        remaining = args.time_limit - (time.time() - t0) - max(3.0, 0.03 * tl)
        if remaining > 3 and nO * nA <= 2_000_000:
            m = gp.Model("tndlp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = remaining
            m.Params.MIPFocus = 1
            nL = len(lines)
            y = m.addVars(nL, vtype=GRB.BINARY)
            f = m.addVars(nL, lb=0.0, ub=F)
            x = m.addVars(nO, nA, lb=0.0)
            m.setAttr("Obj", [y[i] for i in range(nL)],
                      [lam * lines[i]["fc"] for i in range(nL)])
            m.setAttr("Obj", [f[i] for i in range(nL)],
                      [lam * lines[i]["oc"] for i in range(nL)])
            xlist = [x[oi, ai] for oi in range(nO) for ai in range(nA)]
            xobj = [(1.0 - lam) * A[ai][3] for oi in range(nO) for ai in range(nA)]
            m.setAttr("Obj", xlist, xobj)
            for li in range(nL):
                m.addConstr(f[li] <= F * y[li])
            for eid in sorted(covered):
                m.addConstr(gp.quicksum(f[li] for li in lbe[eid])
                            <= float(edges[eid]["edge_capacity"]))
            for ai, (aid, u, v, tt, eid) in enumerate(A):
                m.addConstr(gp.quicksum(x[oi, ai] for oi in range(nO))
                            <= gp.quicksum(lines[li]["cap"] * f[li] for li in lbe[eid]))
            for oi, o in enumerate(origins):
                dm = od[o]
                tot = sum(dm.values())
                for node in node_ids:
                    oa = out_arcs.get(node, [])
                    ia = in_arcs.get(node, [])
                    b = tot if node == o else -dm.get(node, 0.0)
                    if not oa and not ia:
                        continue
                    m.addConstr(gp.quicksum(x[oi, ai] for ai in oa)
                                - gp.quicksum(x[oi, ai] for ai in ia) == b)
            if heur_lfreq is not None:
                m.setAttr("Start", [y[i] for i in range(nL)],
                          [1.0 if heur_lfreq.get(i, 0.0) > 1e-9 else 0.0
                           for i in range(nL)])
                m.setAttr("Start", [f[i] for i in range(nL)],
                          [heur_lfreq.get(i, 0.0) for i in range(nL)])
                xs = [0.0] * (nO * nA)
                oindex = {o: i for i, o in enumerate(origins)}
                for o in origins:
                    for d, q in od[o].items():
                        for ai in opaths[(o, d)]:
                            xs[oindex[o] * nA + ai] += q
                m.setAttr("Start", xlist, xs)

            cb_best = [1e30]

            def cb(mm, where):
                if where == GRB.Callback.MIPSOL:
                    ov = mm.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if ov < cb_best[0] - 1e-9:
                        cb_best[0] = ov
                        if logger:
                            logger.log(ov)

            m.optimize(cb)
            if m.SolCount > 0:
                fv = m.getAttr("X", f)
                xv = m.getAttr("X", x)
                active = {li: fv[li] for li in range(nL) if fv[li] > 1e-7}
                path_recs = decompose(xv)
                obj2, sol2 = build_solution(active, path_recs)
                if best_obj is None or obj2 < best_obj - 1e-9:
                    best_obj, best_sol = obj2, sol2
                    if logger:
                        logger.log_solution(obj2, sol2)
    except Exception:
        pass

    if best_sol is None:
        best_obj = 0.0
        best_sol = {"objective_value": 0.0, "active_lines": [],
                    "active_passenger_paths": []}
        if logger:
            logger.log_solution(best_obj, best_sol)
    write_out(best_obj, best_sol)


if __name__ == "__main__":
    main()