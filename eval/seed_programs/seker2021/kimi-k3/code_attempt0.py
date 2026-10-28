import argparse
import heapq
import json
import random
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


class _Timeout(Exception):
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", required=True, type=int)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    start = time.monotonic()
    time_limit = max(0, int(args.time_limit))
    reserve = min(1.0, max(0.2, 0.05 * time_limit)) if time_limit > 0 else 0.0
    if time_limit > 0 and reserve > 0.8 * time_limit:
        reserve = 0.2 * time_limit
    compute_end = start + time_limit - reserve

    def rem():
        return compute_end - time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    graph = inst["graph"]
    part = inst["partition"]
    n = int(graph["num_vertices"])
    raw_edges = graph["edges"]
    clusters = [list(map(int, vs)) for vs in part["clusters"]]
    m = len(clusters)

    vertex_cluster = [-1] * n
    for ci, vs in enumerate(clusters):
        if not vs:
            raise ValueError("Empty cluster")
        for v in vs:
            if v < 0 or v >= n:
                raise ValueError("Vertex index out of range")
            if vertex_cluster[v] != -1:
                raise ValueError("Vertex appears in more than one cluster")
            vertex_cluster[v] = ci
    if any(c == -1 for c in vertex_cluster):
        raise ValueError("Some vertex is not assigned to a cluster")

    if m == 0:
        sol = {"objective_value": 0.0, "selected_vertices": {}, "coloring": {}}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    full_adj = [set() for _ in range(n)]
    full_edges = set()
    for e in raw_edges:
        u = int(e[0])
        v = int(e[1])
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError("Edge endpoint out of range")
        if u == v:
            continue
        if vertex_cluster[u] == vertex_cluster[v]:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in full_edges:
            continue
        full_edges.add((a, b))
        full_adj[u].add(v)
        full_adj[v].add(u)

    def reduce_cluster(vs):
        isolated = [v for v in vs if not full_adj[v]]
        if isolated:
            return [isolated[0]]
        if len(vs) <= 128 and n <= 100000:
            items = sorted((len(full_adj[v]), v) for v in vs)
            kept = []
            for deg, v in items:
                av = full_adj[v]
                dominated = False
                for u, au in kept:
                    if len(au) <= deg and au.issubset(av):
                        dominated = True
                        break
                if not dominated:
                    kept.append((v, av))
            return [v for v, _ in kept]
        return list(vs)

    if rem() > 0:
        cand_clusters = [reduce_cluster(vs) for vs in clusters]
    else:
        cand_clusters = [list(vs) for vs in clusters]
    cand_clusters = [vs if vs else list(clusters[ci]) for ci, vs in enumerate(cand_clusters)]
    cand_set = set(v for vs in cand_clusters for v in vs)
    n_cand = len(cand_set)

    adj = [set() for _ in range(n)]
    edge_list = []
    for u, v in full_edges:
        if u in cand_set and v in cand_set:
            adj[u].add(v)
            adj[v].add(u)
            edge_list.append((u, v))

    cluster_adj = [set() for _ in range(m)]
    for u, v in edge_list:
        cu = vertex_cluster[u]
        cv = vertex_cluster[v]
        cluster_adj[cu].add(cv)
        cluster_adj[cv].add(cu)
    cluster_deg = [len(s) for s in cluster_adj]
    min_adj = [min((len(adj[v]) for v in vs), default=0) for vs in cand_clusters]

    best_obj = 10 ** 9
    best_selection = None
    best_coloring = None

    def make_solution(selection, coloring):
        if not selection:
            return {"objective_value": 0.0, "selected_vertices": {}, "coloring": {}}
        vals = sorted({coloring[v] for v in selection})
        mp = {old: i + 1 for i, old in enumerate(vals)}
        return {
            "objective_value": float(len(vals)),
            "selected_vertices": {str(i): int(v) for i, v in enumerate(selection)},
            "coloring": {str(v): int(mp[coloring[v]]) for v in selection},
        }

    def consider(selection, coloring):
        nonlocal best_obj, best_selection, best_coloring
        if selection is None:
            return False
        if not selection:
            k = 0
            comp = {}
        else:
            vals = sorted({coloring[v] for v in selection})
            mp = {old: i for i, old in enumerate(vals)}
            comp = {v: mp[coloring[v]] for v in selection}
            k = len(vals)
        if k < best_obj:
            best_obj = k
            best_selection = list(selection)
            best_coloring = comp
            if logger:
                logger.log_solution(float(k), make_solution(best_selection, best_coloring))
            return True
        return False

    def fallback_unique():
        sel = [vs[0] for vs in cand_clusters]
        col = {v: ci for ci, v in enumerate(sel)}
        consider(sel, col)

    def construct_online(order, rng=None, deadline=None):
        selected = [None] * m
        color_of = {}
        processed = [False] * m
        next_color = 0
        for ci in order:
            if deadline is not None and time.monotonic() > deadline:
                raise _Timeout()
            processed[ci] = True
            records = []
            for v in cand_clusters[ci]:
                forb = set()
                for u in adj[v]:
                    cu = color_of.get(u)
                    if cu is not None:
                        forb.add(cu)
                fcount = len(forb)
                if fcount < next_color:
                    c = 0
                    while c in forb:
                        c += 1
                    new = 0
                    acount = next_color - fcount
                else:
                    c = next_color
                    new = 1
                    acount = 0
                unpcs = set()
                uncount = 0
                for u in adj[v]:
                    cu = vertex_cluster[u]
                    if not processed[cu]:
                        uncount += 1
                        unpcs.add(cu)
                key = (new, c, -acount, len(unpcs), uncount, len(adj[v]), v)
                records.append((key, v, c))
            if not records:
                raise ValueError("Cluster with no candidate vertices")
            if rng is None:
                _, v, c = min(records, key=lambda r: r[0])
            else:
                records.sort(key=lambda r: r[0])
                b0 = records[0][0][0]
                b1 = records[0][0][1]
                filt = [r for r in records if r[0][0] == b0 and r[0][1] == b1]
                if len(filt) > 1:
                    _, v, c = filt[rng.randrange(min(len(filt), 4))]
                else:
                    _, v, c = records[0]
            selected[ci] = v
            color_of[v] = c
            if c == next_color:
                next_color += 1
        return selected, color_of, next_color

    def dsatur_color(sel, rng=None, deadline=None):
        if not sel:
            return {}, 0
        sel_set = set(sel)
        deg = {}
        for v in sel:
            d = 0
            for u in adj[v]:
                if u in sel_set:
                    d += 1
            deg[v] = d
        if rng is None:
            tie = {v: v for v in sel}
        else:
            tie = {v: rng.random() for v in sel}
        color = {}
        uncolored = set(sel)
        nbr_cols = {v: set() for v in sel}
        sat = {v: 0 for v in sel}
        heap = []
        for v in sel:
            heapq.heappush(heap, (0, -deg[v], tie[v], v))
        pops = 0
        while uncolored:
            if deadline is not None and (pops & 255) == 0 and time.monotonic() > deadline:
                raise _Timeout()
            nsat, ndeg, tb, v = heapq.heappop(heap)
            if v not in uncolored:
                continue
            if nsat != -sat[v] or ndeg != -deg[v] or tb != tie[v]:
                continue
            used = nbr_cols[v]
            c = 0
            while c in used:
                c += 1
            color[v] = c
            uncolored.remove(v)
            pops += 1
            for u in adj[v]:
                if u in uncolored:
                    s = nbr_cols[u]
                    if c not in s:
                        s.add(c)
                        sat[u] += 1
                        heapq.heappush(heap, (-sat[u], -deg[u], tie[u], u))
        k = max(color.values()) + 1 if color else 0
        return color, k

    def evaluate_selection(sel, online_col, online_k, deadline, rng_for_dsatur=None):
        fcol = online_col
        if best_selection is None or online_k <= best_obj + 1:
            try:
                dscol, dsk = dsatur_color(sel, rng=rng_for_dsatur, deadline=deadline)
                if dsk < online_k:
                    fcol = dscol
            except _Timeout:
                consider(sel, fcol)
                raise
        consider(sel, fcol)

    if rem() <= 0:
        fallback_unique()
        sol = make_solution(best_selection, best_coloring)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    orders = [
        sorted(range(m), key=lambda i: (len(cand_clusters[i]), -cluster_deg[i], min_adj[i], i)),
        sorted(range(m), key=lambda i: (-cluster_deg[i], len(cand_clusters[i]), min_adj[i], i)),
        sorted(range(m), key=lambda i: (min_adj[i], len(cand_clusters[i]), -cluster_deg[i], i)),
        list(range(m)),
    ]
    for order in orders:
        if rem() <= 0.02:
            break
        try:
            sel, col, kol = construct_online(order, rng=None, deadline=compute_end)
            evaluate_selection(sel, col, kol, compute_end, rng_for_dsatur=None)
        except _Timeout:
            break

    rng = random.Random(0)

    def random_improve(phase_end):
        it = 0
        while time.monotonic() < phase_end and rem() > 0:
            it += 1
            if it % 3 == 0:
                order = list(range(m))
                rng.shuffle(order)
            elif it % 3 == 1:
                order = sorted(
                    range(m),
                    key=lambda i: (
                        len(cand_clusters[i]) * (1.0 + rng.random()),
                        -cluster_deg[i] + rng.random(),
                        i,
                    ),
                )
            else:
                order = sorted(
                    range(m),
                    key=lambda i: (
                        -cluster_deg[i] + rng.random() * (cluster_deg[i] + 1.0),
                        len(cand_clusters[i]) * (1.0 + rng.random()),
                        i,
                    ),
                )
            try:
                sel, col, kol = construct_online(order, rng=rng, deadline=phase_end)
                evaluate_selection(sel, col, kol, phase_end, rng_for_dsatur=rng)
            except _Timeout:
                break

    if gp is not None:
        phase_end = min(compute_end, start + min(5.0, max(0.0, 0.25 * time_limit)))
    else:
        phase_end = compute_end
    random_improve(phase_end)

    if best_selection is None:
        fallback_unique()

    def symmetry_clique():
        if m == 1:
            return [0]
        pair_count = {}
        for u, v in edge_list:
            a = vertex_cluster[u]
            b = vertex_cluster[v]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
        comp = [set() for _ in range(m)]
        for (a, b), cnt in pair_count.items():
            if cnt == len(cand_clusters[a]) * len(cand_clusters[b]):
                comp[a].add(b)
                comp[b].add(a)
        order = sorted(range(m), key=lambda i: (-len(comp[i]), len(cand_clusters[i]), i))
        clique = []
        for i in order:
            ok = True
            for j in clique:
                if j not in comp[i]:
                    ok = False
                    break
            if ok:
                clique.append(i)
        return clique

    fix_clusters = symmetry_clique()
    lb = max(1, len(fix_clusters))

    def relabel_with_fixed(selection, coloring, fixed):
        mapping = {}
        ok = True
        for j, ci in enumerate(fixed):
            if ci >= len(selection):
                ok = False
                break
            oc = coloring[selection[ci]]
            if oc in mapping and mapping[oc] != j:
                ok = False
                break
            mapping[oc] = j
        if not ok:
            mapping = {}
            fixed = []
        used_new = set(mapping.values())
        nxt = 0
        for v in selection:
            oc = coloring[v]
            if oc not in mapping:
                while nxt in used_new:
                    nxt += 1
                mapping[oc] = nxt
                used_new.add(nxt)
                nxt += 1
        return {v: mapping[coloring[v]] for v in selection}

    def run_mip(K, fixed):
        cand_vertices = [v for vs in cand_clusters for v in vs]
        model = gp.Model("cluster_select_color")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(0.01, rem())

        z = {}
        for v in cand_vertices:
            for c in range(K):
                z[(v, c)] = model.addVar(vtype=GRB.BINARY, name="z_%d_%d" % (v, c))
        y = model.addVars(K, vtype=GRB.BINARY, name="y")

        for ci, vs in enumerate(cand_clusters):
            model.addConstr(
                gp.quicksum(z[(v, c)] for v in vs for c in range(K)) == 1,
                name="cluster_%d" % ci,
            )
        for c in range(K):
            model.addConstr(
                gp.quicksum(z[(v, c)] for v in cand_vertices) <= m * y[c],
                name="use_%d" % c,
            )
        for c in range(K - 1):
            model.addConstr(y[c] >= y[c + 1], name="ord_%d" % c)
        for j, ci in enumerate(fixed[:K]):
            model.addConstr(
                gp.quicksum(z[(v, j)] for v in cand_clusters[ci]) == 1,
                name="fix_%d" % ci,
            )
        for u, v in edge_list:
            for c in range(K):
                model.addConstr(z[(u, c)] + z[(v, c)] <= 1, name="e_%d_%d_%d" % (u, v, c))

        model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)
        model.update()

        rel = relabel_with_fixed(best_selection, best_coloring, fixed[:K])
        for var in model.getVars():
            var.Start = 0.0
        for v in best_selection:
            z[(v, rel[v])].Start = 1.0
        used = set(rel.values())
        for c in range(K):
            y[c].Start = 1.0 if c in used else 0.0

        cluster_var_lists = [
            [z[(v, c)] for v in vs for c in range(K)] for vs in cand_clusters
        ]

        def _cb(model, where):
            if where == GRB.Callback.MIPSOL:
                sel = []
                col = {}
                for ci, vars_ in enumerate(cluster_var_lists):
                    vals = model.cbGetSolution(vars_)
                    idx = max(range(len(vals)), key=vals.__getitem__)
                    if vals[idx] <= 0.5:
                        return
                    v = cand_clusters[ci][idx // K]
                    c = idx % K
                    sel.append(v)
                    col[v] = c
                consider(sel, col)

        model.optimize(_cb)
        if model.SolCount > 0:
            sel = []
            col = {}
            for ci, vars_ in enumerate(cluster_var_lists):
                vals = [var.X for var in vars_]
                idx = max(range(len(vals)), key=vals.__getitem__)
                if vals[idx] > 0.5:
                    v = cand_clusters[ci][idx // K]
                    c = idx % K
                    sel.append(v)
                    col[v] = c
            if len(sel) == m:
                consider(sel, col)

    mip_done = False
    if gp is not None and GRB is not None and best_obj > lb and best_obj > 1 and rem() > 0.2:
        r = rem()
        if r < 10:
            max_conf = 300000
            max_z = 100000
        elif r < 30:
            max_conf = 800000
            max_z = 250000
        else:
            max_conf = 2000000
            max_z = 600000
        if len(edge_list) * best_obj <= max_conf and n_cand * best_obj <= max_z:
            try:
                run_mip(best_obj, fix_clusters)
                mip_done = True
            except Exception:
                mip_done = False

    if not mip_done and rem() > 0.05:
        random_improve(compute_end)

    if best_selection is None:
        fallback_unique()

    sol = make_solution(best_selection, best_coloring)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()