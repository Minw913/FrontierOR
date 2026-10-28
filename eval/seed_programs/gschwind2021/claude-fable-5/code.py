import argparse
import json
import random
import time
from collections import Counter

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------
def is_valid_group(G, adj, s, conn):
    """Check whether vertex set G is an s-plex (and connected if required)."""
    k = len(G)
    if k <= 1:
        return True
    for v in G:
        if len(adj[v] & G) < k - s:
            return False
    if conn:
        it = iter(G)
        start = next(it)
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in G and w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(seen) != k:
            return False
    return True


def validate_all(groups, vertices, adj, s, conn, covering):
    if not groups:
        return len(vertices) == 0
    vs = set(vertices)
    allv = set()
    tot = 0
    for g in groups:
        gs = set(g)
        if not gs or not gs <= vs:
            return False
        if not is_valid_group(gs, adj, s, conn):
            return False
        allv |= gs
        tot += len(gs)
    if allv != vs:
        return False
    if not covering and tot != len(vs):
        return False
    return True


# ---------------------------------------------------------------------------
# Greedy construction
# ---------------------------------------------------------------------------
def build_group(pool, adj, s, conn, rng, seed_mode):
    """Greedily build one maximal s-plex from vertices in pool."""
    if seed_mode == 'random':
        seed = rng.choice(tuple(pool))
    else:
        best_v = None
        best_d = -1
        for v in pool:
            d = len(adj[v] & pool)
            if d > best_d or (d == best_d and rng.random() < 0.3):
                best_v = v
                best_d = d
        seed = best_v

    G = {seed}
    deg = {seed: 0}          # degree inside G for members
    cand_deg = {}            # degree into G for candidates with >=1 neighbor
    pos = set()
    for w in adj[seed]:
        if w in pool:
            cand_deg[w] = 1
            pos.add(w)
    zero = (pool - adj[seed]) - {seed}   # candidates with no neighbor in G

    while True:
        k = len(G)
        thr = k - s + 1  # minimum required in-degree for a new vertex
        sat = [u for u in G if (k - 1) - deg[u] == s - 1]  # saturated members

        best_c = None
        best_dc = -1
        for c in pos:
            dc = cand_deg[c]
            if dc < thr or dc <= best_dc:
                if not (dc == best_dc and rng.random() < 0.3):
                    continue
                if dc < thr:
                    continue
            ac = adj[c]
            ok = True
            for u in sat:
                if u not in ac:
                    ok = False
                    break
            if ok:
                best_c = c
                best_dc = dc

        if best_c is None and thr <= 0 and not conn:
            # allow adding a vertex with zero neighbors inside the group
            for c in zero:
                if sat:
                    # zero-degree vertex cannot be adjacent to saturated members
                    break
                best_c = c
                best_dc = 0
                break

        if best_c is None:
            break

        G.add(best_c)
        if best_c in pos:
            pos.discard(best_c)
            deg[best_c] = cand_deg.pop(best_c)
        else:
            zero.discard(best_c)
            deg[best_c] = 0
        for w in adj[best_c]:
            if w in deg:
                deg[w] += 1
            elif w in cand_deg:
                cand_deg[w] += 1
            elif w in zero:
                zero.discard(w)
                cand_deg[w] = 1
                pos.add(w)
    return G


def greedy_partition(vertices, adj, s, conn, rng, seed_mode, deadline):
    pool = set(vertices)
    groups = []
    while pool:
        if time.time() > deadline:
            groups.extend({v} for v in pool)
            break
        G = build_group(pool, adj, s, conn, rng, seed_mode)
        groups.append(G)
        pool -= G
    return groups


# ---------------------------------------------------------------------------
# Improvement moves
# ---------------------------------------------------------------------------
def merge_all(groups, adj, s, conn, deadline):
    groups = [set(g) for g in groups]
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        groups.sort(key=len)
        m = len(groups)
        cnt = 0
        for i in range(m - 1):
            if changed:
                break
            for j in range(i + 1, m):
                cnt += 1
                if (cnt & 255) == 0 and time.time() > deadline:
                    return groups
                U = groups[i] | groups[j]
                if is_valid_group(U, adj, s, conn):
                    groups[i] = U
                    groups.pop(j)
                    changed = True
                    break
    return groups


def try_eliminate(groups, adj, s, conn, covering, deadline):
    """Try to remove one group by redistributing its vertices."""
    idxs = sorted(range(len(groups)), key=lambda i: len(groups[i]))
    for gi in idxs:
        if time.time() > deadline:
            return None
        g = groups[gi]
        others = [set(groups[k]) for k in range(len(groups)) if k != gi]
        if not others:
            continue
        others.sort(key=len)
        ok = True
        for v in g:
            if covering and any(v in h for h in others):
                continue
            placed = False
            for h in others:
                if v in h:
                    continue
                if is_valid_group(h | {v}, adj, s, conn):
                    h.add(v)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return others
    return None


def prune_redundant(groups):
    """For covering: drop groups whose vertices are all covered elsewhere."""
    cov = Counter()
    for g in groups:
        for v in g:
            cov[v] += 1
    out = []
    for g in sorted(groups, key=len):
        if all(cov[v] > 1 for v in g):
            for v in g:
                cov[v] -= 1
        else:
            out.append(g)
    return out if out else groups


def improve(groups, adj, s, conn, covering, deadline):
    groups = merge_all(groups, adj, s, conn, deadline)
    while time.time() < deadline:
        ng = try_eliminate(groups, adj, s, conn, covering, deadline)
        if ng is None:
            break
        groups = merge_all(ng, adj, s, conn, deadline)
    if covering:
        groups = prune_redundant(groups)
    return groups


# ---------------------------------------------------------------------------
# Exact ILP (used when connectivity is automatic: conn False or s == 1)
# ---------------------------------------------------------------------------
def run_ilp(vertices, adj, s, covering, ub_groups, time_left, record,
            cur_best_len, conn):
    import gurobipy as gp
    from gurobipy import GRB

    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    K = len(ub_groups)
    if K <= 1:
        return None

    m = gp.Model("splex_decomp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, time_left)

    x = {}
    for i, v in enumerate(vertices):
        for g in range(min(i + 1, K)):
            x[(v, g)] = m.addVar(vtype=GRB.BINARY)
    y = [m.addVar(vtype=GRB.BINARY) for _ in range(K)]
    m.setObjective(gp.quicksum(y), GRB.MINIMIZE)

    for i, v in enumerate(vertices):
        lim = min(i + 1, K)
        expr = gp.quicksum(x[(v, g)] for g in range(lim))
        if covering:
            m.addConstr(expr >= 1)
        else:
            m.addConstr(expr == 1)
        for g in range(lim):
            m.addConstr(x[(v, g)] <= y[g])
    for g in range(K - 1):
        m.addConstr(y[g] >= y[g + 1])

    for g in range(K):
        for v in vertices:
            if (v, g) not in x:
                continue
            terms = [x[(u, g)] for u in vertices
                     if u != v and u not in adj[v] and (u, g) in x]
            if len(terms) > s - 1:
                m.addConstr(gp.quicksum(terms) <= (s - 1) + n * (1 - x[(v, g)]))

    # warm start (order groups by minimum vertex index)
    order = sorted(range(K), key=lambda gi: min(idx[v] for v in ub_groups[gi]))
    for var in x.values():
        var.Start = 0.0
    for newg, gi in enumerate(order):
        for v in ub_groups[gi]:
            key = (v, newg)
            if key in x:
                x[key].Start = 1.0
    for g in range(K):
        y[g].Start = 1.0

    keys = list(x.keys())
    xvars = [x[k] for k in keys]
    state = {'best': cur_best_len}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            ov = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if ov < state['best'] - 0.5:
                try:
                    vals = model.cbGetSolution(xvars)
                    gs = [set() for _ in range(K)]
                    for k, val in zip(keys, vals):
                        if val > 0.5:
                            gs[k[1]].add(k[0])
                    gs = [g for g in gs if g]
                    state['best'] = len(gs)
                    record(gs)
                except Exception:
                    pass

    m.optimize(cb)

    if m.SolCount > 0:
        gs = [set() for _ in range(K)]
        for k, var in x.items():
            if var.X > 0.5:
                gs[k[1]].add(k[0])
        gs = [g for g in gs if g]
        return gs
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        data = json.load(f)
    graph = data['graph']
    ps = data['problem_settings']
    vertices = list(graph['vertices'])
    s = int(ps['parameter_s'])
    conn = bool(ps['connectivity_required'])
    covering = str(ps['decomposition_type']).lower().startswith('cover')

    adj = {v: set() for v in vertices}
    for e in graph.get('edges', []):
        a, b = e[0], e[1]
        if a == b:
            continue
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    n = len(vertices)

    state = {'best': None, 'len': float('inf')}

    def record(groups):
        gs = [set(g) for g in groups]
        if len(gs) >= state['len']:
            return
        if not validate_all(gs, vertices, adj, s, conn, covering):
            return
        state['best'] = gs
        state['len'] = len(gs)
        sol = {"objective_value": float(len(gs)),
               "solution": [sorted(g) for g in gs]}
        if logger:
            try:
                logger.log_solution(float(len(gs)), sol)
            except Exception:
                pass
        try:
            with open(args.solution_path, 'w') as f:
                json.dump(sol, f)
        except Exception:
            pass

    # initial trivial solution: singletons
    record([{v} for v in vertices])

    if n == 0:
        sol = {"objective_value": 0.0, "solution": []}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    # quick check: whole vertex set as one group
    full = set(vertices)
    if is_valid_group(full, adj, s, conn):
        record([full])
        return

    ilp_eligible = (not conn or s == 1) and n <= 100
    if ilp_eligible:
        heur_end = min(deadline, start + max(2.0, 0.35 * args.time_limit))
    else:
        heur_end = deadline

    # heuristic phase: randomized restarts with merge/eliminate improvement
    rng_seed = 0
    non_improve = 0
    max_non_improve = 60 if ilp_eligible else 400
    first = True
    while time.time() < heur_end and state['len'] > 1:
        rng = random.Random(rng_seed)
        rng_seed += 1
        if first:
            mode = 'maxdeg'
            first = False
        else:
            mode = rng.choice(['maxdeg', 'maxdeg', 'random'])
        groups = greedy_partition(vertices, adj, s, conn, rng, mode, deadline)
        groups = improve(groups, adj, s, conn, covering, deadline)
        if len(groups) < state['len']:
            record(groups)
            non_improve = 0
        else:
            non_improve += 1
        if non_improve >= max_non_improve:
            break

    # exact phase (only when connectivity is automatically satisfied)
    if ilp_eligible and state['len'] > 1:
        time_left = deadline - time.time()
        if time_left > 3.0:
            try:
                res = run_ilp(vertices, adj, s, covering, state['best'],
                              time_left, record, state['len'], conn)
                if res is not None and len(res) < state['len']:
                    record(res)
            except Exception:
                pass

    # if time remains and no ILP, keep restarting heuristic
    while time.time() < deadline and state['len'] > 1 and not ilp_eligible:
        rng = random.Random(rng_seed)
        rng_seed += 1
        mode = rng.choice(['maxdeg', 'random'])
        groups = greedy_partition(vertices, adj, s, conn, rng, mode, deadline)
        groups = improve(groups, adj, s, conn, covering, deadline)
        if len(groups) < state['len']:
            record(groups)
            non_improve = 0
        else:
            non_improve += 1
        if non_improve >= 2000:
            break

    # final write (best solution already validated)
    best = state['best'] if state['best'] is not None else [{v} for v in vertices]
    sol = {"objective_value": float(len(best)),
           "solution": [sorted(g) for g in best]}
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()