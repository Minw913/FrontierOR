import argparse
import json
import sys
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GRB = True
except Exception:
    HAVE_GRB = False


# ---------------------------------------------------------------------------
# Bit utilities
# ---------------------------------------------------------------------------

def bits_to_list(bits):
    out = []
    while bits:
        lsb = bits & -bits
        out.append(lsb.bit_length() - 1)
        bits ^= lsb
    return out


def is_connected(bits, adj):
    """Check whether the induced subgraph on `bits` is connected."""
    if bits == 0:
        return False
    seen = bits & -bits
    frontier = seen
    while frontier:
        reach = 0
        f = frontier
        while f:
            lsb = f & -f
            reach |= adj[lsb.bit_length() - 1]
            f ^= lsb
        reach &= bits
        new = reach & ~seen
        seen |= new
        frontier = new
    return seen == bits


# ---------------------------------------------------------------------------
# Enumeration of all valid groups (s-plexes, optionally connected)
# ---------------------------------------------------------------------------

def enumerate_groups(n, adj, max_def, need_conn, deadline, cap):
    groups = []
    defs = {}
    nodes = [0]
    stopped = [False]

    def rec(pos, bits):
        if stopped[0]:
            return
        for v in range(pos, n):
            if stopped[0]:
                return
            missing = bits & ~adj[v]
            k = missing.bit_count()
            if k > max_def:
                continue
            ok = True
            upd = []
            mm = missing
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                mm ^= lsb
                if defs[u] + 1 > max_def:
                    ok = False
                    break
                upd.append(u)
            if not ok:
                continue

            nodes[0] += 1
            if (nodes[0] & 4095) == 0:
                if time.time() > deadline or len(groups) >= cap:
                    stopped[0] = True
                    return

            for u in upd:
                defs[u] += 1
            defs[v] = k
            nbits = bits | (1 << v)
            if (not need_conn) or is_connected(nbits, adj):
                groups.append(nbits)
            rec(v + 1, nbits)
            del defs[v]
            for u in upd:
                defs[u] -= 1

            if len(groups) >= cap:
                stopped[0] = True
                return

    rec(0, 0)
    return groups


# ---------------------------------------------------------------------------
# Greedy partition heuristic (initial incumbent / fallback)
# ---------------------------------------------------------------------------

def greedy_partition(n, adj, max_def, need_conn):
    unassigned = set(range(n))
    deg = [adj[v].bit_count() for v in range(n)]
    result = []
    while unassigned:
        v0 = max(unassigned, key=lambda v: deg[v])
        unassigned.discard(v0)
        g = 1 << v0
        defs = {v0: 0}
        while True:
            best_u = -1
            best_score = -1
            for u in unassigned:
                am = adj[u]
                if need_conn and not (am & g):
                    continue
                missing = g & ~am
                k = missing.bit_count()
                if k > max_def:
                    continue
                ok = True
                mm = missing
                while mm:
                    lsb = mm & -mm
                    w = lsb.bit_length() - 1
                    mm ^= lsb
                    if defs[w] + 1 > max_def:
                        ok = False
                        break
                if not ok:
                    continue
                score = (am & g).bit_count()
                if score > best_score:
                    best_score = score
                    best_u = u
            if best_u < 0:
                break
            u = best_u
            missing = g & ~adj[u]
            mm = missing
            while mm:
                lsb = mm & -mm
                w = lsb.bit_length() - 1
                mm ^= lsb
                defs[w] += 1
            defs[u] = missing.bit_count()
            g |= (1 << u)
            unassigned.discard(u)
        result.append(g)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.setrecursionlimit(100000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = max(1, int(args.time_limit))

    # ---- Load instance ----
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    graph = inst["graph"]
    settings = inst["problem_settings"]

    orig_ids = list(graph["vertices"])
    n = int(graph["num_vertices"])
    edges = graph["edges"]
    s = int(settings.get("parameter_s", 1))
    decomposition = str(settings.get("decomposition_type", "partitioning")).lower()
    need_conn = bool(settings.get("connectivity_required", False))
    is_partition = decomposition != "covering"

    max_def = max(s - 1, 0)

    if n == 0:
        out = {"objective_value": 0.0, "solution": []}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # ---- Build adjacency bitmasks (vertices ordered by degree, desc) ----
    degree = {v: 0 for v in orig_ids}
    edge_set = set()
    for e in edges:
        a, b = e[0], e[1]
        if a == b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key in edge_set:
            continue
        edge_set.add(key)
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    ordered = sorted(orig_ids, key=lambda v: -degree.get(v, 0))
    pos = {v: i for i, v in enumerate(ordered)}
    adj = [0] * n
    for (a, b) in edge_set:
        if a not in pos or b not in pos:
            continue
        ia, ib = pos[a], pos[b]
        adj[ia] |= (1 << ib)
        adj[ib] |= (1 << ia)

    # ---- Greedy initial incumbent ----
    greedy_groups = greedy_partition(n, adj, max_def, need_conn)
    best_solution = greedy_groups
    best_obj = float(len(greedy_groups))

    def make_solution_dict(obj, group_bits):
        sol = [sorted(ordered[v] for v in bits_to_list(g)) for g in group_bits]
        return {"objective_value": float(obj), "solution": sol}

    if logger:
        logger.log_solution(best_obj, make_solution_dict(best_obj, best_solution))

    # ---- Enumerate candidate groups ----
    GROUP_CAP = 200000
    enum_budget = 0.5 * time_limit
    enum_deadline = start_time + enum_budget
    enum_groups = enumerate_groups(n, adj, max_def, need_conn, enum_deadline, GROUP_CAP)

    # Ensure singletons and greedy groups are present (feasibility guarantee)
    group_set = set(enum_groups)
    for v in range(n):
        group_set.add(1 << v)
    for g in greedy_groups:
        group_set.add(g)
    groups = list(group_set)

    # ---- Set partitioning / covering MILP ----
    solved = False
    if HAVE_GRB and n > 0:
        remaining = time_limit - (time.time() - start_time) - 0.2
        if remaining > 0.05:
            try:
                m = gp.Model("splex_decomposition")
                m.setParam("Seed", 0)
                m.setParam("MIPGap", 1e-4)
                m.setParam("NumericFocus", 0)
                m.setParam("Threads", 1)
                m.setParam("TimeLimit", max(remaining, 0.05))
                m.setParam("OutputFlag", 0)

                K = len(groups)
                x = m.addVars(K, vtype=GRB.BINARY, obj=1.0)
                m.ModelSense = GRB.MINIMIZE

                cols = [[] for _ in range(n)]
                for i, g in enumerate(groups):
                    gg = g
                    while gg:
                        lsb = gg & -gg
                        v = lsb.bit_length() - 1
                        cols[v].append(i)
                        gg ^= lsb

                sense = GRB.EQUAL if is_partition else GRB.GREATER_EQUAL
                for v in range(n):
                    expr = gp.quicksum(x[i] for i in cols[v])
                    m.addConstr(expr, sense=sense, rhs=1.0)

                # MIP start from greedy solution
                idx_of = {g: i for i, g in enumerate(groups)}
                for g in greedy_groups:
                    i = idx_of.get(g)
                    if i is not None:
                        x[i].Start = 1.0

                if logger:
                    def callback(model, where):
                        nonlocal best_obj, best_solution
                        if where == GRB.Callback.MIPSOL:
                            try:
                                vals = model.cbGetSolution(x)
                                chosen = [groups[i] for i in range(K) if vals[i] > 0.5]
                                obj = float(len(chosen))
                                if obj < best_obj - 1e-9:
                                    best_obj = obj
                                    best_solution = chosen
                                    logger.log_solution(
                                        obj, make_solution_dict(obj, chosen))
                            except Exception:
                                pass

                    m.optimize(callback)
                else:
                    m.optimize()

                if m.SolCount > 0:
                    vals = m.getAttr("X", x)
                    chosen = [groups[i] for i in range(K) if vals[i] > 0.5]
                    obj = float(len(chosen))
                    if obj < best_obj - 1e-9:
                        best_obj = obj
                        best_solution = chosen
                    solved = True
            except Exception:
                solved = False

    if not solved:
        # Fallback: greedy partition (always feasible)
        best_solution = greedy_groups
        best_obj = float(len(greedy_groups))

    out = make_solution_dict(best_obj, best_solution)
    if logger:
        logger.log_solution(best_obj, out)

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()