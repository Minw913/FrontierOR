import argparse
import json
import time


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enumerate all non-dominated objective vectors of a "
                    "multi-objective 0-1 knapsack problem."
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as fh:
        data = json.load(fh)
    n = int(data["num_items"])
    m = int(data["num_objectives"])
    weights = [int(v) for v in data["weights"]]
    capacity = int(data["capacity"])
    coeffs = [[int(v) for v in row] for row in data["objective_coefficients"]]
    return n, m, weights, capacity, coeffs


def selection_vector(sel, coeffs, n, m):
    vec = []
    for k in range(m):
        ck = coeffs[k]
        tot = 0
        for i in range(n):
            if sel[i]:
                tot += ck[i]
        vec.append(tot)
    return tuple(vec)


# ---------------------------------------------------------------------------
# Dominance pruning (minimization). Input sorted lexicographically; returns
# the unique nondominated vectors together with their selections.
# ---------------------------------------------------------------------------

def prune_nondominated(pairs, m):
    uniq = {}
    for v, s in pairs:
        if v not in uniq:
            uniq[v] = s
    items = sorted(uniq.items(), key=lambda kv: kv[0])
    vecs = [v for v, _ in items]
    sols = [s for _, s in items]
    F = len(vecs)
    if F == 0:
        return [], []
    try:
        import numpy as np
        A = np.asarray(vecs, dtype=np.int64)
        keep_idx = []
        S = None
        B = 256
        for start in range(0, F, B):
            block = A[start:start + B]
            if S is not None:
                dominated = np.zeros(len(block), dtype=bool)
                for s0 in range(0, S.shape[0], 2048):
                    chunk = S[s0:s0 + 2048]
                    dominated |= (
                        (chunk[:, None, :] <= block[None, :, :])
                        .all(axis=2)
                        .any(axis=0)
                    )
                    if dominated.all():
                        break
            else:
                dominated = np.zeros(len(block), dtype=bool)
            local = []
            for j in range(len(block)):
                if dominated[j]:
                    continue
                a = block[j]
                if local:
                    La = np.asarray(local)
                    if bool((La <= a).all(axis=1).any()):
                        continue
                local.append(a)
                keep_idx.append(start + j)
            if local:
                La = np.asarray(local)
                S = La if S is None else np.vstack([S, La])
        return ([vecs[i] for i in keep_idx],
                [sols[i] for i in keep_idx])
    except Exception:
        kept_v, kept_s = [], []
        for v, s in items:
            dominated = False
            for u in kept_v:
                if all(u[k] <= v[k] for k in range(m)):
                    dominated = True
                    break
            if not dominated:
                kept_v.append(v)
                kept_s.append(s)
        return kept_v, kept_s


# ---------------------------------------------------------------------------
# Gurobi-based exact enumeration
# ---------------------------------------------------------------------------

def _new_model(n, weights, capacity):
    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("multiobjective_knapsack")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity,
        name="capacity",
    )
    return model, x, gp, GRB


def _optimize_with_deadline(model, deadline):
    remaining = deadline - time.time()
    if remaining <= 0.3:
        return None
    model.setParam("TimeLimit", max(0.1, remaining - 0.25))
    model.optimize()
    return model.Status


def solve_single(n, weights, capacity, coeffs, deadline, logger, build_dict):
    model, x, gp, GRB = _new_model(n, weights, capacity)
    c1 = coeffs[0]
    f1 = gp.quicksum(c1[i] * x[i] for i in range(n))
    model.setObjective(f1, GRB.MINIMIZE)
    try:
        status = _optimize_with_deadline(model, deadline)
        if status == GRB.OPTIMAL and model.SolCount > 0:
            xv = [1 if x[i].X > 0.5 else 0 for i in range(n)]
            p = (sum(c1[i] for i in range(n) if xv[i]),)
            if logger:
                logger.log_solution(1, build_dict([p], [xv]))
            return [p], [xv], True
    except Exception:
        pass
    return [], [], False


def sweep_biobjective(n, weights, capacity, coeffs, deadline, logger,
                      build_dict):
    """Exact lexicographic epsilon-constraint sweep for two objectives."""
    model, x, gp, GRB = _new_model(n, weights, capacity)
    c1, c2 = coeffs[0], coeffs[1]
    f1 = gp.quicksum(c1[i] * x[i] for i in range(n))
    f2 = gp.quicksum(c2[i] * x[i] for i in range(n))
    UB1 = sum(c for c in c1 if c > 0)
    UB2 = sum(c for c in c2 if c > 0)
    f1bound = model.addConstr(f1 <= UB1 + 1, name="f1bound")
    f2bound = model.addConstr(f2 <= UB2 + 1, name="f2bound")
    points, sols, seen = [], [], set()
    complete = False
    b = UB2 + 1  # current inclusive upper bound on f2
    while True:
        try:
            # Phase 1: minimize f1 subject to f2 <= b
            f1bound.RHS = UB1 + 1
            f2bound.RHS = b
            model.setObjective(f1, GRB.MINIMIZE)
            status = _optimize_with_deadline(model, deadline)
            if status == GRB.INFEASIBLE:
                complete = True
                break
            if status != GRB.OPTIMAL or model.SolCount == 0:
                break
            xv1 = [1 if x[i].X > 0.5 else 0 for i in range(n)]
            v1 = sum(c1[i] for i in range(n) if xv1[i])
            # Phase 2: minimize f2 subject to f2 <= b and f1 <= v1
            f1bound.RHS = v1
            model.setObjective(f2, GRB.MINIMIZE)
            status = _optimize_with_deadline(model, deadline)
            if status != GRB.OPTIMAL or model.SolCount == 0:
                break
            xv = [1 if x[i].X > 0.5 else 0 for i in range(n)]
            p = (sum(c1[i] for i in range(n) if xv[i]),
                 sum(c2[i] for i in range(n) if xv[i]))
            if p in seen:
                break
            seen.add(p)
            points.append(p)
            sols.append(xv)
            if logger:
                try:
                    logger.log_solution(len(points), build_dict(points, sols))
                except Exception:
                    pass
            b = p[1] - 1
        except Exception:
            break
    return points, sols, complete


def sylva_crema(n, m, weights, capacity, coeffs, deadline, logger, build_dict):
    """Exact Pareto-front enumeration for >= 3 objectives via iterative
    exclusion of already-found nondominated points (Sylva & Crema)."""
    model, x, gp, GRB = _new_model(n, weights, capacity)
    fexpr = [gp.quicksum(coeffs[k][i] * x[i] for i in range(n))
             for k in range(m)]
    UB = [sum(c for c in coeffs[k] if c > 0) for k in range(m)]
    model.setObjective(gp.quicksum(fexpr), GRB.MINIMIZE)
    points, sols, seen = [], [], set()
    complete = False
    it = 0
    while True:
        try:
            status = _optimize_with_deadline(model, deadline)
            if status is None:
                break
            if status == GRB.INFEASIBLE:
                complete = True
                break
            if status != GRB.OPTIMAL or model.SolCount == 0:
                break
            xv = [1 if x[i].X > 0.5 else 0 for i in range(n)]
            p = tuple(sum(coeffs[k][i] for i in range(n) if xv[i])
                      for k in range(m))
            if p in seen:
                break
            seen.add(p)
            points.append(p)
            sols.append(xv)
            it += 1
            # Exclude every found point: the next point must beat p somewhere.
            zs = [model.addVar(vtype=GRB.BINARY, name="z%d_%d" % (it, k))
                  for k in range(m)]
            model.update()
            for k in range(m):
                Mk = UB[k] - p[k] + 1
                model.addConstr(fexpr[k] <= p[k] - 1 + Mk * (1 - zs[k]),
                                name="ex%d_%d" % (it, k))
            model.addConstr(gp.quicksum(zs) >= 1, name="exs%d" % it)
        except Exception:
            break
        if logger:
            try:
                logger.log_solution(len(points), build_dict(points, sols))
            except Exception:
                pass
    return points, sols, complete


# ---------------------------------------------------------------------------
# Fallback: exact lexicographic extreme points via pseudo-polynomial DP
# (used only if Gurobi is unavailable / fails before producing any point).
# ---------------------------------------------------------------------------

def lex_extremes(n, m, weights, capacity, coeffs, deadline):
    found = {}
    if capacity < 0 or capacity + 1 > 5_000_000:
        return []
    for k0 in range(m):
        if time.time() > deadline - 0.5:
            break
        order = [k0] + [k for k in range(m) if k != k0]
        dp = [None] * (capacity + 1)
        dp[0] = ((0,) * m, 0)
        completed = True
        for i in range(n):
            wi = weights[i]
            ci = tuple(coeffs[k][i] for k in order)
            bit = 1 << i
            if 0 <= wi <= capacity:
                for wc in range(capacity - wi, -1, -1):
                    cur = dp[wc]
                    if cur is None:
                        continue
                    w2 = wc + wi
                    nt = tuple(cur[0][j] + ci[j] for j in range(m))
                    old = dp[w2]
                    if old is None or nt < old[0]:
                        dp[w2] = (nt, cur[1] | bit)
            if time.time() > deadline - 0.45:
                completed = False
                break
        if not completed:
            break
        best_t = None
        best_mask = 0
        for cur in dp:
            if cur is not None and (best_t is None or cur[0] < best_t):
                best_t = cur[0]
                best_mask = cur[1]
        if best_t is not None:
            sel = [(best_mask >> i) & 1 for i in range(n)]
            vec = selection_vector(sel, coeffs, n, m)
            if vec not in found:
                found[vec] = sel
    return list(found.items())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_solvers(n, m, weights, capacity, coeffs, deadline, logger, build_dict):
    # Trivial case: if every coefficient is nonnegative, the empty selection
    # (objective vector all zeros) dominates every other selection.
    if all(c >= 0 for row in coeffs for c in row):
        sel = [0] * n
        return [(tuple([0] * m), sel)], True

    pairs = []
    complete = False
    try:
        if m == 1:
            pts, sols, complete = solve_single(
                n, weights, capacity, coeffs, deadline, logger, build_dict)
        elif m == 2:
            pts, sols, complete = sweep_biobjective(
                n, weights, capacity, coeffs, deadline, logger, build_dict)
        else:
            pts, sols, complete = sylva_crema(
                n, m, weights, capacity, coeffs, deadline, logger, build_dict)
        pairs = list(zip(pts, sols))
    except Exception:
        pairs = []
        complete = False

    if not pairs:
        try:
            pairs = lex_extremes(n, m, weights, capacity, coeffs, deadline)
        except Exception:
            pairs = []
    if not pairs:
        sel = [0] * n
        pairs = [(selection_vector(sel, coeffs, n, m), sel)]
    return pairs, complete


def main():
    args = parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    start = time.time()
    deadline = start + max(0, int(args.time_limit))

    n, m, weights, capacity, coeffs = load_instance(args.instance_path)

    def build_dict(points, sols):
        return {
            "objective_value": len(points),
            "non_dominated_objective_vectors": [list(p) for p in points],
            "solutions": [list(s) for s in sols],
        }

    pairs, _complete = run_solvers(
        n, m, weights, capacity, coeffs, deadline, logger, build_dict)

    # Verify feasibility and recompute objective vectors from the selections
    # so the output is guaranteed consistent.
    checked = []
    for v, s in pairs:
        s = [1 if t else 0 for t in s]
        tw = 0
        for i in range(n):
            if s[i]:
                tw += weights[i]
        if tw > capacity:
            continue
        checked.append((selection_vector(s, coeffs, n, m), s))
    if not checked:
        sel = [0] * n
        checked = [(selection_vector(sel, coeffs, n, m), sel)]

    vecs, sels = prune_nondominated(checked, m)
    out = build_dict(vecs, sels)

    if logger:
        try:
            logger.log_solution(out["objective_value"], out)
        except Exception:
            pass

    with open(args.solution_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()