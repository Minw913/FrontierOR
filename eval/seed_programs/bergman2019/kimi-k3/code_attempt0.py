import argparse
import json
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_objective(assignment, profits, pairwise):
    """Exact objective of an assignment: item profits + same-knapsack pair profits."""
    by_k = {}
    total = 0.0
    for i, k in assignment:
        total += profits[i]
        if k in by_k:
            by_k[k].append(i)
        else:
            by_k[k] = [i]
    for items in by_k.values():
        L = len(items)
        for a in range(L):
            ia = items[a]
            row = pairwise[ia]
            for b in range(a + 1, L):
                total += row[items[b]]
    return total


def greedy_construct(n, m, profits, pairwise, weights, capacities, feas_k, order, deadline):
    loads = [0] * m
    assign = [-1] * n
    items_in_k = [[] for _ in range(m)]
    for idx, i in enumerate(order):
        if (idx & 63) == 0 and time.time() > deadline:
            break
        fk = feas_k[i]
        if not fk:
            continue
        wi = weights[i]
        row = pairwise[i]
        pi = profits[i]
        best_gain = 1e-9
        best_k = -1
        for k in fk:
            if loads[k] + wi <= capacities[k]:
                g = pi
                for j in items_in_k[k]:
                    g += row[j]
                if g > best_gain:
                    best_gain = g
                    best_k = k
        if best_k >= 0:
            assign[i] = best_k
            loads[best_k] += wi
            items_in_k[best_k].append(i)
    return assign, loads, items_in_k


def local_search(n, m, profits, pairwise, weights, capacities, feas_k,
                 assign, loads, items_in_k, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            wi = weights[i]
            row = pairwise[i]
            pi = profits[i]
            k0 = assign[i]
            cur = 0.0
            if k0 >= 0:
                cur = pi
                for j in items_in_k[k0]:
                    if j != i:
                        cur += row[j]
            best_delta = 1e-9
            best_k = k0
            if k0 >= 0:
                d = -cur
                if d > best_delta:
                    best_delta = d
                    best_k = -1
            for k in feas_k[i]:
                if k == k0:
                    continue
                if loads[k] + wi <= capacities[k]:
                    g = pi
                    for j in items_in_k[k]:
                        g += row[j]
                    d = g - cur
                    if d > best_delta:
                        best_delta = d
                        best_k = k
            if best_k != k0:
                if k0 >= 0:
                    items_in_k[k0].remove(i)
                    loads[k0] -= wi
                if best_k >= 0:
                    items_in_k[best_k].append(i)
                    loads[best_k] += wi
                assign[i] = best_k
                improved = True
            if (i & 63) == 0 and time.time() > deadline:
                return assign, loads, items_in_k
    return assign, loads, items_in_k


def run_heuristics(n, m, profits, pairwise, weights, capacities, feas_k,
                   deadline, restarts, log_fn):
    rng = random.Random(0)
    best_obj = None
    best_assignment = []
    best_assign = [-1] * n
    attempt = 0
    while True:
        attempt += 1
        if attempt == 1:
            order = sorted(range(n), key=lambda i: -profits[i])
        else:
            order = list(range(n))
            rng.shuffle(order)
        assign, loads, items_in_k = greedy_construct(
            n, m, profits, pairwise, weights, capacities, feas_k, order, deadline)
        assign, loads, items_in_k = local_search(
            n, m, profits, pairwise, weights, capacities, feas_k,
            assign, loads, items_in_k, deadline)
        assignment = [[i, assign[i]] for i in range(n) if assign[i] >= 0]
        obj = compute_objective(assignment, profits, pairwise)
        if best_obj is None or obj > best_obj + 1e-9:
            best_obj = obj
            best_assignment = assignment
            best_assign = list(assign)
            log_fn(obj, assignment)
        if not restarts or time.time() >= deadline:
            break
    return best_obj, best_assignment, best_assign


def solve_mip(n, m, profits, pairwise, weights, capacities, feas_k, feas_i,
              start_assign, time_budget, incumbent_obj, log_fn):
    model = gp.Model("quadratic_multiple_knapsack")
    params = model.Params
    params.Seed = 0
    params.MIPGap = 1e-4
    params.NumericFocus = 0
    params.Threads = 1
    params.OutputFlag = 0
    params.TimeLimit = max(1.0, float(time_budget))

    x_index = [(i, k) for i in range(n) for k in feas_k[i]]
    if not x_index:
        return None, None
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")

    # Linearize pair products. y is continuous: with binary x, the sign-based
    # linking constraints force y to the exact product at integer solutions.
    y_index = []
    y_pos = []
    for i in range(n):
        row = pairwise[i]
        fki = feas_k[i]
        if not fki:
            continue
        for j in range(i + 1, n):
            q = row[j]
            if q == 0:
                continue
            wj = weights[j]
            for k in fki:
                if wj <= capacities[k]:
                    y_index.append((i, j, k))
                    y_pos.append(q > 0)

    y = None
    if y_index:
        y = model.addVars(y_index, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

    obj = gp.LinExpr()
    obj.addTerms([float(profits[i]) for (i, k) in x_index],
                 [x[i, k] for (i, k) in x_index])
    if y_index:
        obj.addTerms([float(pairwise[i][j]) for (i, j, k) in y_index],
                     [y[i, j, k] for (i, j, k) in y_index])
    model.setObjective(obj, GRB.MAXIMIZE)

    # Each item in at most one knapsack.
    for i in range(n):
        ks = feas_k[i]
        if len(ks) > 1:
            model.addConstr(gp.quicksum(x[i, k] for k in ks) <= 1.0)

    # Capacity constraints.
    for k in range(m):
        fi = feas_i[k]
        if fi:
            model.addConstr(
                gp.quicksum(float(weights[i]) * x[i, k] for i in fi)
                <= float(capacities[k]))

    # Linking constraints, exploiting the sign of each pairwise coefficient.
    if y_index:
        pos_idx = [t for t, pf in zip(y_index, y_pos) if pf]
        neg_idx = [t for t, pf in zip(y_index, y_pos) if not pf]
        if pos_idx:
            model.addConstrs((y[i, j, k] <= x[i, k] for (i, j, k) in pos_idx),
                             name="pos_i")
            model.addConstrs((y[i, j, k] <= x[j, k] for (i, j, k) in pos_idx),
                             name="pos_j")
        if neg_idx:
            model.addConstrs(
                (x[i, k] + x[j, k] - y[i, j, k] <= 1.0 for (i, j, k) in neg_idx),
                name="neg")

    # MIP warm start from the heuristic solution.
    if start_assign is not None:
        for (i, k) in x_index:
            x[i, k].Start = 1.0 if start_assign[i] == k else 0.0

    cb_state = {"best": incumbent_obj}
    recompute_in_cb = (n <= 300)

    def _cb(model_, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = model_.cbGetSolution(x)
                assignment = [[i, k] for (i, k) in x_index if xv[i, k] > 0.5]
                if recompute_in_cb:
                    val = compute_objective(assignment, profits, pairwise)
                else:
                    val = float(model_.cbGet(GRB.Callback.MIPSOL_OBJ))
                if val > cb_state["best"] + 1e-9:
                    cb_state["best"] = val
                    log_fn(val, assignment)
            except Exception:
                pass

    model.optimize(_cb)

    if model.SolCount > 0:
        xv = model.getAttr("X", x)
        assignment = [[i, k] for (i, k) in x_index if xv[i, k] > 0.5]
        val = compute_objective(assignment, profits, pairwise)
        return val, assignment
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    total_limit = float(args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    log_state = {"last": None}

    def log_fn(obj, assignment):
        if logger is None:
            return
        last = log_state["last"]
        if last is None or obj > last + 1e-9:
            log_state["last"] = obj
            logger.log_solution(float(obj),
                                {"objective_value": float(obj),
                                 "assignment": assignment})

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst["n"])
    m = int(inst["m"])
    profits = inst["profits"]
    pairwise = inst["pairwise_profits"]
    weights = inst["weights"]
    capacities = inst["capacities"]

    best_obj = 0.0
    best_assignment = []

    if n > 0 and m > 0:
        feas_k = [[k for k in range(m) if weights[i] <= capacities[k]]
                  for i in range(n)]
        feas_i = [[i for i in range(n) if weights[i] <= capacities[k]]
                  for k in range(m)]

        # Count nonzero pairs (early exit once the MIP-size threshold is exceeded).
        pair_threshold = 700000
        P = 0
        for i in range(n):
            row = pairwise[i]
            for j in range(i + 1, n):
                if row[j] != 0:
                    P += 1
            if P * m > pair_threshold:
                break

        use_mip = (n * m <= 200000) and (P * m <= pair_threshold)

        elapsed = time.time() - t0
        remaining = total_limit - elapsed
        if use_mip:
            heur_budget = max(0.3, min(0.15 * total_limit, 8.0))
        else:
            heur_budget = max(0.3, remaining - 1.0)
        heur_deadline = time.time() + heur_budget

        best_obj, best_assignment, best_assign = run_heuristics(
            n, m, profits, pairwise, weights, capacities, feas_k,
            heur_deadline, restarts=not use_mip, log_fn=log_fn)

        if use_mip:
            budget = total_limit - (time.time() - t0) - 1.5
            if budget > 2.0:
                try:
                    res_obj, res_assignment = solve_mip(
                        n, m, profits, pairwise, weights, capacities,
                        feas_k, feas_i, best_assign, budget, best_obj, log_fn)
                    if res_obj is not None and res_obj > best_obj + 1e-9:
                        best_obj = res_obj
                        best_assignment = res_assignment
                except Exception:
                    pass

    log_fn(best_obj, best_assignment)

    out = {"objective_value": float(best_obj), "assignment": best_assignment}
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()