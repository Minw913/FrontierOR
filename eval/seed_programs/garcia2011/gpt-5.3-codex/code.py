import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

# Optional Gurobi import (fallback to heuristic-only if unavailable)
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    gp = None
    GRB = None
    HAS_GUROBI = False


def rounded_euclidean_matrix(coords, n):
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj))
            v = int(math.floor(d + 0.5))
            if v <= 0:
                v = 1
            mat[i][j] = v
            mat[j][i] = v
    return mat


def parse_cost_matrix(instance, n):
    raw = instance.get("cost_matrix", None)
    coords_raw = instance.get("node_coordinates", None)

    coords = None
    if isinstance(coords_raw, list) and len(coords_raw) == n:
        ok = True
        temp = []
        for row in coords_raw:
            if not isinstance(row, list) or len(row) != 2:
                ok = False
                break
            try:
                x = float(row[0])
                y = float(row[1])
                temp.append((x, y))
            except Exception:
                ok = False
                break
        if ok:
            coords = temp

    base = rounded_euclidean_matrix(coords, n) if coords is not None else None

    if raw is None:
        if base is None:
            raise ValueError("Instance must provide valid cost_matrix or node_coordinates.")
        return base

    mat = [[None] * n for _ in range(n)]

    if isinstance(raw, list):
        for i in range(min(n, len(raw))):
            row = raw[i]
            if not isinstance(row, list):
                continue
            for j in range(min(n, len(row))):
                v = row[j]
                if v is None:
                    continue
                try:
                    mat[i][j] = int(round(float(v)))
                except Exception:
                    pass

    # Symmetry completion
    for i in range(n):
        for j in range(n):
            if mat[i][j] is None and mat[j][i] is not None:
                mat[i][j] = mat[j][i]

    # Fill from coordinates if available
    if base is not None:
        for i in range(n):
            for j in range(n):
                if mat[i][j] is None:
                    mat[i][j] = base[i][j]

    # Final cleanup
    max_seen = 1
    for i in range(n):
        for j in range(n):
            if mat[i][j] is not None and mat[i][j] > max_seen:
                max_seen = mat[i][j]

    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            elif mat[i][j] is None:
                mat[i][j] = max_seen
            elif mat[i][j] <= 0:
                mat[i][j] = 1

    return mat


def compute_assignment(cost, opens):
    n = len(cost)
    if not opens:
        raise ValueError("Open facility set cannot be empty.")
    assigned_fac = [opens[0]] * n
    assigned_cost = [0] * n
    total = 0
    for i in range(n):
        best_f = opens[0]
        best_c = cost[i][best_f]
        for f in opens[1:]:
            c = cost[i][f]
            if c < best_c:
                best_c = c
                best_f = f
        assigned_fac[i] = best_f
        assigned_cost[i] = best_c
        total += best_c
    return total, assigned_fac, assigned_cost


def greedy_initial_solution(cost, p, rng, time_end):
    n = len(cost)
    selected = [False] * n
    best_cost = [10**18] * n
    opens = []

    for _ in range(p):
        if time.time() >= time_end:
            break

        best_j = -1
        best_gain = -1

        for j in range(n):
            if selected[j]:
                continue
            gain = 0
            for i in range(n):
                cj = cost[i][j]
                bc = best_cost[i]
                if cj < bc:
                    gain += (bc - cj)
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1:
            rem = [j for j in range(n) if not selected[j]]
            best_j = rem[rng.randrange(len(rem))]

        selected[best_j] = True
        opens.append(best_j)
        for i in range(n):
            cj = cost[i][best_j]
            if cj < best_cost[i]:
                best_cost[i] = cj

    if len(opens) < p:
        rem = [j for j in range(n) if not selected[j]]
        rng.shuffle(rem)
        opens.extend(rem[: (p - len(opens))])

    return opens


def local_search_swaps(cost, opens, p, rng, time_end, current_obj, logger, best_logged_ref,
                       max_iter=40, sample_closed_per_open=50):
    n = len(cost)
    open_set = set(opens)

    for _ in range(max_iter):
        if time.time() >= time_end:
            break

        open_list = list(open_set)
        closed_list = [j for j in range(n) if j not in open_set]
        if not closed_list:
            break

        # Best / second-best wrt current open_set
        best_cost = [0] * n
        second_cost = [10**18] * n
        assigned_fac = [open_list[0]] * n

        for i in range(n):
            b1 = 10**18
            b2 = 10**18
            f1 = open_list[0]
            for f in open_list:
                c = cost[i][f]
                if c < b1:
                    b2 = b1
                    b1 = c
                    f1 = f
                elif c < b2:
                    b2 = c
            best_cost[i] = b1
            second_cost[i] = b2
            assigned_fac[i] = f1

        improved = False
        rng.shuffle(open_list)

        for a in open_list:
            if time.time() >= time_end:
                break

            if len(closed_list) > sample_closed_per_open:
                cand_b = rng.sample(closed_list, sample_closed_per_open)
            else:
                cand_b = closed_list[:]
                rng.shuffle(cand_b)

            for b in cand_b:
                if time.time() >= time_end:
                    break

                new_total = 0
                for i in range(n):
                    base = second_cost[i] if assigned_fac[i] == a else best_cost[i]
                    cb = cost[i][b]
                    new_total += cb if cb < base else base
                    if new_total >= current_obj:
                        break

                if new_total < current_obj:
                    open_set.remove(a)
                    open_set.add(b)
                    current_obj = new_total
                    improved = True
                    if current_obj + 1e-9 < best_logged_ref[0]:
                        best_logged_ref[0] = float(current_obj)
                        if logger:
                            logger.log(int(current_obj))
                    break
            if improved:
                break

        if not improved:
            break

    res = list(open_set)
    if len(res) > p:
        res = res[:p]
    elif len(res) < p:
        for j in range(n):
            if j not in open_set:
                res.append(j)
                if len(res) == p:
                    break

    return res, current_obj


def choose_candidates(cost, p, heuristic_opens, max_full_vars=250000):
    n = len(cost)
    if n * n <= max_full_vars:
        return list(range(n))

    cand_limit = min(n, max(120, p + 20))
    scores = []
    for j in range(n):
        s = 0
        for i in range(n):
            s += cost[i][j]
        scores.append((s, j))
    scores.sort()

    cands = set(j for _, j in scores[:cand_limit])
    for j in heuristic_opens:
        cands.add(j)

    if len(cands) > cand_limit:
        keep = set(heuristic_opens)
        for _, j in scores:
            if len(keep) >= cand_limit:
                break
            keep.add(j)
        cands = keep

    return sorted(cands)


def solve_mip(cost, p, candidates, warm_opens, logger, best_logged_ref, time_limit):
    if not HAS_GUROBI:
        return None

    n = len(cost)
    m = len(candidates)
    if m < p or time_limit <= 0.05:
        return None

    cand_pos = {candidates[j]: j for j in range(m)}

    max_x_vars = 400000
    use_restricted = (n * m > max_x_vars)
    allowed = [[] for _ in range(n)]

    if use_restricted:
        k = max(1, min(m, max(10, max_x_vars // max(1, n))))
        for i in range(n):
            arr = [(cost[i][candidates[j]], j) for j in range(m)]
            arr.sort(key=lambda t: t[0])
            allowed[i] = [j for _, j in arr[:k]]
    else:
        full = list(range(m))
        for i in range(n):
            allowed[i] = full

    model = gp.Model("p_median")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.05, float(time_limit))
    model.Params.MIPGap = 1e-4

    y = model.addVars(m, vtype=GRB.BINARY, name="y")
    x = {}
    for i in range(n):
        for j in allowed[i]:
            x[(i, j)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    model.addConstr(gp.quicksum(y[j] for j in range(m)) == p, name="open_p")

    for i in range(n):
        model.addConstr(gp.quicksum(x[(i, j)] for j in allowed[i]) == 1.0, name=f"assign_{i}")
        for j in allowed[i]:
            model.addConstr(x[(i, j)] <= y[j], name=f"link_{i}_{j}")

    model.setObjective(
        gp.quicksum(cost[i][candidates[j]] * x[(i, j)] for i in range(n) for j in allowed[i]),
        GRB.MINIMIZE,
    )

    warm_set = set(warm_opens)
    for j in range(m):
        y[j].Start = 1.0 if candidates[j] in warm_set else 0.0

    warm_inside = [f for f in warm_opens if f in cand_pos]
    if warm_inside:
        for i in range(n):
            best_f = warm_inside[0]
            best_c = cost[i][best_f]
            for f in warm_inside[1:]:
                c = cost[i][f]
                if c < best_c:
                    best_c = c
                    best_f = f
            jj = cand_pos[best_f]
            if jj in allowed[i]:
                x[(i, jj)].Start = 1.0

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < best_logged_ref[0]:
                best_logged_ref[0] = float(obj)
                if logger:
                    logger.log(int(round(obj)))

    model.optimize(cb)

    if model.SolCount == 0:
        return None

    yvals = [y[j].X for j in range(m)]
    open_idx = [j for j in range(m) if yvals[j] > 0.5]
    if len(open_idx) != p:
        order = sorted(range(m), key=lambda j: yvals[j], reverse=True)
        open_idx = order[:p]

    opens = [candidates[j] for j in open_idx]
    obj, assigned_fac, assigned_cost = compute_assignment(cost, opens)
    return obj, opens, assigned_fac, assigned_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    p = int(instance["p"])
    cost = parse_cost_matrix(instance, n)

    start = time.time()
    total_limit = max(1, int(args.time_limit))
    deadline = start + total_limit

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    # Heuristic phase
    heur_time = min(20.0, max(1.0, 0.25 * total_limit))
    heur_deadline = min(deadline - 0.1, time.time() + heur_time)

    opens = greedy_initial_solution(cost, p, rng, heur_deadline)
    best_obj, best_assign_fac, best_assign_cost = compute_assignment(cost, opens)
    best_opens = opens[:]

    best_logged_ref = [float(best_obj)]
    if logger:
        logger.log(int(best_obj))

    improved_opens, improved_obj = local_search_swaps(
        cost=cost,
        opens=best_opens,
        p=p,
        rng=rng,
        time_end=heur_deadline,
        current_obj=best_obj,
        logger=logger,
        best_logged_ref=best_logged_ref,
        max_iter=40,
        sample_closed_per_open=50,
    )

    if improved_obj < best_obj:
        best_opens = improved_opens
        best_obj, best_assign_fac, best_assign_cost = compute_assignment(cost, best_opens)

    # Optional MIP polish (only if gurobi exists)
    remaining = deadline - time.time()
    if HAS_GUROBI and remaining > 0.5:
        candidates = choose_candidates(cost, p, best_opens)
        mip_res = solve_mip(
            cost=cost,
            p=p,
            candidates=candidates,
            warm_opens=best_opens,
            logger=logger,
            best_logged_ref=best_logged_ref,
            time_limit=max(0.05, deadline - time.time() - 0.05),
        )
        if mip_res is not None:
            obj_m, opens_m, fac_m, cst_m = mip_res
            if obj_m < best_obj:
                best_obj = obj_m
                best_opens = opens_m
                best_assign_fac = fac_m
                best_assign_cost = cst_m
                if best_obj + 1e-9 < best_logged_ref[0]:
                    best_logged_ref[0] = float(best_obj)
                    if logger:
                        logger.log(int(best_obj))

    assignments = []
    for i in range(n):
        assignments.append(
            {
                "customer": int(i),
                "facility": int(best_assign_fac[i]),
                "cost": int(best_assign_cost[i]),
            }
        )

    solution = {
        "objective_value": int(best_obj),
        "opened_facilities": [int(x) for x in best_opens],
        "assignments": assignments,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()