import argparse
import json
import time
import random
import sys
from solution_logger import SolutionLogger

# Try to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False


def evaluate(S, weight_dict):
    """Compute total edge weight of a clique S (set of vertex indices)."""
    S_list = list(S)
    total = 0.0
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            u, v = S_list[i], S_list[j]
            if u > v:
                u, v = v, u
            total += weight_dict.get((u, v), 0.0)
    return total


def greedy_from_start(start, adj, weight_dict):
    """Greedy expansion starting from a single vertex."""
    S = {start}
    candidates = set(adj[start])
    while candidates:
        best_v = None
        best_gain = -1.0
        for v in candidates:
            gain = 0.0
            for u in S:
                if u < v:
                    gain += weight_dict.get((u, v), 0.0)
                else:
                    gain += weight_dict.get((v, u), 0.0)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        S.add(best_v)
        candidates = candidates & adj[best_v]
    return S


def local_search(S, adj, weight_dict, n, max_iters=200):
    """Improve a clique using add, swap, and remove+greedy moves."""
    S = set(S)
    current_weight = evaluate(S, weight_dict)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1

        # 1. Try to add any vertex that is connected to all of S
        for v in range(n):
            if v not in S and S <= adj[v]:
                gain = 0.0
                for u in S:
                    if u < v:
                        gain += weight_dict.get((u, v), 0.0)
                    else:
                        gain += weight_dict.get((v, u), 0.0)
                S.add(v)
                current_weight += gain
                improved = True
                break
        if improved:
            continue

        # 2. Try to swap: remove one vertex, add another
        for v in range(n):
            if v in S:
                continue
            blocked = [u for u in S if u not in adj[v]]
            if len(blocked) == 1:
                u = blocked[0]
                gain = 0.0
                for w in S:
                    if w == u:
                        continue
                    if w < v:
                        gain += weight_dict.get((w, v), 0.0)
                    else:
                        gain += weight_dict.get((v, w), 0.0)
                    if w < u:
                        gain -= weight_dict.get((w, u), 0.0)
                    else:
                        gain -= weight_dict.get((u, w), 0.0)
                if gain > 1e-6:
                    S.remove(u)
                    S.add(v)
                    current_weight += gain
                    improved = True
                    break
        if improved:
            continue

        # 3. Try to remove one vertex and then greedily expand
        for u in list(S):
            S_without = S - {u}
            temp_S = set(S_without)
            temp_candidates = [v for v in range(n) if v not in temp_S and temp_S <= adj[v]]
            while temp_candidates:
                best_v = None
                best_gain = -1.0
                for v in temp_candidates:
                    gain = 0.0
                    for w in temp_S:
                        if w < v:
                            gain += weight_dict.get((w, v), 0.0)
                        else:
                            gain += weight_dict.get((v, w), 0.0)
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                temp_S.add(best_v)
                temp_candidates = [v for v in temp_candidates if v != best_v and v in adj[best_v]]
            temp_weight = evaluate(temp_S, weight_dict)
            if temp_weight > current_weight + 1e-6:
                S = temp_S
                current_weight = temp_weight
                improved = True
                break
    return S, current_weight


def solve_mip(n, adj, weight_dict, time_limit, initial_solution=None, use_lazy=True):
    """Solve using Gurobi with the given time limit."""
    if not HAS_GUROBI or n == 0:
        return None

    model = gp.Model("max_weight_clique")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    edge_list = list(weight_dict.keys())
    y = model.addVars(len(edge_list), lb=0, ub=1, name="y")
    model.setObjective(gp.quicksum(weight_dict[edge_list[e]] * y[e] for e in range(len(edge_list))), GRB.MAXIMIZE)

    for e, (i, j) in enumerate(edge_list):
        model.addConstr(y[e] <= x[i])
        model.addConstr(y[e] <= x[j])

    if initial_solution:
        for i in range(n):
            x[i].Start = 1.0 if i in initial_solution else 0.0
        for e, (i, j) in enumerate(edge_list):
            y[e].Start = 1.0 if (i in initial_solution and j in initial_solution) else 0.0

    if use_lazy:
        model.Params.LazyConstraints = 1
        model._x = x
        model._adj = adj
        model._n = n

        def lazy_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                x_vals = model.cbGetSolution(model._x)
                S = [i for i in range(model._n) if x_vals[i] > 0.5]
                count = 0
                for i_idx in range(len(S)):
                    for j_idx in range(i_idx + 1, len(S)):
                        i, j = S[i_idx], S[j_idx]
                        if j not in model._adj[i]:
                            model.cbLazy(model._x[i] + model._x[j] <= 1)
                            count += 1
                            if count >= 1000:
                                return
        model.optimize(lazy_callback)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    model.addConstr(x[i] + x[j] <= 1)
        model.optimize()

    if model.SolCount > 0:
        x_vals = model.getAttr('X', x)
        S = [i for i in range(n) if x_vals[i] > 0.5]
        obj_val = model.ObjVal
        return S, obj_val
    return None


def main():
    parser = argparse.ArgumentParser(description="Maximum Weight Clique")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    vertices = data['vertices']
    n = len(vertices)
    idx_to_label = {i: v for i, v in enumerate(vertices)}
    label_to_idx = {v: i for i, v in enumerate(vertices)}

    adj = [set() for _ in range(n)]
    weight_dict = {}
    for e in data['edges']:
        i = label_to_idx[e['i']]
        j = label_to_idx[e['j']]
        w = float(e['weight'])
        if i > j:
            i, j = j, i
        weight_dict[(i, j)] = w
        adj[i].add(j)
        adj[j].add(i)

    start_time = time.time()
    time_limit = args.time_limit

    best_clique = set()
    best_weight = 0.0

    # ---- Heuristic phase ----
    random.seed(0)
    if n == 0:
        solution = {"objective_value": 0.0, "clique": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0.0, solution)
        return

    # Choose starting vertices for greedy
    if n <= 1000:
        starts = list(range(n))
    else:
        starts = random.sample(range(n), 500)

    heuristic_time = min(time_limit * 0.6, time_limit - 1)
    if heuristic_time < 0:
        heuristic_time = 0

    for start in starts:
        if time.time() - start_time > heuristic_time:
            break
        S = greedy_from_start(start, adj, weight_dict)
        S, w = local_search(S, adj, weight_dict, n, max_iters=100)
        if w > best_weight + 1e-9:
            best_weight = w
            best_clique = S
            if logger:
                solution_dict = {
                    "objective_value": float(best_weight),
                    "clique": [idx_to_label[i] for i in sorted(best_clique)]
                }
                logger.log_solution(best_weight, solution_dict)

    # Iterated local search / random walk for remaining heuristic time
    if best_clique:
        current_S = set(best_clique)
        current_w = best_weight
        while time.time() - start_time < heuristic_time:
            # Perturb: remove 1-2 random vertices, add 1-2 random valid ones
            perturbed = set(current_S)
            num_remove = min(2, len(perturbed))
            for _ in range(num_remove):
                if perturbed:
                    u = random.choice(list(perturbed))
                    perturbed.remove(u)
            candidates = [v for v in range(n) if v not in perturbed and perturbed <= adj[v]]
            random.shuffle(candidates)
            for v in candidates[:2]:
                if perturbed <= adj[v]:
                    perturbed.add(v)
            new_S, new_w = local_search(perturbed, adj, weight_dict, n, max_iters=50)
            if new_w > best_weight + 1e-9:
                best_weight = new_w
                best_clique = new_S
                if logger:
                    solution_dict = {
                        "objective_value": float(best_weight),
                        "clique": [idx_to_label[i] for i in sorted(best_clique)]
                    }
                    logger.log_solution(best_weight, solution_dict)
                current_S = new_S
                current_w = new_w
            elif new_w > current_w * 0.95:
                current_S = new_S
                current_w = new_w

    # ---- Gurobi phase ----
    gurobi_time = time_limit - (time.time() - start_time)
    if HAS_GUROBI and n <= 2000 and gurobi_time > 2:
        total_possible = n * (n - 1) // 2
        num_non_edges = total_possible - len(weight_dict)
        use_lazy = num_non_edges > 100000 or n > 500

        init_S = list(best_clique) if best_clique else None
        res = solve_mip(n, adj, weight_dict, gurobi_time, initial_solution=init_S, use_lazy=use_lazy)
        if res is not None:
            S_mip, w_mip = res
            if w_mip > best_weight + 1e-6:
                best_weight = float(w_mip)
                best_clique = set(S_mip)
                if logger:
                    solution_dict = {
                        "objective_value": best_weight,
                        "clique": [idx_to_label[i] for i in sorted(best_clique)]
                    }
                    logger.log_solution(best_weight, solution_dict)

    # ---- Write final solution ----
    solution = {
        "objective_value": float(best_weight),
        "clique": [idx_to_label[i] for i in sorted(best_clique)]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(best_weight, solution)


if __name__ == "__main__":
    main()