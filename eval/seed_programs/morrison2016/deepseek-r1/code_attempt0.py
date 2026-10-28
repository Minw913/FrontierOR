import argparse
import json
import time
import math
import sys
from collections import defaultdict
from solution_logger import SolutionLogger

def greedy_coloring(n, edges):
    if n == 0:
        return 0, {}
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = e
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort vertices by degree descending
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    colors = [-1] * n
    color_flags = [False] * (n+1)  # to avoid frequent allocation

    for u in order:
        banned = set()
        for v in adj[u]:
            if colors[v] != -1:
                banned.add(colors[v])
        choose_color = 0
        while choose_color in banned:
            choose_color += 1
        colors[u] = choose_color
    
    num_colors = max(colors) + 1 if n > 0 else 0
    return num_colors, {i: colors[i] for i in range(n)}

def run_mip_coloring(n, edges, k, time_limit, initial_sol_dict=None):
    import gurobipy as gp
    try:
        with gp.Env(params={'OutputFlag': 0}) as env:
            model = gp.Model('coloring', env=env)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(0.1, time_limit))

            # Assign variable: x[v, c] for vertex v and color c
            x = model.addVars(n, k, vtype=gp.GRB.BINARY, name="x")

            # Constraint: Each vertex gets exactly one color
            for v in range(n):
                model.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1, name=f"v{v}")

            # Constraint: For each edge and each color, adjacent vertices cannot be colored the same
            constraint_count = 0
            for idx, (u, v) in enumerate(edges):
                for c in range(k):
                    model.addConstr(x[u, c] + x[v, c] <= 1, name=f"e{idx}_{c}")
                constraint_count += 1
                if constraint_count % 10000 == 0:
                    model.update()

            # Symmetry breaking: fix vertex0 to color0
            if n > 0:
                model.addConstr(x[0, 0] == 1, name="sym_fix")

            if initial_sol_dict:
                initial_sol = model.addVars(n, k, vtype=gp.GRB.BINARY, name='init_sol')
                for v in range(n):
                    c_val = initial_sol_dict.get(v, 0)
                    for col in range(k):
                        if col == c_val:
                            initial_sol[v, col].Start = 1
                        else:
                            initial_sol[v, col].Start = 0

            model.update()
            model.optimize()

            sol_found = False
            sol_dict = None
            if model.Status == gp.GRB.OPTIMAL:
                sol_found = True
            elif model.Status == gp.GRB.TIME_LIMIT and model.SolCount > 0:
                sol_found = True
            elif model.Status == gp.GRB.USER_OBJ_LIMIT and model.SolCount > 0:
                sol_found = True

            if sol_found:
                sol_dict = {}
                for v in range(n):
                    for c in range(k):
                        if x[v,c].X >= 0.5:
                            sol_dict[v] = c
                            break
                return sol_dict
            return None
    except gp.GurobiError:
        return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    graph_data = instance_data['graph']
    bounds_data = instance_data['bounds']
    n = graph_data['num_vertices']
    edges = graph_data['edges']
    lower_bound_clique = bounds_data['clique_lower_bound']
    greedy_upper_bound = bounds_data['greedy_upper_bound']

    # Initialize logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    log_first_time = True

    # If n==0, trivial solution
    if n == 0:
        solution_data = {
            'objective_value': 0,
            'selected_sets': [],
            'coloring': {}
        }
        with open(args.solution_path, 'w') as outfile:
            json.dump(solution_data, outfile)
        if logger:
            logger.log(0)
        return

    num_greedy, greedy_coloring_dict = greedy_coloring(n, edges)
    best_coloring_dict = greedy_coloring_dict
    best_k = num_greedy
    if logger and log_first_time:
        logger.log(best_k)
        log_first_time = False

    # Trust the problem's provided greedy_upper_bound and update if a lower greedy is found by us
    if num_greedy != greedy_upper_bound:
        best_k = min(num_greedy, greedy_upper_bound)

    if best_k == lower_bound_clique:
        # Optimal
        pass
    else:
        low = lower_bound_clique
        high = best_k - 1
        remaining_time = args.time_limit - (time.time() - start_time)
        if low <= high and remaining_time > 1:
            # Allocate time: we have max_log_iterations at most
            max_log_iterations = math.ceil(math.log2(high - low + 1)) + 1 if high >= low else 0
            time_per_mip = max(0.2, remaining_time / max_log_iterations) if max_log_iterations > 0 else 0
            for i in range(max_log_iterations):
                current_time_elapsed = time.time() - start_time
                if current_time_elapsed >= args.time_limit:
                    break
                if low > high:
                    break
                mid = (low + high) // 2
                time_alloc = min(args.time_limit - current_time_elapsed, time_per_mip)
                sol_dict = run_mip_coloring(n, edges, mid, time_alloc, best_coloring_dict)
                if sol_dict is not None:
                    best_coloring_dict = sol_dict
                    best_k = mid
                    high = mid - 1
                    if logger:
                        logger.log(best_k)
                else:
                    low = mid + 1

    # Prepare selected_sets: group vertices by color
    color_classes = [[] for _ in range(best_k)]
    for v in range(n):
        c = best_coloring_dict.get(v, 0)
        if c < best_k:
            color_classes[c].append(v)
    for lst in color_classes:
        lst.sort()
    
    coloring_output = {}
    for v in range(n):
        coloring_output[str(v)] = best_coloring_dict[v]

    solution_data = {
        'objective_value': best_k,
        'selected_sets': color_classes,
        'coloring': coloring_output
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_data, f)

if __name__ == "__main__":
    main()