import json
import time
import argparse
import sys

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Solve Minimum Power Spanning Tree (MPST) Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, required=False, help="Path for logging intermediate solutions.")
    args = parser.parse_args()

    # Initialize Logger
    logger = None
    if args.log_path:
        try:
            import os
            sys.path.append(os.getcwd())
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    vertices = data['vertices']
    
    if n == 1:
        # Base case for trivial single-node graph
        solution_dict = {
            "objective_value": 0.0,
            "selected_edges": [],
            "vertex_costs": {str(vertices[0]['id']): 0.0}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)
        return

    # Precompute squared Euclidean distance matrix
    # Based on the problem description, these represent power costs and must function generically 
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = vertices[i]['x'], vertices[i]['y']
        for j in range(i + 1, n):
            xj, yj = vertices[j]['x'], vertices[j]['y']
            w[i][j] = float((xi - xj)**2 + (yi - yj)**2)
            w[j][i] = w[i][j]

    # Heuristic 1: Kruskal's Algorithm
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((w[i][j], i, j))
    edges.sort(key=lambda x: x[0])

    parent = list(range(n))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    mst_edges = set()
    for weight, i, j in edges:
        if union(i, j):
            mst_edges.add((i, j) if i < j else (j, i))
            if len(mst_edges) == n - 1:
                break

    def eval_cost(tree_edges):
        adj_eval = {i: [] for i in range(n)}
        for u, v in tree_edges:
            adj_eval[u].append(w[u][v])
            adj_eval[v].append(w[u][v])
        return sum(max(adj_eval[i]) if adj_eval[i] else 0.0 for i in range(n))

    best_edges = set(mst_edges)
    best_cost = eval_cost(best_edges)
    
    if logger:
        logger.log(best_cost)

    # Heuristic 2: Fast Local Search (Edge Swap)
    adj = {i: {} for i in range(n)}
    for u, v in best_edges:
        adj[u][v] = w[u][v]
        adj[v][u] = w[u][v]

    def get_max(node):
        return max(adj[node].values()) if adj[node] else 0.0

    max_val = {i: get_max(i) for i in range(n)}
    current_cost = best_cost

    improved = True
    ls_start = time.time()
    
    # Cap local search at smaller of 3 secs or 20% of time limit
    ls_time_limit = min(3.0, max(1.0, args.time_limit * 0.2)) 
    
    while improved and time.time() - ls_start < ls_time_limit:
        improved = False
        
        for u, v in list(best_edges):
            # Temporarily remove (u, v)
            old_wu, old_wv = adj[u].pop(v), adj[v].pop(u)
            u_old, v_old = max_val[u], max_val[v]
            max_val[u], max_val[v] = get_max(u), get_max(v)
            
            # Find disjoint components C1 and C2 dynamically
            C1 = {u}
            q = [u]
            while q:
                curr = q.pop()
                for nbr in adj[curr]:
                    if nbr not in C1:
                        C1.add(nbr)
                        q.append(nbr)
            C2 = set(range(n)) - C1
            
            best_diff = 0.0
            best_xy = None
            
            base_diff = max_val[u] - u_old + max_val[v] - v_old
            
            # Evaluate all possible reconnection links between components
            for x in C1:
                mx_x = max_val[x]
                wx = w[x]
                for y in C2:
                    wxy = wx[y]
                    nx = wxy if wxy > mx_x else mx_x
                    ny = wxy if wxy > max_val[y] else max_val[y]
                    
                    diff = base_diff + nx - mx_x + ny - max_val[y]
                    if diff < best_diff:
                        best_diff = diff
                        best_xy = (x, y)
                        
            if best_xy:
                x, y = best_xy
                # Apply strictly improving swap
                best_edges.remove((u, v))
                best_edges.add((x, y) if x < y else (y, x))
                adj[x][y] = w[x][y]
                adj[y][x] = w[x][y]
                max_val[u] = get_max(u)
                max_val[v] = get_max(v)
                max_val[x] = get_max(x)
                max_val[y] = get_max(y)
                current_cost = sum(max_val.values())
                improved = True
                
                if current_cost < best_cost - 1e-5:
                    best_cost = current_cost
                    if logger:
                        logger.log(best_cost)
                break 
            else:
                # Restore removed edge (u, v)
                adj[u][v], adj[v][u] = old_wu, old_wv
                max_val[u], max_val[v] = u_old, v_old

    UB = best_cost
    
    # Prune edges aggressively by mathematical lower bounding theory
    M = [0.0] * n
    for i in range(n):
        M[i] = min((w[i][j] for j in range(n) if i != j), default=0.0)
    base_cost = sum(M)

    candidate_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            # Rigorous proof-derived bound: global tree cost min threshold via element minimums
            if 2 * w[i][j] + base_cost - M[i] - M[j] <= UB + 1e-5:
                candidate_edges.add((i, j))

    # Protect the active incumbent structure natively inside candidate edges
    for u, v in best_edges:
        candidate_edges.add((u, v) if u < v else (v, u))
    
    candidate_edges = list(candidate_edges)

    # Exact MILP via single-core Gurobi solver bounds
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        time_left = args.time_limit - (time.time() - start_time)
        if time_left < 2.0:
            raise Exception("Insufficient remaining time for exact exactitude via Gurobi.")
            
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        
        model = gp.Model("MPSC", env=env)
        model.setParam('TimeLimit', max(1.0, time_left - 1.0))
        model.setParam('LazyConstraints', 1)
        model.setParam('Threads', 1)
        
        x = {}
        for i, j in candidate_edges:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
        p = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
        
        # Fundamental tree connectivity sizing constraint
        model.addConstr(gp.quicksum(x[i, j] for i, j in candidate_edges) == n - 1)
        
        # Force continuous maximum bounds locally upon elements
        adj_c = {i: [] for i in range(n)}
        for i, j in candidate_edges:
            adj_c[i].append((j, x[i, j]))
            adj_c[j].append((i, x[i, j]))
            
        for i in range(n):
            for j, var in adj_c[i]:
                model.addConstr(p[i] >= w[i][j] * var)
                
        model.setObjective(gp.quicksum(p[i] for i in range(n)), GRB.MINIMIZE)
        
        # Pass heuristically optimized tree as MIPStart bounds
        for i, j in candidate_edges:
            if (i, j) in best_edges or (j, i) in best_edges:
                x[i, j].Start = 1.0
            else:
                x[i, j].Start = 0.0
                
        model._best_obj = UB
        model._candidate_edges = candidate_edges
        model._n = n
        model._x = x
        model._logger = logger
        
        def subtour_elim(model, where):
            if where == GRB.Callback.MIPSOL:
                x_vals = model.cbGetSolution(model._x)
                selected = [(i, j) for (i, j) in model._candidate_edges if x_vals[i, j] > 0.5]
                
                n_ = model._n
                adj_m = {i: [] for i in range(n_)}
                for u, v in selected:
                    adj_m[u].append(v)
                    adj_m[v].append(u)
                    
                visited = set()
                components = []
                for i in range(n_):
                    if i not in visited:
                        comp = []
                        q = [i]
                        visited.add(i)
                        while q:
                            curr = q.pop()
                            comp.append(curr)
                            for nbr in adj_m[curr]:
                                if nbr not in visited:
                                    visited.add(nbr)
                                    q.append(nbr)
                        components.append(comp)
                        
                if len(components) > 1:
                    for comp in components:
                        if len(comp) < n_:
                            comp_set = frozenset(comp)
                            edges_in_comp = [(i, j) for (i, j) in model._candidate_edges if i in comp_set and j in comp_set]
                            sel_count = sum(1 for (i, j) in selected if i in comp_set and j in comp_set)
                            if sel_count > len(comp) - 1:
                                model.cbLazy(gp.quicksum(model._x[i, j] for i, j in edges_in_comp) <= len(comp) - 1)
                else:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if model._logger and obj < model._best_obj - 1e-5:
                        model._best_obj = obj
                        model._logger.log(obj)
                        
        model.optimize(subtour_elim)
        
        if model.SolCount > 0:
            best_edges = []
            for i, j in candidate_edges:
                if x[i, j].X > 0.5:
                    best_edges.append((i, j) if i < j else (j, i))
                        
    except Exception as e:
        pass # Rely gracefully onto thoroughly-groomed heuristics

    # Format output natively aligned with explicit schema protocols
    final_adj = {i: [] for i in range(n)}
    for u, v in best_edges:
        final_adj[u].append(w[u][v])
        final_adj[v].append(w[u][v])
        
    vertex_costs = {}
    total_cost = 0.0
    for i in range(n):
        cost = float(max(final_adj[i])) if final_adj[i] else 0.0
        vertex_costs[str(vertices[i]['id'])] = cost
        total_cost += cost
        
    mapped_edges = [[int(vertices[u]['id']), int(vertices[v]['id'])] for u, v in best_edges]
        
    solution_dict = {
        "objective_value": total_cost,
        "selected_edges": mapped_edges,
        "vertex_costs": vertex_costs
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()