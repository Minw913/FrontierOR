import argparse
import json
import itertools
import time
import sys


def solve_tsp(instance, args, logger):
    """Solve TSP. For small n (<=10), brute force. For larger n, use Gurobi with MTZ."""
    n = instance['parameters']['n_cities']
    dist = instance['data']['distance_matrix']
    
    if n <= 10:
        best_tour = None
        best_cost = float('inf')
        solution = {}
        for perm in itertools.permutations(range(n)):
            cost = sum(dist[perm[i]][perm[(i + 1) % n]] for i in range(n))
            if cost < best_cost:
                best_cost = cost
                best_tour = list(perm)
                solution = {}
                for city in range(n):
                    for t in range(n):
                        solution[f'x_{city}_{t}'] = 1 if best_tour[t] == city else 0
                if logger:
                    logger.log_solution(best_cost, solution)
        
        return best_cost, solution, best_tour
    else:
        import gurobipy as gp
        from gurobipy import GRB
        
        env = gp.Env(empty=True)
        env.setParam('Seed', 0)
        env.setParam('Threads', 1)
        env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.Model(env=env)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('TimeLimit', max(1, args.time_limit - 1))
        
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
        
        model.setObjective(
            gp.quicksum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j),
            GRB.MINIMIZE
        )
        
        # MTZ subtour elimination
        u = {}
        for i in range(1, n):
            u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n - 1, name=f'u_{i}')
        
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
        
        x_flat = [x[i, j] for i in range(n) for j in range(n) if i != j]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                vals = model.cbGetSolution(x_flat)
                val_map = {}
                idx = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            val_map[(i, j)] = vals[idx]
                            idx += 1
                # Extract tour
                tour = []
                current = 0
                visited = set()
                for _ in range(n):
                    tour.append(current)
                    visited.add(current)
                    nxt = -1
                    for j in range(n):
                        if j != current and j not in visited and val_map.get((current, j), 0) > 0.5:
                            nxt = j
                            break
                    if nxt == -1:
                        for j in range(n):
                            if j != current and j not in visited and val_map.get((current, j), 0) > 0.5:
                                nxt = j
                                break
                    if nxt == -1:
                        break
                    current = nxt
                sol = {}
                for city in range(n):
                    for t in range(n):
                        sol[f'x_{city}_{t}'] = 1 if t < len(tour) and tour[t] == city else 0
                if logger:
                    logger.log_solution(obj_val, sol)
        
        model._x_flat = x_flat
        model._n = n
        model._logger = logger
        model.optimize(callback)
        
        # Extract final tour
        tour = []
        current = 0
        visited = set()
        for _ in range(n):
            tour.append(current)
            visited.add(current)
            nxt = -1
            for j in range(n):
                if j != current and j not in visited:
                    if x[current, j].X > 0.5:
                        nxt = j
                        break
            if nxt == -1:
                break
            current = nxt
        
        obj = model.ObjVal
        solution = {}
        for city in range(n):
            for t in range(n):
                solution[f'x_{city}_{t}'] = 1 if t < len(tour) and tour[t] == city else 0
        
        if logger:
            logger.log_solution(obj, solution)
        
        return obj, solution, tour


def solve_maxcut(instance, args, logger):
    """Solve MaxCut using Gurobi MIP."""
    import gurobipy as gp
    from gurobipy import GRB
    
    # Try to find graph data in various locations
    data = instance.get('data', instance)
    
    # Detect number of nodes
    if 'n_nodes' in instance:
        n = instance['n_nodes']
    elif 'n_nodes' in data:
        n = data['n_nodes']
    elif 'num_nodes' in instance:
        n = instance['num_nodes']
    elif 'num_nodes' in data:
        n = data['num_nodes']
    elif 'nodes' in data:
        nodes_data = data['nodes']
        if isinstance(nodes_data, int):
            n = nodes_data
        else:
            n = len(nodes_data)
    elif 'nodes' in instance:
        nodes_data = instance['nodes']
        if isinstance(nodes_data, int):
            n = nodes_data
        else:
            n = len(nodes_data)
    else:
        # Infer from edges
        edges = data.get('edges', instance.get('edges', []))
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
    
    # Detect edges
    edges = data.get('edges', instance.get('edges', []))
    
    # Normalize edges to list of (u, v) pairs
    edge_list = []
    for e in edges:
        if isinstance(e, (list, tuple)):
            edge_list.append((int(e[0]), int(e[1])))
        elif isinstance(e, dict):
            u = e.get('u', e.get('source', e.get('from')))
            v = e.get('v', e.get('target', e.get('to')))
            edge_list.append((int(u), int(v)))
    
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('OutputFlag', 0)
    env.start()
    
    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('TimeLimit', max(1, args.time_limit - 1))
    
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}')
    
    # Linearize: cut edge (u,v) = x_u + x_v - 2*z_uv where z_uv = x_u * x_v
    # For maximization, z can be continuous
    z = {}
    for idx, (u, v) in enumerate(edge_list):
        z[idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f'z_{idx}')
        model.addConstr(z[idx] <= x[u])
        model.addConstr(z[idx] <= x[v])
        model.addConstr(z[idx] >= x[u] + x[v] - 1)
    
    model.setObjective(
        gp.quicksum(x[edge_list[idx][0]] + x[edge_list[idx][1]] - 2 * z[idx]
                     for idx in range(len(edge_list))),
        GRB.MAXIMIZE
    )
    
    x_vars = [x[i] for i in range(n)]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            vals = model.cbGetSolution(x_vars)
            sol = {}
            for i in range(n):
                sol[f'x_{i}'] = int(round(vals[i]))
            if logger:
                logger.log_solution(obj_val, sol)
    
    model._x_vars = x_vars
    model._n = n
    model._logger = logger
    model.optimize(callback)
    
    obj = model.ObjVal
    solution = {}
    for i in range(n):
        solution[f'x_{i}'] = int(round(x[i].X))
    
    if logger:
        logger.log_solution(obj, solution)
    
    return obj, solution, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    from solution_logger import SolutionLogger
    
    params = instance.get('parameters', instance)
    
    if 'n_cities' in params:
        # TSP
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
        obj, solution, tour = solve_tsp(instance, args, logger)
        output = {
            'objective_value': obj,
            'solution': solution,
            'tour': tour
        }
    else:
        # MaxCut
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
        obj, solution, _ = solve_maxcut(instance, args, logger)
        output = {
            'objective_value': obj,
            'solution': solution
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()