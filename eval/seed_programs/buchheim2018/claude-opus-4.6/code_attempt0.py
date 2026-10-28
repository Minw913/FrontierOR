import argparse
import json
import time
import itertools
from collections import defaultdict
import numpy as np

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    source = instance['source_node']
    target = instance['target_node']
    arcs = instance['arcs']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']
    nodes_info = instance['nodes']
    
    # Build adjacency structures
    arc_from = [a['from_node'] for a in arcs]
    arc_to = [a['to_node'] for a in arcs]
    
    outgoing = defaultdict(list)  # node -> list of arc indices
    incoming = defaultdict(list)  # node -> list of arc indices
    
    for i, a in enumerate(arcs):
        outgoing[a['from_node']].append(i)
        incoming[a['to_node']].append(i)
    
    all_nodes = set(n['id'] for n in nodes_info)
    
    # For small instances, try enumeration of all paths
    # For larger instances, use Gurobi MIQP
    
    def compute_objective(selected_arcs):
        """Compute objective for a set of selected arc indices."""
        obj = 0.0
        for i in selected_arcs:
            obj += linear_costs[i]
        for i in selected_arcs:
            for j in selected_arcs:
                obj += quadratic_costs[i][j]
        return obj
    
    best_obj = float('inf')
    best_path_arcs = None
    
    def update_best(arc_indices):
        nonlocal best_obj, best_path_arcs
        obj = compute_objective(arc_indices)
        if obj < best_obj:
            best_obj = obj
            best_path_arcs = arc_indices
            if logger:
                logger.log(obj)
    
    # Try path enumeration via DFS for small instances
    # Enumerate all simple paths from source to target
    if num_nodes <= 100 and num_arcs <= 500:
        # DFS enumeration with pruning
        visited = set()
        current_path = []  # list of arc indices
        
        path_count = [0]
        max_paths = 1000000  # limit enumeration
        
        def dfs(node):
            if time.time() - start_time > time_limit * 0.9:
                return
            if path_count[0] > max_paths:
                return
            if node == target:
                path_count[0] += 1
                update_best(list(current_path))
                return
            for arc_idx in outgoing[node]:
                next_node = arc_to[arc_idx]
                if next_node not in visited:
                    visited.add(next_node)
                    current_path.append(arc_idx)
                    dfs(next_node)
                    current_path.pop()
                    visited.discard(next_node)
        
        visited.add(source)
        dfs(source)
        
        # If we enumerated all paths, we have optimal
        if path_count[0] <= max_paths and time.time() - start_time <= time_limit * 0.9:
            # We found optimal
            if best_path_arcs is not None:
                solution = {
                    "objective_value": best_obj,
                    "solution_arcs": [
                        {
                            "id": arcs[i]['id'],
                            "from_node": arcs[i]['from_node'],
                            "to_node": arcs[i]['to_node']
                        }
                        for i in best_path_arcs
                    ]
                }
                with open(solution_path, 'w') as f:
                    json.dump(solution, f)
                return
    
    # If enumeration didn't complete or instance is large, use Gurobi MIQP
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 1:
        remaining_time = 1
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("qspp")
        model.setParam('TimeLimit', max(1, remaining_time - 2))
        model.setParam('Threads', 1)
        model.setParam('OutputFlag', 1)
        
        # Binary variables for each arc
        x = model.addVars(num_arcs, vtype=GRB.BINARY, name="x")
        
        # Flow conservation constraints
        # Source: exactly one arc leaving
        model.addConstr(
            gp.quicksum(x[i] for i in outgoing[source]) == 1,
            name="source_out"
        )
        # No arc entering source (optional but helps)
        if incoming[source]:
            model.addConstr(
                gp.quicksum(x[i] for i in incoming[source]) == 0,
                name="source_in"
            )
        
        # Target: exactly one arc entering
        model.addConstr(
            gp.quicksum(x[i] for i in incoming[target]) == 1,
            name="target_in"
        )
        # No arc leaving target
        if outgoing[target]:
            model.addConstr(
                gp.quicksum(x[i] for i in outgoing[target]) == 0,
                name="target_out"
            )
        
        # Intermediate nodes: flow conservation
        for node in all_nodes:
            if node == source or node == target:
                continue
            in_arcs = incoming[node]
            out_arcs = outgoing[node]
            if in_arcs or out_arcs:
                model.addConstr(
                    gp.quicksum(x[i] for i in in_arcs) == gp.quicksum(x[i] for i in out_arcs),
                    name=f"flow_{node}"
                )
        
        # Objective: linear + quadratic
        # Linear part
        lin_expr = gp.quicksum(linear_costs[i] * x[i] for i in range(num_arcs))
        
        # Quadratic part: sum over all i,j of Q[i][j] * x[i] * x[j]
        # Since Q is symmetric, we can write it as:
        # sum_i Q[i][i] * x[i]^2 + 2 * sum_{i<j} Q[i][j] * x[i] * x[j]
        # But since x[i] is binary, x[i]^2 = x[i], so diagonal terms become linear
        
        Q = quadratic_costs
        
        # Add diagonal quadratic costs as linear terms (since x_i^2 = x_i for binary)
        diag_expr = gp.quicksum(Q[i][i] * x[i] for i in range(num_arcs))
        
        # Off-diagonal quadratic terms
        quad_expr = gp.QuadExpr()
        for i in range(num_arcs):
            for j in range(i+1, num_arcs):
                if Q[i][j] != 0:
                    quad_expr.add(x[i] * x[j], 2 * Q[i][j])
        
        model.setObjective(lin_expr + diag_expr + quad_expr, GRB.MINIMIZE)
        
        # Provide warm start if we have one from enumeration
        if best_path_arcs is not None:
            for i in range(num_arcs):
                x[i].Start = 1 if i in best_path_arcs else 0
        
        # Callback to log incumbents
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                nonlocal best_obj, best_path_arcs
                if obj < best_obj:
                    best_obj = obj
                    vals = model.cbGetSolution(x)
                    best_path_arcs = [i for i in range(num_arcs) if vals[i] > 0.5]
                    if logger:
                        logger.log(obj)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj_val = model.ObjVal
            selected = [i for i in range(num_arcs) if x[i].X > 0.5]
            if obj_val < best_obj:
                best_obj = obj_val
                best_path_arcs = selected
                if logger:
                    logger.log(obj_val)
    
    except Exception as e:
        print(f"Gurobi error: {e}")
    
    # Output solution
    if best_path_arcs is not None:
        solution = {
            "objective_value": best_obj,
            "solution_arcs": [
                {
                    "id": arcs[i]['id'],
                    "from_node": arcs[i]['from_node'],
                    "to_node": arcs[i]['to_node']
                }
                for i in best_path_arcs
            ]
        }
    else:
        solution = {
            "objective_value": 0.0,
            "solution_arcs": []
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)