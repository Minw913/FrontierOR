import argparse
import json
import math
import time
import itertools
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    m = instance['parameters']['m']
    n = instance['parameters']['n']
    target_coords = instance['target_nodes']['coordinates']
    steiner_coords = instance['steiner_nodes']['coordinates']
    node_weights = instance['steiner_nodes']['node_weights']
    assignment_costs = instance['cost_matrices']['assignment_costs']['matrix']
    interconnection_costs = instance['cost_matrices']['interconnection_costs']['matrix']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # For small instances, use Gurobi MIP
    # For larger instances, use a heuristic approach
    
    best_solution = None
    best_obj = float('inf')
    
    def evaluate_solution(active_set, assignments_map):
        """Evaluate objective for a given set of active steiner nodes and assignments."""
        if not active_set:
            return float('inf'), None, None, None
        
        active_list = sorted(active_set)
        
        # Assignment cost
        assign_cost = 0
        for i in range(m):
            assign_cost += assignment_costs[i][assignments_map[i]]
        
        # Node activation cost
        activation_cost = sum(node_weights[j] for j in active_list)
        
        # MST cost among active steiner nodes
        mst_edges = []
        mst_cost = 0
        if len(active_list) > 1:
            # Prim's algorithm
            in_tree = set()
            in_tree.add(active_list[0])
            min_edge = {}
            for v in active_list[1:]:
                min_edge[v] = (interconnection_costs[active_list[0]][v], active_list[0])
            
            while len(in_tree) < len(active_list):
                # Find minimum edge
                best_v = None
                best_cost_e = float('inf')
                best_from = None
                for v, (c, f) in min_edge.items():
                    if v not in in_tree and c < best_cost_e:
                        best_cost_e = c
                        best_v = v
                        best_from = f
                if best_v is None:
                    break
                in_tree.add(best_v)
                mst_edges.append([best_from, best_v])
                mst_cost += best_cost_e
                # Update min_edge
                for v in min_edge:
                    if v not in in_tree:
                        new_cost = interconnection_costs[best_v][v]
                        if new_cost < min_edge[v][0]:
                            min_edge[v] = (new_cost, best_v)
        
        total = assign_cost + mst_cost + activation_cost
        return total, active_list, assignments_map, mst_edges
    
    def greedy_solution():
        """Build a greedy solution."""
        nonlocal best_solution, best_obj
        
        # For each target, find the best steiner node considering assignment cost + node weight
        # Start by assigning each target to its cheapest steiner node
        assignments_map = {}
        for i in range(m):
            best_j = min(range(n), key=lambda j: assignment_costs[i][j] + node_weights[j])
            assignments_map[i] = best_j
        
        active_set = set(assignments_map.values())
        
        obj, active_list, assignments_map, mst_edges = evaluate_solution(active_set, assignments_map)
        
        if obj < best_obj:
            best_obj = obj
            best_solution = (active_list, assignments_map, mst_edges)
            if logger:
                logger.log(best_obj)
        
        return active_set, assignments_map
    
    def try_single_hub():
        """Try using a single steiner node for all targets."""
        nonlocal best_solution, best_obj
        
        for j in range(n):
            cost = node_weights[j] + sum(assignment_costs[i][j] for i in range(m))
            if cost < best_obj:
                best_obj = cost
                assignments_map = {i: j for i in range(m)}
                best_solution = ([j], assignments_map, [])
                if logger:
                    logger.log(best_obj)
    
    def local_search(active_set, assignments_map, time_remaining):
        """Improve solution with local search."""
        nonlocal best_solution, best_obj
        
        current_active = set(active_set)
        current_assign = dict(assignments_map)
        
        improved = True
        while improved and time.time() - start_time < time_limit - 0.5:
            improved = False
            
            # Try reassigning each target to a different steiner node
            for i in range(m):
                old_j = current_assign[i]
                for j in range(n):
                    if j == old_j:
                        continue
                    current_assign[i] = j
                    new_active = set(current_assign.values())
                    obj, al, am, me = evaluate_solution(new_active, current_assign)
                    if obj < best_obj:
                        best_obj = obj
                        best_solution = (al, dict(am), me)
                        current_active = new_active
                        improved = True
                        if logger:
                            logger.log(best_obj)
                    else:
                        current_assign[i] = old_j
            
            # Try adding a steiner node and reassigning some targets to it
            for j in range(n):
                if j in current_active:
                    continue
                # Find targets that would benefit from being assigned to j
                benefit_targets = []
                for i in range(m):
                    old_cost = assignment_costs[i][current_assign[i]]
                    new_cost = assignment_costs[i][j]
                    if new_cost < old_cost:
                        benefit_targets.append((old_cost - new_cost, i))
                
                benefit_targets.sort(reverse=True)
                if not benefit_targets:
                    continue
                
                total_benefit = sum(b for b, _ in benefit_targets)
                if total_benefit <= node_weights[j]:
                    # Even assigning all beneficial targets won't cover activation cost
                    # But MST might change, so still try top candidates
                    pass
                
                # Try assigning the beneficial targets to j
                test_assign = dict(current_assign)
                for _, i in benefit_targets:
                    test_assign[i] = j
                
                new_active = set(test_assign.values())
                obj, al, am, me = evaluate_solution(new_active, test_assign)
                if obj < best_obj:
                    best_obj = obj
                    best_solution = (al, dict(am), me)
                    current_active = new_active
                    current_assign = test_assign
                    improved = True
                    if logger:
                        logger.log(best_obj)
            
            # Try removing a steiner node
            for j in list(current_active):
                targets_on_j = [i for i in range(m) if current_assign[i] == j]
                if not targets_on_j:
                    # Remove unused steiner node
                    test_assign = dict(current_assign)
                    new_active = set(test_assign.values())
                    if j not in new_active:
                        continue
                    # This shouldn't happen if current_active == set of assigned values
                    continue
                
                # Reassign targets on j to their next best steiner node
                test_assign = dict(current_assign)
                for i in targets_on_j:
                    best_alt = None
                    best_alt_cost = float('inf')
                    for jj in current_active:
                        if jj == j:
                            continue
                        if assignment_costs[i][jj] < best_alt_cost:
                            best_alt_cost = assignment_costs[i][jj]
                            best_alt = jj
                    # Also consider non-active steiner nodes
                    for jj in range(n):
                        if jj == j:
                            continue
                        cost_jj = assignment_costs[i][jj]
                        if cost_jj < best_alt_cost:
                            best_alt_cost = cost_jj
                            best_alt = jj
                    test_assign[i] = best_alt
                
                new_active = set(test_assign.values())
                obj, al, am, me = evaluate_solution(new_active, test_assign)
                if obj < best_obj:
                    best_obj = obj
                    best_solution = (al, dict(am), me)
                    current_active = new_active
                    current_assign = test_assign
                    improved = True
                    if logger:
                        logger.log(best_obj)
        
        return current_active, current_assign
    
    def solve_with_gurobi():
        """Solve using Gurobi MIP formulation."""
        nonlocal best_solution, best_obj
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return
        
        remaining_time = time_limit - (time.time() - start_time) - 1
        if remaining_time <= 1:
            return
        
        model = gp.Model("steiner_tree")
        model.setParam('TimeLimit', remaining_time)
        model.setParam('Threads', 1)
        model.setParam('OutputFlag', 0)
        
        # Variables
        # y[j] = 1 if steiner node j is active
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        
        # x[i,j] = 1 if target i assigned to steiner j
        x = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
        
        # e[j1,j2] = 1 if edge between steiner j1 and j2 in spanning tree (j1 < j2)
        edges = [(j1, j2) for j1 in range(n) for j2 in range(j1+1, n)]
        e = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
        
        # Objective
        obj = gp.LinExpr()
        # Assignment costs
        for i in range(m):
            for j in range(n):
                obj += assignment_costs[i][j] * x[i, j]
        # Interconnection costs
        for j1, j2 in edges:
            obj += interconnection_costs[j1][j2] * e[j1, j2]
        # Activation costs
        for j in range(n):
            obj += node_weights[j] * y[j]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        # Each target assigned to exactly one steiner node
        for i in range(m):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        
        # Target can only be assigned to active steiner node
        for i in range(m):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j])
        
        # Number of edges = number of active nodes - 1
        model.addConstr(gp.quicksum(e[j1, j2] for j1, j2 in edges) == gp.quicksum(y[j] for j in range(n)) - 1)
        
        # Edge can only exist if both endpoints are active
        for j1, j2 in edges:
            model.addConstr(e[j1, j2] <= y[j1])
            model.addConstr(e[j1, j2] <= y[j2])
        
        # Subtour elimination: for every subset S of steiner nodes, |S| >= 2,
        # sum of edges within S <= sum of y[j] for j in S minus 1
        # We'll use lazy constraints via callback
        
        # For small n, we can add all subtour elimination constraints
        # For larger n, use lazy constraints
        
        if n <= 20:
            # Add all SEC constraints for subsets of size 2 to n
            # This is exponential but feasible for small n
            for size in range(2, n + 1):
                if time.time() - start_time > time_limit - 2:
                    break
                for subset in itertools.combinations(range(n), size):
                    subset_edges = [(j1, j2) for j1 in subset for j2 in subset if j1 < j2]
                    model.addConstr(
                        gp.quicksum(e[j1, j2] for j1, j2 in subset_edges) <= 
                        gp.quicksum(y[j] for j in subset) - 1
                    )
        else:
            # Use callback for lazy constraints
            model.setParam('LazyConstraints', 1)
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    y_val = model.cbGetSolution(y)
                    e_val = model.cbGetSolution(e)
                    
                    active = [j for j in range(n) if y_val[j] > 0.5]
                    if len(active) <= 1:
                        return
                    
                    # Build adjacency for active steiner nodes
                    adj = {j: [] for j in active}
                    for j1, j2 in edges:
                        if j1 in adj and j2 in adj and e_val[j1, j2] > 0.5:
                            adj[j1].append(j2)
                            adj[j2].append(j1)
                    
                    # Find connected components
                    visited = set()
                    components = []
                    for j in active:
                        if j not in visited:
                            comp = []
                            stack = [j]
                            while stack:
                                node = stack.pop()
                                if node in visited:
                                    continue
                                visited.add(node)
                                comp.append(node)
                                for nb in adj[node]:
                                    if nb not in visited:
                                        stack.append(nb)
                            components.append(comp)
                    
                    if len(components) > 1:
                        for comp in components:
                            if len(comp) >= 2:
                                comp_set = set(comp)
                                subset_edges_list = [(j1, j2) for j1, j2 in edges if j1 in comp_set and j2 in comp_set]
                                model.cbLazy(
                                    gp.quicksum(e[j1, j2] for j1, j2 in subset_edges_list) <= 
                                    gp.quicksum(y[j] for j in comp) - 1
                                )
            
            model._callback = callback
        
        # Provide warm start from best heuristic solution
        if best_solution is not None:
            active_list, assign_map, mst_edges_list = best_solution
            active_set_ws = set(active_list)
            for j in range(n):
                y[j].Start = 1.0 if j in active_set_ws else 0.0
            for i in range(m):
                for j in range(n):
                    x[i, j].Start = 1.0 if assign_map[i] == j else 0.0
            edge_set = set()
            if mst_edges_list:
                for e_pair in mst_edges_list:
                    j1, j2 = min(e_pair), max(e_pair)
                    edge_set.add((j1, j2))
            for j1, j2 in edges:
                e[j1, j2].Start = 1.0 if (j1, j2) in edge_set else 0.0
        
        if n <= 20:
            model.optimize()
        else:
            model.optimize(callback)
        
        if model.SolCount > 0:
            obj_val = model.ObjVal
            if obj_val < best_obj:
                best_obj = obj_val
                
                active_list = [j for j in range(n) if y[j].X > 0.5]
                assign_map = {}
                for i in range(m):
                    for j in range(n):
                        if x[i, j].X > 0.5:
                            assign_map[i] = j
                            break
                
                mst_edges_list = []
                for j1, j2 in edges:
                    if e[j1, j2].X > 0.5:
                        mst_edges_list.append([j1, j2])
                
                best_solution = (active_list, assign_map, mst_edges_list)
                if logger:
                    logger.log(best_obj)
    
    # Run heuristics first
    try_single_hub()
    active_set, assign_map = greedy_solution()
    
    # Local search improvement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2:
        active_set, assign_map = local_search(active_set, assign_map, remaining * 0.3)
    
    # Try enumeration for very small instances
    if n <= 15 and m <= 20:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 2:
            # Enumerate subsets of steiner nodes
            for k in range(1, n + 1):
                if time.time() - start_time > time_limit - 1:
                    break
                for subset in itertools.combinations(range(n), k):
                    if time.time() - start_time > time_limit - 1:
                        break
                    subset_set = set(subset)
                    
                    # Assign each target to cheapest in subset
                    test_assign = {}
                    assign_cost = 0
                    for i in range(m):
                        best_j = min(subset, key=lambda j: assignment_costs[i][j])
                        test_assign[i] = best_j
                        assign_cost += assignment_costs[i][best_j]
                    
                    activation_cost = sum(node_weights[j] for j in subset)
                    
                    # Quick bound check
                    if assign_cost + activation_cost >= best_obj:
                        continue
                    
                    obj, al, am, me = evaluate_solution(subset_set, test_assign)
                    if obj < best_obj:
                        best_obj = obj
                        best_solution = (al, dict(am), me)
                        if logger:
                            logger.log(best_obj)
    
    # Try Gurobi for exact solution
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        solve_with_gurobi()
    
    # Build output
    if best_solution is None:
        # Fallback: use first steiner node
        j = 0
        assign_map = {i: j for i in range(m)}
        cost = node_weights[j] + sum(assignment_costs[i][j] for i in range(m))
        best_solution = ([j], assign_map, [])
        best_obj = cost
    
    active_list, assign_map, mst_edges_list = best_solution
    
    solution = {
        "objective_value": best_obj,
        "active_steiner_nodes": active_list,
        "assignments": {str(i): assign_map[i] for i in range(m)},
        "steiner_tree_edges": mst_edges_list if mst_edges_list else []
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()