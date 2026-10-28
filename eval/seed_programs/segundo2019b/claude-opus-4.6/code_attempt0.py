import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    vertices = graph['vertices']
    edges = graph['edges']
    vertex_weights = instance['vertex_weights']
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Build adjacency set
    adj = {v: set() for v in vertices}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Map vertex to weight
    weight = {v: vertex_weights[i] for i, v in enumerate(vertices)}
    
    # Best solution tracking
    best_clique = []
    best_weight = 0
    
    def update_best(clique):
        nonlocal best_clique, best_weight
        w = sum(weight[v] for v in clique)
        if w > best_weight:
            best_weight = w
            best_clique = list(clique)
            if logger:
                logger.log(best_weight)
    
    def is_clique(vset):
        vlist = list(vset)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                a, b = vlist[i], vlist[j]
                if (min(a, b), max(a, b)) not in edge_set:
                    return False
        return True
    
    # For small graphs, try Gurobi MIP
    # For larger graphs, use greedy + local search
    
    remaining_time = time_limit - (time.time() - start_time)
    
    if num_vertices <= 500 and remaining_time > 2:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Build complement edges for the ILP formulation
            # x[v] = 1 if vertex v is in the clique
            # For every non-edge (u,v), x[u] + x[v] <= 1
            
            model = gp.Model("max_weight_clique")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining_time - 1))
            model.setParam('Threads', 1)
            
            x = {}
            for v in vertices:
                x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}", obj=weight[v])
            
            model.setAttr('ModelSense', GRB.MAXIMIZE)
            model.update()
            
            # Add constraints for non-edges
            vertex_set = set(vertices)
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    if (min(u, v), max(u, v)) not in edge_set:
                        model.addConstr(x[u] + x[v] <= 1)
            
            # Callback to log incumbents
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    vals = model.cbGetSolution(list(x.values()))
                    clique = [v for v, val in zip(x.keys(), vals) if val > 0.5]
                    update_best(clique)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                clique = [v for v in vertices if x[v].X > 0.5]
                update_best(clique)
            
        except Exception as e:
            # Fall through to heuristic
            pass
    
    # Always run heuristic approaches as well (or if MIP didn't work)
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 0.5:
        # Greedy approach: repeatedly pick highest weight vertex compatible with current clique
        # Try multiple orderings
        
        # Sort vertices by weight descending
        sorted_by_weight = sorted(vertices, key=lambda v: weight[v], reverse=True)
        
        # Greedy 1: weight-based
        clique = []
        clique_set = set()
        for v in sorted_by_weight:
            if time.time() - start_time > time_limit - 0.2:
                break
            if all(v in adj[u] for u in clique):
                clique.append(v)
                clique_set.add(v)
        update_best(clique)
        
        # Greedy 2: weight/degree ratio
        sorted_by_ratio = sorted(vertices, key=lambda v: weight[v] / (len(adj[v]) + 1), reverse=True)
        clique = []
        for v in sorted_by_ratio:
            if time.time() - start_time > time_limit - 0.2:
                break
            if all(v in adj[u] for u in clique):
                clique.append(v)
        update_best(clique)
        
        # Greedy 3: degree-based (high degree first, then weight)
        sorted_by_degree = sorted(vertices, key=lambda v: (len(adj[v]), weight[v]), reverse=True)
        clique = []
        for v in sorted_by_degree:
            if time.time() - start_time > time_limit - 0.2:
                break
            if all(v in adj[u] for u in clique):
                clique.append(v)
        update_best(clique)
        
        # Local search: try to improve best clique by swap
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0.5:
            import random
            random.seed(42)
            
            current_clique = set(best_clique)
            current_weight = best_weight
            
            iterations = 0
            while time.time() - start_time < time_limit - 0.3:
                iterations += 1
                improved = False
                
                # Try adding a vertex
                candidates = []
                for v in vertices:
                    if v not in current_clique:
                        if all(v in adj[u] for u in current_clique):
                            candidates.append(v)
                
                if candidates:
                    # Add the best candidate
                    best_add = max(candidates, key=lambda v: weight[v])
                    current_clique.add(best_add)
                    current_weight += weight[best_add]
                    improved = True
                    if current_weight > best_weight:
                        update_best(list(current_clique))
                
                if not improved:
                    # Try swap: remove one vertex, add two or more
                    found_swap = False
                    clique_list = list(current_clique)
                    random.shuffle(clique_list)
                    
                    for remove_v in clique_list:
                        if time.time() - start_time > time_limit - 0.3:
                            break
                        # Remove remove_v and see what we can add
                        test_clique = current_clique - {remove_v}
                        # Find all vertices that can be added
                        can_add = []
                        for v in vertices:
                            if v not in test_clique:
                                if all(v in adj[u] for u in test_clique):
                                    can_add.append(v)
                        
                        # Greedily add from can_add
                        new_clique = set(test_clique)
                        can_add.sort(key=lambda v: weight[v], reverse=True)
                        for v in can_add:
                            if all(v in adj[u] for u in new_clique):
                                new_clique.add(v)
                        
                        new_weight = sum(weight[v] for v in new_clique)
                        if new_weight > current_weight:
                            current_clique = new_clique
                            current_weight = new_weight
                            found_swap = True
                            if current_weight > best_weight:
                                update_best(list(current_clique))
                            break
                    
                    if not found_swap:
                        # Random restart
                        random.shuffle(sorted_by_weight)
                        clique = []
                        for v in sorted_by_weight:
                            if all(v in adj[u] for u in clique):
                                clique.append(v)
                        current_clique = set(clique)
                        current_weight = sum(weight[v] for v in current_clique)
                        if current_weight > best_weight:
                            update_best(list(current_clique))
    
    # If no clique found, pick the highest weight single vertex
    if not best_clique and vertices:
        best_v = max(vertices, key=lambda v: weight[v])
        best_clique = [best_v]
        best_weight = weight[best_v]
        if logger:
            logger.log(best_weight)
    
    # Write solution
    solution = {
        "objective_value": float(best_weight),
        "clique_vertices": best_clique
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


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