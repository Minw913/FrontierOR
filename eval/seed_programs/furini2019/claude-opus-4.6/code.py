import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve_max_clique(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_vertices = instance["num_vertices"]
    num_edges = instance["num_edges"]
    adjacency_list = instance["adjacency_list"]
    
    # Build adjacency sets for fast lookup
    adj_sets = [set(neighbors) for neighbors in adjacency_list]
    
    # Edge set for quick edge existence check
    edge_set = set()
    for u, v in instance["edges"]:
        edge_set.add((min(u, v), max(u, v)))
    
    best_clique = []
    
    def is_clique(vertices):
        vlist = list(vertices)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                if vlist[j] not in adj_sets[vlist[i]]:
                    return False
        return True
    
    # Greedy initial solution
    def greedy_clique():
        # Order vertices by degree (descending)
        vertices_by_degree = sorted(range(num_vertices), key=lambda v: len(adj_sets[v]), reverse=True)
        clique = []
        for v in vertices_by_degree:
            if time.time() - start_time > time_limit * 0.95:
                break
            # Check if v is connected to all vertices in current clique
            if all(v in adj_sets[u] for u in clique):
                clique.append(v)
        return clique
    
    best_clique = greedy_clique()
    if logger:
        logger.log(len(best_clique))
    
    # For small graphs, try exact approach with Gurobi
    # For larger graphs, use branch-and-bound with pruning
    
    if num_vertices <= 500 and time.time() - start_time < time_limit * 0.8:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time) - 1
            if remaining_time > 2:
                model = gp.Model("max_clique")
                model.setParam("OutputFlag", 0)
                model.setParam("TimeLimit", remaining_time)
                model.setParam("Threads", 1)
                
                # Variables
                x = model.addVars(num_vertices, vtype=GRB.BINARY, name="x")
                
                # Objective: maximize sum of x
                model.setObjective(gp.quicksum(x[i] for i in range(num_vertices)), GRB.MAXIMIZE)
                
                # Constraints: for every non-edge, at most one vertex can be selected
                # This is the independent set formulation on the complement graph
                for i in range(num_vertices):
                    for j in range(i + 1, num_vertices):
                        if j not in adj_sets[i]:
                            model.addConstr(x[i] + x[j] <= 1)
                
                # Warm start with greedy solution
                for v in range(num_vertices):
                    x[v].Start = 1.0 if v in set(best_clique) else 0.0
                
                # Callback for logging
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJBST)))
                        nonlocal best_clique
                        if obj > len(best_clique):
                            vals = model.cbGetSolution(x)
                            clique = [i for i in range(num_vertices) if vals[i] > 0.5]
                            if is_clique(clique):
                                best_clique = clique
                                if logger:
                                    logger.log(len(best_clique))
                
                model.optimize(callback)
                
                if model.SolCount > 0:
                    clique = [i for i in range(num_vertices) if x[i].X > 0.5]
                    if is_clique(clique) and len(clique) > len(best_clique):
                        best_clique = clique
                        if logger:
                            logger.log(len(best_clique))
        except Exception as e:
            pass
    
    return best_clique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    clique = solve_max_clique(instance, args.time_limit, logger)
    
    solution = {
        "objective_value": len(clique),
        "clique_vertices": sorted(clique)
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()