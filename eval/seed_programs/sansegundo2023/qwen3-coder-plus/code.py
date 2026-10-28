import argparse
import json
import time
from itertools import combinations
from solution_logger import SolutionLogger
def bron_kerbosch_with_pivot(graph, R, P, X, max_cliques):
    """
    Find all maximal cliques using Bron-Kerbosch algorithm with pivoting.
    """
    if not P and not X:
        max_cliques.append(set(R))
        return
    
    # Choose a pivot vertex u from P ∪ X
    u = next(iter(P)) if P else next(iter(X))
    
    # Iterate through P \ N(u)
    for v in P - set(graph.get(u, [])):
        nbrs = set(graph.get(v, []))
        bron_kerbosch_with_pivot(graph, R | {v}, P & nbrs, X & nbrs, max_cliques)
        P = P - {v}
        X = X | {v}


def bron_kerbosch_find_all_maximal_cliques(adjacency_list):
    """
    Wrapper function to run Bron-Kerbosch to find maximal cliques.
    """
    vertices = list(adjacency_list.keys())
    graph = {int(v): adjacency_list[v] for v in vertices}
    
    max_cliques = []
    R = set()
    P = set(vertices)
    X = set()
    
    bron_kerbosch_with_pivot(graph, R, P, X, max_cliques)
    return max_cliques


def greedy_maximal_clique(adjacency_list, start_vertex=None):
    """
    Find a maximal clique greedily starting from a random/default vertex.
    """
    if not adjacency_list:
        return []
    
    if start_vertex is None:
        start_vertex = int(list(adjacency_list.keys())[0])
        
    clique = {start_vertex}
    candidates = set(adjacency_list.get(start_vertex, []))
    
    while candidates:
        # Pick the next vertex that is connected to all current clique vertices
        next_candidate = None
        for v in candidates:
            if all(w in set(adjacency_list[v]) for w in clique):
                next_candidate = v
                break
        
        if next_candidate is None:
            break
            
        clique.add(next_candidate)
        # Update candidates: keep only vertices adjacent to the newly added vertex
        candidates = candidates.intersection(set(adjacency_list[next_candidate]))
        candidates.discard(next_candidate)
    
    return list(clique)


def approximate_maximum_clique_greedy(adjacency_list, start_from_largest_degree=True):
    """
    A greedy heuristic to find a high-quality clique more quickly.
    """
    if not adjacency_list:
        return []
    
    # Initialize with the best known solution so far
    best_clique = []
    
    vertices_by_degree = sorted(
        ((v, len(neighbors)) for v, neighbors in adjacency_list.items()),
        key=lambda x: x[1],
        reverse=True
    )

    for v, _ in vertices_by_degree[:20]:  # Only top 20 degree vertices to save time
        if start_from_largest_degree and len(best_clique) > len(greedy_maximal_clique(adjacency_list, v)):
            continue
        clique = greedy_maximal_clique(adjacency_list, v)
        if len(clique) > len(best_clique):
            best_clique = clique
            
    return best_clique


def gurobi_solve(instance_data, time_limit):
    """
    Use Gurobi to solve the maximum clique problem optimally.
    For each vertex, we have a binary variable indicating selection.
    For each pair of non-adjacent vertices, we add a constraint that both cannot be selected.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("maximum_clique")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)
        
        vertices = instance_data['vertices']
        
        # Create a binary variable for each vertex
        x = {}
        for v in vertices:
            x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")
            
        # Optimization objective: maximize sum of selected vertices
        model.setObjective(gp.quicksum(x[v] for v in vertices), GRB.MAXIMIZE)
        
        # Constraint: for each pair of non-adjacent vertices, at most one can be selected
        adj_list = {int(k): v for k, v in instance_data['adjacency_list'].items()}
        total_pairs = 0
        
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v1, v2 = vertices[i], vertices[j]
                
                # Check if there's no edge between v1 and v2 (they are NOT connected)
                if v2 not in adj_list.get(v1, []) or v1 not in adj_list.get(v2, []):
                    model.addConstr(x[v1] + x[v2] <= 1)
                    total_pairs += 1
                    if total_pairs > 2 * len(vertices) * len(vertices):  # Avoid too many constraints
                        print(f"Too many non-edges ({total_pairs}), skipping further...")
                        break
        
        model.optimize()
        
        if model.Status == GRB.INFEASIBLE or model.SolCount == 0:
            return {"status": "infeasible", "obj_val": 0, "clique": []}
        
        obj_val = int(model.ObjVal) if model.SolCount > 0 else 0
        solution = [v for v in vertices if x[v].X > 0.5]
        
        return {"status": model.Status, "obj_val": obj_val, "clique": solution}
    
    except Exception as e:
        print(f"Error in calling Gurobi: {e}")
        return {"status": "error", "obj_val": 0, "clique": []}


def main():
    parser = argparse.ArgumentParser(description="Solve the Maximum Clique Problem")
    parser.add_argument("--instance_path", required=True, help="Path to the JSON instance file")
    parser.add_argument("--solution_path", required=True, help="Path to write the solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum time allowed in seconds")
    parser.add_argument("--log_path", default=None, help="Path to log intermediate solutions")

    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Start with the greedy heuristic solution provided in the instance
    best_solution = instance_data.get("greedy_clique", [])
    best_objective = len(best_solution)
    
    if logger:
        logger.log(best_objective)
    
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time <= 60:  # Skip refinement if not enough time leftover
        # Return initial solution if time is tight
        result = {
            "objective_value": best_objective,
            "clique_vertices": best_solution
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(result, f)
        return
        
    
    # Apply greedy improvement
    try:
        improved_greedy_clique = approximate_maximum_clique_greedy(instance_data['adjacency_list'])
        if len(improved_greedy_clique) > best_objective:
            best_solution = improved_greedy_clique
            best_objective = len(improved_greedy_clique)
            if logger:
                logger.log(best_objective)
    except:
        pass  # If there is an error in greedy improvement, continue with initial solution
    
    # Try to use Gurobi if there's sufficient time remaining
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time >= 5:  # We need some meaningful time to get benefits from Gurobi
        try:
            gurobi_result = gurobi_solve(instance_data, remaining_time)
            if gurobi_result["obj_val"] > best_objective:
                best_solution = gurobi_result["clique"]
                best_objective = gurobi_result["obj_val"]
                if logger:
                    logger.log(best_objective)
        except:
            pass  # If Gurobi fails or isn't available, we continue with prior solution

    # Final safety check ensuring correctness
    # Verify that the returned solution is indeed a valid clique
    def is_valid_clique(clique, adjacency_list):
        clique_set = set(clique)
        for u, v in combinations(clique_set, 2):
            if v not in set(adjacency_list[str(u)]):
                return False
        return True
    
    if not is_valid_clique(best_solution, {str(k): v for k, v in instance_data['adjacency_list'].items()}):
        # Fallback to empty solution if validation failed
        best_solution = []
        best_objective = 0
    
    result = {
        "objective_value": best_objective,
        "clique_vertices": best_solution
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()