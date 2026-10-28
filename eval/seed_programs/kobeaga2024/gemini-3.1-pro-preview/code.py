import argparse
import json
import os
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, log_path, sense="minimize"):
            pass
        def log(self, objective_value):
            pass

def solve_orienteering(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
        
    n = instance['num_nodes']
    budget = instance['distance_limitation_d0']
    depot = instance['depot_vertex']
    scores = instance['scores']
    dist = instance['distance_matrix']
    
    # Initialize logger (we maximize profit)
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Create Gurobi model
    model = gp.Model("Orienteering")
    model.setParam('Threads', 1)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    model.setParam('LazyConstraints', 1)
    
    # Variables
    # x[i, j] = 1 if edge from i to j is traversed
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    # y[i] = 1 if node i is visited
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Constraints
    # 1. Depot must be visited
    model.addConstr(y[depot] == 1, name="depot_visited")
    
    # 2. Prevent self-loops for any node except possibly the depot (if no other nodes are visited)
    model.addConstrs((x[i, i] == 0 for i in range(n) if i != depot), name="no_self_loops")
    
    # 3. Flow conservation: in-degree = out-degree = y[i]
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == y[i] for i in range(n)), name="out_degree")
    model.addConstrs((gp.quicksum(x[j, i] for j in range(n)) == y[i] for i in range(n)), name="in_degree")
    
    # 4. Total distance limit
    model.addConstr(
        gp.quicksum(dist[i][j] * x[i, j] for i in range(n) for j in range(n)) <= budget, 
        name="distance_limit"
    )
    
    # Objective: Maximize gathered scores
    model.setObjective(gp.quicksum(scores[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Provide an initial feasible solution (empty tour)
    for i in range(n):
        y[i].Start = 1.0 if i == depot else 0.0
        for j in range(n):
            x[i, j].Start = 1.0 if (i == depot and j == depot) else 0.0
            
    # Subtour elimination callback
    model._n = n
    model._depot = depot
    model._x = x
    model._y = y
    model._best_obj = -1e100
    model._logger = logger
    
    def sec_callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                x_sol = mdl.cbGetSolution(mdl._x)
                y_sol = mdl.cbGetSolution(mdl._y)
                
                # Fetch only edges with value securely above 0.5
                edges = [(i, j) for i in range(mdl._n) for j in range(mdl._n) if x_sol[i, j] > 0.5]
                
                # Build adjacency list
                adj = {i: [] for i in range(mdl._n)}
                for i, j in edges:
                    adj[i].append(j)
                    
                visited_nodes = [i for i in range(mdl._n) if y_sol[i] > 0.5]
                visited = set()
                cycles = []
                
                # Extract connected disjoint cycles strictly
                for i in visited_nodes:
                    if i not in visited:
                        curr = i
                        path = []
                        while curr not in visited:
                            visited.add(curr)
                            path.append(curr)
                            if adj[curr]:
                                curr = adj[curr][0]
                            else:
                                break
                        if curr in path:
                            idx = path.index(curr)
                            cycles.append(path[idx:])
                            
                cuts_added = False
                for cycle in cycles:
                    # Enforce connectivity: Any cycle without the depot is an invalid subtour
                    if mdl._depot not in cycle:
                        cuts_added = True
                        cycle_set = set(cycle)
                        # Directed generalized subtour elimination
                        cut_expr = gp.quicksum(mdl._x[i, j] for i in cycle_set for j in range(mdl._n) if j not in cycle_set)
                        for k in cycle:
                            mdl.cbLazy(cut_expr >= mdl._y[k])
                            
                # Note the solution if feasible and represents a progressive standard best
                if not cuts_added:
                    obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val > mdl._best_obj + 1e-5:
                        mdl._best_obj = obj_val
                        if mdl._logger is not None:
                            mdl._logger.log(obj_val)
                            
            except Exception as e:
                print(f"Exception in Subtour Elimination Callback: {e}")
                
    # Optimize utilizing configured solver and SEC mechanism
    model.optimize(sec_callback)
    
    # Process final evaluated solution route output
    if model.SolCount > 0:
        obj_eval = model.ObjVal
        final_edges = [[i, j] for i in range(n) for j in range(n) if x[i, j].X > 0.5 and i != j]
        final_edges_dict = {i: j for i, j in final_edges}
        
        if final_edges_dict:
            curr = depot
            ordered_visited = []
            visited_in_reconstruct = set()
            while True:
                ordered_visited.append(curr)
                visited_in_reconstruct.add(curr)
                if curr in final_edges_dict:
                    curr = final_edges_dict[curr]
                else:
                    break
                # End condition of the exactly verified single interconnected route cycle
                if curr == depot or curr in visited_in_reconstruct:
                    break
        else:
            ordered_visited = [depot]
            
        solution = {
            "objective_value": float(obj_eval),
            "visited_vertices": ordered_visited,
            "tour_edges": final_edges
        }
    else:
        # Failsafe structural default handling
        solution = {
            "objective_value": 0.0,
            "visited_vertices": [depot],
            "tour_edges": []
        }
        
    # Serialize schema conforming solution safely
    os.makedirs(os.path.dirname(solution_path) or '.', exist_ok=True)
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve the Orienteering Problem via lazy subtour generation.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the problem instance json')
    parser.add_argument('--solution_path', type=str, required=True, help='Path for outputting final solution json')
    parser.add_argument('--time_limit', type=int, required=True, help='Target time limit restrict in execution seconds')
    parser.add_argument('--log_path', type=str, required=False, default=None, help='Tracking path for logger module outputs')
    
    args = parser.parse_args()
    solve_orienteering(args.instance_path, args.solution_path, args.time_limit, args.log_path)