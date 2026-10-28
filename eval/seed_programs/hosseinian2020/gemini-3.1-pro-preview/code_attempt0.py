import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

# Attempt to import SolutionLogger as required by the environment
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Maximize Edge-Weight Clique")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    
    args = parser.parse_args()
    
    # Initialize the external logger
    logger = None
    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
        
    # Read the instance data
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    num_vertices = instance["num_vertices"]
    num_edges = instance["num_edges"]
    vertices_list = instance["vertices"]
    
    # Edge case: If there are no positive-weighted edges, an empty clique is trivially optimal.
    if num_edges == 0:
        sol = {"objective_value": 0.0, "clique": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        if logger:
            logger.log(0.0)
        return

    # Build an O(1) lookup adjacency list
    adj = {v: set() for v in vertices_list}
    for e in instance["edges"]:
        u = e["i"]
        v = e["j"]
        adj[u].add(v)
        adj[v].add(u)
        
    # Initialize Gurobi model setup
    m = gp.Model("MEWCP")
    
    # Strictly enforce time limits and multi-threading boundaries given single CPU rule
    m.Params.TimeLimit = args.time_limit
    m.Params.Threads = 1
    m.Params.LogToConsole = 1
    
    # x[v] is a binary decision variable indicating whether vertex v is part of the clique
    x = m.addVars(vertices_list, vtype=GRB.BINARY, name="x")
    
    # y[k] is a continuous decision variable representing whether edge k is within the selected clique.
    # It acts exactly like a boolean AND condition via limits. Since we are MAXIMIZING, 
    # y[k] will naturally be pushed to its maximum supported bound (i.e., 1.0 IF both endpoints are chosen).
    y = m.addVars(num_edges, vtype=GRB.CONTINUOUS, ub=1.0, name="y")
    
    # Tie the y bounds to the endpoints linearly
    for k, e in enumerate(instance["edges"]):
        u = e["i"]
        v = e["j"]
        m.addConstr(y[k] <= x[u], name=f"yL_{k}")
        m.addConstr(y[k] <= x[v], name=f"yR_{k}")
        
    # Objective Formulation: Maximize the summation of edge weights formed within the chosen clique
    obj_expr = gp.quicksum(e["weight"] * y[k] for k, e in enumerate(instance["edges"]))
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Explicit independent-set constraints over the complementary graph edges (Non-edges).
    # If no edge perfectly couples nodes (u, v), they fundamentally cannot both concurrently exist inside the clique.
    non_edges_iter = ((vertices_list[i], vertices_list[j])
                      for i in range(num_vertices)
                      for j in range(i + 1, num_vertices)
                      if vertices_list[j] not in adj[vertices_list[i]])
                      
    m.addConstrs((x[u] + x[v] <= 1 for u, v in non_edges_iter), name="ne")
    
    # Setup Gurobi callback for tracking incumbent solutions dynamically 
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # We are guaranteed an integral, integer-feasible valid state in MIPSOL phase
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Fire up the optimizer sequence
    m.optimize(callback)
    
    # Extraction Phase
    if m.SolCount > 0:
        obj_val = float(m.ObjVal)
        selected_clique = [v for v in vertices_list if x[v].X > 0.5]
    else:
        obj_val = 0.0
        selected_clique = []
        
    sol = {
        "objective_value": obj_val,
        "clique": selected_clique
    }
    
    # Dump out valid JSON formatted payload aligned tightly to strictly mandated properties
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()