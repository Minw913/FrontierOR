import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, log_path, sense="minimize"):
            pass
        def log(self, obj):
            pass

def solve(instance, time_limit, logger):
    n_clients = instance["n_clients"]
    n_servers = instance["n_servers"]
    linear_costs = instance["linear_costs"]
    quadratic_costs = instance.get("quadratic_costs", [])

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()

    model = gp.Model("DistributedProcessing", env=env)
    
    # Adhere to maximum concurrency specified
    model.setParam("Threads", 1)
    model.setParam("NonConvex", 2) # Ensures full support for generic quadratic objective matrices
    
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
        
    # Variables representing assignment of client i to server j
    x = {}
    for i in range(n_clients):
        for j in range(n_servers):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
    # Constraints: exactly one server assigned per client
    for i in range(n_clients):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_servers)) == 1, name=f"assign_{i}")
        
    qobj = gp.QuadExpr()
    
    # Fast bulk addition for linear component
    n_lin = n_clients * n_servers
    lin_vars = [None] * n_lin
    lin_coeffs = [0.0] * n_lin
    idx = 0
    for i in range(n_clients):
        for j in range(n_servers):
            lin_vars[idx] = x[i, j]
            lin_coeffs[idx] = float(linear_costs[i][j])
            idx += 1
            
    qobj.addTerms(lin_coeffs, lin_vars)
    
    # Fast bulk addition for quadratic interactions terms
    if quadratic_costs:
        n_qc = len(quadratic_costs)
        q_vars1 = [None] * n_qc
        q_vars2 = [None] * n_qc
        q_coeffs = [0.0] * n_qc
        for q_idx, entry in enumerate(quadratic_costs):
            q_vars1[q_idx] = x[int(entry[0]), int(entry[1])]
            q_vars2[q_idx] = x[int(entry[2]), int(entry[3])]
            q_coeffs[q_idx] = float(entry[4])
            
        qobj.addTerms(q_coeffs, q_vars1, q_vars2)
        
    model.setObjective(qobj, GRB.MINIMIZE)
    
    # Establish a fast initial feasible limit (Greedy solution base on lowest linear cost)
    for i in range(n_clients):
        best_server = min(range(n_servers), key=lambda j: linear_costs[i][j])
        for j in range(n_servers):
            x[i, j].Start = (1.0 if j == best_server else 0.0)
            
    # Submitting the intermediate bounds explicitly
    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            objval = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objval)
                
    model.optimize(callback)
    
    # Inquire best feasible objective solution captured from the Gurobi space
    if model.SolCount > 0:
        assignment = {}
        for i in range(n_clients):
            for j in range(n_servers):
                if x[i, j].X > 0.5:
                    assignment[str(i)] = j
        obj_val = model.ObjVal
    else:
        # Reliable fallback: Return our greedy initial feasible logic directly if optimization couldn't yield any states
        assignment = {}
        obj_val = 0.0
        for i in range(n_clients):
            best_server = min(range(n_servers), key=lambda j: linear_costs[i][j])
            assignment[str(i)] = best_server
            obj_val += float(linear_costs[i][best_server])
            
        for entry in quadratic_costs:
            c1, s1, c2, s2, cost = entry
            if assignment[str(int(c1))] == int(s1) and assignment[str(int(c2))] == int(s2):
                obj_val += float(cost)
                
    return {
        "objective_value": obj_val,
        "assignment": assignment
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Quadratic Semi-Assignment Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum solving time in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to write intermediate solutions JSONL (optional)")
    
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    solution_data = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, "w") as f:
        json.dump(solution_data, f, indent=2)