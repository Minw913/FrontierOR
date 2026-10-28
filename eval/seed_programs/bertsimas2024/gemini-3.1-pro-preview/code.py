import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass
        def log_solution(self, *args, **kwargs):
            pass

def main():
    parser = argparse.ArgumentParser(description="Solve Network Design with Scenarios")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to solution logger JSONL file")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_nodes = data['num_nodes']
    num_commodities = data['num_commodities']
    num_scenarios = data['num_scenarios']
    num_edges = data['num_edges']
    gamma = data['gamma']
    c_0 = data['c_0']
    
    edges = data['edges']
    construction_costs = data['construction_costs']
    flow_costs = data['flow_costs']
    capacities = data['capacities']
    demands = data['demands']
    
    # Initialize Gurobi model
    model = gp.Model("Network_Design_Problem")
    
    # Set requested Gurobi parameters
    model.setParam('Threads', 1)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    if args.time_limit is not None:
        model.setParam('TimeLimit', args.time_limit)
        
    # Variables
    z_vars = []
    tf_vars = {}
    x_vars = {}
    
    for e in range(num_edges):
        z_vars.append(model.addVar(vtype=GRB.BINARY, name=f"z[{e}]"))
        
    for r in range(num_scenarios):
        for e in range(num_edges):
            tf_vars[r, e] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=capacities[e], name=f"tf[{r},{e}]")
            for k in range(num_commodities):
                x_vars[k, r, e] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x[{k},{r},{e}]")
                
    # Build adjacency lists for flow conservation
    out_edges = [[] for _ in range(num_nodes)]
    in_edges = [[] for _ in range(num_nodes)]
    for e, (u, v) in enumerate(edges):
        out_edges[u].append(e)
        in_edges[v].append(e)
        
    # Constraints: Flow conservation
    for r in range(num_scenarios):
        for k in range(num_commodities):
            for i in range(num_nodes):
                demand = demands[r][k][i]
                out_flow = gp.quicksum(x_vars[k, r, e] for e in out_edges[i])
                in_flow = gp.quicksum(x_vars[k, r, e] for e in in_edges[i])
                model.addConstr(out_flow - in_flow == demand, name=f"flow_cons_{r}_{k}_{i}")
                
    # Constraints: Total flow definition
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(tf_vars[r, e] == gp.quicksum(x_vars[k, r, e] for k in range(num_commodities)), name=f"tf_def_{r}_{e}")
            
    # Constraints: Capacity
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(tf_vars[r, e] <= capacities[e] * z_vars[e], name=f"cap_{r}_{e}")
            
    # Constraint: Maximum constructed edges
    model.addConstr(gp.quicksum(z_vars) <= c_0, name="max_edges")
    
    # Objective Formulation
    # 1. Total construction cost
    obj = gp.quicksum(construction_costs[e] * z_vars[e] for e in range(num_edges))
    
    # 2. Expected transportation cost
    if num_scenarios > 0:
        trans_cost = gp.quicksum(flow_costs[e] * tf_vars[r, e] for r in range(num_scenarios) for e in range(num_edges))
        obj += (1.0 / num_scenarios) * trans_cost
        
        # 3. Expected quadratic congestion penalty
        quad_pen = gp.quicksum(tf_vars[r, e] * tf_vars[r, e] for r in range(num_scenarios) for e in range(num_edges))
        obj += (1.0 / num_scenarios) * (1.0 / (2.0 * gamma)) * quad_pen
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Organize all decision variables for fast extraction inside the callback
    all_vars = list(z_vars)
    for r in range(num_scenarios):
        for e in range(num_edges):
            all_vars.append(tf_vars[r, e])
            for k in range(num_commodities):
                all_vars.append(x_vars[k, r, e])
                
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    vals = model.cbGetSolution(all_vars)
                    
                    sol_vars = {}
                    for v, val in zip(all_vars, vals):
                        if val > 1e-5:
                            sol_vars[v.VarName] = val
                            
                    z_dict = {}
                    for e in range(num_edges):
                        z_dict[str(e)] = 1 if vals[e] > 0.5 else 0
                        
                    solution_dict = {
                        "objective_value": obj_val,
                        "model_variables": sol_vars,
                        "z": z_dict,
                        "model_variable_name_patterns": {
                            "z[e]": "binary design variable for edge e",
                            "x[k,r,e]": "commodity k, scenario r, edge e flow",
                            "tf[r,e]": "total scenario flow on edge e"
                        }
                    }
                    logger.log_solution(obj_val, solution_dict)
                except Exception:
                    logger.log(obj_val)
                    
    # Optimize
    model.optimize(callback)
    
    # Write the best solution found to the output file
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol_vars = {}
        z_dict = {}
        
        # Extract z variables
        for e in range(num_edges):
            val = z_vars[e].X
            if val > 1e-5:
                sol_vars[z_vars[e].VarName] = val
            z_dict[str(e)] = 1 if val > 0.5 else 0
            
        # Extract flow variables
        for r in range(num_scenarios):
            for e in range(num_edges):
                val = tf_vars[r, e].X
                if val > 1e-5:
                    sol_vars[tf_vars[r, e].VarName] = val
                for k in range(num_commodities):
                    val = x_vars[k, r, e].X
                    if val > 1e-5:
                        sol_vars[x_vars[k, r, e].VarName] = val
                        
        solution_dict = {
            "objective_value": obj_val,
            "model_variables": sol_vars,
            "z": z_dict,
            "model_variable_name_patterns": {
                "z[e]": "binary design variable for edge e",
                "x[k,r,e]": "commodity k, scenario r, edge e flow",
                "tf[r,e]": "total scenario flow on edge e"
            }
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()