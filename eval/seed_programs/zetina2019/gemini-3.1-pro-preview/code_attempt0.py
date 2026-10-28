import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Fixed-Charge Multicommodity Network Flow Solver")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, required=False, default=None, help="Path to the logging file.")
    args = parser.parse_args()

    # Initialize solution logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Set up Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("fixed_charge_mcf", env=env)
    
    if args.time_limit is not None and args.time_limit > 0:
        m.setParam('TimeLimit', args.time_limit)
    
    # Restrict to a single core per requirements
    m.setParam('Threads', 1)

    num_nodes = data['num_nodes']
    commodities = data['commodities']
    num_k = len(commodities)
    
    arcs = [(u, v) for u, v in data['arcs']]
    
    # Pre-process network for flow constraints
    adj_out = {i: [] for i in range(num_nodes)}
    adj_in = {i: [] for i in range(num_nodes)}
    for u, v in arcs:
        adj_out[u].append(v)
        adj_in[v].append(u)
        
    fixed_costs = data['fixed_costs']
    y_obj = {(u, v): fixed_costs[f"{u}_{v}"] for u, v in arcs}
    
    # Arc installation variables
    y = m.addVars(arcs, vtype=GRB.BINARY, obj=y_obj, name="y")
    
    # Routing allocation variables
    obj_x = {}
    variable_costs = data['variable_costs']
    for k in range(num_k):
        demand = commodities[k]['demand']
        for u, v in arcs:
            arc_key = f"{u}_{v}"
            # Transportation cost = demand * per-unit cost on the arc
            cost = variable_costs[arc_key][k] * demand
            obj_x[k, u, v] = cost
            
    x = m.addVars(
        [(k, u, v) for k in range(num_k) for u, v in arcs],
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        obj=obj_x,
        name="x"
    )
    
    # Flow cannot occur on uninstalled arcs (Capacity scaling bounds)
    m.addConstrs((x[k, u, v] <= y[u, v] for k in range(num_k) for u, v in arcs), name="link")
            
    # Flow conservation constraints
    # Net flow (in - out) equals -1 at origin, 1 at destination, 0 otherwise.
    for k in range(num_k):
        origin = commodities[k]['origin']
        dest = commodities[k]['destination']
        
        for i in range(num_nodes):
            if not adj_in[i] and not adj_out[i]:
                continue
                
            in_vars = [x[k, j, i] for j in adj_in[i]]
            out_vars = [x[k, i, j] for j in adj_out[i]]
            
            in_sum = gp.quicksum(in_vars)
            out_sum = gp.quicksum(out_vars)
            
            if i == origin and i == dest:
                rhs = 0
            elif i == origin:
                rhs = -1
            elif i == dest:
                rhs = 1
            else:
                rhs = 0
                
            m.addConstr(in_sum - out_sum == rhs)

    # Callback definition for solution logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Optimize model
    m.optimize(callback)
    
    # Process solution
    if m.SolCount > 0:
        obj_val = m.objVal
        open_arcs = {}
        
        # Record open arcs
        for u, v in arcs:
            arc_key = f"{u}_{v}"
            if y[u, v].X > 0.5:
                open_arcs[arc_key] = 1
                
        # Record routings (nonzero fractions only)
        routings = {}
        for k in range(num_k):
            routings[str(k)] = {}
            for u, v in arcs:
                val = x[k, u, v].X
                if val > 1e-6:
                    arc_key = f"{u}_{v}"
                    routings[str(k)][arc_key] = val
                    
        solution = {
            "objective_value": obj_val,
            "open_arcs": open_arcs,
            "routings": routings
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()