import argparse
import json
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Hub Location under Uncertainty")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the solution JSON should be written.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the logging JSONL file.")
    args = parser.parse_args()

    # Setup the logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    num_scenarios = instance['num_scenarios']
    cost_params = instance['cost_parameters']
    chi = cost_params['chi']
    alpha = cost_params['alpha']
    delta = cost_params['delta']
    
    p_hubs = instance.get('p_hubs')
    distances = instance['distances']
    fixed_costs = instance['fixed_costs']
    hub_capacities = instance.get('hub_capacities')
    scenarios = instance['scenarios']

    # Precompute outgoing and incoming flow requirements explicitly from scenario demands
    W = [scenarios[s]["demands"] for s in range(num_scenarios)]
    O = [[sum(W[s][i][j] for j in range(n)) for i in range(n)] for s in range(num_scenarios)]
    I = [[sum(W[s][j][i] for j in range(n)) for i in range(n)] for s in range(num_scenarios)]

    # Initialize Environment and Model restricting to a single thread
    env = gp.Env()
    model = gp.Model("hub_location_uncertainty", env=env)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    # Variables: Open hubs
    z_obj = [fixed_costs[k] if p_hubs is None else 0.0 for k in range(n)]
    z = model.addVars(n, vtype=GRB.BINARY, obj=z_obj, name="z")

    # Variables: Assignment of node i to hub k in scenario s
    def x_coeff(s, i, k):
        return scenarios[s]["probability"] * (chi * O[s][i] + delta * I[s][i]) * distances[i][k]
    
    x_indices = [(s, i, k) for s in range(num_scenarios) for i in range(n) for k in range(n)]
    x_coeffs = {idx: x_coeff(*idx) for idx in x_indices}
    x = model.addVars(x_indices, vtype=GRB.BINARY, obj=x_coeffs, name="x")

    # Variables: Inter-hub flow transfer from origin i departing its hub k to destination hub m in scenario s
    if alpha > 0:
        def y_coeff(s, i, k, m):
            return scenarios[s]["probability"] * alpha * distances[k][m]
            
        y_indices = [(s, i, k, m) for s in range(num_scenarios) for i in range(n) if O[s][i] > 0 
                     for k in range(n) for m in range(n)]
        y_coeffs = {idx: y_coeff(*idx) for idx in y_indices}
        Y = model.addVars(y_indices, vtype=GRB.CONTINUOUS, lb=0.0, obj=y_coeffs, name="Y")

    # Constraints
    # 1. A node acting as a hub must be assigned to itself
    model.addConstrs(
        (x[s, k, k] == z[k] for s in range(num_scenarios) for k in range(n)),
        name="hub_self_assign"
    )

    # 2. Every node must be assigned to exactly one hub in each scenario
    model.addConstrs(
        (x.sum(s, i, '*') == 1 for s in range(num_scenarios) for i in range(n)),
        name="assign_once"
    )

    # 3. Nodes can only be assigned to open hubs
    model.addConstrs(
        (x[s, i, k] <= z[k] for s in range(num_scenarios) for i in range(n) for k in range(n) if i != k),
        name="assign_to_open"
    )

    # 4 & 5. Inter-hub Flow routing logic to linearly compute precise transfer cost
    if alpha > 0:
        model.addConstrs(
            (Y.sum(s, i, k, '*') == O[s][i] * x[s, i, k] 
             for s in range(num_scenarios) for i in range(n) if O[s][i] > 0 for k in range(n)),
            name="flow_out"
        )

        for s in range(num_scenarios):
            for i in range(n):
                if O[s][i] > 0:
                    for m in range(n):
                        rhs = gp.quicksum(W[s][i][j] * x[s, j, m] for j in range(n) if W[s][i][j] > 0)
                        lhs = Y.sum(s, i, '*', m)
                        model.addConstr(lhs == rhs, name=f"flow_in_{s}_{i}_{m}")

    # 6. Capacitated variant
    if hub_capacities is not None:
        for s in range(num_scenarios):
            for k in range(n):
                model.addConstr(
                    gp.quicksum(O[s][i] * x[s, i, k] for i in range(n) if O[s][i] > 0) <= hub_capacities[k] * z[k],
                    name=f"cap_{s}_{k}"
                )

    # 7 & 8. Require exactly p hubs or at least one opened
    if p_hubs is not None:
        model.addConstr(z.sum() == p_hubs, name="p_hubs")
    else:
        model.addConstr(z.sum() >= 1, name="at_least_one")

    # MIPSOL Callback to log converging bounds over search
    def mip_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)
            except Exception:
                pass

    # Optimization Execution
    model.optimize(mip_callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        hubs = [k for k in range(n) if z[k].X > 0.5]
        z_vals = [1.0 if z[k].X > 0.5 else 0.0 for k in range(n)]
        
        allocations = {}
        x_vals = {}
        
        for s in range(num_scenarios):
            scen_str = str(s)
            allocations[scen_str] = {}
            x_vals[scen_str] = {}
            
            for i in range(n):
                i_str = str(i)
                # Recover the single chosen hub for this node using standard maximal float search resolving precisions
                assigned_k = max(range(n), key=lambda k: x[s, i, k].X)
                allocations[scen_str][i_str] = assigned_k
                
                x_vals_scen_i = {}
                for k in range(n):
                    if i != k:
                        x_vals_scen_i[str(k)] = 1.0 if k == assigned_k else 0.0
                
                x_vals[scen_str][i_str] = x_vals_scen_i
                
        solution = {
            "objective_value": float(obj_val),
            "hubs": hubs,
            "allocations": allocations,
            "z_values": z_vals,
            "x_values": x_vals
        }
        
        # Output optimal assignments fitting specified solution schema
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
            
    else:
        print("No feasible solution was determined within the allotted time limit.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()