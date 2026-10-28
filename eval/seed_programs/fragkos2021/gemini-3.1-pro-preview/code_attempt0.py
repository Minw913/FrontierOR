import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Reliable Facility Location Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to save the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, required=False, help="Path to log intermediate solutions.")
    args = parser.parse_args()

    # Initialize solution logger if provided
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    nodes = data['nodes']
    tc_matrix = data['transportation_cost_matrix']
    penalties = data['penalty_cost']

    # Initialize Gurobi model
    model = gp.Model("Reliable_Facility_Location")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Single core environment
    model.setParam('OutputFlag', 1)

    # Decision variables for opening facilities
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")

    # Ensure the solver always has a valid trivial upper bound (open no facilities -> pay all penalties)
    for j in range(num_facilities):
        x[j].Start = 0

    expected_costs = []

    # The independent RFLP formulation efficiently computes exact probabilities.
    # W_{i, k} = probability that all opened facilities among the first k fail.
    # W_k = W_{k-1} * (1 - s_j * x_j). We linearize this using continuous variable V_k representing W_{k-1} * x_j.
    # Since objective minimizes W, V is naturally pushed up to its bounds, 
    # seamlessly computing the product of probabilities exactly without an exponential number of scenarios.
    
    for i in range(num_customers):
        d_i = nodes[i]['demand']
        p_i = penalties[i]
        
        # Skip customers with absolutely zero demand to reduce model size
        if d_i <= 0:
            continue
            
        # Filter and sort candidate facilities by shipping cost
        fac_costs = []
        for j in range(num_facilities):
            c_ij = tc_matrix[i][j]
            # Only consider facility if it is strictly cheaper than the penalty
            if c_ij < p_i:
                fac_costs.append((c_ij, j))
        
        # Sort facilities in ascending order of shipping cost to this customer
        fac_costs.sort(key=lambda item: item[0])
        
        K_i = len(fac_costs)
        if K_i == 0:
            # No facility can improve over the penalty option
            expected_costs.append(d_i * p_i)
            continue
            
        C_i = [item[0] for item in fac_costs]
        
        # Cost increments
        Deltas = []
        for k in range(K_i - 1):
            Deltas.append(C_i[k+1] - C_i[k])
        Deltas.append(p_i - C_i[-1])
        
        # Marginal probability cascading variables
        W = model.addVars(K_i + 1, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"W_{i}")
        V = model.addVars(K_i, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"V_{i}")
        
        model.addConstr(W[0] == 1.0, name=f"W_{i}_init")
        
        for k in range(K_i):
            j = fac_costs[k][1]
            s_j = nodes[j]['survival_probability']
            
            # W[k+1] = W[k] - s_j * W[k] * x[j] (Linearized via V[k])
            model.addConstr(W[k+1] == W[k] - s_j * V[k], name=f"W_def_{i}_{k}")
            
            # Envelope bounds for V[k] representing W[k] * x[j]
            model.addConstr(V[k] <= W[k], name=f"V_ub_W_{i}_{k}")
            model.addConstr(V[k] <= x[j], name=f"V_ub_x_{i}_{k}")
            model.addConstr(V[k] >= W[k] + x[j] - 1, name=f"V_lb_{i}_{k}")
            
        # Add the expected fractional unit cost for the i-th customer
        expr = C_i[0] + gp.quicksum(Deltas[k] * W[k+1] for k in range(K_i))
        expected_costs.append(d_i * expr)

    # Set up objective: Fixed Setup Costs + Total Expected Fulfillment Cost
    obj = gp.quicksum(nodes[j]['fixed_setup_cost'] * x[j] for j in range(num_facilities)) + gp.quicksum(expected_costs)
    model.setObjective(obj, GRB.MINIMIZE)

    # Callback logging
    def optimize_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Run solver
    model.optimize(optimize_callback)

    # Retrieve and dump solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        fac_locations = [int(round(x[j].X)) for j in range(num_facilities)]
        
        solution = {
            "objective_value": obj_val,
            "facility_locations": fac_locations
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        # Failsafe dummy fallback scenario if absolutely necessary
        # Technically 'SolCount == 0' won't happen because of `.Start = 0` trivial bounds
        solution = {
            "objective_value": sum(nodes[i]['demand'] * penalties[i] for i in range(num_customers)),
            "facility_locations": [0] * num_facilities
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()