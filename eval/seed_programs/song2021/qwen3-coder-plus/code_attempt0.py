import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB


def solve_order_scheduling(instance_data):
    # Extract instance data
    T = instance_data["planning_horizon_H"]
    n = instance_data["num_orders_n"]
    sigma = instance_data["unit_cost_nonregular_capacity_sigma"]
    capacities = instance_data["capacities"]
    orders = instance_data["orders"]

    # Create Gurobi model
    model = gp.Model("OrderScheduling")
    model.setParam(GRB.Param.OutputFlag, 0)

    # Define binary variables x[i,s,f] for each order i, start s, finish f
    # x[i,s,f] = 1 if order i starts at period s and finishes at period f  
    binary_vars = {}
    
    for i, order_info in enumerate(orders):
        feasible_intervals = []
        
        p_i = order_info["work_content_p"]
        r_i = order_info["release_date_r"]
        d_i = order_info["due_date_d"]
        UB_i = order_info["intensity_upper_bound_UB"]
        LB_i = order_info["intensity_lower_bound_LB"]
        
        min_length = order_info["lower_exec_interval_length_l"]
        max_length = order_info["upper_exec_interval_length_l_bar"]
        
        # Enumerate all feasible start-finish combinations
        for length in range(min_length, min(max_length + 1, T + 1)):
            for s in range(r_i, T - length + 2):  # ensure finish <= T
                f = s + length - 1
                binary_vars[i, s, f] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}_{f}")
    
    # Constraint: Each order must be assigned to exactly one start-finish interval
    for i in range(n):
        model.addConstr(
            gp.quicksum(binary_vars[i, s, f] for (i_, s, f) in binary_vars if i_ == i),
            GRB.EQUAL,
            1
        )

    # Variables for period-by-period usage of each order 
    # intensity[i,t] - fraction of work of order i in period t
    intensity = {}  # intensity[order_idx, period]
    for i, order_info in enumerate(orders):
        for t in range(1, T+1):
            intensity[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"intensity_{i}_{t}")

    # Link intensity variables to binary selection variables 
    # If order i is scheduled in [s, f], intensities in [s, f] must sum to 1 and obey bounds
    for i, order_info in enumerate(orders):
        LB_i = order_info['intensity_lower_bound_LB']
        UB_i = order_info['intensity_upper_bound_UB']
        
        for s_start in range(1, T+1):
            for f_finish in range(s_start, T+1):
                if (i, s_start, f_finish) not in binary_vars:
                    continue
                
                # For each period t in [s_start, f_finish]
                for t in range(s_start, f_finish+1):
                    model.addConstr(
                        intensity[i, t] >= LB_i * binary_vars[i, s_start, f_finish],
                        name=f"LBIntensity_{i}_{t}_{s_start}_{f_finish}"
                    )
                    model.addConstr(
                        intensity[i, t] <= UB_i * binary_vars[i, s_start, f_finish],
                        name=f"UBIntensity_{i}_{t}_{s_start}_{f_finish}"
                    )
                
                # For each period t NOT in [s_start, f_finish]
                for t in range(1, T+1):
                    if not (s_start <= t <= f_finish):
                        model.addConstr(
                            intensity[i, t] <= (1 - binary_vars[i, s_start, f_finish]),  # intensity will be 0 if not scheduled in this interval
                            name=f"ZeroIntensityOutside_{i}_{t}_{s_start}_{f_finish}"
                        )
        
        # Ensure intensities of order i sum to 1 over its execution window
        total_intensity_expr = gp.quicksum(intensity[i, t] for t in range(1, T+1))
        model.addConstr(total_intensity_expr == 1)

    # Resource capacity constraints with overflow variables
    nonregular = {}  # nonregular[t] - excess in period t
    for t in range(1, T+1):
        nonregular[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"nonregular_{t}")
        
        # Define the constraint: total_work_in_t + overflow_in_t >= capacity_t
        work_in_t = gp.quicksum(
            orders[i]["work_content_p"] * intensity[i, t]
            for i in range(n)
        )
        model.addConstr(
            work_in_t - nonregular[t] <= capacities[t-1],  # capacity array is 0-indexed
            name=f"CapacityOverPeriod_{t}"
        )

    # Calculate tardiness
    tardiness = {}  # tardiness[i] - tardiness of order i
    for i, order_info in enumerate(orders):
        d_i = order_info["due_date_d"]
        
        # Binary variable to capture finish-period when x[i,s,f] is chosen
        finish_period_for_order = gp.quicksum(f * binary_vars[i, s, f] for (i_, s, f) in binary_vars if i_ == i)
        
        # Calculate tardiness using binary representation of finish period
        # We'll introduce aux var for max(0, finish_period - d_i)
        tardiness[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"tardiness_{i}")
        
        # tardiness[i] >= finish_period - d_i  
        model.addConstr(tardiness[i] >= finish_period_for_order - d_i, name=f"TardinessGeq_{i}")
        
    # Objective function
    total_tardiness_cost = gp.quicksum(
        orders[i]["tardiness_cost_w"] * tardiness[i]
        for i in range(n)
    )
    total_nonregular_cost = sigma * gp.quicksum(nonregular[t] for t in range(1, T+1))
    
    obj = total_tardiness_cost + total_nonregular_cost
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    return model, orders, binary_vars, intensity, nonregular, tardiness, T, n


def extract_solution(model, orders, binary_vars, intensity, nonregular, tardiness, T, n):
    sol = {"objective_value": model.objVal, "orders_solution": [], "nonregular_capacity_by_period": {}}
    
    chosen_intervals = {}
    intensity_values = defaultdict(dict)
    
    # Extract selected intervals and intensities
    for (i, s, f), var in binary_vars.items():
        if var.x > 0.5:  # Binary var is essentially 1
            chosen_intervals[i] = (s, f)
            
            # Get actual intensity values for periods during execution
            for t in range(s, f+1):
                if (i, t) in intensity:
                    intensity_val = intensity[i, t].x
                    intensity_values[i][str(t)] = intensity_val

    # Formulate solution for orders
    for i in range(n):
        s, f = chosen_intervals[i]
        order_sol = {
            "order_id": i,
            "chosen_interval": [s, f],
            "intensities": intensity_values[i]
        }
        sol["orders_solution"].append(order_sol)

    # Get nonregular capacity by period
    for t in range(1, T+1):
        val = nonregular[t].x
        if val > 1e-6:  # Include only if positive (round off near-zero)
            sol["nonregular_capacity_by_period"][str(t)] = val
    
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path where the solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", help="Path to a JSONL file for logging intermediate solutions (optional).")

    args = parser.parse_args()

    # Initialize logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve
    model, orders, binary_vars, intensity, nonregular, tardiness, T, n = solve_order_scheduling(instance_data)
    
    # Set time limit
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    
    # Callback function to log objectives of feasible solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)
    
    model.optimize(mycallback)
    
    # Check if model was solved
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.solCount >= 1:  # At least one solution exists
            solution = extract_solution(model, orders, binary_vars, intensity, nonregular, tardiness, T, n)
        else:
            # Handle case when no feasible solution found before timeout
            raise ValueError("No feasible solution found")
    elif model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model.ilp")
        raise ValueError("Model is infeasible")
    else:
        raise ValueError(f"Model status: {model.status}")

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()