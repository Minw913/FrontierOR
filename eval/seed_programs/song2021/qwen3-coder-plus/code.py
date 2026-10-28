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
    orders_data = instance_data["orders"]

    # Map order_id to index for easier handling
    order_id_to_idx = {}
    orders = {}
    for idx, order_info in enumerate(orders_data):
        oid = order_info["order_id"]
        order_id_to_idx[oid] = idx
        orders[idx] = order_info

    # Create model
    model = gp.Model("OrderScheduling")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.OutputFlag, 0)

    # Identify feasible intervals for each order
    feasible_intervals = defaultdict(list)  # feasible_intervals[i] = [(start, len), ...]
    
    for i in range(n):
        order_info = orders[i]
        
        p_i = order_info["work_content_p"]
        r_i = order_info["release_date_r"]
        d_i = order_info["due_date_d"]
        LB_i = order_info["intensity_lower_bound_LB"]
        UB_i = order_info["intensity_upper_bound_UB"]
        min_len = order_info["lower_exec_interval_length_l"]
        max_len = order_info["upper_exec_interval_length_l_bar"]
        
        # Enumerate candidate start times
        for start_t in range(r_i, T+1):  # Release time constraint
            for length in range(min_len, max_len + 1):
                finish_t = start_t + length - 1
                # Check if finish is in time frame
                if finish_t > T:
                    continue
                # Check intensity bounds feasibility for this duration
                # At least 1/duration >= LB -> duration <= 1/LB
                # At most 1/duration <= UB -> duration >= 1/UB
                if 1/LB_i >= length and 1/UB_i <= length:
                    feasible_intervals[i].append((start_t, length))

    # Variables
    # Binary var: x[i, s, l] = 1 if order i starts at s with duration l
    x = {}
    intensity = {}  # intensity[i, t] = fraction of order i done in period t
    
    for i in range(n):
        for (s, l) in feasible_intervals[i]:
            x[i, s, l] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}_{l}")
    
    # Intensity vars for orders and periods
    for i in range(n):
        order_info = orders[i]
        for t in range(1, T+1):
            intensity[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"intensity_{i}_{t}")
    
    # Non-regular capacity vars
    nr_cap = {}
    for t in range(1, T+1):
        nr_cap[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"nr_cap_{t}")
    
    # Tardiness vars
    tardiness = {}
    for i in range(n):
        tardiness[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"tardiness_{i}")

    # Constraints
    # Each order must be scheduled exactly once
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, s, l] for (s, l) in feasible_intervals[i]) == 1, name=f"schedule_{i}")

    # Set tardiness
    for i in range(n):
        order_info = orders[i]
        d_i = order_info["due_date_d"]
        expr = gp.quicksum(max(0, s+l-1-d_i) * x[i, s, l] for (s, l) in feasible_intervals[i])
        model.addConstr(tardiness[i] >= expr, name=f"tardiness_def_{i}")

    # Set intensity values
    for i in range(n):
        order_info = orders[i]
        LB_i = order_info["intensity_lower_bound_LB"]
        UB_i = order_info["intensity_upper_bound_UB"]
        
        for t in range(1, T+1):
            # Only allow intensity for periods in scheduled execution window
            terms_list = []
            for (s, l) in feasible_intervals[i]:
                if s <= t <= s+l-1:
                    terms_list.append(x[i, s, l])
            
            if terms_list:
                model.addConstr(
                    intensity[i, t] >= LB_i * gp.quicksum(terms_list),
                    name=f"intensity_lb_{i}_{t}"
                )
                model.addConstr(
                    intensity[i, t] <= UB_i * gp.quicksum(terms_list),
                    name=f"intensity_ub_{i}_{t}"
                )
            else:
                # Period is not in any possible execution window
                model.addConstr(intensity[i, t] == 0, name=f"intensity_zero_{i}_{t}")
    
    # Intensities must sum to 1 for each order
    for i in range(n):
        model.addConstr(
            gp.quicksum(intensity[i, t] for t in range(1, T+1)) == 1,
            name=f"sum_one_{i}"
        )
    
    # Capacity constraint: workload can be covered by regular + nonregular capacity
    for t in range(1, T+1):
        workload_t = gp.quicksum(
            orders[i]["work_content_p"] * intensity[i, t] for i in range(n)
        )
        model.addConstr(
            workload_t <= capacities[t-1] + nr_cap[t],
            name=f"capacity_{t}"
        )

    # Objective function
    obj_expr = gp.quicksum(orders[i]["tardiness_cost_w"] * tardiness[i] for i in range(n)) \
              + sigma * gp.quicksum(nr_cap[t] for t in range(1, T+1))
              
    model.setObjective(obj_expr, GRB.MINIMIZE)

    return model, orders, n, T, x, intensity, nr_cap, tardiness, feasible_intervals


def extract_solution(model, orders, n, T, x, intensity, nr_cap, tardiness, feasible_intervals):
    objective_value = model.ObjVal
    
    solution_list = []
    
    for i in range(n):
        # Find the scheduled start and duration
        scheduled = None
        for (s, l) in feasible_intervals[i]:
            if abs(x[i, s, l].X - 1) < 1e-6:
                scheduled = (s, l)
                break
        
        if scheduled is None:
            raise ValueError(f"Order {i} not scheduled!")
        
        s, l = scheduled
        f = s + l - 1
        intensity_dict = {}
        
        for t in range(s, f + 1):
            # Only non-zero intensity periods matter, and only in the current execution window
            int_val = round(intensity[i, t].X, 8)
            if int_val > 1e-6:
                intensity_dict[str(t)] = int_val

        solution_list.append({
            "order_id": i,
            "chosen_interval": [s, f],
            "intensities": intensity_dict
        })

    # Nonregular capacity by period
    nr_dict = {}
    for t in range(1, T+1):
        val = round(nr_cap[t].X, 6)
        if val > 1e-6:
            nr_dict[str(t)] = val

    solution_output = {
        "objective_value": objective_value,
        "orders_solution": solution_list,
        "nonregular_capacity_by_period": nr_dict
    }

    return solution_output


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

    # Build model
    try:
        model, orders, n, T, x, intensity, nr_cap, tardiness, feasible_intervals = solve_order_scheduling(instance_data)
    except Exception as e:
        print(f"Error building model: {e}")
        raise e

    # Optimization callbacks
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)
    
    model.setParam('TimeLimit', args.time_limit)
    model.optimize(callback)

    # Check status and extract solution
    status = model.Status
    if status == GRB.OPTIMAL or (status == GRB.TIME_LIMIT and model.SolCount > 0):
        solution = extract_solution(model, orders, n, T, x, intensity, nr_cap, tardiness, feasible_intervals)
    elif status == GRB.TIME_LIMIT and model.SolCount == 0:
        raise ValueError("No feasible solution found within time limit.")
    elif status in [GRB.INFEASIBLE, GRB.UNBOUNDED]:
        raise ValueError(f"Model infeasible or unbounded. Status: {status}")
    else:
        raise ValueError(f"Optimization finished with status: {status}, SolCount: {model.SolCount}")

    # Save solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()