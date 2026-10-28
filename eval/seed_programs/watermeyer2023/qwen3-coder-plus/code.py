import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_rcpsp_prct(instance_path, solution_path, time_limit, log_path):
    # Read the instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Extract data
    num_real_activities = instance["num_real_activities"]
    num_nodes = instance["num_nodes"] # total number of activities including dummy start/end
    num_resources = instance["num_partially_renewable_resources"]
    deadline = instance["deadline"]
    processing_times = instance["processing_times"]
    temporal_constraints = instance["temporal_constraints"]
    resources = instance["resources"]

    # Initialize logger if provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create Gurobi model
    model = gp.Model("RCPSPrct")
    model.setParam("Threads", 1)

    # Set time limit
    model.setParam('TimeLimit', time_limit)

    # Decision variables: start times for each activity (including dummies)
    max_time = deadline + max(processing_times)  # heuristic upper bound
    start_times = {}
    for i in range(num_nodes):
        start_times[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_time, name=f"start_{i}")

    # Minimize the start time of the end activity (project duration)
    model.setObjective(start_times[num_nodes - 1], GRB.MINIMIZE)

    # Constraint: dummy start activity starts at time 0
    model.addConstr(start_times[0] == 0)

    # Temporal Constraints
    for tc in temporal_constraints:
        from_activity = tc['from']
        to_activity = tc['to']
        delta = tc['delta']
        
        # S_i + delta <= S_j
        model.addConstr(start_times[from_activity] + delta <= start_times[to_activity])

    # Deadline constraint: the last dummy activity must start by deadline
    model.addConstr(start_times[num_nodes-1] <= deadline)

    # Partially renewable resource constraints
    for r in range(num_resources):
        res = resources[r]
        capacity = res['capacity']
        demands = res['demands']
        periods = set(res['time_periods'])  # designated periods for this resource

        # Add constraints for each time period in the designated set
        if periods:
            for t in periods:
                # Calculate resource consumption at time t
                resource_usage_expr = 0
                for i in range(num_nodes):
                    demand = demands[i]
                    if demand > 0:
                        # Indicator: whether activity i uses this period t
                        # This is true if: start_i < t <= start_i + p_i
                        # Equivalent to: start_i < t AND t - p_i <= start_i
                        # Or: t - p_i <= start_i <= t - 1
                        
                        # We can represent this using binary variables and constraints
                        uses_period_t = model.addVar(vtype=GRB.BINARY, name=f'use_{i}_{t}')
                        
                        # If activity i's execution includes period t, then uses_period_t = 1
                        # This is done by the logical condition that: t is in (start_i, start_i + duration_i]
                        # Which is equivalent to: start_i < t <= start_i + duration_i
                        
                        # To model: uses_period_t == 1 iff start_i < t and t <= start_i + p_i 
                        # This is equivalent to:
                        # (start_i <= t - 1) AND (start_i >= t - p_i)
                        # 
                        # So:
                        #   t - p_i <= start_i + (1 - uses_period_t) * big_M  (1)
                        #   start_i <= t - 1 + (1 - uses_period_t) * big_M  (2)
                        #   
                        # And to prevent false positives (when conditions met but uses_period_t is 0):
                        #    start_i >= t - p_i - uses_period_t * big_M  (3)
                        #    start_i <= t - 1 + uses_period_t * big_M  (4)
                        
                        # Use big-M for domain [0, max_time]
                        big_m = max_time
                        
                        # If uses_period_t = 1, then both conditions of time period must hold
                        model.addConstr(start_times[i] + demand <= t + (1 - uses_period_t) * big_m)
                        model.addConstr(t - processing_times[i] <= start_times[i] + (1 - uses_period_t) * big_m)

                        # If uses_period_t = 0, at least one condition must be violated
                        # This is handled through the two inequalities above.
                        
                        resource_usage_expr += demand * uses_period_t
        
                model.addConstr(resource_usage_expr <= capacity, name=f'res_{r}_period_{t}')

    # Set callback to log incumbent solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    # Optimize the model
    if logger:
        model.optimize(mycallback)
    else:
        model.optimize()

    # Extract the solution
    if model.SolCount > 0:
        objective_value = int(round(model.objVal))
        start_times_solution = [int(round(start_times[i].X)) for i in range(num_nodes)]
        
        # Prepare the solution dictionary
        solution = {
            "objective_value": objective_value,
            "start_times": start_times_solution
        }
        
        # Write the solution to the file
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # In case no solution was found, return a suitable error state or empty result
        solution = {
            "objective_value": -1,  # indicates no solution found
            "start_times": []
        }
        
        # Write the solution to the file
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description="Solve Resource-Constrained Project Scheduling Problem with partially Renewable Capacity Timetabling")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the log file for incumbent solutions")

    args = parser.parse_args()
    solve_rcpsp_prct(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()