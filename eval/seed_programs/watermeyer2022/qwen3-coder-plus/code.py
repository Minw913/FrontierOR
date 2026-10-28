import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description='RCPSP-PR Solver')
    parser.add_argument('--instance_path', required=True, help='Path to the JSON instance file')
    parser.add_argument('--solution_path', required=True, help='Path to save the JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum time to run solver in seconds')
    parser.add_argument('--log_path', required=False, help='Path to log solutions')

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Extract data from instance
    num_real_activities = data['num_real_activities']
    num_activities_total = data['num_activities_total']
    num_resources = data['num_resources']
    max_duration = data['max_project_duration']
    activities_data = {a['id']: a for a in data['activities']}
    temporal_constraints = data['temporal_constraints']
    resources_data = data['resources']

    # Build Gurobi model
    model = gp.Model("RCPSP-PR")
    model.setParam("Threads", 1)
    
    # Set Gurobi parameters
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.LogToConsole, 0)

    # Callback function for logging if needed
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj)

    # Decision variables: Start time for each activity
    T = max_duration
    start_vars = {}
    for i in range(num_activities_total):
        activity_duration = activities_data[i]['processing_time']
        max_start_time = max(0, T - activity_duration) if activity_duration <= T else 0
        start_var = model.addVar(
            vtype=GRB.INTEGER,
            name=f"S_{i}",
            lb=0, 
            ub=max_start_time
        )
        start_vars[i] = start_var

    # Constraint 1: Project start activity must start at time 0
    model.addConstr(start_vars[0] == 0, name="project_start_at_0")

    # Constraint 2: Temporal constraints (start-to-start with lags)
    for idx, tc in enumerate(temporal_constraints):
        from_activity = tc['from_activity']
        to_activity = tc['to_activity']
        time_lag = tc['time_lag']
        model.addConstr(
            start_vars[to_activity] >= start_vars[from_activity] + time_lag,
            name=f"temporal_{idx}"
        )

    # Constraint 3: Partially renewable resource constraints
    for res_idx, res in enumerate(resources_data):
        res_periods = set(res['Pi_k'])
        res_capacity = res['capacity']
        
        # Compute resource consumption for each activity
        total_resource_usage = []
        
        for act_idx in range(num_activities_total):
            if act_idx not in res['demands']:
                continue
            
            demand = res['demands'][str(act_idx)]
            if demand == 0:
                continue
            
            activity_duration = activities_data[act_idx]['processing_time']
            
            # For each possible start time, compute how many resource periods are consumed
            if activity_duration == 0:  # Fictitious activities
                consumed_periods = 0
            else:
                # Use a piecewise linear relaxation or a direct computation
                # Use binaries to represent start times
                possible_starts = range(0, T - activity_duration + 1) if activity_duration <= T else []
                
                if activity_duration == 0:  # Handle zero duration as special case
                    possible_starts = range(0, T + 1)
                
                binary_vars = []
                for s in possible_starts:
                    binary = model.addVar(vtype=GRB.BINARY, name=f"start_binary_{act_idx}_{s}_res_{res_idx}")
                    binary_vars.append((binary, s))
                
                if binary_vars:
                    # Exactly one start time chosen
                    model.addConstr(gp.quicksum(b for b, _ in binary_vars) == 1, name=f"exactly_one_start_{act_idx}_res_{res_idx}")
                    
                    # Link start variable to binaries
                    model.addConstr(
                        start_vars[act_idx] == gp.quicksum(s * b for b, s in binary_vars),
                        name=f"link_start_to_binaries_{act_idx}_res_{res_idx}"
                    )
                    
                    # Calculate resource consumption based on start time
                    local_usage_per_binary = []
                    
                    for b_var, start_t in binary_vars:
                        # Calculate how many periods of resource will be consumed
                        # The activity is executed from [start_t, start_t + duration)
                        executed_periods = set()
                        if activity_duration > 0:
                            executed_periods = set(range(start_t, min(start_t + activity_duration, T + 50)))  # Safe upper bound
                        # Count intersection with resource availability periods
                        consumed_count = len(executed_periods.intersection(res_periods))
                        local_usage = demand * consumed_count * b_var
                        local_usage_per_binary.append(local_usage)
                    
                    # Total consumption for this activity
                    if local_usage_per_binary:
                        act_total_usage = gp.quicksum(local_usage_per_binary)
                        total_resource_usage.append(act_total_usage)
        
        # Add total resource constraint
        if total_resource_usage:
            model.addConstr(
                gp.quicksum(total_resource_usage) <= res_capacity,
                name=f"resource_capacity_{res_idx}"
            )

    # Objective: Minimize start time of project end activity (which is num_real_activities + 1)
    model.setObjective(start_vars[num_real_activities + 1], GRB.MINIMIZE)

    # Optimize the model
    model.optimize(my_callback)

    # Extract and prepare solution
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        objective_value = model.ObjVal
        schedule = {}
        for i in range(num_activities_total):
            schedule[i] = int(round(start_vars[i].X))
    else:
        # Infeasible or timeout with no feasible solution
        objective_value = float('inf')
        schedule = {}

    solution = {
        "objective_value": float(objective_value),
        "schedule": schedule
    }

    # Write the solution to the output file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()