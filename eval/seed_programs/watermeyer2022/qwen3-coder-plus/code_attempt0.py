import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_rcpsp_pr(data):
    """
    Solve Resource-Constrained Project Scheduling Problem with Partially Renewable Resources
    """

    # Extract data from instance
    num_real_activities = data['num_real_activities']
    max_duration = data['max_project_duration']
    activities_data = {a['id']: a for a in data['activities']}
    
    # Create activity objects with all necessary components included
    activities = {}
    for a in data['activities']:
        activ_id = a['id']
        activities[activ_id] = {
            'id': activ_id,
            'duration': a['processing_time'],
            'start': 0,  # Will be decided by the optimization
            'demands': {}  # Will populate demand info by resource for this activity
        }
    
    # Process resource constraints 
    resources = []
    for res_data in data['resources']:
        resource_info = {
            'id': res_data['resource_id'],
            'periods': set(res_data['Pi_k']),  # Time periods where this resource applies
            'capacity': res_data['capacity'],
            'demands_by_activity': res_data['demands']  # Maps activity -> per_period_demand
        }
        resources.append(resource_info)
        
        # Update activity demand information
        for act_id_str, demand_val in res_data['demands'].items():
            act_id = int(act_id_str)
            if act_id >= 0 and act_id <= num_real_activities + 1:
                activities[act_id]['demands'][res_data['resource_id']] = demand_val

    # Process temporal constraints
    temporal_constraints = data['temporal_constraints']
    
    # Start optimization model using Gurobi
    model = gp.Model("RCPSP-PR")
    
    # Set Gurobi parameters related to time limits
    model.setParam(GRB.Param.TimeLimit, data.get('time_limit', float('inf')))
    model.setParam(GRB.Param.LogToConsole, 0)  # Suppress console logs, handle with our SolutionLogger
    
    # Variables: For each activity, we create a variable for its start time
    T = max_duration
    start_vars = {}
    for i in range(num_real_activities + 2):  # Activity IDs: 0 .. num_real_activities+1
        # The upper bound on start time could be T, but often making it smaller helps
        max_start_time = T - activities[i]['duration'] if activities[i]['duration'] <= T else 0
        if activities[i]['duration'] == 0:  # Fictitious activities have zero duration
            max_start_time = T
            
        start_vars[i] = model.addVar(
            vtype=GRB.INTEGER,
            name=f"S_{i}",
            lb=0, ub=max_start_time
        )
    
    # Constraint 1: Project Start must happen at time 0
    model.addConstr(start_vars[0] == 0, name="project_start_at_0")
    
    # Constraint 2: Temporal constraints (Start-to-Start with lags)
    for idx, tc in enumerate(temporal_constraints):
        # S[to] - S[from] >= lag
        # => S[to] >= S[from] + lag
        from_activity = tc['from_activity']
        to_activity = tc['to_activity']
        time_lag = tc['time_lag']
        
        # We add the constraint S[to_activity] >= S[from_activity] + time_lag
        model.addConstr(
            start_vars[to_activity] >= start_vars[from_activity] + time_lag,
            name=f"temporal_{idx}"
        )
        
    # Prepare BigM values for resource constraints
    # We need binary variables for overlap checking - if activity j starts during a specific frame around activity i
    resource_vars = {} # Store binary indicators for each activity-resource-time combination
    
    # Constraint 3: Partially Renewable Resources
    for r_idx, resource in enumerate(resources):
        # For resource r, the total consumption of activity i is:
        # demand(i, r) * |execution_periods_for_activity_i INTERSECT Pi_r|
        
        # To compute how many periods a task i overlaps with the Pi_k set:
        # Let O_i(t) = |{ p \in Pi_k : t < p <= t + d_i }|, where t is start of activity i and d_i is duration
        
        # To handle non-linear term, we precalculate for each activity i and possible starting time s
        # O_i(s, k) = how much activity i starting at s consumes resource k (from its Pi_k set)
        resource_consumption_expr_parts = []
        
        for activity_id in range(num_real_activities + 2):
            duration = activities[activity_id]['duration']
            
            # We need to determine how much of resource `r` a job consumes given its start time
            # This creates an indicator constraint to calculate resource consumption
            
            # Big-M approach might be complex here due to time-indexing.
            # A different approach is to create helper variables and constraints
            # We'll use piecewise functions or indicator constraints if available.
            
            # Let O_i_s_k be the number of periods resource is consumed when activity i starts at period s
            # So if activity i starts at S_i, the resource usage is:
            # demand(i,r) * O_i_{S_i}_k
            
            # We will introduce variables and constraints to linearize this relationship
            # Create a variable for each resource and activity representing total resource consumption
            
            # Determine all unique start points relevant for resource calculation
            # This is the max period where this activity could conceivably start
            max_start = T - duration if duration <= T else 0
            if duration == 0:  # Special cases for durations of zero
                max_start = T
                
            periods_in_horizon = set(range(T + 1))
            possible_starts_for_activity = range(max_start + 1)
            
            # Calculate the resource consumption for activity `activity_id` for each possible start time s
            O_map = {}  # Map for calculating periods of overlap for this activity
            for s in possible_starts_for_activity:
                exec_end = s + duration  # Activity executes from [s+1, exec_end] for non-fictitious
                if exec_end > T:
                    continue # Would overflow
                exec_periods = set()
                if duration > 0:
                    exec_periods = set(range(max(0, s), min(exec_end, T + 1)))
                
                intersection = exec_periods.intersection(resource['periods'])
                O_i_at_s = len(intersection)
                
                O_map[s] = O_i_at_s
            
            if O_map:
                demand = activities[activity_id]['demands'].get(resource['id'], 0)
                if demand > 0:
                    # We want to link the continuous decision (consumption) of activity `activity_id` with resource r
                    # We add constraints using big-M formulations with binaries to connect start time and consumption
                    
                    # Create a disjunction over all possible start times
                    binaries = []
                    for s, overlap in O_map.items():
                        b = model.addVar(vtype=GRB.BINARY, name=f'aux_start_{activity_id}_at_{s}_for_res_{r_idx}')
                        binaries.append((b, s, overlap))
                    
                    # Exactly one start time selection for this activity
                    model.addConstr(gp.quicksum(b for b, s, o in binaries) == 1, name=f'exactly_one_start_{activity_id}_res_{r_idx}')
                    
                    # Link binaries to actual start variable
                    # S[activity_id] = sum_s ( s * b_s )
                    model.addConstr(
                        start_vars[activity_id] == gp.quicksum(s * b for b, s, o in binaries),
                        name=f'link_start_to_binaries_{activity_id}_res_{r_idx}'
                    )
                    
                    # Total consumption of this activity for resource r
                    # Consumption_{i,r} = Sum_s (b_s * O_{i,s,r} * demand_{i,r})
                    total_contribution = gp.quicksum(demand * overlap * b for b, s, overlap in binaries)
                    resource_consumption_expr_parts.append(total_contribution)
        
        # Finally, ensure the total resource capacity is respected
        if resource_consumption_expr_parts:
            consumption_expr = gp.quicksum(resource_consumption_expr_parts)
            model.addConstr(consumption_expr <= resource['capacity'], name=f'resource_capacity_{r_idx}')

    # Objective: Minimize the project completion time (start time of the project_end activity)
    objective = start_vars[num_real_activities + 1]
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Solve the model
    model.optimize()
    
    # Check status
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract solution
        schedule = {}
        for activity_id in range(num_real_activities + 2):
            schedule[activity_id] = int(round(start_vars[activity_id].x))
        
        # Ensure the project completion time respects the deadline
        best_objective = min(model.objVal, max_duration) if model.solCount > 0 else float('inf')
        
        return best_objective, schedule
    
    else:
        # No solution found within limits
        return float('inf'), {}


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
    
    # Modify data to add time limit for solver
    data['time_limit'] = args.time_limit

    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback to log every time a new incumbent solution is found
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Call the solver function
    try:
        # We need to pass the callback to the optimization if logging is needed
        if logger:
            # We'll adjust implementation to allow for callback integration
            # Since our solver now incorporates all logic in one model, let's adjust it to allow callbacks
            
            # Extract data
            num_real_activities = data['num_real_activities']
            max_duration = data['max_project_duration']
            activities_data = {a['id']: a for a in data['activities']}

            # Create activity objects with all necessary components included
            activities = {}
            for a in data['activities']:
                activ_id = a['id']
                activities[activ_id] = {
                    'id': activ_id,
                    'duration': a['processing_time'],
                    'start': 0,
                    'demands': {}
                }

            # Process resource constraints 
            resources = []
            for res_data in data['resources']:
                resource_info = {
                    'id': res_data['resource_id'],
                    'periods': set(res_data['Pi_k']),
                    'capacity': res_data['capacity'],
                    'demands_by_activity': res_data['demands']
                }
                resources.append(resource_info)

                # Update activity demand information
                for act_id_str, demand_val in res_data['demands'].items():
                    act_id = int(act_id_str)
                    if act_id >= 0 and act_id <= num_real_activities + 1:
                        activities[act_id]['demands'][res_data['resource_id']] = demand_val

            temporal_constraints = data['temporal_constraints']

            # Start optimization model using Gurobi
            model = gp.Model("RCPSP-PR")

            # Set Gurobi parameters
            model.setParam(GRB.Param.TimeLimit, args.time_limit)
            model.setParam(GRB.Param.LogToConsole, 0)

            # Variables
            T = max_duration
            start_vars = {}
            for i in range(num_real_activities + 2):
                max_start_time = T - activities[i]['duration'] if activities[i]['duration'] <= T else 0
                if activities[i]['duration'] == 0:
                    max_start_time = T

                start_vars[i] = model.addVar(
                    vtype=GRB.INTEGER,
                    name=f"S_{i}",
                    lb=0, ub=max_start_time
                )

            # Constraint 1: Project Start must happen at time 0
            model.addConstr(start_vars[0] == 0, name="project_start_at_0")

            # Constraint 2: Temporal constraints
            for idx, tc in enumerate(temporal_constraints):
                from_activity = tc['from_activity']
                to_activity = tc['to_activity']
                time_lag = tc['time_lag']

                model.addConstr(
                    start_vars[to_activity] >= start_vars[from_activity] + time_lag,
                    name=f"temporal_{idx}"
                )

            # Constraint 3: Partially Renewable Resources
            # Pre-process resource requirements
            for r_idx, resource in enumerate(resources):
                resource_vars = []
                
                for activity_id in range(num_real_activities + 2):
                    duration = activities[activity_id]['duration']
                    demand = activities[activity_id]['demands'].get(resource['id'], 0)
                    
                    if demand == 0:
                        continue
                    
                    max_start = T - duration if duration <= T else 0
                    if duration == 0:
                        max_start = T
                        
                    possible_starts = list(range(max_start + 1)) if max_start >= 0 else []
                    
                    if not possible_starts:
                        continue

                    # Calculation map
                    O_map = {}
                    for s in possible_starts:
                        exec_periods = set()
                        if duration > 0:
                            exec_periods = set(range(max(s + 1), min(s + 1 + duration, T + 2)))
                        
                        intersection = exec_periods.intersection(resource['periods'])
                        O_map[s] = len(intersection)

                    if not O_map:
                        continue
                    
                    # Introduce binaries: x[a_id][s] = 1 if activity starts at s
                    binaries = {}
                    for s in O_map.keys():
                        binaries[s] = model.addVar(vtype=GRB.BINARY, name=f'starts_{activity_id}_{s}')
                    
                    # Activity must start exactly once
                    if binaries:
                        model.addConstr(gp.quicksum(binaries.values()) == 1, name=f'assign_activity_{activity_id}_to_start_r_{r_idx}')
                        
                        # Link with start var
                        model.addConstr(
                            start_vars[activity_id] == gp.quicksum(s * binaries[s] for s in binaries),
                            name=f'link_start_{activity_id}_r_{r_idx}'
                        )
                        
                        # Add contribution to resource consp
                        contrib = gp.quicksum(demand * O_map[s] * binaries[s] for s in binaries)
                        resource_vars.append(contrib)
                
                if resource_vars:
                    model.addConstr(gp.quicksum(resource_vars) <= resource['capacity'], name=f'resource_cap_{r_idx}')

            # Objective: Minimize the project end activity start time
            objective = start_vars[num_real_activities + 1]
            model.setObjective(objective, GRB.MINIMIZE)

            # Optimize with callback
            model.optimize(my_callback)
            
            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.solCount > 0:
                schedule = {}
                for activity_id in range(num_real_activities + 2):
                    schedule[activity_id] = int(round(start_vars[activity_id].x))
                
                solution = {
                    "objective_value": start_vars[num_real_activities + 1].x,
                    "schedule": schedule
                }
            else:
                # No feasible solution found
                solution = {
                    "objective_value": float('inf'),
                    "schedule": {}
                }

        else:
            # No logging needed, solve normally
            objective_value, schedule = solve_rcpsp_pr(data)
            
            solution = {
                "objective_value": float(objective_value),
                "schedule": schedule
            }

        # Write solution to file
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

        print("Solution saved to:", args.solution_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    main()