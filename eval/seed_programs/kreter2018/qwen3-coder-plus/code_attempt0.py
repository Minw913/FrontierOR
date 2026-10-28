import argparse
import json
import math
from collections import defaultdict
from solution_logger import SolutionLogger

def calculate_early_late_times(num_activities, activities_data, precedence_relations, max_project_duration):
    # Create adjacency lists for graph traversal
    forward_adj = [[] for _ in range(num_activities)]
    backward_adj = [[] for _ in range(num_activities)]
    
    for pred, succ in precedence_relations:
        forward_adj[pred].append(succ)
        backward_adj[succ].append(pred)
    
    # Earliest start (forward pass)
    early_start = [0] * num_activities
    queue = [0]  # Start with project start node (ID 0 assumed to be start)
    visited = set(queue)
    
    while queue:
        current = queue.pop(0)
        current_early_finish = early_start[current] + activities_data[current]['duration']
        
        for next_activity in forward_adj[current]:
            early_start[next_activity] = max(early_start[next_activity], current_early_finish)
            if next_activity not in visited:
                visited.add(next_activity)
                queue.append(next_activity)

    # Latest start (backward pass)
    late_start = [max_project_duration] * num_activities
    
    # Find end activity (usually the node with highest topological position or with no successors)
    end_node = -1
    for i in range(len(forward_adj)):
        if not forward_adj[i]:  # has no outgoing edges
            end_node = i
            break
            
    # For the end node:
    end_id = end_node
    late_start[end_id] = max_project_duration - activities_data[end_id]['duration']
    # We'll adjust late start time to consider the actual possible finish time

    queue = [end_id]
    visited = set(queue)
    
    while queue:
        current = queue.pop(0)
        current_late_start_time = late_start[current] - activities_data[current]['duration']
        
        for prev_activity in backward_adj[current]:
            late_start[prev_activity] = min(late_start[prev_activity], current_late_start_time)
            if prev_activity not in visited:
                visited.add(prev_activity)
                queue.append(prev_activity)
                
    return early_start, late_start


def solve_rcpsp(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_real_activities = instance['num_real_activities']
    num_total_activities = instance['num_total_activities']
    num_resource_types = instance['num_resource_types']
    max_project_duration = instance['max_project_duration']
    activities = instance['activities']
    precedence_relations = instance['precedence_relations']
    resource_costs = instance['resource_costs']
    
    # Map activities by ID
    activities_map = {}
    for act in activities:
        activities_map[act['id']] = act
    
    activities_by_index = {}
    for idx, act in enumerate(activities):
        activities_by_index[idx] = act
    
    # Determine early/late times to get time bounds for each activity
    # Assuming that first activity has id 0 and is the start, last is end
    early_start, late_start = calculate_early_late_times(
        num_total_activities,
        {i: activities_by_index[i] for i in range(num_total_activities)},
        precedence_relations,
        max_project_duration
    )

    import gurobipy as gp
    from gurobipy import GRB
    
    # Calculate an upper bound for time horizon
    max_possible_early = max(early_start)
    max_possible_late = max(late_start)
    T = min(max_project_duration, max_possible_late + max_possible_early)
    
    # Create model
    model = gp.Model("rcpsp")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    
    # Binary variables x[i,t] = 1 if activity i starts at time t
    x_vars = {}         # x[i,t]
    r_vars = {}         # r[k] resource provision for resource k
    
    # Activity start time bounds
    for i in range(num_total_activities):
        # Skip if out of valid activities
        duration = activities_map.get(i, {}).get('duration', 0)
        res_req = activities_map.get(i, {}).get('resource_requirements', [0]*num_resource_types)
        
        # If it's a fictitious activity not in activities list, fill with defaults
        if i == 0:  # Project start
            duration = 0
            res_req = [0] * num_resource_types
        
        # If end activity not explicitly in the list, assume it's the one with no successors
        if i not in activities_map and len(r_vars) == 0 and any(i == prec[1] for prec in precedence_relations):
            pass  # Just making sure we handle correctly
        
        start_min = early_start[i]
        start_max = max(0, late_start[i])
        start_max = min(start_max, max_project_duration - duration)  # Ensure activity fits in deadline
        
        if start_min <= start_max:
            x_vars[i] = {}
            for t in range(start_min, start_max + 1):
                x_vars[i][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # Resource provisions
    max_resource_req_sum = []
    for k in range(num_resource_types):
        # Sum of all demands per period could be max of resource requirements across all activities
        total_max = sum(activities_map.get(i, {}).get('resource_requirements', [0]*num_resource_types)[k] 
                       for i in range(num_total_activities))
        max_k = min(int(sum(activities_map.get(i, {}).get('resource_requirements', [0]*num_resource_types)[k] 
                           for i in range(num_total_activities))), 
                    1000)  # A reasonable bound might prevent scalability issues
        
        upper_bound = 0
        for i in range(num_total_activities):
            res_req_for_k = activities_map.get(i, {}).get('resource_requirements', [0]*num_resource_types)
            if k < len(res_req_for_k):
                upper_bound = max(upper_bound, res_req_for_k[k])
        upper_bound *= num_total_activities  # Heuristic upper bound
        r_vars[k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=upper_bound, name=f"r_{k}")

    # Each activity starts exactly once if it's in x_vars range
    for i in range(num_total_activities):
        if i in x_vars and len(x_vars[i]) > 0:
            model.addConstr(gp.quicksum(x_vars[i][t] for t in x_vars[i]) == 1, name=f"start_once_{i}")
    
    # Precedence constraints
    for pred, succ in precedence_relations:
        # Time when pred finishes = start_time(pred) + duration(pred)
        pred_expr = gp.LinExpr()
        for t in x_vars.get(pred, {}):
            pred_expr += t * x_vars[pred][t]
        
        succ_expr = gp.LinExpr()
        for t in x_vars.get(succ, {}):
            succ_expr += t * x_vars[succ][t]
            
        # Ensure succ starts after pred finishes
        duration_pred = activities_map.get(pred, {'duration': 0})['duration']
        model.addConstr(succ_expr >= pred_expr + duration_pred, name=f"prec_{pred}_{succ}")
    
    # Resources per period constraints
    for k in range(num_resource_types):
        # For each period p
        for p in range(max_project_duration + 1):
            resource_usage_expr = gp.LinExpr()
            for i in range(num_total_activities):
                for t in x_vars.get(i, {}):
                    # For activity i starting at t, it's active in [t, t+duration_i-1]
                    dur_i = activities_map.get(i, {'duration': 0})['duration']
                    res_req_k = activities_map.get(i, {}).get('resource_requirements', [0]*num_resource_types)[k]
                    if t <= p <= t + dur_i - 1:
                        resource_usage_expr += res_req_k * x_vars[i][t]

            model.addConstr(resource_usage_expr <= r_vars[k], name=f"resource_cons_{k}_{p}")

    # Objective: minimize total resource cost
    obj_expr = gp.LinExpr()
    for k in range(num_resource_types):
        obj_expr += resource_costs[k] * r_vars[k]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Set up logger if needed
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objval)
                
    # Solve the model
    model.optimize(callback if logger else None)

    # Prepare solution
    solution = {
        "start_times": {},
        "resource_amounts": {}
    }

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract start times
        for i in range(num_total_activities):
            for t in x_vars.get(i, {}):
                if x_vars[i][t].x > 0.9:  # Binary variable is essentially 1
                    solution["start_times"][str(i)] = int(t)
                    break
            else:
                # This happens when there are no feasible start times in the range (e.g., fictitious nodes)
                if str(i) not in solution["start_times"]:
                    if i == 0:  # Assumed to start at 0
                        solution["start_times"][str(i)] = 0

        # Extract resource amounts
        for k in range(num_resource_types):
            solution["resource_amounts"][str(k)] = int(r_vars[k].x)

        solution["objective_value"] = model.objVal if model.solCount > 0 else float('inf')
    else:
        solution["objective_value"] = float('inf')
        solution["start_times"] = {}
        solution["resource_amounts"] = {}

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_rcpsp(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )