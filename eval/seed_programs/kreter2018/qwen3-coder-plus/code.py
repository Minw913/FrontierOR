import argparse
import json
import math
from collections import defaultdict
from solution_logger import SolutionLogger

def calculate_early_late_times(num_activities, activities_data, precedence_relations, max_project_duration):
    """Calculate earliest and latest start times through forward and backward passes."""
    
    # Get duration and resource requirements for all activities
    durations = [0] * num_activities
    for act_data in activities_data.values():
        durations[act_data["id"]] = act_data["duration"]
    
    # Build adjacency lists
    succs = defaultdict(list)
    preds = defaultdict(list)
    for pred, succ in precedence_relations:
        succs[pred].append(succ)
        preds[succ].append(pred)
    
    # Forward pass (earliest start times)
    es = [0] * num_activities
    # Topologically sorted order (for this case project start = 0 and project end likely = n-1)
    # So for now, let's do a full topo sort just in case
    
    in_degree = [0] * num_activities
    for pred, succ in precedence_relations:
        in_degree[succ] += 1
        
    queue = [i for i in range(num_activities) if in_degree[i] == 0]
    while queue:
        current = queue.pop(0)
        for next_act in succs[current]:
            in_degree[next_act] -= 1
            # Update ES(next_act) based on max finish time of predecessors
            es[next_act] = max(es[next_act], es[current] + durations[current])
            if in_degree[next_act] == 0:
                queue.append(next_act)

    # Backward pass (latest start times)
    ls = [0] * num_activities
    # Initialize all to a large positive value
    ls = [max_project_duration] * num_activities
    # End activity is the one with no successors; usually it's the highest numbered one
    # Find nodes with no successors - these need to end by the deadline
    has_successor = set()
    for pred, succ in precedence_relations:
        has_successor.add(pred)
        has_successor.add(succ)
    sinks = [i for i in range(num_activities) if i not in succs]
    # Among them, the one with max_project_duration constraint
    for sink in sinks:
        ls[sink] = max_project_duration - durations[sink]

    # Rebuild in_degrees for reverse
    out_degree = [0] * num_activities
    for pred, succ in precedence_relations:
        out_degree[pred] += 1

    queue = [i for i in range(num_activities) if succs[i] == [] and durations[i] > 0 or i == max([act["id"] for act in activities_data.values()])]  # Likely end node
    # Use more robust method: find ones with no outgoing edges
    queue = []
    for i in range(num_activities):
        if not succs[i]:  # no successors
            if i < num_activities - 1:  # not fictitious end activity (usually last)
                # Actually identify end by seeing which node is mentioned in constraints
                # Let's simply set all ends to max_proj_dur minus dur
                ls[i] = max(0, max_project_duration - durations[i])
    
    # To properly calculate LST in reverse, build reverse graph
    # Use reverse topo sort
    out_degree = [0] * num_activities
    rev_graph = [[] for _ in range(num_activities)]
    for pred, succ in precedence_relations:
        out_degree[pred] += 1
        rev_graph[succ].append(pred)
        
    # Start from end-nodes
    queue = []
    for i in range(num_activities):
        if succs[i] == []:
            ls[i] = max(0, max_project_duration - durations[i])
            queue.append(i)
    
    in_degree_reverse = out_degree[:]  # How many successors still remaining
    processing_queue = queue[:]
    while processing_queue:
        current = processing_queue.pop(0)
        for pred in rev_graph[current]:
            # Current activity is successor of pred
            # Current finishes at es[current] + dur, so pred finishes at most at that - dur[pred]
            ls[pred] = min(ls[pred], ls[current] - durations[pred])
            in_degree_reverse[pred] -= 1
            if in_degree_reverse[pred] == 0:
                processing_queue.append(pred)

    return es, ls


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

    # Convert activities to a dictionary mapping id -> activity object
    activity_dict = {act["id"]: act for act in activities}

    # Fictitious activities (project start and end) are numbered sequentially
    # The start activity is typically 0, and the end is typically last id in sequence
    # Fill missing in dict so all IDs exist
    for i in range(num_total_activities):
        if i not in activity_dict:
            activity_dict[i] = {"id": i, "duration": 0, "resource_requirements": [0] * num_resource_types}

    # Calculate earliest and latest start times
    es, ls = calculate_early_late_times(num_total_activities, activity_dict, precedence_relations, max_project_duration)
    
    # Create start time windows
    start_time_indices = {}
    for i in range(num_total_activities):
        duration = activity_dict[i]["duration"]
        es_i = es[i]
        ls_i = ls[i]
        
        # Ensure start time doesn't cause activity to go beyond deadline
        latest_start_validated = min(ls_i, max_project_duration - duration)
        
        if es_i <= latest_start_validated:
            # Only keep times in the range [es_i, latest_start_validated]
            start_time_indices[i] = list(range(es_i, latest_start_validated + 1))
        else:
            start_time_indices[i] = []

    # Import and create Gurobi model
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("rcpsp_discrete")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    
    # Variables: x[i,t] = 1 if activity i starts at time t
    x = {}
    for i in range(num_total_activities):
        if i not in x:
            x[i] = {}
        for t in start_time_indices[i]:
            x[i][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
    
    # Variables for resource provision for each resource type
    r = {}
    for k in range(num_resource_types):
        # We estimate upper bound for each resource based on max possible needs
        # Use 100*number of activities as rough bound to avoid very large domains for big instances
        max_resource_use_in_single_run = sum(
            max(activity_dict[j].get("resource_requirements", [0]*num_resource_types)[k] for k_idx in [k] if k_idx < len(activity_dict[j].get("resource_requirements", [])))
            for j in range(num_total_activities)
        )
        resource_ub = max(max_resource_use_in_single_run, 50)  # Minimum cap to prevent too small bounds
        
        r[k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=resource_ub, name=f"r_{k}")

    # Constraint: Each activity starts exactly once
    for i in range(num_total_activities):
        if len(start_time_indices[i]) > 0:
            model.addConstr(gp.quicksum(x[i][t] for t in start_time_indices[i]) == 1, name=f"assign_{i}")

    # Precedence constraints
    for prec, succ in precedence_relations:
        # Expected time when precursor finishes
        prec_finish_expr = gp.LinExpr()
        for t in start_time_indices[prec]:
            prec_finish_expr += (t + activity_dict[prec]["duration"]) * x[prec][t]

        succ_start_expr = gp.LinExpr()
        for t in start_time_indices[succ]:
            succ_start_expr += t * x[succ][t]
            
        model.addConstr(succ_start_expr >= prec_finish_expr, name=f"prec_{prec}_{succ}")

    # Resource capacity constraints: for each resource type & time period
    HORIZON = max_project_duration
    for k in range(num_resource_types):
        for t_period in range(HORIZON + 1):
            resource_demand = gp.LinExpr()
            for i in range(num_total_activities):
                req_i_k = activity_dict[i].get("resource_requirements", [0]*num_resource_types)[k]
                for s_time in start_time_indices[i]:
                    dur_i = activity_dict[i]["duration"]
                    # Activity i is active in period t_period if s_time <= t_period < s_time + dur_i
                    if s_time <= t_period < s_time + dur_i:
                        resource_demand += req_i_k * x[i][s_time]

            model.addConstr(resource_demand <= r[k], name=f"res_{k}_period_{t_period}")

    # Objective function: minimize total resource costs
    obj = gp.quicksum(resource_costs[k] * r[k] for k in range(num_resource_types))
    model.setObjective(obj, GRB.MINIMIZE)

    # Set up logger if provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objval)
    
    # Optimize
    model.optimize(callback if logger else None)
    
    # Extract solution
    solution = {
        "objective_value": float('inf'),
        "start_times": {},
        "resource_amounts": {}
    }

    # Handle different solution statuses
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        for i in range(num_total_activities):
            if i in x:
                for t in start_time_indices[i]:
                    if x[i][t].X > 0.9:
                        solution["start_times"][str(i)] = t
                        break
        
        # Add default start time for those not in x
        for i in range(num_total_activities):
            if str(i) not in solution["start_times"]:
                # Default to ES for consistency, but since this indicates no feasible start time in model, skip handling
                if len(start_time_indices[i]) > 0:
                    pass  # Handled above
                else:
                    # This should happen only for problematic cases, set to ES as fallback
                    solution["start_times"][str(i)] = es[i]
                    
        for k in range(num_resource_types):
            solution["resource_amounts"][str(k)] = int(r[k].X)
    elif model.SolCount > 0:
        # A feasible solution was found but with status not among main ones
        solution["objective_value"] = model.ObjVal
        for i in range(num_total_activities):
            if i in x:
                for t in start_time_indices[i]:
                    if x[i][t].X > 0.9:
                        solution["start_times"][str(i)] = t
                        break
        for k in range(num_resource_types):
            solution["resource_amounts"][str(k)] = int(r[k].X)
    else:
        # No solution found
        solution["objective_value"] = float('inf')

    # Output solution
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