import argparse
import bisect
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n_real = instance['num_real_activities']
    n_total = instance['num_activities_total']
    n_res = instance['num_resources']
    T_max = instance['max_project_duration']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources_list = instance['resources']
    
    # Extract processing times
    p = [0] * n_total
    for act in activities:
        i = act['id']
        p[i] = act['processing_time']
    
    # End activity ID is n_real + 1
    end_activity_id = n_real + 1
    
    # Build resource data: capacities, Pi_k, and demands
    capacities = [0] * n_res
    Pi_k = [None] * n_res
    d = [[0] * n_res for _ in range(n_total)]
    
    for k, res in enumerate(resources_list):
        capacities[k] = res['capacity']
        Pi_k[k] = sorted(res['Pi_k'])
        demands_dict = res['demands']
        for i in range(n_total):
            key = str(i)
            if key in demands_dict:
                d[i][k] = demands_dict[key]
    
    # Compute earliest start times (ES) using Bellman-Ford
    ES = [-10**9] * n_total
    ES[0] = 0  # Project-start at time 0
    
    for _ in range(n_total - 1):
        updated = False
        for edge in temporal_constraints:
            i = edge['from_activity']
            j = edge['to_activity']
            lag = edge['time_lag']
            if ES[i] + lag > ES[j]:
                ES[j] = ES[i] + lag
                updated = True
        if not updated:
            break
    
    # Compute latest start times (LS) using Bellman-Ford in reverse
    LS = [10**9] * n_total
    LS[0] = 0
    LS[end_activity_id] = T_max  # Deadline constraint
    
    for _ in range(n_total - 1):
        updated = False
        for edge in temporal_constraints:
            i = edge['from_activity']
            j = edge['to_activity']
            lag = edge['time_lag']
            if LS[j] < 10**9:  # If not infinity
                candidate = LS[j] - lag
                if candidate < LS[i]:
                    LS[i] = candidate
                    updated = True
        if not updated:
            break
    
    # Check temporal feasibility
    temporally_infeasible = False
    for i in range(n_total):
        if ES[i] > LS[i]:
            temporally_infeasible = True
            break
    
    # Fallback schedule: earliest start times
    fallback_schedule = {i: ES[i] for i in range(n_total)}
    fallback_obj = ES[end_activity_id]
    
    if temporally_infeasible:
        solution = {
            "objective_value": fallback_obj,
            "schedule": fallback_schedule
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log(fallback_obj)
        return
    
    # Precompute resource coefficients for the MIP model
    resource_terms = [[] for _ in range(n_res)]  # For each resource: list of (i, s, coeff)
    
    for i in range(n_total):
        for k in range(n_res):
            periods = Pi_k[k]
            for s in range(ES[i], LS[i] + 1):
                a = s + 1
                b = s + p[i]
                left = bisect.bisect_left(periods, a)
                right = bisect.bisect_right(periods, b)
                count = right - left
                coeff = d[i][k] * count
                if coeff > 0:
                    resource_terms[k].append((i, s, coeff))
    
    # Build and solve MIP model
    try:
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', args.time_limit)
        model.setParam('Threads', 1)
        
        # Create variables
        x = {}
        for i in range(n_total):
            for s in range(ES[i], LS[i] + 1):
                x[i, s] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{s}')
        
        # Assignment constraints
        for i in range(n_total):
            model.addConstr(
                gp.quicksum(x[i, s] for s in range(ES[i], LS[i] + 1)) == 1,
                name=f'assignment_{i}'
            )
        
        # Temporal constraints
        for edge in temporal_constraints:
            i = edge['from_activity']
            j = edge['to_activity']
            lag = edge['time_lag']
            expr = gp.quicksum(s * x[j, s] for s in range(ES[j], LS[j] + 1)) - \
                   gp.quicksum(s * x[i, s] for s in range(ES[i], LS[i] + 1)) >= lag
            model.addConstr(expr, name=f'temp_{i}_{j}')
        
        # Resource constraints
        for k in range(n_res):
            if resource_terms[k]:
                total = gp.quicksum(coeff * x[i, s] for (i, s, coeff) in resource_terms[k])
                model.addConstr(total <= capacities[k], name=f'res_{k}')
        
        # Objective: minimize project duration (start time of end activity)
        obj = gp.quicksum(s * x[end_activity_id, s] for s in range(ES[end_activity_id], LS[end_activity_id] + 1))
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Optimize with callback for logging incumbents
        if logger:
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(objval)
            model.optimize(callback)
        else:
            model.optimize()
        
        # Extract solution if available
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            schedule = {}
            for i in range(n_total):
                for s in range(ES[i], LS[i] + 1):
                    if abs(x[i, s].x - 1.0) < 1e-6:
                        schedule[i] = s
            obj_val = schedule[end_activity_id]
            solution = {
                "objective_value": float(obj_val),
                "schedule": schedule
            }
        else:
            solution = {
                "objective_value": fallback_obj,
                "schedule": fallback_schedule
            }
    except gp.GurobiError as e:
        # Fallback in case of any error during MIP setup/solve
        solution = {
            "objective_value": fallback_obj,
            "schedule": fallback_schedule
        }
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    # Log fallback if not logged during optimization
    if logger and solution["objective_value"] == fallback_obj:
        logger.log(fallback_obj)

if __name__ == '__main__':
    main()