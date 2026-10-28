import argparse
import json
import time
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_real = data['num_real_activities']
    num_total = data['num_activities_total']
    num_resources = data['num_resources']
    max_duration = data['max_project_duration']
    
    activities = {}
    for act in data['activities']:
        activities[act['id']] = act['processing_time']
    
    temporal_constraints = data['temporal_constraints']
    resources = data['resources']
    
    # Activity IDs: 0 (start), 1..num_real (real), num_real+1 (end)
    project_start = 0
    project_end = num_real + 1
    
    # Try using Gurobi MIP
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 1:
        remaining_time = 1
    
    model = gp.Model("RCPSP_PR")
    model.setParam('TimeLimit', remaining_time - 1)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables: start times
    S = {}
    for aid in activities:
        if aid == project_start:
            S[aid] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name=f"S_{aid}")
        elif aid == project_end:
            S[aid] = model.addVar(lb=0, ub=max_duration, vtype=GRB.INTEGER, name=f"S_{aid}")
        else:
            S[aid] = model.addVar(lb=0, ub=max_duration, vtype=GRB.INTEGER, name=f"S_{aid}")
    
    # Objective: minimize start time of project end
    model.setObjective(S[project_end], GRB.MINIMIZE)
    
    # Temporal constraints
    for tc in temporal_constraints:
        fr = tc['from_activity']
        to = tc['to_activity']
        lag = tc['time_lag']
        model.addConstr(S[to] - S[fr] >= lag, name=f"temp_{fr}_{to}")
    
    # Binary variables for activity execution in each period
    # Activity i is executing in period t if S[i] + 1 <= t <= S[i] + p[i]
    # i.e., S[i] <= t - 1 and S[i] >= t - p[i]
    # i.e., t - p[i] <= S[i] <= t - 1
    
    # For partially renewable resources, we need to know which activities
    # are executing in which periods from the resource's designated subset.
    
    # For each resource, for each activity with nonzero demand, for each period in Pi_k,
    # create a binary variable x[i,t] = 1 if activity i is executing in period t
    
    # Collect which (activity, period) pairs we need
    needed_pairs = set()
    for res in resources:
        Pi_k = res['Pi_k']
        demands = res['demands']
        for aid_str, dem in demands.items():
            aid = int(aid_str)
            if dem > 0 and activities[aid] > 0:
                for t in Pi_k:
                    needed_pairs.add((aid, t))
    
    # Create binary variables for needed pairs
    x = {}
    for (aid, t) in needed_pairs:
        p_i = activities[aid]
        # Activity aid executes in period t iff t - p_i <= S[aid] <= t - 1
        # If t - 1 < 0 or t - p_i > max_duration, then impossible
        lb_s = max(0, t - p_i)
        ub_s = t - 1
        if ub_s < 0 or lb_s > max_duration:
            # This pair is impossible, x = 0
            x[(aid, t)] = 0
        else:
            x[(aid, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{aid}_{t}")
    
    model.update()
    
    # Link x variables to S variables
    # x[i,t] = 1 iff S[i] >= t - p_i and S[i] <= t - 1
    # We use indicator-style or big-M constraints
    
    # For each binary x[i,t]:
    # x[i,t] = 1 => S[i] >= t - p_i  AND  S[i] <= t - 1
    # x[i,t] = 0 => S[i] < t - p_i  OR  S[i] > t - 1
    #   which is: NOT (S[i] >= t - p_i AND S[i] <= t-1)
    
    # This is equivalent to:
    # S[i] >= t - p_i - M*(1 - x[i,t])
    # S[i] <= t - 1 + M*(1 - x[i,t])
    # And if x[i,t] = 0, at least one must be violated:
    # S[i] <= t - p_i - 1 + M*x[i,t]   OR   S[i] >= t + M*(... not easy)
    
    # Better approach: x[i,t] = 1 iff t - p_i <= S[i] <= t - 1
    # Let's define two helper binary variables or use a different formulation.
    
    # Actually, let's use the formulation:
    # Define y[i,t] = 1 if S[i] <= t-1 (i.e., activity started by period t)
    # Define z[i,t] = 1 if S[i] <= t - p_i - 1 (i.e., activity finished before period t)
    # Then x[i,t] = y[i,t] - z[i,t]  (but z might not be needed for all)
    
    # This gets complex. Let me use a different approach.
    # 
    # For each activity with nonzero processing time that has demand on some resource,
    # define binary variables: u[i,s] = 1 if activity i starts at time s.
    # Then S[i] = sum_s s * u[i,s], and sum_s u[i,s] = 1.
    # x[i,t] = sum_{s=t-p_i}^{t-1} u[i,s]
    
    # This might create many variables but is clean.
    
    # Let's figure out which activities need this treatment
    activities_needing_binary = set()
    for (aid, t) in needed_pairs:
        if isinstance(x.get((aid, t), 0), gp.Var):
            activities_needing_binary.add(aid)
    
    # Remove x variables, we'll reformulate
    model.remove([v for v in x.values() if isinstance(v, gp.Var)])
    x = {}
    model.update()
    
    # For activities needing binary decomposition, create u[i,s] variables
    u = {}
    for aid in activities_needing_binary:
        p_i = activities[aid]
        # Determine feasible start time range from temporal constraints
        # For now use [0, max_duration - p_i] but could be tighter
        earliest = 0
        latest = max_duration  # could be max_duration - p_i but let constraints handle it
        
        # Compute tighter bounds using forward/backward pass on temporal constraints
        # Simple approach: just use 0 to max_duration
        for s in range(0, max_duration - p_i + 1):
            u[(aid, s)] = model.addVar(vtype=GRB.BINARY, name=f"u_{aid}_{s}")
        
        # sum u[i,s] = 1
        model.addConstr(
            gp.quicksum(u[(aid, s)] for s in range(0, max_duration - p_i + 1)) == 1,
            name=f"assign_{aid}"
        )
        
        # Link S[i] = sum s * u[i,s]
        model.addConstr(
            S[aid] == gp.quicksum(s * u[(aid, s)] for s in range(0, max_duration - p_i + 1)),
            name=f"link_S_{aid}"
        )
    
    model.update()
    
    # Now define x[i,t] for needed pairs
    for (aid, t) in needed_pairs:
        p_i = activities[aid]
        lb_s = max(0, t - p_i)
        ub_s = min(t - 1, max_duration - p_i)
        if ub_s < lb_s or ub_s < 0:
            x[(aid, t)] = 0
        else:
            # x[i,t] = sum_{s=lb_s}^{ub_s} u[i,s]
            vars_in_range = [u[(aid, s)] for s in range(lb_s, ub_s + 1) if (aid, s) in u]
            if len(vars_in_range) == 0:
                x[(aid, t)] = 0
            else:
                x[(aid, t)] = gp.quicksum(vars_in_range)
    
    # Resource constraints
    for res in resources:
        Pi_k = res['Pi_k']
        capacity = res['capacity']
        demands = res['demands']
        
        terms = []
        for aid_str, dem in demands.items():
            aid = int(aid_str)
            if dem > 0 and activities[aid] > 0:
                for t in Pi_k:
                    val = x.get((aid, t), 0)
                    if isinstance(val, (int, float)):
                        if val > 0:
                            terms.append(dem * val)
                    else:
                        terms.append(dem * val)
        
        if terms:
            # Separate constant and variable terms
            const_sum = sum(t for t in terms if isinstance(t, (int, float)))
            var_terms = [t for t in terms if not isinstance(t, (int, float))]
            
            if var_terms:
                model.addConstr(
                    gp.quicksum(var_terms) + const_sum <= capacity,
                    name=f"res_{res['resource_id']}"
                )
            else:
                if const_sum > capacity:
                    # Infeasible
                    pass
    
    model.update()
    
    # Check model size - if too large, try to compute bounds first
    num_vars = model.NumVars
    num_constrs = model.NumConstrs
    
    # Callback to log solutions
    best_obj = [float('inf')]
    best_schedule = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGetSolution(S[project_end])
            if obj < best_obj[0]:
                best_obj[0] = obj
                schedule = {}
                for aid in activities:
                    schedule[str(aid)] = int(round(model.cbGetSolution(S[aid])))
                best_schedule[0] = schedule
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    if model.SolCount > 0:
        schedule = {}
        for aid in activities:
            schedule[str(aid)] = int(round(S[aid].X))
        obj_val = int(round(S[project_end].X))
    elif best_schedule[0] is not None:
        schedule = best_schedule[0]
        obj_val = int(best_obj[0])
    else:
        # Fallback: use max_duration as project end
        # Try a simple heuristic
        schedule = {}
        schedule[str(project_start)] = 0
        schedule[str(project_end)] = max_duration
        for aid in activities:
            if aid != project_start and aid != project_end:
                schedule[str(aid)] = 0
        obj_val = max_duration
    
    solution = {
        "objective_value": float(obj_val),
        "schedule": schedule
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)