import json
import argparse
from collections import deque
import gurobipy as gp
from gurobipy import GRB
import os
import sys
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main(instance_path, solution_path, time_limit, log_path=None):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_real_activities = data['num_real_activities']
    num_total_activities = data['num_total_activities']
    num_resource_types = data['num_resource_types']
    max_project_duration = data['max_project_duration']
    activities = data['activities']
    precedence_relations = data['precedence_relations']
    resource_costs = data['resource_costs']
    
    # Build activity dictionary by ID
    activity_dict = {act['id']: act for act in activities}
    
    # Build graph: successors and predecessors
    successors = {act['id']: [] for act in activities}
    predecessors = {act['id']: [] for act in activities}
    for rel in precedence_relations:
        i, j = rel
        successors[i].append(j)
        predecessors[j].append(i)
    
    # Identify start and end activities (fictitious)
    start_activity_id = None
    end_activity_id = None
    for act in activities:
        if len(predecessors[act['id']]) == 0:
            start_activity_id = act['id']
        if len(successors[act['id']]) == 0:
            end_activity_id = act['id']
    
    if start_activity_id is None or end_activity_id is None:
        raise ValueError("Start or end activity not found")
    
    # Compute earliest start times (est) using forward pass
    est = {act['id']: 0 for act in activities}
    in_degree = {act['id']: len(predecessors[act['id']]) for act in activities}
    q = deque()
    q.append(start_activity_id)
    while q:
        i = q.popleft()
        for j in successors[i]:
            new_est = est[i] + activity_dict[i]['duration']
            if new_est > est[j]:
                est[j] = new_est
            in_degree[j] -= 1
            if in_degree[j] == 0:
                q.append(j)
    
    # Compute latest start times (lst) using backward pass
    lst = {act['id']: max_project_duration for act in activities}
    out_degree = {act['id']: len(successors[act['id']]) for act in activities}
    q = deque()
    q.append(end_activity_id)
    lst[end_activity_id] = max_project_duration
    while q:
        j = q.popleft()
        for i in predecessors[j]:
            new_lst = lst[j] - activity_dict[i]['duration']
            if new_lst < lst[i]:
                lst[i] = new_lst
            out_degree[i] -= 1
            if out_degree[i] == 0:
                q.append(i)
    
    # Fix start and end activities
    est[start_activity_id] = 0
    lst[start_activity_id] = 0
    lst[end_activity_id] = max_project_duration
    if est[end_activity_id] > max_project_duration:
        est[end_activity_id] = max_project_duration
        lst[end_activity_id] = max_project_duration
    
    # Initialize Gurobi model
    model = gp.Model("RCPSP")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    # Create variables for start times (binary) and resource amounts (integer)
    y = {}
    for act in activities:
        i = act['id']
        if est[i] <= lst[i]:
            for t in range(est[i], lst[i] + 1):
                y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
    
    # Resource provision variables
    R = {}
    R_upper = [0] * num_resource_types
    for k in range(num_resource_types):
        total = 0
        for act in activities:
            if act['duration'] > 0:
                total += act['resource_requirements'][k]
        R_upper[k] = total
        R[k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=total, name=f"R_{k}")
    
    # Assignment constraints: each activity starts exactly once
    for act in activities:
        i = act['id']
        if est[i] <= lst[i]:
            model.addConstr(
                gp.quicksum(y[i, t] for t in range(est[i], lst[i] + 1)) == 1,
                name=f"assign_{i}"
            )
    
    # Precedence constraints
    for rel in precedence_relations:
        i, j = rel
        if est[i] > lst[i] or est[j] > lst[j]:
            continue
        S_i = gp.quicksum(t * y[i, t] for t in range(est[i], lst[i] + 1))
        S_j = gp.quicksum(t * y[j, t] for t in range(est[j], lst[j] + 1))
        model.addConstr(
            S_j >= S_i + activity_dict[i]['duration'],
            name=f"prec_{i}_{j}"
        )
    
    # Resource constraints: for each time period and resource type
    # Precompute non-zero resource requirements per activity
    non_zero_ks_per_act = {}
    for act in activities:
        i = act['id']
        if act['duration'] == 0:
            non_zero_ks_per_act[i] = []
            continue
        non_zero_ks = [k for k, req in enumerate(act['resource_requirements']) if req != 0]
        non_zero_ks_per_act[i] = non_zero_ks
    
    # Build resource constraints
    expr_res = [[gp.LinExpr() for _ in range(num_resource_types)] for _ in range(max_project_duration)]
    
    for act in activities:
        i = act['id']
        d_i = act['duration']
        if d_i == 0:
            continue
        non_zero_ks = non_zero_ks_per_act[i]
        if not non_zero_ks:
            continue
        res_req = act['resource_requirements']
        for t in range(est[i], lst[i] + 1):
            start_u = t
            end_u = min(t + d_i - 1, max_project_duration - 1)
            if start_u > end_u:
                continue
            for k in non_zero_ks:
                coef = res_req[k]
                for u in range(start_u, end_u + 1):
                    expr_res[u][k].addTerms(coef, y[i, t])
    
    for u in range(max_project_duration):
        for k in range(num_resource_types):
            if expr_res[u][k].size() > 0:
                model.addConstr(expr_res[u][k] <= R[k], name=f"res_{u}_{k}")
    
    # Objective: minimize total resource cost
    model.setObjective(
        gp.quicksum(resource_costs[k] * R[k] for k in range(num_resource_types)),
        GRB.MINIMIZE
    )
    
    # Initialize solution logger
    logger = None
    if log_path and SolutionLogger is not None:
        logger = SolutionLogger(log_path, sense="minimize")
    
    # Callback for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    # Optimize
    if logger:
        model.optimize(callback)
    else:
        model.optimize()
    
    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        start_times = {}
        for act in activities:
            i = act['id']
            found = False
            if est[i] <= lst[i]:
                for t in range(est[i], lst[i] + 1):
                    if abs(y[i, t].X - 1) < 1e-6:
                        start_times[str(i)] = int(t)
                        found = True
                        break
            if not found:
                start_times[str(i)] = est[i]
        
        resource_amounts = {}
        for k in range(num_resource_types):
            resource_amounts[str(k)] = int(round(R[k].X))
        
        objective_value = model.ObjVal
    else:
        # No feasible solution found, return default
        start_times = {str(act['id']): est[act['id']] for act in activities}
        resource_amounts = {str(k): R_upper[k] for k in range(num_resource_types)}
        objective_value = sum(resource_costs[k] * R_upper[k] for k in range(num_resource_types))
    
    # Write solution
    solution = {
        "objective_value": float(objective_value),
        "start_times": start_times,
        "resource_amounts": resource_amounts
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)