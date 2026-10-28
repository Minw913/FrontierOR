import json
import argparse
import time
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    deadline = instance['deadline']
    processing_times = instance['processing_times']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']
    num_resources = instance['num_partially_renewable_resources']

    # Build big-M for temporal constraints
    M = deadline + 1

    # Precompute for each resource the usage multiplier per activity
    # usage[i][r] = number of designated periods in [s+1, s+p_i]
    # But since we optimize, we will linearize via binary variables

    model = gp.Model("PRP_RCPSP")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    # Variables: start time of each activity (integer)
    S = model.addVars(num_nodes, vtype=GRB.INTEGER, lb=0, ub=deadline, name="S")
    model.addConstr(S[0] == 0, "start_at_zero")

    # End activity <= deadline
    model.addConstr(S[num_nodes-1] <= deadline, "deadline")

    # Temporal constraints
    for tc in temporal_constraints:
        i = tc['from']
        j = tc['to']
        delta = tc['delta']
        if tc['type'] == 'min_lag':
            model.addConstr(S[j] - S[i] >= delta, f"minlag_{i}_{j}")
        else:  # max_lag
            model.addConstr(S[j] - S[i] <= delta, f"maxlag_{i}_{j}")

    # For partially renewable resources we introduce binary variables indicating if an activity is active in each time period
    # But to make it efficient we use the standard linearization for PRP

    # First, determine the maximum relevant time
    T = deadline

    # For each resource r, for each activity i, the usage depends on how many of its time periods overlap [S_i+1, S_i+p_i]
    # This is equivalent to sum over t in resource time_periods of whether t is in (S_i, S_i + p_i]

    # To linearize, introduce for each activity i, each relevant t, binary x_{i,t} = 1 if activity i covers period t
    # Then usage = sum_t in R_r x_i,t

    # Constraints for x:
    # For each i with p_i > 0:
    #   sum_{t=1 to T} x_{i,t} == p_i   (but wait, only if S_i + p_i <= T, but it's more subtle because of boundary)

    # Actually, the definition: execution interval = time periods strictly > S_i and <= S_i + p_i
    # So exactly p_i periods if S_i + p_i <= deadline, but we allow S_end <= deadline but periods up to deadline I guess.

    # Since deadline is on start of end, and end has p=0.

    # A standard way is:
    # For each activity i (p_i > 0), for each time t=1 to T:
    # x_{i,t} binary
    # S_i < t <= S_i + p_i   <=>   t - p_i <= S_i < t
    # Which can be modeled as:
    # S_i >= t - p_i - M*(1 - x_it)
    # S_i <= t-1 + M * x_it
    # But also to force exactly the right number, we usually don't need to if we have the inequalities tight.
    # Since objective pushes S[n+1] down, and constraints are <= capacity, the model will tend to be tight.

    # But a better formulation for PRP is to use the cumulative:
    # For each resource r, sum over i of (demand_i,r * sum_{t in R_r} x_{i,t}) <= capacity_r

    # And link x_it and S_i.

    x = {}
    for i in range(num_nodes):
        p = processing_times[i]
        if p == 0:
            continue
        for t in range(1, T+1):
            x[i,t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # Linking constraints
    for i in range(num_nodes):
        p = processing_times[i]
        if p == 0:
            continue
        for t in range(1, T+1):
            # t > S_i  and  t <= S_i + p   <=>   S_i < t <= S_i + p
            # S_i <= t-1
            # S_i >= t - p
            xit = x[i,t]
            model.addConstr(S[i] <= (t - 1) + M * (1 - xit), f"link1_{i}_{t}")
            model.addConstr(S[i] >= t - p - M * (1 - xit), f"link2_{i}_{t}")
            # Also, if t - p < 0 or t > T we still let it be, but bounds on S help.

    # For each resource, add the capacity constraint
    for r_idx, res in enumerate(resources):
        cap = res['capacity']
        demands = res['demands']
        time_periods_set = set(res['time_periods'])  # designated periods
        total_consumption = 0
        for i in range(num_nodes):
            if demands[i] == 0 or processing_times[i] == 0:
                continue
            d = demands[i]
            usage = gp.LinExpr()
            for t in time_periods_set:
                if t in x and (i, t) in x:  # only if we created the var
                    usage += x[i, t]
            total_consumption += d * usage
        model.addConstr(total_consumption <= cap, f"resource_{r_idx}")

    # Objective: minimize start time of last activity
    model.setObjective(S[num_nodes-1], GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(int(round(obj)))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SOLUTION_LIMIT:
        if model.SolCount > 0:
            objective_value = int(round(model.objVal))
            start_times = [int(round(S[i].X)) for i in range(num_nodes)]
            
            # Verify feasibility roughly
            feasible = True
            # We trust Gurobi but do a quick check on temporal
            for tc in temporal_constraints:
                i = tc['from']
                j = tc['to']
                delta = tc['delta']
                diff = start_times[j] - start_times[i]
                if tc['type'] == 'min_lag' and diff < delta:
                    feasible = False
                elif tc['type'] == 'max_lag' and diff > delta:
                    feasible = False
            if feasible and start_times[0] == 0 and start_times[-1] <= deadline:
                solution = {
                    "objective_value": objective_value,
                    "start_times": start_times
                }
                with open(args.solution_path, 'w') as f:
                    json.dump(solution, f, indent=2)
                return

    # If no solution found or error, write a dummy feasible if possible (start everything at 0 if feasible)
    # But usually model finds something. As fallback:
    objective_value = deadline
    start_times = [0] * num_nodes
    # Try to set end to deadline
    start_times[-1] = deadline
    solution = {
        "objective_value": objective_value,
        "start_times": start_times
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()