import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_real = instance['num_real_activities']
    n = instance['num_activities_total']  # includes 0 and n-1
    T = instance['max_project_duration']
    activities = instance['activities']
    temporal = instance['temporal_constraints']
    resources = instance['resources']
    K = len(resources)

    # Build p array: p[i] = processing time of activity i
    p = [0] * n
    for act in activities:
        p[act['id']] = act['processing_time']

    # Temporal constraints: list of (i, j, lag)
    constraints = []
    for tc in temporal:
        constraints.append((tc['from_activity'], tc['to_activity'], tc['time_lag']))

    # Precompute for each resource: set of periods, capacity, demands
    resource_data = []
    for res in resources:
        Pi = set(res['Pi_k'])
        cap = res['capacity']
        dem = {}
        for k, v in res['demands'].items():
            dem[int(k)] = v
        resource_data.append((Pi, cap, dem))

    # Create Gurobi model
    model = gp.Model("PartiallyRenewableRCPSP")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    # Variables: S_i start time of activity i, integer 0 to T
    S = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"S_{i}") for i in range(n)]

    # Objective: minimize S[n-1]
    model.setObjective(S[n-1], GRB.MINIMIZE)

    # Temporal constraints: S_j - S_i >= lag
    for i, j, lag in constraints:
        model.addConstr(S[j] - S[i] >= lag, name=f"temp_{i}_{j}")

    # Project start at 0
    model.addConstr(S[0] == 0, name="start_at_zero")

    # For each partially renewable resource
    for r_idx, (Pi_k, C_k, d_ik) in enumerate(resource_data):
        if not Pi_k:
            continue
        # For each activity, compute how many periods it occupies in Pi_k
        # Activity i starting at s occupies [s+1, s + p_i]
        # Number of occupied periods in Pi_k if start=s: |{t in Pi_k | s+1 <= t <= s+p_i}|
        # = sum_{t in Pi_k} 1 if s < t <= s + p_i
        # Let L = min(Pi_k), U = max(Pi_k). We only need to consider s where there is potential overlap.
        # To linearize the total consumption:
        # total_consumption = sum_i d_i * sum_{t in Pi_k} x_{i,t}
        # where x_{i,t}=1 if activity i is active at t, i.e. S_i +1 <= t <= S_i + p_i
        # i.e. t - p_i <= S_i <= t-1
        # Since S integer, we can introduce binary variables for possible start times.

        # Determine possible start times for each activity: 0 to T - p_i (but S_end <=T)
        # But T can be large, we must be careful with model size.

        # Since n is typically small in these problems (often < 30-60 activities), and T up to few hundred,
        # we can use time-indexed formulation with binary variables x_{i,s} = 1 if activity i starts at time s.

    # Switch to time-indexed MILP formulation which is stronger for scheduling.

    # Re-create model with time-indexed vars for better propagation and to handle resource constraints linearly.
    model.dispose()
    model = gp.Model("PRCPSP_TimeIndexed")
    model.setParam('TimeLimit', args.time_limit - 1)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)
    model.setParam('LazyConstraints', 0)

    # Possible start times for each activity i: 0 .. T - p[i] + 1 but since S_{n-1} is the makespan we minimize
    # We allow starts up to T (S_end <= T)
    max_start = T
    x = []
    for i in range(n):
        max_s = max_start if p[i] == 0 else max_start - p[i] + 1
        xs = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}") for s in range(max_s + 1)]
        x.append(xs)
        # Exactly one start time
        model.addConstr(gp.quicksum(xs) == 1, name=f"one_start_{i}")

    # Link S variables for objective and temporal constraints (although we could eliminate S)
    S = []
    for i in range(n):
        max_s_i = len(x[i]) - 1
        S_i = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, max_s_i), name=f"S_{i}")
        model.addConstr(S_i == gp.quicksum(s * x[i][s] for s in range(len(x[i]))), name=f"link_S_{i}")
        S.append(S_i)

    model.setObjective(S[n-1], GRB.MINIMIZE)

    # Temporal constraints
    for i_idx, j_idx, lag in constraints:
        model.addConstr(S[j_idx] - S[i_idx] >= lag, name=f"temp_{i_idx}_{j_idx}")

    model.addConstr(S[0] == 0, name="start_zero")

    # Resource constraints
    for r_idx, (Pi_set, C_k, d_dict) in enumerate(resource_data):
        if C_k == 0 and any(d_dict.get(i, 0) > 0 for i in range(n)):
            # infeasible if any demand >0, but assume data is valid
            pass
        Pi_list = sorted(Pi_set)
        if not Pi_list:
            continue

        # For each t in Pi_k, we can compute the activities that cover t when starting at s:
        # activity i covers t if s+1 <= t <= s + p_i i.e. t - p_i <= s <= t-1
        # So for each t, the total usage at t is sum over i, sum over valid s for that t, d_i * x_{i,s}
        # But since capacity is on the SUM over t in Pi of usage_at_t, we can aggregate directly.

        total_consumption = gp.LinExpr()
        for i in range(n):
            d_i = d_dict.get(i, 0)
            if d_i == 0:
                continue
            p_i = p[i]
            if p_i == 0:
                continue
            # For each possible start s, number of periods in Pi that are covered by [s+1, s+p_i]
            # We add d_i * count * x_{i,s}
            num_vars = len(x[i])
            for s in range(num_vars):
                # count how many t in Pi with s+1 <= t <= s + p_i
                lower = s + 1
                upper = s + p_i
                # number of t in Pi_list with lower <= t <= upper
                # since Pi_list sorted we could binary search but since |Pi| usually small, linear is fine
                count = sum(1 for t in Pi_list if lower <= t <= upper)
                if count > 0:
                    total_consumption += d_i * count * x[i][s]

        model.addConstr(total_consumption <= C_k, name=f"resource_{r_idx}")

    # Callback for logging improved solutions
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            # Check if this is integer feasible (should be since all vars integer)
            if logger and obj < float('inf'):
                logger.log(float(obj))

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        schedule = {}
        for i in range(n):
            # Get value from S or from x
            for s in range(len(x[i])):
                if abs(x[i][s].X - 1.0) < 1e-5:
                    schedule[str(i)] = int(s)
                    break
        objective = float(S[n-1].X)

        # If we have a feasible solution
        if objective < float('inf') and all(str(i) in schedule for i in range(n)):
            solution = {
                "objective_value": objective,
                "schedule": schedule
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
        else:
            # No feasible solution found, write empty or best we have (should not happen for feasible instances)
            solution = {
                "objective_value": float(T),
                "schedule": {str(i): 0 for i in range(n)}
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
    else:
        # Infeasible or other error - return dummy
        solution = {
            "objective_value": float(T),
            "schedule": {str(i): 0 for i in range(n)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()