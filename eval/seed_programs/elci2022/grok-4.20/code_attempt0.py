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

    n = instance['n']
    m = instance['m']
    S = instance['num_scenarios']
    facility_capacity_limits = instance['facility_capacity_limits']
    capacity_requirements = instance['capacity_requirements']
    release_times = instance['release_times']
    mean_processing_times = instance['mean_processing_times']
    scenarios = instance['scenarios']

    # Precompute scenario data
    probabilities = [sc['probability'] for sc in scenarios]
    processing_times = [sc['processing_times'] for sc in scenarios]

    # Big-M for start times (loose upper bound)
    max_release = max(release_times)
    max_p = max(max(max(pt[j][i] for j in range(m)) for i in range(n)) for pt in processing_times)
    T_upper = max_release + S * n * max_p + 1000  # conservative

    # Create model
    model = gp.Model("StochasticScheduling")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 0.01  # Allow some gap for larger instances

    # First-stage: assignment (binary)
    assign = model.addVars(m, n, vtype=GRB.BINARY, name="assign")

    # Second-stage per scenario: start times
    start = {}
    for s in range(S):
        for j in range(n):
            for f in range(m):
                start[s, f, j] = model.addVar(lb=0, ub=T_upper, vtype=GRB.CONTINUOUS, name=f"start_{s}_{f}_{j}")

    # Makespan per scenario
    makespan = model.addVars(S, lb=0, ub=T_upper, vtype=GRB.CONTINUOUS, name="makespan")

    # Objective: expected makespan
    obj = gp.quicksum(probabilities[s] * makespan[s] for s in range(S))
    model.setObjective(obj, GRB.MINIMIZE)

    # Assignment constraint
    for j in range(n):
        model.addConstr(gp.quicksum(assign[f, j] for f in range(m)) == 1, name=f"assign_{j}")

    # Linking start times to assignment (only start on assigned facility)
    for s in range(S):
        for j in range(n):
            for f in range(m):
                model.addConstr(start[s, f, j] <= T_upper * assign[f, j], name=f"link_start_{s}_{f}_{j}")
                model.addConstr(start[s, f, j] >= release_times[j] * assign[f, j], name=f"release_{s}_{f}_{j}")

    # Makespan definition
    for s in range(S):
        for j in range(n):
            for f in range(m):
                model.addConstr(makespan[s] >= start[s, f, j] + processing_times[s][f][j] * assign[f, j],
                                name=f"makespan_def_{s}_{f}_{j}")

    # Capacity constraints - discretized time points
    # Use a reasonable discretization to keep model size manageable
    # Time points: all release times + a few multiples
    all_releases = sorted(set(release_times))
    time_points = set(all_releases)
    for r in all_releases:
        for k in range(1, 6):  # add some points after each release
            time_points.add(r + k * 5)
            time_points.add(r + k * 20)
    for pt in [0, 10, 20, 50, 100, 200, 500, 1000, 2000]:
        time_points.add(pt)
    time_points = sorted([t for t in time_points if t <= T_upper])
    time_points = time_points[:min(len(time_points), 80)]  # limit to avoid explosion

    for s in range(S):
        for f in range(m):
            for t_idx, t in enumerate(time_points):
                # Jobs active at time t: those with start <= t < start + p
                active = []
                for j in range(n):
                    p = processing_times[s][f][j]
                    if p == 0:
                        continue
                    start_var = start[s, f, j]
                    # start <= t and start > t - p  =>  t - p < start <= t
                    active.append((start_var, p, capacity_requirements[f][j]))

                if not active:
                    continue

                # We use big-M formulation for each possible start time point
                # Instead of indicator, we use a set of inequalities with big-M
                # This is a common linearization but can be weak; we tighten with multiple t
                expr = gp.LinExpr()
                for start_var, p_val, req in active:
                    # If a job is assigned (via bigM from assign), then if it overlaps t it contributes
                    # We discretize and check for each possible "start bucket"
                    # A simpler but still MIP way: use binary variables for discretized start times
                    # To keep it efficient, we instead add a constraint using:
                    # For each job, if assigned, its contribution is bounded using:
                    # But to accurately model capacity, we introduce binary variables for time periods

    # Better approach: Discretize possible start times per facility per scenario and use assignment of start bins
    # But to avoid too many variables, we use a time-indexed formulation with limited time horizon discretization

    # Let's define a set of candidate start times (release + multiples)
    candidate_starts = []
    for j in range(n):
        r = release_times[j]
        candidates = [r]
        for k in range(1, 12):
            candidates.append(r + k * 5)
            candidates.append(r + k * 10)
            candidates.append(r + k * 25)
        candidate_starts.append(sorted(set(candidates)))

    # We will use binary variables for each job-facility-scenario-start_time_index
    # This can be large but for moderate n,m,S it may work
    # Limit number of candidate starts per job to 8
    start_candidates = []
    for j in range(n):
        r = release_times[j]
        cands = [r, r+5, r+10, r+20, r+30, r+50, r+80, r+120, r+200, r+300]
        start_candidates.append([c for c in cands if c <= 2000])  # cap at reasonable time

    # Number of discrete start options per job
    K = [len(start_candidates[j]) for j in range(n)]

    # Binary variable: whether job j on facility f starts at candidate k in scenario s
    y = model.addVars([(s, f, j, k) for s in range(S) for f in range(m) for j in range(n) for k in range(K[j])],
                      vtype=GRB.BINARY, name="y")

    # Reset start variables to be calculated from y
    model.remove(start)  # remove previous continuous start
    start = {}  # redefine as expr

    # Each job on each scenario must be assigned exactly one start time on its facility
    for s in range(S):
        for j in range(n):
            model.addConstr(gp.quicksum(y[s, f, j, k] for f in range(m) for k in range(K[j])) == 1,
                            name=f"one_start_{s}_{j}")

    # Link to first stage assignment
    for j in range(n):
        for f in range(m):
            sum_over_starts = gp.quicksum(y[s, f, j, k] for s in range(S) for k in range(K[j])) 
            model.addConstr(sum_over_starts == S * assign[f, j], name=f"link_assign_{j}_{f}")

    # Define start time as linear expression
    start_expr = {}
    for s in range(S):
        for f in range(m):
            for j in range(n):
                expr = gp.LinExpr()
                for k in range(K[j]):
                    expr += start_candidates[j][k] * y[s, f, j, k]
                start_expr[s, f, j] = expr

    # Redefine makespan constraints using expressions
    for s in range(S):
        for j in range(n):
            for f in range(m):
                p = processing_times[s][f][j]
                model.addConstr(makespan[s] >= start_expr[s, f, j] + p * assign[f, j],
                                name=f"ms_{s}_{f}_{j}")

    # Capacity constraints using time points
    # Use fewer time points for efficiency
    check_times = [0]
    for j_cands in start_candidates:
        for t in j_cands:
            check_times.append(t)
            check_times.append(t + 5)
            check_times.append(t + 15)
    check_times = sorted(set(check_times))[:60]  # limit to 60 points

    for s_idx in range(S):
        p_scen = processing_times[s_idx]
        for f in range(m):
            cap = facility_capacity_limits[f]
            for t in check_times:
                overlapping = []
                for j in range(n):
                    req = capacity_requirements[f][j]
                    if req == 0:
                        continue
                    p = p_scen[f][j]
                    for k_idx, sk in enumerate(start_candidates[j]):
                        finish_time = sk + p
                        if sk <= t < finish_time:
                            overlapping.append((j, k_idx, req))

                if not overlapping:
                    continue

                # Sum of requirements for all selected (start, job) that cover t
                usage = gp.LinExpr()
                for j, k_idx, req in overlapping:
                    usage += req * y[s_idx, f, j, k_idx]
                model.addConstr(usage <= cap, name=f"cap_{s_idx}_{f}_{t}")

    # Also link start times to release (already implicit in candidate starts)

    # Callback for logging incumbent solutions
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.SolCount == 0:
        # No feasible solution found
        print("No feasible solution found within time limit.")
        return

    # Get assignment
    assignment = {}
    for j in range(n):
        for f in range(m):
            if assign[f, j].X > 0.5:
                assignment[str(j)] = f
                break

    # Get schedule per scenario
    schedule = {}
    final_makespans = [0.0] * S
    for s in range(S):
        scen_key = str(s)
        schedule[scen_key] = {}
        for j in range(n):
            job_key = str(j)
            # Find which facility and start was chosen
            chosen_f = None
            chosen_start = None
            for f in range(m):
                for k_idx in range(K[j]):
                    if y[s, f, j, k_idx].X > 0.5:
                        chosen_f = f
                        chosen_start = start_candidates[j][k_idx]
                        break
                if chosen_f is not None:
                    break
            if chosen_f is None:
                # fallback
                chosen_f = assignment[job_key]
                chosen_start = release_times[j]
            p_val = processing_times[s][chosen_f][j]
            finish = chosen_start + p_val
            schedule[scen_key][job_key] = {
                "facility": int(chosen_f),
                "start": int(chosen_start),
                "finish": int(finish)
            }
            final_makespans[s] = max(final_makespans[s], finish)

    expected_makespan = sum(probabilities[s] * final_makespans[s] for s in range(S))

    # Build output
    output = {
        "objective_value": float(expected_makespan),
        "assignment": assignment,
        "schedule": schedule
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Solution written to {args.solution_path} with objective {expected_makespan}")

if __name__ == "__main__":
    main()