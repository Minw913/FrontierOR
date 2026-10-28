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
    max_release = max(release_times) if release_times else 0
    max_p = 0
    for pt in processing_times:
        for f_vals in pt:
            for pval in f_vals:
                if pval > max_p:
                    max_p = pval
    T_upper = max_release + n * max_p * 2 + 1000

    # Create model
    model = gp.Model("StochasticScheduling")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4
    model.Params.Heuristics = 0.2
    model.Params.MIPFocus = 1

    # First-stage: assignment (binary)
    assign = model.addVars(m, n, vtype=GRB.BINARY, name="assign")

    # Candidate start times per job (discretized)
    start_candidates = []
    for j in range(n):
        r = release_times[j]
        cands = [r]
        deltas = [0, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
        for d in deltas:
            cands.append(r + d)
        start_candidates.append(sorted(set([c for c in cands if c <= T_upper])))

    K = [len(start_candidates[j]) for j in range(n)]

    # Binary variable: y[s,f,j,k] = 1 if job j on facility f starts at candidate k in scenario s
    y = model.addVars([(s, f, j, k) for s in range(S) for f in range(m) for j in range(n) for k in range(K[j])],
                      vtype=GRB.BINARY, name="y")

    # Each job-scenario must start exactly once
    for s in range(S):
        for j in range(n):
            model.addConstr(gp.quicksum(y[s, f, j, k] for f in range(m) for k in range(K[j])) == 1,
                            name=f"one_start_{s}_{j}")

    # Link assignment to usage across scenarios
    for j in range(n):
        for f in range(m):
            model.addConstr(gp.quicksum(y[s, f, j, k] for s in range(S) for k in range(K[j])) == S * assign[f, j],
                            name=f"link_assign_{j}_{f}")

    # Makespan per scenario
    makespan = model.addVars(S, lb=0, ub=T_upper*2, vtype=GRB.CONTINUOUS, name="makespan")

    # Objective: expected makespan
    obj = gp.quicksum(probabilities[s] * makespan[s] for s in range(S))
    model.setObjective(obj, GRB.MINIMIZE)

    # Makespan constraints
    for s in range(S):
        for j in range(n):
            for f in range(m):
                for k in range(K[j]):
                    start_t = start_candidates[j][k]
                    p = processing_times[s][f][j]
                    model.addConstr(makespan[s] >= (start_t + p) * y[s, f, j, k], name=f"ms_{s}_{f}_{j}_{k}")

    # Capacity constraints - sampled time points
    check_times = set()
    for j in range(n):
        for st in start_candidates[j]:
            check_times.add(st)
            check_times.add(st + 1)
            for delta in [5, 10, 20, 40]:
                check_times.add(st + delta)
    check_times = sorted(list(check_times))[:50]

    for s in range(S):
        for f in range(m):
            cap = facility_capacity_limits[f]
            for t in check_times:
                usage = gp.LinExpr(0)
                for j in range(n):
                    req = capacity_requirements[f][j]
                    if req <= 0:
                        continue
                    p = processing_times[s][f][j]
                    for k in range(K[j]):
                        sk = start_candidates[j][k]
                        if sk <= t < sk + p:
                            usage += req * y[s, f, j, k]
                model.addConstr(usage <= cap, name=f"cap_s{s}_f{f}_t{t}")

    # Callback for logging
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < GRB.INFINITY and logger:
                logger.log(obj_val)

    # Optimize
    model.optimize(callback)

    if model.SolCount == 0:
        print("No feasible solution found within time limit.")
        # Write dummy feasible if possible, but for now just exit
        dummy_out = {
            "objective_value": 999999.0,
            "assignment": {str(j): 0 for j in range(n)},
            "schedule": {}
        }
        for s in range(S):
            dummy_out["schedule"][str(s)] = {}
            for j_idx in range(n):
                dummy_out["schedule"][str(s)][str(j_idx)] = {
                    "facility": 0,
                    "start": release_times[j_idx],
                    "finish": release_times[j_idx] + processing_times[0][0][j_idx]
                }
        with open(args.solution_path, 'w') as f:
            json.dump(dummy_out, f, indent=2)
        return

    # Extract assignment
    assignment = {}
    for j in range(n):
        assigned = False
        for f in range(m):
            if assign[f, j].X > 0.4:
                assignment[str(j)] = int(f)
                assigned = True
                break
        if not assigned:
            assignment[str(j)] = 0

    # Extract schedule
    schedule = {}
    final_makespans = [0.0] * S
    for s in range(S):
        scen_key = str(s)
        schedule[scen_key] = {}
        for j in range(n):
            job_key = str(j)
            chosen_f = None
            chosen_start = None
            chosen_k = None
            for f in range(m):
                for k in range(K[j]):
                    if y[s, f, j, k].X > 0.4:
                        chosen_f = f
                        chosen_start = start_candidates[j][k]
                        chosen_k = k
                        break
                if chosen_f is not None:
                    break
            if chosen_f is None:
                chosen_f = assignment[job_key]
                chosen_start = release_times[j]
            p_val = processing_times[s][chosen_f][j]
            finish_t = chosen_start + p_val
            schedule[scen_key][job_key] = {
                "facility": int(chosen_f),
                "start": int(chosen_start),
                "finish": int(finish_t)
            }
            if finish_t > final_makespans[s]:
                final_makespans[s] = finish_t

    expected_makespan = sum(probabilities[s] * final_makespans[s] for s in range(S))

    output = {
        "objective_value": float(expected_makespan),
        "assignment": assignment,
        "schedule": schedule
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Solution written with objective {expected_makespan}")

if __name__ == "__main__":
    main()