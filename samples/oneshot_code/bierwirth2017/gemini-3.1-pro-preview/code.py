import json
import argparse
import time
import random
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time_total = time.time()

    # Initialize SolutionLogger if requested
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Parse input data
    with open(args.instance_path) as f:
        data = json.load(f)

    num_jobs = data["num_jobs"]
    num_machines = data["num_machines"]
    jobs = sorted(data["jobs"], key=lambda k: k["job_id"])

    job_ops = [j_data["operations"] for j_data in jobs]
    num_ops = [len(ops) for ops in job_ops]

    # Pre-calculate remaining processing times for ATC heuristics
    rem_p_cache = []
    for j in range(num_jobs):
        p_arr = [op["processing_time"] for op in job_ops[j]]
        suf = [0] * num_ops[j]
        s = 0
        for o in range(num_ops[j]-1, -1, -1):
            s += p_arr[o]
            suf[o] = s
        rem_p_cache.append(suf)

    best_obj = float('inf')
    best_starts = None

    # Constructive heuristic - Active Schedule Generation (ASG)
    def get_schedule(seed, rule="random", K=1.0):
        random.seed(seed)
        op_idx = [0] * num_jobs
        avail_job = [jobs[j]["release_date"] for j in range(num_jobs)]
        avail_mach = [0] * num_machines
        
        start_times = {}
        unscheduled_ops = sum(num_ops)
        
        while unscheduled_ops > 0:
            min_ect = float('inf')
            m_star = -1
            
            # Identify the machine with the minimum earliest completion time
            for j in range(num_jobs):
                if op_idx[j] < num_ops[j]:
                    idx = op_idx[j]
                    m = job_ops[j][idx]["machine"]
                    p = job_ops[j][idx]["processing_time"]
                    e = avail_job[j] if avail_job[j] > avail_mach[m] else avail_mach[m]
                    c = e + p
                    if c < min_ect:
                        min_ect = c
                        m_star = m
                        
            # Determine schedulable candidates
            candidates = []
            for j in range(num_jobs):
                if op_idx[j] < num_ops[j]:
                    idx = op_idx[j]
                    m = job_ops[j][idx]["machine"]
                    if m == m_star:
                        e = avail_job[j] if avail_job[j] > avail_mach[m] else avail_mach[m]
                        p = job_ops[j][idx]["processing_time"]
                        if e < min_ect or (e == min_ect and p == 0):
                            candidates.append((j, e))
                            
            # Dispatching rules
            if rule == "spt":
                j_star = min(candidates, key=lambda x: job_ops[x[0]][op_idx[x[0]]]["processing_time"])[0]
            elif rule == "lpt":
                j_star = max(candidates, key=lambda x: job_ops[x[0]][op_idx[x[0]]]["processing_time"])[0]
            elif rule == "random":
                j_star = random.choice(candidates)[0]
            elif rule == "atc" or rule == "gr_atc":
                avg_p = sum(job_ops[x[0]][op_idx[x[0]]]["processing_time"] for x in candidates) / len(candidates)
                if avg_p == 0: avg_p = 1.0
                
                if rule == "atc":
                    best_atc = -1.0
                    j_star = candidates[0][0]
                    for j, e in candidates:
                        idx = op_idx[j]
                        p = job_ops[j][idx]["processing_time"]
                        w = jobs[j]["weight"]
                        d = jobs[j]["due_date"]
                        rem = rem_p_cache[j][idx] - p
                        slack = max(0, d - e - p - rem)
                        
                        if p > 0: val = (w / p) * math.exp(-slack / (K * avg_p))
                        else: val = w * 1e6 * math.exp(-slack / (K * avg_p))
                            
                        if val > best_atc:
                            best_atc = val
                            j_star = j
                else: # GRASP ATC
                    weights = []
                    for j, e in candidates:
                        idx = op_idx[j]
                        p = job_ops[j][idx]["processing_time"]
                        w = jobs[j]["weight"]
                        d = jobs[j]["due_date"]
                        rem = rem_p_cache[j][idx] - p
                        slack = max(0, d - e - p - rem)
                        
                        if p > 0: val = (w / p) * math.exp(-slack / (K * avg_p))
                        else: val = w * 1e6 * math.exp(-slack / (K * avg_p))
                        weights.append(val)
                        
                    sum_w = sum(weights)
                    if sum_w <= 1e-9:
                        j_star = random.choice(candidates)[0]
                    else:
                        r = random.uniform(0, sum_w)
                        acc = 0.0
                        for i, (j, e) in enumerate(candidates):
                            acc += weights[i]
                            if acc >= r:
                                j_star = j
                                break
                        else:
                            j_star = candidates[-1][0]
            else:
                j_star = random.choice(candidates)[0]
                
            # Schedule chosen operation
            idx = op_idx[j_star]
            p = job_ops[j_star][idx]["processing_time"]
            m = job_ops[j_star][idx]["machine"]
            e = avail_job[j_star] if avail_job[j_star] > avail_mach[m] else avail_mach[m]
            
            start_times[(j_star, idx)] = e
            avail_job[j_star] = e + p
            avail_mach[m] = e + p
            op_idx[j_star] += 1
            unscheduled_ops -= 1
            
        # Compute objective value
        obj = 0.0
        for j in range(num_jobs):
            if num_ops[j] > 0:
                last_idx = num_ops[j] - 1
                comp_time = start_times[(j, last_idx)] + job_ops[j][last_idx]["processing_time"]
            else:
                comp_time = jobs[j]["release_date"]
            t = comp_time - jobs[j]["due_date"]
            if t > 0:
                obj += jobs[j]["weight"] * t
                
        return obj, start_times

    # 1. Heuristic Phase
    # Dedicate up to 20% of the maximum computation time (max 15 seconds) running multiple heuristic evaluations.
    heuristic_time_limit = min(15.0, args.time_limit * 0.2)
    rules = [
        {"rule": "atc", "K": 1.0},
        {"rule": "atc", "K": 0.5},
        {"rule": "atc", "K": 2.0},
        {"rule": "spt", "K": 1.0},
        {"rule": "lpt", "K": 1.0},
    ]
    
    iteration = 0
    while time.time() - start_time_total < heuristic_time_limit:
        if iteration < len(rules):
            r = rules[iteration]["rule"]
            k = rules[iteration]["K"]
        else:
            r = "gr_atc"
            k = random.choice([0.5, 1.0, 2.0, 3.0])
            
        obj, starts = get_schedule(seed=iteration, rule=r, K=k)
        if obj < best_obj:
            best_obj = obj
            best_starts = starts
            if logger:
                logger.log(best_obj)
            # If a perfect solution is found, skip further optimization
            if best_obj == 0.0:
                break
        iteration += 1

    final_starts = best_starts
    final_obj = best_obj

    # Calculate remaining time margin for solver
    time_left = args.time_limit - (time.time() - start_time_total) - 0.5
    
    # 2. Exact Search Phase (Gurobi MIP)
    # Passed remaining available time buffer
    if time_left > 0 and best_obj > 0:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model("JSSP_WT", env=env)
        model.setParam('Threads', 1) 
        model.setParam('TimeLimit', time_left)

        # Variables
        S = {}
        for j in range(num_jobs):
            for o in range(num_ops[j]):
                S[j, o] = model.addVar(vtype=GRB.CONTINUOUS, name=f"S_{j}_{o}")
        
        T = {}
        for j in range(num_jobs):
            T[j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"T_{j}")

        # Machine Disjunctive constraints Setup
        machine_ops = defaultdict(list)
        for j in range(num_jobs):
            for o in range(num_ops[j]):
                m = job_ops[j][o]["machine"]
                machine_ops[m].append((j, o))

        y = {}
        for m, ops in machine_ops.items():
            for i in range(len(ops)):
                for k in range(i + 1, len(ops)):
                    j1, o1 = ops[i]
                    j2, o2 = ops[k]
                    y_var = model.addVar(vtype=GRB.BINARY, name=f"y_{m}_{j1}_{o1}_{j2}_{o2}")
                    y[(j1, o1, j2, o2)] = y_var
                    
                    p1 = job_ops[j1][o1]["processing_time"]
                    p2 = job_ops[j2][o2]["processing_time"]
                    
                    # Indicator constraints to avoid Big-M numerical issues
                    model.addGenConstrIndicator(y_var, 1, S[j2, o2] - S[j1, o1] >= p1)
                    model.addGenConstrIndicator(y_var, 0, S[j1, o1] - S[j2, o2] >= p2)

        # Job topological constraints Setup
        for j in range(num_jobs):
            if num_ops[j] == 0: 
                continue
            
            # First operation release date
            model.addConstr(S[j, 0] >= jobs[j]["release_date"])
            
            # Ordering constraint within job
            for o in range(1, num_ops[j]):
                p_prev = job_ops[j][o-1]["processing_time"]
                model.addConstr(S[j, o] >= S[j, o-1] + p_prev)

            # Job Weighted Tardiness penalty constraints
            last_o = num_ops[j] - 1
            p_last = job_ops[j][last_o]["processing_time"]
            due = jobs[j]["due_date"]
            model.addConstr(T[j] >= S[j, last_o] + p_last - due)
            model.addConstr(T[j] >= 0)

        model.setObjective(gp.quicksum(jobs[j]["weight"] * T[j] for j in range(num_jobs)), GRB.MINIMIZE)

        # Load constructive heuristic solution as a jump-start (MIP START)
        if best_starts is not None:
            for j in range(num_jobs):
                for o in range(num_ops[j]):
                    S[j, o].Start = best_starts[(j, o)]
                    
            for m, ops in machine_ops.items():
                for i in range(len(ops)):
                    for k in range(i + 1, len(ops)):
                        j1, o1 = ops[i]
                        j2, o2 = ops[k]
                        if best_starts[(j1, o1)] + job_ops[j1][o1]["processing_time"] <= best_starts[(j2, o2)]:
                            y[(j1, o1, j2, o2)].Start = 1
                        else:
                            y[(j1, o1, j2, o2)].Start = 0

        # Solution logging callback hook during MIP evaluation
        model._best_obj = best_obj
        def cb(mdl, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger and obj_val < mdl._best_obj - 1e-5:
                    mdl._best_obj = obj_val
                    logger.log(obj_val)

        model.optimize(cb)

        # Parse final results
        if model.SolCount > 0:
            final_obj = model.ObjVal
            final_starts = {}
            for j in range(num_jobs):
                for o in range(num_ops[j]):
                    # Revert possible Gurobi numeric discrepancies using floor/max boundaries bounding start at strictly >=0
                    final_starts[(j, o)] = float(max(0.0, S[j, o].X))
        else:
            final_obj = best_obj
            final_starts = best_starts

    # 3. Output Translation Projection Schema Mapping
    schedule = []
    for j in range(num_jobs):
        if num_ops[j] == 0:
            schedule.append({
                "job_id": int(jobs[j]["job_id"]),
                "completion_time": float(jobs[j]["release_date"]),
                "tardiness": float(max(0, jobs[j]["release_date"] - jobs[j]["due_date"])),
                "operations": []
            })
            continue

        last_o = num_ops[j] - 1
        comp_time = final_starts[(j, last_o)] + job_ops[j][last_o]["processing_time"]
        tardiness = max(0.0, comp_time - jobs[j]["due_date"])
        
        ops_out = []
        for o in range(num_ops[j]):
            ops_out.append({
                "machine": int(job_ops[j][o]["machine"]),
                "start_time": float(final_starts[(j, o)])
            })
            
        schedule.append({
            "job_id": int(jobs[j]["job_id"]),
            "completion_time": float(comp_time),
            "tardiness": float(tardiness),
            "operations": ops_out
        })

    out_dict = {
        "objective_value": float(final_obj),
        "schedule": schedule
    }

    with open(args.solution_path, 'w') as f:
        json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    solve()