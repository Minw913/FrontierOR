import argparse
import json
import time
from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None

def compute_completion_times(sequence, processing_times, no_idle_flags):
    n_jobs = len(sequence)
    n_machines = len(processing_times[0])
    C = [[0] * n_jobs for _ in range(n_machines)]
    
    # Machine 0
    cum_sum = 0
    for k in range(n_jobs):
        job = sequence[k]
        cum_sum += processing_times[job][0]
        C[0][k] = cum_sum

    # Subsequent machines
    for j in range(1, n_machines):
        if no_idle_flags[j]:
            s_j = 0
            cum_sum = 0
            for k in range(n_jobs):
                s_j = max(s_j, C[j-1][k] - cum_sum)
                cum_sum += processing_times[sequence[k]][j]
                C[j][k] = s_j + cum_sum
        else:
            job0 = sequence[0]
            C[j][0] = C[j-1][0] + processing_times[job0][j]
            for k in range(1, n_jobs):
                jobk = sequence[k]
                start_time = max(C[j][k-1], C[j-1][k])
                C[j][k] = start_time + processing_times[jobk][j]
    
    makespan = C[-1][-1]
    return makespan, C

def run_NEH(n_jobs, n_machines, no_idle_flags, processing_times, start_time, time_limit):
    total_processing = [sum(processing_times[i]) for i in range(n_jobs)]
    jobs_sorted = sorted(range(n_jobs), key=lambda i: total_processing[i], reverse=True)
    
    sequence = []
    for idx, job in enumerate(jobs_sorted):
        if time.time() - start_time > time_limit:
            # Time limit reached, append remaining jobs arbitrarily
            sequence += jobs_sorted[idx:]
            break
            
        best_candidate = None
        best_makespan = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            makespan, _ = compute_completion_times(candidate, processing_times, no_idle_flags)
            if makespan < best_makespan:
                best_makespan = makespan
                best_candidate = candidate
        sequence = best_candidate
    
    final_makespan, C = compute_completion_times(sequence, processing_times, no_idle_flags)
    return sequence, final_makespan

def build_mip_model(n_jobs, n_machines, no_idle_flags, processing_times):
    model = gp.Model("flowshop_no_idle")
    model.setParam('OutputFlag', 0)
    
    # Variables
    y = {}
    for i in range(n_jobs):
        for k in range(n_jobs):
            y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
    
    s = {}
    for j in range(n_machines):
        for k in range(n_jobs):
            s[j, k] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_{j}_{k}")
    
    # Assignment constraints
    for i in range(n_jobs):
        model.addConstr(gp.quicksum(y[i, k] for k in range(n_jobs)) == 1, name=f"assign_job_{i}")
    for k in range(n_jobs):
        model.addConstr(gp.quicksum(y[i, k] for i in range(n_jobs)) == 1, name=f"assign_pos_{k}")
    
    # Machine 0
    model.addConstr(s[0, 0] == 0, name="s0_0")
    for k in range(1, n_jobs):
        expr = gp.LinExpr()
        for i in range(n_jobs):
            expr += y[i, k-1] * processing_times[i][0]
        model.addConstr(s[0, k] == s[0, k-1] + expr, name=f"machine0_pos{k}")
    
    # Subsequent machines
    for j in range(1, n_machines):
        # Position 0
        expr0 = gp.LinExpr()
        for i in range(n_jobs):
            expr0 += y[i, 0] * processing_times[i][j-1]
        model.addConstr(s[j, 0] >= s[j-1, 0] + expr0, name=f"machine{j}_pos0")
        
        if no_idle_flags[j]:
            for k in range(1, n_jobs):
                # Recurrence constraint
                expr1 = gp.LinExpr()
                for i in range(n_jobs):
                    expr1 += y[i, k-1] * processing_times[i][j]
                model.addConstr(s[j, k] == s[j, k-1] + expr1, name=f"no_idle_recur_{j}_{k}")
                
                # Availability constraint
                expr2 = gp.LinExpr()
                for i in range(n_jobs):
                    expr2 += y[i, k] * processing_times[i][j-1]
                model.addConstr(s[j, k] >= s[j-1, k] + expr2, name=f"no_idle_avail_{j}_{k}")
        else:
            for k in range(1, n_jobs):
                # Machine availability
                expr1 = gp.LinExpr()
                for i in range(n_jobs):
                    expr1 += y[i, k-1] * processing_times[i][j]
                model.addConstr(s[j, k] >= s[j, k-1] + expr1, name=f"reg_machine_avail_{j}_{k}")
                
                # Job availability
                expr2 = gp.LinExpr()
                for i in range(n_jobs):
                    expr2 += y[i, k] * processing_times[i][j-1]
                model.addConstr(s[j, k] >= s[j-1, k] + expr2, name=f"reg_job_avail_{j}_{k}")
    
    # Makespan: completion time of last job on last machine
    makespan = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="makespan")
    expr_obj = gp.LinExpr()
    for i in range(n_jobs):
        expr_obj += y[i, n_jobs-1] * processing_times[i][n_machines-1]
    model.addConstr(makespan == s[n_machines-1, n_jobs-1] + expr_obj, name="def_makespan")
    model.setObjective(makespan, GRB.MINIMIZE)
    
    return model, y, s, makespan

def extract_sequence_from_mip(y, n_jobs):
    sequence = [-1] * n_jobs
    for k in range(n_jobs):
        for i in range(n_jobs):
            if abs(y[i, k].X - 1.0) < 1e-5:
                sequence[k] = i
                break
    return sequence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    no_idle_flags = instance['no_idle_flags']
    processing_times = instance['processing_times']
    
    start_time = time.time()
    
    # Run NEH for initial solution
    sequence, makespan = run_NEH(n_jobs, n_machines, no_idle_flags, processing_times, start_time, args.time_limit)
    if logger:
        logger.log(makespan)
    
    best_sequence = sequence
    best_makespan = makespan
    
    # If small and time remains, run MIP
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    use_mip = (gp is not None) and (n_jobs <= 15) and (remaining_time > 1)
    
    if use_mip:
        try:
            model, y, s, makespan_var = build_mip_model(n_jobs, n_machines, no_idle_flags, processing_times)
            model.setParam('TimeLimit', max(1, remaining_time))
            
            # Callback for logging
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj)
            
            model.optimize(callback if logger else None)
            
            if model.SolCount > 0:
                mip_sequence = extract_sequence_from_mip(y, n_jobs)
                mip_makespan = makespan_var.X
                if mip_makespan < best_makespan:
                    best_sequence = mip_sequence
                    best_makespan = mip_makespan
        except Exception as e:
            # Fall back to NEH if MIP fails
            print(f"MIP failed: {e}")
            pass
    
    # Compute final completion times
    final_makespan, completion_times_matrix = compute_completion_times(best_sequence, processing_times, no_idle_flags)
    
    # Format completion times for output
    completion_times_dict = {}
    for j in range(n_machines):
        completion_times_dict[f"machine_{j}"] = completion_times_matrix[j]
    
    solution = {
        "objective_value": final_makespan,
        "sequence": best_sequence,
        "completion_times": completion_times_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()