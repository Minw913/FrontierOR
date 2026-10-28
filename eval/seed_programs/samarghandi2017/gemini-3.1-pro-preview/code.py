import argparse
import json
import time
import random
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass

def find_cycles(edges, n_jobs):
    """
    Finds cycles in a directed graph given as a list of edges.
    Assumes vertices are perfectly chained (1 in, 1 out max) as typical in TSP tours.
    """
    adj = {u: v for u, v in edges}
    visited = set()
    cycles = []
    
    for start_node in adj:
        if start_node not in visited:
            curr = start_node
            path = []
            path_set = set()
            while curr in adj and curr not in visited:
                path.append(curr)
                path_set.add(curr)
                visited.add(curr)
                curr = adj[curr]
            
            if curr in path_set:
                idx = path.index(curr)
                cycle_nodes = path[idx:]
                cycle_edges = [(cycle_nodes[k], cycle_nodes[(k+1) % len(cycle_nodes)]) for k in range(len(cycle_nodes))]
                cycles.append(cycle_edges)
    return cycles

def check_sequence_feasibility(seq, D, P, due_dates, n_jobs):
    starts = [0] * n_jobs
    starts[seq[0]] = 0
    for idx in range(1, len(seq)):
        prev = seq[idx-1]
        curr = seq[idx]
        starts[curr] = starts[prev] + D[prev][curr]
    
    for j in seq:
        if starts[j] + P[j] > due_dates[j]:
            return False, None, None
            
    mksp = max(starts[j] + P[j] for j in seq)
    return True, starts, mksp

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Solve NO-WAIT flow shop scheduling.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file.")
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n_jobs = data["n_jobs"]
    n_machines = data["n_machines"]
    processing_times = data["processing_times"]
    due_dates = data["due_dates"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Precompute per-job total processing time
    P = [sum(processing_times[k][i] for k in range(n_machines)) for i in range(n_jobs)]

    # Precompute wait-time delays between jobs: D[i][j]
    D = [[0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j: 
                continue
            max_val = -1
            sum_i = 0
            sum_j = 0
            for k in range(n_machines):
                sum_i += processing_times[k][i]
                if k > 0:
                    sum_j += processing_times[k-1][j]
                diff = sum_i - sum_j
                if diff > max_val:
                    max_val = diff
            D[i][j] = max_val

    # GRASP Heuristic to obtain a good initial bound
    random.seed(42)
    best_seq = None
    best_mksp = float('inf')
    best_starts = None

    # Earliest Due Date check
    seq_edd = sorted(range(n_jobs), key=lambda idx: due_dates[idx])
    ok, st, mk = check_sequence_feasibility(seq_edd, D, P, due_dates, n_jobs)
    if ok:
        best_seq = seq_edd
        best_starts = st
        best_mksp = mk

    # Simple GRASP loop
    for _ in range(150):
        if time.time() - start_time > 2.0:
            break
        cand_seq = []
        unassigned = set(range(n_jobs))
        curr_time = 0
        curr_job = None
        possible = True
        starts_tmp = [0] * n_jobs
        
        while unassigned:
            candidates = []
            for j in unassigned:
                st_c = 0 if curr_job is None else curr_time + D[curr_job][j]
                if st_c + P[j] <= due_dates[j]:
                    candidates.append((j, st_c))
            
            if not candidates:
                possible = False
                break
                
            candidates.sort(key=lambda item: item[1] + 0.5 * due_dates[item[0]])
            top = candidates[:3]
            nxt, nxt_time = random.choice(top)
            cand_seq.append(nxt)
            starts_tmp[nxt] = nxt_time
            curr_time = nxt_time
            curr_job = nxt
            unassigned.remove(nxt)
            
        if possible:
            last = cand_seq[-1]
            mk_val = max(starts_tmp[j] + P[j] for j in cand_seq)
            if mk_val < best_mksp:
                best_mksp = mk_val
                best_seq = list(cand_seq)
                best_starts = list(starts_tmp)

    # Initialize Gurobi model (ATSP with sequence constraints and MTZ Big-M logic embedded)
    model = gp.Model("NoWaitFlowShop")
    model.setParam('OutputFlag', 1)
    model.setParam('LazyConstraints', 1)
    model.setParam('Threads', 1)
    
    elapsed = time.time() - start_time
    model.setParam('TimeLimit', max(1.0, args.time_limit - elapsed))

    # Binary variables for job sequencing
    x = {}
    for j in range(n_jobs):
        x[n_jobs, j] = model.addVar(vtype=GRB.BINARY, name=f"x_start_{j}")
        x[j, n_jobs+1] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_end")
        for i in range(n_jobs):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Degree constraints equivalent to TSP
    model.addConstr(gp.quicksum(x[n_jobs, j] for j in range(n_jobs)) == 1, name="start_out")
    model.addConstr(gp.quicksum(x[i, n_jobs+1] for i in range(n_jobs)) == 1, name="end_in")

    for k in range(n_jobs):
        model.addConstr(
            x[n_jobs, k] + gp.quicksum(x[i, k] for i in range(n_jobs) if i != k) == 1, 
            name=f"in_{k}"
        )
        model.addConstr(
            x[k, n_jobs+1] + gp.quicksum(x[k, j] for j in range(n_jobs) if j != k) == 1, 
            name=f"out_{k}"
        )

    # Continuous time tracking variables
    S = {}
    for j in range(n_jobs):
        ub_val = due_dates[j] - P[j]
        S[j] = model.addVar(lb=0, ub=max(0, ub_val), vtype=GRB.CONTINUOUS, name=f"S_{j}")
        if ub_val < 0:
            model.addConstr(S[j] <= ub_val)

    S[n_jobs+1] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="makespan")
    model.setObjective(S[n_jobs+1], GRB.MINIMIZE)

    # Time separation constraints
    for i in range(n_jobs):
        model.addConstr(S[n_jobs+1] >= S[i] + P[i], name=f"makespan_def_{i}")
        for j in range(n_jobs):
            if i != j:
                M_ij = due_dates[i] - P[i] + D[i][j]
                if M_ij < 0:
                    M_ij = 0
                model.addConstr(S[j] >= S[i] + D[i][j] - M_ij * (1 - x[i, j]), name=f"time_{i}_{j}")

    model._x = x
    
    # Inject MIP start based off GRASP outcome
    if best_seq is not None:
        for (i, j), var in model._x.items():
            var.Start = 0
        for idx in range(1, len(best_seq)):
            model._x[best_seq[idx-1], best_seq[idx]].Start = 1
        model._x[n_jobs, best_seq[0]].Start = 1
        model._x[best_seq[-1], n_jobs+1].Start = 1
        for j in range(n_jobs):
            S[j].Start = best_starts[j]
        S[n_jobs+1].Start = best_mksp

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                x_vals = model.cbGetSolution(model._x)
                edges = []
                for (i, j) in model._x:
                    if i < n_jobs and j < n_jobs:
                        if x_vals[i, j] > 0.5:
                            edges.append((i, j))
                
                cycles = find_cycles(edges, n_jobs)
                if len(cycles) > 0:
                    for cycle in cycles:
                        model.cbLazy(gp.quicksum(model._x[u, v] for u, v in cycle) <= len(cycle) - 1)
                else:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger is not None:
                        logger.log(obj)
            except Exception:
                pass

    model.optimize(callback)

    # Retrieve optimal or best found feasible solution
    sequence = []
    if model.SolCount > 0:
        obj_val = model.ObjVal
        curr = n_jobs
        visited_count = 0
        while curr != n_jobs + 1 and visited_count <= n_jobs + 1:
            next_node = None
            for j in range(n_jobs + 2):
                if (curr, j) in x and x[curr, j].X > 0.5:
                    next_node = j
                    break
            if next_node is not None and next_node != n_jobs + 1:
                sequence.append(next_node + 1)
            curr = next_node
            visited_count += 1
    else:
        obj_val = best_mksp if best_mksp != float('inf') else -1.0
        sequence = [job + 1 for job in best_seq] if best_seq is not None else [i+1 for i in range(n_jobs)]

    # Final Output Formatting
    out_dict = {
        "objective_value": float(obj_val),
        "sequence": sequence
    }
    with open(args.solution_path, "w") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()