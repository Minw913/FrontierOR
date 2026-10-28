import json
import argparse
import sys
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_distance_matrix(pt, n_machines, n_jobs):
    """Compute the minimum start-time gap d[i][j] when job i immediately precedes job j."""
    d = [[0.0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                continue
            max_gap = float('-inf')
            ci = 0.0
            cj_prev = 0.0
            for k in range(n_machines):
                ci += pt[k][i]
                gap = ci - cj_prev
                if gap > max_gap:
                    max_gap = gap
                cj_prev += pt[k][j]
            d[i][j] = max_gap
    return d


def eval_seq(seq, d, T, due_dates):
    """Evaluate makespan and feasibility of a sequence (0-based job indices)."""
    S = 0.0
    makespan = 0.0
    for idx, j in enumerate(seq):
        comp = S + T[j]
        if comp > due_dates[j] + 1e-6:
            return float('inf'), False
        makespan = max(makespan, comp)
        if idx < len(seq) - 1:
            S += d[j][seq[idx + 1]]
    return makespan, True


def local_search(seq, d, T, due_dates, deadline):
    """First-improvement local search: 2-opt (segment reversal) + pairwise swaps."""
    best_ms, best_feas = eval_seq(seq, d, T, due_dates)
    if not best_feas:
        return seq, best_ms, False
    n = len(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 2-opt: reverse segment [i..j]
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                new_seq = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                ms, feas = eval_seq(new_seq, d, T, due_dates)
                if feas and ms < best_ms - 1e-9:
                    best_ms = ms
                    seq = new_seq
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Pairwise swaps
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                new_seq = seq[:]
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                ms, feas = eval_seq(new_seq, d, T, due_dates)
                if feas and ms < best_ms - 1e-9:
                    best_ms = ms
                    seq = new_seq
                    improved = True
                    break
            if improved:
                break
    return seq, best_ms, True


def try_heuristics(pt, n_machines, n_jobs, due_dates, d, T, deadline):
    """Try multiple construction heuristics + local search, return best feasible."""
    best_ms = float('inf')
    best_seq = None

    orderings = []
    orderings.append(sorted(range(n_jobs), key=lambda j: due_dates[j]))
    orderings.append(sorted(range(n_jobs), key=lambda j: due_dates[j] - T[j]))
    orderings.append(sorted(range(n_jobs), key=lambda j: -T[j]))
    orderings.append(sorted(range(n_jobs), key=lambda j: T[j]))
    # Greedy nearest-neighbor from each start
    for start in range(min(n_jobs, 10)):
        remaining = set(range(n_jobs))
        seq = [start]
        remaining.discard(start)
        while remaining:
            last = seq[-1]
            best_j = None
            best_gap = float('inf')
            for j in remaining:
                if d[last][j] < best_gap:
                    best_gap = d[last][j]
                    best_j = j
            seq.append(best_j)
            remaining.discard(best_j)
        orderings.append(seq)

    for ordering in orderings:
        if time.time() > deadline:
            break
        ms, feas = eval_seq(ordering, d, T, due_dates)
        if feas and ms < best_ms:
            best_ms = ms
            best_seq = ordering[:]

    if best_seq is not None:
        best_seq, best_ms, _ = local_search(best_seq, d, T, due_dates, deadline)

    return best_ms, best_seq


class GurobiCallback:
    def __init__(self, x_vars, n_jobs, logger):
        self.x_vars = x_vars
        self.n_jobs = n_jobs
        self.logger = logger
        self.best_obj = float('inf')

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < self.best_obj - 1e-9:
                self.best_obj = obj
                succ = {}
                for i in range(self.n_jobs):
                    for j in range(self.n_jobs):
                        if i != j:
                            val = model.cbGetSolution(self.x_vars[i, j])
                            if val > 0.5:
                                succ[i] = j
                has_pred = set(succ.values())
                first = None
                for j in range(self.n_jobs):
                    if j not in has_pred:
                        first = j
                        break
                if first is not None:
                    seq = [first]
                    current = first
                    visited = {first}
                    while current in succ and succ[current] not in visited:
                        current = succ[current]
                        seq.append(current)
                        visited.add(current)
                    if len(seq) == self.n_jobs:
                        solution_dict = {
                            "objective_value": float(obj),
                            "sequence": [j + 1 for j in seq]
                        }
                        if self.logger:
                            self.logger.log_solution(float(obj), solution_dict)


def extract_sequence_from_model(x_vars, n_jobs):
    succ = {}
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
    has_pred = set(succ.values())
    first = None
    for j in range(n_jobs):
        if j not in has_pred:
            first = j
            break
    if first is None:
        return None
    seq = [first]
    current = first
    visited = {first}
    while current in succ and succ[current] not in visited:
        current = succ[current]
        seq.append(current)
        visited.add(current)
    if len(seq) == n_jobs:
        return seq
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    pt = instance['processing_times']  # [n_machines][n_jobs]
    due_dates = instance['due_dates']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Total processing time per job
    T = [sum(pt[k][j] for k in range(n_machines)) for j in range(n_jobs)]

    # Compute distance matrix
    d = compute_distance_matrix(pt, n_machines, n_jobs)

    # Allocate time: ~15% for heuristics, rest for MIP
    heur_deadline = start_time + max(1, args.time_limit * 0.15)

    # Run heuristics
    heur_ms, heur_seq = try_heuristics(pt, n_machines, n_jobs, due_dates, d, T, heur_deadline)

    if heur_seq is not None:
        solution_dict = {
            "objective_value": float(heur_ms),
            "sequence": [j + 1 for j in heur_seq]
        }
        if logger:
            logger.log_solution(float(heur_ms), solution_dict)

    # Set up Gurobi MIP
    remaining_time = max(1, args.time_limit - (time.time() - start_time) - 1)

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('NumericFocus', 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('TimeLimit', remaining_time)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

    S = {}
    for j in range(n_jobs):
        S[j] = model.addVar(lb=0.0, name=f'S_{j}')

    C = model.addVar(lb=0.0, name='C')
    model.setObjective(C, GRB.MINIMIZE)

    # At most one successor per job
    for i in range(n_jobs):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_jobs) if i != j) <= 1)

    # At most one predecessor per job
    for j in range(n_jobs):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n_jobs) if i != j) <= 1)

    # Total links = n - 1
    model.addConstr(gp.quicksum(x[i, j] for i in range(n_jobs) for j in range(n_jobs) if i != j) == n_jobs - 1)

    # Big M
    max_d = max(d[i][j] for i in range(n_jobs) for j in range(n_jobs) if i != j)
    max_due = max(due_dates[j] for j in range(n_jobs))
    M_val = max_due + max_d + 1

    # MTZ subtour elimination constraints
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j:
                model.addConstr(S[j] >= S[i] + d[i][j] - M_val * (1 - x[i, j]))

    # Due date constraints
    for j in range(n_jobs):
        model.addConstr(S[j] + T[j] <= due_dates[j])

    # Makespan definition
    for j in range(n_jobs):
        model.addConstr(C >= S[j] + T[j])

    # Warm start from heuristic
    if heur_seq is not None:
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i != j:
                    x[i, j].Start = 0.0
        for idx in range(len(heur_seq) - 1):
            x[heur_seq[idx], heur_seq[idx + 1]].Start = 1.0
        cum = 0.0
        for idx, j in enumerate(heur_seq):
            S[j].Start = cum
            if idx < len(heur_seq) - 1:
                cum += d[heur_seq[idx]][heur_seq[idx + 1]]
        C.Start = heur_ms

    # Callback for logging
    callback = GurobiCallback(x, n_jobs, logger)
    model.setCallback(callback)

    model.optimize()

    # Determine best solution
    best_obj = heur_ms if heur_seq is not None else float('inf')
    best_seq = heur_seq

    if model.SolCount > 0:
        obj_val = model.ObjVal
        mip_seq = extract_sequence_from_model(x, n_jobs)
        if mip_seq is not None and obj_val < best_obj - 1e-9:
            best_obj = obj_val
            best_seq = mip_seq
        elif mip_seq is not None and best_seq is None:
            best_obj = obj_val
            best_seq = mip_seq

    # Fallback: if no feasible solution found, output any sequence (ignoring due dates)
    if best_seq is None:
        best_seq = list(range(n_jobs))
        ms, feas = eval_seq(best_seq, d, T, due_dates)
        best_obj = ms if feas else float('inf')
        # If infeasible, still compute makespan
        if not feas:
            S_val = 0.0
            mk = 0.0
            for idx, j in enumerate(best_seq):
                comp = S_val + T[j]
                mk = max(mk, comp)
                if idx < len(best_seq) - 1:
                    S_val += d[j][best_seq[idx + 1]]
            best_obj = mk

    solution = {
        "objective_value": float(best_obj),
        "sequence": [j + 1 for j in best_seq]
    }

    if logger and (best_seq is not None):
        logger.log_solution(float(best_obj), solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()