import argparse
import json
import sys
import time


def compute_makespan(sequence, processing_times, no_idle_flags):
    """Compute makespan and completion times for a given sequence.
    
    For no-idle machines, processing is continuous: once the first job starts,
    every subsequent job starts immediately after the previous one finishes.
    The start time of the first job on a no-idle machine is chosen as the
    smallest value that satisfies all job-precedence constraints.
    """
    n = len(sequence)
    m = len(no_idle_flags)
    C = [[0] * m for _ in range(n)]

    # Machine 0: no job-precedence, so start at time 0 regardless of idle/no-idle.
    cumul = 0
    for k in range(n):
        cumul += processing_times[sequence[k]][0]
        C[k][0] = cumul

    # Machines 1 .. m-1
    for machine in range(1, m):
        if no_idle_flags[machine]:
            # Compute minimal start time S >= 0 such that
            #   S + sum_{i<=k} p[seq[i]][m]  >=  C[k][m-1] + p[seq[k]][m]   for all k
            # i.e. S >= C[k][m-1] - sum_{i<k} p[seq[i]][m]
            S = 0
            cumul_before = 0
            for k in range(n):
                lower = C[k][machine - 1] - cumul_before
                if lower > S:
                    S = lower
                cumul_before += processing_times[sequence[k]][machine]
            cumul = 0
            for k in range(n):
                cumul += processing_times[sequence[k]][machine]
                C[k][machine] = S + cumul
        else:
            for k in range(n):
                p = processing_times[sequence[k]][machine]
                if k == 0:
                    C[k][machine] = C[k][machine - 1] + p
                else:
                    C[k][machine] = max(C[k - 1][machine] + p,
                                        C[k][machine - 1] + p)

    return C[n - 1][m - 1], C


def neh_heuristic(processing_times, no_idle_flags):
    """NEH insertion heuristic: insert jobs in decreasing total processing time."""
    n = len(processing_times)
    m = len(no_idle_flags)
    total = [sum(processing_times[j][mm] for mm in range(m)) for j in range(n)]
    order = sorted(range(n), key=lambda j: -total[j])

    best_seq = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(best_seq) + 1):
            trial = best_seq[:pos] + [job] + best_seq[pos:]
            ms, _ = compute_makespan(trial, processing_times, no_idle_flags)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        best_seq = best_seq[:best_pos] + [job] + best_seq[best_pos:]

    return best_seq


def local_search(sequence, processing_times, no_idle_flags, deadline):
    """First-improvement local search with insert and swap moves."""
    n = len(sequence)
    best_ms, _ = compute_makespan(sequence, processing_times, no_idle_flags)
    best_seq = list(sequence)
    improved = True

    while improved and time.time() < deadline:
        improved = False

        # Insert moves
        for i in range(n):
            if time.time() >= deadline:
                break
            job = best_seq[i]
            trial = best_seq[:i] + best_seq[i + 1:]
            for j in range(n):
                new_seq = trial[:j] + [job] + trial[j:]
                ms, _ = compute_makespan(new_seq, processing_times, no_idle_flags)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # Swap moves
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                new_seq = list(best_seq)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                ms, _ = compute_makespan(new_seq, processing_times, no_idle_flags)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break

    return best_seq


def build_solution_dict(sequence, C, n, m):
    completion_times = {}
    for machine in range(m):
        completion_times[f'machine_{machine}'] = [int(round(C[k][machine])) for k in range(n)]
    return {
        'objective_value': float(C[n - 1][m - 1]),
        'sequence': [int(j) for j in sequence],
        'completion_times': completion_times
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    n = instance['n_jobs']
    m = instance['n_machines']
    no_idle_flags = instance['no_idle_flags']
    processing_times = instance['processing_times']

    start = time.time()
    deadline = start + args.time_limit

    # ---- Edge cases ----
    if n == 0:
        solution = {
            'objective_value': 0.0,
            'sequence': [],
            'completion_times': {f'machine_{i}': [] for i in range(m)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0.0, solution)
        return

    if n == 1:
        ms, C = compute_makespan([0], processing_times, no_idle_flags)
        solution = build_solution_dict([0], C, n, m)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(ms, solution)
        return

    # ---- Phase 1: NEH heuristic ----
    seq = neh_heuristic(processing_times, no_idle_flags)
    ms, C = compute_makespan(seq, processing_times, no_idle_flags)
    best_seq = seq
    best_ms = ms
    best_C = C
    if logger:
        logger.log_solution(ms, build_solution_dict(seq, C, n, m))

    # ---- Phase 2: Local search ----
    ls_deadline = min(start + args.time_limit * 0.4, deadline)
    seq = local_search(best_seq, processing_times, no_idle_flags, ls_deadline)
    ms, C = compute_makespan(seq, processing_times, no_idle_flags)
    if ms < best_ms - 1e-9:
        best_seq = seq
        best_ms = ms
        best_C = C
        if logger:
            logger.log_solution(ms, build_solution_dict(seq, C, n, m))

    # ---- Phase 3: Gurobi MIP (for small instances) ----
    remaining = deadline - time.time()
    use_mip = remaining > 3 and n <= 40

    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()

            model = gp.Model(env=env)
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(remaining - 2, 1))

            x = {}
            for j in range(n):
                for k in range(n):
                    x[j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{j}_{k}')

            Cvar = {}
            for k in range(n):
                for machine in range(m):
                    Cvar[k, machine] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                     name=f'C_{k}_{machine}')

            # Assignment constraints
            for j in range(n):
                model.addConstr(gp.quicksum(x[j, k] for k in range(n)) == 1)
            for k in range(n):
                model.addConstr(gp.quicksum(x[j, k] for j in range(n)) == 1)

            # Machine 0
            for k in range(n):
                p_expr = gp.quicksum(processing_times[j][0] * x[j, k]
                                     for j in range(n))
                if k == 0:
                    model.addConstr(Cvar[k, 0] == p_expr)
                else:
                    if no_idle_flags[0]:
                        model.addConstr(Cvar[k, 0] == Cvar[k - 1, 0] + p_expr)
                    else:
                        model.addConstr(Cvar[k, 0] >= Cvar[k - 1, 0] + p_expr)

            # Machines 1 .. m-1
            for machine in range(1, m):
                for k in range(n):
                    p_expr = gp.quicksum(processing_times[j][machine] * x[j, k]
                                         for j in range(n))
                    # Job precedence
                    model.addConstr(Cvar[k, machine] >= Cvar[k, machine - 1] + p_expr)
                    if k > 0:
                        if no_idle_flags[machine]:
                            model.addConstr(Cvar[k, machine] == Cvar[k - 1, machine] + p_expr)
                        else:
                            model.addConstr(Cvar[k, machine] >= Cvar[k - 1, machine] + p_expr)

            model.setObjective(Cvar[n - 1, m - 1], GRB.MINIMIZE)

            # Warm start
            for j in range(n):
                for k in range(n):
                    x[j, k].Start = 1 if best_seq[k] == j else 0

            # Callback for logging incumbants found during B&B
            state = {'best_ms': best_ms}

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < state['best_ms'] - 1e-9:
                        x_vals = model.cbGetSolution(list(x.values()))
                        x_val_dict = {}
                        idx = 0
                        for j in range(n):
                            for k in range(n):
                                x_val_dict[j, k] = x_vals[idx]
                                idx += 1
                        new_seq = [0] * n
                        for j in range(n):
                            for k in range(n):
                                if x_val_dict[j, k] > 0.5:
                                    new_seq[k] = j
                                    break
                        ms2, C2 = compute_makespan(new_seq, processing_times, no_idle_flags)
                        if ms2 < state['best_ms']:
                            state['best_ms'] = ms2
                            if logger:
                                logger.log_solution(ms2, build_solution_dict(new_seq, C2, n, m))

            model.optimize(callback)

            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val < best_ms - 1e-9:
                    new_seq = [0] * n
                    for j in range(n):
                        for k in range(n):
                            if x[j, k].X > 0.5:
                                new_seq[k] = j
                    ms2, C2 = compute_makespan(new_seq, processing_times, no_idle_flags)
                    if ms2 < best_ms:
                        best_seq = new_seq
                        best_ms = ms2
                        best_C = C2
                        if logger:
                            logger.log_solution(ms2, build_solution_dict(new_seq, C2, n, m))

        except Exception as e:
            print(f"Gurobi error: {e}", file=sys.stderr)

    # ---- Phase 4: Final local search with remaining time ----
    remaining = deadline - time.time()
    if remaining > 1:
        seq = local_search(best_seq, processing_times, no_idle_flags, deadline)
        ms, C = compute_makespan(seq, processing_times, no_idle_flags)
        if ms < best_ms - 1e-9:
            best_seq = seq
            best_ms = ms
            best_C = C
            if logger:
                logger.log_solution(ms, build_solution_dict(seq, C, n, m))

    # ---- Write final solution ----
    solution = build_solution_dict(best_seq, best_C, n, m)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()