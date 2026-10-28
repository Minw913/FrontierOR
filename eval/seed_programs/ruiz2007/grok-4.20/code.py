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

    n = instance['n_jobs']
    m = instance['n_machines']
    p = instance['processing_times']
    no_idle_flags = instance['no_idle_flags']
    no_idle_set = set(instance.get('no_idle_machines', []))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    best_obj = float('inf')
    best_seq = None
    best_C = None

    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', 1)
        env.setParam('TimeLimit', args.time_limit)
        env.start()
        model = gp.Model("no_idle_flowshop", env=env)

        # Variables
        C = model.addVars(n, m, vtype=GRB.CONTINUOUS, lb=0, name="C")
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        y = model.addVars(n, n, m, vtype=GRB.BINARY, name="y")
        makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="makespan")

        # Assignment constraints
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

        # Linearization of product x[i,j] * p[i,k]
        BigM = sum(sum(p[i][k] for k in range(m)) for i in range(n)) + 1
        for i in range(n):
            for j in range(n):
                for k in range(m):
                    model.addConstr(y[i, j, k] <= x[i, j])
                    model.addConstr(y[i, j, k] <= 1)
                    model.addConstr(y[i, j, k] >= 0)
                    model.addConstr(y[i, j, k] * BigM >= x[i, j] * p[i][k] - (1 - x[i, j]) * BigM)  # loose bound

        # Completion time constraints
        for j in range(n):
            # First machine
            model.addConstr(C[j, 0] == gp.quicksum(y[i, j, 0] for i in range(n)))

            # Subsequent machines
            for k in range(1, m):
                model.addConstr(C[j, k] >= C[j, k-1] + gp.quicksum(y[i, j, k] for i in range(n)))

        # Machine precedence and no-idle constraints
        for k in range(m):
            is_no_idle = no_idle_flags[k]
            for j in range(1, n):
                # Regular precedence on machine
                model.addConstr(C[j, k] >= C[j-1, k] + gp.quicksum(y[i, j, k] for i in range(n)))

                if is_no_idle:
                    # No idle: exact equality
                    model.addConstr(C[j, k] == C[j-1, k] + gp.quicksum(y[i, j, k] for i in range(n)))

        # Makespan
        model.addConstr(makespan == C[n-1, m-1])
        model.setObjective(makespan, GRB.MINIMIZE)

        # Callback to log incumbents
        def callback(model, where):
            nonlocal best_obj, best_seq, best_C
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj_val < best_obj - 1e-6:
                    best_obj = obj_val
                    xval = model.cbGetSolution(x)
                    seq = []
                    for j in range(n):
                        for i in range(n):
                            if xval[i, j] > 0.5:
                                seq.append(i)
                                break
                    if logger:
                        logger.log(float(obj_val))
                    # Also try to capture completion times
                    Cval = model.cbGetSolution(C)
                    C_dict = {}
                    for k in range(m):
                        clist = [Cval[j, k] for j in range(n)]
                        C_dict[f"machine_{k}"] = [float(v) for v in clist]
                    best_seq = seq
                    best_C = C_dict

        model.optimize(callback)

        # Extract solution
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            if model.SolCount > 0:
                obj_val = model.objVal
                xval = model.getAttr('X', x)
                seq = []
                for j_pos in range(n):
                    for i_job in range(n):
                        if xval[i_job, j_pos] > 0.5:
                            seq.append(i_job)
                            break
                Cval = model.getAttr('X', C)
                C_dict = {}
                for k in range(m):
                    clist = [Cval[j, k] for j in range(n)]
                    C_dict[f"machine_{k}"] = [float(round(v, 6)) for v in clist]

                final_obj = float(obj_val)
                if logger and (best_obj > final_obj + 1e-6):
                    logger.log(final_obj)

                solution = {
                    "objective_value": final_obj,
                    "sequence": seq,
                    "completion_times": C_dict
                }

                with open(args.solution_path, 'w') as f:
                    json.dump(solution, f, indent=2)
                return

    except Exception as e:
        print("Error during optimization:", str(e), file=sys.stderr)

    # Fallback: simple heuristic - use Johnson's rule approximation or random
    # For general m we use NEH-like greedy
    if best_seq is None:
        # Very basic: sort by total processing time
        total_p = [sum(p[i]) for i in range(n)]
        seq = sorted(range(n), key=lambda i: total_p[i])
        # Evaluate this sequence
        C_eval = compute_completion_times(seq, p, no_idle_flags, n, m)
        obj = C_eval[n-1][m-1]
        C_dict = {}
        for k in range(m):
            C_dict[f"machine_{k}"] = [float(C_eval[j][k]) for j in range(n)]
        solution = {
            "objective_value": float(obj),
            "sequence": [int(s) for s in seq],
            "completion_times": C_dict
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log(float(obj))
        return

    # If we have a solution from MIP but didn't write it
    if best_seq is not None:
        solution = {
            "objective_value": float(best_obj),
            "sequence": best_seq,
            "completion_times": best_C if best_C is not None else {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

def compute_completion_times(seq, p, no_idle_flags, n, m):
    C = [[0.0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        job = seq[j]
        C[j][0] = p[job][0]
        if j > 0:
            C[j][0] = max(C[j][0], C[j-1][0] + p[job][0]) if not no_idle_flags[0] else C[j-1][0] + p[job][0]
    for k in range(1, m):
        job = seq[0]
        C[0][k] = C[0][k-1] + p[job][k]
        for j in range(1, n):
            job = seq[j]
            finish_prev_machine = C[j][k-1] + p[job][k]
            finish_prev_job = C[j-1][k] + p[job][k]
            C[j][k] = max(finish_prev_machine, finish_prev_job)
            if no_idle_flags[k]:
                C[j][k] = C[j-1][k] + p[job][k]
    return C

if __name__ == "__main__":
    main()