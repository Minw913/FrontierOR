import argparse
import json
import time
import sys
import bisect

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path) as f:
        data = json.load(f)
    n = data['n']
    m = data['m']
    jobs = data['jobs']
    p = [0] * n
    w = [0] * n
    j_orig = [0] * n
    for idx, job in enumerate(jobs):
        j_orig[idx] = job['j']
        p[idx] = job['p']
        w[idx] = job['w']
    return n, m, p, w, j_orig


def compute_C_and_obj(assign, n, m, p, w):
    C = [0] * n
    obj = 0
    machine_jobs = [[] for _ in range(m)]
    for i in range(n):
        machine_jobs[assign[i]].append(i)
    for k in range(m):
        t = 0
        for i in machine_jobs[k]:
            t += p[i]
            C[i] = t
            obj += w[i] * C[i]
    return C, obj


def greedy_heuristic(n, m, p, w):
    loads = [0] * m
    assign = [-1] * n
    for i in range(n):
        min_load = loads[0]
        min_k = 0
        for k in range(1, m):
            if loads[k] < min_load:
                min_load = loads[k]
                min_k = k
        assign[i] = min_k
        loads[min_k] += p[i]
    C, obj = compute_C_and_obj(assign, n, m, p, w)
    return assign, C, obj


def local_search(assign, n, m, p, w, max_passes=10):
    # Build machine data structures
    mach_jobs = [[] for _ in range(m)]
    for i in range(n):
        mach_jobs[assign[i]].append(i)
    mach_pref_p = []
    mach_pref_w = []
    mach_total_w = []
    for k in range(m):
        jobs = mach_jobs[k]
        pref_p = [0] * (len(jobs) + 1)
        pref_w = [0] * (len(jobs) + 1)
        for idx, j in enumerate(jobs):
            pref_p[idx + 1] = pref_p[idx] + p[j]
            pref_w[idx + 1] = pref_w[idx] + w[j]
        mach_pref_p.append(pref_p)
        mach_pref_w.append(pref_w)
        mach_total_w.append(pref_w[-1])

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            A = assign[i]
            for B in range(m):
                if B == A:
                    continue
                jobsA = mach_jobs[A]
                posA = bisect.bisect_left(jobsA, i)
                w_after_A = mach_total_w[A] - mach_pref_w[A][posA + 1]
                jobsB = mach_jobs[B]
                posB = bisect.bisect_left(jobsB, i)
                p_before_B = mach_pref_p[B][posB]
                w_after_B = mach_total_w[B] - mach_pref_w[B][posB]
                old_C_i = mach_pref_p[A][posA + 1]
                new_C_i = p_before_B + p[i]
                delta = w[i] * (new_C_i - old_C_i) - p[i] * w_after_A + p[i] * w_after_B
                if delta < -1e-6:
                    # Apply move
                    jobsA.pop(posA)
                    jobsB.insert(posB, i)
                    assign[i] = B
                    # Recompute A
                    pref_pA = [0] * (len(jobsA) + 1)
                    pref_wA = [0] * (len(jobsA) + 1)
                    for idx, j in enumerate(jobsA):
                        pref_pA[idx + 1] = pref_pA[idx] + p[j]
                        pref_wA[idx + 1] = pref_wA[idx] + w[j]
                    mach_pref_p[A] = pref_pA
                    mach_pref_w[A] = pref_wA
                    mach_total_w[A] = pref_wA[-1]
                    # Recompute B
                    pref_pB = [0] * (len(jobsB) + 1)
                    pref_wB = [0] * (len(jobsB) + 1)
                    for idx, j in enumerate(jobsB):
                        pref_pB[idx + 1] = pref_pB[idx] + p[j]
                        pref_wB[idx + 1] = pref_wB[idx] + w[j]
                    mach_pref_p[B] = pref_pB
                    mach_pref_w[B] = pref_wB
                    mach_total_w[B] = pref_wB[-1]
                    improved = True
                    break
        if not improved:
            break
    C, obj = compute_C_and_obj(assign, n, m, p, w)
    return assign, C, obj


def build_solution_dict(assign, C, n, m, p, w, j_orig):
    machine_jobs = [[] for _ in range(m)]
    for i in range(n):
        machine_jobs[assign[i]].append(i)
    machines = []
    for k in range(m):
        jobs_on_k = machine_jobs[k]
        schedule_cost = 0
        job_list = []
        for i in jobs_on_k:
            job_list.append({
                "j_orig": j_orig[i],
                "p": p[i],
                "w": w[i],
                "C_j": int(C[i])
            })
            schedule_cost += w[i] * C[i]
        machines.append({
            "schedule_cost": int(schedule_cost),
            "jobs": job_list
        })
    total_obj = sum(mach["schedule_cost"] for mach in machines)
    return {
        "objective_value": float(total_obj),
        "machines": machines
    }


def solve_with_gurobi(n, m, p, w, time_limit, start_assign, start_C, start_obj, logger):
    if not GUROBI_AVAILABLE:
        return None, None, None
    try:
        env = gp.Env(empty=True)
        env.setParam('Seed', 0)
        env.setParam('MIPGap', 1e-4)
        env.setParam('NumericFocus', 0)
        env.setParam('Threads', 1)
        env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model(env=env)
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0

        # Variables
        x = {}
        for i in range(n):
            for k in range(m):
                x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
        S = {}
        for i in range(n):
            for k in range(m):
                S[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{k}")
        C = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="C")

        # Constraints
        for i in range(n):
            for k in range(m):
                if i == 0:
                    model.addConstr(S[i, k] == p[i] * x[i, k])
                else:
                    model.addConstr(S[i, k] == S[i - 1, k] + p[i] * x[i, k])
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, k] for k in range(m)) == 1)
        for i in range(n):
            for k in range(m):
                model.addConstr((x[i, k] == 1) >> (C[i] == S[i, k]))

        model.setObjective(gp.quicksum(w[i] * C[i] for i in range(n)), GRB.MINIMIZE)

        # Start solution
        if start_assign is not None:
            for i in range(n):
                for k in range(m):
                    x[i, k].Start = 1 if start_assign[i] == k else 0
            for i in range(n):
                C[i].Start = start_C[i]
            for k in range(m):
                s = 0
                for i in range(n):
                    if start_assign[i] == k:
                        s += p[i]
                    S[i, k].Start = s

        # Callback for logging incumbents
        class IncumbentCallback:
            def __init__(self, logger, best_obj):
                self.logger = logger
                self.best_obj = best_obj

            def __call__(self, model, where):
                if where == GRB.Callback.MIP:
                    try:
                        obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                        if obj < GRB.INFINITY and obj < self.best_obj - 1e-6:
                            self.best_obj = obj
                            if self.logger:
                                self.logger.log(obj)
                    except:
                        pass

        callback = IncumbentCallback(logger, start_obj)
        model.optimize(callback)

        if model.SolCount > 0:
            assign = [0] * n
            C_val = [0] * n
            for i in range(n):
                for k in range(m):
                    if round(x[i, k].X) > 0.5:
                        assign[i] = k
                        C_val[i] = round(C[i].X)
                        break
            obj_val = model.ObjVal
            return assign, C_val, obj_val
        else:
            return None, None, None
    except Exception as e:
        print(f"Gurobi error: {e}", file=sys.stderr)
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    n, m, p, w, j_orig = read_instance(args.instance_path)
    if n == 0:
        sol = {"objective_value": 0.0, "machines": []}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # Greedy heuristic
    assign_heur, C_heur, obj_heur = greedy_heuristic(n, m, p, w)
    # Local search improvement
    assign_ls, C_ls, obj_ls = local_search(assign_heur, n, m, p, w, max_passes=10)

    best_assign = assign_ls
    best_C = C_ls
    best_obj = obj_ls

    sol_dict = build_solution_dict(best_assign, best_C, n, m, p, w, j_orig)
    if logger:
        logger.log_solution(best_obj, sol_dict)

    # Gurobi MIP
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 1
    if remaining > 1:
        assign_mip, C_mip, obj_mip = solve_with_gurobi(
            n, m, p, w, remaining, best_assign, best_C, best_obj, logger
        )
        if assign_mip is not None and obj_mip < best_obj - 1e-6:
            best_assign = assign_mip
            best_C = C_mip
            best_obj = obj_mip
            sol_dict = build_solution_dict(best_assign, best_C, n, m, p, w, j_orig)
            if logger:
                logger.log_solution(best_obj, sol_dict)

    # Write final solution
    final_sol = build_solution_dict(best_assign, best_C, n, m, p, w, j_orig)
    with open(args.solution_path, 'w') as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()