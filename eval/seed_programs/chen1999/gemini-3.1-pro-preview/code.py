import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path where the solution JSON file will be saved.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the JSONL logging file.")
    return parser.parse_args()

def solve_weighted_completion_time(instance, time_limit, log_path, solution_path):
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    w = instance["jobs"]["weights"]
    p = instance["jobs"]["processing_times"]

    logger = None
    if log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(log_path, sense="minimize")
        except ImportError:
            pass

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj_val)

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 1)
    env.start()

    model = gp.Model("WeightedCompletionTime", env=env)
    
    # Binary variable: x[j, m] = 1 if job j is assigned to machine m
    x = model.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name="x")
    
    # Each job must be assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[j, m] for m in range(num_machines)) == 1, name=f"assign_{j}")
        # Provide a trivial starting solution (assign all to machine 0)
        x[j, 0].Start = 1
        
    coeffs = []
    vars1 = []
    vars2 = []
    
    # Collect linear and quadratic terms for the objective
    # Total weighted completion time can be evaluated exactly by Smith's ratio rule sequences.
    for j in range(num_jobs):
        for m in range(num_machines):
            coeffs.append(w[j] * p[j][m])
            vars1.append(x[j, m])
            vars2.append(x[j, m])
            
    for m in range(num_machines):
        # Determine the Smith's optimal processing sequence if assigned to machine m
        sorted_jobs = sorted(range(num_jobs), key=lambda j: (p[j][m] / w[j] if w[j] > 0 else float('inf'), j))
        for idx_j in range(len(sorted_jobs)):
            j = sorted_jobs[idx_j]
            for idx_i in range(idx_j):
                i = sorted_jobs[idx_i]
                coeff = w[j] * p[i][m]
                if coeff > 0:
                    coeffs.append(coeff)
                    vars1.append(x[i, m])
                    vars2.append(x[j, m])
                    
    obj = gp.QuadExpr()
    obj.addTerms(coeffs, vars1, vars2)
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    model.setParam('PreQLinearize', -1)  # Allow Gurobi algorithm choice for MIQP linearization
    
    model.optimize(cb)
    
    # Information extraction logic (Fallback to dummy assignment if optimal wasn't found within time)
    schedule = {str(m): [] for m in range(num_machines)}
    completion_times = {}
    total_obj = 0.0
    
    assigned_machine = [-1] * num_jobs
    if model.SolCount > 0:
        for j in range(num_jobs):
            for m in range(num_machines):
                if x[j, m].X > 0.5:
                    assigned_machine[j] = m
                    break
    else:
        for j in range(num_jobs): assigned_machine[j] = 0
            
    for m in range(num_machines):
        m_jobs = [j for j in range(num_jobs) if assigned_machine[j] == m]
        # Re-sort using Smith's exact sequence rule
        m_jobs.sort(key=lambda j: (p[j][m] / w[j] if w[j] > 0 else float('inf'), j))
        schedule[str(m)] = m_jobs
        
        curr_time = 0
        for j in m_jobs:
            curr_time += p[j][m]
            completion_times[str(j)] = curr_time
            total_obj += w[j] * curr_time
            
    solution = {
        "objective_value": float(total_obj),
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [],
        "on_time_jobs": list(range(num_jobs))
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def solve_weighted_tardy_jobs(instance, time_limit, log_path, solution_path):
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    w = instance["jobs"]["weights"]
    p = instance["jobs"]["processing_times"]
    d = instance["jobs"]["due_dates"]

    logger = None
    if log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(log_path, sense="minimize")
        except ImportError:
            pass

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj_val)

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 1)
    env.start()

    model = gp.Model("WeightedTardyJobs", env=env)
    
    # Binary variables encoding processing status and continuous sequence tracking
    x = model.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name="x")
    t = model.addVars(num_jobs, vtype=GRB.BINARY, name="t")
    C = model.addVars(num_jobs, num_machines, vtype=GRB.CONTINUOUS, lb=0, name="C")
    
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[j, m] for m in range(num_machines)) + t[j] == 1, name=f"assign_{j}")
        # Trivial start logic (all jobs tardy initially)
        t[j].Start = 1
        for m in range(num_machines):
            x[j, m].Start = 0

    # Adhere statically to Moore-Hodgson/EDD optimality rule 
    sorted_jobs = sorted(range(num_jobs), key=lambda j: (d[j], j))
    
    for m in range(num_machines):
        cum_p = 0
        for idx, j in enumerate(sorted_jobs):
            p_jm = p[j][m]
            cum_p += p_jm
            
            if idx == 0:
                model.addConstr(C[j, m] == p_jm * x[j, m])
            else:
                prev_j = sorted_jobs[idx - 1]
                model.addConstr(C[j, m] == C[prev_j, m] + p_jm * x[j, m])
                
            M_jm = max(0, cum_p - d[j])
            if M_jm == 0:
                model.addConstr(C[j, m] <= d[j])
            else:
                model.addConstr(C[j, m] <= d[j] + M_jm * (1 - x[j, m]))
                
    # Search space constraint tightening based on individual impossibility
    for j in range(num_jobs):
        for m in range(num_machines):
            if p[j][m] > d[j]:
                model.addConstr(x[j, m] == 0)

    model.setObjective(gp.quicksum(w[j] * t[j] for j in range(num_jobs)), GRB.MINIMIZE)

    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    
    model.optimize(cb)
    
    # Information extraction logic (Fallback to trivial all-tardy if optimal wasn't reachable)
    schedule = {str(m): [] for m in range(num_machines)}
    completion_times = {}
    tardy_jobs = []
    on_time_jobs = []
    total_obj = 0.0
    
    if model.SolCount > 0:
        for j in range(num_jobs):
            is_on_time = False
            for m in range(num_machines):
                if x[j, m].X > 0.5:
                    is_on_time = True
                    on_time_jobs.append(j)
                    schedule[str(m)].append(j)
                    break
            if not is_on_time:
                tardy_jobs.append(j)
                total_obj += w[j]
    else:
        tardy_jobs = list(range(num_jobs))
        total_obj = sum(w)
        
    for m in range(num_machines):
        # Sort effectively via exact EDD to reconstruct timeline
        m_jobs = schedule[str(m)]
        m_jobs.sort(key=lambda j: (d[j], j))
        schedule[str(m)] = m_jobs
        
        curr_time = 0
        for j in m_jobs:
            curr_time += p[j][m]
            completion_times[str(j)] = curr_time
            
    solution = {
        "objective_value": float(total_obj),
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": tardy_jobs,
        "on_time_jobs": on_time_jobs
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    args = parse_args()
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    problem_type = instance.get("problem_type")
    
    if problem_type == "weighted_completion_time":
        solve_weighted_completion_time(instance, args.time_limit, args.log_path, args.solution_path)
    elif problem_type == "weighted_tardy_jobs":
        solve_weighted_tardy_jobs(instance, args.time_limit, args.log_path, args.solution_path)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

if __name__ == '__main__':
    main()