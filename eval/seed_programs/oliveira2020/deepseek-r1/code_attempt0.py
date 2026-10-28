import argparse
import json
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['parameters']['n']
    m = instance['parameters']['m']
    jobs = instance['jobs']
    H = instance['time_horizon']
    
    p = [job['processing_time'] for job in jobs]
    w_list = [job['weight'] for job in jobs]
    d = [job['due_date'] for job in jobs]
    
    model = grb.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    x = []
    s = []
    T = []
    y = {}
    
    for i in range(n):
        x_i = []
        for k in range(m):
            x_i.append(model.addVar(vtype=grb.GRB.BINARY, name=f"x_{i}_{k}"))
        x.append(x_i)
        s.append(model.addVar(lb=0, ub=H - p[i], vtype=grb.GRB.INTEGER, name=f"s_{i}"))
        T.append(model.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name=f"T_{i}"))
    
    for i in range(n):
        for j in range(i+1, n):
            y[(i, j)] = model.addVar(vtype=grb.GRB.BINARY, name=f"y_{i}_{j}")
    
    for i in range(n):
        model.addConstr(grb.quicksum(x[i][k] for k in range(m)) == 1, name=f"assign_{i}")
    
    for i in range(n):
        model.addConstr(T[i] >= s[i] + p[i] - d[i], name=f"tardiness_{i}")
    
    for i in range(n):
        for j in range(i+1, n):
            model.addConstr(s[i] + p[i] <= s[j] + H * (1 - y[(i, j)]), name=f"disj1_{i}_{j}")
            model.addConstr(s[j] + p[j] <= s[i] + H * y[(i, j)], name=f"disj2_{i}_{j}")
    
    model.setObjective(grb.quicksum(w_list[i] * T[i] for i in range(n)), grb.GRB.MINIMIZE)
    
    cumulative = 0
    for i in range(n):
        for k in range(m):
            if k == 0:
                x[i][k].start = 1
            else:
                x[i][k].start = 0
        s[i].start = cumulative
        cumulative += p[i]
        T[i].start = max(0, s[i].start + p[i] - d[i])
    
    for i in range(n):
        for j in range(i+1, n):
            y[(i, j)].start = 1
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    def callback(model, where):
        if logger and where == grb.GRB.Callback.MIPSOL:
            obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
            logger.log(obj)
    
    if logger:
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        obj_val = model.objVal
        x_vals = [None] * n
        s_vals = [0] * n
        for i in range(n):
            s_vals[i] = round(s[i].X)
            for k in range(m):
                if x[i][k].X > 0.5:
                    x_vals[i] = k
        
        machine_jobs = [[] for _ in range(m)]
        for i in range(n):
            k = x_vals[i]
            machine_jobs[k].append((s_vals[i], i))
        
        for k in range(m):
            machine_jobs[k].sort(key=lambda job: job[0])
        
        schedule_list = []
        for k in range(m):
            prev_id = 0
            for idx, (start, i) in enumerate(machine_jobs[k]):
                job_id = jobs[i]['id']
                completion = start + p[i]
                tardiness = max(0, completion - d[i])
                weighted_tardiness = w_list[i] * tardiness
                entry = {
                    "machine_id": k,
                    "prev_job": prev_id,
                    "job": job_id,
                    "start_time": start,
                    "completion_time": completion,
                    "weighted_tardiness": float(weighted_tardiness)
                }
                schedule_list.append(entry)
                prev_id = job_id
        
        solution_dict = {
            "objective_value": float(obj_val),
            "schedule": schedule_list
        }
    else:
        machine_jobs = [[] for _ in range(m)]
        cumulative = 0
        for i in range(n):
            machine_jobs[0].append((cumulative, i))
            cumulative += p[i]
        
        machine_jobs[0].sort(key=lambda job: job[0])
        schedule_list = []
        prev_id = 0
        for idx, (start, i) in enumerate(machine_jobs[0]):
            job_id = jobs[i]['id']
            completion = start + p[i]
            tardiness = max(0, completion - d[i])
            weighted_tardiness = w_list[i] * tardiness
            entry = {
                "machine_id": 0,
                "prev_job": prev_id,
                "job": job_id,
                "start_time": start,
                "completion_time": completion,
                "weighted_tardiness": float(weighted_tardiness)
            }
            schedule_list.append(entry)
            prev_id = job_id
        
        obj_val = sum(w_list[i] * max(0, start + p[i] - d[i]) for i in range(n))
        solution_dict = {
            "objective_value": float(obj_val),
            "schedule": schedule_list
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == '__main__':
    main()