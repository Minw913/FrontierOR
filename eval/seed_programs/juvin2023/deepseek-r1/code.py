import argparse
import json
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from solution_logger import SolutionLogger

def solve_with_heuristic(data):
    num_jobs = data['num_jobs']
    num_machines = data['num_machines']
    jobs = data['jobs']
    
    job_next = {}
    machine_next = [1] * num_machines
    assignment_dict = {}
    schedule_dict = {}
    
    for job in jobs:
        job_id = job['job_id']
        job_next[job_id] = 1
    
    for job in jobs:
        job_id = job['job_id']
        for op in job['operations']:
            op_id = op['operation_id']
            min_time = float('inf')
            best_machine = None
            
            for m in op['eligible_machines']:
                m_str = str(m)
                if m_str in op['processing_times']:
                    pt = op['processing_times'][m_str]
                    if pt < min_time:
                        min_time = pt
                        best_machine = m
                else:
                    pt = float('inf')
                    if pt < min_time:
                        min_time = pt
                        best_machine = m
            
            if best_machine is None:
                if op['eligible_machines']:
                    best_machine = op['eligible_machines'][0]
                    m_str = str(best_machine)
                    if m_str in op['processing_times']:
                        min_time = op['processing_times'][m_str]
                    else:
                        min_time = 0
                else:
                    best_machine = 0
                    min_time = 0
            
            p = min_time
            start_time = max(job_next[job_id], machine_next[best_machine])
            
            if p > 0:
                end_time = start_time + p - 1
                time_periods = list(range(int(start_time), int(end_time) + 1))
                job_next[job_id] = end_time + 1
                machine_next[best_machine] = end_time + 1
            else:
                time_periods = []
                job_next[job_id] = start_time + 1
            
            key = f"({job_id},{op_id})"
            assignment_dict[key] = best_machine
            schedule_dict[key] = time_periods
    
    makespan = max(job_next.values())
    
    solution = {
        "objective_value": makespan,
        "assignment": assignment_dict,
        "schedule": schedule_dict
    }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    if GUROBI_AVAILABLE:
        try:
            num_jobs = data['num_jobs']
            num_machines = data['num_machines']
            jobs = data['jobs']
            
            T = 0
            for job in jobs:
                for op in job['operations']:
                    max_time = 0
                    for m in op['eligible_machines']:
                        m_str = str(m)
                        if m_str in op['processing_times']:
                            pt = op['processing_times'][m_str]
                            if pt > max_time:
                                max_time = pt
                        else:
                            pt = max(op['processing_times'].values()) if op['processing_times'] else 0
                            max_time = max(max_time, pt)
                    T += max_time
            T = max(1, T)
            
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', args.time_limit)
            
            x = {}
            y = {}
            S = {}
            C = {}
            makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")
            
            machine_ops = {m: [] for m in range(num_machines)}
            for job in jobs:
                job_id = job['job_id']
                for op in job['operations']:
                    op_id = op['operation_id']
                    for m in op['eligible_machines']:
                        if m < num_machines:
                            machine_ops[m].append((job_id, op_id))
            
            for job in jobs:
                job_id = job['job_id']
                for op in job['operations']:
                    op_id = op['operation_id']
                    eligible_machines = [m for m in op['eligible_machines'] if m < num_machines]
                    
                    S[(job_id, op_id)] = model.addVar(lb=1, ub=T, vtype=GRB.CONTINUOUS, name=f"S_{job_id}_{op_id}")
                    C[(job_id, op_id)] = model.addVar(lb=1, ub=T, vtype=GRB.CONTINUOUS, name=f"C_{job_id}_{op_id}")
                    
                    for m in eligible_machines:
                        x[(job_id, op_id, m)] = model.addVar(vtype=GRB.BINARY, name=f"x_{job_id}_{op_id}_{m}")
                        for t in range(1, T+1):
                            y[(job_id, op_id, m, t)] = model.addVar(vtype=GRB.BINARY, name=f"y_{job_id}_{op_id}_{m}_{t}")
                    
                    model.addConstr(
                        gp.quicksum(x[(job_id, op_id, m)] for m in eligible_machines) == 1,
                        name=f"assign_{job_id}_{op_id}"
                    )
                    
                    for m in eligible_machines:
                        m_str = str(m)
                        pt = op['processing_times'].get(m_str, 0)
                        model.addConstr(
                            gp.quicksum(y[(job_id, op_id, m, t)] for t in range(1, T+1)) == pt * x[(job_id, op_id, m)],
                            name=f"process_{job_id}_{op_id}_{m}"
                        )
                        for t in range(1, T+1):
                            model.addConstr(
                                y[(job_id, op_id, m, t)] <= x[(job_id, op_id, m)],
                                name=f"link_{job_id}_{op_id}_{m}_{t}"
                            )
            
            for m in range(num_machines):
                for t in range(1, T+1):
                    y_vars = []
                    for (job_id, op_id) in machine_ops[m]:
                        if (job_id, op_id, m, t) in y:
                            y_vars.append(y[(job_id, op_id, m, t)])
                    if y_vars:
                        model.addConstr(
                            gp.quicksum(y_vars) <= 1,
                            name=f"machine_{m}_time_{t}"
                        )
            
            for job in jobs:
                job_id = job['job_id']
                operations = job['operations']
                for i in range(len(operations)-1):
                    op_id1 = operations[i]['operation_id']
                    op_id2 = operations[i+1]['operation_id']
                    model.addConstr(
                        C[(job_id, op_id1)] + 1 <= S[(job_id, op_id2)],
                        name=f"precedence_{job_id}_{op_id1}_{op_id2}"
                    )
            
            for job in jobs:
                job_id = job['job_id']
                for op in job['operations']:
                    op_id = op['operation_id']
                    s_var = S[(job_id, op_id)]
                    c_var = C[(job_id, op_id)]
                    for m in op['eligible_machines']:
                        if m < num_machines:
                            for t in range(1, T+1):
                                model.addConstr(
                                    s_var <= t + T * (1 - y.get((job_id, op_id, m, t), 0)),
                                    name=f"start_def_{job_id}_{op_id}_{m}_{t}"
                                )
                                model.addConstr(
                                    c_var >= t * y.get((job_id, op_id, m, t), 0),
                                    name=f"complete_def_{job_id}_{op_id}_{m}_{t}"
                                )
            
            for job in jobs:
                job_id = job['job_id']
                last_op_id = job['operations'][-1]['operation_id']
                model.addConstr(
                    makespan >= C[(job_id, last_op_id)] + 1,
                    name=f"makespan_job_{job_id}"
                )
            
            model.setObjective(makespan, GRB.MINIMIZE)
            
            if logger:
                model._logger = logger
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        model._logger.log(obj)
                model.optimize(callback)
            else:
                model.optimize()
            
            solution = {
                "objective_value": float('inf'),
                "assignment": {},
                "schedule": {}
            }
            
            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
                solution["objective_value"] = makespan.X
                for job in jobs:
                    job_id = job['job_id']
                    for op in job['operations']:
                        op_id = op['operation_id']
                        key = f"({job_id},{op_id})"
                        assigned_machine = None
                        for m in op['eligible_machines']:
                            if m < num_machines and (job_id, op_id, m) in x and x[(job_id, op_id, m)].X > 0.5:
                                assigned_machine = m
                                solution["assignment"][key] = assigned_machine
                                break
                        if assigned_machine is not None:
                            time_periods = []
                            for t in range(1, T+1):
                                if (job_id, op_id, assigned_machine, t) in y and y[(job_id, op_id, assigned_machine, t)].X > 0.5:
                                    time_periods.append(t)
                            solution["schedule"][key] = time_periods
                        else:
                            if op['eligible_machines']:
                                solution["assignment"][key] = op['eligible_machines'][0]
                            solution["schedule"][key] = []
            else:
                solution = solve_with_heuristic(data)
        except Exception as e:
            print(f"Error during MIP solving: {e}", file=sys.stderr)
            solution = solve_with_heuristic(data)
    else:
        solution = solve_with_heuristic(data)
    
    if logger:
        logger.log(solution['objective_value'])
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()