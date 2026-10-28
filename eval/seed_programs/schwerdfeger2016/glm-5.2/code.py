import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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

    m = instance['num_machines']
    n = instance['num_jobs']
    p = instance['processing_times']

    total_p = sum(p)
    if m == 0 or total_p == 0:
        assignment = {str(i): [] for i in range(m)}
        machine_completion_times = {str(i): 0.0 for i in range(m)}
        sol_dict = {
            "objective_value": 0.0,
            "assignment": assignment,
            "machine_completion_times": machine_completion_times
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)
        return

    avg_c = total_p / m

    model = gp.Model("load_balancing")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    x = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    C = {}
    for i in range(m):
        C[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{i}")
        model.addConstr(C[i] == gp.quicksum(p[j] * x[i, j] for j in range(n)))

    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == 1)

    model.setObjective(gp.quicksum(C[i] * C[i] for i in range(m)), GRB.MINIMIZE)

    # Provide an initial LPT (Longest Processing Time first) solution
    loads = [0] * m
    for j in range(n):
        min_idx = loads.index(min(loads))
        x[min_idx, j].Start = 1
        loads[min_idx] += p[j]
    for i in range(m):
        C[i].Start = loads[i]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                x_vals = model.cbGetSolution(list(x.values()))
                C_vals = model.cbGetSolution(list(C.values()))
                
                x_keys = list(x.keys())
                C_keys = list(C.keys())
                
                assignment = {str(i): [] for i in range(m)}
                machine_completion_times = {str(i): 0.0 for i in range(m)}
                
                for idx, key in enumerate(C_keys):
                    machine_completion_times[str(key)] = float(C_vals[idx])
                
                for idx, key in enumerate(x_keys):
                    i, j = key
                    if x_vals[idx] > 0.5:
                        assignment[str(i)].append(j)
                
                norm_obj = ((sum((C_vals[idx] - avg_c)**2 for idx in range(m)))**0.5) / avg_c
                sol_dict = {
                    "objective_value": float(norm_obj),
                    "assignment": assignment,
                    "machine_completion_times": machine_completion_times
                }
                logger.log_solution(float(norm_obj), sol_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        x_vals = {k: v.X for k, v in x.items()}
        C_vals = {k: v.X for k, v in C.items()}
        
        assignment = {str(i): [] for i in range(m)}
        machine_completion_times = {str(i): 0.0 for i in range(m)}
        
        for i in range(m):
            machine_completion_times[str(i)] = float(C_vals[i])
            for j in range(n):
                if x_vals[i, j] > 0.5:
                    assignment[str(i)].append(j)
        
        norm_obj = ((sum((C_vals[i] - avg_c)**2 for i in range(m)))**0.5) / avg_c
        sol_dict = {
            "objective_value": float(norm_obj),
            "assignment": assignment,
            "machine_completion_times": machine_completion_times
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)
    else:
        # Fallback heuristic if no solution found
        loads = [0] * m
        assignment = {str(i): [] for i in range(m)}
        machine_completion_times = {str(i): 0.0 for i in range(m)}
        for j in range(n):
            min_idx = loads.index(min(loads))
            assignment[str(min_idx)].append(j)
            loads[min_idx] += p[j]
            machine_completion_times[str(min_idx)] = float(loads[min_idx])
        
        norm_obj = ((sum((loads[i] - avg_c)**2 for i in range(m)))**0.5) / avg_c
        sol_dict = {
            "objective_value": float(norm_obj),
            "assignment": assignment,
            "machine_completion_times": machine_completion_times
        }
        if logger:
            logger.log_solution(float(norm_obj), sol_dict)
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)

if __name__ == '__main__':
    main()