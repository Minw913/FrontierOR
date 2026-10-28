import argparse
import json
import time
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_tasks = data['num_tasks']
    cycle_time = data['cycle_time']
    num_stations = data['num_stations']
    total_processing_time = data['total_processing_time']
    task_processing_times = {int(k): v for k, v in data['task_processing_times'].items()}
    precedence_arcs = data['precedence_arcs']
    earliest_station = {int(k): v for k, v in data['earliest_station'].items()}
    latest_station = {int(k): v for k, v in data['latest_station'].items()}
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    tasks = list(range(1, num_tasks + 1))
    
    # Build predecessor/successor lists
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for arc in precedence_arcs:
        pred, succ = arc[0], arc[1]
        successors[pred].append(succ)
        predecessors[succ].append(pred)
    
    # Compute lower bound on objective
    total_slack = num_stations * cycle_time - total_processing_time
    lb_objective = total_slack // cycle_time  # floor division
    
    # Try Gurobi MIP approach
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1:
            raise Exception("No time left")
        
        model = gp.Model("assembly_line_balancing")
        model.setParam('TimeLimit', max(1, remaining_time - 2))
        model.setParam('Threads', 1)
        model.setParam('OutputFlag', 0)
        
        # Decision variables: x[task, station] = 1 if task assigned to station
        x = {}
        for t in tasks:
            es = earliest_station[t]
            ls = latest_station[t]
            for s in range(es, ls + 1):
                x[t, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{s}")
        
        # Objective variable: worst-case idle time
        z = model.addVar(vtype=GRB.INTEGER, lb=lb_objective, ub=cycle_time, name="z")
        
        model.setObjective(z, GRB.MINIMIZE)
        
        # Each task assigned to exactly one station
        for t in tasks:
            es = earliest_station[t]
            ls = latest_station[t]
            model.addConstr(
                gp.quicksum(x[t, s] for s in range(es, ls + 1)) == 1,
                name=f"assign_{t}"
            )
        
        # Station capacity: sum of processing times <= cycle_time
        # Also: idle time at each station <= z
        # i.e., cycle_time - sum_processing <= z
        # i.e., sum_processing >= cycle_time - z
        for s in range(1, num_stations + 1):
            tasks_at_s = [t for t in tasks if earliest_station[t] <= s <= latest_station[t]]
            if tasks_at_s:
                load = gp.quicksum(task_processing_times[t] * x[t, s] for t in tasks_at_s)
                model.addConstr(load <= cycle_time, name=f"cap_{s}")
                model.addConstr(load + z >= cycle_time, name=f"idle_{s}")
            else:
                # Empty station: idle time = cycle_time, so z >= cycle_time
                model.addConstr(z >= cycle_time, name=f"empty_{s}")
        
        # Precedence constraints: station(pred) <= station(succ)
        # Using cumulative formulation for tighter bounds
        for arc in precedence_arcs:
            pred, succ = arc[0], arc[1]
            es_p = earliest_station[pred]
            ls_p = latest_station[pred]
            es_s = earliest_station[succ]
            ls_s = latest_station[succ]
            
            # sum_s(s * x[pred,s]) <= sum_s(s * x[succ,s])
            model.addConstr(
                gp.quicksum(s * x[pred, s] for s in range(es_p, ls_p + 1)) <=
                gp.quicksum(s * x[succ, s] for s in range(es_s, ls_s + 1)),
                name=f"prec_{pred}_{succ}"
            )
        
        # Callback to log incumbent solutions
        best_obj = [float('inf')]
        best_assignment = [None]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGetSolution(z)
                if obj_val < best_obj[0] - 0.5:
                    best_obj[0] = obj_val
                    assignment = {}
                    for t in tasks:
                        es = earliest_station[t]
                        ls = latest_station[t]
                        for s in range(es, ls + 1):
                            val = model.cbGetSolution(x[t, s])
                            if val > 0.5:
                                assignment[t] = s
                                break
                    best_assignment[0] = assignment
                    if logger:
                        logger.log(int(round(obj_val)))
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj_val = int(round(z.X))
            assignment = {}
            for t in tasks:
                es = earliest_station[t]
                ls = latest_station[t]
                for s in range(es, ls + 1):
                    if x[t, s].X > 0.5:
                        assignment[t] = s
                        break
            
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                best_assignment[0] = assignment
                if logger:
                    logger.log(obj_val)
            
            solution = {
                "objective_value": float(best_obj[0]),
                "task_assignments": {str(t): best_assignment[0][t] for t in tasks}
            }
            
            with open(solution_path, 'w') as f:
                json.dump(solution, f)
            return
        
        # If Gurobi found no solution, fall through to heuristic
        if best_assignment[0] is not None:
            solution = {
                "objective_value": float(best_obj[0]),
                "task_assignments": {str(t): best_assignment[0][t] for t in tasks}
            }
            with open(solution_path, 'w') as f:
                json.dump(solution, f)
            return
            
    except Exception as e:
        pass
    
    # Fallback: greedy heuristic
    # Topological sort, then assign tasks greedily
    in_degree = defaultdict(int)
    for t in tasks:
        in_degree[t] = 0
    for arc in precedence_arcs:
        in_degree[arc[1]] += 1
    
    queue = [t for t in tasks if in_degree[t] == 0]
    queue.sort(key=lambda t: -task_processing_times[t])
    topo_order = []
    while queue:
        queue.sort(key=lambda t: (earliest_station[t], -task_processing_times[t]))
        t = queue.pop(0)
        topo_order.append(t)
        for s in successors[t]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    assignment = {}
    station_load = defaultdict(int)
    
    for t in topo_order:
        es = earliest_station[t]
        ls = latest_station[t]
        # Find earliest feasible station respecting predecessors
        min_station = es
        for p in predecessors[t]:
            if p in assignment:
                min_station = max(min_station, assignment[p])
        
        assigned = False
        for s in range(min_station, ls + 1):
            if station_load[s] + task_processing_times[t] <= cycle_time:
                assignment[t] = s
                station_load[s] += task_processing_times[t]
                assigned = True
                break
        
        if not assigned:
            assignment[t] = ls
            station_load[ls] += task_processing_times[t]
    
    max_idle = 0
    for s in range(1, num_stations + 1):
        idle = cycle_time - station_load[s]
        max_idle = max(max_idle, idle)
    
    if logger:
        logger.log(max_idle)
    
    solution = {
        "objective_value": float(max_idle),
        "task_assignments": {str(t): assignment[t] for t in tasks}
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)