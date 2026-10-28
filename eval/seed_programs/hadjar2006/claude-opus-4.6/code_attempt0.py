import argparse
import json
import time
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_tasks = data['num_tasks']
    num_depots = data['num_depots']
    depot_penalty = data['depot_penalty']
    
    depots = {d['id']: d for d in data['depots']}
    tasks = {t['id']: t for t in data['tasks']}
    
    task_ids = sorted(tasks.keys())
    depot_ids = sorted(depots.keys())
    
    # Build compatible pairs lookup
    compatible_pairs = data['compatible_pairs']
    depot_to_task = data['depot_to_task_costs']
    task_to_depot = data['task_to_depot_costs']
    
    # Build cost dictionaries
    # task-to-task costs
    tt_cost = {}
    tt_travel = {}
    for cp in compatible_pairs:
        tt_cost[(cp['task_i'], cp['task_j'])] = cp['cost']
        tt_travel[(cp['task_i'], cp['task_j'])] = cp['travel_time']
    
    # depot-to-task costs
    dt_cost = {}
    dt_travel = {}
    for dt in depot_to_task:
        dt_cost[(dt['depot_id'], dt['task_id'])] = dt['cost']
        dt_travel[(dt['depot_id'], dt['task_id'])] = dt['travel_time']
    
    # task-to-depot costs
    td_cost = {}
    td_travel = {}
    for td in task_to_depot:
        td_cost[(td['task_id'], td['depot_id'])] = td['cost']
        td_travel[(td['task_id'], td['depot_id'])] = td['travel_time']
    
    # Use Gurobi to solve as a min-cost flow / set partitioning problem
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        remaining_time = 1
    
    model = gp.Model("MDVSP")
    model.Params.TimeLimit = remaining_time - 2  # Leave some time for solution extraction
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # Focus on finding feasible solutions
    
    # Decision variables
    # x[d, i, j] = 1 if a vehicle from depot d travels from task i to task j
    # y[d, i] = 1 if a vehicle from depot d starts with task i (depot->task)
    # z[d, i] = 1 if a vehicle from depot d ends with task i (task->depot)
    
    # For efficiency, we create variables indexed by arcs
    # Binary/integer variables
    
    # Variables for depot-to-task arcs
    y = {}
    for (d, t), cost in dt_cost.items():
        y[d, t] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"y_{d}_{t}")
    
    # Variables for task-to-depot arcs
    z = {}
    for (t, d), cost in td_cost.items():
        z[t, d] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"z_{t}_{d}")
    
    # Variables for task-to-task arcs
    x = {}
    for (i, j), cost in tt_cost.items():
        x[i, j] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"x_{i}_{j}")
    
    model.update()
    
    # Constraint 1: Each task must be covered exactly once
    # For each task t: sum over all incoming arcs = 1
    # Incoming arcs: depot-to-task y[d,t] for all d, or task-to-task x[i,t] for all i
    for t in task_ids:
        incoming = []
        for d in depot_ids:
            if (d, t) in y:
                incoming.append(y[d, t])
        for (i, j) in tt_cost:
            if j == t:
                incoming.append(x[i, j])
        model.addConstr(gp.quicksum(incoming) == 1, name=f"cover_in_{t}")
    
    # For each task t: sum over all outgoing arcs = 1
    for t in task_ids:
        outgoing = []
        for d in depot_ids:
            if (t, d) in z:
                outgoing.append(z[t, d])
        for (i, j) in tt_cost:
            if i == t:
                outgoing.append(x[i, j])
        model.addConstr(gp.quicksum(outgoing) == 1, name=f"cover_out_{t}")
    
    # Constraint 2: Flow conservation at each task for each depot
    # For each depot d and task t:
    # inflow from d = outflow to d
    # This is the key constraint that ensures vehicles return to their starting depot
    
    # Build index structures for efficiency
    # For each task, which depots can serve it
    task_depots_in = defaultdict(list)  # task -> list of depots with y[d,t]
    task_depots_out = defaultdict(list)  # task -> list of depots with z[t,d]
    
    for (d, t) in dt_cost:
        task_depots_in[t].append(d)
    for (t, d) in td_cost:
        task_depots_out[t].append(d)
    
    # For each task, which tasks can precede/follow it
    task_predecessors = defaultdict(list)  # task -> list of predecessor tasks
    task_successors = defaultdict(list)  # task -> list of successor tasks
    
    for (i, j) in tt_cost:
        task_successors[i].append(j)
        task_predecessors[j].append(i)
    
    # We need per-depot flow variables. The current formulation with binary x[i,j] 
    # doesn't track which depot a vehicle belongs to on task-to-task arcs.
    # We need to reformulate with depot-indexed task-to-task variables.
    
    # Remove old x variables and recreate with depot index
    model.remove(list(x.values()))
    model.update()
    
    # Recreate x with depot index: x[d, i, j]
    x = {}
    for d in depot_ids:
        for (i, j), cost in tt_cost.items():
            x[d, i, j] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"x_{d}_{i}_{j}")
    
    model.update()
    
    # Remove old coverage constraints
    model.remove(model.getConstrs())
    model.update()
    
    # Constraint 1: Each task covered exactly once (incoming)
    for t in task_ids:
        incoming = []
        for d in depot_ids:
            if (d, t) in y:
                incoming.append(y[d, t])
            for i in task_predecessors[t]:
                incoming.append(x[d, i, t])
        model.addConstr(gp.quicksum(incoming) == 1, name=f"cover_{t}")
    
    # Constraint 2: Flow conservation per depot per task
    for d in depot_ids:
        for t in task_ids:
            inflow = []
            outflow = []
            
            if (d, t) in y:
                inflow.append(y[d, t])
            for i in task_predecessors[t]:
                inflow.append(x[d, i, t])
            
            if (t, d) in z:
                outflow.append(z[t, d])
            for j in task_successors[t]:
                outflow.append(x[d, t, j])
            
            model.addConstr(gp.quicksum(inflow) == gp.quicksum(outflow), 
                          name=f"flow_{d}_{t}")
    
    # Constraint 3: Depot capacity
    for d in depot_ids:
        vehicles_out = []
        for t in task_ids:
            if (d, t) in y:
                vehicles_out.append(y[d, t])
        model.addConstr(gp.quicksum(vehicles_out) <= depots[d]['capacity'], 
                       name=f"cap_{d}")
    
    # Constraint 4: Depot balance (vehicles leaving = vehicles returning)
    for d in depot_ids:
        leaving = []
        returning = []
        for t in task_ids:
            if (d, t) in y:
                leaving.append(y[d, t])
            if (t, d) in z:
                returning.append(z[t, d])
        model.addConstr(gp.quicksum(leaving) == gp.quicksum(returning),
                       name=f"depot_bal_{d}")
    
    model.ModelSense = GRB.MINIMIZE
    
    # Check if problem is small enough for direct MIP, otherwise try LP relaxation first
    num_vars = model.NumVars
    
    # For large instances, relax integrality first
    if num_tasks > 200 and num_depots * len(tt_cost) > 100000:
        model.Params.MIPGap = 0.01
    
    # Callback for logging
    best_obj = [float('inf')]
    best_solution = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract solution
    routes = []
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        # Extract which depot each task's vehicle comes from
        task_depot_assignment = {}
        task_next = {}  # (depot, task) -> next task or None
        task_is_first = {}  # (depot, task) -> True if first task from depot
        
        for d in depot_ids:
            for t in task_ids:
                if (d, t) in y and y[d, t].X > 0.5:
                    task_is_first[(d, t)] = True
                
                for j in task_successors[t]:
                    if x[d, t, j].X > 0.5:
                        task_next[(d, t)] = j
        
        # Build routes
        for d in depot_ids:
            for t in task_ids:
                if (d, t) in task_is_first:
                    # Start a new route from depot d
                    route_tasks = [t]
                    current = t
                    route_cost = dt_cost[(d, t)]
                    
                    while (d, current) in task_next:
                        next_t = task_next[(d, current)]
                        route_cost += tt_cost[(current, next_t)]
                        route_tasks.append(next_t)
                        current = next_t
                    
                    # Add cost of returning to depot
                    route_cost += td_cost[(current, d)]
                    
                    routes.append({
                        "depot": d,
                        "tasks": route_tasks,
                        "cost": route_cost
                    })
        
        total_cost = sum(r['cost'] for r in routes)
        
        solution = {
            "objective_value": total_cost,
            "routes": routes
        }
    else:
        # No solution found - return empty
        solution = {
            "objective_value": 0,
            "routes": []
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)