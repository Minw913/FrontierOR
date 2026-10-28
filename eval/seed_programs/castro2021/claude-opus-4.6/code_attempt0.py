import argparse
import json
import time
import numpy as np

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    cost_type = data['cost_type']
    supplies = data['supplies']
    demands = data['demands']
    arc_capacity = data['arc_capacity']
    linear_costs = data['linear_costs']
    quadratic_costs = data['quadratic_costs']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Use Gurobi to solve
    import gurobipy as gp
    from gurobipy import GRB
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Threads', 1)
    env.setParam('TimeLimit', max(1, time_limit - int(time.time() - start_time) - 2))
    env.start()
    
    model = gp.Model(env=env)
    
    # Create flow variables
    # x[i][j] = flow from supplier i to customer j
    x = {}
    for i in range(n):
        for j in range(m):
            ub = min(supplies[i], demands[j], arc_capacity)
            x[i, j] = model.addVar(lb=0.0, ub=ub, name=f'x_{i}_{j}')
    
    model.update()
    
    # Demand constraints: sum over suppliers for each customer = demand[j]
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == demands[j],
            name=f'demand_{j}'
        )
    
    # Supply constraints: sum over customers for each supplier <= supply[i]
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(m)) <= supplies[i],
            name=f'supply_{i}'
        )
    
    # Objective
    if cost_type == 'quadratic':
        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(m):
                lc = linear_costs[i][j]
                qc = quadratic_costs[i][j]
                if qc != 0:
                    obj += lc * x[i, j] + qc * x[i, j] * x[i, j]
                else:
                    obj += lc * x[i, j]
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        # Linear case
        obj = gp.LinExpr()
        for i in range(n):
            for j in range(m):
                lc = linear_costs[i][j]
                if lc != 0:
                    obj += lc * x[i, j]
        model.setObjective(obj, GRB.MINIMIZE)
    
    # For linear integer costs, variables could potentially be integer,
    # but the transportation problem with integer supplies/demands and 
    # integer costs has integer optimal solutions with LP relaxation.
    # So we solve as continuous LP/QP.
    
    # Set method based on problem type
    if cost_type == 'quadratic':
        model.setParam('Method', 2)  # Barrier for QP
    else:
        # For linear, network simplex or primal simplex
        model.setParam('Method', -1)  # Auto
    
    # Add callback for logging
    def callback(model, where):
        if where == GRB.Callback.BARRIER:
            try:
                obj_val = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
                if logger:
                    logger.log(obj_val)
            except:
                pass
    
    if cost_type == 'quadratic':
        model.optimize(callback)
    else:
        model.optimize()
    
    # Extract solution
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        obj_val = model.ObjVal
        if logger:
            logger.log(obj_val)
        
        flows = {}
        for i in range(n):
            for j in range(m):
                flows[f'x_{i}_{j}'] = x[i, j].X
        
        solution = {
            'objective_value': obj_val,
            'flows': flows
        }
    else:
        # Try to return something
        solution = {
            'objective_value': float('inf'),
            'flows': {f'x_{i}_{j}': 0.0 for i in range(n) for j in range(m)}
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)
    
    env.dispose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)