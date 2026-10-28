import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

class CallbackState:
    def __init__(self, logger):
        self.logger = logger
        self.best_obj = float('inf')

def simplex_callback(model, where, state):
    if where == GRB.Callback.SIMPLEX:
        obj = model.cbGet(GRB.Callback.SPX_OBJVAL)
        if obj < state.best_obj:
            state.best_obj = obj
            if state.logger:
                state.logger.log(obj)

def main():
    parser = argparse.ArgumentParser(description='Transportation Network Optimization')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    arc_capacity = data['arc_capacity']
    supplies = data['supplies']
    demands = data['demands']
    linear_costs = data['linear_costs']
    quadratic_costs = data['quadratic_costs'] if 'quadratic_costs' in data else None
    cost_type = data['cost_type']
    
    model = gp.Model('transportation')
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    x = {}
    for i in range(n):
        for j in range(m):
            var_name = f'x_{i}_{j}'
            x[i, j] = model.addVar(lb=0.0, ub=arc_capacity, name=var_name, vtype=GRB.CONTINUOUS)
    
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) <= supplies[i], name=f'supply_{i}')
    
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == demands[j], name=f'demand_{j}')
    
    if cost_type in ['linear_integer', 'linear_fractional']:
        obj_expr = gp.quicksum(linear_costs[i][j] * x[i, j] for i in range(n) for j in range(m))
    elif cost_type == 'quadratic':
        obj_expr = gp.QuadExpr()
        for i in range(n):
            for j in range(m):
                qcoef = quadratic_costs[i][j]
                lcoef = linear_costs[i][j]
                obj_expr.add(lcoef * x[i, j] + qcoef * x[i, j] * x[i, j])
    else:
        raise ValueError(f"Unsupported cost type: {cost_type}")
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    if cost_type in ['linear_integer', 'linear_fractional'] and logger is not None:
        model.setParam('Method', 0)
        state = CallbackState(logger)
        model._state = state
        model.optimize(lambda model, where: simplex_callback(model, where, state))
    else:
        model.optimize()
    
    if model.SolCount > 0:
        objective_value = model.ObjVal
        flows_dict = {}
        for i in range(n):
            for j in range(m):
                flows_dict[f'x_{i}_{j}'] = x[i, j].X
        if cost_type == 'quadratic' and logger is not None:
            logger.log(objective_value)
    else:
        objective_value = 0.0
        flows_dict = {}
        for i in range(n):
            for j in range(m):
                flows_dict[f'x_{i}_{j}'] = 0.0
    
    solution = {
        'objective_value': objective_value,
        'flows': flows_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()