import json
import argparse
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_LOC(instance):
    n = instance['n']
    m = instance['m']
    k = instance['k']
    data_matrix = np.array(instance['data_matrix'])
    model = Model()
    model.setParam("Threads", 1)
    model.Params.OutputFlag = 0
    x = model.addVars(n, vtype=GRB.BINARY)
    y = model.addVars(m, vtype=GRB.CONTINUOUS)
    model.addConstrs((y[i] <= quicksum(data_matrix[i, j] * x[j] for j in range(n)) for i in range(m)), name='benefit_bound')
    model.addConstr(x.sum() <= k)
    model.setObjective(y.sum(), GRB.MAXIMIZE)
    return model, x

def solve_COV(instance):
    n = instance['n']
    m = instance['m']
    k = instance['k']
    data_matrix = np.array(instance['data_matrix'])
    node_weights = np.array(instance['node_weights'])
    model = Model()
    model.setParam("Threads", 1)
    model.Params.OutputFlag = 0
    x = model.addVars(n, vtype=GRB.BINARY)
    y = model.addVars(m, vtype=GRB.BINARY)
    model.addConstrs((y[i] <= quicksum(data_matrix[i, j] * x[j] for j in range(n)) for i in range(m)), name='coverage')
    model.addConstr(x.sum() <= k)
    model.setObjective(quicksum(node_weights[i] * y[i] for i in range(m)), GRB.MAXIMIZE)
    return model, x

def solve_INF(instance):
    n = instance['n']
    m = instance['m']
    k = instance['k']
    data_matrix = np.array(instance['data_matrix'])
    element_weights = np.array(instance['element_weights'])
    model = Model()
    model.setParam("Threads", 1)
    model.Params.OutputFlag = 0
    x = model.addVars(n, vtype=GRB.BINARY)
    y = model.addVars(m, vtype=GRB.CONTINUOUS)
    z = model.addVars(m, vtype=GRB.CONTINUOUS)
    model.addConstrs((z[i] == quicksum((1 - element_weights[j] * data_matrix[i, j]) * x[j] for j in range(n)) for i in range(m)), name='influence_product')
    model.addConstrs((y[i] == 1 - GRB.Exp(-z[i]) for i in range(m)), name='influence_prob')
    model.addConstr(x.sum() <= k)
    model.setObjective(y.sum(), GRB.MAXIMIZE)
    return model, x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    n = instance['n']
    k = instance['k']
    instance_type = instance['instance_type']
    
    if instance_type == 'LOC':
        model, x = solve_LOC(instance)
    elif instance_type == 'COV':
        model, x = solve_COV(instance)
    elif instance_type == 'INF':
        model, x = solve_INF(instance)
    else:
        raise ValueError('Invalid instance type')
    
    model.Params.TimeLimit = args.time_limit
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_elements = [i for i in range(n) if x[i].x > 0.5]
        objective_value = model.ObjVal
        
        if logger:
            logger.log(objective_value)
        
        solution = {
            'objective_value': objective_value,
            'selected_elements': selected_elements
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print('No solution found')

if __name__ == '__main__':
    main()