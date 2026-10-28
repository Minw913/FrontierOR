import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_instance(instance, time_limit):
    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False
    
    # Extract relevant data from instance
    N = instance['N']
    b = instance['b']
    classes = instance['classes']
    
    # Create variables
    y = {}  # Class activation variables
    x = {}  # Item selection variables
    
    for class_data in classes:
        class_id = class_data['class_id']
        y[class_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{class_id}")
        x[class_id] = {}
        for item in class_data['items']:
            item_id = item['item_id']
            x[class_id][item_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{class_id}_{item_id}")
    
    # Objective function
    objective = gp.quicksum(class_data['f_i'] * y[class_data['class_id']] for class_data in classes)
    for class_data in classes:
        class_id = class_data['class_id']
        for item in class_data['items']:
            item_id = item['item_id']
            objective += item['c_ij'] * x[class_id][item_id]
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Constraints
    for class_data in classes:
        class_id = class_data['class_id']
        for item in class_data['items']:
            item_id = item['item_id']
            model.addConstr(x[class_id][item_id] <= y[class_id], name=f"link_{class_id}_{item_id}")
    
    model.addConstr(gp.quicksum(class_data['d_i'] * y[class_data['class_id']] for class_data in classes) +
                    gp.quicksum(item['a_ij'] * x[class_data['class_id']][item['item_id']] 
                                for class_data in classes 
                                for item in class_data['items']), GRB.LESS_EQUAL, b, name="capacity")
    
    # Optimize
    model.optimize()
    
    # Extract solution
    solution = {'classes': []}
    objective_value = model.getObjective().getValue()
    for class_data in classes:
        class_id = class_data['class_id']
        class_solution = {'class_id': class_id, 'y_i': int(y[class_id].X), 'items': []}
        for item in class_data['items']:
            item_id = item['item_id']
            class_solution['items'].append({'item_id': item_id, 'x_ij': int(x[class_id][item_id].X)})
        solution['classes'].append(class_solution)
    solution['objective_value'] = objective_value
    
    return solution, objective_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    solution, objective_value = solve_instance(instance, args.time_limit)
    
    if logger:
        logger.log(objective_value)
    
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()