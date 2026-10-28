import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_instance(instance, time_limit):
    try:
        model = gp.Model()
        model.setParam("Threads", 1)
        model.Params.TimeLimit = time_limit
        model.Params.LogToConsole = False
        
        N = instance['N']
        b = instance['b']
        classes = instance['classes']
        
        y = {}
        x = {}
        
        for class_data in classes:
            class_id = class_data['class_id']
            y[class_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{class_id}")
            x[class_id] = {}
            for item in class_data['items']:
                item_id = item['item_id']
                x[class_id][item_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{class_id}_{item_id}")
        
        objective = gp.quicksum(class_data['f_i'] * y[class_data['class_id']] for class_data in classes)
        for class_data in classes:
            class_id = class_data['class_id']
            for item in class_data['items']:
                item_id = item['item_id']
                objective += item['c_ij'] * x[class_id][item_id]
        model.setObjective(objective, GRB.MAXIMIZE)
        
        for class_data in classes:
            class_id = class_data['class_id']
            for item in class_data['items']:
                item_id = item['item_id']
                model.addConstr(x[class_id][item_id] <= y[class_id], name=f"link_{class_id}_{item_id}")
        
        lhs = gp.quicksum(class_data['d_i'] * y[class_data['class_id']] for class_data in classes)
        lhs += gp.quicksum(item['a_ij'] * x[class_data['class_id']][item['item_id']] 
                           for class_data in classes 
                           for item in class_data['items'])
        model.addConstr(lhs, GRB.LESS_EQUAL, b, name="capacity")
        
        model.optimize()
        
        solution = {'classes': []}
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if model.getObjective().getValue() is not None:
                objective_value = model.getObjective().getValue()
                for class_data in classes:
                    class_id = class_data['class_id']
                    class_solution = {'class_id': class_id, 'y_i': int(round(y[class_id].X)), 'items': []}
                    for item in class_data['items']:
                        item_id = item['item_id']
                        class_solution['items'].append({'item_id': item_id, 'x_ij': int(round(x[class_id][item_id].X))})
                    solution['classes'].append(class_solution)
                solution['objective_value'] = objective_value
            else:
                solution['objective_value'] = float('-inf')
        else:
            solution['objective_value'] = float('-inf')
        
        model.dispose()
        return solution
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return {'classes': [], 'objective_value': float('-inf')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    solution = solve_instance(instance, args.time_limit)
    
    if logger and 'objective_value' in solution:
        logger.log(solution['objective_value'])
    
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()