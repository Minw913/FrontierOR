import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance_data, time_limit, logger):
    panel_length = instance_data['panel']['length']
    panel_width = instance_data['panel']['width']
    items = instance_data['items']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  

    x = {}
    for item in items:
        x[item['id']] = model.addVar(vtype=GRB.INTEGER, name=f"x_{item['id']}", lb=0, ub=item['copies'])

    model.setObjective(gp.quicksum(item['profit'] * x[item['id']] for item in items), GRB.MAXIMIZE)

    plate_vars = generate_plate_vars(model, panel_length, panel_width, items, x)
    add_constraints(model, plate_vars, items, panel_length, panel_width, x)

    model.optimize()
    if logger and model.SolCount > 0:
        logger.log(model.ObjVal)

    return model, x

def generate_plate_vars(model, panel_length, panel_width, items, x):
    plates = {}
    def add_plate(length, width):
        if (length, width) not in plates:
            plates[(length, width)] = model.addVar(vtype=GRB.BINARY, name=f"plate_{length}_{width}")
        return plates[(length, width)]

    def generate_plates(length, width):
        if (length, width) in plates:
            return plates[(length, width)]
        plate = add_plate(length, width)
        for item in items:
            if item['length'] == length and item['width'] == width:
                model.addConstr(plate >= x[item['id']])
        for i in range(1, length):
            left_plate = generate_plates(i, width)
            right_plate = generate_plates(length - i, width)
            model.addConstr(plate <= left_plate + right_plate)
        for j in range(1, width):
            top_plate = generate_plates(length, j)
            bottom_plate = generate_plates(length, width - j)
            model.addConstr(plate <= top_plate + bottom_plate)
        return plate

    generate_plates(panel_length, panel_width)
    return plates

def add_constraints(model, plates, items, panel_length, panel_width, x):
    model.addConstr(plates[(panel_length, panel_width)] == 1)

def extract_solution(instance_data, model, x_vars):
    items = instance_data['items']
    solution = {'objective_value': model.ObjVal if model.SolCount > 0 else 0, 'items_selected': []}
    for item in items:
        if model.SolCount > 0:
            copies = int(x_vars[item['id']].X)
        else:
            copies = 0
        if copies > 0:
            solution['items_selected'].append({
                'item_id': item['id'],
                'plate_dims': [item['length'], item['width']],
                'copies': copies
            })
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    try:
        model, x_vars = solve(instance_data, args.time_limit, logger)
        solution = extract_solution(instance_data, model, x_vars)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()