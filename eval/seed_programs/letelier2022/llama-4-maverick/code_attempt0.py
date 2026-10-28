import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, log_path):
    num_real_items = instance['num_real_items']
    num_total_items = instance['num_total_items']
    source_item_id = instance['source_item_id']
    sink_item_id = instance['sink_item_id']
    items = instance['items']
    arcs = instance['arcs']
    T = instance['parameters']['T']
    W = instance['parameters']['W']
    L = instance['parameters']['L']

    # Convert 'L' to integer if it's not 'infinity'
    if L != 'infinity':
        L = int(L)

    model = Model('BinPacking')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Variables
    x = {}
    for item in items:
        x[item['item_id']] = model.addVar(vtype=GRB.INTEGER, name=f"x_{item['item_id']}")

    y = {}
    for item in items:
        for t in range(1, T + 1):
            for b in range(1, T + 1):  # Assuming maximum number of bins is T
                y[(item['item_id'], t, b)] = model.addVar(vtype=GRB.BINARY, name=f"y_{item['item_id']}_{t}_{b}")

    z = {}
    for t in range(1, T + 1):
        for b in range(1, T + 1):
            z[(t, b)] = model.addVar(vtype=GRB.BINARY, name=f"z_{t}_{b}")

    # Objective
    model.setObjective(quicksum(z[(t, b)] for t in range(1, T + 1) for b in range(1, T + 1)), GRB.MINIMIZE)

    # Constraints
    for item in items:
        if item['type'] == 'treatment':
            model.addConstr(quicksum(y[(item['item_id'], t, b)] for t in range(1, T + 1) for b in range(1, T + 1)) == 1, f"Assign_{item['item_id']}")

    for t in range(1, T + 1):
        for b in range(1, T + 1):
            model.addConstr(quicksum(item['weight'] * y[(item['item_id'], t, b)] for item in items if item['type'] == 'treatment') <= W * z[(t, b)], f"Capacity_{t}_{b}")

    for arc in arcs:
        model.addConstr(x[arc['to']] - x[arc['from']] >= arc['lag'], f"Precedence_{arc['from']}_{arc['to']}")

    for item in items:
        model.addConstr(x[item['item_id']] == quicksum(t * y[(item['item_id'], t, b)] for t in range(1, T + 1) for b in range(1, T + 1)), f"TimeAssignment_{item['item_id']}")

    for item in items:
        for t in range(1, T + 1):
            for b in range(1, T + 1):
                model.addConstr(y[(item['item_id'], t, b)] * (t - x[item['item_id']]) == 0, f"TimeConsistency_{item['item_id']}_{t}_{b}")

    if L != 'infinity':
        for t in range(1, T + 1):
            model.addConstr(quicksum(z[(t, b)] for b in range(1, T + 1)) <= L, f"BinLimit_{t}")

    # Fix source and sink times
    model.addConstr(x[source_item_id] == 0, "SourceTime")
    model.addConstr(x[sink_item_id] == T + 1, "SinkTime")

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignments = []
        active_bin_periods = []
        num_active_bin_periods = 0

        for t in range(1, T + 1):
            for b in range(1, T + 1):
                if z[(t, b)].x > 0.5:
                    num_active_bin_periods += 1
                    bin_period_items = []
                    total_weight = 0
                    for item in items:
                        if item['type'] == 'treatment' and y[(item['item_id'], t, b)].x > 0.5:
                            assignments.append({
                                'item_id': item['item_id'],
                                'bin': b,
                                'period': t,
                                'weight': item['weight']
                            })
                            bin_period_items.append(item['item_id'])
                            total_weight += item['weight']
                    active_bin_periods.append({
                        'bin': b,
                        'period': t,
                        'items': bin_period_items,
                        'total_weight': total_weight
                    })

        solution = {
            'objective_value': model.objVal,
            'assignments': assignments,
            'active_bin_periods': active_bin_periods,
            'num_active_bin_periods': num_active_bin_periods
        }
        return solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)

    if solution:
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()