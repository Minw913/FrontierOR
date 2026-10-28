import argparse
import json
from gurobipy import Model, GRB, quicksum
import math
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, log_path=None):
    H = instance['planning_horizon_H']
    n = instance['num_orders_n']
    sigma = instance['unit_cost_nonregular_capacity_sigma']
    capacities = instance['capacities']
    orders = instance['orders']

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables
    x = {}  # Binary variable indicating if order i is executed in interval [s, f]
    for order in orders:
        order_id = order['order_id']
        release_date = order['release_date_r']
        due_date = order['due_date_d']
        lower_exec_interval_length = order['lower_exec_interval_length_l']
        upper_exec_interval_length = order['upper_exec_interval_length_l_bar']
        x[order_id] = {}
        for s in range(release_date, H + 1):
            for f in range(s, H + 1):
                if f - s + 1 >= lower_exec_interval_length and f - s + 1 <= upper_exec_interval_length:
                    x[order_id][(s, f)] = model.addVar(vtype=GRB.BINARY, name=f"x_{order_id}_{s}_{f}")

    intensity = {}  # Fraction of order i's work content performed in period t
    for order in orders:
        order_id = order['order_id']
        work_content = order['work_content_p']
        intensity_lower_bound = order['intensity_lower_bound_LB']
        intensity_upper_bound = order['intensity_upper_bound_UB']
        intensity[order_id] = {}
        for t in range(1, H + 1):
            intensity[order_id][t] = model.addVar(lb=0, ub=1, name=f"intensity_{order_id}_{t}")

    nonregular_capacity = {}  # Amount of nonregular capacity used in period t
    for t in range(1, H + 1):
        nonregular_capacity[t] = model.addVar(lb=0, name=f"nonregular_capacity_{t}")

    tardiness = {}  # Tardiness of order i
    for order in orders:
        order_id = order['order_id']
        tardiness[order_id] = model.addVar(lb=0, name=f"tardiness_{order_id}")

    # Constraints
    for order in orders:
        order_id = order['order_id']
        work_content = order['work_content_p']
        model.addConstr(quicksum(x[order_id][(s, f)] for s in x[order_id] for f in range(s, s + len(x[order_id]) // len(set(s for s, _ in x[order_id])))) == 1, name=f"one_interval_{order_id}")

        for s in x[order_id]:
            for f in range(s, s + 1):
                if (s, f) in x[order_id]:
                    for t in range(1, H + 1):
                        if t >= s and t <= f:
                            model.addConstr(intensity[order_id][t] >= x[order_id][(s, f)] * order['intensity_lower_bound_LB'], name=f"intensity_lower_bound_{order_id}_{t}")
                            model.addConstr(intensity[order_id][t] <= x[order_id][(s, f)] * order['intensity_upper_bound_UB'], name=f"intensity_upper_bound_{order_id}_{t}")
                        else:
                            model.addConstr(intensity[order_id][t] == 0, name=f"intensity_zero_{order_id}_{t}")
                    model.addConstr(quicksum(intensity[order_id][t] for t in range(s, f + 1)) == 1, name=f"intensity_sum_{order_id}_{s}_{f}")

        model.addConstr(tardiness[order_id] >= quicksum(f * x[order_id][(s, f)] for s in x[order_id] for f in range(s, s + len(x[order_id]) // len(set(s for s, _ in x[order_id]))) if (s, f) in x[order_id]) - order['due_date_d'], name=f"tardiness_def_{order_id}")

    for t in range(1, H + 1):
        model.addConstr(nonregular_capacity[t] >= quicksum(intensity[order_id][t] * order['work_content_p'] for order in orders for order_id in [order['order_id']]) - capacities[t - 1], name=f"nonregular_capacity_def_{t}")
        model.addConstr(nonregular_capacity[t] >= 0, name=f"nonregular_capacity_nonneg_{t}")

    # Objective
    model.setObjective(quicksum(order['tardiness_cost_w'] * tardiness[order['order_id']] for order in orders) + sigma * quicksum(nonregular_capacity[t] for t in range(1, H + 1)), GRB.MINIMIZE)

    # Solve
    model.optimize()
    incumbent_count = 0
    def log_incumbent(model, where):
        nonlocal incumbent_count
        if where == GRB.Callback.MIPSOL:
            incumbent_count += 1
            obj = model.getObjective().getValue()
            if logger:
                logger.log(obj)

    model.Params.LazyConstraints = 0
    model.optimize(log_incumbent)

    # Extract solution
    solution = {'orders_solution': []}
    for order in orders:
        order_id = order['order_id']
        chosen_interval = None
        for s in x[order_id]:
            for f in range(s, s + 1):
                if (s, f) in x[order_id] and x[order_id][(s, f)].x > 0.5:
                    chosen_interval = [s, f]
        if chosen_interval:
            intensities = {t: intensity[order_id][t].x for t in range(1, H + 1) if intensity[order_id][t].x > 0}
            solution['orders_solution'].append({
                'order_id': order_id,
                'chosen_interval': chosen_interval,
                'intensities': intensities
            })

    solution['nonregular_capacity_by_period'] = {t: nonregular_capacity[t].x for t in range(1, H + 1) if nonregular_capacity[t].x > 0}
    solution['objective_value'] = model.getObjective().getValue()

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, type=str)
    parser.add_argument('--solution_path', required=True, type=str)
    parser.add_argument('--time_limit', required=True, type=int)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()