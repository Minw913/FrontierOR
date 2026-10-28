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
    try:
        H = instance['planning_horizon_H']
        n = instance['num_orders_n']
        sigma = instance['unit_cost_nonregular_capacity_sigma']
        capacities = instance['capacities']
        orders = instance['orders']

        logger = SolutionLogger(log_path, sense="minimize") if log_path else None

        model = Model()
        model.setParam("Threads", 1)
        model.Params.TimeLimit = time_limit
        model.Params.LogToConsole = 0

        x = {}  
        for order in orders:
            order_id = order['order_id']
            release_date = order['release_date_r']
            lower_exec_interval_length = math.ceil(1 / order['intensity_upper_bound_UB'])
            upper_exec_interval_length = math.floor(1 / order['intensity_lower_bound_LB'])
            x[order_id] = {}
            for s in range(release_date, H + 1):
                for f in range(s, min(H + 1, s + upper_exec_interval_length)):
                    if f - s + 1 >= lower_exec_interval_length and f <= H:
                        x[order_id][(s, f)] = model.addVar(vtype=GRB.BINARY, name=f"x_{order_id}_{s}_{f}")

        intensity = {}  
        for order in orders:
            order_id = order['order_id']
            intensity[order_id] = {}
            for t in range(1, H + 1):
                intensity[order_id][t] = model.addVar(lb=0, ub=1, name=f"intensity_{order_id}_{t}")

        nonregular_capacity = {}  
        for t in range(1, H + 1):
            nonregular_capacity[t] = model.addVar(lb=0, name=f"nonregular_capacity_{t}")

        tardiness = {}  
        for order in orders:
            order_id = order['order_id']
            tardiness[order_id] = model.addVar(lb=0, name=f"tardiness_{order_id}")

        for order in orders:
            order_id = order['order_id']
            if x[order_id]:
                model.addConstr(quicksum(x[order_id][interval] for interval in x[order_id]) == 1, name=f"one_interval_{order_id}")
                
                for interval in x[order_id]:
                    s, f = interval
                    for t in range(1, H + 1):
                        if t >= s and t <= f:
                            model.addConstr(intensity[order_id][t] >= x[order_id][interval] * order['intensity_lower_bound_LB'], name=f"intensity_lower_bound_{order_id}_{t}_{s}_{f}")
                            model.addConstr(intensity[order_id][t] <= x[order_id][interval] * order['intensity_upper_bound_UB'], name=f"intensity_upper_bound_{order_id}_{t}_{s}_{f}")
                        else:
                            model.addConstr(intensity[order_id][t] == 0, name=f"intensity_zero_{order_id}_{t}_{s}_{f}")

                model.addConstr(quicksum(intensity[order_id][t] for t in range(1, H + 1)) == 1, name=f"intensity_sum_{order_id}")

                model.addConstr(tardiness[order_id] >= quicksum(f * x[order_id][(s, f)] for (s, f) in x[order_id]) - order['due_date_d'], name=f"tardiness_def_{order_id}")
            else:
                model.addConstr(tardiness[order_id] == 0, name=f"tardiness_def_{order_id}")
                for t in range(1, H + 1):
                    model.addConstr(intensity[order_id][t] == 0, name=f"intensity_zero_{order_id}_{t}")

        for t in range(1, H + 1):
            model.addConstr(nonregular_capacity[t] >= quicksum(intensity[order['order_id']][t] * order['work_content_p'] for order in orders) - capacities[t - 1], name=f"nonregular_capacity_def_{t}")

        model.setObjective(quicksum(order['tardiness_cost_w'] * tardiness[order['order_id']] for order in orders) + sigma * quicksum(nonregular_capacity[t] for t in range(1, H + 1)), GRB.MINIMIZE)

        incumbent_count = 0
        def log_incumbent(model, where):
            nonlocal incumbent_count
            if where == GRB.Callback.MIPSOL:
                incumbent_count += 1
                obj = model.getObjective().getValue()
                if logger:
                    logger.log(obj)

        model.optimize(log_incumbent)
        
        if model.status == GRB.INFEASIBLE:
            return {'objective_value': float('inf'), 'orders_solution': [], 'nonregular_capacity_by_period': {}}

        solution = {'orders_solution': []}
        for order in orders:
            order_id = order['order_id']
            chosen_interval = None
            max_value = -1
            if x[order_id]:
                for interval in x[order_id]:
                    if x[order_id][interval].x > max_value:
                        max_value = x[order_id][interval].x
                        chosen_interval = list(interval)
            if chosen_interval:
                s, f = chosen_interval
                intensities = {str(t): intensity[order_id][t].x for t in range(s, f + 1)}
                solution['orders_solution'].append({
                    'order_id': order_id,
                    'chosen_interval': chosen_interval,
                    'intensities': intensities
                })

        solution['nonregular_capacity_by_period'] = {str(t): nonregular_capacity[t].x for t in range(1, H + 1) if nonregular_capacity[t].x > 0.001}
        solution['objective_value'] = model.getObjective().getValue() if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT else float('inf')

        return solution
    except Exception as e:
        print(f"An error occurred: {e}")
        return {'objective_value': float('inf'), 'orders_solution': [], 'nonregular_capacity_by_period': {}}

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