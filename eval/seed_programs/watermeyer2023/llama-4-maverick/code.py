import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_nodes = instance['num_nodes']
    deadline = instance['deadline']
    processing_times = instance['processing_times']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  

    start_times = model.addVars(num_nodes, vtype=GRB.INTEGER, name='start_times')
    model.setObjective(start_times[num_nodes-1], GRB.MINIMIZE)

    model.addConstr(start_times[0] == 0, name='start_at_zero')

    for constraint in temporal_constraints:
        from_node = constraint['from']
        to_node = constraint['to']
        delta = constraint['delta']
        if constraint['type'] == 'min_lag':
            model.addConstr(start_times[to_node] >= start_times[from_node] + delta, name=f'temporal_{from_node}_{to_node}')
        elif constraint['type'] == 'max_lag':
            model.addConstr(start_times[from_node] >= start_times[to_node] + delta, name=f'temporal_max_{from_node}_{to_node}')

    model.addConstr(start_times[num_nodes-1] <= deadline - processing_times[num_nodes-1], name='deadline_constraint')

    for resource in resources:
        resource_id = resource['resource_id']
        capacity = resource['capacity']
        demands = resource['demands']
        time_periods = resource['time_periods']
        resource_consumption = gp.LinExpr()
        for activity in range(num_nodes):
            demand = demands[activity]
            if demand > 0:
                for t in time_periods:
                    if t > 0:  # Ensure t is within valid range
                        usage = model.addVar(vtype=GRB.BINARY, name=f'activity_{activity}_period_{t}')
                        model.addConstr((start_times[activity] + processing_times[activity] >= t) >= usage, name=f'usage_constr1_{activity}_{t}')
                        model.addConstr((start_times[activity] < t) >= usage, name=f'usage_constr2_{activity}_{t}')
                        model.addConstr((start_times[activity] + processing_times[activity] >= t) + (start_times[activity] < t) - 1 <= usage, name=f'usage_constr3_{activity}_{t}')
                        resource_consumption += demand * usage
        model.addConstr(resource_consumption <= capacity, name=f'resource_{resource_id}')

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.getAttr('SolCount') > 0:
            objective_value = model.getObjective().getValue()
            start_times_solution = [start_times[i].X for i in range(num_nodes)]
            if logger:
                logger.log(objective_value)
            return objective_value, start_times_solution
        else:
            return None, None
    else:
        return None, None

def write_solution(solution_path, objective_value, start_times):
    solution = {
        'objective_value': int(objective_value),
        'start_times': [int(start_time) for start_time in start_times]
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, type=str)
    parser.add_argument('--solution_path', required=True, type=str)
    parser.add_argument('--time_limit', required=True, type=int)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, start_times = solve(instance, args.time_limit, logger)
    if objective_value is not None and start_times is not None:
        write_solution(args.solution_path, objective_value, start_times)

if __name__ == "__main__":
    main()