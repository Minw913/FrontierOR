import argparse
import json
from gurobipy import Model, GRB, quicksum
from collections import defaultdict
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path=None):
    num_periods = instance['problem_parameters']['num_periods']
    num_containers = instance['problem_parameters']['num_containers']
    num_trades = instance['problem_parameters']['num_trades']
    containers = instance['containers']
    trades = instance['trades']

    container_available = {(c['id'], t): 1 if c['availability_start'] <= t <= c['availability_end'] else 0 for c in containers for t in range(num_periods)}
    trade_active = {(tr['id'], t): 1 if tr['start_period'] <= t <= tr['end_period'] else 0 for tr in trades for t in range(num_periods)}

    model = Model('ContainerAssignment')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable console logging

    # Variables
    assign = model.addVars([(c['id'], tr['id'], t) for c in containers for tr in trades for t in range(num_periods)], vtype=GRB.BINARY, name='assign')
    reassign = model.addVars([(tr['id'], t) for tr in trades for t in range(1, num_periods)], vtype=GRB.INTEGER, name='reassign', lb=0)
    group = model.addVars([(c['id'], c2['id'], tr['id'], t) for c in containers for c2 in containers for tr in trades for t in range(num_periods)], vtype=GRB.BINARY, name='group')
    is_group_leader = model.addVars([(c['id'], tr['id'], t) for c in containers for tr in trades for t in range(num_periods)], vtype=GRB.BINARY, name='is_group_leader')

    # Objective: Minimize total re-assignments
    model.setObjective(quicksum(reassign[tr['id'], t] for tr in trades for t in range(1, num_periods)), GRB.MINIMIZE)

    # Constraints
    for tr in trades:
        for t in range(num_periods):
            if trade_active[(tr['id'], t)] == 1:
                # Demand constraint
                model.addConstr(quicksum(assign[c['id'], tr['id'], t] for c in containers) == tr['container_demand'])
                # Container availability constraint
                for c in containers:
                    model.addConstr(assign[c['id'], tr['id'], t] <= container_available[(c['id'], t)])
                # Container assignment exclusivity
                for c in containers:
                    model.addConstr(quicksum(assign[c['id'], tr2['id'], t] for tr2 in trades) <= 1)
                # Dispersion constraint
                model.addConstr(quicksum(is_group_leader[c['id'], tr['id'], t] for c in containers) <= tr['max_dispersion'])
                for c in containers:
                    for c2 in containers:
                        if c['id'] < c2['id']:  # Avoid symmetry
                            if c2['id'] in c['adjacent_containers']:
                                model.addConstr(group[c['id'], c2['id'], tr['id'], t] >= assign[c['id'], tr['id'], t] + assign[c2['id'], tr['id'], t] - 1)
                                model.addConstr(group[c2['id'], c['id'], tr['id'], t] == group[c['id'], c2['id'], tr['id'], t])
                    model.addConstr(is_group_leader[c['id'], tr['id'], t] + quicksum(group[c2['id'], c['id'], tr['id'], t] for c2 in containers if c2['id'] != c['id']) >= assign[c['id'], tr['id'], t])
            else:
                for c in containers:
                    model.addConstr(assign[c['id'], tr['id'], t] == 0)

    for tr in trades:
        for t in range(1, num_periods):
            if trade_active[(tr['id'], t)] == 1 and trade_active[(tr['id'], t-1)] == 1:
                model.addConstr(reassign[tr['id'], t] >= quicksum(assign[c['id'], tr['id'], t] for c in containers) - quicksum(assign[c['id'], tr['id'], t-1] for c in containers))
                model.addConstr(reassign[tr['id'], t] >= quicksum(assign[c['id'], tr['id'], t-1] for c in containers) - quicksum(assign[c['id'], tr['id'], t] for c in containers))

    # Solve and log
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(log_callback)

    # Extract solution
    assignments = defaultdict(dict)
    for tr in trades:
        for t in range(num_periods):
            if trade_active[(tr['id'], t)] == 1:
                assignments[tr['id']][t] = [c['id'] for c in containers if assign[c['id'], tr['id'], t].X > 0.5]

    objective_value = model.ObjVal if model.SolCount > 0 else float('inf')

    return objective_value, dict(assignments)

def write_solution(solution_path, objective_value, assignments):
    solution = {
        "objective_value": objective_value,
        "assignments": assignments
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, assignments = solve(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, assignments)

if __name__ == "__main__":
    main()