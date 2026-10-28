import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    num_sources = instance['num_sources']
    num_sinks = instance['num_sinks']
    supply = instance['supply']
    demand = instance['demand']
    fixed_cost = instance['fixed_cost']
    variable_cost = instance['variable_cost']
    capacity = instance['capacity']

    model = gp.Model()
    model.Params.LogToConsole = 0  # Silence Gurobi output

    # Decision variables
    flow = model.addVars(num_sources, num_sinks, vtype=GRB.CONTINUOUS, name='flow')
    arcs_used = model.addVars(num_sources, num_sinks, vtype=GRB.BINARY, name='arcs_used')

    # Objective function
    model.setObjective(
        gp.quicksum(variable_cost[i][j] * flow[i, j] for i in range(num_sources) for j in range(num_sinks)) +
        gp.quicksum(fixed_cost[i][j] * arcs_used[i, j] for i in range(num_sources) for j in range(num_sinks)),
        GRB.MINIMIZE
    )

    # Constraints
    for i in range(num_sources):
        model.addConstr(gp.quicksum(flow[i, j] for j in range(num_sinks)) == supply[i], name=f'supply_{i}')
    for j in range(num_sinks):
        model.addConstr(gp.quicksum(flow[i, j] for i in range(num_sources)) == demand[j], name=f'demand_{j}')
    for i in range(num_sources):
        for j in range(num_sinks):
            model.addConstr(flow[i, j] <= capacity[i][j] * arcs_used[i, j], name=f'capacity_{i}_{j}')

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        flow_sol = [[flow[i, j].X for j in range(num_sinks)] for i in range(num_sources)]
        arcs_used_sol = [[1 if flow[i, j].X > 1e-6 else 0 for j in range(num_sinks)] for i in range(num_sources)]
        return objective_value, flow_sol, arcs_used_sol
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

def write_solution(solution_path, objective_value, flow, arcs_used):
    solution = {
        'objective_value': objective_value,
        'flow': flow,
        'arcs_used': arcs_used
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)

    def log_callback(model, where):
        if where == GRB.Callback.MIP and logger:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            if model._last_log is None or objbst < model._last_log:
                logger.log(objbst)
                model._last_log = objbst

    model = gp.Model()
    model._last_log = float('inf')
    num_sources = instance['num_sources']
    num_sinks = instance['num_sinks']
    supply = instance['supply']
    demand = instance['demand']
    fixed_cost = instance['fixed_cost']
    variable_cost = instance['variable_cost']
    capacity = instance['capacity']

    flow = model.addVars(num_sources, num_sinks, vtype=GRB.CONTINUOUS, name='flow')
    arcs_used = model.addVars(num_sources, num_sinks, vtype=GRB.BINARY, name='arcs_used')

    model.setObjective(
        gp.quicksum(variable_cost[i][j] * flow[i, j] for i in range(num_sources) for j in range(num_sinks)) +
        gp.quicksum(fixed_cost[i][j] * arcs_used[i, j] for i in range(num_sources) for j in range(num_sinks)),
        GRB.MINIMIZE
    )

    for i in range(num_sources):
        model.addConstr(gp.quicksum(flow[i, j] for j in range(num_sinks)) == supply[i], name=f'supply_{i}')
    for j in range(num_sinks):
        model.addConstr(gp.quicksum(flow[i, j] for i in range(num_sources)) == demand[j], name=f'demand_{j}')
    for i in range(num_sources):
        for j in range(num_sinks):
            model.addConstr(flow[i, j] <= capacity[i][j] * arcs_used[i, j], name=f'capacity_{i}_{j}')

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0

    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        flow_sol = [[flow[i, j].X for j in range(num_sinks)] for i in range(num_sources)]
        arcs_used_sol = [[1 if flow[i, j].X > 1e-6 else 0 for j in range(num_sinks)] for i in range(num_sources)]
        write_solution(args.solution_path, objective_value, flow_sol, arcs_used_sol)
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

if __name__ == "__main__":
    main()