import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_sources = instance['num_sources']
    num_sinks = instance['num_sinks']
    supply = instance['supply']
    demand = instance['demand']
    fixed_cost = instance['fixed_cost']
    variable_cost = instance['variable_cost']
    capacity = instance['capacity']
    total_supply = instance['total_supply']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("FixedChargeTransportation")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.0)
    model.setParam('LogToConsole', 0)

    flow = {}
    use = {}
    for i in range(num_sources):
        for j in range(num_sinks):
            flow[i, j] = model.addVar(lb=0, ub=capacity[i][j], vtype=GRB.CONTINUOUS, name=f"flow_{i}_{j}")
            use[i, j] = model.addVar(vtype=GRB.BINARY, name=f"use_{i}_{j}")

    # Supply constraints
    for i in range(num_sources):
        model.addConstr(gp.quicksum(flow[i, j] for j in range(num_sinks)) == supply[i])

    # Demand constraints
    for j in range(num_sinks):
        model.addConstr(gp.quicksum(flow[i, j] for i in range(num_sources)) == demand[j])

    # Linking constraints
    M = total_supply
    for i in range(num_sources):
        for j in range(num_sinks):
            model.addConstr(flow[i, j] <= capacity[i][j] * use[i, j])

    # Objective
    obj = gp.quicksum(variable_cost[i][j] * flow[i, j] + fixed_cost[i][j] * use[i, j]
                      for i in range(num_sources) for j in range(num_sinks))
    model.setObjective(obj, GRB.MINIMIZE)

    # Callback to log improving solutions
    best_obj = float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = float(model.objVal)
        flow_sol = [[0.0 for _ in range(num_sinks)] for _ in range(num_sources)]
        arcs_used = [[0 for _ in range(num_sinks)] for _ in range(num_sources)]

        for i in range(num_sources):
            for j in range(num_sinks):
                flow_val = flow[i, j].X
                flow_sol[i][j] = float(flow_val)
                if flow_val > 1e-6:
                    arcs_used[i][j] = 1
                else:
                    arcs_used[i][j] = 0

        solution = {
            "objective_value": objective_value,
            "flow": flow_sol,
            "arcs_used": arcs_used
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found - write a dummy feasible solution (should not happen for valid instances)
        flow_sol = [[0.0 for _ in range(num_sinks)] for _ in range(num_sources)]
        arcs_used = [[0 for _ in range(num_sinks)] for _ in range(num_sources)]
        # Try to create a basic feasible solution by assigning greedily
        remaining_supply = supply[:]
        remaining_demand = demand[:]
        for i in range(num_sources):
            for j in range(num_sinks):
                if remaining_supply[i] > 0 and remaining_demand[j] > 0:
                    assign = min(remaining_supply[i], remaining_demand[j], capacity[i][j])
                    flow_sol[i][j] = float(assign)
                    arcs_used[i][j] = 1
                    remaining_supply[i] -= assign
                    remaining_demand[j] -= assign
        objective_value = 0.0
        for i in range(num_sources):
            for j in range(num_sinks):
                if arcs_used[i][j] == 1:
                    objective_value += fixed_cost[i][j] + variable_cost[i][j] * flow_sol[i][j]
        solution = {
            "objective_value": float(objective_value),
            "flow": flow_sol,
            "arcs_used": arcs_used
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()