import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, logger):
    num_nodes = instance['num_nodes']
    num_commodities = instance['num_commodities']
    num_scenarios = instance['num_scenarios']
    num_edges = instance['num_edges']
    gamma = instance['gamma']
    c_0 = instance['c_0']
    edges = instance['edges']
    construction_costs = instance['construction_costs']
    flow_costs = instance['flow_costs']
    capacities = instance['capacities']
    demands = instance['demands']

    try:
        model = gp.Model("Logistics_Planner")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = time_limit

        # Variables
        z = model.addVars(num_edges, vtype=GRB.BINARY, name='z')
        x = model.addVars(num_commodities, num_scenarios, num_edges, vtype=GRB.CONTINUOUS, name='x')
        tf = model.addVars(num_scenarios, num_edges, vtype=GRB.CONTINUOUS, name='tf')

        # Objective Function
        construction_cost = gp.quicksum(construction_costs[e] * z[e] for e in range(num_edges))
        transportation_cost = (1 / num_scenarios) * gp.quicksum(flow_costs[e] * x[k, r, e] for k in range(num_commodities) for r in range(num_scenarios) for e in range(num_edges))
        congestion_penalty = (1 / num_scenarios) * gp.quicksum((1 / (2 * gamma)) * tf[r, e] * tf[r, e] for r in range(num_scenarios) for e in range(num_edges))
        model.setObjective(construction_cost + transportation_cost + congestion_penalty, GRB.MINIMIZE)

        # Constraints
        model.addConstr(gp.quicksum(z[e] for e in range(num_edges)) <= c_0, name='max_edges')
        for k in range(num_commodities):
            for r in range(num_scenarios):
                for n in range(num_nodes):
                    inflow = gp.quicksum(x[k, r, e] for e in range(num_edges) if edges[e][1] == n)
                    outflow = gp.quicksum(x[k, r, e] for e in range(num_edges) if edges[e][0] == n)
                    model.addConstr(inflow - outflow == demands[r][k][n], name=f'flow_balance_{k}_{r}_{n}')

        for r in range(num_scenarios):
            for e in range(num_edges):
                model.addConstr(gp.quicksum(x[k, r, e] for k in range(num_commodities)) == tf[r, e], name=f'total_flow_{r}_{e}')
                model.addConstr(tf[r, e] <= capacities[e] * z[e], name=f'capacity_{r}_{e}')

        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            objective_value = model.ObjVal
            solution_dict = {
                'objective_value': objective_value,
                'model_variables': {},
                'z': {str(e): z[e].X for e in range(num_edges)},
            }
            for k in range(num_commodities):
                for r in range(num_scenarios):
                    for e in range(num_edges):
                        var_name = f'x[{k},{r},{e}]'
                        solution_dict['model_variables'][var_name] = x[k, r, e].X
            for r in range(num_scenarios):
                for e in range(num_edges):
                    var_name = f'tf[{r},{e}]'
                    solution_dict['model_variables'][var_name] = tf[r, e].X

            if logger:
                logger.log_solution(objective_value, solution_dict)

            return solution_dict
        else:
            return None

    except gp.GurobiError as e:
        print(f'Gurobi Error: {e}')
        return None

def write_solution(solution_path, solution_dict):
    with open(solution_path, 'w') as file:
        json.dump(solution_dict, file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution_dict = solve_instance(instance, args.time_limit, logger)

    if solution_dict is not None:
        write_solution(args.solution_path, solution_dict)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()