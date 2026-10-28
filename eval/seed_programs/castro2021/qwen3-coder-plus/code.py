import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_transportation_problem(instance_data, time_limit, log_path):
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    n = instance_data["n"]  # number of suppliers
    m = instance_data["m"]  # number of customers
    supplies = instance_data["supplies"]  # supply capacity for each supplier
    demands = instance_data["demands"]  # demand for each customer
    linear_costs = instance_data["linear_costs"]
    quadratic_costs = instance_data.get("quadratic_costs")
    arc_capacities = [[instance_data["arc_capacity"]] * m for _ in range(n)]  # assuming same arc cap for all

    # Create model
    model = gp.Model("Transportation")
    model.setParam("Threads", 1)

    # Set Gurobi parameters
    model.setParam(GRB.Param.TimeLimit, time_limit)
    if logger:
        def log_solution(model, where):
            if where == GRB.Callback.MIP or where == GRB.Callback.BEST:
                obj_val = model.cbGet(GRB.Callback.MIP_OBJBST)
                if model.Status != GRB.INF_OR_UNBD:
                    logger.log(obj_val)
        
        model.optimize(log_solution)
    else:
        model.optimize()

    # Add variables: x[i][j] is the flow from supplier i to customer j
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(
                lb=0,
                ub=arc_capacities[i][j],
                name=f"x_{i}_{j}",
                vtype=GRB.CONTINUOUS
            )

    # Add constraints
    # 1. Supply constraints: total flow out of supplier i <= supply capacity of i
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(m)) <= supplies[i],
            name=f"supply_{i}"
        )
    
    # 2. Demand constraints: total flow into customer j == demand of j
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == demands[j],
            name=f"demand_{j}"
        )

    # Add objective
    if instance_data["cost_type"] in ['linear_integer', 'linear_fractional']:
        # Linear cost: sum of linear_costs[i][j] * x[i,j]
        obj = gp.quicksum(linear_costs[i][j] * x[i, j] for i in range(n) for j in range(m))
    elif instance_data["cost_type"] == 'quadratic':
        # Quadratic cost: sum of (linear_costs[i][j] * x[i,j] + quadratic_costs[i][j] * x[i,j]^2)
        obj = gp.quicksum(
            linear_costs[i][j] * x[i, j] + quadratic_costs[i][j] * (x[i, j]**2)
            for i in range(n) for j in range(m)
        )
    else:
        raise ValueError(f"Unknown cost type: {instance_data['cost_type']}")
    
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize
    if logger:
        def log_solution_callback(model, where):
            if where == GRB.Callback.MIP or where == GRB.Callback.BEST:
                try:
                    obj_val = model.cbGet(GRB.Callback.MIP_OBJBST)
                    if model.Status != GRB.INF_OR_UNBD:
                        logger.log(obj_val)
                except:
                    pass  # Ignore errors in callback
        
        model.optimize(log_solution_callback)
    else:
        model.optimize()
    
    # Prepare solution
    solution = {"objective_value": model.objVal, "flows": {}}
    
    for i in range(n):
        for j in range(m):
            flow_value = x[i, j].X
            solution["flows"][f"x_{i}_{j}"] = flow_value

    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Transportation Network Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions')

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_transportation_problem(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()