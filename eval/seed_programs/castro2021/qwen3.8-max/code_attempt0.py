import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Solve convex transportation problem")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    supplies = instance['supplies']
    demands = instance['demands']
    arc_capacity = instance['arc_capacity']
    linear_costs = instance['linear_costs']
    cost_type = instance['cost_type']
    quadratic_costs = instance.get('quadratic_costs', None)

    # Build Gurobi model
    model = gp.Model("transportation")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, args.time_limit))
    model.setParam('OutputFlag', 0)

    # Create flow variables: x[i,j] = flow from supplier i to customer j
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(lb=0.0, ub=float(arc_capacity), 
                                    name=f"x_{i}_{j}", vtype=GRB.CONTINUOUS)

    model.update()

    # Build objective function
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(m):
            lc = linear_costs[i][j]
            obj += lc * x[i, j]
            if cost_type == 'quadratic' and quadratic_costs is not None:
                qc = quadratic_costs[i][j]
                obj += qc * x[i, j] * x[i, j]

    model.setObjective(obj, GRB.MINIMIZE)

    # Demand constraints: for each customer j, total inflow = demand_j
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == demands[j],
            name=f"demand_{j}"
        )

    # Supply constraints: for each supplier i, total outflow <= supply_i
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(m)) <= supplies[i],
            name=f"supply_{i}"
        )

    # Solve
    model.optimize()

    elapsed = time.time() - start_time

    # Extract solution
    solution = None
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0:
        flows = {}
        obj_val = 0.0
        for i in range(n):
            for j in range(m):
                flow_val = x[i, j].X
                flows[f"x_{i}_{j}"] = flow_val
                lc = linear_costs[i][j]
                obj_val += lc * flow_val
                if cost_type == 'quadratic' and quadratic_costs is not None:
                    qc = quadratic_costs[i][j]
                    obj_val += qc * flow_val * flow_val

        solution = {
            "objective_value": obj_val,
            "flows": flows
        }

        if logger:
            logger.log_solution(obj_val, solution)

    else:
        # Fallback: try to construct a feasible solution greedily
        # This handles edge cases where Gurobi might fail
        flows = {}
        remaining_demand = list(demands)
        remaining_supply = list(supplies)
        obj_val = 0.0

        for j in range(m):
            for i in range(n):
                if remaining_demand[j] <= 0:
                    break
                flow_val = min(remaining_demand[j], remaining_supply[i], arc_capacity)
                flow_val = max(0, flow_val)
                flows[f"x_{i}_{j}"] = flow_val
                remaining_demand[j] -= flow_val
                remaining_supply[i] -= flow_val
                lc = linear_costs[i][j]
                obj_val += lc * flow_val
                if cost_type == 'quadratic' and quadratic_costs is not None:
                    qc = quadratic_costs[i][j]
                    obj_val += qc * flow_val * flow_val

        # Fill in any zero flows not yet set
        for i in range(n):
            for j in range(m):
                if f"x_{i}_{j}" not in flows:
                    flows[f"x_{i}_{j}"] = 0.0

        solution = {
            "objective_value": obj_val,
            "flows": flows
        }

        if logger:
            logger.log_solution(obj_val, solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()