import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    cost_type = instance['cost_type']
    supplies = instance['supplies']
    demands = instance['demands']
    arc_capacity = instance['arc_capacity']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance.get('quadratic_costs', None)

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('MIPGap', 1e-4)
    env.setParam('NumericFocus', 0)
    env.setParam('Threads', 1)
    env.start()

    model = gp.Model(env=env)
    elapsed = time.time() - start_time
    remaining_time = max(1, args.time_limit - elapsed - 2)
    model.setParam('TimeLimit', remaining_time)
    model.setParam('OutputFlag', 0)

    # Create flow variables
    x = model.addVars(n, m, lb=0, ub=arc_capacity, name='x')

    # Supply capacity constraints: sum_j x_{i,j} <= supplies[i]
    for i in range(n):
        model.addConstr(x.sum(i, '*') <= supplies[i], name=f'supply_{i}')

    # Demand constraints: sum_i x_{i,j} == demands[j]
    for j in range(m):
        model.addConstr(x.sum('*', j) == demands[j], name=f'demand_{j}')

    # Build objective
    if cost_type == 'quadratic':
        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(m):
                lc = linear_costs[i][j] if linear_costs else 0
                qc = quadratic_costs[i][j] if quadratic_costs else 0
                obj += lc * x[i, j]
                if qc != 0:
                    obj += qc * x[i, j] * x[i, j]
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        obj = gp.LinExpr()
        for i in range(n):
            for j in range(m):
                lc = linear_costs[i][j] if linear_costs else 0
                obj += lc * x[i, j]
        model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    solution_dict = {}
    obj_value = 0.0

    if model.SolCount > 0:
        for i in range(n):
            for j in range(m):
                solution_dict[f'x_{i}_{j}'] = float(x[i, j].X)
        obj_value = float(model.ObjVal)
    else:
        # Fallback: greedy heuristic
        solution_dict, obj_value = greedy_solution(
            n, m, supplies, demands, arc_capacity,
            linear_costs, quadratic_costs, cost_type
        )

    if logger and solution_dict:
        logger.log_solution(obj_value, solution_dict)

    solution = {
        'objective_value': obj_value,
        'flows': solution_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def greedy_solution(n, m, supplies, demands, arc_capacity,
                    linear_costs, quadratic_costs, cost_type):
    """Simple greedy heuristic: sort arcs by linear cost, assign greedily."""
    arcs = []
    for i in range(n):
        for j in range(m):
            lc = linear_costs[i][j] if linear_costs else 0
            arcs.append((lc, i, j))
    arcs.sort()

    rem_supply = list(supplies) if supplies else [float('inf')] * n
    rem_demand = list(demands) if demands else [0] * m
    flows = {}

    for lc, i, j in arcs:
        alloc = min(rem_supply[i], rem_demand[j], arc_capacity)
        if alloc > 0:
            flows[f'x_{i}_{j}'] = float(alloc)
            rem_supply[i] -= alloc
            rem_demand[j] -= alloc
        else:
            flows[f'x_{i}_{j}'] = 0.0

    # Fill in zeros for any missing arcs
    for i in range(n):
        for j in range(m):
            key = f'x_{i}_{j}'
            if key not in flows:
                flows[key] = 0.0

    # Compute objective
    obj_value = 0.0
    for i in range(n):
        for j in range(m):
            val = flows[f'x_{i}_{j}']
            lc = linear_costs[i][j] if linear_costs else 0
            if cost_type == 'quadratic':
                qc = quadratic_costs[i][j] if quadratic_costs else 0
                obj_value += lc * val + qc * val * val
            else:
                obj_value += lc * val

    return flows, obj_value


if __name__ == '__main__':
    main()