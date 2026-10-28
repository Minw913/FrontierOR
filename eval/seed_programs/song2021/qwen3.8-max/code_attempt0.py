import json
import argparse
import math
import sys
from gurobipy import *
from solution_logger import SolutionLogger


def extract_solution_from_callback(m, order_ids, intervals, x, y, ot, H, order_map):
    orders_solution = []
    for o in order_ids:
        chosen = None
        best_val = -1
        for (s, f) in intervals[o]:
            val = m.cbGetSolution(x[o, s, f])
            if val > best_val:
                best_val = val
                chosen = (s, f)

        s, f = chosen
        intensities = {}
        for t in range(s, f + 1):
            val = m.cbGetSolution(y[o, t])
            intensities[str(t)] = max(0.0, min(1.0, val))

        orders_solution.append({
            "order_id": o,
            "chosen_interval": [s, f],
            "intensities": intensities
        })

    nonregular = {}
    for t in range(1, H + 1):
        val = m.cbGetSolution(ot[t])
        if val > 1e-6:
            nonregular[str(t)] = round(val, 6)

    obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)

    return {
        "objective_value": obj_val,
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonregular
    }


def extract_final_solution(model, order_ids, intervals, x, y, ot, H, order_map):
    orders_solution = []
    for o in order_ids:
        chosen = None
        best_val = -1
        for (s, f) in intervals[o]:
            val = x[o, s, f].X
            if val > best_val:
                best_val = val
                chosen = (s, f)

        s, f = chosen
        intensities = {}
        for t in range(s, f + 1):
            val = y[o, t].X
            intensities[str(t)] = max(0.0, min(1.0, val))

        orders_solution.append({
            "order_id": o,
            "chosen_interval": [s, f],
            "intensities": intensities
        })

    nonregular = {}
    for t in range(1, H + 1):
        val = ot[t].X
        if val > 1e-6:
            nonregular[str(t)] = round(val, 6)

    return {
        "objective_value": model.ObjVal,
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonregular
    }


def main():
    parser = argparse.ArgumentParser(description="Order Scheduling Optimizer")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path) as f:
        instance = json.load(f)

    H = instance['planning_horizon_H']
    n = instance['num_orders_n']
    sigma = instance['unit_cost_nonregular_capacity_sigma']
    capacities = instance['capacities']
    orders = instance['orders']

    # Build order lookup
    order_map = {o['order_id']: o for o in orders}
    order_ids = [o['order_id'] for o in orders]

    # Generate feasible intervals for each order
    intervals = {}
    for order in orders:
        o = order['order_id']
        r = order['release_date_r']
        UB = order['intensity_upper_bound_UB']
        LB = order['intensity_lower_bound_LB']
        min_len = math.ceil(1.0 / UB - 1e-9)
        max_len = int(math.floor(1.0 / LB + 1e-9))

        feasible = []
        for s in range(r, H + 1):
            for length in range(min_len, max_len + 1):
                f = s + length - 1
                if f <= H:
                    feasible.append((s, f))
        intervals[o] = feasible

    # Check feasibility
    for o in order_ids:
        if len(intervals[o]) == 0:
            # Infeasible - output empty solution
            solution = {"objective_value": float('inf'), "orders_solution": [], "nonregular_capacity_by_period": {}}
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
            return

    # Create Gurobi model
    model = Model("order_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Binary variables: x[o, s, f] = 1 if order o is scheduled in interval [s, f]
    x = {}
    for o in order_ids:
        for (s, f) in intervals[o]:
            x[o, s, f] = model.addVar(vtype=GRB.BINARY, name=f"x_{o}_{s}_{f}")

    # Continuous variables: y[o, t] = intensity of order o in period t
    y = {}
    for o in order_ids:
        for t in range(1, H + 1):
            y[o, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y_{o}_{t}")

    # Tardiness variables
    T = {}
    for o in order_ids:
        T[o] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_{o}")

    # Overtime variables
    ot = {}
    for t in range(1, H + 1):
        ot[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ot_{t}")

    model.update()

    # Constraint: exactly one interval per order
    for o in order_ids:
        model.addConstr(
            quicksum(x[o, s, f] for (s, f) in intervals[o]) == 1,
            name=f"one_interval_{o}"
        )

    # Constraint: intensity bounds linked to active periods
    for o in order_ids:
        LB = order_map[o]['intensity_lower_bound_LB']
        UB = order_map[o]['intensity_upper_bound_UB']
        for t in range(1, H + 1):
            covering = [(s, f) for (s, f) in intervals[o] if s <= t <= f]
            if covering:
                z = quicksum(x[o, s, f] for (s, f) in covering)
                model.addConstr(y[o, t] >= LB * z, name=f"lb_{o}_{t}")
                model.addConstr(y[o, t] <= UB * z, name=f"ub_{o}_{t}")
            else:
                model.addConstr(y[o, t] == 0, name=f"zero_{o}_{t}")

    # Constraint: intensities sum to 1 for each order
    for o in order_ids:
        model.addConstr(
            quicksum(y[o, t] for t in range(1, H + 1)) == 1,
            name=f"sum_y_{o}"
        )

    # Constraint: tardiness linking
    for o in order_ids:
        d = order_map[o]['due_date_d']
        for (s, f) in intervals[o]:
            tard = f - d
            if tard > 0:
                model.addConstr(T[o] >= tard * x[o, s, f], name=f"tard_{o}_{s}_{f}")

    # Constraint: capacity
    for t in range(1, H + 1):
        load = quicksum(y[o, t] * order_map[o]['work_content_p'] for o in order_ids)
        model.addConstr(load <= capacities[t - 1] + ot[t], name=f"cap_{t}")

    # Objective: minimize weighted tardiness + overtime cost
    obj = quicksum(order_map[o]['tardiness_cost_w'] * T[o] for o in order_ids) + \
          sigma * quicksum(ot[t] for t in range(1, H + 1))
    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbent solutions
    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = extract_solution_from_callback(m, order_ids, intervals, x, y, ot, H, order_map)
            if logger:
                logger.log_solution(obj_val, sol)

    # Solve
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        solution = extract_final_solution(model, order_ids, intervals, x, y, ot, H, order_map)
    else:
        # No feasible solution found within time limit
        # Try a greedy heuristic as fallback
        solution = greedy_fallback(order_ids, order_map, intervals, H, capacities, sigma)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def greedy_fallback(order_ids, order_map, intervals, H, capacities, sigma):
    """Simple greedy fallback if MIP doesn't find a solution."""
    orders_solution = []
    remaining_cap = list(capacities)

    # Sort orders by due date (earliest first)
    sorted_orders = sorted(order_ids, key=lambda o: order_map[o]['due_date_d'])

    for o in sorted_orders:
        d = order_map[o]['due_date_d']
        p = order_map[o]['work_content_p']
        LB = order_map[o]['intensity_lower_bound_LB']
        UB = order_map[o]['intensity_upper_bound_UB']

        # Pick the interval that minimizes tardiness, then resource usage
        best_interval = None
        best_score = float('inf')
        for (s, f) in intervals[o]:
            tard = max(0, f - d)
            score = tard * 1000 + f  # prefer early finish
            if score < best_score:
                best_score = score
                best_interval = (s, f)

        if best_interval is None:
            continue

        s, f = best_interval
        length = f - s + 1
        # Uniform intensity
        intensity = 1.0 / length

        intensities = {}
        for t in range(s, f + 1):
            intensities[str(t)] = intensity
            remaining_cap[t - 1] -= intensity * p

        orders_solution.append({
            "order_id": o,
            "chosen_interval": [s, f],
            "intensities": intensities
        })

    # Compute overtime
    nonregular = {}
    total_ot = 0
    for t in range(1, H + 1):
        if remaining_cap[t - 1] < 0:
            ot_val = -remaining_cap[t - 1]
            nonregular[str(t)] = round(ot_val, 6)
            total_ot += ot_val

    # Compute objective
    total_tard_cost = 0
    for sol in orders_solution:
        o = sol['order_id']
        f = sol['chosen_interval'][1]
        d = order_map[o]['due_date_d']
        tard = max(0, f - d)
        total_tard_cost += order_map[o]['tardiness_cost_w'] * tard

    obj_val = total_tard_cost + sigma * total_ot

    return {
        "objective_value": obj_val,
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonregular
    }


if __name__ == "__main__":
    main()