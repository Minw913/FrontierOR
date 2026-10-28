import json
import argparse
import time
import sys
from gurobipy import *

def main():
    parser = argparse.ArgumentParser(description="Capacitated Lot-Sizing Problem Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL log file")
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_items = data['num_items']
    num_periods = data['num_periods']
    demand = data['demand']
    setup_cost = data['setup_cost']
    holding_cost = data['holding_cost']
    variable_cost = data['variable_cost']
    setup_time = data['setup_time']
    variable_time = data['variable_time']
    capacity = data['capacity']
    initial_inventory_cost = data['initial_inventory_cost']

    # Precompute remaining demand from period t to end for each item
    remaining_demand = [[0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        remaining_demand[i][num_periods - 1] = demand[i][num_periods - 1]
        for t in range(num_periods - 2, -1, -1):
            remaining_demand[i][t] = remaining_demand[i][t + 1] + demand[i][t]

    # Create Gurobi model
    model = Model("CLSP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    # X[i,t]: production quantity of item i in period t
    X = {}
    for i in range(num_items):
        for t in range(num_periods):
            X[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{i}_{t}")

    # Y[i,t]: setup indicator for item i in period t
    Y = {}
    for i in range(num_items):
        for t in range(num_periods):
            Y[i, t] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"Y_{i}_{t}")

    # I[i,t]: inventory of item i at time t (t=0 is initial, t=1..T is end of period)
    I = {}
    for i in range(num_items):
        for t in range(num_periods + 1):
            I[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"I_{i}_{t}")

    model.update()

    # Objective function
    obj = LinExpr()
    for i in range(num_items):
        obj += initial_inventory_cost[i] * I[i, 0]
    for i in range(num_items):
        for t in range(num_periods):
            obj += setup_cost[i] * Y[i, t]
            obj += variable_cost[i] * X[i, t]
    for i in range(num_items):
        for t in range(1, num_periods + 1):
            obj += holding_cost[i] * I[i, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints

    # Inventory balance: I[i,t] + X[i,t] = demand[i][t] + I[i,t+1]
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(
                I[i, t] + X[i, t] == demand[i][t] + I[i, t + 1],
                name=f"balance_{i}_{t}"
            )

    # No ending inventory after last period
    for i in range(num_items):
        model.addConstr(I[i, num_periods] == 0, name=f"no_end_{i}")

    # Setup forcing constraint: X[i,t] <= M[i,t] * Y[i,t]
    for i in range(num_items):
        for t in range(num_periods):
            if variable_time[i] > 0:
                cap_bound = (capacity[t] - setup_time[i]) / variable_time[i]
            else:
                cap_bound = remaining_demand[i][t]
            M_it = min(max(cap_bound, 0), remaining_demand[i][t])
            if M_it > 0:
                model.addConstr(X[i, t] <= M_it * Y[i, t], name=f"setup_{i}_{t}")
            else:
                model.addConstr(X[i, t] <= 0, name=f"setup_{i}_{t}")
                model.addConstr(Y[i, t] == 0, name=f"nosetup_{i}_{t}")

    # Capacity constraint for each period
    for t in range(num_periods):
        lhs = LinExpr()
        for i in range(num_items):
            lhs += setup_time[i] * Y[i, t]
            lhs += variable_time[i] * X[i, t]
        model.addConstr(lhs <= capacity[t], name=f"cap_{t}")

    # Optional: if demand is zero from some point, force no production
    for i in range(num_items):
        for t in range(num_periods):
            if remaining_demand[i][t] == 0:
                model.addConstr(X[i, t] == 0, name=f"zero_prod_{i}_{t}")
                model.addConstr(Y[i, t] == 0, name=f"zero_setup_{i}_{t}")

    model.update()

    # Callback to log incumbent solutions
    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = extract_solution(model_cb)
                if sol is not None and logger:
                    logger.log_solution(obj_val, sol)
            except:
                pass

    def extract_solution(m):
        try:
            production = []
            for i in range(num_items):
                row = []
                for t in range(num_periods):
                    row.append(X[i, t].X if hasattr(X[i, t], 'X') else m.cbGetSolution(X[i, t]))
                production.append(row)

            setup = []
            for i in range(num_items):
                row = []
                for t in range(num_periods):
                    val = Y[i, t].X if hasattr(Y[i, t], 'X') else m.cbGetSolution(Y[i, t])
                    row.append(int(round(val)))
                setup.append(row)

            inventory = []
            for i in range(num_items):
                row = []
                for t in range(num_periods + 1):
                    row.append(I[i, t].X if hasattr(I[i, t], 'X') else m.cbGetSolution(I[i, t]))
                inventory.append(row)

            # Compute objective value
            total_cost = 0.0
            for i in range(num_items):
                total_cost += initial_inventory_cost[i] * inventory[i][0]
            for i in range(num_items):
                for t in range(num_periods):
                    total_cost += setup_cost[i] * setup[i][t]
                    total_cost += variable_cost[i] * production[i][t]
            for i in range(num_items):
                for t in range(1, num_periods + 1):
                    total_cost += holding_cost[i] * inventory[i][t]

            solution_dict = {
                "objective_value": total_cost,
                "production": production,
                "setup": setup,
                "inventory": inventory
            }
            return solution_dict
        except:
            return None

    # Optimize with callback
    model.optimize(callback)

    # Extract final solution
    solution_dict = None

    if model.SolCount > 0:
        try:
            production = []
            for i in range(num_items):
                row = []
                for t in range(num_periods):
                    row.append(X[i, t].X)
                production.append(row)

            setup = []
            for i in range(num_items):
                row = []
                for t in range(num_periods):
                    row.append(int(round(Y[i, t].X)))
                setup.append(row)

            inventory = []
            for i in range(num_items):
                row = []
                for t in range(num_periods + 1):
                    row.append(I[i, t].X)
                inventory.append(row)

            # Compute objective
            total_cost = 0.0
            for i in range(num_items):
                total_cost += initial_inventory_cost[i] * inventory[i][0]
            for i in range(num_items):
                for t in range(num_periods):
                    total_cost += setup_cost[i] * setup[i][t]
                    total_cost += variable_cost[i] * production[i][t]
            for i in range(num_items):
                for t in range(1, num_periods + 1):
                    total_cost += holding_cost[i] * inventory[i][t]

            solution_dict = {
                "objective_value": total_cost,
                "production": production,
                "setup": setup,
                "inventory": inventory
            }

            if logger:
                logger.log_solution(total_cost, solution_dict)

        except Exception as e:
            print(f"Error extracting solution: {e}", file=sys.stderr)
    else:
        # Try to construct a feasible solution heuristically
        solution_dict = construct_feasible_solution(data)
        if solution_dict and logger:
            logger.log_solution(solution_dict["objective_value"], solution_dict)

    # Write solution
    if solution_dict is None:
        # Fallback: zero solution (may not be feasible but prevents crash)
        solution_dict = {
            "objective_value": 0.0,
            "production": [[0.0] * num_periods for _ in range(num_items)],
            "setup": [[0] * num_periods for _ in range(num_items)],
            "inventory": [[0.0] * (num_periods + 1) for _ in range(num_items)]
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


def construct_feasible_solution(data):
    """Construct a simple feasible solution by producing exactly what's needed each period."""
    num_items = data['num_items']
    num_periods = data['num_periods']
    demand = data['demand']
    setup_cost = data['setup_cost']
    holding_cost = data['holding_cost']
    variable_cost = data['variable_cost']
    setup_time = data['setup_time']
    variable_time = data['variable_time']
    capacity = data['capacity']
    initial_inventory_cost = data['initial_inventory_cost']

    # Simple lot-for-lot: produce exactly demand each period
    production = [[0.0] * num_periods for _ in range(num_items)]
    setup = [[0] * num_periods for _ in range(num_items)]
    inventory = [[0.0] * (num_periods + 1) for _ in range(num_items)]

    # Check if lot-for-lot is feasible
    for t in range(num_periods):
        time_used = 0.0
        for i in range(num_items):
            if demand[i][t] > 0:
                time_used += setup_time[i] + variable_time[i] * demand[i][t]
                production[i][t] = float(demand[i][t])
                setup[i][t] = 1
        if time_used > capacity[t]:
            return None  # Not feasible with lot-for-lot

    # All inventory is zero (lot-for-lot with no initial inventory)
    # Compute objective
    total_cost = 0.0
    for i in range(num_items):
        for t in range(num_periods):
            total_cost += setup_cost[i] * setup[i][t]
            total_cost += variable_cost[i] * production[i][t]

    solution_dict = {
        "objective_value": total_cost,
        "production": production,
        "setup": setup,
        "inventory": inventory
    }
    return solution_dict


if __name__ == "__main__":
    main()