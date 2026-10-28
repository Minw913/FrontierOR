import json
import argparse
import time
from gurobipy import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_items = data['num_items']
    num_periods = data['num_periods']
    num_machines = data['num_machines']
    max_machines = data['max_machines_per_item_per_period']
    machine_capabilities = data['machine_capabilities']
    demand = data['demand']
    holding_cost = data['holding_cost']
    capacity = data['capacity']
    initial_inventory = data.get('initial_inventory', [0] * num_items)

    # Parse dictionaries
    production_time = {}
    setup_cost = {}
    setup_time = {}
    production_cost = {}

    for key, val in data['production_time'].items():
        parts = key.replace('item_', '').split('_machine_')
        i, m = int(parts[0]), int(parts[1])
        production_time[(i, m)] = val

    for key, val in data['setup_cost'].items():
        parts = key.replace('item_', '').split('_machine_')
        i, m = int(parts[0]), int(parts[1])
        setup_cost[(i, m)] = val

    for key, val in data['setup_time'].items():
        parts = key.replace('item_', '').split('_machine_')
        i, m = int(parts[0]), int(parts[1])
        setup_time[(i, m)] = val

    for key, val in data['production_cost'].items():
        parts = key.replace('item_', '').split('_machine_')
        i, m = int(parts[0]), int(parts[1])
        production_cost[(i, m)] = val

    items = list(range(1, num_items + 1))
    machines = list(range(1, num_machines + 1))
    periods = list(range(1, num_periods + 1))

    # Pre-compute reverse mapping: which items each machine can produce
    machine_items = {m: [] for m in machines}
    for i in items:
        for m in machine_capabilities[i-1]:
            machine_items[m].append(i)

    # Compute cumulative demand from period t onwards for each item
    cum_demand = {}
    for i in items:
        cum = 0
        for t in range(num_periods, 0, -1):
            cum += demand[i-1][t-1]
            cum_demand[(i, t)] = cum

    # Build model
    model = Model("CLSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Time management: leave buffer for solution extraction
    gurobi_time = max(args.time_limit - 5, 1)
    model.Params.TimeLimit = gurobi_time

    # Decision variables
    X = {}  # production quantity
    I = {}  # inventory
    Y = {}  # setup binary

    for i in items:
        for t in periods:
            I[(i, t)] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"I_{i}_{t}")

    for i in items:
        capable_machines = machine_capabilities[i-1]
        for m in capable_machines:
            for t in periods:
                X[(i, m, t)] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"X_{i}_{m}_{t}")
                Y[(i, m, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{m}_{t}")

    model.update()

    # Objective function
    obj = LinExpr()
    for i in items:
        capable_machines = machine_capabilities[i-1]
        for m in capable_machines:
            for t in periods:
                obj += setup_cost[(i, m)] * Y[(i, m, t)]
                obj += production_cost[(i, m)] * X[(i, m, t)]
        for t in periods:
            obj += holding_cost[i-1] * I[(i, t)]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Flow balance
    for i in items:
        for t in periods:
            lhs = LinExpr()
            if t == 1:
                lhs += initial_inventory[i-1]
            else:
                lhs += I[(i, t-1)]

            capable_machines = machine_capabilities[i-1]
            for m in capable_machines:
                lhs += X[(i, m, t)]

            lhs -= I[(i, t)]
            model.addConstr(lhs == demand[i-1][t-1], name=f"flow_{i}_{t}")

    # Constraint 2: Machine capacity
    for m in machines:
        for t in periods:
            lhs = LinExpr()
            for i in machine_items[m]:
                lhs += setup_time[(i, m)] * Y[(i, m, t)]
                lhs += production_time[(i, m)] * X[(i, m, t)]
            model.addConstr(lhs <= capacity[m-1][t-1], name=f"cap_{m}_{t}")

    # Constraint 3: Linking (big-M) with tight bounds
    for i in items:
        capable_machines = machine_capabilities[i-1]
        for m in capable_machines:
            for t in periods:
                M1 = cum_demand[(i, t)]
                remaining_cap = capacity[m-1][t-1] - setup_time[(i, m)]
                if remaining_cap > 0 and production_time[(i, m)] > 0:
                    M2 = remaining_cap / production_time[(i, m)]
                else:
                    M2 = 0
                M_val = min(M1, M2)
                if M_val > 0:
                    model.addConstr(X[(i, m, t)] <= M_val * Y[(i, m, t)], name=f"link_{i}_{m}_{t}")
                else:
                    model.addConstr(X[(i, m, t)] == 0, name=f"link_{i}_{m}_{t}")

    # Constraint 4: Max machines per item per period
    for i in items:
        capable_machines = machine_capabilities[i-1]
        if len(capable_machines) > max_machines[i-1]:
            for t in periods:
                lhs = LinExpr()
                for m in capable_machines:
                    lhs += Y[(i, m, t)]
                model.addConstr(lhs <= max_machines[i-1], name=f"maxmach_{i}_{t}")

    # Valid inequality: terminal inventory should be 0
    for i in items:
        model.addConstr(I[(i, num_periods)] == 0, name=f"terminal_{i}")

    model.update()

    # Optimize
    model.optimize()

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal

        production = {}
        inventory = {}
        setups = {}

        for i in items:
            capable_machines = machine_capabilities[i-1]
            for m in capable_machines:
                for t in periods:
                    key = f"X_item{i}_machine{m}_period{t}"
                    production[key] = X[(i, m, t)].X
                    key_y = f"Y_item{i}_machine{m}_period{t}"
                    setups[key_y] = int(round(Y[(i, m, t)].X))

            for t in periods:
                key = f"I_item{i}_period{t}"
                inventory[key] = I[(i, t)].X

        solution = {
            "objective_value": obj_val,
            "production": production,
            "inventory": inventory,
            "setups": setups
        }

        if logger:
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Fallback: try a simple greedy heuristic
        solution = greedy_heuristic(
            num_items, num_periods, num_machines, items, machines, periods,
            machine_capabilities, machine_items, demand, initial_inventory,
            production_time, setup_cost, setup_time, production_cost,
            holding_cost, capacity, max_machines, cum_demand
        )

        if solution is not None and logger:
            logger.log_solution(solution["objective_value"], solution)

        if solution is None:
            solution = {
                "objective_value": 1e20,
                "production": {},
                "inventory": {},
                "setups": {}
            }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


def greedy_heuristic(num_items, num_periods, num_machines, items, machines, periods,
                     machine_capabilities, machine_items, demand, initial_inventory,
                     production_time, setup_cost, setup_time, production_cost,
                     holding_cost, capacity, max_machines, cum_demand):
    """Simple greedy heuristic to construct a feasible solution."""
    # Track remaining capacity
    rem_cap = [[capacity[m-1][t-1] for t in periods] for m in machines]

    # Track inventory
    inv = {i: initial_inventory[i-1] for i in items}

    production = {}
    inventory = {}
    setups = {}
    total_cost = 0.0

    # Initialize all to zero
    for i in items:
        capable_machines = machine_capabilities[i-1]
        for m in capable_machines:
            for t in periods:
                key = f"X_item{i}_machine{m}_period{t}"
                production[key] = 0.0
                key_y = f"Y_item{i}_machine{m}_period{t}"
                setups[key_y] = 0
        for t in periods:
            key = f"I_item{i}_period{t}"
            inventory[key] = 0.0

    for t in periods:
        for i in items:
            # How much do we need to produce?
            needed = demand[i-1][t-1] - inv[i]
            if needed <= 0:
                inv[i] -= demand[i-1][t-1]
                key = f"I_item{i}_period{t}"
                inventory[key] = inv[i]
                total_cost += holding_cost[i-1] * inv[i]
                continue

            # Need to produce 'needed' units
            # Sort capable machines by production cost
            capable_machines = machine_capabilities[i-1]
            sorted_machines = sorted(capable_machines,
                                     key=lambda m: production_cost[(i, m)])

            machines_used = 0
            for m in sorted_machines:
                if needed <= 0:
                    break
                if machines_used >= max_machines[i-1]:
                    break

                # Check if setup fits
                st = setup_time[(i, m)]
                if rem_cap[m-1][t-1] < st:
                    continue

                # How much can we produce?
                available_time = rem_cap[m-1][t-1] - st
                if production_time[(i, m)] > 0:
                    max_prod = available_time / production_time[(i, m)]
                else:
                    max_prod = float('inf')

                # Also limited by cumulative remaining demand
                max_prod = min(max_prod, cum_demand[(i, t)])

                qty = min(needed, max_prod)
                if qty <= 0:
                    continue

                # Produce
                key = f"X_item{i}_machine{m}_period{t}"
                production[key] = qty
                key_y = f"Y_item{i}_machine{m}_period{t}"
                setups[key_y] = 1

                rem_cap[m-1][t-1] -= st + production_time[(i, m)] * qty
                total_cost += setup_cost[(i, m)] + production_cost[(i, m)] * qty

                needed -= qty
                machines_used += 1

            if needed > 0:
                # Infeasible - couldn't meet demand
                return None

            inv[i] = 0  # All produced to meet demand
            key = f"I_item{i}_period{t}"
            inventory[key] = inv[i]

    solution = {
        "objective_value": total_cost,
        "production": production,
        "inventory": inventory,
        "setups": setups
    }
    return solution


if __name__ == '__main__':
    main()