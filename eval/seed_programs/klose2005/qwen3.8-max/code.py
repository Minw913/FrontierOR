import json
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Capacitated Facility Location Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    num_plants = instance['num_plants']
    demands = instance['customers']['demands']
    fixed_costs = instance['plants']['fixed_costs']
    capacities = instance['plants']['capacities']
    shipping_costs = instance['shipping_costs']

    from gurobipy import Model, GRB, quicksum

    # Build MILP model
    model = Model("CFLP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    y = model.addVars(num_plants, vtype=GRB.BINARY, name="open")
    x = model.addVars(num_customers, num_plants, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="assign")

    # Objective: minimize fixed costs + shipping costs
    obj = quicksum(fixed_costs[j] * y[j] for j in range(num_plants))
    obj += quicksum(shipping_costs[i][j] * x[i, j] for i in range(num_customers) for j in range(num_plants))
    model.setObjective(obj, GRB.MINIMIZE)

    # Demand satisfaction: each customer's fractions sum to 1
    for i in range(num_customers):
        model.addConstr(quicksum(x[i, j] for j in range(num_plants)) == 1.0, name=f"demand_{i}")

    # Capacity constraints
    for j in range(num_plants):
        model.addConstr(
            quicksum(demands[i] * x[i, j] for i in range(num_customers)) <= capacities[j] * y[j],
            name=f"cap_{j}"
        )

    # Linking constraints: x[i,j] <= y[j] (strengthens LP relaxation)
    for i in range(num_customers):
        for j in range(num_plants):
            model.addConstr(x[i, j] <= y[j], name=f"link_{i}_{j}")

    model.update()

    # Callback to log incumbent solutions
    best_obj = [float('inf')]

    def incumbent_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    # Extract solution from callback
                    y_vals = model.cbGetSolution(y)
                    x_vals = model.cbGetSolution(x)
                    open_plants = [j for j in range(num_plants) if y_vals[j] > 0.5]
                    assignments = {}
                    for i in range(num_customers):
                        assignments[str(i)] = {}
                        for j in range(num_plants):
                            val = x_vals[i, j]
                            if val > 1e-6:
                                assignments[str(i)][str(j)] = val
                    solution_dict = {
                        "objective_value": obj_val,
                        "open_plants": open_plants,
                        "assignments": assignments
                    }
                    logger.log_solution(obj_val, solution_dict)

    model.optimize(incumbent_callback)

    # Extract final solution
    if model.SolCount > 0:
        open_plants = [j for j in range(num_plants) if y[j].X > 0.5]
        assignments = {}
        for i in range(num_customers):
            assignments[str(i)] = {}
            for j in range(num_plants):
                val = x[i, j].X
                if val > 1e-6:
                    assignments[str(i)][str(j)] = val

        # Compute objective value precisely
        total_fixed = sum(fixed_costs[j] for j in open_plants)
        total_shipping = sum(
            shipping_costs[i][j] * assignments[str(i)].get(str(j), 0.0)
            for i in range(num_customers) for j in range(num_plants)
        )
        objective_value = total_fixed + total_shipping

        solution = {
            "objective_value": objective_value,
            "open_plants": open_plants,
            "assignments": assignments
        }

        if logger:
            logger.log_solution(objective_value, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Fallback heuristic: greedy assignment
        # Open all plants, assign each customer to cheapest plant respecting capacity
        remaining_cap = list(capacities)
        assignments = {}
        open_plants_set = set()

        # Sort customers by demand (largest first for better packing)
        cust_order = sorted(range(num_customers), key=lambda i: -demands[i])

        for i in cust_order:
            assignments[str(i)] = {}
            remaining_demand = demands[i]
            # Sort plants by shipping cost for this customer
            plant_order = sorted(range(num_plants), key=lambda j: shipping_costs[i][j])
            for j in plant_order:
                if remaining_demand <= 1e-9:
                    break
                if remaining_cap[j] > 1e-9:
                    frac = min(1.0, remaining_cap[j] / demands[i]) if demands[i] > 0 else 0.0
                    if frac > 1e-9:
                        assignments[str(i)][str(j)] = frac
                        remaining_cap[j] -= frac * demands[i]
                        remaining_demand -= frac * demands[i]
                        open_plants_set.add(j)

        open_plants = sorted(open_plants_set)
        total_fixed = sum(fixed_costs[j] for j in open_plants)
        total_shipping = sum(
            shipping_costs[i][j] * assignments[str(i)].get(str(j), 0.0)
            for i in range(num_customers) for j in range(num_plants)
        )
        objective_value = total_fixed + total_shipping

        solution = {
            "objective_value": objective_value,
            "open_plants": open_plants,
            "assignments": assignments
        }

        if logger:
            logger.log_solution(objective_value, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()