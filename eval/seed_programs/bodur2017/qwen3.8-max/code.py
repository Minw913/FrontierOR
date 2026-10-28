import argparse
import json
import time
from gurobipy import *


def main():
    parser = argparse.ArgumentParser(description="Stochastic Facility Location Problem")
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

    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    num_scenarios = instance['num_scenarios']

    facilities = instance['facilities']
    customers = instance['customers']
    scenarios = instance['scenarios']
    trans_costs = instance['transportation_costs']

    # Precompute max total demand across scenarios
    max_total_demand = 0.0
    for s in range(num_scenarios):
        total_d = sum(scenarios[s]['demands'][j] for j in range(num_customers))
        if total_d > max_total_demand:
            max_total_demand = total_d

    # Build Gurobi model
    model = Model("stochastic_facility_location")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for i in range(num_facilities):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    y = {}
    for s in range(num_scenarios):
        for i in range(num_facilities):
            for j in range(num_customers):
                y[s, i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"y_{s}_{i}_{j}")

    model.update()

    # Objective: fixed costs + expected transportation costs
    obj = LinExpr()
    for i in range(num_facilities):
        obj += facilities[i]['opening_cost'] * x[i]

    for s in range(num_scenarios):
        prob_s = scenarios[s]['probability']
        for i in range(num_facilities):
            for j in range(num_customers):
                if trans_costs[i][j] > 0:
                    obj += prob_s * trans_costs[i][j] * y[s, i, j]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints: demand satisfaction
    for s in range(num_scenarios):
        demands_s = scenarios[s]['demands']
        for j in range(num_customers):
            if demands_s[j] > 0:
                model.addConstr(
                    quicksum(y[s, i, j] for i in range(num_facilities)) >= demands_s[j],
                    name=f"demand_{s}_{j}"
                )

    # Constraints: capacity linking
    for s in range(num_scenarios):
        for i in range(num_facilities):
            model.addConstr(
                quicksum(y[s, i, j] for j in range(num_customers)) <= facilities[i]['capacity'] * x[i],
                name=f"capacity_{s}_{i}"
            )

    # Constraint: total capacity >= max total demand
    model.addConstr(
        quicksum(facilities[i]['capacity'] * x[i] for i in range(num_facilities)) >= max_total_demand,
        name="total_capacity"
    )

    model.update()

    # Store best solution info for callback
    best_info = {'obj': None, 'x_vals': None, 'y_vals': None}

    def extract_solution(mdl):
        """Extract solution from model."""
        x_vals = {}
        for i in range(num_facilities):
            x_vals[i] = 1 if x[i].X > 0.5 else 0

        y_vals = {}
        for s in range(num_scenarios):
            y_vals[s] = {}
            for i in range(num_facilities):
                y_vals[s][i] = {}
                for j in range(num_customers):
                    y_vals[s][i][j] = y[s, i, j].X

        return x_vals, y_vals

    def build_solution_dict(obj_val, x_vals, y_vals):
        """Build the solution dictionary for output."""
        open_facs = [i for i in range(num_facilities) if x_vals[i] == 1]

        x_dict = {str(i): x_vals[i] for i in range(num_facilities)}

        y_dict = {}
        for s in range(num_scenarios):
            y_dict[str(s)] = {}
            for i in range(num_facilities):
                y_dict[str(s)][str(i)] = {}
                for j in range(num_customers):
                    y_dict[str(s)][str(i)][str(j)] = y_vals[s][i][j]

        return {
            "objective_value": obj_val,
            "open_facilities": open_facs,
            "x": x_dict,
            "y": y_dict
        }

    def my_callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best_info['obj'] is None or obj_val < best_info['obj']:
                best_info['obj'] = obj_val
                # Extract variable values from callback
                x_vals = {}
                for i in range(num_facilities):
                    x_vals[i] = 1 if mdl.cbGetSolution(x[i]) > 0.5 else 0

                y_vals = {}
                for s in range(num_scenarios):
                    y_vals[s] = {}
                    for i in range(num_facilities):
                        y_vals[s][i] = {}
                        for j in range(num_customers):
                            y_vals[s][i][j] = mdl.cbGetSolution(y[s, i, j])

                best_info['x_vals'] = x_vals
                best_info['y_vals'] = y_vals

                if logger:
                    sol_dict = build_solution_dict(obj_val, x_vals, y_vals)
                    logger.log_solution(obj_val, sol_dict)

    model.optimize(my_callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_vals, y_vals = extract_solution(model)
        solution = build_solution_dict(obj_val, x_vals, y_vals)

        # Log final solution if not already logged via callback
        if logger and (best_info['obj'] is None or abs(obj_val - best_info['obj']) > 1e-6):
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    elif best_info['obj'] is not None:
        # Use best solution found during search
        solution = build_solution_dict(best_info['obj'], best_info['x_vals'], best_info['y_vals'])
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Fallback: try a greedy heuristic
        # Open all facilities and assign greedily
        x_vals = {i: 1 for i in range(num_facilities)}
        y_vals = {}
        total_cost = sum(facilities[i]['opening_cost'] for i in range(num_facilities))

        for s in range(num_scenarios):
            y_vals[s] = {}
            remaining_capacity = {i: facilities[i]['capacity'] for i in range(num_facilities)}
            for j in range(num_customers):
                y_vals[s][i] = {} if i not in y_vals[s] else y_vals[s][i]

            # Initialize
            for i in range(num_facilities):
                y_vals[s][i] = {j: 0.0 for j in range(num_customers)}

            # Sort customers by demand (descending) for greedy assignment
            customer_order = sorted(range(num_customers),
                                    key=lambda j: scenarios[s]['demands'][j], reverse=True)

            for j in customer_order:
                demand_j = scenarios[s]['demands'][j]
                remaining_demand = demand_j
                # Sort facilities by transportation cost to this customer
                fac_order = sorted(range(num_facilities), key=lambda i: trans_costs[i][j])
                for i in fac_order:
                    if remaining_demand <= 0:
                        break
                    if remaining_capacity[i] > 0:
                        ship = min(remaining_demand, remaining_capacity[i])
                        y_vals[s][i][j] = ship
                        remaining_capacity[i] -= ship
                        remaining_demand -= ship
                        total_cost += scenarios[s]['probability'] * trans_costs[i][j] * ship

        solution = build_solution_dict(total_cost, x_vals, y_vals)
        if logger:
            logger.log_solution(total_cost, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == '__main__':
    main()