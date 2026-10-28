import json
import argparse
import sys
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobi not available", file=sys.stderr)
    sys.exit(1)

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Stochastic facility location optimization")
    parser.add_argument('--instance_path', required=True, help="Path to the instance JSON file")
    parser.add_argument('--solution_path', required=True, help="Path to write the solution JSON file")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', default=None, help="Path to the JSONL log file (optional)")
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    num_scenarios = instance['num_scenarios']
    max_total_demand = instance['max_total_demand']

    fixed_costs = instance['facilities']['fixed_costs']
    capacities = instance['facilities']['capacities']
    routing_costs = instance['routing_costs']
    probabilities = instance['scenarios']['probabilities']
    demands = instance['scenarios']['demands']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Precompute sorted facility order for each customer (by routing cost)
    sorted_order = []
    for j in range(num_customers):
        order = sorted(range(num_facilities), key=lambda i: routing_costs[i][j])
        sorted_order.append(order)

    # ------------------------------------------------------------------
    # Greedy fallback solution: open all facilities, assign demands greedily
    # ------------------------------------------------------------------
    greedy_obj = float('inf')
    greedy_y = [1] * num_facilities

    if sum(capacities) >= max_total_demand:
        total_fixed = sum(fixed_costs)
        # x_flows[s][i][j]
        x_flows = [[[0.0] * num_customers for _ in range(num_facilities)] for _ in range(num_scenarios)]
        for s in range(num_scenarios):
            remaining_cap = list(capacities)
            for j in range(num_customers):
                demand = demands[s][j]
                if demand == 0:
                    continue
                for i in sorted_order[j]:
                    if remaining_cap[i] <= 0:
                        continue
                    ship = min(remaining_cap[i], demand)
                    if ship > 0:
                        x_flows[s][i][j] = ship
                        remaining_cap[i] -= ship
                        demand -= ship
                        if demand == 0:
                            break
        routing_total = 0.0
        for s in range(num_scenarios):
            prob = probabilities[s]
            for i in range(num_facilities):
                for j in range(num_customers):
                    if x_flows[s][i][j] > 0:
                        routing_total += prob * routing_costs[i][j] * x_flows[s][i][j]
        greedy_obj = total_fixed + routing_total
        if logger:
            logger.log_solution(greedy_obj, {"objective_value": greedy_obj, "y": greedy_y})
    else:
        print("Instance may be infeasible (total capacity < max total demand)", file=sys.stderr)
        greedy_obj = 0.0
        greedy_y = [0] * num_facilities

    best_obj = greedy_obj
    best_y = greedy_y

    # ------------------------------------------------------------------
    # Gurobi MIP model
    # ------------------------------------------------------------------
    try:
        model = gp.Model("stochastic_facility")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = args.time_limit
        model.Params.OutputFlag = 0

        # Variables
        y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
        x = model.addVars(num_facilities, num_customers, num_scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        # Objective: fixed costs + expected routing costs
        fixed_expr = gp.quicksum(fixed_costs[i] * y[i] for i in range(num_facilities))
        routing_expr = gp.quicksum(
            probabilities[s] * routing_costs[i][j] * x[i, j, s]
            for i in range(num_facilities)
            for j in range(num_customers)
            for s in range(num_scenarios)
        )
        model.setObjective(fixed_expr + routing_expr, GRB.MINIMIZE)

        # Demand constraints: sum_i x_{i,j,s} >= d_{s,j}
        model.addConstrs(
            (gp.quicksum(x[i, j, s] for i in range(num_facilities)) >= demands[s][j]
             for s in range(num_scenarios) for j in range(num_customers)),
            name="demand"
        )

        # Capacity constraints: sum_j x_{i,j,s} <= cap_i * y_i
        model.addConstrs(
            (gp.quicksum(x[i, j, s] for j in range(num_customers)) <= capacities[i] * y[i]
             for s in range(num_scenarios) for i in range(num_facilities)),
            name="capacity"
        )

        # Aggregate capacity: sum_i cap_i * y_i >= max_total_demand
        model.addConstr(
            gp.quicksum(capacities[i] * y[i] for i in range(num_facilities)) >= max_total_demand,
            name="total_capacity"
        )

        # MIP start: open all facilities (Gurobi will complete the continuous part)
        for i in range(num_facilities):
            y[i].Start = 1

        # Callback to log incumbent solutions
        class IncumbentCallback:
            def __init__(self, logger, y_vars, num_facilities, initial_best):
                self.logger = logger
                self.y_vars = y_vars
                self.num_facilities = num_facilities
                self.best_obj = initial_best

            def __call__(self, model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < self.best_obj - 1e-9:
                        self.best_obj = obj
                        y_vals = [model.cbGetSolution(self.y_vars[i]) for i in range(self.num_facilities)]
                        solution_dict = {
                            "objective_value": obj,
                            "y": [int(round(v)) for v in y_vals]
                        }
                        if self.logger:
                            self.logger.log_solution(obj, solution_dict)

        callback = IncumbentCallback(logger, y, num_facilities, best_obj)
        model.optimize(callback)

        # Retrieve best solution from Gurobi if available
        if model.SolCount > 0:
            gurobi_obj = model.ObjVal
            if gurobi_obj < best_obj - 1e-9:
                best_obj = gurobi_obj
                best_y = [int(round(y[i].X)) for i in range(num_facilities)]

    except Exception as e:
        print(f"Gurobi optimization failed: {e}", file=sys.stderr)

    # Write final solution
    solution = {
        "objective_value": best_obj,
        "y": best_y
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()