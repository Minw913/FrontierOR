import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger


def solve_instance(instance_data, time_limit, log_path):
    # Extract instance data
    n = instance_data["n"]  # number of customers
    T = instance_data["T"]  # number of time periods
    m = instance_data["m"]  # number of vehicles
    Q = instance_data["Q"]  # vehicle capacity
    C = instance_data["C"]  # production capacity per period
    f = instance_data["f"]  # fixed setup cost
    u = instance_data["u"]  # unit production cost
    epsilon = instance_data["epsilon"]  # uncertainty half-width fraction
    alpha_hat = instance_data["alpha_hat"]  # penalty cost multiplier
    coordinates = instance_data["coordinates"]
    transportation_costs = instance_data["transportation_costs"]
    h = instance_data["h"]  # holding costs
    L = instance_data["L"]  # storage capacities
    I0 = instance_data["I0"]  # initial inventories
    d_bar = instance_data["d_bar"]  # nominal demands
    sigma = instance_data["sigma"]  # penalty costs for unmet demand
    n_scenarios = instance_data["n_scenarios"]
    scenario_probabilities = instance_data["scenario_probabilities"]
    demand_scenarios = instance_data["demand_scenarios"]

    # Create model
    model = gp.Model("ProductionRouting")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Define decision variables
    # First stage variables (non-anticipative)
    y = {}  # production setup
    z = {}  # visits
    x = {}  # edge traversals

    for t in range(1, T + 1):
        y[t] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}")
        
        for k in range(1, m + 1):
            for i in range(n + 1):  # 0 is plant, 1..n are customers
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")

    for t in range(1, T + 1):
        for k in range(1, m + 1):
            for i in range(n + 1):
                for j in range(n + 1):
                    if i != j:
                        x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")

    # Second stage variables (adaptive to scenarios)
    p = {}  # production quantities
    I = {}  # inventory levels
    e = {}  # unmet demand
    q = {}  # delivery quantities

    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            p[t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=C, name=f"p_{t}_{omega}")

            for i in range(n + 1):
                I[i, t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=L[i], name=f"I_{i}_{t}_{omega}")

            for i in range(1, n + 1):
                e[i, t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"e_{i}_{t}_{omega}")

    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                for i in range(1, n + 1):
                    q[i, k, t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{i}_{k}_{t}_{omega}")

    # Constraints
    # Production setup constraint
    for t in range(1, T + 1):
        model.addConstr(p[t, 0] <= C * y[t])  # Use scenario 0 for setup constraints

    # Vehicle capacity constraints
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                model.addConstr(
                    gp.quicksum(q[i, k, t, omega] for i in range(1, n + 1)) <= Q * z[0, k, t],
                    name=f"vehicle_capacity_{k}_{t}_{omega}"
                )

    # Visit requirement constraints
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for i in range(1, n + 1):
                model.addConstr(
                    gp.quicksum(q[i, k, t, omega] for k in range(1, m + 1)) <= 
                    gp.quicksum(z[i, k, t] for k in range(1, m + 1)) * sum(demand_scenarios[omega][i-1][s] for s in range(t-1, T)),
                    name=f"visit_requirement_{i}_{t}_{omega}"
                )

    # Each customer visited by at most one vehicle per period
    for t in range(1, T + 1):
        for i in range(1, n + 1):
            model.addConstr(
                gp.quicksum(z[i, k, t] for k in range(1, m + 1)) <= 1,
                name=f"customer_visit_once_{i}_{t}"
            )

    # Flow conservation constraints for routing
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                # Flow out of plant equals flow into plant (for vehicle k)
                model.addConstr(
                    gp.quicksum(x[0, j, k, t] for j in range(1, n + 1)) == z[0, k, t],
                    name=f"flow_out_plant_{k}_{t}_{omega}"
                )
                
                model.addConstr(
                    gp.quicksum(x[i, 0, k, t] for i in range(1, n + 1)) == z[0, k, t],
                    name=f"flow_in_plant_{k}_{t}_{omega}"
                )
                
                # Flow conservation at customers
                for i in range(1, n + 1):
                    model.addConstr(
                        gp.quicksum(x[j, i, k, t] for j in range(n + 1) if j != i) == z[i, k, t],
                        name=f"flow_in_customer_{i}_{k}_{t}_{omega}"
                    )
                    
                    model.addConstr(
                        gp.quicksum(x[i, j, k, t] for j in range(n + 1) if j != i) == z[i, k, t],
                        name=f"flow_out_customer_{i}_{k}_{t}_{omega}"
                    )

    # Subtour elimination constraints (MTZ-style)
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                u_vars = {}
                for i in range(1, n + 1):
                    u_vars[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n, name=f"u_{i}_{k}_{t}_{omega}")
                
                for i in range(1, n + 1):
                    for j in range(1, n + 1):
                        if i != j:
                            model.addConstr(
                                u_vars[i] - u_vars[j] + n * x[i, j, k, t] <= n - 1,
                                name=f"subtour_elimination_{i}_{j}_{k}_{t}_{omega}"
                            )

    # Inventory balance at plant
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            lhs = I[0, t-1, omega] if t > 1 else I0[0]
            lhs += p[t, omega]
            
            rhs = gp.quicksum(gp.quicksum(q[i, k, t, omega] for k in range(1, m + 1)) for i in range(1, n + 1))
            rhs += I[0, t, omega]
            
            model.addConstr(lhs == rhs, name=f"inventory_balance_plant_{t}_{omega}")

    # Inventory balance at customers
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for i in range(1, n + 1):
                lhs = I[i, t-1, omega] if t > 1 else I0[i]
                lhs += gp.quicksum(q[i, k, t, omega] for k in range(1, m + 1))
                lhs += e[i, t, omega]
                
                rhs = demand_scenarios[omega][i-1][t-1]  # demand for customer i-1 in period t-1
                rhs += I[i, t, omega]
                
                model.addConstr(lhs == rhs, name=f"inventory_balance_customer_{i}_{t}_{omega}")

    # Storage capacity constraints
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for i in range(n + 1):
                # Ensure inventory doesn't exceed capacity
                model.addConstr(I[i, t, omega] <= L[i], name=f"storage_capacity_{i}_{t}_{omega}")

    # Delivery quantity bounds
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                for i in range(1, n + 1):
                    # Delivery cannot exceed remaining storage capacity
                    model.addConstr(
                        q[i, k, t, omega] <= (L[i] - I[i, t-1, omega]) * z[i, k, t] if t > 1 else (L[i] - I0[i]) * z[i, k, t],
                        name=f"delivery_capacity_{i}_{k}_{t}_{omega}"
                    )
                    
                    # Delivery cannot exceed cumulative future demand
                    future_demand = sum(demand_scenarios[omega][i-1][s] for s in range(t-1, T))
                    model.addConstr(
                        q[i, k, t, omega] <= future_demand * z[i, k, t],
                        name=f"delivery_future_demand_{i}_{k}_{t}_{omega}"
                    )

    # Objective function
    # Expected cost = setup costs + production costs + inventory costs + transportation costs + penalty costs
    obj_expr = gp.LinExpr()
    
    # Setup costs
    obj_expr.addTerms([f] * T, [y[t] for t in range(1, T + 1)])
    
    # Production costs
    for omega in range(n_scenarios):
        prob = scenario_probabilities[omega]
        obj_expr.addTerms(
            [prob * u] * T,
            [p[t, omega] for t in range(1, T + 1)]
        )
    
    # Transportation costs
    for omega in range(n_scenarios):
        prob = scenario_probabilities[omega]
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                for i in range(n + 1):
                    for j in range(n + 1):
                        if i != j:
                            obj_expr.addTerm(
                                prob * transportation_costs[i][j],
                                x[i, j, k, t]
                            )
    
    # Inventory holding costs
    for omega in range(n_scenarios):
        prob = scenario_probabilities[omega]
        for t in range(1, T + 1):
            for i in range(n + 1):
                obj_expr.addTerm(prob * h[i], I[i, t, omega])
    
    # Penalty costs for unmet demand
    for omega in range(n_scenarios):
        prob = scenario_probabilities[omega]
        for t in range(1, T + 1):
            for i in range(1, n + 1):
                obj_expr.addTerm(prob * sigma[i-1], e[i, t, omega])

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Set up callback to log solutions
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def solution_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Construct partial solution dict for logging
                sol_dict = {"objective_value": obj_val}
                logger.log_solution(obj_val, sol_dict)

    # Optimize
    model.optimize(solution_callback)

    # Extract solution
    solution = {
        "objective_value": model.objVal if model.solCount > 0 else float('inf'),
        "model_variables": {},
        "y": {},
        "z": {},
        "x": {},
        "p": {},
        "I": {},
        "e": {},
        "q": {}
    }

    # Populate first stage variables
    for t in range(1, T + 1):
        solution["y"][str(t)] = int(y[t].x) if y[t].x is not None else 0
        
        for k in range(1, m + 1):
            for i in range(n + 1):
                key = f"{i}_{k}_{t}"
                solution["z"][key] = int(z[i, k, t].x) if z[i, k, t].x is not None else 0

    for t in range(1, T + 1):
        for k in range(1, m + 1):
            for i in range(n + 1):
                for j in range(n + 1):
                    if i != j:
                        key = f"{i}_{j}_{k}_{t}"
                        solution["x"][key] = int(x[i, j, k, t].x) if x[i, j, k, t].x is not None else 0

    # Populate second stage variables
    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            solution["p"][f"{t}_{omega}"] = p[t, omega].x if p[t, omega].x is not None else 0.0
            
            for i in range(n + 1):
                key = f"{i}_{t}_{omega}"
                solution["I"][key] = I[i, t, omega].x if I[i, t, omega].x is not None else 0.0

            for i in range(1, n + 1):
                key = f"{i}_{t}_{omega}"
                solution["e"][key] = e[i, t, omega].x if e[i, t, omega].x is not None else 0.0

    for omega in range(n_scenarios):
        for t in range(1, T + 1):
            for k in range(1, m + 1):
                for i in range(1, n + 1):
                    key = f"{i}_{k}_{t}_{omega}"
                    solution["q"][key] = q[i, k, t, omega].x if q[i, k, t, omega].x is not None else 0.0

    # Add model variables in sparse format
    model_vars = {}
    for var in model.getVars():
        val = var.x
        if abs(val) > 1e-6:  # Only include non-zero variables
            model_vars[var.varName] = val
    solution["model_variables"] = model_vars

    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Production Routing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()