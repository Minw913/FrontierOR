import gurobipy as gp
import json
import argparse
import os
from collections import defaultdict

def main(instance_path, solution_path, time_limit):
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']  # number of customers
    T = instance['T']  # number of periods
    m = instance['m']  # number of vehicles
    Q = instance['Q']  # vehicle capacity
    C = instance['C']  # production capacity
    f_cost = instance['f']  # setup cost
    u_cost = instance['u']  # unit production cost
    transportation_costs = instance['transportation_costs']  # (n+1) x (n+1) matrix
    h = instance['h']  # holding costs [0..n]
    L = instance['L']  # storage capacities [0..n]
    I0 = instance['I0']  # initial inventory [0..n]
    sigma = instance['sigma']  # penalty costs for customers [0..n-1]? Note: per customer, length n
    n_scenarios = instance['n_scenarios']
    scenario_probabilities = instance['scenario_probabilities']
    demand_scenarios = instance['demand_scenarios']  # [scenario][customer][period]
    
    # Create model
    model = gp.Model('ProductionRouting')
    
    # Stage 1 variables
    y = {}  # y_t for t in 1..T
    for t in range(1, T+1):
        y[t] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{t}")
    
    z = {}  # z_{i,k,t} for i in [0..n], k in [1..m], t in [1..T]
    x = {}  # x_{i,j,k,t} for i,j in [0..n], i != j, k in [1..m], t in [1..T]
    for t in range(1, T+1):
        for k in range(1, m+1):
            for i in range(0, n+1):
                z[i, k, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}_{k}_{t}")
            for i in range(0, n+1):
                for j in range(0, n+1):
                    if i != j:
                        x[i, j, k, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")
    
    # Stage 2 variables
    p = {}  # p_{t,omega} for t in [1..T], omega in [0..n_scenarios-1]
    I = {}  # I_{i,t,omega} for i in [0..n], t in [1..T], omega in [0..n_scenarios-1]
    e = {}  # e_{i,t,omega} for i in [1..n], t in [1..T], omega in [0..n_scenarios-1]
    q = {}  # q_{i,k,t,omega} for i in [1..n], k in [1..m], t in [1..T], omega in [0..n_scenarios-1]
    for omega in range(n_scenarios):
        for t in range(1, T+1):
            p[t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"p_{t}_{omega}")
            for i in range(0, n+1):
                I[i, t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"I_{i}_{t}_{omega}")
            for i in range(1, n+1):
                e[i, t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"e_{i}_{t}_{omega}")
            for k in range(1, m+1):
                for i in range(1, n+1):
                    q[i, k, t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"q_{i}_{k}_{t}_{omega}")
    
    # Constraints
    
    # Stage 1: Each customer visited by at most one vehicle per period
    for i in range(1, n+1):
        for t in range(1, T+1):
            model.addConstr(gp.quicksum(z[i, k, t] for k in range(1, m+1)) <= 1, f"visit_once_{i}_{t}")
    
    # Stage 1: Routing constraints per vehicle and period
    for t in range(1, T+1):
        for k in range(1, m+1):
            # Plant flow conservation
            model.addConstr(gp.quicksum(x[0, j, k, t] for j in range(1, n+1)) == z[0, k, t], f"plant_out_{k}_{t}")
            model.addConstr(gp.quicksum(x[j, 0, k, t] for j in range(1, n+1)) == z[0, k, t], f"plant_in_{k}_{t}")
            
            # Customer flow conservation
            for i in range(1, n+1):
                model.addConstr(gp.quicksum(x[j, i, k, t] for j in range(0, n+1) if j != i) == z[i, k, t], f"in_flow_{i}_{k}_{t}")
                model.addConstr(gp.quicksum(x[i, j, k, t] for j in range(0, n+1) if j != i) == z[i, k, t], f"out_flow_{i}_{k}_{t}")
            
            # MTZ subtour elimination for customers
            u = {}
            for i in range(1, n+1):
                u[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=n, name=f"u_{i}_{k}_{t}")
            for i in range(1, n+1):
                for j in range(1, n+1):
                    if i != j:
                        model.addConstr(u[i] - u[j] + n * x[i, j, k, t] <= n-1, f"mtz_{i}_{j}_{k}_{t}")
    
    # Stage 2 constraints per scenario
    for omega in range(n_scenarios):
        # Initial inventory constraints
        for i in range(0, n+1):
            model.addConstr(I[i, 0, omega] == I0[i], f"init_inv_{i}_{omega}")
        
        for t in range(1, T+1):
            # Plant inventory balance
            total_shipment = gp.quicksum(q[i, k, t, omega] for i in range(1, n+1) for k in range(1, m+1))
            model.addConstr(I[0, t-1, omega] + p[t, omega] == I[0, t, omega] + total_shipment, f"plant_balance_{t}_{omega}")
            
            # Production constraints
            model.addConstr(p[t, omega] <= C * y[t], f"prod_setup_{t}_{omega}")
            model.addConstr(p[t, omega] <= C, f"prod_cap_{t}_{omega}")
            
            # Customer inventory balance and constraints
            for i in range(1, n+1):
                # Demand for customer i in period t, scenario omega
                d_it = demand_scenarios[omega][i-1][t-1]  # assuming period index 0-based in list
                total_delivery = gp.quicksum(q[i, k, t, omega] for k in range(1, m+1))
                model.addConstr(I[i, t-1, omega] + total_delivery + e[i, t, omega] == d_it + I[i, t, omega], f"cust_balance_{i}_{t}_{omega}")
                model.addConstr(I[i, t, omega] + d_it <= L[i], f"storage_{i}_{t}_{omega}")
                model.addConstr(I[i, t, omega] <= L[i], f"end_inv_{i}_{t}_{omega}")
            
            # Vehicle capacity constraints
            for k in range(1, m+1):
                model.addConstr(gp.quicksum(q[i, k, t, omega] for i in range(1, n+1)) <= Q, f"vehicle_cap_{k}_{t}_{omega}")
            
            # Delivery constraints: only if visited
            for k in range(1, m+1):
                for i in range(1, n+1):
                    model.addConstr(q[i, k, t, omega] <= L[i] * z[i, k, t], f"delivery_bound_{i}_{k}_{t}_{omega}")
    
    # Objective
    transportation_cost = gp.quicksum(
        transportation_costs[i][j] * x[i, j, k, t]
        for i in range(0, n+1) for j in range(0, n+1) if i != j
        for k in range(1, m+1) for t in range(1, T+1)
    )
    
    expected_production_cost = gp.quicksum(
        scenario_probabilities[omega] * u_cost * p[t, omega]
        for t in range(1, T+1) for omega in range(n_scenarios)
    )
    
    expected_holding_cost = gp.quicksum(
        scenario_probabilities[omega] * h[i] * I[i, t, omega]
        for i in range(0, n+1) for t in range(1, T+1) for omega in range(n_scenarios)
    )
    
    expected_penalty_cost = gp.quicksum(
        scenario_probabilities[omega] * sigma[i-1] * e[i, t, omega]
        for i in range(1, n+1) for t in range(1, T+1) for omega in range(n_scenarios)
    )
    
    setup_cost = gp.quicksum(f_cost * y[t] for t in range(1, T+1))
    
    total_cost = setup_cost + transportation_cost + expected_production_cost + expected_holding_cost + expected_penalty_cost
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    
    # Set parameters
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Optimize
    model.optimize()
    
    # Write solution
    solution = {
        "objective_value": model.ObjVal if model.status == gp.GRB.OPTIMAL or (model.status == gp.GRB.TIME_LIMIT and model.SolCount > 0) else None,
        "y": {},
        "z": {},
        "x": {},
        "model_variables": {}
    }
    
    if model.SolCount > 0:
        # Stage 1: y
        for t in range(1, T+1):
            solution["y"][str(t)] = int(round(y[t].X))
        
        # Stage 1: z (only nonzero)
        for t in range(1, T+1):
            for k in range(1, m+1):
                for i in range(0, n+1):
                    if abs(z[i, k, t].X) > 0.5:
                        key = f"{i}_{k}_{t}"
                        solution["z"][key] = 1
        
        # Stage 1: x (only nonzero)
        for t in range(1, T+1):
            for k in range(1, m+1):
                for i in range(0, n+1):
                    for j in range(0, n+1):
                        if i != j and abs(x[i, j, k, t].X) > 0.5:
                            key = f"{i}_{j}_{k}_{t}"
                            solution["x"][key] = 1
        
        # Stage 2: p, I, e, q (only nonzero)
        for omega in range(n_scenarios):
            for t in range(1, T+1):
                if abs(p[t, omega].X) > 1e-6:
                    key = f"p_{t}_{omega}"
                    solution["model_variables"][key] = p[t, omega].X
                for i in range(0, n+1):
                    if abs(I[i, t, omega].X) > 1e-6:
                        key = f"I_{i}_{t}_{omega}"
                        solution["model_variables"][key] = I[i, t, omega].X
                for i in range(1, n+1):
                    if abs(e[i, t, omega].X) > 1e-6:
                        key = f"e_{i}_{t}_{omega}"
                        solution["model_variables"][key] = e[i, t, omega].X
                    for k in range(1, m+1):
                        if abs(q[i, k, t, omega].X) > 1e-6:
                            key = f"q_{i}_{k}_{t}_{omega}"
                            solution["model_variables"][key] = q[i, k, t, omega].X
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_path', type=str)
    parser.add_argument('solution_path', type=str)
    parser.add_argument('time_limit', type=float)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit)