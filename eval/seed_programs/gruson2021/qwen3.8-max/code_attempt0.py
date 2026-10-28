import json
import argparse
import time
from gurobipy import *

def solve(instance_path, solution_path, time_limit, log_path=None):
    # Read instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    T = data['num_periods']
    W = data['num_warehouses']
    R = data['num_retailers']
    S = data['num_scenarios']
    init_inv = data['initial_inventory']
    init_setups = data['initial_setups_imposed']
    ret_to_wh = data['retailer_to_warehouse']  # 0-indexed or 1-indexed?
    scen_prob = data['scenario_probabilities']
    h_plant = data['holding_cost_plant']
    h_wh = data['holding_cost_warehouse']
    h_ret = data['holding_costs_retailers']
    sc_plant = data['setup_costs_plant']
    sc_wh = data['setup_costs_warehouses']
    sc_ret = data['setup_costs_retailers']
    demands = data['demands']  # [scenario][retailer][period]
    
    # Create model
    model = Model("stochastic_lotsizing")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # Setup variables (first stage)
    y_plant = [model.addVar(vtype=GRB.BINARY, name=f"y_plant_{t}") for t in range(T)]
    y_wh = [[model.addVar(vtype=GRB.BINARY, name=f"y_wh_{w}_{t}") for t in range(T)] for w in range(W)]
    y_ret = [[model.addVar(vtype=GRB.BINARY, name=f"y_ret_{r}_{t}") for t in range(T)] for r in range(R)]
    
    # Second-stage variables (per scenario)
    # Production at plant
    X_plant = [[model.addVar(lb=0, name=f"X_plant_{s}_{t}") for t in range(T)] for s in range(S)]
    # Delivery to warehouse
    X_wh = [[[model.addVar(lb=0, name=f"X_wh_{s}_{w}_{t}") for t in range(T)] for w in range(W)] for s in range(S)]
    # Delivery to retailer
    X_ret = [[[model.addVar(lb=0, name=f"X_ret_{s}_{r}_{t}") for t in range(T)] for r in range(R)] for s in range(S)]
    # Inventory at plant
    I_plant = [[model.addVar(lb=0, name=f"I_plant_{s}_{t}") for t in range(T)] for s in range(S)]
    # Inventory at warehouse
    I_wh = [[[model.addVar(lb=0, name=f"I_wh_{s}_{w}_{t}") for t in range(T)] for w in range(W)] for s in range(S)]
    # Inventory at retailer
    I_ret = [[[model.addVar(lb=0, name=f"I_ret_{s}_{r}_{t}") for t in range(T)] for r in range(R)] for s in range(S)]
    
    model.update()
    
    # Initial setup constraints
    if init_setups:
        model.addConstr(y_plant[0] == 1)
        for w in range(W):
            model.addConstr(y_wh[w][0] == 1)
        for r in range(R):
            model.addConstr(y_ret[r][0] == 1)
    
    # Compute big-M values
    # For plant: max total demand from period t onwards in scenario s
    M_plant = [[sum(demands[s][r][t2] for r in range(R) for t2 in range(t, T)) for t in range(T)] for s in range(S)]
    # For warehouse: max demand for retailers of warehouse w from period t onwards
    M_wh = [[[sum(demands[s][r][t2] for r in range(R) if ret_to_wh[r] == w for t2 in range(t, T)) for t in range(T)] for w in range(W)] for s in range(S)]
    # For retailer: max demand for retailer r from period t onwards
    M_ret = [[[sum(demands[s][r][t2] for t2 in range(t, T)) for t in range(T)] for r in range(R)] for s in range(S)]
    
    # Add constraints and objective
    obj = LinExpr()
    
    # Setup costs
    for t in range(T):
        obj += sc_plant[t] * y_plant[t]
    for w in range(W):
        for t in range(T):
            obj += sc_wh[w][t] * y_wh[w][t]
    for r in range(R):
        for t in range(T):
            obj += sc_ret[r][t] * y_ret[r][t]
    
    # Second-stage constraints and holding costs
    for s in range(S):
        for t in range(T):
            # Plant inventory balance
            if t == 0:
                model.addConstr(I_plant[s][t] == init_inv + X_plant[s][t] - quicksum(X_wh[s][w][t] for w in range(W)))
            else:
                model.addConstr(I_plant[s][t] == I_plant[s][t-1] + X_plant[s][t] - quicksum(X_wh[s][w][t] for w in range(W)))
            
            # Plant setup constraint
            if M_plant[s][t] > 0:
                model.addConstr(X_plant[s][t] <= M_plant[s][t] * y_plant[t])
            else:
                model.addConstr(X_plant[s][t] == 0)
            
            # Holding cost at plant
            obj += scen_prob[s] * h_plant * I_plant[s][t]
        
        for w in range(W):
            for t in range(T):
                # Warehouse inventory balance
                retailers_of_w = [r for r in range(R) if ret_to_wh[r] == w]
                if t == 0:
                    model.addConstr(I_wh[s][w][t] == init_inv + X_wh[s][w][t] - quicksum(X_ret[s][r][t] for r in retailers_of_w))
                else:
                    model.addConstr(I_wh[s][w][t] == I_wh[s][w][t-1] + X_wh[s][w][t] - quicksum(X_ret[s][r][t] for r in retailers_of_w))
                
                # Warehouse setup constraint
                if M_wh[s][w][t] > 0:
                    model.addConstr(X_wh[s][w][t] <= M_wh[s][w][t] * y_wh[w][t])
                else:
                    model.addConstr(X_wh[s][w][t] == 0)
                
                # Holding cost at warehouse
                obj += scen_prob[s] * h_wh * I_wh[s][w][t]
        
        for r in range(R):
            for t in range(T):
                # Retailer inventory balance
                if t == 0:
                    model.addConstr(I_ret[s][r][t] == init_inv + X_ret[s][r][t] - demands[s][r][t])
                else:
                    model.addConstr(I_ret[s][r][t] == I_ret[s][r][t-1] + X_ret[s][r][t] - demands[s][r][t])
                
                # Retailer setup constraint
                if M_ret[s][r][t] > 0:
                    model.addConstr(X_ret[s][r][t] <= M_ret[s][r][t] * y_ret[r][t])
                else:
                    model.addConstr(X_ret[s][r][t] == 0)
                
                # Holding cost at retailer
                obj += scen_prob[s] * h_ret[r] * I_ret[s][r][t]
    
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    # Extract solution
    ...