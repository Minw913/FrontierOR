import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    T = data['num_periods']
    W = data['num_warehouses']
    R = data['num_retailers']
    S = data['num_scenarios']
    r2w = data['retailer_to_warehouse']
    probs = data['scenario_probabilities']
    hc_plant = data['holding_cost_plant']
    hc_wh = data['holding_cost_warehouse']
    hc_ret = data['holding_costs_retailers']
    sc_plant = data['setup_costs_plant']
    sc_wh = data['setup_costs_warehouses']
    sc_ret = data['setup_costs_retailers']
    demands = data['demands']  # [scenario][retailer][period]
    initial_inv = data.get('initial_inventory', 0)
    initial_setups = data.get('initial_setups_imposed', True)

    start_time = time.time()

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Build a more compact formulation
    # Instead of tracking per retailer-period demand through the network,
    # we use aggregate flows and inventories per scenario.
    
    # Setup variables are first-stage (scenario-independent)
    # Flow/inventory variables are second-stage (per scenario)
    
    m = gp.Model("lot_sizing")
    m.setParam('TimeLimit', max(1, time_limit - 5))
    m.setParam('Threads', 1)
    m.setParam('MIPFocus', 1)  # Focus on feasibility
    
    # Setup variables: y[facility][period]
    # facility 0 = plant, 1..W = warehouses, W+1..W+R = retailers
    y = {}
    for t in range(T):
        y[0, t] = m.addVar(vtype=GRB.BINARY, name=f"y_0_{t}", obj=sc_plant[t])
        for w in range(W):
            y[w + 1, t] = m.addVar(vtype=GRB.BINARY, name=f"y_{w+1}_{t}", obj=sc_wh[w][t])
        for r in range(R):
            y[W + 1 + r, t] = m.addVar(vtype=GRB.BINARY, name=f"y_{W+1+r}_{t}", obj=sc_ret[r][t])

    # For the detailed formulation tracking retailer-period demands:
    # Variables per scenario, retailer, demand_period, production_period
    # q_plant[s,r,dp,t] = quantity produced at plant in period t for retailer r's demand in period dp, scenario s
    # q_wh[s,r,dp,t] = quantity shipped from plant to warehouse in period t for (r,dp,s)
    # q_ret[s,r,dp,t] = quantity shipped from warehouse to retailer in period t for (r,dp,s)
    # inv_plant[s,r,dp,t], inv_wh[s,r,dp,t], inv_ret[s,r,dp,t]
    
    # This can be very large. Let's see the dimensions.
    # For each scenario s, retailer r, demand period dp, we have periods t=0..dp
    # Total number of (r, dp) pairs with nonzero demand varies.
    
    # To keep it manageable, let's use a more compact formulation:
    # Aggregate production, delivery, and inventory per scenario.
    
    # Actually, let me use the echelon stock formulation or the 
    # standard lot-sizing formulation with aggregate quantities.
    
    # Standard approach: For each scenario, we have aggregate quantities.
    # prod[s,t] = total production at plant in period t, scenario s
    # del_wh[s,w,t] = total delivery from plant to warehouse w in period t, scenario s  
    # del_ret[s,r,t] = total delivery from warehouse to retailer r in period t, scenario s
    # inv_p[s,t], inv_w[s,w,t], inv_r[s,r,t] = aggregate inventories
    
    # The setup constraints link to aggregate quantities via big-M.
    # Big-M for plant in period t: sum of all future demands across all retailers
    # Big-M for warehouse w in period t: sum of future demands of retailers assigned to w
    # Big-M for retailer r in period t: sum of future demands of retailer r
    
    # Compute big-M values per scenario
    # For plant: M_plant[s,t] = sum over all r, all dp >= t of demands[s][r][dp]
    # For warehouse w: M_wh[s,w,t] = sum over r in w's retailers, dp >= t of demands[s][r][dp]
    # For retailer r: M_ret[s,r,t] = sum over dp >= t of demands[s][r][dp]
    
    # But setup is scenario-independent, so big-M should be max over scenarios
    
    # Compute max demands for big-M
    max_total_demand = [[0]*T for _ in range(T)]  # not needed in this form
    
    # Group retailers by warehouse
    wh_retailers = [[] for _ in range(W)]
    for r in range(R):
        wh_retailers[r2w[r]].append(r)
    
    # Big-M for plant in period t (scenario-independent upper bound)
    M_plant = [0] * T
    for t in range(T):
        max_val = 0
        for s in range(S):
            val = sum(demands[s][r][dp] for r in range(R) for dp in range(t, T))
            max_val = max(max_val, val)
        M_plant[t] = max_val
    
    M_wh = [[0]*T for _ in range(W)]
    for w in range(W):
        for t in range(T):
            max_val = 0
            for s in range(S):
                val = sum(demands[s][r][dp] for r in wh_retailers[w] for dp in range(t, T))
                max_val = max(max_val, val)
            M_wh[w][t] = max_val
    
    M_ret = [[0]*T for _ in range(R)]
    for r in range(R):
        for t in range(T):
            max_val = 0
            for s in range(S):
                val = sum(demands[s][r][dp] for dp in range(t, T))
                max_val = max(max_val, val)
            M_ret[r][t] = max_val
    
    # Second-stage variables
    prod = {}  # prod[s,t]
    del_wh = {}  # del_wh[s,w,t]
    del_ret = {}  # del_ret[s,r,t]
    inv_p = {}  # inv_p[s,t]
    inv_w = {}  # inv_w[s,w,t]
    inv_r = {}  # inv_r[s,r,t]
    
    for s in range(S):
        p = probs[s]
        for t in range(T):
            prod[s, t] = m.addVar(lb=0, name=f"prod_{s}_{t}")
            inv_p[s, t] = m.addVar(lb=0, name=f"inv_p_{s}_{t}", obj=p * hc_plant)
            for w in range(W):
                del_wh[s, w, t] = m.addVar(lb=0, name=f"del_wh_{s}_{w}_{t}")
                inv_w[s, w, t] = m.addVar(lb=0, name=f"inv_w_{s}_{w}_{t}", obj=p * hc_wh)
            for r in range(R):
                del_ret[s, r, t] = m.addVar(lb=0, name=f"del_ret_{s}_{r}_{t}")
                inv_r[s, r, t] = m.addVar(lb=0, name=f"inv_r_{s}_{r}_{t}", obj=p * hc_ret[r])
    
    m.update()
    
    # Initial setups
    if initial_setups:
        m.addConstr(y[0, 0] == 1)
        for w in range(W):
            m.addConstr(y[w + 1, 0] == 1)
        for r in range(R):
            m.addConstr(y[W + 1 + r, 0] == 1)
    
    # Constraints per scenario
    for s in range(S):
        for t in range(T):
            # Plant inventory balance
            # inv_p[s,t] = inv_p[s,t-1] + prod[s,t] - sum_w del_wh[s,w,t]
            prev_inv_p = initial_inv if t == 0 else inv_p[s, t - 1]
            m.addConstr(
                inv_p[s, t] == prev_inv_p + prod[s, t] - gp.quicksum(del_wh[s, w, t] for w in range(W))
            )
            
            # Setup constraint for plant
            m.addConstr(prod[s, t] <= M_plant[t] * y[0, t])
            
            for w in range(W):
                # Warehouse inventory balance
                prev_inv_w = initial_inv if t == 0 else inv_w[s, w, t - 1]
                m.addConstr(
                    inv_w[s, w, t] == prev_inv_w + del_wh[s, w, t] - gp.quicksum(del_ret[s, r, t] for r in wh_retailers[w])
                )
                
                # Setup constraint for warehouse
                m.addConstr(del_wh[s, w, t] <= M_wh[w][t] * y[w + 1, t])
            
            for r in range(R):
                # Retailer inventory balance
                prev_inv_r = initial_inv if t == 0 else inv_r[s, r, t - 1]
                m.addConstr(
                    inv_r[s, r, t] == prev_inv_r + del_ret[s, r, t] - demands[s][r][t]
                )
                
                # Setup constraint for retailer
                m.addConstr(del_ret[s, r, t] <= M_ret[r][t] * y[W + 1 + r, t])
    
    m.setAttr('ModelSense', GRB.MINIMIZE)
    
    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    remaining_time = max(1, time_limit - (time.time() - start_time) - 2)
    m.setParam('TimeLimit', remaining_time)
    
    m.optimize(callback)
    
    # Extract solution
    if m.SolCount == 0:
        # No feasible solution found, try to return something
        raise RuntimeError("No feasible solution found")
    
    obj_val = m.ObjVal
    
    # Build output
    solution = {
        "objective_value": obj_val,
        "setup_variables": {},
        "production_plant": {},
        "delivery_warehouse": {},
        "delivery_retailer": {},
        "inventory_plant": {},
        "inventory_warehouse": {},
        "inventory_retailer": {}
    }
    
    # Setup variables
    for t in range(T):
        solution["setup_variables"][f"y_0_{t}"] = round(y[0, t].X)
        for w in range(W):
            solution["setup_variables"][f"y_{w+1}_{t}"] = round(y[w + 1, t].X)
        for r in range(R):
            solution["setup_variables"][f"y_{W+1+r}_{t}"] = round(y[W + 1 + r, t].X)
    
    # Production, delivery, inventory
    for s in range(S):
        for t in range(T):
            solution["production_plant"][f"{t}_{s}"] = max(0, prod[s, t].X)
            solution["inventory_plant"][f"{t}_{s}"] = max(0, inv_p[s, t].X)
            for w in range(W):
                solution["delivery_warehouse"][f"{w}_{t}_{s}"] = max(0, del_wh[s, w, t].X)
                solution["inventory_warehouse"][f"{w}_{t}_{s}"] = max(0, inv_w[s, w, t].X)
            for r in range(R):
                solution["delivery_retailer"][f"{r}_{t}_{s}"] = max(0, del_ret[s, r, t].X)
                solution["inventory_retailer"][f"{r}_{t}_{s}"] = max(0, inv_r[s, r, t].X)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    if logger:
        logger.log(obj_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()