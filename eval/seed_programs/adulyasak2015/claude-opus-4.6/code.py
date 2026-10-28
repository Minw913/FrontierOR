import argparse
import json
import math
import time
import sys
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    T = data['T']
    m = data['m']
    Q = data['Q']
    C = data['C']
    f_cost = data['f']
    u_cost = data['u']
    coords = data['coordinates']
    tc = data['transportation_costs']
    h = data['h']
    L = data['L']
    I0 = data['I0']
    d_bar = data['d_bar']
    sigma = data['sigma']
    n_scenarios = data['n_scenarios']
    scenario_probs = data['scenario_probabilities']
    demand_scenarios = data['demand_scenarios']
    
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("IRP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(time_limit - 10, 10)
    
    # Index sets
    customers = list(range(1, n + 1))
    nodes = list(range(0, n + 1))  # 0 = plant
    vehicles = list(range(1, m + 1))
    periods = list(range(1, T + 1))
    scenarios = list(range(n_scenarios))
    
    # First stage variables
    y = {}  # setup
    z = {}  # visit: z[i,k,t]
    x = {}  # edge traversal: x[i,j,k,t] where i < j
    
    for t in periods:
        y[t] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}")
    
    for t in periods:
        for k in vehicles:
            z[0, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_0_{k}_{t}")
            for i in customers:
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")
    
    for t in periods:
        for k in vehicles:
            for i in nodes:
                for j in nodes:
                    if i < j:
                        if i == 0:
                            x[i, j, k, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"x_{i}_{j}_{k}_{t}")
                        else:
                            x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")
    
    # Second stage variables (per scenario)
    p = {}  # production
    I_var = {}  # inventory
    e = {}  # unmet demand
    q = {}  # delivery quantity
    
    for omega in scenarios:
        for t in periods:
            p[t, omega] = model.addVar(lb=0, ub=C, name=f"p_{t}_{omega}")
        
        # Plant inventory
        for t in range(0, T + 1):  # t=0 is initial
            I_var[0, t, omega] = model.addVar(lb=0, ub=L[0], name=f"I_0_{t}_{omega}")
        
        # Customer inventory and unmet demand
        for i in customers:
            for t in range(0, T + 1):
                I_var[i, t, omega] = model.addVar(lb=0, ub=L[i], name=f"I_{i}_{t}_{omega}")
            for t in periods:
                e[i, t, omega] = model.addVar(lb=0, name=f"e_{i}_{t}_{omega}")
        
        # Delivery quantities
        for t in periods:
            for k in vehicles:
                for i in customers:
                    q[i, k, t, omega] = model.addVar(lb=0, name=f"q_{i}_{k}_{t}_{omega}")
    
    model.update()
    
    # === CONSTRAINTS ===
    
    # Initial inventories (for all scenarios)
    for omega in scenarios:
        model.addConstr(I_var[0, 0, omega] == I0[0])
        for i in customers:
            model.addConstr(I_var[i, 0, omega] == I0[i])
    
    # Each customer visited by at most one vehicle per period
    for t in periods:
        for i in customers:
            model.addConstr(gp.quicksum(z[i, k, t] for k in vehicles) <= 1)
    
    # z[0,k,t] = 1 if vehicle k is dispatched
    for t in periods:
        for k in vehicles:
            model.addConstr(z[0, k, t] == gp.quicksum(z[i, k, t] for i in customers) / max(1, min(n, 1)))
            # Actually: z[0,k,t] >= z[i,k,t] for all i, and z[0,k,t] <= sum z[i,k,t]
            for i in customers:
                model.addConstr(z[0, k, t] >= z[i, k, t])
            model.addConstr(z[0, k, t] <= gp.quicksum(z[i, k, t] for i in customers))
    
    # Routing constraints - degree constraints
    for t in periods:
        for k in vehicles:
            # For plant (node 0): degree = 2 * z[0,k,t]
            model.addConstr(
                gp.quicksum(x[0, j, k, t] for j in customers if (0, j, k, t) in x) == 2 * z[0, k, t]
            )
            # For each customer: degree = 2 * z[i,k,t]
            for i in customers:
                edges_incident = []
                for j in nodes:
                    if i != j:
                        a, b = min(i, j), max(i, j)
                        if (a, b, k, t) in x:
                            edges_incident.append(x[a, b, k, t])
                model.addConstr(gp.quicksum(edges_incident) == 2 * z[i, k, t])
    
    # Subtour elimination: We'll use lazy constraints via callback
    # For small n, we can add all subset constraints upfront; for larger n use lazy
    
    # Symmetry breaking: order vehicles
    for t in periods:
        for k in vehicles:
            if k > 1:
                model.addConstr(z[0, k, t] <= z[0, k-1, t])
    
    # Production setup
    for t in periods:
        for omega in scenarios:
            model.addConstr(p[t, omega] <= C * y[t])
    
    # Plant inventory balance
    for t in periods:
        for omega in scenarios:
            total_shipped = gp.quicksum(q[i, k, t, omega] for i in customers for k in vehicles)
            model.addConstr(
                I_var[0, t-1, omega] + p[t, omega] == total_shipped + I_var[0, t, omega]
            )
    
    # Customer inventory balance
    for t in periods:
        for omega in scenarios:
            for i in customers:
                d_it = demand_scenarios[omega][i-1][t-1]
                total_delivered = gp.quicksum(q[i, k, t, omega] for k in vehicles)
                model.addConstr(
                    I_var[i, t-1, omega] + total_delivered + e[i, t, omega] == d_it + I_var[i, t, omega]
                )
    
    # Delivery capacity: customer storage
    for t in periods:
        for omega in scenarios:
            for i in customers:
                d_it = demand_scenarios[omega][i-1][t-1]
                total_delivered = gp.quicksum(q[i, k, t, omega] for k in vehicles)
                # ending inventory + demand <= L[i] => I_prev + delivered <= L[i]
                # Actually: I_var[i,t,omega] + d_it <= L[i] but d_it is consumed
                # The constraint says: ending inventory + period demand <= storage capacity
                # Which means: I_var[i,t,omega] <= L[i] (already bounded) but also
                # deliveries should respect capacity
                model.addConstr(I_var[i, t-1, omega] + total_delivered <= L[i])
    
    # Delivery only if visited
    for t in periods:
        for k in vehicles:
            for i in customers:
                for omega in scenarios:
                    model.addConstr(q[i, k, t, omega] <= L[i] * z[i, k, t])
    
    # Vehicle capacity
    for t in periods:
        for k in vehicles:
            for omega in scenarios:
                model.addConstr(
                    gp.quicksum(q[i, k, t, omega] for i in customers) <= Q * z[0, k, t]
                )
    
    # Delivery bounded by future demand
    for t in periods:
        for omega in scenarios:
            for i in customers:
                future_demand = sum(demand_scenarios[omega][i-1][t2-1] for t2 in range(t, T+1))
                total_delivered = gp.quicksum(q[i, k, t, omega] for k in vehicles)
                model.addConstr(total_delivered <= future_demand)
    
    # Edge only if both endpoints visited
    for t in periods:
        for k in vehicles:
            for i in nodes:
                for j in nodes:
                    if i < j and (i, j, k, t) in x:
                        if i == 0:
                            model.addConstr(x[i, j, k, t] <= 2 * z[j, k, t])
                        else:
                            model.addConstr(x[i, j, k, t] <= z[i, k, t])
                            model.addConstr(x[i, j, k, t] <= z[j, k, t])
    
    # Objective
    obj = gp.LinExpr()
    
    # Fixed setup costs
    for t in periods:
        obj += f_cost * y[t]
    
    # Transportation costs (first stage, not scenario dependent)
    for t in periods:
        for k in vehicles:
            for i in nodes:
                for j in nodes:
                    if i < j and (i, j, k, t) in x:
                        obj += tc[i][j] * x[i, j, k, t]
    
    # Scenario-dependent costs
    for omega in scenarios:
        prob = scenario_probs[omega]
        # Production costs
        for t in periods:
            obj += prob * u_cost * p[t, omega]
        # Plant inventory holding
        for t in periods:
            obj += prob * h[0] * I_var[0, t, omega]
        # Customer inventory holding and unmet demand penalties
        for i in customers:
            for t in periods:
                obj += prob * h[i] * I_var[i, t, omega]
                obj += prob * sigma[i-1] * e[i, t, omega]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Subtour elimination via lazy constraints
    model.Params.LazyConstraints = 1
    
    def find_subtours(z_vals, x_vals, k, t):
        """Find connected components among visited customers for vehicle k in period t."""
        visited = [i for i in customers if z_vals.get((i, k, t), 0) > 0.5]
        if len(visited) <= 1:
            return []
        
        # Build adjacency
        adj = {i: [] for i in visited}
        for i in nodes:
            for j in nodes:
                if i < j and (i, j, k, t) in x_vals:
                    val = x_vals[(i, j, k, t)]
                    if val > 0.5:
                        if i in adj and j in adj:
                            adj[i].append(j)
                            adj[j].append(i)
                        elif i == 0:
                            pass  # plant connection
                        elif j == 0:
                            pass
        
        # Find components not connected to plant
        # BFS from each node
        plant_connected = set()
        for i in visited:
            for j in nodes:
                a, b = min(0, i), max(0, i)
                if a == 0 and (a, b, k, t) in x_vals and x_vals[(a, b, k, t)] > 0.5:
                    plant_connected.add(i)
        
        # BFS to find all plant-connected nodes
        queue = list(plant_connected)
        seen = set(plant_connected)
        while queue:
            node = queue.pop(0)
            for neighbor in adj.get(node, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        
        # Nodes not connected to plant form subtours
        disconnected = [i for i in visited if i not in seen]
        if not disconnected:
            return []
        
        # Find connected components among disconnected
        remaining = set(disconnected)
        subtours = []
        while remaining:
            start = next(iter(remaining))
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in remaining:
                    remaining.discard(node)
                    component.add(node)
                    for neighbor in adj.get(node, []):
                        if neighbor in remaining:
                            queue.append(neighbor)
            subtours.append(component)
        
        return subtours
    
    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            z_vals = {}
            x_vals = {}
            for key, var in z.items():
                z_vals[key] = model_cb.cbGetSolution(var)
            for key, var in x.items():
                x_vals[key] = model_cb.cbGetSolution(var)
            
            for t in periods:
                for k in vehicles:
                    subtours = find_subtours(z_vals, x_vals, k, t)
                    for S in subtours:
                        # SEC: sum of edges within S <= |S| - 1
                        S_list = list(S)
                        edges_in_S = []
                        for i_idx in range(len(S_list)):
                            for j_idx in range(i_idx + 1, len(S_list)):
                                a, b = min(S_list[i_idx], S_list[j_idx]), max(S_list[i_idx], S_list[j_idx])
                                if (a, b, k, t) in x:
                                    edges_in_S.append(x[a, b, k, t])
                        if edges_in_S:
                            model_cb.cbLazy(
                                gp.quicksum(edges_in_S) <= len(S) - 1
                            )
    
    # For small instances, add SEC constraints directly
    if n <= 8:
        for t in periods:
            for k in vehicles:
                for size in range(2, n + 1):
                    for S in combinations(customers, size):
                        edges_in_S = []
                        for i_idx in range(len(S)):
                            for j_idx in range(i_idx + 1, len(S)):
                                a, b = min(S[i_idx], S[j_idx]), max(S[i_idx], S[j_idx])
                                if (a, b, k, t) in x:
                                    edges_in_S.append(x[a, b, k, t])
                        if edges_in_S:
                            model.addConstr(gp.quicksum(edges_in_S) <= len(S) - 1)
        model.Params.LazyConstraints = 0
        model.optimize()
    else:
        model.optimize(callback)
    
    # Extract solution
    solution = {
        "objective_value": 0.0,
        "model_variables": {},
        "y": {},
        "z": {},
        "x": {}
    }
    
    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        mv = {}
        
        # y variables
        y_dict = {}
        for t in periods:
            val = y[t].X
            if abs(val) > 1e-6:
                y_dict[str(t)] = round(val)
                mv[f"y_{t}"] = round(val)
        solution["y"] = y_dict
        
        # z variables
        z_dict = {}
        for (i, k, t), var in z.items():
            val = var.X
            if abs(val) > 1e-6:
                key = f"{i}_{k}_{t}"
                z_dict[key] = round(val)
                mv[f"z_{i}_{k}_{t}"] = round(val)
        solution["z"] = z_dict
        
        # x variables
        x_dict = {}
        for (i, j, k, t), var in x.items():
            val = var.X
            if abs(val) > 1e-6:
                key = f"{i}_{j}_{k}_{t}"
                x_dict[key] = round(val)
                mv[f"x_{i}_{j}_{k}_{t}"] = round(val)
        solution["x"] = x_dict
        
        # Scenario variables
        for omega in scenarios:
            for t in periods:
                val = p[t, omega].X
                if abs(val) > 1e-6:
                    mv[f"p_{t}_{omega+1}"] = val
            
            for i in nodes:
                for t in periods:
                    val = I_var[i, t, omega].X
                    if abs(val) > 1e-6:
                        mv[f"I_{i}_{t}_{omega+1}"] = val
            
            for i in customers:
                for t in periods:
                    val = e[i, t, omega].X
                    if abs(val) > 1e-6:
                        mv[f"e_{i}_{t}_{omega+1}"] = val
            
            for t in periods:
                for k in vehicles:
                    for i in customers:
                        val = q[i, k, t, omega].X
                        if abs(val) > 1e-6:
                            mv[f"q_{i}_{k}_{t}_{omega+1}"] = val
        
        solution["model_variables"] = mv
        
        if logger:
            logger.log_solution(solution["objective_value"], solution)
    else:
        # No solution found - return empty
        solution["objective_value"] = float('inf')
        solution["model_variables"] = {}
        solution["y"] = {}
        solution["z"] = {}
        solution["x"] = {}
    
    with open(solution_path, 'w') as f_out:
        json.dump(solution, f_out, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == "__main__":
    main()