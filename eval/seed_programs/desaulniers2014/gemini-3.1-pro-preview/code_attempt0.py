import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_customers']
    T = data['num_periods']
    V = data['num_vehicles']
    Q = data['vehicle_capacity']
    depot = data['depot']
    customers = data['customers']
    dist = data['distance_matrix']

    C_dict = {c['id']: c for c in customers}
    
    m = gp.Model()
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', args.time_limit)
    
    # Track inventory at the end of each period (including initial state as period 0)
    I_d = {}
    I_d[0] = depot['initial_inventory']
    for t in range(T):
        I_d[t+1] = m.addVar(lb=0, ub=depot['max_inventory'], vtype=GRB.CONTINUOUS, name=f"I_d_{t+1}")

    I_c = {}
    for i in range(1, N+1):
        I_c[0, i] = C_dict[i]['initial_inventory']
        for t in range(T):
            I_c[t+1, i] = m.addVar(lb=C_dict[i]['min_inventory'], ub=C_dict[i]['max_inventory'], 
                                   vtype=GRB.CONTINUOUS, name=f"I_c_{t+1}_{i}")

    # Routing Configuration and Graph Network Flow Formulation Variables
    x_vars = [(t, k, i, j) for t in range(T) for k in range(V) for i in range(N+1) for j in range(N+1) if i != j]
    x = m.addVars(x_vars, vtype=GRB.BINARY, name="x")
    
    y = m.addVars(T, V, N+1, vtype=GRB.BINARY, name="y") 
    z = m.addVars(T, V, vtype=GRB.BINARY, name="z")
    qv = m.addVars(T, V, N+1, vtype=GRB.CONTINUOUS, name="qv")
    f = m.addVars(x_vars, vtype=GRB.CONTINUOUS, name="f")
    q_total = m.addVars(T, N+1, vtype=GRB.CONTINUOUS, name="q_total")

    # Flow and Balance Constraints
    for t in range(T):
        m.addConstr(I_d[t+1] == I_d[t] + depot['production_per_period'] - gp.quicksum(q_total[t, i] for i in range(1, N+1)))
        
        for i in range(1, N+1):
            m.addConstr(q_total[t, i] == gp.quicksum(qv[t, k, i] for k in range(V)))
            m.addConstr(gp.quicksum(y[t, k, i] for k in range(V)) <= 1)
            
            m.addConstr(I_c[t+1, i] == I_c[t, i] + q_total[t, i] - C_dict[i]['demand_per_period'])
            m.addConstr(I_c[t, i] + q_total[t, i] <= C_dict[i]['max_inventory']) # Inventory capacity is never exceeded
            
        for k in range(V):
            if k > 0:
                # Symmetry Breaking checks matching empty loads and vehicle ordering permutations correctly
                m.addConstr(z[t, k] <= z[t, k-1])
                m.addConstr(gp.quicksum(qv[t, k, i] for i in range(1, N+1)) <= gp.quicksum(qv[t, k-1, i] for i in range(1, N+1)))
                
            m.addConstr(gp.quicksum(qv[t, k, i] for i in range(1, N+1)) <= Q * z[t, k])
            
            m.addConstr(gp.quicksum(x[t, k, 0, j] for j in range(1, N+1)) == z[t, k])
            m.addConstr(gp.quicksum(x[t, k, j, 0] for j in range(1, N+1)) == z[t, k])
            
            # Sub-tour elimination via Network constraints / Loaded Single-Commodity Delivery Formulations guarantees validity
            m.addConstr(gp.quicksum(f[t, k, 0, j] for j in range(1, N+1)) == gp.quicksum(qv[t, k, i] for i in range(1, N+1)))
            
            for i in range(1, N+1):
                m.addConstr(y[t, k, i] <= z[t, k])
                m.addConstr(qv[t, k, i] <= min(C_dict[i]['max_inventory'], Q) * y[t, k, i])
                
                m.addConstr(gp.quicksum(x[t, k, i, j] for j in range(N+1) if j != i) == y[t, k, i])
                m.addConstr(gp.quicksum(x[t, k, j, i] for j in range(N+1) if j != i) == y[t, k, i])
                
                m.addConstr(gp.quicksum(f[t, k, j, i] for j in range(N+1) if j != i) - 
                            gp.quicksum(f[t, k, i, j] for j in range(N+1) if j != i) == qv[t, k, i])
                
                m.addConstr(f[t, k, i, 0] == 0)
                
            for i in range(N+1):
                for j in range(N+1):
                    if i != j:
                        m.addConstr(f[t, k, i, j] <= Q * x[t, k, i, j])

    travel_cost = gp.quicksum(
        dist[i][j] * x[t, k, i, j] 
        for t in range(T) 
        for k in range(V) 
        for i in range(N+1) 
        for j in range(N+1) 
        if i != j
    )

    holding_cost = gp.quicksum(
        depot['holding_cost'] * I_d[t+1] 
        for t in range(T)
    ) + gp.quicksum(
        C_dict[i]['holding_cost'] * I_c[t+1, i] 
        for t in range(T) 
        for i in range(1, N+1)
    )

    m.setObjective(travel_cost + holding_cost, GRB.MINIMIZE)
    
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    m.optimize(cb)
    
    if m.SolCount > 0:
        sol = {
            "objective_value": float(m.ObjVal),
            "solution_details": {
                "periods": {}
            }
        }
        
        for t in range(T):
            period_str = str(t + 1)
            routes = []
            
            for k in range(V):
                if z[t, k].X > 0.5:
                    route = [0]
                    curr = 0
                    visited = {0}
                    while True:
                        next_node = None
                        for j in range(N+1):
                            if j != curr and x[t, k, curr, j].X > 0.5:
                                next_node = j
                                break
                        
                        if next_node is None or next_node == 0:
                            route.append(0)
                            break
                        if next_node in visited:
                            route.append(0)
                            break
                            
                        route.append(next_node)
                        visited.add(next_node)
                        curr = next_node
                        
                    routes.append(route)
                    
            deliveries = {}
            for i in range(1, N+1):
                q_val = sum(qv[t, k, i].X for k in range(V))
                if q_val > 1e-4:
                    deliveries[str(i)] = float(round(q_val, 2))
                    
            inventories = {}
            inventories["depot"] = float(round(I_d[t+1].X, 2))
            for i in range(1, N+1):
                inventories[str(i)] = float(round(I_c[t+1, i].X, 2))
                
            sol["solution_details"]["periods"][period_str] = {
                "routes": routes,
                "deliveries": deliveries,
                "inventories": inventories
            }
            
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
    else:
        sys.exit("No feasible solution was found.")

if __name__ == "__main__":
    main()