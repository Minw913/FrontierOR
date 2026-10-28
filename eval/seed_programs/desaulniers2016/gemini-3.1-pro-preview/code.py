import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Inventory Routing Problem Optimizer")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, help="Path to logger file")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_customers = data["num_customers"]
    num_periods = data["num_periods"]
    num_vehicles = data["num_vehicles"]
    vehicle_capacity = data["vehicle_capacity"]
    num_nodes = data["num_nodes"]
    dist = data["distance_matrix"]
    holding_costs = data["holding_costs"]
    customer_demands = data["customer_demands"]
    supplier_production = data["supplier_production"]
    inv_cap = data["inventory_capacities"]
    init_inv = data["initial_inventories"]

    # Configure SolutionLogger if provided
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    # Sets
    N = num_customers
    T = num_periods
    K = num_vehicles

    model = gp.Model("IRP")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)
    model.setParam("LazyConstraints", 1)

    # ---------- VARIABLES ----------
    # y[i,k,t]: 1 if node i (0=depot, 1..N=customers) visited by vehicle k in period t
    y = {}
    for i in range(N + 1):
        for k in range(1, K + 1):
            for t in range(1, T + 1):
                y[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{t}")

    # x[i,j,k,t]: 1 if vehicle k traverses arc i->j in period t
    x = {}
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                for k in range(1, K + 1):
                    for t in range(1, T + 1):
                        x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")

    # q[i,k,t]: amount delivered to customer i by vehicle k in period t
    q = {}
    for i in range(1, N + 1):
        for k in range(1, K + 1):
            for t in range(1, T + 1):
                q[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{i}_{k}_{t}")

    # I[i,t]: inventory at location i at end of period t
    I = {}
    for i in range(N + 1):
        for t in range(T + 1):
            I[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{t}")

    model.update()

    # ---------- OBJECTIVE ----------
    travel_cost = gp.quicksum(
        dist[i][j] * x[i, j, k, t]
        for i in range(N + 1) for j in range(N + 1) if i != j
        for k in range(1, K + 1) for t in range(1, T + 1)
    )
    holding_cost = gp.quicksum(
        holding_costs[i] * I[i, t]
        for i in range(N + 1) for t in range(1, T + 1)
    )
    model.setObjective(travel_cost + holding_cost, GRB.MINIMIZE)

    # ---------- CONSTRAINTS ----------
    # 1. Initial Inventory
    for i in range(N + 1):
        model.addConstr(I[i, 0] == init_inv[i], name=f"InitInv_{i}")

    # 2. Inventory Balance and Capacity
    for t in range(1, T + 1):
        # Depot Balance
        model.addConstr(
            I[0, t] == I[0, t - 1] + supplier_production[t - 1] - gp.quicksum(q[i, k, t] for i in range(1, N + 1) for k in range(1, K + 1)),
            name=f"DepotBal_{t}"
        )
        # Depot Capacity
        model.addConstr(I[0, t] <= inv_cap[0], name=f"DepotCap_{t}")

        for i in range(1, N + 1):
            # Customer Balance
            model.addConstr(
                I[i, t] == I[i, t - 1] + gp.quicksum(q[i, k, t] for k in range(1, K + 1)) - customer_demands[i - 1][t - 1],
                name=f"CustBal_{i}_{t}"
            )
            # Immediate after-delivery capacity at customer
            model.addConstr(
                I[i, t - 1] + gp.quicksum(q[i, k, t] for k in range(1, K + 1)) <= inv_cap[i],
                name=f"CustImmedCap_{i}_{t}"
            )

    # 3. Deliveries and Vehicle Capacity
    for t in range(1, T + 1):
        for k in range(1, K + 1):
            # Vehicle overall capacity
            model.addConstr(
                gp.quicksum(q[i, k, t] for i in range(1, N + 1)) <= vehicle_capacity * y[0, k, t],
                name=f"VehCap_{k}_{t}"
            )
            # Deliveries bounded by visit indicator and customer capacity
            for i in range(1, N + 1):
                model.addConstr(
                    q[i, k, t] <= min(vehicle_capacity, inv_cap[i]) * y[i, k, t],
                    name=f"DelBound_{i}_{k}_{t}"
                )

    # 4. Routing and Visits
    for t in range(1, T + 1):
        for i in range(1, N + 1):
            # At most one visit per customer per period
            model.addConstr(gp.quicksum(y[i, k, t] for k in range(1, K + 1)) <= 1, name=f"MaxVisit_{i}_{t}")
            
        for k in range(1, K + 1):
            for i in range(1, N + 1):
                # Cannot visit customer i if depot is not visited by vehicle k
                model.addConstr(y[i, k, t] <= y[0, k, t], name=f"DepotLink_{i}_{k}_{t}")

    # 5. Network flow consistency per vehicle
    for t in range(1, T + 1):
        for k in range(1, K + 1):
            for i in range(N + 1):
                out_degree = gp.quicksum(x[i, j, k, t] for j in range(N + 1) if j != i)
                in_degree = gp.quicksum(x[j, i, k, t] for j in range(N + 1) if j != i)
                model.addConstr(out_degree == y[i, k, t], name=f"OutDeg_{i}_{k}_{t}")
                model.addConstr(in_degree == y[i, k, t], name=f"InDeg_{i}_{k}_{t}")

    # 6. Break vehicle symmetry
    for t in range(1, T + 1):
        for k in range(1, K):
            model.addConstr(y[0, k, t] >= y[0, k + 1, t], name=f"Symm_{k}_{t}")

    # Subtour elimination callback setup
    model._x = x
    model._best_obj = float('inf')

    def subtour_elimination_callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                x_vals = m.cbGetSolution(m._x)
                is_feasible = True

                for t in range(1, T + 1):
                    for k in range(1, K + 1):
                        # Extract chosen edges
                        edges = [(i, j) for i in range(N + 1) for j in range(N + 1) if i != j and x_vals[i, j, k, t] > 0.5]
                        if not edges:
                            continue
                        
                        adj = {i: [] for i in range(N + 1)}
                        nodes_in_route = set()
                        for u, v in edges:
                            adj[u].append(v)
                            nodes_in_route.add(u)
                            nodes_in_route.add(v)

                        unvisited = set(nodes_in_route)
                        components = []
                        while unvisited:
                            start = unvisited.pop()
                            comp = [start]
                            q_bfs = [start]
                            while q_bfs:
                                curr = q_bfs.pop(0)
                                for nbr in adj[curr]:
                                    if nbr in unvisited:
                                        unvisited.remove(nbr)
                                        comp.append(nbr)
                                        q_bfs.append(nbr)
                            components.append(comp)

                        for comp in components:
                            if 0 not in comp and len(comp) > 1:
                                is_feasible = False
                                m.cbLazy(
                                    gp.quicksum(m._x[i, j, k, t] for i in comp for j in comp if i != j) <= len(comp) - 1
                                )
                
                # Log improved solution
                if is_feasible and logger is not None:
                    obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < m._best_obj - 1e-5:
                        m._best_obj = obj_val
                        logger.log(obj_val)
            except Exception:
                pass

    model.optimize(subtour_elimination_callback)

    # Proceed to output solution extraction
    if model.SolCount >= 1:
        obj_val = float(model.ObjVal)
        routes = {}
        
        for t in range(1, T + 1):
            period_routes = {}
            for k in range(1, K + 1):
                if y[0, k, t].X > 0.5:
                    route_seq = [0]
                    curr = 0
                    max_steps = N + 2
                    steps = 0
                    while steps < max_steps:
                        next_node = None
                        for j in range(N + 1):
                            if j != curr and x[curr, j, k, t].X > 0.5:
                                next_node = j
                                break
                        if next_node is None:
                            break
                        route_seq.append(next_node)
                        curr = next_node
                        if curr == 0:
                            break
                        steps += 1
                        
                    deliveries = {}
                    for i in range(1, N + 1):
                        if y[i, k, t].X > 0.5:
                            qty = max(0, int(round(q[i, k, t].X)))
                            deliveries[str(i)] = qty

                    period_routes[str(k)] = {
                        "route": route_seq,
                        "deliveries": deliveries
                    }
            if period_routes:
                routes[str(t)] = period_routes

        inventories = {}
        for i in range(N + 1):
            inventories[str(i)] = {}
            for t in range(1, T + 1):
                inventories[str(i)][str(t)] = max(0, int(round(I[i, t].X)))
    else:
        # Fallback empty solution format if timeout before feasibility
        obj_val = 0.0
        routes = {}
        inventories = {str(i): {str(t): init_inv[i] for t in range(1, T + 1)} for i in range(N + 1)}

    solution = {
        "objective_value": obj_val,
        "routes": routes,
        "inventories": inventories
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()