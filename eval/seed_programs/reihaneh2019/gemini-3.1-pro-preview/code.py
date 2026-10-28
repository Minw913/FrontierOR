import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser(description="Distribution Company Location-Routing Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, required=False, help="Path for logging intermediate solutions.")
    args = parser.parse_args()

    # Logger initialization
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_sites = data["num_delivery_sites"]
    num_cust = data["num_customers"]
    Q = data["vehicle_capacity"]
    K = data["num_vehicles"]
    
    depot_id = data["depot"]["id"]
    site_ids = [site["id"] for site in data["delivery_sites"]]
    routing_nodes = [depot_id] + site_ids
    
    customer_ids = [cust["id"] for cust in data["customers"]]
    demands = [cust["demand"] for cust in data["customers"]]

    routing_cost = data["routing_cost_matrix"]
    assignment_cost = data["assignment_cost_matrix"]

    # Initialize Gurobi model
    m = gp.Model("DeliveryOptimization")
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)

    # ------------------
    # Decisions Variables
    # ------------------
    x = {} # binary, 1 if route used between i and j
    f = {} # continuous, payload size carried along route (i, j)
    for i in range(num_sites + 1):
        for j in range(num_sites + 1):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                f[i, j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{i}_{j}")
    
    y = {} # binary, 1 if candidate site s is opened
    for s in range(1, num_sites + 1):
        y[s] = m.addVar(vtype=GRB.BINARY, name=f"y_{s}")

    z = {} # binary, 1 if customer c assigned to site s
    for c in range(num_cust):
        for s in range(1, num_sites + 1):
            z[c, s] = m.addVar(vtype=GRB.BINARY, name=f"z_{c}_{s}")

    # ------------------
    # Objective Formulation
    # ------------------
    obj = gp.LinExpr()
    
    # 1. Vehicle routing cost
    for i in range(num_sites + 1):
        for j in range(num_sites + 1):
            if i != j:
                obj += routing_cost[i][j] * x[i, j]
                
    # 2. Customer assignment cost: notice shape is customers x delivery_sites
    for c in range(num_cust):
        for s in range(1, num_sites + 1):
            obj += assignment_cost[c][s-1] * z[c, s]

    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------
    # Constraints
    # ------------------
    
    # All customers must be assigned precisely to one delivery site
    for c in range(num_cust):
        m.addConstr(gp.quicksum(z[c, s] for s in range(1, num_sites + 1)) == 1, name=f"assign_{c}")

    # Connect site activation binary with customer assignments
    for s in range(1, num_sites + 1):
        for c in range(num_cust):
            m.addConstr(z[c, s] <= y[s], name=f"link_{c}_{s}")
        # Exclude null loops: If site is active, strictly requires at least 1 customer demand
        m.addConstr(y[s] <= gp.quicksum(z[c, s] for c in range(num_cust)), name=f"active_min_{s}")

    # Entering sequence degree equals leaving sequence degree if site actively routed
    for s in range(1, num_sites + 1):
        m.addConstr(gp.quicksum(x[i, s] for i in range(num_sites + 1) if i != s) == y[s], name=f"in_deg_{s}")
        m.addConstr(gp.quicksum(x[s, j] for j in range(num_sites + 1) if j != s) == y[s], name=f"out_deg_{s}")

    # Maximum routes bounds leaving from the depot against fleet size
    m.addConstr(gp.quicksum(x[0, s] for s in range(1, num_sites + 1)) <= K, name="max_vehicles_out")
    # Balancing the loop counts at the depot
    m.addConstr(
        gp.quicksum(x[0, s] for s in range(1, num_sites + 1)) == gp.quicksum(x[s, 0] for s in range(1, num_sites + 1)),
        name="depot_balance"
    )

    # 1-Commodity Delivery Load balances (avoids disconnected subtour and tracks cumulative quantity)
    for s in range(1, num_sites + 1):
        m.addConstr(
            gp.quicksum(f[i, s] for i in range(num_sites + 1) if i != s) - 
            gp.quicksum(f[s, j] for j in range(num_sites + 1) if j != s) == 
            gp.quicksum(demands[c] * z[c, s] for c in range(num_cust)),
            name=f"flow_bal_{s}"
        )

    # Capacity constraints alongside restricting unused physical routes
    for i in range(num_sites + 1):
        for j in range(num_sites + 1):
            if i != j:
                m.addConstr(f[i, j] <= Q * x[i, j], name=f"flow_max_{i}_{j}")
                
    # Tour ends with zero remaining carried demand back at the depot
    for s in range(1, num_sites + 1):
        m.addConstr(f[s, 0] == 0, name=f"flow_ret_{s}")

    # Capacity validation inequalities & Subtour prevention strengthening via delivery drop-off tracking
    for i in range(num_sites + 1):
        for j in range(1, num_sites + 1):
            if i != j:
                m.addConstr(
                    f[i, j] >= gp.quicksum(demands[c] * z[c, j] for c in range(num_cust)) - Q * (1 - x[i, j]),
                    name=f"flow_min_{i}_{j}"
                )

    # Basic topological elimination of 2-pair sequences purely between candidates (non-depot vertices)
    for i in range(1, num_sites + 1):
        for j in range(i + 1, num_sites + 1):
            m.addConstr(x[i, j] + x[j, i] <= 1, name=f"sym_elim_{i}_{j}")

    # Callback declaration for internal incumbent progress
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Start solver routine
    m.optimize(cb)

    # Parse optimization outputs into required structured dictionaries
    if m.SolCount > 0:
        edges = []
        for i in range(num_sites + 1):
            for j in range(num_sites + 1):
                if i != j and x[i, j].X > 0.5:
                    edges.append((i, j))
        
        assignment_res = {}
        site_to_customers = {s: [] for s in range(1, num_sites + 1)}
        for c in range(num_cust):
            for s in range(1, num_sites + 1):
                if z[c, s].X > 0.5:
                    assignment_res[c] = s
                    site_to_customers[s].append(c)

        out_edges_from_depot = [j for (i, j) in edges if i == 0]
        routes_output = {}
        customer_tours = {}
        vehicle_id = 1

        for start_node in out_edges_from_depot:
            curr = start_node
            route_nodes = [0, curr]
            while curr != 0:
                nxt_nodes = [j for (i, j) in edges if i == curr]
                if not nxt_nodes:
                    break
                nxt = nxt_nodes[0]
                route_nodes.append(nxt)
                curr = nxt
            
            v_key = str(vehicle_id)
            arcs = [[routing_nodes[route_nodes[k]], routing_nodes[route_nodes[k+1]]] for k in range(len(route_nodes)-1)]
            seq = [routing_nodes[n] for n in route_nodes]
            
            routes_output[v_key] = {
                "arcs": arcs,
                "sequence": seq
            }
            
            for n in route_nodes:
                if n != 0:
                    for c in site_to_customers[n]:
                        customer_tours[str(customer_ids[c])] = {
                            "vehicle": v_key,
                            "site": routing_nodes[n]
                        }
                        
            vehicle_id += 1

        assignments_output = {str(customer_ids[c]): routing_nodes[s] for c, s in assignment_res.items()}

        sol = {
            "objective_value": m.ObjVal,
            "routes": routes_output,
            "assignments": assignments_output,
            "customer_tours": customer_tours
        }

        # Write solution outputs to file following formatting boundaries strictly
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)

if __name__ == "__main__":
    solve()