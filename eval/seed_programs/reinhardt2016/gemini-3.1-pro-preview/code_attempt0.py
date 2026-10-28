import argparse
import json
import os
import math

import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="VRPTW with Edge Access Costs Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where solution JSON is saved.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for solution logger.")
    args = parser.parse_args()

    # Solution logging setup
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance data
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = instance["num_customers"]
    K = instance["num_vehicles"]
    vehicle_capacity = instance["vehicle_capacity"]
    
    nodes = instance["nodes"]
    # Guarantee node processing in id order
    nodes.sort(key=lambda item: item["id"])
    
    demand = [nodes[i]["demand"] for i in range(n + 1)]
    EW = [nodes[i]["time_window_start"] for i in range(n + 1)]
    LW = [nodes[i]["time_window_end"] for i in range(n + 1)]
    service_time = [nodes[i]["service_time"] for i in range(n + 1)]
    
    distance_matrix = instance["distance_matrix"]
    
    # Pre-process edge sets to map each ID to its cost and a set of frozensets (undirected edges)
    E_s = {}
    for es in instance["edge_sets"]:
        s = es["id"]
        cost = es["access_cost"]
        st = set()
        for e in es["edges"]:
            st.add(tuple(sorted((e[0], e[1]))))
        E_s[s] = {"cost": cost, "edges": st}
        
    def get_dist(i, j):
        # Maps artificial end depot n + 1 back to 0
        if i == 0 and j == n + 1:
            return 0.0
        return distance_matrix[i][0 if j == n + 1 else j]
        
    # Generate unified valid directed arcs space
    arcs = []
    for i in range(n + 1):
        for j in range(1, n + 2):
            if i != j:
                if i == 0 and j == n + 1:
                    arcs.append((0, n + 1))       # Valid transition denoting an unused vehicle
                elif i == 0 and j <= n:
                    arcs.append((0, j))           # Start to customer
                elif i > 0 and j <= n + 1:
                    arcs.append((i, j))           # Customer to customer or to end
                    
    model = gp.Model("VRPTW_Edge_Sets")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)
    
    # -----------------------------
    # VARIABLES
    # -----------------------------
    x = model.addVars(K, arcs, vtype=GRB.BINARY, name="x")
    t = model.addVars(K, n + 2, vtype=GRB.CONTINUOUS, name="t")
    v = model.addVars(K, n + 1, lb=1, ub=n, vtype=GRB.CONTINUOUS, name="v")
    z = model.addVars(E_s.keys(), vtype=GRB.BINARY, name="z")
    
    # -----------------------------
    # CONSTRAINTS
    # -----------------------------
    
    # 1. Time Bounds 
    for k in range(K):
        for i in range(n + 2):
            phys_i = 0 if i == n + 1 else i
            t[k, i].lb = EW[phys_i]
            t[k, i].ub = LW[phys_i]
            
    # 2. Time Progression & Travel Pruning 
    for k in range(K):
        for i, j in arcs:
            if i == 0 and j == n + 1:
                continue
            phys_i = i
            phys_j = 0 if j == n + 1 else j
            T_ij = service_time[phys_i] + distance_matrix[phys_i][phys_j]
            if EW[phys_i] + T_ij > LW[phys_j]:
                model.addConstr(x[k, i, j] == 0) # Infeasible arc mathematically
            else:
                M_ij = LW[phys_i] + T_ij - EW[phys_j]
                if M_ij > 0:
                    model.addConstr(t[k, j] >= t[k, i] + T_ij - M_ij * (1 - x[k, i, j]))

    # 3. MTZ Subtour Elimination 
    for k in range(K):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    model.addConstr(v[k, j] >= v[k, i] + 1 - n * (1 - x[k, i, j]))
                    
    # 4. Mandatory Single Exact Customer Visitation 
    for p in range(1, n + 1):
        model.addConstr(gp.quicksum(x[k, i, p] for k in range(K) for i in range(n + 1) if i != p) == 1)
        
    # 5. Flow Balance for Vehicles 
    for k in range(K):
        model.addConstr(gp.quicksum(x[k, 0, j] for j in range(1, n + 2)) == 1)
        model.addConstr(gp.quicksum(x[k, i, n + 1] for i in range(n + 1)) == 1)
        for p in range(1, n + 1):
            model.addConstr(
                gp.quicksum(x[k, i, p] for i in range(n + 1) if i != p) ==
                gp.quicksum(x[k, p, j] for j in range(1, n + 2) if j != p)
            )

    # 6. Unused Vehicles Zero Active Traversal Linkage 
    for k in range(K):
        for p in range(1, n + 1):
            model.addConstr(
                gp.quicksum(x[k, i, p] for i in range(n + 1) if i != p) <= 1 - x[k, 0, n + 1]
            )
            
    # 7. Capacity Control Over the Active Route Set
    for k in range(K):
        model.addConstr(
            gp.quicksum(
                demand[p] * gp.quicksum(x[k, i, p] for i in range(n + 1) if i != p)
                for p in range(1, n + 1)
            ) <= vehicle_capacity
        )
        
    # 8. Hierarchical Pack Constraint to Subdue Vehicle Permutation Search Redundancy 
    for k in range(1, K):
        model.addConstr(x[k, 0, n + 1] >= x[k - 1, 0, n + 1])
        
    # 9. Enforce Optional Arbitrary Global Network Subsets Costs Formulations (Edge Sets Access Mechanism)
    for s, data in E_s.items():
        s_edges = data["edges"]
        for p in range(1, n + 1):
            subset_in = []
            for i in range(n + 1):
                if i != p:
                    if tuple(sorted((i, p))) in s_edges:
                        subset_in.append(i)
            if subset_in:
                model.addConstr(
                    gp.quicksum(x[k, i, p] for k in range(K) for i in subset_in) <= z[s]
                )
                
            subset_out = []
            for j in range(1, n + 2):
                if j != p:
                    phys_j = 0 if j == n + 1 else j
                    if tuple(sorted((p, phys_j))) in s_edges:
                        subset_out.append(j)
            if subset_out:
                model.addConstr(
                    gp.quicksum(x[k, p, j] for k in range(K) for j in subset_out) <= z[s]
                )
                
    # -----------------------------
    # OPTIMIZATION
    # -----------------------------                
    obj = gp.quicksum(E_s[s]["cost"] * z[s] for s in E_s.keys()) + \
          gp.quicksum(get_dist(i, j) * x[k, i, j] for k in range(K) for i, j in arcs)
    model.setObjective(obj, GRB.MINIMIZE)

    def mip_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(val)

    model.optimize(mip_callback)

    # -----------------------------
    # OUTPUT EXTRACTION
    # -----------------------------    
    if model.SolCount > 0:
        routes = []
        for k in range(K):
            # Evaluate deployment usage structure
            if x[k, 0, n + 1].X < 0.5:
                seq = [0]
                route_arcs = []
                curr = 0
                while True:
                    nxt = None
                    for j in range(1, n + 2):
                        if curr != j and x[k, curr, j].X > 0.5:
                            nxt = j
                            break
                    if nxt is None:
                        break
                    
                    phys_nxt = 0 if nxt == n + 1 else nxt
                    seq.append(phys_nxt)
                    route_arcs.append([curr, phys_nxt])
                    curr = nxt
                    if curr == n + 1:
                        break
                        
                vtimes = {}
                for i in range(n + 1):
                    vtimes[str(i)] = float(t[k, i].X)
                
                routes.append({
                    "vehicle": k,
                    "sequence": seq,
                    "arcs": route_arcs,
                    "visit_times": vtimes
                })
        
        edge_sets_used = [int(s) for s in E_s.keys() if z[s].X > 0.5]
        
        out_dict = {
            "objective_value": float(model.ObjVal),
            "routes": routes,
            "edge_sets_used": edge_sets_used
        }
        
        sol_dir = os.path.dirname(args.solution_path)
        if sol_dir:
            os.makedirs(sol_dir, exist_ok=True)
            
        with open(args.solution_path, "w") as f:
            json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()