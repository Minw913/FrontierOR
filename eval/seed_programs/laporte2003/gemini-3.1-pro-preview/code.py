import argparse
import json
import sys
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

def subtour_elimination(model, where):
    if where == GRB.Callback.MIPSOL:
        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)

        # Build graph of visited edges
        edges = [e for e in model._E if x_val[e] > 0.5]
        
        adj = {i: [] for i in model._V}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
            
        visited = set()
        components = []
        
        # Find all connected components for selected vertices
        for i in model._V:
            if y_val[i] > 0.5 and i not in visited:
                comp = []
                q_list = [i]
                visited.add(i)
                while q_list:
                    u = q_list.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q_list.append(v)
                components.append(comp)
        
        # If there are disconnected subtours, add inequalities to break them
        if len(components) > 1:
            for comp in components:
                # We only need to cut off logic for components that don't contain the depot
                if model._depot not in comp:
                    cut_edges = []
                    comp_set = frozenset(comp)
                    # Sum over all edges exiting this component
                    for i in comp:
                        for j in model._V:
                            if j not in comp_set:
                                e = (i, j) if i < j else (j, i)
                                if e in model._E:
                                    cut_edges.append(model._x[e])
                    
                    # Generate lazy constraints
                    if cut_edges:
                        # For every visited node in the disconnected component, external connectivity must exist
                        for k in comp:
                            if y_val[k] > 0.5:
                                model.cbLazy(gp.quicksum(cut_edges) >= 2 * model._y[k])
        
        else:
            # Fully connected exact simple cycle validation without subtours (feasible solution found!)
            if model._logger is not None:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(obj)

def main():
    parser = argparse.ArgumentParser(description="Solver for the Traveling Purchaser Problem variant.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the output JSON will be saved.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    
    args = parser.parse_args()
    
    # Init Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Load instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    num_vertices = data["num_vertices"]
    depot = data["depot"]
    product_demands = data["product_demands"]
    
    V = list(range(num_vertices))
    M = list(range(len(product_demands)))
    
    # Process edges / travel costs safely enforcing lower_idx < upper_idx
    travel_costs = {}
    for key, cost in data["travel_costs"].items():
        i, j = map(int, key.split(","))
        if i > j:
            i, j = j, i
        travel_costs[(i, j)] = cost
        
    # Process products availability at markets
    product_markets_int = {int(p_str): [int(m) for m in m_list] for p_str, m_list in data["product_markets"].items()}
    
    # Process Purchase properties 
    purchase_costs = {}
    for key, cost in data["purchase_costs"].items():
        p_str, m_str = key.split(",")
        purchase_costs[(int(p_str), int(m_str))] = cost
        
    supplies = {}
    for key, sup in data["supplies"].items():
        p_str, m_str = key.split(",")
        supplies[(int(p_str), int(m_str))] = sup
        
    q_keys = list(purchase_costs.keys())
    x_keys = list(travel_costs.keys())
    
    # Model Setup
    model = gp.Model("Purchaser_RoundTrip")
    
    # Set model parameters compliant to regulations
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1) 
    model.setParam('LazyConstraints', 1)
    
    # Define Decision Variables
    y = model.addVars(V, vtype=GRB.BINARY, name="y")
    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")
    q = model.addVars(q_keys, vtype=GRB.CONTINUOUS, lb=0, name="q")
    
    # Include Object References natively in logic flow limits over MIPSOL callbacks
    model._x = x
    model._y = y
    model._V = V
    model._E = x_keys
    model._depot = depot
    model._logger = logger
    
    # Constraints
    # 1. Depot strictly visited implicitly bounds minimal cyclic depth configuration
    model.addConstr(y[depot] == 1, name="depot_visited")
    
    # 2. Visited vertices MUST structurally complete precisely 2 edges in the network
    for i in V:
        edges_i = []
        for j in V:
            if i == j: 
                continue
            e = (i, j) if i < j else (j, i)
            edges_i.append(x[e])
        model.addConstr(gp.quicksum(edges_i) == 2 * y[i], name=f"degree_{i}")
        
    # 3. Supply limits dynamically conditioned by market visitation rules
    for p, k in q_keys:
        model.addConstr(q[p, k] <= supplies[p, k] * y[k], name=f"supply_limit_{p}_{k}")
        
    # 4. Strict Exact demand aggregation per product index requirement guarantees execution validity 
    for p in M:
        model.addConstr(
            gp.quicksum(q[p, k] for k in product_markets_int[p]) == product_demands[p], 
            name=f"demand_exact_{p}"
        )
        
    # Pre-solve Mandatory Constraints Bounds Deduction enhancing fractional node relaxations 
    for p in M:
        dem = product_demands[p]
        total_sup = sum(supplies[(p, k)] for k in product_markets_int[p])
        for k in product_markets_int[p]:
            min_amount = dem - (total_sup - supplies[(p, k)])
            if min_amount > 0:
                q[p, k].lb = min_amount
                y[k].lb = 1
            q[p, k].ub = min(dem, supplies[(p, k)])
            
    # Formulate Primary Optimizable Objective Metrics Function Cost
    objective = gp.quicksum(travel_costs[e] * x[e] for e in x_keys) + \
                gp.quicksum(purchase_costs[p, k] * q[p, k] for p, k in q_keys)
                
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Execution Procedure triggering constraints handler 
    model.optimize(subtour_elimination)
    
    # Outcome Extract parsing schema mappings
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        visited_nodes = [i for i in V if y[i].X > 0.5]
        visited_markets = [i for i in visited_nodes if i != depot]
        
        active_edges = [e for e in x_keys if x[e].X > 0.5]
        tour_edges_out = [[int(u), int(v)] for u, v in active_edges]
        
        # Sequentially orchestrating the routing map topological dependencies loop 
        adj = {i: [] for i in V}
        for u, v in active_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        tour = [depot]
        curr = depot
        prev = None
        
        while True:
            neighbors = adj[curr]
            next_node = None
            for nbc in neighbors:
                if nbc != prev:
                    next_node = nbc
                    break
            
            if next_node is None: break
                
            tour.append(next_node)
            prev = curr
            curr = next_node
            
            if curr == depot:
                break
                
        # Consolidate Quantifiable Unit Procurements mapped to string tokens definitions
        purchases_out = {str(p): {} for p in M}
        for p, k in q_keys:
            val = q[p, k].X
            val_int = int(round(val))
            if val_int > 0:
                purchases_out[str(p)][str(k)] = val_int

        solution = {
            "objective_value": float(obj_val),
            "visited_markets": visited_markets,
            "tour": tour,
            "tour_edges": tour_edges_out,
            "purchases": purchases_out
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution could be formulated within operational capacity constraints.")
        sys.exit(0)

if __name__ == "__main__":
    main()