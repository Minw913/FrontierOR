import os
import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Solve Net Profit Maximization Hauler Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path for the logging JSONL file")
    args = parser.parse_args()

    # Initialize Logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None

    # Load Instance Data
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_nodes = data["num_nodes"]
    depot = data["depot"]
    num_arcs = data["num_arcs"]
    num_profitable_arcs = data["num_profitable_arcs"]
    arcs = data["arcs"]
    profitable_arcs = data["profitable_arcs"]
    VI_nodes = data.get("VI_nodes", [])
    strong_incompatibilities = data.get("strong_incompatibilities", [])
    weak_incompatibilities = data.get("weak_incompatibilities", [])

    # Initialize Gurobi Model
    model = gp.Model("HaulerTour")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("LazyConstraints", 1)
    # Give solver more focus on finding good feasible solutions if needed
    model.setParam("MIPFocus", 1)

    # Base Variables
    model._x = {}
    for a_idx in range(num_arcs):
        model._x[a_idx] = model.addVar(
            vtype=GRB.INTEGER, lb=0, name=f"x_{arcs[a_idx][0]}_{arcs[a_idx][1]}"
        )
        model._x[a_idx].Start = 0  # Initial Empty Tour

    model._y = {}
    for p_idx in range(num_profitable_arcs):
        model._y[p_idx] = model.addVar(
            vtype=GRB.BINARY, name=f"y_{profitable_arcs[p_idx][0]}_{profitable_arcs[p_idx][1]}"
        )
        model._y[p_idx].Start = 0  # Initial Empty Tour

    z = {}
    for u in VI_nodes:
        z[u] = model.addVar(vtype=GRB.BINARY, name=f"z_{u}")
        z[u].Start = 0

    w = {}
    for idx_w, (u, v, penalty) in enumerate(weak_incompatibilities):
        w[idx_w] = model.addVar(vtype=GRB.BINARY, name=f"w_{u}_{v}")
        w[idx_w].Start = 0

    # 1. Flow Conservation Constraints
    in_arcs = {i: [] for i in range(num_nodes)}
    out_arcs = {i: [] for i in range(num_nodes)}
    for a_idx, a in enumerate(arcs):
        out_arcs[a[0]].append(a_idx)
        in_arcs[a[1]].append(a_idx)

    for i in range(num_nodes):
        model.addConstr(
            gp.quicksum(model._x[a_idx] for a_idx in in_arcs[i]) == 
            gp.quicksum(model._x[a_idx] for a_idx in out_arcs[i]),
            name=f"flow_{i}"
        )

    # 2. Profitable arcs can only be served if traversed at least once
    arc_dict = {(a[0], a[1]): idx for idx, a in enumerate(arcs)}
    model._p_to_a = {}
    for p_idx, p in enumerate(profitable_arcs):
        a_idx = arc_dict[(p[0], p[1])]
        model._p_to_a[p_idx] = a_idx
        model.addConstr(model._x[a_idx] >= model._y[p_idx], name=f"serve_{p_idx}")

    # 3. Active nodes definitions for Incompatibilities
    for p_idx, p in enumerate(profitable_arcs):
        u = p[0]
        if u in z:
            model.addConstr(z[u] >= model._y[p_idx], name=f"active_{u}_{p_idx}")

    # 4. Strong Incompatibilities
    for u, v in strong_incompatibilities:
        if u in z and v in z:
            model.addConstr(z[u] + z[v] <= 1, name=f"strong_{u}_{v}")

    # 5. Weak Incompatibilities
    for idx_w, (u, v, penalty) in enumerate(weak_incompatibilities):
        if u in z and v in z:
            model.addConstr(w[idx_w] >= z[u] + z[v] - 1, name=f"weak_{u}_{v}")

    # Objective: Maximize Profit
    obj = (
        gp.quicksum(profitable_arcs[p_idx][2] * model._y[p_idx] for p_idx in range(num_profitable_arcs))
        - gp.quicksum(arcs[a_idx][2] * model._x[a_idx] for a_idx in range(num_arcs))
        - gp.quicksum(weak_incompatibilities[w_idx][2] * w[w_idx] for w_idx in range(len(weak_incompatibilities)))
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # Heuristic Global Constraint: Force the depot to be active if ANY profit is collected
    for p_idx in range(num_profitable_arcs):
        if len(out_arcs[depot]) > 0:
            model.addConstr(
                gp.quicksum(model._x[a_idx] for a_idx in out_arcs[depot]) >= model._y[p_idx]
            )

    model._added_cuts = set()

    # Subtour Elimination Callback
    def subtour_elimination(model, where):
        if where == GRB.CB_MIPSOL:
            x_val = model.cbGetSolution(model._x)
            y_val = model.cbGetSolution(model._y)
            
            # Form undirected components representation
            adj_undir = {i: [] for i in range(num_nodes)}
            for a_idx, a in enumerate(arcs):
                if x_val[a_idx] > 0.5:
                    adj_undir[a[0]].append(a[1])
                    adj_undir[a[1]].append(a[0])
                    
            visited = set()
            queue = [depot]
            visited.add(depot)
            while queue:
                curr = queue.pop(0)
                for nxt in adj_undir[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
                        
            served_p_indices = [p_idx for p_idx in range(num_profitable_arcs) if y_val[p_idx] > 0.5]
            unconnected_served = [
                p_idx for p_idx in served_p_indices 
                if profitable_arcs[p_idx][0] not in visited or profitable_arcs[p_idx][1] not in visited
            ]
            
            node_to_comp = {}
            components = []
            for p_idx in unconnected_served:
                u, v = profitable_arcs[p_idx][0], profitable_arcs[p_idx][1]
                for node in (u, v):
                    if node not in visited and node not in node_to_comp:
                        comp = set()
                        q2 = [node]
                        comp.add(node)
                        while q2:
                            curr = q2.pop(0)
                            for nxt in adj_undir[curr]:
                                if nxt not in comp:
                                    comp.add(nxt)
                                    q2.append(nxt)
                        components.append(comp)
                        for c_node in comp:
                            node_to_comp[c_node] = len(components) - 1
            
            added_any_cut = False
            for comp in components:
                cut_arcs = [a_idx for a_idx, a in enumerate(arcs) if a[0] in comp and a[1] not in comp]
                comp_tup = tuple(sorted(list(comp)))
                for p2_idx in served_p_indices:
                    if profitable_arcs[p2_idx][0] in comp and profitable_arcs[p2_idx][1] in comp:
                        cut_id = (comp_tup, p2_idx)
                        if cut_id not in model._added_cuts:
                            model._added_cuts.add(cut_id)
                            model.cbLazy(
                                gp.quicksum(model._x[a_idx] for a_idx in cut_arcs) >= model._y[p2_idx]
                            )
                            added_any_cut = True
                            
            if not added_any_cut:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)

        elif where == GRB.CB_MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x_val = model.cbGetNodeRel(model._x)
            y_val = model.cbGetNodeRel(model._y)
            
            for threshold in [0.05, 0.25, 0.5, 0.75]:
                adj_undir = {i: [] for i in range(num_nodes)}
                for a_idx, a in enumerate(arcs):
                    if x_val[a_idx] > threshold:
                        adj_undir[a[0]].append(a[1])
                        adj_undir[a[1]].append(a[0])
                        
                visited = set()
                queue = [depot]
                visited.add(depot)
                while queue:
                    curr = queue.pop(0)
                    for nxt in adj_undir[curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
                            
                node_to_comp = {}
                components = []
                for i in range(num_nodes):
                    if i not in visited and i not in node_to_comp:
                        comp = set()
                        q2 = [i]
                        comp.add(i)
                        while q2:
                            curr = q2.pop(0)
                            for nxt in adj_undir[curr]:
                                if nxt not in comp:
                                    comp.add(nxt)
                                    q2.append(nxt)
                        components.append(comp)
                        for c_node in comp:
                            node_to_comp[c_node] = len(components) - 1
                            
                for comp in components:
                    cut_arcs = [a_idx for a_idx, a in enumerate(arcs) if a[0] in comp and a[1] not in comp]
                    cut_val = sum(x_val[a_idx] for a_idx in cut_arcs)
                    comp_tup = tuple(sorted(list(comp)))
                    
                    for p_idx, p in enumerate(profitable_arcs):
                        if p[0] in comp and p[1] in comp:
                            if cut_val < y_val[p_idx] - 1e-4:
                                cut_id = (comp_tup, p_idx)
                                if cut_id not in model._added_cuts:
                                    model._added_cuts.add(cut_id)
                                    model.cbLazy(
                                        gp.quicksum(model._x[a_idx] for a_idx in cut_arcs) >= model._y[p_idx]
                                    )

    # Optimiziation
    model.optimize(subtour_elimination)

    # Prepare Formatted Outuput
    if model.SolCount > 0:
        output = {
            "objective_value": float(model.ObjVal),
            "served_arcs": [],
            "tour_arcs": []
        }
        for p_idx, p in enumerate(profitable_arcs):
            if model._y[p_idx].X > 0.5:
                a_idx = model._p_to_a[p_idx]
                cost = arcs[a_idx][2]
                output["served_arcs"].append({
                    "from": p[0],
                    "to": p[1],
                    "profit": float(p[2]),
                    "cost": float(cost)
                })
                
        for a_idx, a in enumerate(arcs):
            count = int(round(model._x[a_idx].X))
            if count > 0:
                output["tour_arcs"].append({
                    "from": a[0],
                    "to": a[1],
                    "count": count,
                    "cost": float(a[2])
                })
    else:
        # Trivial empty tour is universally feasible with zero net profit
        output = {
            "objective_value": 0.0,
            "served_arcs": [],
            "tour_arcs": []
        }

    # Write the response back securely into the output specification
    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()