import argparse
import json
import sys

# Try to import SolutionLogger
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser(description="Solve Unsplittable Multi-Commodity Network Flow Problem.")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the JSONL logging file.")

    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    network = instance['network']
    commodities_data = instance['commodities']
    commodities = commodities_data['commodity_list']

    nodes = network['nodes']
    arcs = network['arcs']

    arc_dict = {a['arc_id']: a for a in arcs}

    # Initialize model
    m = gp.Model("Telecomm_Routing")
    
    # Configure model parameters
    m.setParam('Threads', 1)
    if args.time_limit is not None:
        m.setParam('TimeLimit', args.time_limit)
    
    # Focus on finding feasible solutions rather than proving optimality
    m.setParam('MIPFocus', 1)

    # Decision variables
    # reject[k] = 1 if commodity k is rejected, 0 otherwise
    reject = {}
    for c in commodities:
        k = c['commodity_id']
        reject[k] = m.addVar(vtype=GRB.BINARY, name=f"reject_{k}")

    # route[k, a_id] = 1 if commodity k uses arc a_id
    route = {}
    for c in commodities:
        k = c['commodity_id']
        for a in arcs:
            route[k, a['arc_id']] = m.addVar(vtype=GRB.BINARY, name=f"route_{k}_{a['arc_id']}")

    # Objective: Minimize cost of rejected commodities
    obj_expr = gp.LinExpr()
    for c in commodities:
        k = c['commodity_id']
        obj_expr.addTerms(c['artificial_arc_cost'], reject[k])
        
    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints 1: Flow Conservation
    for c in commodities:
        k = c['commodity_id']
        orig = c['origin']
        dest = c['destination']
        
        # Build flow expressions for each node for this commodity
        node_flow = {n: gp.LinExpr() for n in nodes}
        for a in arcs:
            u = a['from_node']
            v = a['to_node']
            a_id = a['arc_id']
            # Flow out from u (+) and into v (-)
            var = route[k, a_id]
            node_flow[u].addTerms(1.0, var)
            node_flow[v].addTerms(-1.0, var)
            
        # Add constraints
        for n in nodes:
            if n == orig and n == dest:
                m.addConstr(node_flow[n] == 0, name=f"flow_{k}_{n}")
            elif n == orig:
                m.addConstr(node_flow[n] == 1 - reject[k], name=f"flow_{k}_{n}")
            elif n == dest:
                m.addConstr(node_flow[n] == reject[k] - 1, name=f"flow_{k}_{n}")
            else:
                m.addConstr(node_flow[n] == 0, name=f"flow_{k}_{n}")

    # Constraints 2: Arc Capacities
    for a in arcs:
        a_id = a['arc_id']
        cap = a['capacity']
        usage_expr = gp.LinExpr()
        for c in commodities:
            k = c['commodity_id']
            demand = c['demand']
            usage_expr.addTerms(demand, route[k, a_id])
        m.addConstr(usage_expr <= cap, name=f"cap_{a_id}")

    # Set up solution logger
    logger = None
    if args.log_path and SolutionLogger is not None:
        # Objective direction is "minimize"
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Callback for logging
    def optimization_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objf = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objf)

    # Solve the model
    m.optimize(optimization_callback)

    # Check available solutions
    if m.SolCount == 0:
        print("No feasible solution found.", file=sys.stderr)
        sys.exit(1)

    # Prepare output
    sol_commodities = []
    
    for c in commodities:
        k = c['commodity_id']
        is_rejected = reject[k].X > 0.5
        
        if is_rejected:
            sol_commodities.append({
                "commodity_id": k,
                "rejected": True,
                "path_arcs": []
            })
        else:
            orig = c['origin']
            dest = c['destination']
            
            # Extract chosen arcs to build an adjacency list
            adj = {}
            for a in arcs:
                a_id = a['arc_id']
                if route[k, a_id].X > 0.5:
                    u = a['from_node']
                    if u not in adj:
                        adj[u] = []
                    adj[u].append((a['to_node'], a_id))
                    
            path_arc_ids = []
            if orig != dest:
                visited = set()
                
                # DFS to reconstruct the route path
                def dfs(u):
                    if u == dest:
                        return True
                    visited.add(u)
                    for v, arc_id in adj.get(u, []):
                        if v not in visited:
                            path_arc_ids.append(arc_id)
                            if dfs(v):
                                return True
                            path_arc_ids.pop()
                    return False
                    
                dfs(orig)
            
            # Format path arcs
            formatted_path = []
            for arc_id in path_arc_ids:
                a = arc_dict[arc_id]
                formatted_path.append({
                    "from": a['from_node'],
                    "to": a['to_node'],
                    "arc_id": arc_id
                })
                
            sol_commodities.append({
                "commodity_id": k,
                "rejected": False,
                "path_arcs": formatted_path
            })

    output = {
        "objective_value": m.ObjVal,
        "commodities": sol_commodities
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()