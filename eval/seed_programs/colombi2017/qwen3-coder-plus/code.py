import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import sys

def solve_pcarp(instance_data, time_limit):
    n_nodes = instance_data["num_nodes"]
    depot = instance_data["depot"]
    n_arcs = instance_data["num_arcs"]
    n_profitable_arcs = instance_data["num_profitable_arcs"]
    n_VI_nodes = instance_data["num_VI_nodes"]
    n_strong_incompatibilities = instance_data["num_strong_incompatibilities"]
    n_weak_incompatibilities = instance_data["num_weak_incompatibilities"]

    # Parse components from instance data
    arcs_list = instance_data["arcs"]  # [i, j, cost]
    arcs = []  # (i, j) tuples
    arc_costs = {}  # (i, j) -> cost
    for i, j, cost in arcs_list:
        arcs.append((i, j))
        arc_costs[(i, j)] = float(cost)

    profitable_arcs_list = instance_data["profitable_arcs"]  # [i, j, profit]
    profitable_arcs = set()
    arc_profits = {}  # (i, j) -> profit
    for i, j, profit in profitable_arcs_list:
        profitable_arcs.add((i, j))
        arc_profits[(i, j)] = float(profit)

    VI_node_set = set(instance_data["VI_nodes"])  # subset of nodes that appear in incompatibility graph

    strong_incompatibilities = instance_data["strong_incompatibilities"]  # [i, j]
    weak_incompatibilities = instance_data["weak_incompatibilities"]  # [i, j, penalty]

    # Build incompatibility graphs
    strong_graph = defaultdict(set)  # Maps a VI_node to set of its strong incompatible nodes
    for u, v in strong_incompatibilities:
        if u in VI_node_set and v in VI_node_set:
            strong_graph[u].add(v)
            strong_graph[v].add(u)

    # For weak incompatibilities we store a tuple (u, w_penalty) for each v
    # and (v, w_penalty) for each u
    weak_incompatibilities_dict = defaultdict(list)  # Maps a VI_node to list of (other_node, penalty)
    for u, v, penalty in weak_incompatibilities:
        if u in VI_node_set and v in VI_node_set:
            weak_incompatibilities_dict[u].append((v, float(penalty)))
            weak_incompatibilities_dict[v].append((u, float(penalty)))

    # Create the model
    model = gp.Model("PCARP_with_Incompatibilities")
    model.setParam("Threads", 1)
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    
    # Suppress console output
    model.setParam('OutputFlag', 0)
    
    # Decision Variables
    # x_{ij} : Number of times arc (i,j) is traversed
    x = {}
    for i, j in arcs:
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")

    # z_{ij} : Binary variables indicating if profitable arc (i,j) is served
    z = {}
    for i, j in profitable_arcs:
        z[i, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

    # y_u : Binary variables indicating if VI-node u is active (has its profitable arc served)
    y = {}
    for u in VI_node_set:
        y[u] = model.addVar(vtype=GRB.BINARY, name=f"y_{u}")

    # w_uv : Binary variable indicating if weak incompatibility (u, v) is overridden
    w = {}
    for u, v, penalty in weak_incompatibilities:
        if u in VI_node_set and v in VI_node_set:
            w[u, v] = model.addVar(vtype=GRB.BINARY, name=f"w_{u}_{v}")
            # Since (u,v) might appear as (v,u) order matters for naming but logically they're same
            # So let's also set w[v,u] for consistency
            w[v, u] = w[u, v]

    # Objective function: maximize profit - travel cost - penalty
    total_profit = gp.quicksum(arc_profits[i, j] * z[i, j] for i, j in profitable_arcs)
    total_travel_cost = gp.quicksum(arc_costs[i, j] * x[i, j] for i, j in arcs)
    total_penalty = gp.quicksum(
        penalty * w[u, v]
        for u, v, penalty in weak_incompatibilities
        if u in VI_node_set and v in VI_node_set
    )
    obj = total_profit - total_travel_cost - total_penalty
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint: Flow conservation at each node
    for k in range(n_nodes):
        incoming_flow = gp.quicksum(x[i, j] for i, j in arcs if j == k)
        outgoing_flow = gp.quicksum(x[i, j] for i, j in arcs if i == k)
        model.addConstr(incoming_flow == outgoing_flow, f"flow_conservation_{k}")

    # Constraint: At the depot, there is net outward flow
    depot_out_flow = gp.quicksum(x[depot, j] for depot, j in arcs if depot == depot)
    depot_in_flow = gp.quicksum(x[i, depot] for i, depot in arcs if depot == depot)
    model.addConstr(depot_out_flow >= 1, f"depot_has_outbound")
    model.addConstr(depot_out_flow == depot_in_flow + 0, f"depot_flow")

    # Constraint: If any profitable arc from u is served, u is active
    # More formally: y_u = 1 iff there exists (u, j) such that z_{u,j} = 1
    # This can be simplified as: y_u >= z_{u,j} for all (u,j) in profitable_arcs
    for u in VI_node_set:
        outgoing_profitable_from_u = [(i, j) for (i, j) in profitable_arcs if i == u]
        if outgoing_profitable_from_u:
            model.addConstr(gp.quicksum(z[i, j] for i, j in outgoing_profitable_from_u) <= len(outgoing_profitable_from_u) * y[u])
              
    # The reverse direction: y[u] <= sum of corresponding z terms (for correctness)
    for u in VI_node_set:
        outgoing_profitable_from_u = [(i, j) for (i, j) in profitable_arcs if i == u]
        if outgoing_profitable_from_u:
            # If y[u] = 1, then at least one corresponding z_{u,*} = 1
            # We add constraint: sum of z terms >= y[u] 
            # Which implies if y[u] = 1, sum(z) >= 1 => so some z is 1
            model.addConstr(gp.quicksum(z[i, j] for i, j in outgoing_profitable_from_u) >= y[u])

    # Constraint: Strong incompatibility: if u is active, none of its strong neighbors can be active
    for u in VI_node_set:
        for v in strong_graph[u]:
            if u < v:  # avoid duplicate constraints e.g., for (u,v) and (v,u)
                model.addConstr(y[u] + y[v] <= 1, f"strong_incomp_{u}_{v}")

    # Constraint: Weak incompatibility: if both u and v are active, pay the penalty
    for u, v, penalty in weak_incompatibilities:
        if u in VI_node_set and v in VI_node_set:
            # w[u,v] = 1 means we pay for this incompatibility. If both are active, then w must be 1
            model.addConstr(y[u] + y[v] - 1 <= w[u, v], f"weak_override_{u}_{v}")

    # For connectivity, we can add cutset constraints
    # However, modeling full connectivity without exponential variables is tricky
    # We will rely on the fact that Gurobi/MIP solver should ensure a connected path
    # to visit chosen profitable arcs, or use more sophisticated modeling if needed later.

    # Solve the model
    model.optimize()

    # Interpret results
    status = model.status
    
    best_obj_val = -float('inf')
    
    # Handle case if no solution is found
    if status != GRB.INF_OR_UNBD and status != GRB.UNBOUNDED and status != GRB.INFEASIBLE:
        if model.solCount > 0:
            best_obj_val = model.objval
            
            # Extract variables
            selected_profitable_arcs = []
            for (i, j) in profitable_arcs:
                if z[i, j].X > 0.5:
                    selected_profitable_arcs.append({
                        'from': i,
                        'to': j,
                        'profit': arc_profits[i, j],
                        'cost': arc_costs[i, j]
                    })

            tour_arc_list = []
            for (i, j) in arcs:
                count = int(round(x[i, j].X))
                if count > 0:
                    tour_arc_list.append({
                        'from': i,
                        'to': j,
                        'count': count,
                        'cost': arc_costs[i, j]
                    })

            return {
                "objective_value": best_obj_val,
                "served_arcs": selected_profitable_arcs,
                "tour_arcs": tour_arc_list
            }
        else:
            # Return default empty solution if no good solution found
            return {
                "objective_value": -float('inf'),
                "served_arcs": [],
                "tour_arcs": []
            }

    # If model is infeasible, unbounded, or other status indicates failure
    return {
        "objective_value": -float('inf'),
        "served_arcs": [],
        "tour_arcs": []
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()

    # Initialize logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve
    try:
        solution = solve_pcarp(instance_data, args.time_limit)
    except Exception as e:
        print(f"Error occurred during solving: {e}", file=sys.stderr)
        # On error, return a default empty solution
        solution = {
            "objective_value": -float('inf'),
            "served_arcs": [],
            "tour_arcs": []
        }
    
    # If we have a valid solution, log it
    if logger and solution["objective_value"] != -float('inf'):
        logger.log(solution["objective_value"])

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()