import argparse
import json
import math
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB

def solve_wind_farm(instance_data: Dict, time_limit: int):
    """
    Solve the wind farm optimization problem.
    """
    # Extract parameters
    nodes = instance_data['nodes']
    edges = instance_data['edges'] 
    quota = instance_data['parameters']['quota']
    alpha = instance_data['parameters']['alpha']
    
    # Create nodemap for quick access
    id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # Separate node types
    turbine_nodes = [i for i, node in enumerate(nodes) if node['type'] == 'potential_turbine']
    substation_nodes = [i for i, node in enumerate(nodes) if node['type'] == 'substation']
    steiner_nodes = [i for i, node in enumerate(nodes) if node['type'] == 'steiner']
    
    # Build adjacency list for edges
    adj_list = [[] for _ in range(n)]
    edge_map = {}
    for e_idx, edge in enumerate(edges):
        u = id_to_idx[edge['from']]
        v = id_to_idx[edge['to']]
        adj_list[u].append((v, e_idx))
        adj_list[v].append((u, e_idx))
        edge_map[(min(u,v), max(u,v))] = e_idx
    
    # Initialize model
    model = gp.Model("windfarm")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # Variables
    y = {}  # 1 if turbine is built at turbine node i
    for i in turbine_nodes:
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        
    x = {}  # 1 if edge e is selected
    for e_idx in range(len(edges)):
        x[e_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{e_idx}")
        
    f = {}  # flow on each edge (for connectivity via network flow)
    for e_idx in range(len(edges)):
        f[e_idx] = model.addVars(2, lb=0, vtype=GRB.CONTINUOUS, name=f"f_{e_idx}")  # Forward and backward flow

    # Add constraints
    
    # 1. Energy quota constraint
    model.addConstr(gp.quicksum(nodes[i]['profit'] * y[i] for i in turbine_nodes) >= quota, name="energy_quota")

    # 2. Connectivity: Ensure substation nodes are connected and all turbines are connected to some substation
    # Use the spanning tree constraint approach with network flows
    # We ensure that there's a flow from a virtual root to all nodes in the solution
    
    # For simplicity, pick first substation as root
    if len(substation_nodes) > 0:
        root = substation_nodes[0]
        
        # Flow balance constraints (excluding root)
        for i in range(n):
            if i == root:
                # Root supplies flow to all nodes in solution
                outflow = sum(f[edge_idx][direction] for v, edge_idx in adj_list[i] for direction in [0, 1])
                total_selected_nodes = sum(y[j] for j in turbine_nodes) + len(substation_nodes) + \
                                       sum(x[edge_idx] for v, edge_idx in adj_list[k] for k in range(n) for direction in [0, 1])
                
                # This is a simplification: we'll enforce total flow = n'
                # where n' is number of selected turbines + substations + enough steiners
                # A more accurate approach would be to have flow from root to active nodes and enforce connectivity
                # Let's reformulate: supply = n-1 units, demand = 1 unit per node in spanning tree
                # Sum of supplies = n-1, thus total flow in = n-1
                flow_sum_expr = gp.LinExpr()
                for u in range(n):
                    for v, edge_idx in adj_list[u]:
                        flow_sum_expr.addTerms([1], [f[edge_idx][0]])
                        flow_sum_expr.addTerms([1], [f[edge_idx][1]])
                
                n_prime = len(substation_nodes) + sum(1 for i in turbine_nodes)  # Upper bound on needed nodes
                model.addConstr(flow_sum_expr <= n_prime, name="flow_root_supply")
                
            else:
                # Non-root node demand constraint: net inbound flow = 1 if the node is selected to be part of the solution
                net_in_flow = gp.LinExpr()
                for u, edge_idx in adj_list[i]:
                    # Flow from u to i contributes negatively (outflow)
                    # Flow from i to u contributes positively (inflow)
                    if id_to_idx[edges[edge_idx]['from']] == i:  # edge is i -> v
                        net_in_flow.addTerms([1], [f[edge_idx][1]])  # incoming to i
                        net_in_flow.addTerms([-1], [f[edge_idx][0]])  # outgoing from i
                    else:  # edge is u -> i
                        net_in_flow.addTerms([1], [f[edge_idx][0]])  # incoming to i
                        net_in_flow.addTerms([-1], [f[edge_idx][1]])  # outgoing from i
                
                # The node participates if it's either a substation or a selected turbine, 
                # or it's a Steiner node that appears as an intermediate in our tree
                # A Steiner node participates = is selected in x => has flow through it
                is_active = 0
                if i in turbine_nodes:
                    is_active = y[i]
                elif i in substation_nodes:
                    is_active = 1  # Always participates
                else:  # Steiner node
                    # Participates if at least one edge incident to it is selected
                    expr = gp.quicksum(x[e_idx] for _, e_idx in adj_list[i])
                    is_active = model.addVar(vtype=GRB.BINARY, name=f"is_active_steiner_{i}")
                    
                    # Link: is_active implies some edge incident to it is used
                    for _, e_idx in adj_list[i]:
                        model.addConstr(is_active <= x[e_idx])
                    
                    # Link: if any edge incident is used, then it's active (approximation)
                    big_m = len(adj_list[i])
                    model.addConstr(big_m * is_active >= gp.quicksum(x[e_idx] for _, e_idx in adj_list[i]))
                
                model.addConstr(net_in_flow == is_active, name=f"flow_balance_node_{i}")

        
        # Force edge selection for active Steiner nodes to be consistent with flow paths
        # When an edge is used (x=1), it must have flow
        for e_idx in range(len(edges)):
            u = id_to_idx[edges[e_idx]['from']]
            v = id_to_idx[edges[e_idx]['to']]
            
            # If edge used then some flow should go through it
            # This creates a lower bound on flow if x is active
            big_M_flow = n
            model.addConstr(f[e_idx][0] + f[e_idx][1] <= big_M_flow * x[e_idx], name=f"edge_implies_flow_{e_idx}")

    # 3. Ensure tree property: |selected_edges| = |selected_nodes| - 1
    # This is harder to express directly in MIP without knowing selected set
    # Instead we rely on flow-based connectivity, which ensures connectivity
    total_nodes_in_tree = len(substation_nodes) + gp.quicksum(y[t] for t in turbine_nodes) + \
                           gp.quicksum(x[edge_idx] for edge_idx in range(len(edges)))
    
    total_edges_used = gp.quicksum(x[e_idx] for e_idx in range(len(edges)))
    # This is not a tight formulation but flow + ensuring |E|=|V|-1 for tree
    
    # To enforce tree |E| = |V| - 1, add constraint
    # Number of selected edges = number of selected nodes - 1
    num_nodes_formula = len(substation_nodes) + gp.quicksum(y[i] for i in turbine_nodes) + \
                       gp.quicksum(
                           x[e_idx] for i in steiner_nodes 
                           for _, e_idx in adj_list[i] if sum(nodes[k]['scenic_impact'] > 0 for k in [id_to_idx[edges[e_idx]['from']], id_to_idx[edges[e_idx]['to']]]) == 2
                       )
    # That above doesn't work well for Steiner. Better to estimate: let's just ensure connectivity and minimal spanning
    
    # For simplicity of formulation, just enforce basic degree constraints
    # At least ensure all substations are connected
    # Connectivity enforced with flow constraints above
    
    # Objective function
    total_installation_cost = gp.quicksum((nodes[i]['cost'] if i < len(nodes) else 0) * y[i] for i in turbine_nodes)
    total_cable_cost = gp.quicksum(edges[e_idx]['cost'] * x[e_idx] for e_idx in range(len(edges)))

    total_energy_scenic = gp.quicksum(nodes[i]['scenic_impact'] * y[i] for i in turbine_nodes)
    total_cable_scenic = gp.quicksum(edges[e_idx]['scenic_impact'] * x[e_idx] for e_idx in range(len(edges)))

    weighted_obj = alpha * (total_installation_cost + total_cable_cost) + \
                  (1 - alpha) * (total_energy_scenic + total_cable_scenic)

    model.setObjective(weighted_obj, GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_turbines = [nodes[i]['id'] for i in turbine_nodes if y[i].x > 0.5]
        selected_edge_ids = [edges[e_idx]['id'] for e_idx in range(len(edges)) if x[e_idx].x > 0.5]

        solution = {
            "objective_value": model.objVal,
            "selected_turbines": selected_turbines,
            "selected_edges": [{"from": edges[e_idx]['from'], "to": edges[e_idx]['to']} 
                              for e_idx in range(len(edges)) if x[e_idx].x > 0.5]
        }
        
        # Return best solution found even if timeout
        if model.SolCount > 0:
            solution["objective_value"] = model.objVal
            return solution
    
    return {"objective_value": float('inf'), "selected_turbines": [], "selected_edges": []}


def solve_wind_farm_reformulated(instance_data: Dict, time_limit: int):
    """
    Reformulate the problem using a more standard spanning tree approach with degree-like constraints.
    """
    # Extract parameters
    nodes = instance_data['nodes']
    edges = instance_data['edges'] 
    quota = instance_data['parameters']['quota']
    alpha = instance_data['parameters']['alpha']
    
    # Create nodemap
    id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # Separate node types
    turbine_nodes = [i for i, node in enumerate(nodes) if node['type'] == 'potential_turbine']
    substation_nodes = [i for i, node in enumerate(nodes) if node['type'] == 'substation']
    steiner_nodes = [i for i, node in enumerate(nodes) if node['type'] == 'steiner']
    
    # Build adjacency list for edges
    adj_list = [[] for _ in range(n)]
    for e_idx, edge in enumerate(edges):
        u = id_to_idx[edge['from']]
        v = id_to_idx[edge['to']]
        adj_list[u].append((v, e_idx))
        adj_list[v].append((u, e_idx))

    # Create Gurobi model
    model = gp.Model("WindFarm_STP")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)

    # Binary variables for turbines
    y = model.addVars(turbine_nodes, vtype=GRB.BINARY, name="build_turbine")
    
    # Binary variables for edges
    x = model.addVars(len(edges), vtype=GRB.BINARY, name="select_edge")
    
    # Binary variables to indicate if steiner nodes are used in the tree
    z = model.addVars(steiner_nodes, vtype=GRB.BINARY, name="use_steiner")

    # Constraints:
    # 1. Meet energy quota
    model.addConstr(gp.quicksum(nodes[i]['profit'] * y[i] for i in turbine_nodes) >= quota, name="energy_quota")

    # 2. If a Steiner node is in the tree, at least 2 edges incident must be in the tree
    # Actually, we need to connect the tree properly. Better to use flow model again.
    
    # Network flow model for connectivity
    # We treat the solution as a tree connecting all needed substation nodes and selected turbines.
    # Root node will supply flow to all others, total flow is #nodes in tree - 1
    
    total_possible_nodes = len(substation_nodes) + len(turbine_nodes)
    
    # Flow variables: for each edge, forward and backward flow
    f = model.addVars(range(len(edges)), 2, lb=0, vtype=GRB.CONTINUOUS, name="flow")
    
    # Pick root node (first substation)
    root_node = substation_nodes[0] if substation_nodes else (turbine_nodes[0] if turbine_nodes else steiner_nodes[0])

    # Calculate which nodes are active in solution
    selected_nodes = []
    for i in substation_nodes:
        selected_nodes.append(i)
    for i in turbine_nodes:
        selected_nodes.append(i)
    for i in steiner_nodes:
        selected_nodes.append(i)
        
    # For each non-root node, add flow conservation
    for i in range(n):
        if i != root_node:
            # Determine if node i must be in solution:
            # - If it's a substation: always yes
            # - If it's a turbine: only if built
            # - If it's a Steiner node: depends on tree structure
            
            net_flow_expr = gp.LinExpr()
            for v, edge_idx in adj_list[i]:
                # If the edge is from i to v, we look at flow i->v (forward: 0) and v->i (backward: 1)
                if id_to_idx[edges[edge_idx]['from']] == i:  # This means we have edge i -> v
                    net_flow_expr.add(f[edge_idx, 1])  # Flow coming into i
                    net_flow_expr.add(-f[edge_idx, 0])  # Flow going out of i
                else:  # edge is v -> i
                    net_flow_expr.add(f[edge_idx, 0])  # Flow coming into i  
                    net_flow_expr.add(-f[edge_idx, 1])  # Flow going out of i

            # RHS = supply/demand of node i
            # Supply: -1 for root (generates flow), +1 for leaf nodes (consumes 1 unit), 0 for intermediates
            # In any tree: root gets -(n'-1), all others get +1 (if present in tree)
            if i in substation_nodes:
                rhs_var = model.addVar(vtype=GRB.BINARY, name=f"is_in_tree_sub_{i}")
                model.addConstr(rhs_var == 1)  # Always satisfied  
                model.addConstr(net_flow_expr == rhs_var, name=f"flow_cons_sub_{i}")
            elif i in turbine_nodes:
                model.addConstr(net_flow_expr == y[i], name=f"flow_cons_turb_{i}")
            elif i in steiner_nodes:
                # If this steiner node is used in tree (selected path), requires flow conservation
                # z_i tells us whether this steiner node is active in tree
                model.addConstr(net_flow_expr == z[i], name=f"flow_cons_stein_{i}")
                
                # If any edge incident to this steiner is used, then z[i] should be 1
                for _, edge_idx in adj_list[i]:
                    model.addConstr(z[i] >= x[edge_idx])
                
                # Conversely, if z[i] = 1, ensure some x[edge] = 1
                model.addConstr(z[i] * len(adj_list[i]) >= gp.quicksum(x[e_idx] for _, e_idx in adj_list[i]))

    # Root node: provides -(total_selected - 1) units of flow
    total_selected = len(substation_nodes) + gp.quicksum(y[i] for i in turbine_nodes) + gp.quicksum(z[i] for i in steiner_nodes)
    root_supply = total_selected - 1  # -ve flow balance since root supplies
    
    root_flow_expr = gp.LinExpr()
    for v, edge_idx in adj_list[root_node]:
        # If edge is root -> v, contribute f[edge_idx, 0] to root outflow
        # If edge is v -> root, contribute f[edge_idx, 1] to root outflow
        if id_to_idx[edges[edge_idx]['from']] == root_node:  # root -> v
            root_flow_expr.add(f[edge_idx, 0])
            root_flow_expr.add(-f[edge_idx, 1])
        else:  # v -> root
            root_flow_expr.add(f[edge_idx, 1])
            root_flow_expr.add(-f[edge_idx, 0])
            
    model.addConstr(root_flow_expr == -root_supply, name="root_flow")

    # Link selected edges to flow: if edge selected, there's flow
    M_flow_upper = len(selected_nodes)  # Big-M upper bound 
    for e_idx in range(len(edges)):
        u = id_to_idx[edges[e_idx]['from']]
        v = id_to_idx[edges[e_idx]['to']]
        
        # if x[e_idx]=1 then f[e_idx,0] or f[e_idx,1] can be positive
        # If x[e_idx] = 0 then both flow vars must be 0
        model.addConstr(f[e_idx, 0] <= M_flow_upper * x[e_idx], name=f"f0_up_{e_idx}")
        model.addConstr(f[e_idx, 1] <= M_flow_upper * x[e_idx], name=f"f1_up_{e_idx}")

    # Objective
    obj = (
        # Costs
        alpha * (gp.quicksum(nodes[i]['cost'] * y[i] for i in turbine_nodes) +
                 gp.quicksum(edges[e_idx]['cost'] * x[e_idx] for e_idx in range(len(edges)))) +
        # Scenic impacts  
        (1 - alpha) * (gp.quicksum(nodes[i]['scenic_impact'] * y[i] for i in turbine_nodes) +
                       gp.quicksum(edges[e_idx]['scenic_impact'] * x[e_idx] for e_idx in range(len(edges))))
    )

    model.setObjective(obj, GRB.MINIMIZE)
    
    model.optimize()

    # Handle results:
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        solution = {
            "objective_value": model.objVal,
            "selected_turbines": [nodes[i]['id'] for i in turbine_nodes if y[i].x > 0.5],
            "selected_edges": [
                {"from": edges[e_idx]['from'], "to": edges[e_idx]['to']}
                for e_idx in range(len(edges)) if x[e_idx].x > 0.5
            ]
        }
        return solution
        
    return {"objective_value": float('inf'), "selected_turbines": [], "selected_edges": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None
        
    # Run solver
    solution = solve_wind_farm_reformulated(instance_data, args.time_limit)
    
    # Log if we have a valid solution
    if logger and solution["objective_value"] != float('inf'):
        logger.log(solution["objective_value"])

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()