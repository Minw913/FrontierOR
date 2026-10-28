import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_weighted_max_clique(instance_data, time_limit, log_path):
    """
    Solves the weighted max clique problem using Gurobi.
    
    :param instance_data: Dictionary containing the graph instance data
    :param time_limit: Time limit in seconds
    :param log_path: Path to log file for intermediate solutions (or None)
    :return: A dictionary with solution data
    """
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Extract data
    n = instance_data["num_vertices"]
    vertices = instance_data["vertices"]
    edges = instance_data["edges"]
    
    # Create mapping from vertex_id to index 0..n-1
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build adjacency list representation - needed for constraints
    adj = {v: set() for v in vertices}
    edge_weights = {}
    edge_list_idx = []  # List of (i_idx, j_idx) tuples representing edges by their indices

    for edge in edges:
        u, v, w = edge["i"], edge["j"], edge["weight"]
        idx_u, idx_v = vertex_to_idx[u], vertex_to_idx[v]
        edge_list_idx.append((idx_u, idx_v))
        adj[u].add(v)
        adj[v].add(u)
        edge_weights[(u, v)] = w
        edge_weights[(v, u)] = w

    # Create Gurobi model
    model = gp.Model("WeightedMaxClique")

    # Turn off verbosity unless debugging
    model.setParam('OutputFlag', 0)

    # Set time limit
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    # x[i] = 1 if vertex i is included in the clique
    x = []
    for i in range(n):
        x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}"))

    # y[e] = 1 if edge e is included in the clique
    y = {}
    for edge_idx, (i_idx, j_idx) in enumerate(edge_list_idx):
        y[edge_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{i_idx}_{j_idx}")

    # Objective function: maximize sum of edge weights in the clique
    obj_expr = gp.LinExpr()
    for edge_idx, (i_idx, j_idx) in enumerate(edge_list_idx):
        u, v = vertices[i_idx], vertices[j_idx]
        weight = edge_weights[(u, v)]
        obj_expr += weight * y[edge_idx]
        
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Constraints: edge constraint - y_e can exist only if both endpoints are in clique
    for edge_idx, (i_idx, j_idx) in enumerate(edge_list_idx):
        model.addConstr(y[edge_idx] <= x[i_idx])
        model.addConstr(y[edge_idx] <= x[j_idx])
        model.addConstr(y[edge_idx] >= x[i_idx] + x[j_idx] - 1)

    # Constraints: No two non-edge vertices can both be in the clique
    # For every pair of vertices that are not connected by an edge, add exclusion constraint
    all_vertices_set = set(vertices)
    added_constraints = set()
    for v in vertices:
        adj_vertices = adj.get(v, set())
        non_adj_vertices = all_vertices_set - adj_vertices - {v}
        for w in non_adj_vertices:
            # Ensure we don't double-add constraints
            vu_pair = tuple(sorted((v, w)))
            if vu_pair in added_constraints:
                continue
            added_constraints.add(vu_pair)
            
            v_idx = vertex_to_idx[v]
            w_idx = vertex_to_idx[w]
            model.addConstr(x[v_idx] + x[w_idx] <= 1)

    # Store best solution found
    best_clique = []
    best_objective_value = 0.0

    # Callback to capture improving solutions
    def mycallback(model, where):
        nonlocal best_clique, best_objective_value
        if where == GRB.Callback.MIPSOL:
            # Retrieve current solution values
            x_vals = model.cbGetSolution(x)
            selected_vertex_indices = [i for i in range(n) if x_vals[i] > 0.5]

            # Compute objective value for this solution
            current_obj = 0
            for edge_idx, (i_idx, j_idx) in enumerate(edge_list_idx):
                u, v = vertices[i_idx], vertices[j_idx]
                y_val = model.cbGetSolution(y[edge_idx])
                if y_val > 0.5:
                    current_obj += edge_weights[(u, v)]

            if current_obj > best_objective_value:
                best_objective_value = current_obj
                best_clique = [vertices[i] for i in selected_vertex_indices]
                if logger:
                    logger.log(current_obj)

    # Solve the model
    model.optimize(mycallback)

    # In case the callback didn't get executed due to the solver's solution path,
    # check the final solution if it's better than what we have.
    if model.SolCount > 0 and model.status != GRB.INFEASIBLE:
        x_vals = model.getAttr('X', x)
        selected_vertex_indices = [i for i in range(n) if x_vals[i] > 0.5]
        final_clique = [vertices[i] for i in selected_vertex_indices]
        
        # Calculate true objective value (just to ensure it matches model.ObjVal)
        final_obj_val = 0
        y_vals = model.getAttr('X', y)
        for edge_idx, (i_idx, j_idx) in enumerate(edge_list_idx):
            u, v = vertices[i_idx], vertices[j_idx]
            if y_vals[edge_idx] > 0.5:
                final_obj_val += edge_weights[(u, v)]

        if final_obj_val > best_objective_value:
            best_clique = final_clique
            best_objective_value = final_obj_val

    return {
        "objective_value": best_objective_value,
        "clique": sorted(best_clique)
    }


def main():
    parser = argparse.ArgumentParser(description='Solve Weighted Max Clique Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file for intermediate solutions (optional)')
    
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_weighted_max_clique(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()