import argparse
import json
import math
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_traveling_salesman_with_budget(instance_data, time_limit, logger=None):
    """
    Solves the Traveling Salesman with Budget constraint problem.
    
    Args:
        instance_data (dict): Dictionary containing problem instance data
        time_limit (int): Time limit in seconds
        logger: Logger to track intermediate solutions
    
    Returns:
        dict: Solution dictionary with objective value, visited vertices and tour edges
    """
    num_nodes = instance_data["num_nodes"]
    dist_limit = instance_data["distance_limitation_d0"]
    depot = instance_data["depot_vertex"]
    scores = instance_data["scores"]
    distance_matrix = instance_data["distance_matrix"]

    # Create model
    model = gp.Model("TSP_with_budget")
    model.setParam('TimeLimit', time_limit)
    # To improve heuristics during search 
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions

    # Variables for edges in the tour
    x = {}
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            x[j,i] = x[i,j]

    # Track whether a vertex is in the tour
    y = {}
    for i in range(num_nodes):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    # Add degree constraints
    for i in range(num_nodes):
        # Every node in the tour must have 2 incident edges
        degree_expr = gp.LinExpr()
        for j in range(num_nodes):
            if i != j:
                degree_expr += x[i,j]
        if i == depot:
            # Depot always in the tour and degree 2
            model.addConstr(degree_expr == 2 * y[i])
            model.addConstr(y[i] == 1)  # depot must be in tour
        else:
            model.addConstr(degree_expr == 2 * y[i])

    # Budget constraint
    dist_expr = gp.LinExpr()
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist_expr += distance_matrix[i][j] * x[i,j]
    model.addConstr(dist_expr <= dist_limit)

    # Subtour elimination using MTZ-like constraints
    u = {}
    for i in range(1, num_nodes):
        u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_nodes-1, name=f"u_{i}")

    # MTZ constraints to prevent subtours (adapted for our problem)
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                # If edge from i to j exists, enforce ordering to eliminate cycles
                # This is modified since x_ij may be binary across whole graph
                model.addConstr(u[i] - u[j] + (num_nodes-1) * x[i,j] <= num_nodes - 2)

    # Objective: maximize collected scores
    obj_expr = gp.LinExpr()
    for i in range(num_nodes):
        obj_expr += scores[i] * y[i]
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Solve the model
    model.optimize()

    best_solution = {
        "objective_value": 0,
        "visited_vertices": [],
        "tour_edges": []
    }

    if model.SolCount > 0:
        # Get the values of the solution
        selected_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if x[i,j].X > 0.5:
                    selected_edges.append((i, j))

        # Find visited vertices from the edges
        visited_vertices = set()
        for u, v in selected_edges:
            visited_vertices.add(u)
            visited_vertices.add(v)
        
        # Ensure depot is in the tour
        visited_vertices.add(depot)
        
        # Build tour edges with directions so we get a consistent path representation
        visited_list = list(visited_vertices)
        
        best_solution = {
            "objective_value": model.ObjVal,
            "visited_vertices": list(visited_vertices),
            "tour_edges": selected_edges
        }

        if logger and model.SolCount > 0:
            logger.log(model.ObjVal)

    return best_solution


def main():
    parser = argparse.ArgumentParser(description="Solve TSP with distance budget")
    parser.add_argument("--instance_path", type=str, required=True, 
                        help="Path to input JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to output JSON solution file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to log file for intermediate solutions")
    
    args = parser.parse_args()
    
    # Initialize logger if path provided
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_traveling_salesman_with_budget(instance_data, args.time_limit, logger)
    
    # Write solution to output file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()