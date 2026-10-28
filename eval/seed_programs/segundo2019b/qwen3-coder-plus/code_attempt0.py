import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_maximum_weighted_clique(instance_data, time_limit, log_path):
    # Extract data
    num_vertices = instance_data["graph"]["num_vertices"]
    edges = instance_data["graph"]["edges"]
    vertex_weights = instance_data["vertex_weights"]
    
    # Create mapping from vertex id to index for convenience
    vertex_to_idx = {vertex_id: idx for idx, vertex_id in enumerate(instance_data["graph"]["vertices"])}
    
    # Create model
    model = gp.Model("MaximumWeightedClique")
    
    # Set Gurobi parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 2)  # Focus on finding good feasible solutions
    
    # Decision variables: x_i = 1 if vertex i is selected
    x = model.addVars(num_vertices, vtype=GRB.BINARY, name="x")
    
    # Objective function: maximize sum of weights of selected vertices
    model.setObjective(gp.quicksum(vertex_weights[i] * x[i] for i in range(num_vertices)), GRB.MAXIMIZE)
    
    # Add constraints: For every pair of vertices NOT connected by an edge,
    # we cannot select both vertices
    edges_set = set()
    for u, v in edges:
        idx_u = vertex_to_idx[u]
        idx_v = vertex_to_idx[v]
        edges_set.add((min(idx_u, idx_v), max(idx_u, idx_v)))
    
    # For pairs of vertices not in the edges list, add constraints
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if (i, j) not in edges_set:
                model.addConstr(x[i] + x[j] <= 1, f"non_adjacency_{i}_{j}")
    
    # Create logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Callback function for logging solutions
    def callback_function(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Optimize
    model.optimize(callback_function)
    
    # Extract solution
    selected_vertices = []
    for i in range(num_vertices):
        if x[i].X > 0.5:
            selected_vertices.append(instance_data["graph"]["vertices"][i])
    
    # Compute objective value
    objective_value = sum(vertex_weights[idx] for idx, _ in enumerate(instance_data["graph"]["vertices"]) if _ in [v for v in selected_vertices])
    
    # Return the result
    return {
        "objective_value": model.objVal if model.solCount > 0 else 0,
        "clique_vertices": selected_vertices
    }


def main():
    parser = argparse.ArgumentParser(description="Solve Maximum Weighted Clique Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the input instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the log file for intermediate solutions")
    
    args = parser.parse_args()
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_maximum_weighted_clique(instance_data, args.time_limit, args.log_path)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()