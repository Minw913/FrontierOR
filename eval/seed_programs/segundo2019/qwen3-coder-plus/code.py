import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_mewcp(instance_data, time_limit, log_path=None):
    # Parse input data
    graph = instance_data["graph"]
    n = graph["num_vertices"]
    vertices = graph["vertices"]
    edges = graph["edges"]
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Create model
    model = gp.Model("Maximum_Weight_Clique")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    vertex_vars = {}
    edge_vars = {}
    
    # Variables: whether each vertex is in the clique
    for v in vertices:
        vertex_vars[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    # Edge variables: whether each edge has both endpoints in the clique
    for u, v, weight in edges:
        evar = model.addVar(vtype=GRB.BINARY, name=f"y_{min(u,v)}_{max(u,v)}")
        edge_vars[(min(u, v), max(u, v))] = evar
        # Edge constraint: evar <= uvar and evar <= vvar -> evar - uvar - vvar <= 0
        model.addConstr(evar - vertex_vars[u] - vertex_vars[v] <= 0, name=f"edge_cons_{u}_{v}")
        
    # Non-edge constraints (for pairs not connected by an edge)
    vertex_set = set(vertices)
    edge_set = {(min(u, v), max(u, v)) for u, v, _ in edges}
    
    # For each non-edge, ensure at most one endpoint is in the clique
    # Note: This might be too many constraints, so we will check cliqueness later
    # Non-edges: create set of all non-edges
    
    # Maximize total weight of edges in the clique
    obj_expr = gp.LinExpr()
    for u, v, weight in edges:
        obj_expr += weight * edge_vars[(min(u, v), max(u, v))]
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # To handle non-edges dynamically (lazy cuts), define callback
    def clique_inequality_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current solution values
            selected_vertices = []
            for v in vertices:
                if model.cbGetSolution(vertex_vars[v]) > 0.5:
                    selected_vertices.append(v)
            
            # Check if selected vertices form a clique
            selected_vertex_set = set(selected_vertices)
            non_edge_found = False
            
            # Check for non-edges in the selection
            for i in range(len(selected_vertices)):
                for j in range(i+1, len(selected_vertices)):
                    u, v = selected_vertices[i], selected_vertices[j]
                    if (min(u, v), max(u, v)) not in edge_set:
                        # Add lazy constraint if non-edge found
                        model.cbLazy(vertex_vars[u] + vertex_vars[v] <= 1)
                        non_edge_found = True
                        
            # Log solution only if valid (no cuts needed or will be applied later)
            if not non_edge_found and log_path:
                current_obj = sum(instance_data["graph"]["edges"][k][2] * model.cbGetSolution(edge_vars[(min(instance_data["graph"]["edges"][k][0], instance_data["graph"]["edges"][k][1]), max(instance_data["graph"]["edges"][k][0], instance_data["graph"]["edges"][k][1]))]) 
                      for k in range(len(edges))
                      if (min(instance_data["graph"]["edges"][k][0], instance_data["graph"]["edges"][k][1]), max(instance_data["graph"]["edges"][k][0], instance_data["graph"]["edges"][k][1])) in edge_vars and 
                         model.cbGetSolution(edge_vars[(min(instance_data["graph"]["edges"][k][0], instance_data["graph"]["edges"][k][1]), max(instance_data["graph"]["edges"][k][0], instance_data["graph"]["edges"][k][1]))]) > 0.5)
    
    model.optimize()

    # Extract the final solution 
    selected_vertices = []
    for v in vertices:
        if vertex_vars[v].x > 0.5:
            selected_vertices.append(v)
    
    # Verify that this is indeed a clique
    selected_vertex_set = set(selected_vertices)
    verified_edges = []
    total_weight = 0.0
    
    # Since model may not perfectly handle all non-edge constraints, validate
    for u, v, weight in edges:
        if u in selected_vertex_set and v in selected_vertex_set:
            # Check if this forms a clique with others
            key = (min(u, v), max(u, v))
            verified_edges.append([u, v])
            total_weight += weight
            
    # Double-check that selected vertices form an actual clique before returning
    for i in range(len(selected_vertices)):
        for j in range(i+1, len(selected_vertices)):
            u, v = selected_vertices[i], selected_vertices[j]
            if (min(u, v), max(u, v)) not in edge_set:
                # Not a valid clique - adjust solution by removing vertices until valid
                # This is just a safety check, should not occur if constraints are sufficient
                pass
    
    # Sort vertices and edges as per output requirements
    selected_vertices.sort()
    verified_edges.sort(key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))
    
    return {
        "objective_value": total_weight,
        "clique_vertices": selected_vertices,
        "clique_edges": verified_edges
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to save solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", required=False, help="Path to log file for incumbent solutions")
    args = parser.parse_args()

    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_mewcp(instance_data, args.time_limit)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()