import argparse
import json
import sys


def solve_with_heuristic(instance_data, time_limit):
    """Solve the weighted max clique problem using a heuristic approach."""
    from solution_logger import SolutionLogger
    
    # Extract instance data
    n = instance_data["num_vertices"]
    vertices = instance_data["vertices"]
    edges = instance_data["edges"]
    
    # Build graph as adjacency list with weights
    graph = {v: {} for v in vertices}
    for edge in edges:
        u, v, w = edge["i"], edge["j"], edge["weight"]
        graph[u][v] = w
        graph[v][u] = w
    
    # Convert list of vertices to a set for O(1) lookup
    vertex_set = set(vertices)
    
    # Greedy algorithm for weighted max clique
    best_clique = []
    best_weight = 0.0
    
    # Start with each vertex as a potential starting point
    for start_vertex in vertices:
        # Greedy growth starting from this vertex
        current_clique = {start_vertex}
        
        remaining_vertices = list(graph[start_vertex].keys())  # Neighbors of start vertex
        
        changed = True
        while changed and remaining_vertices:
            changed = False
            
            # Find best candidate to add to current clique
            best_candidate = None
            best_score = float('-inf')
            
            for v in remaining_vertices:
                # Check if v is fully connected to all vertices already in clique
                if all(vc in graph[v] for vc in current_clique):
                    # Heuristic: pick vertex with high average connection weight to current clique
                    total_weight = sum(graph[v][vc] for vc in current_clique)
                    avg_weight = total_weight / len(current_clique)
                    score = avg_weight  # Or use total_weight, degree, or other criteria
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = v
            
            if best_candidate is not None:
                current_clique.add(best_candidate)
                remaining_vertices.remove(best_candidate)
                remaining_vertices = [v for v in remaining_vertices if v in graph[best_candidate]]
                changed = True
        
        # Calculate weight of the final clique
        current_weight = sum(
            graph[u][v] 
            for i, u in enumerate(current_clique) 
            for v in list(current_clique)[i+1:] 
            if v in graph[u]
        )
        
        # Check if this is the best solution found so far
        if current_weight > best_weight:
            best_weight = current_weight
            best_clique = sorted(list(current_clique))
    
    # Try another strategy: sort vertices by local density and build iteratively
    initial_cliques = []
    scores = []
    
    for start_vertex in vertices:
        # Get neighbors of this vertex
        neighbors = list(graph[start_vertex].keys())
        
        # Greedy search in this neighborhood
        # Sort neighbors by degree in the subgraph
        neighbor_subgraph = {v: {vv for vv in graph[v] if vv in neighbors and vv in graph[v]} for v in neighbors}
        
        # Initialize with the start vertex and one other neighbor
        for next_vertex in neighbors[:5]:  # Limit neighbors to check for performance
            current_clique = {start_vertex, next_vertex}
            
            # Potential additions are intersections of neighborhoods of clique vertices
            potential_additions = set(graph[start_vertex].keys()).intersection(set(graph[next_vertex].keys()))
            potential_additions = [v for v in potential_additions if v != start_vertex and v != next_vertex]
            
            changed = True
            while changed and potential_additions:
                changed = False
                
                # Get candidate which has connections to all in current_clique
                candidates = []
                for v in potential_additions:
                    if all(v in graph[c] for c in current_clique):
                        # Score based on the sum of weights to current clique
                        score = sum(graph[v][c] for c in current_clique)
                        candidates.append((score, v))
                
                if candidates:
                    # Pick highest scoring v
                    candidates.sort(reverse=True)
                    _, best_addition = candidates[0]
                    
                    current_clique.add(best_addition)
                    potential_additions.remove(best_addition)
                    # Remove those no longer valid (not connected to new vertex)
                    potential_additions = [v for v in potential_additions if v in graph[best_addition]]
                    changed = True
            
            # Calculate score for this configuration
            weight = sum(
                graph[u][v] 
                for i, u in enumerate(current_clique) 
                for v in list(current_clique)[i+1:] 
                if v in graph[u]
            )
            
            if weight > best_weight:
                best_weight = weight
                best_clique = sorted(list(current_clique))
    
    return {
        "objective_value": best_weight,
        "clique": best_clique
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

    # Import SolutionLogger after argument parsing
    from solution_logger import SolutionLogger
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    try:
        # Solve using heuristic approach
        solution = solve_with_heuristic(instance_data, args.time_limit)
        
        # Log the final solution if desired
        if logger:
            logger.log(solution["objective_value"])
    except Exception as e:
        print(f"Error during solving: {e}", file=sys.stderr)
        # Default fallback: just one node
        solution = {
            "objective_value": 0.0,
            "clique": [instance_data["vertices"][0]] if instance_data["vertices"] else []
        }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()