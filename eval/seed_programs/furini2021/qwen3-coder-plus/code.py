import argparse
import json
import itertools
import time
import random
from collections import defaultdict

def is_clique(graph, nodes):
    """Check if the subgraph induced by nodes is a clique"""
    for u, v in itertools.combinations(nodes, 2):
        if v not in graph[u]:
            return False
    return True


def find_max_clique(graph, timeout_time):
    """Find maximum clique using Bron-Kerbosch algorithm with timeout"""
    def bron_kerbosch_with_timeout(R, P, X, start_time):
        nonlocal max_clique, time_exceeded
        
        if time.time() - start_time >= timeout_time:
            time_exceeded = True
            return
        
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(max_clique):
                max_clique = R[:]
            return

        # Pivot heuristic for pruning
        pivot_choices = P + X
        if pivot_choices:
            u = next(iter(pivot_choices))
            neighbors_u = graph.get(u, set())
            # Try nodes that are NOT neighbors of u first, then those who are
            for node in list(P):
                if time.time() - start_time >= timeout_time:
                    time_exceeded = True
                    return
                if node in neighbors_u:
                    continue
                
                bron_kerbosch_with_timeout(
                    R + [node],
                    [v for v in P if v in graph[node]],
                    [v for v in X if v in graph[node]],
                    start_time
                )
                
                P.remove(node)
                X.append(node)

        # Handle neighbors of u
        for node in list(P):
            if time.time() - start_time >= timeout_time:
                time_exceeded = True
                return
            
            bron_kerbosch_with_timeout(
                R + [node],
                [v for v in P if v in graph[node]],
                [v for v in X if v in graph[node]],
                start_time
            )
            
            P.remove(node)
            X.append(node)

    max_clique = []
    time_exceeded = False
    start_time = time.time()
    
    nodes = list(graph.keys())
    P = list(nodes)  # Potential nodes
    R = []  # Currently building clique
    X = []  # Excluded nodes
    
    bron_kerbosch_with_timeout(R, P, X, start_time)
    
    return max_clique


def convert_edges_to_graph(edges, vertices):
    """Convert edge list to adjacency dict"""
    graph = {v: set() for v in vertices}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    return graph


def brute_force_edge_removal(starting_graph, budget, vertices):
    """Try all possible edge removal combinations up to budget size"""
    all_edges = []
    for u in starting_graph:
        for v in starting_graph[u]:
            if u < v:  # Only add (u, v) once to remove duplicates
                all_edges.append((u, v))
    
    best_remaining_clique_size = float('inf')
    best_interdicted_edges = []
    
    # Generate all possible sets of edges to remove
    for r in range(min(budget + 1, len(all_edges) + 1)):
        for edges_to_remove in itertools.combinations(all_edges, r):
            current_graph = defaultdict(set)
            for u in starting_graph:
                current_graph[u] = starting_graph[u].copy()
                
            # Remove selected edges
            for u, v in edges_to_remove:
                if v in current_graph[u]:
                    current_graph[u].remove(v)
                if u in current_graph[v]:
                    current_graph[v].remove(u)
            
            # Find max clique in remaining graph
            max_clique = find_max_clique(current_graph, 1)  # Limit time to 1 sec per check
            if max_clique:
                max_clique_size = len(max_clique)
            else:
                max_clique_size = 0
                
            if max_clique_size < best_remaining_clique_size:
                best_remaining_clique_size = max_clique_size
                best_interdicted_edges = list(edges_to_remove)
        
        # Time limit check might go here if needed
        if r == budget:  # Only consider up to full budget size
            break
    
    return best_remaining_clique_size, best_interdicted_edges


def greedy_edge_removal(graph, budget, vertices):
    """Greedy algorithm to iteratively remove edges that most reduce max clique size"""
    interdicted_edges = []
    temp_graph = {}
    for u in graph:
        temp_graph[u] = graph[u].copy()
    
    remaining_budget = budget
    total_nodes = len(vertices)
    
    for _ in range(remaining_budget):
        worst_edge = None
        best_reduction = -1
        
        # Find edge whose removal causes max reduction
        candidate_edges = []
        for u in temp_graph:
            for v in temp_graph[u]:
                if u < v:
                    candidate_edges.append((u, v))
        
        for u, v in candidate_edges:
            # Temporarily remove edge
            temp_graph[u].remove(v)
            temp_graph[v].remove(u)
            
            # Get the max clique in the modified graph
            max_clique = find_max_clique(temp_graph, 0.5)  # Limited time
            clique_size = len(max_clique) if max_clique else 0
            
            # Restore edge to test other candidates
            temp_graph[u].add(v)
            temp_graph[v].add(u)
            
            reduction = total_nodes - clique_size  # Higher reduction is better
            if reduction > best_reduction:
                best_reduction = reduction
                worst_edge = (u, v)
        
        if worst_edge is None:
            break  # No more edges to remove, graph might be disconnected enough
        
        # Actually remove the selected edge
        u, v = worst_edge
        temp_graph[u].remove(v)
        temp_graph[v].remove(u)
        interdicted_edges.append([u, v])
    
    # Calculate final max clique after all removals
    final_max_clique = find_max_clique(temp_graph, 2.0)
    remaining_clique_size = len(final_max_clique) if final_max_clique else 0
    
    return remaining_clique_size, interdicted_edges


def get_edge_betweenness(graph, edges):
    """Calculate approx betweenness for each edge based on shortest paths in cliques"""
    edge_count = {tuple(sorted(e)) if isinstance(e, list) else tuple(sorted(list(e))) : 0 
                  for e in edges}
    
    # For each vertex act as source
    vertices = list(graph.keys())
    for s in vertices:
        # BFS to find shortest paths from s
        queue = [s]
        distances = {v: float('inf') for v in vertices}
        distances[s] = 0
        predecessors = {v: [] for v in vertices}
        
        visited = {v: False for v in vertices}
        visited[s] = True
        
        while queue:
            u = queue.pop(0)
            
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    distances[v] = distances[u] + 1
                    queue.append(v)
                if distances[v] == distances[u] + 1:
                    predecessors[v].append(u)
        
        # Backtrack to count how many shortest paths use each edge
        delta = {v: 0 for v in vertices}
        for v in sorted(distances.keys(), key=lambda x: distances[x], reverse=True):
            for pred in predecessors[v]:
                edge = tuple(sorted([pred, v]))
                if edge in edge_count:
                    c = 1.0 / len(predecessors[v])
                    delta[pred] += c * (1.0 + delta[v])
                    edge_count[edge] += delta[v] * c
    
    return edge_count


def centrality_based_edge_removal(graph, budget, vertices):
    """Remove edges based on centrality measures that affect cliques most"""
    interdicted_edges = []
    current_graph = {v: graph[v].copy() for v in graph}
    
    for _ in range(budget):
        # Calculate which edge removal would most impact cliques
        max_impact_edge = None
        max_impact_score = -1
        
        # Get current edges in the graph
        current_edges = []
        for u in current_graph:
            for v in current_graph[u]:
                if u < v:
                    current_edges.append((u, v))
        
        # Calculate degree centrality to identify important edges
        for u, v in current_edges:
            # Removal of this edge might break high-degree connections
            # Calculate score based on how much it affects connectivity of high-degree nodes
            score = len(current_graph[u]) + len(current_graph[v])
            
            # Also consider if removing this edge breaks a potential large clique
            common_neighbors = current_graph[u] & current_graph[v]
            score += len(common_neighbors)  # Removing connections of hubs that create triangles
            
            if score > max_impact_score:
                max_impact_score = score
                max_impact_edge = (u, v)
        
        if max_impact_edge is None:
            break  # No edges to remove
        
        # Actually remove the selected edge
        u, v = max_impact_edge
        current_graph[u].remove(v)
        current_graph[v].remove(u)
        interdicted_edges.append([u, v])
    
    # Get final max clique after removals
    final_max_clique = find_max_clique(current_graph, 3.0)
    remaining_clique_size = len(final_max_clique) if final_max_clique else 0
    
    return remaining_clique_size, interdicted_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    num_vertices = instance['num_vertices']
    num_edges = instance['num_edges']  
    interdiction_budget_k = instance['interdiction_budget_k']
    vertices = instance['vertices']
    edges = instance['edges']
    
    original_graph = convert_edges_to_graph(edges, vertices)
    
    start_time = time.time()
    
    # Start with a basic greedy approach
    if num_edges <= 20:  # Small instances - try brute force if small
        solution_clique_size, solution_edges = brute_force_edge_removal(original_graph, interdiction_budget_k, vertices)
    else:
        # Apply greedy or centrality-based approaches
        solution_clique_size, solution_edges = centrality_based_edge_removal(original_graph, interdiction_budget_k, vertices)
        
        # Maybe try one more approach for improvement
        greedy_clique_size, greedy_edges = greedy_edge_removal(original_graph, interdiction_budget_k, vertices)
        
        if greedy_clique_size < solution_clique_size:
            solution_clique_size = greedy_clique_size
            solution_edges = greedy_edges
    
    remaining_time = args.time_limit - (time.time() - start_time)
    
    # Log improved solution if found
    if logger:
        logger.log(int(solution_clique_size))
    
    # Write solution
    solution = {
        "objective_value": int(solution_clique_size),
        "interdicted_edges": solution_edges
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()