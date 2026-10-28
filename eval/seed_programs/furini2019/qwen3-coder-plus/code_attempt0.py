import argparse
import json
import time
from solution_logger import SolutionLogger
from collections import defaultdict


def is_clique(vertices, adjacency_set):
    """Check if a set of vertices forms a clique"""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if vertices[j] not in adjacency_set[vertices[i]]:
                return False
    return True


def bron_kerbosch_with_pivot(result, r, p, x, graph_adj, degeneracy_order):
    """
    Bron-Kerbosch algorithm with pivoting for finding all maximal cliques.
    Using degeneracy ordering for better performance.
    """
    if not p and not x:
        result.append(r[:])  # Found a maximal clique
        return

    # Choose pivot vertex u as the one with highest degree among p and x
    candidates = p | x
    u = max(candidates, key=lambda v: len(graph_adj[v]), default=list(p)[0])
    
    # Only consider neighbors of u that are in p
    for v in p - set(graph_adj.get(u, [])):
        r_new = r + [v]
        p_new = p & set(graph_adj.get(v, []))
        x_new = x & set(graph_adj.get(v, []))
        
        bron_kerbosch_with_pivot(result, r_new, p_new, x_new, graph_adj, degeneracy_order)
        
        p.remove(v)
        x.add(v)


def get_deg_order(num_vertices, adj_list):
    """
    Get the degeneracy order of vertices.
    Order vertices by increasing degree of unprocessed graph.
    """
    import heapq
    
    # Copy adjacency lists to work with them
    adjs = [set(neighbors) for neighbors in adj_list]
    order = []
    # Priority queue: (degree, vertex)
    pq = [(len(adjs[i]), i) for i in range(num_vertices)]
    heapq.heapify(pq)
    processed = [False] * num_vertices
    
    while pq:
        _, v = heapq.heappop(pq)
        if processed[v]:
            continue
            
        order.append(v)
        processed[v] = True
        
        # Remove v from its neighbors' adjacency lists
        for neighbor in adjs[v]:
            if not processed[neighbor]:
                adjs[neighbor].remove(v)
                # Update in priority queue (add updated entry)
                heapq.heappush(pq, (len(adjs[neighbor]), neighbor))
    
    return order[::-1]  # Return in reverse order for degeneracy ordering


def find_maximal_cliques_bk(graph_adj, num_vertices, adj_list):
    """
    Find maximal cliques using Bron-Kerbosch with degeneracy ordering
    """
    degeneracy_order = get_deg_order(num_vertices, adj_list)
    result = []
    
    # Prepare adjacency as sets instead of lists for faster operations
    graph_adj_sets = {}
    for v in graph_adj:
        graph_adj_sets[v] = set(graph_adj[v])
    
    # Start bron kerbosch
    p = set(range(num_vertices))
    r = []
    x = set()
    
    bron_kerbosch_with_pivot(result, r, p, x, graph_adj_sets, degeneracy_order)
    return result


def greedy_max_clique(num_vertices, adj_list, max_time=None, logger=None):
    """
    Greedy approach with local search to find large cliques
    """
    start_time = time.time()
    best_clique = []
    
    # Adjacency as a set for O(1) lookup
    adj_dict = {i: set(adj_list[i]) for i in range(num_vertices)}
    
    # Try starting from each node to build a greedy clique
    for start_node in range(num_vertices):
        if max_time and time.time() - start_time > max_time:
            break
            
        # Greedy growth from start_node
        current_clique = [start_node]
        remaining_nodes = set(adj_dict[start_node])
        
        while remaining_nodes:
            if max_time and time.time() - start_time > max_time:
                break

            # Pick node with most connections to current clique
            candidate_scores = {}
            for node in remaining_nodes:
                score = len(adj_dict[node].intersection(current_clique))
                candidate_scores[node] = score
            
            # Find max scoring candidate
            best_candidate = None
            best_score = -1
            for node, score in candidate_scores.items():
                if score > best_score:
                    best_score = score
                    best_candidate = node
            
            # We need to connect to all in current clique to add
            if best_score == len(current_clique):
                current_clique.append(best_candidate)
                # Update remaining nodes to be neighbors of the new node and previous ones
                remaining_nodes &= adj_dict[best_candidate]
            else:
                # No node connects to all in current_clique, exit loop
                break
        
        if len(current_clique) > len(best_clique):
            best_clique = current_clique[:]
            if logger:
                logger.log(len(best_clique))
    
    return best_clique


def local_search_improve(initial_clique, adj_dict, num_vertices, max_time=None, logger=None, best_callback=None):
    """
    Improve a given clique through local search: add, remove, swap moves
    """
    start_time = time.time()
    current_clique = initial_clique[:]
    current_clique_set = set(current_clique)
    best_clique = current_clique[:]
    
    iterations = 0
    improving = True
    
    while improving and (not max_time or time.time() - start_time < max_time):
        improving = False
        iterations += 1
        
        # 1. Try adding nodes from outside
        potential_additions = []
        for v in range(num_vertices):
            if v not in current_clique_set:
                # Check if v can be added to current_clique
                can_add = all(v in adj_dict[n] for n in current_clique)
                if can_add:
                    potential_additions.append(v)
                
        if potential_additions:
            # Add the node that gives the best expansion
            for v in potential_additions:
                new_clique = current_clique + [v]
                if len(new_clique) > len(best_clique):
                    best_clique = new_clique[:]
                    if logger:
                        logger.log(len(best_clique))
                current_clique = new_clique[:]
                current_clique_set.add(v)
                improving = True
                break  # Break to re-evaluate remaining additions
        
        # 2. Try swapping vertices (1-out, 1-in)
        if not improving:
            for i, old_v in enumerate(current_clique):
                for new_v in range(num_vertices):
                    if new_v not in current_clique_set:
                        # Verify removing old_v still connects to new_v's neighbors
                        temp_clique = current_clique[:i] + current_clique[i+1:]
                        all_connected_new = all(new_v in adj_dict[other_v] for other_v in temp_clique)
                        
                        if all_connected_new:
                            # Verify new_v connects to existing clique minus old_v
                            if new_v in adj_dict[old_v]:  # Old and new must be neighbors too for the swap
                                all_connected_old = all(old_v in adj_dict[other_v] for other_v in temp_clique)
                                if all_connected_old:  # Now can perform swap
                                    new_clique = temp_clique + [new_v]
                                    if len(new_clique) == len(current_clique):
                                        current_clique = new_clique[:]
                                        current_clique_set = set(current_clique)
                                        # Could be a different but same-size clique (diversification)
                                        break
        
    return best_clique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Read the instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance["num_vertices"]
    edges = instance["edges"]
    adj_list = instance["adjacency_list"]

    # Start timing
    start_time = time.time()
    remaining_time = args.time_limit

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Build adjacencies as a set for fast lookup
    adj_dict = {i: set(row) for i, row in enumerate(adj_list)}

    # Try different approaches in the remaining time
    best_clique = []

    # Method 1: Greedy
    if remaining_time > 0:
        clique_greedy = greedy_max_clique(
            num_vertices, 
            adj_list, 
            max_time=remaining_time, 
            logger=logger
        )
        if len(clique_greedy) > len(best_clique):
            best_clique = clique_greedy[:]
        elapsed = time.time() - start_time
        remaining_time = args.time_limit - elapsed

    # Method 2: Local search improvement
    if len(best_clique) > 0 and remaining_time > 0:
        improved_clique = local_search_improve(
            best_clique,
            adj_dict,
            num_vertices,
            max_time=remaining_time,
            logger=logger
        )
        if len(improved_clique) > len(best_clique):
            best_clique = improved_clique[:]
        elapsed = time.time() - start_time
        remaining_time = args.time_limit - elapsed

    # If we have a lot of time left and small enough graph, try exact method
    if remaining_time > 5 and num_vertices <= 70:  # Heuristic threshold
        try:
            from itertools import combinations
            # Use greedy + local search as initial solution, skip complex exact methods for large graphs
            max_cliques = find_maximal_cliques_bk(
                {i: adj_list[i] for i in range(len(adj_list))},
                num_vertices,
                adj_list
            )
            
            for clique in max_cliques:
                if len(clique) > len(best_clique):
                    best_clique = clique[:]
                    if logger:
                        logger.log(len(best_clique))
        except:
            # If exact approach fails due to complexity, just use what we have
            pass

    # Final output
    solution = {
        "objective_value": len(best_clique),
        "clique_vertices": best_clique
    }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()