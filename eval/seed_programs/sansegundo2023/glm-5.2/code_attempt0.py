import json
import argparse
import time
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)
    
    V = data["num_vertices"]
    adj_list = data["adjacency_list"]
    
    # Build adjacency masks. adj[v] includes v itself (1 << v) for convenience.
    adj = [0] * V
    for v in range(V):
        mask = 0
        if str(v) in adj_list:
            for u in adj_list[str(v)]:
                mask |= (1 << u)
        adj[v] = mask | (1 << v)
        
    degrees = [0] * V
    for v in range(V):
        degrees[v] = adj[v].bit_count() - 1
        
    best_clique = []
    if "greedy_clique" in data and len(data["greedy_clique"]) > 0:
        best_clique = data["greedy_clique"]
        if logger:
            logger.log_solution(len(best_clique), {"clique_vertices": best_clique})

    start_time = time.time()
    time_limit = args.time_limit
    
    colors = [0] * V
    
    def expand(R, P, R_size):
        nonlocal best_clique
        if time.time() - start_time > time_limit:
            return
            
        if P == 0:
            if R_size > len(best_clique):
                best_clique = []
                temp = R
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    best_clique.append(v)
                    temp -= (1 << v)
                if logger:
                    logger.log_solution(R_size, {"clique_vertices": best_clique})
            return
            
        P_list = []
        temp = P
        while temp:
            v = (temp & -temp).bit_length() - 1
            P_list.append(v)
            temp -= (1 << v)
            
        # Sort by degree descending to get a tighter greedy coloring
        P_list.sort(key=lambda x: degrees[x], reverse=True)
        
        color_classes = []
        for v in P_list:
            assigned = False
            for i in range(len(color_classes)):
                if (color_classes[i] & adj[v]) == 0:
                    color_classes[i] |= (1 << v)
                    colors[v] = i + 1
                    assigned = True
                    break
            if not assigned:
                color_classes.append(1 << v)
                colors[v] = len(color_classes)
                
        max_color = len(color_classes)
        if R_size + max_color <= len(best_clique):
            return
            
        # Branch on vertices in descending order of color
        P_list.sort(key=lambda x: colors[x], reverse=True)
        
        for v in P_list:
            if R_size + colors[v] <= len(best_clique):
                break
                
            new_R = R | (1 << v)
            new_P = (P & adj[v]) - (1 << v)
            
            expand(new_R, new_P, R_size + 1)
            
            P -= (1 << v)
            
            if time.time() - start_time > time_limit:
                return

    P_initial = 0
    for v in range(V):
        P_initial |= (1 << v)
        
    expand(0, P_initial, 0)
    
    output = {
        "objective_value": len(best_clique),
        "clique_vertices": best_clique
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    solve()