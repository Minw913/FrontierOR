import argparse
import json
import time
import sys
from collections import defaultdict

def greedy_coloring(adj, num_nodes, order=None):
    """Greedy coloring with given order. Returns coloring dict and num colors."""
    if order is None:
        order = list(range(num_nodes))
    
    color = [-1] * num_nodes
    num_colors = 0
    
    for node in order:
        # Find colors used by neighbors
        used = set()
        for nb in adj[node]:
            if color[nb] >= 0:
                used.add(color[nb])
        # Assign smallest available color
        c = 0
        while c in used:
            c += 1
        color[node] = c
        if c + 1 > num_colors:
            num_colors = c + 1
    
    return color, num_colors

def dsatur_coloring(adj, num_nodes):
    """DSATUR heuristic for graph coloring."""
    color = [-1] * num_nodes
    saturation = [0] * num_nodes
    degree = [len(adj[i]) for i in range(num_nodes)]
    colored = [False] * num_nodes
    neighbor_colors = [set() for _ in range(num_nodes)]
    num_colors = 0
    
    for _ in range(num_nodes):
        # Pick uncolored vertex with max saturation, break ties by degree
        best = -1
        best_sat = -1
        best_deg = -1
        for v in range(num_nodes):
            if not colored[v]:
                if (saturation[v] > best_sat or 
                    (saturation[v] == best_sat and degree[v] > best_deg)):
                    best = v
                    best_sat = saturation[v]
                    best_deg = degree[v]
                    
        # Assign smallest available color
        c = 0
        while c in neighbor_colors[best]:
            c += 1
        color[best] = c
        colored[best] = True
        if c + 1 > num_colors:
            num_colors = c + 1
        
        # Update saturation of neighbors
        for nb in adj[best]:
            if not colored[nb]:
                if c not in neighbor_colors[nb]:
                    neighbor_colors[nb].add(c)
                    saturation[nb] += 1
    
    return color, num_colors

def try_k_coloring_greedy(adj, num_nodes, k, rng_seed, max_iter=1000):
    """Try to find a k-coloring using iterative greedy with random restarts."""
    import random
    rng = random.Random(rng_seed)
    
    best_conflicts = float('inf')
    best_coloring = None
    
    for iteration in range(max_iter):
        # Random order
        order = list(range(num_nodes))
        rng.shuffle(order)
        
        color = [-1] * num_nodes
        conflicts = 0
        
        for node in order:
            used = set()
            for nb in adj[node]:
                if color[nb] >= 0 and color[nb] < k:
                    used.add(color[nb])
            # Assign smallest available color within k
            assigned = False
            for c in range(k):
                if c not in used:
                    color[node] = c
                    assigned = True
                    break
            if not assigned:
                # Pick least conflicting color
                conflict_count = [0] * k
                for nb in adj[node]:
                    if color[nb] >= 0:
                        conflict_count[color[nb]] += 1
                c = min(range(k), key=lambda x: conflict_count[x])
                color[node] = c
                conflicts += conflict_count[c]
        
        if conflicts == 0:
            return color, True
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_coloring = color[:]
    
    return best_coloring, False

def tabucol(adj, num_nodes, k, max_iter, deadline, rng_seed=42):
    """TabuCol algorithm for k-coloring."""
    import random
    rng = random.Random(rng_seed)
    
    if num_nodes == 0:
        return [], True
    
    # Initialize random coloring
    color = [rng.randint(0, k-1) for _ in range(num_nodes)]
    
    # Compute conflict count for each (node, color) pair
    # conflict[v][c] = number of neighbors of v that have color c
    conflict = [[0] * k for _ in range(num_nodes)]
    for v in range(num_nodes):
        for nb in adj[v]:
            conflict[v][color[nb]] += 1
    
    # Count total conflicts
    total_conflicts = 0
    for v in range(num_nodes):
        total_conflicts += conflict[v][color[v]]
    total_conflicts //= 2  # Each conflict counted twice
    
    if total_conflicts == 0:
        return color, True
    
    # Tabu tenure
    tabu_tenure = max(7, num_nodes // 10)
    tabu = [[0] * k for _ in range(num_nodes)]  # tabu[v][c] = iteration until which (v,c) is tabu
    
    best_conflicts = total_conflicts
    
    for iteration in range(1, max_iter + 1):
        if time.time() > deadline:
            break
            
        # Find best non-tabu move
        best_delta = float('inf')
        best_moves = []
        
        # Only consider conflicting vertices
        for v in range(num_nodes):
            if conflict[v][color[v]] > 0:  # v is in conflict
                current_c = color[v]
                for c in range(k):
                    if c == current_c:
                        continue
                    # Delta = change in conflicts
                    delta = conflict[v][c] - conflict[v][current_c]
                    
                    # Check if move is tabu (unless it leads to best known)
                    new_total = total_conflicts + delta
                    is_tabu = tabu[v][c] > iteration
                    
                    if is_tabu and new_total >= best_conflicts:
                        continue
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_moves = [(v, c)]
                    elif delta == best_delta:
                        best_moves.append((v, c))
        
        if not best_moves:
            # All moves are tabu, pick random conflicting vertex
            conflicting = [v for v in range(num_nodes) if conflict[v][color[v]] > 0]
            if not conflicting:
                return color, True
            v = rng.choice(conflicting)
            c = rng.choice([c for c in range(k) if c != color[v]])
            best_moves = [(v, c)]
            best_delta = conflict[v][c] - conflict[v][color[v]]
        
        # Pick random best move
        v, new_c = rng.choice(best_moves)
        old_c = color[v]
        
        # Apply move
        color[v] = new_c
        total_conflicts += best_delta
        
        # Update conflict table
        for nb in adj[v]:
            conflict[nb][old_c] -= 1
            conflict[nb][new_c] += 1
        
        # Update tabu
        tabu[v][old_c] = iteration + tabu_tenure + rng.randint(0, 3)
        
        if total_conflicts < best_conflicts:
            best_conflicts = total_conflicts
        
        if total_conflicts == 0:
            return color, True
    
    return color, (total_conflicts == 0)

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    deadline = start_time + args.time_limit - 1  # Leave 1 second buffer
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    num_nodes = instance['num_nodes']
    num_edges = instance['num_edges']
    edges = instance['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    if num_nodes == 0:
        solution = {
            "objective_value": 0,
            "coloring": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    if num_edges == 0:
        coloring = {str(i): 0 for i in range(num_nodes)}
        solution = {
            "objective_value": 1,
            "coloring": coloring
        }
        if logger:
            logger.log(1)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Get initial solution with DSATUR
    best_color, best_num_colors = dsatur_coloring(adj, num_nodes)
    
    if logger:
        logger.log(best_num_colors)
    
    # Also try greedy with largest-first ordering
    degree_order = sorted(range(num_nodes), key=lambda v: len(adj[v]), reverse=True)
    color2, nc2 = greedy_coloring(adj, num_nodes, degree_order)
    if nc2 < best_num_colors:
        best_color = color2
        best_num_colors = nc2
        if logger:
            logger.log(best_num_colors)
    
    # Compute a lower bound: clique lower bound (max degree + 1 is upper, but simple LB is max clique)
    # Simple lower bound: at least the size of the largest clique
    # For now, use a simple lower bound based on max degree neighborhood
    max_degree = max(len(adj[v]) for v in range(num_nodes))
    
    # Lower bound: chromatic number >= clique number
    # Simple greedy clique finding
    def find_clique_greedy():
        """Find a clique greedily."""
        # Start from highest degree vertex
        order = sorted(range(num_nodes), key=lambda v: len(adj[v]), reverse=True)
        best_clique = []
        
        for start in order[:min(50, num_nodes)]:
            clique = [start]
            adj_set = set(adj[start])
            candidates = [v for v in adj[start] if v > start or len(best_clique) == 0]
            # Sort candidates by degree in subgraph
            for v in sorted(candidates, key=lambda x: len(adj[x]), reverse=True):
                if all(v in set(adj[u]) for u in clique):
                    clique.append(v)
                    adj_set = adj_set & set(adj[v])
            if len(clique) > len(best_clique):
                best_clique = clique
        
        return best_clique
    
    clique = find_clique_greedy()
    lower_bound = len(clique)
    
    if best_num_colors <= lower_bound:
        # Already optimal
        coloring = {str(i): best_color[i] for i in range(num_nodes)}
        solution = {
            "objective_value": best_num_colors,
            "coloring": coloring
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Try to reduce the number of colors using TabuCol
    target_k = best_num_colors - 1
    
    while target_k >= lower_bound and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        
        # Determine max iterations based on problem size
        if num_nodes <= 100:
            max_iter = 100000
        elif num_nodes <= 500:
            max_iter = 50000
        else:
            max_iter = 20000
        
        # Try multiple seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            
            color_result, success = tabucol(adj, num_nodes, target_k, max_iter, deadline, rng_seed=seed * 1337 + 42)
            
            if success:
                best_color = color_result
                best_num_colors = target_k
                if logger:
                    logger.log(best_num_colors)
                found = True
                break
        
        if not found:
            # Try harder with more iterations if time permits
            remaining = deadline - time.time()
            if remaining > 2:
                extra_iter = min(500000, int(remaining * 10000))
                color_result, success = tabucol(adj, num_nodes, target_k, extra_iter, deadline, rng_seed=9999)
                if success:
                    best_color = color_result
                    best_num_colors = target_k
                    if logger:
                        logger.log(best_num_colors)
                    found = True
        
        if found:
            if best_num_colors <= lower_bound:
                break
            target_k = best_num_colors - 1
        else:
            break
    
    # For small graphs, try exact solution with Gurobi if time permits
    remaining = deadline - time.time()
    if remaining > 2 and num_nodes <= 80 and best_num_colors > lower_bound:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            k = best_num_colors
            
            model = gp.Model("coloring")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining - 0.5)
            model.setParam('Threads', 1)
            
            # x[v][c] = 1 if vertex v has color c
            x = {}
            for v in range(num_nodes):
                for c in range(k):
                    x[v, c] = model.addVar(vtype=GRB.BINARY, name=f'x_{v}_{c}')
            
            # y[c] = 1 if color c is used
            y = {}
            for c in range(k):
                y[c] = model.addVar(vtype=GRB.BINARY, name=f'y_{c}')
            
            model.update()
            
            # Each vertex gets exactly one color
            for v in range(num_nodes):
                model.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1)
            
            # Adjacent vertices different colors
            for u, v in edges:
                for c in range(k):
                    model.addConstr(x[u, c] + x[v, c] <= 1)
            
            # Link x and y
            for v in range(num_nodes):
                for c in range(k):
                    model.addConstr(x[v, c] <= y[c])
            
            # Symmetry breaking: use colors in order
            for c in range(1, k):
                model.addConstr(y[c] <= y[c-1])
            
            # Warm start
            for v in range(num_nodes):
                for c in range(k):
                    x[v, c].Start = 1.0 if best_color[v] == c else 0.0
            for c in range(k):
                y[c].Start = 1.0 if c in set(best_color) else 0.0
            
            model.setObjective(gp.quicksum(y[c] for c in range(k)), GRB.MINIMIZE)
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                    if obj < best_num_colors:
                        if logger:
                            logger.log(obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                obj = int(round(model.ObjVal))
                if obj < best_num_colors:
                    best_num_colors = obj
                    for v in range(num_nodes):
                        for c in range(k):
                            if x[v, c].X > 0.5:
                                best_color[v] = c
                                break
                    if logger:
                        logger.log(best_num_colors)
        except Exception:
            pass
    
    # Remap colors to be contiguous starting from 0
    used_colors = sorted(set(best_color))
    color_map = {c: i for i, c in enumerate(used_colors)}
    final_color = [color_map[best_color[v]] for v in range(num_nodes)]
    
    coloring = {str(i): final_color[i] for i in range(num_nodes)}
    solution = {
        "objective_value": len(used_colors),
        "coloring": coloring
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()