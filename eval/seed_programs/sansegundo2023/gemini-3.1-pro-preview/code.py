import time
import json
import argparse
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class MaxCliqueSolver:
    def __init__(self, adj_sets, initial_clique, logger, start_time, time_limit, vertices):
        self.logger = logger
        self.start_time = start_time
        self.time_limit = time_limit
        self.time_out = False
        
        self.best_clique = list(initial_clique)
        self.max_size = len(initial_clique)
        
        # Precompute bitmasks and degrees for absolute maximum efficiency
        self.adj_mask = {v: 0 for v in vertices}
        self.deg = {v: 0 for v in vertices}
        
        for u, neighbors in adj_sets.items():
            mask = 0
            for v in neighbors:
                if v in self.adj_mask: # filter out invalid nodes safely
                    mask |= (1 << v)
            self.adj_mask[u] = mask
            self.deg[u] = len(neighbors)

    def expand(self, current_clique, candidates):
        # Enforce execution time limit
        if self.time_out or time.time() - self.start_time > self.time_limit:
            self.time_out = True
            return

        # Check for improvement and log 
        if len(current_clique) > self.max_size:
            self.max_size = len(current_clique)
            self.best_clique = list(current_clique)
            if self.logger:
                self.logger.log(float(self.max_size))

        if not candidates:
            return

        # Phase 1: Greedy Graph Coloring using integer bitmasks (Fast Independent Set coloring)
        color_classes = [] 
        colors = {}
        for v in candidates:
            assigned = False
            v_adj = self.adj_mask[v]
            # Try finding a non-conflicting color for vertex v
            for i, class_mask in enumerate(color_classes):
                if (v_adj & class_mask) == 0:
                    color_classes[i] |= (1 << v)
                    colors[v] = i
                    assigned = True
                    break
            # Start a new color class if no existing one fits
            if not assigned:
                color_classes.append(1 << v)
                colors[v] = len(color_classes) - 1

        # Phase 2: Ordering based on assigned colors.
        # Nodes with higher color values are pushed earlier to maximize bound tightness.
        candidates_sorted = sorted(candidates, key=lambda v: colors[v], reverse=True)
        
        # Phase 3: Branching with early pruning based on sequence coloring
        for i, v in enumerate(candidates_sorted):
            if self.time_out:
                return
                
            # Crucial bounding condition: If current theoretical upper limit cannot exceed recorded best, prune!
            if len(current_clique) + colors[v] + 1 <= self.max_size:
                break
                
            v_adj = self.adj_mask[v]
            
            # Form next candidate generation intersected with current vertex's neighborhood.
            # Using bitwise evaluation over elements situated to the sequence's right side strictly
            # averts symmetric or repetitive combinations.
            new_candidates = [u for u in candidates_sorted[i+1:] if (v_adj & (1 << u))]
            
            current_clique.append(v)
            self.expand(current_clique, new_candidates)
            current_clique.pop()

def solve_max_clique(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_vertices = data.get("num_vertices", 0)
    vertices = data.get("vertices", [])
    adjacency_list = data.get("adjacency_list", {})
    
    adj_sets = {int(u): set(neighbors) for u, neighbors in adjacency_list.items()}
    greedy_clique = set(data.get("greedy_clique", []))
    
    logger = None
    if log_path and SolutionLogger:
        logger = SolutionLogger(log_path, sense="maximize")

    # Degenerate case verification
    if not vertices:
        solution = {
            "objective_value": 0.0, 
            "clique_vertices": []
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        return

    # Using Tomita's Branch and Bound Algorithm approach to sidestep constraints from uninstalled solvers (like gurobipy).
    solver = MaxCliqueSolver(
        adj_sets=adj_sets,
        initial_clique=greedy_clique,
        logger=logger,
        start_time=start_time,
        time_limit=max(0.1, time_limit - 0.5), # Provide minor buffer margin
        vertices=vertices
    )
    
    # Initialize the highest probability structural path utilizing deterministic order by descending degrees
    initial_candidates = sorted(vertices, key=lambda v: solver.deg[v], reverse=True)
    solver.expand([], initial_candidates)
    
    clique_vertices = [int(v) for v in solver.best_clique]
    if not clique_vertices and vertices:
        clique_vertices = [int(vertices[0])]
    objective_value = float(len(clique_vertices))
    
    solution = {
        "objective_value": objective_value,
        "clique_vertices": clique_vertices
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Fast Max Clique Optimizer")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to logging file")
    args = parser.parse_args()
    
    solve_max_clique(
        args.instance_path,
        args.solution_path,
        args.time_limit,
        args.log_path
    )