import json
import argparse
import sys
import time
import random

# Increase recursion depth for deep graphs
sys.setrecursionlimit(20000)

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class MaxEdgeWeightClique:
    def __init__(self, vertices, edges, logger, time_limit):
        self.vertices = vertices
        self.edges = edges
        self.logger = logger
        self.time_limit = time_limit
        self.start_time = time.time()
        
        self.adj = {v: set() for v in vertices}
        self.weights = {}
        for e in edges:
            u, v, w = e['i'], e['j'], e['weight']
            self.adj[u].add(v)
            self.adj[v].add(u)
            self.weights[(u, v)] = w
            self.weights[(v, u)] = w
            
        self.best_weight = -1.0
        self.best_clique = []
        
    def check_time(self):
        # Leave a 0.2 seconds buffer to securely unpack and write solutions on timeouts
        return time.time() - self.start_time > self.time_limit - 0.2
        
    def solve(self):
        # Empty graph or zero possible edges automatically equates to an empty clique
        if not self.edges:
            self.update_best(0.0, [])
            return self.best_weight, self.best_clique
            
        # Fast generation of high quality bounds using GRASP-like heuristics
        self.heuristic()
        
        # Rigorous Enumeration framework: Branch-and-Bound Bron-Kerbosch
        C = []
        P = list(self.vertices)
        w_C = {v: 0.0 for v in self.vertices}
        X = []
        
        try:
            self.search(C, 0.0, P, w_C, X)
        except TimeoutError:
            pass
            
        return self.best_weight, self.best_clique
        
    def update_best(self, weight, clique):
        if weight > self.best_weight:
            self.best_weight = weight
            self.best_clique = list(clique)
            if self.logger:
                self.logger.log(float(self.best_weight))
                
    def heuristic(self):
        start_heur = time.time()
        max_heur_time = self.time_limit * 0.2
        if max_heur_time < 0.5:
            max_heur_time = self.time_limit - 0.5
            if max_heur_time <= 0:
                max_heur_time = 0.1
                
        iter_count = 0
        vertices_list = list(self.vertices)
        # Prioritize root candidates dynamically by incident weight topology
        v_weights = {v: sum(self.weights[(v, u)] for u in self.adj[v]) for v in vertices_list}
        vertices_list.sort(key=lambda v: v_weights[v], reverse=True)
        
        # 1. Deterministic pure-greedy sweeps
        for start_v in vertices_list:
            if time.time() - start_heur > max_heur_time:
                break
            self._greedy_build(start_v, randomize=False)
            iter_count += 1
            
        # 2. Randomized adaptive sweeps to escape early local-optima sets
        while time.time() - start_heur < max_heur_time:
            # Recompute randomization limits repeatedly evaluating clock drift limit
            start_v = random.choice(vertices_list)
            self._greedy_build(start_v, randomize=True)
            
    def _greedy_build(self, start_v, randomize=False):
        C = [start_v]
        curr_w = 0.0
        candidates = set(self.adj[start_v])
        
        while candidates:
            cand_scores = []
            for v in candidates:
                w_to_c = sum(self.weights[(v, u)] for u in C)
                w_to_cand = sum(self.weights[(v, u)] for u in candidates if u in self.adj[v])
                # Multi-metric node score incorporating clique weight vs forward depth potential 
                score = w_to_c + 0.3 * w_to_cand
                cand_scores.append((score, w_to_c, v))
                
            cand_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Simulated pseudo-random restricted choice phase 
            if randomize and len(cand_scores) > 2:
                limit = min(3, len(cand_scores))
                chosen = random.choice(cand_scores[:limit])
            else:
                chosen = cand_scores[0]
                
            v = chosen[2]
            w_to_c = chosen[1]
            
            C.append(v)
            curr_w += w_to_c
            candidates = candidates.intersection(self.adj[v])
            
        if curr_w > self.best_weight:
            self.update_best(curr_w, C)

    def search(self, C, current_weight, P, w_C, X):
        if self.check_time():
            raise TimeoutError()
            
        if not P and not X:
            return
            
        if not P:
            return
            
        # Optimization: Tomita pivot tracking to maximize graph coloring skips efficiently 
        best_u = None
        max_intersect = -1
        
        for u in P:
            intersect = 0
            u_adj = self.adj[u]
            for v in P:
                if v in u_adj:
                    intersect += 1
            if intersect > max_intersect:
                max_intersect = intersect
                best_u = u
                
        for x in X:
            intersect = 0
            x_adj = self.adj[x]
            for v in P:
                if v in x_adj:
                    intersect += 1
            if intersect > max_intersect:
                max_intersect = intersect
                best_u = x
                
        # Sub-branch subsumption pruning
        if max_intersect == len(P) and best_u in X:
            return
            
        P_iter = []
        if best_u is not None:
            best_u_adj = self.adj[best_u]
            for v in P:
                if v not in best_u_adj:
                    P_iter.append(v)
        else:
            P_iter = list(P)
            
        # Sequence generation targeting maximal bounds reduction earliest
        def v_score(v):
            score = w_C[v]
            v_adj = self.adj[v]
            for u in P:
                if u in v_adj:
                    score += self.weights[(v, u)]
            return score
            
        P_iter.sort(key=v_score, reverse=True)
        
        X_local = list(X)
        P_local = list(P)
        
        for v in P_iter:
            new_P = []
            new_w_C = {}
            v_adj = self.adj[v]
            for u in P_local:
                if u != v and u in v_adj:
                    new_P.append(u)
                    new_w_C[u] = w_C[u] + self.weights[(v, u)]
                    
            new_X = []
            for x in X_local:
                if x in v_adj:
                    new_X.append(x)
                    
            new_weight = current_weight + w_C[v]
            if new_weight > self.best_weight:
                self.update_best(new_weight, C + [v])
            
            # Inbound sub-tree topological capacity limits constraint check
            ub_branch = new_weight
            for u in new_P:
                ub_branch += new_w_C[u]
                
            np_len = len(new_P)
            for i_idx in range(np_len - 1):
                u1 = new_P[i_idx]
                u1_adj = self.adj[u1]
                for j_idx in range(i_idx + 1, np_len):
                    u2 = new_P[j_idx]
                    if u2 in u1_adj:
                        ub_branch += self.weights[(u1, u2)]
                        
            # Execute descent safely if ceiling limits aren't violated
            if ub_branch > self.best_weight:
                C.append(v)
                self.search(C, new_weight, new_P, new_w_C, new_X)
                C.pop()
                
            P_local.remove(v)
            X_local.append(v)

def main():
    parser = argparse.ArgumentParser(description="Maximize Edge-Weight Clique")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    
    args = parser.parse_args()
    
    # Secure solution external logging instance injection layer
    logger = None
    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
        
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    # Spin-Up analytical logic model limits 
    solver = MaxEdgeWeightClique(
        vertices=instance["vertices"],
        edges=instance["edges"],
        logger=logger,
        time_limit=args.time_limit
    )
    
    best_w, best_c = solver.solve()
    
    sol = {
        "objective_value": float(max(0.0, best_w)),
        "clique": best_c
    }
    
    # Synchronized JSON solution export phase output payload mapping formatting
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()