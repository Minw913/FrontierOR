import json
import argparse
import time
import os
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    vertices = instance['vertices']
    edges_data = instance['edges']

    # Build adjacency list and edge weights
    adj = {v: set() for v in vertices}
    weight = {}
    for e in edges_data:
        i = e['i']
        j = e['j']
        w = e['weight']
        adj[i].add(j)
        adj[j].add(i)
        weight[(min(i,j), max(i,j))] = w

    n = len(vertices)
    vlist = sorted(vertices)
    idx = {vlist[i]: i for i in range(n)}

    # Greedy randomized adaptive search procedure (GRASP) with local search
    best_clique = []
    best_obj = 0.0
    start_time = time.time()
    max_iter = 1000
    alpha = 0.5
    elapsed = 0.0

    def compute_clique_weight(clq):
        if len(clq) < 2:
            return 0.0
        total = 0.0
        for i in range(len(clq)):
            for j in range(i+1, len(clq)):
                a, b = clq[i], clq[j]
                if (min(a,b), max(a,b)) in weight:
                    total += weight[(min(a,b), max(a,b))]
        return total

    def is_clique(clq):
        for i in range(len(clq)):
            for j in range(i+1, len(clq)):
                if clq[j] not in adj[clq[i]]:
                    return False
        return True

    def local_search(current):
        improved = True
        clq_set = set(current)
        while improved and (time.time() - start_time < args.time_limit):
            improved = False
            candidates = []
            for v in vertices:
                if v not in clq_set:
                    if all(nb in clq_set for nb in adj[v] if nb in clq_set) and all(v in adj[u] for u in clq_set):
                        gain = sum(weight.get((min(v,u), max(v,u)), 0) for u in clq_set)
                        candidates.append((gain, v))
            if candidates:
                candidates.sort(reverse=True)
                best_gain, best_v = candidates[0]
                if best_gain > 0:
                    clq_set.add(best_v)
                    improved = True
        return list(clq_set)

    while elapsed < args.time_limit and max_iter > 0:
        max_iter -= 1
        # Greedy randomized construction
        C = []
        candidates = list(vertices)
        while candidates:
            gains = []
            for v in candidates:
                gain = sum(weight.get((min(v,u), max(v,u)), 0) for u in C)
                gains.append((gain, v))
            if not gains:
                break
            gains.sort(reverse=True)
            max_gain = gains[0][0]
            min_gain = gains[-1][0]
            threshold = max_gain - alpha * (max_gain - min_gain)
            rcl = [v for g, v in gains if g >= threshold]
            # pick randomly from RCL
            import random
            chosen = random.choice(rcl)
            C.append(chosen)
            # update candidates: only those connected to all in C
            new_candidates = []
            for v in candidates:
                if v == chosen:
                    continue
                if all(v in adj[u] for u in C):
                    new_candidates.append(v)
            candidates = new_candidates

        if not is_clique(C):
            # this should not happen with the filter above, but safety
            C = []

        # Local search
        improved_clique = local_search(C)
        obj = compute_clique_weight(improved_clique)

        if obj > best_obj + 1e-6:
            best_obj = obj
            best_clique = improved_clique[:]
            if logger:
                logger.log(float(best_obj))

        elapsed = time.time() - start_time
        if elapsed >= args.time_limit:
            break

    # Final greedy improvement on best
    if best_clique:
        final_clique = local_search(best_clique)
        final_obj = compute_clique_weight(final_clique)
        if final_obj > best_obj:
            best_obj = final_obj
            best_clique = final_clique
            if logger:
                logger.log(float(best_obj))

    # If nothing found, take highest weight single edge or single vertex
    if not best_clique:
        max_w = 0
        best_edge = None
        for (i,j), w in weight.items():
            if w > max_w:
                max_w = w
                best_edge = (i,j)
        if best_edge:
            best_clique = list(best_edge)
            best_obj = max_w
            if logger:
                logger.log(float(best_obj))
        elif vertices:
            best_clique = [vertices[0]]
            best_obj = 0.0

    solution = {
        "objective_value": float(best_obj),
        "clique": [int(v) for v in sorted(best_clique)]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and best_obj > 0:
        logger.log(float(best_obj))

if __name__ == "__main__":
    # Ensure random seed is somewhat reproducible but diverse
    import random
    random.seed(42)
    main()