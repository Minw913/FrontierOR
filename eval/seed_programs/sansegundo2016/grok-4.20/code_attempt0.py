import json
import argparse
import time
from typing import List, Dict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path: str) -> Dict:
    with open(instance_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    graph = instance['graph']
    n = graph['num_vertices']
    adj = graph['adjacency_list']

    # Build adjacency matrix for quick lookup (0-based)
    is_adj = [[False] * n for _ in range(n)]
    for u_str, neighbors in adj.items():
        u = int(u_str)
        for v in neighbors:
            is_adj[u][v] = True
            is_adj[v][u] = True

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    best_obj = 0
    best_clique = []

    # Simple but effective heuristic: greedy with multiple random starts
    # Followed by exact MIP if time permits (with early termination)

    def run_greedy(random_order=False):
        if random_order:
            import random
            vertices = list(range(n))
            random.shuffle(vertices)
        else:
            # sort by degree descending
            degrees = [(len(adj[str(i)]), i) for i in range(n)]
            degrees.sort(reverse=True)
            vertices = [v for _, v in degrees]

        clique = []
        candidates = set(range(n))

        for v in vertices:
            if v not in candidates:
                continue
            clique.append(v)
            to_remove = []
            for c in candidates:
                if c != v and not is_adj[v][c]:
                    to_remove.append(c)
            for c in to_remove:
                candidates.remove(c)
            candidates.remove(v)

        return len(clique), sorted(clique)

    # Multiple greedy runs
    for _ in range(50):
        elapsed = time.time() - start_time
        if elapsed >= args.time_limit * 0.7:  # reserve time for MIP
            break
        obj, clique = run_greedy(random_order=True)
        if obj > best_obj:
            best_obj = obj
            best_clique = clique
            if logger:
                logger.log(best_obj)

    # Deterministic greedy
    obj, clique = run_greedy(random_order=False)
    if obj > best_obj:
        best_obj = obj
        best_clique = clique
        if logger:
            logger.log(best_obj)

    # Now solve with MIP using Gurobi if time remains
    elapsed = time.time() - start_time
    if elapsed < args.time_limit - 5:
        remaining_time = max(1, int(args.time_limit - elapsed))
        try:
            m = gp.Model("MaxClique")
            m.setParam('TimeLimit', remaining_time)
            m.setParam('Threads', 1)
            m.setParam('MIPFocus', 1)  # focus on finding feasible solutions
            m.setParam('Heuristics', 0.8)
            m.setParam('Cuts', 2)
            m.setParam('LogToConsole', 0)

            x = [m.addVar(vtype=GRB.BINARY, name=f'x_{i}') for i in range(n)]

            # Objective
            m.setObjective(gp.quicksum(x), GRB.MAXIMIZE)

            # Edge constraints: for every non-edge, at most one can be chosen
            for i in range(n):
                for j in range(i + 1, n):
                    if not is_adj[i][j]:
                        m.addConstr(x[i] + x[j] <= 1)

            # Optional: symmetry breaking - prefer higher degree vertices
            degrees = [len(adj[str(i)]) for i in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if degrees[i] < degrees[j]:
                        m.addConstr(x[i] <= x[j])  # weak symmetry break

            m.optimize()

            if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.FEASIBLE):
                if m.objVal > best_obj + 1e-6:
                    best_obj = int(round(m.objVal))
                    best_clique = [i for i in range(n) if x[i].X > 0.5]
                    if logger:
                        logger.log(best_obj)
        except Exception:
            # If Gurobi fails for any reason, fall back to heuristic solution
            pass

    # Final output
    solution = {
        "objective_value": int(best_obj),
        "clique_vertices": [int(v) for v in best_clique]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()