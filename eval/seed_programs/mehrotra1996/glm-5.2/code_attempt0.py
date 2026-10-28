import argparse
import json
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    graph = instance["graph"]
    nodes = graph["nodes"]
    edges = [tuple(e) for e in graph["edges"]]
    num_nodes = len(nodes)
    num_edges = len(edges)

    start_time = time.time()

    # Greedy DSATUR Algorithm
    def dsatur():
        adj = {v: set() for v in nodes}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        color = {}
        saturation = {v: 0 for v in nodes}
        degree = {v: len(adj[v]) for v in nodes}
        uncolored = set(nodes)

        while uncolored:
            # Pick vertex with max saturation, break ties by degree
            v = max(uncolored, key=lambda x: (saturation[x], degree[x]))
            used_colors = {color[u] for u in adj[v] if u in color}
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            uncolored.remove(v)
            # Update saturation for neighbors
            for u in adj[v]:
                if u in uncolored:
                    saturation[u] = len({color[w] for w in adj[u] if w in color})
        return color

    current_color = dsatur()
    ub = max(current_color.values()) + 1 if current_color else 0

    def get_solution_dict(color):
        obj_val = max(color.values()) + 1 if color else 0
        color_classes = [[] for _ in range(obj_val)]
        for v, c in color.items():
            color_classes[c].append(v)
        return {
            "objective_value": obj_val,
            "vertex_colors": {str(v): c for v, c in color.items()},
            "color_classes": color_classes
        }

    if logger:
        logger.log_solution(ub, get_solution_dict(current_color))

    # Iterative improvement
    while ub > 1 and time.time() - start_time < args.time_limit - 1:
        remaining = args.time_limit - (time.time() - start_time) - 1
        if remaining <= 0:
            break
            
        k = ub - 1
        found = False

        # Try Gurobi if problem size is manageable
        if num_nodes * k <= 200000 and num_edges * k <= 1000000:
            try:
                import gurobipy as gp
                from gurobipy import GRB
                model = gp.Model("k_coloring")
                model.setParam("Seed", 0)
                model.setParam("MIPGap", 1e-4)
                model.setParam("NumericFocus", 0)
                model.setParam("Threads", 1)
                model.setParam("TimeLimit", max(1, remaining * 0.9))
                
                x = {}
                for v in nodes:
                    for c in range(k):
                        x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
                
                for v in nodes:
                    model.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1)
                
                for u, v in edges:
                    for c in range(k):
                        model.addConstr(x[u, c] + x[v, c] <= 1)
                
                model.setObjective(0, GRB.MINIMIZE)
                model.optimize()
                
                if model.status in [GRB.Status.OPTIMAL, GRB.Status.FEASIBLE]:
                    found = True
                    current_color = {}
                    for v in nodes:
                        for c in range(k):
                            if x[v, c].X > 0.5:
                                current_color[v] = c
                                break
                    ub = k
                    if logger:
                        logger.log_solution(ub, get_solution_dict(current_color))
                elif model.status == GRB.Status.INFEASIBLE:
                    break
            except Exception:
                pass

        if not found and time.time() - start_time < args.time_limit - 1:
            remaining = args.time_limit - (time.time() - start_time) - 1
            # Fallback to Tabu Search Local Search
            adj = {v: set() for v in nodes}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            random.seed(0)
            ls_color = {v: random.randint(0, k-1) for v in nodes}
            conflicts = 0
            for u, v in edges:
                if ls_color[u] == ls_color[v]:
                    conflicts += 1
            best_conflicts = conflicts
            best_color = ls_color.copy()
            tabu_list = {}
            tabu_tenure = 10
            iteration = 0
            ls_start = time.time()
            
            while time.time() - ls_start < remaining and best_conflicts > 0:
                iteration += 1
                best_delta = float('inf')
                best_moves = []
                conflicting_vertices = []
                for u, v in edges:
                    if ls_color[u] == ls_color[v]:
                        conflicting_vertices.append(u)
                if not conflicting_vertices:
                    break
                v = random.choice(conflicting_vertices)
                old_c = ls_color[v]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    delta = 0
                    for neighbor in adj[v]:
                        if ls_color[neighbor] == old_c:
                            delta -= 1
                        if ls_color[neighbor] == new_c:
                            delta += 1
                    is_tabu = (v, new_c) in tabu_list and tabu_list[(v, new_c)] > iteration
                    if is_tabu and conflicts + delta >= best_conflicts:
                        continue
                    if delta < best_delta:
                        best_delta = delta
                        best_moves = [(v, new_c)]
                    elif delta == best_delta:
                        best_moves.append((v, new_c))
                if best_moves:
                    v, new_c = random.choice(best_moves)
                    ls_color[v] = new_c
                    conflicts += best_delta
                    tabu_list[(v, old_c)] = iteration + tabu_tenure
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_color = ls_color.copy()
                        
            if best_conflicts == 0:
                found = True
                current_color = best_color
                ub = k
                if logger:
                    logger.log_solution(ub, get_solution_dict(current_color))
            else:
                break

    solution = get_solution_dict(current_color)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()