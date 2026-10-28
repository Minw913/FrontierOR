import argparse
import json
import time
import sys
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    vertices = graph['vertices']
    edges = graph['edges']

    start_time = time.time()

    # Build adjacency structures
    adj = defaultdict(set)
    edge_weight = {}
    for e in edges:
        u, v, w = e[0], e[1], e[2]
        adj[u].add(v)
        adj[v].add(u)
        a, b = min(u, v), max(u, v)
        edge_weight[(a, b)] = w

    vertex_set = set(vertices)

    def clique_weight(clique_list):
        total = 0
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j])
                total += edge_weight.get((a, b), 0)
        return total

    def get_clique_edges(clique_list):
        result = []
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j])
                result.append([a, b])
        return result

    # Compute weighted degree for each vertex
    weighted_deg = defaultdict(float)
    for e in edges:
        u, v, w = e[0], e[1], e[2]
        weighted_deg[u] += w
        weighted_deg[v] += w

    best_clique = []
    best_obj = 0.0

    def update_best(clique):
        nonlocal best_clique, best_obj
        obj = clique_weight(clique)
        if obj > best_obj:
            best_obj = obj
            best_clique = list(clique)
            if logger:
                logger.log(best_obj)

    # Start with each vertex as a single clique
    if vertices:
        update_best([vertices[0]])

    # Greedy construction: start from each vertex, greedily add vertex maximizing weight gain
    def greedy_clique(start_vertex):
        clique = [start_vertex]
        candidates = set(adj[start_vertex])
        while candidates:
            best_v = None
            best_gain = -1
            for v in candidates:
                gain = 0
                for c in clique:
                    a, b = min(v, c), max(v, c)
                    gain += edge_weight.get((a, b), 0)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v is None or best_gain <= 0:
                break
            clique.append(best_v)
            candidates = candidates & adj[best_v]
        return clique

    # Sort vertices by weighted degree descending for greedy starts
    sorted_verts = sorted(vertices, key=lambda v: weighted_deg[v], reverse=True)

    # Phase 1: Greedy from top vertices
    num_greedy_starts = min(len(sorted_verts), max(50, num_vertices))
    for i, v in enumerate(sorted_verts[:num_greedy_starts]):
        if time.time() - start_time > args.time_limit * 0.3:
            break
        c = greedy_clique(v)
        update_best(c)

    # Phase 2: Try to use Gurobi for smaller instances or as improvement
    remaining_time = args.time_limit - (time.time() - start_time)
    
    if remaining_time > 2 and num_vertices <= 5000:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("MEWCP")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining_time - 1))
            model.setParam('Threads', 1)
            
            # Vertex variables
            x = {}
            for v in vertices:
                x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

            # Edge variables and objective
            y = {}
            obj_expr = 0
            edge_list = []
            for e in edges:
                u, v, w = e[0], e[1], e[2]
                a, b = min(u, v), max(u, v)
                if (a, b) not in y:
                    y[(a, b)] = model.addVar(vtype=GRB.BINARY, name=f"y_{a}_{b}")
                    edge_list.append((a, b, w))
                    obj_expr += w * y[(a, b)]

            model.setObjective(obj_expr, GRB.MAXIMIZE)

            # Edge linking constraints: y_e <= x_u, y_e <= x_v
            for a, b, w in edge_list:
                model.addConstr(y[(a, b)] <= x[a])
                model.addConstr(y[(a, b)] <= x[b])

            # Clique constraints: for non-edges, x_u + x_v <= 1
            # Build edge set for quick lookup
            edge_set = set()
            for e in edges:
                u, v = e[0], e[1]
                a, b = min(u, v), max(u, v)
                edge_set.add((a, b))

            # For large graphs, we can't enumerate all non-edges
            # Instead use a smarter approach: complement graph clique constraints
            # But for moderate sizes, enumerate non-edges
            if num_vertices <= 500:
                for i in range(len(vertices)):
                    for j in range(i + 1, len(vertices)):
                        a, b = min(vertices[i], vertices[j]), max(vertices[i], vertices[j])
                        if (a, b) not in edge_set:
                            model.addConstr(x[a] + x[b] <= 1)
            else:
                # Use lazy constraints or just add for vertices with missing edges
                # Add constraints only for neighbors of selected vertices
                # Use a different approach: for each vertex, sum of selected non-neighbors <= 0 if vertex is selected
                # x[v] + x[u] <= 1 for non-edges
                # This is too many. Use separation approach via callbacks or add most important ones.
                
                # Alternative: add constraints for non-edges involving high-degree vertices
                # Actually, let's use a different formulation with callbacks
                
                # For now, add non-edge constraints in batches
                non_edge_count = 0
                max_non_edges = 500000
                for i in range(len(sorted_verts)):
                    if non_edge_count >= max_non_edges:
                        break
                    vi = sorted_verts[i]
                    for j in range(i + 1, len(sorted_verts)):
                        vj = sorted_verts[j]
                        a, b = min(vi, vj), max(vi, vj)
                        if (a, b) not in edge_set:
                            model.addConstr(x[a] + x[b] <= 1)
                            non_edge_count += 1
                            if non_edge_count >= max_non_edges:
                                break

            # Warm start with best known clique
            for v in vertices:
                x[v].Start = 1 if v in best_clique else 0
            for (a, b), var in y.items():
                var.Start = 1 if (a in best_clique and b in best_clique) else 0

            model.optimize()

            if model.SolCount > 0:
                sol_verts = [v for v in vertices if x[v].X > 0.5]
                # Verify it's a clique
                is_clique = True
                for i in range(len(sol_verts)):
                    for j in range(i + 1, len(sol_verts)):
                        a, b = min(sol_verts[i], sol_verts[j]), max(sol_verts[i], sol_verts[j])
                        if (a, b) not in edge_set:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    update_best(sol_verts)

        except Exception as e:
            pass

    # Phase 3: Local search improvement
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 1:
        edge_set_quick = set()
        for e in edges:
            u, v = e[0], e[1]
            edge_set_quick.add((min(u, v), max(u, v)))

        def is_connected_to_all(v, clique_set):
            for c in clique_set:
                a, b = min(v, c), max(v, c)
                if (a, b) not in edge_set_quick:
                    return False
            return True

        def add_weight(v, clique_list):
            w = 0
            for c in clique_list:
                a, b = min(v, c), max(v, c)
                w += edge_weight.get((a, b), 0)
            return w

        import random
        random.seed(42)

        current_clique = list(best_clique)
        current_obj = best_obj

        iterations = 0
        while time.time() - start_time < args.time_limit - 0.5:
            iterations += 1
            improved = False

            # Try swap: remove one vertex, add another
            clique_set = set(current_clique)
            
            # Try adding a vertex
            candidates_add = []
            for v in vertices:
                if v not in clique_set and is_connected_to_all(v, clique_set):
                    candidates_add.append(v)
            
            if candidates_add:
                best_add = None
                best_add_gain = 0
                for v in candidates_add:
                    gain = add_weight(v, current_clique)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = v
                if best_add is not None:
                    current_clique.append(best_add)
                    current_obj += best_add_gain
                    update_best(current_clique)
                    improved = True
                    continue

            # Try swap: remove one, add one or two
            if len(current_clique) > 1:
                best_swap_gain = 0
                best_swap = None  # (remove_idx, add_vertices)
                
                for ridx in range(len(current_clique)):
                    if time.time() - start_time > args.time_limit - 0.5:
                        break
                    removed = current_clique[ridx]
                    remaining = [current_clique[k] for k in range(len(current_clique)) if k != ridx]
                    remaining_set = set(remaining)
                    loss = add_weight(removed, remaining)

                    # Find vertices connectable to remaining
                    cands = []
                    for v in vertices:
                        if v not in clique_set and is_connected_to_all(v, remaining_set):
                            cands.append(v)

                    # Try adding one
                    for v in cands:
                        gain = add_weight(v, remaining) - loss
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (ridx, [v])

                    # Try adding two
                    if len(cands) <= 200:
                        for ci in range(len(cands)):
                            for cj in range(ci + 1, len(cands)):
                                va, vb = cands[ci], cands[cj]
                                a2, b2 = min(va, vb), max(va, vb)
                                if (a2, b2) in edge_set_quick:
                                    gain = add_weight(va, remaining) + add_weight(vb, remaining) + edge_weight.get((a2, b2), 0) - loss
                                    if gain > best_swap_gain:
                                        best_swap_gain = gain
                                        best_swap = (ridx, [va, vb])

                if best_swap is not None and best_swap_gain > 0:
                    ridx, add_vs = best_swap
                    current_clique = [current_clique[k] for k in range(len(current_clique)) if k != ridx] + add_vs
                    current_obj = clique_weight(current_clique)
                    update_best(current_clique)
                    improved = True
                    continue

            if not improved:
                # Random perturbation: remove a random vertex and try rebuilding
                if len(current_clique) > 2:
                    ridx = random.randint(0, len(current_clique) - 1)
                    current_clique = [current_clique[k] for k in range(len(current_clique)) if k != ridx]
                    current_obj = clique_weight(current_clique)
                    # Try greedy expansion
                    clique_set = set(current_clique)
                    while True:
                        best_v = None
                        best_gain = 0
                        for v in vertices:
                            if v not in clique_set and is_connected_to_all(v, clique_set):
                                gain = add_weight(v, current_clique)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_v = v
                        if best_v is None:
                            break
                        current_clique.append(best_v)
                        clique_set.add(best_v)
                        current_obj += best_gain
                    update_best(current_clique)
                else:
                    # Restart from a random vertex
                    v = random.choice(sorted_verts[:max(1, len(sorted_verts)//2)])
                    current_clique = greedy_clique(v)
                    current_obj = clique_weight(current_clique)
                    update_best(current_clique)

    # Output
    best_clique_sorted = sorted(best_clique)
    clique_edges = get_clique_edges(best_clique_sorted)

    solution = {
        "objective_value": best_obj,
        "clique_vertices": best_clique_sorted,
        "clique_edges": clique_edges
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()