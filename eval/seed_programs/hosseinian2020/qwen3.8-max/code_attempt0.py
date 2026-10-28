import json
import argparse
import time
from solution_logger import SolutionLogger


def greedy_heuristic(vertices, edges, adj, weight_map):
    """Greedy heuristic to find a good initial clique."""
    if not edges:
        return [], 0.0

    # Compute for each vertex, the sum of weights of incident edges
    vertex_weight = {v: 0 for v in vertices}
    for (i, j, w) in edges:
        vertex_weight[i] += w
        vertex_weight[j] += w

    sorted_vertices = sorted(vertices, key=lambda v: vertex_weight[v], reverse=True)

    best_clique = []
    best_value = 0.0

    # Try starting from top vertices
    num_starts = min(50, len(sorted_vertices))
    for start_v in sorted_vertices[:num_starts]:
        current_clique = {start_v}
        current_value = 0.0

        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_vertex = None

            for v in sorted_vertices:
                if v in current_clique:
                    continue
                # Check if v is connected to all vertices in current clique
                can_add = True
                for u in current_clique:
                    if (min(v, u), max(v, u)) not in adj:
                        can_add = False
                        break

                if can_add:
                    gain = sum(weight_map.get((min(v, u), max(v, u)), 0) for u in current_clique)
                    if gain > best_gain:
                        best_gain = gain
                        best_vertex = v

            if best_vertex is not None and best_gain > 0:
                current_clique.add(best_vertex)
                current_value += best_gain
                improved = True

        if current_value > best_value:
            best_value = current_value
            best_clique = list(current_clique)

    return best_clique, best_value


def local_search(clique, vertices, edges, adj, weight_map):
    """Try to improve clique by swapping vertices."""
    if not clique:
        return clique, 0.0

    clique_set = set(clique)
    current_value = sum(w for (i, j, w) in edges if i in clique_set and j in clique_set)

    improved = True
    while improved:
        improved = False
        # Try removing a vertex and adding another
        for v in list(clique_set):
            remaining = clique_set - {v}
            if not remaining:
                continue
            # Value without v
            val_without_v = sum(w for (i, j, w) in edges if i in remaining and j in remaining)

            # Try adding each non-clique vertex
            for u in vertices:
                if u in clique_set:
                    continue
                # Check if u connects to all in remaining
                can_add = True
                for r in remaining:
                    if (min(u, r), max(u, r)) not in adj:
                        can_add = False
                        break
                if can_add:
                    new_clique = remaining | {u}
                    new_val = sum(w for (i, j, w) in edges if i in new_clique and j in new_clique)
                    if new_val > current_value:
                        clique_set = new_clique
                        current_value = new_val
                        improved = True
                        break
            if improved:
                break

    return list(clique_set), current_value


def solve_with_gurobi(vertices, edges, adj, time_limit, logger, initial_clique=None):
    """Solve using Gurobi MIP."""
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.start()

    model = gp.Model(env=env)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)

    # Decision variables
    x = {}
    for v in vertices:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f'x_{v}')

    y = {}
    edge_list = []
    for (i, j, w) in edges:
        key = (min(i, j), max(i, j))
        if key not in y:
            y[key] = model.addVar(vtype=GRB.BINARY, name=f'y_{key[0]}_{key[1]}')
            edge_list.append((i, j, w, key))

    model.update()

    # Objective: maximize sum of weights of included edges
    model.setObjective(
        gp.quicksum(w * y[key] for (i, j, w, key) in edge_list),
        GRB.MAXIMIZE
    )

    # Constraints: y_e <= x_i and y_e <= x_j
    for (i, j, w, key) in edge_list:
        model.addConstr(y[key] <= x[i])
        model.addConstr(y[key] <= x[j])

    # Constraints: for non-edges, x_i + x_j <= 1
    edge_set = set(y.keys())
    n = len(vertices)
    for idx1 in range(n):
        for idx2 in range(idx1 + 1, n):
            v1, v2 = vertices[idx1], vertices[idx2]
            if (min(v1, v2), max(v1, v2)) not in edge_set:
                model.addConstr(x[v1] + x[v2] <= 1)

    model.update()

    # Set initial solution if available
    if initial_clique:
        for v in vertices:
            x[v].Start = 1.0 if v in initial_clique else 0.0
        clique_set = set(initial_clique)
        for (i, j, w, key) in edge_list:
            y[key].Start = 1.0 if (i in clique_set and j in clique_set) else 0.0

    # Callback to log solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            clique = [v for v in vertices if model.cbGetSolution(x[v]) > 0.5]
            sol_dict = {
                "objective_value": float(obj),
                "clique": clique
            }
            if logger:
                logger.log_solution(obj, sol_dict)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        clique = [v for v in vertices if x[v].X > 0.5]
        clique_set = set(clique)
        obj_value = 0.0
        for (i, j, w) in edges:
            if i in clique_set and j in clique_set:
                obj_value += w
        return clique, obj_value

    return [], 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance['num_vertices']
    num_edges = instance['num_edges']
    vertices = instance['vertices']
    edges = [(e['i'], e['j'], e['weight']) for e in instance['edges']]

    # Build adjacency set and weight map
    adj = set()
    weight_map = {}
    for (i, j, w) in edges:
        key = (min(i, j), max(i, j))
        adj.add(key)
        weight_map[key] = w

    # Greedy heuristic for initial solution
    greedy_clique, greedy_value = greedy_heuristic(vertices, edges, adj, weight_map)

    # Local search improvement
    if greedy_clique:
        improved_clique, improved_value = local_search(
            greedy_clique, vertices, edges, adj, weight_map
        )
        if improved_value > greedy_value:
            greedy_clique = improved_clique
            greedy_value = improved_value

    if logger and greedy_value > 0:
        sol_dict = {
            "objective_value": float(greedy_value),
            "clique": greedy_clique
        }
        logger.log_solution(greedy_value, sol_dict)

    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    best_clique = greedy_clique
    best_value = greedy_value

    if remaining_time > 2:
        # Use Gurobi for exact/near-optimal solution
        try:
            gurobi_clique, gurobi_value = solve_with_gurobi(
                vertices, edges, adj, remaining_time - 1, logger,
                initial_clique=set(greedy_clique) if greedy_clique else None
            )

            if gurobi_value > best_value:
                best_clique = gurobi_clique
                best_value = gurobi_value
        except Exception:
            pass

    # Write solution
    solution = {
        "objective_value": float(best_value),
        "clique": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()