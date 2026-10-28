import json
import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="Maximum Weight Clique Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance['graph']['num_vertices']
    num_edges = instance['graph']['num_edges']
    vertices = instance['graph']['vertices']
    edges = instance['graph']['edges']
    vertex_weights = instance['vertex_weights']

    # Build adjacency structures
    adj_set = set()
    adj_list = {v: set() for v in vertices}
    for u, v in edges:
        key = (min(u, v), max(u, v))
        adj_set.add(key)
        adj_list[u].add(v)
        adj_list[v].add(u)

    def clique_weight(clique):
        return sum(vertex_weights[v] for v in clique)

    def make_maximal(clique):
        """Extend clique to be maximal by adding compatible vertices."""
        clique_set = set(clique)
        for v in sorted(vertices, key=lambda v: vertex_weights[v], reverse=True):
            if v not in clique_set and all(v in adj_list[u] for u in clique):
                clique.append(v)
                clique_set.add(v)
        return clique

    # --- Greedy heuristics with multiple orderings ---
    best_clique = []
    best_weight = 0

    orderings = [
        sorted(vertices, key=lambda v: vertex_weights[v], reverse=True),
        sorted(vertices, key=lambda v: (vertex_weights[v], len(adj_list[v])), reverse=True),
        sorted(vertices, key=lambda v: len(adj_list[v]), reverse=True),
        sorted(vertices, key=lambda v: vertex_weights[v] * len(adj_list[v]), reverse=True),
        sorted(vertices, key=lambda v: vertex_weights[v] / max(len(vertices) - len(adj_list[v]), 1), reverse=True),
    ]

    for ordering in orderings:
        if time.time() - start_time > args.time_limit * 0.2:
            break
        clique = []
        for v in ordering:
            if all(v in adj_list[u] for u in clique):
                clique.append(v)
        # Make it maximal
        clique = make_maximal(clique)
        w = clique_weight(clique)
        if w > best_weight:
            best_weight = w
            best_clique = clique[:]
            if logger:
                logger.log_solution(best_weight, {
                    "objective_value": float(best_weight),
                    "clique_vertices": best_clique
                })

    # --- Local search: try swaps to improve ---
    if best_clique and (time.time() - start_time) < args.time_limit * 0.3:
        improved = True
        while improved and (time.time() - start_time) < args.time_limit * 0.3:
            improved = False
            clique_set = set(best_clique)
            candidates = [v for v in vertices if v not in clique_set]
            # Try 1-for-1 swaps
            for i in range(len(best_clique)):
                v_out = best_clique[i]
                for v_in in candidates:
                    if all(v_in in adj_list[u] for u in best_clique if u != v_out):
                        new_weight = best_weight - vertex_weights[v_out] + vertex_weights[v_in]
                        if new_weight > best_weight:
                            best_clique[i] = v_in
                            best_weight = new_weight
                            clique_set.remove(v_out)
                            clique_set.add(v_in)
                            candidates = [v for v in vertices if v not in clique_set]
                            improved = True
                            break
                if improved:
                    break
            # Try adding
            if not improved:
                for v in candidates:
                    if all(v in adj_list[u] for u in best_clique):
                        best_clique.append(v)
                        best_weight += vertex_weights[v]
                        improved = True
                        break

        if logger:
            logger.log_solution(best_weight, {
                "objective_value": float(best_weight),
                "clique_vertices": best_clique
            })

    # --- Gurobi MIP ---
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("max_weight_clique")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(remaining - 2, 1))
            model.setParam('OutputFlag', 0)

            # Decision variables
            x = {}
            for v in vertices:
                x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

            # Objective: maximize total weight
            model.setObjective(
                gp.quicksum(vertex_weights[v] * x[v] for v in vertices),
                GRB.MAXIMIZE
            )

            # Constraints: for each non-edge (u, v), at most one can be selected
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    if (min(u, v), max(u, v)) not in adj_set:
                        model.addConstr(x[u] + x[v] <= 1)

            # Warm start with best heuristic solution
            best_set = set(best_clique)
            for v in vertices:
                x[v].Start = 1 if v in best_set else 0

            # Callback to log incumbent improvements
            current_best = [best_weight]

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > current_best[0]:
                        current_best[0] = obj
                        sol_clique = [v for v in vertices if model.cbGetSolution(x[v]) > 0.5]
                        if logger:
                            logger.log_solution(obj, {
                                "objective_value": float(obj),
                                "clique_vertices": sol_clique
                            })

            model.optimize(callback)

            # Extract solution
            if model.SolCount > 0:
                gurobi_clique = [v for v in vertices if x[v].X > 0.5]
                gurobi_weight = clique_weight(gurobi_clique)
                if gurobi_weight > best_weight:
                    best_weight = gurobi_weight
                    best_clique = gurobi_clique

        except Exception as e:
            print(f"Gurobi error: {e}", file=sys.stderr)

    # --- Final log and output ---
    if logger:
        logger.log_solution(best_weight, {
            "objective_value": float(best_weight),
            "clique_vertices": best_clique
        })

    solution = {
        "objective_value": float(best_weight),
        "clique_vertices": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()