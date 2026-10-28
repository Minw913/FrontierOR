import json
import argparse
import time
import heapq
import sys


def main():
    parser = argparse.ArgumentParser(description="Steiner Network Optimization")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    m = instance['parameters']['m']
    n = instance['parameters']['n']
    node_weights = instance['steiner_nodes']['node_weights']
    assignment_costs = instance['cost_matrices']['assignment_costs']['matrix']
    interconnection_costs = instance['cost_matrices']['interconnection_costs']['matrix']

    # Handle trivial case: no targets
    if m == 0:
        solution = {
            "objective_value": 0.0,
            "active_steiner_nodes": [],
            "assignments": {},
            "steiner_tree_edges": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0.0, solution)
        return

    # Phase 1: Greedy heuristic for initial solution
    best_solution = greedy_heuristic(m, n, assignment_costs, interconnection_costs, node_weights)
    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    # Phase 2: MIP with Gurobi
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 2:
        try:
            mip_solution = solve_mip(m, n, assignment_costs, interconnection_costs,
                                     node_weights, remaining, logger, best_solution)
            if mip_solution is not None and mip_solution["objective_value"] < best_solution["objective_value"]:
                best_solution = mip_solution
                if logger:
                    logger.log_solution(best_solution["objective_value"], best_solution)
        except Exception:
            pass

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


def greedy_heuristic(m, n, assignment_costs, interconnection_costs, node_weights):
    """Greedy heuristic: try single nodes, then add/remove greedily."""
    # Find best single node
    best_obj = float('inf')
    best_active = [0]
    for j in range(n):
        cost = node_weights[j] + sum(assignment_costs[i][j] for i in range(m))
        if cost < best_obj:
            best_obj = cost
            best_active = [j]

    # Greedy addition
    active = list(best_active)
    improved = True
    while improved:
        improved = False
        current_cost = compute_total_cost(active, m, assignment_costs, interconnection_costs, node_weights)
        best_add = None
        best_add_cost = current_cost
        active_set = set(active)
        for j in range(n):
            if j not in active_set:
                new_active = active + [j]
                new_cost = compute_total_cost(new_active, m, assignment_costs, interconnection_costs, node_weights)
                if new_cost < best_add_cost:
                    best_add_cost = new_cost
                    best_add = j
        if best_add is not None:
            active.append(best_add)
            improved = True

    # Greedy removal
    improved = True
    while improved:
        improved = False
        current_cost = compute_total_cost(active, m, assignment_costs, interconnection_costs, node_weights)
        best_remove = None
        best_remove_cost = current_cost
        for j in list(active):
            if len(active) > 1:
                new_active = [k for k in active if k != j]
                new_cost = compute_total_cost(new_active, m, assignment_costs, interconnection_costs, node_weights)
                if new_cost < best_remove_cost:
                    best_remove_cost = new_cost
                    best_remove = j
        if best_remove is not None:
            active.remove(best_remove)
            improved = True

    return build_solution(active, m, assignment_costs, interconnection_costs, node_weights)


def compute_total_cost(active, m, assignment_costs, interconnection_costs, node_weights):
    """Compute total cost for a given set of active nodes."""
    if len(active) == 0:
        return float('inf')
    cost = sum(node_weights[j] for j in active)
    for i in range(m):
        cost += min(assignment_costs[i][j] for j in active)
    if len(active) > 1:
        cost += mst_cost(active, interconnection_costs)
    return cost


def mst_cost(active, interconnection_costs):
    """Compute MST cost using Prim's algorithm."""
    if len(active) <= 1:
        return 0.0
    in_tree = {active[0]}
    heap = [(interconnection_costs[active[0]][j], j) for j in active if j != active[0]]
    heapq.heapify(heap)
    total = 0.0
    while len(in_tree) < len(active) and heap:
        w, node = heapq.heappop(heap)
        if node in in_tree:
            continue
        in_tree.add(node)
        total += w
        for j in active:
            if j not in in_tree:
                heapq.heappush(heap, (interconnection_costs[node][j], j))
    return total


def build_mst_edges(active, interconnection_costs):
    """Build MST edges using Prim's algorithm."""
    if len(active) <= 1:
        return []
    in_tree = {active[0]}
    heap = [(interconnection_costs[active[0]][j], active[0], j) for j in active if j != active[0]]
    heapq.heapify(heap)
    edges = []
    while len(in_tree) < len(active) and heap:
        w, u, v = heapq.heappop(heap)
        if v in in_tree:
            continue
        in_tree.add(v)
        edges.append([u, v])
        for j in active:
            if j not in in_tree:
                heapq.heappush(heap, (interconnection_costs[v][j], v, j))
    return edges


def build_solution(active, m, assignment_costs, interconnection_costs, node_weights):
    """Build a complete solution dictionary from a set of active nodes."""
    assignments = {}
    for i in range(m):
        best_j = min(active, key=lambda j: assignment_costs[i][j])
        assignments[str(i)] = best_j
    tree_edges = build_mst_edges(active, interconnection_costs)
    obj = compute_total_cost(active, m, assignment_costs, interconnection_costs, node_weights)
    return {
        "objective_value": obj,
        "active_steiner_nodes": sorted(active),
        "assignments": assignments,
        "steiner_tree_edges": tree_edges
    }


def solve_mip(m, n, assignment_costs, interconnection_costs, node_weights,
              time_limit, logger, initial_solution):
    """Solve using Gurobi MIP with lazy connectivity constraints."""
    from gurobipy import Model, GRB, quicksum

    model = Model("steiner_network")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    e = {}
    for j in range(n):
        for k in range(j + 1, n):
            e[j, k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"e_{j}_{k}")

    model.update()

    # Objective
    obj_expr = quicksum(assignment_costs[i][j] * x[i, j] for i in range(m) for j in range(n))
    obj_expr += quicksum(interconnection_costs[j][k] * e[j, k] for j in range(n) for k in range(j + 1, n))
    obj_expr += quicksum(node_weights[j] * y[j] for j in range(n))
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Each target assigned to exactly one Steiner node
    for i in range(m):
        model.addConstr(quicksum(x[i, j] for j in range(n)) == 1, name=f"assign_{i}")

    # Assignment only to active nodes
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"link_{i}_{j}")

    # Aggregated linking constraints
    for j in range(n):
        model.addConstr(quicksum(x[i, j] for i in range(m)) <= m * y[j], name=f"agg_{j}")

    # Edge only between active nodes
    for j in range(n):
        for k in range(j + 1, n):
            model.addConstr(e[j, k] <= y[j], name=f"ey_{j}_{k}_j")
            model.addConstr(e[j, k] <= y[k], name=f"ey_{j}_{k}_k")

    # Tree size constraint
    model.addConstr(
        quicksum(e[j, k] for j in range(n) for k in range(j + 1, n)) ==
        quicksum(y[j] for j in range(n)) - 1,
        name="tree_size"
    )

    # Warm start from initial solution
    if initial_solution:
        active_set = set(initial_solution["active_steiner_nodes"])
        edge_set = set()
        for edge in initial_solution["steiner_tree_edges"]:
            edge_set.add((min(edge[0], edge[1]), max(edge[0], edge[1])))

        for j in range(n):
            y[j].Start = 1.0 if j in active_set else 0.0
        for i in range(m):
            assigned_to = initial_solution["assignments"].get(str(i), -1)
            for j in range(n):
                x[i, j].Start = 1.0 if j == assigned_to else 0.0
        for j in range(n):
            for k in range(j + 1, n):
                e[j, k].Start = 1.0 if (j, k) in edge_set else 0.0

    # Track best solution
    best_obj = [initial_solution["objective_value"] if initial_solution else float('inf')]
    best_sol = [initial_solution]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            y_sol = model.cbGetSolution(y)
            e_sol = model.cbGetSolution(e)

            active = [j for j in range(n) if y_sol[j] > 0.5]

            if len(active) > 1:
                active_set = set(active)
                adj = {j: set() for j in active}

                for j in range(n):
                    for k in range(j + 1, n):
                        if e_sol[j, k] > 0.5:
                            if j in active_set and k in active_set:
                                adj[j].add(k)
                                adj[k].add(j)

                # BFS connectivity check
                visited = set()
                queue = [active[0]]
                visited.add(active[0])
                while queue:
                    node = queue.pop(0)
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)

                if len(visited) < len(active):
                    # Disconnected: add connectivity cut
                    S = visited
                    T = active_set - S
                    cut_vars = []
                    for j in S:
                        for k in T:
                            if j < k:
                                cut_vars.append(e[j, k])
                            else:
                                cut_vars.append(e[k, j])
                    model.cbLazy(quicksum(cut_vars) >= 1)
                    return

            # Feasible solution - check if it improves best
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-9:
                x_sol = model.cbGetSolution(x)
                sol = extract_solution(m, n, y_sol, x_sol, e_sol,
                                       assignment_costs, interconnection_costs, node_weights)
                if sol is not None:
                    best_obj[0] = obj_val
                    best_sol[0] = sol
                    if logger:
                        logger.log_solution(obj_val, sol)

    model.setParam('LazyConstraints', 1)
    model.optimize(callback)

    # If callback didn't capture best, try model solution
    if model.SolCount > 0:
        try:
            y_val = {j: y[j].X for j in range(n)}
            x_val = {(i, j): x[i, j].X for i in range(m) for j in range(n)}
            e_val = {(j, k): e[j, k].X for j in range(n) for k in range(j + 1, n)}
            sol = extract_solution(m, n, y_val, x_val, e_val,
                                   assignment_costs, interconnection_costs, node_weights)
            if sol is not None and sol["objective_value"] < best_obj[0]:
                best_obj[0] = sol["objective_value"]
                best_sol[0] = sol
                if logger:
                    logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass

    return best_sol[0]


def extract_solution(m, n, y_val, x_val, e_val, assignment_costs, interconnection_costs, node_weights):
    """Extract a solution dictionary from variable values."""
    active = [j for j in range(n) if y_val[j] > 0.5]
    if len(active) == 0:
        return None

    active_set = set(active)
    assignments = {}
    for i in range(m):
        best_j = active[0]
        best_val = -1.0
        for j in active:
            key = (i, j)
            val = x_val[key] if key in x_val else 0.0
            if val > best_val:
                best_val = val
                best_j = j
        assignments[str(i)] = best_j

    tree_edges = []
    for j in range(n):
        for k in range(j + 1, n):
            key = (j, k)
            val = e_val[key] if key in e_val else 0.0
            if val > 0.5:
                tree_edges.append([j, k])

    # Compute objective from scratch for accuracy
    obj = sum(node_weights[j] for j in active)
    for i in range(m):
        obj += assignment_costs[i][assignments[str(i)]]
    for edge in tree_edges:
        obj += interconnection_costs[edge[0]][edge[1]]

    return {
        "objective_value": obj,
        "active_steiner_nodes": sorted(active),
        "assignments": assignments,
        "steiner_tree_edges": tree_edges
    }


if __name__ == "__main__":
    main()