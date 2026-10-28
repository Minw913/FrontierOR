import argparse
import json
import time
import heapq
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    source = instance['source_node']
    target = instance['target_node']
    arcs = instance['arcs']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']

    # Build adjacency lists
    out_arcs = [[] for _ in range(num_nodes)]
    in_arcs = [[] for _ in range(num_nodes)]
    for arc in arcs:
        out_arcs[arc['from_node']].append(arc['id'])
        in_arcs[arc['to_node']].append(arc['id'])

    def compute_objective(path):
        lin = sum(linear_costs[a] for a in path)
        quad = 0
        for i in range(len(path)):
            for j in range(len(path)):
                quad += quadratic_costs[path[i]][path[j]]
        return lin + quad

    def make_solution_dict(path):
        obj = compute_objective(path)
        return {
            "objective_value": obj,
            "solution_arcs": [
                {"id": arcs[a]['id'], "from_node": arcs[a]['from_node'], "to_node": arcs[a]['to_node']}
                for a in path
            ]
        }

    # Heuristic: Dijkstra by linear cost
    def dijkstra_linear():
        dist = [float('inf')] * num_nodes
        dist[source] = 0
        prev_arc = [-1] * num_nodes
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == target:
                break
            for a_id in out_arcs[u]:
                v = arcs[a_id]['to_node']
                nd = d + linear_costs[a_id]
                if nd < dist[v]:
                    dist[v] = nd
                    prev_arc[v] = a_id
                    heapq.heappush(pq, (nd, v))
        if dist[target] == float('inf'):
            return None
        path = []
        node = target
        while node != source:
            a_id = prev_arc[node]
            if a_id == -1:
                return None
            path.append(a_id)
            node = arcs[a_id]['from_node']
        path.reverse()
        return path

    best_path = dijkstra_linear()
    best_obj = compute_objective(best_path) if best_path else float('inf')

    if best_path and logger:
        logger.log_solution(best_obj, make_solution_dict(best_path))

    # Try to improve heuristic with local search (2-opt style on path)
    def local_search(path, max_iter=100):
        if path is None:
            return path
        current_path = list(path)
        current_obj = compute_objective(current_path)
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            if time.time() - start_time > args.time_limit * 0.3:
                break
            improved = False
            iteration += 1
            # Try replacing subpaths
            # Build node sequence from path
            nodes_seq = [source]
            for a in current_path:
                nodes_seq.append(arcs[a]['to_node'])
            
            # Try all pairs of positions to reconnect
            for i in range(len(nodes_seq) - 1):
                for j in range(i + 2, len(nodes_seq)):
                    # Try to find alternative path from nodes_seq[i] to nodes_seq[j]
                    # using BFS/shortest path
                    s_node = nodes_seq[i]
                    t_node = nodes_seq[j]
                    # Find shortest path from s_node to t_node by linear cost
                    d = [float('inf')] * num_nodes
                    d[s_node] = 0
                    prev = [-1] * num_nodes
                    pq2 = [(0, s_node)]
                    while pq2:
                        dd, u = heapq.heappop(pq2)
                        if dd > d[u]:
                            continue
                        if u == t_node:
                            break
                        for a_id in out_arcs[u]:
                            v = arcs[a_id]['to_node']
                            nd = dd + linear_costs[a_id]
                            if nd < d[v]:
                                d[v] = nd
                                prev[v] = a_id
                                heapq.heappush(pq2, (nd, v))
                    if d[t_node] < float('inf'):
                        # Reconstruct subpath
                        subpath = []
                        node = t_node
                        while node != s_node:
                            a_id = prev[node]
                            subpath.append(a_id)
                            node = arcs[a_id]['from_node']
                        subpath.reverse()
                        # Check no repeated nodes
                        sub_nodes = set([s_node])
                        valid = True
                        n = s_node
                        for a in subpath:
                            n = arcs[a]['to_node']
                            if n in sub_nodes:
                                valid = False
                                break
                            sub_nodes.add(n)
                        if not valid:
                            continue
                        # Check no overlap with rest of path
                        rest_nodes = set(nodes_seq[:i+1]) | set(nodes_seq[j:])
                        if sub_nodes & rest_nodes - {s_node, t_node}:
                            continue
                        new_path = current_path[:i] + subpath + current_path[j-1:]
                        # Verify it's a valid path
                        new_nodes = [source]
                        for a in new_path:
                            new_nodes.append(arcs[a]['to_node'])
                        if new_nodes[-1] == target and len(set(new_nodes)) == len(new_nodes):
                            new_obj = compute_objective(new_path)
                            if new_obj < current_obj:
                                current_path = new_path
                                current_obj = new_obj
                                improved = True
                                break
                if improved:
                    break
        return current_path

    if best_path and time.time() - start_time < args.time_limit * 0.3:
        improved_path = local_search(best_path, max_iter=20)
        if improved_path:
            improved_obj = compute_objective(improved_path)
            if improved_obj < best_obj:
                best_path = improved_path
                best_obj = improved_obj
                if logger:
                    logger.log_solution(best_obj, make_solution_dict(best_path))

    # MIQP with Gurobi
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 2:
        try:
            from gurobipy import *

            env = Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Seed", 0)
            env.setParam("MIPGap", 1e-4)
            env.setParam("NumericFocus", 0)
            env.setParam("Threads", 1)
            env.setParam("TimeLimit", max(1, remaining_time - 2))
            env.start()

            model = Model("QSPP", env=env)

            # Binary variables for arc selection
            x = model.addVars(num_arcs, vtype=GRB.BINARY, name="x")

            # MTZ variables for subtour elimination
            u = model.addVars(num_nodes, lb=0, ub=num_nodes - 1, vtype=GRB.CONTINUOUS, name="u")

            # Objective: linear + quadratic
            obj_expr = LinExpr()
            for a in range(num_arcs):
                obj_expr += linear_costs[a] * x[a]

            qexpr = QuadExpr()
            for i in range(num_arcs):
                qi = quadratic_costs[i]
                for j in range(num_arcs):
                    if qi[j] != 0:
                        qexpr += qi[j] * x[i] * x[j]

            model.setObjective(obj_expr + qexpr, GRB.MINIMIZE)

            # Flow conservation constraints
            for v in range(num_nodes):
                if v == source:
                    model.addConstr(quicksum(x[a] for a in out_arcs[v]) == 1, name=f"src_out")
                    if in_arcs[v]:
                        model.addConstr(quicksum(x[a] for a in in_arcs[v]) == 0, name=f"src_in")
                elif v == target:
                    model.addConstr(quicksum(x[a] for a in in_arcs[v]) == 1, name=f"tgt_in")
                    if out_arcs[v]:
                        model.addConstr(quicksum(x[a] for a in out_arcs[v]) == 0, name=f"tgt_out")
                else:
                    model.addConstr(
                        quicksum(x[a] for a in out_arcs[v]) - quicksum(x[a] for a in in_arcs[v]) == 0,
                        name=f"flow_{v}"
                    )

            # Degree constraints for simple path
            for v in range(num_nodes):
                if v != source and in_arcs[v]:
                    model.addConstr(quicksum(x[a] for a in in_arcs[v]) <= 1, name=f"in_deg_{v}")
                if v != target and out_arcs[v]:
                    model.addConstr(quicksum(x[a] for a in out_arcs[v]) <= 1, name=f"out_deg_{v}")

            # MTZ constraints for cycle elimination
            model.addConstr(u[source] == 0, name="mtz_source")
            for arc in arcs:
                i, j = arc['from_node'], arc['to_node']
                a_id = arc['id']
                if j != source:
                    model.addConstr(
                        u[i] - u[j] + num_nodes * x[a_id] <= num_nodes - 1,
                        name=f"mtz_{a_id}"
                    )

            # Warm start
            if best_path:
                path_set = set(best_path)
                for a in range(num_arcs):
                    x[a].Start = 1.0 if a in path_set else 0.0
                # Set MTZ values based on path order
                if best_path:
                    node_order = {source: 0}
                    pos = 1
                    for a in best_path:
                        node_order[arcs[a]['to_node']] = pos
                        pos += 1
                    for v in range(num_nodes):
                        if v in node_order:
                            u[v].Start = node_order[v]
                        else:
                            u[v].Start = 0

            # Callback to log incumbent solutions
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    sol_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    sol_x = model.cbGetSolution(x)
                    selected = [a for a in range(num_arcs) if sol_x[a] > 0.5]
                    sol_dict = {
                        "objective_value": sol_obj,
                        "solution_arcs": [
                            {"id": arcs[a]['id'], "from_node": arcs[a]['from_node'], "to_node": arcs[a]['to_node']}
                            for a in selected
                        ]
                    }
                    if logger:
                        logger.log_solution(sol_obj, sol_dict)

            model.optimize(callback)

            if model.SolCount > 0:
                sol_x = model.getAttr("X", x)
                selected = [a for a in range(num_arcs) if sol_x[a] > 0.5]
                obj_val = model.ObjVal

                if obj_val < best_obj:
                    best_obj = obj_val
                    best_path = selected

                solution_dict = {
                    "objective_value": obj_val,
                    "solution_arcs": [
                        {"id": arcs[a]['id'], "from_node": arcs[a]['from_node'], "to_node": arcs[a]['to_node']}
                        for a in selected
                    ]
                }
                if logger:
                    logger.log_solution(obj_val, solution_dict)

            env.dispose()

        except Exception as e:
            # If Gurobi fails, use heuristic solution
            pass

    # Write final solution
    if best_path:
        solution_dict = make_solution_dict(best_path)
    else:
        solution_dict = {"objective_value": 0, "solution_arcs": []}

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == "__main__":
    main()