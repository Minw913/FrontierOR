import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # -----------------------------
    # 1. Read instance
    # -----------------------------
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    n = graph['num_vertices']
    m = graph['num_edges']
    vertex_ids = graph['vertices']
    edges_raw = graph['edges']

    # Map vertex identifiers to 0..n-1
    id_to_idx = {vid: i for i, vid in enumerate(vertex_ids)}
    idx_to_id = {i: vid for i, vid in enumerate(vertex_ids)}

    # Build adjacency and weight dictionary
    adj = [set() for _ in range(n)]
    weight_dict = {}
    edges = []
    for e in edges_raw:
        u_id, v_id, w = e[0], e[1], e[2]
        u = id_to_idx[u_id]
        v = id_to_idx[v_id]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in weight_dict:
            # Keep the maximum weight if duplicate edges appear
            if w > weight_dict[(u, v)]:
                weight_dict[(u, v)] = w
        else:
            weight_dict[(u, v)] = w
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v, w))

    m_actual = len(edges)

    # -----------------------------
    # 2. Heuristic
    # -----------------------------
    start_time = time.time()
    best_clique = [0] if n > 0 else []
    best_obj = 0.0

    # Total incident weight per vertex (for ordering starts)
    incident_weight = [0.0] * n
    for (u, v, w) in edges:
        incident_weight[u] += w
        incident_weight[v] += w

    # Choose starting vertices for greedy
    if n <= 500:
        start_vertices = list(range(n))
    else:
        start_vertices = sorted(range(n), key=lambda i: incident_weight[i], reverse=True)[:100]

    def greedy_from(start):
        clique = [start]
        candidates = set(adj[start])
        obj = 0.0
        while candidates:
            best_v = None
            best_gain = -1.0
            for v in candidates:
                gain = 0.0
                for u in clique:
                    key = (u, v) if u < v else (v, u)
                    w = weight_dict.get(key)
                    if w is not None:
                        gain += w
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v is None or best_gain <= 0:
                break
            clique.append(best_v)
            obj += best_gain
            candidates.intersection_update(adj[best_v])
            candidates.discard(best_v)
        return clique, obj

    # Run greedy with time budget
    heuristic_time_budget = min(10.0, max(1.0, args.time_limit * 0.2))
    for sv in start_vertices:
        if time.time() - start_time > heuristic_time_budget:
            break
        c, o = greedy_from(sv)
        if o > best_obj:
            best_obj = o
            best_clique = c

    # Local search: try to add any vertex that connects to all clique members
    def try_add(clique, obj):
        changed = True
        while changed:
            changed = False
            if not clique:
                break
            candidates = set(range(n)) - set(clique)
            for u in clique:
                candidates &= adj[u]
            for v in candidates:
                gain = 0.0
                for u in clique:
                    key = (u, v) if u < v else (v, u)
                    w = weight_dict.get(key)
                    if w is not None:
                        gain += w
                if gain > 1e-6:
                    clique.append(v)
                    obj += gain
                    changed = True
                    break
        return clique, obj

    if best_clique:
        best_clique, best_obj = try_add(best_clique, best_obj)

    # Log heuristic incumbent
    if best_clique and logger:
        clique_vertices_ids = sorted(idx_to_id[i] for i in best_clique)
        clique_edges_ids = []
        for i in range(len(best_clique)):
            for j in range(i + 1, len(best_clique)):
                u, v = best_clique[i], best_clique[j]
                if v in adj[u]:
                    clique_edges_ids.append([idx_to_id[min(u, v)], idx_to_id[max(u, v)]])
        sol_dict = {
            "objective_value": float(best_obj),
            "clique_vertices": clique_vertices_ids,
            "clique_edges": clique_edges_ids
        }
        logger.log_solution(float(best_obj), sol_dict)

    # -----------------------------
    # 3. Exact MIP (Gurobi)
    # -----------------------------
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    if remaining_time > 1 and n > 0:
        model = gp.Model()
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining_time
        model.Params.LazyConstraints = 1
        model.Params.OutputFlag = 0

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(m_actual, vtype=GRB.BINARY, name="y")

        # Linking constraints
        for i, (u, v, w) in enumerate(edges):
            model.addConstr(y[i] <= x[u])
            model.addConstr(y[i] <= x[v])
            model.addConstr(y[i] >= x[u] + x[v] - 1)

        model.setObjective(gp.quicksum(w * y[i] for i, (u, v, w) in enumerate(edges)), GRB.MAXIMIZE)

        # MIP start
        if best_clique:
            for i in range(n):
                x[i].start = 1.0 if i in best_clique else 0.0
            for i, (u, v, w) in enumerate(edges):
                if u in best_clique and v in best_clique:
                    y[i].start = 1.0
                else:
                    y[i].start = 0.0

        # Store data for callback
        model._x = x
        model._adj = adj
        model._idx_to_id = idx_to_id
        model._logger = logger
        model._best_obj = float(best_obj)

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                x_vals = model.cbGetSolution(model._x)
                selected = [i for i in range(n) if x_vals[i] > 0.5]
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                violated = False
                # Check all pairs; add lazy cuts for non-edges
                for ii in range(len(selected)):
                    u = selected[ii]
                    for jj in range(ii + 1, len(selected)):
                        v = selected[jj]
                        if v not in model._adj[u]:
                            model.cbLazy(model._x[u] + model._x[v] <= 1)
                            violated = True
                if not violated and obj_val > model._best_obj + 1e-9:
                    model._best_obj = obj_val
                    if model._logger:
                        clique_vertices_ids = sorted(model._idx_to_id[i] for i in selected)
                        clique_edges_ids = []
                        for i in range(len(selected)):
                            for j in range(i + 1, len(selected)):
                                u, v = selected[i], selected[j]
                                if v in model._adj[u]:
                                    clique_edges_ids.append([model._idx_to_id[min(u, v)], model._idx_to_id[max(u, v)]])
                        sol_dict = {
                            "objective_value": float(obj_val),
                            "clique_vertices": clique_vertices_ids,
                            "clique_edges": clique_edges_ids
                        }
                        model._logger.log_solution(float(obj_val), sol_dict)

        model.optimize(callback)

        if model.SolCount > 0:
            x_vals = model.getAttr('X', x)
            selected = [i for i in range(n) if x_vals[i] > 0.5]
            # Recompute exact objective from the solution
            obj = 0.0
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    u, v = selected[i], selected[j]
                    if v in adj[u]:
                        key = (u, v) if u < v else (v, u)
                        obj += weight_dict[key]
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_clique = selected

    # -----------------------------
    # 4. Final output
    # -----------------------------
    clique_vertices_ids = sorted(idx_to_id[i] for i in best_clique)
    clique_edges_ids = []
    obj_final = 0.0
    for i in range(len(best_clique)):
        for j in range(i + 1, len(best_clique)):
            u, v = best_clique[i], best_clique[j]
            if v in adj[u]:
                key = (u, v) if u < v else (v, u)
                w = weight_dict[key]
                obj_final += w
                clique_edges_ids.append([idx_to_id[min(u, v)], idx_to_id[max(u, v)]])

    solution = {
        "objective_value": float(obj_final),
        "clique_vertices": clique_vertices_ids,
        "clique_edges": clique_edges_ids
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()