import argparse
import json
import os
import time

from solution_logger import SolutionLogger


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    best_value = 0.0
    best_clique = []
    best_solution = {
        "objective_value": 0.0,
        "clique": []
    }

    if logger:
        logger.log_solution(0.0, best_solution)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = list(data.get("vertices", []))
    n = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Consolidate accidental duplicate edge records by retaining the last weight.
    edge_map = {}
    for edge in data.get("edges", []):
        li = edge["i"]
        lj = edge["j"]
        if li == lj or li not in label_to_index or lj not in label_to_index:
            continue
        u = label_to_index[li]
        v = label_to_index[lj]
        if u > v:
            u, v = v, u
        edge_map[(u, v)] = int(edge["weight"])

    edges = [(u, v, w) for (u, v), w in edge_map.items()]
    m = len(edges)

    adjacency = [set() for _ in range(n)]
    weight_to_neighbor = [dict() for _ in range(n)]
    weighted_degree = [0 for _ in range(n)]

    for u, v, w in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
        weight_to_neighbor[u][v] = w
        weight_to_neighbor[v][u] = w
        weighted_degree[u] += w
        weighted_degree[v] += w

    def evaluate_indices(clique_indices):
        """Return the exact objective, or None if the vertices are not a clique."""
        total = 0
        for p, u in enumerate(clique_indices):
            wu = weight_to_neighbor[u]
            for v in clique_indices[p + 1:]:
                w = wu.get(v)
                if w is None:
                    return None
                total += w
        return float(total)

    def accept_solution(clique_indices):
        nonlocal best_value, best_clique, best_solution

        clique_indices = sorted(set(clique_indices))
        value = evaluate_indices(clique_indices)
        if value is None or value <= best_value + 1e-7:
            return False

        best_value = value
        best_clique = clique_indices
        best_solution = {
            "objective_value": float(value),
            "clique": [labels[i] for i in clique_indices]
        }
        if logger:
            logger.log_solution(float(value), best_solution)
        return True

    def greedy_complete(initial):
        clique = list(initial)
        if not clique:
            return clique

        if len(clique) == 1:
            candidates = set(adjacency[clique[0]])
        else:
            candidates = set(adjacency[clique[0]])
            for u in clique[1:]:
                candidates.intersection_update(adjacency[u])
            candidates.difference_update(clique)

        while candidates and time.monotonic() < heuristic_deadline:
            best_vertex = None
            best_increment = -1
            best_tiebreak = -1

            for v in candidates:
                increment = 0
                wv = weight_to_neighbor[v]
                for u in clique:
                    increment += wv[u]

                tie = weighted_degree[v]
                if (increment > best_increment or
                        (increment == best_increment and tie > best_tiebreak) or
                        (increment == best_increment and tie == best_tiebreak and
                         (best_vertex is None or v < best_vertex))):
                    best_vertex = v
                    best_increment = increment
                    best_tiebreak = tie

            if best_vertex is None:
                break

            clique.append(best_vertex)
            candidates.intersection_update(adjacency[best_vertex])
            candidates.discard(best_vertex)

        return clique

    # Spend only a limited fraction of the budget on deterministic construction.
    remaining_initial = max(0.0, deadline - time.monotonic())
    heuristic_budget = min(3.0, max(0.02, remaining_initial * 0.12))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    if n > 0 and time.monotonic() < heuristic_deadline:
        degree_order = sorted(
            range(n),
            key=lambda i: (weighted_degree[i], len(adjacency[i]), -i),
            reverse=True
        )

        if n <= 500:
            starts = degree_order
        else:
            starts = degree_order[:150]

        heavy_edges = sorted(edges, key=lambda e: (e[2], weighted_degree[e[0]] +
                                                   weighted_degree[e[1]]),
                             reverse=True)[:100]
        start_set = set(starts)
        for u, v, _ in heavy_edges:
            start_set.add(u)
            start_set.add(v)

        starts = sorted(
            start_set,
            key=lambda i: (weighted_degree[i], len(adjacency[i]), -i),
            reverse=True
        )

        for s in starts:
            if time.monotonic() >= heuristic_deadline:
                break
            accept_solution(greedy_complete([s]))

        # Forced heavy-edge starts can avoid a poor first greedy choice.
        for u, v, _ in heavy_edges:
            if time.monotonic() >= heuristic_deadline:
                break
            accept_solution(greedy_complete([u, v]))

    # If no optimization time remains, return the best heuristic incumbent.
    if time.monotonic() >= deadline or n == 0 or m == 0:
        write_solution(args.solution_path, best_solution)
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("maximum_edge_weight_clique")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        # Edge variables are continuous because the AND constraints make them
        # integral whenever their endpoint variables are integral.
        y = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

        for e, (u, v, _) in enumerate(edges):
            model.addConstr(y[e] <= x[u])
            model.addConstr(y[e] <= x[v])
            model.addConstr(y[e] >= x[u] + x[v] - 1)

        model.setObjective(
            gp.quicksum(float(w) * y[e] for e, (_, _, w) in enumerate(edges)),
            GRB.MAXIMIZE
        )

        possible_pairs = n * (n - 1) // 2
        nonedge_count = possible_pairs - m

        # Explicit nonedge constraints are strong, but can be prohibitively
        # numerous on large sparse graphs. For such cases, use an exact compact
        # cardinality formulation.
        remaining = max(0.0, deadline - time.monotonic())
        explicit_threshold = 1_500_000 if remaining >= 20.0 else 350_000
        use_explicit_nonedges = nonedge_count <= explicit_threshold

        z = None
        if use_explicit_nonedges:
            for u in range(n):
                neighbors = adjacency[u]
                for v in range(u + 1, n):
                    if v not in neighbors:
                        model.addConstr(x[u] + x[v] <= 1)
        else:
            # A selected set of k vertices is a clique exactly when it contains
            # choose(k,2) graph edges. y is the indicator of each existing
            # selected edge, so this gives a compact exact formulation.
            z = model.addVars(n + 1, vtype=GRB.BINARY, name="cardinality")
            model.addConstr(gp.quicksum(z[k] for k in range(n + 1)) == 1)
            model.addConstr(
                gp.quicksum(x[i] for i in range(n)) ==
                gp.quicksum(k * z[k] for k in range(n + 1))
            )
            model.addConstr(
                gp.quicksum(y[e] for e in range(m)) ==
                gp.quicksum((k * (k - 1) // 2) * z[k]
                            for k in range(n + 1))
            )

        # Warm start from the best heuristic solution.
        selected_start = set(best_clique)
        for i in range(n):
            x[i].Start = 1.0 if i in selected_start else 0.0

        for e, (u, v, _) in enumerate(edges):
            y[e].Start = 1.0 if u in selected_start and v in selected_start else 0.0

        if z is not None:
            start_size = len(selected_start)
            for k in range(n + 1):
                z[k].Start = 1.0 if k == start_size else 0.0

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            model.Params.TimeLimit = max(0.001, remaining)

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution(x)
                    clique_indices = [i for i in range(n) if values[i] > 0.5]
                    accept_solution(clique_indices)
                except gp.GurobiError:
                    return

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                final_indices = [i for i in range(n) if x[i].X > 0.5]
                accept_solution(final_indices)

    except Exception:
        # The heuristic solution remains valid if model construction, licensing,
        # or optimization fails.
        pass

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()