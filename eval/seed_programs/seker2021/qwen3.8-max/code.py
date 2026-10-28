import json
import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description='Solve cluster selection and coloring on perfect graph')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path) as f:
        instance = json.load(f)

    num_vertices = instance['graph']['num_vertices']
    edges = instance['graph']['edges']
    num_clusters = instance['partition']['num_clusters']
    clusters = instance['partition']['clusters']

    # Build adjacency structure
    adj = [set() for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def greedy_color(vertex_set, order=None):
        """Greedy coloring of a set of vertices."""
        if order is None:
            order = sorted(vertex_set)
        color = {}
        for v in order:
            used = set()
            for u in adj[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return color

    def num_colors_used(color):
        if not color:
            return 0
        return max(color.values()) + 1

    def make_solution(selected_list, color, obj):
        return {
            "objective_value": float(obj),
            "selected_vertices": {str(k): selected_list[k] for k in range(num_clusters)},
            "coloring": {str(v): color[v] for v in selected_list}
        }

    # Phase 1: Greedy selection - pick vertex with fewest conflicts to already selected
    selected = []
    selected_set = set()
    for k in range(num_clusters):
        best_v = None
        best_conflicts = float('inf')
        for v in clusters[k]:
            conflicts = len(adj[v] & selected_set)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_v = v
        selected.append(best_v)
        selected_set.add(best_v)

    # Try multiple coloring orders to get best initial coloring
    best_coloring = None
    best_obj = float('inf')

    # Order 1: sorted by vertex index
    c1 = greedy_color(selected_set)
    o1 = num_colors_used(c1)
    if o1 < best_obj:
        best_obj = o1
        best_coloring = c1

    # Order 2: by degree within selected subgraph (descending)
    order2 = sorted(selected_set, key=lambda v: len(adj[v] & selected_set), reverse=True)
    c2 = greedy_color(selected_set, order2)
    o2 = num_colors_used(c2)
    if o2 < best_obj:
        best_obj = o2
        best_coloring = c2

    # Order 3: by degree in original graph (descending)
    order3 = sorted(selected_set, key=lambda v: len(adj[v]), reverse=True)
    c3 = greedy_color(selected_set, order3)
    o3 = num_colors_used(c3)
    if o3 < best_obj:
        best_obj = o3
        best_coloring = c3

    best_selected = selected[:]
    best_solution = make_solution(best_selected, best_coloring, best_obj)

    if logger:
        logger.log_solution(best_obj, best_solution)

    # Phase 2: Local search - try swapping vertices within clusters
    local_deadline = start_time + min(args.time_limit * 0.3, max(args.time_limit - 3, 1))

    improved = True
    while improved and time.time() < local_deadline:
        improved = False
        for k in range(num_clusters):
            if time.time() >= local_deadline:
                break
            current_v = selected[k]
            for v in clusters[k]:
                if v == current_v:
                    continue
                if time.time() >= local_deadline:
                    break
                selected[k] = v
                new_set = set(selected)

                # Try multiple coloring orders
                new_coloring = greedy_color(new_set)
                new_obj = num_colors_used(new_coloring)

                # Also try degree-ordered coloring
                ord_d = sorted(new_set, key=lambda x: len(adj[x] & new_set), reverse=True)
                alt_coloring = greedy_color(new_set, ord_d)
                alt_obj = num_colors_used(alt_coloring)
                if alt_obj < new_obj:
                    new_obj = alt_obj
                    new_coloring = alt_coloring

                if new_obj < best_obj:
                    best_obj = new_obj
                    best_selected = selected[:]
                    best_coloring = dict(new_coloring)
                    improved = True
                    best_solution = make_solution(best_selected, best_coloring, best_obj)
                    if logger:
                        logger.log_solution(best_obj, best_solution)

                selected[k] = current_v

    # Phase 3: MIP with Gurobi
    remaining = args.time_limit - (time.time() - start_time)

    if remaining > 2 and best_obj > 1:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            max_colors = best_obj

            # Check if problem size is tractable
            num_y_vars = num_vertices * max_colors
            if num_y_vars <= 2000000:
                model = gp.Model()
                model.setParam('Seed', 0)
                model.setParam('MIPGap', 1e-4)
                model.setParam('NumericFocus', 0)
                model.setParam('Threads', 1)
                model.setParam('TimeLimit', max(remaining - 1, 1))
                model.setParam('OutputFlag', 0)

                # Decision variables
                x = {v: model.addVar(vtype=GRB.BINARY, name=f"x_{v}")
                     for v in range(num_vertices)}

                y = {(v, c): model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{c}")
                     for v in range(num_vertices) for c in range(max_colors)}

                z = {c: model.addVar(vtype=GRB.BINARY, name=f"z_{c}")
                     for c in range(max_colors)}

                # Objective: minimize number of colors used
                model.setObjective(gp.quicksum(z[c] for c in range(max_colors)), GRB.MINIMIZE)

                # Exactly one vertex per cluster
                for k in range(num_clusters):
                    model.addConstr(
                        gp.quicksum(x[v] for v in clusters[k]) == 1,
                        name=f"cluster_{k}"
                    )

                # Each selected vertex gets exactly one color
                for v in range(num_vertices):
                    model.addConstr(
                        gp.quicksum(y[v, c] for c in range(max_colors)) == x[v],
                        name=f"assign_{v}"
                    )

                # Proper coloring: adjacent selected vertices get different colors
                for u, v in edges:
                    for c in range(max_colors):
                        model.addConstr(
                            y[u, c] + y[v, c] <= 1,
                            name=f"edge_{u}_{v}_c{c}"
                        )

                # Link y and z: color c is used if any vertex gets it
                for v in range(num_vertices):
                    for c in range(max_colors):
                        model.addConstr(y[v, c] <= z[c], name=f"link_{v}_{c}")

                # Symmetry breaking: use colors in order
                for c in range(max_colors - 1):
                    model.addConstr(z[c] >= z[c + 1], name=f"sym_{c}")

                # Warm start from heuristic solution
                sel_set = set(best_selected)
                for v in range(num_vertices):
                    if v in sel_set:
                        x[v].Start = 1
                        cv = best_coloring[v]
                        for c in range(max_colors):
                            y[v, c].Start = 1 if c == cv else 0
                    else:
                        x[v].Start = 0
                        for c in range(max_colors):
                            y[v, c].Start = 0

                for c in range(max_colors):
                    z[c].Start = 1 if c < best_obj else 0

                # State tracking for callback
                state = {
                    'best_obj': best_obj,
                    'best_selected': best_selected[:],
                    'best_coloring': dict(best_coloring),
                    'best_solution': best_solution
                }

                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if obj_val < state['best_obj'] - 0.5:
                            x_vals = model.cbGetSolution(x)
                            y_vals = model.cbGetSolution(y)

                            sel = []
                            for k in range(num_clusters):
                                for v in clusters[k]:
                                    if x_vals[v] > 0.5:
                                        sel.append(v)
                                        break

                            col = {}
                            for v in sel:
                                for c in range(max_colors):
                                    if y_vals[v, c] > 0.5:
                                        col[v] = c
                                        break

                            new_obj = int(round(obj_val))
                            state['best_obj'] = new_obj
                            state['best_selected'] = sel
                            state['best_coloring'] = col
                            state['best_solution'] = make_solution(sel, col, new_obj)

                            if logger:
                                logger.log_solution(new_obj, state['best_solution'])

                model.optimize(callback)

                # Extract final solution from model
                if model.SolCount > 0:
                    obj_val = model.ObjVal
                    if obj_val < state['best_obj'] - 0.5:
                        sel = []
                        for k in range(num_clusters):
                            for v in clusters[k]:
                                if x[v].X > 0.5:
                                    sel.append(v)
                                    break

                        col = {}
                        for v in sel:
                            for c in range(max_colors):
                                if y[v, c].X > 0.5:
                                    col[v] = c
                                    break

                        new_obj = int(round(obj_val))
                        state['best_obj'] = new_obj
                        state['best_selected'] = sel
                        state['best_coloring'] = col
                        state['best_solution'] = make_solution(sel, col, new_obj)

                        if logger:
                            logger.log_solution(new_obj, state['best_solution'])

                best_obj = state['best_obj']
                best_selected = state['best_selected']
                best_coloring = state['best_coloring']
                best_solution = state['best_solution']

        except Exception as e:
            print(f"MIP error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()