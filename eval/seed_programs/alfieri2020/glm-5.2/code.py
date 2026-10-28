import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def dsatur(num_nodes, edges):
    """DSATUR heuristic for graph coloring."""
    adj = [set() for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    color = [-1] * num_nodes
    neighbor_colors = [set() for _ in range(num_nodes)]
    degrees = [len(adj[i]) for i in range(num_nodes)]
    uncolored = set(range(num_nodes))

    while uncolored:
        # pick vertex with highest saturation, tie-break by degree
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), degrees[x]))
        c = 0
        while c in neighbor_colors[v]:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if color[u] == -1:
                neighbor_colors[u].add(c)

    if num_nodes == 0:
        return {}, 0
    max_c = max(color)
    return color, max_c + 1


def greedy_clique(num_nodes, edges):
    """Simple greedy maximum clique heuristic for lower bound."""
    adj = [set() for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(num_nodes)]
    order = sorted(range(num_nodes), key=lambda x: degrees[x], reverse=True)
    best = []
    for v in order:
        clique = [v]
        for u in order:
            if u == v:
                continue
            if all(u in adj[w] for w in clique):
                clique.append(u)
        if len(clique) > len(best):
            best = clique
    return len(best)


def solve_mip(num_nodes, edges, init_color, init_num_colors,
              total_time_limit, logger, start_time):
    """MIP to minimize number of colors used."""
    UB = init_num_colors
    if num_nodes == 0 or UB == 0:
        return 0, {}

    remaining = total_time_limit - (time.time() - start_time)
    if remaining <= 0:
        return init_num_colors, {str(i): init_color[i] for i in range(num_nodes)}

    model = gp.Model("graph_coloring")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", max(0.0, remaining))
    model.setParam("OutputFlag", 0)

    x = model.addVars(num_nodes, UB, vtype=GRB.BINARY, name="x")
    y = model.addVars(UB, vtype=GRB.BINARY, name="y")

    # each vertex gets exactly one color
    for v in range(num_nodes):
        model.addConstr(gp.quicksum(x[v, c] for c in range(UB)) == 1)

    # adjacent vertices must not share a color
    for u, v in edges:
        for c in range(UB):
            model.addConstr(x[u, c] + x[v, c] <= 1)

    # link x and y
    for v in range(num_nodes):
        for c in range(UB):
            model.addConstr(x[v, c] <= y[c])

    # symmetry breaking: colors are used in decreasing order
    for c in range(UB - 1):
        model.addConstr(y[c] >= y[c + 1])

    # lower bound from greedy clique
    lb = greedy_clique(num_nodes, edges)
    if lb > 1:
        model.addConstr(y.sum() >= lb)

    model.setObjective(y.sum(), GRB.MINIMIZE)

    # MIP start from heuristic
    if init_color:
        for v in range(num_nodes):
            c = init_color[v]
            for cc in range(UB):
                x[v, cc].Start = 1.0 if cc == c else 0.0
        used = set(init_color)
        for c in range(UB):
            y[c].Start = 1.0 if c in used else 0.0

    best_obj = init_num_colors
    best_coloring = {str(i): init_color[i] for i in range(num_nodes)}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            nonlocal best_obj
            if obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(obj)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        if obj_val < best_obj - 1e-6:
            best_obj = obj_val
            x_vals = model.getAttr("X", x)
            coloring = {}
            for v in range(num_nodes):
                for c in range(UB):
                    if x_vals[v, c] > 0.5:
                        coloring[str(v)] = c
                        break
            best_coloring = coloring
            if logger:
                logger.log_solution(best_obj, best_coloring)

    return int(round(best_obj)), best_coloring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    data = read_instance(args.instance_path)
    num_nodes = data["num_nodes"]
    num_edges = data["num_edges"]
    edges = data["edges"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # heuristic solution
    init_color, init_num_colors = dsatur(num_nodes, edges)
    best_obj = init_num_colors
    best_coloring = {str(i): init_color[i] for i in range(num_nodes)} if num_nodes > 0 else {}

    if logger and best_coloring:
        logger.log_solution(best_obj, best_coloring)

    # improve with MIP if time allows
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 1 and num_nodes > 0:
        obj, coloring = solve_mip(num_nodes, edges, init_color, init_num_colors,
                                  args.time_limit, logger, start_time)
        if obj < best_obj:
            best_obj = obj
            best_coloring = coloring
            if logger:
                logger.log_solution(best_obj, best_coloring)

    solution = {
        "objective_value": float(best_obj),
        "coloring": best_coloring
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()