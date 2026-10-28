import json
import argparse
import time
import heapq
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def find_shortest_path_solution(instance):
    """Find a feasible solution using shortest paths for each commodity."""
    num_nodes = instance['num_nodes']
    arcs = instance['arcs']
    fixed_costs = instance['fixed_costs']
    variable_costs = instance['variable_costs']
    commodities = instance['commodities']
    num_commodities = instance['num_commodities']

    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    for idx, (i, j) in enumerate(arcs):
        adj[i].append((j, idx))

    open_arcs_indices = set()
    routings = {}

    for k in range(num_commodities):
        origin = commodities[k]['origin']
        destination = commodities[k]['destination']

        if origin == destination:
            routings[str(k)] = {}
            continue

        # Dijkstra with variable costs for this commodity
        dist = [float('inf')] * num_nodes
        dist[origin] = 0
        prev_arc = [-1] * num_nodes
        pq = [(0, origin)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == destination:
                break
            for v, arc_idx in adj[u]:
                key = f"{arcs[arc_idx][0]}_{arcs[arc_idx][1]}"
                cost = variable_costs[key][k]
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev_arc[v] = arc_idx
                    heapq.heappush(pq, (dist[v], v))

        routings[str(k)] = {}
        if dist[destination] < float('inf'):
            node = destination
            while node != origin:
                arc_idx = prev_arc[node]
                open_arcs_indices.add(arc_idx)
                i, j = arcs[arc_idx]
                key = f"{i}_{j}"
                routings[str(k)][key] = 1.0
                node = i  # Move back to the source of this arc

    # Calculate objective
    obj = 0.0
    open_arcs = {}
    for idx in open_arcs_indices:
        i, j = arcs[idx]
        key = f"{i}_{j}"
        open_arcs[key] = 1
        obj += fixed_costs[key]

    for k in range(num_commodities):
        demand = commodities[k]['demand']
        for key, frac in routings.get(str(k), {}).items():
            obj += demand * variable_costs[key][k] * frac

    return obj, open_arcs, routings


def solve_mip(instance, time_limit, logger=None, initial_solution=None):
    """Solve the FCNDP using Gurobi MIP."""
    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    num_commodities = instance['num_commodities']
    arcs = instance['arcs']
    fixed_costs = instance['fixed_costs']
    variable_costs = instance['variable_costs']
    commodities = instance['commodities']

    model = gp.Model("FCNDP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Binary variables: y[idx] = 1 if arc idx is opened
    y = {}
    for idx in range(num_arcs):
        i, j = arcs[idx]
        key = f"{i}_{j}"
        y[idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{key}")

    # Continuous variables: x[k, idx] = fraction of commodity k on arc idx
    x = {}
    for k in range(num_commodities):
        for idx in range(num_arcs):
            x[k, idx] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                     name=f"x_{k}_{idx}")

    model.update()

    # Objective
    obj = gp.LinExpr()
    for idx in range(num_arcs):
        i, j = arcs[idx]
        key = f"{i}_{j}"
        obj += fixed_costs[key] * y[idx]

    for k in range(num_commodities):
        demand_k = commodities[k]['demand']
        for idx in range(num_arcs):
            i, j = arcs[idx]
            key = f"{i}_{j}"
            obj += demand_k * variable_costs[key][k] * x[k, idx]

    model.setObjective(obj, GRB.MINIMIZE)

    # Precompute incoming and outgoing arcs for each node
    in_arcs = [[] for _ in range(num_nodes)]
    out_arcs = [[] for _ in range(num_nodes)]
    for idx in range(num_arcs):
        i, j = arcs[idx]
        out_arcs[i].append(idx)
        in_arcs[j].append(idx)

    # Flow conservation constraints
    for k in range(num_commodities):
        origin = commodities[k]['origin']
        destination = commodities[k]['destination']

        for node in range(num_nodes):
            flow_expr = gp.LinExpr()
            for idx in in_arcs[node]:
                flow_expr += x[k, idx]
            for idx in out_arcs[node]:
                flow_expr -= x[k, idx]

            if node == origin:
                rhs = -1.0
            elif node == destination:
                rhs = 1.0
            else:
                rhs = 0.0

            model.addConstr(flow_expr == rhs, name=f"flow_{k}_{node}")

    # Linking constraints: x[k,a] <= y[a]
    for k in range(num_commodities):
        for idx in range(num_arcs):
            model.addConstr(x[k, idx] <= y[idx], name=f"link_{k}_{idx}")

    # Add cutset inequalities for strengthening
    for k in range(num_commodities):
        origin = commodities[k]['origin']
        destination = commodities[k]['destination']
        if origin == destination:
            continue

        # BFS from origin to find layers
        visited = [False] * num_nodes
        visited[origin] = True
        queue = [origin]
        layers = [set([origin])]
        layer_of = {origin: 0}

        while queue:
            next_queue = []
            next_layer = set()
            for u in queue:
                for idx in out_arcs[u]:
                    v = arcs[idx][1]
                    if not visited[v]:
                        visited[v] = True
                        next_queue.append(v)
                        next_layer.add(v)
                        layer_of[v] = len(layers)
            if next_layer:
                layers.append(next_layer)
            queue = next_queue

        # Add cutset inequality for each layer boundary
        if destination in layer_of:
            dest_layer = layer_of[destination]
            for l in range(dest_layer):
                S = set()
                for node in range(num_nodes):
                    if node in layer_of and layer_of[node] <= l:
                        S.add(node)
                if origin in S and destination not in S:
                    cut_expr = gp.LinExpr()
                    for idx in range(num_arcs):
                        i, j = arcs[idx]
                        if i in S and j not in S:
                            cut_expr += y[idx]
                    if cut_expr.size() > 0:
                        model.addConstr(cut_expr >= 1, name=f"cutset_{k}_{l}")

    # Set initial solution
    if initial_solution:
        _, init_open, init_routings = initial_solution
        for idx in range(num_arcs):
            i, j = arcs[idx]
            key = f"{i}_{j}"
            y[idx].Start = 1.0 if key in init_open else 0.0

        for k in range(num_commodities):
            for idx in range(num_arcs):
                i, j = arcs[idx]
                key = f"{i}_{j}"
                if str(k) in init_routings and key in init_routings[str(k)]:
                    x[k, idx].Start = init_routings[str(k)][key]
                else:
                    x[k, idx].Start = 0.0

    # Callback for logging solutions
    best_logged = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_logged[0] - 1e-8:
                best_logged[0] = obj_val
                if logger:
                    open_arcs = {}
                    for idx in range(num_arcs):
                        i, j = arcs[idx]
                        key = f"{i}_{j}"
                        if model.cbGetSolution(y[idx]) > 0.5:
                            open_arcs[key] = 1

                    routings = {}
                    for k in range(num_commodities):
                        routings[str(k)] = {}
                        for idx in range(num_arcs):
                            i, j = arcs[idx]
                            key = f"{i}_{j}"
                            val = model.cbGetSolution(x[k, idx])
                            if val > 1e-6:
                                routings[str(k)][key] = val

                    solution = {
                        "objective_value": obj_val,
                        "open_arcs": open_arcs,
                        "routings": routings
                    }
                    logger.log_solution(obj_val, solution)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        open_arcs = {}
        for idx in range(num_arcs):
            i, j = arcs[idx]
            key = f"{i}_{j}"
            if y[idx].X > 0.5:
                open_arcs[key] = 1

        routings = {}
        for k in range(num_commodities):
            routings[str(k)] = {}
            for idx in range(num_arcs):
                i, j = arcs[idx]
                key = f"{i}_{j}"
                if x[k, idx].X > 1e-6:
                    routings[str(k)][key] = x[k, idx].X

        return {
            "objective_value": model.ObjVal,
            "open_arcs": open_arcs,
            "routings": routings
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="Solve Fixed-Charge Network Design Problem")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    start_time = time.time()

    # Get heuristic solution first
    heuristic_obj, heuristic_open, heuristic_routings = find_shortest_path_solution(instance)
    initial_solution = (heuristic_obj, heuristic_open, heuristic_routings)

    if logger:
        heuristic_solution = {
            "objective_value": heuristic_obj,
            "open_arcs": heuristic_open,
            "routings": heuristic_routings
        }
        logger.log_solution(heuristic_obj, heuristic_solution)

    remaining_time = args.time_limit - (time.time() - start_time)

    solution = None
    if remaining_time > 1:
        solution = solve_mip(instance, remaining_time, logger, initial_solution)

    if solution is None:
        solution = {
            "objective_value": heuristic_obj,
            "open_arcs": heuristic_open,
            "routings": heuristic_routings
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()