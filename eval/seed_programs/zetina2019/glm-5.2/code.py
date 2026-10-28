import json
import argparse
import heapq
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    num_commodities = instance['num_commodities']
    arcs = instance['arcs']
    fixed_costs = instance['fixed_costs']
    variable_costs = instance['variable_costs']
    commodities = instance['commodities']

    arc_keys = [f"{a[0]}_{a[1]}" for a in arcs]

    # Adjacency lists for flow conservation
    outgoing = {i: [] for i in range(num_nodes)}
    incoming = {i: [] for i in range(num_nodes)}
    for i, a in enumerate(arcs):
        u, v = a
        outgoing[u].append(i)
        incoming[v].append(i)

    # Build model
    model = Model("FCMCF")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Variables
    y = {}
    for key in arc_keys:
        y[key] = model.addVar(vtype=GRB.BINARY, name=f"y_{key}")

    x = {}
    for k in range(num_commodities):
        for key in arc_keys:
            x[(k, key)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x_{k}_{key}")

    model.update()

    # Flow conservation: outflow - inflow = 1 at origin, -1 at dest, 0 otherwise
    for k in range(num_commodities):
        origin = commodities[k]['origin']
        dest = commodities[k]['destination']
        for i in range(num_nodes):
            out_expr = quicksum(x[(k, arc_keys[a])] for a in outgoing[i])
            in_expr = quicksum(x[(k, arc_keys[a])] for a in incoming[i])
            if i == origin:
                rhs = 1.0
            elif i == dest:
                rhs = -1.0
            else:
                rhs = 0.0
            model.addConstr(out_expr - in_expr == rhs, name=f"flow_{k}_{i}")

    # Linking constraints: flow only on installed arcs
    for k in range(num_commodities):
        for key in arc_keys:
            model.addConstr(x[(k, key)] <= y[key], name=f"link_{k}_{key}")

    # Objective: fixed costs + variable transportation costs
    obj = quicksum(fixed_costs[key] * y[key] for key in arc_keys)
    for k in range(num_commodities):
        demand = commodities[k]['demand']
        for key in arc_keys:
            var_cost = variable_costs[key][k]
            obj += demand * var_cost * x[(k, key)]
    model.setObjective(obj, GRB.MINIMIZE)

    # Initial feasible solution: shortest paths on full graph ignoring fixed costs
    adj = {i: [] for i in range(num_nodes)}
    for i, a in enumerate(arcs):
        u, v = a
        adj[u].append((v, i))

    init_y = {key: 0 for key in arc_keys}
    init_x = {}
    feasible_init = True

    for k in range(num_commodities):
        origin = commodities[k]['origin']
        dest = commodities[k]['destination']
        dist = {i: float('inf') for i in range(num_nodes)}
        dist[origin] = 0
        prev = {i: -1 for i in range(num_nodes)}
        pq = [(0, origin)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dest:
                break
            for v, a_idx in adj[u]:
                key = arc_keys[a_idx]
                w = variable_costs[key][k]
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = a_idx
                    heapq.heappush(pq, (nd, v))
        if dist[dest] == float('inf'):
            feasible_init = False
            break
        # Reconstruct path
        path_arcs = []
        curr = dest
        while curr != origin:
            a_idx = prev[curr]
            if a_idx == -1:
                feasible_init = False
                break
            path_arcs.append(a_idx)
            u, v = arcs[a_idx]
            curr = u
        if not feasible_init:
            break
        path_arcs.reverse()
        for a_idx in path_arcs:
            key = arc_keys[a_idx]
            init_y[key] = 1
            init_x[(k, key)] = 1.0

    if feasible_init:
        for key in arc_keys:
            y[key].Start = init_y[key]
        for k in range(num_commodities):
            for key in arc_keys:
                x[(k, key)].Start = init_x.get((k, key), 0.0)

    # Prepare variable lists for callback
    y_vars = [y[key] for key in arc_keys]
    x_vars = [x[(k, key)] for k in range(num_commodities) for key in arc_keys]
    x_keys_list = [(k, key) for k in range(num_commodities) for key in arc_keys]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                y_vals = model.cbGetSolution(y_vars)
                x_vals = model.cbGetSolution(x_vars)

                open_arcs = {}
                for i, key in enumerate(arc_keys):
                    if y_vals[i] > 0.5:
                        open_arcs[key] = 1

                routings = {}
                for i, (k, key) in enumerate(x_keys_list):
                    val = x_vals[i]
                    if val > 1e-6:
                        k_str = str(k)
                        if k_str not in routings:
                            routings[k_str] = {}
                        routings[k_str][key] = val

                sol_dict = {
                    "objective_value": obj_val,
                    "open_arcs": open_arcs,
                    "routings": routings
                }
                logger.log_solution(obj_val, sol_dict)

    model.optimize(callback=callback)

    # Retrieve final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        y_vals = model.getAttr('X', y)
        x_vals = model.getAttr('X', x)

        open_arcs = {}
        for key in arc_keys:
            if y_vals[key] > 0.5:
                open_arcs[key] = 1

        routings = {}
        for k in range(num_commodities):
            for key in arc_keys:
                val = x_vals[(k, key)]
                if val > 1e-6:
                    k_str = str(k)
                    if k_str not in routings:
                        routings[k_str] = {}
                    routings[k_str][key] = val

        solution = {
            "objective_value": obj_val,
            "open_arcs": open_arcs,
            "routings": routings
        }
    else:
        solution = {
            "objective_value": float('inf'),
            "open_arcs": {},
            "routings": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()