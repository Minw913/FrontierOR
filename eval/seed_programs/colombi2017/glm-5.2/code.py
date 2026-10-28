import json
import argparse
from gurobipy import Model, GRB, quicksum

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log_solution(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    depot = instance['depot']
    arcs = instance['arcs']
    profitable_arcs = instance['profitable_arcs']
    vi_nodes = instance['VI_nodes']
    strong_incompat = instance['strong_incompatibilities']
    weak_incompat = instance['weak_incompatibilities']

    # Match each profitable arc to a road arc (handle parallel arcs by order)
    arc_map = {}
    for idx, (i, j, c) in enumerate(arcs):
        if (i, j) not in arc_map:
            arc_map[(i, j)] = []
        arc_map[(i, j)].append(idx)

    profitable_to_arc = []
    for (i, j, profit) in profitable_arcs:
        if (i, j) in arc_map and len(arc_map[(i, j)]) > 0:
            arc_idx = arc_map[(i, j)].pop(0)
            profitable_to_arc.append(arc_idx)
        else:
            profitable_to_arc.append(-1)

    model = Model("haul")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LazyConstraints', 1)
    model.setParam('OutputFlag', 0)

    # Variables
    x_vars = []
    for idx, (i, j, c) in enumerate(arcs):
        x = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{idx}")
        x_vars.append(x)

    y_vars = []
    for p_idx, (i, j, profit) in enumerate(profitable_arcs):
        y = model.addVar(vtype=GRB.BINARY, name=f"y_{p_idx}")
        y_vars.append(y)

    z_vars = {}
    for node in vi_nodes:
        z_vars[node] = model.addVar(vtype=GRB.BINARY, name=f"z_{node}")

    weak_vars = []
    for idx, (i, j, penalty) in enumerate(weak_incompat):
        if i in z_vars and j in z_vars:
            w = model.addVar(vtype=GRB.BINARY, name=f"w_{idx}")
            weak_vars.append((w, penalty, i, j))
        else:
            weak_vars.append((None, penalty, i, j))

    model.update()

    # Flow conservation
    out_arcs = {i: [] for i in range(num_nodes)}
    in_arcs = {i: [] for i in range(num_nodes)}
    for idx, (i, j, c) in enumerate(arcs):
        out_arcs[i].append(idx)
        in_arcs[j].append(idx)

    for i in range(num_nodes):
        model.addConstr(quicksum(x_vars[idx] for idx in out_arcs[i]) == quicksum(x_vars[idx] for idx in in_arcs[i]))

    # Profitable arc served only if traversed
    for p_idx, arc_idx in enumerate(profitable_to_arc):
        if arc_idx != -1:
            model.addConstr(y_vars[p_idx] <= x_vars[arc_idx])

    # Active node definition
    profit_by_start = {}
    for p_idx, (i, j, profit) in enumerate(profitable_arcs):
        if i not in profit_by_start:
            profit_by_start[i] = []
        profit_by_start[i].append(p_idx)

    for node in vi_nodes:
        if node in profit_by_start:
            for p_idx in profit_by_start[node]:
                model.addConstr(z_vars[node] >= y_vars[p_idx])
            model.addConstr(z_vars[node] <= quicksum(y_vars[p_idx] for p_idx in profit_by_start[node]))
        else:
            model.addConstr(z_vars[node] == 0)

    # Strong incompatibilities
    for i, j in strong_incompat:
        if i in z_vars and j in z_vars:
            model.addConstr(z_vars[i] + z_vars[j] <= 1)

    # Weak incompatibilities
    for w, penalty, i, j in weak_vars:
        if w is not None:
            model.addConstr(z_vars[i] + z_vars[j] - 1 <= w)

    # Objective
    obj = quicksum(profitable_arcs[p_idx][2] * y_vars[p_idx] for p_idx in range(len(profitable_arcs)))
    obj -= quicksum(arcs[idx][2] * x_vars[idx] for idx in range(len(arcs)))
    for w, penalty, i, j in weak_vars:
        if w is not None:
            obj -= penalty * w
    model.setObjective(obj, GRB.MAXIMIZE)

    # MIP Start (empty tour is always feasible)
    for v in x_vars:
        v.Start = 0
    for v in y_vars:
        v.Start = 0
    for v in z_vars.values():
        v.Start = 0
    for w, _, _, _ in weak_vars:
        if w is not None:
            w.Start = 0

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(x_vars)
            y_vals = model.cbGetSolution(y_vars)
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)

            if logger:
                served_arcs = []
                for p_idx, (i, j, profit) in enumerate(profitable_arcs):
                    if y_vals[p_idx] > 0.5:
                        arc_idx = profitable_to_arc[p_idx]
                        cost = arcs[arc_idx][2] if arc_idx != -1 else 0
                        served_arcs.append({'from': i, 'to': j, 'profit': profit, 'cost': cost})
                tour_arcs = []
                for a_idx, (i, j, c) in enumerate(arcs):
                    count = int(round(x_vals[a_idx]))
                    if count > 0:
                        tour_arcs.append({'from': i, 'to': j, 'count': count, 'cost': c})
                sol_dict = {
                    'objective_value': float(obj_val),
                    'served_arcs': served_arcs,
                    'tour_arcs': tour_arcs
                }
                logger.log_solution(float(obj_val), sol_dict)

            # Subtour elimination: add cut for any disconnected component not containing the depot
            adj = [[] for _ in range(num_nodes)]
            for idx, (i, j, c) in enumerate(arcs):
                if x_vals[idx] > 0.5:
                    adj[i].append(j)
                    adj[j].append(i)
            visited = [False] * num_nodes
            for start in range(num_nodes):
                if not visited[start]:
                    comp = []
                    stack = [start]
                    visited[start] = True
                    while stack:
                        u = stack.pop()
                        comp.append(u)
                        for v in adj[u]:
                            if not visited[v]:
                                visited[v] = True
                                stack.append(v)
                    if depot not in comp:
                        comp_set = set(comp)
                        expr = 0
                        for idx, (i, j, c) in enumerate(arcs):
                            if i in comp_set and j not in comp_set:
                                expr += x_vars[idx]
                        model.cbLazy(expr >= 1)

    model.optimize(callback)

    # Retrieve final solution
    if model.SolCount > 0:
        x_vals = [round(v.X) for v in x_vars]
        y_vals = [round(v.X) for v in y_vars]
        w_vals = [round(w.X) if w is not None else 0 for w, _, _, _ in weak_vars]
        obj = sum(profitable_arcs[p_idx][2] * y_vals[p_idx] for p_idx in range(len(profitable_arcs)))
        obj -= sum(arcs[idx][2] * x_vals[idx] for idx in range(len(arcs)))
        for (w, penalty, i, j), w_val in zip(weak_vars, w_vals):
            if w_val > 0.5:
                obj -= penalty
        obj = float(obj)
    else:
        obj = 0.0
        x_vals = [0] * len(arcs)
        y_vals = [0] * len(profitable_arcs)

    served_arcs = []
    for p_idx, (i, j, profit) in enumerate(profitable_arcs):
        if y_vals[p_idx] > 0.5:
            arc_idx = profitable_to_arc[p_idx]
            cost = arcs[arc_idx][2] if arc_idx != -1 else 0
            served_arcs.append({'from': i, 'to': j, 'profit': profit, 'cost': cost})

    tour_arcs = []
    for a_idx, (i, j, c) in enumerate(arcs):
        count = int(x_vals[a_idx])
        if count > 0:
            tour_arcs.append({'from': i, 'to': j, 'count': count, 'cost': c})

    solution = {
        'objective_value': obj,
        'served_arcs': served_arcs,
        'tour_arcs': tour_arcs
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(obj, solution)


if __name__ == '__main__':
    main()