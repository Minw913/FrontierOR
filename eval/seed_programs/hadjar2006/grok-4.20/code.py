import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    num_depots = instance['num_depots']
    depots = instance['depots']
    tasks = instance['tasks']
    compatible_pairs = instance['compatible_pairs']
    depot_to_task_costs = instance['depot_to_task_costs']
    task_to_depot_costs = instance['task_to_depot_costs']

    task_id_to_idx = {t['id']: i for i, t in enumerate(tasks)}
    depot_id_to_idx = {d['id']: i for i, d in enumerate(depots)}
    task_idx_to_id = {i: t['id'] for i, t in enumerate(tasks)}
    depot_idx_to_id = {i: d['id'] for i, d in enumerate(depots)}

    model = gp.Model("MDVSP")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.LogToConsole = 0
    model.Params.Method = 3  # concurrent
    model.Params.Heuristics = 0.8

    x_po = {}
    x_pi = {}
    x_arc = {}

    for entry in depot_to_task_costs:
        d_idx = depot_id_to_idx[entry['depot_id']]
        t_idx = task_id_to_idx[entry['task_id']]
        x_po[(d_idx, t_idx)] = model.addVar(vtype=GRB.BINARY, obj=entry['cost'])

    for entry in task_to_depot_costs:
        t_idx = task_id_to_idx[entry['task_id']]
        d_idx = depot_id_to_idx[entry['depot_id']]
        x_pi[(t_idx, d_idx)] = model.addVar(vtype=GRB.BINARY, obj=entry['cost'])

    for entry in compatible_pairs:
        i_idx = task_id_to_idx[entry['task_i']]
        j_idx = task_id_to_idx[entry['task_j']]
        cost = entry['cost']
        for d in range(num_depots):
            x_arc[(d, i_idx, j_idx)] = model.addVar(vtype=GRB.BINARY, obj=cost)

    # Each task is covered exactly once
    for t in range(num_tasks):
        inflow = gp.LinExpr()
        for d in range(num_depots):
            if (d, t) in x_po:
                inflow.add(x_po[(d, t)])
        for i in range(num_tasks):
            for d in range(num_depots):
                if (d, i, t) in x_arc:
                    inflow.add(x_arc[(d, i, t)])
        model.addConstr(inflow == 1, name=f"cover_{t}")

    # Flow conservation per depot per task
    for d in range(num_depots):
        for t in range(num_tasks):
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()

            if (d, t) in x_po:
                inflow.add(x_po[(d, t)])
            for i in range(num_tasks):
                if (d, i, t) in x_arc:
                    inflow.add(x_arc[(d, i, t)])

            for j in range(num_tasks):
                if (d, t, j) in x_arc:
                    outflow.add(x_arc[(d, t, j)])
            if (t, d) in x_pi:
                outflow.add(x_pi[(t, d)])

            if inflow.size() > 0 or outflow.size() > 0:
                model.addConstr(inflow == outflow, name=f"flow_d{d}_t{t}")

    # Depot capacity
    for d in range(num_depots):
        outflow = gp.LinExpr()
        for t in range(num_tasks):
            if (d, t) in x_po:
                outflow.add(x_po[(d, t)])
        model.addConstr(outflow <= depots[d]['capacity'], name=f"cap_{d}")

    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount >= 1:
        objective_value = model.ObjVal

        routes = []
        for d in range(num_depots):
            # Build adjacency for this depot
            adj = {i: [] for i in range(num_tasks)}
            starts = []
            for t in range(num_tasks):
                if (d, t) in x_po and x_po[(d, t)].X > 0.5:
                    starts.append(t)
                for j in range(num_tasks):
                    if (d, t, j) in x_arc and x_arc[(d, t, j)].X > 0.5:
                        adj[t].append(j)

            for start in starts:
                path = []
                current = start
                visited = set()
                while current is not None and current not in visited:
                    visited.add(current)
                    path.append(current)
                    next_task = None
                    for nxt in adj[current]:
                        if nxt not in visited:
                            next_task = nxt
                            break
                    current = next_task
                if path:
                    route_cost = 0.0
                    # po cost
                    if (d, path[0]) in x_po:
                        route_cost += x_po[(d, path[0])].Obj
                    # arc costs
                    for k in range(len(path) - 1):
                        i = path[k]
                        j = path[k + 1]
                        if (d, i, j) in x_arc:
                            route_cost += x_arc[(d, i, j)].Obj
                    # pi cost
                    last = path[-1]
                    if (last, d) in x_pi and x_pi[(last, d)].X > 0.5:
                        route_cost += x_pi[(last, d)].Obj

                    routes.append({
                        "depot": int(depot_idx_to_id[d]),
                        "tasks": [int(task_idx_to_id[t]) for t in path],
                        "cost": float(route_cost)
                    })

        solution = {
            "objective_value": float(objective_value),
            "routes": routes
        }
    else:
        solution = {
            "objective_value": 0.0,
            "routes": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()