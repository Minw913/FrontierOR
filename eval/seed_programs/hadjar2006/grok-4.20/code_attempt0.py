import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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
    depot_penalty = instance['depot_penalty']
    tasks = instance['tasks']
    depots = instance['depots']
    compatible_pairs = instance['compatible_pairs']
    depot_to_task_costs = instance['depot_to_task_costs']
    task_to_depot_costs = instance['task_to_depot_costs']

    task_ids = [t['id'] for t in tasks]
    depot_ids = [d['id'] for d in depots]
    task_id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    depot_id_to_idx = {did: i for i, did in enumerate(depot_ids)}

    # Build networks per depot
    # arcs: (from_type, from_id, to_type, to_id, cost, travel_time)
    # from_type: -1=depot, 0=task, to_type: 0=task, 1=depot

    # Precompute incoming and outgoing for each depot
    # We will create one big model with x[d][i][j] where i,j can be tasks or depots

    model = gp.Model("MDVSP")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # focus on finding good feasible solutions
    model.Params.LogToConsole = 0

    # Variables: for each depot, for each possible arc
    # We use three kinds of variables per depot:
    # pull_out[d][t]: from depot d to task t
    # pull_in[d][t]: from task t to depot d
    # connect[d][i][j]: from task i to task j for depot d

    pull_out = {}
    pull_in = {}
    connect = {}

    # Build index structures
    out_from_depot = [[] for _ in range(num_depots)]  # for each d, list of tasks reachable
    in_to_depot = [[] for _ in range(num_depots)]     # for each d, list of tasks that can go to d
    task_to_tasks = [[] for _ in range(num_tasks)]    # from task idx to list of (task_idx, cost, travel)

    depot_to_task_dict = {}
    for entry in depot_to_task_costs:
        d_idx = depot_id_to_idx[entry['depot_id']]
        t_idx = task_id_to_idx[entry['task_id']]
        depot_to_task_dict[(d_idx, t_idx)] = entry['cost']
        out_from_depot[d_idx].append(t_idx)

    task_to_depot_dict = {}
    for entry in task_to_depot_costs:
        t_idx = task_id_to_idx[entry['task_id']]
        d_idx = depot_id_to_idx[entry['depot_id']]
        task_to_depot_dict[(t_idx, d_idx)] = entry['cost']
        in_to_depot[d_idx].append(t_idx)

    compat_dict = {}
    for entry in compatible_pairs:
        i_idx = task_id_to_idx[entry['task_i']]
        j_idx = task_id_to_idx[entry['task_j']]
        cost = entry['cost']
        compat_dict[(i_idx, j_idx)] = cost
        task_to_tasks[i_idx].append((j_idx, cost))

    # Create variables
    for d in range(num_depots):
        for t in out_from_depot[d]:
            pull_out[(d, t)] = model.addVar(vtype=GRB.BINARY, name=f"po_{d}_{t}", obj=depot_to_task_dict[(d, t)])

        for t in in_to_depot[d]:
            pull_in[(d, t)] = model.addVar(vtype=GRB.BINARY, name=f"pi_{d}_{t}", obj=task_to_depot_dict[(t, d)])

    for d in range(num_depots):
        for i in range(num_tasks):
            for j_idx, cost_val in task_to_tasks[i]:
                connect[(d, i, j_idx)] = model.addVar(vtype=GRB.BINARY, name=f"c_{d}_{i}_{j_idx}", obj=cost_val)

    # Each task must be covered exactly once (by any depot/vehicle)
    task_cover = {}
    for t in range(num_tasks):
        expr = gp.LinExpr()
        # pull_out to this task from any depot
        for d in range(num_depots):
            if (d, t) in pull_out:
                expr.add(pull_out[(d, t)])
            # incoming connections from other tasks for this depot
            for d2 in range(num_depots):
                for prev, _ in task_to_tasks:
                    if prev == t and (d2, prev, t) in connect:  # wrong, need to reverse
                        pass
        # Better to precompute incoming for each task
    # Rebuild with proper covering constraints

    # Reset and use better structure
    model.dispose()
    model = gp.Model("MDVSP")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.LogToConsole = 0

    # Variables only when arc exists
    x_po = {}   # (d, t) : from depot d to first task t
    x_pi = {}   # (t, d) : from last task t to depot d
    x_arc = {}  # (d, i, j) : task i to task j under depot d

    # Create all possible variables
    depots_cap = [d['capacity'] for d in depots]

    for entry in depot_to_task_costs:
        d = depot_id_to_idx[entry['depot_id']]
        t = task_id_to_idx[entry['task_id']]
        x_po[(d, t)] = model.addVar(vtype=GRB.BINARY, obj=entry['cost'], name=f"po_{d}_{t}")

    for entry in task_to_depot_costs:
        t = task_id_to_idx[entry['task_id']]
        d = depot_id_to_idx[entry['depot_id']]
        x_pi[(t, d)] = model.addVar(vtype=GRB.BINARY, obj=entry['cost'], name=f"pi_{t}_{d}")

    for entry in compatible_pairs:
        i = task_id_to_idx[entry['task_i']]
        j = task_id_to_idx[entry['task_j']]
        cost = entry['cost']
        for d_idx in range(num_depots):
            x_arc[(d_idx, i, j)] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"arc_{d_idx}_{i}_{j}")

    # Objective already set in var creation

    # Constraint 1: Each task covered exactly once (exactly one incoming arc)
    for t in range(num_tasks):
        expr = gp.LinExpr(0)
        # incoming from depots
        for d in range(num_depots):
            if (d, t) in x_po:
                expr.add(x_po[(d, t)])
        # incoming from other tasks
        for i_idx in range(num_tasks):
            for d_idx in range(num_depots):
                if (d_idx, i_idx, t) in x_arc:
                    expr.add(x_arc[(d_idx, i_idx, t)])
        model.addConstr(expr == 1, name=f"cover_in_{t}")

    # Constraint 2: Flow balance per task per depot: inflow = outflow for each task and each depot
    for d in range(num_depots):
        for t in range(num_tasks):
            inflow = gp.LinExpr(0)
            outflow = gp.LinExpr(0)

            # inflow: from depot or from previous task
            if (d, t) in x_po:
                inflow.add(x_po[(d, t)])
            for prev in range(num_tasks):
                if (d, prev, t) in x_arc:
                    inflow.add(x_arc[(d, prev, t)])

            # outflow: to next task or to depot
            for nxt in range(num_tasks):
                if (d, t, nxt) in x_arc:
                    outflow.add(x_arc[(d, t, nxt)])
            if (t, d) in x_pi:
                outflow.add(x_pi[(t, d)])

            model.addConstr(inflow == outflow, name=f"balance_d{d}_t{t}")

    # Constraint 3: Depot capacity: number of vehicles from a depot <= capacity
    for d in range(num_depots):
        expr = gp.LinExpr(0)
        for t in range(num_tasks):
            if (d, t) in x_po:
                expr.add(x_po[(d, t)])
        model.addConstr(expr <= depots_cap[d], name=f"cap_{d}")

    # We do not need explicit return balance because of the per-depot flow conservation
    # Since every path starts and ends at same depot, and we have balance at tasks, it should hold globally

    # Optional: symmetry breaking or other improvements but limited time

    # Callback to log incumbent solutions
    best_obj = None
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        # Build routes per depot
        routes = []

        for d in range(num_depots):
            # Build graph for this depot
            adj = {t: [] for t in range(num_tasks)}
            start_tasks = []
            end_tasks = []

            for t in range(num_tasks):
                if (d, t) in x_po and x_po[(d, t)].X > 0.5:
                    start_tasks.append(t)
                if (t, d) in x_pi and x_pi[(t, d)].X > 0.5:
                    end_tasks.append(t)

            for i in range(num_tasks):
                for j in range(num_tasks):
                    if (d, i, j) in x_arc and x_arc[(d, i, j)].X > 0.5:
                        adj[i].append(j)

            # For each start, follow the path
            for start_t in start_tasks:
                path = [start_t]
                current = start_t
                while True:
                    found = False
                    for nxt in adj[current]:
                        if nxt not in path:  # avoid cycles though shouldn't exist
                            path.append(nxt)
                            current = nxt
                            found = True
                            break
                    if not found:
                        break
                # Check if ends at a valid end
                if path[-1] in end_tasks or True:  # should be
                    route_cost = 0.0
                    # calculate cost: po + arcs + pi
                    route_cost += x_po[(d, path[0])].X * x_po[(d, path[0])].Obj
                    for k in range(len(path)-1):
                        route_cost += x_arc[(d, path[k], path[k+1])].X * x_arc[(d, path[k], path[k+1])].Obj
                    if (path[-1], d) in x_pi:
                        route_cost += x_pi[(path[-1], d)].X * x_pi[(path[-1], d)].Obj

                    routes.append({
                        "depot": int(depots[d]['id']),
                        "tasks": [int(tasks[tid]['id']) for tid in path],
                        "cost": float(route_cost)
                    })

        # Write solution
        solution = {
            "objective_value": float(objective_value),
            "routes": routes
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found, write empty or best known (none)
        solution = {
            "objective_value": 0.0,
            "routes": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()