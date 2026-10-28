import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Multi-Depot Vehicle Scheduling Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    num_depots = instance['num_depots']
    depots = instance['depots']
    tasks = instance['tasks']
    compatible_pairs = instance['compatible_pairs']
    depot_to_task_costs_list = instance['depot_to_task_costs']
    task_to_depot_costs_list = instance['task_to_depot_costs']

    # Build lookups
    depot_ids = [d['id'] for d in depots]
    task_ids = [t['id'] for t in tasks]
    depot_capacity = {d['id']: d['capacity'] for d in depots}

    pull_out_cost = {}
    for entry in depot_to_task_costs_list:
        pull_out_cost[(entry['depot_id'], entry['task_id'])] = entry['cost']

    pull_in_cost = {}
    for entry in task_to_depot_costs_list:
        pull_in_cost[(entry['task_id'], entry['depot_id'])] = entry['cost']

    connect_cost = {}
    for entry in compatible_pairs:
        connect_cost[(entry['task_i'], entry['task_j'])] = entry['cost']

    # Build adjacency lists
    task_successors = {t: [] for t in task_ids}
    task_predecessors = {t: [] for t in task_ids}
    for entry in compatible_pairs:
        task_successors[entry['task_i']].append(entry['task_j'])
        task_predecessors[entry['task_j']].append(entry['task_i'])

    # Handle trivial case
    if num_tasks == 0:
        solution = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Build Gurobi model
    model = gp.Model("MDVSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    pull_out = {}   # (depot_id, task_id) -> binary variable
    pull_in = {}    # (task_id, depot_id) -> binary variable
    connect = {}    # (depot_id, task_i, task_j) -> binary variable

    for d in depot_ids:
        for t in task_ids:
            pull_out[(d, t)] = model.addVar(vtype=GRB.BINARY, name=f"po_{d}_{t}")
            pull_in[(t, d)] = model.addVar(vtype=GRB.BINARY, name=f"pi_{t}_{d}")

    for d in depot_ids:
        for entry in compatible_pairs:
            i, j = entry['task_i'], entry['task_j']
            connect[(d, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"cn_{d}_{i}_{j}")

    model.update()

    # Objective: minimize total cost
    obj = gp.LinExpr()
    for (d, t), var in pull_out.items():
        obj += pull_out_cost[(d, t)] * var
    for (t, d), var in pull_in.items():
        obj += pull_in_cost[(t, d)] * var
    for (d, i, j), var in connect.items():
        obj += connect_cost[(i, j)] * var

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Each task is covered exactly once
    for t in task_ids:
        expr = gp.LinExpr()
        for d in depot_ids:
            expr += pull_out[(d, t)]
            for i in task_predecessors[t]:
                expr += connect[(d, i, t)]
        model.addConstr(expr == 1, name=f"cover_{t}")

    # Constraint 2: Flow conservation at each task for each depot
    for d in depot_ids:
        for t in task_ids:
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()

            # Inflow to task t from depot d's vehicles
            inflow += pull_out[(d, t)]
            for i in task_predecessors[t]:
                inflow += connect[(d, i, t)]

            # Outflow from task t for depot d's vehicles
            for j in task_successors[t]:
                outflow += connect[(d, t, j)]
            outflow += pull_in[(t, d)]

            model.addConstr(inflow == outflow, name=f"flow_{d}_{t}")

    # Constraint 3: Depot capacity
    for d in depot_ids:
        expr = gp.LinExpr()
        for t in task_ids:
            expr += pull_out[(d, t)]
        model.addConstr(expr <= depot_capacity[d], name=f"cap_{d}")

    model.update()

    # Helper function to extract routes from variable values
    def extract_routes_from_values(get_val):
        routes = []
        for d in depot_ids:
            succ = {}
            for entry in compatible_pairs:
                i, j = entry['task_i'], entry['task_j']
                if get_val(connect[(d, i, j)]) > 0.5:
                    succ[i] = j

            starts = [t for t in task_ids if get_val(pull_out[(d, t)]) > 0.5]

            for start in starts:
                route_tasks = [start]
                current = start
                while current in succ:
                    current = succ[current]
                    route_tasks.append(current)

                route_cost = pull_out_cost[(d, start)]
                for idx in range(len(route_tasks) - 1):
                    route_cost += connect_cost[(route_tasks[idx], route_tasks[idx + 1])]
                route_cost += pull_in_cost[(route_tasks[-1], d)]

                routes.append({
                    "depot": int(d),
                    "tasks": [int(t) for t in route_tasks],
                    "cost": float(route_cost)
                })
        return routes

    # Callback for logging intermediate incumbent solutions
    best_logged_obj = [float('inf')]

    def mipsol_callback(cb_model, where):
        if where == GRB.CALLBACK_MIPSOL:
            obj_val = cb_model.cbGet(GRB.CB_MIPSOL_OBJ)
            if obj_val < best_logged_obj[0] - 1e-6:
                best_logged_obj[0] = obj_val
                routes = extract_routes_from_values(lambda v: cb_model.cbGetSolution(v))
                solution_dict = {"objective_value": float(obj_val), "routes": routes}
                if logger:
                    logger.log_solution(float(obj_val), solution_dict)

    if logger:
        model.setCallback(mipsol_callback)

    # Solve
    model.optimize()

    # Extract final solution
    if model.SolCount > 0:
        routes = extract_routes_from_values(lambda v: v.X)
        total_cost = float(model.ObjVal)

        solution = {
            "objective_value": total_cost,
            "routes": routes
        }

        # Log final solution if not already logged via callback
        if logger and total_cost < best_logged_obj[0] - 1e-6:
            logger.log_solution(total_cost, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No feasible solution found within time limit
        solution = {"objective_value": 1e10, "routes": []}
        if logger:
            logger.log_solution(1e10, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()