import argparse
import json
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
    
    tasks = instance['tasks']
    depots = instance['depots']
    num_tasks = instance['num_tasks']
    num_depots = instance['num_depots']
    compatible_pairs = instance['compatible_pairs']
    depot_to_task_costs = instance['depot_to_task_costs']
    task_to_depot_costs = instance['task_to_depot_costs']

    task_id_to_index = {task['id']: idx for idx, task in enumerate(tasks)}
    depot_id_to_index = {depot['id']: idx for idx, depot in enumerate(depots)}
    task_index_to_id = [task['id'] for task in tasks]
    depot_index_to_id = [depot['id'] for depot in depots]

    pull_out_arcs = []
    for arc in depot_to_task_costs:
        depot_idx = depot_id_to_index[arc['depot_id']]
        task_idx = task_id_to_index[arc['task_id']]
        cost = arc['cost']
        pull_out_arcs.append((depot_idx, task_idx, cost))
    
    pull_in_arcs = []
    for arc in task_to_depot_costs:
        task_idx = task_id_to_index[arc['task_id']]
        depot_idx = depot_id_to_index[arc['depot_id']]
        cost = arc['cost']
        pull_in_arcs.append((task_idx, depot_idx, cost))
    
    deadhead_arcs = []
    for pair in compatible_pairs:
        task_i_idx = task_id_to_index[pair['task_i']]
        task_j_idx = task_id_to_index[pair['task_j']]
        cost = pair['cost']
        deadhead_arcs.append((task_i_idx, task_j_idx, cost))
    
    pull_out_cost_dict = {}
    for (d_idx, t_idx, cost) in pull_out_arcs:
        pull_out_cost_dict[(d_idx, t_idx)] = cost
    
    pull_in_cost_dict = {}
    for (t_idx, d_idx, cost) in pull_in_arcs:
        pull_in_cost_dict[(d_idx, t_idx)] = cost
    
    deadhead_cost_dict = {}
    for (i, j, cost) in deadhead_arcs:
        deadhead_cost_dict[(i, j)] = cost

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    f = {}
    total_cost = 0
    flow_in_dt = {(d, t): [] for d in range(num_depots) for t in range(num_tasks)}
    flow_out_dt = {(d, t): [] for d in range(num_depots) for t in range(num_tasks)}
    pull_out_vars = {d: [] for d in range(num_depots)}
    
    for d in range(num_depots):
        source_d = num_tasks + 2 * d
        sink_d = num_tasks + 2 * d + 1
        
        for t in range(num_tasks):
            if (d, t) in pull_out_cost_dict:
                cost_val = pull_out_cost_dict[(d, t)]
                var = model.addVar(vtype=GRB.BINARY, name=f"f_d{d}_pullout_{t}")
                f[(d, source_d, t)] = var
                flow_in_dt[(d, t)].append(var)
                pull_out_vars[d].append(var)
                total_cost += cost_val * var
        
        for t in range(num_tasks):
            if (d, t) in pull_in_cost_dict:
                cost_val = pull_in_cost_dict[(d, t)]
                var = model.addVar(vtype=GRB.BINARY, name=f"f_d{d}_pullin_{t}")
                f[(d, t, sink_d)] = var
                flow_out_dt[(d, t)].append(var)
                total_cost += cost_val * var
        
        for (i, j, cost_val) in deadhead_arcs:
            var = model.addVar(vtype=GRB.BINARY, name=f"f_d{d}_deadhead_{i}_{j}")
            f[(d, i, j)] = var
            flow_out_dt[(d, i)].append(var)
            flow_in_dt[(d, j)].append(var)
            total_cost += cost_val * var
    
    for t in range(num_tasks):
        total_in = []
        for d in range(num_depots):
            total_in.extend(flow_in_dt[(d, t)])
        model.addConstr(gp.quicksum(total_in) == 1, name=f"cover_trip_{t}")
    
    for d in range(num_depots):
        for t in range(num_tasks):
            in_list = flow_in_dt[(d, t)]
            out_list = flow_out_dt[(d, t)]
            model.addConstr(gp.quicksum(in_list) == gp.quicksum(out_list), name=f"flow_d{d}_t{t}")
    
    for d in range(num_depots):
        cap = depots[d]['capacity']
        model.addConstr(gp.quicksum(pull_out_vars[d]) <= cap, name=f"cap_d{d}")
    
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()
    
    solution = {"objective_value": 0.0, "routes": []}
    routes = []
    
    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        model.setParam(GRB.Param.SolutionNumber, 0)
        
        for d in range(num_depots):
            source_d = num_tasks + 2 * d
            sink_d = num_tasks + 2 * d + 1
            covered_in_depot = set()
            
            for t in range(num_tasks):
                arc_key = (d, source_d, t)
                if arc_key in f and f[arc_key].X > 0.5:
                    if t in covered_in_depot:
                        continue
                    route_tasks = [t]
                    covered_in_depot.add(t)
                    current = t
                    while True:
                        found_next = False
                        arc_key_pullin = (d, current, sink_d)
                        if arc_key_pullin in f and f[arc_key_pullin].X > 0.5:
                            break
                        for next_task in range(num_tasks):
                            arc_key_deadhead = (d, current, next_task)
                            if arc_key_deadhead in f and f[arc_key_deadhead].X > 0.5:
                                if next_task in covered_in_depot:
                                    break
                                route_tasks.append(next_task)
                                covered_in_depot.add(next_task)
                                current = next_task
                                found_next = True
                                break
                        if not found_next:
                            break
                    task_ids = [task_index_to_id[i] for i in route_tasks]
                    cost_route = pull_out_cost_dict.get((d, route_tasks[0]), 0)
                    for idx in range(len(route_tasks) - 1):
                        i = route_tasks[idx]
                        j = route_tasks[idx+1]
                        cost_route += deadhead_cost_dict.get((i, j), 0)
                    last_task = route_tasks[-1]
                    cost_route += pull_in_cost_dict.get((d, last_task), 0)
                    routes.append({
                        "depot": depot_index_to_id[d],
                        "tasks": task_ids,
                        "cost": cost_route
                    })
        solution['routes'] = routes
    
    with open(args.solution_path, 'w') as f_out:
        json.dump(solution, f_out)

if __name__ == "__main__":
    main()