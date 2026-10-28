import argparse
import json
import time
from collections import defaultdict
import gurobipy as gp
from solution_logger import SolutionLogger

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_items = data['num_items']
    bin_capacity = data['bin_capacity']
    items = data['items']
    conflict_edges = data['conflict_edges']
    
    if num_items == 0:
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": 0, "bins": {}}, f)
        return
    
    weight_dict = {item['id']: item['weight'] for item in items}
    item_ids = [item['id'] for item in items]
    
    conflict_graph = defaultdict(set)
    for edge in conflict_edges:
        u, v = edge
        conflict_graph[u].add(v)
        conflict_graph[v].add(u)
    
    sorted_items = sorted(items, key=lambda x: x['weight'], reverse=True)
    bins_greedy = []
    
    for item in sorted_items:
        item_id = item['id']
        item_weight = item['weight']
        placed = False
        for idx in range(len(bins_greedy)):
            remaining, items_set = bins_greedy[idx]
            if remaining < item_weight:
                continue
            if conflict_graph[item_id] & items_set:
                continue
            remaining -= item_weight
            items_set.add(item_id)
            bins_greedy[idx] = (remaining, items_set)
            placed = True
            break
        if not placed:
            bins_greedy.append((bin_capacity - item_weight, {item_id}))
    
    greedy_objective = len(bins_greedy)
    if logger:
        logger.log(greedy_objective)
    
    best_objective = greedy_objective
    best_solution = [items_set for (_, items_set) in bins_greedy]
    
    remaining_time_after_greedy = args.time_limit - (time.time() - start_time)
    skip_ilp = True
    if remaining_time_after_greedy > 1:
        n = num_items
        bin_ub = greedy_objective
        num_conflict_edges = len(conflict_edges)
        if n * bin_ub <= 1000000 and bin_ub * num_conflict_edges <= 1000000:
            skip_ilp = False
    
    if not skip_ilp:
        try:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model("bppc", env=env)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            
            y = model.addVars(bin_ub, vtype=gp.GRB.BINARY, name="y")
            x = {}
            for item in items:
                item_id = item['id']
                for j in range(bin_ub):
                    x[item_id, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{item_id}_{j}")
            
            for item in items:
                item_id = item['id']
                model.addConstr(gp.quicksum(x[item_id, j] for j in range(bin_ub)) == 1, name=f"assign_{item_id}")
            
            for j in range(bin_ub):
                model.addConstr(
                    gp.quicksum(weight_dict[item_id] * x[item_id, j] for item in items) <= bin_capacity * y[j],
                    name=f"cap_{j}"
                )
            
            for edge in conflict_edges:
                u, v = edge
                for j in range(bin_ub):
                    model.addConstr(x[u, j] + x[v, j] <= 1, name=f"conflict_{u}_{v}_{j}")
            
            model.setObjective(gp.quicksum(y[j] for j in range(bin_ub)), gp.GRB.MINIMIZE)
            
            for j in range(bin_ub):
                if j < len(bins_greedy):
                    y[j].start = 1.0
                    bin_set = bins_greedy[j][1]
                    for item_id in bin_set:
                        x[item_id, j].start = 1.0
            
            model.setParam('TimeLimit', max(0.1, remaining_time_after_greedy))
            
            if logger:
                def callback(model, where):
                    if where == gp.GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                        logger.log(int(obj_val))
                model.optimize(callback)
            else:
                model.optimize()
            
            if model.SolCount > 0:
                bin_list = []
                for j in range(bin_ub):
                    if y[j].X > 0.5:
                        items_in_bin = []
                        for item in items:
                            item_id = item['id']
                            if x[item_id, j].X > 0.5:
                                items_in_bin.append(item_id)
                        bin_list.append(set(items_in_bin))
                ilp_objective = len(bin_list)
                if ilp_objective < best_objective:
                    best_objective = ilp_objective
                    best_solution = bin_list
        except Exception:
            pass
    
    bins_dict_output = {}
    for idx, items_set in enumerate(best_solution):
        bin_id = str(idx)
        total_weight = sum(weight_dict[item_id] for item_id in items_set)
        bins_dict_output[bin_id] = {
            "items": list(items_set),
            "total_weight": total_weight
        }
    
    solution = {
        "objective_value": best_objective,
        "bins": bins_dict_output
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()