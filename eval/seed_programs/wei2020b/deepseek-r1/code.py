import argparse
import json
import time
import math
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Bin Packing Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log JSONL file for intermediate solutions')
    args = parser.parse_args()

    start_time = time.time()
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    n = instance_data['n']
    W = instance_data['W']
    V = instance_data['V']
    items = instance_data['items']

    if n == 0:
        output = {
            "objective_value": 0,
            "bins": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output, f)
        return

    total_weight = sum(item['w'] for item in items)
    total_volume = sum(item['v'] for item in items)
    lb_weight = math.ceil(total_weight / W)
    lb_volume = math.ceil(total_volume / V)
    lb = max(int(lb_weight), int(lb_volume))

    items_with_index = [(idx, item) for idx, item in enumerate(items)]
    sorted_items = sorted(items_with_index, 
                         key=lambda x: max(x[1]['w'] / W, x[1]['v'] / V), 
                         reverse=True)
    
    bins_ffd = []
    for idx, item in sorted_items:
        w_i, v_i = item['w'], item['v']
        placed = False
        for bin in bins_ffd:
            if bin[0] + w_i <= W and bin[1] + v_i <= V:
                bin[0] += w_i
                bin[1] += v_i
                bin[2].append(idx)
                placed = True
                break
        if not placed:
            bins_ffd.append([w_i, v_i, [idx]])
    
    UB = len(bins_ffd)
    
    if logger:
        logger.log(UB)
    
    if lb == UB:
        solution_obj = UB
        solution_bins = []
        for bin in bins_ffd:
            solution_bins.append({
                "items": bin[2],
                "total_weight": int(bin[0]),
                "total_volume": int(bin[1])
            })
    else:
        time_elapsed = time.time() - start_time
        time_left = args.time_limit - time_elapsed
        if time_left <= 0.1:
            solution_obj = UB
            solution_bins = []
            for bin in bins_ffd:
                solution_bins.append({
                    "items": bin[2],
                    "total_weight": int(bin[0]),
                    "total_volume": int(bin[1])
                })
        elif n * UB <= 500000:
            try:
                import gurobipy as gp
                time_elapsed_model_start = time.time()
                time_remaining_for_model = time_left - (time_elapsed_model_start - start_time)
                if time_remaining_for_model <= 0.01:
                    solution_obj = UB
                    solution_bins = []
                    for bin in bins_ffd:
                        solution_bins.append({
                            "items": bin[2],
                            "total_weight": int(bin[0]),
                            "total_volume": int(bin[1])
                        })
                else:
                    model = gp.Model("bin_packing")
                    model.setParam('Threads', 1)
                    model.setParam('TimeLimit', time_remaining_for_model)
                    model.setParam('OutputFlag', 0)

                    y = model.addVars(UB, vtype=gp.GRB.BINARY, name="y")
                    x = model.addVars(n, UB, vtype=gp.GRB.BINARY, name="x")

                    for i in range(n):
                        model.addConstr(gp.quicksum(x[i,j] for j in range(UB)) == 1, f"assign_item_{i}")
                    
                    for j in range(UB):
                        model.addConstr(gp.quicksum(items[i]['w'] * x[i,j] for i in range(n)) <= W * y[j], f"weight_limit_bin{j}")
                        model.addConstr(gp.quicksum(items[i]['v'] * x[i,j] for i in range(n)) <= V * y[j], f"volume_limit_bin{j}")

                    for j in range(UB-1):
                        model.addConstr(y[j] >= y[j+1], f"symmetry_{j}")
                    
                    obj = gp.quicksum(y[j] for j in range(UB))
                    model.setObjective(obj, gp.GRB.MINIMIZE)

                    assignment = [-1] * n
                    for bin_idx, bin in enumerate(bins_ffd):
                        for item_idx in bin[2]:
                            assignment[item_idx] = bin_idx

                    for j in range(UB):
                        y[j].Start = 1.0
                    for i in range(n):
                        for j in range(UB):
                            if j == assignment[i]:
                                x[i, j].Start = 1.0
                            else:
                                x[i, j].Start = 0.0

                    if logger:
                        def mycallback(model_in, where):
                            if where == gp.GRB.Callback.MIPSOL:
                                obj_val = model_in.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                                logger.log(int(round(obj_val)))
                        model.optimize(mycallback)
                    else:
                        model.optimize()

                    solution_obj = UB
                    if model.SolCount > 0:
                        solution_obj = int(round(model.ObjVal))
                        solution_bins = []
                        for j in range(UB):
                            if y[j].X < 0.5:
                                break
                            items_in_bin = []
                            total_w = 0
                            total_v = 0
                            for i in range(n):
                                if x[i,j].X > 0.5:
                                    items_in_bin.append(i)
                                    total_w += items[i]['w']
                                    total_v += items[i]['v']
                            solution_bins.append({
                                "items": items_in_bin,
                                "total_weight": int(total_w),
                                "total_volume": int(total_v)
                            })
                    else:
                        solution_bins = []
                        for bin in bins_ffd:
                            solution_bins.append({
                                "items": bin[2],
                                "total_weight": int(bin[0]),
                                "total_volume": int(bin[1])
                            })
            except Exception as e:
                solution_obj = UB
                solution_bins = []
                for bin in bins_ffd:
                    solution_bins.append({
                        "items": bin[2],
                        "total_weight": int(bin[0]),
                        "total_volume": int(bin[1])
                    })
        else:
            solution_obj = UB
            solution_bins = []
            for bin in bins_ffd:
                solution_bins.append({
                    "items": bin[2],
                    "total_weight": int(bin[0]),
                    "total_volume": int(bin[1])
                })

    output = {
        "objective_value": solution_obj,
        "bins": solution_bins
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

if __name__ == '__main__':
    main()