import json
import time
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def build_arcflow_model(W, items, logger, remaining_time):
    model = Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(0.1, remaining_time))

    nodes = list(range(W + 1))
    arcs = []
    for i in range(W):
        arcs.append(('loss', i, i + 1))
    for item in items:
        w = item['width']
        type_id = item['type_id']
        for i in range(0, W - w + 1):
            arcs.append((type_id, i, i + w))

    f = {}
    for arc in arcs:
        f[arc] = model.addVar(vtype=GRB.INTEGER, name=f"f_{arc[0]}_{arc[1]}_{arc[2]}", lb=0)
    total_flow = model.addVar(vtype=GRB.INTEGER, name="total_flow", lb=0)

    for i in range(1, W):
        inflow = []
        outflow = []
        for arc in arcs:
            if arc[2] == i:
                inflow.append(f[arc])
            if arc[1] == i:
                outflow.append(f[arc])
        model.addConstr(quicksum(inflow) == quicksum(outflow))

    outflow_0 = [f[arc] for arc in arcs if arc[1] == 0]
    inflow_W = [f[arc] for arc in arcs if arc[2] == W]
    model.addConstr(quicksum(outflow_0) == total_flow)
    model.addConstr(quicksum(inflow_W) == total_flow)

    for item in items:
        type_id = item['type_id']
        demand = item['demand']
        item_arcs = [f[arc] for arc in arcs if arc[0] == type_id]
        model.addConstr(quicksum(item_arcs) >= demand)

    model.setObjective(total_flow, GRB.MINIMIZE)
    return model, f, arcs, total_flow

def arcflow_callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(int(round(obj)))

def extract_solution(model, f, arcs, total_flow):
    objective_value = int(round(total_flow.X))
    model_variables = {"total_flow": objective_value}
    for arc in arcs:
        val = f[arc].X
        if val > 0.5:
            if arc[0] == 'loss':
                key = f"loss_{arc[1]}_{arc[2]}"
            else:
                key = f"item_{arc[0]}_{arc[1]}_{arc[2]}"
            model_variables[key] = int(round(val))
    return objective_value, model_variables

def run_ffd(W, items):
    items_list = []
    for item in items:
        for _ in range(item['demand']):
            items_list.append((item['width'], item['type_id']))
    items_list.sort(key=lambda x: x[0], reverse=True)
    
    bins = []
    bin_totals = []
    
    for width, type_id in items_list:
        found = False
        for i in range(len(bins)):
            if bin_totals[i] + width <= W:
                bins[i].append((width, type_id))
                bin_totals[i] += width
                found = True
                break
        if not found:
            bins.append([(width, type_id)])
            bin_totals.append(width)
    
    total_flow = len(bins)
    model_variables = {"total_flow": total_flow}
    arc_flow = {}
    
    for bin_items in bins:
        current = 0
        for width, type_id in bin_items:
            next_state = current + width
            arc = (type_id, current, next_state)
            arc_flow[arc] = arc_flow.get(arc, 0) + 1
            current = next_state
        for i in range(current, W):
            arc = ('loss', i, i + 1)
            arc_flow[arc] = arc_flow.get(arc, 0) + 1
    
    for arc, count in arc_flow.items():
        if arc[0] == 'loss':
            key = f"loss_{arc[1]}_{arc[2]}"
        else:
            key = f"item_{arc[0]}_{arc[1]}_{arc[2]}"
        model_variables[key] = count
    
    return total_flow, model_variables

def main(args):
    start_time = time.time()
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    W = instance['bin_capacity']
    items = instance['items']
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    total_arcs = W
    for item in items:
        total_arcs += (W - item['width'] + 1)
    
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    
    if total_arcs <= 10000000 and remaining_time > 0.1:
        try:
            model, f, arcs, total_flow = build_arcflow_model(W, items, logger, remaining_time)
            model.optimize(lambda model, where: arcflow_callback(model, where, logger))
            
            if model.SolCount > 0:
                objective_value, model_variables = extract_solution(model, f, arcs, total_flow)
                if logger:
                    logger.log_solution(objective_value, model_variables)
            else:
                objective_value, model_variables = run_ffd(W, items)
                if logger:
                    logger.log_solution(objective_value, model_variables)
        except Exception as e:
            objective_value, model_variables = run_ffd(W, items)
            if logger:
                logger.log_solution(objective_value, model_variables)
    else:
        objective_value, model_variables = run_ffd(W, items)
        if logger:
            logger.log_solution(objective_value, model_variables)
    
    solution = {
        "objective_value": objective_value,
        "model_variables": model_variables
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args)