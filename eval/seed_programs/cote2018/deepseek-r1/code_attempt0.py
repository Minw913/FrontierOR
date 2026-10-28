import json
import sys
import time
import bisect
from gurobipy import Model, GRB, quicksum

def main(instance_path, solution_path, time_limit):
    start_time = time.time()
    with open(instance_path) as f:
        instance = json.load(f)
    
    W = instance['bin_capacity']
    items = instance['items']
    
    # Calculate total arcs for exact model
    total_arcs = W  # loss arcs
    for item in items:
        total_arcs += (W - item['width'] + 1)
    
    if total_arcs <= 10000000:
        # Build arc-flow model
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)
        model.setParam('Threads', 1)
        
        # Create nodes: 0 to W
        nodes = list(range(W + 1))
        
        # Create arcs: loss arcs and item arcs
        arcs = []
        # Loss arcs
        for i in range(W):
            arcs.append(('loss', i, i + 1))
        # Item arcs
        for item in items:
            w = item['width']
            type_id = item['type_id']
            for i in range(0, W - w + 1):
                arcs.append((type_id, i, i + w))
        
        # Create variables
        f = {}
        for arc in arcs:
            f[arc] = model.addVar(vtype=GRB.INTEGER, name=f"f_{arc[0]}_{arc[1]}_{arc[2]}")
        total_flow = model.addVar(vtype=GRB.INTEGER, name="total_flow")
        
        # Flow conservation constraints
        for i in range(1, W):
            inflow = []
            outflow = []
            for arc in arcs:
                if arc[2] == i:  # arc ends at i
                    inflow.append(f[arc])
                if arc[1] == i:  # arc starts at i
                    outflow.append(f[arc])
            model.addConstr(quicksum(inflow) == quicksum(outflow))
        
        # Source and sink constraints
        outflow_0 = [f[arc] for arc in arcs if arc[1] == 0]
        inflow_W = [f[arc] for arc in arcs if arc[2] == W]
        model.addConstr(quicksum(outflow_0) == total_flow)
        model.addConstr(quicksum(inflow_W) == total_flow)
        
        # Demand constraints
        for item in items:
            type_id = item['type_id']
            demand = item['demand']
            item_arcs = [f[arc] for arc in arcs if arc[0] == type_id]
            model.addConstr(quicksum(item_arcs) >= demand)
        
        # Objective
        model.setObjective(total_flow, GRB.MINIMIZE)
        
        # Optimize
        model.optimize()
        
        # Extract solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
            objective_value = int(round(total_flow.X))
            model_variables = {"total_flow": objective_value}
            for arc in arcs:
                if f[arc].X > 0.5:
                    if arc[0] == 'loss':
                        key = f"loss_{arc[1]}_{arc[2]}"
                    else:
                        key = f"item_{arc[0]}_{arc[1]}_{arc[2]}"
                    model_variables[key] = int(round(f[arc].X))
        else:
            # Fallback to FFD if no solution found
            objective_value, model_variables = run_ffd(W, items)
    else:
        objective_value, model_variables = run_ffd(W, items)
    
    # Write solution
    solution = {
        "objective_value": objective_value,
        "model_variables": model_variables
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def run_ffd(W, items):
    # Expand items
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

if __name__ == '__main__':
    instance_path = sys.argv[1]
    solution_path = sys.argv[2]
    time_limit = int(sys.argv[3])
    main(instance_path, solution_path, time_limit)