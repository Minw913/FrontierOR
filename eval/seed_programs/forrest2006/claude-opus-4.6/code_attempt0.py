import argparse
import json
import time
import sys
from collections import defaultdict
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_slabs = data['num_slabs']
    num_orders = data['num_orders']
    num_colors = data['num_colors']
    max_colors = data['max_colors_per_knapsack']
    slab_weights = data['slab_weights']
    order_weights = data['order_weights']
    order_colors = data['order_colors']
    eligible_slabs_per_order = data['eligible_slabs_per_order']
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    start_time = time.time()
    
    # Build reverse mapping: for each slab, which orders are eligible
    orders_per_slab = defaultdict(list)
    for o in range(num_orders):
        for s in eligible_slabs_per_order[o]:
            orders_per_slab[s].append(o)
    
    # Colors available per slab
    colors_per_slab = {}
    for s in range(num_slabs):
        colors_per_slab[s] = set(order_colors[o] for o in orders_per_slab[s])
    
    # Try Gurobi MIP formulation
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1:
            remaining_time = 1
        
        m = gp.Model("steel_mill")
        m.Params.TimeLimit = remaining_time - 1
        m.Params.Threads = 1
        m.Params.MIPFocus = 1  # Focus on finding feasible solutions
        
        # Decision variables
        # x[o, s] = 1 if order o is assigned to slab s
        x = {}
        for o in range(num_orders):
            for s in eligible_slabs_per_order[o]:
                x[o, s] = m.addVar(vtype=GRB.BINARY, name=f"x_{o}_{s}")
        
        # y[s] = 1 if slab s is in service
        y = {}
        for s in range(num_slabs):
            if orders_per_slab[s]:
                y[s] = m.addVar(vtype=GRB.BINARY, name=f"y_{s}")
        
        # z[s, c] = 1 if color c is used on slab s
        z = {}
        for s in range(num_slabs):
            if orders_per_slab[s]:
                for c in colors_per_slab[s]:
                    z[s, c] = m.addVar(vtype=GRB.BINARY, name=f"z_{s}_{c}")
        
        m.update()
        
        # Objective: maximize 2 * sum(order_weight * assigned) - sum(slab_weight * used)
        obj = gp.LinExpr()
        for o in range(num_orders):
            for s in eligible_slabs_per_order[o]:
                obj += 2.0 * order_weights[o] * x[o, s]
        for s in range(num_slabs):
            if s in y:
                obj -= slab_weights[s] * y[s]
        
        m.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraint: each order assigned to at most one slab
        for o in range(num_orders):
            if eligible_slabs_per_order[o]:
                m.addConstr(
                    gp.quicksum(x[o, s] for s in eligible_slabs_per_order[o]) <= 1,
                    name=f"order_once_{o}"
                )
        
        # Constraint: capacity - total weight of orders on slab <= slab weight
        for s in range(num_slabs):
            if orders_per_slab[s]:
                m.addConstr(
                    gp.quicksum(order_weights[o] * x[o, s] for o in orders_per_slab[s]) <= slab_weights[s] * y[s],
                    name=f"capacity_{s}"
                )
        
        # Constraint: at most max_colors distinct colors per slab
        for s in range(num_slabs):
            if s in y and colors_per_slab[s]:
                m.addConstr(
                    gp.quicksum(z[s, c] for c in colors_per_slab[s]) <= max_colors,
                    name=f"max_colors_{s}"
                )
        
        # Constraint: if order o with color c is assigned to slab s, then z[s, c] = 1
        for s in range(num_slabs):
            if s in y:
                # Group orders by color
                color_orders = defaultdict(list)
                for o in orders_per_slab[s]:
                    color_orders[order_colors[o]].append(o)
                
                for c, orders in color_orders.items():
                    for o in orders:
                        m.addConstr(x[o, s] <= z[s, c], name=f"color_link_{o}_{s}_{c}")
        
        # Callback to log incumbent solutions
        best_obj = [float('-inf')]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val > best_obj[0]:
                    best_obj[0] = obj_val
                    if logger:
                        logger.log(obj_val)
        
        m.optimize(callback)
        
        # Extract solution
        assignments = []
        slabs_used = set()
        
        if m.SolCount > 0:
            objective_value = m.ObjVal
            
            for o in range(num_orders):
                for s in eligible_slabs_per_order[o]:
                    if x[o, s].X > 0.5:
                        assignments.append({'order': o, 'slab': s})
                        slabs_used.add(s)
                        break
            
            slabs_used = sorted(slabs_used)
        else:
            # No solution found, return empty
            objective_value = 0.0
            slabs_used = []
            if logger:
                logger.log(0.0)
        
    except ImportError:
        # Fallback greedy if Gurobi not available
        objective_value = 0.0
        assignments = []
        slabs_used = []
        if logger:
            logger.log(0.0)
    
    solution = {
        'objective_value': objective_value,
        'assignments': assignments,
        'slabs_used': slabs_used
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)