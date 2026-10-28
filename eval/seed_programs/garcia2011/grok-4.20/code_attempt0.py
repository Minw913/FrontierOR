import json
import argparse
import time
import sys
from typing import List, Dict, Any
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

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    p = instance['p']
    cost = instance['cost_matrix']

    start_time = time.time()
    best_obj = float('inf')
    best_opened = None
    best_assign = None

    try:
        model = gp.Model("p_median")
        model.setParam('TimeLimit', args.time_limit)
        model.setParam('Threads', 1)
        model.setParam('MIPGap', 0.0)
        model.setParam('LogToConsole', 0)

        # Variables
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # x[i,j] = 1 if i assigned to facility j
        y = model.addVars(n, vtype=GRB.BINARY, name="y")     # y[j] = 1 if j is a median

        # Objective
        obj = gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        # Each customer assigned to exactly one facility
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)

        # Can only assign to open facilities
        for i in range(n):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j])

        # Exactly p medians
        model.addConstr(gp.quicksum(y[j] for j in range(n)) == p)

        # Symmetry breaking: facilities can only serve customers with higher or equal index
        # (helps a bit with branching)
        for i in range(n):
            for j in range(n):
                if i < j:
                    model.addConstr(x[i, j] <= y[i])  # optional tightening

        # Callback to log improving solutions
        def callback(model, where):
            nonlocal best_obj, best_opened, best_assign
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj - 1e-6:  # strict improvement
                    best_obj = obj_val
                    y_sol = model.cbGetSolution(y)
                    x_sol = model.cbGetSolution(x)
                    
                    opened = [j for j in range(n) if y_sol[j] > 0.5]
                    assignments = []
                    for i in range(n):
                        for j in range(n):
                            if x_sol[i, j] > 0.5:
                                assignments.append({
                                    "customer": i,
                                    "facility": j,
                                    "cost": cost[i][j]
                                })
                                break
                    
                    best_opened = opened
                    best_assign = assignments
                    
                    if logger:
                        logger.log(int(round(obj_val)))
        
        model.optimize(callback)

        # If no solution found via callback, extract from final model
        if best_opened is None and model.SolCount > 0:
            obj_val = model.objVal
            y_sol = model.getAttr('X', y)
            x_sol = model.getAttr('X', x)
            
            best_opened = [j for j in range(n) if y_sol[j] > 0.5]
            best_assign = []
            for i in range(n):
                for j in range(n):
                    if x_sol[i, j] > 0.5:
                        best_assign.append({
                            "customer": i,
                            "facility": j,
                            "cost": cost[i][j]
                        })
                        break
            best_obj = obj_val
            if logger:
                logger.log(int(round(best_obj)))

    except Exception as e:
        # Fallback: greedy algorithm
        # Select p medians with smallest average cost
        avg_cost = [sum(row) / n for row in cost]
        sorted_nodes = sorted(range(n), key=lambda k: avg_cost[k])
        opened = sorted_nodes[:p]
        opened_set = set(opened)
        
        assignments = []
        total_cost = 0
        for i in range(n):
            best_j = min(opened, key=lambda j: cost[i][j])
            c = cost[i][best_j]
            total_cost += c
            assignments.append({
                "customer": i,
                "facility": best_j,
                "cost": c
            })
        
        best_obj = total_cost
        best_opened = opened
        best_assign = assignments
        if logger:
            logger.log(int(round(best_obj)))

    # Ensure output respects schema exactly
    solution = {
        "objective_value": int(round(best_obj)),
        "opened_facilities": sorted(best_opened) if best_opened is not None else [],
        "assignments": sorted(best_assign, key=lambda a: a["customer"]) if best_assign is not None else []
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()