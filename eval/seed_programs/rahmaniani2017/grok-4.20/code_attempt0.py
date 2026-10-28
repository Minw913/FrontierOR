import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import math

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

    n = instance['num_nodes']
    m = instance['num_commodities']
    tau = instance['parameters']['tau']
    chi = instance['parameters']['chi']
    delta = instance['parameters']['delta']
    TC = instance['parameters']['TC']
    nodes = instance['nodes']
    distances = instance['distances']
    commodities = instance['commodities']

    fixed_cost = [nodes[i]['fixed_cost'] for i in range(n)]
    flow = [commodities[k]['flow'] for k in range(m)]
    o = [commodities[k]['origin'] for k in range(m)]
    d = [commodities[k]['destination'] for k in range(m)]

    # Precompute effective distances
    disc = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                disc[i][j] = 0.0
            else:
                disc[i][j] = tau * distances[i][j]

    start_time = time.time()
    best_obj = float('inf')
    best_open = []
    best_assign = []

    model = gp.Model("HubLocation")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, n, n, vtype=GRB.BINARY, name="x")

    # Objective
    trans_cost = gp.quicksum(flow[k] * (
        chi * distances[o[k]][i] + disc[i][j] + delta * distances[j][d[k]]
    ) * x[k, i, j] for k in range(m) for i in range(n) for j in range(n))
    fixed = gp.quicksum(fixed_cost[i] * y[i] for i in range(n))
    model.setObjective(trans_cost + fixed, GRB.MINIMIZE)

    # Constraints
    for k in range(m):
        model.addConstr(gp.quicksum(x[k, i, j] for i in range(n) for j in range(n)) == 1)

    for k in range(m):
        for i in range(n):
            for j in range(n):
                model.addConstr(x[k, i, j] <= y[i])
                model.addConstr(x[k, i, j] <= y[j])

    # If origin and destination are same, still allow single or two hubs
    # No additional constraints needed

    # Callback for logging incumbents
    def callback(model, where):
        nonlocal best_obj, best_open, best_assign
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-6:
                best_obj = obj
                yval = model.cbGetSolution(y)
                open_hubs = [i for i in range(n) if yval[i] > 0.5]
                if logger:
                    logger.log(float(obj))
                
                # Extract assignment (using symmetry, prefer i <= j)
                assign_list = []
                xval = model.cbGetSolution(x)
                for k in range(m):
                    found = False
                    for i in range(n):
                        for j in range(n):
                            if xval[k, i, j] > 0.5:
                                if i == j:
                                    hub_edge = [i]
                                else:
                                    hub_edge = [i, j] if i <= j else [j, i]
                                trans_cost_k = flow[k] * (
                                    chi * distances[o[k]][hub_edge[0]] +
                                    (0 if len(hub_edge)==1 else tau * distances[hub_edge[0]][hub_edge[1]]) +
                                    delta * distances[hub_edge[-1]][d[k]]
                                )
                                assign_list.append({
                                    "commodity": k,
                                    "origin": o[k],
                                    "destination": d[k],
                                    "hub_edge": hub_edge,
                                    "cost": float(trans_cost_k)
                                })
                                found = True
                                break
                        if found:
                            break
                best_open = open_hubs[:]
                best_assign = assign_list[:]

    model.optimize(callback)

    # If no solution found by callback (rare), get from final solution
    if best_obj == float('inf') and model.SolCount > 0:
        best_obj = model.ObjVal
        yval = model.getAttr('X', y)
        best_open = [i for i in range(n) if yval[i] > 0.5]
        xval = model.getAttr('X', x)
        best_assign = []
        for k_idx in range(m):
            found = False
            for i in range(n):
                for j in range(n):
                    if xval[k_idx, i, j] > 0.5:
                        if i == j:
                            hub_edge = [i]
                        else:
                            hub_edge = [min(i,j), max(i,j)]
                        trans = flow[k_idx] * (
                            chi * distances[o[k_idx]][hub_edge[0]] + 
                            (0.0 if len(hub_edge) == 1 else tau * distances[hub_edge[0]][hub_edge[1]]) +
                            delta * distances[hub_edge[-1]][d[k_idx]]
                        )
                        best_assign.append({
                            "commodity": k_idx,
                            "origin": o[k_idx],
                            "destination": d[k_idx],
                            "hub_edge": hub_edge,
                            "cost": float(trans)
                        })
                        found = True
                        break
                if found: break

    # If still no solution (e.g. timeout before any feasible), use a greedy heuristic
    if best_obj == float('inf'):
        # Simple greedy: open all nodes as hubs, assign each commodity to closest single hub
        best_open = list(range(n))
        best_assign = []
        total_cost = sum(fixed_cost)
        for k in range(m):
            min_c = float('inf')
            best_h = 0
            for h in range(n):
                c = flow[k] * (chi * distances[o[k]][h] + delta * distances[h][d[k]])
                if c < min_c:
                    min_c = c
                    best_h = h
            best_assign.append({
                "commodity": k,
                "origin": o[k],
                "destination": d[k],
                "hub_edge": [best_h],
                "cost": float(min_c)
            })
            total_cost += min_c
        best_obj = total_cost
        if logger:
            logger.log(float(best_obj))

    # Write solution
    solution = {
        "objective_value": float(best_obj),
        "open_hubs": sorted(best_open),
        "assignments": best_assign
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()