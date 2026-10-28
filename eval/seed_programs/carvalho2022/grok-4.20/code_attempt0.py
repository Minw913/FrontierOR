import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['dimensions']['n']
    m = data['dimensions']['m']
    p = data['dimensions']['p']

    demands = data['demands']
    inventory_costs = data['inventory_costs']
    processing_time = data['processing_time']
    setup_times = data['setup_times']
    setup_costs = data['setup_costs']
    machine_capacities = data['machine_capacities']
    max_production = data['max_production']
    max_setups_per_item = data['max_setups_per_item']
    minimum_lot_sizes = data['minimum_lot_sizes']
    shortcut_items = data['shortcut_items']
    machine_eligibility = data['machine_eligibility']

    # Precompute cumulative demands and big-M like values for linking
    total_demand = [[0] * p for _ in range(n)]
    for i in range(n):
        for t in range(p - 1, -1, -1):
            total_demand[i][t] = demands[i][t] + (total_demand[i][t + 1] if t + 1 < p else 0)

    model = gp.Model("MultiItemLotSizingWithSetupCarryover")

    # Decision variables
    # Production: x[i][k][t][s] : amount of item i produced on machine k in period t to satisfy demand in s >= t
    x = [[[[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}_{k}_{t}_{s}")
            for s in range(t, p)] for t in range(p)] for k in range(m)] for i in range(n)]

    # y[i][j][k][t] : number of setups from i to j on machine k in period t (integer)
    y = [[[[model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_setups_per_item[i][k][t] if i != j else 0,
            name=f"y_{i}_{j}_{k}_{t}")
            for t in range(p)] for k in range(m)] for j in range(n)] for i in range(n)]

    # z[i][k][t] : 1 if machine k starts period t configured for item i
    z = [[[model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")
           for t in range(p)] for k in range(m)] for i in range(n)]

    # Inventory I[i][t] after period t (for t=0..p, I[p]=0)
    I = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{t}")
          for t in range(p + 1)] for i in range(n)]

    # Auxiliary: production quantity of item i on machine k in period t
    prod_qty = [[[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"prod_{i}_{k}_{t}")
                 for t in range(p)] for k in range(m)] for i in range(n)]

    # Binary for whether item i is produced on machine k in t (via setup or carry)
    w = [[[model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{k}_{t}")
           for t in range(p)] for k in range(m)] for i in range(n)]

    # Objective: inventory + setup costs
    inventory_cost = gp.quicksum(inventory_costs[i] * (I[i][t] - I[i][t+1]) * (p - t) for i in range(n) for t in range(p))
    setup_cost = gp.quicksum(setup_costs[i][j][k][t] * y[i][j][k][t]
                             for i in range(n) for j in range(n) for k in range(m) for t in range(p) if i != j)
    model.setObjective(inventory_cost + setup_cost, GRB.MINIMIZE)

    # Constraints

    # 1. Inventory balance
    for i in range(n):
        # Initial inventory = 0
        model.addConstr(I[i][0] == 0)
        for t in range(p):
            inflow = gp.quicksum(x[i][k][t][s] for k in range(m) for s in range(t, p))
            model.addConstr(I[i][t] + inflow == demands[i][t] + I[i][t + 1])
        model.addConstr(I[i][p] == 0)

    # 2. Link prod_qty to x
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(prod_qty[i][k][t] == gp.quicksum(x[i][k][t][s] for s in range(t, p)))

    # 3. Capacity constraints
    for k in range(m):
        for t in range(p):
            prod_time = processing_time * gp.quicksum(prod_qty[i][k][t] for i in range(n))
            setup_time = gp.quicksum(setup_times[i][j][k][t] * y[i][j][k][t] for i in range(n) for j in range(n))
            model.addConstr(prod_time + setup_time <= machine_capacities[k][t])

    # 4. Production only if set up (w indicator)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(prod_qty[i][k][t] <= total_demand[i][0] * w[i][k][t])

    # 5. Each machine starts with exactly one item configured
    for k in range(m):
        for t in range(p):
            model.addConstr(gp.quicksum(z[i][k][t] for i in range(n)) == 1)

    # 6. Setup flow and carryover linking
    for k in range(m):
        for t in range(p):
            for j in range(n):
                # Incoming setups + carryin = outgoing setups + production indicator
                inflow = gp.quicksum(y[i][j][k][t] for i in range(n) if i != j) + z[j][k][t]
                outflow = gp.quicksum(y[j][l][k][t] for l in range(n) if l != j)
                model.addConstr(inflow == outflow + w[j][k][t])

    # 7. Carryover from previous period
    for k in range(m):
        for t in range(1, p):
            for i in range(n):
                model.addConstr(z[i][k][t] <= gp.quicksum(y[j][i][k][t-1] for j in range(n)) + z[i][k][t-1])

    # 8. Initial period carryin: only possible if eligible (but z free, will be constrained by eligibility)
    for k in range(m):
        for i in range(n):
            if machine_eligibility[i][k] == 0:
                model.addConstr(z[i][k][0] == 0)

    # 9. Eligibility and max setups already encoded in var ub and here
    for i in range(n):
        for k in range(m):
            for t in range(p):
                if machine_eligibility[i][k] == 0:
                    model.addConstr(w[i][k][t] == 0)
                    model.addConstr(gp.quicksum(y[j][i][k][t] for j in range(n)) == 0)
                    model.addConstr(z[i][k][t] == 0)

    # 10. Minimum lot size for non-carryover production
    for i in range(n):
        for k in range(m):
            for t in range(p):
                # If produced without carry-over, then at least min lot if w=1 and not carried
                # We use: prod_qty >= min_lot * (w - z)
                minlot = minimum_lot_sizes[i]
                if minlot > 0:
                    model.addConstr(prod_qty[i][k][t] >= minlot * (w[i][k][t] - z[i][k][t]))

    # 11. Max production bounds
    for i in range(n):
        for t in range(p):
            total_prod = gp.quicksum(x[i][k][t][s] for k in range(m) for s in range(t, p))
            model.addConstr(total_prod <= max_production[i][t])

    # 12. Max setups per item already in variable bounds

    # Symmetry breaking / tightening: limit setups
    for i in range(n):
        for j in range(n):
            for k in range(m):
                for t in range(p):
                    model.addConstr(y[i][j][k][t] <= 2)  # practical upper bound

    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = 0.01
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Callback to log improved solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < model._best_obj or model._best_obj is None:
                model._best_obj = obj
                if logger:
                    logger.log(float(obj))

    model._best_obj = None
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        production = {}
        for i in range(n):
            for k in range(m):
                for t in range(p):
                    for s in range(t, p):
                        val = x[i][k][t][s].X
                        if val > 1e-6:
                            key = f"x_{i}_{k}_{t}_{s}"
                            production[key] = float(val)

        setups = {}
        for i_idx in range(n):
            for j_idx in range(n):
                for k_idx in range(m):
                    for t_idx in range(p):
                        val = y[i_idx][j_idx][k_idx][t_idx].X
                        if val > 0.5:
                            key = f"y_{i_idx}_{j_idx}_{k_idx}_{t_idx}"
                            setups[key] = int(round(val))

        carryover = {}
        for i_idx in range(n):
            for k_idx in range(m):
                for t_idx in range(p):
                    val = z[i_idx][k_idx][t_idx].X
                    if abs(val - 1.0) < 1e-5:
                        key = f"z_{i_idx}_{k_idx}_{t_idx}"
                        carryover[key] = 1

        solution = {
            "objective_value": float(objective_value),
            "production": production,
            "setups": setups,
            "carryover": carryover
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or no solution - write empty
        solution = {
            "objective_value": 0.0,
            "production": {},
            "setups": {},
            "carryover": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()