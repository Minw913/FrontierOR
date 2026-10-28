import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["dimensions"]["n"]
    m = inst["dimensions"]["m"]
    p = inst["dimensions"]["p"]
    
    demands = inst["demands"]
    h = inst["inventory_costs"]
    pt = inst["processing_time"]
    st = inst["setup_times"]
    sc = inst["setup_costs"]
    cap = inst["machine_capacities"]
    max_prod = inst["max_production"]
    max_setups = inst["max_setups_per_item"]
    min_lot = inst["minimum_lot_sizes"]
    elig = inst["machine_eligibility"]

    model = gp.Model("lot_sizing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Variables
    z = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")

    w = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                ub_w = max_setups[i][k][t]
                if elig[i][k] == 0:
                    ub_w = 0
                w[i, k, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_w, name=f"w_{i}_{k}_{t}")

    y = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(m):
                    for t in range(p):
                        y[i, j, k, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{j}_{k}_{t}")

    x = {}
    for i in range(n):
        for k in range(m):
            for t1 in range(p):
                for t2 in range(t1, p):
                    ub_x = demands[i][t2] if elig[i][k] == 1 else 0
                    x[i, k, t1, t2] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub_x, name=f"x_{i}_{k}_{t1}_{t2}")

    model.update()

    # Constraints
    # 1. Demand
    for i in range(n):
        for t2 in range(p):
            model.addConstr(gp.quicksum(x[i, k, t1, t2] for k in range(m) for t1 in range(t2+1)) == demands[i][t2])

    # 2. State conservation
    for k in range(m):
        model.addConstr(gp.quicksum(z[i, k, 0] for i in range(n)) == 1)
        for t in range(1, p):
            model.addConstr(gp.quicksum(z[i, k, t] for i in range(n)) == 1)

    # 3. Flow conservation
    for i in range(n):
        for k in range(m):
            for t in range(p-1):
                model.addConstr(z[i, k, t] + gp.quicksum(y[j, i, k, t] for j in range(n) if j != i) 
                                == gp.quicksum(y[i, j, k, t] for j in range(n) if j != i) + z[i, k, t+1])
            model.addConstr(gp.quicksum(y[i, j, k, p-1] for j in range(n) if j != i) 
                            <= z[i, k, p-1] + gp.quicksum(y[j, i, k, p-1] for j in range(n) if j != i))

    # 4. Link w and y
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(gp.quicksum(y[j, i, k, t] for j in range(n) if j != i) >= w[i, k, t] - z[i, k, t])

    # 5. Max production cumulative
    for i in range(n):
        for t in range(p):
            model.addConstr(gp.quicksum(x[i, k, t1, t2] for k in range(m) for t1 in range(t, p) for t2 in range(t1, p)) <= max_prod[i][t])

    # 6. Production and w
    for i in range(n):
        for k in range(m):
            for t in range(p):
                if elig[i][k] == 1:
                    model.addConstr(gp.quicksum(x[i, k, t, t2] for t2 in range(t, p)) <= demands[i][t] * w[i, k, t] + gp.quicksum(demands[i][t2] for t2 in range(t+1, p)) * w[i, k, t])

    # 7. Min lot size
    for i in range(n):
        for k in range(m):
            for t in range(p):
                if min_lot[i] > 0:
                    model.addConstr(gp.quicksum(x[i, k, t, t2] for t2 in range(t, p)) >= min_lot[i] * gp.quicksum(y[j, i, k, t] for j in range(n) if j != i))

    # 8. Machine capacity
    for k in range(m):
        for t in range(p):
            setup_time = gp.quicksum(st[i][j][k] * y[i, j, k, t] for i in range(n) for j in range(n) if i != j)
            prod_time = pt * gp.quicksum(x[i, k, t, t2] for i in range(n) for t2 in range(t, p))
            model.addConstr(setup_time + prod_time <= cap[k][t])

    # Objective
    holding_cost = gp.quicksum((t2 - t1) * h[i] * x[i, k, t1, t2] for i in range(n) for k in range(m) for t1 in range(p) for t2 in range(t1, p))
    setup_cost = gp.quicksum(sc[i][j][k] * y[i, j, k, t] for i in range(n) for j in range(n) if i != j for k in range(m) for t in range(p))
    
    model.setObjective(holding_cost + setup_cost, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    if model.SolCount == 0:
        solution = {
            "objective_value": 0,
            "production": {},
            "setups": {},
            "carryover": {}
        }
    else:
        obj = model.ObjVal
        production = {}
        for i in range(n):
            for k in range(m):
                for t1 in range(p):
                    for t2 in range(t1, p):
                        val = x[i, k, t1, t2].X
                        if val > 1e-6:
                            production[f"x_{i}_{k}_{t1}_{t2}"] = val
        setups = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    for k in range(m):
                        for t in range(p):
                            val = y[i, j, k, t].X
                            if val > 1e-6:
                                setups[f"y_{i}_{j}_{k}_{t}"] = int(round(val))
        carryover = {}
        for i in range(n):
            for k in range(m):
                for t in range(p):
                    val = z[i, k, t].X
                    if val > 0.5:
                        carryover[f"z_{i}_{k}_{t}"] = 1
        
        solution = {
            "objective_value": obj,
            "production": production,
            "setups": setups,
            "carryover": carryover
        }
        if logger:
            logger.log_solution(obj, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()