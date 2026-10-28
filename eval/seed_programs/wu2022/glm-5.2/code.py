import argparse
import json
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("gurobipy not available", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    num_items = inst["dimensions"]["num_items"]
    num_periods = inst["dimensions"]["num_periods"]
    num_plants = inst["dimensions"]["num_plants"]
    num_customers = inst["dimensions"]["num_customers"]

    plant_capabilities = inst["plant_capabilities"]
    customer_item_demands = inst["customer_item_demands"]
    demand = inst["demand"]
    setup_cost = inst["setup_cost"]
    production_cost = inst["production_cost"]
    holding_cost = inst["holding_cost"]
    setup_time = inst["setup_time"]
    production_time = inst["production_time"]
    capacity = inst["capacity"]
    plant_opening_cost = inst["plant_opening_cost"]
    transportation_cost = inst["transportation_cost"]

    # Capable (item, plant) pairs
    capable = [(i, p) for i in range(num_items) for p in plant_capabilities[i]]

    # Total demand per item (for big-M)
    total_demand_item = [0] * num_items
    for i in range(num_items):
        s = 0
        for c in range(num_customers):
            for t in range(num_periods):
                s += demand[i][c][t]
        total_demand_item[i] = s

    # Set of customers that order each item
    item_customers = {i: [] for i in range(num_items)}
    for c in range(num_customers):
        for i in customer_item_demands[c]:
            item_customers[i].append(c)

    # Transportation variable keys
    S_keys = []
    for i in range(num_items):
        for c in item_customers[i]:
            for p in plant_capabilities[i]:
                for t in range(num_periods):
                    if demand[i][c][t] > 0 or True:
                        S_keys.append((i, c, p, t))

    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("MIPGap", 1e-4)
    env.setParam("NumericFocus", 0)
    env.setParam("Threads", 1)
    env.start()

    model = gp.Model(env=env)
    model.Params.TimeLimit = max(1, args.time_limit - 2)
    model.Params.OutputFlag = 0

    # Decision variables
    Z = model.addVars(num_plants, vtype=GRB.BINARY, name="Z")
    U = model.addVars(capable, vtype=GRB.BINARY, name="U")
    Y_keys = [(i, p, t) for (i, p) in capable for t in range(num_periods)]
    Y = model.addVars(Y_keys, vtype=GRB.BINARY, name="Y")
    X = model.addVars(Y_keys, vtype=GRB.CONTINUOUS, lb=0, name="X")
    Inv = model.addVars(Y_keys, vtype=GRB.CONTINUOUS, lb=0, name="Inv")
    S = model.addVars(S_keys, vtype=GRB.CONTINUOUS, lb=0, name="S")

    # Objective
    obj = gp.LinExpr()
    for p in range(num_plants):
        obj += plant_opening_cost[p] * Z[p]
    for (i, p, t) in Y_keys:
        obj += setup_cost[i][p][t] * Y[i, p, t]
        obj += production_cost[i][p][t] * X[i, p, t]
        obj += holding_cost[i][p][t] * Inv[i, p, t]
    for (i, c, p, t) in S_keys:
        obj += transportation_cost[i][c][p][t] * S[i, c, p, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Single sourcing: each item assigned to exactly one capable plant
    for i in range(num_items):
        model.addConstr(gp.quicksum(U[i, p] for p in plant_capabilities[i]) == 1)

    # U <= Z
    for (i, p) in capable:
        model.addConstr(U[i, p] <= Z[p])

    # Y <= U
    for (i, p, t) in Y_keys:
        model.addConstr(Y[i, p, t] <= U[i, p])

    # X <= M * Y
    for i in range(num_items):
        M = total_demand_item[i] if total_demand_item[i] > 0 else 1
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                model.addConstr(X[i, p, t] <= M * Y[i, p, t])

    # Capacity per plant per period
    cap_plant_items = {p: [i for i in range(num_items) if p in plant_capabilities[i]] for p in range(num_plants)}
    for p in range(num_plants):
        for t in range(num_periods):
            setup_part = gp.quicksum(setup_time[i][p][t] * Y[i, p, t] for i in cap_plant_items[p])
            prod_part = gp.quicksum(production_time[i][p][t] * X[i, p, t] for i in cap_plant_items[p])
            model.addConstr(setup_part + prod_part <= capacity[p][t])

    # Inventory balance
    for (i, p) in capable:
        for t in range(num_periods):
            prev = Inv[i, p, t - 1] if t > 0 else 0
            shipped = gp.quicksum(S[i, c, p, t] for c in item_customers[i])
            model.addConstr(Inv[i, p, t] == prev + X[i, p, t] - shipped)

    # Demand satisfaction
    for c in range(num_customers):
        for i in customer_item_demands[c]:
            for t in range(num_periods):
                model.addConstr(gp.quicksum(S[i, c, p, t] for p in plant_capabilities[i]) == demand[i][c][t])

    # S <= demand * U (only assigned plant ships)
    for (i, c, p, t) in S_keys:
        model.addConstr(S[i, c, p, t] <= demand[i][c][t] * U[i, p])

    # Callback for logging incumbents
    best_obj = [float("inf")]

    def callback(cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-9:
                best_obj[0] = obj_val
                if logger:
                    try:
                        z_vals = cb.cbGetSolution([Z[p] for p in range(num_plants)])
                        u_vals = cb.cbGetSolution([U[i, p] for (i, p) in capable])
                        y_vals = cb.cbGetSolution([Y[i, p, t] for (i, p, t) in Y_keys])
                        Z_dict = {str(p): int(round(z_vals[k])) for k, p in enumerate(range(num_plants))}
                        U_dict = {f"{i}_{p}": int(round(u_vals[k])) for k, (i, p) in enumerate(capable)}
                        Y_dict = {f"{i}_{p}_{t}": int(round(y_vals[k])) for k, (i, p, t) in enumerate(Y_keys)}
                        sol = {"objective_value": float(obj_val), "Z": Z_dict, "U": U_dict, "Y": Y_dict}
                        logger.log_solution(float(obj_val), sol)
                    except Exception:
                        logger.log(float(obj_val))

    model.optimize(callback)

    def val(v):
        try:
            return v.X
        except Exception:
            return 0

    status = model.Status
    if model.SolCount > 0:
        obj_val = model.ObjVal
    else:
        obj_val = 0.0

    Z_dict = {str(p): int(round(val(Z[p]))) for p in range(num_plants)}
    U_dict = {f"{i}_{p}": int(round(val(U[i, p]))) for (i, p) in capable}
    Y_dict = {f"{i}_{p}_{t}": int(round(val(Y[i, p, t]))) for (i, p, t) in Y_keys}

    solution = {
        "objective_value": float(obj_val),
        "Z": Z_dict,
        "U": U_dict,
        "Y": Y_dict,
    }

    if logger and model.SolCount > 0:
        if obj_val < best_obj[0] - 1e-9 or best_obj[0] == float("inf"):
            logger.log_solution(float(obj_val), solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()