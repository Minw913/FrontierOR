import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    num_plants = instance["num_plants"]
    demands = instance["customers"]["demands"]
    fixed_costs = instance["plants"]["fixed_costs"]
    capacities = instance["plants"]["capacities"]
    shipping_costs = instance["shipping_costs"]

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("Threads", 1)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("cflp", env=env)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("TimeLimit", args.time_limit)

    y = model.addVars(num_plants, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_customers, num_plants, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Objective: fixed opening costs + proportional shipping costs
    model.setObjective(
        gp.quicksum(fixed_costs[j] * y[j] for j in range(num_plants))
        + gp.quicksum(
            shipping_costs[i][j] * x[i, j]
            for i in range(num_customers)
            for j in range(num_plants)
        ),
        GRB.MINIMIZE,
    )

    # Each customer's demand fractions must sum to 1
    for i in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_plants)) == 1.0)

    # Capacity constraints (only open plants can serve)
    for j in range(num_plants):
        model.addConstr(
            gp.quicksum(demands[i] * x[i, j] for i in range(num_customers))
            <= capacities[j] * y[j]
        )

    best_obj = [float("inf")]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-9:
                best_obj[0] = obj
                if logger:
                    yv = model.cbGetSolution([y[j] for j in range(num_plants)])
                    x_list = [x[i, j] for i in range(num_customers) for j in range(num_plants)]
                    xv = model.cbGetSolution(x_list)
                    open_plants = [j for j in range(num_plants) if yv[j] > 0.5]
                    assignments = {}
                    idx = 0
                    for i in range(num_customers):
                        inner = {}
                        for j in range(num_plants):
                            val = xv[idx]
                            idx += 1
                            if val > 1e-6:
                                inner[str(j)] = val
                        if inner:
                            assignments[str(i)] = inner
                    sol = {
                        "objective_value": obj,
                        "open_plants": open_plants,
                        "assignments": assignments,
                    }
                    logger.log_solution(obj, sol)

    model.optimize(callback)

    status = model.Status
    if status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
        # Should not happen given problem guarantee, but handle gracefully
        solution = {
            "objective_value": 0.0,
            "open_plants": [],
            "assignments": {},
        }
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
        return

    obj_val = model.ObjVal
    open_plants = [j for j in range(num_plants) if y[j].X > 0.5]
    assignments = {}
    for i in range(num_customers):
        inner = {}
        for j in range(num_plants):
            val = x[i, j].X
            if val > 1e-6:
                inner[str(j)] = val
        if inner:
            assignments[str(i)] = inner

    solution = {
        "objective_value": obj_val,
        "open_plants": open_plants,
        "assignments": assignments,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()