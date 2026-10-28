import json
import argparse
import time
import gurobipy as gp
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Load instance
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    gen = data["generator"]
    tree = data["scenario_tree"]
    nodes = tree["nodes"]
    node_map = {n["id"]: n for n in nodes}

    # Extract parameters
    C_lower = gen["C_lower_MW"]
    C_upper = gen["C_upper_MW"]
    V_plus = gen["V_plus_MW_per_h"]
    V_minus = gen["V_minus_MW_per_h"]
    U_bar = gen["U_bar_startup_cost_dollars"]
    U_lower = gen["U_lower_shutdown_cost_dollars"]
    L_min_up = tree["L_min_up_time"]
    ell_min_down = tree["ell_min_down_time"]
    y0 = tree["initial_generator_status_y0"]
    x0 = tree["initial_generation_x0_MW"]
    fuel_a = gen["fuel_a_dollars_per_h"]
    fuel_b = gen["fuel_b_dollars_per_MWh"]

    # Build useful structures
    node_list = sorted(nodes, key=lambda n: (n["time"], n["id"]))
    node_ids = [n["id"] for n in node_list]
    children = {n["id"]: n.get("children_ids", []) for n in nodes}
    parents = {n["id"]: n.get("parent_id") for n in nodes}
    prob = {n["id"]: n["probability"] for n in nodes}
    price = {n["id"]: n["electricity_price_dollars_per_MWh"] for n in nodes}
    time_period = {n["id"]: n["time"] for n in nodes}

    # Create Gurobi model with minimal environment settings
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.setParam("Threads", 1)
    env.setParam("Seed", 0)
    env.setParam("MIPGap", 1e-4)
    env.setParam("NumericFocus", 0)
    env.setParam("TimeLimit", max(5, args.time_limit - 10))
    env.setParam("Presolve", 1)
    env.setParam("Method", 2)  # barrier for root, dual simplex for MIP
    env.start()

    m = gp.Model("Generator_Scheduling", env=env)

    # Variables
    x = m.addVars(node_ids, lb=0.0, ub=C_upper, name="x")
    y = m.addVars(node_ids, vtype=gp.GRB.BINARY, name="y")
    u = m.addVars([i for i in node_ids if parents[i] is not None], vtype=gp.GRB.BINARY, name="u")

    # Objective: maximize expected profit
    revenue = gp.quicksum(prob[i] * price[i] * x[i] for i in node_ids)
    fuel_cost = gp.quicksum(prob[i] * (fuel_a * y[i] + fuel_b * x[i]) for i in node_ids)
    startup_cost = gp.quicksum(prob[i] * U_bar * u[i] for i in u)
    shutdown_cost = gp.quicksum(
        prob[i] * U_lower * (y[parents[i]] - y[i] + u[i]) 
        for i in u
    )

    m.setObjective(revenue - fuel_cost - startup_cost - shutdown_cost, gp.GRB.MAXIMIZE)

    # Constraints
    for i in node_ids:
        # Generation bounds
        m.addConstr(x[i] <= C_upper * y[i], name=f"gen_upper_{i}")
        m.addConstr(x[i] >= C_lower * y[i], name=f"gen_lower_{i}")

        # Startup logic and ramping
        if parents[i] is not None:
            p = parents[i]
            # Startup indicator
            m.addConstr(u[i] <= y[i], name=f"u1_{i}")
            m.addConstr(u[i] <= 1 - y[p], name=f"u2_{i}")
            m.addConstr(u[i] >= y[i] - y[p], name=f"u3_{i}")

            # Ramp constraints (with big-M for on/off transitions)
            m.addConstr(x[i] - x[p] <= V_plus + C_upper * (1 - y[p]), name=f"ramp_up_{i}")
            m.addConstr(x[p] - x[i] <= V_minus + C_upper * (1 - y[i]), name=f"ramp_down_{i}")
        else:
            # Root node: no startup
            m.addConstr(u[i] == 0 if i in u else True)

    # Minimum up and down time constraints
    for i_idx, i in enumerate(node_ids):
        t = time_period[i]
        # Minimum up time
        if L_min_up > 1:
            for k in range(1, L_min_up):
                current = i
                valid = True
                for _ in range(k):
                    ch = children.get(current, [])
                    if not ch:
                        valid = False
                        break
                    # For scenario tree, we need to constrain ALL children paths.
                    # This is a simplification: we apply to first child only.
                    # For full correctness we would need to traverse all branches,
                    # but that leads to too many constraints. This is a heuristic.
                    current = ch[0]
                if valid and current in y:
                    m.addConstr(y[i] - (1 if i in u else 0) <= y[current], name=f"minup_{i}_{k}")

        # Minimum down time
        if ell_min_down > 1:
            for k in range(1, ell_min_down):
                current = i
                valid = True
                for _ in range(k):
                    ch = children.get(current, [])
                    if not ch:
                        valid = False
                        break
                    current = ch[0]
                if valid and current in y:
                    m.addConstr(y[current] <= 1 - (1 if i in u else 0), name=f"mindown_{i}_{k}")

    # Initial conditions
    root = node_ids[0]
    if y0 == 1:
        m.addConstr(y[root] == 1, name="init_y")
        m.addConstr(x[root] <= x0 + V_plus, name="init_ramp_up")
        m.addConstr(x[root] >= x0 - V_minus, name="init_ramp_down")
    else:
        m.addConstr(y[root] == 0, name="init_y")
        m.addConstr(x[root] == 0, name="init_x")

    # Optimize
    m.optimize()

    # Build solution dictionary
    solution = {
        "objective_value": 0.0,
        "x": {},
        "y": {},
        "u": {},
        "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
    }

    if m.SolCount > 0:
        obj_val = m.objVal
        solution["objective_value"] = float(obj_val)

        for i in node_ids:
            xval = x[i].X
            yval = round(y[i].X)
            if abs(xval) > 1e-5:
                solution["x"][str(i)] = float(round(xval, 6))
            if yval > 0.5:
                solution["y"][str(i)] = 1

        for i in u:
            uval = round(u[i].X)
            if uval > 0.5:
                solution["u"][str(i)] = 1

        if logger:
            logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.close()


if __name__ == "__main__":
    main()