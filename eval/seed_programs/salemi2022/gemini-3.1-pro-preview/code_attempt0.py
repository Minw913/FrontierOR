import argparse
import json
import gurobipy as gp
from gurobipy import GRB


def solve_power_system(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n_g = data["n_generators"]
    n_s = data["n_scenarios"]
    T = data["T"]
    generators = data["generators"]
    demand = data["demand_scenarios"]
    reserve = data["reserve_scenarios"]

    # Initialize logger if provided
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    except ImportError:
        logger = None

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()

    model = gp.Model("PowerSystem_StochasticUC", env=env)

    if time_limit is not None and time_limit > 0:
        model.setParam("TimeLimit", time_limit)

    # First-stage decision variables (scenario-independent)
    x = model.addVars(n_g, T, vtype=GRB.BINARY, name="x")           # Commitment status
    y = model.addVars(n_g, T, vtype=GRB.BINARY, name="y")           # Start-up indicator
    y_bar = model.addVars(n_g, T, vtype=GRB.BINARY, name="y_bar")   # Shut-down indicator
    q = model.addVars(n_g, T, vtype=GRB.CONTINUOUS, lb=0, name="q") # Start-up cost

    # Second-stage decision variables (scenario-dependent)
    p = model.addVars(n_g, n_s, T, vtype=GRB.CONTINUOUS, lb=0, name="p")           # Production
    p_bar = model.addVars(n_g, n_s, T, vtype=GRB.CONTINUOUS, lb=0, name="p_bar")   # Available capacity

    # Objective Function
    obj_fixed = gp.quicksum(generators[g]["c_f"] * x[g, t] for g in range(n_g) for t in range(T))
    obj_startup = gp.quicksum(q[g, t] for g in range(n_g) for t in range(T))
    if n_s > 0:
        obj_prod = gp.quicksum(generators[g]["c_g"] * p[g, s, t] for g in range(n_g) for s in range(n_s) for t in range(T)) / n_s
    else:
        obj_prod = 0.0

    model.setObjective(obj_fixed + obj_startup + obj_prod, GRB.MINIMIZE)

    # Constraints
    for g in range(n_g):
        L = generators[g]["L"]
        ell = generators[g]["ell"]
        M = generators[g]["M"]
        m = generators[g]["m"]
        RU = generators[g]["RU"]
        SU = generators[g]["SU"]
        RD = generators[g]["RD"]
        SD = generators[g]["SD"]
        K = generators[g]["startup_costs_K"]

        for t in range(T):
            val_x_prev = x[g, t-1] if t > 0 else 0
            
            # Logic relationship linking commitment, start-up, and shut-down
            model.addConstr(y[g, t] - y_bar[g, t] == x[g, t] - val_x_prev, name=f"logic_{g}_{t}")

            # Minimum up-time
            start_up_sum = gp.quicksum(y[g, t_prime] for t_prime in range(max(0, t - L + 1), t + 1))
            model.addConstr(start_up_sum <= x[g, t], name=f"min_up_{g}_{t}")

            # Minimum down-time
            shut_down_sum = gp.quicksum(y_bar[g, t_prime] for t_prime in range(max(0, t - ell + 1), t + 1))
            model.addConstr(shut_down_sum <= 1 - x[g, t], name=f"min_down_{g}_{t}")

            # Time-dependent start-up costs
            for k in range(1, t + 2):
                sum_x_prev = gp.quicksum(x[g, t - j] for j in range(1, k + 1) if (t - j) >= 0)
                model.addConstr(q[g, t] >= K[k-1] * (x[g, t] - sum_x_prev), name=f"startup_cost_{g}_{t}_{k}")

            for s in range(n_s):
                # Production limits and minimum uptime constraints
                model.addConstr(p[g, s, t] >= m * x[g, t], name=f"min_prod_{g}_{s}_{t}")
                model.addConstr(p[g, s, t] <= p_bar[g, s, t], name=f"max_prod_{g}_{s}_{t}")
                model.addConstr(p_bar[g, s, t] <= M * x[g, t], name=f"max_cap_{g}_{s}_{t}")

                # Ramping rate limits
                val_p_prev = p[g, s, t-1] if t > 0 else 0
                model.addConstr(p[g, s, t] - val_p_prev <= RU * val_x_prev + SU * y[g, t], name=f"ramp_up_{g}_{s}_{t}")
                model.addConstr(val_p_prev - p[g, s, t] <= RD * x[g, t] + SD * y_bar[g, t], name=f"ramp_down_{g}_{s}_{t}")

    # System requirements (demand and reserve)
    for s in range(n_s):
        for t in range(T):
            model.addConstr(gp.quicksum(p[g, s, t] for g in range(n_g)) >= demand[s][t], name=f"demand_{s}_{t}")
            model.addConstr(gp.quicksum(p_bar[g, s, t] for g in range(n_g)) >= demand[s][t] + reserve[s][t], name=f"reserve_{s}_{t}")

    # Callback mechanism for iterative logging
    def make_callback(cb_logger):
        best_obj = [float('inf')]
        def callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if obj < best_obj[0] - 1e-5:
                    best_obj[0] = obj
                    if cb_logger:
                        cb_logger.log(obj)
        return callback

    # Optimization
    model.optimize(make_callback(logger))

    if model.SolCount == 0:
        raise Exception("No feasible solution found within the allowed time limit.")

    # Projecting outputs to match solution schema exactly
    obj_val = model.ObjVal

    out_x = [[int(round(x[g, t].X)) for t in range(T)] for g in range(n_g)]
    out_y = [[int(round(y[g, t].X)) for t in range(T)] for g in range(n_g)]
    out_y_bar = [[int(round(y_bar[g, t].X)) for t in range(T)] for g in range(n_g)]
    out_q = [[float(max(0.0, q[g, t].X)) for t in range(T)] for g in range(n_g)]

    out_p = [[[float(max(0.0, p[g, s, t].X)) for t in range(T)] for s in range(n_s)] for g in range(n_g)]
    out_p_bar = [[[float(max(0.0, p_bar[g, s, t].X)) for t in range(T)] for s in range(n_s)] for g in range(n_g)]

    solution = {
        "objective_value": obj_val,
        "x": out_x,
        "y": out_y,
        "y_bar": out_y_bar,
        "q": out_q,
        "p": out_p,
        "p_bar": out_p_bar
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    solve_power_system(args.instance_path, args.solution_path, args.time_limit, args.log_path)