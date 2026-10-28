import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    G = instance["n_generators"]
    S = instance["n_scenarios"]
    T = instance["T"]

    gens = instance["generators"]
    demand = instance["demand_scenarios"]
    reserve = instance["reserve_scenarios"]

    model = Model("stochastic_unit_commitment")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1

    # Decision variables
    x = model.addVars(G, T, vtype=GRB.BINARY, name="x")        # on/off
    y = model.addVars(G, T, vtype=GRB.BINARY, name="y")        # startup
    z = model.addVars(G, T, vtype=GRB.BINARY, name="y_bar")    # shutdown
    q = model.addVars(G, T, lb=0.0, vtype=GRB.CONTINUOUS, name="q")  # startup cost

    p = model.addVars(G, S, T, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    pb = model.addVars(G, S, T, lb=0.0, vtype=GRB.CONTINUOUS, name="p_bar")

    # Commitment transitions + startup/shutdown logic
    for g in range(G):
        for t in range(T):
            if t == 0:
                model.addConstr(y[g, t] - z[g, t] == x[g, t], name=f"logic_{g}_{t}")
            else:
                model.addConstr(
                    y[g, t] - z[g, t] == x[g, t] - x[g, t - 1],
                    name=f"logic_{g}_{t}"
                )
            model.addConstr(y[g, t] + z[g, t] <= 1, name=f"no_simul_{g}_{t}")

    # Minimum up/down time
    for g in range(G):
        L = int(gens[g]["L"])
        ell = int(gens[g]["ell"])

        for t in range(T):
            up_start = max(0, t - L + 1)
            model.addConstr(
                quicksum(y[g, i] for i in range(up_start, t + 1)) <= x[g, t],
                name=f"min_up_{g}_{t}"
            )

            down_start = max(0, t - ell + 1)
            model.addConstr(
                quicksum(z[g, i] for i in range(down_start, t + 1)) <= 1 - x[g, t],
                name=f"min_down_{g}_{t}"
            )

    # Startup cost constraints
    for g in range(G):
        K = gens[g]["startup_costs_K"]  # length T, K[k-1] cost after k inactive periods
        for t in range(T):
            # Always charge at least the first startup level when startup happens
            model.addConstr(q[g, t] >= K[0] * y[g, t], name=f"q_base_{g}_{t}")

            # Additional lower bounds depending on consecutive prior off periods
            # k = number of preceding periods examined (1..t)
            for k in range(1, t + 1):
                expr = x[g, t] - quicksum(x[g, t - n] for n in range(1, k + 1))
                model.addConstr(
                    q[g, t] >= K[k - 1] * expr,
                    name=f"q_lag_{g}_{t}_{k}"
                )

    # Production/availability constraints and ramps
    for g in range(G):
        M = float(gens[g]["M"])
        m = float(gens[g]["m"])
        RU = float(gens[g]["RU"])
        SU = float(gens[g]["SU"])
        RD = float(gens[g]["RD"])
        SD = float(gens[g]["SD"])

        for s in range(S):
            for t in range(T):
                model.addConstr(p[g, s, t] <= pb[g, s, t], name=f"p_le_pb_{g}_{s}_{t}")
                model.addConstr(pb[g, s, t] <= M * x[g, t], name=f"pb_cap_{g}_{s}_{t}")
                model.addConstr(p[g, s, t] >= m * x[g, t], name=f"p_min_{g}_{s}_{t}")

                if t == 0:
                    # Initial condition: all generators off and no prior output
                    model.addConstr(
                        p[g, s, t] <= SU * y[g, t],
                        name=f"ramp_up_init_{g}_{s}_{t}"
                    )
                else:
                    model.addConstr(
                        p[g, s, t] - p[g, s, t - 1]
                        <= RU * x[g, t - 1] + SU * y[g, t],
                        name=f"ramp_up_{g}_{s}_{t}"
                    )
                    model.addConstr(
                        p[g, s, t - 1] - p[g, s, t]
                        <= RD * x[g, t] + SD * z[g, t],
                        name=f"ramp_down_{g}_{s}_{t}"
                    )

    # Demand and reserve coverage
    for s in range(S):
        for t in range(T):
            model.addConstr(
                quicksum(p[g, s, t] for g in range(G)) >= float(demand[s][t]),
                name=f"demand_{s}_{t}"
            )
            model.addConstr(
                quicksum(pb[g, s, t] for g in range(G)) >= float(demand[s][t]) + float(reserve[s][t]),
                name=f"reserve_{s}_{t}"
            )

    # Objective
    fixed_and_start = quicksum(
        float(gens[g]["c_f"]) * x[g, t] + q[g, t]
        for g in range(G) for t in range(T)
    )
    expected_prod = (1.0 / S) * quicksum(
        float(gens[g]["c_g"]) * p[g, s, t]
        for g in range(G) for s in range(S) for t in range(T)
    )
    model.setObjective(fixed_and_start + expected_prod, GRB.MINIMIZE)

    # Incumbent logging callback
    model._logger = logger
    model._best_logged = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._best_logged:
                m._best_logged = obj
                m._logger.log(float(obj))

    model.optimize(cb)

    # Build output
    if model.SolCount > 0:
        objective_value = float(model.ObjVal)
        x_sol = [[int(round(x[g, t].X)) for t in range(T)] for g in range(G)]
        y_sol = [[int(round(y[g, t].X)) for t in range(T)] for g in range(G)]
        z_sol = [[int(round(z[g, t].X)) for t in range(T)] for g in range(G)]
        q_sol = [[float(q[g, t].X) for t in range(T)] for g in range(G)]

        p_sol = [[[float(p[g, s, t].X) for t in range(T)] for s in range(S)] for g in range(G)]
        pb_sol = [[[float(pb[g, s, t].X) for t in range(T)] for s in range(S)] for g in range(G)]
    else:
        # Fallback (no feasible solution found)
        objective_value = 1e100
        x_sol = [[0 for _ in range(T)] for _ in range(G)]
        y_sol = [[0 for _ in range(T)] for _ in range(G)]
        z_sol = [[0 for _ in range(T)] for _ in range(G)]
        q_sol = [[0.0 for _ in range(T)] for _ in range(G)]
        p_sol = [[[0.0 for _ in range(T)] for _ in range(S)] for _ in range(G)]
        pb_sol = [[[0.0 for _ in range(T)] for _ in range(S)] for _ in range(G)]

    return {
        "objective_value": objective_value,
        "x": x_sol,
        "y": y_sol,
        "y_bar": z_sol,
        "q": q_sol,
        "p": p_sol,
        "p_bar": pb_sol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    sol = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()