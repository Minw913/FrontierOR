import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=3600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    N = instance["N"]
    S = instance["S"]
    k = instance["k"]
    beta = instance["beta"]
    gamma = instance["gamma"]
    mu_bar = instance["mu_bar"]
    mu = instance["mu"]
    p_s = instance["p_s"]
    scenarios = instance["scenarios"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Build MIQP model
    model = gp.Model("portfolio_cvar")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("OutputFlag", 0)

    # Decision variables
    x = model.addVars(N, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
    t = model.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

    # Objective: (1/(2*gamma)) * sum(x_i^2) + a + (1/(1-beta)) * sum_s p_s * t_s
    obj = gp.QuadExpr()
    for i in range(N):
        obj += (1.0 / (2.0 * gamma)) * x[i] * x[i]
    obj += a
    for s in range(S):
        obj += (p_s / (1.0 - beta)) * t[s]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint: sum(x_i) = 1
    model.addConstr(gp.quicksum(x[i] for i in range(N)) == 1.0, name="budget")

    # Constraint: sum(z_i) <= k
    model.addConstr(gp.quicksum(z[i] for i in range(N)) <= k, name="cardinality")

    # Constraint: x_i <= z_i (linking)
    for i in range(N):
        model.addConstr(x[i] <= z[i], name=f"link_{i}")

    # Constraint: expected return >= mu_bar
    model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(N)) >= mu_bar, name="return_req")

    # Constraint: t_s >= -sum_i r_{s,i} * x_i - a for each scenario s
    for s in range(S):
        scenario_return = gp.quicksum(scenarios[s][i] * x[i] for i in range(N))
        model.addConstr(t[s] >= -scenario_return - a, name=f"cvar_{s}")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution([x[i] for i in range(N)])
            z_vals = model.cbGetSolution([z[i] for i in range(N)])
            a_val = model.cbGetSolution(a)
            t_vals = model.cbGetSolution([t[s] for s in range(S)])

            sol_dict = {
                "objective_value": obj_val,
                "x": [float(v) for v in x_vals],
                "z": [int(round(v)) for v in z_vals],
                "a": float(a_val),
                "v": float(sum(p_s * t_vals[s] for s in range(S)) / (1.0 - beta))
            }
            if logger:
                logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        x_vals = [x[i].X for i in range(N)]
        z_vals = [int(round(z[i].X)) for i in range(N)]
        a_val = a.X
        t_vals = [t[s].X for s in range(S)]

        # Compute v from t values
        v_val = sum(p_s * t_vals[s] for s in range(S)) / (1.0 - beta)

        # Compute objective
        reg_term = sum(x_vals[i] ** 2 for i in range(N)) / (2.0 * gamma)
        obj_val = reg_term + a_val + v_val

        solution = {
            "objective_value": obj_val,
            "x": x_vals,
            "z": z_vals,
            "a": a_val,
            "v": v_val
        }
    else:
        # Fallback: try to construct a feasible solution
        # Pick top-k assets by expected return
        sorted_indices = sorted(range(N), key=lambda i: mu[i], reverse=True)[:k]
        x_vals = [0.0] * N
        z_vals = [0] * N
        for i in sorted_indices:
            z_vals[i] = 1
            x_vals[i] = 1.0 / k

        # Compute scenario losses
        a_val = 0.0
        max_loss = 0.0
        for s in range(S):
            loss = -sum(scenarios[s][i] * x_vals[i] for i in range(N))
            if loss > max_loss:
                max_loss = loss
        a_val = max_loss

        v_val = 0.0
        for s in range(S):
            loss = -sum(scenarios[s][i] * x_vals[i] for i in range(N))
            excess = max(loss - a_val, 0.0)
            v_val += p_s * excess
        v_val /= (1.0 - beta)

        reg_term = sum(x_vals[i] ** 2 for i in range(N)) / (2.0 * gamma)
        obj_val = reg_term + a_val + v_val

        solution = {
            "objective_value": obj_val,
            "x": x_vals,
            "z": z_vals,
            "a": a_val,
            "v": v_val
        }

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    # Write solution
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()