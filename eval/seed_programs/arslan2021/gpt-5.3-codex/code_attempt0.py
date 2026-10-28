import argparse
import json
import math
import sys
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    items = instance["items"]
    params = instance["problem_parameters"]

    C = float(params["knapsack_capacity"])
    Gamma = float(params["uncertainty_budget"])
    n = len(items)

    w = []
    p = []
    d = []
    o = []
    q = []
    for it in items:
        w.append(float(it["weight"]))
        p.append(float(it["nominal_profit"]))
        d.append(float(it["max_degradation"]))
        o.append(float(it["outsource_penalty"]))
        q.append(float(it["repair_capacity"]))

    m = Model("robust_knapsack_lp_recourse")

    # Required solver settings
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = int(time_limit)
    m.Params.OutputFlag = 0

    # Variables
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    y = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    r = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="r")

    # Dual variables for budgeted uncertainty max problem:
    # max sum d_i * (y_i-r_i) * xi_i
    # s.t. 0<=xi_i<=1, sum xi_i <= Gamma
    # dual: min Gamma*pi + sum rho_i
    #       pi + rho_i >= d_i*(y_i-r_i), pi,rho_i >=0
    pi = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="pi")
    rho = m.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="rho")

    # Constraints
    for i in range(n):
        m.addConstr(y[i] <= x[i], name=f"link_y_x_{i}")
        m.addConstr(r[i] <= y[i], name=f"link_r_y_{i}")
        m.addConstr(pi + rho[i] >= d[i] * (y[i] - r[i]), name=f"robust_dual_{i}")

    m.addConstr(quicksum(w[i] * y[i] + q[i] * r[i] for i in range(n)) <= C, name="capacity")

    # Objective:
    # first-stage baseline + deterministic recourse part + worst-case degradation term (dualized)
    obj = (
        quicksum((o[i] - p[i]) * x[i] for i in range(n))
        + quicksum((-o[i]) * y[i] for i in range(n))
        + Gamma * pi
        + quicksum(rho[i] for i in range(n))
    )
    m.setObjective(obj, GRB.MINIMIZE)

    cb_state = {"best": float("inf")}

    def mip_callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-12 < cb_state["best"]:
                cb_state["best"] = obj_val
                x_sol = {}
                for i in range(n):
                    xv = model.cbGetSolution(x[i])
                    x_sol[str(i)] = int(xv >= 0.5)
                sol_dict = {"objective_value": float(obj_val), "x": x_sol}
                logger.log_solution(float(obj_val), sol_dict)

    m.optimize(mip_callback)

    if m.SolCount > 0:
        x_out = {str(i): int(x[i].X >= 0.5) for i in range(n)}
        objective_value = float(m.ObjVal)
    else:
        # Fallback if no feasible solution (should be rare if instance is feasible)
        x_out = {str(i): 0 for i in range(n)}
        objective_value = 1e30

    return {"objective_value": objective_value, "x": x_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()