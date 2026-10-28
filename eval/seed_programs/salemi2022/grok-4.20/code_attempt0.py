import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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

    n_g = instance['n_generators']
    n_s = instance['n_scenarios']
    T = instance['T']
    gens = instance['generators']
    demand = instance['demand_scenarios']
    reserve = instance['reserve_scenarios']

    # Precompute startup cost lookup: startup_costs[g][k-1] = cost after exactly k off periods
    startup = []
    for g in range(n_g):
        startup.append(gens[g]['startup_costs_K'])

    model = gp.Model("UnitCommitment")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.001)
    model.setParam('LogToConsole', 0)

    # First-stage variables (common to all scenarios)
    x = model.addVars(n_g, T, vtype=GRB.BINARY, name="x")  # on
    y = model.addVars(n_g, T, vtype=GRB.BINARY, name="y")  # startup
    ybar = model.addVars(n_g, T, vtype=GRB.BINARY, name="ybar")  # shutdown
    q = model.addVars(n_g, T, lb=0, name="q")  # startup cost

    # Second-stage variables
    p = model.addVars(n_g, n_s, T, lb=0, name="p")
    pbar = model.addVars(n_g, n_s, T, lb=0, name="pbar")

    # Objective: fixed + startup + expected production cost
    fixed_cost = gp.quicksum(gens[g]['c_f'] * x[g, t] for g in range(n_g) for t in range(T))
    startup_cost = gp.quicksum(q[g, t] for g in range(n_g) for t in range(T))
    prod_cost = (1.0 / n_s) * gp.quicksum(
        gens[g]['c_g'] * p[g, s, t] for g in range(n_g) for s in range(n_s) for t in range(T)
    )
    model.setObjective(fixed_cost + startup_cost + prod_cost, GRB.MINIMIZE)

    # Initial state: all generators offline at t=-1
    for g in range(n_g):
        model.addConstr(y[g, 0] - ybar[g, 0] == x[g, 0])
        model.addConstr(y[g, 0] == 0)  # cannot startup from negative time, effectively handled by below

    # Commitment logic and minimum up/down time
    for g in range(n_g):
        Lg = gens[g]['L']
        ellg = gens[g]['ell']
        for t in range(T):
            # x[t] = x[t-1] + y[t] - ybar[t]
            if t > 0:
                model.addConstr(x[g, t] == x[g, t-1] + y[g, t] - ybar[g, t])

            # Minimum up time
            for ell in range(1, Lg + 1):
                if t - ell + 1 >= 0:
                    sum_y = gp.quicksum(y[g, t - k] for k in range(ell))
                    model.addConstr(sum_y <= x[g, t])

            # Minimum down time
            for ell in range(1, ellg + 1):
                if t - ell + 1 >= 0:
                    sum_ybar = gp.quicksum(ybar[g, t - k] for k in range(ell))
                    model.addConstr(sum_ybar <= 1 - x[g, t])

    # Startup cost constraints (McCormick linearization of piecewise)
    for g in range(n_g):
        for t in range(T):
            for k in range(1, t + 2):  # k=1..t+1
                if k > len(startup[g]):
                    continue
                cost_k = startup[g][k - 1]
                if t - k + 1 >= 0:
                    sum_x_past = gp.quicksum(x[g, t - j] for j in range(1, k))
                    model.addConstr(q[g, t] >= cost_k * (x[g, t] - sum_x_past))
                else:
                    # before horizon: all previous were off
                    model.addConstr(q[g, t] >= cost_k * x[g, t])

    # Production and capacity limits
    for g in range(n_g):
        Mg = gens[g]['M']
        mg = gens[g]['m']
        SUg = gens[g]['SU']
        SDg = gens[g]['SD']
        for s in range(n_s):
            for t in range(T):
                model.addConstr(p[g, s, t] <= pbar[g, s, t])
                model.addConstr(p[g, s, t] <= Mg * x[g, t])
                model.addConstr(pbar[g, s, t] <= Mg * x[g, t])
                model.addConstr(p[g, s, t] >= mg * x[g, t])
                model.addConstr(p[g, s, t] == 0 if not x[g, t] else 0)  # redundant but ok
                # when off, force to zero
                model.addConstr(p[g, s, t] <= Mg * x[g, t])
                model.addConstr(pbar[g, s, t] <= Mg * x[g, t])

    # Ramping constraints
    for g in range(n_g):
        RUg = gens[g]['RU']
        SUg = gens[g]['SU']
        RDg = gens[g]['RD']
        SDg = gens[g]['SD']
        for s in range(n_s):
            for t in range(T):
                # from t-1 to t
                if t == 0:
                    # initial p[-1]=0
                    model.addConstr(p[g, s, 0] <= SUg * y[g, 0])
                    model.addConstr(p[g, s, 0] <= SDg * (1 - ybar[g, 0]) + 0)  # not shutting down at 0 typically
                else:
                    # ramp up
                    model.addConstr(
                        p[g, s, t] - p[g, s, t-1] <= RUg * x[g, t-1] + SUg * y[g, t]
                    )
                    # ramp down
                    model.addConstr(
                        p[g, s, t-1] - p[g, s, t] <= RDg * x[g, t] + SDg * ybar[g, t]
                    )

    # Demand and reserve
    for s in range(n_s):
        for t in range(T):
            model.addConstr(
                gp.quicksum(p[g, s, t] for g in range(n_g)) >= demand[s][t]
            )
            model.addConstr(
                gp.quicksum(pbar[g, s, t] for g in range(n_g)) >= demand[s][t] + reserve[s][t]
            )

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
        sol_x = [[int(x[g, t].X + 0.5) for t in range(T)] for g in range(n_g)]
        sol_y = [[int(y[g, t].X + 0.5) for t in range(T)] for g in range(n_g)]
        sol_ybar = [[int(ybar[g, t].X + 0.5) for t in range(T)] for g in range(n_g)]
        sol_q = [[float(q[g, t].X) for t in range(T)] for g in range(n_g)]
        sol_p = [[[float(p[g, s, t].X) for t in range(T)] for s in range(n_s)] for g in range(n_g)]
        sol_pbar = [[[float(pbar[g, s, t].X) for t in range(T)] for s in range(n_s)] for g in range(n_g)]
        obj_val = float(model.objVal)
    else:
        # Infeasible - return dummy (should not happen for valid instances)
        obj_val = 1e9
        sol_x = [[0] * T for _ in range(n_g)]
        sol_y = [[0] * T for _ in range(n_g)]
        sol_ybar = [[0] * T for _ in range(n_g)]
        sol_q = [[0.0] * T for _ in range(n_g)]
        sol_p = [[[0.0] * T for _ in range(n_s)] for _ in range(n_g)]
        sol_pbar = [[[0.0] * T for _ in range(n_s)] for _ in range(n_g)]

    solution = {
        "objective_value": obj_val,
        "x": sol_x,
        "y": sol_y,
        "y_bar": sol_ybar,
        "q": sol_q,
        "p": sol_p,
        "p_bar": sol_pbar
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.close()

if __name__ == "__main__":
    main()