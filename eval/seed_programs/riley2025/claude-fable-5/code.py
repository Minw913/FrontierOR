import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    I = data["num_demand_sites"]
    J = data["num_production_sites"]
    S = data["num_scenarios"]
    T = data["num_time_periods"]
    M = data["num_module_sizes"]
    K = data["units_per_module_size"]
    u_max = float(data["u_max"])

    # module capacities
    cap = [0.0] * M
    size_cost = [0.0] * M
    for ms in data["module_sizes"]:
        cap[ms["id"]] = float(ms["capacity"])
        size_cost[ms["id"]] = float(ms.get("capital_cost_per_unit", 0.0))

    # per-unit capital costs
    unit_cost = [[size_cost[m] for _ in range(K)] for m in range(M)]
    if "module_units" in data and data["module_units"]:
        for mu in data["module_units"]:
            m = mu["module_size_id"]
            k = mu["unit_id"]
            if 0 <= m < M and 0 <= k < K:
                unit_cost[m][k] = float(mu["capital_cost"])

    # order units within a size by ascending cost (buy cheapest first)
    order = [sorted(range(K), key=lambda k: unit_cost[m][k]) for m in range(M)]

    c = data["production_costs"]  # [I][J]
    pen = data["penalty_costs"]   # [I][T]

    probs = []
    dem = []  # dem[s][i][t]
    for s_idx, sc in enumerate(data["scenarios"]):
        p = sc.get("probability", None)
        if p is None:
            p = data["scenario_probabilities"][s_idx]
        probs.append(float(p))
        dem.append(sc["demands"])

    # ------------------------------------------------------------------
    # Model.  Key observation: relocation costs are only charged between
    # ordered pairs of production sites; depot <-> site moves are free.
    # Since routing any move through the depot within the same period is
    # free and satisfies all cumulative balance constraints, the optimal
    # solution never needs direct site-to-site moves (assuming costs >= 0).
    # Therefore stationing decisions v[j,m,t,s] only need to satisfy
    # sum_j v[j,m,t,s] <= (# purchased units of size m), and w_fd / w_td
    # are recovered from period-to-period deltas of v.
    # ------------------------------------------------------------------
    env = gp.Env()
    model = gp.Model("modular_supply_chain", env=env)
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # first-stage purchase
    y = {}
    for m in range(M):
        for k in range(K):
            y[m, k] = model.addVar(vtype=GRB.BINARY, obj=unit_cost[m][k],
                                   name=f"y[{m},{k}]")
    # symmetry breaking: buy cheapest units first
    for m in range(M):
        for r in range(K - 1):
            model.addConstr(y[m, order[m][r]] >= y[m, order[m][r + 1]])

    # second-stage variables
    x = {}
    q = {}
    v = {}
    for s in range(S):
        p = probs[s]
        for i in range(I):
            for t in range(T):
                d = dem[s][i][t]
                for j in range(J):
                    x[i, j, t, s] = model.addVar(
                        lb=0.0, ub=1.0, obj=p * c[i][j] * d,
                        name=f"x[{i},{j},{t},{s}]")
                q[i, t, s] = model.addVar(
                    lb=0.0, ub=1.0, obj=p * pen[i][t] * d,
                    name=f"q[{i},{t},{s}]")
        for j in range(J):
            for m in range(M):
                for t in range(T):
                    v[j, m, t, s] = model.addVar(
                        vtype=GRB.INTEGER, lb=0, ub=K, obj=0.0,
                        name=f"v[{j},{m},{t},{s}]")

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # demand split constraints
    for s in range(S):
        for i in range(I):
            for t in range(T):
                model.addConstr(
                    gp.quicksum(x[i, j, t, s] for j in range(J)) + q[i, t, s] == 1.0)

    # capacity constraints
    for s in range(S):
        for j in range(J):
            for t in range(T):
                prod = gp.quicksum(dem[s][i][t] * x[i, j, t, s] for i in range(I))
                model.addConstr(
                    prod <= gp.quicksum(cap[m] * v[j, m, t, s] for m in range(M)))
                model.addConstr(prod <= u_max)

    # availability constraints
    for s in range(S):
        for m in range(M):
            n_m = gp.quicksum(y[m, k] for k in range(K))
            for t in range(T):
                model.addConstr(
                    gp.quicksum(v[j, m, t, s] for j in range(J)) <= n_m)

    # ------------------------------------------------------------------
    # Solution reconstruction utilities
    # ------------------------------------------------------------------
    def build_solution(yvals, xvals, qvals, vvals):
        mv = {}
        yfield = {}
        capital = 0.0
        Nm = [0] * M
        for m in range(M):
            for k in range(K):
                b = 1 if yvals[m, k] > 0.5 else 0
                yfield[f"y_{m}_{k}"] = b
                if b:
                    mv[f"y[{m},{k}]"] = 1
                    capital += unit_cost[m][k]
                    Nm[m] += 1

        total = capital
        for s in range(S):
            p = probs[s]
            op = 0.0
            for i in range(I):
                for t in range(T):
                    d = dem[s][i][t]
                    qv = qvals[i, t, s]
                    if qv < 0:
                        qv = 0.0
                    if qv > 1:
                        qv = 1.0
                    if qv > 1e-9:
                        mv[f"q[{i},{t},{s}]"] = qv
                        op += pen[i][t] * d * qv
                    for j in range(J):
                        xv = xvals[i, j, t, s]
                        if xv < 0:
                            xv = 0.0
                        if xv > 1:
                            xv = 1.0
                        if xv > 1e-9:
                            mv[f"x[{i},{j},{t},{s}]"] = xv
                            op += c[i][j] * d * xv
            # stationing & depot; relocations recovered from deltas
            for m in range(M):
                if Nm[m] > 0:
                    mv[f"z[{m},{s}]"] = Nm[m]
                site_tot = [0] * T
                for j in range(J):
                    prev = 0
                    for t in range(T):
                        vv = int(round(vvals[j, m, t, s]))
                        if vv < 0:
                            vv = 0
                        if vv > 0:
                            mv[f"v[{j},{m},{t},{s}]"] = vv
                        site_tot[t] += vv
                        delta = vv - prev
                        if delta > 0:
                            mv[f"w_fd[{j},{m},{t},{s}]"] = delta
                        elif delta < 0:
                            mv[f"w_td[{j},{m},{t},{s}]"] = -delta
                        prev = vv
                for t in range(T):
                    dep = Nm[m] - site_tot[t]
                    if dep > 0:
                        mv[f"v_depot[{m},{t},{s}]"] = dep
            total += p * op

        return total, {"objective_value": total,
                       "model_variables": mv,
                       "y": yfield}

    def extract_cb(model_):
        yv = {key: model_.cbGetSolution(var) for key, var in y.items()}
        xv = {key: model_.cbGetSolution(var) for key, var in x.items()}
        qv = {key: model_.cbGetSolution(var) for key, var in q.items()}
        vv = {key: model_.cbGetSolution(var) for key, var in v.items()}
        return yv, xv, qv, vv

    best = {"obj": float("inf"), "sol": None}

    def callback(model_, where):
        if where == GRB.Callback.MIPSOL:
            try:
                yv, xv, qv, vv = extract_cb(model_)
                obj, sol = build_solution(yv, xv, qv, vv)
                if obj < best["obj"] - 1e-9:
                    best["obj"] = obj
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    elapsed = time.time() - t_start
    remaining = max(5.0, args.time_limit - elapsed - 5.0)
    model.Params.TimeLimit = remaining

    model.optimize(callback)

    # final extraction
    if model.SolCount > 0:
        yv = {key: var.X for key, var in y.items()}
        xv = {key: var.X for key, var in x.items()}
        qv = {key: var.X for key, var in q.items()}
        vv = {key: var.X for key, var in v.items()}
        obj, sol = build_solution(yv, xv, qv, vv)
        if obj < best["obj"] - 1e-9 or best["sol"] is None:
            best["obj"] = obj
            best["sol"] = sol
            if logger:
                logger.log_solution(obj, sol)

    if best["sol"] is None:
        # trivial fallback: buy nothing, all demand unmet
        yv = {(m, k): 0.0 for m in range(M) for k in range(K)}
        xv = {key: 0.0 for key in x}
        qv = {key: 1.0 for key in q}
        vv = {key: 0.0 for key in v}
        obj, sol = build_solution(yv, xv, qv, vv)
        best["obj"] = obj
        best["sol"] = sol
        if logger:
            logger.log_solution(obj, sol)

    with open(args.solution_path, "w") as f:
        json.dump(best["sol"], f)


if __name__ == "__main__":
    main()