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
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    T = data["num_periods"]
    W = data["num_warehouses"]
    R = data["num_retailers"]
    S = data["num_scenarios"]
    r2w = list(data["retailer_to_warehouse"])
    # normalize warehouse indices to 0-based if they appear 1-based
    if len(r2w) > 0 and max(r2w) >= W:
        r2w = [x - 1 for x in r2w]
    p = data["scenario_probabilities"]
    h0 = float(data["holding_cost_plant"])
    h1 = float(data["holding_cost_warehouse"])
    h2 = [float(x) for x in data["holding_costs_retailers"]]
    K0 = data["setup_costs_plant"]
    K1 = data["setup_costs_warehouses"]
    K2 = data["setup_costs_retailers"]
    d = data["demands"]  # [s][r][t]
    init_setups = bool(data.get("initial_setups_imposed", True))

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    NF = 1 + W + R  # facilities

    def assemble(objval, yv, prod, delw, delr, invp, invw, invr):
        def rnd(x):
            v = max(0.0, x)
            return round(v, 6)

        sol = {"objective_value": float(objval)}
        sv = {}
        for fac in range(NF):
            for t in range(T):
                sv["y_%d_%d" % (fac, t)] = int(round(yv[(fac, t)]))
        sol["setup_variables"] = sv
        sol["production_plant"] = {
            "%d_%d" % (t, s): rnd(prod[t][s]) for t in range(T) for s in range(S)
        }
        sol["delivery_warehouse"] = {
            "%d_%d_%d" % (w, t, s): rnd(delw[w][t][s])
            for w in range(W) for t in range(T) for s in range(S)
        }
        sol["delivery_retailer"] = {
            "%d_%d_%d" % (r, t, s): rnd(delr[r][t][s])
            for r in range(R) for t in range(T) for s in range(S)
        }
        sol["inventory_plant"] = {
            "%d_%d" % (t, s): rnd(invp[t][s]) for t in range(T) for s in range(S)
        }
        sol["inventory_warehouse"] = {
            "%d_%d_%d" % (w, t, s): rnd(invw[w][t][s])
            for w in range(W) for t in range(T) for s in range(S)
        }
        sol["inventory_retailer"] = {
            "%d_%d_%d" % (r, t, s): rnd(invr[r][t][s])
            for r in range(R) for t in range(T) for s in range(S)
        }
        return sol

    def zeros2():
        return [[0.0] * S for _ in range(T)]

    def zeros3(n):
        return [[[0.0] * S for _ in range(T)] for _ in range(n)]

    # ------------------------------------------------------------------
    # Fallback feasible solution: everything produced/shipped in period 0
    # ------------------------------------------------------------------
    fb_prod = zeros2()
    fb_delw = zeros3(W)
    fb_delr = zeros3(R)
    fb_invp = zeros2()
    fb_invw = zeros3(W)
    fb_invr = zeros3(R)
    fb_hold = 0.0
    for s in range(S):
        for r in range(R):
            w = r2w[r]
            tot = 0.0
            for t in range(T):
                dv = d[s][r][t]
                tot += dv
                fb_hold += p[s] * h2[r] * t * dv
            fb_prod[0][s] += tot
            fb_delw[w][0][s] += tot
            fb_delr[r][0][s] += tot
            # retailer inventory: sum of demands due strictly after k
            rem = tot
            for k in range(T):
                rem -= d[s][r][k]
                fb_invr[r][k][s] += rem
    fb_setup = float(K0[0]) + sum(float(K1[w][0]) for w in range(W)) + \
        sum(float(K2[r][0]) for r in range(R))
    fb_obj = fb_setup + fb_hold
    fb_yv = {(fac, t): (1 if t == 0 else 0) for fac in range(NF) for t in range(T)}
    fb_sol = assemble(fb_obj, fb_yv, fb_prod, fb_delw, fb_delr,
                      fb_invp, fb_invw, fb_invr)
    if logger:
        logger.log_solution(fb_obj, fb_sol)

    # ------------------------------------------------------------------
    # Build MIP (per retailer-period-demand disaggregated flow model)
    # ------------------------------------------------------------------
    m = gp.Model("sowmr")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    # setup variables with objective coefficients
    y = {}
    for t in range(T):
        y[(0, t)] = m.addVar(vtype=GRB.BINARY, obj=float(K0[t]), name="y_0_%d" % t)
    for w in range(W):
        for t in range(T):
            y[(1 + w, t)] = m.addVar(vtype=GRB.BINARY, obj=float(K1[w][t]),
                                     name="y_%d_%d" % (1 + w, t))
    for r in range(R):
        for t in range(T):
            y[(1 + W + r, t)] = m.addVar(vtype=GRB.BINARY, obj=float(K2[r][t]),
                                         name="y_%d_%d" % (1 + W + r, t))
    if init_setups:
        for fac in range(NF):
            y[(fac, 0)].LB = 1.0

    triples = []
    for s in range(S):
        for r in range(R):
            for t in range(T):
                if d[s][r][t] > 0:
                    triples.append((s, r, t))

    keys = [(s, r, t, k) for (s, r, t) in triples for k in range(t + 1)]
    keys_s2 = [(s, r, t, k) for (s, r, t) in triples for k in range(t)]

    q0 = m.addVars(keys, lb=0.0, name="q0")
    q1 = m.addVars(keys, lb=0.0, name="q1")
    q2 = m.addVars(keys, lb=0.0, name="q2")
    s0 = m.addVars(keys, lb=0.0, obj=[p[kk[0]] * h0 for kk in keys], name="s0")
    s1 = m.addVars(keys, lb=0.0, obj=[p[kk[0]] * h1 for kk in keys], name="s1")
    s2 = m.addVars(keys_s2, lb=0.0, obj=[p[kk[0]] * h2[kk[1]] for kk in keys_s2],
                   name="s2")

    for (s, r, t) in triples:
        dv = float(d[s][r][t])
        w = r2w[r]
        for k in range(t + 1):
            prev0 = s0[s, r, t, k - 1] if k > 0 else 0.0
            prev1 = s1[s, r, t, k - 1] if k > 0 else 0.0
            m.addConstr(q1[s, r, t, k] + s0[s, r, t, k] == prev0 + q0[s, r, t, k])
            m.addConstr(q2[s, r, t, k] + s1[s, r, t, k] == prev1 + q1[s, r, t, k])
            if k < t:
                prev2 = s2[s, r, t, k - 1] if k > 0 else 0.0
                m.addConstr(s2[s, r, t, k] == prev2 + q2[s, r, t, k])
            else:
                prev2 = s2[s, r, t, t - 1] if t > 0 else 0.0
                m.addConstr(prev2 + q2[s, r, t, t] == dv)
            m.addConstr(q0[s, r, t, k] <= dv * y[(0, k)])
            m.addConstr(q1[s, r, t, k] <= dv * y[(1 + w, k)])
            m.addConstr(q2[s, r, t, k] <= dv * y[(1 + W + r, k)])
        m.addConstr(gp.quicksum(q0[s, r, t, k] for k in range(t + 1)) == dv)
        m.addConstr(gp.quicksum(q1[s, r, t, k] for k in range(t + 1)) == dv)

    m.ModelSense = GRB.MINIMIZE
    m.update()

    # MIP start = fallback solution
    for (fac, t), var in y.items():
        var.Start = 1.0 if t == 0 else 0.0
    for (s, r, t, k) in keys:
        dv = float(d[s][r][t])
        v = dv if k == 0 else 0.0
        q0[s, r, t, k].Start = v
        q1[s, r, t, k].Start = v
        q2[s, r, t, k].Start = v
        s0[s, r, t, k].Start = 0.0
        s1[s, r, t, k].Start = 0.0
    for (s, r, t, k) in keys_s2:
        s2[s, r, t, k].Start = float(d[s][r][t])

    def build_from_vals(vals, objval):
        def get(v):
            return vals[v.index]

        prod = zeros2()
        delw = zeros3(W)
        delr = zeros3(R)
        invp = zeros2()
        invw = zeros3(W)
        invr = zeros3(R)
        for (s, r, t, k), var in q0.items():
            prod[k][s] += get(var)
        for (s, r, t, k), var in q1.items():
            delw[r2w[r]][k][s] += get(var)
        for (s, r, t, k), var in q2.items():
            delr[r][k][s] += get(var)
        for (s, r, t, k), var in s0.items():
            invp[k][s] += get(var)
        for (s, r, t, k), var in s1.items():
            invw[r2w[r]][k][s] += get(var)
        for (s, r, t, k), var in s2.items():
            invr[r][k][s] += get(var)
        yv = {key: get(var) for key, var in y.items()}
        return assemble(objval, yv, prod, delw, delr, invp, invw, invr)

    best_obj = [fb_obj]
    best_sol = [fb_sol]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                try:
                    vals = model.cbGetSolution(model.getVars())
                    sol = build_from_vals(vals, obj)
                    best_obj[0] = obj
                    best_sol[0] = sol
                    if logger:
                        logger.log_solution(obj, sol)
                except Exception:
                    if logger:
                        logger.log(obj)
                    best_obj[0] = obj

    remaining = args.time_limit - (time.time() - start_time) - 2.0
    m.Params.TimeLimit = max(1.0, remaining)

    try:
        m.optimize(cb)
    except gp.GurobiError:
        pass

    final_sol = best_sol[0]
    if m.SolCount > 0:
        try:
            vals = m.getAttr("X", m.getVars())
            obj = m.ObjVal
            sol = build_from_vals(vals, obj)
            final_sol = sol
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log_solution(obj, sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()