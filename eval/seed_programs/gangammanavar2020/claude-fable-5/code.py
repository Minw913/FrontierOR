import argparse
import json
import time
import numpy as np


def find_seed(obj):
    """Recursively search the instance dict for a key containing 'seed'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and "seed" in k.lower():
                if isinstance(v, (int, float)):
                    return int(v)
        for v in obj.values():
            r = find_seed(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_seed(v)
            if r is not None:
                return r
    return None


def build_heuristic_solution(N, K, lb, h, pen, demands, probs):
    """Newsvendor-style heuristic: per-retailer quantile stock, no transshipment."""
    s = np.zeros(N)
    for i in range(N):
        d_i = demands[:, i]
        order = np.argsort(d_i)
        cum = np.cumsum(probs[order])
        target = pen[i] / (pen[i] + h[i]) if (pen[i] + h[i]) > 0 else 1.0
        idx = np.searchsorted(cum, target)
        idx = min(idx, K - 1)
        s[i] = d_i[order[idx]]
    s = np.maximum(s, lb)

    scen_out = []
    obj = 0.0
    for k in range(K):
        d = demands[k]
        f = np.minimum(s, d)
        e = s - f
        r = d - f
        q = s - e
        obj += probs[k] * (float(h @ e) + float(pen @ r))
        scen_out.append({
            "e": e.tolist(),
            "f": f.tolist(),
            "q": q.tolist(),
            "r": r.tolist(),
            "t": [[0.0] * N for _ in range(N)],
        })
    sol = {"objective_value": float(obj), "s": s.tolist(), "scenarios": scen_out}
    return float(obj), sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    N = int(inst["N"])
    fs = inst.get("first_stage", {})
    lb = np.array(fs.get("variable_lower_bounds", [0.0] * N), dtype=float)
    lb = np.maximum(lb, 0.0)

    ss = inst["second_stage"]
    h = np.array(ss["holding_costs"], dtype=float)
    pen = np.array(ss["penalty_costs"], dtype=float)
    cmean = np.array(ss["transshipment_costs_mean"], dtype=float)
    cstd = np.array(ss.get("transshipment_costs_std",
                           np.zeros((N, N)).tolist()), dtype=float)
    links = ss.get("random_cost_links", []) or []
    frac = float(ss.get("random_cost_std_fraction", 0.0) or 0.0)

    dd = ss["demand_distribution"]
    K = int(dd["num_scenarios"])
    demands = np.array(dd["scenarios"], dtype=float)
    probs = np.array(dd["probabilities"], dtype=float)

    seed = find_seed(inst)
    if seed is None:
        seed = 0
    rng = np.random.default_rng(seed)

    # Generate realized transshipment costs per scenario
    C = np.repeat(cmean[None, :, :], K, axis=0)
    if links:
        for k in range(K):
            for (i, j) in links:
                i, j = int(i), int(j)
                sd = cstd[i, j]
                if sd <= 0:
                    sd = frac * cmean[i, j]
                if sd > 0:
                    C[k, i, j] = max(rng.normal(cmean[i, j], sd), 0.0)

    # Heuristic fallback / initial incumbent
    heur_obj, heur_sol = build_heuristic_solution(N, K, lb, h, pen, demands, probs)
    best_sol = heur_sol
    if logger:
        logger.log_solution(heur_obj, heur_sol)

    # Build the extensive-form LP with Gurobi
    try:
        import gurobipy as gp
        from gurobipy import GRB

        elapsed = time.time() - t_start
        remaining = max(args.time_limit - elapsed - 3.0, 5.0)

        m = gp.Model("transshipment")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = remaining

        s_var = m.addMVar(N, lb=lb, name="s")

        ones = np.ones(N)
        ub_t = np.full((N, N), GRB.INFINITY)
        np.fill_diagonal(ub_t, 0.0)

        f_vars, e_vars, q_vars, r_vars, t_vars = [], [], [], [], []
        obj_expr = 0
        for k in range(K):
            f = m.addMVar(N, lb=0.0)
            e = m.addMVar(N, lb=0.0)
            q = m.addMVar(N, lb=0.0)
            r = m.addMVar(N, lb=0.0)
            t = m.addMVar((N, N), lb=0.0, ub=ub_t)
            d = demands[k]

            # 1) f_i + outbound + e_i = s_i
            m.addConstr(f + (t @ ones) + e == s_var)
            # 2) f_i + inbound + r_i = d_i
            m.addConstr(f + (ones @ t) + r == d)
            # 3) e_i + q_i = s_i
            m.addConstr(e + q == s_var)
            # 4) system balance (implied but included)
            m.addConstr(ones @ r + ones @ q == float(d.sum()))

            p = probs[k]
            obj_expr = obj_expr + p * (h @ e) + p * (pen @ r) \
                + p * (C[k].reshape(-1) @ t.reshape(-1))

            f_vars.append(f); e_vars.append(e); q_vars.append(q)
            r_vars.append(r); t_vars.append(t)

        m.setObjective(obj_expr, GRB.MINIMIZE)
        m.optimize()

        has_sol = m.Status == GRB.OPTIMAL or (m.SolCount > 0)
        if has_sol:
            s_val = np.maximum(np.array(s_var.X), 0.0)
            scen_out = []
            for k in range(K):
                scen_out.append({
                    "e": np.maximum(np.array(e_vars[k].X), 0.0).tolist(),
                    "f": np.maximum(np.array(f_vars[k].X), 0.0).tolist(),
                    "q": np.maximum(np.array(q_vars[k].X), 0.0).tolist(),
                    "r": np.maximum(np.array(r_vars[k].X), 0.0).tolist(),
                    "t": np.maximum(np.array(t_vars[k].X), 0.0).tolist(),
                })
            lp_obj = float(m.ObjVal)
            lp_sol = {"objective_value": lp_obj, "s": s_val.tolist(),
                      "scenarios": scen_out}
            if lp_obj <= heur_obj + 1e-9:
                best_sol = lp_sol
                if logger:
                    logger.log_solution(lp_obj, lp_sol)
    except Exception:
        # keep heuristic solution
        pass

    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh)


if __name__ == "__main__":
    main()