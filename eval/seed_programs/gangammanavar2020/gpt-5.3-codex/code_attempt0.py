import argparse
import json
import math
import time
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_seed(instance: Dict[str, Any]) -> int:
    # Try several possible locations for seed to be robust.
    if "random_seed" in instance:
        return int(instance["random_seed"])
    if "seed" in instance:
        return int(instance["seed"])
    ss = instance.get("second_stage", {})
    if "random_seed" in ss:
        return int(ss["random_seed"])
    if "seed" in ss:
        return int(ss["seed"])
    return 0


def build_cost_scenarios(instance: Dict[str, Any]) -> List[List[List[float]]]:
    """
    Build one realized transshipment cost matrix per demand scenario.
    Uncertain links are sampled from Normal(mean, std), clamped at 0.
    Deterministic links use mean.
    """
    ss = instance["second_stage"]
    mean_c = ss["transshipment_costs_mean"]
    std_c = ss.get("transshipment_costs_std", None)
    random_links = ss.get("random_cost_links", [])
    std_frac = float(ss.get("random_cost_std_fraction", 0.0))
    K = int(ss["demand_distribution"]["num_scenarios"])
    N = int(instance["N"])

    # Determine uncertain link set
    uncertain = set()
    for pair in random_links:
        if len(pair) == 2:
            uncertain.add((int(pair[0]), int(pair[1])))

    # If random_cost_links absent/empty but std matrix has positives, infer uncertain links
    if not uncertain and std_c is not None:
        for i in range(N):
            for j in range(N):
                if i != j and float(std_c[i][j]) > 0:
                    uncertain.add((i, j))

    # RNG
    try:
        import numpy as np
        rng = np.random.default_rng(get_seed(instance))
        normal_sample = lambda mu, sd: float(rng.normal(mu, sd))
    except Exception:
        # Fallback to python random if numpy unavailable
        import random
        random.seed(get_seed(instance))
        normal_sample = lambda mu, sd: float(random.gauss(mu, sd))

    scenarios_c = []
    for _k in range(K):
        mat = [[float(mean_c[i][j]) for j in range(N)] for i in range(N)]
        for i in range(N):
            mat[i][i] = 0.0
        for (i, j) in uncertain:
            mu = float(mean_c[i][j])
            if std_c is not None:
                sd = float(std_c[i][j])
            else:
                sd = abs(std_frac * mu)
            val = normal_sample(mu, sd)
            if val < 0.0:
                val = 0.0
            mat[i][j] = val
        scenarios_c.append(mat)
    return scenarios_c


def heuristic_solution(instance: Dict[str, Any], probs: List[float]) -> Dict[str, Any]:
    """
    Always-feasible fallback:
      s = lower bounds
      e = s, f = 0, q = 0, r = demand, t = 0
    """
    N = int(instance["N"])
    ss = instance["second_stage"]
    K = int(ss["demand_distribution"]["num_scenarios"])
    demands = ss["demand_distribution"]["scenarios"]
    holding = [float(x) for x in ss["holding_costs"]]
    penalty = [float(x) for x in ss["penalty_costs"]]
    lbs = [float(x) for x in instance["first_stage"]["variable_lower_bounds"]]

    s = lbs[:]
    scenarios_out = []
    obj = 0.0

    for k in range(K):
        d = [float(x) for x in demands[k]]
        e = s[:]
        f = [0.0] * N
        q = [0.0] * N
        r = d[:]
        t = [[0.0] * N for _ in range(N)]

        scen_cost = 0.0
        for i in range(N):
            scen_cost += holding[i] * e[i] + penalty[i] * r[i]
        obj += probs[k] * scen_cost

        scenarios_out.append({"e": e, "f": f, "q": q, "r": r, "t": t})

    return {
        "objective_value": float(obj),
        "s": s,
        "scenarios": scenarios_out,
    }


def solve_instance(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    N = int(instance["N"])
    I = range(N)

    ss = instance["second_stage"]
    dd = ss["demand_distribution"]
    K = int(dd["num_scenarios"])
    Kset = range(K)

    demands = [[float(x) for x in dd["scenarios"][k]] for k in Kset]
    raw_probs = [float(x) for x in dd["probabilities"]]
    p_sum = sum(raw_probs)
    if p_sum > 0:
        probs = [x / p_sum for x in raw_probs]
    else:
        probs = [1.0 / K] * K

    holding = [float(x) for x in ss["holding_costs"]]
    penalty = [float(x) for x in ss["penalty_costs"]]
    lbs = [float(x) for x in instance["first_stage"]["variable_lower_bounds"]]

    cost_scenarios = build_cost_scenarios(instance)

    # Initial feasible solution for logging/fallback
    best_sol = heuristic_solution(instance, probs)
    best_obj = best_sol["objective_value"]
    if logger:
        logger.log(best_obj)

    model = gp.Model("stochastic_transshipment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))

    # Variables
    s = model.addVars(N, lb=0.0, name="s")
    for i in I:
        s[i].LB = lbs[i]

    e = model.addVars(((k, i) for k in Kset for i in I), lb=0.0, name="e")
    f = model.addVars(((k, i) for k in Kset for i in I), lb=0.0, name="f")
    q = model.addVars(((k, i) for k in Kset for i in I), lb=0.0, name="q")
    r = model.addVars(((k, i) for k in Kset for i in I), lb=0.0, name="r")
    t = model.addVars(((k, i, j) for k in Kset for i in I for j in I if i != j), lb=0.0, name="t")

    # Constraints
    for k in Kset:
        total_demand_k = sum(demands[k])

        # 1) local fulfillment + outbound + ending inventory = s
        for i in I:
            model.addConstr(
                f[k, i] + gp.quicksum(t[k, i, j] for j in I if j != i) + e[k, i] == s[i],
                name=f"stock_balance_{k}_{i}",
            )

        # 2) local fulfillment + inbound + backlog = realized demand
        for i in I:
            model.addConstr(
                f[k, i] + gp.quicksum(t[k, j, i] for j in I if j != i) + r[k, i] == demands[k][i],
                name=f"demand_balance_{k}_{i}",
            )

        # 3) ending inventory + replenishment quantity = s
        for i in I:
            model.addConstr(
                e[k, i] + q[k, i] == s[i],
                name=f"replen_balance_{k}_{i}",
            )

        # 4) system-wide backlog + replenishment = system demand
        model.addConstr(
            gp.quicksum(r[k, i] for i in I) + gp.quicksum(q[k, i] for i in I) == total_demand_k,
            name=f"global_balance_{k}",
        )

    # Objective
    obj = gp.LinExpr()
    for k in Kset:
        pk = probs[k]
        for i in I:
            obj += pk * holding[i] * e[k, i]
            obj += pk * penalty[i] * r[k, i]
        cmat = cost_scenarios[k]
        for i in I:
            for j in I:
                if i != j:
                    obj += pk * float(cmat[i][j]) * t[k, i, j]

    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    if model.SolCount > 0:
        sol_obj = float(model.ObjVal)

        # Extract solution
        s_val = [float(s[i].X) for i in I]
        scenarios_out = []

        for k in Kset:
            e_val = [float(e[k, i].X) for i in I]
            f_val = [float(f[k, i].X) for i in I]
            q_val = [float(q[k, i].X) for i in I]
            r_val = [float(r[k, i].X) for i in I]
            t_mat = [[0.0] * N for _ in I]
            for i in I:
                for j in I:
                    if i != j:
                        t_mat[i][j] = float(t[k, i, j].X)

            scenarios_out.append({
                "e": e_val,
                "f": f_val,
                "q": q_val,
                "r": r_val,
                "t": t_mat
            })

        candidate = {
            "objective_value": sol_obj,
            "s": s_val,
            "scenarios": scenarios_out
        }

        if sol_obj + 1e-9 < best_obj:
            best_sol = candidate
            best_obj = sol_obj
            if logger:
                logger.log(best_obj)
        else:
            # still return optimizer solution if at least as good numerically
            best_sol = candidate
            best_obj = sol_obj

    return best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger=logger)

    # Ensure schema keys exist and proper types
    out = {
        "objective_value": float(solution["objective_value"]),
        "s": [float(x) for x in solution["s"]],
        "scenarios": []
    }

    for scen in solution["scenarios"]:
        out["scenarios"].append({
            "e": [float(x) for x in scen["e"]],
            "f": [float(x) for x in scen["f"]],
            "q": [float(x) for x in scen["q"]],
            "r": [float(x) for x in scen["r"]],
            "t": [[float(v) for v in row] for row in scen["t"]],
        })

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()