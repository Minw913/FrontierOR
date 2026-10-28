import argparse
import json
import math
import time
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    # ----------------------------
    # Read basic dimensions
    # ----------------------------
    I = instance["num_demand_sites"]
    J = instance["num_production_sites"]
    S = instance["num_scenarios"]
    T = instance["num_time_periods"]
    M = instance["num_module_sizes"]
    K = instance["units_per_module_size"]
    u_max = float(instance["u_max"])

    scenarios = instance["scenarios"]
    scenario_probabilities = instance.get("scenario_probabilities", [1.0 / S] * S)

    # Prefer scenario-specific probabilities when available
    p_s = []
    for s in range(S):
        p_s.append(float(scenarios[s].get("probability", scenario_probabilities[s])))

    production_costs = instance["production_costs"]
    relocation_costs = instance["relocation_costs"]
    penalty_costs = instance["penalty_costs"]

    capacities = [float(ms["capacity"]) for ms in instance["module_sizes"]]
    default_capex = [float(ms["capital_cost_per_unit"]) for ms in instance["module_sizes"]]

    # Unit-level capital costs (fallback to module default)
    capex_mk = [[default_capex[m] for _ in range(K)] for m in range(M)]
    for rec in instance.get("module_units", []):
        m = int(rec["module_size_id"])
        k = int(rec["unit_id"])
        if 0 <= m < M and 0 <= k < K:
            capex_mk[m][k] = float(rec.get("capital_cost", default_capex[m]))

    demands = [[[float(scenarios[s]["demands"][i][t]) for t in range(T)] for i in range(I)] for s in range(S)]

    def prod_cost(i, j, t):
        v = production_costs[i][j]
        if isinstance(v, list):
            return float(v[t])
        return float(v)

    def reloc_cost(j, jp, m, t):
        v = relocation_costs[j][jp][m]
        if isinstance(v, list):
            return float(v[t])
        return float(v)

    # ----------------------------
    # Build model
    # ----------------------------
    model = Model("modular_supply_chain")

    # Required parameters
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)
    model.Params.OutputFlag = 0

    # First-stage: purchase decisions
    y = {}
    for m in range(M):
        for k in range(K):
            y[m, k] = model.addVar(vtype=GRB.BINARY, name=f"y[{m},{k}]")

    # Symmetry breaking for identical units within each module size
    for m in range(M):
        for k in range(K - 1):
            model.addConstr(y[m, k] >= y[m, k + 1], name=f"sym_y[{m},{k}]")

    # Scenario copies of purchased count (for schema compatibility)
    z = {}
    for m in range(M):
        for s in range(S):
            z[m, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"z[{m},{s}]")

    # Second-stage variables
    x = {}        # fraction served from production site
    q = {}        # unmet fraction
    v = {}        # units at production site
    v_depot = {}  # units at depot
    w = {}        # relocation prod->prod
    w_fd = {}     # depot->prod
    w_td = {}     # prod->depot

    for s in range(S):
        for t in range(T):
            for i in range(I):
                q[i, t, s] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"q[{i},{t},{s}]")
                for j in range(J):
                    x[i, j, t, s] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x[{i},{j},{t},{s}]")

            for m in range(M):
                v_depot[m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"v_depot[{m},{t},{s}]")
                for j in range(J):
                    v[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"v[{j},{m},{t},{s}]")
                    w_fd[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"w_fd[{j},{m},{t},{s}]")
                    w_td[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"w_td[{j},{m},{t},{s}]")

                for j in range(J):
                    for jp in range(J):
                        if j == jp:
                            continue
                        w[j, jp, m, t, s] = model.addVar(
                            vtype=GRB.INTEGER, lb=0, ub=K, name=f"w[{j},{jp},{m},{t},{s}]"
                        )

    model.update()

    # z ties to first-stage y
    for m in range(M):
        ysum_m = quicksum(y[m, k] for k in range(K))
        for s in range(S):
            model.addConstr(z[m, s] == ysum_m, name=f"z_link[{m},{s}]")

    # Demand fulfillment fractions
    for s in range(S):
        for t in range(T):
            for i in range(I):
                model.addConstr(
                    quicksum(x[i, j, t, s] for j in range(J)) + q[i, t, s] == 1.0,
                    name=f"demand_frac[{i},{t},{s}]"
                )

    # Inventory flow balances
    for s in range(S):
        for m in range(M):
            total_purchased_m = quicksum(y[m, k] for k in range(K))
            for t in range(T):
                inflow_depot = quicksum(w_td[j, m, t, s] for j in range(J))
                outflow_depot = quicksum(w_fd[j, m, t, s] for j in range(J))

                if t == 0:
                    model.addConstr(
                        v_depot[m, t, s] == total_purchased_m + inflow_depot - outflow_depot,
                        name=f"depot_bal[{m},{t},{s}]"
                    )
                else:
                    model.addConstr(
                        v_depot[m, t, s] == v_depot[m, t - 1, s] + inflow_depot - outflow_depot,
                        name=f"depot_bal[{m},{t},{s}]"
                    )

                for j in range(J):
                    inflow_prod = w_fd[j, m, t, s] + quicksum(
                        w[jp, j, m, t, s] for jp in range(J) if jp != j
                    )
                    outflow_prod = w_td[j, m, t, s] + quicksum(
                        w[j, jp, m, t, s] for jp in range(J) if jp != j
                    )

                    if t == 0:
                        model.addConstr(
                            v[j, m, t, s] == inflow_prod - outflow_prod,
                            name=f"prod_bal[{j},{m},{t},{s}]"
                        )
                    else:
                        model.addConstr(
                            v[j, m, t, s] == v[j, m, t - 1, s] + inflow_prod - outflow_prod,
                            name=f"prod_bal[{j},{m},{t},{s}]"
                        )

    # Capacity constraints + global production limit at each site
    for s in range(S):
        for t in range(T):
            for j in range(J):
                produced_jts = quicksum(demands[s][i][t] * x[i, j, t, s] for i in range(I))
                cap_jts = quicksum(capacities[m] * v[j, m, t, s] for m in range(M))
                model.addConstr(produced_jts <= cap_jts, name=f"cap[{j},{t},{s}]")
                model.addConstr(produced_jts <= u_max, name=f"u_max[{j},{t},{s}]")

    # Objective
    capex_term = quicksum(capex_mk[m][k] * y[m, k] for m in range(M) for k in range(K))

    expected_opex = 0
    for s in range(S):
        prod_term_s = quicksum(
            prod_cost(i, j, t) * demands[s][i][t] * x[i, j, t, s]
            for i in range(I) for j in range(J) for t in range(T)
        )
        reloc_term_s = quicksum(
            reloc_cost(j, jp, m, t) * w[j, jp, m, t, s]
            for j in range(J) for jp in range(J) if jp != j
            for m in range(M) for t in range(T)
        )
        penalty_term_s = quicksum(
            float(penalty_costs[i][t]) * demands[s][i][t] * q[i, t, s]
            for i in range(I) for t in range(T)
        )
        expected_opex += p_s[s] * (prod_term_s + reloc_term_s + penalty_term_s)

    model.setObjective(capex_term + expected_opex, GRB.MINIMIZE)

    # Incumbent logging callback (objective only during search)
    model._logger = logger
    model._best_inc = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < m._best_inc - 1e-9:
                m._best_inc = obj
                m._logger.log(float(obj))

    model.optimize(cb)

    return model, (y, z, x, q, v, v_depot, w, w_fd, w_td), (I, J, S, T, M, K)


def extract_solution(model, vars_pack, dims):
    y, z, x, q, v, v_depot, w, w_fd, w_td = vars_pack
    I, J, S, T, M, K = dims

    model_vars = {}

    def add_if_nonzero(name, val, is_int=False):
        if abs(val) > 1e-9:
            model_vars[name] = int(round(val)) if is_int else float(val)

    # y for dedicated output
    y_out = {}
    for m in range(M):
        for k in range(K):
            val = y[m, k].X if model.SolCount > 0 else 0.0
            y_out[f"y_{m}_{k}"] = int(round(val))
            add_if_nonzero(f"y[{m},{k}]", val, is_int=True)

    # z
    for m in range(M):
        for s in range(S):
            val = z[m, s].X if model.SolCount > 0 else 0.0
            add_if_nonzero(f"z[{m},{s}]", val, is_int=True)

    # x, q
    for s in range(S):
        for t in range(T):
            for i in range(I):
                valq = q[i, t, s].X if model.SolCount > 0 else 0.0
                add_if_nonzero(f"q[{i},{t},{s}]", valq, is_int=False)
                for j in range(J):
                    valx = x[i, j, t, s].X if model.SolCount > 0 else 0.0
                    add_if_nonzero(f"x[{i},{j},{t},{s}]", valx, is_int=False)

    # v, v_depot, w, w_fd, w_td
    for s in range(S):
        for t in range(T):
            for m in range(M):
                valvd = v_depot[m, t, s].X if model.SolCount > 0 else 0.0
                add_if_nonzero(f"v_depot[{m},{t},{s}]", valvd, is_int=True)

                for j in range(J):
                    valv = v[j, m, t, s].X if model.SolCount > 0 else 0.0
                    add_if_nonzero(f"v[{j},{m},{t},{s}]", valv, is_int=True)

                    valfd = w_fd[j, m, t, s].X if model.SolCount > 0 else 0.0
                    valtd = w_td[j, m, t, s].X if model.SolCount > 0 else 0.0
                    add_if_nonzero(f"w_fd[{j},{m},{t},{s}]", valfd, is_int=True)
                    add_if_nonzero(f"w_td[{j},{m},{t},{s}]", valtd, is_int=True)

                for j in range(J):
                    for jp in range(J):
                        if j == jp:
                            continue
                        valw = w[j, jp, m, t, s].X if model.SolCount > 0 else 0.0
                        add_if_nonzero(f"w[{j},{jp},{m},{t},{s}]", valw, is_int=True)

    patterns = {
        "y[m,k]": "m=0..num_module_sizes-1, k=0..units_per_module_size-1",
        "z[m,s]": "module size m, scenario s",
        "x[i,j,t,s]": "demand site i, production site j, period t, scenario s",
        "q[i,t,s]": "unmet-demand fraction",
        "v[j,m,t,s]": "units at production site",
        "v_depot[m,t,s]": "units at depot",
        "w[j,jp,m,t,s]": "relocation between distinct production sites j and jp",
        "w_fd[j,m,t,s]": "depot-to-production-site relocation",
        "w_td[j,m,t,s]": "production-site-to-depot relocation"
    }

    if model.SolCount > 0:
        obj = float(model.ObjVal)
    else:
        obj = 1e100

    solution = {
        "objective_value": obj,
        "model_variables": model_vars,
        "y": y_out,
        "model_variable_name_patterns": patterns
    }
    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve modular supply chain stochastic optimization.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    model, vars_pack, dims = build_and_solve(instance, args.time_limit, logger=logger)
    solution = extract_solution(model, vars_pack, dims)

    # Log final complete incumbent if available
    if logger is not None and model.SolCount > 0:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()