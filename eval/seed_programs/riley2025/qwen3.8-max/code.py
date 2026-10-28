import argparse
import json
import math
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=600.0)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    I = int(instance.get("num_demand_sites", len(instance.get("demand_sites", []))))
    J = int(instance.get("num_production_sites", len(instance.get("production_sites", []))))
    S = int(instance.get("num_scenarios", len(instance.get("scenarios", []))))
    T = int(instance.get("num_time_periods", 0))
    M = int(instance.get("num_module_sizes", len(instance.get("module_sizes", []))))
    U = int(instance.get("units_per_module_size", 0))

    scenarios = instance.get("scenarios", [])
    if scenarios and S != len(scenarios):
        S = len(scenarios)

    if T == 0 and scenarios:
        d0 = scenarios[0].get("demands", [])
        if d0 and isinstance(d0[0], list):
            T = len(d0[0])

    try:
        u_max = float(instance.get("u_max", float("inf")))
    except Exception:
        u_max = float("inf")

    if math.isfinite(u_max):
        u_max_finite = True
        if u_max < 0.0:
            u_max = 0.0
    else:
        u_max_finite = False
        u_max = float("inf")

    if instance.get("scenario_probabilities"):
        probs = [float(p) for p in instance["scenario_probabilities"]]
    else:
        probs = [float(sc.get("probability", 0.0)) for sc in scenarios]

    if len(probs) < S:
        probs.extend([0.0] * (S - len(probs)))
    elif len(probs) > S:
        probs = probs[:S]

    module_sizes = instance.get("module_sizes", [])
    caps = []
    default_costs = []

    for m in range(M):
        if m < len(module_sizes):
            try:
                cap = float(module_sizes[m].get("capacity", 0.0))
            except Exception:
                cap = 0.0
            if cap < 0.0:
                cap = 0.0

            try:
                cost = float(module_sizes[m].get("capital_cost_per_unit", 0.0))
            except Exception:
                cost = 0.0

            caps.append(cap)
            default_costs.append(cost)
        else:
            caps.append(0.0)
            default_costs.append(0.0)

    unit_costs_by_k = [dict() for _ in range(M)]
    for entry in instance.get("module_units", []):
        try:
            m = int(entry.get("module_size_id", -1))
            k = int(entry.get("unit_id", -1))
            cost = float(entry.get("capital_cost", default_costs[m] if 0 <= m < M else 0.0))
            if 0 <= m < M and 0 <= k < U:
                unit_costs_by_k[m][k] = cost
        except Exception:
            pass

    sorted_units = []
    prefix = []
    is_uniform = []
    uniform_cost = []

    for m in range(M):
        lst = []
        for k in range(U):
            lst.append((k, unit_costs_by_k[m].get(k, default_costs[m])))
        lst.sort(key=lambda x: (x[1], x[0]))
        sorted_units.append(lst)

        pref = [0.0]
        for _, cost in lst:
            pref.append(pref[-1] + cost)
        prefix.append(pref)

        if U > 0:
            costs = [c for _, c in lst]
            if max(costs) - min(costs) <= 1e-8:
                is_uniform.append(True)
                uniform_cost.append(costs[0])
            else:
                is_uniform.append(False)
                uniform_cost.append(None)
        else:
            is_uniform.append(True)
            uniform_cost.append(0.0)

    penalty_raw = instance.get("penalty_costs", [])
    penalty = []
    for i in range(I):
        row = []
        if i < len(penalty_raw):
            rawrow = penalty_raw[i]
            for t in range(T):
                try:
                    val = float(rawrow[t]) if t < len(rawrow) else 0.0
                except Exception:
                    val = 0.0
                row.append(val)
        else:
            row = [0.0] * T
        penalty.append(row)

    demands = []
    for s in range(S):
        sc = scenarios[s] if s < len(scenarios) else {}
        d_raw = sc.get("demands", [])
        dd = []
        for i in range(I):
            row = []
            if i < len(d_raw):
                rawrow = d_raw[i]
                for t in range(T):
                    try:
                        val = float(rawrow[t]) if t < len(rawrow) else 0.0
                    except Exception:
                        val = 0.0
                    if val < 0.0:
                        val = 0.0
                    row.append(val)
            else:
                row = [0.0] * T
            dd.append(row)
        demands.append(dd)

    pc_raw = instance.get("production_costs", [])
    pc_2d = None
    pc_3d = None
    pc_orient = "IJT"

    if pc_raw:
        first = pc_raw[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
            pc_3d = pc_raw
            if (
                len(pc_raw) >= I
                and len(pc_raw[0]) == J
                and (T == 0 or (len(pc_raw[0]) > 0 and len(pc_raw[0][0]) == T))
            ):
                pc_orient = "IJT"
            elif (
                len(pc_raw) >= I
                and len(pc_raw[0]) == T
                and (J == 0 or (len(pc_raw[0]) > 0 and len(pc_raw[0][0]) == J))
            ):
                pc_orient = "ITJ"
            else:
                pc_orient = "IJT"
        else:
            pc_2d = pc_raw

    base_cost = 0.0
    total_demand_st = [[0.0] * T for _ in range(S)]

    for s in range(S):
        prob = probs[s] if s < len(probs) else 0.0
        for t in range(T):
            tot = 0.0
            for i in range(I):
                d = demands[s][i][t]
                tot += d
                base_cost += prob * penalty[i][t] * d
            total_demand_st[s][t] = tot

    initial_Z = [0] * M
    for m in range(M):
        cnt = 0
        for _, cost in sorted_units[m]:
            if cost < -1e-8:
                cnt += 1
            else:
                break
        initial_Z[m] = cnt

    x_info = []
    x_var_list = []
    x_by_its = defaultdict(list)
    x_by_jts = defaultdict(list)
    x_by_ts = defaultdict(list)

    v_info = []
    v_var_list = []
    v_by_jts = defaultdict(list)
    v_by_mts = defaultdict(list)

    VAR_EPS = 1e-6

    def construct_solution(z_vals, x_vals, v_vals):
        Zint = []
        if z_vals is None:
            z_vals = []

        for m in range(M):
            try:
                zv = float(z_vals[m]) if m < len(z_vals) else 0.0
            except Exception:
                zv = 0.0
            zi = int(round(zv))
            if zi < 0:
                zi = 0
            if zi > U:
                zi = U
            Zint.append(zi)

        v_int = []
        if v_vals is None:
            v_vals = []

        for idx in range(len(v_info)):
            try:
                val = float(v_vals[idx]) if idx < len(v_vals) else 0.0
            except Exception:
                val = 0.0
            rv = int(round(val))
            if rv < 0:
                rv = 0
            if rv > U:
                rv = U
            v_int.append(rv)

        max_needed = [0] * M
        for key, idxs in v_by_mts.items():
            m = key[0]
            tot = 0
            for vi in idxs:
                tot += v_int[vi]
            if tot > max_needed[m]:
                max_needed[m] = tot

        for m in range(M):
            if max_needed[m] > Zint[m]:
                Zint[m] = min(U, max_needed[m])

        for key, idxs in v_by_mts.items():
            m = key[0]
            limit = Zint[m]
            tot = 0
            for vi in idxs:
                tot += v_int[vi]

            if tot > limit:
                excess = tot - limit
                sorted_idxs = sorted(idxs, key=lambda vi: v_int[vi], reverse=True)
                for vi in sorted_idxs:
                    if excess <= 0:
                        break
                    dec = min(v_int[vi], excess)
                    if dec > 0:
                        v_int[vi] -= dec
                        excess -= dec

        x_arr = [0.0] * len(x_info)
        if x_vals is None:
            x_vals = []

        for idx in range(len(x_info)):
            try:
                val = float(x_vals[idx]) if idx < len(x_vals) else 0.0
            except Exception:
                val = 0.0
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            x_arr[idx] = val

        for idxs in x_by_its.values():
            tot = 0.0
            for idx in idxs:
                tot += x_arr[idx]
            if tot > 1.0 + 1e-7:
                factor = 1.0 / tot
                for idx in idxs:
                    x_arr[idx] *= factor

        for key, idxs in x_by_jts.items():
            prod = 0.0
            for idx in idxs:
                prod += x_info[idx][4] * x_arr[idx]
            if prod <= 1e-9:
                continue

            cap = 0.0
            for vi in v_by_jts.get(key, []):
                m = v_info[vi][1]
                cap += caps[m] * v_int[vi]

            limit = cap
            if u_max_finite and u_max < limit:
                limit = u_max
            if limit < 0.0:
                limit = 0.0

            if prod > limit + 1e-6:
                if limit <= 1e-9:
                    for idx in idxs:
                        x_arr[idx] = 0.0
                else:
                    factor = limit / prod
                    if factor > 1.0:
                        factor = 1.0
                    for idx in idxs:
                        x_arr[idx] *= factor

        for idx in range(len(x_arr)):
            if x_arr[idx] <= VAR_EPS:
                x_arr[idx] = 0.0

        sum_x_its = {}
        for key, idxs in x_by_its.items():
            tot = 0.0
            for idx in idxs:
                tot += x_arr[idx]
            if tot > 1.0:
                factor = 1.0 / tot
                for idx in idxs:
                    x_arr[idx] *= factor
                tot = 1.0
            elif tot < 0.0:
                tot = 0.0
            sum_x_its[key] = tot

        op = base_cost
        for idx, info in enumerate(x_info):
            xv = x_arr[idx]
            if xv > 0.0:
                op += info[5] * xv

        capital = 0.0
        for m in range(M):
            capital += prefix[m][Zint[m]]

        obj = capital + op

        mv = {}
        y_dict = {}

        for m in range(M):
            n = Zint[m]
            selected = set()
            if n > 0:
                for k, _ in sorted_units[m][:n]:
                    selected.add(k)

            for k in range(U):
                val = 1 if k in selected else 0
                y_dict[f"y_{m}_{k}"] = val
                if val:
                    mv[f"y[{m},{k}]"] = 1

            if n > 0:
                for s in range(S):
                    mv[f"z[{m},{s}]"] = n

        for idx, info in enumerate(x_info):
            xv = x_arr[idx]
            if xv > VAR_EPS:
                mv[f"x[{info[0]},{info[1]},{info[2]},{info[3]}]"] = round(xv, 9)

        for s in range(S):
            for t in range(T):
                for i in range(I):
                    tot = sum_x_its.get((i, t, s), 0.0)
                    qv = 1.0 - tot
                    if qv > VAR_EPS:
                        mv[f"q[{i},{t},{s}]"] = round(qv, 9)

        for idx, val in enumerate(v_int):
            if val > 0:
                j, m, t, s = v_info[idx]
                mv[f"v[{j},{m},{t},{s}]"] = int(val)

        if any(n > 0 for n in Zint):
            for s in range(S):
                for m in range(M):
                    nZ = Zint[m]
                    if nZ == 0:
                        continue

                    prev = [0] * J
                    for t in range(T):
                        curr = [0] * J
                        for vi in v_by_mts.get((m, t, s), []):
                            j = v_info[vi][0]
                            curr[j] = v_int[vi]

                        sum_curr = sum(curr)
                        depot = nZ - sum_curr
                        if depot < 0:
                            depot = 0

                        if depot > 0:
                            mv[f"v_depot[{m},{t},{s}]"] = int(depot)

                        for j in range(J):
                            delta = curr[j] - prev[j]
                            if delta > 0:
                                mv[f"w_fd[{j},{m},{t},{s}]"] = int(delta)
                            elif delta < 0:
                                mv[f"w_td[{j},{m},{t},{s}]"] = int(-delta)

                        prev = curr

        sol = {
            "objective_value": float(obj),
            "model_variables": mv,
            "y": y_dict,
        }
        return sol, float(obj)

    initial_sol, initial_obj = construct_solution(initial_Z, [], [])
    if logger:
        try:
            logger.log_solution(initial_obj, initial_sol)
        except Exception:
            pass

    state = {
        "best_obj": initial_obj,
        "best_sol": initial_sol,
    }

    total_cap_max = sum(caps[m] * U for m in range(M))
    allow_x = (
        M > 0
        and U > 0
        and total_cap_max > 1e-9
        and (not u_max_finite or u_max > 1e-9)
    )

    elapsed = time.time() - start_time
    remaining = max(0.0, float(args.time_limit) - elapsed - 1.0)

    if not allow_x or remaining <= 0.0:
        final_sol = state["best_sol"]
        if os.path.dirname(args.solution_path):
            os.makedirs(os.path.dirname(args.solution_path), exist_ok=True)
        with open(args.solution_path, "w") as f:
            json.dump(final_sol, f)
        return

    model = gp.Model("modular_supply_chain")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", 0)
    model.setParam("Threads", 1)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)

    Z_vars = []
    capex_vars = {}

    for m in range(M):
        if U <= 0:
            var = model.addVar(lb=0.0, ub=0.0, vtype=GRB.INTEGER, obj=0.0, name=f"Z_{m}")
            Z_vars.append(var)
            continue

        if is_uniform[m]:
            var = model.addVar(
                lb=0.0,
                ub=float(U),
                vtype=GRB.INTEGER,
                obj=float(uniform_cost[m]),
                name=f"Z_{m}",
            )
            Z_vars.append(var)
        else:
            var = model.addVar(lb=0.0, ub=float(U), vtype=GRB.INTEGER, obj=0.0, name=f"Z_{m}")
            lb = min(prefix[m])
            ub = max(prefix[m])
            capvar = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, obj=1.0, name=f"capex_{m}")
            model.addGenConstrPWL(var, capvar, list(range(U + 1)), prefix[m], f"capex_pwl_{m}")
            Z_vars.append(var)
            capex_vars[m] = capvar

    cap_pos = [m for m in range(M) if caps[m] > 1e-9]

    if pc_2d is not None:
        for s in range(S):
            prob = probs[s] if s < len(probs) else 0.0
            if abs(prob) <= 1e-12:
                continue
            dem_s = demands[s]
            for t in range(T):
                for i in range(I):
                    d = dem_s[i][t]
                    if d <= 1e-9:
                        continue
                    pen = penalty[i][t]
                    row = pc_2d[i] if i < len(pc_2d) else []
                    for j in range(J):
                        try:
                            c = float(row[j]) if j < len(row) else 0.0
                        except Exception:
                            c = 0.0
                        net = c - pen
                        if net < -1e-8:
                            coeff = prob * d * net
                            if coeff < -1e-12:
                                var = model.addVar(
                                    lb=0.0,
                                    ub=1.0,
                                    vtype=GRB.CONTINUOUS,
                                    obj=coeff,
                                    name=f"x_{i}_{j}_{t}_{s}",
                                )
                                idx = len(x_info)
                                x_info.append((i, j, t, s, d, coeff))
                                x_var_list.append(var)
                                x_by_its[(i, t, s)].append(idx)
                                x_by_jts[(j, t, s)].append(idx)
                                x_by_ts[(t, s)].append(idx)
    elif pc_3d is not None:
        for s in range(S):
            prob = probs[s] if s < len(probs) else 0.0
            if abs(prob) <= 1e-12:
                continue
            dem_s = demands[s]
            for t in range(T):
                for i in range(I):
                    d = dem_s[i][t]
                    if d <= 1e-9:
                        continue
                    pen = penalty[i][t]
                    for j in range(J):
                        try:
                            if pc_orient == "ITJ":
                                c = float(pc_3d[i][t][j])
                            else:
                                c = float(pc_3d[i][j][t])
                        except Exception:
                            c = 0.0
                        net = c - pen
                        if net < -1e-8:
                            coeff = prob * d * net
                            if coeff < -1e-12:
                                var = model.addVar(
                                    lb=0.0,
                                    ub=1.0,
                                    vtype=GRB.CONTINUOUS,
                                    obj=coeff,
                                    name=f"x_{i}_{j}_{t}_{s}",
                                )
                                idx = len(x_info)
                                x_info.append((i, j, t, s, d, coeff))
                                x_var_list.append(var)
                                x_by_its[(i, t, s)].append(idx)
                                x_by_jts[(j, t, s)].append(idx)
                                x_by_ts[(t, s)].append(idx)
    else:
        for s in range(S):
            prob = probs[s] if s < len(probs) else 0.0
            if abs(prob) <= 1e-12:
                continue
            dem_s = demands[s]
            for t in range(T):
                for i in range(I):
                    d = dem_s[i][t]
                    if d <= 1e-9:
                        continue
                    pen = penalty[i][t]
                    for j in range(J):
                        c = 0.0
                        net = c - pen
                        if net < -1e-8:
                            coeff = prob * d * net
                            if coeff < -1e-12:
                                var = model.addVar(
                                    lb=0.0,
                                    ub=1.0,
                                    vtype=GRB.CONTINUOUS,
                                    obj=coeff,
                                    name=f"x_{i}_{j}_{t}_{s}",
                                )
                                idx = len(x_info)
                                x_info.append((i, j, t, s, d, coeff))
                                x_var_list.append(var)
                                x_by_its[(i, t, s)].append(idx)
                                x_by_jts[(j, t, s)].append(idx)
                                x_by_ts[(t, s)].append(idx)

    if not x_info:
        final_sol = state["best_sol"]
        if os.path.dirname(args.solution_path):
            os.makedirs(os.path.dirname(args.solution_path), exist_ok=True)
        with open(args.solution_path, "w") as f:
            json.dump(final_sol, f)
        return

    for key in list(x_by_jts.keys()):
        j, t, s = key
        for m in cap_pos:
            var = model.addVar(
                lb=0.0,
                ub=float(U),
                vtype=GRB.INTEGER,
                name=f"v_{j}_{m}_{t}_{s}",
            )
            vi = len(v_info)
            v_info.append((j, m, t, s))
            v_var_list.append(var)
            v_by_jts[key].append(vi)
            v_by_mts[(m, t, s)].append(vi)

    model.update()

    for idxs in x_by_its.values():
        if idxs:
            model.addConstr(gp.quicksum(x_var_list[idx] for idx in idxs) <= 1.0)

    for key, idxs in x_by_jts.items():
        prod_expr = gp.quicksum(x_info[idx][4] * x_var_list[idx] for idx in idxs)

        v_idxs = v_by_jts.get(key, [])
        if v_idxs:
            cap_expr = gp.quicksum(caps[v_info[vi][1]] * v_var_list[vi] for vi in v_idxs)
            model.addConstr(prod_expr <= cap_expr)
        else:
            model.addConstr(prod_expr <= 0.0)

        if u_max_finite:
            j, t, s = key
            if s < S and t < T and u_max < total_demand_st[s][t] - 1e-8:
                model.addConstr(prod_expr <= u_max)

    for key, idxs in v_by_mts.items():
        m, t, s = key
        if idxs:
            model.addConstr(gp.quicksum(v_var_list[vi] for vi in idxs) <= Z_vars[m])

    if cap_pos:
        rhs_total_cap = gp.quicksum(caps[m] * Z_vars[m] for m in cap_pos)
        for key, idxs in x_by_ts.items():
            if idxs:
                model.addConstr(
                    gp.quicksum(x_info[idx][4] * x_var_list[idx] for idx in idxs) <= rhs_total_cap
                )

    model.update()

    try:
        model.setObjective(model.getObjective() + base_cost, GRB.MINIMIZE)
        constant_set = True
    except Exception:
        model.ModelSense = GRB.MINIMIZE
        constant_set = False

    try:
        for m, var in enumerate(Z_vars):
            var.Start = float(initial_Z[m])
        for m, var in capex_vars.items():
            var.Start = float(prefix[m][initial_Z[m]])
    except Exception:
        pass

    def gurobi_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                sol_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if not constant_set:
                    sol_obj += base_cost

                if sol_obj >= state["best_obj"] - 1e-6:
                    return

                z_vals = cb_model.cbGetSolution(Z_vars) if Z_vars else []
                x_vals = cb_model.cbGetSolution(x_var_list) if x_var_list else []
                v_vals = cb_model.cbGetSolution(v_var_list) if v_var_list else []

                sol, obj = construct_solution(z_vals, x_vals, v_vals)

                if obj < state["best_obj"] - 1e-6:
                    state["best_obj"] = obj
                    state["best_sol"] = sol
                    if logger:
                        try:
                            logger.log_solution(obj, sol)
                        except Exception:
                            pass
            except Exception:
                pass

    elapsed = time.time() - start_time
    remaining = max(0.0, float(args.time_limit) - elapsed - 1.0)

    if remaining > 0.0:
        try:
            model.setParam("TimeLimit", remaining)
            model.optimize(gurobi_callback)
        except Exception:
            pass

    try:
        if model.SolCount > 0:
            z_vals = [var.X for var in Z_vars] if Z_vars else []
            x_vals = [var.X for var in x_var_list] if x_var_list else []
            v_vals = [var.X for var in v_var_list] if v_var_list else []

            sol, obj = construct_solution(z_vals, x_vals, v_vals)

            if obj < state["best_obj"] - 1e-6:
                state["best_obj"] = obj
                state["best_sol"] = sol
                if logger:
                    try:
                        logger.log_solution(obj, sol)
                    except Exception:
                        pass
    except Exception:
        pass

    final_sol = state["best_sol"]

    if os.path.dirname(args.solution_path):
        os.makedirs(os.path.dirname(args.solution_path), exist_ok=True)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()