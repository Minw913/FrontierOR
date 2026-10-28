import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated berth/QC/yard assignment solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    # -----------------------------
    # Read instance data
    # -----------------------------
    num_vessels = inst["num_vessels"]
    num_berths = inst["num_berths"]
    num_subblocks = inst["num_subblocks"]
    H = inst["H"]
    E = inst["E"]
    T = H + E

    co = float(inst["penalty_cost_coefficient_co"])
    qc_cap = inst["QC_capacity_per_timestep"]  # length T
    D_L = inst["loading_route_lengths_D_L"]    # [subblock][berth]
    D_U = inst["average_unloading_route_lengths_D_U"]  # [berth]
    vessels = inst["vessels"]

    # -----------------------------
    # Build feasible modes per vessel
    # mode = (berth, profile_idx, alpha, beta, handling_time, qc_profile_list, penalty_cost, unload_cost)
    # -----------------------------
    global_modes = []
    vessel_mode_ids = [[] for _ in range(num_vessels)]
    modes_by_berth = [[] for _ in range(num_berths)]

    for v in range(num_vessels):
        vdat = vessels[v]
        f_start, f_end = vdat["feasible_time_window"]
        e_start, e_end = vdat["expected_time_window"]
        pcoef = float(vdat["penalty_cost_coefficient"])
        unload_cont = vdat["unloading_containers"]

        profiles = vdat["qc_profiles"]
        for b in range(num_berths):
            unload_cost = co * D_U[b] * unload_cont
            for k, prof in enumerate(profiles):
                ht = prof["handling_time"]
                qc_vec = prof["qc_per_timestep"]

                lb = max(1, f_start)
                ub = min(T, f_end - ht + 1)
                if lb > ub:
                    continue

                for alpha in range(lb, ub + 1):
                    beta = alpha + ht - 1
                    early_pen = max(0, e_start - alpha)
                    late_pen = max(0, beta - e_end)
                    penalty = pcoef * (early_pen + late_pen)

                    m_id = len(global_modes)
                    global_modes.append({
                        "v": v,
                        "berth": b,
                        "profile": k,
                        "alpha": alpha,
                        "beta": beta,
                        "handling_time": ht,
                        "qc_vec": qc_vec,
                        "penalty": penalty,
                        "unload_cost": unload_cost
                    })
                    vessel_mode_ids[v].append(m_id)
                    modes_by_berth[b].append(m_id)

    # Basic feasibility check: each vessel must have at least one mode
    all_have_mode = all(len(vessel_mode_ids[v]) > 0 for v in range(num_vessels))

    # -----------------------------
    # Build MIP model
    # -----------------------------
    model = gp.Model("terminal_planning")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, int(args.time_limit))
    model.Params.OutputFlag = 0

    M = T

    # Variables
    y = []  # mode selection binaries
    for m in range(len(global_modes)):
        y.append(model.addVar(vtype=GRB.BINARY, name=f"y_{m}"))

    g = [[model.addVar(vtype=GRB.BINARY, name=f"g_{v}_{s}") for s in range(num_subblocks)] for v in range(num_vessels)]
    z = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"z_{v}_{b}") for b in range(num_berths)] for v in range(num_vessels)]
    w = [[[model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"w_{v}_{s}_{b}")
           for b in range(num_berths)] for s in range(num_subblocks)] for v in range(num_vessels)]

    rho = [model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=T, name=f"rho_{b}") for b in range(num_berths)]
    varsigma = [model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=T, name=f"varsigma_{b}") for b in range(num_berths)]

    model.update()

    # Exactly one mode per vessel
    for v in range(num_vessels):
        if len(vessel_mode_ids[v]) == 0:
            continue
        model.addConstr(gp.quicksum(y[m] for m in vessel_mode_ids[v]) == 1, name=f"one_mode_{v}")

    # Link z[v,b] to selected mode berth
    for v in range(num_vessels):
        for b in range(num_berths):
            mids = [m for m in vessel_mode_ids[v] if global_modes[m]["berth"] == b]
            if mids:
                model.addConstr(z[v][b] == gp.quicksum(y[m] for m in mids), name=f"z_link_{v}_{b}")
            else:
                model.addConstr(z[v][b] == 0.0, name=f"z_zero_{v}_{b}")

    # Subblock assignment
    for v in range(num_vessels):
        req = vessels[v]["reserved_subblocks"]
        model.addConstr(gp.quicksum(g[v][s] for s in range(num_subblocks)) == req, name=f"subblock_count_{v}")

    for s in range(num_subblocks):
        model.addConstr(gp.quicksum(g[v][s] for v in range(num_vessels)) <= 1, name=f"subblock_unique_{s}")

    # Linearization w = g * z
    for v in range(num_vessels):
        for s in range(num_subblocks):
            for b in range(num_berths):
                model.addConstr(w[v][s][b] <= g[v][s], name=f"w_le_g_{v}_{s}_{b}")
                model.addConstr(w[v][s][b] <= z[v][b], name=f"w_le_z_{v}_{s}_{b}")
                model.addConstr(w[v][s][b] >= g[v][s] + z[v][b] - 1, name=f"w_ge_{v}_{s}_{b}")

    # Berth occupancy: at most one vessel per berth-time
    occupancy = [[[] for _ in range(T + 1)] for _ in range(num_berths)]  # 1..T
    for m, md in enumerate(global_modes):
        b = md["berth"]
        for t in range(md["alpha"], md["beta"] + 1):
            occupancy[b][t].append(m)

    for b in range(num_berths):
        for t in range(1, T + 1):
            mids = occupancy[b][t]
            if len(mids) > 1:
                model.addConstr(gp.quicksum(y[m] for m in mids) <= 1, name=f"berth_occ_{b}_{t}")

    # QC usage constraints with periodicity
    qc_terms = [[] for _ in range(T + 1)]  # each item: (m, coeff)
    for m, md in enumerate(global_modes):
        alpha = md["alpha"]
        for r, q in enumerate(md["qc_vec"]):
            if q != 0:
                t = alpha + r
                if 1 <= t <= T:
                    qc_terms[t].append((m, q))

    # t = E+1..H
    for t in range(E + 1, H + 1):
        expr = gp.LinExpr()
        for (m, q) in qc_terms[t]:
            expr.addTerms(q, y[m])
        model.addConstr(expr <= qc_cap[t - 1], name=f"qc_mid_{t}")

    # t = 1..E : usage(t) + usage(t+H) <= cap(t)
    for t in range(1, E + 1):
        expr = gp.LinExpr()
        for (m, q) in qc_terms[t]:
            expr.addTerms(q, y[m])
        twin = t + H
        if 1 <= twin <= T:
            for (m, q) in qc_terms[twin]:
                expr.addTerms(q, y[m])
        model.addConstr(expr <= qc_cap[t - 1], name=f"qc_wrap_{t}")

    # Berth cyclic span constraints via rho/varsigma
    for b in range(num_berths):
        for m in modes_by_berth[b]:
            alpha = global_modes[m]["alpha"]
            beta = global_modes[m]["beta"]
            model.addConstr(rho[b] <= alpha + M * (1 - y[m]), name=f"rho_def_{b}_{m}")
            model.addConstr(varsigma[b] >= beta - M * (1 - y[m]), name=f"vars_def_{b}_{m}")
        model.addConstr(varsigma[b] - rho[b] <= H - 1, name=f"berth_span_{b}")

    # Objective
    obj = gp.LinExpr()

    # Penalty + unloading (mode-dependent)
    for m, md in enumerate(global_modes):
        coeff = md["penalty"] + md["unload_cost"]
        if coeff != 0:
            obj.addTerms(coeff, y[m])

    # Loading transport via w
    for v in range(num_vessels):
        load = vessels[v]["loading_containers"]
        req = vessels[v]["reserved_subblocks"]
        if req > 0 and load > 0:
            factor = co * (load / req)
            for s in range(num_subblocks):
                for b in range(num_berths):
                    c = factor * D_L[s][b]
                    if c != 0:
                        obj.addTerms(c, w[v][s][b])

    model.setObjective(obj, GRB.MINIMIZE)

    # -----------------------------
    # Solution extraction helpers
    # -----------------------------
    vessel_ids = [vessels[v]["vessel_id"] for v in range(num_vessels)]

    def decode_solution(y_vals, g_vals, obj_value=None):
        # chosen mode per vessel
        chosen_mode = [None] * num_vessels
        for v in range(num_vessels):
            mids = vessel_mode_ids[v]
            if not mids:
                continue
            best_m = mids[0]
            best_val = -1.0
            for m in mids:
                val = y_vals[m]
                if val > best_val:
                    best_val = val
                    best_m = m
            chosen_mode[v] = best_m

        # chosen subblocks per vessel
        chosen_subblocks = [[] for _ in range(num_vessels)]
        for v in range(num_vessels):
            req = vessels[v]["reserved_subblocks"]
            vals = [(g_vals[v][s], s) for s in range(num_subblocks)]
            vals.sort(reverse=True)
            picked = [s for val, s in vals[:req]]
            chosen_subblocks[v] = picked

        # vessel outputs
        vessel_out = []
        berth_to_intervals = [[] for _ in range(num_berths)]
        qc_usage = defaultdict(int)

        for v in range(num_vessels):
            m = chosen_mode[v]
            if m is None:
                # fallback dummy
                berth = 0
                alpha = 1
                prof = 0
                ht = 1
                qc_vec = [0]
                beta = alpha + ht - 1
            else:
                md = global_modes[m]
                berth = md["berth"]
                alpha = md["alpha"]
                beta = md["beta"]
                prof = md["profile"]
                ht = md["handling_time"]
                qc_vec = md["qc_vec"]

                berth_to_intervals[berth].append((alpha, beta))
                for r, q in enumerate(qc_vec):
                    if q != 0:
                        t = alpha + r
                        if 1 <= t <= T:
                            qc_usage[t] += int(q)

            vessel_out.append({
                "vessel_id": vessel_ids[v],
                "alpha": int(alpha),
                "beta": int(beta),
                "berth": int(berth),
                "profile": int(prof),
                "subblocks": [int(s) for s in chosen_subblocks[v]],
                "handling_time": int(ht),
                "qc_per_timestep": [int(x) for x in qc_vec]
            })

        # berth outputs
        berths_out = []
        for b in range(num_berths):
            if berth_to_intervals[b]:
                rho_b = min(a for a, _ in berth_to_intervals[b])
                vars_b = max(e for _, e in berth_to_intervals[b])
            else:
                rho_b = 0
                vars_b = 0
            berths_out.append({
                "berth": int(b),
                "rho": int(rho_b),
                "varsigma": int(vars_b)
            })

        qc_usage_out = {str(int(t)): int(q) for t, q in sorted(qc_usage.items()) if q != 0}

        if obj_value is None:
            # rough recompute (for fallback / safety)
            obj_tmp = 0.0
            for v in range(num_vessels):
                m = chosen_mode[v]
                if m is None:
                    continue
                md = global_modes[m]
                obj_tmp += md["penalty"] + md["unload_cost"]
                load = vessels[v]["loading_containers"]
                req = vessels[v]["reserved_subblocks"]
                if req > 0 and load > 0:
                    factor = co * (load / req)
                    b = md["berth"]
                    for s in chosen_subblocks[v]:
                        obj_tmp += factor * D_L[s][b]
            obj_value = obj_tmp

        return {
            "objective_value": float(obj_value),
            "vessels": vessel_out,
            "berths": berths_out,
            "qc_usage": qc_usage_out
        }

    # Callback for incumbent logging
    if logger:
        model._logger = logger
        model._best_logged = float("inf")
        model._y = y
        model._g = g
        model._decode_solution = decode_solution

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val + 1e-9 < m._best_logged:
                    y_vals = m.cbGetSolution(m._y)
                    g_vals = []
                    for vv in range(num_vessels):
                        g_vals.append(m.cbGetSolution(m._g[vv]))
                    sol = m._decode_solution(y_vals, g_vals, obj_val)
                    m._logger.log_solution(float(obj_val), sol)
                    m._best_logged = float(obj_val)

        callback = cb
    else:
        callback = None

    # -----------------------------
    # Optimize
    # -----------------------------
    if all_have_mode:
        model.optimize(callback)
    else:
        model.optimize(callback)  # likely infeasible, but run anyway

    # -----------------------------
    # Final solution
    # -----------------------------
    if model.SolCount > 0:
        y_vals = [var.X for var in y]
        g_vals = [[g[v][s].X for s in range(num_subblocks)] for v in range(num_vessels)]
        final_obj = float(model.ObjVal)
        solution = decode_solution(y_vals, g_vals, final_obj)

        if logger:
            # Ensure final best is logged
            logger.log_solution(solution["objective_value"], solution)
    else:
        # Fallback schema-compliant solution
        # Greedy placeholder
        y_vals = [0.0] * len(global_modes)
        for v in range(num_vessels):
            if vessel_mode_ids[v]:
                y_vals[vessel_mode_ids[v][0]] = 1.0
        g_vals = [[0.0] * num_subblocks for _ in range(num_vessels)]
        used = set()
        for v in range(num_vessels):
            req = vessels[v]["reserved_subblocks"]
            cnt = 0
            for s in range(num_subblocks):
                if s not in used:
                    g_vals[v][s] = 1.0
                    used.add(s)
                    cnt += 1
                    if cnt == req:
                        break
        solution = decode_solution(y_vals, g_vals, 1e30)
        if logger:
            logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()