import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Unpack instance
    num_vessels = instance['num_vessels']
    num_berths = instance['num_berths']
    num_subblocks = instance['num_subblocks']
    H = instance['H']
    E = instance['E']
    penalty_co = instance['penalty_cost_coefficient_co']
    QC_capacity = instance['QC_capacity_per_timestep']
    D_L = instance['loading_route_lengths_D_L']
    D_U = instance['average_unloading_route_lengths_D_U']
    vessels = instance['vessels']

    vessel_data = []
    for v in vessels:
        vessel_data.append({
            'id': v['vessel_id'],
            'load': v['loading_containers'],
            'unload': v['unloading_containers'],
            'res_sub': v['reserved_subblocks'],
            'pen': v['penalty_cost_coefficient'],
            'feas_s': v['feasible_time_window'][0],
            'feas_e': v['feasible_time_window'][1],
            'exp_s': v['expected_time_window'][0],
            'exp_e': v['expected_time_window'][1],
            'profiles': v['qc_profiles']
        })

    T = H + E
    model = gp.Model("Berth_QC_Yard")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # Decision variables
    x = {}      # (v, b, p, t) binary
    y = {}      # (v, s) binary
    rho = {b: model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"rho_{b}") for b in range(num_berths)}
    varsigma = {b: model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"varsigma_{b}") for b in range(num_berths)}

    for v_idx, v in enumerate(vessel_data):
        for b in range(num_berths):
            for p_idx, prof in enumerate(v['profiles']):
                ht = prof['handling_time']
                for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                    x[(v_idx, b, p_idx, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{v_idx}_{b}_{p_idx}_{t}")

        for s in range(num_subblocks):
            y[(v_idx, s)] = model.addVar(vtype=GRB.BINARY, name=f"y_{v_idx}_{s}")

    # Objective: build with quicksum to avoid LinExpr.add() issues
    penalty_terms = []
    transport_terms = []

    for v_idx, v in enumerate(vessel_data):
        # Penalty for deviation from expected window
        for t in range(v['feas_s'], v['feas_e'] + 1):
            early = model.addVar(lb=0.0, name=f"early_{v_idx}_{t}")
            late = model.addVar(lb=0.0, name=f"late_{v_idx}_{t}")
            model.addConstr(early >= v['exp_s'] - t)
            model.addConstr(late >= t - v['exp_e'])

            sum_x_t = gp.quicksum(
                x.get((v_idx, b, p, t), 0)
                for b in range(num_berths)
                for p, prof in enumerate(v['profiles'])
                if (ht := prof['handling_time']) and t <= v['feas_e'] - ht + 1
                and (v_idx, b, p, t) in x
            )
            model.addConstr(early <= T * (1 - sum_x_t))
            model.addConstr(late <= T * (1 - sum_x_t))
            penalty_terms.append(v['pen'] * (early + late))

        # Loading transportation
        load_per_sub = v['load'] / v['res_sub']
        for s in range(num_subblocks):
            for b in range(num_berths):
                transport_terms.append(
                    y[(v_idx, s)] * (penalty_co * D_L[s][b] * load_per_sub)
                )

        # Unloading transportation
        for b in range(num_berths):
            unload_cost = penalty_co * D_U[b] * v['unload']
            sum_x_b = gp.quicksum(
                x.get((v_idx, b, p, t), 0)
                for p, prof in enumerate(v['profiles'])
                for t in range(v['feas_s'], v['feas_e'] - prof['handling_time'] + 2)
                if (v_idx, b, p, t) in x
            )
            transport_terms.append(unload_cost * sum_x_b)

    model.setObjective(gp.quicksum(penalty_terms) + gp.quicksum(transport_terms), GRB.MINIMIZE)

    # 1. Each vessel assigned exactly once
    for v_idx in range(num_vessels):
        model.addConstr(
            gp.quicksum(x[k] for k in x if k[0] == v_idx) == 1,
            name=f"assign_v{v_idx}"
        )

    # 2. Yard subblocks
    for v_idx, v in enumerate(vessel_data):
        model.addConstr(
            gp.quicksum(y[(v_idx, s)] for s in range(num_subblocks)) == v['res_sub'],
            name=f"sub_req_{v_idx}"
        )
    for s in range(num_subblocks):
        model.addConstr(
            gp.quicksum(y[(v, s)] for v in range(num_vessels)) <= 1,
            name=f"sub_excl_{s}"
        )

    # 3. No overlapping on same berth
    for b in range(num_berths):
        for v1 in range(num_vessels):
            v1d = vessel_data[v1]
            for p1, prof1 in enumerate(v1d['profiles']):
                ht1 = prof1['handling_time']
                for t1 in range(v1d['feas_s'], v1d['feas_e'] - ht1 + 2):
                    k1 = (v1, b, p1, t1)
                    if k1 not in x: continue
                    for v2 in range(v1 + 1, num_vessels):
                        v2d = vessel_data[v2]
                        for p2, prof2 in enumerate(v2d['profiles']):
                            ht2 = prof2['handling_time']
                            for t2 in range(v2d['feas_s'], v2d['feas_e'] - ht2 + 2):
                                k2 = (v2, b, p2, t2)
                                if k2 not in x: continue
                                M = T + 2
                                model.addConstr(t1 + ht1 <= t2 + M * (2 - x[k1] - x[k2]))
                                model.addConstr(t2 + ht2 <= t1 + M * (2 - x[k1] - x[k2]))

    # 4. Berth time window (cyclic)
    for b in range(num_berths):
        for v_idx, v in enumerate(vessel_data):
            for p_idx, prof in enumerate(v['profiles']):
                ht = prof['handling_time']
                for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                    k = (v_idx, b, p_idx, t)
                    if k not in x: continue
                    model.addConstr(rho[b] <= t + T * (1 - x[k]))
                    model.addConstr(varsigma[b] >= t + ht - 1 - T * (1 - x[k]))
        model.addConstr(varsigma[b] - rho[b] <= H - 1, name=f"berth_span_{b}")

    # 5. Quay crane capacity (periodic)
    for tau in range(T):
        qc_expr = gp.LinExpr()
        for v_idx, v in enumerate(vessel_data):
            for b in range(num_berths):
                for p_idx, prof in enumerate(v['profiles']):
                    ht = prof['handling_time']
                    for t_start in range(max(0, tau - ht + 1), tau + 1):
                        if t_start < v['feas_s'] or t_start > v['feas_e'] - ht + 1:
                            continue
                        k = (v_idx, b, p_idx, t_start)
                        if k in x:
                            step = tau - t_start
                            qc_expr.add(x[k] * prof['qc_per_timestep'][step])

        cap = QC_capacity[tau % H]
        if tau < H:
            twin = (tau + H) % T
            twin_expr = gp.LinExpr()
            # We reuse the same logic for twin (simpler to recompute for small instances)
            for v_idx, v in enumerate(vessel_data):
                for b_idx in range(num_berths):
                    for p_idx, prof in enumerate(v['profiles']):
                        ht = prof['handling_time']
                        for ts in range(max(0, twin - ht + 1), twin + 1):
                            if ts < v['feas_s'] or ts > v['feas_e'] - ht + 1: continue
                            k = (v_idx, b_idx, p_idx, ts)
                            if k in x:
                                step = twin - ts
                                twin_expr.add(x[k] * prof['qc_per_timestep'][step])
            model.addConstr(qc_expr + twin_expr <= cap, name=f"qc_cap_{tau}")
        else:
            model.addConstr(qc_expr <= cap, name=f"qc_cap_{tau}")

    # Callback for incumbent logging
    best_obj = float('inf')
    best_sol = None

    def callback(model, where):
        nonlocal best_obj, best_sol
        if where != GRB.Callback.MIPSOL:
            return
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj >= best_obj - 1e-6:
            return
        best_obj = obj

        x_sol = model.cbGetSolution(x)
        y_sol = model.cbGetSolution(y)
        rho_sol = model.cbGetSolution(rho)
        varsigma_sol = model.cbGetSolution(varsigma)

        sol = {
            "objective_value": float(obj),
            "vessels": [],
            "berths": [],
            "qc_usage": {}
        }

        for v_idx, v in enumerate(vessel_data):
            for b in range(num_berths):
                for p_idx, prof in enumerate(v['profiles']):
                    ht = prof['handling_time']
                    for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                        key = (v_idx, b, p_idx, t)
                        if key in x_sol and x_sol[key] > 0.5:
                            subblocks = [s for s in range(num_subblocks) if y_sol.get((v_idx, s), 0.0) > 0.5]
                            sol["vessels"].append({
                                "vessel_id": v['id'],
                                "alpha": int(t),
                                "beta": int(t + ht - 1),
                                "berth": int(b),
                                "profile": int(p_idx),
                                "subblocks": [int(s) for s in subblocks],
                                "handling_time": int(ht),
                                "qc_per_timestep": [int(q) for q in prof['qc_per_timestep']]
                            })
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

        for b in range(num_berths):
            sol["berths"].append({
                "berth": int(b),
                "rho": int(rho_sol.get(b, 0)),
                "varsigma": int(varsigma_sol.get(b, 0))
            })

        # Compute QC usage from solution
        qc_usage = {}
        for tau in range(T):
            usage = 0
            for v_idx, v in enumerate(vessel_data):
                for b in range(num_berths):
                    for p_idx, prof in enumerate(v['profiles']):
                        ht = prof['handling_time']
                        for t_start in range(max(0, tau - ht + 1), tau + 1):
                            if t_start < v['feas_s'] or t_start > v['feas_e'] - ht + 1:
                                continue
                            if x_sol.get((v_idx, b, p_idx, t_start), 0.0) > 0.5:
                                step = tau - t_start
                                usage += prof['qc_per_timestep'][step]
            if usage > 0.1:
                qc_usage[str(tau + 1)] = int(round(usage))
        sol["qc_usage"] = qc_usage

        best_sol = sol
        if logger:
            logger.log_solution(float(obj), sol)

    model.optimize(callback)

    # Final solution extraction if callback did not capture the best
    if model.SolCount > 0:
        final_obj = model.ObjVal
        if best_sol is None or abs(final_obj - best_obj) > 1e-4:
            x_sol = model.getAttr('X', x)
            y_sol = model.getAttr('X', y)
            rho_sol = model.getAttr('X', rho)
            varsigma_sol = model.getAttr('X', varsigma)

            sol = {
                "objective_value": float(final_obj),
                "vessels": [],
                "berths": [],
                "qc_usage": {}
            }

            for v_idx, v in enumerate(vessel_data):
                for b in range(num_berths):
                    for p_idx, prof in enumerate(v['profiles']):
                        ht = prof['handling_time']
                        for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                            key = (v_idx, b, p_idx, t)
                            if key in x_sol and x_sol[key] > 0.5:
                                subblocks = [s for s in range(num_subblocks)
                                           if (v_idx, s) in y_sol and y_sol[(v_idx, s)] > 0.5]
                                sol["vessels"].append({
                                    "vessel_id": v['id'],
                                    "alpha": int(t),
                                    "beta": int(t + ht - 1),
                                    "berth": int(b),
                                    "profile": int(p_idx),
                                    "subblocks": [int(s) for s in subblocks],
                                    "handling_time": int(ht),
                                    "qc_per_timestep": [int(q) for q in prof['qc_per_timestep']]
                                })
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

            for b in range(num_berths):
                sol["berths"].append({
                    "berth": int(b),
                    "rho": int(rho_sol[b]),
                    "varsigma": int(varsigma_sol[b])
                })

            qc_usage = {}
            for tau in range(T):
                usage = 0.0
                for v_idx, v in enumerate(vessel_data):
                    for b_idx in range(num_berths):
                        for p_idx, prof in enumerate(v['profiles']):
                            ht = prof['handling_time']
                            for ts in range(max(0, tau - ht + 1), tau + 1):
                                if ts < v['feas_s'] or ts > v['feas_e'] - ht + 1:
                                    continue
                                key = (v_idx, b_idx, p_idx, ts)
                                if key in x_sol and x_sol[key] > 0.5:
                                    step = tau - ts
                                    usage += prof['qc_per_timestep'][step]
                if usage > 0.1:
                    qc_usage[str(tau + 1)] = int(round(usage))
            sol["qc_usage"] = qc_usage
            best_sol = sol

    if best_sol is None:
        best_sol = {
            "objective_value": 0.0,
            "vessels": [],
            "berths": [{"berth": b, "rho": 0, "varsigma": 0} for b in range(num_berths)],
            "qc_usage": {}
        }

    if logger and best_sol.get("objective_value") is not None:
        logger.log_solution(best_sol["objective_value"], best_sol)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()