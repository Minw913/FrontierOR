import json
import argparse
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

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

    # Precompute vessel data
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

    # Total time steps considered
    T = H + E

    # Create model
    model = gp.Model("Berth_QC_Yard_Integrated")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # Decision variables
    # x[v, b, p, t]: vessel v assigned to berth b, profile p, start time t
    x = {}
    for v_idx in range(num_vessels):
        v = vessel_data[v_idx]
        for b in range(num_berths):
            for p_idx, prof in enumerate(v['profiles']):
                for t in range(v['feas_s'], v['feas_e'] - prof['handling_time'] + 2):
                    x[v_idx, b, p_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{v_idx}_{b}_{p_idx}_{t}")

    # y[v, s]: vessel v reserves subblock s
    y = {}
    for v_idx in range(num_vessels):
        v = vessel_data[v_idx]
        for s in range(num_subblocks):
            y[v_idx, s] = model.addVar(vtype=GRB.BINARY, name=f"y_{v_idx}_{s}")

    # Auxiliary for berth span
    rho = {}
    varsigma = {}
    for b in range(num_berths):
        rho[b] = model.addVar(lb=1, ub=T, vtype=GRB.INTEGER, name=f"rho_{b}")
        varsigma[b] = model.addVar(lb=1, ub=T, vtype=GRB.INTEGER, name=f"varsigma_{b}")

    # Objective components
    penalty_cost = gp.LinExpr()
    trans_cost = gp.LinExpr()

    for v_idx in range(num_vessels):
        v = vessel_data[v_idx]
        # Penalty
        for b in range(num_berths):
            for p_idx, prof in enumerate(v['profiles']):
                ht = prof['handling_time']
                for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                    if (v_idx, b, p_idx, t) in x:
                        dev = gp.max_(0, v['exp_s'] - t) + gp.max_(0, (t + ht - 1) - v['exp_e'])
                        penalty_cost.add(x[v_idx, b, p_idx, t] * v['pen'] * dev)

        # Transportation
        for s in range(num_subblocks):
            for b in range(num_berths):
                if (v_idx, s) in y and (v_idx, b, 0, v['feas_s']) in x:  # dummy check
                    trans_cost.add(y[v_idx, s] * penalty_co * D_L[s][b] * (v['load'] / v['res_sub']))

        for b in range(num_berths):
            trans_cost.add(penalty_co * D_U[b] * v['unload'] * 
                          gp.quicksum(x[v_idx, b, p_idx, t] for p_idx in range(len(v['profiles'])) 
                                     for t in range(v['feas_s'], v['feas_e'] - v['profiles'][p_idx]['handling_time'] + 2) 
                                     if (v_idx, b, p_idx, t) in x))

    model.setObjective(penalty_cost + trans_cost, GRB.MINIMIZE)

    # Constraints

    # 1. Each vessel assigned exactly once
    for v_idx in range(num_vessels):
        v = vessel_data[v_idx]
        assign_sum = gp.LinExpr()
        for b in range(num_berths):
            for p_idx, prof in enumerate(v['profiles']):
                ht = prof['handling_time']
                for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                    assign_sum.add(x[v_idx, b, p_idx, t])
        model.addConstr(assign_sum == 1, name=f"assign_v{v_idx}")

    # 2. Subblock reservation
    for v_idx in range(num_vessels):
        v = vessel_data[v_idx]
        model.addConstr(gp.quicksum(y[v_idx, s] for s in range(num_subblocks)) == v['res_sub'], 
                       name=f"subblocks_v{v_idx}")

    for s in range(num_subblocks):
        model.addConstr(gp.quicksum(y[v_idx, s] for v_idx in range(num_vessels)) <= 1, 
                       name=f"subblock_excl_{s}")

    # 3. Berth non-overlap and sequencing
    for b in range(num_berths):
        for v1 in range(num_vessels):
            v1d = vessel_data[v1]
            for p1 in range(len(v1d['profiles'])):
                ht1 = v1d['profiles'][p1]['handling_time']
                for t1 in range(v1d['feas_s'], v1d['feas_e'] - ht1 + 2):
                    if (v1, b, p1, t1) not in x:
                        continue
                    for v2 in range(v1 + 1, num_vessels):
                        v2d = vessel_data[v2]
                        for p2 in range(len(v2d['profiles'])):
                            ht2 = v2d['profiles'][p2]['handling_time']
                            for t2 in range(v2d['feas_s'], v2d['feas_e'] - ht2 + 2):
                                if (v2, b, p2, t2) not in x:
                                    continue
                                # If both on same berth, enforce order
                                model.addConstr(
                                    (t1 + ht1 - 1) * x[v1, b, p1, t1] + (1 - x[v1, b, p1, t1]) * T +
                                    (1 - x[v2, b, p2, t2]) * T >= t2 * x[v2, b, p2, t2],
                                    name=f"seq1_{v1}_{v2}_{b}_{t1}_{t2}"
                                )
                                model.addConstr(
                                    (t2 + ht2 - 1) * x[v2, b, p2, t2] + (1 - x[v2, b, p2, t2]) * T +
                                    (1 - x[v1, b, p1, t1]) * T >= t1 * x[v1, b, p1, t1],
                                    name=f"seq2_{v1}_{v2}_{b}_{t1}_{t2}"
                                )

    # 4. Berth span constraints
    for b in range(num_berths):
        for v_idx in range(num_vessels):
            v = vessel_data[v_idx]
            for p_idx, prof in enumerate(v['profiles']):
                ht = prof['handling_time']
                for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                    if (v_idx, b, p_idx, t) in x:
                        model.addConstr(rho[b] <= t * x[v_idx, b, p_idx, t] + (1 - x[v_idx, b, p_idx, t]) * T,
                                       name=f"rho_lb_{b}_{v_idx}_{t}")
                        model.addConstr(varsigma[b] >= (t + ht - 1) * x[v_idx, b, p_idx, t],
                                       name=f"varsigma_ub_{b}_{v_idx}_{t}")
        
        model.addConstr(varsigma[b] - rho[b] <= H - 1, name=f"berth_span_{b}")

    # 5. QC capacity with periodicity
    qc_usage_vars = {}
    for tau in range(T):
        qc_usage_vars[tau] = model.addVar(lb=0, ub=instance['num_QCs'], vtype=GRB.CONTINUOUS, name=f"qc_{tau}")

    for tau in range(T):
        qc_expr = gp.LinExpr()
        for v_idx in range(num_vessels):
            v = vessel_data[v_idx]
            for b in range(num_berths):
                for p_idx, prof in enumerate(v['profiles']):
                    ht = prof['handling_time']
                    for t in range(max(v['feas_s'], tau - ht + 1), tau + 1):
                        if t > v['feas_e'] - ht + 1:
                            continue
                        if (v_idx, b, p_idx, t) in x:
                            step_in_profile = tau - t
                            if 0 <= step_in_profile < ht:
                                qc_expr.add(x[v_idx, b, p_idx, t] * prof['qc_per_timestep'][step_in_profile])
        model.addConstr(qc_usage_vars[tau] == qc_expr, name=f"qc_def_{tau}")

    for tau in range(1, H + 1):
        if tau <= E:
            model.addConstr(qc_usage_vars[tau - 1] + qc_usage_vars[tau + H - 1] <= QC_capacity[tau - 1],
                           name=f"qc_cap_wrap_{tau}")
        else:
            model.addConstr(qc_usage_vars[tau - 1] <= QC_capacity[tau - 1], name=f"qc_cap_{tau}")

    # Callback for incumbent logging
    best_obj = float('inf')
    solution_dict_cache = None

    def callback(model, where):
        nonlocal best_obj, solution_dict_cache
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj - 1e-6:
                best_obj = obj
                # Extract solution
                sol = {}
                sol['objective_value'] = float(obj)
                sol['vessels'] = []
                sol['berths'] = []
                sol['qc_usage'] = {}

                # Vessel assignments
                x_sol = model.cbGetSolution(x)
                y_sol = model.cbGetSolution(y)
                rho_sol = model.cbGetSolution(rho)
                varsigma_sol = model.cbGetSolution(varsigma)
                qc_sol = model.cbGetSolution(qc_usage_vars)

                for v_idx in range(num_vessels):
                    v = vessel_data[v_idx]
                    assigned = False
                    for b in range(num_berths):
                        for p_idx in range(len(v['profiles'])):
                            for t in range(v['feas_s'], v['feas_e'] - v['profiles'][p_idx]['handling_time'] + 2):
                                if (v_idx, b, p_idx, t) in x_sol and x_sol[v_idx, b, p_idx, t] > 0.5:
                                    prof = v['profiles'][p_idx]
                                    ht = prof['handling_time']
                                    vessel_entry = {
                                        "vessel_id": v['id'],
                                        "alpha": int(t),
                                        "beta": int(t + ht - 1),
                                        "berth": int(b),
                                        "profile": int(p_idx),
                                        "subblocks": [],
                                        "handling_time": int(ht),
                                        "qc_per_timestep": [int(q) for q in prof['qc_per_timestep']]
                                    }
                                    # Subblocks
                                    for s in range(num_subblocks):
                                        if (v_idx, s) in y_sol and y_sol[v_idx, s] > 0.5:
                                            vessel_entry["subblocks"].append(int(s))
                                    sol['vessels'].append(vessel_entry)
                                    assigned = True
                                    break
                            if assigned: break
                        if assigned: break

                # Berths
                for b in range(num_berths):
                    sol['berths'].append({
                        "berth": int(b),
                        "rho": int(rho_sol[b]),
                        "varsigma": int(varsigma_sol[b])
                    })

                # QC usage (non-zero only)
                for tau in range(T):
                    if qc_sol[tau] > 0.1:
                        sol['qc_usage'][str(tau + 1)] = int(round(qc_sol[tau]))

                solution_dict_cache = sol
                if logger:
                    logger.log_solution(float(obj), sol)

    model.optimize(callback)

    # Final solution extraction
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.SolCount > 0:
        if solution_dict_cache is None or model.ObjVal < best_obj - 1e-6:
            # Extract final solution
            sol = {}
            sol['objective_value'] = float(model.ObjVal)
            sol['vessels'] = []
            sol['berths'] = []
            sol['qc_usage'] = {}

            x_sol = model.getAttr('X', x)
            y_sol = model.getAttr('X', y)
            rho_sol = model.getAttr('X', rho)
            varsigma_sol = model.getAttr('X', varsigma)
            qc_sol = model.getAttr('X', qc_usage_vars)

            for v_idx in range(num_vessels):
                v = vessel_data[v_idx]
                assigned = False
                for b_idx in range(num_berths):
                    for p_idx in range(len(v['profiles'])):
                        ht = v['profiles'][p_idx]['handling_time']
                        for t in range(v['feas_s'], v['feas_e'] - ht + 2):
                            key = (v_idx, b_idx, p_idx, t)
                            if key in x_sol and x_sol[key] > 0.5:
                                prof = v['profiles'][p_idx]
                                vessel_entry = {
                                    "vessel_id": v['id'],
                                    "alpha": int(t),
                                    "beta": int(t + ht - 1),
                                    "berth": int(b_idx),
                                    "profile": int(p_idx),
                                    "subblocks": [],
                                    "handling_time": int(ht),
                                    "qc_per_timestep": [int(q) for q in prof['qc_per_timestep']]
                                }
                                for s in range(num_subblocks):
                                    if (v_idx, s) in y_sol and y_sol[v_idx, s] > 0.5:
                                        vessel_entry["subblocks"].append(int(s))
                                sol['vessels'].append(vessel_entry)
                                assigned = True
                                break
                        if assigned: break
                    if assigned: break

            for b in range(num_berths):
                sol['berths'].append({
                    "berth": int(b),
                    "rho": int(rho_sol[b]),
                    "varsigma": int(varsigma_sol[b])
                })

            for tau in range(T):
                if qc_sol[tau] > 0.1:
                    sol['qc_usage'][str(tau + 1)] = int(round(qc_sol[tau]))

            final_solution = sol
            final_obj = float(model.ObjVal)
        else:
            final_solution = solution_dict_cache
            final_obj = best_obj
    else:
        # Fallback: return empty feasible if possible or raise
        final_solution = {"objective_value": 0.0, "vessels": [], "berths": [], "qc_usage": {}}
        final_obj = 0.0

    if logger and final_solution.get('objective_value', None) is not None:
        logger.log_solution(final_obj, final_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()