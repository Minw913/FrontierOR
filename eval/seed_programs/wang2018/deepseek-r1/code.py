import json
import argparse
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description='Container Terminal Planning')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    data = read_instance(args.instance_path)
    
    num_vessels = data['num_vessels']
    num_berths = data['num_berths']
    num_subblocks = data['num_subblocks']
    H = data['H']
    E = data['E']
    penalty_co_co = data['penalty_cost_coefficient_co']
    QC_capacity = data['QC_capacity_per_timestep']
    loading_distances = data['loading_route_lengths_D_L']
    unloading_distances_avg = data['average_unloading_route_lengths_D_U'] 
    vessels_data = data['vessels']

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    T_total = H + E  # total time steps: 1 to H+E

    # Precompute for vessels:
    # T_ip[i][p] = list of feasible start times for vessel i and profile p? 
    # But we precomputed set of start times for each vessel and profile? 
    T_ip = {}          # for vessel i, profile index p -> the list of start times (from a_i to b_i - handling_time + 1)
    T_ip_set = {}       # set for faster inclusion check

    for i, vessel in enumerate(vessels_data):
        feasible_start = vessel['feasible_time_window'][0]
        feasible_end = vessel['feasible_time_window'][1]
        profiles = vessel['qc_profiles']
        T_ip[i] = []
        T_ip_set[i] = []
        for p_idx, profile in enumerate(profiles):
            handling_time = profile['handling_time']
            lb = feasible_start
            ub = feasible_end - handling_time + 1
            t0_list = []
            t0_set = set()
            if lb <= ub:
                t0_list = list(range(lb, ub+1))
                t0_set = set(range(lb, ub+1))
            T_ip[i].append(t0_list)
            T_ip_set[i].append(t0_set)

    # Create model
    model = gp.Model("ContainerTerminal")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Variables
    y = {}
    for i in range(num_vessels):
        for b in range(num_berths):
            y[i, b] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{b}")

    x = {}
    for i in range(num_vessels):
        n_i = vessels_data[i]['reserved_subblocks']
        for s in range(num_subblocks):
            x[i, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")

    # u: for vessel i, for profile p, for start time t0 in T_ip[i][p]
    u = {}
    for i in range(num_vessels):
        profiles = vessels_data[i]['qc_profiles']
        for p_idx in range(len(profiles)):
            t0_list = T_ip[i][p_idx]
            for t0 in t0_list:
                u[i, p_idx, t0] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{p_idx}_{t0}")

    # Variables for penalty
    dev_start = {}
    dev_end = {}
    for i in range(num_vessels):
        dev_start[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"dev_start_{i}")
        dev_end[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"dev_end_{i}")

    # Variables for sub-block and berth costs (the product x[i,s]*y[i,b] is linearized by u_x)
    u_x = {}  # u_x[i, s, b] = x[i,s] * y[i,b]
    for i in range(num_vessels):
        for s in range(num_subblocks):
            for b in range(num_berths):
                u_x[i, s, b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"ux_{i}_{s}_{b}")

    # Variables for the start time (alpha_i) and handling_time (h_i) for each vessel i
    alpha_i = {}
    h_i = {}
    end_i = {}
    for i in range(num_vessels):
        alpha_i[i] = model.addVar(vtype=GRB.INTEGER, name=f"alpha_{i}")
        h_i[i] = model.addVar(vtype=GRB.INTEGER, name=f"h_i_{i}")
        end_i[i] = model.addVar(vtype=GRB.INTEGER, name=f"end_{i}")

    # Variables for non-overlapping in berths
    w_berth = {}  # w_berth[i,j,b] for i<j
    for i in range(num_vessels):
        for j in range(i+1, num_vessels):
            for b in range(num_berths):
                w_berth[i,j,b] = model.addVar(vtype=GRB.BINARY, name=f"w_berth_{i}_{j}_{b}")
    w = {} # w[i,j]: sum_b w_berth[i,j,b] -> which is 0 or 1
    v = {} # v[i,j] for i<j: binary to indicate ordering
    for i in range(num_vessels):
        for j in range(i+1, num_vessels):
            w[i,j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_berths, name=f"w_{i}_{j}")
            v[i,j] = model.addVar(vtype=GRB.BINARY, name=f"v_{i}_{j}")

    # Each vessel to exactly one berth and exactly one profile and start time
    for i in range(num_vessels):
        model.addConstr(gp.quicksum(y[i, b] for b in range(num_berths)) == 1, name=f"one_berth_{i}")
        model.addConstr(gp.quicksum(u[i, p_idx, t0] 
                                    for p_idx in range(len(vessels_data[i]['qc_profiles']))
                                    for t0 in T_ip[i][p_idx]) == 1, name=f"one_profile_start_{i}")
        model.addConstr(gp.quicksum(x[i,s] for s in range(num_subblocks)) == vessels_data[i]['reserved_subblocks'],
                        name=f"reserve_subblocks_{i}")

    # Ensure subblocks not overlapping: each subblock at most one vessel? 
    for s in range(num_subblocks):
        model.addConstr(gp.quicksum(x[i,s] for i in range(num_vessels)) <= 1, name=f"subblock_unique_{s}")

    # Link u to alpha_i and h_i
    for i in range(num_vessels):
        num_profiles = len(vessels_data[i]['qc_profiles'])
        # For alpha_i: 
        alpha_expr = 0
        h_expr = 0
        for p_idx in range(num_profiles):
            handling_time = vessels_data[i]['qc_profiles'][p_idx]['handling_time']
            for t0 in T_ip[i][p_idx]:
                alpha_expr += u[i, p_idx, t0] * t0
                h_expr += u[i, p_idx, t0] * handling_time
        model.addConstr(alpha_i[i] == alpha_expr, name=f"def_alpha_{i}")
        model.addConstr(h_i[i] == h_expr, name=f"def_h_i_{i}")
        model.addConstr(end_i[i] == alpha_i[i] + h_i[i] - 1, name=f"def_end_{i}")

    # Penalty constraints for each vessel
    for i, vessel in enumerate(vessels_data):
        exp_start = vessel['expected_time_window'][0]
        exp_end = vessel['expected_time_window'][1]
        model.addConstr(dev_start[i] >= exp_start - alpha_i[i], name=f"dev_start_constraint_{i}")
        model.addConstr(dev_start[i] >= 0, name=f"dev_start_nonneg_{i}")
        model.addConstr(dev_end[i] >= end_i[i] - exp_end, name=f"dev_end_constraint_{i}")
        model.addConstr(dev_end[i] >= 0, name=f"dev_end_nonneg_{i}")

    # Linearize u_x = x * y
    for i in range(num_vessels):
        for s in range(num_subblocks):
            for b in range(num_berths):
                model.addConstr(u_x[i, s, b] <= x[i, s], name=f"ux1_{i}_{s}_{b}")
                model.addConstr(u_x[i, s, b] <= y[i, b], name=f"ux2_{i}_{s}_{b}")
                model.addConstr(u_x[i, s, b] >= x[i, s] + y[i, b] - 1, name=f"ux3_{i}_{s}_{b}")

    # Non-overlapping for vessels at the same berth: setup w_berth
    for i in range(num_vessels):
        for j in range(i+1, num_vessels):
            sum_w_berth = 0
            for b in range(num_berths):
                model.addConstr(w_berth[i,j,b] <= y[i,b], name=f"w_berth_1_{i}_{j}_{b}")
                model.addConstr(w_berth[i,j,b] <= y[j,b], name=f"w_berth_2_{i}_{j}_{b}")
                model.addConstr(w_berth[i,j,b] >= y[i,b] + y[j,b] - 1, name=f"w_berth_3_{i}_{j}_{b}") 
                sum_w_berth += w_berth[i,j,b]
            model.addConstr(w[i,j] == sum_w_berth, name=f"w_{i}_{j}")

    # Non-overlapping: for each pair, if w[i,j] >=1 (at least one common berth) then enforce either i before j or j before i
    M = T_total + 100   # big-M, T_total is H+E
    for i in range(num_vessels):
        for j in range(i+1, num_vessels):
            # Condition: if w[i,j] >= 1, then enforce either end_i <= alpha_j-1 OR end_j <= alpha_i-1
            # But note: our variables: we have binary v to choose which one is before.

            # Scenario 1: i before j: 
            #    end_i <= alpha_j-1
            # Scenario 2: j before i: 
            #    end_j <= alpha_i-1

            model.addConstr(end_i[i] <= alpha_i[j] - 1 + M * (1 - v[i,j]) + M * (1 - w[i,j]), 
                         name=f"non_overlap_1_{i}_{j}")
            model.addConstr(end_i[j] <= alpha_i[i] - 1 + M * (v[i,j]) + M * (1 - w[i,j]), 
                         name=f"non_overlap_2_{i}_{j}")

    # QC usage per time step t for t=1 to T_total = H+E
    # We create a LinExpr for each time step: 
    qc_usage_exprs = [0.0] * T_total   # index 0 to T_total-1 represent time step 1 to T_total
    for t_val in range(1, T_total+1):
        expr = 0.0
        for i in range(num_vessels):
            vessel = vessels_data[i]
            profiles = vessel['qc_profiles']
            for p_idx, profile in enumerate(profiles):
                handling_time = profile['handling_time']
                lb_t0 = max(vessel['feasible_time_window'][0], t_val - handling_time + 1)
                ub_t0 = min(vessel['feasible_time_window'][1] - handling_time + 1, t_val)
                if lb_t0 <= ub_t0:
                    for t0 in T_ip[i][p_idx]:
                        if t0 < lb_t0 or t0 > ub_t0:
                            continue
                        # The relative time step in the handling period: k = t_val - t0 (which is >=0 and <= handling_time-1)
                        if (t0 <= t_val and t_val <= t0 + handling_time - 1):
                            k = t_val - t0  # time into the handling: 0-indexed relative
                            qc_val = profile['qc_per_timestep'][k]
                            expr += u[i, p_idx, t0] * qc_val
        qc_usage_exprs[t_val-1] = expr

    # QC capacity constraints
    for t_index in range(T_total):
        time_step = t_index+1
        # For time_step in [1, E]: check the sum current and twin (in next week: t_index+H) 
        if time_step <= E:
            time_step_twin = time_step + H   # must be in [H+1, H+E] (since H>=time_step and time_step<=E, so time_step_twin<= H+E)
            expr_here = qc_usage_exprs[t_index]
            expr_twin = qc_usage_exprs[t_index + H]
            total_here = expr_here + expr_twin
            model.addConstr(total_here <= QC_capacity[t_index], name=f"qc_periodic_{time_step}")
        elif time_step <= H:  # from E+1 to H
            model.addConstr(qc_usage_exprs[t_index] <= QC_capacity[t_index], name=f"qc_regular_{time_step}")
        # For time steps > H (if any) do nothing? because they are only included via the periodic constraint? 
        # Note: our indexes for time_step_twin in [H+1, H+E] (which is covered by the periodic constraint for the first E time steps) 
        # and we did not get constraint for time_step in [E+1, H] and beyond? But beyond H? 
        # Actually, time_step > H: we don't have a constraint? that includes time_step in the range [H+1, H+E]? but they are not constrained individually? 
        # and the problem does not say so? 
        # Also, in the periodic constraint, time_step in [1, E] are constrained with their twin in [H+1, H+E]. 
        # So we skip t_index>=H? because we have time steps only up to H?

    # Part 9: Objective function
    penalty_cost = 0.0
    for i, vessel in enumerate(vessels_data):
        coeff = vessel['penalty_cost_coefficient']
        penalty_cost += coeff * (dev_start[i] + dev_end[i])

    # Unloading cost: per vessel, co_global * unloading_containers_i * (average_unloading_route_length for the assigned berth)
    unloading_cost = 0.0
    for i, vessel in enumerate(vessels_data):
        unloading_containers = vessel['unloading_containers']
        expr_berth_dist = 0.0
        for b in range(num_berths):
            # distance for this berth: unloading_distances_avg has length = number of berths? 
            expr_berth_dist += y[i,b] * unloading_distances_avg[b]
        unloading_cost += penalty_co_co * unloading_containers * expr_berth_dist

    # Loading cost: co_global * (loading_containers_i / reserved_subblocks_i) * sum_{s, b} u_x[i,s,b] * D_L[s][b]
    loading_cost = 0.0
    for i, vessel in enumerate(vessels_data):
        loading_containers = vessel['loading_containers']
        num_reserved = vessel['reserved_subblocks']
        sum_uxd = 0.0
        for s in range(num_subblocks):
            for b in range(num_berths):
                dist = loading_distances[s][b]
                sum_uxd += u_x[i,s,b] * dist
        if num_reserved > 0:
            loading_cost_unit = penalty_co_co * (loading_containers / num_reserved)
        else:
            loading_cost_unit = 0
        loading_cost += loading_cost_unit * sum_uxd

    total_cost = penalty_cost + unloading_cost + loading_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback for logging
    if logger:
        def mycallback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)
                except AttributeError as e:
                    pass
            elif where == GRB.Callback.MESSAGE:
                pass

        model._mycallback = mycallback
        model.optimize(mycallback)
    else:
        model.optimize()

    # Recover solution
    status = model.status
    if status == GRB.INFEASIBLE:
        # Infeasible? no solution? but try to return anyhow? 
        print("Infeasible model!")
        solution = {
            "objective_value": 0.0,
            "vessels": [],
            "berths": [],
            "qc_usage": {}
        }
    elif status == GRB.OPTIMAL or status == GRB.TIME_LIMIT:
        # Output best solution
        solution = {"objective_value": model.ObjVal}
        sol_vessels = []

        # For each vessel, get assignments
        for i, vessel in enumerate(vessels_data):
            vessel_id = vessel['vessel_id']
            # Find the berth b for which y[i,b] is 1
            chosen_berth = None
            for b in range(num_berths):
                if y[i, b].X > 0.5:
                    chosen_berth = b
                    break

            chosen_profile_idx = None
            chosen_t0 = None
            for p_idx in range(len(vessel['qc_profiles'])):
                for t0 in T_ip[i][p_idx]:
                    if (i, p_idx, t0) in u and u[i,p_idx,t0].X > 0.5:
                        chosen_profile_idx = p_idx
                        chosen_t0 = t0
            profile = vessel['qc_profiles'][chosen_profile_idx]
            handling_time = profile['handling_time']
            qc_per_timestep_list = profile['qc_per_timestep']
            start_time = chosen_t0
            end_time = chosen_t0 + handling_time - 1

            # subblocks: which s are chosen? 
            subblock_indices = []
            for s in range(num_subblocks):
                if (i,s) in x and x[i,s].X > 0.5:
                    subblock_indices.append(s)

            vessel_sol = {
                "vessel_id": vessel_id,
                "alpha": start_time,
                "beta": end_time, 
                "berth": chosen_berth,
                "profile": chosen_profile_idx,
                "subblocks": subblock_indices,
                "handling_time": handling_time,
                "qc_per_timestep": qc_per_timestep_list
            }
            sol_vessels.append(vessel_sol)
        solution["vessels"] = sol_vessels

        # Compute for berths: for each berth, the earliest start (rho) and the latest end (varsigma) for the vessels assigned to it.
        # We group by berth.
        berth_assignments = defaultdict(list)
        for vessel_sol in sol_vessels:
            berth = vessel_sol["berth"]
            alpha = vessel_sol["alpha"]
            beta = vessel_sol["beta"]
            berth_assignments[berth].append( (alpha, beta) )

        sol_berths = []
        for b in range(num_berths):
            if b not in berth_assignments:
                # but we have at least one vessel? skip? 
                rho = 0
                varsigma = 0
            else:
                start_times = [alpha for (alpha, beta) in berth_assignments[b]]
                end_times = [beta for (alpha, beta) in berth_assignments[b]]
                rho = min(start_times) if start_times else 0
                varsigma = max(end_times) if end_times else 0
            sol_berths.append({"berth": b, "rho": rho, "varsigma": varsigma})
        solution["berths"] = sol_berths

        # QC_usage: for each time step from 1 to T_total, compute the total QC usage
        qc_usage_dict = {}
        for t in range(1, T_total+1):
            usage = 0.0
            for vessel_sol in sol_vessels:
                start_time = vessel_sol["alpha"]
                end_time = vessel_sol["beta"]
                handling_time = vessel_sol["handling_time"]
                qc_per_timestep = vessel_sol["qc_per_timestep"]
                if start_time <= t and t <= end_time:
                    # relative step = t - start_time (from 0 to handling_time-1)
                    offset = t - start_time
                    usage += qc_per_timestep[offset]
            if usage > 1e-5:
                qc_usage_dict[str(t)] = round(usage) if round(usage) != 0 else 0
        solution["qc_usage"] = qc_usage_dict

    else:
        print(f"Unexpected status: {status}")
        solution = {
            "objective_value": 0.0,
            "vessels": [],
            "berths": [],
            "qc_usage": {}
        }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        if solution: 
            try:
                logger.log_solution(solution['objective_value'], solution)
            except:
                pass

if __name__ == '__main__':
    main()