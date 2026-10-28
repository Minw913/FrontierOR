import json
import argparse
import time
import math
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

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_patients = instance['num_patients']
    num_hospitals = instance['num_hospitals']
    num_days = instance['num_days']
    num_ors_per_hospital = instance['num_ors_per_hospital']
    num_scenarios = instance['num_scenarios']
    global_params = instance['global_parameters']
    patients = instance['patients']
    hospitals = instance['hospitals']
    scenarios = instance['scenarios']

    hospital_ids = sorted([h['hospital_id'] for h in hospitals])
    day_ids = list(range(num_days))

    G = {}
    F = {}
    B = {}
    for h in hospitals:
        h_id = h['hospital_id']
        for day_info in h['days']:
            d_id = day_info['day_id']
            G[(h_id, d_id)] = day_info['G_hd']
            F[(h_id, d_id)] = day_info['F_hd']
            B[(h_id, d_id)] = day_info['B_hd']

    day_id_to_label = {}
    for day_info in hospitals[0]['days']:
        day_id = day_info['day_id']
        day_label = day_info['day_label']
        day_id_to_label[day_id] = day_label

    c_sched_dict = {}
    for p in patients:
        p_id = p['patient_id']
        c_sched_dict[p_id] = {}
        for d_id in day_ids:
            day_label = day_id_to_label[d_id]
            s_key = str(day_label)
            c_sched_dict[p_id][d_id] = p['c_sched_per_day'].get(s_key, 0)

    patient_id_to_index = {p['patient_id']: i for i, p in enumerate(patients)}
    scenario_durations = []
    for scenario in scenarios:
        durs = scenario['surgery_durations_minutes']
        dur_map = {}
        for p in patients:
            p_id = p['patient_id']
            idx = patient_id_to_index[p_id]
            dur_map[p_id] = durs[idx]
        scenario_durations.append(dur_map)

    mandatory_patients = [p for p in patients if p['is_mandatory']]
    non_mandatory_patients = [p for p in patients if not p['is_mandatory']]
    mandatory_patient_ids = [p['patient_id'] for p in mandatory_patients]
    non_mandatory_patient_ids = [p['patient_id'] for p in non_mandatory_patients]

    total_z_vars = num_hospitals * num_days * num_ors_per_hospital * num_patients * num_scenarios
    if total_z_vars <= 500000:
        scenario_indices = list(range(num_scenarios))
    else:
        scenario_indices = list(range(min(10, num_scenarios)))

    model = gp.Model()
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)

    u = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            u_key = (h_id, d_id)
            u[u_key] = model.addVar(vtype=GRB.BINARY, name=f"u_{h_id}_{d_id}")

    y = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                y_key = (h_id, d_id, k)
                y[y_key] = model.addVar(vtype=GRB.BINARY, name=f"y_{h_id}_{d_id}_{k}")

    x = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                for p in patients:
                    p_id = p['patient_id']
                    x_key = (h_id, d_id, k, p_id)
                    x[x_key] = model.addVar(vtype=GRB.BINARY, name=f"x_{h_id}_{d_id}_{k}_{p_id}")

    w = {}
    for p in non_mandatory_patients:
        p_id = p['patient_id']
        w[p_id] = model.addVar(vtype=GRB.BINARY, name=f"w_{p_id}")

    z = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                for p in patients:
                    p_id = p['patient_id']
                    for s_index in scenario_indices:
                        z_key = (h_id, d_id, k, p_id, s_index)
                        z[z_key] = model.addVar(vtype=GRB.BINARY, name=f"z_{h_id}_{d_id}_{k}_{p_id}_{s_index}")

    model.update()

    obj = 0

    for h_id in hospital_ids:
        for d_id in day_ids:
            u_key = (h_id, d_id)
            obj += G[u_key] * u[u_key]

    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                y_key = (h_id, d_id, k)
                obj += F[(h_id, d_id)] * y[y_key]

    for p in non_mandatory_patients:
        p_id = p['patient_id']
        obj += p['c_unsched'] * w[p_id]

    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                for p in patients:
                    p_id = p['patient_id']
                    x_key = (h_id, d_id, k, p_id)
                    obj += c_sched_dict[p_id][d_id] * x[x_key]

    cancellation_obj = 0
    for s_index in scenario_indices:
        scenario_factor = 1.0 / len(scenario_indices)
        for h_id in hospital_ids:
            for d_id in day_ids:
                for k in range(num_ors_per_hospital):
                    for p in patients:
                        p_id = p['patient_id']
                        z_key = (h_id, d_id, k, p_id, s_index)
                        cancellation_obj += scenario_factor * p['c_cancel'] * z[z_key]
    obj += cancellation_obj

    model.setObjective(obj, GRB.MINIMIZE)

    for p in mandatory_patients:
        p_id = p['patient_id']
        expr = gp.LinExpr()
        for h_id in hospital_ids:
            for d_id in day_ids:
                for k in range(num_ors_per_hospital):
                    x_key = (h_id, d_id, k, p_id)
                    expr += x[x_key]
        model.addConstr(expr == 1, f"mandatory_{p_id}")

    for p in non_mandatory_patients:
        p_id = p['patient_id']
        expr = gp.LinExpr()
        for h_id in hospital_ids:
            for d_id in day_ids:
                for k in range(num_ors_per_hospital):
                    x_key = (h_id, d_id, k, p_id)
                    expr += x[x_key]
        expr += w[p_id]
        model.addConstr(expr == 1, f"nonmandatory_{p_id}")

    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                y_key = (h_id, d_id, k)
                model.addConstr(y[y_key] <= u[(h_id, d_id)], f"suite_open_{h_id}_{d_id}_{k}")

    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                y_key = (h_id, d_id, k)
                for p in patients:
                    p_id = p['patient_id']
                    x_key = (h_id, d_id, k, p_id)
                    model.addConstr(x[x_key] <= y[y_key], f"assign_open_{h_id}_{d_id}_{k}_{p_id}")

    for s_index in scenario_indices:
        dur_map = scenario_durations[s_index]
        for h_id in hospital_ids:
            for d_id in day_ids:
                for k in range(num_ors_per_hospital):
                    expr = gp.LinExpr()
                    for p in patients:
                        p_id = p['patient_id']
                        x_key = (h_id, d_id, k, p_id)
                        z_key = (h_id, d_id, k, p_id, s_index)
                        dur = dur_map[p_id]
                        expr += dur * x[x_key]
                        expr -= dur * z[z_key]
                    model.addConstr(expr <= B[(h_id, d_id)], f"time_{h_id}_{d_id}_{k}_{s_index}")

    for s_index in scenario_indices:
        for h_id in hospital_ids:
            for d_id in day_ids:
                for k in range(num_ors_per_hospital):
                    for p in patients:
                        p_id = p['patient_id']
                        x_key = (h_id, d_id, k, p_id)
                        z_key = (h_id, d_id, k, p_id, s_index)
                        model.addConstr(z[z_key] <= x[x_key], f"cancel_assign_{h_id}_{d_id}_{k}_{p_id}_{s_index}")

    time_remaining = args.time_limit - (time.time() - start_time)
    if time_remaining < 1:
        time_remaining = 1
    model.setParam('TimeLimit', time_remaining)

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    try:
        model.optimize(callback if logger else None)
    except gp.GurobiError as e:
        pass

    u_sol = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            key = f"{h_id},{d_id}"
            u_key = (h_id, d_id)
            if u_key in u:
                u_sol[key] = int(round(u[u_key].X))

    y_sol = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                key = f"{h_id},{d_id},{k}"
                y_key = (h_id, d_id, k)
                if y_key in y:
                    y_sol[key] = int(round(y[y_key].X))

    x_sol = {}
    for h_id in hospital_ids:
        for d_id in day_ids:
            for k in range(num_ors_per_hospital):
                for p in patients:
                    p_id = p['patient_id']
                    x_key = (h_id, d_id, k, p_id)
                    if x_key in x and round(x[x_key].X) == 1:
                        key = f"{h_id},{d_id},{p_id},{k}"
                        x_sol[key] = 1

    w_sol = {}
    for p in non_mandatory_patients:
        p_id = p['patient_id']
        if p_id in w:
            w_sol[str(p_id)] = int(round(w[p_id].X))

    suite_cost = 0
    for key, val in u_sol.items():
        if val == 1:
            parts = key.split(',')
            h_id = int(parts[0])
            d_id = int(parts[1])
            suite_cost += G.get((h_id, d_id), 0)

    or_cost = 0
    for key, val in y_sol.items():
        if val == 1:
            parts = key.split(',')
            h_id = int(parts[0])
            d_id = int(parts[1])
            or_cost += F.get((h_id, d_id), 0)

    postponement_cost = 0
    for p in non_mandatory_patients:
        p_id = p['patient_id']
        if w_sol.get(str(p_id), 0) == 1:
            postponement_cost += p['c_unsched']

    scheduling_benefit = 0
    for key in x_sol:
        parts = key.split(',')
        h_id = int(parts[0])
        d_id = int(parts[1])
        p_id = int(parts[2])
        k = int(parts[3])
        scheduling_benefit += c_sched_dict.get(p_id, {}).get(d_id, 0)

    total_cancellation_cost = 0.0
    for s in range(num_scenarios):
        dur_map = scenario_durations[s]
        for h_id in hospital_ids:
            for d_id in day_ids:
                for k in range(num_ors_per_hospital):
                    key_y = f"{h_id},{d_id},{k}"
                    if y_sol.get(key_y, 0) == 1:
                        patients_in_OR = []
                        total_duration = 0.0
                        for p in patients:
                            p_id = p['patient_id']
                            key_x = f"{h_id},{d_id},{p_id},{k}"
                            if x_sol.get(key_x, 0) == 1:
                                dur = dur_map.get(p_id, 0)
                                c_cancel = p['c_cancel']
                                patients_in_OR.append((p_id, dur, c_cancel))
                                total_duration += dur
                        B_hd = B.get((h_id, d_id), 0)
                        R = total_duration - B_hd
                        if R <= 1e-6:
                            cancellation_this_OR = 0.0
                        else:
                            n = len(patients_in_OR)
                            if n <= 20:
                                best_cost = float('inf')
                                for mask in range(1, 1<<n):
                                    total_dur_cancel = 0.0
                                    total_cost_cancel = 0.0
                                    for i in range(n):
                                        if mask & (1 << i):
                                            dur_i, cost_i = patients_in_OR[i][1], patients_in_OR[i][2]
                                            total_dur_cancel += dur_i
                                            total_cost_cancel += cost_i
                                    if total_dur_cancel >= R and total_cost_cancel < best_cost:
                                        best_cost = total_cost_cancel
                                cancellation_this_OR = best_cost
                            else:
                                R_int = math.ceil(R)
                                dp = [10**18] * (R_int+1)
                                dp[0] = 0
                                for (p_id, dur, c_cancel) in patients_in_OR:
                                    dur_int = math.ceil(dur)
                                    for j in range(R_int, -1, -1):
                                        if dp[j] < 10**18:
                                            j2 = j + dur_int
                                            if j2 > R_int:
                                                j2 = R_int
                                            if dp[j2] > dp[j] + c_cancel:
                                                dp[j2] = dp[j] + c_cancel
                                cancellation_this_OR = dp[R_int]
                        total_cancellation_cost += cancellation_this_OR

    expected_cancellation_cost = total_cancellation_cost / num_scenarios
    total_obj = suite_cost + or_cost + postponement_cost + scheduling_benefit + expected_cancellation_cost

    solution = {
        "objective_value": total_obj,
        "u": u_sol,
        "y": y_sol,
        "x": x_sol,
        "w": w_sol
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()