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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_patients = data['num_patients']
    num_hospitals = data['num_hospitals']
    num_days = data['num_days']
    num_ors_per_h = data['num_ors_per_hospital']
    num_scenarios = data['num_scenarios']
    patients = data['patients']
    hospitals = data['hospitals']
    scenarios = data['scenarios']
    gprm = data['global_parameters']
    kappa1 = gprm['kappa_1']
    kappa2 = gprm['kappa_2']
    kappa3 = gprm['kappa_3']
    kappa4 = gprm['kappa_4']

    # Precompute patient data
    is_mandatory = [p['is_mandatory'] for p in patients]
    c_unsched = [p['c_unsched'] for p in patients]
    c_cancel = [p['c_cancel'] for p in patients]
    omega = [p['omega_p'] for p in patients]
    alpha = [p['alpha_p'] for p in patients]
    rho = [p['rho_p'] for p in patients]

    # Build day labels and scheduling benefits: c_sched[p][d] = patients[p]['c_sched_per_day'][day_label]
    day_labels = []
    B = [[0]*num_days for _ in range(num_hospitals)]
    F = [[0]*num_days for _ in range(num_hospitals)]
    G = [[0]*num_days for _ in range(num_hospitals)]
    for h_idx, hosp in enumerate(hospitals):
        for day_entry in hosp['days']:
            d = day_entry['day_id']
            label = day_entry['day_label']
            B[h_idx][d] = day_entry['B_hd']
            F[h_idx][d] = day_entry['F_hd']
            G[h_idx][d] = day_entry['G_hd']
            if len(day_labels) <= d:
                day_labels.append(label)
    day_label_to_idx = {lbl: idx for idx, lbl in enumerate(day_labels)}

    c_sched = [[0.0 for _ in range(num_days)] for _ in range(num_patients)]
    for p in range(num_patients):
        for lbl_str, val in patients[p]['c_sched_per_day'].items():
            lbl = int(lbl_str)
            if lbl in day_label_to_idx:
                d = day_label_to_idx[lbl]
                c_sched[p][d] = val

    # Scenario durations: dur[s][p]
    dur = []
    for sc in scenarios:
        dur.append(sc['surgery_durations_minutes'])

    # Build model
    model = gp.Model("HospitalScheduling")
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # reserve some time for output

    # Variables
    u = {}  # (h,d)
    y = {}  # (h,d,k)
    x = {}  # (h,d,p,k)
    w = {}  # p (only for non-mandatory)
    z = {}  # (s,h,d,k,p)  -- whether performed in scenario s

    for h in range(num_hospitals):
        for d in range(num_days):
            u[h,d] = model.addVar(vtype=GRB.BINARY, name=f"u_{h}_{d}")
            for k in range(num_ors_per_h):
                y[h,d,k] = model.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{k}")
                for p in range(num_patients):
                    x[h,d,p,k] = model.addVar(vtype=GRB.BINARY, name=f"x_{h}_{d}_{p}_{k}")
                    for s in range(num_scenarios):
                        z[s,h,d,k,p] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}_{k}_{p}")

    for p in range(num_patients):
        if not is_mandatory[p]:
            w[p] = model.addVar(vtype=GRB.BINARY, name=f"w_{p}")

    # Objective components
    suite_open_cost = gp.quicksum(G[h][d] * u[h,d] for h in range(num_hospitals) for d in range(num_days))
    or_open_cost = gp.quicksum(F[h][d] * y[h,d,k] for h in range(num_hospitals) for d in range(num_days) for k in range(num_ors_per_h))

    sched_benefit = gp.quicksum(-kappa1 * omega[p] * x[h,d,p,k] for h in range(num_hospitals) for d in range(num_days) for p in range(num_patients) for k in range(num_ors_per_h))

    postponement_cost = gp.quicksum(kappa2 * c_unsched[p] * w[p] for p in range(num_patients) if not is_mandatory[p])

    cancel_cost_per_sc = []
    for s in range(num_scenarios):
        cancel_expr = gp.quicksum(
            c_cancel[p] * (x[h,d,p,k] - z[s,h,d,k,p])
            for h in range(num_hospitals)
            for d in range(num_days)
            for p in range(num_patients)
            for k in range(num_ors_per_h)
        )
        cancel_cost_per_sc.append(cancel_expr)

    expected_cancel = (1.0 / num_scenarios) * gp.quicksum(cancel_cost_per_sc)

    total_obj = suite_open_cost + or_open_cost + sched_benefit + postponement_cost + expected_cancel
    model.setObjective(total_obj, GRB.MINIMIZE)

    # Constraints

    # 1. Each mandatory patient assigned exactly once, non-mandatory either assigned or postponed
    for p in range(num_patients):
        assign_sum = gp.quicksum(x[h,d,p,k] for h in range(num_hospitals) for d in range(num_days) for k in range(num_ors_per_h))
        if is_mandatory[p]:
            model.addConstr(assign_sum == 1, name=f"assign_mand_{p}")
        else:
            model.addConstr(assign_sum + w[p] == 1, name=f"assign_nonmand_{p}")

    # 2. y only if u open
    for h in range(num_hospitals):
        for d in range(num_days):
            for k in range(num_ors_per_h):
                model.addConstr(y[h,d,k] <= u[h,d], name=f"y_implies_u_{h}_{d}_{k}")

    # 3. x only if y open
    for h in range(num_hospitals):
        for d in range(num_days):
            for k in range(num_ors_per_h):
                for p in range(num_patients):
                    model.addConstr(x[h,d,p,k] <= y[h,d,k], name=f"x_implies_y_{h}_{d}_{k}_{p}")

    # 4. Duration limit per scenario per OR (only performed surgeries)
    for s in range(num_scenarios):
        for h in range(num_hospitals):
            for d in range(num_days):
                for k in range(num_ors_per_h):
                    model.addConstr(
                        gp.quicksum(dur[s][p] * z[s,h,d,k,p] for p in range(num_patients)) <= B[h][d] * y[h,d,k],
                        name=f"dur_limit_s{s}_h{h}_d{d}_k{k}"
                    )

    # 5. z <= x
    for s in range(num_scenarios):
        for h in range(num_hospitals):
            for d in range(num_days):
                for k in range(num_ors_per_h):
                    for p in range(num_patients):
                        model.addConstr(z[s,h,d,k,p] <= x[h,d,p,k], name=f"z_implies_x_s{s}_{h}_{d}_{k}_{p}")

    # 6. At most one patient per OR per day in deterministic sense (no two x=1 for same h,d,k)
    # This helps reduce symmetry and is valid because of the duration constraints and positive durations.
    for h in range(num_hospitals):
        for d in range(num_days):
            for k in range(num_ors_per_h):
                model.addConstr(
                    gp.quicksum(x[h,d,p,k] for p in range(num_patients)) <= 1,
                    name=f"one_patient_per_or_{h}_{d}_{k}"
                )

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible.", file=sys.stderr)
        return
    elif model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        print(f"Warning: Solver status {model.status}", file=sys.stderr)

    obj_val = model.objVal if model.objVal is not None else 0.0

    u_sol = {}
    for h in range(num_hospitals):
        for d in range(num_days):
            val = int(round(u[h,d].X))
            if val > 0:
                u_sol[f"{h},{d}"] = val

    y_sol = {}
    for h in range(num_hospitals):
        for d in range(num_days):
            for k in range(num_ors_per_h):
                val = int(round(y[h,d,k].X))
                if val > 0:
                    y_sol[f"{h},{d},{k}"] = val

    x_sol = {}
    for h in range(num_hospitals):
        for d in range(num_days):
            for p_idx in range(num_patients):
                for k in range(num_ors_per_h):
                    val = int(round(x[h,d,p_idx,k].X))
                    if val > 0:
                        x_sol[f"{h},{d},{p_idx},{k}"] = val

    w_sol = {}
    for p in range(num_patients):
        if not is_mandatory[p]:
            val = int(round(w[p].X))
            if val > 0:
                w_sol[str(p)] = val

    solution = {
        "objective_value": float(obj_val),
        "u": u_sol,
        "y": y_sol,
        "x": x_sol,
        "w": w_sol
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Solved in {elapsed:.2f} seconds. Objective: {obj_val:.4f}")


if __name__ == "__main__":
    main()