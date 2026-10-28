import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    N = data['num_patients']
    M = data['num_hospitals']
    T = data['num_days']
    K = data['num_ors_per_hospital']
    L = data['num_scenarios']

    patients = data['patients']
    hospitals = data['hospitals']
    scenarios = data['scenarios']

    p_ids = [p['patient_id'] for p in patients]
    h_ids = [h['hospital_id'] for h in hospitals]
    d_ids = list(range(T))
    k_ids = list(range(K))
    s_ids = list(range(L))

    p_id_to_idx = {p_id: i for i, p_id in enumerate(p_ids)}
    h_id_to_idx = {h_id: i for i, h_id in enumerate(h_ids)}

    B = [[0] * T for _ in range(M)]
    F = [[0] * T for _ in range(M)]
    G = [[0] * T for _ in range(M)]
    for h_idx, h in enumerate(hospitals):
        for d in h['days']:
            d_idx = d['day_id']
            B[h_idx][d_idx] = d['B_hd']
            F[h_idx][d_idx] = d['F_hd']
            G[h_idx][d_idx] = d['G_hd']

    c_sched = [[0] * T for _ in range(N)]
    c_unsched = [0] * N
    c_cancel = [0] * N
    is_mandatory = [False] * N
    for p_idx, p in enumerate(patients):
        c_unsched[p_idx] = p['c_unsched']
        c_cancel[p_idx] = p['c_cancel']
        is_mandatory[p_idx] = p['is_mandatory']
        for d_label, benefit in p['c_sched_per_day'].items():
            d_idx = int(d_label) - 1
            if 0 <= d_idx < T:
                c_sched[p_idx][d_idx] = benefit

    durations = [[0] * N for _ in range(L)]
    for s_idx, s in enumerate(scenarios):
        for p_idx, dur in enumerate(s['surgery_durations_minutes']):
            durations[s_idx][p_idx] = dur

    model = gp.Model("surgery_scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, args.time_limit - 1))

    u = model.addVars(h_ids, d_ids, vtype=GRB.BINARY, name="u")
    y = model.addVars(h_ids, d_ids, k_ids, vtype=GRB.BINARY, name="y")
    x = model.addVars(h_ids, d_ids, p_ids, k_ids, vtype=GRB.BINARY, name="x")
    w = model.addVars(p_ids, vtype=GRB.BINARY, name="w")
    z = model.addVars(s_ids, p_ids, h_ids, d_ids, k_ids, vtype=GRB.BINARY, name="z")

    obj = gp.LinExpr()
    for h_idx, h in enumerate(h_ids):
        for d in d_ids:
            obj.add(G[h_idx][d] * u[h, d])
            for k in k_ids:
                obj.add(F[h_idx][d] * y[h, d, k])

    for p_idx, p in enumerate(p_ids):
        for h in h_ids:
            for d in d_ids:
                for k in k_ids:
                    obj.add(-c_sched[p_idx][d] * x[h, d, p, k])
        obj.add(c_unsched[p_idx] * w[p])
        for s in s_ids:
            for h in h_ids:
                for d in d_ids:
                    for k in k_ids:
                        obj.add((1.0 / L) * c_cancel[p_idx] * z[s, p, h, d, k])

    model.setObjective(obj, GRB.MINIMIZE)

    for h in h_ids:
        for d in d_ids:
            for k in k_ids:
                model.addConstr(y[h, d, k] <= u[h, d], name=f"y_u_{h}_{d}_{k}")
                for p in p_ids:
                    model.addConstr(x[h, d, p, k] <= y[h, d, k], name=f"x_y_{h}_{d}_{p}_{k}")
                    for s in s_ids:
                        model.addConstr(z[s, p, h, d, k] <= x[h, d, p, k], name=f"z_x_{s}_{p}_{h}_{d}_{k}")

    for p_idx, p in enumerate(p_ids):
        if is_mandatory[p_idx]:
            model.addConstr(gp.quicksum(x[h, d, p, k] for h in h_ids for d in d_ids for k in k_ids) == 1, name=f"assign_m_{p}")
        else:
            model.addConstr(gp.quicksum(x[h, d, p, k] for h in h_ids for d in d_ids for k in k_ids) + w[p] == 1, name=f"assign_nm_{p}")

    for s in s_ids:
        for h_idx, h in enumerate(h_ids):
            for d in d_ids:
                for k in k_ids:
                    model.addConstr(
                        gp.quicksum(durations[s][p_idx] * (x[h, d, p, k] - z[s, p, h, d, k]) for p_idx, p in enumerate(p_ids)) <= B[h_idx][d],
                        name=f"cap_{s}_{h}_{d}_{k}"
                    )

    model._best_obj = float('inf')
    model._vars_u = u
    model._vars_y = y
    model._vars_x = x
    model._vars_w = w

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < model._best_obj:
                model._best_obj = obj_val
                sol_dict = {
                    "objective_value": obj_val,
                    "u": {},
                    "y": {},
                    "x": {},
                    "w": {}
                }
                u_vals = {k: v for k, v in model.cbGetSolution(model._vars_u).items() if v > 0.5}
                y_vals = {k: v for k, v in model.cbGetSolution(model._vars_y).items() if v > 0.5}
                x_vals = {k: v for k, v in model.cbGetSolution(model._vars_x).items() if v > 0.5}
                w_vals = {k: v for k, v in model.cbGetSolution(model._vars_w).items() if v > 0.5}

                for (h, d), v in u_vals.items():
                    sol_dict["u"][f"{h},{d}"] = 1
                for (h, d, k), v in y_vals.items():
                    sol_dict["y"][f"{h},{d},{k}"] = 1
                for (h, d, p, k), v in x_vals.items():
                    sol_dict["x"][f"{h},{d},{p},{k}"] = 1
                for p, v in w_vals.items():
                    sol_dict["w"][f"{p}"] = 1

                if logger:
                    logger.log_solution(obj_val, sol_dict)

    if logger:
        model.optimize(log_callback)
    else:
        model.optimize()

    solution = {
        "objective_value": model.ObjVal if model.SolCount > 0 else 0,
        "u": {},
        "y": {},
        "x": {},
        "w": {}
    }
    if model.SolCount > 0:
        for h in h_ids:
            for d in d_ids:
                if u[h, d].X > 0.5:
                    solution["u"][f"{h},{d}"] = 1
                    for k in k_ids:
                        if y[h, d, k].X > 0.5:
                            solution["y"][f"{h},{d},{k}"] = 1
                            for p in p_ids:
                                if x[h, d, p, k].X > 0.5:
                                    solution["x"][f"{h},{d},{p},{k}"] = 1
        for p in p_ids:
            if w[p].X > 0.5:
                solution["w"][f"{p}"] = 1

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    solve()