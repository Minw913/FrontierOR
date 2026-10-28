import json
import argparse
import time
import math
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except Exception:
        logger = None

    start_time = time.time()

    with open(args.instance_path) as f:
        inst = json.load(f)

    g = inst['global_parameters']
    theta = float(g['theta'])
    sigma = float(g['sigma'])
    alpha = float(g['alpha'])
    beta = float(g['beta'])
    gamma = float(g['gamma'])
    alpha_prime = float(g['alpha_prime'])
    beta_prime = float(g['beta_prime'])
    gamma_prime = float(g['gamma_prime'])

    ip = inst['instance_parameters']
    N1 = int(ip['num_level1_facilities'])
    N2 = int(ip['num_level2_facilities'])
    Nc = int(ip['num_customers'])
    K = int(ip['num_interdiction_levels_K'])
    budget = float(ip['total_interdiction_budget'])

    ip_par = inst['interdiction_parameters']
    cost_l1 = {int(k): float(v) for k, v in ip_par['interdiction_cost_level1_h1'].items()}
    cost_l2 = {int(k): float(v) for k, v in ip_par['interdiction_cost_level2_h2'].items()}
    d1 = {int(k): float(v) for k, v in ip_par['capacity_reduction_level1_d1'].items()}
    d2 = {int(k): float(v) for k, v in ip_par['capacity_reduction_level2_d2'].items()}

    cost_l1[0] = cost_l1.get(0, 0.0)
    cost_l2[0] = cost_l2.get(0, 0.0)
    d1[0] = d1.get(0, 0.0)
    d2[0] = d2.get(0, 0.0)

    levels = list(range(K))

    customers = inst['customers']
    cust_D = [float(d) for d in customers['demands']]

    l1f = inst['level1_facilities']
    cap1_l1 = [float(c) for c in l1f['capacity_type1']]

    l2f = inst['level2_facilities']
    cap1_l2 = [float(c) for c in l2f['capacity_type1']]
    cap2_l2 = [float(c) for c in l2f['capacity_type2']]
    l2_x = l2f['x_coordinates']
    l2_y = l2f['y_coordinates']

    dist = inst['distances']
    c2l1 = dist['customer_to_level1_facility']
    c2l2 = dist['customer_to_level2_facility']
    l1tol2 = dist['level1_to_level2_facility']

    l2tol2 = [[math.hypot(l2_x[i] - l2_x[j], l2_y[i] - l2_y[j]) for j in range(N2)] for i in range(N2)]

    C1l1 = [[cap1_l1[j] * (1.0 - d1[l]) for l in levels] for j in range(N1)]
    C1l2 = [[cap1_l2[j] * (1.0 - d2[l]) for l in levels] for j in range(N2)]
    C2l2 = [[cap2_l2[j] * (1.0 - d2[l]) for l in levels] for j in range(N2)]

    # Build follower LP once (modify RHS each iteration)
    fl_model = gp.Model()
    fl_model.Params.OutputFlag = 0
    fl_model.Params.Method = 1

    x = {}
    yv = {}
    zv = {}
    u = [fl_model.addVar(lb=0, name=f"u_{i}") for i in range(Nc)]
    v = [fl_model.addVar(lb=0, name=f"v_{i}") for i in range(Nc)]
    r = {}
    ro = [fl_model.addVar(lb=0, name=f"ro_{j}") for j in range(N1)]
    r2 = {}
    r2o = [fl_model.addVar(lb=0, name=f"r2o_{j}") for j in range(N2)]

    for i in range(Nc):
        for j in range(N1):
            x[i, j] = fl_model.addVar(lb=0, obj=alpha * c2l1[i][j], name=f"x_{i}_{j}")
        for j in range(N2):
            yv[i, j] = fl_model.addVar(lb=0, obj=beta * c2l2[i][j], name=f"y_{i}_{j}")
            zv[i, j] = fl_model.addVar(lb=0, obj=beta * c2l2[i][j], name=f"z_{i}_{j}")

    for j in range(N1):
        for k in range(N2):
            r[j, k] = fl_model.addVar(lb=0, obj=gamma * l1tol2[j][k], name=f"r_{j}_{k}")

    for j in range(N2):
        for k in range(N2):
            r2[j, k] = fl_model.addVar(lb=0, obj=gamma * l2tol2[j][k], name=f"r2_{j}_{k}")

    for i in range(Nc):
        u[i].Obj = alpha_prime + sigma * gamma_prime
        v[i].Obj = beta_prime
    for j in range(N1):
        ro[j].Obj = gamma_prime
    for j in range(N2):
        r2o[j].Obj = gamma_prime

    fl_model.update()

    demand1 = []
    demand2 = []
    for i in range(Nc):
        demand1.append(fl_model.addConstr(
            gp.quicksum(x[i, j] for j in range(N1)) +
            gp.quicksum(yv[i, j] for j in range(N2)) + u[i] == theta * cust_D[i]
        ))
        demand2.append(fl_model.addConstr(
            gp.quicksum(zv[i, j] for j in range(N2)) + v[i] == (1.0 - theta) * cust_D[i]
        ))

    l1_cap = []
    l1_ref = []
    for j in range(N1):
        l1_cap.append(fl_model.addConstr(
            gp.quicksum(x[i, j] for i in range(Nc)) <= C1l1[j][0]
        ))
        l1_ref.append(fl_model.addConstr(
            sigma * gp.quicksum(x[i, j] for i in range(Nc)) ==
            gp.quicksum(r[j, k] for k in range(N2)) + ro[j]
        ))

    l2_t1_cap = []
    l2_ref = []
    l2_t2_cap = []
    for j in range(N2):
        l2_t1_cap.append(fl_model.addConstr(
            gp.quicksum(yv[i, j] for i in range(Nc)) <= C1l2[j][0]
        ))
        l2_ref.append(fl_model.addConstr(
            sigma * gp.quicksum(yv[i, j] for i in range(Nc)) ==
            gp.quicksum(r2[j, k] for k in range(N2)) + r2o[j]
        ))
        l2_t2_cap.append(fl_model.addConstr(
            gp.quicksum(zv[i, j] for i in range(Nc)) +
            gp.quicksum(r[jp, j] for jp in range(N1)) +
            gp.quicksum(r2[jp, j] for jp in range(N2)) <= C2l2[j][0]
        ))

    fl_model.update()

    def solve_follower(y_l1, y_l2):
        for j in range(N1):
            l1_cap[j].RHS = C1l1[j][y_l1[j]]
        for j in range(N2):
            l2_t1_cap[j].RHS = C1l2[j][y_l2[j]]
            l2_t2_cap[j].RHS = C2l2[j][y_l2[j]]
        fl_model.optimize()
        if fl_model.Status != GRB.OPTIMAL:
            return None, None
        cost = fl_model.ObjVal
        p1 = [demand1[i].Pi for i in range(Nc)]
        p2 = [demand2[i].Pi for i in range(Nc)]
        m1 = [l1_cap[j].Pi for j in range(N1)]
        m2 = [l2_t1_cap[j].Pi for j in range(N2)]
        m3 = [l2_t2_cap[j].Pi for j in range(N2)]
        return cost, (p1, p2, m1, m2, m3)

    # Master MILP
    master = gp.Model()
    master.Params.OutputFlag = 0
    master.Params.Seed = 0
    master.Params.MIPGap = 1e-4
    master.Params.NumericFocus = 0
    master.Params.Threads = 1

    theta_var = master.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    z_l1 = {}
    z_l2 = {}
    for j in range(N1):
        for l in levels:
            z_l1[j, l] = master.addVar(vtype=GRB.BINARY, name=f"zl1_{j}_{l}")
    for j in range(N2):
        for l in levels:
            z_l2[j, l] = master.addVar(vtype=GRB.BINARY, name=f"zl2_{j}_{l}")

    for j in range(N1):
        master.addConstr(gp.quicksum(z_l1[j, l] for l in levels) == 1)
    for j in range(N2):
        master.addConstr(gp.quicksum(z_l2[j, l] for l in levels) == 1)

    budget_expr = gp.LinExpr()
    for j in range(N1):
        for l in levels:
            if l > 0:
                budget_expr += cost_l1[l] * z_l1[j, l]
    for j in range(N2):
        for l in levels:
            if l > 0:
                budget_expr += cost_l2[l] * z_l2[j, l]
    master.addConstr(budget_expr <= budget)

    master.setObjective(theta_var, GRB.MAXIMIZE)
    master.update()

    def add_benders_cut(duals):
        p1, p2, m1, m2, m3 = duals
        const_k = (sum(p1[i] * theta * cust_D[i] for i in range(Nc)) +
                   sum(p2[i] * (1.0 - theta) * cust_D[i] for i in range(Nc)))
        expr = gp.LinExpr()
        for j in range(N1):
            for l in levels:
                if abs(m1[j]) > 1e-12:
                    expr += m1[j] * C1l1[j][l] * z_l1[j, l]
        for j in range(N2):
            for l in levels:
                if abs(m2[j]) > 1e-12:
                    expr += m2[j] * C1l2[j][l] * z_l2[j, l]
                if abs(m3[j]) > 1e-12:
                    expr += m3[j] * C2l2[j][l] * z_l2[j, l]
        master.addConstr(theta_var <= const_k + expr)

    def add_nogood(y_l1, y_l2):
        expr = gp.LinExpr()
        for j in range(N1):
            for l in levels:
                if l == y_l1[j]:
                    expr += (1.0 - z_l1[j, l])
                else:
                    expr += z_l1[j, l]
        for j in range(N2):
            for l in levels:
                if l == y_l2[j]:
                    expr += (1.0 - z_l2[j, l])
                else:
                    expr += z_l2[j, l]
        master.addConstr(expr >= 1)

    def get_master_y():
        y_l1 = []
        for j in range(N1):
            chosen = 0
            for l in levels:
                if z_l1[j, l].X > 0.5:
                    chosen = l
                    break
            y_l1.append(chosen)
        y_l2 = []
        for j in range(N2):
            chosen = 0
            for l in levels:
                if z_l2[j, l].X > 0.5:
                    chosen = l
                    break
            y_l2.append(chosen)
        return y_l1, y_l2

    def make_sol_dict(obj, y_l1, y_l2):
        interdiction = {}
        for j, l in enumerate(y_l1):
            interdiction[f"S1_{j}"] = int(l)
        for j, l in enumerate(y_l2):
            interdiction[f"S2_{j}"] = int(l)
        return {"objective_value": float(obj), "interdiction_pattern": interdiction}

    best_cost = -float('inf')
    best_y_l1 = [0] * N1
    best_y_l2 = [0] * N2

    # Initial exploration seeds
    seed_ys = [([0] * N1, [0] * N2)]
    for j in range(N2):
        for l in range(K - 1, 0, -1):
            if cost_l2[l] <= budget + 1e-9:
                yl2 = [0] * N2
                yl2[j] = l
                seed_ys.append(([0] * N1, yl2[:]))
                break
    for j in range(N1):
        for l in range(K - 1, 0, -1):
            if cost_l1[l] <= budget + 1e-9:
                yl1 = [0] * N1
                yl1[j] = l
                seed_ys.append((yl1[:], [0] * N2))
                break

    seen = set()
    for yl1, yl2 in seed_ys:
        if time.time() - start_time > args.time_limit * 0.5:
            break
        key = (tuple(yl1), tuple(yl2))
        if key in seen:
            continue
        seen.add(key)
        cost, duals = solve_follower(yl1, yl2)
        if cost is None:
            continue
        if duals is not None:
            add_benders_cut(duals)
        add_nogood(yl1, yl2)
        if cost > best_cost + 1e-9:
            best_cost = cost
            best_y_l1 = yl1[:]
            best_y_l2 = yl2[:]
            if logger:
                logger.log_solution(best_cost, make_sol_dict(best_cost, best_y_l1, best_y_l2))

    master.update()

    tol = 1e-4 * max(1.0, abs(best_cost))
    max_iter = 500
    iteration = 0

    while iteration < max_iter:
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed - 1.0
        if remaining <= 0.5:
            break
        master.Params.TimeLimit = max(1.0, remaining)
        master.optimize()
        if master.Status == GRB.INFEASIBLE or master.SolCount == 0:
            break
        ub = master.ObjVal
        if ub <= best_cost + tol:
            break
        y_l1_new, y_l2_new = get_master_y()
        key = (tuple(y_l1_new), tuple(y_l2_new))
        if key in seen:
            add_nogood(y_l1_new, y_l2_new)
            master.update()
            iteration += 1
            continue
        seen.add(key)
        cost, duals = solve_follower(y_l1_new, y_l2_new)
        if cost is None or duals is None:
            add_nogood(y_l1_new, y_l2_new)
            master.update()
            iteration += 1
            continue
        add_benders_cut(duals)
        add_nogood(y_l1_new, y_l2_new)
        master.update()
        if cost > best_cost + 1e-9:
            best_cost = cost
            best_y_l1 = y_l1_new[:]
            best_y_l2 = y_l2_new[:]
            if logger:
                logger.log_solution(best_cost, make_sol_dict(best_cost, best_y_l1, best_y_l2))
        iteration += 1

    sol = make_sol_dict(best_cost, best_y_l1, best_y_l2)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()