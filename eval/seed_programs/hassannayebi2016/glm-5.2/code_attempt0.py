import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_objective(a_vals, T, D, NT, theta, lambdas, C):
    """Compute the exact total waiting time for a given schedule."""
    A = [0.0]
    for p in range(NT):
        A.append(A[-1] + theta * lambdas[p])
    total_demand = A[-1]

    def A_func(t):
        if t <= 0:
            return 0.0
        if t >= T:
            return total_demand
        p = int(t // theta)
        if p >= NT:
            p = NT - 1
        offset = t - p * theta
        return A[p] + offset * lambdas[p]

    obj = 0.0
    q_cur = 0.0
    a_prev = 0.0
    c_prev = 0.0
    for i in range(D):
        a_cur = a_vals[i]
        c_cur = A_func(a_cur)
        h = a_cur - a_prev
        arrivals = c_cur - c_prev
        occ = q_cur + arrivals / 2.0
        obj += occ * h
        s_cur = q_cur + arrivals
        b_cur = min(C, s_cur)
        q_cur = max(0.0, s_cur - b_cur)
        a_prev = a_cur
        c_prev = c_cur
    h = T - a_prev
    arrivals = total_demand - c_prev
    occ = q_cur + arrivals / 2.0
    obj += occ * h
    return obj


class SolutionCallback:
    def __init__(self, a_vars, D, logger):
        self.a_vars = a_vars
        self.D = D
        self.logger = logger

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            a_vals = model.cbGet(self.a_vars)
            solution_dict = {
                "departure_schedule_minutes": {f"train_{i+1}": a_vals[i] for i in range(self.D)}
            }
            if self.logger:
                self.logger.log_solution(obj, solution_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    params = instance['parameters']
    T = params['T_planning_horizon_minutes']
    D = params['D_num_train_services']
    NT = params['NT_num_demand_periods']
    theta = params['theta_period_length_minutes']
    h_min = params['h_min_minutes']
    h_max = params['h_max_minutes']
    C = params['C_train_capacity_passengers']
    demand_periods = instance['demand_periods']
    lambdas = [dp['lambda_pax_per_min'] for dp in demand_periods]

    A = [0.0]
    for p in range(NT):
        A.append(A[-1] + theta * lambdas[p])
    total_demand = A[-1]

    # Construct a feasible start solution
    if D == 1:
        a_start = [T / 2.0]
    else:
        h_interior = T / (D - 1)
        if h_interior < h_min:
            h_interior = h_min
        elif h_interior > h_max:
            h_interior = h_max
        rem = T - (D - 1) * h_interior
        init = rem / 2.0
        a_start = [init + i * h_interior for i in range(D)]

    a_start = [max(0.0, min(t, T - 1e-5)) for t in a_start]
    for i in range(1, D):
        if a_start[i] <= a_start[i-1]:
            a_start[i] = a_start[i-1] + 1e-5

    obj_start = compute_objective(a_start, T, D, NT, theta, lambdas, C)
    if logger:
        logger.log_solution(obj_start, {"departure_schedule_minutes": {f"train_{i+1}": a_start[i] for i in range(D)}})

    # Gurobi model
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('NonConvex', 2)
    model.setParam('OutputFlag', 0)
    model.setParam('Presolve', 2)
    model.setParam('TimeLimit', max(1, args.time_limit - 2))

    w = model.addVars(D, NT + 1, lb=0, name="w")
    a = model.addVars(D, lb=0, ub=T - 1e-5, name="a")
    c = model.addVars(D, lb=0, ub=total_demand, name="c")
    q = model.addVars(D + 1, lb=0, ub=total_demand, name="q")
    b = model.addVars(D, lb=0, ub=C, name="b")
    s = model.addVars(D, lb=0, ub=total_demand, name="s")

    for i in range(D):
        model.addConstr(gp.quicksum(w[i, p] for p in range(NT + 1)) == 1)
        model.addSOS(GRB.SOS_TYPE2, [w[i, p] for p in range(NT + 1)], [float(p) for p in range(NT + 1)])
        model.addConstr(a[i] == gp.quicksum(p * theta * w[i, p] for p in range(NT + 1)))
        model.addConstr(c[i] == gp.quicksum(A[p] * w[i, p] for p in range(NT + 1)))

    for i in range(D - 1):
        model.addConstr(a[i + 1] - a[i] >= h_min)
        model.addConstr(a[i + 1] - a[i] <= h_max)

    for i in range(D):
        model.addConstr(a[i] >= i * h_min)
        model.addConstr(a[i] <= T - (D - 1 - i) * h_min)

    model.addConstr(q[0] == 0)
    for i in range(D):
        c_prev = c[i - 1] if i > 0 else 0
        model.addConstr(s[i] == q[i] + c[i] - c_prev)
        model.addConstr(b[i] <= C)
        model.addConstr(b[i] <= s[i])
        model.addConstr(q[i + 1] == s[i] - b[i])

    obj = gp.QuadExpr()
    for i in range(D + 1):
        a_prev = a[i - 1] if i > 0 else 0
        a_next = a[i] if i < D else T
        c_prev = c[i - 1] if i > 0 else 0
        c_next = c[i] if i < D else total_demand
        q_cur = q[i]
        h = a_next - a_prev
        occ = q_cur + (c_next - c_prev) / 2.0
        obj += occ * h
    model.setObjective(obj, GRB.MINIMIZE)

    # Set MIP start
    q_cur_start = 0.0
    c_prev_start = 0.0
    for i in range(D):
        t = a_start[i]
        p = int(t // theta)
        if p >= NT:
            p = NT - 1
            offset = theta
        else:
            offset = t - p * theta
        if p == NT:
            w[i, NT].Start = 1.0
            c_cur_start = A[NT]
        else:
            w[i, p].Start = 1.0 - offset / theta
            w[i, p + 1].Start = offset / theta
            c_cur_start = A[p] + offset * lambdas[p]
        for q_idx in range(NT + 1):
            if q_idx != p and q_idx != p + 1:
                w[i, q_idx].Start = 0.0
        a[i].Start = t
        c[i].Start = c_cur_start
        s_cur_start = q_cur_start + c_cur_start - c_prev_start
        b_cur_start = min(C, s_cur_start)
        q_next_start = max(0.0, s_cur_start - b_cur_start)
        s[i].Start = s_cur_start
        b[i].Start = b_cur_start
        q[i + 1].Start = q_next_start
        q_cur_start = q_next_start
        c_prev_start = c_cur_start

    a_list = [a[i] for i in range(D)]
    callback = SolutionCallback(a_list, D, logger)
    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        a_vals = [a[i].X for i in range(D)]
    else:
        obj_val = obj_start
        a_vals = a_start

    solution = {
        "objective_value": float(obj_val),
        "departure_schedule_minutes": {f"train_{i+1}": float(a_vals[i]) for i in range(D)}
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()