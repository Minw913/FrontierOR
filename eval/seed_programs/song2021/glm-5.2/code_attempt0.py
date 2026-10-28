import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def generate_start_solution(orders, capacities, H, b, f, x, z, O, T):
    """Generate a feasible start solution: earliest interval with uniform intensity."""
    start_vals = {}
    intervals = []
    for i, order in enumerate(orders):
        r = order["release_date_r"]
        l_min = order["lower_exec_interval_length_l"]
        l_max = order["upper_exec_interval_length_l_bar"]
        d = order["due_date_d"]

        chosen_s = None
        chosen_e = None
        chosen_L = None
        # find the earliest feasible interval
        for L in range(l_min, l_max + 1):
            for s in range(r, H - L + 1):
                e = s + L - 1
                if e <= H - 1:
                    chosen_s = s
                    chosen_e = e
                    chosen_L = L
                    break
            if chosen_s is not None:
                break

        if chosen_s is None:
            continue

        intensity = 1.0 / chosen_L
        intervals.append((i, chosen_s, chosen_e, chosen_L, order["work_content_p"]))

        start_vals[b[i][chosen_s]] = 1.0
        start_vals[f[i][chosen_e]] = 1.0
        for t in range(H):
            if chosen_s <= t <= chosen_e:
                start_vals[z[i][t]] = 1.0
                start_vals[x[i][t]] = intensity
            else:
                start_vals[z[i][t]] = 0.0
                start_vals[x[i][t]] = 0.0
        start_vals[T[i]] = max(0, chosen_e - d)

    # compute overtime
    R = [0.0] * H
    for i, s, e, L, p in intervals:
        intensity = 1.0 / L
        for t in range(s, e + 1):
            R[t] += p * intensity
    for t in range(H):
        O_t = max(0.0, R[t] - capacities[t])
        start_vals[O[t]] = O_t

    return start_vals


class IncumbentLogger(gp.Callback):
    """Callback to log every incumbent solution found by Gurobi."""

    def __init__(self, logger, orders, b, f, x, O, T, H, n):
        self.logger = logger
        self.orders = orders
        self.b = b
        self.f = f
        self.x = x
        self.O = O
        self.T = T
        self.H = H
        self.n = n
        self.last_obj = None

    def _build_solution(self, model, obj):
        orders_solution = []
        for i in range(self.n):
            start = None
            finish = None
            for t in self.b[i]:
                val = model.cbGetSolution(self.b[i][t])
                if val > 0.5:
                    start = t
                    break
            for t in self.f[i]:
                val = model.cbGetSolution(self.f[i][t])
                if val > 0.5:
                    finish = t
                    break

            intensities = {}
            if start is not None and finish is not None:
                for t in range(start, finish + 1):
                    val = model.cbGetSolution(self.x[i][t])
                    if val > 1e-9:
                        intensities[str(t)] = val

            orders_solution.append({
                "order_id": self.orders[i]["order_id"],
                "chosen_interval": [start, finish],
                "intensities": intensities
            })

        nonregular = {}
        for t in range(self.H):
            val = model.cbGetSolution(self.O[t])
            if val > 1e-9:
                nonregular[str(t)] = val

        return {
            "objective_value": obj,
            "orders_solution": orders_solution,
            "nonregular_capacity_by_period": nonregular
        }

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if self.logger and (self.last_obj is None or obj < self.last_obj - 1e-9):
                sol_dict = self._build_solution(model, obj)
                if sol_dict:
                    self.logger.log_solution(obj, sol_dict)
                    self.last_obj = obj


def main():
    parser = argparse.ArgumentParser(description="Solve order scheduling problem with Gurobi")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as fin:
        data = json.load(fin)

    H = data["planning_horizon_H"]
    n = data["num_orders_n"]
    sigma = data["unit_cost_nonregular_capacity_sigma"]
    capacities = data["capacities"]
    orders = data["orders"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create Gurobi model
    model = gp.Model("order_scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    b = {}   # b[i][t] = 1 if order i starts at period t
    f = {}   # f[i][t] = 1 if order i finishes at period t
    x = {}   # x[i][t] = intensity of order i in period t
    z = {}   # z[i][t] = 1 if order i is active in period t
    O = {}   # O[t] = overtime in period t
    T = {}   # T[i] = tardiness of order i

    for i, order in enumerate(orders):
        r = order["release_date_r"]
        UB = order["intensity_upper_bound_UB"]

        b_i = {}
        f_i = {}
        for t in range(r, H):
            b_i[t] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{t}")
            f_i[t] = model.addVar(vtype=GRB.BINARY, name=f"f_{i}_{t}")
        b[i] = b_i
        f[i] = f_i

        x_i = {}
        z_i = {}
        for t in range(H):
            x_i[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB, name=f"x_{i}_{t}")
            z_i[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"z_{i}_{t}")
        x[i] = x_i
        z[i] = z_i

        T[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{i}")

    for t in range(H):
        O[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"O_{t}")

    model.update()

    # Constraints
    for i, order in enumerate(orders):
        r = order["release_date_r"]
        l_min = order["lower_exec_interval_length_l"]
        l_max = order["upper_exec_interval_length_l_bar"]
        UB = order["intensity_upper_bound_UB"]
        LB = order["intensity_lower_bound_LB"]
        d = order["due_date_d"]

        # Exactly one start and one finish
        model.addConstr(gp.quicksum(b[i][t] for t in b[i]) == 1)
        model.addConstr(gp.quicksum(f[i][t] for t in f[i]) == 1)

        # Length constraints
        s_i = gp.quicksum(t * b[i][t] for t in b[i])
        e_i = gp.quicksum(t * f[i][t] for t in f[i])
        model.addConstr(e_i - s_i + 1 >= l_min)
        model.addConstr(e_i - s_i + 1 <= l_max)

        # Activity indicators and intensity bounds
        for t in range(H):
            S_it = gp.quicksum(b[i][s] for s in b[i] if s <= t)
            F_it = gp.quicksum(f[i][e] for e in f[i] if e < t)
            model.addConstr(z[i][t] == S_it - F_it)
            model.addConstr(x[i][t] <= UB * z[i][t])
            model.addConstr(x[i][t] >= LB * z[i][t])

        # Total intensity = 1
        model.addConstr(gp.quicksum(x[i][t] for t in range(H)) == 1)

        # Tardiness
        model.addConstr(T[i] >= e_i - d)

    # Resource constraints
    for t in range(H):
        R_t = gp.quicksum(orders[i]["work_content_p"] * x[i][t] for i in range(n))
        model.addConstr(O[t] >= R_t - capacities[t])

    # Objective
    model.setObjective(
        gp.quicksum(orders[i]["tardiness_cost_w"] * T[i] for i in range(n)) +
        sigma * gp.quicksum(O[t] for t in range(H)),
        GRB.MINIMIZE
    )

    # Provide a feasible start solution
    start_vals = generate_start_solution(orders, capacities, H, b, f, x, z, O, T)
    for var, val in start_vals.items():
        var.Start = val

    # Callback for logging incumbents
    callback = None
    if logger:
        callback = IncumbentLogger(logger, orders, b, f, x, O, T, H, n)

    model.optimize(callback)

    # Write final solution
    if model.SolCount > 0:
        orders_solution = []
        for i in range(n):
            start = None
            finish = None
            for t in b[i]:
                if b[i][t].X > 0.5:
                    start = t
                    break
            for t in f[i]:
                if f[i][t].X > 0.5:
                    finish = t
                    break

            intensities = {}
            if start is not None and finish is not None:
                for t in range(start, finish + 1):
                    val = x[i][t].X
                    if val > 1e-9:
                        intensities[str(t)] = val

            orders_solution.append({
                "order_id": orders[i]["order_id"],
                "chosen_interval": [start, finish],
                "intensities": intensities
            })

        nonregular = {}
        for t in range(H):
            val = O[t].X
            if val > 1e-9:
                nonregular[str(t)] = val

        obj = model.ObjVal
        sol_dict = {
            "objective_value": obj,
            "orders_solution": orders_solution,
            "nonregular_capacity_by_period": nonregular
        }

        with open(args.solution_path, 'w') as fout:
            json.dump(sol_dict, fout, indent=2)

        # Ensure the final incumbent is logged if it wasn't captured by callback
        if logger and (callback.last_obj is None or obj < callback.last_obj - 1e-9):
            logger.log_solution(obj, sol_dict)
    else:
        # No feasible solution found (should not happen with valid instances)
        sol_dict = {
            "objective_value": 1e20,
            "orders_solution": [],
            "nonregular_capacity_by_period": {}
        }
        with open(args.solution_path, 'w') as fout:
            json.dump(sol_dict, fout, indent=2)


if __name__ == "__main__":
    main()