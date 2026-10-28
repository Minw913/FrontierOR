import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB, quicksum


def main():
    parser = argparse.ArgumentParser(description="Unit Commitment Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_g = instance['num_generators']
    num_t = instance['num_time_periods']
    period_hours = instance['time_period_hours']
    generators = instance['generators']
    demand = instance['demand']
    reserve_req = instance['reserve_requirement']

    model = gp.Model("UC")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 1)

    # Decision variables
    u = {}
    v = {}
    w = {}
    p = {}
    pb = {}
    delta = {}
    y = {}

    for g in range(num_g):
        gen = generators[g]
        num_segs = len(gen['production_cost_segments'])
        num_tiers = len(gen['startup_costs'])

        for t in range(num_t):
            u[g, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{g}_{t}")
            v[g, t] = model.addVar(vtype=GRB.BINARY, name=f"v_{g}_{t}")
            w[g, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{g}_{t}")
            p[g, t] = model.addVar(lb=0, ub=gen['p_max'], name=f"p_{g}_{t}")
            pb[g, t] = model.addVar(lb=0, ub=gen['p_max'], name=f"pb_{g}_{t}")

            for k in range(num_segs):
                seg = gen['production_cost_segments'][k]
                seg_width = seg['output_mw_end'] - seg['output_mw_start']
                delta[g, t, k] = model.addVar(lb=0, ub=seg_width, name=f"d_{g}_{t}_{k}")

            for k in range(num_tiers):
                y[g, t, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{g}_{t}_{k}")

    model.update()

    # Objective function
    obj = gp.LinExpr()
    for g in range(num_g):
        gen = generators[g]
        for t in range(num_t):
            # No-load cost
            obj += gen['no_load_cost'] * u[g, t]
            # Piecewise linear production cost
            for k, seg in enumerate(gen['production_cost_segments']):
                obj += seg['marginal_cost_per_mwh'] * period_hours * delta[g, t, k]
            # Startup cost
            for k, tier in enumerate(gen['startup_costs']):
                obj += tier['cost'] * y[g, t, k]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    for g in range(num_g):
        gen = generators[g]
        p_min = gen['p_min']
        p_max_g = gen['p_max']
        RU = gen['ramp_up_limit']
        RD = gen['ramp_down_limit']
        SU = gen['startup_ramp_limit']
        SD = gen['shutdown_ramp_limit']
        UT = gen['min_up_time']
        DT = gen['min_down_time']
        init_status = gen['initial_status']
        init_power = gen['initial_power']
        num_segs = len(gen['production_cost_segments'])
        num_tiers = len(gen['startup_costs'])

        u_init = 1 if init_status > 0 else 0

        for t in range(num_t):
            # Logical linking: u[t] - u[t-1] = v[t] - w[t]
            if t == 0:
                model.addConstr(u[g, t] - u_init == v[g, t] - w[g, t])
            else:
                model.addConstr(u[g, t] - u[g, t - 1] == v[g, t] - w[g, t])

            # Cannot start up and shut down simultaneously
            model.addConstr(v[g, t] + w[g, t] <= 1)

            # Power output bounds
            model.addConstr(p[g, t] >= p_min * u[g, t])
            model.addConstr(p[g, t] <= p_max_g * u[g, t])

            # Reserve capacity bounds
            model.addConstr(pb[g, t] >= p[g, t])
            model.addConstr(pb[g, t] <= p_max_g * u[g, t])

            # Piecewise linear cost segments
            model.addConstr(
                quicksum(delta[g, t, k] for k in range(num_segs)) == p[g, t] - p_min * u[g, t]
            )
            for k in range(num_segs):
                seg_width = gen['production_cost_segments'][k]['output_mw_end'] - gen['production_cost_segments'][k]['output_mw_start']
                model.addConstr(delta[g, t, k] <= seg_width * u[g, t])

            # Startup cost tier assignment
            model.addConstr(quicksum(y[g, t, k] for k in range(num_tiers)) == v[g, t])

            for k, tier in enumerate(gen['startup_costs']):
                if tier['max_downtime_hours'] is not None:
                    D_k = tier['max_downtime_hours'] // period_hours
                    # Sum of shutdowns in the relevant window
                    lo = max(0, t - D_k)
                    if lo < t:
                        shutdown_sum = quicksum(w[g, tau] for tau in range(lo, t))
                    else:
                        shutdown_sum = gp.LinExpr(0)

                    # Check initial condition contribution
                    if init_status < 0:
                        initial_off = abs(init_status)
                        total_off_at_t = initial_off + t
                        if total_off_at_t * period_hours <= tier['max_downtime_hours']:
                            # Linearized: y <= shutdown_sum + 1 - sum(v before t)
                            if t > 0:
                                v_sum = quicksum(v[g, tau] for tau in range(0, t))
                            else:
                                v_sum = gp.LinExpr(0)
                            model.addConstr(y[g, t, k] <= shutdown_sum + 1 - v_sum)
                        else:
                            model.addConstr(y[g, t, k] <= shutdown_sum)
                    else:
                        model.addConstr(y[g, t, k] <= shutdown_sum)

        # Ramping constraints
        for t in range(num_t):
            if t == 0:
                if u_init == 1:
                    # Ramp up from initial state
                    model.addConstr(p[g, 0] - init_power <= RU + SU * v[g, 0])
                    # Ramp down from initial state
                    model.addConstr(init_power - p[g, 0] <= RD * u[g, 0] + SD * w[g, 0])
                # If initially off, startup ramp is handled by the general constraint
                # (p[g,0] <= SU when v[g,0]=1 via ramp-up constraint with u[g,-1]=0)
            else:
                # Ramp up: p[t] - p[t-1] <= RU * u[t-1] + SU * v[t]
                model.addConstr(p[g, t] - p[g, t - 1] <= RU * u[g, t - 1] + SU * v[g, t])
                # Ramp down: p[t-1] - p[t] <= RD * u[t] + SD * w[t]
                model.addConstr(p[g, t - 1] - p[g, t] <= RD * u[g, t] + SD * w[g, t])

        # Initial minimum up/down time enforcement
        if init_status > 0:
            remaining_on = UT - init_status
            if remaining_on > 0:
                for t in range(min(remaining_on, num_t)):
                    model.addConstr(u[g, t] == 1)
        elif init_status < 0:
            remaining_off = DT - abs(init_status)
            if remaining_off > 0:
                for t in range(min(remaining_off, num_t)):
                    model.addConstr(u[g, t] == 0)

        # Minimum up time constraints
        for t in range(num_t):
            end_t = min(t + UT, num_t)
            model.addConstr(
                quicksum(u[g, tau] for tau in range(t, end_t)) >= (end_t - t) * v[g, t]
            )

        # Minimum down time constraints
        for t in range(num_t):
            end_t = min(t + DT, num_t)
            model.addConstr(
                quicksum(1 - u[g, tau] for tau in range(t, end_t)) >= (end_t - t) * w[g, t]
            )

    # System-wide constraints
    for t in range(num_t):
        # Power balance
        model.addConstr(
            quicksum(p[g, t] for g in range(num_g)) == demand[t]
        )
        # Reserve requirement
        model.addConstr(
            quicksum(pb[g, t] for g in range(num_g)) >= demand[t] + reserve_req[t]
        )

    model.update()

    # Callback to log incumbent solutions
    best_obj = float('inf')

    def callback(model_cb, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                solution = extract_solution_cb(model_cb, generators, num_g, num_t, u, p, pb)
                if logger:
                    logger.log_solution(obj_val, solution)

    model.optimize(callback)

    # Write final solution
    if model.SolCount > 0:
        solution = extract_solution(model, generators, num_g, num_t, u, p, pb)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Write an empty/infeasible indicator
        solution = {"objective_value": None, "schedule": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


def extract_solution(model, generators, num_g, num_t, u, p, pb):
    solution = {
        "objective_value": model.ObjVal,
        "schedule": {}
    }
    for g in range(num_g):
        gen = generators[g]
        solution["schedule"][gen['id']] = {
            "u": [int(round(u[g, t].X)) for t in range(num_t)],
            "p": [round(p[g, t].X, 6) for t in range(num_t)],
            "pb": [round(pb[g, t].X, 6) for t in range(num_t)]
        }
    return solution


def extract_solution_cb(model, generators, num_g, num_t, u, p, pb):
    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
    solution = {
        "objective_value": obj_val,
        "schedule": {}
    }
    for g in range(num_g):
        gen = generators[g]
        solution["schedule"][gen['id']] = {
            "u": [int(round(model.cbGetSolution(u[g, t]))) for t in range(num_t)],
            "p": [round(model.cbGetSolution(p[g, t]), 6) for t in range(num_t)],
            "pb": [round(model.cbGetSolution(pb[g, t]), 6) for t in range(num_t)]
        }
    return solution


if __name__ == "__main__":
    main()