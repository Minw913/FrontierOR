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

    num_g = data['num_generators']
    T = data['num_time_periods']
    generators = data['generators']
    demand = data['demand']
    reserve = data['reserve_requirement']
    hours_per_period = data['time_period_hours']

    gen_dict = {g['id']: g for g in generators}
    gen_ids = [g['id'] for g in generators]

    model = gp.Model("UnitCommitment")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.001
    model.Params.LogToConsole = 0

    # Variables
    u = {}
    p = {}
    pb = {}
    s = {}
    startup_cost = {}

    for gid in gen_ids:
        g = gen_dict[gid]
        for t in range(T):
            u[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{gid}_{t}")
            p[gid, t] = model.addVar(lb=0, ub=g['p_max'], name=f"p_{gid}_{t}")
            pb[gid, t] = model.addVar(lb=0, ub=g['p_max'], name=f"pb_{gid}_{t}")
        for t in range(T):
            s[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"s_{gid}_{t}")

    # Initial status processing
    initial_on = {}
    initial_off = {}
    must_on_init = {}
    must_off_init = {}
    initial_p = {}

    for gid in gen_ids:
        g = gen_dict[gid]
        init = g['initial_status']
        initial_p[gid] = g['initial_power']
        if init > 0:
            initial_on[gid] = True
            initial_off[gid] = False
            must_on_init[gid] = max(0, g['min_up_time'] - init)
        else:
            initial_on[gid] = False
            initial_off[gid] = True
            must_off_init[gid] = max(0, g['min_down_time'] + init)  # init is negative

    # Objective: no-load + production + startup
    total_cost = 0.0

    for gid in gen_ids:
        g = gen_dict[gid]
        # No-load
        for t in range(T):
            total_cost += g['no_load_cost'] * u[gid, t]

        # Production cost (piecewise linear)
        segments = g['production_cost_segments']
        for t in range(T):
            pvar = p[gid, t]
            prev = 0.0
            for seg in segments:
                start = seg['output_mw_start']
                end = seg['output_mw_end']
                mc = seg['marginal_cost_per_mwh']
                width = end - start
                if width > 1e-6:
                    delta = model.addVar(lb=0, ub=width, name=f"delta_{gid}_{t}_{start}")
                    total_cost += mc * delta * hours_per_period
                    model.addConstr(pvar >= prev)
                    model.addConstr(pvar <= prev + delta)
                    prev = end
            # Force to zero if off
            model.addConstr(pvar <= g['p_max'] * u[gid, t])

        # Startup costs - use auxiliary continuous variables with big-M
        startup_levels = g['startup_costs']
        if startup_levels:
            max_dt = max((s['max_downtime_hours'] or 10000) for s in startup_levels)
            bigM = max_dt + 1
            downtime = {}
            for t in range(T):
                downtime[t] = model.addVar(lb=0, ub=bigM, name=f"down_{gid}_{t}")
                startup_cost[gid, t] = model.addVar(lb=0, name=f"scost_{gid}_{t}")
                total_cost += startup_cost[gid, t]

            for t in range(T):
                # Link s to u
                if t == 0:
                    if initial_on[gid]:
                        model.addConstr(s[gid, t] == 0)
                    else:
                        model.addConstr(s[gid, t] == u[gid, t])
                else:
                    model.addConstr(s[gid, t] <= u[gid, t])
                    model.addConstr(s[gid, t] <= 1 - u[gid, t-1])
                    model.addConstr(s[gid, t] >= u[gid, t] - u[gid, t-1])

                # Downtime estimation
                if t == 0:
                    if initial_off[gid]:
                        model.addConstr(downtime[t] >= -initial_off_periods(gid, data))  # will handle via cost tiers
                        model.addConstr(downtime[t] <= bigM)
                    else:
                        model.addConstr(downtime[t] == 0)
                else:
                    model.addConstr(downtime[t] >= downtime[t-1] + 1 - bigM * u[gid, t-1])
                    model.addConstr(downtime[t] <= downtime[t-1] + 1)
                    model.addConstr(downtime[t] <= bigM * (1 - u[gid, t]))

                # Startup cost linearization (tiered)
                cost_expr = 0.0
                prev_max_h = 0
                for tier in startup_levels:
                    c = tier['cost']
                    max_h = tier['max_downtime_hours']
                    if max_h is None:
                        max_h = bigM * hours_per_period
                    hours_var = downtime[t] * hours_per_period
                    # Use indicator for tier
                    in_tier = model.addVar(vtype=GRB.BINARY)
                    model.addConstr(hours_var <= max_h + bigM * (1 - in_tier))
                    model.addConstr(hours_var >= prev_max_h + 1 - bigM * in_tier)
                    cost_expr += c * in_tier
                    prev_max_h = max_h
                model.addConstr(startup_cost[gid, t] == cost_expr * s[gid, t])

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Power balance and reserve
    for t in range(T):
        model.addConstr(gp.quicksum(p[gid, t] for gid in gen_ids) == demand[t])
        model.addConstr(gp.quicksum(pb[gid, t] for gid in gen_ids) >= demand[t] + reserve[t])

    # Reserve and power linking
    for gid in gen_ids:
        g = gen_dict[gid]
        for t in range(T):
            model.addConstr(pb[gid, t] >= p[gid, t])
            model.addConstr(pb[gid, t] <= g['p_max'] * u[gid, t])
            model.addConstr(p[gid, t] >= g['p_min'] * u[gid, t])

    # Ramping and min up/down
    for gid_idx, gid in enumerate(gen_ids):
        g = gen_dict[gid]
        pmin = g['p_min']
        pmax = g['p_max']
        rup = g['ramp_up_limit']
        rdown = g['ramp_down_limit']
        srup = g['startup_ramp_limit']
        sdown = g['shutdown_ramp_limit']
        minup = g['min_up_time']
        mindown = g['min_down_time']

        for t in range(T):
            # Startup and shutdown power limits
            if t == 0:
                if initial_on[gid]:
                    model.addConstr(p[gid, t] <= initial_p[gid] + rup)
                    model.addConstr(p[gid, t] >= initial_p[gid] - rdown)
                else:
                    model.addConstr(p[gid, t] <= srup * s[gid, t])
            else:
                # Ramp when staying on
                model.addConstr(p[gid, t] - p[gid, t-1] <= rup * u[gid, t-1] + srup * s[gid, t])
                model.addConstr(p[gid, t-1] - p[gid, t] <= rdown * u[gid, t] + sdown * (u[gid, t-1] - u[gid, t]))

            # Min up time
            if minup > 1:
                for k in range(1, min(minup, T - t)):
                    model.addConstr(u[gid, t + k] >= s[gid, t])

            # Min down time
            if mindown > 1:
                for k in range(1, min(mindown, T - t)):
                    model.addConstr(u[gid, t + k] <= 1 - s[gid, t])

        # Initial must-run constraints
        init_on = must_on_init.get(gid, 0)
        for t in range(min(init_on, T)):
            model.addConstr(u[gid, t] == 1)

        init_off = must_off_init.get(gid, 0)
        for t in range(min(init_off, T)):
            model.addConstr(u[gid, t] == 0)

        # Initial ramp from t=0
        if initial_on[gid] and T > 0:
            model.addConstr(p[gid, 0] <= initial_p[gid] + rup)
            model.addConstr(p[gid, 0] >= initial_p[gid] - rdown)
            model.addConstr(pb[gid, 0] <= initial_p[gid] + rup)  # conservative

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    status = model.status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        # If infeasible, try to return something (though should not happen)
        objective = 1e10
        solution = {gid: {"u": [0]*T, "p": [0.0]*T, "pb": [0.0]*T} for gid in gen_ids}
    else:
        objective = model.objVal
        solution = {}
        for gid in gen_ids:
            u_sol = [int(u[gid, t].X + 0.5) for t in range(T)]
            p_sol = [float(p[gid, t].X) for t in range(T)]
            pb_sol = [float(pb[gid, t].X) for t in range(T)]
            solution[gid] = {
                "u": u_sol,
                "p": p_sol,
                "pb": pb_sol
            }

    result = {
        "objective_value": float(objective),
        "schedule": solution
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

def initial_off_periods(gid, data):
    g = next(g for g in data['generators'] if g['id'] == gid)
    init = g['initial_status']
    if init < 0:
        return -init
    return 0

if __name__ == "__main__":
    main()