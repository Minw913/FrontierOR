import argparse
import json
import math
from typing import Dict, List, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_production_pwl_points(gen: Dict[str, Any], hours: float) -> Tuple[List[float], List[float]]:
    """
    Build (x,y) points for production variable cost as a function of p (MW).
    y is period cost ($) for that period, so marginal costs are multiplied by period hours.
    """
    p_max = float(gen["p_max"])
    segments = sorted(gen.get("production_cost_segments", []), key=lambda s: float(s["output_mw_start"]))

    x_pts = [0.0]
    y_pts = [0.0]

    if not segments:
        if p_max > 0:
            x_pts.append(p_max)
            y_pts.append(0.0)
        return x_pts, y_pts

    curr_x = 0.0
    curr_cost = 0.0
    last_slope = 0.0

    for seg in segments:
        a = float(seg["output_mw_start"])
        b = float(seg["output_mw_end"])
        m = float(seg["marginal_cost_per_mwh"]) * hours
        last_slope = m

        if b <= curr_x + 1e-12:
            continue

        if a > curr_x + 1e-12:
            x_pts.append(a)
            y_pts.append(curr_cost)
            curr_x = a

        if b > curr_x + 1e-12:
            curr_cost += m * (b - curr_x)
            x_pts.append(b)
            y_pts.append(curr_cost)
            curr_x = b

    if curr_x < p_max - 1e-12:
        curr_cost += last_slope * (p_max - curr_x)
        x_pts.append(p_max)
        y_pts.append(curr_cost)
    elif curr_x > p_max + 1e-12:
        # If segment data exceeds p_max, clip by interpolation on last segment
        # (rare; we keep function consistent up to p_max).
        pass

    # Remove duplicate x points if any
    clean_x = [x_pts[0]]
    clean_y = [y_pts[0]]
    for i in range(1, len(x_pts)):
        if abs(x_pts[i] - clean_x[-1]) > 1e-12:
            clean_x.append(x_pts[i])
            clean_y.append(y_pts[i])
        else:
            clean_y[-1] = y_pts[i]

    # Ensure last point reaches p_max
    if clean_x[-1] < p_max - 1e-12:
        clean_x.append(p_max)
        clean_y.append(clean_y[-1])

    return clean_x, clean_y


def sort_startup_tiers(startup_costs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def k(t):
        mdh = t.get("max_downtime_hours", None)
        return float("inf") if mdh is None else float(mdh)
    return sorted(startup_costs, key=k)


def solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    G = int(instance["num_generators"])
    T = int(instance["num_time_periods"])
    hours = float(instance["time_period_hours"])

    gens = instance["generators"]
    demand = [float(x) for x in instance["demand"]]
    reserve = [float(x) for x in instance["reserve_requirement"]]

    model = gp.Model("unit_commitment")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    # Index mapping
    gen_ids = [g["id"] for g in gens]

    # Decision variables
    u = {}       # on/off
    v = {}       # startup
    w = {}       # shutdown
    p = {}       # power output
    pb = {}      # available power for reserve
    d_off = {}   # consecutive off periods up to t (including t)
    pcost = {}   # production cost var per (g,t)
    delta = {}   # startup tier selection binaries (g,t,l)

    # Build vars
    for i, g in enumerate(gens):
        gid = g["id"]
        startup_tiers = sort_startup_tiers(g.get("startup_costs", []))
        for t in range(T):
            u[i, t] = model.addVar(vtype=GRB.BINARY, name=f"u[{gid},{t}]")
            v[i, t] = model.addVar(vtype=GRB.BINARY, name=f"v[{gid},{t}]")
            w[i, t] = model.addVar(vtype=GRB.BINARY, name=f"w[{gid},{t}]")
            p[i, t] = model.addVar(lb=0.0, ub=float(g["p_max"]), vtype=GRB.CONTINUOUS, name=f"p[{gid},{t}]")
            pb[i, t] = model.addVar(lb=0.0, ub=float(g["p_max"]), vtype=GRB.CONTINUOUS, name=f"pb[{gid},{t}]")
            d_off[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"doff[{gid},{t}]")
            pcost[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"pcost[{gid},{t}]")

            for l, tier in enumerate(startup_tiers):
                delta[i, t, l] = model.addVar(vtype=GRB.BINARY, name=f"delta[{gid},{t},{l}]")

    model.update()

    # Constraints
    for i, g in enumerate(gens):
        p_min = float(g["p_min"])
        p_max = float(g["p_max"])
        ru = float(g["ramp_up_limit"])
        rd = float(g["ramp_down_limit"])
        su = float(g["startup_ramp_limit"])
        sd = float(g["shutdown_ramp_limit"])
        min_up = int(g["min_up_time"])
        min_down = int(g["min_down_time"])
        initial_status = int(g["initial_status"])
        p0 = float(g["initial_power"])
        u0 = 1 if initial_status > 0 else 0

        # on/off and reserve linkage
        for t in range(T):
            model.addConstr(p[i, t] >= p_min * u[i, t], name=f"pmin[{i},{t}]")
            model.addConstr(p[i, t] <= p_max * u[i, t], name=f"pmax[{i},{t}]")
            model.addConstr(pb[i, t] >= p[i, t], name=f"pb_ge_p[{i},{t}]")
            model.addConstr(pb[i, t] <= p_max * u[i, t], name=f"pb_le_pmaxu[{i},{t}]")

        # startup/shutdown transition
        for t in range(T):
            if t == 0:
                model.addConstr(u[i, t] - u0 == v[i, t] - w[i, t], name=f"logic0[{i}]")
            else:
                model.addConstr(u[i, t] - u[i, t - 1] == v[i, t] - w[i, t], name=f"logic[{i},{t}]")
            model.addConstr(v[i, t] + w[i, t] <= 1, name=f"no_simul_sw[{i},{t}]")

        # ramping with startup/shutdown allowances
        for t in range(T):
            if t == 0:
                model.addConstr(
                    p[i, t] - p0 <= ru * u0 + su * v[i, t],
                    name=f"ramp_up0[{i}]"
                )
                model.addConstr(
                    p0 - p[i, t] <= rd * u[i, t] + sd * w[i, t],
                    name=f"ramp_down0[{i}]"
                )
            else:
                model.addConstr(
                    p[i, t] - p[i, t - 1] <= ru * u[i, t - 1] + su * v[i, t],
                    name=f"ramp_up[{i},{t}]"
                )
                model.addConstr(
                    p[i, t - 1] - p[i, t] <= rd * u[i, t] + sd * w[i, t],
                    name=f"ramp_down[{i},{t}]"
                )

        # minimum up/down times
        if min_up > 0:
            for t in range(T):
                lhs = gp.quicksum(v[i, k] for k in range(max(0, t - min_up + 1), t + 1))
                model.addConstr(lhs <= u[i, t], name=f"minup[{i},{t}]")

        if min_down > 0:
            for t in range(T):
                lhs = gp.quicksum(w[i, k] for k in range(max(0, t - min_down + 1), t + 1))
                model.addConstr(lhs <= 1 - u[i, t], name=f"mindown[{i},{t}]")

        # Initial residual up/down requirements
        if u0 == 1:
            remaining_on = max(0, min_up - initial_status)
            for t in range(min(T, remaining_on)):
                model.addConstr(u[i, t] == 1, name=f"init_on_fix[{i},{t}]")
        else:
            off_hist = -initial_status if initial_status < 0 else 0
            remaining_off = max(0, min_down - off_hist)
            for t in range(min(T, remaining_off)):
                model.addConstr(u[i, t] == 0, name=f"init_off_fix[{i},{t}]")

        # Consecutive off-time tracking d_off
        d_init = (-initial_status) if initial_status < 0 else 0
        Moff = float(T + abs(initial_status) + 5)

        for t in range(T):
            prev = d_init if t == 0 else d_off[i, t - 1]
            model.addConstr(d_off[i, t] <= Moff * (1 - u[i, t]), name=f"doff_zero_on[{i},{t}]")
            model.addConstr(d_off[i, t] >= prev + 1 - Moff * u[i, t], name=f"doff_lb[{i},{t}]")
            model.addConstr(d_off[i, t] <= prev + 1 + Moff * u[i, t], name=f"doff_ub[{i},{t}]")

        # Startup tier selection and validity by prior downtime
        startup_tiers = sort_startup_tiers(g.get("startup_costs", []))
        for t in range(T):
            if startup_tiers:
                model.addConstr(
                    gp.quicksum(delta[i, t, l] for l in range(len(startup_tiers))) == v[i, t],
                    name=f"tier_select[{i},{t}]"
                )
            prev_off = d_init if t == 0 else d_off[i, t - 1]
            Mh = Moff * hours
            prev_max_hours = None

            for l, tier in enumerate(startup_tiers):
                max_h = tier.get("max_downtime_hours", None)

                # lower bound: downtime_hours > prev_max_hours
                if prev_max_hours is not None:
                    eps = 1e-4
                    model.addConstr(
                        hours * prev_off >= (float(prev_max_hours) + eps) - Mh * (1 - delta[i, t, l]),
                        name=f"tier_lb[{i},{t},{l}]"
                    )

                # upper bound: downtime_hours <= max_h (if finite)
                if max_h is not None:
                    model.addConstr(
                        hours * prev_off <= float(max_h) + Mh * (1 - delta[i, t, l]),
                        name=f"tier_ub[{i},{t},{l}]"
                    )

                if max_h is not None:
                    prev_max_hours = float(max_h)

        # Production cost PWL constraint
        x_pts, y_pts = build_production_pwl_points(g, hours)
        for t in range(T):
            model.addGenConstrPWL(p[i, t], pcost[i, t], x_pts, y_pts, name=f"pwl[{i},{t}]")

    # System constraints
    for t in range(T):
        model.addConstr(gp.quicksum(p[i, t] for i in range(G)) == demand[t], name=f"demand[{t}]")
        model.addConstr(gp.quicksum(pb[i, t] for i in range(G)) >= demand[t] + reserve[t], name=f"reserve[{t}]")

    # Objective
    obj = gp.LinExpr()
    for i, g in enumerate(gens):
        no_load = float(g["no_load_cost"])
        startup_tiers = sort_startup_tiers(g.get("startup_costs", []))

        for t in range(T):
            obj += no_load * u[i, t]
            obj += pcost[i, t]
            for l, tier in enumerate(startup_tiers):
                obj += float(tier["cost"]) * delta[i, t, l]

    model.setObjective(obj, GRB.MINIMIZE)

    # Incumbent logging callback
    best_logged = [float("inf")]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-6 < best_logged[0]:
                best_logged[0] = val
                logger.log(float(val))

    model.optimize(cb)

    if model.SolCount == 0:
        # Fallback (should be rare)
        schedule = {}
        for g in gens:
            schedule[g["id"]] = {
                "u": [0] * T,
                "p": [0.0] * T,
                "pb": [0.0] * T
            }
        return {"objective_value": float("inf"), "schedule": schedule}

    if logger is not None and model.ObjVal + 1e-6 < best_logged[0]:
        logger.log(float(model.ObjVal))

    # Build output
    schedule = {}
    for i, g in enumerate(gens):
        gid = g["id"]
        u_list = [int(round(u[i, t].X)) for t in range(T)]
        p_list = [float(p[i, t].X) for t in range(T)]
        pb_list = [float(pb[i, t].X) for t in range(T)]
        schedule[gid] = {"u": u_list, "p": p_list, "pb": pb_list}

    return {
        "objective_value": float(model.ObjVal),
        "schedule": schedule
    }


def main():
    parser = argparse.ArgumentParser(description="Solve thermal unit commitment with reserves.")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()