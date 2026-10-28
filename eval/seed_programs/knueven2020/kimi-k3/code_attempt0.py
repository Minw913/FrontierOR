import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-7


def normalize_segments(raw_segments, p_min, p_max):
    if p_max < p_min - TOL:
        raise ValueError("Generator p_max is smaller than p_min.")

    total_width = p_max - p_min
    if total_width <= 1e-10:
        return []

    raw = []
    for seg in raw_segments or []:
        start = float(seg["output_mw_start"])
        end = float(seg["output_mw_end"])
        slope = float(seg["marginal_cost_per_mwh"])
        if end > start + TOL:
            raw.append((start, end, slope))
    raw.sort(key=lambda x: (x[0], x[1]))

    pieces = []
    cursor = p_min
    last_slope = 0.0

    for start, end, slope in raw:
        start = max(start, p_min)
        end = min(end, p_max)
        if end <= p_min + TOL or start >= p_max - TOL:
            continue
        if end <= cursor + TOL:
            last_slope = slope
            continue

        if start > cursor + TOL:
            pieces.append((start - cursor, slope))
            cursor = start

        actual_start = max(start, cursor)
        if end > actual_start + TOL:
            pieces.append((end - actual_start, slope))
            cursor = end
            last_slope = slope

    if cursor < p_max - TOL:
        pieces.append((p_max - cursor, last_slope))

    merged = []
    for width, slope in pieces:
        if width <= 1e-10:
            continue
        if merged and abs(slope - merged[-1][1]) <= 1e-11:
            merged[-1] = (merged[-1][0] + width, merged[-1][1])
        else:
            merged.append((width, slope))

    width_sum = sum(w for w, _ in merged)
    if width_sum < total_width - 1e-8:
        slope = merged[-1][1] if merged else 0.0
        merged.append((total_width - width_sum, slope))
    elif width_sum > total_width + 1e-8 and merged:
        excess = width_sum - total_width
        w, s = merged[-1]
        merged[-1] = (max(0.0, w - excess), s)
        merged = [x for x in merged if x[0] > 1e-10]

    return merged


def prepare_instance(data):
    num_periods = int(data["num_time_periods"])
    period_hours = float(data.get("time_period_hours", 1))
    if period_hours <= 0:
        raise ValueError("time_period_hours must be positive.")

    demand = [float(x) for x in data["demand"]]
    reserve = [float(x) for x in data["reserve_requirement"]]
    if len(demand) != num_periods or len(reserve) != num_periods:
        raise ValueError("Demand/reserve vector length does not match num_time_periods.")

    generators = []
    for raw in data["generators"]:
        status = int(raw.get("initial_status", 0))
        p_min = float(raw["p_min"])
        p_max = float(raw["p_max"])
        min_up = max(0, int(raw["min_up_time"]))
        min_down = max(0, int(raw["min_down_time"]))

        tiers = []
        for tier in raw.get("startup_costs", []) or []:
            max_dt = tier.get("max_downtime_hours")
            tiers.append(
                {
                    "max": None if max_dt is None else float(max_dt),
                    "cost": float(tier["cost"]),
                }
            )
        tiers.sort(
            key=lambda x: (
                x["max"] is None,
                float("inf") if x["max"] is None else x["max"],
            )
        )
        if tiers and all(t["max"] is not None for t in tiers):
            tiers[-1]["max"] = None

        segments = normalize_segments(
            raw.get("production_cost_segments", []), p_min, p_max
        )
        convex = all(
            segments[i][1] >= segments[i - 1][1] - 1e-9
            for i in range(1, len(segments))
        )

        u_init = 1 if status > 0 else 0
        init_off = abs(status) if status <= 0 else 0
        rem_up = max(0, min_up - status) if status > 0 else 0
        rem_down = max(0, min_down - init_off) if status <= 0 else 0

        generators.append(
            {
                "id": str(raw["id"]),
                "p_min": p_min,
                "p_max": p_max,
                "ramp_up": float(raw["ramp_up_limit"]),
                "ramp_down": float(raw["ramp_down_limit"]),
                "startup_ramp": float(raw["startup_ramp_limit"]),
                "shutdown_ramp": float(raw["shutdown_ramp_limit"]),
                "min_up": min_up,
                "min_down": min_down,
                "no_load": float(raw["no_load_cost"]),
                "tiers": tiers,
                "segments": segments,
                "convex": convex,
                "initial_status": status,
                "initial_power": float(raw.get("initial_power", 0.0)),
                "u_init": u_init,
                "init_off": init_off,
                "rem_up": rem_up,
                "rem_down": rem_down,
            }
        )

    return generators, demand, reserve, period_hours


def production_cost_at(g, power):
    if power <= g["p_min"] + TOL:
        return 0.0
    remaining = min(power, g["p_max"]) - g["p_min"]
    cost = 0.0
    for width, slope in g["segments"]:
        take = min(width, remaining)
        if take > 0:
            cost += slope * take
            remaining -= take
        if remaining <= TOL:
            break
    return cost


def downtime_before_start(g, u, t):
    if t == 0:
        return g["init_off"]

    downtime = 0
    j = t - 1
    while j >= 0 and u[j] == 0:
        downtime += 1
        j -= 1
    if j < 0 and g["u_init"] == 0:
        downtime += g["init_off"]
    return downtime


def startup_tier_index(g, downtime_periods, period_hours):
    if not g["tiers"]:
        return -1
    downtime_hours = downtime_periods * period_hours
    for idx, tier in enumerate(g["tiers"]):
        if tier["max"] is None or downtime_hours <= tier["max"] + 1e-9:
            return idx
    return len(g["tiers"]) - 1


def startup_cost_for(g, downtime_periods, period_hours):
    idx = startup_tier_index(g, downtime_periods, period_hours)
    return 0.0 if idx < 0 else g["tiers"][idx]["cost"]


def make_solution(gens, u_values, p_values, period_hours):
    total = 0.0
    schedule = {}
    num_periods = len(u_values[0]) if u_values else 0

    for gi, g in enumerate(gens):
        u_clean = []
        p_clean = []
        pb_clean = []

        for t in range(num_periods):
            u_val = 1 if u_values[gi][t] >= 0.5 else 0
            p_val = float(p_values[gi][t])

            if u_val == 0:
                p_val = 0.0
                pb_val = 0.0
            else:
                if p_val < g["p_min"] and g["p_min"] - p_val <= 1e-5:
                    p_val = g["p_min"]
                if p_val > g["p_max"] and p_val - g["p_max"] <= 1e-5:
                    p_val = g["p_max"]
                if abs(p_val) < 1e-10:
                    p_val = 0.0
                pb_val = g["p_max"]

            u_clean.append(u_val)
            p_clean.append(float(p_val))
            pb_clean.append(float(pb_val))

            if u_val:
                total += g["no_load"]
                total += production_cost_at(g, p_val) * period_hours

        prev = g["u_init"]
        for t in range(num_periods):
            if u_clean[t] == 1 and prev == 0:
                downtime = downtime_before_start(g, u_clean, t)
                total += startup_cost_for(g, downtime, period_hours)
            prev = u_clean[t]

        schedule[g["id"]] = {
            "u": u_clean,
            "p": p_clean,
            "pb": pb_clean,
        }

    return float(total), {
        "objective_value": float(total),
        "schedule": schedule,
    }


def validate_schedule(gens, u, p, demand, reserve):
    tol = 2e-5
    num_periods = len(demand)

    for gi, g in enumerate(gens):
        for t in range(min(g["rem_up"], num_periods)):
            if g["u_init"] == 1 and u[gi][t] != 1:
                return False
        for t in range(min(g["rem_down"], num_periods)):
            if g["u_init"] == 0 and u[gi][t] != 0:
                return False

        prev_u = g["u_init"]
        prev_p = g["initial_power"]

        for t in range(num_periods):
            u_t = u[gi][t]
            p_t = p[gi][t]
            start = 1 if (u_t == 1 and prev_u == 0) else 0
            shutdown = 1 if (u_t == 0 and prev_u == 1) else 0

            if u_t:
                if p_t < g["p_min"] - tol or p_t > g["p_max"] + tol:
                    return False
            else:
                if abs(p_t) > tol:
                    return False

            if start and p_t > g["startup_ramp"] + tol:
                return False
            if t + 1 < num_periods and u_t == 1 and u[gi][t + 1] == 0:
                if p_t > g["shutdown_ramp"] + tol:
                    return False

            if p_t - prev_p > g["ramp_up"] * prev_u + g["startup_ramp"] * start + tol:
                return False
            if prev_p - p_t > g["ramp_down"] * u_t + g["shutdown_ramp"] * shutdown + tol:
                return False

            if start:
                for j in range(t, min(num_periods, t + g["min_up"])):
                    if u[gi][j] != 1:
                        return False
            if shutdown:
                for j in range(t, min(num_periods, t + g["min_down"])):
                    if u[gi][j] != 0:
                        return False

            prev_u = u_t
            prev_p = p_t

        if num_periods > 0 and g["u_init"] == 1 and u[gi][0] == 0:
            if g["initial_power"] > g["shutdown_ramp"] + tol:
                return False

    for t in range(num_periods):
        produced = sum(p[gi][t] for gi in range(len(gens)))
        scale = max(1.0, abs(demand[t]))
        if abs(produced - demand[t]) > 1e-4 * scale:
            return False

        available = sum(
            gens[gi]["p_max"] * u[gi][t] for gi in range(len(gens))
        )
        if available + 1e-4 * scale < demand[t] + reserve[t]:
            return False

    return True


def dispatch_for_commitment(gens, u, demand, mode):
    num_gens = len(gens)
    num_periods = len(demand)

    base = []
    for gi, g in enumerate(gens):
        row = []
        prev = g["u_init"]
        for t in range(num_periods):
            if u[gi][t] == 0:
                row.append((0.0, 0.0))
            else:
                lo = g["p_min"]
                hi = g["p_max"]
                if prev == 0:
                    hi = min(hi, g["startup_ramp"])
                if t + 1 < num_periods and u[gi][t + 1] == 0:
                    hi = min(hi, g["shutdown_ramp"])
                if lo > hi + 1e-7:
                    return None
                row.append((lo, hi))
            prev = u[gi][t]
        base.append(row)

        if num_periods > 0 and g["u_init"] == 1 and u[gi][0] == 0:
            if g["initial_power"] > g["shutdown_ramp"] + 1e-7:
                return None

    backward = [[(0.0, 0.0) for _ in range(num_periods)] for _ in range(num_gens)]
    for gi, g in enumerate(gens):
        for t in range(num_periods - 1, -1, -1):
            if u[gi][t] == 0:
                backward[gi][t] = (0.0, 0.0)
                continue
            lo, hi = base[gi][t]
            if t + 1 < num_periods and u[gi][t + 1] == 1:
                next_lo, next_hi = backward[gi][t + 1]
                lo = max(lo, next_lo - g["ramp_up"])
                hi = min(hi, next_hi + g["ramp_down"])
            if lo > hi + 1e-7:
                return None
            backward[gi][t] = (lo, hi)

    p_by_time = []

    for t in range(num_periods):
        lows = []
        highs = []

        for gi, g in enumerate(gens):
            if u[gi][t] == 0:
                lows.append(0.0)
                highs.append(0.0)
                continue

            lo, hi = backward[gi][t]
            if t == 0:
                if g["u_init"] == 1:
                    lo = max(lo, g["initial_power"] - g["ramp_down"])
                    hi = min(hi, g["initial_power"] + g["ramp_up"])
            elif u[gi][t - 1] == 1:
                prev_power = p_by_time[t - 1][gi]
                lo = max(lo, prev_power - g["ramp_down"])
                hi = min(hi, prev_power + g["ramp_up"])

            if lo > hi + 1e-6:
                return None
            lows.append(max(g["p_min"], lo))
            highs.append(min(g["p_max"], hi))

        total_low = sum(lows)
        total_high = sum(highs)
        if total_low > demand[t] + 1e-6 or total_high < demand[t] - 1e-6:
            return None

        remaining = demand[t] - total_low
        row = list(lows)

        if remaining < 0:
            if remaining < -1e-6:
                return None
            remaining = 0.0

        if mode == "proportional":
            spans = [max(0.0, highs[gi] - lows[gi]) for gi in range(num_gens)]
            span_total = sum(spans)
            if remaining > 1e-7:
                if span_total <= 1e-10:
                    return None
                for gi in range(num_gens):
                    row[gi] += spans[gi] * remaining / span_total
        else:
            chunks = []
            for gi, g in enumerate(gens):
                if u[gi][t] == 0 or highs[gi] <= lows[gi] + TOL:
                    continue
                cursor = g["p_min"]
                for width, slope in g["segments"]:
                    seg_lo = cursor
                    seg_hi = cursor + width
                    lo = max(lows[gi], seg_lo)
                    hi = min(highs[gi], seg_hi)
                    if hi > lo + TOL:
                        chunks.append((slope, gi, hi - lo))
                    cursor = seg_hi

            chunks.sort(key=lambda x: (x[0], x[1]))
            for _, gi, capacity in chunks:
                if remaining <= 1e-8:
                    break
                take = min(capacity, remaining)
                row[gi] += take
                remaining -= take

            if remaining > 1e-5:
                return None

        p_by_time.append(row)

    return [
        [p_by_time[t][gi] for t in range(num_periods)]
        for gi in range(num_gens)
    ]


def build_heuristic(gens, demand, reserve, period_hours):
    num_gens = len(gens)
    num_periods = len(demand)

    if num_periods == 0:
        u = [[] for _ in gens]
        p = [[] for _ in gens]
        return make_solution(gens, u, p, period_hours)

    requirement = [demand[t] + reserve[t] for t in range(num_periods)]

    locked = set()
    initial_on = set()
    for gi, g in enumerate(gens):
        if g["u_init"] == 1:
            initial_on.add(gi)
            if (
                g["rem_up"] > 0
                or g["initial_power"] > g["shutdown_ramp"] + 1e-7
            ):
                locked.add(gi)

    def active(gi, t):
        g = gens[gi]
        return g["u_init"] == 1 or t >= g["rem_down"]

    def capacity_ok(selected):
        for t in range(num_periods):
            cap = sum(
                gens[gi]["p_max"] for gi in selected if active(gi, t)
            )
            if cap + 1e-7 < requirement[t]:
                return False
        return True

    def minimum_ok(selected):
        for t in range(num_periods):
            minimum = sum(
                gens[gi]["p_min"] for gi in selected if active(gi, t)
            )
            if minimum > demand[t] + 1e-7:
                return False
        return True

    def make_commitment(selected):
        return [
            [
                1 if gi in selected and active(gi, t) else 0
                for t in range(num_periods)
            ]
            for gi in range(num_gens)
        ]

    avg_cost = []
    for g in gens:
        startup_est = min((x["cost"] for x in g["tiers"]), default=0.0)
        denom = max(g["p_max"], 1e-6)
        avg_cost.append(
            (
                g["no_load"]
                + production_cost_at(g, g["p_max"])
                + startup_est / max(1, num_periods)
            )
            / denom
        )

    def repair_subset(selected, order):
        selected = set(selected)
        for _ in range(2 * num_gens + 10):
            if minimum_ok(selected):
                return selected

            removable = [
                gi
                for gi in selected
                if gi not in locked and capacity_ok(selected - {gi})
            ]
            if removable:
                selected.remove(
                    max(
                        removable,
                        key=lambda gi: (
                            gens[gi]["p_min"],
                            avg_cost[gi],
                            gens[gi]["p_max"],
                        ),
                    )
                )
                continue

            swapped = False
            for add_gi in order:
                if add_gi in selected:
                    continue
                selected.add(add_gi)
                candidates = [
                    gi
                    for gi in selected
                    if gi != add_gi
                    and gi not in locked
                    and gens[gi]["p_min"] > gens[add_gi]["p_min"] + TOL
                    and capacity_ok(selected - {gi})
                ]
                if candidates:
                    selected.remove(
                        max(
                            candidates,
                            key=lambda gi: (
                                gens[gi]["p_min"],
                                avg_cost[gi],
                            ),
                        )
                    )
                    swapped = True
                    break
                selected.remove(add_gi)

            if not swapped:
                return None

        return selected if minimum_ok(selected) else None

    best = None

    def attempt(selected):
        nonlocal best
        selected = set(selected)
        if not capacity_ok(selected) or not minimum_ok(selected):
            return

        u = make_commitment(selected)
        for mode in ("cost", "proportional"):
            p = dispatch_for_commitment(gens, u, demand, mode)
            if p is None:
                continue
            obj, sol = make_solution(gens, u, p, period_hours)
            clean_u = [sol["schedule"][g["id"]]["u"] for g in gens]
            clean_p = [sol["schedule"][g["id"]]["p"] for g in gens]
            if not validate_schedule(gens, clean_u, clean_p, demand, reserve):
                continue
            if best is None or obj < best[0] - 1e-7:
                best = (obj, sol)

    orders = [
        sorted(range(num_gens), key=lambda gi: (avg_cost[gi], gens[gi]["p_min"])),
        sorted(range(num_gens), key=lambda gi: (gens[gi]["p_min"], avg_cost[gi])),
        sorted(
            range(num_gens),
            key=lambda gi: (
                -(gens[gi]["p_max"] - gens[gi]["p_min"])
                / (avg_cost[gi] + 1e-3),
                gens[gi]["p_min"],
            ),
        ),
        sorted(range(num_gens), key=lambda gi: (-gens[gi]["p_max"], avg_cost[gi])),
        sorted(
            range(num_gens),
            key=lambda gi: (
                gens[gi]["no_load"]
                + production_cost_at(gens[gi], gens[gi]["p_max"]),
                gens[gi]["p_min"],
            ),
        ),
    ]

    unique_orders = []
    seen_orders = set()
    for order in orders:
        key = tuple(order)
        if key not in seen_orders:
            seen_orders.add(key)
            unique_orders.append(order)

    base_options = [set(locked), set(locked) | initial_on]

    for base in base_options:
        for order in unique_orders:
            selected = set(base)
            for gi in order:
                if capacity_ok(selected):
                    break
                selected.add(gi)

            if not capacity_ok(selected):
                continue
            selected = repair_subset(selected, order)
            if selected is not None:
                attempt(selected)

    attempt(set(range(num_gens)))
    return best


def build_model(gens, demand, reserve, period_hours):
    num_gens = len(gens)
    num_periods = len(demand)

    model = gp.Model("thermal_unit_commitment")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0

    u_vars = []
    v_vars = []
    w_vars = []
    p_vars = []
    pb_vars = []
    q_vars = []
    y_vars = []
    z_vars = []

    for gi, g in enumerate(gens):
        u_row = [
            model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            for _ in range(num_periods)
        ]
        v_row = [
            model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            for _ in range(num_periods)
        ]
        w_row = [
            model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            for _ in range(num_periods)
        ]
        p_row = [
            model.addVar(lb=0.0, ub=max(0.0, g["p_max"]))
            for _ in range(num_periods)
        ]
        pb_row = [
            model.addVar(lb=0.0, ub=max(0.0, g["p_max"]))
            for _ in range(num_periods)
        ]

        for t in range(num_periods):
            u_row[t].BranchPriority = 20
            v_row[t].BranchPriority = 10
            w_row[t].BranchPriority = 10

        q_row = []
        y_row = []
        for _ in range(num_periods):
            q_row.append(
                [
                    model.addVar(lb=0.0, ub=width)
                    for width, _ in g["segments"]
                ]
            )
            if not g["convex"] and g["segments"]:
                y_row.append(
                    [
                        model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                        for _ in g["segments"]
                    ]
                )
            else:
                y_row.append([])

        z_row = []
        for _ in range(num_periods):
            z_row.append(
                [
                    model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                    for _ in g["tiers"]
                ]
            )

        if num_periods > 0:
            if g["rem_up"] > 0:
                for t in range(min(g["rem_up"], num_periods)):
                    u_row[t].LB = 1.0
                    u_row[t].UB = 1.0
            if g["rem_down"] > 0:
                for t in range(min(g["rem_down"], num_periods)):
                    u_row[t].LB = 0.0
                    u_row[t].UB = 0.0

        u_vars.append(u_row)
        v_vars.append(v_row)
        w_vars.append(w_row)
        p_vars.append(p_row)
        pb_vars.append(pb_row)
        q_vars.append(q_row)
        y_vars.append(y_row)
        z_vars.append(z_row)

    objective = gp.quicksum(
        g["no_load"] * u_vars[gi][t]
        for gi, g in enumerate(gens)
        for t in range(num_periods)
    )
    objective += gp.quicksum(
        period_hours * slope * q_vars[gi][t][k]
        for gi, g in enumerate(gens)
        for t in range(num_periods)
        for k, (_, slope) in enumerate(g["segments"])
    )
    objective += gp.quicksum(
        tier["cost"] * z_vars[gi][t][k]
        for gi, g in enumerate(gens)
        for t in range(num_periods)
        for k, tier in enumerate(g["tiers"])
    )
    model.setObjective(objective, GRB.MINIMIZE)

    for gi, g in enumerate(gens):
        for t in range(num_periods):
            u_t = u_vars[gi][t]
            p_t = p_vars[gi][t]
            pb_t = pb_vars[gi][t]

            model.addConstr(p_t >= g["p_min"] * u_t)
            model.addConstr(p_t <= g["p_max"] * u_t)
            model.addConstr(pb_t >= p_t)
            model.addConstr(pb_t <= g["p_max"] * u_t)

            model.addConstr(
                p_t
                == g["p_min"] * u_t
                + gp.quicksum(q_vars[gi][t][k] for k in range(len(g["segments"])))
            )

            if g["convex"]:
                for k, (width, _) in enumerate(g["segments"]):
                    model.addConstr(q_vars[gi][t][k] <= width * u_t)
            elif g["segments"]:
                y_row = y_vars[gi][t]
                model.addConstr(gp.quicksum(y_row) == u_t)
                for k, (width, _) in enumerate(g["segments"]):
                    current_or_later = gp.quicksum(
                        y_row[j] for j in range(k, len(y_row))
                    )
                    model.addConstr(q_vars[gi][t][k] <= width * current_or_later)
                    if k + 1 < len(y_row):
                        later = gp.quicksum(
                            y_row[j] for j in range(k + 1, len(y_row))
                        )
                        model.addConstr(q_vars[gi][t][k] >= width * later)

    for t in range(num_periods):
        model.addConstr(
            gp.quicksum(p_vars[gi][t] for gi in range(num_gens)) == demand[t]
        )
        model.addConstr(
            gp.quicksum(pb_vars[gi][t] for gi in range(num_gens))
            >= demand[t] + reserve[t]
        )
        model.addConstr(
            gp.quicksum(
                gens[gi]["p_max"] * u_vars[gi][t] for gi in range(num_gens)
            )
            >= demand[t] + reserve[t]
        )
        model.addConstr(
            gp.quicksum(
                gens[gi]["p_min"] * u_vars[gi][t] for gi in range(num_gens)
            )
            <= demand[t]
        )

    for gi, g in enumerate(gens):
        if num_periods > 0:
            model.addConstr(
                v_vars[gi][0] - w_vars[gi][0]
                == u_vars[gi][0] - g["u_init"]
            )
            model.addConstr(
                p_vars[gi][0] - g["initial_power"]
                <= g["ramp_up"] * g["u_init"]
                + g["startup_ramp"] * v_vars[gi][0]
            )
            model.addConstr(
                g["initial_power"] - p_vars[gi][0]
                <= g["ramp_down"] * u_vars[gi][0]
                + g["shutdown_ramp"] * w_vars[gi][0]
            )

        for t in range(1, num_periods):
            model.addConstr(
                v_vars[gi][t] - w_vars[gi][t]
                == u_vars[gi][t] - u_vars[gi][t - 1]
            )
            model.addConstr(
                p_vars[gi][t] - p_vars[gi][t - 1]
                <= g["ramp_up"] * u_vars[gi][t - 1]
                + g["startup_ramp"] * v_vars[gi][t]
            )
            model.addConstr(
                p_vars[gi][t - 1] - p_vars[gi][t]
                <= g["ramp_down"] * u_vars[gi][t]
                + g["shutdown_ramp"] * w_vars[gi][t]
            )

        for t in range(num_periods):
            if g["min_up"] > 0:
                for j in range(t, min(num_periods, t + g["min_up"])):
                    model.addConstr(v_vars[gi][t] <= u_vars[gi][j])
                first = max(0, t - g["min_up"] + 1)
                model.addConstr(
                    gp.quicksum(v_vars[gi][j] for j in range(first, t + 1))
                    <= u_vars[gi][t]
                )

            if g["min_down"] > 0:
                for j in range(t, min(num_periods, t + g["min_down"])):
                    model.addConstr(w_vars[gi][t] <= 1 - u_vars[gi][j])
                first = max(0, t - g["min_down"] + 1)
                model.addConstr(
                    gp.quicksum(w_vars[gi][j] for j in range(first, t + 1))
                    <= 1 - u_vars[gi][t]
                )

        if g["tiers"]:
            for t in range(num_periods):
                model.addConstr(
                    gp.quicksum(z_vars[gi][t][k] for k in range(len(g["tiers"])))
                    == v_vars[gi][t]
                )

                for k, tier in enumerate(g["tiers"]):
                    z = z_vars[gi][t][k]

                    if k > 0:
                        previous_max = g["tiers"][k - 1]["max"]
                        if previous_max is None:
                            z.UB = 0.0
                        else:
                            lower_periods = (
                                int(math.floor(previous_max / period_hours + 1e-9))
                                + 1
                            )
                            max_possible = (
                                t if g["u_init"] == 1 else t + g["init_off"]
                            )
                            if max_possible < lower_periods:
                                z.UB = 0.0
                            else:
                                start_idx = (
                                    t - lower_periods
                                    if t >= lower_periods
                                    else 0
                                )
                                window_len = t - start_idx
                                if window_len > 0:
                                    model.addConstr(
                                        gp.quicksum(
                                            u_vars[gi][j]
                                            for j in range(start_idx, t)
                                        )
                                        <= window_len * (1 - z)
                                    )

                    if tier["max"] is not None:
                        upper_periods = int(
                            math.floor(tier["max"] / period_hours + 1e-9)
                        )
                        if g["u_init"] == 1:
                            automatic = t <= upper_periods
                        else:
                            automatic = t + g["init_off"] <= upper_periods

                        if not automatic:
                            start_idx = max(0, t - upper_periods)
                            if start_idx >= t:
                                model.addConstr(z <= 0)
                            else:
                                model.addConstr(
                                    z
                                    <= gp.quicksum(
                                        u_vars[gi][j]
                                        for j in range(start_idx, t)
                                    )
                                )

    return {
        "model": model,
        "u": u_vars,
        "v": v_vars,
        "w": w_vars,
        "p": p_vars,
        "pb": pb_vars,
        "q": q_vars,
        "y": y_vars,
        "z": z_vars,
    }


def set_complete_mip_start(ctx, gens, solution, period_hours):
    model = ctx["model"]
    model.update()

    num_periods = len(next(iter(solution["schedule"].values()))["u"]) if solution["schedule"] else 0

    for gi, g in enumerate(gens):
        sched = solution["schedule"][g["id"]]
        u = sched["u"]
        p = sched["p"]

        prev = g["u_init"]
        for t in range(num_periods):
            u_val = int(u[t])
            p_val = float(p[t])
            start = 1 if u_val == 1 and prev == 0 else 0
            shutdown = 1 if u_val == 0 and prev == 1 else 0

            ctx["u"][gi][t].Start = u_val
            ctx["v"][gi][t].Start = start
            ctx["w"][gi][t].Start = shutdown
            ctx["p"][gi][t].Start = p_val
            ctx["pb"][gi][t].Start = g["p_max"] if u_val else 0.0

            for k in range(len(g["tiers"])):
                ctx["z"][gi][t][k].Start = 0.0
            if start and g["tiers"]:
                downtime = downtime_before_start(g, u, t)
                tier_idx = startup_tier_index(g, downtime, period_hours)
                if tier_idx >= 0:
                    ctx["z"][gi][t][tier_idx].Start = 1.0

            remaining = max(0.0, p_val - g["p_min"]) if u_val else 0.0
            q_values = []
            for width, _ in g["segments"]:
                take = min(width, remaining)
                q_values.append(take)
                remaining -= take
            for k, value in enumerate(q_values):
                ctx["q"][gi][t][k].Start = value

            if not g["convex"] and g["segments"]:
                active = 0
                if u_val:
                    cursor = g["p_min"]
                    for k, (width, _) in enumerate(g["segments"]):
                        cursor += width
                        if p_val <= cursor + 1e-7:
                            active = k
                            break
                    else:
                        active = len(g["segments"]) - 1
                for k in range(len(g["segments"])):
                    ctx["y"][gi][t][k].Start = 1.0 if (u_val and k == active) else 0.0

            prev = u_val


def set_partial_mip_start(ctx, gens, period_hours):
    model = ctx["model"]
    model.update()

    num_periods = len(ctx["u"][0]) if ctx["u"] else 0

    for gi, g in enumerate(gens):
        u = [
            1 if (g["u_init"] == 1 or t >= g["rem_down"]) else 0
            for t in range(num_periods)
        ]
        prev = g["u_init"]
        for t in range(num_periods):
            start = 1 if u[t] == 1 and prev == 0 else 0
            shutdown = 1 if u[t] == 0 and prev == 1 else 0

            ctx["u"][gi][t].Start = u[t]
            ctx["v"][gi][t].Start = start
            ctx["w"][gi][t].Start = shutdown

            for k in range(len(g["tiers"])):
                ctx["z"][gi][t][k].Start = 0.0
            if start and g["tiers"]:
                downtime = downtime_before_start(g, u, t)
                tier_idx = startup_tier_index(g, downtime, period_hours)
                if tier_idx >= 0:
                    ctx["z"][gi][t][tier_idx].Start = 1.0

            prev = u[t]


def solution_from_flat(gens, values, num_gens, num_periods, period_hours):
    block = num_gens * num_periods
    u_vals = values[:block]
    p_vals = values[block : 2 * block]

    u = []
    p = []
    idx = 0
    for _ in range(num_gens):
        u.append(u_vals[idx : idx + num_periods])
        p.append(p_vals[idx : idx + num_periods])
        idx += num_periods

    return make_solution(gens, u, p, period_hours)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", required=False, default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    wall_start = time.monotonic()
    deadline = wall_start + max(0.0, float(args.time_limit) - 0.05)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    gens, demand, reserve, period_hours = prepare_instance(instance)
    num_gens = len(gens)
    num_periods = len(demand)

    best_obj = float("inf")
    best_solution = None
    last_logged = None

    try:
        heuristic = build_heuristic(gens, demand, reserve, period_hours)
        if heuristic is not None:
            best_obj, best_solution = heuristic
            if logger:
                logger.log_solution(best_obj, best_solution)
            last_logged = best_obj
    except Exception:
        pass

    ctx = None
    output_vars = []

    if time.monotonic() < deadline:
        try:
            ctx = build_model(gens, demand, reserve, period_hours)
            if best_solution is not None:
                set_complete_mip_start(ctx, gens, best_solution, period_hours)
            else:
                set_partial_mip_start(ctx, gens, period_hours)

            flat_u = [var for row in ctx["u"] for var in row]
            flat_p = [var for row in ctx["p"] for var in row]
            flat_pb = [var for row in ctx["pb"] for var in row]
            output_vars = flat_u + flat_p + flat_pb
        except Exception:
            if best_solution is None:
                raise
            ctx = None

    if ctx is not None:
        model = ctx["model"]

        def improved(new_obj, old_obj):
            if old_obj == float("inf"):
                return True
            return new_obj < old_obj - 1e-7 * max(1.0, abs(old_obj))

        def gurobi_callback(cb_model, where):
            nonlocal best_obj, best_solution, last_logged

            if time.monotonic() >= deadline:
                cb_model.terminate()
                return

            if where == GRB.Callback.MIPSOL:
                try:
                    values = cb_model.cbGetSolution(output_vars)
                    obj, sol = solution_from_flat(
                        gens,
                        values,
                        num_gens,
                        num_periods,
                        period_hours,
                    )
                    if improved(obj, best_obj):
                        best_obj = obj
                        best_solution = sol
                        if logger:
                            logger.log_solution(obj, sol)
                        last_logged = obj
                except Exception:
                    if logger:
                        try:
                            logger.log(
                                float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                            )
                        except Exception:
                            pass

        try:
            remaining = deadline - time.monotonic()
            if remaining > 0.05:
                if remaining > 4.0:
                    first_budget = min(10.0, max(0.2, 0.20 * remaining))
                    first_budget = min(first_budget, max(0.05, remaining - 0.02))
                    model.Params.MIPFocus = 1
                    model.Params.Heuristics = 0.25
                    model.Params.SolutionLimit = 1
                    model.Params.TimeLimit = max(0.05, first_budget)
                    model.optimize(gurobi_callback)

                terminal_statuses = {
                    GRB.OPTIMAL,
                    GRB.INFEASIBLE,
                    GRB.INF_OR_UNBOUND,
                }
                remaining = deadline - time.monotonic()
                if model.Status not in terminal_statuses and remaining > 0.05:
                    model.Params.SolutionLimit = 2000000000
                    if best_solution is not None:
                        model.Params.MIPFocus = 0
                        model.Params.Heuristics = 0.08
                    else:
                        model.Params.MIPFocus = 1
                        model.Params.Heuristics = 0.20
                    model.Params.TimeLimit = max(0.05, remaining - 0.02)
                    model.optimize(gurobi_callback)

        except gp.GurobiError:
            if best_solution is None:
                raise

        try:
            if model.SolCount > 0:
                values = [var.X for var in output_vars]
                obj, sol = solution_from_flat(
                    gens,
                    values,
                    num_gens,
                    num_periods,
                    period_hours,
                )
                if obj < best_obj - 1e-7 * max(1.0, abs(best_obj)):
                    best_obj = obj
                    best_solution = sol
        except Exception:
            pass

    if best_solution is None:
        raise RuntimeError("No feasible solution was found within the time limit.")

    if logger and (
        last_logged is None
        or best_obj < last_logged - 1e-7 * max(1.0, abs(last_logged))
    ):
        logger.log_solution(best_obj, best_solution)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()