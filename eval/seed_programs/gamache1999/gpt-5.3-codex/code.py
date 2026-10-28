import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def intervals_overlap(s1, e1, s2, e2):
    return (s1 < e2) and (s2 < e1)


def build_and_solve(instance, time_limit, logger=None):
    t0 = time.time()

    D = int(instance["planning_horizon_days"])
    crew_list = instance["crew_members"]
    pairing_list = instance["pairings"]

    num_crew = len(crew_list)
    num_pairings = len(pairing_list)

    wr = instance["work_rules"]
    monthly_flight_limit = float(wr["monthly_flight_limit_hours"])
    rolling_limit = float(wr["rolling_flight_limit_hours"])
    max_cross = int(wr["max_cross_pairings_per_month"])
    max_consec_early = int(wr["max_consecutive_early_departures_before_8am"])  # usually 2

    # Pairing indexed data
    p_id = []
    p_start_day = []
    p_dep_hour = []
    p_duration_min = []
    p_flight_hours = []
    p_aircraft = []
    p_cross = []
    p_req = []
    p_start_abs = []
    p_end_abs = []
    p_early = []

    for p in pairing_list:
        p_id.append(int(p["id"]))
        sd = int(p["start_day"])
        dh = float(p["departure_hour"])
        durm = int(p["duration_minutes"])
        p_start_day.append(sd)
        p_dep_hour.append(dh)
        p_duration_min.append(durm)
        p_flight_hours.append(float(p["flight_hours"]))
        p_aircraft.append(int(p["aircraft_type"]))
        p_cross.append(bool(p["is_cross_pairing"]))
        p_req.append(int(p["coverage_requirement"]))

        sabs = sd * 24.0 + dh
        eabs = sabs + durm / 60.0
        p_start_abs.append(sabs)
        p_end_abs.append(eabs)
        p_early.append(dh < 8.0)

    # Crew indexed data
    c_id = []
    c_quals = []
    c_prev15 = []
    c_preassigned_intervals = []
    c_preassigned_minutes = []
    c_reserve_forbid_next_early = []

    for c in crew_list:
        c_id.append(int(c["id"]))
        quals = set(int(q) for q in c.get("qualifications", []))
        c_quals.append(quals)
        c_prev15.append(float(c.get("flight_hours_previous_15_days", 0.0)))

        intervals = []
        pre_minutes = 0
        forbid_days = set()

        for a in c.get("pre_assigned_activities", []):
            sday = int(a["start_day"])
            eday = int(a["end_day"])
            adur = int(a.get("duration_days", eday - sday + 1))
            atype = str(a.get("type", "")).lower()

            astart = sday * 24.0
            aend = (eday + 1) * 24.0
            intervals.append((astart, aend))
            pre_minutes += adur * 1440

            if "reserve" in atype:
                forbid_days.add(eday + 1)

        c_preassigned_intervals.append(intervals)
        c_preassigned_minutes.append(pre_minutes)
        c_reserve_forbid_next_early.append(forbid_days)

    # Feasible assignment arcs
    cand_by_crew = [[] for _ in range(num_crew)]
    cand_by_pair = [[] for _ in range(num_pairings)]

    for ci in range(num_crew):
        quals = c_quals[ci]
        intervals = c_preassigned_intervals[ci]
        forbid_days = c_reserve_forbid_next_early[ci]

        for pi in range(num_pairings):
            if p_aircraft[pi] not in quals:
                continue

            # Reserve-following rule approximation:
            # if next day after reserve starts pairing, require departure >= 12
            if p_start_day[pi] in forbid_days and p_dep_hour[pi] < 12.0:
                continue

            # No overlap with pre-assigned activities (treated as full-day blocks)
            ok = True
            ps = p_start_abs[pi]
            pe = p_end_abs[pi]
            for astart, aend in intervals:
                if intervals_overlap(ps, pe, astart, aend):
                    ok = False
                    break
            if not ok:
                continue

            cand_by_crew[ci].append(pi)
            cand_by_pair[pi].append(ci)

    # Build model
    model = gp.Model("crew_pairing_cover")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    # Variables x(ci,pi) only on feasible arcs
    x = {}
    for ci in range(num_crew):
        for pi in cand_by_crew[ci]:
            x[(ci, pi)] = model.addVar(vtype=GRB.BINARY, name=f"x_{ci}_{pi}")

    u = {}
    for pi in range(num_pairings):
        u[pi] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=p_req[pi], name=f"u_{pi}")

    model.update()

    # Coverage constraints
    for pi in range(num_pairings):
        expr = gp.quicksum(x[(ci, pi)] for ci in cand_by_pair[pi] if (ci, pi) in x)
        model.addConstr(expr + u[pi] == p_req[pi], name=f"cover_{pi}")

    # Crew constraints
    for ci in range(num_crew):
        vars_ci = [x[(ci, pi)] for pi in cand_by_crew[ci]]

        if vars_ci:
            # Monthly flight limit
            model.addConstr(
                gp.quicksum(p_flight_hours[pi] * x[(ci, pi)] for pi in cand_by_crew[ci]) <= monthly_flight_limit,
                name=f"month_fh_{ci}",
            )

            # Rolling limit for days 0..14 (approx using start day)
            rhs = max(0.0, rolling_limit - c_prev15[ci])
            model.addConstr(
                gp.quicksum(
                    p_flight_hours[pi] * x[(ci, pi)]
                    for pi in cand_by_crew[ci]
                    if p_start_day[pi] < 15
                ) <= rhs,
                name=f"roll_fh_{ci}",
            )

            # Max cross pairings
            model.addConstr(
                gp.quicksum(x[(ci, pi)] for pi in cand_by_crew[ci] if p_cross[pi]) <= max_cross,
                name=f"cross_{ci}",
            )

            # Early departure consecutive-days rule approximation:
            # in any 3-day window, at most 2 early departures (for max_consec_early=2)
            if max_consec_early >= 0 and D >= max_consec_early + 1:
                window = max_consec_early + 1
                for d in range(0, D - window + 1):
                    rel = [
                        pi for pi in cand_by_crew[ci]
                        if p_early[pi] and d <= p_start_day[pi] <= d + window - 1
                    ]
                    if rel:
                        model.addConstr(
                            gp.quicksum(x[(ci, pi)] for pi in rel) <= max_consec_early,
                            name=f"early_{ci}_{d}",
                        )

            # No-overlap between assigned pairings for same crew
            plist = sorted(cand_by_crew[ci], key=lambda pi: p_start_abs[pi])
            n = len(plist)
            for i in range(n):
                pi = plist[i]
                end_i = p_end_abs[pi]
                j = i + 1
                while j < n and p_start_abs[plist[j]] < end_i:
                    pj = plist[j]
                    if intervals_overlap(p_start_abs[pi], p_end_abs[pi], p_start_abs[pj], p_end_abs[pj]):
                        model.addConstr(x[(ci, pi)] + x[(ci, pj)] <= 1, name=f"ov_{ci}_{pi}_{pj}")
                    j += 1

    # Objective: minimize uncovered pairing duration
    model.setObjective(gp.quicksum(p_duration_min[pi] * u[pi] for pi in range(num_pairings)), GRB.MINIMIZE)

    # Greedy MIP start
    assigned = [set() for _ in range(num_crew)]
    intervals_ci = [[] for _ in range(num_crew)]
    month_hours = [0.0] * num_crew
    roll_hours = [0.0] * num_crew
    cross_cnt = [0] * num_crew
    early_days = [set() for _ in range(num_crew)]
    rem = p_req[:]

    order = list(range(num_pairings))
    order.sort(key=lambda pi: (len(cand_by_pair[pi]) / max(1, p_req[pi]), -p_duration_min[pi]))

    def can_assign(ci, pi):
        # overlap with existing assigned pairings
        s, e = p_start_abs[pi], p_end_abs[pi]
        for (s2, e2) in intervals_ci[ci]:
            if intervals_overlap(s, e, s2, e2):
                return False

        # flight limits
        fh = p_flight_hours[pi]
        if month_hours[ci] + fh > monthly_flight_limit + 1e-9:
            return False
        if p_start_day[pi] < 15:
            rhs = max(0.0, rolling_limit - c_prev15[ci])
            if roll_hours[ci] + fh > rhs + 1e-9:
                return False

        # cross limit
        if p_cross[pi] and cross_cnt[ci] + 1 > max_cross:
            return False

        # early consecutive rule (3-day windows)
        if p_early[pi]:
            d = p_start_day[pi]
            for w in range(d - max_consec_early, d + 1):
                cnt = 1  # this assignment
                for dd in range(w, w + max_consec_early + 1):
                    if dd == d:
                        continue
                    if dd in early_days[ci]:
                        cnt += 1
                if cnt > max_consec_early:
                    return False

        return True

    for pi in order:
        while rem[pi] > 0:
            best_ci = None
            best_key = None
            for ci in cand_by_pair[pi]:
                if can_assign(ci, pi):
                    key = (month_hours[ci], len(cand_by_crew[ci]))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_ci = ci
            if best_ci is None:
                break

            ci = best_ci
            assigned[ci].add(pi)
            intervals_ci[ci].append((p_start_abs[pi], p_end_abs[pi]))
            month_hours[ci] += p_flight_hours[pi]
            if p_start_day[pi] < 15:
                roll_hours[ci] += p_flight_hours[pi]
            if p_cross[pi]:
                cross_cnt[ci] += 1
            if p_early[pi]:
                early_days[ci].add(p_start_day[pi])

            rem[pi] -= 1

    # Set starts
    for (ci, pi), var in x.items():
        var.Start = 1.0 if pi in assigned[ci] else 0.0
    for pi in range(num_pairings):
        varu = rem[pi]
        if varu < 0:
            varu = 0
        u[pi].Start = float(varu)

    greedy_obj = float(sum(p_duration_min[pi] * rem[pi] for pi in range(num_pairings)))
    if logger:
        logger.log(greedy_obj)

    # Optimize with remaining time
    elapsed = time.time() - t0
    remaining = max(1, int(time_limit - elapsed))
    model.Params.TimeLimit = remaining

    best_logged = {"v": float("inf")}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
            if obj + 1e-9 < best_logged["v"]:
                best_logged["v"] = obj
                if logger:
                    logger.log(obj)

    model.optimize(cb)

    # Extract solution (MIP incumbent if available, else greedy)
    final_assign = [set() for _ in range(num_crew)]
    uncovered = [p_req[pi] for pi in range(num_pairings)]

    if model.SolCount > 0:
        for (ci, pi), var in x.items():
            if var.X > 0.5:
                final_assign[ci].add(pi)
        for pi in range(num_pairings):
            covered = sum(1 for ci in cand_by_pair[pi] if (ci, pi) in x and x[(ci, pi)].X > 0.5)
            uncovered[pi] = max(0, p_req[pi] - covered)
    else:
        final_assign = assigned
        uncovered = rem

    objective_value = float(sum(p_duration_min[pi] * uncovered[pi] for pi in range(num_pairings)))

    # Build output
    horizon_minutes = D * 1440
    rosters = {}
    for ci in range(num_crew):
        plist = sorted(list(final_assign[ci]), key=lambda pi: (p_start_day[pi], p_dep_hour[pi], p_id[pi]))
        pair_ids = [p_id[pi] for pi in plist]
        productive_pair_minutes = sum(p_duration_min[pi] for pi in plist)
        unprod = horizon_minutes - c_preassigned_minutes[ci] - productive_pair_minutes
        if unprod < 0:
            unprod = 0

        rosters[str(c_id[ci])] = {
            "roster_index": 0,
            "pairings": pair_ids,
            "unproductive_time": int(unprod),
        }

    uncovered_list = []
    for pi in range(num_pairings):
        if uncovered[pi] > 0:
            uncovered_list.append({
                "pairing_id": p_id[pi],
                "uncovered_count": int(uncovered[pi]),
                "duration_minutes": int(p_duration_min[pi]),
            })

    solution = {
        "objective_value": objective_value,
        "rosters": rosters,
        "uncovered_pairings": uncovered_list,
    }
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()