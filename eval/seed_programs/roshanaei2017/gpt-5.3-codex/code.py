import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def safe_log_solution(logger, objective_value, solution_dict=None):
    """Compatibility wrapper: some environments expose only logger.log()."""
    if logger is None:
        return
    if solution_dict is not None and hasattr(logger, "log_solution"):
        logger.log_solution(objective_value, solution_dict)
    else:
        logger.log(objective_value)


def detect_days(instance):
    num_days = int(instance["num_days"])
    seen = set()

    for s in instance.get("surgeons", []):
        for d in s.get("operating_days", []):
            seen.add(int(d))
        for d in s.get("availability_by_day", {}).keys():
            seen.add(int(d))

    for h in instance.get("hospitals", []):
        for orr in h.get("ORs", []):
            for d in orr.get("daily", {}).keys():
                seen.add(int(d))

    if not seen:
        return list(range(1, num_days + 1))

    d1 = set(range(1, num_days + 1))
    d0 = set(range(0, num_days))
    if seen.issubset(d1):
        return sorted(d1)
    if seen.issubset(d0):
        return sorted(d0)
    return sorted(seen)


def get_or_day_params(or_data, day):
    day_info = or_data.get("daily", {}).get(str(day), {})
    reg = int(day_info.get("regular_time", or_data.get("regular_time", 0)))
    mot = int(day_info.get("max_overtime", or_data.get("max_overtime", 0)))
    return reg, mot


def build_empty_solution(obj=1e18):
    return {
        "objective_value": float(obj),
        "assignments": [],
        "opened_ors": [],
        "surgeon_assignments": [],
        "or_sequences": [],
        "surgeon_sequences": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_all = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    alpha = float(inst["alpha"])
    days = detect_days(inst)

    # ---- Parse data ----
    patients = {}
    for p in inst["patients"]:
        pid = int(p["patient_id"])
        patients[pid] = {
            "due_date": int(p["due_date"]),
            "mandatory": bool(p["is_mandatory"]),
            "prep": float(p["preparation_time"]),
            "clean": float(p["cleaning_time"]),
            "eligible_surgeons": set(int(x) for x in p.get("eligible_surgeons", [])),
            "eligible_ORs_by_hospital": {
                int(h): set(int(r) for r in rs)
                for h, rs in p.get("eligible_ORs_by_hospital", {}).items()
            },
            "surgeon_times": {int(s): float(v) for s, v in p.get("surgeon_specific_times", {}).items()},
            "reward": float(p.get("reward", 0.0)),
        }

    surgeons = {}
    for s in inst["surgeons"]:
        sid = int(s["surgeon_id"])
        surgeons[sid] = {
            "operating_days": set(int(d) for d in s.get("operating_days", [])),
            "availability": {int(d): float(v) for d, v in s.get("availability_by_day", {}).items()},
        }

    hospitals = {}
    hospital_ids = []
    surgeon_fixed_cost = {}  # (s,h)->cost
    or_regular = {}          # (h,d,r)->minutes
    or_maxot = {}            # (h,d,r)->minutes
    or_fixed_open_cost = {}  # (h,d,r)->cost/day
    or_ot_cost_hour = {}     # (h,d,r)->cost/hour
    ors_by_hospital = defaultdict(list)

    for h in inst["hospitals"]:
        hid = int(h["hospital_id"])
        hospital_ids.append(hid)
        hospitals[hid] = h

        sfix = {int(k): float(v) for k, v in h.get("surgeon_fixed_costs_per_day", {}).items()}
        for s, c in sfix.items():
            surgeon_fixed_cost[(s, hid)] = c

        for orr in h.get("ORs", []):
            rid = int(orr["or_id"])
            ors_by_hospital[hid].append(rid)

            fixed_hour = float(orr.get("fixed_cost_per_hour", 0.0))
            ot_hour = float(orr.get("overtime_cost_per_hour", 0.0))
            for d in days:
                reg, mot = get_or_day_params(orr, d)
                or_regular[(hid, d, rid)] = float(reg)
                or_maxot[(hid, d, rid)] = float(mot)
                or_fixed_open_cost[(hid, d, rid)] = fixed_hour * (reg / 60.0)
                or_ot_cost_hour[(hid, d, rid)] = ot_hour

    patient_ids = sorted(patients.keys())
    surgeon_ids = sorted(surgeons.keys())
    hospital_ids = sorted(hospital_ids)

    # ---- Build feasible x tuples ----
    # x[p,s,h,d,r]
    x_keys = []
    by_patient = defaultdict(list)
    by_shd = defaultdict(list)   # (s,h,d)->x-keys
    by_hdr = defaultdict(list)   # (h,d,r)->x-keys
    feasible_for_patient = defaultdict(int)

    for p in patient_ids:
        pdata = patients[p]
        for s in pdata["eligible_surgeons"]:
            if s not in surgeons:
                continue
            if s not in pdata["surgeon_times"]:
                continue
            sdata = surgeons[s]

            for d in days:
                if d not in sdata["operating_days"]:
                    continue
                if float(sdata["availability"].get(d, 0.0)) <= 0:
                    continue
                if pdata["mandatory"] and d > pdata["due_date"]:
                    continue

                for h in hospital_ids:
                    elig_ors = pdata["eligible_ORs_by_hospital"].get(h, set())
                    if not elig_ors:
                        continue
                    for r in elig_ors:
                        if r not in ors_by_hospital[h]:
                            continue
                        k = (p, s, h, d, r)
                        x_keys.append(k)
                        by_patient[p].append(k)
                        by_shd[(s, h, d)].append(k)
                        by_hdr[(h, d, r)].append(k)
                        feasible_for_patient[p] += 1

    # Hard infeasibility check: mandatory unschedulable
    if any(patients[p]["mandatory"] and feasible_for_patient[p] == 0 for p in patient_ids):
        sol = build_empty_solution(1e18)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # ---- Build MIP (compact) ----
    m = gp.Model("hospital_or_scheduling_compact")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    elapsed = time.time() - start_all
    remain = max(1, int(args.time_limit - elapsed))
    m.Params.TimeLimit = remain

    x = {k: m.addVar(vtype=GRB.BINARY, name=f"x_{k[0]}_{k[1]}_{k[2]}_{k[3]}_{k[4]}") for k in x_keys}
    a = {p: m.addVar(vtype=GRB.BINARY, name=f"a_{p}") for p in patient_ids}

    y = {}
    ot = {}
    for h in hospital_ids:
        for d in days:
            for r in ors_by_hospital[h]:
                y[(h, d, r)] = m.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{r}")
                ot[(h, d, r)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ot_{h}_{d}_{r}")

    z = {}
    for s in surgeon_ids:
        for h in hospital_ids:
            for d in days:
                z[(s, h, d)] = m.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}")

    m.update()

    # Assignment constraints
    for p in patient_ids:
        expr = gp.quicksum(x[k] for k in by_patient[p])
        m.addConstr(a[p] == expr, name=f"assign_link_{p}")
        if patients[p]["mandatory"]:
            m.addConstr(expr == 1, name=f"mand_{p}")
        else:
            m.addConstr(expr <= 1, name=f"opt_{p}")

    # Link to OR open and surgeon assignment
    for (p, s, h, d, r), var in x.items():
        m.addConstr(var <= y[(h, d, r)], name=f"link_y_{p}_{s}_{h}_{d}_{r}")
        m.addConstr(var <= z[(s, h, d)], name=f"link_z_{p}_{s}_{h}_{d}_{r}")

    # Surgeon at most one hospital/day + off-day zero
    for s in surgeon_ids:
        sdata = surgeons[s]
        for d in days:
            avail = float(sdata["availability"].get(d, 0.0))
            if d in sdata["operating_days"] and avail > 0:
                m.addConstr(gp.quicksum(z[(s, h, d)] for h in hospital_ids) <= 1, name=f"oneh_{s}_{d}")
            else:
                for h in hospital_ids:
                    m.addConstr(z[(s, h, d)] == 0, name=f"off_{s}_{h}_{d}")

    # OR capacity and overtime
    for h in hospital_ids:
        for d in days:
            for r in ors_by_hospital[h]:
                reg = or_regular[(h, d, r)]
                mot = or_maxot[(h, d, r)]

                load = gp.quicksum(
                    (patients[p]["prep"] + patients[p]["surgeon_times"][s] + patients[p]["clean"]) * x[(p, s, h, d, r)]
                    for (p, s, hh, dd, rr) in by_hdr.get((h, d, r), [])
                )
                m.addConstr(load <= reg * y[(h, d, r)] + ot[(h, d, r)], name=f"orcap_{h}_{d}_{r}")
                m.addConstr(ot[(h, d, r)] <= mot * y[(h, d, r)], name=f"otmax_{h}_{d}_{r}")

    # Surgeon load capacity
    for s in surgeon_ids:
        sdata = surgeons[s]
        for h in hospital_ids:
            for d in days:
                avail = float(sdata["availability"].get(d, 0.0))
                terms = []
                for (p, ss, hh, dd, r) in by_shd.get((s, h, d), []):
                    proc = patients[p]["surgeon_times"][s]
                    tps = patients[p]["prep"] + proc + patients[p]["clean"]
                    coeff = alpha * proc + (1.0 - alpha) * tps
                    terms.append(coeff * x[(p, s, h, d, r)])
                if terms:
                    m.addConstr(gp.quicksum(terms) <= avail * z[(s, h, d)], name=f"sload_{s}_{h}_{d}")
                else:
                    m.addConstr(z[(s, h, d)] == 0, name=f"zempty_{s}_{h}_{d}")

    # Objective
    open_cost = gp.quicksum(or_fixed_open_cost[(h, d, r)] * y[(h, d, r)] for (h, d, r) in y.keys())
    s_cost = gp.quicksum(surgeon_fixed_cost.get((s, h), 0.0) * z[(s, h, d)] for (s, h, d) in z.keys())
    ot_cost = gp.quicksum(or_ot_cost_hour[(h, d, r)] * (ot[(h, d, r)] / 60.0) for (h, d, r) in ot.keys())
    reward = gp.quicksum(patients[p]["reward"] * a[p] for p in patient_ids if not patients[p]["mandatory"])
    m.setObjective(open_cost + s_cost + ot_cost - reward, GRB.MINIMIZE)

    # Callback logging incumbents
    m._logger = logger
    m._best = float("inf")

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if obj + 1e-9 < model._best:
                model._best = obj
                safe_log_solution(model._logger, obj, None)

    m.optimize(cb)

    if m.SolCount == 0:
        sol = build_empty_solution(1e18)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # ---- Extract selected assignments ----
    selected = []
    for k, var in x.items():
        if var.X > 0.5:
            selected.append(k)  # (p,s,h,d,r)

    # ---- Heuristic timing construction ----
    jobs_by_hd = defaultdict(list)
    for (p, s, h, d, r) in selected:
        pdata = patients[p]
        proc = pdata["surgeon_times"][s]
        jobs_by_hd[(h, d)].append({
            "p": p, "s": s, "r": r,
            "prep": pdata["prep"], "proc": proc, "clean": pdata["clean"],
            "due": pdata["due_date"], "mandatory": pdata["mandatory"]
        })

    job_times = {}
    or_order = defaultdict(list)        # (h,d,r) -> ordered patient list
    surgeon_order = defaultdict(list)   # (s,h,d) -> ordered patient list

    for (h, d), jobs in jobs_by_hd.items():
        or_avail = {r: 0.0 for r in ors_by_hospital[h]}
        surg_avail = {}
        for jb in jobs:
            surg_avail[jb["s"]] = 0.0

        unscheduled = jobs[:]
        while unscheduled:
            best_idx = None
            best_key = None
            best_times = None

            for idx, jb in enumerate(unscheduled):
                p, s, r = jb["p"], jb["s"], jb["r"]
                prep, proc, clean = jb["prep"], jb["proc"], jb["clean"]

                room_entry = max(or_avail[r], surg_avail[s] - prep)
                surg_start = max(room_entry + prep, surg_avail[s])
                finish = surg_start + proc
                room_exit = finish + clean

                key = (0 if jb["mandatory"] else 1, jb["due"], surg_start, finish)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
                    best_times = (room_entry, surg_start, finish, room_exit)

            jb = unscheduled.pop(best_idx)
            p, s, r = jb["p"], jb["s"], jb["r"]
            room_entry, surg_start, finish, room_exit = best_times

            job_times[p] = {
                "room_entry": float(room_entry),
                "surgery_start": float(surg_start),
                "finish": float(finish),
                "room_exit": float(room_exit),
                "s": s, "h": h, "d": d, "r": r
            }

            or_avail[r] = room_exit
            surg_avail[s] = finish
            or_order[(h, d, r)].append(p)
            surgeon_order[(s, h, d)].append(p)

    # ---- Build output tables ----
    assignments = []
    for (p, s, h, d, r) in selected:
        pdata = patients[p]
        proc = pdata["surgeon_times"][s]
        tps = pdata["prep"] + proc + pdata["clean"]

        tm = job_times.get(p, None)
        if tm is None:
            tm = {
                "room_entry": 0.0,
                "surgery_start": pdata["prep"],
                "finish": pdata["prep"] + proc,
                "room_exit": tps
            }

        assignments.append({
            "patient_id": int(p),
            "surgeon_id": int(s),
            "hospital_id": int(h),
            "day": int(d),
            "or_id": int(r),
            "is_mandatory": bool(pdata["mandatory"]),
            "T_ps": float(tps),
            "finish_time": float(tm["finish"]),
            "surgery_start_time": float(tm["surgery_start"]),
            "room_entry_time": float(tm["room_entry"]),
            "room_exit_time": float(tm["room_exit"]),
        })

    opened_ors = []
    for (h, d, r), yv in y.items():
        if yv.X > 0.5:
            plist = or_order.get((h, d, r), [])
            completion = max((job_times[p]["room_exit"] for p in plist), default=0.0)
            reg = or_regular[(h, d, r)]
            overtime = max(0.0, completion - reg)

            opened_ors.append({
                "hospital_id": int(h),
                "day": int(d),
                "or_id": int(r),
                "regular_time": int(reg),
                "completion_time": float(completion),
                "overtime": float(overtime),
            })

    surgeon_assignments = []
    for (s, h, d), zv in z.items():
        if zv.X > 0.5:
            plist = surgeon_order.get((s, h, d), [])
            if plist:
                st = min(job_times[p]["surgery_start"] for p in plist)
                en = max(job_times[p]["finish"] for p in plist)
            else:
                st = 0.0
                en = 0.0
            surgeon_assignments.append({
                "surgeon_id": int(s),
                "hospital_id": int(h),
                "day": int(d),
                "start_time": float(st),
                "end_time": float(en),
            })

    or_sequences = []
    for (h, d, r), plist in or_order.items():
        pos = {p: i for i, p in enumerate(plist)}
        n = len(plist)
        for i in range(n):
            for j in range(i + 1, n):
                p = plist[i]
                k = plist[j]
                or_sequences.append({
                    "hospital_id": int(h),
                    "day": int(d),
                    "or_id": int(r),
                    "patient_p": int(p),
                    "patient_k": int(k),
                    "p_after_k": int(1 if pos[p] > pos[k] else 0),
                })

    surgeon_sequences = []
    for (s, h, d), plist in surgeon_order.items():
        pos = {p: i for i, p in enumerate(plist)}
        n = len(plist)
        for i in range(n):
            for j in range(i + 1, n):
                p = plist[i]
                k = plist[j]
                surgeon_sequences.append({
                    "hospital_id": int(h),
                    "day": int(d),
                    "surgeon_id": int(s),
                    "patient_p": int(p),
                    "patient_k": int(k),
                    "p_after_k": int(1 if pos[p] > pos[k] else 0),
                })

    # Objective value (consistent with output)
    open_cost_val = 0.0
    for rec in opened_ors:
        h, d, r = rec["hospital_id"], rec["day"], rec["or_id"]
        open_cost_val += or_fixed_open_cost[(h, d, r)]

    surgeon_cost_val = 0.0
    for rec in surgeon_assignments:
        s, h = rec["surgeon_id"], rec["hospital_id"]
        surgeon_cost_val += surgeon_fixed_cost.get((s, h), 0.0)

    overtime_cost_val = 0.0
    for rec in opened_ors:
        h, d, r = rec["hospital_id"], rec["day"], rec["or_id"]
        overtime_cost_val += or_ot_cost_hour[(h, d, r)] * (rec["overtime"] / 60.0)

    reward_val = 0.0
    for rec in assignments:
        p = rec["patient_id"]
        if not rec["is_mandatory"]:
            reward_val += patients[p]["reward"]

    objective_value = open_cost_val + surgeon_cost_val + overtime_cost_val - reward_val

    solution = {
        "objective_value": float(objective_value),
        "assignments": assignments,
        "opened_ors": opened_ors,
        "surgeon_assignments": surgeon_assignments,
        "or_sequences": or_sequences,
        "surgeon_sequences": surgeon_sequences,
    }

    safe_log_solution(logger, float(objective_value), solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()