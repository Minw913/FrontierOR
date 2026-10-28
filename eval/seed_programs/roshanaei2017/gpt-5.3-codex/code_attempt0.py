import argparse
import json
import math
import time
from collections import defaultdict
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def to_int_dict(d):
    return {int(k): v for k, v in d.items()}


def detect_days(instance):
    num_days = int(instance["num_days"])
    seen = set()

    for s in instance.get("surgeons", []):
        for d in s.get("operating_days", []):
            seen.add(int(d))
        for k in s.get("availability_by_day", {}).keys():
            seen.add(int(k))

    for h in instance.get("hospitals", []):
        for orr in h.get("ORs", []):
            for k in orr.get("daily", {}).keys():
                seen.add(int(k))

    if not seen:
        return list(range(1, num_days + 1))

    d1 = set(range(1, num_days + 1))
    d0 = set(range(0, num_days))

    if seen.issubset(d1):
        return list(range(1, num_days + 1))
    if seen.issubset(d0):
        return list(range(0, num_days))

    # fallback: use observed days
    return sorted(seen)


def get_or_day_params(or_data, day):
    daily = or_data.get("daily", {})
    day_info = daily.get(str(day), {})
    regular = int(day_info.get("regular_time", or_data.get("regular_time", 0)))
    max_ot = int(day_info.get("max_overtime", or_data.get("max_overtime", 0)))
    return regular, max_ot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    t0 = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    alpha = float(inst["alpha"])
    days = detect_days(inst)

    # ---------------------------
    # Parse entities
    # ---------------------------
    patients = {}
    for p in inst["patients"]:
        pid = int(p["patient_id"])
        patients[pid] = {
            "due_date": int(p["due_date"]),
            "is_mandatory": bool(p["is_mandatory"]),
            "prep": float(p["preparation_time"]),
            "clean": float(p["cleaning_time"]),
            "eligible_surgeons": set(int(x) for x in p.get("eligible_surgeons", [])),
            "eligible_ORs_by_hospital": {
                int(h): set(int(r) for r in rs) for h, rs in p.get("eligible_ORs_by_hospital", {}).items()
            },
            "surgeon_specific_times": {int(s): float(v) for s, v in p.get("surgeon_specific_times", {}).items()},
            "reward": float(p.get("reward", 0.0)),
        }

    surgeons = {}
    for s in inst["surgeons"]:
        sid = int(s["surgeon_id"])
        availability = to_int_dict(s.get("availability_by_day", {}))
        surgeons[sid] = {
            "operating_days": set(int(d) for d in s.get("operating_days", [])),
            "availability": {int(d): float(v) for d, v in availability.items()},
        }

    hospitals = {}
    all_or_keys = []  # (h, r)
    or_day_regular = {}
    or_day_maxot = {}
    or_fixed_cost_day = {}
    or_ot_cost = {}

    for h in inst["hospitals"]:
        hid = int(h["hospital_id"])
        s_cost = {int(k): float(v) for k, v in h.get("surgeon_fixed_costs_per_day", {}).items()}
        or_map = {}
        for orr in h["ORs"]:
            rid = int(orr["or_id"])
            or_map[rid] = orr
            all_or_keys.append((hid, rid))

            for d in days:
                reg, mot = get_or_day_params(orr, d)
                or_day_regular[(hid, d, rid)] = float(reg)
                or_day_maxot[(hid, d, rid)] = float(mot)
                # fixed opening cost interpreted as hourly * regular session hours
                fixed_per_hour = float(orr.get("fixed_cost_per_hour", 0.0))
                or_fixed_cost_day[(hid, d, rid)] = fixed_per_hour * (reg / 60.0)
                or_ot_cost[(hid, d, rid)] = float(orr.get("overtime_cost_per_hour", 0.0))

        hospitals[hid] = {
            "ORs": or_map,
            "surgeon_fixed_costs": s_cost
        }

    patient_ids = sorted(patients.keys())
    surgeon_ids = sorted(surgeons.keys())
    hospital_ids = sorted(hospitals.keys())

    # ---------------------------
    # Build feasible assignment tuples
    # x[p,s,h,d,r]
    # ---------------------------
    feasible_tuples = []
    by_patient = defaultdict(list)
    by_hdr_patient = defaultdict(list)   # (p,h,d,r)->[(tuplekeys)]
    by_pshd_patient = defaultdict(list)  # (p,s,h,d)->[(tuplekeys)]
    patients_possible_in_hdr = defaultdict(set)   # (h,d,r)->set(p)
    patients_possible_in_shd = defaultdict(set)   # (s,h,d)->set(p)

    max_proc = 0.0
    max_prep = 0.0
    max_clean = 0.0
    max_or_span = 0.0

    for p in patient_ids:
        pdata = patients[p]
        max_prep = max(max_prep, pdata["prep"])
        max_clean = max(max_clean, pdata["clean"])
        is_mand = pdata["is_mandatory"]

        for s in pdata["eligible_surgeons"]:
            if s not in surgeons:
                continue
            if s not in pdata["surgeon_specific_times"]:
                continue
            proc = pdata["surgeon_specific_times"][s]
            max_proc = max(max_proc, proc)

            sdata = surgeons[s]
            for d in days:
                if d not in sdata["operating_days"]:
                    continue
                avail = float(sdata["availability"].get(d, 0.0))
                if avail <= 0:
                    continue

                if is_mand and d > pdata["due_date"]:
                    continue  # mandatory must be no later than due date

                for h in hospital_ids:
                    eligible_ors = pdata["eligible_ORs_by_hospital"].get(h, set())
                    if not eligible_ors:
                        continue
                    for r in eligible_ors:
                        if r not in hospitals[h]["ORs"]:
                            continue
                        key = (p, s, h, d, r)
                        feasible_tuples.append(key)
                        by_patient[p].append(key)
                        by_hdr_patient[(p, h, d, r)].append(key)
                        by_pshd_patient[(p, s, h, d)].append(key)
                        patients_possible_in_hdr[(h, d, r)].add(p)
                        patients_possible_in_shd[(s, h, d)].add(p)

    for h in hospital_ids:
        for d in days:
            for r in hospitals[h]["ORs"].keys():
                max_or_span = max(max_or_span, or_day_regular[(h, d, r)] + or_day_maxot[(h, d, r)])

    BIG_M = max_or_span + max_prep + max_proc + max_clean + 100.0
    if BIG_M < 1000:
        BIG_M = 1000.0

    # quick infeasibility check: mandatory without any feasible assignment
    mandatory_no_choice = [p for p in patient_ids if patients[p]["is_mandatory"] and len(by_patient[p]) == 0]
    if mandatory_no_choice:
        # Output empty infeasible-like solution
        out = {
            "objective_value": 1e18,
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": [],
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    # ---------------------------
    # Model
    # ---------------------------
    m = gp.Model("Hospital_OR_Scheduling")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, int(args.time_limit))
    m.Params.MIPFocus = 1

    # Variables
    x = {}
    for key in feasible_tuples:
        x[key] = m.addVar(vtype=GRB.BINARY, name=f"x_{key}")

    a = {p: m.addVar(vtype=GRB.BINARY, name=f"a_{p}") for p in patient_ids}
    finish = {p: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"F_{p}") for p in patient_ids}
    surg_dur = {p: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"dur_{p}") for p in patient_ids}

    y = {}
    ot = {}
    comp = {}
    for h in hospital_ids:
        for d in days:
            for r in hospitals[h]["ORs"].keys():
                y[(h, d, r)] = m.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{r}")
                ot[(h, d, r)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ot_{h}_{d}_{r}")
                comp[(h, d, r)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"comp_{h}_{d}_{r}")

    z = {}
    st = {}
    en = {}
    for s in surgeon_ids:
        for h in hospital_ids:
            for d in days:
                z[(s, h, d)] = m.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}")
                st[(s, h, d)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"st_{s}_{h}_{d}")
                en[(s, h, d)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"en_{s}_{h}_{d}")

    # Sequencing binaries
    b_or = {}
    for (h, d, r), pset in patients_possible_in_hdr.items():
        plist = sorted(pset)
        if len(plist) <= 1:
            continue
        for p, k in combinations(plist, 2):
            b_or[(p, k, h, d, r)] = m.addVar(vtype=GRB.BINARY, name=f"bor_{p}_{k}_{h}_{d}_{r}")

    b_surg = {}
    for (s, h, d), pset in patients_possible_in_shd.items():
        plist = sorted(pset)
        if len(plist) <= 1:
            continue
        for p, k in combinations(plist, 2):
            b_surg[(p, k, s, h, d)] = m.addVar(vtype=GRB.BINARY, name=f"bs_{p}_{k}_{s}_{h}_{d}")

    m.update()

    # ---------------------------
    # Core assignment constraints
    # ---------------------------
    for p in patient_ids:
        expr = gp.quicksum(x[key] for key in by_patient[p])
        m.addConstr(a[p] == expr, name=f"assign_link_{p}")
        if patients[p]["is_mandatory"]:
            m.addConstr(expr == 1, name=f"mandatory_{p}")
        else:
            m.addConstr(expr <= 1, name=f"optional_{p}")

    # surgeon duration per patient
    for p in patient_ids:
        m.addConstr(
            surg_dur[p] == gp.quicksum(patients[p]["surgeon_specific_times"][key[1]] * x[key] for key in by_patient[p]),
            name=f"surgdur_def_{p}"
        )
        # finish-time lower/upper bounds conditioned on assignment
        m.addConstr(
            finish[p] >= patients[p]["prep"] + surg_dur[p] - BIG_M * (1 - a[p]),
            name=f"finish_lb_{p}"
        )
        m.addConstr(
            finish[p] <= BIG_M * a[p],
            name=f"finish_ub_{p}"
        )

    # Link x to OR open and surgeon assignment
    for (p, s, h, d, r), xv in x.items():
        m.addConstr(xv <= y[(h, d, r)], name=f"x_y_{p}_{s}_{h}_{d}_{r}")
        m.addConstr(xv <= z[(s, h, d)], name=f"x_z_{p}_{s}_{h}_{d}_{r}")

    # Surgeon at most one hospital per day
    for s in surgeon_ids:
        sdata = surgeons[s]
        for d in days:
            if d in sdata["operating_days"] and float(sdata["availability"].get(d, 0.0)) > 0:
                m.addConstr(gp.quicksum(z[(s, h, d)] for h in hospital_ids) <= 1, name=f"one_hosp_{s}_{d}")
            else:
                for h in hospital_ids:
                    m.addConstr(z[(s, h, d)] == 0, name=f"z_off_{s}_{h}_{d}")

    # OR capacity and overtime
    for h in hospital_ids:
        for d in days:
            for r in hospitals[h]["ORs"].keys():
                reg = or_day_regular[(h, d, r)]
                mot = or_day_maxot[(h, d, r)]

                load_expr = gp.quicksum(
                    (patients[p]["prep"] + patients[p]["surgeon_specific_times"][s] + patients[p]["clean"]) * x[(p, s, h, d, r)]
                    for (p, s, hh, dd, rr) in feasible_tuples if hh == h and dd == d and rr == r
                )

                m.addConstr(load_expr <= reg * y[(h, d, r)] + ot[(h, d, r)], name=f"or_cap_{h}_{d}_{r}")
                m.addConstr(ot[(h, d, r)] <= mot * y[(h, d, r)], name=f"ot_max_{h}_{d}_{r}")
                m.addConstr(ot[(h, d, r)] >= comp[(h, d, r)] - reg, name=f"ot_def_{h}_{d}_{r}")
                m.addConstr(comp[(h, d, r)] <= (reg + mot) * y[(h, d, r)], name=f"comp_bound_{h}_{d}_{r}")

    # completion time >= finish + cleaning for each assigned patient in OR
    for (p, h, d, r), keys in by_hdr_patient.items():
        assign_expr = gp.quicksum(x[k] for k in keys)
        m.addConstr(
            comp[(h, d, r)] >= finish[p] + patients[p]["clean"] - BIG_M * (1 - assign_expr),
            name=f"comp_lb_{p}_{h}_{d}_{r}"
        )

    # Surgeon workload and time-span constraints
    for s in surgeon_ids:
        sdata = surgeons[s]
        for h in hospital_ids:
            for d in days:
                avail = float(sdata["availability"].get(d, 0.0))

                load = gp.quicksum(
                    (
                        alpha * patients[p]["surgeon_specific_times"][s]
                        + (1 - alpha) * (patients[p]["prep"] + patients[p]["surgeon_specific_times"][s] + patients[p]["clean"])
                    ) * x[(p, s, h, d, r)]
                    for (p, ss, hh, dd, r) in feasible_tuples if ss == s and hh == h and dd == d
                )
                m.addConstr(load <= avail * z[(s, h, d)], name=f"sload_{s}_{h}_{d}")

                m.addConstr(en[(s, h, d)] - st[(s, h, d)] <= avail + BIG_M * (1 - z[(s, h, d)]),
                            name=f"span_{s}_{h}_{d}")
                m.addConstr(en[(s, h, d)] <= BIG_M * z[(s, h, d)], name=f"en_b_{s}_{h}_{d}")
                m.addConstr(st[(s, h, d)] <= BIG_M * z[(s, h, d)], name=f"st_b_{s}_{h}_{d}")
                m.addConstr(st[(s, h, d)] <= en[(s, h, d)] + BIG_M * (1 - z[(s, h, d)]),
                            name=f"stleqen_{s}_{h}_{d}")

    # start/end bounds from assigned cases
    for (p, s, h, d), keys in by_pshd_patient.items():
        assign_expr = gp.quicksum(x[k] for k in keys)
        proc = patients[p]["surgeon_specific_times"][s]
        m.addConstr(en[(s, h, d)] >= finish[p] - BIG_M * (1 - assign_expr), name=f"en_case_{p}_{s}_{h}_{d}")
        m.addConstr(st[(s, h, d)] <= finish[p] - proc + BIG_M * (1 - assign_expr),
                    name=f"st_case_{p}_{s}_{h}_{d}")

    # OR sequencing disjunctive constraints
    for (p, k, h, d, r), b in b_or.items():
        A_p = gp.quicksum(x[var] for var in by_hdr_patient.get((p, h, d, r), []))
        A_k = gp.quicksum(x[var] for var in by_hdr_patient.get((k, h, d, r), []))

        # if b=0 => p before k
        m.addConstr(
            finish[p] + patients[p]["clean"] + patients[k]["prep"] + surg_dur[k] - finish[k]
            <= BIG_M * (b + (1 - A_p) + (1 - A_k)),
            name=f"or_seq1_{p}_{k}_{h}_{d}_{r}"
        )
        # if b=1 => k before p
        m.addConstr(
            finish[k] + patients[k]["clean"] + patients[p]["prep"] + surg_dur[p] - finish[p]
            <= BIG_M * ((1 - b) + (1 - A_p) + (1 - A_k)),
            name=f"or_seq2_{p}_{k}_{h}_{d}_{r}"
        )

    # Surgeon sequencing disjunctive constraints
    for (p, k, s, h, d), b in b_surg.items():
        A_p = gp.quicksum(x[var] for var in by_pshd_patient.get((p, s, h, d), []))
        A_k = gp.quicksum(x[var] for var in by_pshd_patient.get((k, s, h, d), []))

        tps = patients[p]["surgeon_specific_times"][s]
        tks = patients[k]["surgeon_specific_times"][s]

        # if b=0 => p before k in surgeon timeline
        m.addConstr(
            finish[p] + tks - finish[k] <= BIG_M * (b + (1 - A_p) + (1 - A_k)),
            name=f"s_seq1_{p}_{k}_{s}_{h}_{d}"
        )
        # if b=1 => k before p
        m.addConstr(
            finish[k] + tps - finish[p] <= BIG_M * ((1 - b) + (1 - A_p) + (1 - A_k)),
            name=f"s_seq2_{p}_{k}_{s}_{h}_{d}"
        )

    # ---------------------------
    # Objective
    # ---------------------------
    open_cost = gp.quicksum(
        or_fixed_cost_day[(h, d, r)] * y[(h, d, r)]
        for h in hospital_ids for d in days for r in hospitals[h]["ORs"].keys()
    )
    surgeon_cost = gp.quicksum(
        float(hospitals[h]["surgeon_fixed_costs"].get(s, 0.0)) * z[(s, h, d)]
        for s in surgeon_ids for h in hospital_ids for d in days
    )
    overtime_cost = gp.quicksum(
        or_ot_cost[(h, d, r)] * (ot[(h, d, r)] / 60.0)
        for h in hospital_ids for d in days for r in hospitals[h]["ORs"].keys()
    )
    reward = gp.quicksum(
        patients[p]["reward"] * a[p]
        for p in patient_ids if not patients[p]["is_mandatory"]
    )

    m.setObjective(open_cost + surgeon_cost + overtime_cost - reward, GRB.MINIMIZE)

    # incumbent logger callback
    m._logger = logger
    m._best_logged = float("inf")

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < model._best_logged:
                model._best_logged = obj
                model._logger.log(float(obj))

    # Respect global time limit from CLI
    elapsed = time.time() - t0
    remaining = max(1, int(args.time_limit - elapsed))
    m.Params.TimeLimit = remaining

    m.optimize(cb)

    # ---------------------------
    # Build output
    # ---------------------------
    if m.SolCount == 0:
        out = {
            "objective_value": 1e18,
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": [],
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    obj_val = float(m.ObjVal)

    # capture selected assignments
    selected = {}
    assignments = []
    for (p, s, h, d, r), var in x.items():
        if var.X > 0.5:
            selected[p] = (s, h, d, r)

            proc = patients[p]["surgeon_specific_times"][s]
            prep = patients[p]["prep"]
            clean = patients[p]["clean"]
            fin = float(finish[p].X)
            surg_start = fin - proc
            room_entry = surg_start - prep
            room_exit = fin + clean
            T_ps = prep + proc + clean

            assignments.append({
                "patient_id": int(p),
                "surgeon_id": int(s),
                "hospital_id": int(h),
                "day": int(d),
                "or_id": int(r),
                "is_mandatory": bool(patients[p]["is_mandatory"]),
                "T_ps": float(T_ps),
                "finish_time": float(fin),
                "surgery_start_time": float(surg_start),
                "room_entry_time": float(room_entry),
                "room_exit_time": float(room_exit),
            })

    opened_ors = []
    for h in hospital_ids:
        for d in days:
            for r in hospitals[h]["ORs"].keys():
                if y[(h, d, r)].X > 0.5:
                    opened_ors.append({
                        "hospital_id": int(h),
                        "day": int(d),
                        "or_id": int(r),
                        "regular_time": int(or_day_regular[(h, d, r)]),
                        "completion_time": float(comp[(h, d, r)].X),
                        "overtime": float(max(0.0, ot[(h, d, r)].X)),
                    })

    surgeon_assignments = []
    for s in surgeon_ids:
        for h in hospital_ids:
            for d in days:
                if z[(s, h, d)].X > 0.5:
                    surgeon_assignments.append({
                        "surgeon_id": int(s),
                        "hospital_id": int(h),
                        "day": int(d),
                        "start_time": float(st[(s, h, d)].X),
                        "end_time": float(en[(s, h, d)].X),
                    })

    # or_sequences output only for actually co-assigned pairs
    by_hdr_assigned = defaultdict(list)
    by_shd_assigned = defaultdict(list)
    for p, (s, h, d, r) in selected.items():
        by_hdr_assigned[(h, d, r)].append(p)
        by_shd_assigned[(s, h, d)].append(p)

    or_sequences = []
    for (p, k, h, d, r), var in b_or.items():
        plist = by_hdr_assigned.get((h, d, r), [])
        pset = set(plist)
        if p in pset and k in pset:
            or_sequences.append({
                "hospital_id": int(h),
                "day": int(d),
                "or_id": int(r),
                "patient_p": int(p),
                "patient_k": int(k),
                "p_after_k": int(round(var.X)),
            })

    surgeon_sequences = []
    for (p, k, s, h, d), var in b_surg.items():
        plist = by_shd_assigned.get((s, h, d), [])
        pset = set(plist)
        if p in pset and k in pset:
            surgeon_sequences.append({
                "hospital_id": int(h),
                "day": int(d),
                "surgeon_id": int(s),
                "patient_p": int(p),
                "patient_k": int(k),
                "p_after_k": int(round(var.X)),
            })

    out = {
        "objective_value": obj_val,
        "assignments": assignments,
        "opened_ors": opened_ors,
        "surgeon_assignments": surgeon_assignments,
        "or_sequences": or_sequences,
        "surgeon_sequences": surgeon_sequences,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()