import argparse
import json
import math
import time
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    # -----------------------------
    # Parse sets and basic mappings
    # -----------------------------
    patients = instance["patients"]
    hospitals = instance["hospitals"]
    scenarios = instance["scenarios"]

    P = [p["patient_id"] for p in patients]
    H = [h["hospital_id"] for h in hospitals]
    S = [s["scenario_id"] for s in scenarios]
    num_scenarios = instance["num_scenarios"]

    # Map patient_id -> position in patient list (for scenario duration lookup)
    ppos = {p["patient_id"]: i for i, p in enumerate(patients)}

    # Build day sets and parameters
    # Assume all hospitals share same day_id universe
    day_ids = sorted({d["day_id"] for h in hospitals for d in h["days"]})
    D = day_ids

    # day_id -> day_label (from first occurrence)
    day_label_of = {}
    for h in hospitals:
        for d in h["days"]:
            if d["day_id"] not in day_label_of:
                day_label_of[d["day_id"]] = d["day_label"]

    # num ORs per hospital (can read from hospital records)
    R_by_h = {h["hospital_id"]: list(range(h["num_ors"])) for h in hospitals}

    # Parameters per hospital/day
    B = {}  # time limit per OR
    F = {}  # OR opening cost
    G = {}  # suite opening cost
    for h in hospitals:
        hid = h["hospital_id"]
        for d in h["days"]:
            did = d["day_id"]
            B[(hid, did)] = float(d["B_hd"])
            F[(hid, did)] = float(d["F_hd"])
            G[(hid, did)] = float(d["G_hd"])

    # Patient parameters
    mandatory = {p["patient_id"]: bool(p["is_mandatory"]) for p in patients}
    c_unsched = {p["patient_id"]: float(p["c_unsched"]) for p in patients}
    c_cancel = {p["patient_id"]: float(p["c_cancel"]) for p in patients}

    # Scheduling benefit c_sched[p,d]
    c_sched = {}
    for p in patients:
        pid = p["patient_id"]
        per_day = p["c_sched_per_day"]
        for did in D:
            dl = day_label_of[did]
            c_sched[(pid, did)] = float(per_day[str(dl)] if str(dl) in per_day else per_day[dl])

    # Scenario durations dur[(s,p)]
    dur = {}
    for s in scenarios:
        sid = s["scenario_id"]
        arr = s["surgery_durations_minutes"]
        for pid in P:
            dur[(sid, pid)] = float(arr[ppos[pid]])

    M = [pid for pid in P if mandatory[pid]]
    N = [pid for pid in P if not mandatory[pid]]

    # -----------------------------
    # Build model
    # -----------------------------
    m = Model("hospital_or_scheduling_stochastic")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1, int(time_limit))
    m.Params.Threads = 1

    # Variables
    # u[h,d] suite open
    u = m.addVars([(h, d) for h in H for d in D], vtype=GRB.BINARY, name="u")

    # y[h,d,r] OR open
    y = m.addVars(
        [(h, d, r) for h in H for d in D for r in R_by_h[h]],
        vtype=GRB.BINARY,
        name="y",
    )

    # x[h,d,p,r] assignment
    x_keys = [(h, d, p, r) for h in H for d in D for p in P for r in R_by_h[h]]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")

    # w[p] postpone only for non-mandatory
    w = m.addVars(N, vtype=GRB.BINARY, name="w")

    # q[h,d,p,r,s] performed in scenario s
    q_keys = [(h, d, p, r, s) for h in H for d in D for p in P for r in R_by_h[h] for s in S]
    q = m.addVars(q_keys, vtype=GRB.BINARY, name="q")

    # -----------------------------
    # Constraints
    # -----------------------------
    # OR open only if suite open
    for h in H:
        for d in D:
            for r in R_by_h[h]:
                m.addConstr(y[h, d, r] <= u[h, d], name=f"or_if_suite_{h}_{d}_{r}")

    # Assignment only to open OR
    for h in H:
        for d in D:
            for r in R_by_h[h]:
                for p in P:
                    m.addConstr(x[h, d, p, r] <= y[h, d, r], name=f"assign_if_or_{h}_{d}_{p}_{r}")

    # Mandatory assignment exactly once
    for p in M:
        m.addConstr(
            quicksum(x[h, d, p, r] for h in H for d in D for r in R_by_h[h]) == 1,
            name=f"mandatory_once_{p}",
        )

    # Non-mandatory either scheduled once or postponed
    for p in N:
        m.addConstr(
            quicksum(x[h, d, p, r] for h in H for d in D for r in R_by_h[h]) + w[p] == 1,
            name=f"nonmandatory_sched_or_postpone_{p}",
        )

    # Per-scenario execution can occur only if assigned
    for h in H:
        for d in D:
            for r in R_by_h[h]:
                for p in P:
                    for s in S:
                        m.addConstr(
                            q[h, d, p, r, s] <= x[h, d, p, r],
                            name=f"perform_if_assigned_{h}_{d}_{p}_{r}_{s}",
                        )

    # Per-scenario OR capacity
    for h in H:
        for d in D:
            for r in R_by_h[h]:
                for s in S:
                    m.addConstr(
                        quicksum(dur[(s, p)] * q[h, d, p, r, s] for p in P) <= B[(h, d)] * y[h, d, r],
                        name=f"capacity_{h}_{d}_{r}_{s}",
                    )

    # -----------------------------
    # Objective
    # -----------------------------
    # Opening costs
    obj_open_suite = quicksum(G[(h, d)] * u[h, d] for h in H for d in D)
    obj_open_or = quicksum(F[(h, d)] * y[h, d, r] for h in H for d in D for r in R_by_h[h])

    # Scheduling benefit/cost
    obj_sched = quicksum(c_sched[(p, d)] * x[h, d, p, r] for h in H for d in D for p in P for r in R_by_h[h])

    # Postponement
    obj_postpone = quicksum(c_unsched[p] * w[p] for p in N)

    # Expected cancellation:
    # (1/|S|) * sum_s sum_{assigned} c_cancel[p] * (x - q)
    # = sum c_cancel*x - (1/|S|) * sum_s c_cancel*q
    obj_cancel_x = quicksum(c_cancel[p] * x[h, d, p, r] for h in H for d in D for p in P for r in R_by_h[h])
    obj_cancel_q = (1.0 / num_scenarios) * quicksum(
        c_cancel[p] * q[h, d, p, r, s] for h in H for d in D for p in P for r in R_by_h[h] for s in S
    )

    m.setObjective(obj_open_suite + obj_open_or + obj_sched + obj_postpone + obj_cancel_x - obj_cancel_q, GRB.MINIMIZE)

    # -----------------------------
    # Simple feasible MIP start
    # -----------------------------
    # Choose cheapest (h,d) to open
    best_hd = min(((h, d) for h in H for d in D), key=lambda t: G[t] + F[t])
    h0, d0 = best_hd
    r0 = R_by_h[h0][0]

    for h in H:
        for d in D:
            u[h, d].Start = 1.0 if (h, d) == (h0, d0) else 0.0
            for r in R_by_h[h]:
                y[h, d, r].Start = 1.0 if (h, d, r) == (h0, d0, r0) else 0.0

    for h, d, p, r in x_keys:
        x[h, d, p, r].Start = 0.0

    for p in M:
        x[h0, d0, p, r0].Start = 1.0

    for p in N:
        # crude rule: postpone unless scheduling+possible cancellation seems clearly better
        best_sched_cost = min(c_sched[(p, d)] + c_cancel[p] for d in D)
        if c_unsched[p] <= best_sched_cost:
            w[p].Start = 1.0
        else:
            w[p].Start = 0.0
            x[h0, d0, p, r0].Start = 1.0

    for key in q_keys:
        q[key].Start = 0.0

    # -----------------------------
    # Incumbent logging callback
    # -----------------------------
    m._logger = logger

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            model._logger.log(float(obj))

    m.optimize(cb)

    # -----------------------------
    # Extract best solution found
    # -----------------------------
    sol = {
        "objective_value": float("inf"),
        "u": {},
        "y": {},
        "x": {},
        "w": {},
    }

    if m.SolCount > 0:
        sol["objective_value"] = float(m.ObjVal)

        # u, y: include all keys
        for h in H:
            for d in D:
                sol["u"][f"{h},{d}"] = int(round(u[h, d].X))
                for r in R_by_h[h]:
                    sol["y"][f"{h},{d},{r}"] = int(round(y[h, d, r].X))

        # x: only non-zero entries
        for (h, d, p, r) in x_keys:
            val = x[h, d, p, r].X
            if val > 0.5:
                sol["x"][f"{h},{d},{p},{r}"] = 1

        # w: non-mandatory patients
        for p in N:
            sol["w"][f"{p}"] = int(round(w[p].X))
    else:
        # fallback (unlikely)
        for h in H:
            for d in D:
                sol["u"][f"{h},{d}"] = 0
                for r in R_by_h[h]:
                    sol["y"][f"{h},{d},{r}"] = 0
        for p in N:
            sol["w"][f"{p}"] = 1

    return sol


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

    solution = build_and_solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()