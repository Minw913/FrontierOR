import argparse
import json
import math
import random
import time
from collections import defaultdict
from solution_logger import SolutionLogger


def get_day_cost(c_sched_per_day, day_label):
    if str(day_label) in c_sched_per_day:
        return float(c_sched_per_day[str(day_label)])
    return float(c_sched_per_day[day_label])


def build_data(instance):
    patients = instance["patients"]
    hospitals = instance["hospitals"]
    scenarios = instance["scenarios"]

    P = [p["patient_id"] for p in patients]
    H = [h["hospital_id"] for h in hospitals]
    S = [s["scenario_id"] for s in scenarios]

    p_by_id = {p["patient_id"]: p for p in patients}
    h_by_id = {h["hospital_id"]: h for h in hospitals}

    # day universe and labels
    D = sorted({d["day_id"] for h in hospitals for d in h["days"]})
    day_label_of = {}
    for h in hospitals:
        for d in h["days"]:
            if d["day_id"] not in day_label_of:
                day_label_of[d["day_id"]] = d["day_label"]

    # hospital-day params
    B = {}
    F = {}
    G = {}
    num_ors = {}
    for h in hospitals:
        hid = h["hospital_id"]
        num_ors[hid] = int(h["num_ors"])
        for d in h["days"]:
            did = d["day_id"]
            B[(hid, did)] = float(d["B_hd"])
            F[(hid, did)] = float(d["F_hd"])
            G[(hid, did)] = float(d["G_hd"])

    # patient params
    mandatory = {}
    c_unsched = {}
    c_cancel = {}
    c_sched = {}
    for p in patients:
        pid = p["patient_id"]
        mandatory[pid] = bool(p["is_mandatory"])
        c_unsched[pid] = float(p["c_unsched"])
        c_cancel[pid] = float(p["c_cancel"])
        for d in D:
            c_sched[(pid, d)] = get_day_cost(p["c_sched_per_day"], day_label_of[d])

    M = [p for p in P if mandatory[p]]
    N = [p for p in P if not mandatory[p]]

    # scenario durations
    ppos = {pid: i for i, pid in enumerate(P)}
    dur = {}  # dur[(s,p)]
    for sc in scenarios:
        sid = sc["scenario_id"]
        arr = sc["surgery_durations_minutes"]
        for pid in P:
            dur[(sid, pid)] = float(arr[ppos[pid]])

    # mean duration
    mean_dur = {}
    denom = max(1, len(S))
    for pid in P:
        mean_dur[pid] = sum(dur[(s, pid)] for s in S) / denom

    return {
        "P": P, "H": H, "D": D, "S": S,
        "patients": patients, "hospitals": hospitals,
        "B": B, "F": F, "G": G, "num_ors": num_ors,
        "mandatory": mandatory, "M": M, "N": N,
        "c_unsched": c_unsched, "c_cancel": c_cancel, "c_sched": c_sched,
        "dur": dur, "mean_dur": mean_dur
    }


def greedy_construct(data, rng, overload_coeff=2.0):
    P, H, D = data["P"], data["H"], data["D"]
    B, F, G, num_ors = data["B"], data["F"], data["G"], data["num_ors"]
    mandatory, M, N = data["mandatory"], data["M"], data["N"]
    c_unsched, c_cancel, c_sched = data["c_unsched"], data["c_cancel"], data["c_sched"]
    mean_dur = data["mean_dur"]

    # decision structures
    suite_open = {(h, d): 0 for h in H for d in D}
    open_ors_count = {(h, d): 0 for h in H for d in D}  # how many ORs opened at (h,d)
    slot_patients = defaultdict(list)  # key (h,d,r) -> [p]
    slot_load = defaultdict(float)     # expected load

    # assignment output maps
    assign = {}  # p -> (h,d,r)
    w = {p: 0 for p in N}

    def open_incremental_cost(h, d):
        if open_ors_count[(h, d)] >= num_ors[h]:
            return float("inf")
        return F[(h, d)] + (0.0 if suite_open[(h, d)] == 1 else G[(h, d)])

    def candidate_cost_existing(p, h, d, r):
        b = B[(h, d)]
        new_load = slot_load[(h, d, r)] + mean_dur[p]
        overload = max(0.0, new_load - b)
        return c_sched[(p, d)] + overload_coeff * (overload / max(1.0, b)) * c_cancel[p]

    def candidate_cost_open_new(p, h, d):
        b = B[(h, d)]
        overload = max(0.0, mean_dur[p] - b)
        return c_sched[(p, d)] + open_incremental_cost(h, d) + overload_coeff * (overload / max(1.0, b)) * c_cancel[p]

    def assign_to_slot(p, h, d, r):
        assign[p] = (h, d, r)
        slot_patients[(h, d, r)].append(p)
        slot_load[(h, d, r)] += mean_dur[p]

    def open_and_assign(p, h, d):
        r = open_ors_count[(h, d)]
        open_ors_count[(h, d)] += 1
        suite_open[(h, d)] = 1
        assign_to_slot(p, h, d, r)

    # Mandatory first: high cancellation penalty and long surgeries first
    mand_order = sorted(M, key=lambda p: (c_cancel[p], mean_dur[p]), reverse=True)

    for p in mand_order:
        best = (float("inf"), None)  # (cost, ("existing"/"new", h,d,r_or_none))

        # existing opened ORs
        for h in H:
            for d in D:
                k = open_ors_count[(h, d)]
                for r in range(k):
                    cost = candidate_cost_existing(p, h, d, r)
                    if cost < best[0]:
                        best = (cost, ("existing", h, d, r))

        # opening new OR
        for h in H:
            for d in D:
                if open_ors_count[(h, d)] < num_ors[h]:
                    cost = candidate_cost_open_new(p, h, d)
                    if cost < best[0]:
                        best = (cost, ("new", h, d, None))

        mode, h, d, r = best[1]
        if mode == "existing":
            assign_to_slot(p, h, d, r)
        else:
            open_and_assign(p, h, d)

    # Non-mandatory: score by how attractive scheduling is vs postponement
    # lower min schedule cost compared to postponement -> schedule earlier
    non_order = sorted(
        N,
        key=lambda p: (c_unsched[p] - min(c_sched[(p, d)] for d in D)),
        reverse=True
    )

    for p in non_order:
        best_sched = (float("inf"), None)

        # existing ORs
        for h in H:
            for d in D:
                k = open_ors_count[(h, d)]
                for r in range(k):
                    cost = candidate_cost_existing(p, h, d, r)
                    if cost < best_sched[0]:
                        best_sched = (cost, ("existing", h, d, r))

        # opening new OR
        for h in H:
            for d in D:
                if open_ors_count[(h, d)] < num_ors[h]:
                    cost = candidate_cost_open_new(p, h, d)
                    if cost < best_sched[0]:
                        best_sched = (cost, ("new", h, d, None))

        # postpone option
        if c_unsched[p] <= best_sched[0]:
            w[p] = 1
        else:
            mode, h, d, r = best_sched[1]
            if mode == "existing":
                assign_to_slot(p, h, d, r)
            else:
                open_and_assign(p, h, d)
            w[p] = 0

    # Build u,y,x
    u = {(h, d): int(suite_open[(h, d)]) for h in H for d in D}
    y = {}
    for h in H:
        for d in D:
            k = open_ors_count[(h, d)]
            for r in range(num_ors[h]):
                y[(h, d, r)] = 1 if r < k else 0

    x = {}
    for p, (h, d, r) in assign.items():
        x[(h, d, p, r)] = 1

    return u, y, x, w, slot_patients


def best_value_knapsack(weights, values, capacity):
    """
    Maximize sum(values) with sum(weights)<=capacity.
    Hybrid:
      - exact DP when moderate
      - greedy fallback when large
    """
    n = len(weights)
    C = max(0, int(capacity))
    if n == 0 or C <= 0:
        return 0.0

    # Exact DP threshold
    if n * C <= 250000:
        dp = [0.0] * (C + 1)
        for w, v in zip(weights, values):
            if w <= 0:
                # zero-weight positive value can always be taken
                if v > 0:
                    for c in range(C + 1):
                        dp[c] += v
                continue
            if w > C:
                continue
            for c in range(C, w - 1, -1):
                nv = dp[c - w] + v
                if nv > dp[c]:
                    dp[c] = nv
        return max(dp)

    # Greedy fallback (value/weight)
    items = []
    zero_val = 0.0
    for w, v in zip(weights, values):
        if w <= 0:
            if v > 0:
                zero_val += v
        else:
            items.append((v / w, w, v))
    items.sort(reverse=True)
    rem = C
    val = zero_val
    for _, w, v in items:
        if w <= rem:
            rem -= w
            val += v
    return val


def evaluate_objective(data, u, y, x, w, slot_patients):
    H, D, S = data["H"], data["D"], data["S"]
    B, F, G = data["B"], data["F"], data["G"]
    c_sched, c_unsched, c_cancel = data["c_sched"], data["c_unsched"], data["c_cancel"]
    dur = data["dur"]
    N = data["N"]

    # Opening costs
    open_suite_cost = sum(G[(h, d)] * u[(h, d)] for h in H for d in D)
    open_or_cost = sum(F[(h, d)] * y[(h, d, r)] for (h, d, r) in y)

    # Scheduling cost
    sched_cost = 0.0
    for (h, d, p, r), val in x.items():
        if val == 1:
            sched_cost += c_sched[(p, d)]

    # Postponement
    postpone_cost = sum(c_unsched[p] * w.get(p, 0) for p in N)

    # Expected cancellation cost
    # For each scenario and OR, choose subset to perform (knapsack maximize saved cancel penalty)
    scenario_cancel_total = 0.0
    num_s = max(1, len(S))

    for s in S:
        canc_s = 0.0
        for (h, d, r), plist in slot_patients.items():
            if not plist:
                continue
            total_pen = sum(c_cancel[p] for p in plist)
            capacity = int(round(B[(h, d)]))
            weights = [int(math.ceil(dur[(s, p)])) for p in plist]
            values = [c_cancel[p] for p in plist]
            saved = best_value_knapsack(weights, values, capacity)
            canc_s += (total_pen - saved)
        scenario_cancel_total += canc_s

    expected_cancel = scenario_cancel_total / num_s

    return open_suite_cost + open_or_cost + sched_cost + postpone_cost + expected_cancel


def solve_heuristic(instance, time_limit, logger=None):
    data = build_data(instance)

    start = time.time()
    deadline = start + max(1, int(time_limit))

    # Multi-start greedy with tiny randomness in patient ordering via jitter
    best = None
    best_obj = float("inf")

    # At least one iteration
    it = 0
    while True:
        now = time.time()
        if it > 0 and now >= deadline:
            break

        rng = random.Random(12345 + 7919 * it)
        overload_coeff = 1.5 + 1.5 * rng.random()  # [1.5,3.0]

        u, y, x, w, slot_patients = greedy_construct(data, rng, overload_coeff=overload_coeff)

        obj = evaluate_objective(data, u, y, x, w, slot_patients)

        if obj < best_obj:
            best_obj = obj
            best = (u, y, x, w)
            if logger:
                logger.log(float(obj))

        it += 1
        # Keep some time for output
        if time.time() + 0.2 >= deadline:
            break

    # Fallback safety
    if best is None:
        u = {(h["hospital_id"], d["day_id"]): 0 for h in instance["hospitals"] for d in h["days"]}
        y = {}
        for h in instance["hospitals"]:
            hid = h["hospital_id"]
            for d in h["days"]:
                did = d["day_id"]
                for r in range(h["num_ors"]):
                    y[(hid, did, r)] = 0
        x = {}
        w = {str(p["patient_id"]): 1 for p in instance["patients"] if not p["is_mandatory"]}
        best_obj = float("inf")
        best = (u, y, x, w)

    u, y, x, w = best
    data = build_data(instance)

    # Build output schema
    out = {
        "objective_value": float(best_obj),
        "u": {},
        "y": {},
        "x": {},
        "w": {}
    }

    # u and y should include all keys
    for h in data["H"]:
        for d in data["D"]:
            out["u"][f"{h},{d}"] = int(u.get((h, d), 0))
            for r in range(data["num_ors"][h]):
                out["y"][f"{h},{d},{r}"] = int(y.get((h, d, r), 0))

    # x only non-zero entries
    for (h, d, p, r), val in x.items():
        if val:
            out["x"][f"{h},{d},{p},{r}"] = 1

    # w for non-mandatory patients
    for p in data["N"]:
        out["w"][f"{p}"] = int(w.get(p, 0))

    return out


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

    solution = solve_heuristic(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()