import argparse
import json
import math
import random
import time
from itertools import product
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def infer_time_base(instance):
    vals = []
    for v in instance.get("vessels", []):
        for k in ("a", "b", "t_bar"):
            if k in v and isinstance(v[k], int):
                vals.append(v[k])
    for b in instance.get("berths", []):
        for k in ("a_k", "b_k"):
            if k in b and isinstance(b[k], int):
                vals.append(b[k])
    if not vals:
        return 0
    return 0 if min(vals) == 0 else 1


def within_shift_pos(t, time_base, steps_per_shift):
    # Returns 1..steps_per_shift
    return ((t - time_base) % steps_per_shift) + 1


def parse_seed(instance):
    for k in ("seed", "random_seed", "profile_seed", "qc_profile_seed"):
        if k in instance:
            try:
                return int(instance[k])
            except Exception:
                pass
    # Optional nested location
    if "qc_profiles" in instance:
        qcp = instance["qc_profiles"]
        for k in ("seed", "random_seed", "profile_seed", "qc_profile_seed"):
            if k in qcp:
                try:
                    return int(qcp[k])
                except Exception:
                    pass
    return 0


def generate_shift_profiles(profile_cfg, requested_n, rng):
    """
    Generate admissible shift-level profiles:
    sequence length in [min_handling_shifts, max_handling_shifts],
    each entry in [min_qc, max_qc].
    Enumerated lexicographically by (length, tuple values), then sampled.
    """
    min_qc = int(profile_cfg["min_qc"])
    max_qc = int(profile_cfg["max_qc"])
    min_h = int(profile_cfg["min_handling_shifts"])
    max_h = int(profile_cfg["max_handling_shifts"])

    all_profiles = []
    qc_vals = list(range(min_qc, max_qc + 1))
    for L in range(min_h, max_h + 1):
        for seq in product(qc_vals, repeat=L):
            all_profiles.append(tuple(int(x) for x in seq))

    if requested_n >= len(all_profiles):
        return all_profiles

    idxs = rng.sample(range(len(all_profiles)), requested_n)
    return [all_profiles[i] for i in idxs]


def choose_vessel_class(v, big_cfg, small_cfg, hours_per_shift):
    # If explicit type is provided, use it
    for key in ("size", "type", "class"):
        if key in v:
            val = str(v[key]).lower()
            if "big" in val or "large" in val:
                return "big"
            if "small" in val:
                return "small"

    duration_hours = float(v.get("duration_hours", 0.0))
    if hours_per_shift <= 0:
        # Fallback
        return "big" if duration_hours >= 0 else "small"

    exp_shifts = duration_hours / hours_per_shift

    bmin, bmax = big_cfg["min_handling_shifts"], big_cfg["max_handling_shifts"]
    smin, smax = small_cfg["min_handling_shifts"], small_cfg["max_handling_shifts"]

    in_big = (bmin <= exp_shifts <= bmax)
    in_small = (smin <= exp_shifts <= smax)
    if in_big and not in_small:
        return "big"
    if in_small and not in_big:
        return "small"

    # If both or none: choose closest midpoint
    bmid = 0.5 * (bmin + bmax)
    smid = 0.5 * (smin + smax)
    return "big" if abs(exp_shifts - bmid) <= abs(exp_shifts - smid) else "small"


def expand_profile_to_steps(shift_seq, start_t, time_base, steps_per_shift):
    """
    Given shift-level QC seq and a start time step, return:
    - duration in time steps
    - list of qc usage per relative step 1..duration
    End-of-shift rule: QC count only changes at shift boundaries.
    """
    offset = (start_t - time_base) % steps_per_shift  # 0..S-1
    first_seg_len = steps_per_shift - offset
    L = len(shift_seq)
    duration = first_seg_len + (L - 1) * steps_per_shift

    qc_steps = []
    for r in range(1, duration + 1):
        if r <= first_seg_len:
            idx = 0
        else:
            idx = 1 + (r - first_seg_len - 1) // steps_per_shift
        qc_steps.append(int(shift_seq[idx]))
    return duration, qc_steps


def build_solution_dict(num_vessels, selected_option_by_vessel, objective_value):
    vessels_out = []
    for i in range(num_vessels):
        opt = selected_option_by_vessel.get(i)
        if opt is None:
            # Should not happen in feasible solution; keep schema-compliant fallback.
            vessels_out.append({
                "id": i,
                "berth": 0,
                "start_time": 0,
                "end_time": 0,
                "profile": 0
            })
        else:
            vessels_out.append({
                "id": i,
                "berth": int(opt["berth"]),
                "start_time": int(opt["start_time"]),
                "end_time": int(opt["end_time"]),
                "profile": int(opt["profile_out"])
            })
    return {
        "objective_value": None if objective_value is None else float(objective_value),
        "vessels": vessels_out
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    instance = load_instance(args.instance_path)

    num_vessels = int(instance["num_vessels"])
    num_berths = int(instance["num_berths"])

    vessels = sorted(instance["vessels"], key=lambda x: x["id"])
    berths = sorted(instance["berths"], key=lambda x: x["id"])

    c1 = float(instance["c1"])
    c2 = float(instance["c2"])

    steps_per_shift = int(instance["time_steps_per_shift"])
    total_time_steps = int(instance["total_time_steps"])
    num_shifts = int(instance.get("num_shifts_per_step", instance.get("num_shifts", 1)))
    time_horizon_hours = float(instance.get("time_horizon_hours", total_time_steps))
    hours_per_shift = (time_horizon_hours / num_shifts) if num_shifts > 0 else 1.0

    time_base = infer_time_base(instance)
    T_start = time_base
    T_end = time_base + total_time_steps - 1

    # QC capacity (constant by schema; allow optional overrides)
    qc_cap_default = int(instance["total_qc"])
    qc_cap_map = {}
    if "qc_available" in instance and isinstance(instance["qc_available"], list):
        arr = instance["qc_available"]
        for idx, val in enumerate(arr):
            qc_cap_map[time_base + idx] = int(val)
    else:
        for t in range(T_start, T_end + 1):
            qc_cap_map[t] = qc_cap_default

    # Build shift-level profile pools
    qcp = instance["qc_profiles"]
    big_cfg = qcp["big"]
    small_cfg = qcp["small"]

    rng = random.Random(parse_seed(instance))
    n_big = int(instance["num_big_profiles"])
    n_small = int(instance["num_small_profiles"])

    big_profiles = generate_shift_profiles(big_cfg, n_big, rng)
    small_profiles = generate_shift_profiles(small_cfg, n_small, rng)

    # Prepare berth windows
    berth_window = {}
    for b in berths:
        k = int(b["id"])
        berth_window[k] = (int(b["a_k"]), int(b["b_k"]))

    # Build assignment options
    # option fields:
    # id, vessel, berth, start_time, end_time, profile_out, cost, qc_by_time (dict t->qc)
    options = []
    options_by_vessel = defaultdict(list)
    berth_time_to_opts = defaultdict(list)
    qc_time_coeffs = defaultdict(list)

    # Optional duration filters if present
    def get_duration_bounds_steps(v):
        # optional fields
        min_steps = v.get("min_duration_steps", None)
        max_steps = v.get("max_duration_steps", None)
        if min_steps is not None and max_steps is not None:
            return int(min_steps), int(max_steps)
        min_h = v.get("min_duration_hours", None)
        max_h = v.get("max_duration_hours", None)
        if min_h is not None and max_h is not None:
            h_per_step = time_horizon_hours / total_time_steps if total_time_steps > 0 else 1.0
            return int(math.floor(min_h / h_per_step)), int(math.ceil(max_h / h_per_step))
        return None, None

    for v in vessels:
        i = int(v["id"])
        a_i = int(v["a"])
        b_i = int(v["b"])
        k_bar = int(v["k_bar"])
        t_bar = int(v["t_bar"])

        vclass = choose_vessel_class(v, big_cfg, small_cfg, hours_per_shift)
        shift_profiles = big_profiles if vclass == "big" else small_profiles

        min_d_steps, max_d_steps = get_duration_bounds_steps(v)

        # Feasible start range clipped to horizon
        start_lo = max(a_i, T_start)
        start_hi = min(b_i, T_end)
        if start_lo > start_hi:
            continue

        for k in range(num_berths):
            a_k, b_k = berth_window[k]
            t_lo = max(start_lo, a_k)
            t_hi = min(start_hi, b_k)
            if t_lo > t_hi:
                continue

            berth_pen = c1 * abs(k - k_bar)

            for t in range(t_lo, t_hi + 1):
                # For this start t, every shift-profile determines duration + per-step QC
                for p_idx, seq in enumerate(shift_profiles):
                    dur, qc_steps = expand_profile_to_steps(seq, t, time_base, steps_per_shift)
                    end_t = t + dur - 1

                    # Must stay within planning horizon and berth availability window
                    if end_t > T_end:
                        continue
                    if end_t > b_k:
                        continue

                    if min_d_steps is not None and dur < min_d_steps:
                        continue
                    if max_d_steps is not None and dur > max_d_steps:
                        continue

                    time_pen = c2 * abs(t - t_bar)
                    cost = berth_pen + time_pen

                    qc_by_time = {}
                    feasible_qc = True
                    for r, q in enumerate(qc_steps, start=0):
                        tt = t + r
                        if tt < T_start or tt > T_end:
                            feasible_qc = False
                            break
                        # individual vessel q usage positive; capacity checked globally
                        qc_by_time[tt] = int(q)
                    if not feasible_qc:
                        continue

                    oid = len(options)
                    opt = {
                        "id": oid,
                        "vessel": i,
                        "berth": k,
                        "start_time": t,
                        "end_time": end_t,
                        "profile_out": p_idx,
                        "cost": float(cost),
                        "qc_by_time": qc_by_time
                    }
                    options.append(opt)
                    options_by_vessel[i].append(oid)

                    for tt in range(t, end_t + 1):
                        berth_time_to_opts[(k, tt)].append(oid)
                    for tt, qq in qc_by_time.items():
                        qc_time_coeffs[tt].append((oid, qq))

    # Basic feasibility check
    all_have_options = all(len(options_by_vessel[i]) > 0 for i in range(num_vessels))
    if not all_have_options:
        # Infeasible under reconstructed profile generation/rules
        sol = {
            "objective_value": None,
            "vessels": []
        }
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # Build MIP
    model = gp.Model("berth_qc_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.time() - t0
    remain = max(1.0, args.time_limit - elapsed)
    model.Params.TimeLimit = remain

    y = model.addVars(len(options), vtype=GRB.BINARY, name="y")

    # Objective
    model.setObjective(gp.quicksum(options[o]["cost"] * y[o] for o in range(len(options))), GRB.MINIMIZE)

    # Each vessel exactly one option
    for i in range(num_vessels):
        model.addConstr(gp.quicksum(y[o] for o in options_by_vessel[i]) == 1, name=f"assign_v_{i}")

    # Berth occupancy conflicts: at most one vessel per berth-time
    for (k, tt), olist in berth_time_to_opts.items():
        model.addConstr(gp.quicksum(y[o] for o in olist) <= 1, name=f"berth_{k}_t_{tt}")

    # QC capacity each time step
    for tt in range(T_start, T_end + 1):
        pairs = qc_time_coeffs.get(tt, [])
        if not pairs:
            continue
        model.addConstr(
            gp.quicksum(coeff * y[o] for (o, coeff) in pairs) <= qc_cap_map[tt],
            name=f"qc_t_{tt}"
        )

    # Callback logging incumbents
    best_logged = {"obj": float("inf")}

    def build_incumbent_solution(cb_model):
        vals = cb_model.cbGetSolution([y[o] for o in range(len(options))])
        selected_option_by_vessel = {}
        # Pick selected binary per vessel
        for oid, val in enumerate(vals):
            if val > 0.5:
                i = options[oid]["vessel"]
                selected_option_by_vessel[i] = options[oid]

        # If any vessel missing due tolerance, repair by max value in that vessel's options
        missing = [i for i in range(num_vessels) if i not in selected_option_by_vessel]
        if missing:
            val_map = {oid: vals[oid] for oid in range(len(options))}
            for i in missing:
                cand = options_by_vessel[i]
                if not cand:
                    continue
                best_oid = max(cand, key=lambda oo: val_map.get(oo, -1.0))
                selected_option_by_vessel[i] = options[best_oid]

        obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        sol_dict = build_solution_dict(num_vessels, selected_option_by_vessel, obj)
        return obj, sol_dict

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < best_logged["obj"]:
                    best_logged["obj"] = obj
                    obj2, sol_dict = build_incumbent_solution(cb_model)
                    logger.log_solution(float(obj2), sol_dict)
            except Exception:
                # If full snapshot fails, still log objective
                try:
                    obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj + 1e-9 < best_logged["obj"]:
                        best_logged["obj"] = obj
                        logger.log(float(obj))
                except Exception:
                    pass

    model.optimize(callback if logger is not None else None)

    if model.SolCount > 0:
        selected_option_by_vessel = {}
        for i in range(num_vessels):
            chosen = None
            for oid in options_by_vessel[i]:
                if y[oid].X > 0.5:
                    chosen = options[oid]
                    break
            if chosen is None:
                # fallback
                chosen = max(options_by_vessel[i], key=lambda oo: y[oo].X)
                chosen = options[chosen]
            selected_option_by_vessel[i] = chosen

        objective_value = float(model.ObjVal)
        final_sol = build_solution_dict(num_vessels, selected_option_by_vessel, objective_value)
    else:
        final_sol = {"objective_value": None, "vessels": []}

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()