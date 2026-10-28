import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp_path, path)


def get_seed(instance):
    for key in ("random_seed", "profile_seed", "seed"):
        if key in instance:
            return int(instance[key])
    qc_data = instance.get("qc_profiles", {})
    if isinstance(qc_data, dict):
        for key in ("random_seed", "profile_seed", "seed"):
            if key in qc_data:
                return int(qc_data[key])
    return 0


def available_qcs(instance, horizon):
    raw = instance.get("qc_availability", instance.get("total_qc", 0))
    if isinstance(raw, (int, float)):
        return [int(raw)] * (horizon + 1)

    result = [0] * (horizon + 1)
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            default = int(instance.get("total_qc", 0))
            result = [default] * (horizon + 1)
            for row in raw:
                t = int(row.get("time", row.get("time_step", row.get("id", 0))))
                value = int(row.get("available", row.get("num_qc", row.get("qcs", 0))))
                if 0 <= t <= horizon:
                    result[t] = value
        else:
            if len(raw) == horizon + 1:
                return [int(x) for x in raw]
            for t in range(1, horizon + 1):
                result[t] = int(raw[t - 1]) if t - 1 < len(raw) else int(raw[-1])
    return result


def extract_explicit_profiles(vessel, instance, steps_per_shift):
    """
    Accepts materialized profiles when an instance provides them in addition to,
    or instead of, the compact generation description.
    """
    raw = vessel.get("profiles")
    if raw is None:
        raw = vessel.get("generated_profiles")
    if raw is None and isinstance(vessel.get("qc_profiles"), (list, dict)):
        raw = vessel.get("qc_profiles")

    if raw is None:
        top = instance.get("materialized_qc_profiles")
        if isinstance(top, dict):
            raw = top.get(str(vessel["id"]), top.get(vessel["id"]))
        elif isinstance(top, list) and vessel["id"] < len(top):
            raw = top[vessel["id"]]

    if raw is None:
        return None

    profiles = []

    def add_profile(obj, default_id, forced_offset=None):
        if not isinstance(obj, dict):
            return

        pid = int(obj.get("id", obj.get("profile", obj.get("profile_id", default_id))))
        offset = obj.get(
            "within_shift_index",
            obj.get("start_index", obj.get("offset", forced_offset))
        )
        if offset is not None:
            offset = int(offset)
            if offset == 0:
                offset = 1

        qcs = obj.get(
            "qcs",
            obj.get(
                "qc_usage",
                obj.get("qc_assignment", obj.get("quay_cranes"))
            ),
        )
        duration = obj.get(
            "duration",
            obj.get("handling_duration", obj.get("num_time_steps"))
        )

        if qcs is None:
            shift_qcs = obj.get("shift_qcs", obj.get("qc_per_shift"))
            if shift_qcs is not None and offset is not None:
                expanded = []
                first_length = steps_per_shift - offset + 1
                for i, q in enumerate(shift_qcs):
                    repeat = first_length if i == 0 else steps_per_shift
                    expanded.extend([int(q)] * repeat)
                qcs = expanded

        if qcs is None:
            return

        qcs = [int(x) for x in qcs]
        if duration is None:
            duration = len(qcs)
        duration = int(duration)
        qcs = qcs[:duration]
        if len(qcs) < duration and qcs:
            qcs.extend([qcs[-1]] * (duration - len(qcs)))
        if duration <= 0 or not qcs:
            return

        profiles.append({
            "id": pid,
            "offset": offset,
            "duration": duration,
            "qcs": tuple(qcs),
        })

    if isinstance(raw, list):
        for i, obj in enumerate(raw):
            add_profile(obj, i)
    elif isinstance(raw, dict):
        # Either one profile object or a map from within-shift index to lists.
        if any(k in raw for k in ("qcs", "qc_usage", "qc_assignment", "shift_qcs")):
            add_profile(raw, 0)
        else:
            next_id = 0
            for key, values in raw.items():
                try:
                    offset = int(key)
                    if offset == 0:
                        offset = 1
                except (TypeError, ValueError):
                    offset = None
                if isinstance(values, list):
                    for obj in values:
                        add_profile(obj, next_id, offset)
                        next_id += 1
    return profiles or None


def sampled_shift_sequences(spec, requested_count, seed):
    min_qc = int(spec["min_qc"])
    max_qc = int(spec["max_qc"])
    min_len = int(spec["min_handling_shifts"])
    max_len = int(spec["max_handling_shifts"])

    base = max_qc - min_qc + 1
    lengths = list(range(min_len, max_len + 1))
    block_sizes = [base ** length for length in lengths]
    population = sum(block_sizes)

    if population <= 0 or requested_count <= 0:
        return []

    count = min(int(requested_count), population)
    rng = random.Random(seed)
    selected = rng.sample(range(population), count)

    sequences = []
    for global_index in selected:
        local = global_index
        length = lengths[-1]
        for candidate_length, block_size in zip(lengths, block_sizes):
            if local < block_size:
                length = candidate_length
                break
            local -= block_size

        digits = [0] * length
        value = local
        for pos in range(length - 1, -1, -1):
            digits[pos] = value % base
            value //= base
        sequences.append(tuple(min_qc + d for d in digits))

    return sequences


def classify_vessels(instance):
    vessels = instance["vessels"]
    durations = sorted(float(v.get("duration_hours", 0)) for v in vessels)
    median = durations[len(durations) // 2] if durations else 0.0

    result = {}
    for v in vessels:
        explicit = v.get("size", v.get("vessel_size", v.get("profile_type")))
        if explicit is not None:
            name = str(explicit).lower()
            result[int(v["id"])] = "big" if "big" in name or "large" in name else "small"
        elif "is_big" in v:
            result[int(v["id"])] = "big" if bool(v["is_big"]) else "small"
        else:
            result[int(v["id"])] = (
                "big" if float(v.get("duration_hours", 0)) >= median else "small"
            )
    return result


def build_generated_profiles(instance, vessel, vessel_class, sequence_cache):
    """
    Compact profile generation.

    A vessel's duration_hours is interpreted as crane-hours of work. A sampled
    shift-level sequence is expanded from each possible within-shift start
    position. Service finishes at the first time step at which its crane-hour
    requirement has been processed. Crane assignments only change at shift
    boundaries.
    """
    steps_per_shift = int(instance["time_steps_per_shift"])
    horizon_hours = float(instance["time_horizon_hours"])
    total_steps = int(instance["total_time_steps"])
    hours_per_step = horizon_hours / total_steps if total_steps else 1.0
    required_qc_steps = float(vessel.get("duration_hours", 0.0)) / hours_per_step
    required_qc_steps = max(required_qc_steps, 1e-9)

    sequences = sequence_cache[vessel_class]
    generated = []

    for offset in range(1, steps_per_shift + 1):
        for seq_index, shift_sequence in enumerate(sequences):
            expanded = []
            first_shift_length = steps_per_shift - offset + 1
            for j, qc in enumerate(shift_sequence):
                repeat = first_shift_length if j == 0 else steps_per_shift
                expanded.extend([int(qc)] * repeat)

            cumulative = 0.0
            finish = None
            for i, qc in enumerate(expanded):
                cumulative += qc
                if cumulative + 1e-9 >= required_qc_steps:
                    finish = i + 1
                    break

            if finish is None:
                continue

            # Keep profiles whose sampled final shift is actually used.
            start_of_final_sampled_shift = (
                first_shift_length
                + max(0, len(shift_sequence) - 2) * steps_per_shift
                + 1
            )
            if len(shift_sequence) > 1 and finish < start_of_final_sampled_shift:
                continue

            generated.append({
                "id": (offset - 1) * len(sequences) + seq_index,
                "offset": offset,
                "duration": finish,
                "qcs": tuple(expanded[:finish]),
            })

    # If strict final-shift filtering removed everything, retain any sequence
    # capable of completing the work.
    if not generated:
        for offset in range(1, steps_per_shift + 1):
            for seq_index, shift_sequence in enumerate(sequences):
                expanded = []
                first_shift_length = steps_per_shift - offset + 1
                for j, qc in enumerate(shift_sequence):
                    repeat = first_shift_length if j == 0 else steps_per_shift
                    expanded.extend([int(qc)] * repeat)

                cumulative = 0.0
                for i, qc in enumerate(expanded):
                    cumulative += qc
                    if cumulative + 1e-9 >= required_qc_steps:
                        generated.append({
                            "id": (offset - 1) * len(sequences) + seq_index,
                            "offset": offset,
                            "duration": i + 1,
                            "qcs": tuple(expanded[:i + 1]),
                        })
                        break

    return generated


def make_solution(candidate_by_var, selected_indices, objective):
    rows = []
    by_vessel = {}
    for idx in selected_indices:
        c = candidate_by_var[idx]
        by_vessel[c["vessel"]] = {
            "id": c["vessel"],
            "berth": c["berth"],
            "start_time": c["start"],
            "end_time": c["end"],
            "profile": c["profile"],
        }

    for vessel_id in sorted(by_vessel):
        rows.append(by_vessel[vessel_id])

    return {
        "objective_value": float(objective),
        "vessels": rows,
    }


def greedy_incumbent(candidates_by_vessel, num_vessels, num_berths, horizon, qc_capacity):
    berth_used = [[False] * (horizon + 1) for _ in range(num_berths)]
    qc_used = [0] * (horizon + 1)
    chosen = {}

    order = sorted(
        range(num_vessels),
        key=lambda v: (
            len(candidates_by_vessel[v]),
            min((c["cost"] for c in candidates_by_vessel[v]), default=math.inf),
        ),
    )

    for v in order:
        feasible = []
        for c in candidates_by_vessel[v]:
            ok = True
            for t in range(c["start"], c["end"] + 1):
                if berth_used[c["berth"]][t]:
                    ok = False
                    break
                rel = t - c["start"]
                if qc_used[t] + c["qcs"][rel] > qc_capacity[t]:
                    ok = False
                    break
            if ok:
                resource_score = sum(
                    c["qcs"][t - c["start"]] / max(1, qc_capacity[t])
                    for t in range(c["start"], c["end"] + 1)
                )
                feasible.append((c["cost"], resource_score, c["end"], c))

        if not feasible:
            return None

        feasible.sort(key=lambda x: (x[0], x[1], x[2]))
        c = feasible[0][3]
        chosen[v] = c
        for t in range(c["start"], c["end"] + 1):
            berth_used[c["berth"]][t] = True
            qc_used[t] += c["qcs"][t - c["start"]]

    return chosen


def main():
    args = parse_args()
    start_clock = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    vessels_raw = sorted(instance["vessels"], key=lambda x: int(x["id"]))
    berths_raw = sorted(instance["berths"], key=lambda x: int(x["id"]))
    num_vessels = int(instance["num_vessels"])
    num_berths = int(instance["num_berths"])
    horizon = int(instance["total_time_steps"])
    steps_per_shift = int(instance["time_steps_per_shift"])
    c1 = int(instance["c1"])
    c2 = int(instance["c2"])
    qc_capacity = available_qcs(instance, horizon)

    seed = get_seed(instance)
    profile_specs = instance.get("qc_profiles", {})
    classes = classify_vessels(instance)

    sequence_cache = {}
    if isinstance(profile_specs, dict) and "big" in profile_specs and "small" in profile_specs:
        sequence_cache["big"] = sampled_shift_sequences(
            profile_specs["big"],
            int(instance.get("num_big_profiles", 0)),
            seed,
        )
        sequence_cache["small"] = sampled_shift_sequences(
            profile_specs["small"],
            int(instance.get("num_small_profiles", 0)),
            seed,
        )

    profiles_by_vessel = {}
    for v in vessels_raw:
        vid = int(v["id"])
        explicit = extract_explicit_profiles(v, instance, steps_per_shift)
        if explicit is not None:
            profiles_by_vessel[vid] = explicit
        else:
            vessel_class = classes[vid]
            if vessel_class not in sequence_cache:
                profiles_by_vessel[vid] = []
            else:
                profiles_by_vessel[vid] = build_generated_profiles(
                    instance, v, vessel_class, sequence_cache
                )

    candidates = []
    candidates_by_vessel = [[] for _ in range(num_vessels)]
    berth_time_rows = defaultdict(list)
    qc_time_rows = defaultdict(list)

    berth_data = {
        int(b["id"]): (
            int(b.get("a_k", 1)),
            int(b.get("b_k", horizon)),
        )
        for b in berths_raw
    }

    for v in vessels_raw:
        vid = int(v["id"])
        earliest = int(v["a"])
        latest = int(v["b"])
        preferred_berth = int(v["k_bar"])
        expected_time = int(v["t_bar"])

        by_offset = defaultdict(list)
        for profile in profiles_by_vessel[vid]:
            if profile["offset"] is None:
                for off in range(1, steps_per_shift + 1):
                    by_offset[off].append(profile)
            else:
                by_offset[int(profile["offset"])].append(profile)

        for berth in range(num_berths):
            if berth not in berth_data:
                continue
            berth_start, berth_end = berth_data[berth]
            first_start = max(earliest, berth_start, 1)
            last_start = min(latest, berth_end, horizon)

            for service_start in range(first_start, last_start + 1):
                offset = ((service_start - 1) % steps_per_shift) + 1
                for profile in by_offset.get(offset, []):
                    duration = int(profile["duration"])
                    service_end = service_start + duration - 1
                    if service_end > horizon or service_end > berth_end:
                        continue

                    qcs = profile["qcs"]
                    if len(qcs) < duration:
                        continue

                    individually_feasible = True
                    for rel, t in enumerate(range(service_start, service_end + 1)):
                        if qcs[rel] > qc_capacity[t]:
                            individually_feasible = False
                            break
                    if not individually_feasible:
                        continue

                    cost = (
                        c1 * abs(berth - preferred_berth)
                        + c2 * abs(service_start - expected_time)
                    )
                    idx = len(candidates)
                    candidate = {
                        "index": idx,
                        "vessel": vid,
                        "berth": berth,
                        "start": service_start,
                        "end": service_end,
                        "profile": int(profile["id"]),
                        "qcs": qcs,
                        "cost": float(cost),
                    }
                    candidates.append(candidate)
                    candidates_by_vessel[vid].append(candidate)
                    for t in range(service_start, service_end + 1):
                        berth_time_rows[(berth, t)].append(idx)
                        qc_time_rows[t].append((idx, int(qcs[t - service_start])))

    if any(not candidates_by_vessel[v] for v in range(num_vessels)):
        solution = {"objective_value": None, "vessels": []}
        write_json(args.solution_path, solution)
        return

    model = gp.Model("berth_and_quay_crane_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    elapsed = time.monotonic() - start_clock
    model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)

    x = model.addVars(
        len(candidates),
        vtype=GRB.BINARY,
        obj=[c["cost"] for c in candidates],
        name="x",
    )
    model.ModelSense = GRB.MINIMIZE

    for v in range(num_vessels):
        model.addConstr(
            gp.quicksum(x[c["index"]] for c in candidates_by_vessel[v]) == 1,
            name=f"assign_{v}",
        )

    for (berth, t), indices in berth_time_rows.items():
        if len(indices) > 1:
            model.addConstr(
                gp.quicksum(x[i] for i in indices) <= 1,
                name=f"berth_{berth}_{t}",
            )

    for t, terms in qc_time_rows.items():
        model.addConstr(
            gp.quicksum(coeff * x[i] for i, coeff in terms) <= qc_capacity[t],
            name=f"qc_{t}",
        )

    greedy = greedy_incumbent(
        candidates_by_vessel, num_vessels, num_berths, horizon, qc_capacity
    )
    best_solution = None
    best_objective = math.inf

    if greedy is not None:
        chosen_indices = set(c["index"] for c in greedy.values())
        for i in range(len(candidates)):
            x[i].Start = 1.0 if i in chosen_indices else 0.0
        greedy_objective = sum(c["cost"] for c in greedy.values())
        best_solution = make_solution(candidates, chosen_indices, greedy_objective)
        best_objective = float(greedy_objective)
        if logger:
            logger.log_solution(best_objective, best_solution)

    callback_state = {
        "best": best_objective,
        "solution": best_solution,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective >= callback_state["best"] - 1e-7:
                return
            values = cb_model.cbGetSolution([x[i] for i in range(len(candidates))])
            selected = [i for i, value in enumerate(values) if value > 0.5]
            if len(selected) != num_vessels:
                return
            solution = make_solution(candidates, selected, objective)
            callback_state["best"] = objective
            callback_state["solution"] = solution
            if logger:
                logger.log_solution(objective, solution)
        except Exception:
            # Logging must never interrupt the optimizer.
            pass

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        pass

    final_solution = callback_state["solution"]

    if model.SolCount > 0:
        selected = [i for i in range(len(candidates)) if x[i].X > 0.5]
        if len(selected) == num_vessels:
            exact_objective = sum(candidates[i]["cost"] for i in selected)
            solver_solution = make_solution(candidates, selected, exact_objective)
            if final_solution is None or exact_objective < final_solution["objective_value"] - 1e-7:
                final_solution = solver_solution
                if logger:
                    logger.log_solution(exact_objective, solver_solution)

    if final_solution is None:
        final_solution = {"objective_value": None, "vessels": []}

    write_json(args.solution_path, final_solution)


if __name__ == "__main__":
    main()