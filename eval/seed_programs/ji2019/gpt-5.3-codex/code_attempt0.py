import argparse
import json
import time
from copy import deepcopy

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic solver for lock/transshipment scheduling problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path for incumbent solutions.")
    return parser.parse_args()


def parse_eligible_berths(raw, berth_id_to_idx, K):
    if raw is None:
        return list(range(K))
    if not isinstance(raw, list):
        return list(range(K))

    # Binary mask style
    if len(raw) == K and all(isinstance(v, (int, float)) for v in raw):
        vals = [int(round(v)) for v in raw]
        if all(v in (0, 1) for v in vals):
            return [k for k, v in enumerate(vals) if v == 1]

    # Explicit berth IDs/indices
    elig = []
    for b in raw:
        if b in berth_id_to_idx:
            elig.append(berth_id_to_idx[b])
        elif isinstance(b, (int, float)):
            bb = int(round(b))
            if 0 <= bb < K:
                elig.append(bb)
    return sorted(set(elig))


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    n = int(data["num_ships"])
    K = int(data["num_berths"])

    ships = data["ships"]
    berths = data["berths"]
    handling = data["handling_times"]

    berth_ids = [b["id"] for b in berths]
    berth_id_to_idx = {bid: i for i, bid in enumerate(berth_ids)}

    ship_ids = [s["id"] for s in ships]
    arrivals = [float(s["arrival_time"]) for s in ships]
    widths = [float(s["width"]) for s in ships]
    lengths = [float(s["length"]) for s in ships]
    penalties = [float(s["transshipment_penalty"]) for s in ships]

    start_berth = [0.0] * K
    for b in berths:
        start_berth[berth_id_to_idx[b["id"]]] = float(b["start_availability_sk"])

    eligible = []
    for s in ships:
        eligible.append(parse_eligible_berths(s.get("eligible_berths_B", None), berth_id_to_idx, K))

    H = [[float(handling[i][k]) for k in range(K)] for i in range(n)]

    inst = {
        "n": n,
        "K": K,
        "ship_ids": ship_ids,
        "berth_ids": berth_ids,
        "arrivals": arrivals,
        "widths": widths,
        "lengths": lengths,
        "penalties": penalties,
        "eligible": eligible,
        "H": H,
        "start_berth": start_berth,
        "W": float(data["chamber_width_W"]),
        "L": float(data["chamber_length_L"]),
        "Du": float(data["lockage_duration_Du"]),
        "MT": float(data["min_time_between_lockages_MT"]),
        "sc": float(data["lock_start_availability_sc"]),
        "ek": float(data["berth_end_availability_ek"]),
        "Cmax": float(data["C_max"]),
    }
    return inst


def can_lock(inst, i):
    return inst["widths"][i] <= inst["W"] and inst["lengths"][i] <= inst["L"]


def build_solution_from_modes(inst, modes, strict=True):
    """
    modes[i] = 1 transshipment, 0 lock
    strict=True enforces berth end availability and lock Cmax strictly.
    """
    n, K = inst["n"], inst["K"]
    arrivals = inst["arrivals"]
    penalties = inst["penalties"]
    eligible = inst["eligible"]
    H = inst["H"]
    sk = inst["start_berth"]
    ek = inst["ek"]

    # Lock scheduling (one ship per lockage, ordered by arrival)
    lock_ships = [i for i in range(n) if modes[i] == 0]
    lock_ships.sort(key=lambda i: (arrivals[i], i))

    lockage_assignment = {}
    lockage_completion = {}
    ship_service_completion = [None] * n

    prev_c = None
    for pos, i in enumerate(lock_ships, start=1):
        if not can_lock(inst, i):
            if strict:
                return None
        base = inst["sc"] + inst["Du"] if prev_c is None else prev_c + inst["MT"]
        c = max(base, arrivals[i] + inst["Du"])
        if strict and c > inst["Cmax"]:
            return None
        lockage_assignment[i] = pos
        lockage_completion[pos] = c
        ship_service_completion[i] = c
        prev_c = c

    # Berth scheduling (arrival order among trans ships; greedy best berth)
    trans_ships = [i for i in range(n) if modes[i] == 1]
    trans_ships.sort(key=lambda i: (arrivals[i], i))

    berth_time = sk[:]
    berth_order_count = [0] * K
    berth_assignment = {}
    berth_order = {}

    for i in trans_ships:
        best = None
        best_k = None
        for k in eligible[i]:
            h = H[i][k]
            c = max(berth_time[k], arrivals[i]) + h
            feasible = c <= ek
            if strict and not feasible:
                continue
            # In non-strict, allow overrun if unavoidable
            metric = c
            if (best is None) or (metric < best):
                best = metric
                best_k = k

        if best_k is None:
            # fallback
            if strict:
                return None
            # choose least-overrun berth if possible, else impossible
            if not eligible[i]:
                return None
            candidate = None
            cand_k = None
            for k in eligible[i]:
                c = max(berth_time[k], arrivals[i]) + H[i][k]
                if candidate is None or c < candidate:
                    candidate = c
                    cand_k = k
            best_k = cand_k
            best = candidate

        berth_order_count[best_k] += 1
        berth_assignment[i] = best_k
        berth_order[i] = berth_order_count[best_k]
        berth_time[best_k] = best
        ship_service_completion[i] = best

    # Compute objective
    obj = 0.0
    for i in range(n):
        if modes[i] == 0:
            obj += ship_service_completion[i] - arrivals[i]
        else:
            obj += ship_service_completion[i] - arrivals[i] + penalties[i]

    return {
        "objective": obj,
        "modes": modes[:],
        "ship_service_completion": ship_service_completion,
        "lockage_assignment": lockage_assignment,
        "lockage_completion": lockage_completion,
        "berth_assignment": berth_assignment,
        "berth_order": berth_order,
    }


def construct_initial(inst):
    n, K = inst["n"], inst["K"]
    arrivals = inst["arrivals"]
    penalties = inst["penalties"]
    eligible = inst["eligible"]
    H = inst["H"]
    sk = inst["start_berth"]
    ek = inst["ek"]

    order = sorted(range(n), key=lambda i: (arrivals[i], i))

    modes = [1] * n
    berth_time = sk[:]
    lock_prev = None

    for i in order:
        # lock option
        lock_feas = can_lock(inst, i)
        if lock_feas:
            base = inst["sc"] + inst["Du"] if lock_prev is None else lock_prev + inst["MT"]
            c_lock = max(base, arrivals[i] + inst["Du"])
            if c_lock > inst["Cmax"]:
                lock_feas = False
        else:
            c_lock = float("inf")

        # trans option
        best_trans = None
        best_k = None
        for k in eligible[i]:
            c = max(berth_time[k], arrivals[i]) + H[i][k]
            if c <= ek and (best_trans is None or c < best_trans):
                best_trans = c
                best_k = k

        trans_feas = best_k is not None

        # choose
        if trans_feas and (not lock_feas or (best_trans - arrivals[i] + penalties[i] <= c_lock - arrivals[i])):
            modes[i] = 1
            berth_time[best_k] = best_trans
        elif lock_feas:
            modes[i] = 0
            lock_prev = c_lock
        else:
            # fallback if both infeasible under strict windows: choose trans non-strict if possible, else lock
            if eligible[i]:
                best_any = None
                best_any_k = None
                for k in eligible[i]:
                    c = max(berth_time[k], arrivals[i]) + H[i][k]
                    if best_any is None or c < best_any:
                        best_any = c
                        best_any_k = k
                modes[i] = 1
                berth_time[best_any_k] = best_any
            else:
                modes[i] = 0
                if lock_prev is None:
                    lock_prev = max(inst["sc"] + inst["Du"], arrivals[i] + inst["Du"])
                else:
                    lock_prev = max(lock_prev + inst["MT"], arrivals[i] + inst["Du"])

    sol = build_solution_from_modes(inst, modes, strict=True)
    if sol is None:
        sol = build_solution_from_modes(inst, modes, strict=False)
    return sol


def format_output(inst, sol):
    ship_ids = inst["ship_ids"]
    berth_ids = inst["berth_ids"]
    arrivals = inst["arrivals"]

    modes_out = {}
    completion_times_out = {}
    lockage_assignments_out = {}
    berth_assignments_out = {}
    lockage_completion_times_out = {}

    for i, sid in enumerate(ship_ids):
        m = int(sol["modes"][i])
        modes_out[str(sid)] = m
        if m == 0:
            completion_times_out[str(sid)] = float(sol["ship_service_completion"][i])
            lockage_assignments_out[str(sid)] = int(sol["lockage_assignment"][i])
        else:
            # As described: departure time of transshipment-mode ship equals arrival time
            completion_times_out[str(sid)] = float(arrivals[i])
            k = sol["berth_assignment"][i]
            berth_assignments_out[str(sid)] = {
                "berth": int(berth_ids[k]),
                "order": int(sol["berth_order"][i]),
            }

    for lidx, ct in sol["lockage_completion"].items():
        lockage_completion_times_out[str(lidx)] = float(ct)

    return {
        "objective_value": float(sol["objective"]),
        "modes": modes_out,
        "completion_times": completion_times_out,
        "lockage_assignments": lockage_assignments_out,
        "lockage_completion_times": lockage_completion_times_out,
        "berth_assignments": berth_assignments_out,
    }


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    t0 = time.time()

    inst = load_instance(args.instance_path)

    best = construct_initial(inst)
    if best is None:
        # emergency fallback: all lock if possible else all trans non-strict
        n = inst["n"]
        modes = [0 if can_lock(inst, i) else 1 for i in range(n)]
        best = build_solution_from_modes(inst, modes, strict=False)
    best_obj = best["objective"]
    if logger:
        logger.log(best_obj)

    # Simple 1-flip local search
    n = inst["n"]
    improved = True
    while improved and (time.time() - t0 < max(0, args.time_limit - 1)):
        improved = False
        cur_modes = best["modes"][:]
        for i in range(n):
            if time.time() - t0 >= max(0, args.time_limit - 1):
                break
            trial_modes = cur_modes[:]
            trial_modes[i] = 1 - trial_modes[i]
            trial = build_solution_from_modes(inst, trial_modes, strict=True)
            if trial is None:
                continue
            if trial["objective"] + 1e-9 < best_obj:
                best = trial
                best_obj = trial["objective"]
                improved = True
                if logger:
                    logger.log(best_obj)
                break  # first-improvement restart

    output = format_output(inst, best)

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()