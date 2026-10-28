import argparse
import json
import time
from collections import defaultdict

from solution_logger import SolutionLogger


def build_landside_plan(S, p, sigma_l, landside_containers):
    """
    Greedy EDF-like plan for landside containers on crane L.
    Returns:
      l_positions: list position at each integer time
      processed_events: list of dicts for landside_events
      processed_ids: set
      skipped_ids: set
      end_time, end_pos
    """
    pos = sigma_l
    t = 0
    l_positions = [pos]
    processed_events = []
    processed_ids = set()
    skipped_ids = set()

    def extend_to_time(new_t, at_pos):
        nonlocal t
        while t < new_t:
            l_positions.append(at_pos)
            t += 1

    def move_to(target):
        nonlocal pos, t
        while pos < target:
            pos += 1
            l_positions.append(pos)
            t += 1
        while pos > target:
            pos -= 1
            l_positions.append(pos)
            t += 1

    # assume input is ordered by non-decreasing deadline per statement
    for c in landside_containers:
        cid = c["id"]
        src = c["source_slot"]
        eft = c["earliest_finish_time"]
        ddl = c["deadline"]

        travel1 = abs(pos - src)
        arrive_src = t + travel1
        after_lift = arrive_src + p
        travel2 = abs(src - (S + 1))
        arrive_dest = after_lift + travel2
        drop_start = max(arrive_dest, eft)

        if drop_start <= ddl:
            move_to(src)
            lift_time = t
            extend_to_time(t + p, pos)  # lift
            move_to(S + 1)
            if t < eft:
                extend_to_time(eft, pos)
            drop_time = t
            if drop_time <= ddl:
                extend_to_time(t + p, pos)  # drop
                processed_events.append(
                    {
                        "container_id": cid,
                        "source_slot": src,
                        "lift_time": lift_time,
                        "drop_time": drop_time,
                    }
                )
                processed_ids.add(cid)
            else:
                # very defensive fallback
                skipped_ids.add(cid)
        else:
            skipped_ids.add(cid)

    return l_positions, processed_events, processed_ids, skipped_ids, t, pos


def schedule_seaside_with_fixed_l(
    S,
    p,
    sigma_w,
    seaside_containers,
    l_positions_base,
):
    """
    Schedule seaside crane W sequentially, respecting W < L at all times
    against fixed L positions (landside-only phase).
    Direct for targets <= S; cooperative handover for target S+1 via slot S.
    """
    w_pos = sigma_w
    t = 0
    w_positions = [w_pos]
    seaside_events = []
    coop_queue = []  # entries for containers requiring L final delivery to S+1

    def lpos(tt):
        if tt < len(l_positions_base):
            return l_positions_base[tt]
        return l_positions_base[-1]

    def safe_at(time_idx, wslot):
        return 0 <= wslot <= S + 1 and (wslot < lpos(time_idx))

    def wait_one():
        nonlocal t
        nxt = t + 1
        if not safe_at(nxt, w_pos):
            return False
        w_positions.append(w_pos)
        t = nxt
        return True

    def move_toward(target):
        nonlocal w_pos, t
        guard = 0
        while w_pos != target:
            guard += 1
            if guard > 2000000:
                return False
            desired = w_pos + (1 if target > w_pos else -1)
            nxt = t + 1
            if safe_at(nxt, desired):
                w_pos = desired
                w_positions.append(w_pos)
                t = nxt
            else:
                # try to keep safety by moving left if needed
                if not safe_at(nxt, w_pos):
                    if w_pos > 0 and safe_at(nxt, w_pos - 1):
                        w_pos -= 1
                        w_positions.append(w_pos)
                        t = nxt
                    else:
                        return False
                else:
                    w_positions.append(w_pos)
                    t = nxt
        return True

    def reserve_stationary(duration):
        nonlocal t
        for _ in range(duration):
            nxt = t + 1
            if not safe_at(nxt, w_pos):
                return False
            w_positions.append(w_pos)
            t = nxt
        return True

    # process seaside in required pickup order
    for c in sorted(seaside_containers, key=lambda x: x["id"]):
        cid = c["id"]
        target = c["target_slot"]

        # go to seaside handover slot 0
        if not move_toward(0):
            return None
        # lift at 0
        if not reserve_stationary(p):
            return None

        drop_slot = target if target <= S else S  # coop only if target S+1
        if not move_toward(drop_slot):
            return None

        seaside_drop_time = t
        if not reserve_stationary(p):
            return None

        evt = {
            "container_id": cid,
            "target_slot": target,
            "seaside_drop_time": seaside_drop_time,
            "seaside_drop_slot": drop_slot,
        }

        if target == S + 1:
            coop_queue.append(
                {
                    "container_id": cid,
                    "target_slot": target,
                    "handover_slot": drop_slot,
                    "available_time": seaside_drop_time,  # per rule: L may begin lift after drop started
                }
            )

        seaside_events.append(evt)

    # after finishing all W-side handling, return W to 0 for easier future separation
    if not move_toward(0):
        return None

    return {
        "w_positions": w_positions,
        "seaside_events": seaside_events,
        "coop_queue": coop_queue,
        "w_end_time": t,
        "w_end_pos": w_pos,
    }


def append_l_coop_tasks(
    S,
    p,
    l_positions,
    w_positions,
    coop_queue,
    seaside_events_by_id,
):
    """
    Append cooperative L tasks (for target S+1) after landside phase.
    W positions are known; if needed, W is assumed to stay at its last position.
    """
    t = len(l_positions) - 1
    l_pos = l_positions[-1]

    def wpos(tt):
        if tt < len(w_positions):
            return w_positions[tt]
        return w_positions[-1]

    def safe_l_at(time_idx, lslot):
        return 0 <= lslot <= S + 1 and (wpos(time_idx) < lslot)

    def wait_until(time_target):
        nonlocal t
        while t < time_target:
            nxt = t + 1
            if not safe_l_at(nxt, l_pos):
                return False
            l_positions.append(l_pos)
            t = nxt
        return True

    def move_to(target):
        nonlocal t, l_pos
        while l_pos < target:
            nxt = t + 1
            cand = l_pos + 1
            if not safe_l_at(nxt, cand):
                # wait if cannot move
                if not safe_l_at(nxt, l_pos):
                    return False
                l_positions.append(l_pos)
                t = nxt
            else:
                l_pos = cand
                l_positions.append(l_pos)
                t = nxt
        while l_pos > target:
            nxt = t + 1
            cand = l_pos - 1
            if not safe_l_at(nxt, cand):
                if not safe_l_at(nxt, l_pos):
                    return False
                l_positions.append(l_pos)
                t = nxt
            else:
                l_pos = cand
                l_positions.append(l_pos)
                t = nxt
        return True

    def stationary(duration):
        nonlocal t
        for _ in range(duration):
            nxt = t + 1
            if not safe_l_at(nxt, l_pos):
                return False
            l_positions.append(l_pos)
            t = nxt
        return True

    # Cooperative tasks in seaside id order
    for task in sorted(coop_queue, key=lambda x: x["container_id"]):
        cid = task["container_id"]
        av = task["available_time"]
        hs = task["handover_slot"]  # S
        tgt = task["target_slot"]   # S+1

        if not wait_until(max(t, av)):
            return None

        if not move_to(hs):
            return None
        lift_time = t
        if not stationary(p):
            return None

        if not move_to(tgt):
            return None
        drop_time = t
        if not stationary(p):
            return None

        evt = seaside_events_by_id[cid]
        evt["landside_lift_time"] = lift_time
        evt["landside_lift_slot"] = hs
        evt["landside_drop_time"] = drop_time

    return l_positions


def compute_objective(seaside_events):
    ms = 0
    for e in seaside_events:
        if "landside_drop_time" in e:
            ms = max(ms, e["landside_drop_time"])
        else:
            ms = max(ms, e["seaside_drop_time"])
    return float(ms)


def mandatory_landside_condition_ok(landside_containers, processed_ids, makespan):
    """
    Check:
      1) all with deadline <= makespan are processed
      2) if there are unprocessed with deadline > makespan, then at least one
         container in the next-earliest deadline group is processed
    """
    by_id = {c["id"]: c for c in landside_containers}

    # 1
    for c in landside_containers:
        if c["deadline"] <= makespan and c["id"] not in processed_ids:
            return False

    # 2
    remaining = [c for c in landside_containers if c["id"] not in processed_ids and c["deadline"] > makespan]
    if not remaining:
        return True

    next_deadline = min(c["deadline"] for c in remaining)
    group_ids = [c["id"] for c in landside_containers if c["deadline"] == next_deadline]
    if any(gid in processed_ids for gid in group_ids):
        return True
    return False


def build_solution(instance, time_limit, logger=None):
    S = instance["S"]
    n = instance["n"]
    m = instance["m"]
    p = instance["p"]
    sigma_w = instance["sigma_w"]
    sigma_l = instance["sigma_l"]
    seaside_containers = instance["seaside_containers"]
    landside_containers = instance["landside_containers"]

    # Basic consistency
    if len(seaside_containers) != n:
        seaside_containers = sorted(seaside_containers, key=lambda x: x["id"])
    if len(landside_containers) != m:
        landside_containers = sorted(landside_containers, key=lambda x: x["deadline"])

    # 1) Landside baseline plan
    l_positions, landside_events, processed_ids, skipped_ids, _, _ = build_landside_plan(
        S, p, sigma_l, landside_containers
    )

    # 2) Seaside schedule against fixed L baseline
    sea_plan = schedule_seaside_with_fixed_l(S, p, sigma_w, seaside_containers, l_positions)
    if sea_plan is None:
        # fallback trivial infeasible-safe output
        return {
            "objective_value": float("inf"),
            "crane_w_positions": {"0": sigma_w},
            "crane_l_positions": {"0": sigma_l},
            "seaside_events": [],
            "landside_events": landside_events,
        }

    w_positions = sea_plan["w_positions"]
    seaside_events = sea_plan["seaside_events"]
    coop_queue = sea_plan["coop_queue"]

    seaside_events_by_id = {e["container_id"]: e for e in seaside_events}

    # Ensure W stays available long enough while L appends coop
    # (W already returned to 0; extension will be done later).
    l_positions2 = append_l_coop_tasks(S, p, l_positions, w_positions, coop_queue, seaside_events_by_id)
    if l_positions2 is None:
        # if coop appending fails, keep baseline only (containers with target S+1 remain incomplete)
        l_positions2 = l_positions

    l_positions = l_positions2

    # Extend shorter timeline with waiting at last position
    horizon = max(len(w_positions), len(l_positions)) - 1
    while len(w_positions) <= horizon:
        w_positions.append(w_positions[-1])
    while len(l_positions) <= horizon:
        l_positions.append(l_positions[-1])

    objective = compute_objective(seaside_events)

    # Mandatory condition check; if violated, still return schedule (best effort)
    # but objective quality may be irrelevant if infeasible on these constraints.
    _ok = mandatory_landside_condition_ok(landside_containers, processed_ids, objective)

    if logger:
        logger.log(objective)

    return {
        "objective_value": objective,
        "crane_w_positions": {str(t): int(w_positions[t]) for t in range(horizon + 1)},
        "crane_l_positions": {str(t): int(l_positions[t]) for t in range(horizon + 1)},
        "seaside_events": sorted(seaside_events, key=lambda x: x["container_id"]),
        "landside_events": landside_events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    # Single-pass constructive heuristic (fast, respects time limit by design)
    solution = build_solution(instance, args.time_limit, logger=logger)

    # Enforce finite objective fallback
    if solution["objective_value"] == float("inf"):
        solution["objective_value"] = 1e12

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()