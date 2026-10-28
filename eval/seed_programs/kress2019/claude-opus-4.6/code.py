import json
import argparse
import time
import copy

def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    S = instance["S"]
    n = instance["n"]
    m = instance["m"]
    p = instance["p"]
    sigma_w = instance["sigma_w"]
    sigma_l = instance["sigma_l"]
    seaside_containers = sorted(instance["seaside_containers"], key=lambda c: c["id"])
    landside_containers = sorted(instance["landside_containers"], key=lambda c: c["deadline"])
    
    last_slot = S + 1
    T_MAX = max(3000, (n + m) * (S + 2 + 2 * p) * 4)
    
    def simulate_schedule(plans, l_task_order):
        """
        plans: list of dicts per seaside container:
            {container_id, target_slot, cooperative, intermediate_slot}
        l_task_order: ordered list of tasks for landside crane
            each: {type: 'coop'/'land', container_id, pickup_slot, delivery_slot, 
                   earliest_finish_time (for land), deadline (for land)}
        
        Returns (makespan, seaside_events, landside_events, w_positions, l_positions, final_t) or None
        """
        w_pos = sigma_w
        l_pos = sigma_l
        
        # Build operation sequences for each crane
        # W crane ops: for each seaside container: move_to_0, lift_at_0, move_to_drop, drop_at_drop
        # L crane ops: for each task: move_to_pickup, lift_at_pickup, move_to_delivery, drop_at_delivery
        
        # W crane operation list
        w_ops = []
        for pl in plans:
            drop_slot = pl["intermediate_slot"] if pl["cooperative"] else pl["target_slot"]
            w_ops.append(("move", 0))
            w_ops.append(("lift", 0, pl["container_id"]))
            w_ops.append(("move", drop_slot))
            w_ops.append(("drop", drop_slot, pl["container_id"]))
        
        # L crane operation list
        l_ops = []
        for task in l_task_order:
            l_ops.append(("move", task["pickup_slot"]))
            l_ops.append(("lift", task["pickup_slot"], task["container_id"], task["type"]))
            l_ops.append(("move", task["delivery_slot"]))
            l_ops.append(("drop", task["delivery_slot"], task["container_id"], task["type"]))
        
        w_op_idx = 0
        l_op_idx = 0
        
        w_busy_until = -1  # time step until which w is busy (inclusive)
        l_busy_until = -1
        
        w_positions = {}
        l_positions = {}
        
        # Track events
        seaside_drop_starts = {}
        seaside_drop_slots = {}
        seaside_lift_times = {}
        landside_lift_starts = {}
        landside_drop_starts = {}
        
        last_w_drop_start = -1  # ordering constraint
        
        # Coop dependency: landside lift of coop container can only start after seaside drop starts
        
        for t in range(T_MAX + 1):
            if time.time() - start_time > time_limit - 1:
                return None
            
            w_positions[t] = w_pos
            l_positions[t] = l_pos
            
            # Check if both cranes are done
            if w_op_idx >= len(w_ops) and l_op_idx >= len(l_ops) and t > w_busy_until and t > l_busy_until:
                # Compute makespan
                makespan = 0
                for pl in plans:
                    cid = pl["container_id"]
                    if not pl["cooperative"]:
                        if cid in seaside_drop_starts:
                            makespan = max(makespan, seaside_drop_starts[cid] + p)
                    else:
                        if cid in landside_drop_starts:
                            makespan = max(makespan, landside_drop_starts[cid] + p)
                
                # Check landside deadlines
                for task in l_task_order:
                    if task["type"] == "land":
                        cid = task["container_id"]
                        if cid in landside_drop_starts:
                            drop_finish = landside_drop_starts[cid] + p
                            if drop_finish > task.get("deadline", float('inf')):
                                return None
                            if drop_finish < task.get("earliest_finish_time", 0):
                                return None
                
                # Build events
                s_events = []
                for pl in plans:
                    cid = pl["container_id"]
                    ev = {
                        "container_id": cid,
                        "target_slot": pl["target_slot"],
                        "seaside_drop_time": seaside_drop_starts.get(cid, 0),
                        "seaside_drop_slot": seaside_drop_slots.get(cid, pl["target_slot"]),
                    }
                    if pl["cooperative"]:
                        ev["landside_lift_time"] = landside_lift_starts.get(cid)
                        ev["landside_lift_slot"] = pl["intermediate_slot"]
                        ev["landside_drop_time"] = landside_drop_starts.get(cid)
                    s_events.append(ev)
                
                l_events = []
                for task in l_task_order:
                    if task["type"] == "land":
                        cid = task["container_id"]
                        if cid in landside_lift_starts:
                            l_events.append({
                                "container_id": cid,
                                "source_slot": task["pickup_slot"],
                                "lift_time": landside_lift_starts[cid],
                                "drop_time": landside_drop_starts[cid]
                            })
                
                return (makespan, s_events, l_events, w_positions, l_positions, t)
            
            # ---- Process W crane ----
            w_did_something = False
            if t <= w_busy_until:
                w_did_something = True  # still busy with lift/drop
            elif w_op_idx < len(w_ops):
                op = w_ops[w_op_idx]
                if op[0] == "move":
                    target = op[1]
                    if w_pos == target:
                        w_op_idx += 1
                        # Don't set w_did_something, try next op this same tick
                    else:
                        desired = w_pos + (1 if target > w_pos else -1)
                        # Collision: w must stay < l_pos
                        if desired < l_pos:
                            w_pos = desired
                        # else wait
                        w_did_something = True
                
                # Try again if we just finished a move (w_pos == target)
                if not w_did_something and w_op_idx < len(w_ops):
                    op = w_ops[w_op_idx]
                    if op[0] == "lift":
                        slot = op[1]
                        cid = op[2]
                        if w_pos == slot:
                            w_busy_until = t + p - 1
                            seaside_lift_times[cid] = t
                            w_op_idx += 1
                            w_did_something = True
                        else:
                            w_did_something = True  # wait (shouldn't happen)
                    elif op[0] == "drop":
                        slot = op[1]
                        cid = op[2]
                        if w_pos == slot and t >= last_w_drop_start:
                            w_busy_until = t + p - 1
                            seaside_drop_starts[cid] = t
                            seaside_drop_slots[cid] = slot
                            last_w_drop_start = t
                            w_op_idx += 1
                            w_did_something = True
                        else:
                            w_did_something = True  # wait
                    elif op[0] == "move":
                        # Another move after move completion
                        target = op[1]
                        if w_pos == target:
                            w_op_idx += 1
                        else:
                            desired = w_pos + (1 if target > w_pos else -1)
                            if desired < l_pos:
                                w_pos = desired
                            w_did_something = True
            
            # ---- Process L crane ----
            l_did_something = False
            if t <= l_busy_until:
                l_did_something = True
            elif l_op_idx < len(l_ops):
                op = l_ops[l_op_idx]
                if op[0] == "move":
                    target = op[1]
                    if l_pos == target:
                        l_op_idx += 1
                    else:
                        desired = l_pos + (1 if target > l_pos else -1)
                        # Collision: l must stay > w_pos
                        if desired > w_pos:
                            l_pos = desired
                        l_did_something = True
                
                if not l_did_something and l_op_idx < len(l_ops):
                    op = l_ops[l_op_idx]
                    if op[0] == "lift":
                        slot = op[1]
                        cid = op[2]
                        task_type = op[3]
                        can_lift = True
                        if task_type == "coop":
                            if cid not in seaside_drop_starts or t < seaside_drop_starts[cid]:
                                can_lift = False
                        if can_lift and l_pos == slot:
                            l_busy_until = t + p - 1
                            landside_lift_starts[cid] = t
                            l_op_idx += 1
                            l_did_something = True
                        else:
                            l_did_something = True  # wait
                    elif op[0] == "drop":
                        slot = op[1]
                        cid = op[2]
                        task_type = op[3]
                        can_drop = True
                        if task_type == "land":
                            # Find the task to check earliest_finish_time
                            for task in l_task_order:
                                if task["container_id"] == cid and task["type"] == "land":
                                    eft = task.get("earliest_finish_time", 0)
                                    if t + p < eft:
                                        can_drop = False
                                    break
                        if can_drop and l_pos == slot:
                            l_busy_until = t + p - 1
                            landside_drop_starts[cid] = t
                            l_op_idx += 1
                            l_did_something = True
                        else:
                            l_did_something = True  # wait
                    elif op[0] == "move":
                        target = op[1]
                        if l_pos == target:
                            l_op_idx += 1
                        else:
                            desired = l_pos + (1 if target > l_pos else -1)
                            if desired > w_pos:
                                l_pos = desired
                            l_did_something = True
            
            # Update positions after potential movement
            w_positions[t] = w_pos
            l_positions[t] = l_pos
            
            # Safety check
            if w_pos >= l_pos:
                return None
        
        return None  # exceeded T_MAX
    
    # Determine which landside containers to process
    def get_required_landside(makespan_est):
        required = []
        for lc in landside_containers:
            if lc["deadline"] <= makespan_est:
                required.append(lc)
        remaining = [lc for lc in landside_containers if lc["deadline"] > makespan_est]
        if remaining:
            next_deadline = remaining[0]["deadline"]
            next_group = [lc for lc in remaining if lc["deadline"] == next_deadline]
            if next_group:
                required.append(next_group[0])
        return required
    
    best_result = None
    best_makespan = float('inf')
    
    # Try various strategies
    def try_config(coop_flags, intermediate_slots_map):
        nonlocal best_result, best_makespan
        
        if time.time() - start_time > time_limit - 2:
            return
        
        plans = []
        for i, sc in enumerate(seaside_containers):
            coop = coop_flags[i]
            target = sc["target_slot"]
            inter = intermediate_slots_map.get(i, target)
            plans.append({
                "container_id": sc["id"],
                "target_slot": target,
                "cooperative": coop,
                "intermediate_slot": inter if coop else None
            })
        
        # Estimate makespan for landside container selection
        est = 0
        pos = sigma_w
        for i, sc in enumerate(seaside_containers):
            target = sc["target_slot"]
            drop = intermediate_slots_map.get(i, target) if coop_flags[i] else target
            est += abs(pos - 0) + p + abs(0 - drop) + p
            pos = drop
        est_makespan = est
        
        req_land = get_required_landside(max(est_makespan, best_makespan if best_makespan < float('inf') else est_makespan))
        
        # Build landside task order
        coop_tasks = []
        for i, sc in enumerate(seaside_containers):
            if coop_flags[i]:
                inter = intermediate_slots_map.get(i, sc["target_slot"])
                coop_tasks.append({
                    "type": "coop",
                    "container_id": sc["id"],
                    "pickup_slot": inter,
                    "delivery_slot": sc["target_slot"],
                })
        
        land_tasks = []
        for lc in req_land:
            land_tasks.append({
                "type": "land",
                "container_id": lc["id"],
                "pickup_slot": lc["source_slot"],
                "delivery_slot": last_slot,
                "earliest_finish_time": lc["earliest_finish_time"],
                "deadline": lc["deadline"],
            })
        
        all_l_tasks = coop_tasks + land_tasks
        
        result = simulate_schedule(plans, all_l_tasks)
        if result is not None:
            ms = result[0]
            if ms < best_makespan:
                best_makespan = ms
                best_result = result
                if logger:
                    logger.log(best_makespan)
                
                # Re-check required landside with actual makespan
                req_land2 = get_required_landside(ms)
                if len(req_land2) != len(req_land):
                    land_tasks2 = []
                    for lc in req_land2:
                        land_tasks2.append({
                            "type": "land",
                            "container_id": lc["id"],
                            "pickup_slot": lc["source_slot"],
                            "delivery_slot": last_slot,
                            "earliest_finish_time": lc["earliest_finish_time"],
                            "deadline": lc["deadline"],
                        })
                    all_l_tasks2 = coop_tasks + land_tasks2
                    result2 = simulate_schedule(plans, all_l_tasks2)
                    if result2 is not None:
                        ms2 = result2[0]
                        if ms2 < best_makespan:
                            best_makespan = ms2
                            best_result = result2
                            if logger:
                                logger.log(best_makespan)
        
        # Also try land tasks first, then coop tasks
        if coop_tasks and land_tasks:
            all_l_tasks_rev = land_tasks + coop_tasks
            result3 = simulate_schedule(plans, all_l_tasks_rev)
            if result3 is not None:
                ms3 = result3[0]
                if ms3 < best_makespan:
                    best_makespan = ms3
                    best_result = result3
                    if logger:
                        logger.log(best_makespan)
    
    # Strategy 1: No cooperation at all
    if all(sc["target_slot"] < last_slot for sc in seaside_containers):
        coop_flags = [False] * n
        try_config(coop_flags, {})
    
    # Strategy 2: Cooperate for containers targeting last_slot
    coop_flags2 = [False] * n
    inter_map2 = {}
    for i, sc in enumerate(seaside_containers):
        if sc["target_slot"] == last_slot:
            coop_flags2[i] = True
            inter_map2[i] = max(1, last_slot - 2)
    try_config(coop_flags2, inter_map2)
    
    # Strategy 3: Cooperate for containers with target > S/2
    if time.time() - start_time < time_limit - 3:
        for threshold in [S * 3 // 4, S // 2, S // 4, 1]:
            if time.time() - start_time > time_limit - 2:
                break
            coop_flags3 = [False] * n
            inter_map3 = {}
            for i, sc in enumerate(seaside_containers):
                if sc["target_slot"] >= threshold:
                    coop_flags3[i] = True
                    inter_map3[i] = max(1, min(threshold - 1, sc["target_slot"] - 1, S))
            if any(coop_flags3):
                try_config(coop_flags3, inter_map3)
    
    # Strategy 4: Try various intermediate slots for cooperative containers
    if time.time() - start_time < time_limit - 3:
        for i, sc in enumerate(seaside_containers):
            if time.time() - start_time > time_limit - 2:
                break
            target = sc["target_slot"]
            if target <= 1:
                continue
            
            # Try making just this container cooperative
            for inter in [max(1, target // 2), max(1, target - 1), max(1, min(target, S))]:
                if time.time() - start_time > time_limit - 2:
                    break
                if inter == target and target <= S:
                    continue  # pointless cooperation
                coop_f = [False] * n
                coop_f[i] = True
                im = {i: inter}
                try_config(coop_f, im)
    
    # Strategy 5: All cooperative
    if time.time() - start_time < time_limit - 3:
        coop_all = [True] * n
        im_all = {}
        for i, sc in enumerate(seaside_containers):
            target = sc["target_slot"]
            im_all[i] = max(1, min(target, S // 2 + 1))
        try_config(coop_all, im_all)
    
    if best_result is None:
        # Try very basic: no landside, no cooperation, handle each seaside container simply
        # Ensure targets are valid
        plans_basic = []
        for sc in seaside_containers:
            target = sc["target_slot"]
            if target >= last_slot:
                # Must cooperate
                plans_basic.append({
                    "container_id": sc["id"],
                    "target_slot": target,
                    "cooperative": True,
                    "intermediate_slot": max(1, S)
                })
            else:
                plans_basic.append({
                    "container_id": sc["id"],
                    "target_slot": target,
                    "cooperative": False,
                    "intermediate_slot": None
                })
        
        coop_tasks_b = []
        for pl in plans_basic:
            if pl["cooperative"]:
                coop_tasks_b.append({
                    "type": "coop",
                    "container_id": pl["container_id"],
                    "pickup_slot": pl["intermediate_slot"],
                    "delivery_slot": pl["target_slot"],
                })
        
        result_b = simulate_schedule(plans_basic, coop_tasks_b)
        if result_b is not None:
            best_makespan = result_b[0]
            best_result = result_b
            if logger:
                logger.log(best_makespan)
    
    if best_result is None:
        # Absolute fallback
        w_pos_out = {str(t): sigma_w for t in range(100)}
        l_pos_out = {str(t): sigma_l for t in range(100)}
        return {
            "objective_value": 100.0,
            "crane_w_positions": w_pos_out,
            "crane_l_positions": l_pos_out,
            "seaside_events": [],
            "landside_events": []
        }
    
    makespan, s_ev, l_ev, w_pos_dict, l_pos_dict, final_t = best_result
    
    crane_w = {}
    crane_l = {}
    for t in range(final_t + 1):
        crane_w[str(t)] = w_pos_dict.get(t, sigma_w)
        crane_l[str(t)] = l_pos_dict.get(t, sigma_l)
    
    return {
        "objective_value": float(makespan),
        "crane_w_positions": crane_w,
        "crane_l_positions": crane_l,
        "seaside_events": s_ev,
        "landside_events": l_ev
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()