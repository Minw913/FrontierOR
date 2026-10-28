import json
import argparse
import time
from collections import deque

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path) as f:
        data = json.load(f)
    
    params = data["problem_parameters"]
    num_periods = params["num_periods"]
    containers_data = data["containers"]
    trades_data = data["trades"]
    
    # Build container info
    container_ids = []
    container_avail = {}  # id -> (start, end)
    adjacency = {}  # id -> set of adjacent ids
    
    for c in containers_data:
        cid = c["id"]
        container_ids.append(cid)
        container_avail[cid] = (c["availability_start"], c["availability_end"])
        adjacency[cid] = set(c["adjacent_containers"])
    
    container_id_set = set(container_ids)
    
    # Build trade info
    trades = []
    for t in trades_data:
        trades.append({
            "id": t["id"],
            "start": t["start_period"],
            "end": t["end_period"],
            "demand": t["container_demand"],
            "max_disp": t["max_dispersion"]
        })
    
    def containers_available(period):
        return set(cid for cid in container_ids 
                   if container_avail[cid][0] <= period <= container_avail[cid][1])
    
    def count_clusters(assigned_set):
        if not assigned_set:
            return 0
        visited = set()
        clusters = 0
        for start_node in assigned_set:
            if start_node in visited:
                continue
            clusters += 1
            queue = deque([start_node])
            visited.add(start_node)
            while queue:
                node = queue.popleft()
                for nb in adjacency[node]:
                    if nb in assigned_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        return clusters
    
    def is_connected(assigned_set, max_disp):
        """Check if assigned_set forms at most max_disp clusters."""
        if not assigned_set:
            return True
        visited = set()
        clusters = 0
        for start_node in assigned_set:
            if start_node in visited:
                continue
            clusters += 1
            if clusters > max_disp:
                return False
            queue = deque([start_node])
            visited.add(start_node)
            while queue:
                node = queue.popleft()
                for nb in adjacency[node]:
                    if nb in assigned_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        return True
    
    # Sort trades by start period, then by demand (larger first for priority)
    trades_sorted = sorted(trades, key=lambda t: (t["start"], -t["demand"]))
    
    # For each period, we need to know which trades are active
    period_trades = {}
    for p in range(1, num_periods + 1):
        period_trades[p] = []
    for t in trades:
        for p in range(t["start"], t["end"] + 1):
            if p in period_trades:
                period_trades[p].append(t)
    
    # assignments[trade_id][period] = set of container ids
    assignments = {t["id"]: {} for t in trades}
    
    def find_connected_group(seed, available, size, max_disp):
        """Find a set of 'size' containers from 'available' that forms at most max_disp clusters,
        starting from seed containers using BFS expansion."""
        if size == 0:
            return set()
        if size > len(available):
            return None
        
        # Strategy: grow clusters via BFS from seeds
        result = set()
        remaining = set(available)
        
        if max_disp >= size:
            # Each container can be its own cluster
            return set(list(available)[:size])
        
        # Try to grow connected clusters
        # Start from each seed and expand
        clusters_used = 0
        
        # Sort seeds by number of available neighbors (descending) for better connectivity
        seed_list = sorted(remaining, key=lambda c: len(adjacency[c] & remaining), reverse=True)
        
        for s in seed_list:
            if len(result) >= size:
                break
            if s in result:
                continue
            if clusters_used >= max_disp:
                # Try to expand existing clusters
                break
            # Start new cluster from s
            clusters_used += 1
            queue = deque([s])
            result.add(s)
            remaining.discard(s)
            while queue and len(result) < size:
                node = queue.popleft()
                neighbors = sorted(adjacency[node] & remaining, 
                                   key=lambda c: len(adjacency[c] & remaining), reverse=True)
                for nb in neighbors:
                    if len(result) >= size:
                        break
                    result.add(nb)
                    remaining.discard(nb)
                    queue.append(nb)
        
        if len(result) < size:
            # Try adding remaining containers if dispersion allows
            needed = size - len(result)
            for c in list(remaining):
                if needed <= 0:
                    break
                test = result | {c}
                if count_clusters(test) <= max_disp:
                    result.add(c)
                    remaining.discard(c)
                    needed -= 1
        
        if len(result) == size and count_clusters(result) <= max_disp:
            return result
        
        # Fallback: try all subsets (too expensive for large), just return what we have
        if len(result) >= size:
            result_list = list(result)[:size]
            result_set = set(result_list)
            if count_clusters(result_set) <= max_disp:
                return result_set
        
        return None
    
    def greedy_assign_period(period, prev_assignments):
        """Assign containers to trades for a given period, trying to reuse prev_assignments."""
        avail = containers_available(period)
        active_trades = period_trades.get(period, [])
        # Sort: trades with smaller max_disp first (harder constraint), then larger demand
        active_trades_sorted = sorted(active_trades, key=lambda t: (t["max_disp"], -t["demand"]))
        
        used = set()
        period_assignment = {}
        
        for t in active_trades_sorted:
            tid = t["id"]
            demand = t["demand"]
            max_disp = t["max_disp"]
            
            remaining_avail = avail - used
            
            # Try to reuse previous assignment
            prev = prev_assignments.get(tid, set())
            reusable = prev & remaining_avail
            
            if len(reusable) == demand and is_connected(reusable, max_disp):
                period_assignment[tid] = reusable
                used |= reusable
                continue
            
            # Try to keep as many from prev as possible and fill the rest
            best_assignment = None
            
            if reusable:
                # Keep reusable ones and try to add more
                needed = demand - len(reusable)
                if needed <= 0:
                    # Need to select subset of reusable
                    # Try subsets that are connected
                    candidate = set(list(reusable)[:demand])
                    if is_connected(candidate, max_disp):
                        best_assignment = candidate
                    else:
                        # BFS from reusable
                        best_assignment = find_connected_group(
                            reusable, remaining_avail, demand, max_disp)
                else:
                    # Expand from reusable
                    extra_avail = remaining_avail - reusable
                    result = set(reusable)
                    queue = deque(list(reusable))
                    while queue and len(result) < demand:
                        node = queue.popleft()
                        for nb in adjacency[node]:
                            if nb in extra_avail and nb not in result:
                                result.add(nb)
                                extra_avail.discard(nb)
                                queue.append(nb)
                                if len(result) >= demand:
                                    break
                    
                    if len(result) < demand:
                        # Add disconnected if dispersion allows
                        for c in list(extra_avail):
                            if len(result) >= demand:
                                break
                            result.add(c)
                            extra_avail.discard(c)
                    
                    if len(result) >= demand:
                        candidate = set(list(result)[:demand])
                        if is_connected(candidate, max_disp):
                            best_assignment = candidate
                        else:
                            # Try harder
                            best_assignment = find_connected_group(
                                reusable, remaining_avail, demand, max_disp)
            
            if best_assignment is None:
                best_assignment = find_connected_group(
                    set(), remaining_avail, demand, max_disp)
            
            if best_assignment is not None and len(best_assignment) == demand:
                period_assignment[tid] = best_assignment
                used |= best_assignment
            else:
                # Fallback: just pick any available containers
                picked = set(list(remaining_avail)[:demand])
                period_assignment[tid] = picked
                used |= picked
        
        return period_assignment
    
    # Phase 1: Greedy forward pass
    prev_assign = {}
    for p in range(1, num_periods + 1):
        pa = greedy_assign_period(p, prev_assign)
        for tid, cset in pa.items():
            assignments[tid][p] = cset
        prev_assign = pa
        
        if time.time() - start_time > time_limit * 0.4:
            break
    
    # Fill any missing periods (if we ran out of time above)
    for t in trades:
        tid = t["id"]
        for p in range(t["start"], t["end"] + 1):
            if p not in assignments[tid]:
                prev = assignments[tid].get(p - 1, set()) if p > t["start"] else set()
                pa = greedy_assign_period(p, {tid: prev})
                if tid in pa:
                    assignments[tid][p] = pa[tid]
                else:
                    assignments[tid][p] = set()
    
    def compute_objective():
        total = 0
        for t in trades:
            tid = t["id"]
            for p in range(t["start"] + 1, t["end"] + 1):
                prev_set = assignments[tid].get(p - 1, set())
                curr_set = assignments[tid].get(p, set())
                if curr_set != prev_set:
                    total += 1
        return total
    
    best_obj = compute_objective()
    if logger:
        logger.log(best_obj)
    
    # Phase 2: Try backward pass to improve continuity
    def try_backward_pass():
        nonlocal best_obj
        # For each trade, try to make assignments consistent (same set across periods)
        for t in trades:
            tid = t["id"]
            if t["end"] <= t["start"]:
                continue
            
            # Find the most common assignment
            from collections import Counter
            assignment_counts = Counter()
            for p in range(t["start"], t["end"] + 1):
                key = frozenset(assignments[tid].get(p, set()))
                assignment_counts[key] += 1
            
            most_common = assignment_counts.most_common(1)[0][0]
            target_set = set(most_common)
            
            if not target_set or len(target_set) != t["demand"]:
                continue
            
            if not is_connected(target_set, t["max_disp"]):
                continue
            
            # Try to assign target_set to all periods
            for p in range(t["start"], t["end"] + 1):
                current = assignments[tid].get(p, set())
                if current == target_set:
                    continue
                
                avail = containers_available(p)
                if not target_set.issubset(avail):
                    continue
                
                # Check no conflict with other trades in this period
                conflict = False
                for t2 in trades:
                    if t2["id"] == tid:
                        continue
                    if t2["start"] <= p <= t2["end"]:
                        other_set = assignments[t2["id"]].get(p, set())
                        if other_set & target_set:
                            conflict = True
                            break
                
                if not conflict:
                    assignments[tid][p] = target_set
            
            if time.time() - start_time > time_limit * 0.7:
                break
        
        new_obj = compute_objective()
        if new_obj < best_obj:
            best_obj = new_obj
            if logger:
                logger.log(best_obj)
    
    try_backward_pass()
    
    # Phase 3: Local search - swap assignments between periods to reduce changes
    def local_search():
        nonlocal best_obj
        improved = True
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            iteration += 1
            
            for t in trades:
                tid = t["id"]
                periods = list(range(t["start"], t["end"] + 1))
                
                for i in range(1, len(periods)):
                    if time.time() - start_time > time_limit * 0.95:
                        return
                    
                    p = periods[i]
                    pp = periods[i - 1]
                    
                    curr = assignments[tid].get(p, set())
                    prev = assignments[tid].get(pp, set())
                    
                    if curr == prev:
                        continue
                    
                    # Try to change current to match previous
                    avail = containers_available(p)
                    if not prev.issubset(avail):
                        continue
                    if not is_connected(prev, t["max_disp"]):
                        continue
                    
                    # Check conflicts
                    # What other trades use containers in prev in period p?
                    containers_to_free = prev - curr
                    containers_to_release = curr - prev
                    
                    conflict = False
                    affected_trades = {}
                    
                    for c in containers_to_free:
                        # Find who uses c in period p
                        for t2 in trades:
                            if t2["id"] == tid:
                                continue
                            if t2["start"] <= p <= t2["end"]:
                                t2_set = assignments[t2["id"]].get(p, set())
                                if c in t2_set:
                                    # Can we swap c out and give a released container?
                                    if t2["id"] not in affected_trades:
                                        affected_trades[t2["id"]] = {"remove": set(), "t": t2}
                                    affected_trades[t2["id"]]["remove"].add(c)
                    
                    if not affected_trades:
                        # No conflicts, just assign prev to current
                        old_curr = assignments[tid][p]
                        next_set = assignments[tid].get(p + 1 if p + 1 <= t["end"] else None, None) if i + 1 < len(periods) else None
                        
                        old_changes = (1 if curr != prev else 0)
                        if next_set is not None:
                            old_changes += (1 if curr != next_set else 0)
                        
                        new_changes = 0  # prev == prev so 0
                        if next_set is not None:
                            new_changes += (1 if prev != next_set else 0)
                        
                        if new_changes < old_changes:
                            assignments[tid][p] = set(prev)
                            new_obj = best_obj - (old_changes - new_changes)
                            best_obj = new_obj
                            if logger:
                                logger.log(best_obj)
                            improved = True
                    elif len(affected_trades) == 1:
                        # Try simple swap
                        at_id, at_info = list(affected_trades.items())[0]
                        at = at_info["t"]
                        remove_from_other = at_info["remove"]
                        
                        # Give released containers to the other trade
                        other_set = assignments[at_id].get(p, set())
                        new_other = (other_set - remove_from_other) | (containers_to_release & avail)
                        
                        if len(new_other) != at["demand"]:
                            continue
                        
                        if not is_connected(new_other, at["max_disp"]):
                            continue
                        
                        # Calculate change in objective
                        # For tid
                        old_tid_change_here = (1 if curr != prev else 0)
                        new_tid_change_here = 0  # now matches prev
                        
                        next_p = periods[i + 1] if i + 1 < len(periods) else None
                        if next_p:
                            next_set = assignments[tid].get(next_p, set())
                            old_tid_change_here += (1 if curr != next_set else 0)
                            new_tid_change_here += (1 if prev != next_set else 0)
                        
                        # For affected trade
                        other_prev_p = None
                        other_next_p = None
                        for pp2 in range(at["start"], at["end"] + 1):
                            if pp2 == p:
                                continue
                            if pp2 < p:
                                other_prev_p = pp2
                            elif pp2 > p and other_next_p is None:
                                other_next_p = pp2
                        
                        old_other_changes = 0
                        new_other_changes = 0
                        
                        if other_prev_p is not None:
                            opset = assignments[at_id].get(other_prev_p, set())
                            old_other_changes += (1 if other_set != opset else 0)
                            new_other_changes += (1 if new_other != opset else 0)
                        
                        if other_next_p is not None:
                            onset = assignments[at_id].get(other_next_p, set())
                            old_other_changes += (1 if other_set != onset else 0)
                            new_other_changes += (1 if new_other != onset else 0)
                        
                        delta = (new_tid_change_here + new_other_changes) - (old_tid_change_here + old_other_changes)
                        
                        if delta < 0:
                            assignments[tid][p] = set(prev)
                            assignments[at_id][p] = new_other
                            best_obj += delta
                            if logger:
                                logger.log(best_obj)
                            improved = True
    
    local_search()
    
    # Phase 4: Try propagating forward - for each trade, try to extend each assignment as long as possible
    def propagate_forward():
        nonlocal best_obj
        for t in trades:
            tid = t["id"]
            periods = list(range(t["start"], t["end"] + 1))
            
            if time.time() - start_time > time_limit * 0.98:
                return
            
            i = 0
            while i < len(periods):
                if time.time() - start_time > time_limit * 0.98:
                    return
                    
                p = periods[i]
                curr = assignments[tid].get(p, set())
                
                # Try to extend curr to subsequent periods
                for j in range(i + 1, len(periods)):
                    np = periods[j]
                    next_curr = assignments[tid].get(np, set())
                    
                    if next_curr == curr:
                        continue
                    
                    avail = containers_available(np)
                    if not curr.issubset(avail):
                        break
                    
                    # Check conflicts
                    conflict = False
                    for c in curr - next_curr:
                        for t2 in trades:
                            if t2["id"] == tid:
                                continue
                            if t2["start"] <= np <= t2["end"]:
                                if c in assignments[t2["id"]].get(np, set()):
                                    conflict = True
                                    break
                        if conflict:
                            break
                    
                    if conflict:
                        break
                    
                    # Safe to assign
                    old_changes = 0
                    new_changes = 0
                    
                    prev_np = assignments[tid].get(periods[j-1], set()) if j > 0 else set()
                    old_changes += (1 if next_curr != prev_np else 0)
                    new_changes += (1 if curr != prev_np else 0)
                    
                    if j + 1 < len(periods):
                        next_np = assignments[tid].get(periods[j+1], set())
                        old_changes += (1 if next_curr != next_np else 0)
                        new_changes += (1 if curr != next_np else 0)
                    
                    if new_changes <= old_changes:
                        assignments[tid][np] = set(curr)
                        delta = new_changes - old_changes
                        if delta < 0:
                            best_obj += delta
                            if logger:
                                logger.log(best_obj)
                    else:
                        break
                
                i += 1
    
    propagate_forward()
    
    # Recompute objective to be safe
    best_obj = compute_objective()
    if logger:
        logger.log(best_obj)
    
    # Build output
    output_assignments = {}
    for t in trades:
        tid = t["id"]
        trade_dict = {}
        for p in range(t["start"], t["end"] + 1):
            cset = assignments[tid].get(p, set())
            trade_dict[str(p)] = sorted(list(cset))
        output_assignments[str(tid)] = trade_dict
    
    solution = {
        "objective_value": best_obj,
        "assignments": output_assignments
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)