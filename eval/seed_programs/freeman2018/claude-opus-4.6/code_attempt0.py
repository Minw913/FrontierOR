import argparse
import json
import math
import time
import itertools
import random
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    num_sites = data['num_sites']
    num_customers = data['num_customers']
    num_periods = data['num_periods']
    params = data['parameters']
    T = params['T']
    delta = params['delta']
    epsilon = params['epsilon']
    L = params['L']
    c = params['c']
    q = params['q']
    weekly_cycle = params['weekly_attraction_cycle']
    avg_biz_cust = params['avg_businesses_customers']
    avg_biz_sites = params['avg_businesses_sites']
    
    sites = data['sites']
    customers = data['customers']
    site_to_site = data['site_to_site_distances']
    cust_to_site = data['customer_to_site_distances']
    
    # Parse distances
    def get_ss_dist(i, j):
        key = f"{i}_{j}"
        return site_to_site[key]
    
    def get_cs_dist(c_id, s_id):
        key = f"{c_id}_{s_id}"
        return cust_to_site[key]
    
    # Precompute site data
    site_by_id = {}
    for s in sites:
        site_by_id[s['id']] = s
    
    cust_by_id = {}
    for cu in customers:
        cust_by_id[cu['id']] = cu
    
    site_ids = [s['id'] for s in sites]
    cust_ids = [cu['id'] for cu in customers]
    
    # Precompute attraction values for sites per period
    # Site attraction = num_businesses / avg_businesses_sites
    site_attraction_base = {}
    for s in sites:
        site_attraction_base[s['id']] = s['num_businesses'] / avg_biz_sites if avg_biz_sites > 0 else 1.0
    
    # Self-attraction for customers per period
    # self_attraction = num_businesses / avg_businesses_customers (strictly positive)
    cust_self_attraction = {}
    for cu in customers:
        cust_self_attraction[cu['id']] = cu['num_businesses'] / avg_biz_cust if avg_biz_cust > 0 else 1.0
    
    # Weekly attraction factor for each period (0-indexed period maps to day of week)
    def get_weekly_factor(t):
        # t is 1-indexed period
        return weekly_cycle[(t - 1) % 7]
    
    # Precompute attraction of customer c to site s in period t
    # = site_attraction_base[s] * weekly_factor[t] / dist(c, s)^q
    # Distance from customer to site
    cs_dist_cache = {}
    for cu_id in cust_ids:
        for s_id in site_ids:
            d = get_cs_dist(cu_id, s_id)
            cs_dist_cache[(cu_id, s_id)] = d
    
    def attraction(cu_id, s_id, t):
        d = cs_dist_cache[(cu_id, s_id)]
        if d <= 0:
            d = 0.001  # avoid division by zero
        return site_attraction_base[s_id] * get_weekly_factor(t) / (d ** q)
    
    # Precompute all attraction values
    attr_cache = {}
    for cu_id in cust_ids:
        for s_id in site_ids:
            for t in range(1, T + 1):
                attr_cache[(cu_id, s_id, t)] = attraction(cu_id, s_id, t)
    
    # Precompute ss distances
    ss_dist_cache = {}
    for i in [0] + site_ids:
        for j in site_ids:
            ss_dist_cache[(i, j)] = get_ss_dist(i, j)
    
    # Check which sites are reachable from which (within L distance)
    reachable = {}
    for i in [0] + site_ids:
        reachable[i] = []
        for j in site_ids:
            if ss_dist_cache[(i, j)] <= L:
                reachable[i].append(j)
    
    def compute_attendance(tour, events):
        """
        tour: dict period -> site_id
        events: dict period -> 0/1
        Returns: dict (site_id, period) -> attendance, and total profit
        """
        # Collect event list: (site, period) pairs
        event_list = []
        for t in range(1, T + 1):
            if events.get(t, 0) == 1:
                event_list.append((tour[t], t))
        
        attendance = {}
        total_revenue = 0.0
        total_fixed_cost = 0.0
        total_travel_cost = 0.0
        
        # Compute attendance for each event
        for (s_id, t) in event_list:
            site = site_by_id[s_id]
            total_att = 0.0
            
            for cu_id in cust_ids:
                cu = cust_by_id[cu_id]
                pop = cu['population']
                self_attr = cust_self_attraction[cu_id]
                
                # Numerator: attraction to this event
                numerator = attr_cache[(cu_id, s_id, t)]
                
                # Denominator: self_attraction + sum of attractions to all events within epsilon
                denom = self_attr
                for (s2, t2) in event_list:
                    if abs(t2 - t) <= epsilon:
                        # This event is within the shadow window (including self)
                        pass
                    else:
                        continue
                    denom += attr_cache[(cu_id, s2, t2)]
                
                # Wait, need to include ALL events within epsilon of t, not just this one
                # Let me recalculate: denominator includes self_attraction plus
                # attraction to all events within epsilon periods of t
                # Actually the loop above does that - for all events (s2,t2) where |t2-t| <= epsilon
                
                proportion = numerator / denom if denom > 0 else 0
                proportion = min(proportion, 1.0)
                total_att += pop * proportion
            
            total_att = min(total_att, site['capacity'])
            attendance[(s_id, t)] = total_att
            total_revenue += site['revenue_per_customer'] * total_att
            total_fixed_cost += site['fixed_cost']
        
        # Travel cost
        # Period 1: from dummy (0) to tour[1]
        prev = 0
        for t in range(1, T + 1):
            curr = tour[t]
            total_travel_cost += c * ss_dist_cache[(prev, curr)]
            prev = curr
        
        profit = total_revenue - total_fixed_cost - total_travel_cost
        return attendance, profit
    
    def compute_attendance_fast(tour, events):
        """Optimized version that precomputes event windows."""
        event_list = []
        event_by_period = {}
        for t in range(1, T + 1):
            if events.get(t, 0) == 1:
                event_list.append((tour[t], t))
                event_by_period[t] = tour[t]
        
        # Precompute for each period t, which events are within epsilon
        events_in_window = {}
        for t in range(1, T + 1):
            window = []
            for t2 in range(max(1, t - epsilon), min(T, t + epsilon) + 1):
                if t2 in event_by_period:
                    window.append((event_by_period[t2], t2))
            events_in_window[t] = window
        
        attendance = {}
        total_revenue = 0.0
        total_fixed_cost = 0.0
        total_travel_cost = 0.0
        
        for (s_id, t) in event_list:
            site = site_by_id[s_id]
            total_att = 0.0
            window_events = events_in_window[t]
            
            for cu_id in cust_ids:
                cu = cust_by_id[cu_id]
                pop = cu['population']
                self_attr = cust_self_attraction[cu_id]
                
                numerator = attr_cache[(cu_id, s_id, t)]
                
                denom = self_attr
                for (s2, t2) in window_events:
                    denom += attr_cache[(cu_id, s2, t2)]
                
                proportion = numerator / denom if denom > 0 else 0
                proportion = min(proportion, 1.0)
                total_att += pop * proportion
            
            total_att = min(total_att, site['capacity'])
            attendance[(s_id, t)] = total_att
            total_revenue += site['revenue_per_customer'] * total_att
            total_fixed_cost += site['fixed_cost']
        
        prev = 0
        for t in range(1, T + 1):
            curr = tour[t]
            total_travel_cost += c * ss_dist_cache[(prev, curr)]
            prev = curr
        
        profit = total_revenue - total_fixed_cost - total_travel_cost
        return attendance, profit
    
    def check_consecutive_constraint(events):
        """Check delta constraint: for every window of delta+2 consecutive periods,
        at most delta events."""
        for t_start in range(1, T - delta):  # t from 1 to T-delta-1
            window_end = t_start + delta + 1
            count = 0
            for t2 in range(t_start, min(window_end + 1, T + 1)):
                if events.get(t2, 0) == 1:
                    count += 1
            if count > delta:
                return False
        return True
    
    def check_max_events(tour, events):
        """Check per-site max events constraint."""
        site_event_count = {}
        for t in range(1, T + 1):
            if events.get(t, 0) == 1:
                s = tour[t]
                site_event_count[s] = site_event_count.get(s, 0) + 1
                if site_event_count[s] > site_by_id[s]['max_events']:
                    return False
        return True
    
    # Greedy heuristic approach
    # Strategy: Try to visit high-revenue sites and hold events there
    
    # Estimate site profitability
    def estimate_site_value(s_id):
        site = site_by_id[s_id]
        # Rough estimate: revenue * capacity - fixed_cost
        # Also consider customer attraction
        total_potential = 0
        for cu_id in cust_ids:
            cu = cust_by_id[cu_id]
            d = cs_dist_cache[(cu_id, s_id)]
            if d <= 0:
                d = 0.001
            # Use average weekly factor
            avg_factor = sum(weekly_cycle) / 7.0
            attr_val = site_attraction_base[s_id] * avg_factor / (d ** q)
            self_attr = cust_self_attraction[cu_id]
            prop = attr_val / (self_attr + attr_val)
            total_potential += cu['population'] * prop
        
        total_potential = min(total_potential, site['capacity'])
        value = site['revenue_per_customer'] * total_potential - site['fixed_cost']
        return value, total_potential
    
    site_values = {}
    for s_id in site_ids:
        val, pot = estimate_site_value(s_id)
        site_values[s_id] = (val, pot)
    
    # Sort sites by value
    sorted_sites = sorted(site_ids, key=lambda s: site_values[s][0], reverse=True)
    
    # Simple greedy: stay at the best site and hold events respecting constraints
    best_profit = float('-inf')
    best_tour = None
    best_events = None
    best_attendance = None
    
    # Try a few strategies
    def try_single_site_strategy(target_site):
        """Stay at one site and hold as many events as possible."""
        # Check reachability from dummy
        if target_site not in reachable[0]:
            return None, None, None, float('-inf')
        
        tour = {}
        events = {}
        for t in range(1, T + 1):
            tour[t] = target_site
            events[t] = 0
        
        # Schedule events respecting delta constraint and max_events
        site = site_by_id[target_site]
        max_ev = site['max_events']
        
        # Prefer weekend days (higher weekly factor)
        period_factors = [(t, get_weekly_factor(t)) for t in range(1, T + 1)]
        period_factors.sort(key=lambda x: x[1], reverse=True)
        
        event_count = 0
        for t, factor in period_factors:
            if event_count >= max_ev:
                break
            events[t] = 1
            if not check_consecutive_constraint(events):
                events[t] = 0
            else:
                event_count += 1
        
        att, profit = compute_attendance_fast(tour, events)
        return tour, events, att, profit
    
    def try_multi_site_strategy(site_sequence):
        """Visit sites in sequence, holding events at each."""
        tour = {}
        events = {}
        
        # Build tour: travel to each site in sequence
        t = 1
        prev = 0
        site_event_counts = {s: 0 for s in site_ids}
        
        for target in site_sequence:
            if t > T:
                break
            # Check reachability
            if target not in reachable[prev]:
                # Try to find path through intermediate sites
                # For simplicity, skip this target
                continue
            
            tour[t] = target
            events[t] = 0
            prev = target
            t += 1
        
        # Fill remaining periods staying at last site
        last_site = prev if prev != 0 else sorted_sites[0]
        while t <= T:
            tour[t] = last_site
            events[t] = 0
            t += 1
        
        # Schedule events
        for t in range(1, T + 1):
            s = tour[t]
            site = site_by_id[s]
            if site_event_counts[s] < site['max_events']:
                events[t] = 1
                if not check_consecutive_constraint(events):
                    events[t] = 0
                else:
                    site_event_counts[s] += 1
        
        att, profit = compute_attendance_fast(tour, events)
        return tour, events, att, profit
    
    # Strategy 1: Try each single site
    for s_id in sorted_sites[:min(10, len(sorted_sites))]:
        if time.time() - start_time > time_limit * 0.3:
            break
        tour, events, att, profit = try_single_site_strategy(s_id)
        if profit > best_profit:
            best_profit = profit
            best_tour = tour
            best_events = events
            best_attendance = att
            if logger:
                logger.log(profit)
    
    # Strategy 2: Try visiting top sites in order
    if time.time() - start_time < time_limit * 0.4:
        tour, events, att, profit = try_multi_site_strategy(sorted_sites[:5])
        if profit > best_profit:
            best_profit = profit
            best_tour = tour
            best_events = events
            best_attendance = att
            if logger:
                logger.log(profit)
    
    # Now try a more sophisticated approach: dynamic programming / greedy construction
    def greedy_construct():
        """Build a tour greedily, period by period."""
        tour = {}
        events = {}
        site_event_counts = {s: 0 for s in site_ids}
        
        prev_site = 0
        for t in range(1, T + 1):
            if time.time() - start_time > time_limit * 0.6:
                # Just stay at current site
                curr = prev_site if prev_site != 0 else sorted_sites[0]
                for t2 in range(t, T + 1):
                    tour[t2] = curr
                    events[t2] = 0
                break
            
            best_choice = None
            best_val = float('-inf')
            
            for s_id in reachable[prev_site]:
                site = site_by_id[s_id]
                travel_cost = c * ss_dist_cache[(prev_site, s_id)]
                
                # Estimate value of holding event here
                # Quick estimate: potential attendance * revenue - fixed_cost - travel_cost
                val_no_event = -travel_cost
                
                val_event = float('-inf')
                if site_event_counts[s_id] < site['max_events']:
                    # Check if we can hold event (consecutive constraint)
                    events[t] = 1
                    tour[t] = s_id
                    can_event = check_consecutive_constraint(events)
                    del tour[t]
                    events[t] = 0
                    
                    if can_event:
                        # Estimate attendance
                        factor = get_weekly_factor(t)
                        total_att = 0
                        for cu_id in cust_ids:
                            cu = cust_by_id[cu_id]
                            attr_val = attr_cache[(cu_id, s_id, t)]
                            self_attr = cust_self_attraction[cu_id]
                            # Rough: ignore other events in window for estimation
                            prop = attr_val / (self_attr + attr_val)
                            total_att += cu['population'] * prop
                        total_att = min(total_att, site['capacity'])
                        
                        val_event = site['revenue_per_customer'] * total_att - site['fixed_cost'] - travel_cost
                
                best_option_val = max(val_no_event, val_event)
                hold_event = val_event > val_no_event and val_event > 0
                
                if best_option_val > best_val:
                    best_val = best_option_val
                    best_choice = (s_id, hold_event)
            
            if best_choice is None:
                # Stay at current site (if possible) or pick any reachable
                if prev_site != 0 and prev_site in reachable[prev_site]:
                    best_choice = (prev_site, False)
                elif reachable[prev_site]:
                    best_choice = (reachable[prev_site][0], False)
                else:
                    # Shouldn't happen
                    best_choice = (sorted_sites[0], False)
            
            s_id, hold_event = best_choice
            tour[t] = s_id
            if hold_event:
                events[t] = 1
                site_event_counts[s_id] += 1
            else:
                events[t] = 0
            prev_site = s_id
        
        # Ensure all periods are filled
        for t in range(1, T + 1):
            if t not in tour:
                tour[t] = prev_site if prev_site != 0 else sorted_sites[0]
                events[t] = 0
        
        att, profit = compute_attendance_fast(tour, events)
        return tour, events, att, profit
    
    if time.time() - start_time < time_limit * 0.5:
        tour, events, att, profit = greedy_construct()
        if profit > best_profit:
            best_profit = profit
            best_tour = tour
            best_events = events
            best_attendance = att
            if logger:
                logger.log(profit)
    
    # Local search improvement
    def local_search(tour, events, att, profit, max_time):
        """Improve solution by local moves."""
        best_p = profit
        improved = True
        
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # Move 1: Toggle event at each period
            for t in range(1, T + 1):
                if time.time() - start_time > max_time:
                    break
                
                old_val = events[t]
                new_val = 1 - old_val
                
                if new_val == 1:
                    # Check max events
                    s = tour[t]
                    count = sum(1 for t2 in range(1, T + 1) if events.get(t2, 0) == 1 and tour[t2] == s)
                    if count >= site_by_id[s]['max_events']:
                        continue
                
                events[t] = new_val
                if not check_consecutive_constraint(events):
                    events[t] = old_val
                    continue
                
                new_att, new_profit = compute_attendance_fast(tour, events)
                if new_profit > best_p:
                    best_p = new_profit
                    att = new_att
                    improved = True
                    if logger:
                        logger.log(new_profit)
                else:
                    events[t] = old_val
            
            # Move 2: Change site at each period (if feasible)
            for t in range(1, T + 1):
                if time.time() - start_time > max_time:
                    break
                
                old_site = tour[t]
                prev = tour[t-1] if t > 1 else 0
                next_site = tour[t+1] if t < T else None
                
                candidates = reachable[prev]
                
                for new_site in candidates:
                    if new_site == old_site:
                        continue
                    
                    # Check next period reachability
                    if next_site is not None and next_site not in reachable.get(new_site, []):
                        continue
                    
                    # Check max events if event held
                    if events.get(t, 0) == 1:
                        count = sum(1 for t2 in range(1, T + 1) if t2 != t and events.get(t2, 0) == 1 and tour[t2] == new_site)
                        if count >= site_by_id[new_site]['max_events']:
                            continue
                    
                    tour[t] = new_site
                    new_att, new_profit = compute_attendance_fast(tour, events)
                    if new_profit > best_p:
                        best_p = new_profit
                        att = new_att
                        improved = True
                        if logger:
                            logger.log(new_profit)
                    else:
                        tour[t] = old_site
        
        return tour, events, att, best_p
    
    if best_tour is not None and time.time() - start_time < time_limit * 0.9:
        best_tour, best_events, best_attendance, best_profit = local_search(
            best_tour, best_events, best_attendance, best_profit, 
            start_time + time_limit * 0.85
        )
    
    # Additional local search: swap events between periods
    if best_tour is not None and time.time() - start_time < time_limit * 0.95:
        improved = True
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            for t1 in range(1, T + 1):
                if time.time() - start_time > time_limit * 0.95:
                    break
                for t2 in range(t1 + 1, min(t1 + 2 * epsilon + 3, T + 1)):
                    if best_events.get(t1, 0) == best_events.get(t2, 0):
                        continue
                    
                    # Swap events
                    old1, old2 = best_events[t1], best_events[t2]
                    best_events[t1], best_events[t2] = old2, old1
                    
                    # Check constraints
                    valid = check_consecutive_constraint(best_events) and check_max_events(best_tour, best_events)
                    
                    if valid:
                        new_att, new_profit = compute_attendance_fast(best_tour, best_events)
                        if new_profit > best_profit:
                            best_profit = new_profit
                            best_attendance = new_att
                            improved = True
                            if logger:
                                logger.log(new_profit)
                        else:
                            best_events[t1], best_events[t2] = old1, old2
                    else:
                        best_events[t1], best_events[t2] = old1, old2
    
    # If no feasible solution found, create a trivial one
    if best_tour is None:
        best_tour = {}
        best_events = {}
        best_attendance = {}
        # Just stay at first reachable site with no events
        first_site = reachable[0][0] if reachable[0] else site_ids[0]
        for t in range(1, T + 1):
            best_tour[t] = first_site
            best_events[t] = 0
        best_attendance, best_profit = compute_attendance_fast(best_tour, best_events)
        if logger:
            logger.log(best_profit)
    
    # Format output
    tour_out = {str(t): best_tour[t] for t in range(1, T + 1)}
    events_out = {str(t): best_events.get(t, 0) for t in range(1, T + 1)}
    event_sites_out = {}
    for t in range(1, T + 1):
        if best_events.get(t, 0) == 1:
            event_sites_out[str(t)] = best_tour[t]
        else:
            event_sites_out[str(t)] = 0
    
    attendance_out = {}
    for s_id in site_ids:
        for t in range(1, T + 1):
            key = f"{s_id}_{t}"
            attendance_out[key] = best_attendance.get((s_id, t), 0.0)
    
    solution = {
        "objective_value": best_profit,
        "tour": tour_out,
        "events": events_out,
        "event_sites": event_sites_out,
        "attendance": attendance_out
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)