import argparse
import json
import time
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    params = instance['parameters']
    Q = params['Q']
    T = params['T']
    W = params['W']
    L = params['L']  # 'infinity' or int
    max_bins_per_period = None if L == 'infinity' else int(L)
    
    num_real_items = instance['num_real_items']
    source_id = instance['source_item_id']
    sink_id = instance['sink_item_id']
    
    items_data = {}
    for item in instance['items']:
        items_data[item['item_id']] = item
    
    real_items = [item for item in instance['items'] if item['type'] == 'treatment']
    real_item_ids = [item['item_id'] for item in real_items]
    
    arcs = instance['arcs']
    
    # Compute earliest and latest times for each real item using Bellman-Ford-like approach
    # We have source at time 0, sink at time T+1
    # Arcs define constraints: time[to] >= time[from] + lag
    
    # Build adjacency for forward propagation (earliest times)
    # and backward propagation (latest times)
    
    item_ids_all = [item['item_id'] for item in instance['items']]
    
    # Initialize earliest times
    earliest = {}
    latest = {}
    for item in instance['items']:
        if item['type'] == 'source':
            earliest[item['item_id']] = 0
            latest[item['item_id']] = 0
        elif item['type'] == 'sink':
            earliest[item['item_id']] = T + 1
            latest[item['item_id']] = T + 1
        else:
            earliest[item['item_id']] = 1
            latest[item['item_id']] = T
    
    # Forward pass: propagate earliest times
    # time[to] >= time[from] + lag
    changed = True
    iterations = 0
    while changed and iterations < num_real_items + 10:
        changed = False
        iterations += 1
        for arc in arcs:
            fr = arc['from']
            to = arc['to']
            lag = arc['lag']
            if fr in earliest and to in earliest:
                new_val = earliest[fr] + lag
                if new_val > earliest[to]:
                    # Don't update source/sink
                    if items_data[to]['type'] == 'treatment':
                        earliest[to] = new_val
                        changed = True
    
    # Backward pass: propagate latest times
    # time[from] <= time[to] - lag
    changed = True
    iterations = 0
    while changed and iterations < num_real_items + 10:
        changed = False
        iterations += 1
        for arc in arcs:
            fr = arc['from']
            to = arc['to']
            lag = arc['lag']
            if fr in latest and to in latest:
                new_val = latest[to] - lag
                if new_val < latest[fr]:
                    if items_data[fr]['type'] == 'treatment':
                        latest[fr] = new_val
                        changed = True
    
    # Clamp to [1, T]
    for iid in real_item_ids:
        earliest[iid] = max(1, earliest[iid])
        latest[iid] = min(T, latest[iid])
    
    # Build weight lookup
    item_weight = {}
    for item in real_items:
        item_weight[item['item_id']] = item['weight']
    
    # Build precedence constraints between real items
    prec_arcs = []
    for arc in arcs:
        fr = arc['from']
        to = arc['to']
        if items_data[fr]['type'] == 'treatment' and items_data[to]['type'] == 'treatment':
            prec_arcs.append((fr, to, arc['lag']))
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining < 2 or num_real_items == 0:
        # Use greedy approach
        assignments, active_bps = greedy_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period)
        obj = len(active_bps)
        if logger:
            logger.log(obj)
        write_solution(solution_path, obj, assignments, active_bps)
        return
    
    # Try MIP formulation
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        # For large instances, use a heuristic first, then try MIP
        if num_real_items > 500:
            # Too large for MIP, use heuristic
            assignments, active_bps = heuristic_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period, time_limit - (time.time() - start_time), logger)
            obj = len(active_bps)
            write_solution(solution_path, obj, assignments, active_bps)
            return
        
        # Estimate number of bins needed
        total_weight = sum(item_weight[iid] for iid in real_item_ids)
        min_bins_total = -(-total_weight // W)  # ceil division
        
        # Estimate max bins per period
        max_items_per_period = {}
        for t in range(1, T+1):
            items_in_t = [iid for iid in real_item_ids if earliest[iid] <= t <= latest[iid]]
            max_items_per_period[t] = len(items_in_t)
        
        # Upper bound on bins needed per period
        max_bins_any_period = max(1, -(-num_real_items // 1))  # worst case
        
        # Better estimate: run greedy first to get upper bound
        greedy_assignments, greedy_bps = greedy_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period)
        greedy_obj = len(greedy_bps)
        if logger:
            logger.log(greedy_obj)
        
        best_assignments = greedy_assignments
        best_bps = greedy_bps
        best_obj = greedy_obj
        
        # Determine max bins per period from greedy
        bins_used_per_period = defaultdict(int)
        for bp in greedy_bps:
            bins_used_per_period[bp['period']] = max(bins_used_per_period[bp['period']], bp['bin'])
        
        max_B = max((bp['bin'] for bp in greedy_bps), default=0)
        # Try to reduce: use at most greedy_obj bins total, distributed
        # Actually, let's compute a tighter bound
        
        # For MIP: determine max bins per period
        # We'll use a time-indexed formulation
        
        # If problem is small enough, use full MIP
        total_vars_estimate = 0
        for iid in real_item_ids:
            total_vars_estimate += (latest[iid] - earliest[iid] + 1)
        
        # Estimate number of bins needed
        B_max = min(max_B + 1, num_real_items, greedy_obj)
        if max_bins_per_period is not None:
            B_max = min(B_max, max_bins_per_period)
        
        total_xvars = total_vars_estimate * B_max
        
        if total_xvars > 100000:
            # Too large, stick with heuristic
            # Try to improve with local search
            assignments, active_bps = heuristic_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period, time_limit - (time.time() - start_time), logger, greedy_assignments, greedy_bps)
            obj = len(active_bps)
            if obj < best_obj:
                best_obj = obj
                best_assignments = assignments
                best_bps = active_bps
            write_solution(solution_path, best_obj, best_assignments, best_bps)
            return
        
        # MIP formulation
        model = gp.Model("bin_scheduling")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 2))
        model.setParam('Threads', 1)
        model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
        
        # Variables: x[i,b,t] = 1 if item i assigned to bin b in period t
        x = {}
        for item in real_items:
            iid = item['item_id']
            w = item_weight[iid]
            for t in range(earliest[iid], latest[iid] + 1):
                for b in range(1, B_max + 1):
                    x[iid, b, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{iid}_{b}_{t}")
        
        # y[b,t] = 1 if bin b in period t is used
        y = {}
        active_periods = set()
        for item in real_items:
            iid = item['item_id']
            for t in range(earliest[iid], latest[iid] + 1):
                for b in range(1, B_max + 1):
                    active_periods.add((b, t))
        
        for (b, t) in active_periods:
            y[b, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{b}_{t}")
        
        model.update()
        
        # Objective: minimize sum of y[b,t]
        model.setObjective(gp.quicksum(y[b, t] for (b, t) in active_periods), GRB.MINIMIZE)
        
        # Constraint: each item assigned exactly once
        for item in real_items:
            iid = item['item_id']
            model.addConstr(
                gp.quicksum(x[iid, b, t] 
                           for t in range(earliest[iid], latest[iid] + 1) 
                           for b in range(1, B_max + 1)) == 1,
                name=f"assign_{iid}"
            )
        
        # Constraint: bin capacity
        # For each (b, t), sum of weights <= W * y[b,t]
        bt_items = defaultdict(list)
        for item in real_items:
            iid = item['item_id']
            for t in range(earliest[iid], latest[iid] + 1):
                for b in range(1, B_max + 1):
                    bt_items[b, t].append(iid)
        
        for (b, t) in active_periods:
            if (b, t) in bt_items:
                items_list = bt_items[b, t]
                model.addConstr(
                    gp.quicksum(item_weight[iid] * x[iid, b, t] for iid in items_list) <= W * y[b, t],
                    name=f"cap_{b}_{t}"
                )
        
        # Linking: x[i,b,t] <= y[b,t]
        for item in real_items:
            iid = item['item_id']
            for t in range(earliest[iid], latest[iid] + 1):
                for b in range(1, B_max + 1):
                    model.addConstr(x[iid, b, t] <= y[b, t], name=f"link_{iid}_{b}_{t}")
        
        # Precedence constraints
        # time[to] >= time[from] + lag
        # time[i] = sum_t sum_b t * x[i,b,t]
        # We need auxiliary variables for time
        time_var = {}
        for item in real_items:
            iid = item['item_id']
            time_var[iid] = model.addVar(lb=earliest[iid], ub=latest[iid], vtype=GRB.INTEGER, name=f"t_{iid}")
        
        # Link time_var to x
        for item in real_items:
            iid = item['item_id']
            model.addConstr(
                time_var[iid] == gp.quicksum(t * x[iid, b, t] 
                                             for t in range(earliest[iid], latest[iid] + 1) 
                                             for b in range(1, B_max + 1)),
                name=f"time_link_{iid}"
            )
        
        # Precedence
        for (fr, to, lag) in prec_arcs:
            model.addConstr(time_var[to] >= time_var[fr] + lag, name=f"prec_{fr}_{to}")
        
        # Symmetry breaking: within same period, use bins in order
        # This helps but may be too many constraints
        # Skip for now if problem is large
        
        # Bins per period limit
        if max_bins_per_period is not None:
            for t in range(1, T+1):
                bins_in_t = [(b, t) for b in range(1, B_max + 1) if (b, t) in active_periods]
                if bins_in_t:
                    model.addConstr(
                        gp.quicksum(y[b, t] for (b, t_) in bins_in_t) <= max_bins_per_period,
                        name=f"maxbins_{t}"
                    )
        
        # Symmetry breaking: within a period, lower-numbered bins used first
        for t in range(1, T+1):
            for b in range(2, B_max + 1):
                if (b, t) in active_periods and (b-1, t) in active_periods:
                    model.addConstr(y[b, t] <= y[b-1, t], name=f"sym_{b}_{t}")
        
        # Set initial solution from greedy
        for item in real_items:
            iid = item['item_id']
            for t in range(earliest[iid], latest[iid] + 1):
                for b in range(1, B_max + 1):
                    x[iid, b, t].Start = 0
        
        for a in greedy_assignments:
            iid = a['item_id']
            b = a['bin']
            t = a['period']
            if b <= B_max and (iid, b, t) in x:
                x[iid, b, t].Start = 1
        
        for (b, t) in active_periods:
            y[b, t].Start = 0
        for bp in greedy_bps:
            b = bp['bin']
            t = bp['period']
            if (b, t) in active_periods:
                y[b, t].Start = 1
        
        model.setParam('Cutoff', greedy_obj + 0.5)
        
        # Callback to log solutions
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(int(round(obj_val)))
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj_val = int(round(model.ObjVal))
            if obj_val < best_obj:
                best_obj = obj_val
                # Extract solution
                best_assignments = []
                best_bps = []
                bp_items = defaultdict(list)
                bp_weight = defaultdict(int)
                
                for item in real_items:
                    iid = item['item_id']
                    assigned = False
                    for t in range(earliest[iid], latest[iid] + 1):
                        for b in range(1, B_max + 1):
                            if x[iid, b, t].X > 0.5:
                                best_assignments.append({
                                    'item_id': iid,
                                    'bin': b,
                                    'period': t,
                                    'weight': item_weight[iid]
                                })
                                bp_items[b, t].append(iid)
                                bp_weight[b, t] += item_weight[iid]
                                assigned = True
                                break
                        if assigned:
                            break
                
                for (b, t), items_list in bp_items.items():
                    best_bps.append({
                        'bin': b,
                        'period': t,
                        'items': items_list,
                        'total_weight': bp_weight[b, t]
                    })
        
        write_solution(solution_path, best_obj, best_assignments, best_bps)
        
    except Exception as e:
        # Fallback to greedy
        print(f"MIP failed: {e}", file=sys.stderr)
        assignments, active_bps = greedy_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period)
        obj = len(active_bps)
        if logger:
            logger.log(obj)
        write_solution(solution_path, obj, assignments, active_bps)


def greedy_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period):
    """Greedy: assign each item to its earliest feasible time, packing into bins."""
    
    # Topological sort of items based on precedence
    # Build adjacency
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    item_ids = set(item['item_id'] for item in real_items)
    
    for iid in item_ids:
        in_degree[iid] = 0
    
    for (fr, to, lag) in prec_arcs:
        if lag > 0:
            adj[fr].append((to, lag))
            in_degree[to] += 1
    
    # Also handle lag <= 0 arcs (they still impose constraints but don't define ordering)
    # For scheduling, we process items that have no unscheduled predecessors first
    
    # Compute actual earliest considering precedence
    actual_earliest = {iid: earliest[iid] for iid in item_ids}
    actual_latest = {iid: latest[iid] for iid in item_ids}
    
    # Forward propagation
    from collections import deque
    
    # Build full adjacency for propagation
    fwd_adj = defaultdict(list)
    bwd_adj = defaultdict(list)
    for (fr, to, lag) in prec_arcs:
        fwd_adj[fr].append((to, lag))
        bwd_adj[to].append((fr, lag))
    
    changed = True
    while changed:
        changed = False
        for (fr, to, lag) in prec_arcs:
            new_e = actual_earliest[fr] + lag
            if new_e > actual_earliest[to]:
                actual_earliest[to] = new_e
                changed = True
            new_l = actual_latest[to] - lag
            if new_l < actual_latest[fr]:
                actual_latest[fr] = new_l
                changed = True
    
    # Schedule items: process in order of earliest time, then by weight (heavier first for better packing)
    items_sorted = sorted(real_items, key=lambda it: (actual_earliest[it['item_id']], -item_weight[it['item_id']]))
    
    # Track assigned times
    assigned_time = {}
    assigned_bin = {}
    
    # Bins per period: list of (remaining_capacity, bin_id)
    bins_per_period = defaultdict(list)  # period -> [(remaining_cap, bin_id)]
    next_bin_id = defaultdict(lambda: 1)  # period -> next bin id
    
    assignments = []
    
    for item in items_sorted:
        iid = item['item_id']
        w = item_weight[iid]
        
        # Compute effective earliest based on already-assigned predecessors
        eff_earliest = actual_earliest[iid]
        for (fr, to, lag) in prec_arcs:
            if to == iid and fr in assigned_time:
                eff_earliest = max(eff_earliest, assigned_time[fr] + lag)
        
        eff_latest = actual_latest[iid]
        for (fr, to, lag) in prec_arcs:
            if fr == iid and to in assigned_time:
                eff_latest = min(eff_latest, assigned_time[to] - lag)
        
        eff_earliest = max(1, eff_earliest)
        eff_latest = min(T, eff_latest)
        
        # Try to pack into existing bin-period first
        best_t = None
        best_b = None
        best_remaining = W + 1
        
        for t in range(eff_earliest, eff_latest + 1):
            for j, (rem, bid) in enumerate(bins_per_period[t]):
                if rem >= w and rem - w < best_remaining:
                    best_t = t
                    best_b = bid
                    best_remaining = rem - w
        
        if best_t is not None:
            # Assign to existing bin
            t = best_t
            b = best_b
            # Update remaining capacity
            for j, (rem, bid) in enumerate(bins_per_period[t]):
                if bid == b:
                    bins_per_period[t][j] = (rem - w, bid)
                    break
        else:
            # Need new bin - find period with least new bins or earliest
            t = eff_earliest
            if max_bins_per_period is not None:
                # Find a period that has room for a new bin
                found = False
                for tt in range(eff_earliest, eff_latest + 1):
                    if len(bins_per_period[tt]) < max_bins_per_period:
                        t = tt
                        found = True
                        break
                if not found:
                    # Must fit in existing bin somewhere
                    for tt in range(eff_earliest, eff_latest + 1):
                        for j, (rem, bid) in enumerate(bins_per_period[tt]):
                            if rem >= w:
                                t = tt
                                b = bid
                                bins_per_period[t][j] = (rem - w, bid)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        assigned_time[iid] = t
                        assigned_bin[iid] = b
                        assignments.append({
                            'item_id': iid,
                            'bin': b,
                            'period': t,
                            'weight': w
                        })
                        continue
            
            b = next_bin_id[t]
            next_bin_id[t] += 1
            bins_per_period[t].append((W - w, b))
        
        assigned_time[iid] = t
        assigned_bin[iid] = b
        assignments.append({
            'item_id': iid,
            'bin': b,
            'period': t,
            'weight': w
        })
    
    # Build active bin periods
    bp_items = defaultdict(list)
    bp_weight = defaultdict(int)
    for a in assignments:
        key = (a['bin'], a['period'])
        bp_items[key].append(a['item_id'])
        bp_weight[key] += a['weight']
    
    active_bps = []
    for (b, t), items_list in bp_items.items():
        active_bps.append({
            'bin': b,
            'period': t,
            'items': items_list,
            'total_weight': bp_weight[b, t]
        })
    
    return assignments, active_bps


def heuristic_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period, time_remaining, logger, init_assignments=None, init_bps=None):
    """Improved heuristic with multiple strategies."""
    
    if init_assignments is not None:
        best_assignments = init_assignments
        best_bps = init_bps
        best_obj = len(init_bps)
    else:
        best_assignments, best_bps = greedy_solve(real_items, item_weight, earliest, latest, prec_arcs, W, T, max_bins_per_period)
        best_obj = len(best_bps)
        if logger:
            logger.log(best_obj)
    
    # Try different time assignment strategies
    start = time.time()
    
    # Strategy: assign to latest feasible time
    items_sorted = sorted(real_items, key=lambda it: (-latest[it['item_id']], -item_weight[it['item_id']]))
    
    # Build precedence info
    actual_earliest = {item['item_id']: earliest[item['item_id']] for item in real_items}
    actual_latest = {item['item_id']: latest[item['item_id']] for item in real_items}
    
    changed = True
    while changed:
        changed = False
        for (fr, to, lag) in prec_arcs:
            new_e = actual_earliest[fr] + lag
            if new_e > actual_earliest[to]:
                actual_earliest[to] = new_e
                changed = True
            new_l = actual_latest[to] - lag
            if new_l < actual_latest[fr]:
                actual_latest[fr] = new_l
                changed = True
    
    # Try packing by grouping items with same time window and using FFD
    # Group items by their feasible time periods
    # For each period, collect items that could go there, then do bin packing
    
    # Strategy: First Fit Decreasing per period
    # Assign items to their "best" period to minimize total bins
    
    # Simple: for each possible period assignment, count bins needed
    # This is complex, so let's just try a few random assignments
    
    import random
    random.seed(42)
    
    for attempt in range(100):
        if time.time() - start > time_remaining - 1:
            break
        
        assigned_time = {}
        # Topological order
        fwd_adj = defaultdict(list)
        for (fr, to, lag) in prec_arcs:
            fwd_adj[fr].append((to, lag))
        
        # Random priority
        if attempt == 0:
            order = sorted(real_items, key=lambda it: (actual_earliest[it['item_id']], -item_weight[it['item_id']]))
        elif attempt == 1:
            order = sorted(real_items, key=lambda it: (actual_latest[it['item_id']], -item_weight[it['item_id']]))
        else:
            order = list(real_items)
            random.shuffle(order)
        
        # Resolve precedence: process in topological order
        in_deg = defaultdict(int)
        item_ids = set(it['item_id'] for it in real_items)
        for iid in item_ids:
            in_deg[iid] = 0
        fwd = defaultdict(list)
        bwd = defaultdict(list)
        for (fr, to, lag) in prec_arcs:
            fwd[fr].append((to, lag))
            bwd[to].append((fr, lag))
            if lag > 0:
                in_deg[to] += 1
        
        from collections import deque
        queue = deque()
        topo_order = []
        for iid in item_ids:
            if in_deg[iid] == 0:
                queue.append(iid)
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for (to, lag) in fwd[node]:
                if lag > 0:
                    in_deg[to] -= 1
                    if in_deg[to] == 0:
                        queue.append(to)
        
        # Add remaining (cycles with non-positive lags)
        remaining_items = [iid for iid in item_ids if iid not in set(topo_order)]
        topo_order.extend(remaining_items)
        
        # Assign times in topological order
        assigned_time = {}
        bins_per_period = defaultdict(list)
        next_bin = defaultdict(lambda: 1)
        assignments = []
        valid = True
        
        for iid in topo_order:
            w = item_weight[iid]
            eff_e = actual_earliest[iid]
            eff_l = actual_latest[iid]
            
            for (fr2, to2, lag2) in prec_arcs:
                if to2 == iid and fr2 in assigned_time:
                    eff_e = max(eff_e, assigned_time[fr2] + lag2)
                if fr2 == iid and to2 in assigned_time:
                    eff_l = min(eff_l, assigned_time[to2] - lag2)
            
            eff_e = max(1, eff_e)
            eff_l = min(T, eff_l)
            
            if eff_e > eff_l:
                valid = False
                break
            
            # Try to fit into existing bin-period (best fit)
            best_t = None
            best_b = None
            best_waste = W + 1
            
            for t in range(eff_e, eff_l + 1):
                for j, (rem, bid) in enumerate(bins_per_period[t]):
                    if rem >= w:
                        waste = rem - w
                        if waste < best_waste:
                            best_waste = waste
                            best_t = t
                            best_b = bid
            
            if best_t is not None:
                t = best_t
                b = best_b
                for j, (rem, bid) in enumerate(bins_per_period[t]):
                    if bid == b:
                        bins_per_period[t][j] = (rem - w, bid)
                        break
            else:
                # New bin needed - pick period to minimize total bins
                if attempt % 2 == 0:
                    t = eff_e
                else:
                    t = eff_l
                
                if max_bins_per_period is not None:
                    found = False
                    for tt in (range(eff_e, eff_l + 1) if attempt % 2 == 0 else range(eff_l, eff_e - 1, -1)):
                        if len(bins_per_period[tt]) < max_bins_per_period:
                            t = tt
                            found = True
                            break
                    if not found:
                        # Try fitting in existing bin
                        for tt in range(eff_e, eff_l + 1):
                            for j, (rem, bid) in enumerate(bins_per_period[tt]):
                                if rem >= w:
                                    t = tt
                                    b = bid
                                    bins_per_period[t][j] = (rem - w, bid)
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            valid = False
                            break
                        assigned_time[iid] = t
                        assignments.append({'item_id': iid, 'bin': b, 'period': t, 'weight': w})
                        continue
                
                b = next_bin[t]
                next_bin[t] += 1
                bins_per_period[t].append((W - w, b))
            
            assigned_time[iid] = t
            assignments.append({'item_id': iid, 'bin': b, 'period': t, 'weight': w})
        
        if not valid:
            continue
        
        # Count active bin-periods
        bp_items = defaultdict(list)
        bp_weight = defaultdict(int)
        for a in assignments:
            key = (a['bin'], a['period'])
            bp_items[key].append(a['item_id'])
            bp_weight[key] += a['weight']
        
        active_bps = []
        for (b, t), items_list in bp_items.items():
            active_bps.append({
                'bin': b,
                'period': t,
                'items': items_list,
                'total_weight': bp_weight[b, t]
            })
        
        obj = len(active_bps)
        if obj < best_obj:
            best_obj = obj
            best_assignments = assignments
            best_bps = active_bps
            if logger:
                logger.log(best_obj)
    
    return best_assignments, best_bps


def write_solution(solution_path, obj, assignments, active_bps):
    solution = {
        'objective_value': float(obj),
        'assignments': assignments,
        'active_bin_periods': active_bps,
        'num_active_bin_periods': obj
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()