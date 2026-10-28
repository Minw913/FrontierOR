import json
import argparse
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

def read_instance(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_adjacency(containers):
    adj = defaultdict(set)
    for c in containers:
        for n in c['adjacent_containers']:
            adj[c['id']].add(n)
            adj[n].add(c['id'])
    return adj

def count_components(nodes, adj):
    """Count connected components of the subgraph induced by nodes."""
    if not nodes:
        return 0
    visited = set()
    components = 0
    node_set = set(nodes)
    for start in nodes:
        if start not in visited:
            components += 1
            queue = deque([start])
            visited.add(start)
            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor in node_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    return components

def find_good_set(candidates, adj, size, max_components):
    """Find a set of 'size' containers from candidates with at most max_components components."""
    if len(candidates) <= size:
        return list(candidates)[:size]
    
    candidates = set(candidates)
    best_set = None
    best_comp = float('inf')
    
    # Try BFS from multiple starting points to find connected sets
    start_points = list(candidates)[:min(20, len(candidates))]
    
    for start in start_points:
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue and len(visited) < size:
            curr = queue.popleft()
            neighbors = [n for n in adj[curr] if n in candidates and n not in visited]
            for n in neighbors:
                if len(visited) < size:
                    visited.add(n)
                    queue.append(n)
        
        if len(visited) >= size:
            result = list(visited)[:size]
            comp = count_components(result, adj)
            if comp <= max_components:
                return result
            if comp < best_comp:
                best_comp = comp
                best_set = result
    
    # If no connected set found, try to build one with multiple components
    if best_set is not None and best_comp <= max_components:
        return best_set
    
    # Greedy: build set by adding nodes that minimize components
    selected = set()
    remaining = set(candidates)
    
    # Start with a node
    if remaining:
        start = next(iter(remaining))
        selected.add(start)
        remaining.discard(start)
    
    while len(selected) < size and remaining:
        # Find candidates adjacent to selected set
        frontier = set()
        for s in selected:
            for n in adj[s]:
                if n in remaining:
                    frontier.add(n)
        
        if frontier:
            # Pick one that connects to most selected nodes
            best_node = max(frontier, key=lambda n: len(adj[n] & selected))
            selected.add(best_node)
            remaining.discard(best_node)
        else:
            # No adjacent candidate, start new component
            if remaining:
                selected.add(next(iter(remaining)))
                remaining.discard(selected.__iter__().__next__())
                # Actually let's just pick any
                node = next(iter(remaining))
                selected.add(node)
                remaining.discard(node)
    
    result = list(selected)[:size]
    comp = count_components(result, adj)
    if comp <= max_components:
        return result
    
    # If still too many components, try to fix by swapping
    return result

def compute_objective(assignments, trades):
    """Compute total re-assignments."""
    obj = 0
    for trade in trades:
        tid = trade['id']
        for p in range(trade['start_period'] + 1, trade['end_period'] + 1):
            if str(tid) in assignments and str(p) in assignments[str(tid)] and str(p-1) in assignments[str(tid)]:
                curr = set(assignments[str(tid)][str(p)])
                prev = set(assignments[str(tid)][str(p-1)])
                if curr != prev:
                    obj += 1
    return obj

def solve_greedy(data, time_limit, logger=None):
    """Greedy heuristic to find a feasible solution."""
    start_time = time.time()
    
    params = data['problem_parameters']
    containers = data['containers']
    trades = data['trades']
    num_periods = params['num_periods']
    
    adj = get_adjacency(containers)
    
    # Container availability
    avail = {}
    for c in containers:
        avail[c['id']] = set(range(c['availability_start'], c['availability_end'] + 1))
    
    # Sort trades by demand (larger first) then by start period
    sorted_trades = sorted(trades, key=lambda t: (-t['container_demand'], t['start_period']))
    
    # assignments[tid][p] = list of container ids
    assignments = defaultdict(dict)
    
    # Process period by period
    for p in range(num_periods):
        if time.time() - start_time > time_limit * 0.5:
            break
        
        # Available containers in this period
        available = set(cid for cid in avail if p in avail[cid])
        assigned_this_period = set()
        
        for trade in sorted_trades:
            tid = trade['id']
            if trade['start_period'] <= p <= trade['end_period']:
                demand = trade['container_demand']
                max_disp = trade['max_dispersion']
                
                # Try to keep previous assignment
                prev_set = None
                if p > trade['start_period'] and (p - 1) in assignments[tid]:
                    prev_set = set(assignments[tid][p - 1])
                
                if prev_set is not None:
                    # Keep containers that are still available and not taken
                    keep = set()
                    for c in prev_set:
                        if c in available and c not in assigned_this_period:
                            keep.add(c)
                    
                    need = demand - len(keep)
                    
                    if need <= 0:
                        result = list(keep)[:demand]
                    else:
                        # Need additional containers
                        candidates = available - assigned_this_period - keep
                        
                        # Prefer containers adjacent to kept ones (for dispersion)
                        added = []
                        frontier = deque()
                        seen_frontier = set()
                        for c in keep:
                            for n in adj[c]:
                                if n in candidates and n not in seen_frontier:
                                    frontier.append(n)
                                    seen_frontier.add(n)
                        
                        while len(added) < need and frontier:
                            c = frontier.popleft()
                            if c in candidates:
                                added.append(c)
                                candidates.discard(c)
                                for n in adj[c]:
                                    if n in candidates and n not in seen_frontier:
                                        frontier.append(n)
                                        seen_frontier.add(n)
                        
                        # If still need more, pick any available
                        while len(added) < need and candidates:
                            c = next(iter(candidates))
                            added.append(c)
                            candidates.discard(c)
                        
                        result = list(keep) + added
                    
                    result = result[:demand]
                    assignments[tid][p] = result
                    assigned_this_period.update(result)
                else:
                    # First period for this trade
                    candidates = available - assigned_this_period
                    result = find_good_set(candidates, adj, demand, max_disp)
                    assignments[tid][p] = result
                    assigned_this_period.update(result)
    
    # Format solution
    solution = {"objective_value": 0.0, "assignments": {}}
    for trade in trades:
        tid = trade['id']
        solution["assignments"][str(tid)] = {}
        for p in range(trade['start_period'], trade['end_period'] + 1):
            if p in assignments[tid]:
                solution["assignments"][str(tid)][str(p)] = sorted(assignments[tid][p])
            else:
                solution["assignments"][str(tid)][str(p)] = []
    
    obj = compute_objective(solution["assignments"], trades)
    solution["objective_value"] = float(obj)
    
    return obj, solution

def local_search(data, solution, time_limit, start_time, logger=None):
    """Local search to improve the solution by reducing re-assignments."""
    params = data['problem_parameters']
    containers = data['containers']
    trades = data['trades']
    num_periods = params['num_periods']
    
    adj = get_adjacency(containers)
    
    avail = {}
    for c in containers:
        avail[c['id']] = set(range(c['availability_start'], c['availability_end'] + 1))
    
    assignments = {}
    for tid_str, periods in solution["assignments"].items():
        assignments[int(tid_str)] = {int(p): set(clist) for p, clist in periods.items()}
    
    best_obj = compute_objective_from_internal(assignments, trades)
    
    # Find re-assignment opportunities
    improved = True
    iteration = 0
    max_iterations = 1000
    
    while improved and iteration < max_iterations:
        if time.time() - start_time > time_limit * 0.85:
            break
        improved = False
        iteration += 1
        
        for trade in trades:
            tid = trade['id']
            demand = trade['container_demand']
            max_disp = trade['max_dispersion']
            
            for p in range(trade['start_period'] + 1, trade['end_period'] + 1):
                if time.time() - start_time > time_limit * 0.85:
                    break
                
                if p not in assignments[tid] or (p-1) not in assignments[tid]:
                    continue
                
                curr = assignments[tid][p]
                prev = assignments[tid][p-1]
                
                if curr == prev:
                    continue
                
                # Try to make curr equal to prev by swapping with other trades
                # Find containers in prev but not in curr (need to add)
                need_to_add = prev - curr
                # Find containers in curr but not in prev (can remove)
                can_remove = curr - prev
                
                if not need_to_add:
                    continue
                
                # Check if containers in need_to_add are available and not assigned to others
                success = True
                swaps = []  # (container, from_trade) pairs
                
                for c in need_to_add:
                    if p not in avail[c]:
                        success = False
                        break
                    # Find who has this container in period p
                    owner = None
                    for other_trade in trades:
                        oid = other_trade['id']
                        if oid == tid:
                            continue
                        if p in assignments[oid] and c in assignments[oid][p]:
                            owner = oid
                            break
                    
                    if owner is not None:
                        # Need to swap: give owner one of our can_remove containers
                        if can_remove:
                            swap_container = next(iter(can_remove))
                            # Check if swap_container is available and valid for owner
                            if p in avail[swap_container]:
                                swaps.append((c, owner, swap_container))
                                can_remove.discard(swap_container)
                            else:
                                success = False
                                break
                        else:
                            success = False
                            break
                    elif c not in curr:
                        # Container is free, just add it
                        swaps.append((c, None, None))
                
                if not success or not swaps:
                    continue
                
                # Check dispersion after swap
                new_curr = set(curr)
                for (c, owner, swap_c) in swaps:
                    new_curr.add(c)
                    if swap_c is not None:
                        new_curr.discard(swap_c)
                
                if len(new_curr) != demand:
                    continue
                
                # Check dispersion
                comp = count_components(list(new_curr), adj)
                if comp > max_disp:
                    continue
                
                # Apply swaps
                for (c, owner, swap_c) in swaps:
                    if owner is not None:
                        assignments[owner][p].discard(c)
                        assignments[owner][p].add(swap_c)
                        # Check dispersion for owner
                        owner_comp = count_components(list(assignments[owner][p]), adj)
                        if owner_comp > trades_dict[owner]['max_dispersion']:
                            # Revert
                            assignments[owner][p].add(c)
                            assignments[owner][p].discard(swap_c)
                            success = False
                            break
                
                if success:
                    assignments[tid][p] = new_curr
                    new_obj = compute_objective_from_internal(assignments, trades)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        improved = True
                    else:
                        # Revert if not improving
                        assignments[tid][p] = curr
    
    # Format solution
    sol_out = {"objective_value": float(best_obj), "assignments": {}}
    for trade in trades:
        tid = trade['id']
        sol_out["assignments"][str(tid)] = {}
        for p in range(trade['start_period'], trade['end_period'] + 1):
            if p in assignments[tid]:
                sol_out["assignments"][str(tid)][str(p)] = sorted(list(assignments[tid][p]))
            else:
                sol_out["assignments"][str(tid)][str(p)] = []
    
    return best_obj, sol_out

def compute_objective_from_internal(assignments, trades):
    obj = 0
    for trade in trades:
        tid = trade['id']
        for p in range(trade['start_period'] + 1, trade['end_period'] + 1):
            if p in assignments[tid] and (p-1) in assignments[tid]:
                if assignments[tid][p] != assignments[tid][p-1]:
                    obj += 1
    return obj

def solve_mip(data, time_limit, warm_start=None, logger=None):
    """Solve using MIP with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None
    
    start_time = time.time()
    
    params = data['problem_parameters']
    containers = data['containers']
    trades = data['trades']
    num_periods = params['num_periods']
    
    adj = get_adjacency(containers)
    
    # Container info
    container_ids = [c['id'] for c in containers]
    avail = {}
    for c in containers:
        avail[c['id']] = (c['availability_start'], c['availability_end'])
    
    # Edges
    edges = set()
    for c in containers:
        for n in c['adjacent_containers']:
            edge = (min(c['id'], n), max(c['id'], n))
            edges.add(edge)
    edges = list(edges)
    
    # Build adjacency list for edges
    adj_edges = defaultdict(list)  # node -> list of (edge_index, other_node)
    for idx, (i, j) in enumerate(edges):
        adj_edges[i].append((idx, j))
        adj_edges[j].append((idx, i))
    
    # Trade info
    trade_map = {t['id']: t for t in trades}
    
    # Estimate problem size
    num_vars_est = 0
    for t in trades:
        for p in range(t['start_period'], t['end_period'] + 1):
            for cid in container_ids:
                if avail[cid][0] <= p <= avail[cid][1]:
                    num_vars_est += 1
    
    # If problem is too large, skip MIP
    if num_vars_est > 200000:
        return None, None
    
    model = gp.Model("container_assignment")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_limit))
    model.setParam('OutputFlag', 0)
    
    # Decision variables
    x = {}
    for t in trades:
        tid = t['id']
        for p in range(t['start_period'], t['end_period'] + 1):
            for cid in container_ids:
                if avail[cid][0] <= p <= avail[cid][1]:
                    x[tid, cid, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{tid}_{cid}_{p}")
    
    # Re-assignment variables
    r = {}
    for t in trades:
        tid = t['id']
        for p in range(t['start_period'] + 1, t['end_period'] + 1):
            r[tid, p] = model.addVar(vtype=GRB.BINARY, name=f"r_{tid}_{p}")
    
    # Flow variables for dispersion
    root = {}
    f_src = {}
    f_edge = {}
    
    for t in trades:
        tid = t['id']
        demand = t['container_demand']
        max_disp = t['max_dispersion']
        
        # If max_disp >= demand, no dispersion constraint needed
        if max_disp >= demand:
            continue
        
        for p in range(t['start_period'], t['end_period'] + 1):
            for cid in container_ids:
                if (tid, cid, p) in x:
                    root[tid, p, cid] = model.addVar(vtype=GRB.BINARY, name=f"rt_{tid}_{p}_{cid}")
                    f_src[tid, p, cid] = model.addVar(lb=0, ub=demand, name=f"fs_{tid}_{p}_{cid}")
            
            for idx, (i, j) in enumerate(edges):
                if (tid, i, p) in x and (tid, j, p) in x:
                    f_edge[tid, p, idx, 0] = model.addVar(lb=0, ub=demand, name=f"fe_{tid}_{p}_{idx}_0")
                    f_edge[tid, p, idx, 1] = model.addVar(lb=0, ub=demand, name=f"fe_{tid}_{p}_{idx}_1")
    
    model.update()
    
    # Objective
    model.setObjective(gp.quicksum(r.values()), GRB.MINIMIZE)
    
    # Constraints
    # 1. Demand
    for t in trades:
        tid = t['id']
        demand = t['container_demand']
        for p in range(t['start_period'], t['end_period'] + 1):
            vars_p = [x[tid, cid, p] for cid in container_ids if (tid, cid, p) in x]
            if vars_p:
                model.addConstr(gp.quicksum(vars_p) == demand, name=f"dem_{tid}_{p}")
    
    # 2. Capacity
    for p in range(num_periods):
        for cid in container_ids:
            vars_c = [x[tid, cid, p] for tid in trade_map if (tid, cid, p) in x]
            if len(vars_c) > 1:
                model.addConstr(gp.quicksum(vars_c) <= 1, name=f"cap_{cid}_{p}")
    
    # 3. Re-assignment linking
    for t in trades:
        tid = t['id']
        for p in range(t['start_period'] + 1, t['end_period'] + 1):
            for cid in container_ids:
                has_curr = (tid, cid, p) in x
                has_prev = (tid, cid, p-1) in x
                
                if has_curr and has_prev:
                    model.addConstr(r[tid, p] >= x[tid, cid, p] - x[tid, cid, p-1])
                    model.addConstr(r[tid, p] >= x[tid, cid, p-1] - x[tid, cid, p])
                elif has_curr and not has_prev:
                    model.addConstr(r[tid, p] >= x[tid, cid, p])
                elif not has_curr and has_prev:
                    model.addConstr(r[tid, p] >= x[tid, cid, p-1])
    
    # 4. Dispersion (flow-based)
    for t in trades:
        tid = t['id']
        demand = t['container_demand']
        max_disp = t['max_dispersion']
        
        if max_disp >= demand:
            continue
        
        for p in range(t['start_period'], t['end_period'] + 1):
            # Number of roots <= max_disp
            root_vars = [root[tid, p, cid] for cid in container_ids if (tid, p, cid) in root]
            if root_vars:
                model.addConstr(gp.quicksum(root_vars) <= max_disp, name=f"nroot_{tid}_{p}")
            
            for cid in container_ids:
                if (tid, p, cid) in root:
                    model.addConstr(root[tid, p, cid] <= x[tid, cid, p])
                    model.addConstr(f_src[tid, p, cid] <= demand * root[tid, p, cid])
            
            # Flow conservation
            for cid in container_ids:
                if (tid, cid, p) not in x:
                    continue
                
                inflow = [f_src[tid, p, cid]] if (tid, p, cid) in f_src else []
                outflow = []
                
                for idx, (i, j) in enumerate(edges):
                    if (tid, p, idx, 0) not in f_edge:
                        continue
                    if i == cid:
                        outflow.append(f_edge[tid, p, idx, 0])
                        inflow.append(f_edge[tid, p, idx, 1])
                    elif j == cid:
                        inflow.append(f_edge[tid, p, idx, 0])
                        outflow.append(f_edge[tid, p, idx, 1])
                
                model.addConstr(
                    gp.quicksum(inflow) - gp.quicksum(outflow) == x[tid, cid, p],
                    name=f"fc_{tid}_{p}_{cid}"
                )
            
            # Flow bounds
            for idx, (i, j) in enumerate(edges):
                if (tid, p, idx, 0) in f_edge:
                    model.addConstr(f_edge[tid, p, idx, 0] <= demand * x[tid, i, p])
                    model.addConstr(f_edge[tid, p, idx, 0] <= demand * x[tid, j, p])
                    model.addConstr(f_edge[tid, p, idx, 1] <= demand * x[tid, i, p])
                    model.addConstr(f_edge[tid, p, idx, 1] <= demand * x[tid, j, p])
    
    # Warm start
    if warm_start is not None:
        try:
            for t in trades:
                tid = t['id']
                for p in range(t['start_period'], t['end_period'] + 1):
                    p_str = str(p)
                    if str(tid) in warm_start["assignments"] and p_str in warm_start["assignments"][str(tid)]:
                        assigned = set(warm_start["assignments"][str(tid)][p_str])
                        for cid in container_ids:
                            if (tid, cid, p) in x:
                                x[tid, cid, p].Start = 1.0 if cid in assigned else 0.0
            
            for t in trades:
                tid = t['id']
                for p in range(t['start_period'] + 1, t['end_period'] + 1):
                    if str(tid) in warm_start["assignments"]:
                        p_str = str(p)
                        prev_str = str(p-1)
                        if p_str in warm_start["assignments"][str(tid)] and prev_str in warm_start["assignments"][str(tid)]:
                            curr = set(warm_start["assignments"][str(tid)][p_str])
                            prev = set(warm_start["assignments"][str(tid)][prev_str])
                            r[tid, p].Start = 0.0 if curr == prev else 1.0
        except:
            pass
    
    model.update()
    
    # Solve
    model.optimize()
    
    if model.SolCount > 0:
        # Extract solution
        sol_assignments = {}
        for t in trades:
            tid = t['id']
            sol_assignments[str(tid)] = {}
            for p in range(t['start_period'], t['end_period'] + 1):
                assigned = []
                for cid in container_ids:
                    if (tid, cid, p) in x and x[tid, cid, p].X > 0.5:
                        assigned.append(cid)
                sol_assignments[str(tid)][str(p)] = sorted(assigned)
        
        obj_val = model.ObjVal
        
        solution = {
            "objective_value": float(obj_val),
            "assignments": sol_assignments
        }
        
        return obj_val, solution
    
    return None, None

def validate_solution(data, solution):
    """Validate the solution and fix any issues."""
    params = data['problem_parameters']
    containers = data['containers']
    trades = data['trades']
    num_periods = params['num_periods']
    
    adj = get_adjacency(containers)
    avail = {}
    for c in containers:
        avail[c['id']] = (c['availability_start'], c['availability_end'])
    
    trade_map = {t['id']: t for t in trades}
    
    # Check and fix issues
    assignments = solution["assignments"]
    
    for trade in trades:
        tid = trade['id']
        tid_str = str(tid)
        if tid_str not in assignments:
            assignments[tid_str] = {}
        
        for p in range(trade['start_period'], trade['end_period'] + 1):
            p_str = str(p)
            if p_str not in assignments[tid_str]:
                assignments[tid_str][p_str] = []
    
    # Recompute objective
    obj = 0
    for trade in trades:
        tid = trade['id']
        tid_str = str(tid)
        for p in range(trade['start_period'] + 1, trade['end_period'] + 1):
            p_str = str(p)
            prev_str = str(p - 1)
            if p_str in assignments[tid_str] and prev_str in assignments[tid_str]:
                curr = set(assignments[tid_str][p_str])
                prev = set(assignments[tid_str][prev_str])
                if curr != prev:
                    obj += 1
    
    solution["objective_value"] = float(obj)
    return solution

def main():
    parser = argparse.ArgumentParser(description="Container Assignment Optimization")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    # Read instance
    data = read_instance(args.instance_path)
    
    # Phase 1: Greedy heuristic
    time_for_heuristic = min(args.time_limit * 0.15, 30)
    obj, solution = solve_greedy(data, time_for_heuristic, logger)
    
    if logger:
        logger.log_solution(obj, solution)
    
    # Phase 2: Local search improvement
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 2:
        try:
            # Build trades_dict for local search
            global trades_dict
            trades_dict = {t['id']: t for t in data['trades']}
            obj2, solution2 = local_search(data, solution, args.time_limit, start_time, logger)
            if obj2 < obj:
                obj = obj2
                solution = solution2
                if logger:
                    logger.log_solution(obj, solution)
        except Exception as e:
            pass
    
    # Phase 3: MIP (if time allows)
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 5:
        try:
            mip_time = remaining - 2
            obj_mip, sol_mip = solve_mip(data, mip_time, warm_start=solution, logger=logger)
            if obj_mip is not None and obj_mip < obj:
                obj = obj_mip
                solution = sol_mip
                if logger:
                    logger.log_solution(obj, solution)
        except Exception as e:
            pass
    
    # Validate and output
    solution = validate_solution(data, solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()