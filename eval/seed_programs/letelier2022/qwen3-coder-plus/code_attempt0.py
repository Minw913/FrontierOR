import argparse
import json
import math
from collections import defaultdict
from solution_logger import SolutionLogger

def build_precedence_graph(items, arcs, num_total_items):
    """Build adjacency lists for arcs and determine time bounds."""
    # Build graph: adjacency lists, in-degrees
    adj_list = [[] for _ in range(num_total_items)]
    pred_list = [[] for _ in range(num_total_items)]
    
    for arc_data in arcs:
        # Note: arc.from -> arc.to with lag
        from_node = arc_data['from']
        to_node = arc_data['to']
        lag = arc_data['lag']
        
        adj_list[from_node].append((to_node, lag))
        pred_list[to_node].append((from_node, lag))
    
    return adj_list, pred_list


def compute_time_windows(items, source_item_id, adj_list, pred_list):
    """Compute earliest/latest possible periods for all items."""
    # Use topological sort + dynamic programming to get earliest periods
    num_items = len(items)
    
    # For Earliest: forward pass with source at time 0
    # For Latest: backward pass with sink at time T+1
    
    # Initialize: source = 0, sink = T+1, others = inf/-inf
    earliest = [float('inf')] * num_items
    latest = [-float('inf')] * num_items
    
    # Get sink id from the items data
    sink_id = None
    T = None  # We'll find out from the sink arc
    for item in items:
        if item['item_id'] == items[source_item_id]['sink_item_id']:
            sink_id = item['item_id']
            T = item['item_id'] - 1  # Since sink is at T+1
            break
    
    # Find T and the actual values
    src_id = items[source_item_id]['item_id']
    for i, item in enumerate(items):
        if item['type'] == 'source':
            src_id = i
            continue
        elif item['type'] == 'sink':
            sink_id = i
            T = item['item_id'] - 1  # This is actually T+1 due to special encoding in description
    
    # Actually let's find T from context in arcs instead:
    # The sink is always at time T+1, and we find it
    sink_value = None
    for item in items:
        if item['type'] == 'sink':
            # Since sink is fixed at time T+1, and its item_id corresponds to that
            sink_value = item['item_id']
            T = sink_value - 1
            break
    
    earliest[src_id] = 0
    latest[sink_id] = T + 1  # Fixed to time T+1
    
    # Topological sort for forward propagation
    in_degree = [0] * num_items
    for u in range(num_items):
        for v, _ in adj_list[u]:
            in_degree[v] += 1
    
    queue = []
    for u in range(num_items):
        if in_degree[u] == 0:
            queue.append(u)
    
    while queue:
        u = queue.pop(0)
        for v, lag in adj_list[u]:
            # Update earliest[v] to max of current value and earliest[u] + lag
            earliest[v] = min(earliest[v], earliest[u] + lag)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Calculate latest times - reverse the graph direction and use latest[sink] = T+1
    out_degree = [0] * num_items
    for u in range(num_items):
        for v, _ in pred_list[u]:  # Reverse direction for backwards processing
            out_degree[v] += 1
    
    # Queue with initially leaf nodes in reversed dependency tree (starting from sink)
    leaves = []
    processed_count = [0] * num_items
    
    # We start processing from leaves with initial latest times
    # Backward version requires propagating from sink
    # Rebuild using latest time from sink
    queue_back = [sink_id]
    processed = {sink_id}
    
    # Create reverse adjacency to propagate latests
    rev_adj = [[] for _ in range(num_items)]
    for u in range(num_items):
        for v, lag in pred_list[u]: # edge v->u was originally in adj_list[v] 
            rev_adj[u].append((v, lag))
    
    # Actually let's use full recursive processing with priority on nodes depending on those with computed latests
    # Propagate latest back through reverse graph using sink constraints
    
    # For simplicity, doing brute-force iterations until no updates
    has_changed = True
    while has_changed:
        has_changed = False
        # Reverse arcs: if there's an arc u->v with lag l, then latest_v - l <= latest_u, so latest_u >= latest_v - l
        for u in range(num_items):
            for v, lag in pred_list[u]:  # means there is an arc from v to u with lag
                new_early = latest[v] - lag
                if new_early < latest[u]:
                    latest[u] = new_early
                    has_changed = True
                    
        # Similarly for backward propagation with forward arcs
        for u in range(num_items):
            for v, lag in adj_list[u]:  # Forward arc u to v means time_v >= time_u + lag
                # So to satisfy latest_v we need latest_u <= latest_v - lag
                new_latest_for_u = latest[v] - lag
                if new_latest_for_u < latest[u]:
                    latest[u] = new_latest_for_u
                    has_changed = True
    
    # Actually the above is incorrect logic. Let me recalculate manually and systematically:
    # First reset latest times appropriately
    latest = [T + 1] * num_items  # Start all as high since sink is at T+1
    
    # Now for reverse pass: from sink backwards, update latest times using backward arcs
    # This means using edges in adjacency format in reverse with inverted logic
    # If constraint is that v.time >= u.time + lag (i.e., u->v with lag), then v.latest >= u.latest + lag,
    # which implies u.latest <= v.latest - lag 
    
    # Process queue based on topological order reversed
    outdegree = [len(adj_list[i]) for i in range(len(adj_list))]
    queue = []
    for i in range(num_items):
        if outdegree[i] == 0:  # Sink node initially
            queue.append(i)
            
    while queue:
        u = queue.pop()
        for v, lag in pred_list[u]:  # There is an edge v -> u with lag
            # time[u] >= time[v] + lag  => latest[v] <= latest[u] - lag
            if latest[v] > latest[u] - lag:
                latest[v] = latest[u] - lag
            outdegree[v] -= 1
            if outdegree[v] == 0:
                queue.append(v)

    # Now trim the boundaries
    for i in range(num_items):
        if earliest[i] > latest[i]:
            print(f"ERROR: Earliest > Latest for item {i}: {earliest[i]} > {latest[i]}")
            raise ValueError("Infeasible constraints")
                
        earliest[i] = max(1, earliest[i])  # All actual items have earliest at least 1
        if items[i]['type'] != 'sink':
            latest[i] = min(T, latest[i])
    
    return earliest, latest, T


def solve_instance(instance_path, solution_path, time_limit, log_path=None):
    """Main function to solve the optimization problem."""
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    Q = instance["parameters"]["Q"]
    T = instance["T"]  
    W = instance["parameters"]["W"]
    L_str = instance["parameters"]["L"]
    max_bins_per_period = float('inf') if L_str == 'infinity' else int(L_str)
    
    items = instance["items"]
    num_real_items = instance["num_real_items"]
    arcs = instance["arcs"]
    source_item_id = instance["source_item_id"]
    sink_item_id = instance["sink_item_id"]
    
    # Index items by id for easy lookups
    item_idx_by_id = {item['item_id']: idx for idx, item in enumerate(items)}
    
    # Convert arcs: update 'from' and 'to' to indices in the items array
    new_arcs = []
    for arc in arcs:
        from_idx = item_idx_by_id[arc['from']]
        to_idx = item_idx_by_id[arc['to']]
        new_arcs.append({
            'from': from_idx,
            'to': to_idx,
            'lag': arc['lag']
        })
    arcs = new_arcs
    
    # Also fix source/sink mapping if needed
    for i in range(len(items)):
        if items[i]['type'] == 'source':
            source_item_id = i
        elif items[i]['type'] == 'sink':
            sink_item_id = i
    
    # We need items as they are indexed correctly in arcs now
    # Compute time windows to reduce the number of variables
    adj_list, pred_list = build_precedence_graph(items, arcs, len(items))
    earliest, latest, final_T = compute_time_windows(items, source_item_id, adj_list, pred_list)
    
    # Filter out real items (not source or sink)
    real_items = [i for i in range(len(items)) if items[i]['type'] == 'treatment']
    
    # Determine upper bound on bins needed (can estimate this)
    max_needed_bins = int(math.ceil(sum(items[i]['weight'] for i in real_items) / W))

    if max_bins_per_period == float('inf'):
        num_bins = max_needed_bins
    else:
        num_bins = int(min(max_bins_per_period, max_needed_bins))


    try:
        import gurobipy as gp
        from gurobipy import GRB

        # Create model
        model = gp.Model('capacitated_facility_scheduling')
        model.setParam(GRB.Param.TimeLimit, time_limit)
        
        # Define variable ranges relevant to problem
        x_vars = {}  # x[i,t,b] binary: item i in period t in bin b
        y_vars = {}  # y[t,b] binary: bin b is used in period t
        
        # Only create variables for items that are in reasonable time windows and real treatment items
        for i in real_items:
            e_i = max(1, earliest[i])  # at least 1 for real items
            l_i = min(final_T, latest[i])
            w_i = items[i]['weight']
            
            for t in range(e_i, l_i + 1):
                # Estimate max bins practically needed per period
                for b in range(num_bins):
                    x_vars[i, t, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}_{b}")
        
        # Define y vars for all combinations that could potentially happen
        for t in range(1, final_T + 1):
            for b in range(num_bins):
                y_vars[t, b] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}_{b}")


        # Constraints
        # 1. Each real item is assigned exactly once
        for i in real_items:
            model.addConstr(
                gp.quicksum(x_vars[i, t, b] 
                            for t in range(max(1, earliest[i]), min(final_T, latest[i]) + 1)
                            for b in range(num_bins)
                            if (i, t, b) in x_vars) == 1,
                name=f"assign_once_{i}"
            )
        
        # 2. Capacity: sum of weights in any bin in any period <= W
        for t in range(1, final_T + 1):
            for b in range(num_bins):
                model.addConstr(
                    gp.quicksum(items[i]['weight'] * x_vars[i, t, b]
                                for i in real_items
                                if (i, t, b) in x_vars) <= W * y_vars[t, b],
                    name=f"capacity_{t}_{b}"
                )

        # 3. Links y and x: if any x[i,t,b] = 1 then y[t,b] = 1
        # Done implicitly in constraint above
        
        # 4. Timing precedence constraints: If there's arc (i,j) with lag l, 
        # then period_j >= period_i + l
        for arc in arcs:
            from_item = arc['from']
            to_item = arc['to']
            lag = arc['lag']
            
            # Only for real items
            if items[from_item]['type'] != 'treatment' or items[to_item]['type'] != 'treatment':
                continue
                
            # Create binary variables to encode "if x[f,tf,bf]=1 and x[t,tt,bt]=1 then tf+lag <= tt"
            # Reformulate: sum(tf * x[f,tf,bf]) + lag <= sum(tt * x[t,tt,bt_t] for all valid t assignments)
            
            lhs_expr = gp.LinExpr()
            rhs_expr = gp.LinExpr()
            
            # Left side: period of 'from'
            e_f = max(1, earliest[from_item])
            l_f = min(final_T, latest[from_item])
            for tf in range(e_f, l_f + 1):
                for bf in range(num_bins):
                    if (from_item, tf, bf) in x_vars:
                        lhs_expr.addTerms(-1.0, x_vars[from_item, tf, bf], tf)  # We need to make it >=, so negate
            
            # Right side: period of 'to'
            e_t = max(1, earliest[to_item])
            l_t = min(final_T, latest[to_item])
            for tt in range(e_t, l_t + 1):
                for bt in range(num_bins):
                    if (to_item, tt, bt) in x_vars:
                        rhs_expr.addTerms(-1.0, x_vars[to_item, tt, bt], tt - lag)  # Subtract lag to make it equivalent to "to >= from + lag"
                        
            # So inequality is: -sum(tf * assign_f_tf_bf) >= -sum(tt * assign_t_tt_bt) + l
            # Or rearranged: sum(tt * assign_to_tt_bt) - l >= sum(tf * assign_from_tf_bf)
            model.addConstr(lhs_expr >= rhs_expr, name=f"precedence_{from_item}_{to_item}")

        # Objective: Minimize sum of y[t,b] over all t,b
        total_active = gp.quicksum(y_vars[t, b] for t in range(1, final_T + 1) for b in range(num_bins))
        model.setObjective(total_active, GRB.MINIMIZE)

        # Optimize
        def mycallback(model, where):
            if where == GRB.Callback.MIPSol:
                obj = model.cbGet(GRB.Callback.MIPSolObj)
                if logger:
                    logger.log(obj)

        model.optimize(mycallback)

        # Extract solution
        if model.status == GRB.OPTIMAL or model.SolCount > 0:
            solution = {
                "objective_value": model.objVal,
                "assignments": [],
                "active_bin_periods": [],
                "num_active_bin_periods": 0
            }

            assignments_dict = defaultdict(list)  # Maps (t,b) to list of items
            active_bin_periods_set = set()       # Set of (t,b) tuples

            for (i, t, b), var in x_vars.items():
                if abs(var.x - 1.0) < 1e-5:  # If binary var is 1
                    solution["assignments"].append({
                        "item_id": items[i]['item_id'],
                        "bin": b,
                        "period": t,
                        "weight": items[i]['weight']
                    })
                    assignments_dict[(t, b)].append(items[i]['item_id'])
                    active_bin_periods_set.add((t, b))
            
            active_bp_list = []
            for (t, b) in sorted(active_bin_periods_set):
                item_list = assignments_dict[(t, b)]
                total_wgt = sum(items[item_idx_by_id[iid]]['weight'] for iid in item_list)
                active_bp_list.append({
                    "bin": b,
                    "period": t,
                    "items": item_list,
                    "total_weight": total_wgt
                })

            solution["active_bin_periods"] = active_bp_list
            solution["num_active_bin_periods"] = len(active_bin_periods_set)

            # Write solution
            with open(solution_path, 'w') as f:
                json.dump(solution, f, indent=2)

        else:
            print(f"Model status: {model.status}, no solution found.")
            # Return empty solution
            solution = {
                "objective_value": float('-inf'),
                "assignments": [],
                "active_bin_periods": [],
                "num_active_bin_periods": 0
            }
            with open(solution_path, 'w') as f:
                json.dump(solution, f, indent=2)

    except ImportError:
        # If Gurobi is not available, implement a greedy heuristic approach
        # This is a fallback implementation
        solution = greedy_heuristic_solution(instance, log_path)
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


def greedy_heuristic_solution(instance, log_path=None):
    """Fallback: basic greedy heuristic for the scheduling problem."""
    # This is a simple greedy approach: assign items in order of earliest deadline, 
    # fitting items into earliest possible period/bins
    
    # We will process items and assign each to earliest valid period/bin combination
    import datetime
    start_time = datetime.datetime.now()
    
    Q = instance["parameters"]["Q"]
    T = instance["parameters"]["T"]    
    W = instance["parameters"]["W"]
    
    items = instance["items"]
    arcs = instance["arcs"]
    source_item_id = instance["source_item_id"]
    sink_item_id = instance["sink_item_id"]
    
    # Index items by id for easy lookups
    item_idx_by_id = {item['item_id']: idx for idx, item in enumerate(items)}
    
    # Convert arcs similarly
    new_arcs = []
    for arc in arcs:
        from_idx = item_idx_by_id[arc['from']]
        to_idx = item_idx_by_id[arc['to']]
        new_arcs.append({
            'from': from_idx,
            'to': to_idx,
            'lag': arc['lag']
        })
    arcs = new_arcs
    
    # Identify real treatment items
    real_items = [i for i in range(len(items)) if items[i]['type'] == 'treatment']
    
    # Create precedence graph
    succ = defaultdict(list)
    pred_count = defaultdict(int)
    
    for arc in arcs:
        from_item, to_item, lag = arc['from'], arc['to'], arc['lag']
        if items[from_item]['type'] == 'treatment' and items[to_item]['type'] == 'treatment':
            succ[from_item].append((to_item, lag))
            pred_count[to_item] += 1
    
    # Calculate earliest possible assignment based on dependencies
    earliest = [1]*len(items)
    for arc in arcs:
        if items[arc['from']]['type'] == 'source':
            earliest[arc['to']] = max(earliest[arc['to']], 1)
    
    # Topologically sort to determine order
    from collections import deque
    queue = deque()
    for item_idx in real_items:
        if pred_count[item_idx] == 0:
            queue.append(item_idx)
    
    order = []
    while queue:
        item_idx = queue.popleft()
        order.append(item_idx)
        for next_item, lag in succ[item_idx]:
            pred_count[next_item] -= 1
            earliest[next_item] = max(earliest[next_item], earliest[item_idx] + lag)
            if pred_count[next_item] == 0:
                queue.append(next_item)
    
    # Assign items greedily
    assignment = {}
    bins_used = {}
    
    # Sort items by earliest possible time in order to process
    ordered_items = sorted(real_items, key=lambda x: earliest[x])
    
    for item_idx in ordered_items:
        weight = items[item_idx]['weight']
        assigned = False
        
        # Try periods starting from earliest[item_idx] to T
        for period in range(earliest[item_idx], T + 1):
            period_key = f"{period}"
            if period_key not in bins_used:
                bins_used[period_key] = {}
            
            # Look for the lowest numbered bin that fits this item
            found_bin = None
            for bin_num in sorted(bins_used[period_key].keys(), key=int):
                if bins_used[period_key][bin_num]["weight"] + weight <= W:
                    found_bin = bin_num
                    break
        
            # If no existing bin in this period works
            if found_bin is None:
                # Create a new bin for this period
                new_bin = 0
                while str(new_bin) in bins_used[period_key]:
                    new_bin += 1
                found_bin = new_bin
                bins_used[period_key][str(found_bin)] = {"weight": 0, "items": []}
            
            # Add item to bin in period
            bins_used[period_key][str(found_bin)]["weight"] += weight
            bins_used[period_key][str(found_bin)]["items"].append(items[item_idx]['item_id'])
            assignment[items[item_idx]['item_id']] = {
                "item_id": items[item_idx]['item_id'],
                "bin": int(found_bin),
                "period": period,
                "weight": weight
            }
            assigned = True
            break
    
        # We assume constraints allow at least some assignment
        assert assigned, f"Could not assign item {item_idx} within time limits"

    # Build output solution
    assignments_list = list(assignment.values())
    
    active_bp = []
    for period_str, bins_dict in bins_used.items():
        for bin_str, content in bins_dict.items():
            active_bp.append({
                "bin": int(bin_str),
                "period": int(period_str),
                "items": content["items"],
                "total_weight": content["weight"]
            })
    
    solution = {
        "objective_value": len(active_bp),
        "assignments": assignments_list,
        "active_bin_periods": active_bp,
        "num_active_bin_periods": len(active_bp)
    }
    
    return solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve CapaFacSched problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to the log JSONL file (optional).')
    
    args = parser.parse_args()
    solve_instance(args.instance_path, args.solution_path, args.time_limit, args.log_path)