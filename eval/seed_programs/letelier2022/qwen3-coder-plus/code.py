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


def compute_time_windows(items, num_total_items, arcs):
    """Compute earliest/latest possible periods for all items."""
    # Create adjacency lists again just for this calculation
    adj_list = [[] for _ in range(num_total_items)]
    pred_list = [[] for _ in range(num_total_items)]
    
    for arc in arcs:
        from_node = arc['from']
        to_node = arc['to']
        lag = arc['lag']
        adj_list[from_node].append((to_node, lag))
        pred_list[to_node].append((from_node, lag))
    
    # Identify source and sink
    source_node, sink_node = None, None
    for i, item in enumerate(items):
        if item['type'] == 'source':
            source_node = i
        elif item['type'] == 'sink':
            sink_node = i

    T_val = 0
    if sink_node is not None:
        # Assuming items are arranged such that sink has a special ID indicating the end
        for item in items:
            if item['type'] == 'sink':
                T_val = item['item_id'] - 1  # If sink is at T+1, then T = item_id - 1
                break

    # Initialize earliest and latest times
    earliest = [float('inf')] * num_total_items
    latest = [-float('inf')] * num_total_items
    
    if source_node is not None:
        earliest[source_node] = 0
    if sink_node is not None:
        latest[sink_node] = T_val + 1

    # Forward pass to calculate earliest times
    # Using a simple iterative approach
    changed = True
    while changed:
        changed = False
        for u in range(num_total_items):
            for v, lag in adj_list[u]:
                new_earliest = earliest[u] + lag
                if new_earliest < earliest[v]:
                    earliest[v] = new_earliest
                    changed = True
    
    # Backward pass to calculate latest times
    changed = True
    while changed:
        changed = False
        for u in range(num_total_items):
            for v, lag in adj_list[u]:  # Edge from u to v: time_v >= time_u + lag
                # Implication for latest: time_u <= time_v - lag
                # So latest[u] <= latest[v] - lag
                new_latest = latest[v] - lag
                if new_latest < latest[u]:
                    latest[u] = new_latest
                    changed = True
    
    # Clamp to time window size
    for i in range(num_total_items):
        if items[i]['type'] not in ['source', 'sink']:
            earliest[i] = max(1, earliest[i])
            latest[i] = min(T_val, latest[i])
        elif items[i]['type'] == 'sink':
            latest[i] = T_val + 1


    return earliest, latest, T_val


def solve_instance(instance_path, solution_path, time_limit, log_path=None):
    """Main function to solve the optimization problem."""
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    Q = instance["parameters"]["Q"]
    T = instance["parameters"]["T"]  
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
    
    # Identify items by type
    source_node, sink_node = None, None
    for i in range(len(items)):
        if items[i]['type'] == 'source':
            source_node = i
        elif items[i]['type'] == 'sink':
            sink_node = i
    assert source_node is not None and sink_node is not None, "Missing source or sink"

    # Compute time windows to reduce the number of variables
    earliest, latest, final_T = compute_time_windows(items, len(items), arcs)

    # Filter out real items (not source or sink)
    real_items = [i for i in range(len(items)) if items[i]['type'] == 'treatment']
    
    # Determine upper bound on bins needed (can estimate this)
    total_weight = sum(items[i]['weight'] for i in real_items)
    max_needed_bins = int(math.ceil(total_weight / W))
    if max_bins_per_period == float('inf'):
        num_bins = max_needed_bins
    else:
        num_bins = min(max_bins_per_period, max_needed_bins)

    import gurobipy as gp
    from gurobipy import GRB

    # Create model
    model = gp.Model('capacitated_facility_scheduling')
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    
    # Define variable ranges relevant to problem
    x_vars = {}  # x[i,t,b] binary: item i in period t in bin b
    y_vars = {}  # y[t,b] binary: bin b is used in period t

    # Only create variables for items that are in reasonable time windows and real treatment items
    for i in real_items:
        e_i = max(1, int(earliest[i]))  # at least 1 for real items
        l_i = min(final_T, int(latest[i]))
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
        e_i = max(1, int(earliest[i]))
        l_i = min(final_T, int(latest[i]))
        model.addConstr(
            gp.quicksum(x_vars[i, t, b] 
                        for t in range(e_i, l_i + 1)
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

    # 3. Link y and x: if any x[i,t,b]=1 then y[t,b]=1
    for i in real_items:
        e_i = max(1, int(earliest[i]))
        l_i = min(final_T, int(latest[i]))
        for t in range(e_i, l_i + 1):
            for b in range(num_bins):
                if (i, t, b) in x_vars:
                    model.addConstr(x_vars[i, t, b] <= y_vars[t, b], name=f"link_x_to_y_{i}_{t}_{b}")

    # 4. Timing precedence constraints: If arc (i,j) with lag l exists,
    # then period[j] >= period[i] + l
    for arc in arcs:
        from_item = arc['from']
        to_item = arc['to']
        lag = arc['lag']

        # Only apply to real treatment items
        if items[from_item]['type'] != 'treatment' or items[to_item]['type'] != 'treatment':
            continue

        # Add big-M constraint to enforce precedence:
        # time_var_to >= time_var_from + lag
        # Define helper variables for time of assignment:
        # sum(t * x[from, t, b]) for all valid t,b
        # sum(t * x[to, t, b]) for all valid t,b
        
        # To linearize: for each assignment of to_item, ensure its time >= min(from_times) + lag
        # Actually it's easier to do: sum over assignments of "to" >= sum over assignments of "from" + lag
        
        expr_from = gp.LinExpr()
        expr_to = gp.LinExpr()
        
        e_f = max(1, int(earliest[from_item]))
        l_f = min(final_T, int(latest[from_item]))
        for t in range(e_f, l_f + 1):
            for b in range(num_bins):
                if (from_item, t, b) in x_vars:
                    expr_from.addTerms(t, x_vars[from_item, t, b])
        
        e_t = max(1, int(earliest[to_item]))
        l_t = min(final_T, int(latest[to_item]))
        for t in range(e_t, l_t + 1):
            for b in range(num_bins):
                if (to_item, t, b) in x_vars:
                    expr_to.addTerms(t, x_vars[to_item, t, b])
        
        # The precedence constraint is: expr_to >= expr_from + lag
        model.addConstr(expr_to >= expr_from + lag, name=f"precedence_{from_item}_{to_item}")

    # Objective: Minimize sum of y[t,b] over all t,b
    total_active = gp.quicksum(y_vars[t, b] for t in range(1, final_T + 1) for b in range(num_bins))
    model.setObjective(total_active, GRB.MINIMIZE)

    # Callback to log objectives
    def mycallback(model, where):
        if where == GRB.Callback.MIPSol:
            obj = model.cbGet(GRB.Callback.MIPSolObj)
            if logger:
                logger.log(obj)

    model.optimize(mycallback)

    # Extract solution
    solution = {
        "objective_value": float('inf'),  # Placeholder if no solution
        "assignments": [],
        "active_bin_periods": [],
        "num_active_bin_periods": 0
    }

    if model.solCount > 0:
        solution["objective_value"] = model.objVal
        assignments_dict = defaultdict(lambda: {'items': [], 'weight': 0})
        
        for (i, t, b), var in x_vars.items():
            if var.x > 0.5:  # Binary variable is true
                item_obj = items[i]
                solution["assignments"].append({
                    "item_id": item_obj['item_id'],
                    "bin": b,
                    "period": t,
                    "weight": item_obj['weight']
                })
                
                # Add to assignment dict for active bin periods
                assignments_dict[(t, b)]['items'].append(item_obj['item_id'])
                assignments_dict[(t, b)]['weight'] += item_obj['weight']

        active_bin_periods_list = []
        for (t, b), data in assignments_dict.items():
            active_bin_periods_list.append({
                "bin": b,
                "period": t,
                "items": data['items'],
                "total_weight": data['weight']
            })
        
        solution["active_bin_periods"] = active_bin_periods_list
        solution["num_active_bin_periods"] = len(active_bin_periods_list)

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def greedy_heuristic_solution(instance, log_path=None):
    """Fallback: basic greedy heuristic for the scheduling problem."""
    # This is a simple greedy approach: assign items in order of earliest deadline,
    # fitting items into earliest possible period/bins

    Q = instance["parameters"]["Q"]
    T = instance["parameters"]["T"]    
    W = instance["parameters"]["W"]

    items = instance["items"]
    arcs = instance["arcs"]
    
    # Index items
    item_idx_by_id = {item['item_id']: idx for idx, item in enumerate(items)}
    
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

    # Identify real items
    real_items = [i for i in range(len(items)) if items[i]['type'] == 'treatment']
    
    # Create precedence graph
    succ = defaultdict(list)
    pred_count = {i: 0 for i in real_items}
    
    for arc in arcs:
        from_item, to_item, lag = arc['from'], arc['to'], arc['lag']
        if from_item in pred_count and to_item in pred_count:
            succ[from_item].append((to_item, lag))
            pred_count[to_item] += 1
    
    # Topological sort
    from collections import deque
    queue = deque([i for i in real_items if pred_count[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v, lag in succ[u]:
            pred_count[v] -= 1
            if pred_count[v] == 0:
                queue.append(v)

    # Greedily assign items
    assignment = {}
    bins_used = {}

    # We'll calculate earliest possible times iteratively
    earliest_time = {i: 1 for i in real_items}
    
    # Propagate earliest times according to precedences
    for u in topo_order:
        for v, lag in succ[u]:
            earliest_time[v] = max(earliest_time[v], earliest_time[u] + lag)

    for item_idx in topo_order:
        weight = items[item_idx]['weight']
        start_period = earliest_time[item_idx]

        # Find best period and bin
        assigned = False
        for period in range(start_period, T + 1):
            # Check existing bins in that period for space
            period_buckets = bins_used.setdefault(period, {})
            
            # Check in numeric order of bin index
            possible_bins = sorted(period_buckets.keys()) + [max(period_buckets.keys()) + 1 if period_buckets else 0]
            
            for bin_num in possible_bins:
                if bin_num not in period_buckets:
                    # Create new bin
                    period_buckets[bin_num] = {
                        'weight': 0,
                        'items': []
                    }

                # Try this bin if it has capacity
                current_bin = period_buckets[bin_num]
                
                if current_bin['weight'] + weight <= W:
                    # Assign this item here
                    current_bin['weight'] += weight
                    current_bin['items'].append(items[item_idx]['item_id'])
                    
                    assignment[items[item_idx]['item_id']] = {
                        "item_id": items[item_idx]['item_id'],
                        "bin": bin_num,
                        "period": period,
                        "weight": weight
                    }
                    assigned = True
                    break
            
            if assigned:
                break  # Stop searching for this item
        
        assert assigned, f"Failed to assign item {items[item_idx]['item_id']} within time/bins."

    # Construct the solution
    assignments_list = list(assignment.values())
    
    active_periods = []
    for period, bins_in_period in bins_used.items():
        for bin_num, contents in bins_in_period.items():
            active_periods.append({
                "bin": bin_num,
                "period": period,
                "items": contents['items'],
                "total_weight": contents['weight']
            })

    total_active = len(active_periods)

    solution = {
        "objective_value": total_active,
        "assignments": assignments_list,
        "active_bin_periods": active_periods,
        "num_active_bin_periods": total_active
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