import argparse
import json
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    params = data['parameters']
    T = int(params['T'])
    W = int(params['W'])
    L_str = params['L']
    L = None if L_str == 'infinity' else int(L_str)

    source_id = int(data['source_item_id'])
    sink_id = int(data['sink_item_id'])

    items = data['items']
    real_items = [it for it in items if it['type'] == 'treatment']
    real_item_set = set(int(it['item_id']) for it in real_items)

    arcs = data['arcs']
    
    weights = {int(it['item_id']): int(it['weight']) for it in real_items}

    all_item_ids = set()
    for it in items:
        all_item_ids.add(int(it['item_id']))
    for a in arcs:
        all_item_ids.add(int(a['from']))
        all_item_ids.add(int(a['to']))
    
    id_list = sorted(list(all_item_ids))
    id_to_idx = {iid: i for i, iid in enumerate(id_list)}
    
    src_idx = id_to_idx[source_id]
    sink_idx = id_to_idx[sink_id]

    n = len(id_list)
    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    for a in arcs:
        u = id_to_idx[int(a['from'])]
        v = id_to_idx[int(a['to'])]
        lag = int(a['lag'])
        adj[u].append((v, lag))
        radj[v].append((u, lag))

    INF = 10**9
    earliest = [-INF] * n
    earliest[src_idx] = 0
    for _ in range(n + 1):
        updated = False
        for u in range(n):
            if earliest[u] == -INF: continue
            for v, lag in adj[u]:
                if earliest[u] + lag > earliest[v]:
                    earliest[v] = earliest[u] + lag
                    updated = True
        if not updated:
            break

    latest = [INF] * n
    latest[sink_idx] = T + 1
    for _ in range(n + 1):
        updated = False
        for v in range(n):
            if latest[v] == INF: continue
            for u, lag in radj[v]:
                if latest[v] - lag < latest[u]:
                    latest[u] = latest[v] - lag
                    updated = True
        if not updated:
            break

    e = {}
    l = {}
    for it in real_items:
        iid = int(it['item_id'])
        idx = id_to_idx[iid]
        e[iid] = max(1, earliest[idx])
        l[iid] = min(T, latest[idx])
        if e[iid] > l[iid]:
            e[iid] = 1
            l[iid] = T

    out_arcs = {iid: [] for iid in real_item_set}
    in_arcs = {iid: [] for iid in real_item_set}
    for a in arcs:
        u = int(a['from'])
        v = int(a['to'])
        lag = int(a['lag'])
        if u in real_item_set and v in real_item_set:
            out_arcs[u].append((v, lag))
            in_arcs[v].append((u, lag))

    period = {}
    bin_id = {}

    def check_feasible(u, t):
        for v, lag in out_arcs[u]:
            if v in period:
                if period[v] < t + lag:
                    return False
        for v, lag in in_arcs[u]:
            if v in period:
                if period[v] > t - lag:
                    return False
        return True

    def get_solution():
        assignments = []
        bin_periods = {}
        for iid in real_item_set:
            assignments.append({
                'item_id': iid,
                'bin': bin_id[iid],
                'period': period[iid],
                'weight': weights[iid]
            })
            key = (bin_id[iid], period[iid])
            if key not in bin_periods:
                bin_periods[key] = []
            bin_periods[key].append((iid, weights[iid]))
        
        active = []
        for (b, t) in sorted(bin_periods.keys()):
            lst = bin_periods[(b, t)]
            active.append({
                'bin': b,
                'period': t,
                'items': [x[0] for x in lst],
                'total_weight': sum(x[1] for x in lst)
            })
        obj = len(active)
        return {
            'objective_value': float(obj),
            'assignments': assignments,
            'active_bin_periods': active,
            'num_active_bin_periods': int(obj)
        }

    sorted_items = sorted(real_items, key=lambda x: (l[int(x['item_id'])], e[int(x['item_id'])], -weights[int(x['item_id'])]))
    
    bin_usage = {}
    
    for it in sorted_items:
        iid = int(it['item_id'])
        w = weights[iid]
        placed = False
        
        for t in range(e[iid], l[iid] + 1):
            if not check_feasible(iid, t): continue
            
            best_b = -1
            best_cap = -1
            for (b, bt), cap in bin_usage.items():
                if bt == t and cap > 0 and cap + w <= W:
                    if cap > best_cap:
                        best_cap = cap
                        best_b = b
            
            if best_b != -1:
                period[iid] = t
                bin_id[iid] = best_b
                bin_usage[(best_b, t)] += w
                placed = True
                break
        
        if not placed:
            for t in range(e[iid], l[iid] + 1):
                if not check_feasible(iid, t): continue
                active_bins_in_t = sum(1 for (b, bt) in bin_usage if bt == t and bin_usage[(b, bt)] > 0)
                if L is None or active_bins_in_t < L:
                    new_b = 0
                    while (new_b, t) in bin_usage and bin_usage[(new_b, t)] > 0:
                        new_b += 1
                    period[iid] = t
                    bin_id[iid] = new_b
                    bin_usage[(new_b, t)] = w
                    placed = True
                    break
        
        if not placed:
            best_t = e[iid]
            period[iid] = best_t
            bin_id[iid] = 0
            bin_usage[(0, best_t)] = bin_usage.get((0, best_t), 0) + w

    def get_obj():
        return sum(1 for v in bin_usage.values() if v > 0)

    initial_sol = get_solution()
    best_obj = initial_sol['objective_value']
    best_sol = initial_sol

    if logger:
        logger.log_solution(best_obj, best_sol)

    start_time = time.time()

    def get_bin_fullness(iid):
        return W - bin_usage[(bin_id[iid], period[iid])]

    improved = True
    while improved and time.time() - start_time < args.time_limit:
        improved = False
        items_list = sorted([int(it['item_id']) for it in real_items], key=get_bin_fullness, reverse=True)
        for iid in items_list:
            if time.time() - start_time >= args.time_limit:
                break
            curr_t = period[iid]
            curr_b = bin_id[iid]
            curr_key = (curr_b, curr_t)
            
            for t in range(e[iid], l[iid] + 1):
                if t == curr_t: continue
                if not check_feasible(iid, t): continue
                
                best_b = -1
                best_cap = -1
                for (b, bt), cap in bin_usage.items():
                    if bt == t and cap > 0 and cap + weights[iid] <= W:
                        if cap > best_cap:
                            best_cap = cap
                            best_b = b
                
                if best_b != -1:
                    bin_usage[curr_key] -= weights[iid]
                    period[iid] = t
                    bin_id[iid] = best_b
                    bin_usage[(best_b, t)] += weights[iid]
                    
                    improved = True
                    new_obj = get_obj()
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_sol = get_solution()
                        if logger:
                            logger.log_solution(best_obj, best_sol)
                    break

    final_sol = best_sol
    with open(args.solution_path, 'w') as f:
        json.dump(final_sol, f, indent=2)

if __name__ == '__main__':
    main()