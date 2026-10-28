import argparse
import json
import time
from collections import defaultdict
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = args.time_limit

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    def time_exceeded():
        return time.time() - start_time > time_limit

    try:
        with open(args.instance_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading instance file: {e}")
        return

    parameters = data.get('parameters', {})
    T_horizon = parameters.get('T', 0)
    W = parameters.get('W', 0)
    L_str = parameters.get('L', 'infinity')
    L_val = None if L_str == 'infinity' else int(L_str)
    items = data.get('items', [])
    arcs = data.get('arcs', [])
    source_id = data.get('source_item_id')
    sink_id = data.get('sink_item_id')
    num_total_items = data.get('num_total_items', 0)

    if not items or source_id is None or sink_id is None or T_horizon <= 0 or W <= 0:
        print("Invalid instance data")
        output_solution(args.solution_path, 0.0, [], [])
        return

    node_ids = [item['item_id'] for item in items]
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    index_to_id = node_ids

    edges_forward = []
    edges_reverse = []
    for arc in arcs:
        u = arc['from']
        v = arc['to']
        lag = arc['lag']
        u_idx = id_to_index[u]
        v_idx = id_to_index[v]
        edges_forward.append((u_idx, v_idx, lag))
        edges_reverse.append((v_idx, u_idx, lag))

    n = len(node_ids)
    feasible = True
    dist_early = [-10**18] * n
    source_idx = id_to_index[source_id]
    dist_early[source_idx] = 0

    if time_exceeded():
        output_solution(args.solution_path, 0.0, [], [])
        return

    for _ in range(n-1):
        if time_exceeded():
            feasible = False
            break
        updated = False
        for (u, v, lag) in edges_forward:
            if dist_early[u] != -10**18 and dist_early[u] + lag > dist_early[v]:
                dist_early[v] = dist_early[u] + lag
                updated = True
        if not updated:
            break

    if feasible:
        for (u, v, lag) in edges_forward:
            if time_exceeded():
                feasible = False
                break
            if dist_early[u] != -10**18 and dist_early[u] + lag > dist_early[v]:
                feasible = False
                break

    if feasible:
        dist_rev = [-10**18] * n
        sink_idx = id_to_index[sink_id]
        dist_rev[sink_idx] = 0

        for _ in range(n-1):
            if time_exceeded():
                feasible = False
                break
            updated = False
            for (u, v, lag) in edges_reverse:
                if dist_rev[u] != -10**18 and dist_rev[u] + lag > dist_rev[v]:
                    dist_rev[v] = dist_rev[u] + lag
                    updated = True
            if not updated:
                break

        if feasible:
            for (u, v, lag) in edges_reverse:
                if time_exceeded():
                    feasible = False
                    break
                if dist_rev[u] != -10**18 and dist_rev[u] + lag > dist_rev[v]:
                    feasible = False
                    break

    real_items = []
    if feasible:
        sink_early = dist_early[sink_idx]
        if sink_early > T_horizon + 1:
            feasible = False
        else:
            for item in items:
                if item['type'] != 'treatment':
                    continue
                item_id = item['item_id']
                idx = id_to_index[item_id]
                e = dist_early[idx]
                l_val = T_horizon + 1 - dist_rev[idx]
                if e > l_val:
                    feasible = False
                    break
                window_start = max(1, e)
                window_end = min(T_horizon, l_val)
                if window_start > window_end:
                    feasible = False
                    break
                real_items.append((item_id, item['weight'], window_start, window_end))

    assignments = {}
    active_bin_periods = []
    objective_value = 0.0

    if feasible and real_items:
        real_items_sorted = sorted(real_items, key=lambda x: x[1], reverse=True)
        period_bins = {t: [] for t in range(1, T_horizon + 1)}
        next_bin_id = 0

        for (item_id, weight, window_start, window_end) in real_items_sorted:
            if time_exceeded():
                feasible = False
                break

            candidate_bins = []
            for t in range(window_start, window_end + 1):
                if t < 1 or t > T_horizon:
                    continue
                for bin_info in period_bins[t]:
                    bin_id, rem = bin_info
                    if rem >= weight:
                        candidate_bins.append((rem, t, bin_id))

            if candidate_bins:
                candidate_bins.sort(key=lambda x: x[0])
                rem_val, chosen_t, chosen_bin_id = candidate_bins[0]
                new_list = []
                updated_bin = False
                for (bin_id, rem) in period_bins[chosen_t]:
                    if bin_id == chosen_bin_id and not updated_bin:
                        new_rem = rem - weight
                        new_list.append((bin_id, new_rem))
                        updated_bin = True
                    else:
                        new_list.append((bin_id, rem))
                if not updated_bin:
                    feasible = False
                    break
                period_bins[chosen_t] = new_list
                assignments[item_id] = (chosen_t, chosen_bin_id, weight)
            else:
                candidate_periods = []
                for t in range(window_start, window_end + 1):
                    if t < 1 or t > T_horizon:
                        continue
                    if L_val is None or len(period_bins[t]) < L_val:
                        candidate_periods.append(t)

                if not candidate_periods:
                    feasible = False
                    break

                bin_counts = [(len(period_bins[t]), t) for t in candidate_periods]
                bin_counts.sort(key=lambda x: (x[0], x[1]))
                chosen_t = bin_counts[0][1]
                new_bin_id = next_bin_id
                next_bin_id += 1
                period_bins[chosen_t].append((new_bin_id, W - weight))
                assignments[item_id] = (chosen_t, new_bin_id, weight)

    if feasible and real_items and not time_exceeded():
        assignment_list = []
        bin_period_dict = defaultdict(lambda: {'items': [], 'total_weight': 0})
        for item_id, (period, bin_id, weight) in assignments.items():
            assignment_list.append({
                'item_id': item_id,
                'bin': bin_id,
                'period': period,
                'weight': weight
            })
            key = (bin_id, period)
            bin_period_dict[key]['items'].append(item_id)
            bin_period_dict[key]['total_weight'] += weight

        active_bin_periods = []
        for (bin_id, period), info in bin_period_dict.items():
            active_bin_periods.append({
                'bin': bin_id,
                'period': period,
                'items': info['items'],
                'total_weight': info['total_weight']
            })
        num_active = len(active_bin_periods)
        objective_value = float(num_active)
        if logger:
            logger.log(objective_value)
    else:
        assignment_list = []
        active_bin_periods = []
        objective_value = 0.0

    output_solution(args.solution_path, objective_value, assignment_list, active_bin_periods)

def output_solution(solution_path, objective_value, assignments, active_bin_periods):
    solution = {
        "objective_value": objective_value,
        "assignments": assignments,
        "active_bin_periods": active_bin_periods,
        "num_active_bin_periods": len(active_bin_periods)
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()