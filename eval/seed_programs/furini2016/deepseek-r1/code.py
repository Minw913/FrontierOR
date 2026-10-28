import json
import argparse
from collections import defaultdict
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    panel = instance['panel']
    L = panel['length']
    W = panel['width']
    items = instance['items']

    items_by_dim = defaultdict(list)
    best_item = {}
    item_info_by_id = {}
    for item in items:
        l_i = item['length']
        w_i = item['width']
        key = (l_i, w_i)
        items_by_dim[key].append((item['id'], item['profit'], item['copies']))
        current_best = best_item.get(key, 0)
        if item['profit'] > current_best:
            best_item[key] = item['profit']
        item_info_by_id[item['id']] = item

    for key in items_by_dim:
        items_by_dim[key] = sorted(items_by_dim[key], key=lambda x: x[1], reverse=True)

    start_time = time.time()

    if L == 0 or W == 0:
        dp = [[0] * (W+1) for _ in range(L+1)]
        choice = [['discard'] * (W+1) for _ in range(L+1)]
    else:
        dp = [[0] * (W+1) for _ in range(L+1)]
        choice = [[None] * (W+1) for _ in range(L+1)]

        for l in range(0, L+1):
            for w in range(0, W+1):
                if l == 0 or w == 0:
                    dp[l][w] = 0
                    choice[l][w] = 'discard'
                    continue

                candidate_item = best_item.get((l, w), 0)
                if candidate_item > 0:
                    dp[l][w] = candidate_item
                    choice[l][w] = 'item'
                else:
                    dp[l][w] = 0
                    choice[l][w] = 'discard'

        time_limit_exceeded = False
        for l in range(1, L+1):
            current_time = time.time()
            if current_time - start_time > args.time_limit:
                time_limit_exceeded = True
                break
            for w in range(1, W+1):
                if w % 100 == 0:
                    current_time = time.time()
                    if current_time - start_time > args.time_limit:
                        time_limit_exceeded = True
                        break
                candidate_item = best_item.get((l, w), 0)
                best_val = candidate_item
                if candidate_item > 0:
                    current_choice = 'item'
                else:
                    current_choice = 'discard'

                for k in range(1, l):
                    candidate = dp[k][w] + dp[l-k][w]
                    if candidate > best_val:
                        best_val = candidate
                        current_choice = ('v', k)

                for k in range(1, w):
                    candidate = dp[l][k] + dp[l][w-k]
                    if candidate > best_val:
                        best_val = candidate
                        current_choice = ('h', k)

                dp[l][w] = best_val
                choice[l][w] = current_choice

            if time_limit_exceeded:
                break

    if L < len(dp) and W < len(dp[0]) and dp[L][W] != 0 and (time.time() - start_time <= args.time_limit or dp[L][W] is not None):
        def backtrack(l, w):
            stack = [(l, w)]
            plates = []
            while stack:
                l_curr, w_curr = stack.pop()
                if l_curr < 0 or w_curr < 0 or l_curr > L or w_curr > W:
                    continue
                ch = choice[l_curr][w_curr]
                if ch == 'discard':
                    continue
                elif ch == 'item':
                    plates.append((l_curr, w_curr))
                elif isinstance(ch, tuple):
                    if ch[0] == 'v':
                        k = ch[1]
                        stack.append((k, w_curr))
                        stack.append((l_curr - k, w_curr))
                    elif ch[0] == 'h':
                        k = ch[1]
                        stack.append((l_curr, k))
                        stack.append((l_curr, w_curr - k))
            return plates

        plate_list = backtrack(L, W)
        freq = defaultdict(int)
        for p in plate_list:
            freq[p] += 1

        usage_by_id = defaultdict(int)
        total_profit = 0
        for (l, w), count in freq.items():
            items_list = items_by_dim.get((l, w), [])
            remaining = count
            for (id_i, p_i, copies_i) in items_list:
                if remaining <= 0:
                    break
                use = min(remaining, copies_i)
                usage_by_id[id_i] += use
                total_profit += use * p_i
                remaining -= use
    else:
        total_profit = 0
        usage_by_id = {}

    items_selected = []
    for item_id, copies in usage_by_id.items():
        item = item_info_by_id[item_id]
        items_selected.append({
            "item_id": item_id,
            "plate_dims": [item['length'], item['width']],
            "copies": copies
        })

    solution = {
        "objective_value": total_profit,
        "items_selected": items_selected
    }

    if logger:
        logger.log(total_profit)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, separators=(',', ':'))

if __name__ == "__main__":
    main()