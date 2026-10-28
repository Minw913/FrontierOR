import json
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    panel_L = instance['panel']['length']
    panel_W = instance['panel']['width']
    items_data = instance['items']
    n = len(items_data)
    items = []
    for it in items_data:
        items.append((it['length'], it['width'], it['profit'], it['copies'], it['id']))
    zero = tuple([0] * n)

    item_lens = [it[0] for it in items]
    item_wids = [it[1] for it in items]

    def generate_sums(dims, limit):
        reachable = set([0, limit])
        for d in dims:
            if d <= limit:
                reachable.add(d)
        queue = list(reachable)
        while queue:
            if time.time() - start_time > time_limit * 0.2:
                break
            x = queue.pop()
            for d in dims:
                s = x + d
                if s <= limit and s not in reachable:
                    reachable.add(s)
                    queue.append(s)
        return sorted(reachable)

    xl = generate_sums(item_lens, panel_L)
    yw = generate_sums(item_wids, panel_W)
    xl_idx = {x: i for i, x in enumerate(xl)}
    yw_idx = {y: j for j, y in enumerate(yw)}

    # Precompute horizontal splits (l1 <= l2)
    h_splits = [[] for _ in range(len(xl))]
    for i in range(len(xl)):
        l = xl[i]
        for i1 in range(i):
            l1 = xl[i1]
            l2 = l - l1
            if l2 < 0:
                break
            if l1 > l2:
                continue
            if l2 in xl_idx:
                i2 = xl_idx[l2]
                h_splits[i].append((i1, i2))
            else:
                h_splits[i].append((i1, -1))

    # Precompute vertical splits (w1 <= w2)
    v_splits = [[] for _ in range(len(yw))]
    for j in range(len(yw)):
        w = yw[j]
        for j1 in range(j):
            w1 = yw[j1]
            w2 = w - w1
            if w2 < 0:
                break
            if w1 > w2:
                continue
            if w2 in yw_idx:
                j2 = yw_idx[w2]
                v_splits[j].append((j1, j2))
            else:
                v_splits[j].append((j1, -1))

    def add_counts(c1, c2):
        return tuple(a + b for a, b in zip(c1, c2))

    # Unlimited DP (for upper bound and initial solution)
    U = [[0] * len(yw) for _ in range(len(xl))]
    choice = [[None] * len(yw) for _ in range(len(xl))]

    for i in range(len(xl)):
        l = xl[i]
        for j in range(len(yw)):
            w = yw[j]
            if l == 0 or w == 0:
                U[i][j] = 0
                choice[i][j] = None
                continue
            best = 0
            best_choice = None
            for idx, it in enumerate(items):
                if it[0] == l and it[1] == w:
                    if it[2] > best:
                        best = it[2]
                        best_choice = ('item', idx)
            for i1, i2 in h_splits[i]:
                if i2 == -1:
                    val = U[i1][j]
                    if val > best:
                        best = val
                        best_choice = ('h', i1, -1)
                else:
                    val = U[i1][j] + U[i2][j]
                    if val > best:
                        best = val
                        best_choice = ('h', i1, i2)
            for j1, j2 in v_splits[j]:
                if j2 == -1:
                    val = U[i][j1]
                    if val > best:
                        best = val
                        best_choice = ('v', j1, -1)
                else:
                    val = U[i][j1] + U[i][j2]
                    if val > best:
                        best = val
                        best_choice = ('v', j1, j2)
            U[i][j] = best
            choice[i][j] = best_choice

    def traceback(i, j):
        ch = choice[i][j]
        if ch is None:
            return zero
        if ch[0] == 'item':
            counts = list(zero)
            counts[ch[1]] = 1
            return tuple(counts)
        elif ch[0] == 'h':
            i1 = ch[1]
            i2 = ch[2]
            if i2 == -1:
                return traceback(i1, j)
            else:
                return add_counts(traceback(i1, j), traceback(i2, j))
        else:
            j1 = ch[1]
            j2 = ch[2]
            if j2 == -1:
                return traceback(i, j1)
            else:
                return add_counts(traceback(i, j1), traceback(i, j2))

    best_obj = 0
    best_solution = {
        "objective_value": 0.0,
        "items_selected": []
    }

    if panel_L in xl_idx and panel_W in yw_idx:
        upper_bound = U[xl_idx[panel_L]][yw_idx[panel_W]]
        counts_unlimited = traceback(xl_idx[panel_L], yw_idx[panel_W])
        if all(counts_unlimited[k] <= items[k][3] for k in range(n)):
            best_obj = upper_bound
            sol_items = []
            for k, cnt in enumerate(counts_unlimited):
                if cnt > 0:
                    sol_items.append({
                        "item_id": items[k][4],
                        "plate_dims": [items[k][0], items[k][1]],
                        "copies": cnt
                    })
            best_solution = {
                "objective_value": float(best_obj),
                "items_selected": sol_items
            }
            if logger:
                logger.log_solution(best_obj, best_solution)
    else:
        upper_bound = 0

    # If the unlimited solution is already optimal and feasible, output it
    if best_obj == upper_bound and upper_bound > 0:
        with open(args.solution_path, 'w') as f:
            json.dump(best_solution, f, indent=2)
        return

    # Limited DP with beam search
    state_count = len(xl) * len(yw)
    if state_count <= 10000:
        beam_width = 10 ** 9
    elif state_count <= 50000:
        beam_width = 500
    elif state_count <= 200000:
        beam_width = 100
    else:
        beam_width = 20

    dp = [[None] * len(yw) for _ in range(len(xl))]
    dp_completed = False

    for i in range(len(xl)):
        l = xl[i]
        for j in range(len(yw)):
            w = yw[j]
            if l == 0 or w == 0:
                dp[i][j] = [(0, zero)]
                continue
            if time.time() - start_time > time_limit:
                break
            candidates = []
            # Exact item match
            for idx, it in enumerate(items):
                if it[0] == l and it[1] == w:
                    counts = list(zero)
                    counts[idx] = 1
                    candidates.append((it[2], tuple(counts)))
            # Horizontal cuts
            for i1, i2 in h_splits[i]:
                if i2 == -1:
                    list1 = dp[i1][j]
                    for p1, c1 in list1:
                        candidates.append((p1, c1))
                else:
                    list1 = dp[i1][j]
                    list2 = dp[i2][j]
                    for p1, c1 in list1:
                        for p2, c2 in list2:
                            candidates.append((p1 + p2, add_counts(c1, c2)))
            # Vertical cuts
            for j1, j2 in v_splits[j]:
                if j2 == -1:
                    list1 = dp[i][j1]
                    for p1, c1 in list1:
                        candidates.append((p1, c1))
                else:
                    list1 = dp[i][j1]
                    list2 = dp[i][j2]
                    for p1, c1 in list1:
                        for p2, c2 in list2:
                            candidates.append((p1 + p2, add_counts(c1, c2)))
            # Prune dominated solutions
            candidates.sort(key=lambda x: x[0], reverse=True)
            pruned = []
            for p, c in candidates:
                dom = False
                for p_k, c_k in pruned:
                    if p_k >= p and all(c_k[k] <= c[k] for k in range(n)):
                        dom = True
                        break
                if not dom:
                    new_pruned = []
                    for p_k, c_k in pruned:
                        if p >= p_k and all(c[k] <= c_k[k] for k in range(n)):
                            continue
                        new_pruned.append((p_k, c_k))
                    new_pruned.append((p, c))
                    pruned = new_pruned
            if beam_width < 10 ** 9 and len(pruned) > beam_width:
                pruned.sort(key=lambda x: x[0], reverse=True)
                pruned = pruned[:beam_width]
            dp[i][j] = pruned

            # Check panel solution
            if l == panel_L and w == panel_W:
                for p, c in pruned:
                    if all(c[k] <= items[k][3] for k in range(n)):
                        if p > best_obj:
                            best_obj = p
                            sol_items = []
                            for k, cnt in enumerate(c):
                                if cnt > 0:
                                    sol_items.append({
                                        "item_id": items[k][4],
                                        "plate_dims": [items[k][0], items[k][1]],
                                        "copies": cnt
                                    })
                            best_solution = {
                                "objective_value": float(best_obj),
                                "items_selected": sol_items
                            }
                            if logger:
                                logger.log_solution(best_obj, best_solution)
                            if best_obj == upper_bound:
                                dp_completed = True
                                break
                if dp_completed:
                    break
        if time.time() - start_time > time_limit or dp_completed:
            break
    else:
        dp_completed = True

    # Fallback if no solution was found
    if best_obj == 0 and not dp_completed:
        for idx, it in enumerate(items):
            if it[0] == panel_L and it[1] == panel_W and it[3] > 0:
                if it[2] > best_obj:
                    best_obj = it[2]
                    best_solution = {
                        "objective_value": float(best_obj),
                        "items_selected": [{
                            "item_id": it[4],
                            "plate_dims": [it[0], it[1]],
                            "copies": 1
                        }]
                    }
                    if logger:
                        logger.log_solution(best_obj, best_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == "__main__":
    main()