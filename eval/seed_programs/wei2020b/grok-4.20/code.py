import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    W = instance['W']
    V = instance['V']
    items = instance['items']
    weights = [item['w'] for item in items]
    volumes = [item['v'] for item in items]

    start_time = time.time()

    # Simple lower bound
    total_w = sum(weights)
    total_v = sum(volumes)
    lower_bound = max((total_w + W - 1) // W, (total_v + V - 1) // V)
    best_obj = n
    best_bins = None

    # Use First Fit Decreasing with multiple sorting heuristics
    def solve_with_sort(key_func):
        nonlocal best_obj, best_bins
        if time.time() - start_time > args.time_limit * 0.95:
            return

        indexed = list(range(n))
        indexed.sort(key=key_func, reverse=True)

        bins = []
        bin_w = []
        bin_v = []

        for i in indexed:
            placed = False
            for b_idx in range(len(bins)):
                if bin_w[b_idx] + weights[i] <= W and bin_v[b_idx] + volumes[i] <= V:
                    bins[b_idx].append(i)
                    bin_w[b_idx] += weights[i]
                    bin_v[b_idx] += volumes[i]
                    placed = True
                    break
            if not placed:
                bins.append([i])
                bin_w.append(weights[i])
                bin_v.append(volumes[i])

        num_bins = len(bins)
        if num_bins < best_obj:
            best_obj = num_bins
            solution_bins = []
            for b_idx, item_list in enumerate(bins):
                sorted_items = sorted(item_list)
                solution_bins.append({
                    "items": sorted_items,
                    "total_weight": bin_w[b_idx],
                    "total_volume": bin_v[b_idx]
                })
            best_bins = {
                "objective_value": best_obj,
                "bins": solution_bins
            }
            if logger:
                logger.log(best_obj)

    # Multiple heuristics
    heuristics = [
        lambda i: weights[i],                    # by weight
        lambda i: volumes[i],                    # by volume
        lambda i: weights[i] * 1.0 / W + volumes[i] * 1.0 / V,  # by normalized sum
        lambda i: max(weights[i] * 1.0 / W, volumes[i] * 1.0 / V),  # by max normalized
        lambda i: weights[i] * volumes[i],       # by area
        lambda i: weights[i] + volumes[i]        # by linear sum
    ]

    for h in heuristics:
        solve_with_sort(h)
        if best_obj == lower_bound:
            break

    # If still time left, try local improvement (swap items between bins)
    if time.time() - start_time < args.time_limit * 0.9 and best_bins is not None:
        improved = True
        current_bins = [b["items"][:] for b in best_bins["bins"]]
        current_w = [b["total_weight"] for b in best_bins["bins"]]
        current_v = [b["total_volume"] for b in best_bins["bins"]]
        current_obj = best_obj

        iterations = 0
        while improved and iterations < 100 and (time.time() - start_time < args.time_limit * 0.95):
            iterations += 1
            improved = False
            for i in range(n):
                # Find current bin of i
                src_bin = None
                for b_idx, blist in enumerate(current_bins):
                    if i in blist:
                        src_bin = b_idx
                        break
                if src_bin is None:
                    continue

                for dst_bin in range(len(current_bins)):
                    if dst_bin == src_bin:
                        continue
                    # Try moving i to dst_bin
                    if (current_w[dst_bin] + weights[i] - (weights[i] if False else 0) <= W and
                        current_v[dst_bin] + volumes[i] - (volumes[i] if False else 0) <= V):
                        # Check if removing from src would be ok (always is)
                        current_bins[src_bin].remove(i)
                        current_bins[dst_bin].append(i)
                        current_w[src_bin] -= weights[i]
                        current_v[src_bin] -= volumes[i]
                        current_w[dst_bin] += weights[i]
                        current_v[dst_bin] += volumes[i]

                        # If src bin empty, remove it
                        if not current_bins[src_bin]:
                            current_bins.pop(src_bin)
                            current_w.pop(src_bin)
                            current_v.pop(src_bin)
                            if src_bin < dst_bin:
                                dst_bin -= 1
                            improved = True
                            current_obj -= 1
                            break
                        improved = True
                        break

                if improved:
                    break

            if improved:
                best_obj = current_obj
                solution_bins = []
                for b_idx in range(len(current_bins)):
                    sorted_items = sorted(current_bins[b_idx])
                    solution_bins.append({
                        "items": sorted_items,
                        "total_weight": current_w[b_idx],
                        "total_volume": current_v[b_idx]
                    })
                best_bins = {
                    "objective_value": best_obj,
                    "bins": solution_bins
                }
                if logger:
                    logger.log(best_obj)

    # Final fallback: one item per bin
    if best_bins is None:
        solution_bins = []
        for i in range(n):
            solution_bins.append({
                "items": [i],
                "total_weight": weights[i],
                "total_volume": volumes[i]
            })
        best_bins = {
            "objective_value": n,
            "bins": solution_bins
        }
        if logger:
            logger.log(n)

    with open(args.solution_path, 'w') as f:
        json.dump(best_bins, f, indent=2)

if __name__ == "__main__":
    main()