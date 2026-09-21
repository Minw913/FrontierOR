import argparse
import json
import time
import random
import bisect
import sys

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.5

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    params = data["parameters"]
    n_customers = int(params["n_customers"])
    total_vertices = int(params["total_vertices"])
    H = int(params["H"])
    B = [float(x) for x in params["period_boundaries"]]
    depot = int(data["depot"]["index"])

    D = data["arcs"]["distances"]
    V = data["arcs"]["speeds_v_ijh"]

    n = total_vertices
    EPS = 1e-9

    # ---------------- travel time function ----------------
    def travel_time(i, j, t0):
        d = D[i][j]
        if d <= 0.0:
            return 0.0
        t = t0
        h = bisect.bisect_right(B, t) - 1
        if h < 0:
            h = 0
        if h >= H:
            h = H - 1
        Vij = V[i][j]
        lastH = H - 1
        while True:
            v = Vij[h]
            if v <= 1e-12:
                if h == lastH:
                    return 1e18
                t = B[h + 1]
                h += 1
                continue
            arr = t + d / v
            if h == lastH or arr <= B[h + 1] + 1e-12:
                return arr - t0
            d -= v * (B[h + 1] - t)
            t = B[h + 1]
            h += 1

    def eval_tour(tour):
        t = 0.0
        for k in range(len(tour) - 1):
            t += travel_time(tour[k], tour[k + 1], t)
        return t

    def prefix_times(tour):
        # T[k] = arrival time at tour[k]
        T = [0.0] * len(tour)
        t = 0.0
        for k in range(len(tour) - 1):
            t += travel_time(tour[k], tour[k + 1], t)
            T[k + 1] = t
        return T

    def eval_from(tour, start_idx, t_start, prune=None):
        # Evaluate cost of tour starting at index start_idx with time t_start
        t = t_start
        m = len(tour) - 1
        for k in range(start_idx, m):
            t += travel_time(tour[k], tour[k + 1], t)
            if prune is not None and t >= prune:
                return t
        return t

    # ---------------- trivial cases ----------------
    customers = [v for v in range(n) if v != depot]
    if n_customers == 0 or len(customers) == 0:
        tour = [depot, depot]
        obj = 0.0
        if logger:
            logger.log_solution(obj, {"objective_value": obj, "tour": tour})
        with open(args.solution_path, "w") as f:
            json.dump({"objective_value": obj, "tour": tour}, f)
        return

    # ---------------- construction: nearest neighbor (time-dependent) ----------------
    def nearest_neighbor(start_perm_seed=None):
        unvisited = set(customers)
        tour = [depot]
        t = 0.0
        cur = depot
        while unvisited:
            best_j = None
            best_tt = None
            for j in unvisited:
                tt = travel_time(cur, j, t)
                if best_tt is None or tt < best_tt:
                    best_tt = tt
                    best_j = j
            tour.append(best_j)
            t += best_tt
            unvisited.remove(best_j)
            cur = best_j
        tour.append(depot)
        return tour

    best_tour = nearest_neighbor()
    best_obj = eval_tour(best_tour)
    if logger:
        logger.log_solution(best_obj, {"objective_value": best_obj, "tour": list(best_tour)})

    rng = random.Random(0)
    nc = len(customers)  # number of interior positions

    # ---------------- local search ----------------
    def local_search(tour, obj):
        # tour: list [depot, c..., depot], obj: its cost
        improved_any = True
        while improved_any:
            if time.time() > deadline:
                break
            improved_any = False
            T = prefix_times(tour)
            L = len(tour)

            # ----- 2-opt (segment reversal) -----
            done = False
            for i in range(1, L - 2):
                if time.time() > deadline:
                    done = True
                    break
                for j in range(i + 1, L - 1):
                    # new tour: tour[:i] + reversed(tour[i:j+1]) + tour[j+1:]
                    cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    new_obj = eval_from(cand, i - 1, T[i - 1], prune=obj)
                    if new_obj < obj - 1e-7:
                        tour = cand
                        obj = new_obj
                        T = prefix_times(tour)
                        improved_any = True
                if done:
                    break
            if done:
                break

            # ----- Or-opt: relocate segments of length 1..3 (with optional reversal) -----
            T = prefix_times(tour)
            L = len(tour)
            done = False
            for seg_len in (1, 2, 3):
                if seg_len > nc:
                    break
                for i in range(1, L - 1 - seg_len + 1):
                    if time.time() > deadline:
                        done = True
                        break
                    seg = tour[i:i + seg_len]
                    rest = tour[:i] + tour[i + seg_len:]
                    # insert at positions 1..len(rest)-1
                    for pos in range(1, len(rest)):
                        if pos == i:
                            continue
                        for rev in (False, True):
                            s = seg[::-1] if rev else seg
                            if rev and seg_len == 1:
                                continue
                            cand = rest[:pos] + s + rest[pos:]
                            start = min(i, pos) - 1
                            new_obj = eval_from(cand, start, T[start] if start < len(T) else 0.0, prune=obj)
                            # T corresponds to old tour; prefix up to 'start' identical
                            if new_obj < obj - 1e-7:
                                tour = cand
                                obj = new_obj
                                T = prefix_times(tour)
                                improved_any = True
                                seg = tour[i:i + seg_len] if i + seg_len <= len(tour) - 1 else None
                                rest = None
                                break
                        if rest is None:
                            break
                    if rest is None:
                        # restart or-opt scanning for this i is messy; just continue outer loop
                        break
                if done:
                    break
            if done:
                break
        return tour, obj

    def double_bridge(tour):
        # interior indices 1..L-2
        Ln = len(tour)
        if Ln < 8:
            # simple random swap perturbation
            t = list(tour)
            if Ln > 4:
                a, b = rng.sample(range(1, Ln - 1), 2)
                t[a], t[b] = t[b], t[a]
            return t
        pts = sorted(rng.sample(range(1, Ln - 1), 3))
        a, b, c = pts
        return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]

    # Initial local search
    best_tour, best_obj = local_search(best_tour, best_obj)
    if logger:
        logger.log_solution(best_obj, {"objective_value": best_obj, "tour": list(best_tour)})

    cur_tour, cur_obj = list(best_tour), best_obj

    # ---------------- iterated local search ----------------
    while time.time() < deadline:
        pert = double_bridge(cur_tour)
        pert_obj = eval_tour(pert)
        pert, pert_obj = local_search(pert, pert_obj)
        if pert_obj < cur_obj - 1e-7:
            cur_tour, cur_obj = pert, pert_obj
        else:
            # occasionally accept slightly worse to diversify
            if rng.random() < 0.05:
                cur_tour, cur_obj = pert, pert_obj
            else:
                cur_tour, cur_obj = list(best_tour), best_obj
        if cur_obj < best_obj - 1e-7:
            best_tour, best_obj = list(cur_tour), cur_obj
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj, "tour": list(best_tour)})

    # final verification
    final_obj = eval_tour(best_tour)
    solution = {"objective_value": final_obj, "tour": [int(v) for v in best_tour]}
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()