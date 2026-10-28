import json
import argparse
import time
import random
import math
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    n = instance['num_customers']
    C = instance['vehicle_capacity']
    dist = instance['distance_matrix']

    delivery = [0] * (n + 1)
    pickup = [0] * (n + 1)
    for cust in instance['customers']:
        delivery[cust['id']] = cust['delivery_demand']
        pickup[cust['id']] = cust['pickup_demand']

    random.seed(42)

    def time_left():
        return time_limit - (time.time() - start_time)

    def route_cost(lh, bh):
        if not lh and not bh:
            return 0
        cost = 0
        prev = 0
        for node in lh:
            cost += dist[prev][node]
            prev = node
        for node in bh:
            cost += dist[prev][node]
            prev = node
        cost += dist[prev][0]
        return cost

    def total_cost(routes):
        return sum(route_cost(lh, bh) for lh, bh in routes)

    def route_delivery_load(lh):
        return sum(delivery[c] for c in lh)

    def route_pickup_load(bh):
        return sum(pickup[c] for c in bh)

    def format_solution(routes):
        obj = total_cost(routes)
        routes_out = []
        routes_detailed = []
        for lh, bh in routes:
            route_nodes = [0]
            route_det = [{"node": 0, "role": "depot", "customer": 0, "quantity": 0}]
            for c in lh:
                route_nodes.append(c)
                route_det.append({"node": c, "role": "linehaul", "customer": c, "quantity": delivery[c]})
            for c in bh:
                route_nodes.append(n + c)
                route_det.append({"node": n + c, "role": "backhaul", "customer": c, "quantity": pickup[c]})
            route_nodes.append(0)
            route_det.append({"node": 0, "role": "depot", "customer": 0, "quantity": 0})
            routes_out.append(route_nodes)
            routes_detailed.append(route_det)
        return {
            "objective_value": obj,
            "routes": routes_out,
            "routes_detailed": routes_detailed
        }

    # Construction: Savings-based
    def construct_savings():
        routes = []
        for i in range(1, n + 1):
            if delivery[i] > 0:
                routes.append(([i], []))
            if pickup[i] > 0:
                routes.append(([], [i]))

        improved = True
        while improved and time_left() > 2:
            improved = False
            best_saving = 0
            best_merge = None

            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    lh_i, bh_i = routes[i]
                    lh_j, bh_j = routes[j]

                    if lh_i and lh_j:
                        if route_delivery_load(lh_i) + route_delivery_load(lh_j) <= C:
                            saving = dist[lh_i[-1]][0] + dist[0][lh_j[0]] - dist[lh_i[-1]][lh_j[0]]
                            if saving > best_saving:
                                best_saving = saving
                                best_merge = (i, j, 'lh_lh')

                    if bh_i and bh_j:
                        if route_pickup_load(bh_i) + route_pickup_load(bh_j) <= C:
                            saving = dist[bh_i[-1]][0] + dist[0][bh_j[0]] - dist[bh_i[-1]][bh_j[0]]
                            if saving > best_saving:
                                best_saving = saving
                                best_merge = (i, j, 'bh_bh')

                    if lh_i and bh_j:
                        if route_pickup_load(bh_i) + route_pickup_load(bh_j) <= C:
                            saving = dist[lh_i[-1]][0] + dist[0][bh_j[0]] - dist[lh_i[-1]][bh_j[0]]
                            if saving > best_saving:
                                best_saving = saving
                                best_merge = (i, j, 'lh_bh')

                    if lh_j and bh_i:
                        if route_pickup_load(bh_i) + route_pickup_load(bh_j) <= C:
                            saving = dist[lh_j[-1]][0] + dist[0][bh_i[0]] - dist[lh_j[-1]][bh_i[0]]
                            if saving > best_saving:
                                best_saving = saving
                                best_merge = (i, j, 'lh_bh_rev')

            if best_merge and best_saving > 0:
                i, j, mtype = best_merge
                lh_i, bh_i = routes[i]
                lh_j, bh_j = routes[j]

                if mtype == 'lh_lh':
                    routes[i] = (lh_i + lh_j, bh_i + bh_j)
                elif mtype == 'bh_bh':
                    routes[i] = (lh_i + lh_j, bh_i + bh_j)
                elif mtype == 'lh_bh':
                    routes[i] = (lh_i + lh_j, bh_i + bh_j)
                elif mtype == 'lh_bh_rev':
                    routes[i] = (lh_j + lh_i, bh_j + bh_i)

                routes.pop(j)
                improved = True

        routes = [(lh, bh) for lh, bh in routes if lh or bh]
        return routes

    # Nearest neighbor construction
    def construct_nn():
        routes = []
        rem_d = set(range(1, n + 1))
        rem_p = set(range(1, n + 1))

        while rem_d or rem_p:
            lh = []
            bh = []
            load_d = 0
            load_p = 0
            prev = 0

            while rem_d:
                candidates = [(dist[prev][c], c) for c in rem_d if load_d + delivery[c] <= C]
                if not candidates:
                    break
                candidates.sort()
                _, best = candidates[0]
                lh.append(best)
                load_d += delivery[best]
                prev = best
                rem_d.remove(best)

            while rem_p:
                candidates = [(dist[prev][c], c) for c in rem_p if load_p + pickup[c] <= C]
                if not candidates:
                    break
                candidates.sort()
                _, best = candidates[0]
                bh.append(best)
                load_p += pickup[best]
                prev = best
                rem_p.remove(best)

            if lh or bh:
                routes.append((lh, bh))
            else:
                break

        return routes

    # 2-opt improvement for a sequence
    def two_opt_seq(seq):
        if len(seq) < 3:
            return seq
        improved = True
        while improved:
            improved = False
            for i in range(len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    prev_i = seq[i - 1] if i > 0 else 0
                    next_j = seq[j + 1] if j + 1 < len(seq) else 0
                    old = dist[prev_i][seq[i]] + dist[seq[j]][next_j]
                    new = dist[prev_i][seq[j]] + dist[seq[i]][next_j]
                    if new < old - 1e-10:
                        seq = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                        improved = True
        return seq

    # Or-opt for a sequence
    def or_opt_seq(seq):
        if len(seq) < 2:
            return seq
        for seg_len in [1, 2, 3]:
            if seg_len > len(seq):
                break
            improved = True
            while improved:
                improved = False
                for i in range(len(seq) - seg_len + 1):
                    seg = seq[i:i + seg_len]
                    rest = seq[:i] + seq[i + seg_len:]
                    # Original cost of seq
                    orig_cost = 0
                    prev = 0
                    for node in seq:
                        orig_cost += dist[prev][node]
                        prev = node
                    orig_cost += dist[prev][0]

                    best_cost = orig_cost
                    best_pos = -1
                    for j in range(len(rest) + 1):
                        new_seq = rest[:j] + seg + rest[j:]
                        cost = 0
                        prev = 0
                        for node in new_seq:
                            cost += dist[prev][node]
                            prev = node
                        cost += dist[prev][0]
                        if cost < best_cost - 1e-10:
                            best_cost = cost
                            best_pos = j

                    if best_pos >= 0:
                        seq = rest[:best_pos] + seg + rest[best_pos:]
                        improved = True
        return seq

    # Improve a single route
    def improve_route(lh, bh):
        if len(lh) >= 2:
            lh = two_opt_seq(lh[:])
        if len(bh) >= 2:
            bh = two_opt_seq(bh[:])
        if len(lh) >= 2:
            lh = or_opt_seq(lh[:])
        if len(bh) >= 2:
            bh = or_opt_seq(bh[:])
        return lh, bh

    # Full local search
    def local_search(routes, max_iter=50):
        for iteration in range(max_iter):
            if time_left() < 2:
                break
            improved = False

            # Intra-route improvements
            for i in range(len(routes)):
                old_cost = route_cost(routes[i][0], routes[i][1])
                lh, bh = improve_route(routes[i][0][:], routes[i][1][:])
                new_cost = route_cost(lh, bh)
                if new_cost < old_cost - 1e-10:
                    routes[i] = (lh, bh)
                    improved = True

            # Inter-route relocate (linehaul) - use while loop to avoid index issues
            for i in range(len(routes)):
                if time_left() < 2:
                    break
                k = 0
                while k < len(routes[i][0]):
                    lh_i, bh_i = routes[i]
                    c = lh_i[k]
                    best_delta = 0
                    best_j = -1
                    best_pos = -1
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        lh_j, bh_j = routes[j]
                        if route_delivery_load(lh_j) + delivery[c] > C:
                            continue
                        new_lh_i = lh_i[:k] + lh_i[k + 1:]
                        for pos in range(len(lh_j) + 1):
                            new_lh_j = lh_j[:pos] + [c] + lh_j[pos:]
                            delta = (route_cost(new_lh_i, bh_i) + route_cost(new_lh_j, bh_j)) - \
                                    (route_cost(lh_i, bh_i) + route_cost(lh_j, bh_j))
                            if delta < best_delta:
                                best_delta = delta
                                best_j = j
                                best_pos = pos
                    if best_j >= 0:
                        lh_i, bh_i = routes[i]
                        new_lh_i = lh_i[:k] + lh_i[k + 1:]
                        lh_j, bh_j = routes[best_j]
                        new_lh_j = lh_j[:best_pos] + [c] + lh_j[best_pos:]
                        routes[i] = (new_lh_i, bh_i)
                        routes[best_j] = (new_lh_j, bh_j)
                        improved = True
                        # Don't increment k since the list shifted
                    else:
                        k += 1

            # Inter-route relocate (backhaul) - use while loop
            for i in range(len(routes)):
                if time_left() < 2:
                    break
                k = 0
                while k < len(routes[i][1]):
                    lh_i, bh_i = routes[i]
                    c = bh_i[k]
                    best_delta = 0
                    best_j = -1
                    best_pos = -1
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        lh_j, bh_j = routes[j]
                        if route_pickup_load(bh_j) + pickup[c] > C:
                            continue
                        new_bh_i = bh_i[:k] + bh_i[k + 1:]
                        for pos in range(len(bh_j) + 1):
                            new_bh_j = bh_j[:pos] + [c] + bh_j[pos:]
                            delta = (route_cost(lh_i, new_bh_i) + route_cost(lh_j, new_bh_j)) - \
                                    (route_cost(lh_i, bh_i) + route_cost(lh_j, bh_j))
                            if delta < best_delta:
                                best_delta = delta
                                best_j = j
                                best_pos = pos
                    if best_j >= 0:
                        lh_i, bh_i = routes[i]
                        new_bh_i = bh_i[:k] + bh_i[k + 1:]
                        lh_j, bh_j = routes[best_j]
                        new_bh_j = bh_j[:best_pos] + [c] + bh_j[best_pos:]
                        routes[i] = (lh_i, new_bh_i)
                        routes[best_j] = (lh_j, new_bh_j)
                        improved = True
                    else:
                        k += 1

            # Remove empty routes
            routes = [(lh, bh) for lh, bh in routes if lh or bh]

            if not improved:
                break

        return routes

    # Simulated Annealing
    def simulated_annealing(routes, temp_start=50.0, cooling=0.9998, min_temp=0.01):
        current = deepcopy(routes)
        current_cost = total_cost(current)
        best = deepcopy(current)
        best_cost = current_cost
        temp = temp_start
        no_improve = 0

        while time_left() > 3 and temp > min_temp:
            new_routes = deepcopy(current)
            move_type = random.choice(['relocate_lh', 'relocate_bh', 'swap_lh', 'swap_bh', '2opt_lh', '2opt_bh', 'or_opt'])

            valid = True
            if move_type == 'relocate_lh':
                non_empty = [i for i in range(len(new_routes)) if new_routes[i][0]]
                if len(non_empty) < 2:
                    valid = False
                else:
                    i = random.choice(non_empty)
                    others = [x for x in range(len(new_routes)) if x != i]
                    j = random.choice(others)
                    k = random.randint(0, len(new_routes[i][0]) - 1)
                    c = new_routes[i][0][k]
                    if route_delivery_load(new_routes[j][0]) + delivery[c] > C:
                        valid = False
                    else:
                        lh_i = new_routes[i][0][:k] + new_routes[i][0][k + 1:]
                        pos = random.randint(0, len(new_routes[j][0]))
                        lh_j = new_routes[j][0][:pos] + [c] + new_routes[j][0][pos:]
                        new_routes[i] = (lh_i, new_routes[i][1])
                        new_routes[j] = (lh_j, new_routes[j][1])

            elif move_type == 'relocate_bh':
                non_empty = [i for i in range(len(new_routes)) if new_routes[i][1]]
                if len(non_empty) < 2:
                    valid = False
                else:
                    i = random.choice(non_empty)
                    others = [x for x in range(len(new_routes)) if x != i]
                    j = random.choice(others)
                    k = random.randint(0, len(new_routes[i][1]) - 1)
                    c = new_routes[i][1][k]
                    if route_pickup_load(new_routes[j][1]) + pickup[c] > C:
                        valid = False
                    else:
                        bh_i = new_routes[i][1][:k] + new_routes[i][1][k + 1:]
                        pos = random.randint(0, len(new_routes[j][1]))
                        bh_j = new_routes[j][1][:pos] + [c] + new_routes[j][1][pos:]
                        new_routes[i] = (new_routes[i][0], bh_i)
                        new_routes[j] = (new_routes[j][0], bh_j)

            elif move_type == 'swap_lh':
                non_empty = [i for i in range(len(new_routes)) if new_routes[i][0]]
                if len(non_empty) < 2:
                    valid = False
                else:
                    i, j = random.sample(non_empty, 2)
                    ki = random.randint(0, len(new_routes[i][0]) - 1)
                    kj = random.randint(0, len(new_routes[j][0]) - 1)
                    ci = new_routes[i][0][ki]
                    cj = new_routes[j][0][kj]
                    load_i = route_delivery_load(new_routes[i][0]) - delivery[ci] + delivery[cj]
                    load_j = route_delivery_load(new_routes[j][0]) - delivery[cj] + delivery[ci]
                    if load_i > C or load_j > C:
                        valid = False
                    else:
                        lh_i = new_routes[i][0][:ki] + [cj] + new_routes[i][0][ki + 1:]
                        lh_j = new_routes[j][0][:kj] + [ci] + new_routes[j][0][kj + 1:]
                        new_routes[i] = (lh_i, new_routes[i][1])
                        new_routes[j] = (lh_j, new_routes[j][1])

            elif move_type == 'swap_bh':
                non_empty = [i for i in range(len(new_routes)) if new_routes[i][1]]
                if len(non_empty) < 2:
                    valid = False
                else:
                    i, j = random.sample(non_empty, 2)
                    ki = random.randint(0, len(new_routes[i][1]) - 1)
                    kj = random.randint(0, len(new_routes[j][1]) - 1)
                    ci = new_routes[i][1][ki]
                    cj = new_routes[j][1][kj]
                    load_i = route_pickup_load(new_routes[i][1]) - pickup[ci] + pickup[cj]
                    load_j = route_pickup_load(new_routes[j][1]) - pickup[cj] + pickup[ci]
                    if load_i > C or load_j > C:
                        valid = False
                    else:
                        bh_i = new_routes[i][1][:ki] + [cj] + new_routes[i][1][ki + 1:]
                        bh_j = new_routes[j][1][:kj] + [ci] + new_routes[j][1][kj + 1:]
                        new_routes[i] = (new_routes[i][0], bh_i)
                        new_routes[j] = (new_routes[j][0], bh_j)

            elif move_type == '2opt_lh':
                candidates = [i for i in range(len(new_routes)) if len(new_routes[i][0]) >= 3]
                if not candidates:
                    valid = False
                else:
                    i = random.choice(candidates)
                    lh = new_routes[i][0][:]
                    idx1 = random.randint(0, len(lh) - 2)
                    idx2 = random.randint(idx1 + 1, len(lh) - 1)
                    lh = lh[:idx1] + lh[idx1:idx2 + 1][::-1] + lh[idx2 + 1:]
                    new_routes[i] = (lh, new_routes[i][1])

            elif move_type == '2opt_bh':
                candidates = [i for i in range(len(new_routes)) if len(new_routes[i][1]) >= 3]
                if not candidates:
                    valid = False
                else:
                    i = random.choice(candidates)
                    bh = new_routes[i][1][:]
                    idx1 = random.randint(0, len(bh) - 2)
                    idx2 = random.randint(idx1 + 1, len(bh) - 1)
                    bh = bh[:idx1] + bh[idx1:idx2 + 1][::-1] + bh[idx2 + 1:]
                    new_routes[i] = (new_routes[i][0], bh)

            elif move_type == 'or_opt':
                candidates = [i for i in range(len(new_routes)) if len(new_routes[i][0]) >= 2 or len(new_routes[i][1]) >= 2]
                if not candidates:
                    valid = False
                else:
                    i = random.choice(candidates)
                    lh, bh = new_routes[i]
                    if lh and (not bh or random.random() < 0.5):
                        seg_len = min(random.randint(1, 3), len(lh))
                        start = random.randint(0, len(lh) - seg_len)
                        seg = lh[start:start + seg_len]
                        rest = lh[:start] + lh[start + seg_len:]
                        pos = random.randint(0, len(rest))
                        lh = rest[:pos] + seg + rest[pos:]
                        new_routes[i] = (lh, bh)
                    elif bh:
                        seg_len = min(random.randint(1, 3), len(bh))
                        start = random.randint(0, len(bh) - seg_len)
                        seg = bh[start:start + seg_len]
                        rest = bh[:start] + bh[start + seg_len:]
                        pos = random.randint(0, len(rest))
                        bh = rest[:pos] + seg + rest[pos:]
                        new_routes[i] = (lh, bh)
                    else:
                        valid = False

            if not valid:
                temp *= cooling
                continue

            new_routes = [(lh, bh) for lh, bh in new_routes if lh or bh]
            new_cost = total_cost(new_routes)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = new_routes
                current_cost = new_cost
                if current_cost < best_cost:
                    best = deepcopy(current)
                    best_cost = current_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            temp *= cooling

            if no_improve > 5000:
                temp = temp_start * 0.5
                no_improve = 0

        return best, best_cost

    # Main algorithm
    routes1 = construct_nn()
    routes1 = local_search(routes1, max_iter=20)
    cost1 = total_cost(routes1)

    routes2 = construct_savings()
    routes2 = local_search(routes2, max_iter=20)
    cost2 = total_cost(routes2)

    if cost1 <= cost2:
        best_routes = routes1
        best_cost = cost1
    else:
        best_routes = routes2
        best_cost = cost2

    if logger:
        logger.log_solution(best_cost, format_solution(best_routes))

    # Run SA if time permits
    if time_left() > 5:
        sa_routes, sa_cost = simulated_annealing(deepcopy(best_routes))
        if sa_cost < best_cost:
            best_routes = sa_routes
            best_cost = sa_cost
            if logger:
                logger.log_solution(best_cost, format_solution(best_routes))

    # Final local search polish
    if time_left() > 2:
        best_routes = local_search(best_routes, max_iter=30)
        final_cost = total_cost(best_routes)
        if final_cost < best_cost:
            best_cost = final_cost
            if logger:
                logger.log_solution(best_cost, format_solution(best_routes))

    # Write output
    solution = format_solution(best_routes)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()