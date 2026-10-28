import argparse
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Solution logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    m = instance['num_periods']
    k = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']

    I0 = depot['initial_inventory']
    Imax0 = depot['max_inventory']
    prod = depot['production_per_period']
    h0 = depot['holding_cost']

    I = {c['id']: c['initial_inventory'] for c in customers}
    Imax = {c['id']: c['max_inventory'] for c in customers}
    Imin = {c['id']: c['min_inventory'] for c in customers}
    demand = {c['id']: c['demand_per_period'] for c in customers}
    h = {c['id']: c['holding_cost'] for c in customers}
    C = list(I.keys())

    start_time = time.time()
    periods_out = {}
    total_obj = 0.0

    for t in range(1, m + 1):
        I0 += prod

        mandatory = []
        optional = []
        for i in C:
            min_q = max(0, Imin[i] - (I[i] - demand[i]))
            max_q = Imax[i] - I[i]
            if min_q > 0:
                mandatory.append((i, min_q, max_q))
            elif max_q > 0:
                optional.append((i, 0, max_q))

        deliveries = {}
        I0_proj = I0
        for i, min_q, max_q in mandatory:
            deliveries[i] = max_q
            I0_proj -= max_q

        opt_sorted = sorted(optional, key=lambda x: x[2], reverse=True)
        opt_idx = 0
        while I0_proj > Imax0 and opt_idx < len(opt_sorted):
            i, _, max_q = opt_sorted[opt_idx]
            if i not in deliveries:
                deliveries[i] = max_q
                I0_proj -= max_q
            opt_idx += 1

        if I0_proj < 0:
            for i, _, _ in opt_sorted:
                if i in deliveries and I[i] - demand[i] >= Imin[i]:
                    reduce = min(deliveries[i], -I0_proj)
                    deliveries[i] -= reduce
                    I0_proj += reduce
                    if deliveries[i] == 0: del deliveries[i]
                    if I0_proj >= 0: break
            if I0_proj < 0:
                for i, min_q, max_q in mandatory:
                    if i in deliveries and deliveries[i] > min_q:
                        reduce = min(deliveries[i] - min_q, -I0_proj)
                        deliveries[i] -= reduce
                        I0_proj += reduce
                        if I0_proj >= 0: break
            if I0_proj < 0:
                for i, min_q, max_q in mandatory:
                    if i in deliveries and deliveries[i] > 0:
                        reduce = min(deliveries[i], -I0_proj)
                        deliveries[i] -= reduce
                        I0_proj += reduce
                        if I0_proj >= 0: break

        total_q = sum(deliveries.values())
        if total_q > k * Q:
            for i, _, _ in opt_sorted:
                if i in deliveries and I[i] - demand[i] >= Imin[i]:
                    reduce = min(deliveries[i], total_q - k * Q)
                    deliveries[i] -= reduce
                    total_q -= reduce
                    if deliveries[i] == 0: del deliveries[i]
                    if total_q <= k * Q: break
            if total_q > k * Q:
                for i, min_q, max_q in mandatory:
                    if i in deliveries and deliveries[i] > min_q:
                        reduce = min(deliveries[i] - min_q, total_q - k * Q)
                        deliveries[i] -= reduce
                        total_q -= reduce
                        if total_q <= k * Q: break
            if total_q > k * Q:
                for i, min_q, max_q in mandatory:
                    if i in deliveries and deliveries[i] > 0:
                        reduce = min(deliveries[i], total_q - k * Q)
                        deliveries[i] -= reduce
                        total_q -= reduce
                        if total_q <= k * Q: break

        routes = []
        loads = []
        remaining_cust = set(deliveries.keys())
        while remaining_cust and len(routes) < k:
            curr = 0
            route = [0]
            load = 0
            while True:
                best_c = -1
                best_dist = float('inf')
                for c in remaining_cust:
                    if load + deliveries[c] <= Q:
                        if dist[curr][c] < best_dist:
                            best_dist = dist[curr][c]
                            best_c = c
                if best_c == -1: break
                route.append(best_c)
                load += deliveries[best_c]
                remaining_cust.remove(best_c)
                curr = best_c
            route.append(0)
            routes.append(route)
            loads.append(load)

        if remaining_cust:
            for c in list(remaining_cust):
                for r_idx in range(len(routes)):
                    if loads[r_idx] + deliveries[c] <= Q:
                        best_pos = 1
                        best_cost = float('inf')
                        for pos in range(1, len(routes[r_idx])):
                            prev = routes[r_idx][pos-1]
                            nxt = routes[r_idx][pos]
                            cost = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                        routes[r_idx].insert(best_pos, c)
                        loads[r_idx] += deliveries[c]
                        remaining_cust.remove(c)
                        break
                else:
                    if routes:
                        r_idx = min(range(len(routes)), key=lambda i: loads[i])
                        best_pos = 1
                        best_cost = float('inf')
                        for pos in range(1, len(routes[r_idx])):
                            prev = routes[r_idx][pos-1]
                            nxt = routes[r_idx][pos]
                            cost = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                        routes[r_idx].insert(best_pos, c)
                        loads[r_idx] += deliveries[c]
                        remaining_cust.remove(c)

        optional_remaining = [c for c in C if c not in deliveries and Imax[c] - I[c] > 0]
        improved = True
        while improved:
            improved = False
            best_insertion = None
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                load = loads[r_idx]
                for c in optional_remaining:
                    max_q = Imax[c] - I[c]
                    if max_q > 0 and max_q <= Q - load and max_q <= I0:
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            cost = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                            if best_insertion is None or cost < best_insertion[0]:
                                best_insertion = (cost, r_idx, c, pos, max_q)
            if best_insertion:
                cost, r_idx, c, pos, max_q = best_insertion
                if I0 - sum(deliveries.values()) > Imax0 or cost <= 0:
                    routes[r_idx].insert(pos, c)
                    loads[r_idx] += max_q
                    deliveries[c] = max_q
                    optional_remaining.remove(c)
                    improved = True

        final_routes = []
        for route in routes:
            if len(route) <= 3:
                final_routes.append(route)
                continue
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        a, b = route[i-1], route[i]
                        c, dd = route[j], route[j+1]
                        if dist[a][c] + dist[b][dd] < dist[a][b] + dist[c][dd]:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
            final_routes.append(route)

        route_cost = 0.0
        for route in final_routes:
            for i in range(len(route) - 1):
                route_cost += dist[route[i]][route[i+1]]

        total_delivered = sum(deliveries.values())
        I0 -= total_delivered
        if I0 < 0: I0 = 0
        
        inv_cost = h0 * I0
        for i in C:
            I[i] = I[i] + deliveries.get(i, 0) - demand[i]
            if I[i] < 0: I[i] = 0
            inv_cost += h[i] * I[i]

        total_obj += route_cost + inv_cost

        periods_out[str(t)] = {
            "routes": final_routes,
            "deliveries": {str(k): v for k, v in deliveries.items()},
            "inventories": {
                "depot": I0,
                **{str(i): I[i] for i in C}
            }
        }

    solution = {
        "objective_value": total_obj,
        "solution_details": {
            "periods": periods_out
        }
    }

    if logger:
        logger.log_solution(total_obj, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()