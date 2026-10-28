import argparse
import json
import time
import random
import math
from solution_logger import SolutionLogger

# Global variables (populated in main)
depot = 0
dist = []
linehaul_demand = {}
backhaul_demand = {}
all_linehaul = []
all_backhaul = []
Q = 0
K = 0
total_customers = 0

class Route:
    def __init__(self, depot_idx):
        self.nodes = [depot_idx, depot_idx]
        self.linehaul_load = 0
        self.backhaul_load = 0

    def is_empty(self):
        return len(self.nodes) == 2 and self.nodes[0] == depot and self.nodes[1] == depot

    def has_linehaul(self):
        return self.linehaul_load > 0

    def has_backhaul(self):
        return self.backhaul_load > 0

    def get_backhaul_start(self):
        """Return the index of the first backhaul node, or the end depot if none."""
        for i in range(1, len(self.nodes)):
            if self.nodes[i] in backhaul_demand:
                return i
        return len(self.nodes) - 1

    def cost(self):
        c = 0
        n = self.nodes
        for i in range(len(n) - 1):
            c += dist[n[i]][n[i+1]]
        return c


class Solution:
    def __init__(self, routes):
        self.routes = routes
        self.cost = sum(r.cost() for r in routes)

    def copy(self):
        new_routes = []
        for r in self.routes:
            new_r = Route(depot)
            new_r.nodes = r.nodes[:]
            new_r.linehaul_load = r.linehaul_load
            new_r.backhaul_load = r.backhaul_load
            new_routes.append(new_r)
        return Solution(new_routes)


def solution_cost(routes):
    return sum(r.cost() for r in routes)


def solution_to_dict(sol):
    return {
        "objective_value": float(sol.cost),
        "routes": [r.nodes for r in sol.routes]
    }


def get_all_customers(routes):
    customers = []
    for r in routes:
        for node in r.nodes[1:-1]:
            customers.append(node)
    return customers


def feasible_insertions(cust, route):
    """Return list of (position, insertion_cost) for feasible insertions."""
    if cust in linehaul_demand:
        if route.linehaul_load + linehaul_demand[cust] > Q:
            return []
        end = route.get_backhaul_start()
        positions = range(1, end + 1)
    else:
        if route.backhaul_load + backhaul_demand[cust] > Q:
            return []
        if not route.has_linehaul():
            return []
        start = route.get_backhaul_start()
        positions = range(start, len(route.nodes))

    results = []
    for pos in positions:
        prev = route.nodes[pos - 1]
        nxt = route.nodes[pos]
        cost = dist[prev][cust] + dist[cust][nxt] - dist[prev][nxt]
        results.append((pos, cost))
    return results


def insert_customer_to_route(route, pos, cust):
    route.nodes.insert(pos, cust)
    if cust in linehaul_demand:
        route.linehaul_load += linehaul_demand[cust]
    else:
        route.backhaul_load += backhaul_demand[cust]


def remove_customer_from_route(route, cust):
    idx = route.nodes.index(cust)
    route.nodes.pop(idx)
    if cust in linehaul_demand:
        route.linehaul_load -= linehaul_demand[cust]
    else:
        route.backhaul_load -= backhaul_demand[cust]


# ------------------- Construction -------------------

def construct_initial_solution(time_limit):
    start = time.time()
    for attempt in range(50):
        if time.time() - start > time_limit * 0.3:
            break
        routes = [Route(depot) for _ in range(K)]
        line_order = all_linehaul[:]
        random.shuffle(line_order)
        back_order = all_backhaul[:]
        random.shuffle(back_order)

        success = True
        # Insert linehaul
        for cust in line_order:
            best_route = None
            best_pos = None
            best_cost = float('inf')
            for r in routes:
                feasible = feasible_insertions(cust, r)
                for pos, cost in feasible:
                    if cost < best_cost:
                        best_cost = cost
                        best_route = r
                        best_pos = pos
            if best_route is None:
                # Fallback: best fit by capacity
                best_route = None
                best_remaining = -1
                for r in routes:
                    if r.linehaul_load + linehaul_demand[cust] <= Q:
                        rem = Q - (r.linehaul_load + linehaul_demand[cust])
                        if best_route is None or rem < best_remaining:
                            best_route = r
                            best_remaining = rem
                            best_pos = r.get_backhaul_start()
                if best_route is None:
                    success = False
                    break
            insert_customer_to_route(best_route, best_pos, cust)

        if not success:
            continue

        # Insert backhaul
        for cust in back_order:
            best_route = None
            best_pos = None
            best_cost = float('inf')
            for r in routes:
                if not r.has_linehaul():
                    continue
                feasible = feasible_insertions(cust, r)
                for pos, cost in feasible:
                    if cost < best_cost:
                        best_cost = cost
                        best_route = r
                        best_pos = pos
            if best_route is None:
                best_route = None
                best_remaining = -1
                for r in routes:
                    if not r.has_linehaul():
                        continue
                    if r.backhaul_load + backhaul_demand[cust] <= Q:
                        rem = Q - (r.backhaul_load + backhaul_demand[cust])
                        if best_route is None or rem < best_remaining:
                            best_route = r
                            best_remaining = rem
                            best_pos = len(r.nodes) - 1
                if best_route is None:
                    success = False
                    break
            insert_customer_to_route(best_route, best_pos, cust)

        if not success:
            continue

        # Verify all inserted
        inserted_line = set()
        inserted_back = set()
        for r in routes:
            for node in r.nodes[1:-1]:
                if node in linehaul_demand:
                    inserted_line.add(node)
                else:
                    inserted_back.add(node)
        if len(inserted_line) == len(all_linehaul) and len(inserted_back) == len(all_backhaul):
            return routes

    # Deterministic fallback
    routes = [Route(depot) for _ in range(K)]
    line_sorted = sorted(all_linehaul, key=lambda c: linehaul_demand[c], reverse=True)
    for cust in line_sorted:
        for r in routes:
            if r.linehaul_load + linehaul_demand[cust] <= Q:
                insert_customer_to_route(r, r.get_backhaul_start(), cust)
                break
        else:
            # Force into first route (should not happen on feasible instances)
            insert_customer_to_route(routes[0], routes[0].get_backhaul_start(), cust)

    back_sorted = sorted(all_backhaul, key=lambda c: backhaul_demand[c], reverse=True)
    for cust in back_sorted:
        for r in routes:
            if r.has_linehaul() and r.backhaul_load + backhaul_demand[cust] <= Q:
                insert_customer_to_route(r, len(r.nodes) - 1, cust)
                break
        else:
            for r in routes:
                if r.has_linehaul():
                    insert_customer_to_route(r, len(r.nodes) - 1, cust)
                    break
    return routes


# ------------------- Removal operators -------------------

def random_removal(routes, q):
    customers = []
    for r_idx, r in enumerate(routes):
        for pos, node in enumerate(r.nodes):
            if pos == 0 or pos == len(r.nodes) - 1:
                continue
            customers.append((node, r_idx, pos))
    random.shuffle(customers)
    removed = []
    linehaul_counts = [sum(1 for n in r.nodes[1:-1] if n in linehaul_demand) for r in routes]
    for cust, r_idx, pos in customers:
        if len(removed) >= q:
            break
        r = routes[r_idx]
        if cust in linehaul_demand:
            if linehaul_counts[r_idx] == 1 and r.has_backhaul():
                continue
            linehaul_counts[r_idx] -= 1
        removed.append(cust)
    return removed


def worst_removal(routes, q):
    customers = []
    for r_idx, r in enumerate(routes):
        for pos, node in enumerate(r.nodes):
            if pos == 0 or pos == len(r.nodes) - 1:
                continue
            prev = r.nodes[pos - 1]
            nxt = r.nodes[pos + 1]
            gain = dist[prev][node] + dist[node][nxt] - dist[prev][nxt]
            customers.append((node, r_idx, pos, gain))
    customers.sort(key=lambda x: -x[3])
    removed = []
    linehaul_counts = [sum(1 for n in r.nodes[1:-1] if n in linehaul_demand) for r in routes]
    for cust, r_idx, pos, gain in customers:
        if len(removed) >= q:
            break
        r = routes[r_idx]
        if cust in linehaul_demand:
            if linehaul_counts[r_idx] == 1 and r.has_backhaul():
                continue
            linehaul_counts[r_idx] -= 1
        removed.append(cust)
    if len(removed) > q:
        removed = removed[:q]
    return removed


def related_removal(routes, q):
    customers = []
    for r_idx, r in enumerate(routes):
        for pos, node in enumerate(r.nodes):
            if pos == 0 or pos == len(r.nodes) - 1:
                continue
            customers.append((node, r_idx, pos))
    seed = random.choice(customers)
    seed_node = seed[0]

    def relatedness(c):
        return (dist[seed_node][c] + dist[c][seed_node]) / 2.0

    customers.sort(key=lambda x: relatedness(x[0]))
    removed = []
    linehaul_counts = [sum(1 for n in r.nodes[1:-1] if n in linehaul_demand) for r in routes]
    for cust, r_idx, pos in customers:
        if len(removed) >= q:
            break
        r = routes[r_idx]
        if cust in linehaul_demand:
            if linehaul_counts[r_idx] == 1 and r.has_backhaul():
                continue
            linehaul_counts[r_idx] -= 1
        removed.append(cust)
    return removed


def cluster_removal(routes, q):
    eligible = [i for i, r in enumerate(routes) if len(r.nodes) > 2]
    if not eligible:
        return []
    r_idx = random.choice(eligible)
    r = routes[r_idx]
    route_customers = r.nodes[1:-1]
    q_actual = min(q, len(route_customers))
    if len(route_customers) <= q_actual:
        removed = route_customers[:]
    else:
        start = random.randint(0, len(route_customers) - q_actual)
        removed = route_customers[start:start + q_actual]

    remaining_linehaul = [c for c in route_customers if c not in removed and c in linehaul_demand]
    remaining_backhaul = [c for c in route_customers if c not in removed and c in backhaul_demand]
    if len(remaining_linehaul) == 0 and len(remaining_backhaul) > 0:
        removed.extend(remaining_backhaul)
    return removed


# ------------------- Insertion operators -------------------

def insert_greedy(routes, cust):
    best_route = None
    best_pos = None
    best_cost = float('inf')
    for r in routes:
        feasible = feasible_insertions(cust, r)
        for pos, cost in feasible:
            if cost < best_cost:
                best_cost = cost
                best_route = r
                best_pos = pos
    if best_route is None:
        return False
    insert_customer_to_route(best_route, best_pos, cust)
    return True


def insert_regret(routes, removed_list):
    removed = list(removed_list)
    while removed:
        best_regret = -1
        best_cust = None
        best_insertion = None
        for cust in removed:
            insertions = []
            for r in routes:
                feasible = feasible_insertions(cust, r)
                for pos, cost in feasible:
                    insertions.append((r, pos, cost))
            if not insertions:
                continue
            insertions.sort(key=lambda x: x[2])
            best_cost = insertions[0][2]
            second_cost = insertions[1][2] if len(insertions) > 1 else best_cost
            regret = second_cost - best_cost
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_insertion = insertions[0]
        if best_cust is None:
            return False
        r, pos, _ = best_insertion
        insert_customer_to_route(r, pos, best_cust)
        removed.remove(best_cust)
    return True


# ------------------- Local search -------------------

def local_search(routes, start_time=None, time_limit=None, max_passes=20):
    for _ in range(max_passes):
        if start_time is not None and time.time() - start_time > time_limit:
            break
        improved = False
        customers = get_all_customers(routes)
        for cust in customers:
            for r_idx, r in enumerate(routes):
                if cust in r.nodes:
                    pos = r.nodes.index(cust)
                    break
            else:
                continue
            r1 = routes[r_idx]
            prev = r1.nodes[pos - 1]
            nxt = r1.nodes[pos + 1]
            curr_cost = dist[prev][cust] + dist[cust][nxt] - dist[prev][nxt]
            best_delta = 0
            best_move = None
            for r2_idx, r2 in enumerate(routes):
                if r2_idx == r_idx:
                    continue
                feasible = feasible_insertions(cust, r2)
                for pos2, cost in feasible:
                    delta = cost - curr_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (r2_idx, pos2)
            if best_move:
                r2_idx, pos2 = best_move
                remove_customer_from_route(r1, cust)
                insert_customer_to_route(routes[r2_idx], pos2, cust)
                improved = True
                break
        if not improved:
            break


# ------------------- Main -------------------

def main():
    global depot, dist, linehaul_demand, backhaul_demand, all_linehaul, all_backhaul, Q, K, total_customers

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    depot = instance['depot']['index']
    dist = instance['distance_matrix']
    linehaul = instance['linehaul_customers']
    backhaul = instance['backhaul_customers']
    Q = instance['vehicle_capacity']
    K = instance['num_vehicles']

    linehaul_demand = {c['index']: c['demand'] for c in linehaul}
    backhaul_demand = {c['index']: c['demand'] for c in backhaul}
    all_linehaul = [c['index'] for c in linehaul]
    all_backhaul = [c['index'] for c in backhaul]
    total_customers = len(all_linehaul) + len(all_backhaul)

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    random.seed(0)

    initial_routes = construct_initial_solution(args.time_limit)
    current = Solution(initial_routes)
    best = current.copy()
    best_cost = best.cost

    if logger:
        logger.log_solution(best_cost, solution_to_dict(best))

    q_min = 1
    q_max = min(max(1, int(0.2 * total_customers)), 30)
    if q_max < 1:
        q_max = 1

    removal_ops = ['random', 'worst', 'related', 'cluster']
    insertion_ops = ['greedy', 'regret']
    removal_weights = {op: 1.0 for op in removal_ops}
    insertion_weights = {op: 1.0 for op in insertion_ops}
    removal_scores = {op: 0.0 for op in removal_ops}
    insertion_scores = {op: 0.0 for op in insertion_ops}
    removal_usage = {op: 0 for op in removal_ops}
    insertion_usage = {op: 0 for op in insertion_ops}

    rho = 0.1
    segment_size = 100
    temp = 0.05 * best_cost if best_cost > 0 else 1.0
    min_temp = 1e-3
    temp_factor = 0.9999

    start_time = time.time()
    iteration = 0
    local_search_interval = 20

    while time.time() - start_time < args.time_limit:
        iteration += 1

        rem_op = random.choices(removal_ops, weights=[removal_weights[op] for op in removal_ops])[0]
        q = random.randint(q_min, q_max)

        candidate = current.copy()
        routes = candidate.routes

        if rem_op == 'random':
            removed = random_removal(routes, q)
        elif rem_op == 'worst':
            removed = worst_removal(routes, q)
        elif rem_op == 'related':
            removed = related_removal(routes, q)
        else:
            removed = cluster_removal(routes, q)

        if not removed:
            continue

        for cust in removed:
            for r in routes:
                if cust in r.nodes:
                    remove_customer_from_route(r, cust)
                    break

        ins_op = random.choices(insertion_ops, weights=[insertion_weights[op] for op in insertion_ops])[0]
        if ins_op == 'greedy':
            random.shuffle(removed)
            success = True
            for cust in removed:
                if not insert_greedy(routes, cust):
                    success = False
                    break
        else:
            success = insert_regret(routes, removed)

        if not success:
            continue

        candidate.cost = solution_cost(routes)

        if iteration % local_search_interval == 0 and time.time() - start_time < args.time_limit:
            local_search(routes, start_time=start_time, time_limit=args.time_limit)
            candidate.cost = solution_cost(routes)

        if candidate.cost < best_cost:
            best = candidate.copy()
            best_cost = best.cost
            if logger:
                logger.log_solution(best_cost, solution_to_dict(best))
            removal_scores[rem_op] += 33
            insertion_scores[ins_op] += 33
        elif candidate.cost < current.cost:
            current = candidate
            removal_scores[rem_op] += 9
            insertion_scores[ins_op] += 9
        else:
            if current.cost > 0:
                delta = candidate.cost - current.cost
                prob = math.exp(-delta / max(temp, 1e-6))
            else:
                prob = 1.0
            if random.random() < prob:
                current = candidate
                removal_scores[rem_op] += 13
                insertion_scores[ins_op] += 13

        removal_usage[rem_op] += 1
        insertion_usage[ins_op] += 1

        if iteration % segment_size == 0:
            for op in removal_ops:
                if removal_usage[op] > 0:
                    avg = removal_scores[op] / removal_usage[op]
                    removal_weights[op] = removal_weights[op] * (1 - rho) + rho * avg
                    removal_weights[op] = max(0.1, removal_weights[op])
                removal_scores[op] = 0.0
                removal_usage[op] = 0
            for op in insertion_ops:
                if insertion_usage[op] > 0:
                    avg = insertion_scores[op] / insertion_usage[op]
                    insertion_weights[op] = insertion_weights[op] * (1 - rho) + rho * avg
                    insertion_weights[op] = max(0.1, insertion_weights[op])
                insertion_scores[op] = 0.0
                insertion_usage[op] = 0

        temp = max(min_temp, temp * temp_factor)

    output = {
        "objective_value": float(best.cost),
        "routes": [r.nodes for r in best.routes]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()