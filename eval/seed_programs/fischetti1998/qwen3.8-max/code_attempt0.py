#!/usr/bin/env python3
import json
import argparse
import time
import sys
import random
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    start_time = time.time()
    time_limit = args.time_limit

    # Parse instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    depot = data['depot']
    t0 = data['t0']

    prizes = {}
    for k, v in data['prizes'].items():
        prizes[int(k)] = v

    # Travel time matrix (1-indexed)
    travel = [[0] * (n + 1) for _ in range(n + 1)]
    for key, val in data['travel_times'].items():
        key_clean = key.strip('()')
        parts = key_clean.split(',')
        i, j = int(parts[0].strip()), int(parts[1].strip())
        travel[i][j] = val
        travel[j][i] = val

    non_depot = [c for c in range(1, n + 1) if c != depot]

    def get_cost(tour):
        cost = 0
        for i in range(len(tour) - 1):
            cost += travel[tour[i]][tour[i + 1]]
        return cost

    def get_prize(tour):
        return sum(prizes.get(c, 0) for c in set(tour))

    def make_solution(tour):
        visited = sorted(set(tour))
        edges = []
        for i in range(len(tour) - 1):
            edges.append([tour[i], tour[i + 1]])
        obj = get_prize(tour)
        return {
            "objective_value": float(obj),
            "visited_nodes": visited,
            "edges": edges,
            "tour": tour
        }

    # ===================== HEURISTIC METHODS =====================

    def greedy_construct(strategy='ratio'):
        """Greedy construction heuristic."""
        tour = [depot, depot]
        current_cost = 0
        visited_set = {depot}

        while True:
            best_city = -1
            best_pos = -1
            best_score = -1
            best_delta = float('inf')

            for c in non_depot:
                if c in visited_set:
                    continue
                p = prizes.get(c, 0)
                if p == 0 and strategy == 'ratio':
                    continue

                for pos in range(len(tour) - 1):
                    delta = travel[tour[pos]][c] + travel[c][tour[pos + 1]] - travel[tour[pos]][tour[pos + 1]]
                    if current_cost + delta <= t0:
                        if strategy == 'ratio':
                            score = p / max(delta, 0.001)
                        elif strategy == 'prize':
                            score = p
                        else:  # nearest
                            score = -delta
                        if score > best_score or (score == best_score and delta < best_delta):
                            best_score = score
                            best_city = c
                            best_pos = pos
                            best_delta = delta

            if best_city == -1:
                break

            tour.insert(best_pos + 1, best_city)
            current_cost += best_delta
            visited_set.add(best_city)

        return tour

    def two_opt(tour):
        """2-opt local search to reduce tour cost."""
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    delta = (travel[tour[i - 1]][tour[j]] + travel[tour[i]][tour[j + 1]]
                             - travel[tour[i - 1]][tour[i]] - travel[tour[j]][tour[j + 1]])
                    if delta < -1e-10:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True
        return tour

    def try_insert_all(tour):
        """Try inserting all unvisited cities at best position."""
        visited_set = set(tour)
        current_cost = get_cost(tour)
        improved = True
        while improved:
            improved = False
            best_city = -1
            best_pos = -1
            best_score = -1

            for c in non_depot:
                if c in visited_set:
                    continue
                p = prizes.get(c, 0)
                if p == 0:
                    continue
                for pos in range(1, len(tour)):
                    delta = travel[tour[pos - 1]][c] + travel[c][tour[pos]] - travel[tour[pos - 1]][tour[pos]]
                    if current_cost + delta <= t0:
                        score = p / max(delta, 0.001)
                        if score > best_score:
                            best_score = score
                            best_city = c
                            best_pos = pos

            if best_city != -1:
                delta = travel[tour[best_pos - 1]][best_city] + travel[best_city][tour[best_pos]] - travel[tour[best_pos - 1]][tour[best_pos]]
                tour.insert(best_pos, best_city)
                visited_set.add(best_city)
                current_cost += delta
                improved = True

        return tour

    def or_opt(tour):
        """Or-opt: relocate segments of 1, 2, or 3 cities."""
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(1, len(tour) - seg_len):
                    if i + seg_len >= len(tour) - 1:
                        continue
                    # Cost of removing segment tour[i:i+seg_len]
                    cost_remove = (travel[tour[i - 1]][tour[i + seg_len]]
                                   - travel[tour[i - 1]][tour[i]]
                                   - travel[tour[i + seg_len - 1]][tour[i + seg_len]])
                    for s in range(seg_len - 1):
                        cost_remove -= travel[tour[i + s]][tour[i + s + 1]]

                    segment = tour[i:i + seg_len]
                    remaining = tour[:i] + tour[i + seg_len:]

                    for j in range(1, len(remaining)):
                        cost_insert = (travel[remaining[j - 1]][segment[0]]
                                       + travel[segment[-1]][remaining[j]]
                                       - travel[remaining[j - 1]][remaining[j]])
                        for s in range(seg_len - 1):
                            cost_insert += travel[segment[s]][segment[s + 1]]

                        if cost_remove + cost_insert < -1e-10:
                            new_tour = remaining[:j] + segment + remaining[j:]
                            if get_cost(new_tour) <= t0:
                                tour = new_tour
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return tour

    def local_search(tour):
        """Full local search procedure."""
        if len(tour) < 4:
            return tour
        tour = two_opt(tour)
        tour = try_insert_all(tour)
        tour = two_opt(tour)
        tour = or_opt(tour)
        tour = try_insert_all(tour)
        return tour

    def remove_reinsert(tour):
        """Try removing each visited city and reinserting a better one."""
        best_tour = tour[:]
        best_prize = get_prize(tour)

        for i in range(1, len(tour) - 1):
            if len(tour) <= 4:
                break
            new_tour = tour[:i] + tour[i + 1:]
            new_cost = get_cost(new_tour)
            new_visited = set(new_tour)

            for c in non_depot:
                if c in new_visited:
                    continue
                p = prizes.get(c, 0)
                for pos in range(1, len(new_tour)):
                    delta = travel[new_tour[pos - 1]][c] + travel[c][new_tour[pos]] - travel[new_tour[pos - 1]][new_tour[pos]]
                    if new_cost + delta <= t0:
                        candidate = new_tour[:pos] + [c] + new_tour[pos:]
                        cp = get_prize(candidate)
                        if cp > best_prize:
                            best_prize = cp
                            best_tour = candidate[:]
                            break

        return best_tour

    # ===================== INITIAL SOLUTIONS =====================

    best_tour = None
    best_prize_val = -1

    # Try multiple greedy strategies
    for strategy in ['ratio', 'prize', 'nearest']:
        tour = greedy_construct(strategy)
        if len(tour) >= 4:
            tour = local_search(tour)
            if get_cost(tour) <= t0:
                p = get_prize(tour)
                if p > best_prize_val:
                    best_prize_val = p
                    best_tour = tour[:]
                    sol = make_solution(best_tour)
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)

    # If no solution found yet, try all pairs
    if best_tour is None or len(best_tour) < 4:
        for c1 in non_depot:
            for c2 in non_depot:
                if c1 >= c2:
                    continue
                cost = travel[depot][c1] + travel[c1][c2] + travel[c2][depot]
                if cost <= t0:
                    candidate = [depot, c1, c2, depot]
                    p = get_prize(candidate)
                    if p > best_prize_val:
                        best_prize_val = p
                        best_tour = candidate[:]

        if best_tour is not None:
            best_tour = local_search(best_tour)
            if get_cost(best_tour) <= t0:
                p = get_prize(best_tour)
                if p > best_prize_val:
                    best_prize_val = p
                sol = make_solution(best_tour)
                if logger:
                    logger.log_solution(sol["objective_value"], sol)

    # ===================== ILS PHASE =====================

    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed

    if remaining_time > 2 and best_tour is not None and len(best_tour) >= 4:
        random.seed(42)
        ils_tour = best_tour[:]
        ils_prize = best_prize_val
        no_improve = 0

        while time.time() - start_time < time_limit - max(2, time_limit * 0.3):
            # Perturbation
            perturbed = ils_tour[:]
            num_remove = max(1, min(3, len(perturbed) // 4))
            for _ in range(num_remove):
                if len(perturbed) <= 4:
                    break
                idx = random.randint(1, len(perturbed) - 2)
                perturbed.pop(idx)

            # Re-optimize
            perturbed = try_insert_all(perturbed)
            perturbed = two_opt(perturbed)
            perturbed = or_opt(perturbed)
            perturbed = try_insert_all(perturbed)

            if get_cost(perturbed) <= t0 and len(perturbed) >= 4:
                p = get_prize(perturbed)
                if p > ils_prize:
                    ils_prize = p
                    ils_tour = perturbed[:]
                    no_improve = 0
                    if p > best_prize_val:
                        best_prize_val = p
                        best_tour = perturbed[:]
                        sol = make_solution(best_tour)
                        if logger:
                            logger.log_solution(sol["objective_value"], sol)
                else:
                    no_improve += 1
                    # Accept with some probability for diversification
                    if p >= ils_prize - prizes.get(ils_tour[1], 0) * 0.5:
                        ils_tour = perturbed[:]
                        ils_prize = p
            else:
                no_improve += 1

            if no_improve > 50:
                # Restart from best
                ils_tour = best_tour[:]
                ils_prize = best_prize_val
                no_improve = 0

    # ===================== MIP PHASE =====================

    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed

    if n <= 80 and remaining_time > 5:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            mip_time = max(5, remaining_time - 2)

            model = gp.Model("OP")
            model.setParam('OutputFlag', 0)
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', mip_time)

            # Directed arc variables
            x = {}
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i != j:
                        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

            y = {}
            for i in range(1, n + 1):
                y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

            u = {}
            for i in range(1, n + 1):
                if i != depot:
                    u[i] = model.addVar(lb=2, ub=n, vtype=GRB.CONTINUOUS, name=f"u_{i}")

            # Objective: maximize prizes
            model.setObjective(
                gp.quicksum(prizes.get(i, 0) * y[i] for i in range(1, n + 1)),
                GRB.MAXIMIZE
            )

            # Depot is always visited
            model.addConstr(y[depot] == 1)

            # Degree constraints
            for i in range(1, n + 1):
                if i == depot:
                    model.addConstr(
                        gp.quicksum(x[depot, j] for j in range(1, n + 1) if j != depot) == 1,
                        name="depot_out"
                    )
                    model.addConstr(
                        gp.quicksum(x[j, depot] for j in range(1, n + 1) if j != depot) == 1,
                        name="depot_in"
                    )
                else:
                    model.addConstr(
                        gp.quicksum(x[j, i] for j in range(1, n + 1) if j != i) == y[i],
                        name=f"in_{i}"
                    )
                    model.addConstr(
                        gp.quicksum(x[i, j] for j in range(1, n + 1) if j != i) == y[i],
                        name=f"out_{i}"
                    )

            # Time budget
            model.addConstr(
                gp.quicksum(travel[i][j] * x[i, j]
                            for i in range(1, n + 1)
                            for j in range(1, n + 1) if i != j) <= t0,
                name="budget"
            )

            # MTZ subtour elimination
            for i in range(1, n + 1):
                if i == depot:
                    continue
                for j in range(1, n + 1):
                    if j == depot or j == i:
                        continue
                    model.addConstr(
                        u[i] - u[j] + 1 <= (n - 1) * (1 - x[i, j]),
                        name=f"mtz_{i}_{j}"
                    )

            # At least 3 cities
            model.addConstr(
                gp.quicksum(y[i] for i in range(1, n + 1)) >= 3,
                name="min_cities"
            )

            # Warm start from heuristic
            if best_tour is not None and len(best_tour) >= 4:
                visited_set = set(best_tour)
                for i in range(1, n + 1):
                    y[i].Start = 1.0 if i in visited_set else 0.0

                for i in range(1, n + 1):
                    for j in range(1, n + 1):
                        if i != j:
                            in_tour = False
                            for k in range(len(best_tour) - 1):
                                if best_tour[k] == i and best_tour[k + 1] == j:
                                    in_tour = True
                                    break
                            x[i, j].Start = 1.0 if in_tour else 0.0

                # Set u values
                for idx, city in enumerate(best_tour):
                    if city != depot and city in u:
                        u[city].Start = float(idx + 1)

            model.optimize()

            if model.SolCount > 0:
                # Extract solution
                mip_visited = [i for i in range(1, n + 1) if y[i].X > 0.5]

                if len(mip_visited) >= 3:
                    # Build adjacency from directed arcs
                    adj = {}
                    for i in range(1, n + 1):
                        for j in range(1, n + 1):
                            if i != j and x[i, j].X > 0.5:
                                if i not in adj:
                                    adj[i] = []
                                adj[i].append(j)

                    # Trace tour from depot
                    mip_tour = [depot]
                    current = depot
                    visited_in_trace = {depot}

                    while True:
                        nexts = adj.get(current, [])
                        found_next = False
                        for nxt in nexts:
                            if nxt == depot and len(mip_tour) >= 3:
                                mip_tour.append(depot)
                                found_next = True
                                break
                            if nxt not in visited_in_trace:
                                mip_tour.append(nxt)
                                visited_in_trace.add(nxt)
                                current = nxt
                                found_next = True
                                break
                        if not found_next:
                            break

                    if mip_tour[-1] == depot and len(mip_tour) >= 4:
                        if get_cost(mip_tour) <= t0 + 0.01:
                            mip_prize = get_prize(mip_tour)
                            if mip_prize > best_prize_val:
                                best_prize_val = mip_prize
                                best_tour = mip_tour[:]
                                sol = make_solution(best_tour)
                                if logger:
                                    logger.log_solution(sol["objective_value"], sol)

        except Exception as e:
            pass  # Fall back to heuristic solution

    # ===================== FINAL OUTPUT =====================

    if best_tour is None or len(best_tour) < 4:
        # Last resort fallback
        best_tour = [depot, depot, depot]
        best_prize_val = 0
        for c1 in non_depot:
            for c2 in non_depot:
                if c1 >= c2:
                    continue
                cost = travel[depot][c1] + travel[c1][c2] + travel[c2][depot]
                if cost <= t0:
                    best_tour = [depot, c1, c2, depot]
                    best_prize_val = get_prize(best_tour)
                    break
            if len(best_tour) >= 4:
                break

    sol = make_solution(best_tour)

    if logger:
        logger.log_solution(sol["objective_value"], sol)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()