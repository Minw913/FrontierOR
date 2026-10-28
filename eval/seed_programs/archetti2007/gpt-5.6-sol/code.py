import argparse
import itertools
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-8


def normalized_number(value):
    value = float(value)
    nearest = round(value)
    if abs(value - nearest) <= 1e-7:
        return int(nearest)
    return round(value, 10)


class InventoryRoutingSolver:
    def __init__(self, instance, time_limit, logger):
        self.instance = instance
        self.n = int(instance["num_retailers"])
        self.H = int(instance["time_horizon"])
        self.capacity = float(instance["vehicle_capacity"])
        self.supplier = instance["supplier"]
        self.retailers = instance["retailers"]
        self.dist = instance["distance_matrix"]
        self.logger = logger

        self.supplier_id = self.supplier["id"]
        self.retailer_ids = [retailer["id"] for retailer in self.retailers]

        self.initial_supplier_inventory = float(
            self.supplier["starting_inventory"]
        )
        self.production = float(self.supplier["production_rate"])
        self.supplier_holding_cost = float(
            self.supplier["inventory_cost"]
        )

        self.initial_inventory = tuple(
            float(retailer["starting_inventory"])
            for retailer in self.retailers
        )
        self.demand = tuple(
            float(retailer["demand"]) for retailer in self.retailers
        )
        self.maximum = tuple(
            float(retailer["max_inventory"])
            for retailer in self.retailers
        )
        self.retailer_holding_cost = tuple(
            float(retailer["inventory_cost"])
            for retailer in self.retailers
        )

        self.start_time = time.monotonic()
        self.deadline = self.start_time + max(0.01, float(time_limit))
        self.random = random.Random(0)

        self.route_cache = {}
        self.max_distance = max(
            (float(v) for row in self.dist for v in row),
            default=1.0,
        )
        self.penalty_scale = max(
            1.0,
            self.max_distance * max(2, self.n),
            sum(self.maximum) * max(
                [self.supplier_holding_cost]
                + list(self.retailer_holding_cost)
                + [1.0]
            ),
        )

        self.best_solution = None
        self.best_objective = math.inf

    def out_of_time(self):
        return time.monotonic() >= self.deadline

    def tour_cost(self, order):
        if not order:
            return 0.0
        total = float(self.dist[0][order[0] + 1])
        for first, second in zip(order, order[1:]):
            total += float(self.dist[first + 1][second + 1])
        total += float(self.dist[order[-1] + 1][0])
        return total

    def exact_tour(self, nodes):
        """Held-Karp solution for a small retailer subset."""
        k = len(nodes)
        if k == 0:
            return 0.0, []
        if k == 1:
            node = nodes[0]
            return 2.0 * float(self.dist[0][node + 1]), [node]

        # dp[(mask, last_position)] = (cost, previous_position)
        dp = {}
        for j, node in enumerate(nodes):
            dp[(1 << j, j)] = (
                float(self.dist[0][node + 1]),
                -1,
            )

        for size in range(2, k + 1):
            for combination in itertools.combinations(range(k), size):
                mask = 0
                for index in combination:
                    mask |= 1 << index

                for last in combination:
                    previous_mask = mask ^ (1 << last)
                    best_cost = math.inf
                    best_previous = -1

                    for previous in combination:
                        if previous == last:
                            continue
                        previous_entry = dp.get((previous_mask, previous))
                        if previous_entry is None:
                            continue
                        cost = (
                            previous_entry[0]
                            + float(
                                self.dist[
                                    nodes[previous] + 1
                                ][nodes[last] + 1]
                            )
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_previous = previous

                    dp[(mask, last)] = (best_cost, best_previous)

        full_mask = (1 << k) - 1
        best_last = min(
            range(k),
            key=lambda j: (
                dp[(full_mask, j)][0]
                + float(self.dist[nodes[j] + 1][0])
            ),
        )
        best_cost = (
            dp[(full_mask, best_last)][0]
            + float(self.dist[nodes[best_last] + 1][0])
        )

        reversed_positions = []
        mask = full_mask
        last = best_last
        while last >= 0:
            reversed_positions.append(last)
            previous = dp[(mask, last)][1]
            mask ^= 1 << last
            last = previous

        order = [nodes[position] for position in reversed(reversed_positions)]
        return best_cost, order

    def heuristic_tour(self, nodes):
        if not nodes:
            return 0.0, []
        if len(nodes) == 1:
            node = nodes[0]
            return 2.0 * float(self.dist[0][node + 1]), [node]

        unvisited = set(nodes)
        order = []
        current_matrix_index = 0

        while unvisited:
            nxt = min(
                unvisited,
                key=lambda node: (
                    float(self.dist[current_matrix_index][node + 1]),
                    node,
                ),
            )
            order.append(nxt)
            unvisited.remove(nxt)
            current_matrix_index = nxt + 1

        # Improve the nearest-neighbor route by 2-opt.
        improved = True
        passes = 0
        while improved and passes < 8 and not self.out_of_time():
            improved = False
            passes += 1

            for i in range(len(order) - 1):
                before_i = 0 if i == 0 else order[i - 1] + 1
                first_i = order[i] + 1

                for j in range(i + 1, len(order)):
                    last_j = order[j] + 1
                    after_j = (
                        0 if j == len(order) - 1 else order[j + 1] + 1
                    )

                    old_cost = (
                        float(self.dist[before_i][first_i])
                        + float(self.dist[last_j][after_j])
                    )
                    new_cost = (
                        float(self.dist[before_i][last_j])
                        + float(self.dist[first_i][after_j])
                    )

                    if new_cost + EPS < old_cost:
                        order[i:j + 1] = reversed(order[i:j + 1])
                        improved = True

        return self.tour_cost(order), order

    def route_for_subset(self, subset):
        key = tuple(sorted(subset))
        cached = self.route_cache.get(key)
        if cached is not None:
            return cached

        if len(key) <= 11 and not self.out_of_time():
            answer = self.exact_tour(list(key))
        else:
            answer = self.heuristic_tour(list(key))

        self.route_cache[key] = answer
        return answer

    def candidate_subsets(self, inventory):
        mandatory = {
            i for i in range(self.n)
            if inventory[i] + EPS < self.demand[i]
        }

        mandatory_load = sum(
            self.maximum[i] - inventory[i] for i in mandatory
        )
        if mandatory_load > self.capacity + EPS:
            return []

        optional = [
            i for i in range(self.n)
            if i not in mandatory
            and self.maximum[i] - inventory[i] > EPS
        ]

        candidates = {tuple(sorted(mandatory))}
        residual = self.capacity - mandatory_load

        def add_prefixes(order):
            chosen = set(mandatory)
            used = mandatory_load
            for i in order:
                quantity = self.maximum[i] - inventory[i]
                if used + quantity <= self.capacity + EPS:
                    chosen.add(i)
                    used += quantity
                    candidates.add(tuple(sorted(chosen)))

        # Several deterministic orderings expose different inventory/routing
        # tradeoffs to the beam search.
        urgency_order = sorted(
            optional,
            key=lambda i: (
                inventory[i] / self.demand[i]
                if self.demand[i] > EPS else math.inf,
                self.maximum[i] - inventory[i],
                i,
            ),
        )
        quantity_order = sorted(
            optional,
            key=lambda i: (
                -(self.maximum[i] - inventory[i]),
                i,
            ),
        )
        retailer_cost_order = sorted(
            optional,
            key=lambda i: (
                self.retailer_holding_cost[i]
                - self.supplier_holding_cost,
                inventory[i] / self.demand[i]
                if self.demand[i] > EPS else math.inf,
                i,
            ),
        )
        depot_distance_order = sorted(
            optional,
            key=lambda i: (
                float(self.dist[0][i + 1]),
                inventory[i] / self.demand[i]
                if self.demand[i] > EPS else math.inf,
                i,
            ),
        )

        add_prefixes(urgency_order)
        add_prefixes(quantity_order)
        add_prefixes(retailer_cost_order)
        add_prefixes(depot_distance_order)

        # Single additions are particularly important when a full prefix does
        # not fit.
        for i in optional:
            quantity = self.maximum[i] - inventory[i]
            if quantity <= residual + EPS:
                candidates.add(tuple(sorted(mandatory | {i})))

        # Enumerate all optional choices only when this is inexpensive.
        if len(optional) <= 8:
            for mask in range(1 << len(optional)):
                chosen = set(mandatory)
                load = mandatory_load
                feasible = True
                for position, i in enumerate(optional):
                    if mask & (1 << position):
                        load += self.maximum[i] - inventory[i]
                        if load > self.capacity + EPS:
                            feasible = False
                            break
                        chosen.add(i)
                if feasible:
                    candidates.add(tuple(sorted(chosen)))
        else:
            # Deterministic randomized subsets increase diversity on larger
            # instances without exponential enumeration.
            trials = min(28, 4 + len(optional))
            for _ in range(trials):
                shuffled = list(optional)
                self.random.shuffle(shuffled)
                chosen = set(mandatory)
                load = mandatory_load
                for i in shuffled:
                    quantity = self.maximum[i] - inventory[i]
                    urgency = (
                        self.demand[i] / max(inventory[i], EPS)
                        if self.demand[i] > EPS else 0.0
                    )
                    probability = min(0.85, 0.18 + 0.25 * urgency)
                    if (
                        load + quantity <= self.capacity + EPS
                        and self.random.random() < probability
                    ):
                        chosen.add(i)
                        load += quantity
                candidates.add(tuple(sorted(chosen)))

        return list(candidates)

    def risk_penalty(self, inventory, remaining_periods):
        """
        Estimate imminent capacity collisions. This does not affect the
        reported objective; it is used only to rank beam states.
        """
        lookahead = min(4, remaining_periods)
        if lookahead <= 0:
            return 0.0

        projected = list(inventory)
        penalty = 0.0

        for offset in range(lookahead):
            required_load = 0.0
            for i in range(self.n):
                if projected[i] + EPS < self.demand[i]:
                    required_load += self.maximum[i] - projected[i]
                    projected[i] = self.maximum[i]
                projected[i] -= self.demand[i]

            overload = max(0.0, required_load - self.capacity)
            penalty += overload * self.penalty_scale * (lookahead - offset)

        return penalty

    def state_key(self, inventory, supplier_inventory):
        return (
            tuple(round(value, 7) for value in inventory),
            round(supplier_inventory, 7),
        )

    def initial_cost(self):
        cost = self.supplier_holding_cost * self.initial_supplier_inventory
        cost += sum(
            self.retailer_holding_cost[i] * self.initial_inventory[i]
            for i in range(self.n)
        )
        return cost

    def simulate_path(self, path):
        supplier_inventory = self.initial_supplier_inventory
        retailer_inventory = list(self.initial_inventory)

        supplier_history = {1: supplier_inventory}
        retailer_history = {
            i: {1: retailer_inventory[i]} for i in range(self.n)
        }
        deliveries = {}
        routes = {}
        objective = self.initial_cost()

        for t, visited_tuple in enumerate(path, start=1):
            visited = set(visited_tuple)
            period_deliveries = {}

            total_shipment = 0.0
            for i in visited:
                shipment = self.maximum[i] - retailer_inventory[i]
                if shipment < 0.0 and shipment > -EPS:
                    shipment = 0.0
                total_shipment += shipment
                period_deliveries[str(self.retailer_ids[i])] = (
                    normalized_number(shipment)
                )

            deliveries[str(t)] = period_deliveries

            _, order = self.route_for_subset(visited_tuple)
            route_edges = []
            if len(order) == 1:
                route_edges.append(
                    [self.supplier_id, self.retailer_ids[order[0]], 2]
                )
            elif len(order) >= 2:
                route_edges.append(
                    [self.supplier_id, self.retailer_ids[order[0]], 1]
                )
                for first, second in zip(order, order[1:]):
                    route_edges.append(
                        [self.retailer_ids[first],
                         self.retailer_ids[second], 1]
                    )
                route_edges.append(
                    [self.retailer_ids[order[-1]], self.supplier_id, 1]
                )
            routes[str(t)] = route_edges

            next_supplier = (
                supplier_inventory + self.production - total_shipment
            )
            next_retailer = []
            for i in range(self.n):
                shipment = (
                    self.maximum[i] - retailer_inventory[i]
                    if i in visited else 0.0
                )
                next_inventory = (
                    retailer_inventory[i] + shipment - self.demand[i]
                )
                next_retailer.append(next_inventory)
                retailer_history[i][t + 1] = next_inventory

            supplier_inventory = next_supplier
            retailer_inventory = next_retailer
            supplier_history[t + 1] = supplier_inventory

            objective += (
                self.supplier_holding_cost * supplier_inventory
            )
            objective += sum(
                self.retailer_holding_cost[i] * retailer_inventory[i]
                for i in range(self.n)
            )
            objective += self.route_for_subset(visited_tuple)[0]

        solution = {
            "objective_value": float(objective),
            "deliveries": deliveries,
            "routes": routes,
            "supplier_inventory": {
                str(t): normalized_number(supplier_history[t])
                for t in range(1, self.H + 2)
            },
            "retailer_inventory": {
                str(self.retailer_ids[i]): {
                    str(t): normalized_number(retailer_history[i][t])
                    for t in range(1, self.H + 2)
                }
                for i in range(self.n)
            },
        }
        return solution

    def record_solution(self, path):
        solution = self.simulate_path(path)
        objective = float(solution["objective_value"])

        if objective + 1e-7 < self.best_objective:
            self.best_objective = objective
            self.best_solution = solution
            if self.logger:
                self.logger.log_solution(objective, solution)

    def greedy_solution(self):
        """
        Quickly obtain an incumbent. Retailers that are mandatory now are
        visited, followed by retailers likely to become mandatory soon.
        """
        inventory = list(self.initial_inventory)
        supplier_inventory = self.initial_supplier_inventory
        path = []

        for t in range(1, self.H + 1):
            mandatory = {
                i for i in range(self.n)
                if inventory[i] + EPS < self.demand[i]
            }
            load = sum(
                self.maximum[i] - inventory[i] for i in mandatory
            )

            if (
                load > self.capacity + EPS
                or load > supplier_inventory + EPS
            ):
                return None

            optional = [
                i for i in range(self.n)
                if i not in mandatory
                and self.maximum[i] - inventory[i] > EPS
            ]
            optional.sort(
                key=lambda i: (
                    inventory[i] / self.demand[i]
                    if self.demand[i] > EPS else math.inf,
                    float(self.dist[0][i + 1]),
                    i,
                )
            )

            visited = set(mandatory)
            for i in optional:
                shipment = self.maximum[i] - inventory[i]
                becomes_mandatory_next = (
                    inventory[i] - self.demand[i]
                    < self.demand[i] - EPS
                )
                if (
                    becomes_mandatory_next
                    and load + shipment <= self.capacity + EPS
                    and load + shipment <= supplier_inventory + EPS
                ):
                    visited.add(i)
                    load += shipment

            path.append(tuple(sorted(visited)))

            next_inventory = []
            for i in range(self.n):
                shipment = (
                    self.maximum[i] - inventory[i]
                    if i in visited else 0.0
                )
                value = inventory[i] + shipment - self.demand[i]
                if value < -EPS:
                    return None
                next_inventory.append(max(0.0, value))

            supplier_inventory += self.production - load
            inventory = next_inventory

        return path

    def solve(self):
        greedy_path = self.greedy_solution()
        if greedy_path is not None:
            self.record_solution(greedy_path)

        if self.out_of_time():
            return self.best_solution

        # State: (true_cost, inventory_tuple, supplier_inventory, path)
        beam = [(
            self.initial_cost(),
            self.initial_inventory,
            self.initial_supplier_inventory,
            tuple(),
        )]

        if self.n <= 8:
            beam_width = 1800
        elif self.n <= 15:
            beam_width = 800
        elif self.n <= 30:
            beam_width = 350
        else:
            beam_width = 140

        # Keep runtime proportional to the supplied budget.
        remaining_time = max(0.0, self.deadline - time.monotonic())
        if remaining_time < 5.0:
            beam_width = min(beam_width, 100)
        elif remaining_time < 15.0:
            beam_width = min(beam_width, 300)

        for t in range(1, self.H + 1):
            if self.out_of_time():
                break

            successors = {}
            complete_paths = []

            for true_cost, inventory, supplier_inventory, path in beam:
                if self.out_of_time():
                    break

                subsets = self.candidate_subsets(inventory)
                for visited_tuple in subsets:
                    shipment = sum(
                        self.maximum[i] - inventory[i]
                        for i in visited_tuple
                    )

                    if (
                        shipment > self.capacity + EPS
                        or shipment > supplier_inventory + EPS
                    ):
                        continue

                    visited = set(visited_tuple)
                    next_inventory = []
                    feasible = True

                    for i in range(self.n):
                        delivered = (
                            self.maximum[i] - inventory[i]
                            if i in visited else 0.0
                        )
                        value = inventory[i] + delivered - self.demand[i]
                        if value < -EPS:
                            feasible = False
                            break
                        next_inventory.append(max(0.0, value))

                    if not feasible:
                        continue

                    next_supplier = (
                        supplier_inventory + self.production - shipment
                    )
                    if next_supplier < -EPS:
                        continue
                    next_supplier = max(0.0, next_supplier)

                    route_cost, _ = self.route_for_subset(visited_tuple)
                    next_cost = true_cost + route_cost
                    next_cost += (
                        self.supplier_holding_cost * next_supplier
                    )
                    next_cost += sum(
                        self.retailer_holding_cost[i] * next_inventory[i]
                        for i in range(self.n)
                    )

                    next_path = path + (visited_tuple,)
                    next_inventory_tuple = tuple(next_inventory)

                    key = self.state_key(
                        next_inventory_tuple, next_supplier
                    )
                    previous = successors.get(key)
                    if previous is None or next_cost < previous[0] - EPS:
                        successors[key] = (
                            next_cost,
                            next_inventory_tuple,
                            next_supplier,
                            next_path,
                        )

                    if t == self.H:
                        complete_paths.append((next_cost, next_path))

            if not successors:
                break

            if t == self.H:
                if complete_paths:
                    complete_paths.sort(key=lambda item: item[0])
                    self.record_solution(complete_paths[0][1])
                break

            remaining_periods = self.H - t
            ranked = list(successors.values())
            ranked.sort(
                key=lambda state: (
                    state[0]
                    + self.risk_penalty(
                        state[1], remaining_periods
                    ),
                    state[0],
                )
            )

            # Most states are chosen by risk-adjusted cost. A smaller portion
            # is retained by pure objective value to avoid excessive heuristic
            # bias.
            primary_count = max(1, int(beam_width * 0.82))
            selected = ranked[:primary_count]
            selected_keys = {
                self.state_key(state[1], state[2]) for state in selected
            }

            by_cost = sorted(successors.values(), key=lambda state: state[0])
            for state in by_cost:
                if len(selected) >= beam_width:
                    break
                key = self.state_key(state[1], state[2])
                if key not in selected_keys:
                    selected.append(state)
                    selected_keys.add(key)

            beam = selected

        return self.best_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    solver = InventoryRoutingSolver(
        instance=instance,
        time_limit=args.time_limit,
        logger=logger,
    )
    solution = solver.solve()

    if solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit. "
            "The instance may be infeasible."
        )

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2)


if __name__ == "__main__":
    main()