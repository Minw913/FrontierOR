import argparse
import json
import math
import random
import time
from typing import Dict, List, Tuple, Optional

from solution_logger import SolutionLogger


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


class DARPHeuristicSolver:
    def __init__(self, instance: dict, time_limit: int, logger: Optional[SolutionLogger] = None):
        self.inst = instance
        self.time_limit = int(time_limit)
        self.logger = logger

        self.K = int(instance["num_vehicles"])
        self.C = int(instance["vehicle_capacity"])
        self.max_ride = float(instance["maximum_ride_time"])
        self.max_route_duration = float(instance["maximum_route_duration"])

        self.nodes = {int(n["node_id"]): n for n in instance["nodes"]}
        self.node_ids = sorted(self.nodes.keys())

        self.origin = next(i for i in self.node_ids if self.nodes[i]["node_type"] == "origin_depot")
        self.destination = next(i for i in self.node_ids if self.nodes[i]["node_type"] == "destination_depot")
        self.pickups = [i for i in self.node_ids if self.nodes[i]["node_type"] == "pickup"]
        self.dropoffs = [i for i in self.node_ids if self.nodes[i]["node_type"] == "dropoff"]

        self.user_pickup = {}
        self.user_dropoff = {}
        for p in self.pickups:
            u = int(self.nodes[p]["user_id"])
            self.user_pickup[u] = p
        for d in self.dropoffs:
            u = int(self.nodes[d]["user_id"])
            self.user_dropoff[u] = d
        self.users = sorted(self.user_pickup.keys())

        self.E = {i: float(self.nodes[i]["earliest_time"]) for i in self.node_ids}
        self.L = {i: float(self.nodes[i]["latest_time"]) for i in self.node_ids}
        self.S = {i: float(self.nodes[i]["service_duration"]) for i in self.node_ids}
        self.load = {i: int(self.nodes[i]["load"]) for i in self.node_ids}

        self.dist = {}
        for i in self.node_ids:
            for j in self.node_ids:
                if i != j:
                    self.dist[(i, j)] = euclidean(self.nodes[i], self.nodes[j])

    def route_cost(self, route: List[int]) -> float:
        c = 0.0
        for i in range(len(route) - 1):
            c += self.dist[(route[i], route[i + 1])]
        return c

    def evaluate_route(self, route: List[int]):
        """
        Checks feasibility of a single route and returns:
        (feasible, cost, service_times, ride_times_on_route)
        service_times includes origin and destination nodes.
        """
        if not route or route[0] != self.origin or route[-1] != self.destination:
            return False, None, None, None

        # Start at earliest feasible depot start
        t0 = self.E[self.origin]
        if t0 > self.L[self.origin] + 1e-9:
            return False, None, None, None

        service_times = {self.origin: t0}
        t = t0
        q = 0

        pickup_start = {}
        ride_times = {}

        seen_pickups = set()
        seen_dropoffs = set()

        for idx in range(1, len(route)):
            i = route[idx - 1]
            j = route[idx]

            arrival = t + self.S[i] + self.dist[(i, j)]
            t = max(arrival, self.E[j])
            if t > self.L[j] + 1e-9:
                return False, None, None, None

            service_times[j] = t

            # update load after visiting node j
            q += self.load[j]
            if q < -1e-9 or q > self.C + 1e-9:
                return False, None, None, None

            ntype = self.nodes[j]["node_type"]
            if ntype == "pickup":
                u = int(self.nodes[j]["user_id"])
                pickup_start[u] = t
                seen_pickups.add(u)
            elif ntype == "dropoff":
                u = int(self.nodes[j]["user_id"])
                if u not in pickup_start:
                    return False, None, None, None  # drop before pickup
                rt = t - (pickup_start[u] + self.S[self.user_pickup[u]])
                direct = self.dist[(self.user_pickup[u], self.user_dropoff[u])]
                if rt + 1e-9 < direct:
                    return False, None, None, None
                if rt > self.max_ride + 1e-9:
                    return False, None, None, None
                ride_times[u] = rt
                seen_dropoffs.add(u)

        # Route duration
        duration = service_times[self.destination] - service_times[self.origin]
        if duration > self.max_route_duration + 1e-9:
            return False, None, None, None

        # every seen pickup must have seen dropoff and vice versa
        if seen_pickups != seen_dropoffs:
            return False, None, None, None

        return True, self.route_cost(route), service_times, ride_times

    def evaluate_solution(self, routes: Dict[int, List[int]]):
        all_pickup_served = set()
        all_drop_served = set()

        total_cost = 0.0
        per_vehicle_times = {}
        all_ride = {}

        for k in range(self.K):
            route = routes[k]
            feasible, rcost, stimes, rides = self.evaluate_route(route)
            if not feasible:
                return False, None, None, None
            total_cost += rcost
            per_vehicle_times[k] = stimes

            # collect served nodes
            for nid in route:
                ntype = self.nodes[nid]["node_type"]
                if ntype == "pickup":
                    all_pickup_served.add(nid)
                elif ntype == "dropoff":
                    all_drop_served.add(nid)

            for u, rt in rides.items():
                if u in all_ride:
                    return False, None, None, None
                all_ride[u] = rt

        if set(self.pickups) != all_pickup_served:
            return False, None, None, None
        if set(self.dropoffs) != all_drop_served:
            return False, None, None, None
        if set(self.users) != set(all_ride.keys()):
            return False, None, None, None

        return True, total_cost, per_vehicle_times, all_ride

    def construct_greedy(self, order: List[int], deadline: float, randomize: bool = False):
        routes = {k: [self.origin, self.destination] for k in range(self.K)}
        route_costs = {k: self.route_cost(routes[k]) for k in range(self.K)}

        for u in order:
            if time.time() >= deadline:
                return None

            p = self.user_pickup[u]
            d = self.user_dropoff[u]

            candidates = []
            for k in range(self.K):
                r = routes[k]
                base_cost = route_costs[k]
                m = len(r)

                # insert pickup before index i, dropoff before index j (after pickup)
                for i in range(1, m):
                    for j in range(i + 1, m + 1):
                        nr = r[:]
                        nr.insert(i, p)
                        nr.insert(j, d)

                        feas, ncost, _, _ = self.evaluate_route(nr)
                        if feas:
                            delta = ncost - base_cost
                            candidates.append((delta, k, nr, ncost))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[0])
            if randomize and len(candidates) > 1:
                # pick among top few for diversification
                top = min(5, len(candidates))
                pick_idx = random.randint(0, top - 1)
                chosen = candidates[pick_idx]
            else:
                chosen = candidates[0]

            _, best_k, best_route, best_ncost = chosen
            routes[best_k] = best_route
            route_costs[best_k] = best_ncost

        return routes

    def solve(self):
        start = time.time()
        deadline = start + self.time_limit - 0.02

        # base order: tighter requests first
        def tightness(u):
            p = self.user_pickup[u]
            d = self.user_dropoff[u]
            return (self.L[d] - self.E[p], self.L[p], self.L[d])

        base_order = sorted(self.users, key=tightness)

        best_obj = float("inf")
        best_routes = None
        best_stimes = None
        best_rides = None

        iteration = 0
        while time.time() < deadline:
            iteration += 1

            if iteration == 1:
                order = base_order[:]
                randomize = False
            else:
                order = base_order[:]
                random.shuffle(order)
                randomize = True

            routes = self.construct_greedy(order, deadline, randomize=randomize)
            if routes is None:
                continue

            feasible, obj, stimes_by_k, rides = self.evaluate_solution(routes)
            if not feasible:
                continue

            if obj + 1e-9 < best_obj:
                best_obj = obj
                best_routes = routes
                best_stimes = stimes_by_k
                best_rides = rides
                if self.logger:
                    self.logger.log(float(best_obj))

            # if very little time left, stop
            if time.time() > deadline:
                break

        # Fallback if no feasible found
        if best_routes is None:
            routes = {str(k): [self.origin, self.destination] for k in range(self.K)}
            service_times = {}
            for k in range(self.K):
                service_times[f"depot_start_{k}"] = float(self.E[self.origin])
                service_times[f"depot_end_{k}"] = float(self.E[self.origin])
            return {
                "objective_value": 1e18,
                "routes": routes,
                "service_times": service_times,
                "ride_times": {},
            }

        # Build output
        out_routes = {str(k): best_routes[k] for k in range(self.K)}

        service_times = {}
        for k in range(self.K):
            st = best_stimes[k]
            service_times[f"depot_start_{k}"] = float(st[self.origin])
            service_times[f"depot_end_{k}"] = float(st[self.destination])

        # node-level service times for pickups/dropoffs
        # each such node appears in exactly one vehicle
        for k in range(self.K):
            for nid, t in best_stimes[k].items():
                ntype = self.nodes[nid]["node_type"]
                if ntype in ("pickup", "dropoff"):
                    service_times[str(nid)] = float(t)

        ride_times = {str(u): float(best_rides[u]) for u in self.users}

        return {
            "objective_value": float(best_obj),
            "routes": out_routes,
            "service_times": service_times,
            "ride_times": ride_times,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = DARPHeuristicSolver(instance, args.time_limit, logger)
    solution = solver.solve()

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()