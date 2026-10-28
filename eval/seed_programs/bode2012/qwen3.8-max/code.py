import json
import argparse
import time
import heapq
import random
from collections import defaultdict
from solution_logger import SolutionLogger


class CARPSolver:
    def __init__(self, instance):
        self.instance = instance
        self.n = instance['graph']['num_nodes']
        self.depot = instance['depot']
        self.num_vehicles = instance['fleet']['num_vehicles']
        self.capacity = instance['fleet']['vehicle_capacity']
        self.edges = instance['edges']

        # Build adjacency list and edge lookup
        self.adj = [[] for _ in range(self.n)]
        self.edge_map = {}
        self.edge_between = defaultdict(list)

        for e in self.edges:
            u, v = e['endpoints']
            self.adj[u].append((v, e['cost']))
            self.adj[v].append((u, e['cost']))
            self.edge_map[e['edge_id']] = e
            key = (min(u, v), max(u, v))
            self.edge_between[key].append(e['edge_id'])

        # Required edges
        self.required_edges = [e for e in self.edges if e['is_required']]
        self.num_required = len(self.required_edges)
        self.total_service_cost = sum(e['service_cost'] for e in self.required_edges)

        # Compute all-pairs shortest paths
        self.dist = []
        for s in range(self.n):
            self.dist.append(self._dijkstra(s))

    def _dijkstra(self, source):
        dist = [float('inf')] * self.n
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        return dist

    def route_deadheading_cost(self, route):
        """Compute minimum deadheading cost for a route (list of indices into required_edges)."""
        if not route:
            return 0

        e0 = self.required_edges[route[0]]
        u0, v0 = e0['endpoints']
        # dp0: end at u0 (approached from v0), dp1: end at v0 (approached from u0)
        dp0 = self.dist[self.depot][v0]
        dp1 = self.dist[self.depot][u0]

        for i in range(1, len(route)):
            e_prev = self.required_edges[route[i - 1]]
            e_cur = self.required_edges[route[i]]
            u_p, v_p = e_prev['endpoints']
            u_c, v_c = e_cur['endpoints']

            new_dp0 = min(dp0 + self.dist[u_p][v_c], dp1 + self.dist[v_p][v_c])
            new_dp1 = min(dp0 + self.dist[u_p][u_c], dp1 + self.dist[v_p][u_c])
            dp0, dp1 = new_dp0, new_dp1

        e_last = self.required_edges[route[-1]]
        u_l, v_l = e_last['endpoints']
        return min(dp0 + self.dist[u_l][self.depot], dp1 + self.dist[v_l][self.depot])

    def path_scanning(self, criterion='nearest'):
        """Construct routes using path scanning heuristic."""
        unserviced = set(range(self.num_required))
        routes = [[] for _ in range(self.num_vehicles)]
        demands = [0] * self.num_vehicles

        for k in range(self.num_vehicles):
            if not unserviced:
                break
            current_node = self.depot

            while unserviced:
                best_idx = None
                best_val = None

                for idx in unserviced:
                    e = self.required_edges[idx]
                    if demands[k] + e['demand'] > self.capacity:
                        continue
                    u, v = e['endpoints']
                    cost_to_reach = min(self.dist[current_node][u], self.dist[current_node][v])

                    if criterion == 'nearest':
                        val = cost_to_reach
                    elif criterion == 'farthest':
                        val = -cost_to_reach
                    elif criterion == 'demand':
                        val = -e['demand']
                    elif criterion == 'ratio':
                        val = cost_to_reach / max(e['demand'], 1)
                    else:
                        val = cost_to_reach

                    if best_val is None or val < best_val:
                        best_val = val
                        best_idx = idx

                if best_idx is None:
                    break

                routes[k].append(best_idx)
                demands[k] += self.required_edges[best_idx]['demand']

                u, v = self.required_edges[best_idx]['endpoints']
                if self.dist[current_node][u] <= self.dist[current_node][v]:
                    current_node = v
                else:
                    current_node = u

                unserviced.remove(best_idx)

        # Handle remaining edges
        for idx in list(unserviced):
            for k in range(self.num_vehicles):
                if demands[k] + self.required_edges[idx]['demand'] <= self.capacity:
                    routes[k].append(idx)
                    demands[k] += self.required_edges[idx]['demand']
                    unserviced.remove(idx)
                    break

        return routes

    def optimize_route_2opt(self, route):
        """Optimize edge ordering within a route using 2-opt."""
        if len(route) <= 2:
            return route

        best_route = route[:]
        best_cost = self.route_deadheading_cost(best_route)

        improved = True
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    new_cost = self.route_deadheading_cost(new_route)
                    if new_cost < best_cost - 1e-10:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def optimize_route_or_opt(self, route):
        """Or-opt: move segments of 1, 2, or 3 edges."""
        if len(route) <= 3:
            return route

        best_route = route[:]
        best_cost = self.route_deadheading_cost(best_route)

        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(len(best_route) - seg_len + 1):
                    segment = best_route[i:i + seg_len]
                    remaining = best_route[:i] + best_route[i + seg_len:]
                    for j in range(len(remaining) + 1):
                        new_route = remaining[:j] + segment + remaining[j:]
                        new_cost = self.route_deadheading_cost(new_route)
                        if new_cost < best_cost - 1e-10:
                            best_route = new_route
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return best_route

    def local_search_inter(self, routes, time_limit, start_time):
        """Inter-route local search: relocate and swap."""
        # Relocate
        improved = True
        max_iter = 200
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.8 and iteration < max_iter:
            improved = False
            iteration += 1

            demands = [sum(self.required_edges[i]['demand'] for i in routes[k])
                       for k in range(len(routes))]

            for k1 in range(len(routes)):
                if time.time() - start_time > time_limit * 0.8:
                    break
                for i in range(len(routes[k1])):
                    edge_idx = routes[k1][i]
                    edge = self.required_edges[edge_idx]

                    for k2 in range(len(routes)):
                        if k1 == k2:
                            continue
                        if demands[k2] + edge['demand'] > self.capacity:
                            continue

                        cost_k1 = self.route_deadheading_cost(routes[k1])
                        cost_k2 = self.route_deadheading_cost(routes[k2]) if routes[k2] else 0

                        new_route_k1 = routes[k1][:i] + routes[k1][i + 1:]
                        new_cost_k1 = self.route_deadheading_cost(new_route_k1)

                        best_insert_cost = float('inf')
                        best_pos = 0
                        for pos in range(len(routes[k2]) + 1):
                            new_route_k2 = routes[k2][:pos] + [edge_idx] + routes[k2][pos:]
                            c = self.route_deadheading_cost(new_route_k2)
                            if c < best_insert_cost:
                                best_insert_cost = c
                                best_pos = pos

                        if new_cost_k1 + best_insert_cost < cost_k1 + cost_k2 - 1e-10:
                            routes[k1] = new_route_k1
                            routes[k2] = routes[k2][:best_pos] + [edge_idx] + routes[k2][best_pos:]
                            demands[k1] -= edge['demand']
                            demands[k2] += edge['demand']
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # Swap
        improved = True
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.85 and iteration < 100:
            improved = False
            iteration += 1

            demands = [sum(self.required_edges[i]['demand'] for i in routes[k])
                       for k in range(len(routes))]

            for k1 in range(len(routes)):
                if time.time() - start_time > time_limit * 0.85:
                    break
                for i in range(len(routes[k1])):
                    for k2 in range(k1 + 1, len(routes)):
                        for j in range(len(routes[k2])):
                            e1_idx = routes[k1][i]
                            e2_idx = routes[k2][j]
                            e1 = self.required_edges[e1_idx]
                            e2 = self.required_edges[e2_idx]

                            new_dk1 = demands[k1] - e1['demand'] + e2['demand']
                            new_dk2 = demands[k2] - e2['demand'] + e1['demand']
                            if new_dk1 > self.capacity or new_dk2 > self.capacity:
                                continue

                            cost_k1 = self.route_deadheading_cost(routes[k1])
                            cost_k2 = self.route_deadheading_cost(routes[k2])

                            new_route_k1 = routes[k1][:i] + [e2_idx] + routes[k1][i + 1:]
                            new_route_k2 = routes[k2][:j] + [e1_idx] + routes[k2][j + 1:]

                            new_cost_k1 = self.route_deadheading_cost(new_route_k1)
                            new_cost_k2 = self.route_deadheading_cost(new_route_k2)

                            if new_cost_k1 + new_cost_k2 < cost_k1 + cost_k2 - 1e-10:
                                routes[k1] = new_route_k1
                                routes[k2] = new_route_k2
                                demands[k1] = new_dk1
                                demands[k2] = new_dk2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return routes

    def _get_optimal_directions(self, route):
        """Get optimal traversal directions via DP backtracking."""
        if not route:
            return []

        n = len(route)
        if n == 1:
            e = self.required_edges[route[0]]
            u, v = e['endpoints']
            if self.dist[self.depot][v] <= self.dist[self.depot][u]:
                return [0]
            else:
                return [1]

        # Forward DP
        dp = [[0.0, 0.0] for _ in range(n)]
        e0 = self.required_edges[route[0]]
        u0, v0 = e0['endpoints']
        dp[0][0] = self.dist[self.depot][v0]
        dp[0][1] = self.dist[self.depot][u0]

        for i in range(1, n):
            e_prev = self.required_edges[route[i - 1]]
            e_cur = self.required_edges[route[i]]
            u_p, v_p = e_prev['endpoints']
            u_c, v_c = e_cur['endpoints']
            dp[i][0] = min(dp[i - 1][0] + self.dist[u_p][v_c], dp[i - 1][1] + self.dist[v_p][v_c])
            dp[i][1] = min(dp[i - 1][0] + self.dist[u_p][u_c], dp[i - 1][1] + self.dist[v_p][u_c])

        # Choose best final state
        e_last = self.required_edges[route[-1]]
        u_l, v_l = e_last['endpoints']
        total_0 = dp[n - 1][0] + self.dist[u_l][self.depot]
        total_1 = dp[n - 1][1] + self.dist[v_l][self.depot]

        directions = [0] * n
        directions[n - 1] = 0 if total_0 <= total_1 else 1

        # Backtrack
        for i in range(n - 2, -1, -1):
            next_dir = directions[i + 1]
            e_prev = self.required_edges[route[i]]
            e_next = self.required_edges[route[i + 1]]
            u_p, v_p = e_prev['endpoints']
            u_n, v_n = e_next['endpoints']

            if next_dir == 0:
                val_0 = dp[i][0] + self.dist[u_p][v_n]
                val_1 = dp[i][1] + self.dist[v_p][v_n]
            else:
                val_0 = dp[i][0] + self.dist[u_p][u_n]
                val_1 = dp[i][1] + self.dist[v_p][u_n]

            directions[i] = 0 if val_0 <= val_1 else 1

        return directions

    def _shortest_path(self, source, target):
        """Reconstruct shortest path from source to target."""
        if source == target:
            return [source]

        dist = [float('inf')] * self.n
        prev = [-1] * self.n
        dist[source] = 0
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == target:
                break
            for v, w in self.adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        if dist[target] == float('inf'):
            return [source]

        path = []
        node = target
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def _find_edge_id(self, u, v):
        """Find cheapest edge ID between nodes u and v."""
        key = (min(u, v), max(u, v))
        if key in self.edge_between:
            best_eid = None
            best_cost = float('inf')
            for eid in self.edge_between[key]:
                e = self.edge_map[eid]
                if e['cost'] < best_cost:
                    best_cost = e['cost']
                    best_eid = eid
            return best_eid
        return None

    def build_solution(self, routes):
        """Build the output solution dictionary."""
        solution_routes = []

        for k, route in enumerate(routes):
            if not route:
                solution_routes.append({
                    "vehicle": k,
                    "serviced_edges": [],
                    "deadheaded_edges": []
                })
                continue

            directions = self._get_optimal_directions(route)
            deadhead_count = defaultdict(int)
            current_node = self.depot

            for pos, idx in enumerate(route):
                e = self.required_edges[idx]
                u, v = e['endpoints']

                if directions[pos] == 0:
                    approach_node = v
                    end_node = u
                else:
                    approach_node = u
                    end_node = v

                sp = self._shortest_path(current_node, approach_node)
                for i in range(len(sp) - 1):
                    eid = self._find_edge_id(sp[i], sp[i + 1])
                    if eid is not None:
                        deadhead_count[eid] += 1

                current_node = end_node

            # Return to depot
            sp = self._shortest_path(current_node, self.depot)
            for i in range(len(sp) - 1):
                eid = self._find_edge_id(sp[i], sp[i + 1])
                if eid is not None:
                    deadhead_count[eid] += 1

            deadheaded_edges = [{"edge_id": eid, "times": cnt}
                                for eid, cnt in sorted(deadhead_count.items()) if cnt > 0]

            solution_routes.append({
                "vehicle": k,
                "serviced_edges": [self.required_edges[i]['edge_id'] for i in route],
                "deadheaded_edges": deadheaded_edges
            })

        total_deadheading = sum(self.route_deadheading_cost(r) for r in routes)
        objective = self.total_service_cost + total_deadheading

        return {
            "objective_value": float(objective),
            "routes": solution_routes
        }

    def compute_objective(self, routes):
        """Compute total objective value."""
        total_dh = sum(self.route_deadheading_cost(r) for r in routes)
        return self.total_service_cost + total_dh

    def _perturb_solution(self, routes):
        """Random perturbation: move a random edge to another route."""
        non_empty = [k for k in range(len(routes)) if routes[k]]
        if not non_empty:
            return None

        k1 = random.choice(non_empty)
        if not routes[k1]:
            return None

        i = random.randint(0, len(routes[k1]) - 1)
        edge_idx = routes[k1][i]
        edge = self.required_edges[edge_idx]

        demands = [sum(self.required_edges[idx]['demand'] for idx in routes[k])
                   for k in range(len(routes))]

        candidates = [k for k in range(len(routes))
                      if k != k1 and demands[k] + edge['demand'] <= self.capacity]

        if not candidates:
            return None

        k2 = random.choice(candidates)
        pos = random.randint(0, len(routes[k2]))

        new_routes = [r[:] for r in routes]
        new_routes[k1] = new_routes[k1][:i] + new_routes[k1][i + 1:]
        new_routes[k2] = new_routes[k2][:pos] + [edge_idx] + new_routes[k2][pos:]

        return new_routes

    def solve(self, time_limit, start_time, logger=None):
        """Main solving method."""
        best_routes = None
        best_cost = float('inf')

        # Phase 1: Constructive heuristics
        criteria = ['nearest', 'farthest', 'demand', 'ratio']
        for criterion in criteria:
            if time.time() - start_time > time_limit * 0.25:
                break
            routes = self.path_scanning(criterion)

            # Optimize each route
            for k in range(len(routes)):
                if time.time() - start_time > time_limit * 0.25:
                    break
                routes[k] = self.optimize_route_2opt(routes[k])
                routes[k] = self.optimize_route_or_opt(routes[k])

            cost = self.compute_objective(routes)
            if cost < best_cost:
                best_cost = cost
                best_routes = [r[:] for r in routes]
                if logger:
                    sol = self.build_solution(best_routes)
                    logger.log_solution(sol['objective_value'], sol)

        # Phase 2: Inter-route local search
        if time.time() - start_time < time_limit * 0.5:
            best_routes = self.local_search_inter(best_routes, time_limit, start_time)
            # Re-optimize intra-route after inter-route changes
            for k in range(len(best_routes)):
                if time.time() - start_time > time_limit * 0.7:
                    break
                best_routes[k] = self.optimize_route_2opt(best_routes[k])

            cost = self.compute_objective(best_routes)
            if cost < best_cost:
                best_cost = cost
                if logger:
                    sol = self.build_solution(best_routes)
                    logger.log_solution(sol['objective_value'], sol)

        # Phase 3: Random restarts with perturbation
        random.seed(42)
        while time.time() - start_time < time_limit * 0.85:
            perturbed = self._perturb_solution(best_routes)
            if perturbed is None:
                break

            # Optimize perturbed solution
            for k in range(len(perturbed)):
                if time.time() - start_time > time_limit * 0.85:
                    break
                perturbed[k] = self.optimize_route_2opt(perturbed[k])

            if time.time() - start_time < time_limit * 0.8:
                perturbed = self.local_search_inter(perturbed, time_limit, start_time)

            cost = self.compute_objective(perturbed)
            if cost < best_cost:
                best_cost = cost
                best_routes = perturbed
                if logger:
                    sol = self.build_solution(best_routes)
                    logger.log_solution(sol['objective_value'], sol)

        # Phase 4: Try Gurobi MIP for small instances
        if self.num_required <= 12 and self.num_vehicles <= 5 and time.time() - start_time < time_limit * 0.5:
            mip_result = self._solve_mip(time_limit - (time.time() - start_time))
            if mip_result is not None:
                mip_cost = self.compute_objective(mip_result)
                if mip_cost < best_cost:
                    best_cost = mip_cost
                    best_routes = mip_result
                    if logger:
                        sol = self.build_solution(best_routes)
                        logger.log_solution(sol['objective_value'], sol)

        return best_routes, best_cost

    def _solve_mip(self, remaining_time):
        """Attempt MIP solution for small instances using Gurobi."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None

        if remaining_time < 5:
            return None

        try:
            E = self.num_required
            K = self.num_vehicles

            # Precompute transition costs between states
            # State (e, d): d=0 means end at endpoints[0], d=1 means end at endpoints[1]
            # Cost from depot to state (e, d)
            depot_to = [[0, 0] for _ in range(E)]
            for i in range(E):
                u, v = self.required_edges[i]['endpoints']
                depot_to[i][0] = self.dist[self.depot][v]  # approach from v, end at u
                depot_to[i][1] = self.dist[self.depot][u]  # approach from u, end at v

            # Cost from state (e, d) to depot
            to_depot = [[0, 0] for _ in range(E)]
            for i in range(E):
                u, v = self.required_edges[i]['endpoints']
                to_depot[i][0] = self.dist[u][self.depot]
                to_depot[i][1] = self.dist[v][self.depot]

            # Cost from state (e1, d1) to state (e2, d2)
            trans = [[[[0, 0] for _ in range(2)] for _ in range(E)] for _ in range(E)]
            for i in range(E):
                u_i, v_i = self.required_edges[i]['endpoints']
                for j in range(E):
                    if i == j:
                        continue
                    u_j, v_j = self.required_edges[j]['endpoints']
                    # From (i, 0) [at u_i] to (j, 0) [approach v_j]: dist[u_i][v_j]
                    trans[i][j][0][0] = self.dist[u_i][v_j]
                    # From (i, 0) [at u_i] to (j, 1) [approach u_j]: dist[u_i][u_j]
                    trans[i][j][0][1] = self.dist[u_i][u_j]
                    # From (i, 1) [at v_i] to (j, 0) [approach v_j]: dist[v_i][v_j]
                    trans[i][j][1][0] = self.dist[v_i][v_j]
                    # From (i, 1) [at v_i] to (j, 1) [approach u_j]: dist[v_i][u_j]
                    trans[i][j][1][1] = self.dist[v_i][u_j]

            model = gp.Model("CARP")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(remaining_time * 0.8, 1))
            model.setParam('OutputFlag', 0)

            # Variables: x[i][d][k] = 1 if vehicle k services edge i in direction d
            x = {}
            for i in range(E):
                for d in range(2):
                    for k in range(K):
                        x[i, d, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{d}_{k}")

            # y[i][di][j][dj][k] = 1 if state (i,di) immediately precedes (j,dj) in vehicle k's route
            y = {}
            for i in range(E):
                for di in range(2):
                    for j in range(E):
                        if i == j:
                            continue
                        for dj in range(2):
                            for k in range(K):
                                y[i, di, j, dj, k] = model.addVar(
                                    vtype=GRB.BINARY, name=f"y_{i}_{di}_{j}_{dj}_{k}")

            # s[i][k] = position of edge i in vehicle k's route (for MTZ)
            s = {}
            for i in range(E):
                for k in range(K):
                    s[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=E, name=f"s_{i}_{k}")

            model.update()

            # Objective: minimize deadheading
            obj = gp.LinExpr()
            for i in range(E):
                for d in range(2):
                    for k in range(K):
                        obj += depot_to[i][d] * x[i, d, k]
                        obj += to_depot[i][d] * x[i, d, k]

            for i in range(E):
                for di in range(2):
                    for j in range(E):
                        if i == j:
                            continue
                        for dj in range(2):
                            for k in range(K):
                                obj += trans[i][j][di][dj] * y[i, di, j, dj, k]

            model.setObjective(obj, GRB.MINIMIZE)

            # Constraints
            # 1. Each edge serviced exactly once
            for i in range(E):
                model.addConstr(
                    gp.quicksum(x[i, d, k] for d in range(2) for k in range(K)) == 1)

            # 2. Capacity
            for k in range(K):
                model.addConstr(
                    gp.quicksum(self.required_edges[i]['demand'] * (x[i, 0, k] + x[i, 1, k])
                                for i in range(E)) <= self.capacity)

            # 3. Flow conservation: if edge i is in direction d for vehicle k,
            #    it has exactly one successor (or is last)
            for i in range(E):
                for d in range(2):
                    for k in range(K):
                        # Outgoing arcs from (i, d, k)
                        model.addConstr(
                            gp.quicksum(y[i, d, j, dj, k]
                                        for j in range(E) if j != i
                                        for dj in range(2)) <= x[i, d, k])

            for j in range(E):
                for dj in range(2):
                    for k in range(K):
                        # Incoming arcs to (j, dj, k)
                        model.addConstr(
                            gp.quicksum(y[i, di, j, dj, k]
                                        for i in range(E) if i != j
                                        for di in range(2)) <= x[j, dj, k])

            # 4. MTZ subtour elimination
            for i in range(E):
                for j in range(E):
                    if i == j:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            for k in range(K):
                                model.addConstr(
                                    s[i, k] + 1 <= s[j, k] + E * (1 - y[i, di, j, dj, k]))

            # 5. Link s to x
            for i in range(E):
                for k in range(K):
                    model.addConstr(s[i, k] <= E * (x[i, 0, k] + x[i, 1, k]))

            model.optimize()

            if model.Status == GRB.INFEASIBLE:
                return None

            if model.SolCount == 0:
                return None

            # Extract solution
            routes = [[] for _ in range(K)]
            for i in range(E):
                for d in range(2):
                    for k in range(K):
                        if x[i, d, k].X > 0.5:
                            routes[k].append((i, s[i, k].X))
                            break

            # Sort by position
            for k in range(K):
                routes[k].sort(key=lambda t: t[1])
                routes[k] = [t[0] for t in routes[k]]

            return routes

        except Exception:
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    start_time = time.time()
    solver = CARPSolver(instance)

    best_routes, best_cost = solver.solve(args.time_limit, start_time, logger)

    solution = solver.build_solution(best_routes)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(solution['objective_value'], solution)


if __name__ == '__main__':
    main()