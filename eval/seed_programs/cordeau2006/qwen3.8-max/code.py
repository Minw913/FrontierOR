#!/usr/bin/env python3
"""
Dial-a-Ride Problem (DARP) Solver
Uses MIP (Gurobi) for small instances and heuristic methods for larger ones.
"""

import json
import argparse
import math
import time
import random
import sys

from solution_logger import SolutionLogger


def euclidean_distance(n1, n2):
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)


class DARPInstance:
    def __init__(self, data):
        self.num_vehicles = data['num_vehicles']
        self.num_users = data['num_users']
        self.capacity = data['vehicle_capacity']
        self.max_ride_time = data['maximum_ride_time']
        self.max_route_duration = data['maximum_route_duration']
        self.nodes = data['nodes']
        self.num_nodes = data['num_nodes']

        # Build distance matrix
        self.dist = [[0.0] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                d = euclidean_distance(self.nodes[i], self.nodes[j])
                self.dist[i][j] = d
                self.dist[j][i] = d

        self.origin_depot = 0
        self.dest_depot = self.num_nodes - 1
        self.pickup_nodes = []
        self.dropoff_nodes = []
        self.user_pickup = {}
        self.user_dropoff = {}
        self.node_user = {}

        for node in self.nodes:
            if node['node_type'] == 'pickup':
                self.pickup_nodes.append(node['node_id'])
                self.user_pickup[node['user_id']] = node['node_id']
                self.node_user[node['node_id']] = node['user_id']
            elif node['node_type'] == 'dropoff':
                self.dropoff_nodes.append(node['node_id'])
                self.user_dropoff[node['user_id']] = node['node_id']
                self.node_user[node['node_id']] = node['user_id']

        # Collect actual user IDs from node data
        self.users = sorted(self.user_pickup.keys())

    def earliest(self, node):
        return self.nodes[node]['earliest_time']

    def latest(self, node):
        return self.nodes[node]['latest_time']

    def service_dur(self, node):
        return self.nodes[node]['service_duration']

    def load(self, node):
        return self.nodes[node]['load']


def check_route_feasibility(inst, sequence):
    """Check feasibility of a route. Returns (service_times, end_time) or (None, None)."""
    if not sequence or len(sequence) < 2:
        return None, None
    if sequence[0] != inst.origin_depot or sequence[-1] != inst.dest_depot:
        return None, None
    if len(sequence) == 2:
        return {}, inst.earliest(inst.origin_depot)

    current_time = inst.earliest(inst.origin_depot)
    current_load = 0
    pickup_end_times = {}
    service_times = {}

    for i in range(1, len(sequence)):
        prev = sequence[i - 1]
        curr = sequence[i]

        arrival = current_time + inst.service_dur(prev) + inst.dist[prev][curr]
        service_start = max(arrival, inst.earliest(curr))

        if service_start > inst.latest(curr) + 1e-6:
            return None, None

        current_load += inst.load(curr)
        if current_load < -1e-6 or current_load > inst.capacity + 1e-6:
            return None, None

        ntype = inst.nodes[curr]['node_type']
        if ntype == 'pickup':
            uid = inst.node_user[curr]
            pickup_end_times[uid] = service_start + inst.service_dur(curr)
            service_times[curr] = service_start
        elif ntype == 'dropoff':
            uid = inst.node_user[curr]
            if uid not in pickup_end_times:
                return None, None
            ride_time = service_start - pickup_end_times[uid]
            direct = inst.dist[inst.user_pickup[uid]][inst.user_dropoff[uid]]
            if ride_time < direct - 1e-6 or ride_time > inst.max_ride_time + 1e-6:
                return None, None
            service_times[curr] = service_start

        current_time = service_start

    if current_time - inst.earliest(inst.origin_depot) > inst.max_route_duration + 1e-6:
        return None, None

    return service_times, current_time


def compute_route_cost(inst, sequence):
    if len(sequence) < 2:
        return 0.0
    return sum(inst.dist[sequence[i]][sequence[i + 1]] for i in range(len(sequence) - 1))


def try_insert_user(inst, sequence, user_id):
    """Insert user into route at best feasible position. Returns (new_seq, cost_inc) or (None, inf)."""
    if user_id not in inst.user_pickup or user_id not in inst.user_dropoff:
        return None, float('inf')
    pickup = inst.user_pickup[user_id]
    dropoff = inst.user_dropoff[user_id]
    n = len(sequence)
    old_cost = compute_route_cost(inst, sequence)
    best_inc = float('inf')
    best_seq = None

    for pi in range(1, n):
        for di in range(pi + 1, n + 1):
            new_seq = sequence[:pi] + [pickup] + sequence[pi:di - 1] + [dropoff] + sequence[di - 1:]
            new_cost = compute_route_cost(inst, new_seq)
            inc = new_cost - old_cost
            if inc < best_inc:
                if check_route_feasibility(inst, new_seq)[0] is not None:
                    best_inc = inc
                    best_seq = new_seq
    return best_seq, best_inc


def construct_solution(inst, user_order=None):
    """Cheapest insertion heuristic."""
    if user_order is None:
        user_order = inst.users[:]
    routes_seq = [[inst.origin_depot, inst.dest_depot] for _ in range(inst.num_vehicles)]
    unserved = []

    for uid in user_order:
        best_v, best_seq, best_inc = None, None, float('inf')
        for v in range(inst.num_vehicles):
            new_seq, inc = try_insert_user(inst, routes_seq[v], uid)
            if new_seq is not None and inc < best_inc:
                best_inc = inc
                best_v = v
                best_seq = new_seq
        if best_v is not None:
            routes_seq[best_v] = best_seq
        else:
            unserved.append(uid)
    return routes_seq, unserved


def remove_user_from_seq(inst, seq, uid):
    """Remove user's pickup and dropoff from sequence."""
    p, d = inst.user_pickup[uid], inst.user_dropoff[uid]
    if p not in seq or d not in seq:
        return seq[:]
    pi = seq.index(p)
    di = seq.index(d)
    if pi > di:
        pi, di = di, pi
    return seq[:pi] + seq[pi + 1:di] + seq[di + 1:]


def local_search(inst, routes_seq, time_limit, start_time):
    """Improve with intra-route and inter-route moves."""
    improved = True
    while improved and time.time() - start_time < time_limit - 2:
        improved = False

        # Intra-route 2-opt
        for v in range(inst.num_vehicles):
            if time.time() - start_time > time_limit - 2:
                break
            seq = routes_seq[v]
            if len(seq) <= 3:
                continue
            best_imp = 1e-10
            best_move = None
            for i in range(1, len(seq) - 2):
                for j in range(i + 1, len(seq) - 1):
                    new_seq = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    imp = compute_route_cost(inst, seq) - compute_route_cost(inst, new_seq)
                    if imp > best_imp:
                        if check_route_feasibility(inst, new_seq)[0] is not None:
                            best_imp = imp
                            best_move = (i, j)
            if best_move:
                i, j = best_move
                routes_seq[v] = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                improved = True

        if improved:
            continue

        # Intra-route Or-opt (relocate single node)
        for v in range(inst.num_vehicles):
            if time.time() - start_time > time_limit - 2:
                break
            seq = routes_seq[v]
            if len(seq) <= 3:
                continue
            found = False
            for i in range(1, len(seq) - 1):
                node = seq[i]
                old_cost = compute_route_cost(inst, seq)
                for j in range(1, len(seq)):
                    if j == i or j == i + 1:
                        continue
                    new_seq = seq[:i] + seq[i + 1:]
                    pos = j - 1 if j > i else j
                    if pos > len(new_seq):
                        pos = len(new_seq)
                    new_seq = new_seq[:pos] + [node] + new_seq[pos:]
                    if compute_route_cost(inst, new_seq) < old_cost - 1e-10:
                        if check_route_feasibility(inst, new_seq)[0] is not None:
                            routes_seq[v] = new_seq
                            improved = True
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if improved:
            continue

        # Inter-route relocation
        for v1 in range(inst.num_vehicles):
            if time.time() - start_time > time_limit - 2:
                break
            seq1 = routes_seq[v1]
            users_v1 = set()
            for n_id in seq1:
                if n_id in inst.node_user:
                    users_v1.add(inst.node_user[n_id])
            for uid in sorted(users_v1):
                removed = remove_user_from_seq(inst, seq1, uid)
                if check_route_feasibility(inst, removed)[0] is None:
                    continue
                for v2 in range(inst.num_vehicles):
                    if v2 == v1:
                        continue
                    new_seq2, _ = try_insert_user(inst, routes_seq[v2], uid)
                    if new_seq2 is not None:
                        old_c = compute_route_cost(inst, seq1) + compute_route_cost(inst, routes_seq[v2])
                        new_c = compute_route_cost(inst, removed) + compute_route_cost(inst, new_seq2)
                        if new_c < old_c - 1e-10:
                            routes_seq[v1] = removed
                            routes_seq[v2] = new_seq2
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Inter-route swap
        for v1 in range(inst.num_vehicles):
            if improved or time.time() - start_time > time_limit - 2:
                break
            seq1 = routes_seq[v1]
            users_v1 = sorted(set(inst.node_user[n] for n in seq1 if n in inst.node_user))
            for v2 in range(v1 + 1, inst.num_vehicles):
                if improved:
                    break
                seq2 = routes_seq[v2]
                users_v2 = sorted(set(inst.node_user[n] for n in seq2 if n in inst.node_user))
                for u1 in users_v1:
                    if improved:
                        break
                    for u2 in users_v2:
                        s1_no_u1 = remove_user_from_seq(inst, seq1, u1)
                        s2_no_u2 = remove_user_from_seq(inst, seq2, u2)
                        if check_route_feasibility(inst, s1_no_u1)[0] is None:
                            continue
                        if check_route_feasibility(inst, s2_no_u2)[0] is None:
                            continue
                        new_s1, _ = try_insert_user(inst, s1_no_u1, u2)
                        if new_s1 is None:
                            continue
                        new_s2, _ = try_insert_user(inst, s2_no_u2, u1)
                        if new_s2 is None:
                            continue
                        old_c = compute_route_cost(inst, seq1) + compute_route_cost(inst, seq2)
                        new_c = compute_route_cost(inst, new_s1) + compute_route_cost(inst, new_s2)
                        if new_c < old_c - 1e-10:
                            routes_seq[v1] = new_s1
                            routes_seq[v2] = new_s2
                            improved = True
                            break

    return routes_seq


def lns_iteration(inst, routes_seq, num_remove=3):
    """Large Neighborhood Search: remove and reinsert users."""
    all_users = set()
    user_route = {}
    for v in range(inst.num_vehicles):
        for node in routes_seq[v]:
            if node in inst.node_user:
                uid = inst.node_user[node]
                all_users.add(uid)
                user_route[uid] = v

    if len(all_users) <= num_remove:
        return routes_seq

    to_remove = random.sample(sorted(all_users), min(num_remove, len(all_users)))
    new_routes = [seq[:] for seq in routes_seq]

    for uid in to_remove:
        v = user_route[uid]
        new_routes[v] = remove_user_from_seq(inst, new_routes[v], uid)

    random.shuffle(to_remove)
    for uid in to_remove:
        best_v, best_seq, best_cost = None, None, float('inf')
        for v in range(inst.num_vehicles):
            ns, _ = try_insert_user(inst, new_routes[v], uid)
            if ns is not None:
                c = compute_route_cost(inst, ns)
                if c < best_cost:
                    best_cost = c
                    best_v = v
                    best_seq = ns
        if best_v is not None:
            new_routes[best_v] = best_seq
        else:
            return routes_seq  # Failed, revert

    return new_routes


def build_solution_dict(inst, routes_seq):
    """Build output solution dictionary."""
    total_cost = 0.0
    routes_dict = {}
    service_times_dict = {}
    ride_times_dict = {}

    for v in range(inst.num_vehicles):
        seq = routes_seq[v]
        routes_dict[str(v)] = seq
        total_cost += compute_route_cost(inst, seq)

        current_time = inst.earliest(inst.origin_depot)
        service_times_dict[f'depot_start_{v}'] = current_time
        pickup_end_times = {}

        for i in range(1, len(seq)):
            prev = seq[i - 1]
            curr = seq[i]
            arrival = current_time + inst.service_dur(prev) + inst.dist[prev][curr]
            service_start = max(arrival, inst.earliest(curr))

            ntype = inst.nodes[curr]['node_type']
            if ntype in ('pickup', 'dropoff'):
                service_times_dict[str(curr)] = service_start

            if ntype == 'pickup':
                uid = inst.node_user[curr]
                pickup_end_times[uid] = service_start + inst.service_dur(curr)
            elif ntype == 'dropoff':
                uid = inst.node_user[curr]
                ride_times_dict[str(uid)] = service_start - pickup_end_times[uid]

            current_time = service_start

        service_times_dict[f'depot_end_{v}'] = current_time

    return {
        "objective_value": total_cost,
        "routes": routes_dict,
        "service_times": service_times_dict,
        "ride_times": ride_times_dict
    }


def solve_with_gurobi(inst, time_limit):
    """MIP formulation for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    try:
        n = inst.num_nodes
        K = inst.num_vehicles
        V = list(range(n))
        P = inst.pickup_nodes
        D = inst.dropoff_nodes
        PD = P + D

        model = gp.Model("DARP")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = time_limit
        model.Params.OutputFlag = 0

        # Create arc variables only for potentially feasible arcs
        x = {}
        arc_M = {}
        for i in V:
            if i == inst.dest_depot:
                continue
            for j in V:
                if j == inst.origin_depot or i == j:
                    continue
                M_ij = inst.latest(j) - inst.earliest(i) - inst.service_dur(i) - inst.dist[i][j]
                if M_ij < -1e-6:
                    continue
                arc_M[i, j] = M_ij
                for k in range(K):
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        t = {}
        for i in V:
            for k in range(K):
                t[i, k] = model.addVar(lb=inst.earliest(i), ub=inst.latest(i), name=f"t_{i}_{k}")

        L = {}
        for i in V:
            for k in range(K):
                lb = max(0, inst.load(i))
                ub = min(inst.capacity, inst.capacity + inst.load(i))
                L[i, k] = model.addVar(lb=lb, ub=ub, name=f"L_{i}_{k}")

        model.update()

        # Objective
        model.setObjective(gp.quicksum(inst.dist[i][j] * x[i, j, k] for (i, j, k) in x), GRB.MINIMIZE)

        # Each pickup visited exactly once
        for i in P:
            model.addConstr(gp.quicksum(x[i, j, k] for j in V for k in range(K) if (i, j, k) in x) == 1)

        # Flow conservation at request nodes
        for i in PD:
            for k in range(K):
                model.addConstr(
                    gp.quicksum(x[j, i, k] for j in V if (j, i, k) in x) ==
                    gp.quicksum(x[i, j, k] for j in V if (i, j, k) in x))

        # Depot constraints
        for k in range(K):
            model.addConstr(gp.quicksum(x[inst.origin_depot, j, k] for j in V if (inst.origin_depot, j, k) in x) <= 1)
            model.addConstr(gp.quicksum(x[i, inst.dest_depot, k] for i in V if (i, inst.dest_depot, k) in x) <= 1)

        # Same vehicle for pickup and dropoff
        for uid in inst.users:
            pi, di = inst.user_pickup[uid], inst.user_dropoff[uid]
            for k in range(K):
                model.addConstr(
                    gp.quicksum(x[pi, j, k] for j in V if (pi, j, k) in x) ==
                    gp.quicksum(x[di, j, k] for j in V if (di, j, k) in x))

        # Time constraints
        for (i, j, k) in x:
            M_ij = arc_M[i, j]
            model.addConstr(t[j, k] >= t[i, k] + inst.service_dur(i) + inst.dist[i][j] - M_ij * (1 - x[i, j, k]))

        # Load constraints
        for (i, j, k) in x:
            model.addConstr(L[j, k] >= L[i, k] + inst.load(j) - inst.capacity * (1 - x[i, j, k]))

        # Route duration
        for k in range(K):
            model.addConstr(t[inst.dest_depot, k] - t[inst.origin_depot, k] <= inst.max_route_duration)

        # Ride time constraints
        for uid in inst.users:
            pi, di = inst.user_pickup[uid], inst.user_dropoff[uid]
            direct = inst.dist[pi][di]
            for k in range(K):
                y_uk = gp.quicksum(x[pi, j, k] for j in V if (pi, j, k) in x)
                big_M = inst.latest(di) - inst.earliest(pi)
                model.addConstr(t[di, k] - t[pi, k] - inst.service_dur(pi) >= direct - big_M * (1 - y_uk))
                model.addConstr(t[di, k] - t[pi, k] - inst.service_dur(pi) <= inst.max_ride_time + big_M * (1 - y_uk))

        model.optimize()

        if model.SolCount > 0:
            routes_seq = []
            for k in range(K):
                seq = [inst.origin_depot]
                current = inst.origin_depot
                visited = {inst.origin_depot}
                while True:
                    found = False
                    for j in V:
                        if j != current and j not in visited and (current, j, k) in x:
                            if x[current, j, k].X > 0.5:
                                seq.append(j)
                                visited.add(j)
                                current = j
                                found = True
                                break
                    if not found:
                        break
                if seq[-1] != inst.dest_depot:
                    seq.append(inst.dest_depot)
                routes_seq.append(seq)

            # Verify all routes
            all_ok = True
            for s in routes_seq:
                if check_route_feasibility(inst, s)[0] is None:
                    all_ok = False
                    break
            if all_ok:
                return routes_seq

    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="DARP Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    random.seed(42)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    inst = DARPInstance(data)
    best_solution = None
    best_cost = float('inf')

    def update_best(routes_seq):
        nonlocal best_solution, best_cost
        sol = build_solution_dict(inst, routes_seq)
        if sol['objective_value'] < best_cost:
            best_cost = sol['objective_value']
            best_solution = sol
            if logger:
                logger.log_solution(best_cost, best_solution)
            return True
        return False

    # Try MIP for small instances
    if inst.num_users <= 12:
        remaining = args.time_limit - (time.time() - start_time) - 2
        if remaining > 5:
            mip_routes = solve_with_gurobi(inst, min(remaining, args.time_limit * 0.6))
            if mip_routes is not None:
                update_best(mip_routes)

    # Heuristic construction and improvement
    if time.time() - start_time < args.time_limit - 2:
        num_attempts = max(3, min(15, int((args.time_limit - (time.time() - start_time)) / 3)))
        for attempt in range(num_attempts):
            if time.time() - start_time > args.time_limit - 3:
                break
            if attempt == 0:
                order = inst.users[:]
            else:
                order = inst.users[:]
                random.shuffle(order)

            routes_seq, unserved = construct_solution(inst, order)

            # Try to insert unserved users
            still_unserved = []
            for uid in unserved:
                inserted = False
                for v in range(inst.num_vehicles):
                    ns, _ = try_insert_user(inst, routes_seq[v], uid)
                    if ns is not None:
                        routes_seq[v] = ns
                        inserted = True
                        break
                if not inserted:
                    still_unserved.append(uid)

            if not still_unserved:
                routes_seq = local_search(inst, routes_seq, args.time_limit, start_time)
                update_best(routes_seq)

    # LNS phase
    if best_solution is not None and time.time() - start_time < args.time_limit - 2:
        best_routes_seq = [best_solution['routes'][str(v)] for v in range(inst.num_vehicles)]
        current_routes = [s[:] for s in best_routes_seq]
        current_cost = best_cost

        while time.time() - start_time < args.time_limit - 1:
            num_rm = random.randint(1, min(4, max(1, len(inst.users))))
            new_routes = lns_iteration(inst, current_routes, num_rm)
            if new_routes is not None:
                new_routes = local_search(inst, new_routes, args.time_limit, start_time)
                new_cost = sum(compute_route_cost(inst, s) for s in new_routes)
                if new_cost < current_cost - 1e-10:
                    current_routes = new_routes
                    current_cost = new_cost
                    update_best(current_routes)
                elif random.random() < 0.1:
                    current_routes = new_routes
                    current_cost = new_cost

    # Fallback: ensure we have a solution
    if best_solution is None:
        routes_seq, unserved = construct_solution(inst)
        if unserved:
            # Try harder to insert
            for uid in unserved[:]:
                for v in range(inst.num_vehicles):
                    ns, _ = try_insert_user(inst, routes_seq[v], uid)
                    if ns is not None:
                        routes_seq[v] = ns
                        unserved.remove(uid)
                        break
        best_solution = build_solution_dict(inst, routes_seq)
        if logger:
            logger.log_solution(best_solution['objective_value'], best_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()