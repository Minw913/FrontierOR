import argparse
import json
import math
import time
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def route_cost(route, coords):
    return sum(dist(coords[route[i]], coords[route[i + 1]]) for i in range(len(route) - 1))


def two_opt(route, coords, deadline=None):
    best = list(route)
    n = len(best)
    if n < 4:
        return best
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                delta = (dist(coords[a], coords[c]) + dist(coords[b], coords[d])
                         - dist(coords[a], coords[b]) - dist(coords[c], coords[d]))
                if delta < -1e-9:
                    best = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    improved = True
        if deadline and time.time() > deadline:
            break
    return best


def clarke_wright(depot_id, customer_ids, coords, Q, demands, forbid_pairs=None):
    if not customer_ids:
        return []
    routes = {}
    route_of = {}
    route_load = {}
    for c in customer_ids:
        routes[c] = [depot_id, c, depot_id]
        route_of[c] = c
        route_load[c] = demands[c]

    savings = []
    for i in range(len(customer_ids)):
        for j in range(i + 1, len(customer_ids)):
            c1, c2 = customer_ids[i], customer_ids[j]
            if forbid_pairs:
                p = (min(c1, c2), max(c1, c2))
                if p in forbid_pairs:
                    continue
            s = (dist(coords[c1], coords[depot_id]) + dist(coords[c2], coords[depot_id])
                 - dist(coords[c1], coords[c2]))
            savings.append((s, c1, c2))
    savings.sort(reverse=True)

    for s, c1, c2 in savings:
        if s <= 0:
            break
        r1, r2 = route_of[c1], route_of[c2]
        if r1 == r2:
            continue
        e1s = routes[r1][1] == c1
        e1e = routes[r1][-2] == c1
        e2s = routes[r2][1] == c2
        e2e = routes[r2][-2] == c2
        if not (e1s or e1e) or not (e2s or e2e):
            continue
        new_load = route_load[r1] + route_load[r2]
        if new_load > Q:
            continue
        if e1e and e2s:
            new_route = routes[r1][:-1] + routes[r2][1:]
        elif e1s and e2e:
            new_route = routes[r2][:-1] + routes[r1][1:]
        elif e1e and e2e:
            new_route = routes[r1][:-1] + routes[r2][::-1][1:]
        else:
            new_route = routes[r1][::-1][:-1] + routes[r2][1:]
        del routes[r2]
        routes[r1] = new_route
        route_load[r1] = new_load
        for c in new_route:
            if c != depot_id:
                route_of[c] = r1
    return list(routes.values())


def force_merge(routes, depot_id, coords, Q, demands, max_vehicles, sat_of=None):
    routes = [list(r) for r in routes]
    while len(routes) > max_vehicles:
        best_delta = float('inf')
        best_pair = None
        best_new = None
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                r1, r2 = routes[i], routes[j]
                if sat_of is not None:
                    sats1 = set(sat_of[c] for c in r1 if c != depot_id)
                    sats2 = set(sat_of[c] for c in r2 if c != depot_id)
                    if sats1 & sats2:
                        continue
                load1 = sum(demands[c] for c in r1 if c != depot_id)
                load2 = sum(demands[c] for c in r2 if c != depot_id)
                if load1 + load2 > Q:
                    continue
                for a, b in [(r1, r2), (r2, r1)]:
                    for ra in (False, True):
                        for rb in (False, True):
                            aa = a[::-1] if ra else a
                            bb = b[::-1] if rb else b
                            new_route = aa[:-1] + bb[1:]
                            old_cost = route_cost(a, coords) + route_cost(b, coords)
                            new_cost = route_cost(new_route, coords)
                            delta = new_cost - old_cost
                            if delta < best_delta - 1e-12:
                                best_delta = delta
                                best_pair = (i, j)
                                best_new = new_route
        if best_pair is None:
            break
        i, j = best_pair
        routes = [r for k, r in enumerate(routes) if k != i and k != j]
        routes.append(best_new)
    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit - 1)

    with open(args.instance_path) as f:
        inst = json.load(f)

    depot = inst['depot']
    satellites = inst['satellites']
    customers = inst['customers']
    groups_data = inst['groups']

    coords = {depot['id']: (depot['x'], depot['y'])}
    for s in satellites:
        coords[s['id']] = (s['x'], s['y'])
    for c in customers:
        coords[c['id']] = (c['x'], c['y'])

    sat_by_id = {s['id']: s for s in satellites}
    cust_by_id = {c['id']: c for c in customers}

    Q1 = inst['Q1']
    Q2 = inst['Q2']
    T1 = inst['T1']
    T2 = inst['T2']
    sat_ids = [s['id'] for s in satellites]

    group_ids = []
    group_custs = {}
    group_demand = {}
    for g in groups_data:
        gid = g['group_id']
        group_ids.append(gid)
        group_custs[gid] = list(g['customer_ids'])
        group_demand[gid] = sum(cust_by_id[c]['demand'] for c in group_custs[gid])

    # Estimate per (group, satellite)
    est_cost = {}
    est_veh = {}
    for gid in group_ids:
        gd = {c: cust_by_id[c]['demand'] for c in group_custs[gid]}
        for sid in sat_ids:
            routes = clarke_wright(sid, group_custs[gid], coords, Q2, gd)
            routes = [two_opt(r, coords) for r in routes]
            est_cost[(gid, sid)] = sum(route_cost(r, coords) for r in routes)
            est_veh[(gid, sid)] = len(routes)

    assignment = {}
    if HAS_GUROBI and group_ids and sat_ids:
        try:
            m = gp.Model("assign")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = max(1, min(30.0, args.time_limit * 0.3))

            x = {}
            feasible_pairs = {gid: [] for gid in group_ids}
            for gid in group_ids:
                for sid in sat_ids:
                    if group_demand[gid] > sat_by_id[sid]['capacity']:
                        continue
                    if est_veh[(gid, sid)] > sat_by_id[sid]['max_vehicles']:
                        continue
                    v = m.addVar(vtype=GRB.BINARY, name=f"x_{gid}_{sid}")
                    x[(gid, sid)] = v
                    feasible_pairs[gid].append(sid)

            # Ensure each group has at least one feasible satellite
            for gid in group_ids:
                if not feasible_pairs[gid]:
                    # relax: pick any satellite ignoring vehicle limit
                    for sid in sat_ids:
                        if group_demand[gid] <= sat_by_id[sid]['capacity']:
                            v = m.addVar(vtype=GRB.BINARY, name=f"xr_{gid}_{sid}")
                            x[(gid, sid)] = v
                            feasible_pairs[gid].append(sid)
                    if not feasible_pairs[gid]:
                        # ignore capacity too
                        for sid in sat_ids:
                            v = m.addVar(vtype=GRB.BINARY, name=f"xx_{gid}_{sid}")
                            x[(gid, sid)] = v
                            feasible_pairs[gid].append(sid)

            for gid in group_ids:
                m.addConstr(gp.quicksum(x[(gid, sid)] for sid in feasible_pairs[gid]) == 1)

            for sid in sat_ids:
                m.addConstr(gp.quicksum(
                    group_demand[gid] * x[(gid, sid)] for gid in group_ids if (gid, sid) in x
                ) <= sat_by_id[sid]['capacity'])
                m.addConstr(gp.quicksum(
                    est_veh[(gid, sid)] * x[(gid, sid)] for gid in group_ids if (gid, sid) in x
                ) <= sat_by_id[sid]['max_vehicles'])

            m.addConstr(gp.quicksum(
                est_veh[(gid, sid)] * x[(gid, sid)] for gid in group_ids for sid in sat_ids if (gid, sid) in x
            ) <= T2)

            obj = gp.quicksum(
                (sat_by_id[sid]['handling_cost'] * group_demand[gid] + est_cost[(gid, sid)]) * x[(gid, sid)]
                for gid in group_ids for sid in sat_ids if (gid, sid) in x
            )
            m.setObjective(obj, GRB.MINIMIZE)
            m.optimize()

            for gid in group_ids:
                assigned = None
                for sid in feasible_pairs[gid]:
                    if x[(gid, sid)].X > 0.5:
                        assigned = sid
                        break
                if assigned is None:
                    assigned = min(feasible_pairs[gid],
                                   key=lambda s: est_cost[(gid, s)] + sat_by_id[s]['handling_cost'] * group_demand[gid])
                assignment[gid] = assigned
        except Exception:
            assignment = {}

    if not assignment:
        # Greedy fallback
        sat_load = defaultdict(float)
        sat_veh = defaultdict(int)
        # sort groups by descending demand
        order = sorted(group_ids, key=lambda g: -group_demand[g])
        for gid in order:
            best_sid = None
            best_score = float('inf')
            for sid in sat_ids:
                cap_ok = sat_load[sid] + group_demand[gid] <= sat_by_id[sid]['capacity']
                veh_ok = sat_veh[sid] + est_veh[(gid, sid)] <= sat_by_id[sid]['max_vehicles']
                t2_ok = sum(sat_veh.values()) + est_veh[(gid, sid)] <= T2
                score = est_cost[(gid, sid)] + sat_by_id[sid]['handling_cost'] * group_demand[gid]
                if cap_ok and veh_ok and t2_ok and score < best_score:
                    best_score = score
                    best_sid = sid
            if best_sid is None:
                # relax
                for sid in sat_ids:
                    score = est_cost[(gid, sid)] + sat_by_id[sid]['handling_cost'] * group_demand[gid]
                    if score < best_score:
                        best_score = score
                        best_sid = sid
            assignment[gid] = best_sid
            if best_sid is not None:
                sat_load[best_sid] += group_demand[gid]
                sat_veh[best_sid] += est_veh[(gid, best_sid)]

    # Second echelon routing per satellite
    sat_to_custs = defaultdict(list)
    for gid in group_ids:
        sat_to_custs[assignment[gid]].extend(group_custs[gid])

    second_echelon_routes = []
    sat_total_demand = {sid: 0 for sid in sat_ids}
    for sid in sat_ids:
        custs = sat_to_custs.get(sid, [])
        if not custs:
            continue
        gd = {c: cust_by_id[c]['demand'] for c in custs}
        routes = clarke_wright(sid, custs, coords, Q2, gd)
        routes = [two_opt(r, coords, deadline) for r in routes]
        max_v = sat_by_id[sid]['max_vehicles']
        if len(routes) > max_v:
            routes = force_merge(routes, sid, coords, Q2, gd, max_v)
            routes = [two_opt(r, coords, deadline) for r in routes]
        for r in routes:
            second_echelon_routes.append({'satellite': sid, 'route': r})
        sat_total_demand[sid] = sum(gd[c] for c in custs)

    # First echelon: split demands into virtual nodes
    virtual_nodes = []
    next_id = max(coords.keys()) + 1
    sat_virtual_map = defaultdict(list)
    for sid in sat_ids:
        d = sat_total_demand.get(sid, 0)
        if d <= 0:
            continue
        n = max(1, math.ceil(d / Q1))
        base = d // n
        rem = d - base * n
        for i in range(n):
            load = base + (1 if i < rem else 0)
            vid = next_id
            virtual_nodes.append((vid, sid, load))
            coords[vid] = coords[sid]
            sat_virtual_map[sid].append(vid)
            next_id += 1

    forbid = set()
    for sid, vids in sat_virtual_map.items():
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                forbid.add((min(vids[i], vids[j]), max(vids[i], vids[j])))

    vdemands = {vid: load for vid, _, load in virtual_nodes}
    vids = [vid for vid, _, _ in virtual_nodes]
    sat_of_virtual = {vid: sid for vid, sid, _ in virtual_nodes}

    first_routes_raw = []
    if vids:
        first_routes_raw = clarke_wright(depot['id'], vids, coords, Q1, vdemands, forbid_pairs=forbid)
        first_routes_raw = [two_opt(r, coords, deadline) for r in first_routes_raw]
        if len(first_routes_raw) > T1:
            first_routes_raw = force_merge(first_routes_raw, depot['id'], coords, Q1, vdemands, T1, sat_of=sat_of_virtual)
            first_routes_raw = [two_opt(r, coords, deadline) for r in first_routes_raw]

    first_echelon_routes = []
    for vi, route in enumerate(first_routes_raw):
        deliveries = {}
        mapped_route = [depot['id']]
        for node in route[1:-1]:
            sid = sat_of_virtual.get(node)
            if sid is not None:
                deliveries[sid] = deliveries.get(sid, 0) + vdemands[node]
                mapped_route.append(sid)
        mapped_route.append(depot['id'])
        cleaned = [mapped_route[0]]
        for n in mapped_route[1:]:
            if n != cleaned[-1]:
                cleaned.append(n)
        if cleaned[-1] != depot['id']:
            cleaned.append(depot['id'])
        # String keys for deliveries per schema
        deliveries_str = {str(k): v for k, v in deliveries.items()}
        first_echelon_routes.append({
            'vehicle': vi,
            'route': cleaned,
            'deliveries': deliveries_str
        })

    # Compute total cost
    total_cost = 0.0
    for r in first_echelon_routes:
        total_cost += route_cost(r['route'], coords)
    for sid in sat_ids:
        d = sat_total_demand.get(sid, 0)
        total_cost += sat_by_id[sid]['handling_cost'] * d
    for sr in second_echelon_routes:
        total_cost += route_cost(sr['route'], coords)

    solution = {
        'objective_value': total_cost,
        'first_echelon_routes': first_echelon_routes,
        'second_echelon_routes': [{'satellite': sr['satellite'], 'route': sr['route']}
                                  for sr in second_echelon_routes],
        'group_assignments': {str(gid): assignment[gid] for gid in group_ids}
    }

    if logger:
        logger.log_solution(total_cost, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()