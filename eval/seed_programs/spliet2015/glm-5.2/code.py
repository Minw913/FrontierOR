import json
import argparse
import time
import random
from solution_logger import SolutionLogger

def parse_instance(path):
    with open(path) as f:
        return json.load(f)

def compute_service_times(route, tw, dist, depot_exo, sd, ed):
    st = {}
    t = float(depot_exo[0])
    st[str(sd)] = t
    prev = sd
    for i in range(1, len(route)):
        node = route[i]
        arrival = t + dist[prev][node]
        if node == ed:
            if arrival > depot_exo[1] + 1e-6:
                return None
            st[str(node)] = arrival
            t = arrival
        else:
            tws, twe = tw[node]
            if arrival > twe + 1e-6:
                return None
            service = max(arrival, tws)
            st[str(node)] = service
            t = service
        prev = node
    return st

def route_cost(route, dist):
    return sum(dist[route[i]][route[i+1]] for i in range(len(route) - 1))

def solve_vrptw(custs, tw, dems, cap, dist, depot_exo, sd, ed, sort_key=None):
    if sort_key is None:
        sort_key = lambda c: tw[c][0]
    ordered = sorted(custs, key=sort_key)
    routes = []
    for cust in ordered:
        best = None
        best_inc = float('inf')
        for ri, r in enumerate(routes):
            cur_dem = sum(dems[n - 1] for n in r[1:-1])
            if cur_dem + dems[cust - 1] > cap:
                continue
            for pos in range(1, len(r)):
                nr = r[:pos] + [cust] + r[pos:]
                if compute_service_times(nr, tw, dist, depot_exo, sd, ed) is None:
                    continue
                inc = route_cost(nr, dist) - route_cost(r, dist)
                if inc < best_inc:
                    best_inc = inc
                    best = (ri, pos)
        if best is not None:
            ri, pos = best
            routes[ri] = routes[ri][:pos] + [cust] + routes[ri][pos:]
        else:
            routes.append([sd, cust, ed])
    return routes

def improve_2opt(routes, tw, dist, depot_exo, sd, ed):
    improved = True
    while improved:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            n = len(r)
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    nr = r[:i] + r[i:j+1][::-1] + r[j+1:]
                    if route_cost(nr, dist) < route_cost(r, dist) - 1e-9:
                        if compute_service_times(nr, tw, dist, depot_exo, sd, ed) is not None:
                            routes[ri] = nr
                            r = nr
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return routes

def improve_relocate(routes, tw, dems, cap, dist, depot_exo, sd, ed):
    improved = True
    while improved:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            if len(r) <= 3:
                continue
            for pos in range(1, len(r) - 1):
                cust = r[pos]
                r_wo = r[:pos] + r[pos+1:]
                if compute_service_times(r_wo, tw, dist, depot_exo, sd, ed) is None:
                    continue
                base = route_cost(r, dist)
                best_delta = -1e-9
                best_move = None
                for rj in range(len(routes)):
                    if rj == ri:
                        target = r_wo
                    else:
                        target = routes[rj]
                        dem = sum(dems[n-1] for n in target[1:-1])
                        if dem + dems[cust-1] > cap:
                            continue
                    for posj in range(1, len(target)):
                        if rj == ri and posj == pos:
                            continue
                        nrj = target[:posj] + [cust] + target[posj:]
                        if compute_service_times(nrj, tw, dist, depot_exo, sd, ed) is None:
                            continue
                        if rj == ri:
                            delta = route_cost(nrj, dist) - base
                        else:
                            delta = (route_cost(r_wo, dist) - base +
                                    route_cost(nrj, dist) - route_cost(routes[rj], dist))
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (rj, nrj)
                if best_move is not None:
                    rj, nrj = best_move
                    if rj == ri:
                        routes[ri] = nrj
                    else:
                        routes[ri] = r_wo
                        routes[rj] = nrj
                    routes = [r for r in routes if len(r) > 2]
                    improved = True
                    break
            if improved:
                break
    return routes

def evaluate(tw, instance, dist, sort_key=None, do_2opt=True, do_relocate=True):
    sd, ed = 0, instance['num_customers'] + 1
    depot_exo = instance['depot']['exogenous_time_window']
    cap = instance['vehicle_capacity']
    custs = [c['id'] for c in instance['customers']]
    
    total = 0.0
    sr = {}
    for sc in instance['scenarios']:
        dems = sc['demands']
        routes = solve_vrptw(custs, tw, dems, cap, dist, depot_exo, sd, ed, sort_key)
        if do_2opt:
            routes = improve_2opt(routes, tw, dist, depot_exo, sd, ed)
        if do_relocate:
            routes = improve_relocate(routes, tw, dems, cap, dist, depot_exo, sd, ed)
            if do_2opt:
                routes = improve_2opt(routes, tw, dist, depot_exo, sd, ed)
        sc_cost = 0.0
        rl = []
        for r in routes:
            st = compute_service_times(r, tw, dist, depot_exo, sd, ed)
            c = route_cost(r, dist)
            sc_cost += c
            rl.append({'route': list(r), 'arrival_times': st if st else {}, 'cost': c})
        sr[sc['scenario_id']] = rl
        total += sc['probability'] * sc_cost
    return total, sr

def make_solution(obj, tw, routes):
    return {
        'objective_value': obj,
        'time_windows': {str(k): [v[0], v[1]] for k, v in tw.items()},
        'routes': {str(sid): rl for sid, rl in routes.items()}
    }

def get_tw(instance, strategy='centered', offset=0, st_info=None):
    tw = {}
    for c in instance['customers']:
        exo = c['exogenous_time_window']
        w = c['endogenous_time_window_width']
        if strategy == 'service_times' and st_info and c['id'] in st_info:
            times = st_info[c['id']]
            tmin, tmax = min(times), max(times)
            if tmax - tmin <= w:
                start = max(float(exo[0]), min(tmin, tmax - w))
            else:
                center = sum(times) / len(times)
                start = center - w / 2
        elif strategy == 'centered':
            start = (float(exo[0]) + float(exo[1]) - w) / 2
        elif strategy == 'earliest':
            start = float(exo[0])
        elif strategy == 'latest':
            start = float(exo[1]) - w
        elif strategy == 'offset':
            start = float(exo[0]) + offset * (float(exo[1]) - float(exo[0]) - w)
        else:
            start = (float(exo[0]) + float(exo[1]) - w) / 2
        start = max(float(exo[0]), min(start, float(exo[1]) - w))
        tw[c['id']] = [start, start + w]
    return tw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = parse_instance(args.instance_path)
    dist = instance['distance_matrix']
    sd, ed = 0, instance['num_customers'] + 1
    depot_exo = instance['depot']['exogenous_time_window']
    cap = instance['vehicle_capacity']
    custs = [c['id'] for c in instance['customers']]
    
    t0 = time.time()
    best_obj = float('inf')
    best_tw = None
    best_routes = None
    
    def try_update(obj, tw, routes):
        nonlocal best_obj, best_tw, best_routes
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_tw = {k: [v[0], v[1]] for k, v in tw.items()}
            best_routes = {}
            for sid, rl in routes.items():
                best_routes[sid] = [{'route': list(r['route']), 'arrival_times': dict(r['arrival_times']), 'cost': r['cost']} for r in rl]
            if logger:
                logger.log_solution(best_obj, make_solution(best_obj, best_tw, best_routes))
    
    def time_left():
        return time.time() - t0 < args.time_limit
    
    # Compute service times from full exogenous routing
    all_st = {}
    if time_left():
        full_tw = {c['id']: [float(c['exogenous_time_window'][0]), float(c['exogenous_time_window'][1])] for c in instance['customers']}
        for sc in instance['scenarios']:
            if not time_left():
                break
            dems = sc['demands']
            routes = solve_vrptw(custs, full_tw, dems, cap, dist, depot_exo, sd, ed)
            routes = improve_2opt(routes, full_tw, dist, depot_exo, sd, ed)
            for r in routes:
                st = compute_service_times(r, full_tw, dist, depot_exo, sd, ed)
                if st:
                    for ns, t in st.items():
                        nid = int(ns)
                        if nid != sd and nid != ed:
                            all_st.setdefault(nid, []).append(t)
    
    # Try different TW strategies
    strategies = [
        ('centered', 0, None),
        ('earliest', 0, None),
        ('latest', 0, None),
        ('offset', 0.25, None),
        ('offset', 0.5, None),
        ('offset', 0.75, None),
        ('service_times', 0, all_st),
    ]
    
    for strat, off, st_info in strategies:
        if not time_left():
            break
        tw = get_tw(instance, strat, off, st_info)
        obj, routes = evaluate(tw, instance, dist, do_2opt=True, do_relocate=True)
        try_update(obj, tw, routes)
        if not time_left():
            break
        sk = lambda c: dist[sd][c]
        obj, routes = evaluate(tw, instance, dist, sort_key=sk, do_2opt=True, do_relocate=True)
        try_update(obj, tw, routes)
    
    # Local search on TW starts
    if best_tw and time_left():
        tw = {k: [v[0], v[1]] for k, v in best_tw.items()}
        fast_obj, _ = evaluate(tw, instance, dist, do_2opt=True, do_relocate=False)
        
        improved = True
        while improved and time_left():
            improved = False
            for c in instance['customers']:
                if not time_left():
                    break
                cid = c['id']
                exo = c['exogenous_time_window']
                w = c['endogenous_time_window_width']
                cur = tw[cid][0]
                rng = float(exo[1]) - float(exo[0]) - w
                if rng <= 1e-9:
                    continue
                step = max(rng / 10, 0.1)
                best_shift = 0
                best_s_obj = fast_obj
                for shift in [-step, step, -2*step, 2*step]:
                    if not time_left():
                        break
                    ns = max(float(exo[0]), min(cur + shift, float(exo[1]) - w))
                    if abs(ns - cur) < 1e-9:
                        continue
                    tw[cid] = [ns, ns + w]
                    obj, _ = evaluate(tw, instance, dist, do_2opt=True, do_relocate=False)
                    if obj < best_s_obj - 1e-9:
                        best_s_obj = obj
                        best_shift = shift
                ns = max(float(exo[0]), min(cur + best_shift, float(exo[1]) - w))
                tw[cid] = [ns, ns + w]
                if best_shift != 0:
                    improved = True
                    fast_obj = best_s_obj
                    obj, routes = evaluate(tw, instance, dist, do_2opt=True, do_relocate=True)
                    try_update(obj, tw, routes)
    
    # Random perturbation search
    random.seed(0)
    while time_left():
        if best_tw:
            tw = {k: [v[0], v[1]] for k, v in best_tw.items()}
        else:
            tw = get_tw(instance, 'centered')
        for c in instance['customers']:
            if random.random() < 0.3:
                exo = c['exogenous_time_window']
                w = c['endogenous_time_window_width']
                shift = random.uniform(-0.3, 0.3) * (float(exo[1]) - float(exo[0]))
                ns = max(float(exo[0]), min(tw[c['id']][0] + shift, float(exo[1]) - w))
                tw[c['id']] = [ns, ns + w]
        obj, routes = evaluate(tw, instance, dist, do_2opt=True, do_relocate=True)
        try_update(obj, tw, routes)
    
    # Write solution
    if best_tw is None:
        best_tw = get_tw(instance, 'centered')
        best_obj, best_routes = evaluate(best_tw, instance, dist, do_2opt=False, do_relocate=False)
    
    sol = make_solution(best_obj, best_tw, best_routes)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == '__main__':
    main()