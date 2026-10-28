import json
import argparse
import time
import math
import random
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:
    HAVE_GUROBI = False

from solution_logger import SolutionLogger


# -----------------------------
# Global problem data (filled in main)
# -----------------------------
N = 0
K = 0
Q = 0
rho = 0.0
start_id = 0
end_id = 0
e0 = 0.0
l0 = 0.0
e_end = 0.0
l_end = 0.0
dist = []
demand = {}
tw_e = {}
tw_l = {}
serv = {}
pref = {}
cust_ids = []

EPS = 1e-6
_CACHE_MAX = 300000
_fb_cache = {}
_rc_cache = {}
_eval_cache = {}

# Gurobi environment (reused)
_genv = None


def init_gurobi_env():
    global _genv, HAVE_GUROBI
    if not HAVE_GUROBI:
        return
    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        _genv = env
    except Exception:
        _genv = None
        HAVE_GUROBI = False


# -----------------------------
# Route feasibility (forward-backward) with caching
# -----------------------------
def fb(route):
    key = tuple(route)
    cached = _fb_cache.get(key)
    if cached is not None:
        return cached
    if len(_fb_cache) > _CACHE_MAX:
        _fb_cache.clear()
    k = len(route)
    if k == 0:
        res = (True, [], [])
        _fb_cache[key] = res
        return res

    # capacity
    tot = 0
    for c in route:
        tot += demand[c]
        if tot > Q + EPS:
            res = (False, None, None)
            _fb_cache[key] = res
            return res

    E = [0.0] * k
    c = route[0]
    E[0] = max(float(tw_e[c]), e0 + dist[start_id][c])
    if E[0] > tw_l[c] + EPS:
        res = (False, E, None)
        _fb_cache[key] = res
        return res
    for j in range(1, k):
        cj = route[j]
        pj = route[j - 1]
        E[j] = max(float(tw_e[cj]), E[j - 1] + serv[pj] + dist[pj][cj])
        if E[j] > tw_l[cj] + EPS:
            res = (False, E, None)
            _fb_cache[key] = res
            return res

    last = route[-1]
    arr = E[-1] + serv[last] + dist[last][end_id]
    if arr > l_end + EPS:
        res = (False, E, None)
        _fb_cache[key] = res
        return res

    # return earliest: can wait at last customer
    min_last = e_end - serv[last] - dist[last][end_id]
    if E[-1] < min_last:
        E[-1] = min_last
        if E[-1] > tw_l[last] + EPS or E[-1] + serv[last] + dist[last][end_id] > l_end + EPS:
            res = (False, E, None)
            _fb_cache[key] = res
            return res

    L = [0.0] * k
    L[-1] = min(float(tw_l[last]), l_end - serv[last] - dist[last][end_id])
    ok = (L[-1] >= E[-1] - EPS)
    for j in range(k - 2, -1, -1):
        cj = route[j]
        nj = route[j + 1]
        L[j] = min(float(tw_l[cj]), L[j + 1] - serv[cj] - dist[cj][nj])
        if L[j] < E[j] - EPS:
            ok = False
    res = (ok, E, L)
    _fb_cache[key] = res
    return res


def routing_cost(route):
    key = tuple(route)
    cached = _rc_cache.get(key)
    if cached is not None:
        return cached
    if not route:
        _rc_cache[key] = 0.0
        return 0.0
    rc = dist[start_id][route[0]]
    for j in range(len(route) - 1):
        rc += dist[route[j]][route[j + 1]]
    rc += dist[route[-1]][end_id]
    _rc_cache[key] = rc
    return rc


def greedy_schedule(E, L, route):
    k = len(route)
    if k == 0:
        return []
    g = [0.0] * k
    g[0] = min(max(pref[route[0]], E[0]), L[0])
    for j in range(1, k):
        earliest = g[j - 1] + serv[route[j - 1]] + dist[route[j - 1]][route[j]]
        lower = max(earliest, E[j])
        g[j] = min(max(pref[route[j]], lower), L[j])
    return g


def solve_schedule_qp(route, E, L):
    """Return (inconvenience_cost, schedule_list)."""
    k = len(route)
    if k == 0:
        return 0.0, []
    if rho == 0.0 or not HAVE_GUROBI or _genv is None:
        g = greedy_schedule(E, L, route)
        inc = rho * sum((g[j] - pref[route[j]]) ** 2 for j in range(k))
        return inc, g

    try:
        m = gp.Model('sched', env=_genv)
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.NumericFocus = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.TimeLimit = 3.0
        t = m.addVars(k, lb=[E[j] for j in range(k)], ub=[L[j] for j in range(k)], name='t')
        for j in range(k - 1):
            m.addConstr(t[j + 1] >= t[j] + serv[route[j]] + dist[route[j]][route[j + 1]])
        obj = gp.quicksum((t[j] - pref[route[j]]) * (t[j] - pref[route[j]]) for j in range(k))
        m.setObjective(rho * obj, GRB.MINIMIZE)
        m.optimize()
        if m.SolCount > 0:
            vals = [t[j].X for j in range(k)]
            inc = rho * sum((vals[j] - pref[route[j]]) ** 2 for j in range(k))
            return inc, vals
    except Exception:
        pass
    g = greedy_schedule(E, L, route)
    inc = rho * sum((g[j] - pref[route[j]]) ** 2 for j in range(k))
    return inc, g


def evaluate_route(route):
    key = tuple(route)
    cached = _eval_cache.get(key)
    if cached is not None:
        return cached
    if len(_eval_cache) > _CACHE_MAX:
        _eval_cache.clear()
    feas, E, L = fb(route)
    if not feas:
        res = {'feasible': False}
        _eval_cache[key] = res
        return res
    inc, sched = solve_schedule_qp(route, E, L)
    rc = routing_cost(route)
    res = {'feasible': True, 'routing': rc, 'inc': inc, 'sched': sched}
    _eval_cache[key] = res
    return res


def proxy_inc(route):
    if not route:
        return 0.0
    feas, E, L = fb(route)
    if not feas:
        return float('inf')
    g = greedy_schedule(E, L, route)
    return rho * sum((g[j] - pref[route[j]]) ** 2 for j in range(len(route)))


def compute_solution(routes):
    nonempty = [r for r in routes if r]
    if len(nonempty) > K:
        return False, None, None, None
    total = 0.0
    sched = {}
    seen = set()
    for r in nonempty:
        er = evaluate_route(r)
        if not er['feasible']:
            return False, None, None, None
        total += er['routing'] + er['inc']
        for j, cid in enumerate(r):
            if cid in seen:
                return False, None, None, None
            seen.add(cid)
            sched[cid] = er['sched'][j]
    if len(seen) != N:
        return False, None, None, None
    routes_out = [[start_id] + r + [end_id] for r in nonempty]
    return True, float(total), sched, routes_out


# -----------------------------
# Insertion
# -----------------------------
def best_insertion(c, routes, allow_new=True):
    best = None
    best_delta = float('inf')
    olds = []
    for r in routes:
        if r:
            rc_old = routing_cost(r)
            pinc_old = proxy_inc(r)
        else:
            rc_old = 0.0
            pinc_old = 0.0
        olds.append((rc_old, pinc_old))

    for ri, r in enumerate(routes):
        n = len(r)
        rc_old, pinc_old = olds[ri]
        for pos in range(n + 1):
            newr = r[:pos] + [c] + r[pos:]
            feas, E, L = fb(newr)
            if not feas:
                continue
            g = greedy_schedule(E, L, newr)
            pinc_new = rho * sum((g[j] - pref[newr[j]]) ** 2 for j in range(len(newr)))
            rc_new = routing_cost(newr)
            delta = (rc_new - rc_old) + (pinc_new - pinc_old)
            if delta < best_delta - 1e-12:
                best_delta = delta
                best = (ri, pos)

    if allow_new and len(routes) < K:
        newr = [c]
        feas, E, L = fb(newr)
        if feas:
            g = greedy_schedule(E, L, newr)
            pinc = rho * sum((g[j] - pref[newr[j]]) ** 2 for j in range(len(newr)))
            rc = routing_cost(newr)
            delta = rc + pinc
            if delta < best_delta - 1e-12:
                best_delta = delta
                best = ('new',)
    return best, best_delta


def build_initial():
    orderings = []
    orderings.append(sorted(cust_ids, key=lambda c: tw_e[c]))
    orderings.append(sorted(cust_ids, key=lambda c: pref[c]))
    orderings.append(sorted(cust_ids, key=lambda c: -demand[c]))
    rnd = list(cust_ids)
    random.shuffle(rnd)
    orderings.append(rnd)

    for order in orderings:
        routes = []
        ok = True
        for c in order:
            best, _ = best_insertion(c, routes, allow_new=(len(routes) < K))
            if best is None:
                ok = False
                break
            if best[0] == 'new':
                routes.append([c])
            else:
                ri, pos = best
                routes[ri].insert(pos, c)
        if ok:
            feas, _, _, _ = compute_solution(routes)
            if feas:
                return routes
    return None


# -----------------------------
# LNS perturbation
# -----------------------------
def perturb(routes, sched):
    op = random.choice(['random', 'worst', 'related', 'route'])
    q = random.randint(1, max_remove)
    work = [list(r) for r in routes]
    allc = [c for r in work for c in r]
    if not allc:
        return None

    removed = []
    if op == 'random':
        q = min(q, len(allc))
        removed = random.sample(allc, q)
    elif op == 'worst':
        incs = {c: rho * ((sched[c] - pref[c]) ** 2) for c in allc}
        order = sorted(allc, key=lambda c: incs[c], reverse=True)
        top = order[:max(q * 2, q + 2)]
        q = min(q, len(top))
        removed = random.sample(top, q)
    elif op == 'related':
        seed = random.choice(allc)
        order = sorted(allc, key=lambda c: dist[seed][c] + abs(sched[c] - sched[seed]))
        q = min(q, len(order))
        removed = order[:q]
    else:  # route
        nonempty = [ri for ri, r in enumerate(work) if r]
        ri = random.choice(nonempty)
        r = work[ri]
        if len(r) <= q:
            removed = list(r)
        else:
            removed = random.sample(r, q)

    rem_set = set(removed)
    newroutes = []
    for r in work:
        nr = [x for x in r if x not in rem_set]
        if nr:
            newroutes.append(nr)

    for c in removed:
        best, _ = best_insertion(c, newroutes, allow_new=(len(newroutes) < K))
        if best is None:
            return None
        if best[0] == 'new':
            newroutes.append([c])
        else:
            ri, pos = best
            newroutes[ri].insert(pos, c)
    return newroutes


# -----------------------------
# Main
# -----------------------------
def main():
    global N, K, Q, rho, start_id, end_id, e0, l0, e_end, l_end, dist
    global demand, tw_e, tw_l, serv, pref, cust_ids, max_remove

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    N = int(inst['num_customers'])
    K = int(inst['num_vehicles'])
    Q = int(inst['vehicle_capacity'])
    rho = float(inst['inconvenience_cost_function']['rho'])

    depot = inst['depot']
    depot_end = inst['depot_end']
    start_id = int(depot['id'])
    end_id = int(depot_end['id'])
    e0 = float(depot['time_window'][0])
    l0 = float(depot['time_window'][1])
    e_end = float(depot_end['time_window'][0])
    l_end = float(depot_end['time_window'][1])
    if 'planning_horizon' in inst:
        l_end = min(l_end, float(inst['planning_horizon']))

    dist = inst['distance_matrix']

    demand = {}
    tw_e = {}
    tw_l = {}
    serv = {}
    pref = {}
    cust_ids = []
    for c in inst['customers']:
        cid = int(c['id'])
        cust_ids.append(cid)
        demand[cid] = int(c['demand'])
        tw_e[cid] = float(c['time_window'][0])
        tw_l[cid] = float(c['time_window'][1])
        serv[cid] = float(c['service_time'])
        pref[cid] = float(c['preferred_time'])

    max_remove = min(N, max(2, min(20, int(0.1 * N) + 2)))

    init_gurobi_env()
    random.seed(0)

    start = time.time()
    margin = max(0.3, min(1.5, args.time_limit * 0.1))
    stop_time = start + max(0.5, args.time_limit - margin)

    # Initial solution
    initial = build_initial()
    if initial is None:
        # last-resort: try to force single-customer routes where possible
        initial = []
        rem = list(cust_ids)
        random.shuffle(rem)
        for c in rem:
            placed = False
            for r in initial:
                feas, _, _ = fb(r + [c])
                if feas:
                    r.append(c)
                    placed = True
                    break
            if not placed and len(initial) < K:
                initial.append([c])
                placed = True
            if not placed:
                # force into a new route anyway (may be infeasible)
                initial.append([c])

    feas, best_cost, best_sched, best_routes_out = compute_solution(initial)
    best = [list(r) for r in initial]
    if not feas:
        # try to salvage: output whatever
        best_cost = float('inf')

    if best_cost < float('inf') and logger:
        sol_dict = {
            "objective_value": float(best_cost),
            "routes": best_routes_out,
            "schedule": {str(c): float(t) for c, t in best_sched.items()}
        }
        logger.log_solution(best_cost, sol_dict)

    current = [list(r) for r in best]
    cur_cost = best_cost
    cur_sched = dict(best_sched) if best_sched else {}

    if best_cost < float('inf'):
        T0 = max(1e-6, 0.02 * best_cost) if best_cost > 0 else 1.0
        Tmin = max(1e-9, 1e-5 * best_cost) if best_cost > 0 else 1e-6
    else:
        T0 = 1.0
        Tmin = 1e-6
    T = T0

    no_improve = 0
    restart_limit = max(150, 3 * N)
    last_log_cost = best_cost

    while time.time() < stop_time:
        newr = perturb(current, cur_sched)
        if newr is None:
            no_improve += 1
            continue
        feas, nc, ns, _ = compute_solution(newr)
        if not feas:
            no_improve += 1
            continue

        delta = nc - cur_cost
        accept = delta < -1e-9
        if not accept:
            if T > 0:
                ratio = delta / T
                if ratio < 700:
                    if random.random() < math.exp(-ratio):
                        accept = True
                # else effectively 0

        if accept:
            current = [list(r) for r in newr]
            cur_cost = nc
            cur_sched = ns
            if nc < best_cost - 1e-9:
                best = [list(r) for r in newr]
                best_cost = nc
                best_sched = ns
                no_improve = 0
                if logger and best_cost < last_log_cost - 1e-9:
                    last_log_cost = best_cost
                    _, _, _, best_routes_out = compute_solution(best)
                    sol_dict = {
                        "objective_value": float(best_cost),
                        "routes": best_routes_out,
                        "schedule": {str(c): float(t) for c, t in best_sched.items()}
                    }
                    logger.log_solution(best_cost, sol_dict)
            else:
                no_improve += 1
        else:
            no_improve += 1

        progress = (time.time() - start) / (stop_time - start) if stop_time > start else 1.0
        progress = min(1.0, max(0.0, progress))
        if T0 > Tmin:
            T = max(Tmin, T0 * (Tmin / T0) ** progress)
        else:
            T = Tmin

        if no_improve > restart_limit:
            current = [list(r) for r in best]
            cur_cost = best_cost
            cur_sched = dict(best_sched)
            no_improve = 0

    # Finalize
    feas, final_cost, final_sched, final_routes_out = compute_solution(best)
    if not feas:
        # fallback: try initial again
        feas, final_cost, final_sched, final_routes_out = compute_solution(initial)
    if not feas:
        # produce a trivial output to satisfy schema
        final_routes_out = []
        final_sched = {}
        final_cost = 0.0
        for c in cust_ids:
            final_sched[str(c)] = float(pref[c])

    solution = {
        "objective_value": float(final_cost),
        "routes": final_routes_out,
        "schedule": {str(c): float(t) for c, t in final_sched.items()}
    }

    if logger:
        logger.log_solution(final_cost, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()