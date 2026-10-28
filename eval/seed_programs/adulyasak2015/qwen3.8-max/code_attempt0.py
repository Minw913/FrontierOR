import argparse
import json
import math
import time
import itertools
import bisect

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI = True
except Exception:
    gp = None
    GRB = None
    GUROBI = False

START_TIME = 0.0
TIME_LIMIT = 0.0
ROUND = 8

N = 0
T = 0
M = 0
Q = 0.0
CAP_PROD = 0.0
SETUP_COST = 0.0
PROD_COST = 0.0

HOLD = []
CAP_INV = []
INIT_INV = []
PENALTY = []
COST_MAT = []

DEMANDS = []
PROBS = []
S = 0
EXP_DEMAND = []
FUTURE_ALL = []
MAX_FUTURE = []


def remaining():
    return max(0.0, TIME_LIMIT - (time.time() - START_TIME))


def pad_float_list(lst, length, default=0.0):
    if not isinstance(lst, (list, tuple)):
        lst = []
    out = []
    for i in range(length):
        try:
            val = float(lst[i]) if i < len(lst) else default
        except Exception:
            val = default
        out.append(val)
    return out


def compute_future_from_dem(dem):
    n = len(dem)
    if n == 0:
        return []
    tlen = len(dem[0]) if dem else 0
    fut = [[0.0] * tlen for _ in range(n)]
    for i in range(n):
        run = 0.0
        for t in range(tlen - 1, -1, -1):
            run += float(dem[i][t])
            fut[i][t] = run
    return fut


class Route:
    __slots__ = ("seq", "custs", "cost")

    def __init__(self, seq, cost):
        self.seq = tuple(int(c) for c in seq)
        self.custs = tuple(sorted(self.seq))
        self.cost = float(cost)


def route_cost(seq):
    if not seq:
        return 0.0
    c = float(COST_MAT[0][int(seq[0]) + 1])
    for idx in range(len(seq) - 1):
        c += float(COST_MAT[int(seq[idx]) + 1][int(seq[idx + 1]) + 1])
    c += float(COST_MAT[int(seq[-1]) + 1][0])
    return c


def two_opt(seq):
    seq = list(seq)
    if len(seq) < 3:
        return seq
    improved = True
    passes = 0
    while improved and passes < 2:
        improved = False
        passes += 1
        current_cost = route_cost(seq)
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                new_seq = seq[:i + 1] + seq[i + 1:j + 1][::-1] + seq[j + 1:]
                new_cost = route_cost(new_seq)
                if new_cost < current_cost - 1e-10:
                    seq = new_seq
                    current_cost = new_cost
                    improved = True
    return seq


def tsp_seq_for_set(custs):
    custs = list(custs)
    k = len(custs)
    if k == 0:
        return (), 0.0
    if k == 1:
        seq = (custs[0],)
        return seq, route_cost(seq)

    if k <= 5:
        best_seq = None
        best_cost = float("inf")
        for perm in itertools.permutations(custs):
            c = route_cost(perm)
            if c < best_cost:
                best_cost = c
                best_seq = perm
        return tuple(best_seq), float(best_cost)

    starts = sorted(custs, key=lambda i: COST_MAT[0][i + 1])[:min(3, k)]
    best_seq = None
    best_cost = float("inf")
    for st in starts:
        unvisited = set(custs)
        unvisited.remove(st)
        seq = [st]
        cur = st
        while unvisited:
            nxt = min(unvisited, key=lambda j: COST_MAT[cur + 1][j + 1])
            seq.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        seq = two_opt(seq)
        c = route_cost(seq)
        if c < best_cost:
            best_cost = c
            best_seq = seq
    return tuple(best_seq), float(best_cost)


def add_route_to_dict(routes_dict, custs):
    cleaned = []
    seen = set()
    for c in custs:
        ci = int(c)
        if 0 <= ci < N and ci not in seen:
            cleaned.append(ci)
            seen.add(ci)
    if not cleaned:
        return
    key = tuple(sorted(cleaned))
    seq, cost = tsp_seq_for_set(key)
    if key not in routes_dict or cost < routes_dict[key][0] - 1e-10:
        routes_dict[key] = (cost, tuple(seq))


def generate_route_pool(max_routes, max_len):
    if max_routes <= 0 or N == 0 or M == 0:
        return []

    priorities = []
    for i in range(N):
        dem_sum = sum(EXP_DEMAND[i]) if T > 0 else 0.0
        priorities.append(float(PENALTY[i]) * dem_sum + dem_sum)
    order = sorted(range(N), key=lambda i: -priorities[i])

    routes_dict = {}

    single_limit = min(N, max(max_routes, 50))
    for i in order[:single_limit]:
        add_route_to_dict(routes_dict, [i])

    pair_customers = order[:min(N, max(20, max_routes // 2))]
    pair_candidates = []
    for idx, a in enumerate(pair_customers):
        for b in pair_customers[idx + 1:]:
            saving = COST_MAT[0][a + 1] + COST_MAT[0][b + 1] - COST_MAT[a + 1][b + 1]
            pair_candidates.append((saving, a, b))
    pair_candidates.sort(key=lambda x: x[0], reverse=True)

    for _, a, b in pair_candidates[:max(50, max_routes * 2)]:
        add_route_to_dict(routes_dict, [a, b])

    cluster_pairs = pair_candidates[:min(len(pair_candidates), max_routes * 5)]
    for ml in range(3, min(int(max_len), 6) + 1):
        clusters = [{i} for i in pair_customers]
        for _, a, b in cluster_pairs:
            ca = -1
            cb = -1
            for idx, cl in enumerate(clusters):
                if a in cl:
                    ca = idx
                if b in cl:
                    cb = idx
            if ca < 0 or cb < 0 or ca == cb:
                continue
            if len(clusters[ca]) + len(clusters[cb]) <= ml:
                new_cl = clusters[ca] | clusters[cb]
                add_route_to_dict(routes_dict, list(new_cl))
                clusters[ca] = new_cl
                clusters[cb] = set()
        for cl in clusters:
            if 2 <= len(cl) <= ml:
                add_route_to_dict(routes_dict, list(cl))

    candidate_customers = order[:min(N, max(50, max_routes))]
    seeds = order[:min(N, max(10, max_routes // 4))]
    for st in seeds:
        unvisited = set(candidate_customers)
        if st in unvisited:
            unvisited.remove(st)
        seq = [st]
        cur = st
        for _ in range(min(int(max_len), N) - 1):
            if not unvisited:
                break
            nxt = min(unvisited, key=lambda j: COST_MAT[cur + 1][j + 1])
            seq.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
            if len(seq) >= 2:
                add_route_to_dict(routes_dict, seq[:])

    routes = [Route(seq, cost) for cost, seq in routes_dict.values()]
    if len(routes) <= max_routes:
        return routes

    singletons = [r for r in routes if len(r.seq) == 1]
    multis = [r for r in routes if len(r.seq) > 1]
    singletons.sort(key=lambda r: -priorities[r.seq[0]])

    if len(singletons) <= max_routes:
        selected = singletons[:]
    else:
        keep_s = min(len(singletons), max(5, max_routes // 2))
        selected = singletons[:keep_s]

    remaining_slots = max(0, max_routes - len(selected))
    if remaining_slots > 0:
        def route_score(r):
            singleton_sum = sum(2.0 * COST_MAT[0][i + 1] for i in r.custs)
            savings = max(0.0, singleton_sum - r.cost)
            prio = sum(priorities[i] for i in r.custs)
            return (savings, prio, -r.cost)

        multis.sort(key=route_score, reverse=True)
        selected.extend(multis[:remaining_slots])

    return selected[:max_routes]


def sanitize_assignments(assignments):
    clean = []
    for t in range(T):
        seen = set()
        new = []
        if t < len(assignments):
            for seq in assignments[t]:
                if not seq:
                    continue
                valid = []
                local = set()
                for c in seq:
                    ci = int(c)
                    if 0 <= ci < N and ci not in local:
                        valid.append(ci)
                        local.add(ci)
                if not valid:
                    continue
                if any(c in seen for c in valid):
                    continue
                if len(new) >= M:
                    break
                new.append(tuple(valid))
                seen.update(valid)
        clean.append(new)
    return clean


def ensure_y(y_vals):
    if y_vals is None:
        return [0] * T
    y_vals = list(y_vals)
    if len(y_vals) < T:
        y_vals.extend([0] * (T - len(y_vals)))
    return [1 if y_vals[t] > 0.5 else 0 for t in range(T)]


def build_first_stage(y_vals, assignments):
    y_vals = ensure_y(y_vals)
    assignments = sanitize_assignments(assignments)

    y_dict = {str(t + 1): int(y_vals[t]) for t in range(T)}
    z_dict = {}
    x_dict = {}
    trans_cost = 0.0

    setup_cost = sum(SETUP_COST for t in range(T) if y_vals[t] > 0.5)

    for t, assign in enumerate(assignments):
        for k, seq in enumerate(assign):
            if not seq:
                continue
            z_dict[f"0_{k + 1}_{t + 1}"] = 1
            nodes = [0] + [int(c) + 1 for c in seq] + [0]
            for idx in range(len(nodes) - 1):
                a = nodes[idx]
                b = nodes[idx + 1]
                if a == b:
                    continue
                trans_cost += float(COST_MAT[a][b])
                if a < b:
                    key = f"{a}_{b}_{k + 1}_{t + 1}"
                else:
                    key = f"{b}_{a}_{k + 1}_{t + 1}"
                x_dict[key] = x_dict.get(key, 0) + 1
            for c in seq:
                z_dict[f"{int(c) + 1}_{k + 1}_{t + 1}"] = 1

    fixed_cost = float(setup_cost) + float(trans_cost)
    return y_dict, z_dict, x_dict, fixed_cost


def add_first_stage_mv(mv, y_dict, z_dict, x_dict):
    for key, val in y_dict.items():
        if val:
            mv[f"y_{key}"] = 1
    for key, val in z_dict.items():
        if val:
            mv[f"z_{key}"] = 1
    for key, val in x_dict.items():
        if val:
            mv[f"x_{key}"] = int(val)


def make_solution(objective_value, model_variables, y_dict, z_dict, x_dict):
    return {
        "objective_value": float(objective_value),
        "model_variables": model_variables,
        "y": y_dict,
        "z": z_dict,
        "x": x_dict,
    }


def evaluate_routes_greedy(assignments, y_vals):
    assignments = sanitize_assignments(assignments)
    y_vals = ensure_y(y_vals)

    y_dict, z_dict, x_dict, fixed_cost = build_first_stage(y_vals, assignments)
    mv = {}
    add_first_stage_mv(mv, y_dict, z_dict, x_dict)

    var_cost = 0.0

    visit_periods = [[] for _ in range(N)]
    for t, assign in enumerate(assignments):
        for seq in assign:
            for i in seq:
                visit_periods[i].append(t)

    for s in range(S):
        prob = PROBS[s]
        plant_inv = float(INIT_INV[0])
        cust_inv = [float(INIT_INV[i + 1]) for i in range(N)]

        for t in range(T):
            desired = {}
            current = []
            for seq in assignments[t]:
                for i in seq:
                    current.append(i)

            for i in current:
                lst = visit_periods[i]
                pos = bisect.bisect_right(lst, t)
                nxt = lst[pos] if pos < len(lst) else T

                cover_dem = 0.0
                for tau in range(t, min(nxt, T)):
                    cover_dem += DEMANDS[s][i][tau]

                fut_dem = FUTURE_ALL[s][i][t] if s < len(FUTURE_ALL) else 0.0
                max_storage = max(0.0, float(CAP_INV[i + 1]) - cust_inv[i])
                target = max(0.0, cover_dem - cust_inv[i])
                desired[i] = min(target, max_storage, fut_dem, float(Q))

            alloc = []
            for k, seq in enumerate(assignments[t]):
                cap = float(Q)
                for i in sorted(seq, key=lambda x: (-PENALTY[x], -desired.get(x, 0.0))):
                    amt = min(desired.get(i, 0.0), cap)
                    if amt > 1e-7:
                        alloc.append([float(PENALTY[i]), float(amt), k, i])
                        cap -= amt
                        desired[i] = 0.0

            total_alloc = sum(a[1] for a in alloc)
            max_prod = float(CAP_PROD) if y_vals[t] else 0.0
            max_ship = min(total_alloc, plant_inv + max_prod)

            if total_alloc > max_ship + 1e-8 and total_alloc > 0.0:
                scale = max(0.0, max_ship / total_alloc)
                for a in alloc:
                    a[1] *= scale
                total_alloc = sum(a[1] for a in alloc)

            p_val = min(max_prod, max(0.0, total_alloc - plant_inv))
            cap_avail = plant_inv + p_val

            if total_alloc > cap_avail + 1e-8 and total_alloc > 0.0:
                scale = max(0.0, cap_avail / total_alloc)
                for a in alloc:
                    a[1] *= scale
                total_alloc = sum(a[1] for a in alloc)

            plant_inv = plant_inv + p_val - total_alloc
            if plant_inv < 0.0:
                plant_inv = 0.0
            if plant_inv > CAP_INV[0]:
                plant_inv = float(CAP_INV[0])

            if p_val > 1e-6:
                mv[f"p_{t + 1}_{s + 1}"] = round(p_val, ROUND)
            if plant_inv > 1e-6:
                mv[f"I_0_{t + 1}_{s + 1}"] = round(plant_inv, ROUND)

            var_cost += prob * (PROD_COST * p_val + HOLD[0] * plant_inv)

            q_by_i = {}
            q_out = {}
            for _, amt, k, i in alloc:
                if amt <= 1e-7:
                    continue
                q_by_i[i] = q_by_i.get(i, 0.0) + amt
                q_out[(i, k)] = q_out.get((i, k), 0.0) + amt

            for (i, k), amt in q_out.items():
                if amt > 1e-6:
                    mv[f"q_{i + 1}_{k + 1}_{t + 1}_{s + 1}"] = round(amt, ROUND)

            for i in range(N):
                qi = q_by_i.get(i, 0.0)
                avail = cust_inv[i] + qi
                d = DEMANDS[s][i][t]

                if avail >= d:
                    e_val = 0.0
                    new_inv = avail - d
                else:
                    e_val = d - avail
                    new_inv = 0.0

                if new_inv > CAP_INV[i + 1]:
                    new_inv = float(CAP_INV[i + 1])

                cust_inv[i] = new_inv

                if new_inv > 1e-6:
                    mv[f"I_{i + 1}_{t + 1}_{s + 1}"] = round(new_inv, ROUND)
                if e_val > 1e-6:
                    mv[f"e_{i + 1}_{t + 1}_{s + 1}"] = round(e_val, ROUND)

                var_cost += prob * (HOLD[i + 1] * new_inv + PENALTY[i] * e_val)

    objective = fixed_cost + var_cost
    return make_solution(objective, mv, y_dict, z_dict, x_dict)


def estimate_lp_variables(assignments, scen_count):
    entries = sum(len(seq) for assign in assignments for seq in assign)
    return scen_count * (T * (2 + 2 * N) + entries)


def evaluate_routes_lp(assignments, y_vals, time_budget):
    if not GUROBI or time_budget <= 0.0:
        return None

    assignments = sanitize_assignments(assignments)
    y_vals = ensure_y(y_vals)
    y_dict, z_dict, x_dict, fixed_cost = build_first_stage(y_vals, assignments)

    mv = {}
    add_first_stage_mv(mv, y_dict, z_dict, x_dict)

    model = None
    try:
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", 0)
        model.setParam("Threads", 1)
        model.setParam("NumericFocus", 0)
        model.setParam("TimeLimit", max(0.1, float(time_budget)))

        p = {}
        I0 = {}
        I = {}
        e = {}
        q_vars = {}
        q_at = [[[] for _ in range(T)] for _ in range(S)]
        q_cust = {}

        for s in range(S):
            for t in range(T):
                ub_p = float(CAP_PROD) if y_vals[t] else 0.0
                p[s, t] = model.addVar(lb=0.0, ub=ub_p)
                I0[s, t] = model.addVar(lb=0.0, ub=float(CAP_INV[0]))
                for i in range(N):
                    I[s, i, t] = model.addVar(lb=0.0, ub=float(CAP_INV[i + 1]))
                    e[s, i, t] = model.addVar(lb=0.0)

        for s in range(S):
            for t in range(T):
                for k, seq in enumerate(assignments[t]):
                    for i in seq:
                        ub = min(float(Q), float(CAP_INV[i + 1]), float(MAX_FUTURE[i][t]))
                        if ub > 1e-8:
                            var = model.addVar(lb=0.0, ub=ub)
                            q_vars[s, t, k, i] = var
                            q_at[s][t].append(var)
                            q_cust.setdefault((s, i, t), []).append(var)

        model.update()

        for s in range(S):
            for t in range(T):
                Iprev0 = float(INIT_INV[0]) if t == 0 else I0[s, t - 1]
                ship = gp.quicksum(q_at[s][t]) if q_at[s][t] else 0.0
                model.addConstr(Iprev0 + p[s, t] == ship + I0[s, t])

                for k, seq in enumerate(assignments[t]):
                    qv = [q_vars[s, t, k, i] for i in seq if (s, t, k, i) in q_vars]
                    if qv:
                        model.addConstr(gp.quicksum(qv) <= float(Q))

                for i in range(N):
                    qv = q_cust.get((s, i, t), [])
                    Iprev = float(INIT_INV[i + 1]) if t == 0 else I[s, i, t - 1]
                    if qv:
                        sum_q = gp.quicksum(qv)
                        model.addConstr(Iprev + sum_q + e[s, i, t] == DEMANDS[s][i][t] + I[s, i, t])
                        model.addConstr(Iprev + sum_q <= float(CAP_INV[i + 1]))
                        model.addConstr(sum_q <= FUTURE_ALL[s][i][t])
                    else:
                        model.addConstr(Iprev + e[s, i, t] == DEMANDS[s][i][t] + I[s, i, t])

        obj = gp.LinExpr()
        for s in range(S):
            w = PROBS[s]
            if w == 0.0:
                continue
            for t in range(T):
                obj += w * (PROD_COST * p[s, t] + HOLD[0] * I0[s, t])
                for i in range(N):
                    obj += w * (HOLD[i + 1] * I[s, i, t] + PENALTY[i] * e[s, i, t])

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        if model.SolCount == 0:
            return None

        var_cost = 0.0
        for s in range(S):
            w = PROBS[s]
            for t in range(T):
                pval = p[s, t].X
                I0val = I0[s, t].X

                if pval > 1e-6:
                    mv[f"p_{t + 1}_{s + 1}"] = round(pval, ROUND)
                if I0val > 1e-6:
                    mv[f"I_0_{t + 1}_{s + 1}"] = round(I0val, ROUND)

                var_cost += w * (PROD_COST * pval + HOLD[0] * I0val)

                for i in range(N):
                    Ival = I[s, i, t].X
                    eval_ = e[s, i, t].X

                    if Ival > 1e-6:
                        mv[f"I_{i + 1}_{t + 1}_{s + 1}"] = round(Ival, ROUND)
                    if eval_ > 1e-6:
                        mv[f"e_{i + 1}_{t + 1}_{s + 1}"] = round(eval_, ROUND)

                    var_cost += w * (HOLD[i + 1] * Ival + PENALTY[i] * eval_)

                for k, seq in enumerate(assignments[t]):
                    for i in seq:
                        var = q_vars.get((s, t, k, i))
                        if var is None:
                            continue
                        qval = var.X
                        if qval > 1e-6:
                            mv[f"q_{i + 1}_{k + 1}_{t + 1}_{s + 1}"] = round(qval, ROUND)

        objective = fixed_cost + var_cost
        return make_solution(objective, mv, y_dict, z_dict, x_dict)

    except Exception:
        return None
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


def select_opt_scenarios(K):
    if K <= 0:
        return [], []

    exp = [row[:] for row in EXP_DEMAND]
    if K == 1:
        return [exp], [1.0]

    dems = [exp]
    weights = [1.0]

    if S > 0:
        totals = []
        for s in range(S):
            total = 0.0
            for i in range(N):
                total += sum(DEMANDS[s][i])
            totals.append(total)

        idxs = []
        idxs.append(max(range(S), key=lambda x: PROBS[x]))
        idxs.append(max(range(S), key=lambda x: totals[x]))
        idxs.append(min(range(S), key=lambda x: totals[x]))
        idxs.extend(sorted(range(S), key=lambda x: -PROBS[x]))

        seen = set()
        for idx in idxs:
            if idx in seen:
                continue
            seen.add(idx)
            dems.append([row[:] for row in DEMANDS[idx]])
            weights.append(PROBS[idx] if PROBS[idx] > 0.0 else 1.0 / max(1, S))
            if len(dems) >= K:
                break

    total_w = sum(weights)
    if total_w <= 0.0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / total_w for w in weights]

    return dems[:K], weights[:K]


def solve_route_mip(pool, scen_demands, scen_weights, time_budget):
    if not GUROBI or time_budget <= 0.0:
        return None

    K = len(scen_demands)
    R = len(pool)
    if R == 0 or K == 0 or T == 0 or M == 0:
        return None

    model = None
    try:
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", 0)
        model.setParam("Threads", 1)
        model.setParam("NumericFocus", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("TimeLimit", max(0.1, float(time_budget)))

        use = {}
        for r in range(R):
            for t in range(T):
                use[r, t] = model.addVar(vtype=GRB.BINARY)

        y = {t: model.addVar(vtype=GRB.BINARY) for t in range(T)}

        p = {}
        I0 = {}
        I = {}
        e = {}

        for a in range(K):
            for t in range(T):
                p[a, t] = model.addVar(lb=0.0, ub=float(CAP_PROD))
                I0[a, t] = model.addVar(lb=0.0, ub=float(CAP_INV[0]))
                for i in range(N):
                    I[a, i, t] = model.addVar(lb=0.0, ub=float(CAP_INV[i + 1]))
                    e[a, i, t] = model.addVar(lb=0.0)

        future_max = [[0.0] * T for _ in range(N)]
        scen_futures = []
        for a, dem in enumerate(scen_demands):
            fut = compute_future_from_dem(dem)
            scen_futures.append(fut)
            for i in range(N):
                for t in range(T):
                    if fut[i][t] > future_max[i][t]:
                        future_max[i][t] = fut[i][t]

        M_i_t = [[min(float(Q), float(CAP_INV[i + 1]), future_max[i][t]) for t in range(T)] for i in range(N)]

        q = {}
        q_at = [[[] for _ in range(T)] for _ in range(K)]
        q_route = {}
        q_cust = {}

        for a in range(K):
            for t in range(T):
                for r, route in enumerate(pool):
                    qvars_route = []
                    for i in route.custs:
                        if M_i_t[i][t] > 1e-8:
                            var = model.addVar(lb=0.0, ub=M_i_t[i][t])
                            q[a, r, i, t] = var
                            q_at[a][t].append(var)
                            qvars_route.append(var)
                            q_cust.setdefault((a, i, t), []).append(var)
                    if qvars_route:
                        q_route[a, r, t] = qvars_route

        model.update()

        for t in range(T):
            model.addConstr(gp.quicksum(use[r, t] for r in range(R)) <= M)

        routes_by_cust = [[] for _ in range(N)]
        for r, route in enumerate(pool):
            for i in route.custs:
                routes_by_cust[i].append(r)

        for i in range(N):
            if routes_by_cust[i]:
                for t in range(T):
                    model.addConstr(gp.quicksum(use[r, t] for r in routes_by_cust[i]) <= 1)

        for a in range(K):
            dem = scen_demands[a]
            fut = scen_futures[a]
            for t in range(T):
                Iprev0 = float(INIT_INV[0]) if t == 0 else I0[a, t - 1]
                ship = gp.quicksum(q_at[a][t]) if q_at[a][t] else 0.0
                model.addConstr(Iprev0 + p[a, t] == ship + I0[a, t])
                model.addConstr(p[a, t] <= float(CAP_PROD) * y[t])

                for r in range(R):
                    qvars = q_route.get((a, r, t))
                    if qvars:
                        model.addConstr(gp.quicksum(qvars) <= float(Q) * use[r, t])

                for i in range(N):
                    qvars = q_cust.get((a, i, t), [])
                    Iprev = float(INIT_INV[i + 1]) if t == 0 else I[a, i, t - 1]
                    if qvars:
                        sum_q = gp.quicksum(qvars)
                        model.addConstr(Iprev + sum_q + e[a, i, t] == dem[i][t] + I[a, i, t])
                        model.addConstr(Iprev + sum_q <= float(CAP_INV[i + 1]))
                        model.addConstr(sum_q <= fut[i][t])
                    else:
                        model.addConstr(Iprev + e[a, i, t] == dem[i][t] + I[a, i, t])

        obj = gp.LinExpr()
        for t in range(T):
            obj += SETUP_COST * y[t]
        for r, route in enumerate(pool):
            for t in range(T):
                obj += route.cost * use[r, t]

        for a in range(K):
            w = scen_weights[a]
            for t in range(T):
                obj += w * (PROD_COST * p[a, t] + HOLD[0] * I0[a, t])
                for i in range(N):
                    obj += w * (HOLD[i + 1] * I[a, i, t] + PENALTY[i] * e[a, i, t])

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        if model.SolCount == 0:
            return None

        y_vals = [1 if y[t].X > 0.5 else 0 for t in range(T)]
        assignments = []

        for t in range(T):
            sel = []
            for r, route in enumerate(pool):
                try:
                    val = use[r, t].X
                except Exception:
                    val = 0.0
                if val > 0.5:
                    sel.append(route)

            sel.sort(key=lambda rt: (len(rt.seq), rt.cost))
            chosen = []
            covered = set()
            for rt in sel:
                if len(chosen) >= M:
                    break
                if any(c in covered for c in rt.custs):
                    continue
                chosen.append(tuple(rt.seq))
                covered.update(rt.custs)
            assignments.append(chosen)

        return sanitize_assignments(assignments), y_vals

    except Exception:
        return None
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


def optimize_y_fixed_routes(assignments, initial_y, time_budget):
    initial_y = ensure_y(initial_y)

    if not GUROBI or time_budget <= 1.0 or T == 0:
        return initial_y

    assignments = sanitize_assignments(assignments)
    if not any(assignments[t] for t in range(T)):
        return [0] * T

    total_entries = sum(len(seq) for assign in assignments for seq in assign)
    max_cont = 80000 if time_budget > 20 else 30000
    denom = max(1, T * (total_entries + 2 * N + 2))
    K = max(1, int(max_cont // denom))
    K = min(K, 5 if time_budget > 20 else 2, S + 1)

    dems, weights = select_opt_scenarios(K)
    K = len(dems)
    if K == 0:
        return initial_y

    model = None
    try:
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", 0)
        model.setParam("Threads", 1)
        model.setParam("NumericFocus", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("TimeLimit", max(0.1, float(time_budget)))

        y = {t: model.addVar(vtype=GRB.BINARY) for t in range(T)}

        p = {}
        I0 = {}
        I = {}
        e = {}
        q_vars = {}
        q_at = [[[] for _ in range(T)] for _ in range(K)]
        q_cust = {}

        future_max = [[0.0] * T for _ in range(N)]
        futures = []
        for a, dem in enumerate(dems):
            fut = compute_future_from_dem(dem)
            futures.append(fut)
            for i in range(N):
                for t in range(T):
                    if fut[i][t] > future_max[i][t]:
                        future_max[i][t] = fut[i][t]

        for a in range(K):
            for t in range(T):
                p[a, t] = model.addVar(lb=0.0, ub=float(CAP_PROD))
                I0[a, t] = model.addVar(lb=0.0, ub=float(CAP_INV[0]))
                for i in range(N):
                    I[a, i, t] = model.addVar(lb=0.0, ub=float(CAP_INV[i + 1]))
                    e[a, i, t] = model.addVar(lb=0.0)

        for a in range(K):
            for t in range(T):
                for k, seq in enumerate(assignments[t]):
                    for i in seq:
                        ub = min(float(Q), float(CAP_INV[i + 1]), future_max[i][t])
                        if ub > 1e-8:
                            var = model.addVar(lb=0.0, ub=ub)
                            q_vars[a, t, k, i] = var
                            q_at[a][t].append(var)
                            q_cust.setdefault((a, i, t), []).append(var)

        model.update()

        for a in range(K):
            fut = futures[a]
            for t in range(T):
                Iprev0 = float(INIT_INV[0]) if t == 0 else I0[a, t - 1]
                ship = gp.quicksum(q_at[a][t]) if q_at[a][t] else 0.0
                model.addConstr(Iprev0 + p[a, t] == ship + I0[a, t])
                model.addConstr(p[a, t] <= float(CAP_PROD) * y[t])

                for k, seq in enumerate(assignments[t]):
                    qv = [q_vars[a, t, k, i] for i in seq if (a, t, k, i) in q_vars]
                    if qv:
                        model.addConstr(gp.quicksum(qv) <= float(Q))

                for i in range(N):
                    qv = q_cust.get((a, i, t), [])
                    Iprev = float(INIT_INV[i + 1]) if t == 0 else I[a, i, t - 1]
                    if qv:
                        sum_q = gp.quicksum(qv)
                        model.addConstr(Iprev + sum_q + e[a, i, t] == dems[a][i][t] + I[a, i, t])
                        model.addConstr(Iprev + sum_q <= float(CAP_INV[i + 1]))
                        model.addConstr(sum_q <= fut[i][t])
                    else:
                        model.addConstr(Iprev + e[a, i, t] == dems[a][i][t] + I[a, i, t])

        obj = gp.LinExpr()
        for t in range(T):
            obj += SETUP_COST * y[t]

        for a in range(K):
            w = weights[a]
            for t in range(T):
                obj += w * (PROD_COST * p[a, t] + HOLD[0] * I0[a, t])
                for i in range(N):
                    obj += w * (HOLD[i + 1] * I[a, i, t] + PENALTY[i] * e[a, i, t])

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        if model.SolCount == 0:
            return initial_y

        return [1 if y[t].X > 0.5 else 0 for t in range(T)]

    except Exception:
        return initial_y
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


def build_heuristic_routes(pool):
    assignments = [[] for _ in range(T)]
    if not pool or M == 0 or N == 0:
        return assignments

    cust_inv = [float(INIT_INV[i + 1]) for i in range(N)]
    plant_inv = float(INIT_INV[0])

    for t in range(T):
        scores = [0.0] * N
        for i in range(N):
            shortage = max(0.0, EXP_DEMAND[i][t] - cust_inv[i])
            scores[i] = float(PENALTY[i]) * shortage + shortage

        selected = []
        covered = set()

        while len(selected) < M:
            best = None
            best_ratio = 1e-12
            for r in pool:
                if any(c in covered for c in r.custs):
                    continue
                score = sum(scores[i] for i in r.custs)
                if score <= 1e-7:
                    continue
                ratio = score / (r.cost + 1e-6)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = r
            if best is None:
                break
            selected.append(best)
            covered.update(best.custs)

        assignments[t] = [tuple(r.seq) for r in selected]

        total_q = 0.0
        for seq in assignments[t]:
            cap = float(Q)
            for i in sorted(seq, key=lambda x: -scores[x]):
                need = max(0.0, EXP_DEMAND[i][t] - cust_inv[i])
                qi = min(need, cap)
                cap -= qi
                total_q += qi
                cust_inv[i] += qi

        if plant_inv + float(CAP_PROD) >= total_q:
            prod = min(float(CAP_PROD), max(0.0, total_q - plant_inv))
            plant_inv = plant_inv + prod - total_q
            if plant_inv < 0.0:
                plant_inv = 0.0
            if plant_inv > CAP_INV[0]:
                plant_inv = float(CAP_INV[0])
        else:
            plant_inv = 0.0

        for i in range(N):
            d = EXP_DEMAND[i][t]
            if cust_inv[i] >= d:
                cust_inv[i] -= d
            else:
                cust_inv[i] = 0.0

    return sanitize_assignments(assignments)


def evaluate_solution(assignments, y_vals, time_budget):
    if assignments is None:
        return None

    y_vals = ensure_y(y_vals)

    if GUROBI and time_budget >= 3.0 and S > 0:
        est = estimate_lp_variables(assignments, S)
        if est <= 100000:
            sol = evaluate_routes_lp(assignments, y_vals, time_budget)
            if sol is not None:
                return sol

    return evaluate_routes_greedy(assignments, y_vals)


def main():
    global START_TIME, TIME_LIMIT
    global N, T, M, Q, CAP_PROD, SETUP_COST, PROD_COST
    global HOLD, CAP_INV, INIT_INV, PENALTY, COST_MAT
    global DEMANDS, PROBS, S, EXP_DEMAND, FUTURE_ALL, MAX_FUTURE

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", "--instance", dest="instance_path", required=True)
    parser.add_argument("--solution_path", "--solution", dest="solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=300.0)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    START_TIME = time.time()
    TIME_LIMIT = max(0.0, float(args.time_limit))

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = max(0, int(float(data.get("n", 0))))
    T = max(0, int(float(data.get("T", 0))))
    M = max(0, int(float(data.get("m", 0))))
    Q = max(0.0, float(data.get("Q", 0.0)))
    CAP_PROD = max(0.0, float(data.get("C", 0.0)))
    SETUP_COST = float(data.get("f", 0.0))
    PROD_COST = float(data.get("u", 0.0))

    HOLD = pad_float_list(data.get("h", []), N + 1, 0.0)
    CAP_INV = pad_float_list(data.get("L", []), N + 1, 0.0)
    INIT_INV = pad_float_list(data.get("I0", []), N + 1, 0.0)

    CAP_INV = [max(0.0, float(x)) for x in CAP_INV]
    INIT_INV = [max(0.0, min(float(INIT_INV[i]), CAP_INV[i])) for i in range(N + 1)]

    if "sigma" in data:
        PENALTY = pad_float_list(data.get("sigma", []), N, 0.0)
    else:
        alpha_hat = float(data.get("alpha_hat", 0.0))
        PENALTY = [alpha_hat * PROD_COST] * N

    PENALTY = [max(0.0, float(x)) for x in PENALTY]

    costs = data.get("transportation_costs")
    valid_costs = (
        isinstance(costs, list)
        and len(costs) == N + 1
        and all(isinstance(row, list) and len(row) == N + 1 for row in costs)
    )

    if valid_costs:
        COST_MAT = [[float(costs[i][j]) for j in range(N + 1)] for i in range(N + 1)]
    else:
        coords = data.get("coordinates")
        if isinstance(coords, list) and len(coords) >= N + 1:
            COST_MAT = []
            for i in range(N + 1):
                row = []
                xi, yi = float(coords[i][0]), float(coords[i][1])
                for j in range(N + 1):
                    xj, yj = float(coords[j][0]), float(coords[j][1])
                    row.append(math.hypot(xi - xj, yi - yj))
                COST_MAT.append(row)
        else:
            COST_MAT = [[0.0] * (N + 1) for _ in range(N + 1)]

    raw_scenarios = data.get("demand_scenarios")
    if not raw_scenarios:
        d_bar = data.get("d_bar", [[0.0] * T for _ in range(N)])
        raw_scenarios = [d_bar]
        raw_probs = [1.0]
    else:
        raw_probs = data.get("scenario_probabilities", [1.0 / len(raw_scenarios)] * len(raw_scenarios))

    S = len(raw_scenarios)
    if S == 0:
        S = 1
        raw_scenarios = [[[0.0] * T for _ in range(N)]]
        raw_probs = [1.0]

    DEMANDS = []
    for s in range(S):
        raw_s = raw_scenarios[s] if s < len(raw_scenarios) else []
        sc = []
        for i in range(N):
            raw_row = raw_s[i] if i < len(raw_s) else []
            sc.append([max(0.0, float(raw_row[t] if t < len(raw_row) else 0.0)) for t in range(T)])
        DEMANDS.append(sc)

    PROBS = []
    for s in range(S):
        try:
            val = float(raw_probs[s]) if s < len(raw_probs) else 0.0
        except Exception:
            val = 0.0
        PROBS.append(max(0.0, val))

    total_prob = sum(PROBS)
    if total_prob <= 0.0:
        PROBS = [1.0 / S] * S
    else:
        PROBS = [p / total_prob for p in PROBS]

    EXP_DEMAND = [[0.0] * T for _ in range(N)]
    for s in range(S):
        w = PROBS[s]
        if w == 0.0:
            continue
        for i in range(N):
            for t in range(T):
                EXP_DEMAND[i][t] += w * DEMANDS[s][i][t]

    FUTURE_ALL = []
    MAX_FUTURE = [[0.0] * T for _ in range(N)]
    for s in range(S):
        fut = compute_future_from_dem(DEMANDS[s])
        FUTURE_ALL.append(fut)
        for i in range(N):
            for t in range(T):
                if fut[i][t] > MAX_FUTURE[i][t]:
                    MAX_FUTURE[i][t] = fut[i][t]

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    best_sol = evaluate_routes_greedy([[] for _ in range(T)], [0] * T)
    best_obj = best_sol["objective_value"]
    baseline_obj = best_obj

    if logger is not None:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    def consider(sol):
        nonlocal best_obj, best_sol
        if sol is None:
            return
        obj = sol.get("objective_value", float("inf"))
        if obj < best_obj - 1e-7:
            best_obj = obj
            best_sol = sol
            if logger is not None:
                try:
                    logger.log_solution(obj, sol)
                except Exception:
                    pass

    if N > 0 and M > 0 and T > 0 and GUROBI and remaining() > 15.0:
        rem = remaining()
        max_binaries = 3500 if rem > 180.0 else (1800 if rem > 60.0 else 800)
        R_limit = max(1, int(max_binaries // max(1, T)))
        R_limit = max(R_limit, min(N, 80))
        R_limit = min(R_limit, 450)
        max_len = min(N, 6)

        pool = generate_route_pool(R_limit, max_len)

        if pool:
            entries = sum(len(r.seq) for r in pool)
            max_cont = 120000 if rem > 120.0 else 50000
            denom = max(1, T * (entries + 2 * N + 2))
            K = max(1, int(max_cont // denom))
            K = min(K, 8 if rem > 120.0 else 3, S + 1)

            opt_dems, opt_weights = select_opt_scenarios(K)
            time_budget = max(5.0, min(rem - 10.0, rem * 0.5))

            if time_budget > 1.0:
                result = solve_route_mip(pool, opt_dems, opt_weights, time_budget)
                if result is not None:
                    assignments, y_mip = result

                    if remaining() > 3.0:
                        sol = evaluate_solution(
                            assignments,
                            y_mip,
                            min(remaining() - 2.0, max(5.0, rem * 0.15)),
                        )
                        consider(sol)

                    if remaining() > 10.0:
                        y2 = optimize_y_fixed_routes(
                            assignments,
                            y_mip,
                            min(remaining() - 5.0, max(5.0, rem * 0.10)),
                        )
                        if y2 != y_mip and remaining() > 3.0:
                            sol2 = evaluate_solution(
                                assignments,
                                y2,
                                min(remaining() - 2.0, max(3.0, rem * 0.10)),
                            )
                            consider(sol2)

    if N > 0 and M > 0 and T > 0 and remaining() > 20.0 and best_obj >= baseline_obj - 1e-6:
        rem = remaining()
        max_binaries = 1200
        R_limit = max(1, int(max_binaries // max(1, T)))
        R_limit = max(R_limit, min(N, 50))
        R_limit = min(R_limit, 200)
        max_len = min(N, 5)

        pool = generate_route_pool(R_limit, max_len)
        heur = build_heuristic_routes(pool)

        if any(heur[t] for t in range(T)):
            y_init = [1 if heur[t] and CAP_PROD > 0.0 else 0 for t in range(T)]
            y_use = y_init

            if GUROBI and remaining() > 10.0:
                y_opt = optimize_y_fixed_routes(
                    heur,
                    y_init,
                    min(remaining() - 5.0, 10.0),
                )
                if y_opt is not None:
                    y_use = y_opt

            if remaining() > 3.0:
                sol = evaluate_solution(
                    heur,
                    y_use,
                    min(remaining() - 2.0, 10.0),
                )
                consider(sol)

    if best_sol is None:
        best_sol = {
            "objective_value": 0.0,
            "model_variables": {},
            "y": {},
            "z": {},
            "x": {},
        }

    with open(args.solution_path, "w") as out:
        json.dump(best_sol, out)


if __name__ == "__main__":
    main()