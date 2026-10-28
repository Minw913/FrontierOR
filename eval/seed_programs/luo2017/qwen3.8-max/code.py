import argparse
import json
import math
import time
from itertools import combinations

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

try:
    from solution_logger import SolutionLogger
except Exception:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_solution(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass


EPS = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = max(0.0, float(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = int(data.get("num_customers", 0))
    Q = float(data.get("vehicle_capacity", 0.0))
    horizon = float(data.get("scheduling_horizon", 1e100))

    cost_function = data.get("cost_function", {})
    a = float(cost_function.get("a", 0.0))
    b = float(cost_function.get("b", 0.0))

    depot = data.get("depot", {})
    depot_id = int(depot.get("id", 0))
    depot_idx = depot_id

    dist = data.get("distance_matrix", [])
    if not dist:
        coords = data.get("coordinates", [])
        m = len(coords) if coords else (N + 1)
        dist = [[0.0] * m for _ in range(m)]
        if coords:
            for i in range(m):
                xi, yi = float(coords[i][0]), float(coords[i][1])
                for j in range(m):
                    xj, yj = float(coords[j][0]), float(coords[j][1])
                    dist[i][j] = math.hypot(xi - xj, yi - yj)

    if depot_idx < 0 or depot_idx >= len(dist):
        depot_idx = 0

    depot_tw = depot.get("time_window", [0.0, horizon])
    depot_e = float(depot_tw[0])
    depot_l = float(min(float(depot_tw[1]), horizon))
    depot_service = float(depot.get("service_time", 0.0))

    demand = {}
    service = {}
    tw_early = {}
    tw_late = {}
    cust_ids = []

    for c in data.get("customers", []):
        cid = int(c["id"])
        cust_ids.append(cid)
        demand[cid] = float(c.get("demand", 0.0))
        service[cid] = float(c.get("service_time", 0.0))
        ctw = c.get("time_window", [0.0, horizon])
        tw_early[cid] = float(ctw[0])
        tw_late[cid] = float(min(float(ctw[1]), horizon))

    active = [i for i in cust_ids if demand.get(i, 0.0) > EPS]

    def write_solution(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)

    if not active or Q <= EPS:
        sol = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(sol)
        return

    depot_start = depot_e + depot_service

    def is_feasible(seq):
        if depot_start > depot_l + EPS:
            return False
        t = depot_start
        loc = depot_idx
        for c in seq:
            t += float(dist[loc][c])
            if t < tw_early[c]:
                t = tw_early[c]
            if t > tw_late[c] + EPS:
                return False
            t += service[c]
            loc = c
        t += float(dist[loc][depot_idx])
        return t <= depot_l + EPS

    def cumdist_total(seq):
        cum = {}
        d = 0.0
        loc = depot_idx
        for c in seq:
            d += float(dist[loc][c])
            cum[c] = d
            loc = c
        if seq:
            d += float(dist[loc][depot_idx])
        return cum, d

    def total_distance(seq):
        return cumdist_total(seq)[1]

    def route_cost(seq, qty):
        if not seq or not qty:
            return 0.0
        cum, total = cumdist_total(seq)
        cost = b * total
        if abs(a) > EPS:
            for c in seq:
                q = qty.get(c, 0.0)
                if q > EPS:
                    cost += a * q * cum[c]
        return cost

    direct_routes = []
    for i in active:
        rem = demand[i]
        while rem > EPS:
            take = min(Q, rem)
            direct_routes.append(([i], {i: take}))
            rem -= take

    def finalize_routes(routes):
        cleaned = []

        for seq, qty in routes:
            q = {int(c): float(v) for c, v in qty.items() if float(v) > EPS}
            if not q:
                continue
            s = [c for c in seq if c in q]
            if not s:
                continue

            load = sum(q.values())
            if Q > EPS and load > Q + 1e-5:
                remaining = dict(q)
                while sum(v for v in remaining.values() if v > EPS) > EPS:
                    cap = Q
                    qq = {}
                    for c in s:
                        rem = remaining.get(c, 0.0)
                        if rem > EPS and cap > EPS:
                            take = min(rem, cap)
                            qq[c] = take
                            remaining[c] = rem - take
                            cap -= take
                    if qq:
                        ss = [c for c in s if c in qq]
                        cleaned.append((ss, qq))
                    else:
                        break
            else:
                cleaned.append((s, q))

        delivered = {i: 0.0 for i in active}
        for s, q in cleaned:
            for c, v in q.items():
                delivered[c] = delivered.get(c, 0.0) + v

        for c in active:
            dv = delivered.get(c, 0.0)
            if dv > demand[c] + 1e-5:
                factor = demand[c] / dv if dv > EPS else 1.0
                if factor < 1.0:
                    for s, q in cleaned:
                        if c in q:
                            q[c] *= factor
                            if q[c] <= EPS:
                                del q[c]
                    delivered[c] = demand[c]

        cleaned2 = []
        for s, q in cleaned:
            qq = {c: float(v) for c, v in q.items() if float(v) > EPS}
            if not qq:
                continue
            ss = [c for c in s if c in qq]
            if ss:
                cleaned2.append((ss, qq))
        cleaned = cleaned2

        delivered = {i: 0.0 for i in active}
        for s, q in cleaned:
            for c, v in q.items():
                delivered[c] = delivered.get(c, 0.0) + v

        if Q > EPS:
            for c in active:
                rem = demand[c] - delivered.get(c, 0.0)
                if rem > 1e-6:
                    while rem > EPS:
                        take = min(Q, rem)
                        cleaned.append(([c], {c: take}))
                        rem -= take

        return cleaned

    def build_solution(routes):
        objective = 0.0
        out_routes = []
        virtual_depot = N + 1
        vehicle_id = 0

        for seq, qty in routes:
            q = {c: float(v) for c, v in qty.items() if float(v) > EPS}
            if not q:
                continue
            seq = [c for c in seq if c in q]
            if not seq:
                continue

            load = sum(q.values())
            edges = []
            route_obj = 0.0
            prev_mat = depot_idx
            prev_id = depot_id

            for c in seq:
                dij = float(dist[prev_mat][c])
                edges.append({
                    "from": prev_id,
                    "to": c,
                    "load_weight": float(load)
                })
                route_obj += dij * (a * load + b)

                load -= q[c]
                if load < 0.0 and load > -1e-6:
                    load = 0.0

                prev_mat = c
                prev_id = c

            d_back = float(dist[prev_mat][depot_idx])
            if load < EPS:
                load = 0.0

            edges.append({
                "from": prev_id,
                "to": virtual_depot,
                "load_weight": float(load)
            })
            route_obj += d_back * (a * load + b)

            objective += route_obj
            out_routes.append({
                "vehicle": vehicle_id,
                "route": [depot_id] + seq + [virtual_depot],
                "deliveries": {str(c): float(q[c]) for c in seq},
                "edges": edges
            })
            vehicle_id += 1

        return {"objective_value": float(objective), "routes": out_routes}

    best_routes = finalize_routes(direct_routes)
    best_solution = build_solution(best_routes)
    best_obj = float(best_solution["objective_value"])

    if logger:
        logger.log_solution(best_obj, best_solution)

    def update_best(routes):
        nonlocal best_routes, best_solution, best_obj
        cleaned = finalize_routes(routes)
        sol = build_solution(cleaned)
        obj = float(sol["objective_value"])
        tol = 1e-6 * max(1.0, abs(best_obj))
        if obj < best_obj - tol:
            best_routes = cleaned
            best_solution = sol
            best_obj = obj
            if logger:
                logger.log_solution(obj, sol)
            return True
        return False

    def intra_improve(r, stop_time):
        seq = r["seq"]
        qty = r["qty"]
        if len(seq) <= 1:
            return False

        base_cost = route_cost(seq, qty)

        # Relocate one customer inside the same route.
        best_seq = None
        best_cost = base_cost

        for i, c in enumerate(seq):
            if time.time() >= stop_time:
                break
            removed = seq[:i] + seq[i + 1:]
            for pos in range(len(removed) + 1):
                if pos == i:
                    continue
                ns = removed[:pos] + [c] + removed[pos:]
                if ns == seq:
                    continue
                if not is_feasible(ns):
                    continue
                cc = route_cost(ns, qty)
                if cc < best_cost - 1e-7:
                    best_cost = cc
                    best_seq = ns

        if best_seq is not None:
            r["seq"] = best_seq
            return True

        # 2-opt for short routes.
        if len(seq) <= 8:
            best_seq = None
            best_cost = base_cost
            L = len(seq)
            for i in range(L - 1):
                if time.time() >= stop_time:
                    break
                for j in range(i + 1, L):
                    ns = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    if ns == seq:
                        continue
                    if not is_feasible(ns):
                        continue
                    cc = route_cost(ns, qty)
                    if cc < best_cost - 1e-7:
                        best_cost = cc
                        best_seq = ns
            if best_seq is not None:
                r["seq"] = best_seq
                return True

        return False

    def improve_routes(initial_routes, stop_time):
        routes = []
        for seq, qty in initial_routes:
            q = {int(c): float(v) for c, v in qty.items() if float(v) > EPS}
            if not q:
                continue
            s = [c for c in seq if c in q]
            if s:
                routes.append({"seq": s, "qty": q})

        if not routes:
            return []

        changed = True
        while changed and time.time() < stop_time:
            changed = False

            # Intra-route improvements.
            for r in routes:
                if time.time() >= stop_time:
                    break
                if intra_improve(r, stop_time):
                    changed = True

            # Inter-route partial/full relocation.
            inter_changed = False
            for ri in range(len(routes)):
                if time.time() >= stop_time:
                    break
                if ri >= len(routes):
                    continue

                r = routes[ri]
                if not r["seq"]:
                    continue

                r_cost = route_cost(r["seq"], r["qty"])
                moved = False

                for c in list(r["seq"]):
                    if moved or time.time() >= stop_time:
                        break

                    q = r["qty"].get(c, 0.0)
                    if q <= EPS:
                        continue

                    for ti in range(len(routes)):
                        if ti == ri or time.time() >= stop_time:
                            continue

                        t = routes[ti]
                        if not t["seq"]:
                            continue

                        load_t = sum(t["qty"].values())
                        cap_left = Q - load_t
                        if cap_left <= EPS:
                            continue

                        take = min(q, cap_left)
                        if take <= 1e-6:
                            continue

                        t_cost = route_cost(t["seq"], t["qty"])

                        if c in t["qty"]:
                            new_qty_t = dict(t["qty"])
                            new_qty_t[c] = new_qty_t.get(c, 0.0) + take
                            new_seq_t = list(t["seq"])
                            cost_t_new = route_cost(new_seq_t, new_qty_t)
                            best_info = (new_seq_t, new_qty_t, cost_t_new)
                        else:
                            best_info = None
                            for pos in range(len(t["seq"]) + 1):
                                ns = t["seq"][:pos] + [c] + t["seq"][pos:]
                                if not is_feasible(ns):
                                    continue
                                nq = dict(t["qty"])
                                nq[c] = take
                                cc = route_cost(ns, nq)
                                if best_info is None or cc < best_info[2] - EPS:
                                    best_info = (ns, nq, cc)

                            if best_info is None:
                                continue

                        new_qty_r = dict(r["qty"])
                        rem = q - take
                        if rem > EPS:
                            new_qty_r[c] = rem
                        else:
                            new_qty_r.pop(c, None)

                        new_seq_r = [x for x in r["seq"] if x in new_qty_r]
                        cost_r_new = route_cost(new_seq_r, new_qty_r) if new_seq_r else 0.0

                        delta = best_info[2] + cost_r_new - (r_cost + t_cost)
                        if delta < -1e-7:
                            t["seq"] = list(best_info[0])
                            t["qty"] = dict(best_info[1])

                            if new_seq_r:
                                r["seq"] = list(new_seq_r)
                                r["qty"] = dict(new_qty_r)
                            else:
                                routes.pop(ri)

                            changed = True
                            inter_changed = True
                            moved = True
                            break

                if moved:
                    break

            if inter_changed:
                routes = [r for r in routes if r["seq"]]

        return [(r["seq"], r["qty"]) for r in routes]

    na = len(active)

    if na <= 7:
        max_columns = 200000
    elif na <= 9 and time_limit >= 120:
        max_columns = 30000
    else:
        max_columns = int(min(20000, max(3000, time_limit * 80)))
    max_columns = max(max_columns, na + 100)

    columns = []
    col_index = {}

    def add_column(seq, prechecked=False):
        if not seq:
            return False

        tup = tuple(int(c) for c in seq)
        if len(set(tup)) != len(tup):
            return False

        if tup in col_index:
            return True

        if len(columns) >= max_columns:
            return False

        if not all(demand.get(c, 0.0) > EPS for c in tup):
            return False

        if not prechecked and not is_feasible(tup):
            return False

        cum, total = cumdist_total(tup)
        fixed = b * total
        unit = {c: a * cum[c] for c in tup}

        col_index[tup] = len(columns)
        columns.append({
            "seq": tup,
            "fixed": fixed,
            "unit": unit,
            "total": total
        })
        return True

    def add_subsequences(seq, max_len, limit, stop_time):
        added = 0
        L = len(seq)
        max_r = min(max_len, L)
        for r in range(1, max_r + 1):
            for idxs in combinations(range(L), r):
                if added >= limit or len(columns) >= max_columns or time.time() > stop_time:
                    return
                sub = [seq[i] for i in idxs]
                if add_column(sub, prechecked=True):
                    added += 1

    def enumerate_columns(max_len, stop_time):
        if depot_start > depot_l + EPS:
            return

        feas_order = sorted(active, key=lambda x: (tw_late[x], -demand[x], x))
        path = []
        path_set = set()

        def dfs(loc, t):
            if len(path) >= max_len or len(columns) >= max_columns or time.time() > stop_time:
                return

            for c in feas_order:
                if c in path_set:
                    continue
                if len(columns) >= max_columns or time.time() > stop_time:
                    return

                arrival = t + float(dist[loc][c])
                start = arrival if arrival >= tw_early[c] else tw_early[c]
                if start > tw_late[c] + EPS:
                    continue

                finish = start + service[c]
                if finish + float(dist[c][depot_idx]) > depot_l + EPS:
                    continue

                path.append(c)
                path_set.add(c)
                add_column(path, prechecked=True)
                dfs(c, finish)
                path.pop()
                path_set.remove(c)

        dfs(depot_idx, depot_start)

    def generate_insertion_columns(stop_time):
        if len(columns) >= max_columns:
            return

        if na <= 12:
            max_len_ins = 6
        elif na <= 25:
            max_len_ins = 5
        else:
            max_len_ins = 4

        seed_order = sorted(active, key=lambda x: (-demand[x], tw_late[x], x))
        seed_limit = min(na, 30 if na > 30 else na)

        for seed in seed_order[:seed_limit]:
            if time.time() > stop_time or len(columns) >= max_columns:
                break

            if not is_feasible([seed]):
                continue

            route = [seed]
            add_column(route, prechecked=True)
            old_total = total_distance(route)

            while len(route) < max_len_ins and time.time() < stop_time and len(columns) < max_columns:
                best_seq = None
                best_score = float("inf")
                best_total = old_total

                for c in active:
                    if c in route:
                        continue

                    for pos in range(len(route) + 1):
                        prev = depot_idx if pos == 0 else route[pos - 1]
                        nxt = depot_idx if pos == len(route) else route[pos]

                        delta = (
                            float(dist[prev][c])
                            + float(dist[c][nxt])
                            - float(dist[prev][nxt])
                        )

                        if delta >= best_score - EPS:
                            continue

                        ns = route[:pos] + [c] + route[pos:]
                        if not is_feasible(ns):
                            continue

                        best_score = delta
                        best_seq = ns
                        best_total = old_total + delta

                if best_seq is None:
                    break

                route = best_seq
                old_total = best_total
                add_column(route, prechecked=True)

                if len(route) <= 5:
                    add_subsequences(route, min(4, len(route)), 10, stop_time)

    def solve_with_gurobi(cols, mip_stop):
        if gp is None or GRB is None or not cols:
            return None

        covered = set()
        for col in cols:
            covered.update(col["seq"])
        if not all(c in covered for c in active):
            return None

        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model(env=env)
        except Exception:
            try:
                model = gp.Model()
                model.setParam("OutputFlag", 0)
            except Exception:
                return None

        try:
            y_by_col = {}
            q_by_col = {}

            for idx, col in enumerate(cols):
                seq = col["seq"]
                seq_demand = sum(demand.get(c, 0.0) for c in seq)
                if seq_demand <= EPS:
                    continue

                ub = int(math.ceil(seq_demand / Q - 1e-9))
                if ub < 1:
                    ub = 1

                y = model.addVar(
                    lb=0.0,
                    ub=float(ub),
                    vtype=GRB.INTEGER,
                    obj=float(col["fixed"]),
                    name=f"y_{idx}"
                )

                qd = {}
                for c in seq:
                    qv = model.addVar(
                        lb=0.0,
                        ub=float(demand[c]),
                        obj=float(col["unit"].get(c, 0.0)),
                        name=f"q_{idx}_{c}"
                    )
                    qd[c] = qv

                y_by_col[idx] = y
                q_by_col[idx] = qd

            model.update()

            if y_by_col:
                model.setAttr("Start", list(y_by_col.values()), [0.0] * len(y_by_col))

            all_q = [qv for qd in q_by_col.values() for qv in qd.values()]
            if all_q:
                model.setAttr("Start", all_q, [0.0] * len(all_q))

            direct_col = {}
            for idx, col in enumerate(cols):
                if len(col["seq"]) == 1:
                    direct_col[col["seq"][0]] = idx

            for c in active:
                idx = direct_col.get(c)
                if idx is not None and idx in y_by_col:
                    m = int(math.ceil(demand[c] / Q - 1e-9))
                    if m > 0:
                        y_by_col[idx].Start = float(m)
                        q_by_col[idx][c].Start = float(demand[c])

            cols_by_cust = {c: [] for c in active}
            for idx, col in enumerate(cols):
                if idx not in y_by_col:
                    continue
                for c in col["seq"]:
                    if c in cols_by_cust:
                        cols_by_cust[c].append(idx)

            for c in active:
                terms = [
                    q_by_col[idx][c]
                    for idx in cols_by_cust[c]
                    if idx in q_by_col and c in q_by_col[idx]
                ]
                if not terms:
                    return None
                model.addConstr(gp.quicksum(terms) == float(demand[c]), name=f"demand_{c}")

            for idx, y in y_by_col.items():
                seq = cols[idx]["seq"]
                terms = [q_by_col[idx][c] for c in seq if c in q_by_col[idx]]
                if terms:
                    model.addConstr(gp.quicksum(terms) <= Q * y, name=f"cap_{idx}")

            remaining = mip_stop - time.time()
            if remaining <= 0.2:
                return None

            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", max(0.1, remaining))

            model.optimize()

            if model.SolCount == 0:
                return None

            extracted = []

            for idx, y in y_by_col.items():
                try:
                    _ = float(y.X)
                except Exception:
                    continue

                seq = cols[idx]["seq"]
                agg = {}
                total_q = 0.0

                for c in seq:
                    try:
                        val = float(q_by_col[idx][c].X)
                    except Exception:
                        val = 0.0
                    if val > 1e-6:
                        agg[c] = val
                        total_q += val

                if total_q <= EPS:
                    continue

                k = int(math.ceil(total_q / Q - 1e-9))
                if k < 1:
                    k = 1

                remaining_agg = dict(agg)

                for _ in range(k):
                    cap = Q
                    qty = {}
                    for c in seq:
                        if cap <= EPS:
                            break
                        rem = remaining_agg.get(c, 0.0)
                        if rem > EPS:
                            take = min(rem, cap)
                            qty[c] = take
                            remaining_agg[c] = rem - take
                            cap -= take
                    if qty:
                        extracted.append((list(seq), qty))

                rem_total = sum(v for v in remaining_agg.values() if v > EPS)
                if rem_total > 1e-5:
                    for c, rem in remaining_agg.items():
                        while rem > EPS:
                            take = min(Q, rem)
                            extracted.append(([c], {c: take}))
                            rem -= take

            return extracted

        except Exception:
            return None

    # Add direct columns for the set-covering model.
    if gp is not None:
        for i in active:
            add_column([i], prechecked=False)

    # First heuristic improvement phase.
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed

    if remaining > 1.0 and len(direct_routes) <= 250:
        budget = min(max(0.5, 0.2 * time_limit), max(0.5, 0.4 * remaining))
        stop = time.time() + budget
        improved = improve_routes(direct_routes, stop)
        update_best(improved)

        if gp is not None:
            for seq, _ in best_routes:
                if add_column(seq, prechecked=False) and len(seq) <= 5:
                    add_subsequences(seq, min(4, len(seq)), 10, time.time() + 0.25)

    # Generate columns and solve a set-covering MIP if Gurobi is available.
    if gp is not None:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        reserve = max(2.0, min(30.0, 0.1 * time_limit))

        if remaining > reserve + 1.0:
            gen_budget = min(
                max(0.5, 0.15 * time_limit),
                max(0.2, (remaining - reserve) * 0.3)
            )
            gen_stop = time.time() + gen_budget

            if na <= 7:
                max_len_enum = na
            elif na <= 10:
                max_len_enum = 5
            elif na <= 15:
                max_len_enum = 4
            elif na <= 25:
                max_len_enum = 3
            else:
                max_len_enum = 2

            if time.time() < gen_stop and len(columns) < max_columns:
                enumerate_columns(max_len_enum, gen_stop)

            if time.time() < gen_stop and len(columns) < max_columns:
                generate_insertion_columns(gen_stop)

            # Make sure heuristic routes are present as columns.
            for seq, _ in best_routes:
                if add_column(seq, prechecked=False) and len(seq) <= 5:
                    add_subsequences(seq, min(4, len(seq)), 10, time.time() + 0.25)

            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            reserve = max(2.0, min(30.0, 0.1 * time_limit))
            mip_stop = start_time + time_limit - reserve

            if remaining > reserve + 1.0 and len(columns) > len(active):
                extracted = solve_with_gurobi(columns, mip_stop)
                if extracted:
                    update_best(extracted)

    # Final short local-search pass on the best known solution.
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed

    if remaining > 1.5 and len(best_routes) <= 250:
        stop = time.time() + min(remaining - 0.5, max(0.5, 0.1 * time_limit))
        improved = improve_routes(best_routes, stop)
        update_best(improved)

    write_solution(best_solution)


if __name__ == "__main__":
    main()