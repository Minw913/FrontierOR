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
    parser.add_argument("instance_path")
    parser.add_argument("solution_path")
    parser.add_argument("time_limit", type=float, nargs="?", default=60.0)
    parser.add_argument("log_path", nargs="?", default=None)
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
    depot_tw = depot.get("time_window", [0.0, horizon])
    depot_e = float(depot_tw[0])
    depot_l = float(min(float(depot_tw[1]), horizon))
    depot_service = float(depot.get("service_time", 0.0))

    dist = data.get("distance_matrix", [])
    customers = data.get("customers", [])

    if not dist:
        coords = data.get("coordinates", [])
        if coords:
            m = len(coords)
            dist = [[0.0] * m for _ in range(m)]
            for i in range(m):
                xi, yi = float(coords[i][0]), float(coords[i][1])
                for j in range(m):
                    xj, yj = float(coords[j][0]), float(coords[j][1])
                    dist[i][j] = math.hypot(xi - xj, yi - yj)
        else:
            dist = [[0.0] * (N + 1) for _ in range(N + 1)]

    demand = {}
    service = {}
    tw_early = {}
    tw_late = {}
    cust_ids = []

    for c in customers:
        cid = int(c["id"])
        cust_ids.append(cid)
        demand[cid] = float(c.get("demand", 0.0))
        service[cid] = float(c.get("service_time", 0.0))
        ctw = c.get("time_window", [0.0, horizon])
        tw_early[cid] = float(ctw[0])
        tw_late[cid] = float(ctw[1])

    active = [i for i in cust_ids if demand.get(i, 0.0) > EPS]

    def write_solution(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)

    if not active:
        sol = {"objective_value": 0.0, "routes": []}
        if logger is not None:
            logger.log_solution(0.0, sol)
        write_solution(sol)
        return

    if Q <= EPS:
        sol = {"objective_value": 0.0, "routes": []}
        if logger is not None:
            logger.log_solution(0.0, sol)
        write_solution(sol)
        return

    depot_start = depot_e + depot_service

    def is_feasible(seq):
        if depot_start > depot_l + EPS:
            return False
        t = depot_start
        loc = 0
        for c in seq:
            t += float(dist[loc][c])
            if t < tw_early[c]:
                t = tw_early[c]
            if t > tw_late[c] + EPS:
                return False
            t += service[c]
            loc = c
        t += float(dist[loc][0])
        return t <= depot_l + EPS

    def cumdist_total(seq):
        cum = {}
        d = 0.0
        loc = 0
        for c in seq:
            d += float(dist[loc][c])
            cum[c] = d
            loc = c
        total = d + (float(dist[loc][0]) if seq else 0.0)
        return cum, total

    def total_distance(seq):
        d = 0.0
        loc = 0
        for c in seq:
            d += float(dist[loc][c])
            loc = c
        if seq:
            d += float(dist[loc][0])
        return d

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

    def finalize_and_build(routes):
        cleaned = []

        for seq, qty in routes:
            if not seq:
                continue
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
                if rem > 1e-5:
                    while rem > EPS:
                        take = min(Q, rem)
                        cleaned.append(([c], {c: take}))
                        rem -= take

        objective = 0.0
        out_routes = []
        virtual_depot = N + 1
        vehicle_id = 0

        for seq, qty in cleaned:
            q = {c: float(v) for c, v in qty.items() if float(v) > EPS}
            if not q:
                continue
            seq = [c for c in seq if c in q]
            if not seq:
                continue

            load = sum(q.values())
            edges = []
            route_obj = 0.0
            prev_mat = 0
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

            d_back = float(dist[prev_mat][0])
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

    direct_routes = []
    for i in active:
        rem = demand[i]
        while rem > EPS:
            take = min(Q, rem)
            direct_routes.append(([i], {i: take}))
            rem -= take

    best_solution = finalize_and_build(direct_routes)
    best_obj = float(best_solution["objective_value"])
    if logger is not None:
        logger.log_solution(best_obj, best_solution)

    if abs(b) <= 1e-9 and a >= -EPS:
        write_solution(best_solution)
        return

    if time_limit <= (time.time() - start_time):
        write_solution(best_solution)
        return

    na = len(active)
    if na <= 7:
        max_columns = 200000
    elif na <= 9 and time_limit >= 120:
        max_columns = 30000
    else:
        max_columns = int(min(20000, max(3000, time_limit * 80)))

    columns = []
    col_index = {}

    def add_column(seq, prechecked=False):
        if not seq:
            return False
        tup = tuple(int(c) for c in seq)
        if tup in col_index:
            return True
        if len(columns) >= max_columns:
            return False
        if all(demand.get(c, 0.0) <= EPS for c in tup):
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

    for i in active:
        add_column([i])

    for i in active:
        if (i,) not in col_index:
            add_column([i], prechecked=True)

    def greedy_improve(initial_routes, stop_time):
        routes = []
        for seq, qty in initial_routes:
            q = {c: float(v) for c, v in qty.items() if float(v) > EPS}
            if not q:
                continue
            s = [c for c in seq if c in q]
            if not s:
                continue
            routes.append({
                "seq": list(s),
                "qty": dict(q),
                "cost": route_cost(s, q)
            })

        changed = True
        while changed and time.time() < stop_time:
            changed = False
            for ri in range(len(routes)):
                if time.time() >= stop_time:
                    break
                r = routes[ri]
                r_cost = route_cost(r["seq"], r["qty"])

                for c in list(r["seq"]):
                    q = r["qty"].get(c, 0.0)
                    if q <= EPS:
                        continue

                    for ti in range(len(routes)):
                        if ti == ri:
                            continue
                        t = routes[ti]
                        t_load = sum(t["qty"].values())
                        if t_load + q > Q + EPS:
                            continue

                        t_cost = route_cost(t["seq"], t["qty"])

                        if c in t["qty"]:
                            new_seq_t = list(t["seq"])
                            new_qty_t = dict(t["qty"])
                            new_qty_t[c] = new_qty_t.get(c, 0.0) + q
                            if sum(new_qty_t.values()) > Q + EPS:
                                continue
                            cost_t_new = route_cost(new_seq_t, new_qty_t)
                            new_info = (new_seq_t, new_qty_t, cost_t_new)
                        else:
                            best = None
                            for pos in range(len(t["seq"]) + 1):
                                new_seq = t["seq"][:pos] + [c] + t["seq"][pos:]
                                if not is_feasible(new_seq):
                                    continue
                                new_qty = dict(t["qty"])
                                new_qty[c] = q
                                cost = route_cost(new_seq, new_qty)
                                if best is None or cost < best[2] - EPS:
                                    best = (new_seq, new_qty, cost)
                            if best is None:
                                continue
                            new_info = best

                        new_seq_r = [x for x in r["seq"] if x != c]
                        new_qty_r = {k: v for k, v in r["qty"].items() if k != c}
                        cost_r_new = route_cost(new_seq_r, new_qty_r) if new_seq_r else 0.0

                        delta = new_info[2] + cost_r_new - (r_cost + t_cost)
                        if delta < -1e-7:
                            t["seq"] = list(new_info[0])
                            t["qty"] = dict(new_info[1])
                            t["cost"] = new_info[2]

                            if new_seq_r:
                                r["seq"] = list(new_seq_r)
                                r["qty"] = dict(new_qty_r)
                                r["cost"] = cost_r_new
                            else:
                                routes.pop(ri)

                            changed = True
                            break

                    if changed:
                        break
                if changed:
                    break

        return [(r["seq"], r["qty"]) for r in routes]

    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed

    if len(direct_routes) <= 120 and remaining_time > 3.0:
        greedy_stop = time.time() + min(3.0, max(0.5, 0.1 * time_limit))
        greedy_routes = greedy_improve(direct_routes, greedy_stop)
        greedy_solution = finalize_and_build(greedy_routes)
        greedy_obj = float(greedy_solution["objective_value"])
        tol = 1e-6 * max(1.0, abs(best_obj))
        if greedy_obj < best_obj - tol:
            best_obj = greedy_obj
            best_solution = greedy_solution
            if logger is not None:
                logger.log_solution(best_obj, best_solution)

        for seq, qty in greedy_routes:
            if seq:
                add_column(seq, prechecked=True)

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

    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    reserve = max(2.0, min(30.0, 0.1 * time_limit))

    if remaining_time <= reserve + 0.5:
        write_solution(best_solution)
        return

    gen_budget = min(15.0, max(0.5, 0.15 * time_limit), remaining_time - reserve - 0.5)
    gen_stop = time.time() + max(0.2, gen_budget)

    search_order = sorted(active, key=lambda x: (-demand[x], x))

    def enumerate_columns(max_len, stop_time):
        if depot_start > depot_l + EPS:
            return

        path = []
        path_set = set()

        def dfs(loc, t):
            if len(columns) >= max_columns or time.time() > stop_time:
                return
            if len(path) >= max_len:
                return

            for c in search_order:
                if c in path_set:
                    continue
                if len(columns) >= max_columns or time.time() > stop_time:
                    return

                arrival = t + float(dist[loc][c])
                start = arrival if arrival >= tw_early[c] else tw_early[c]
                if start > tw_late[c] + EPS:
                    continue

                finish = start + service[c]
                if finish + float(dist[c][0]) > depot_l + EPS:
                    continue

                path.append(c)
                path_set.add(c)
                add_column(path, prechecked=True)
                dfs(c, finish)
                path.pop()
                path_set.remove(c)

        dfs(0, depot_start)

    enumerate_columns(max_len_enum, gen_stop)

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

    def generate_insertion_columns(stop_time):
        if len(columns) >= max_columns:
            return

        if na <= 12:
            max_len_ins = 6
        elif na <= 25:
            max_len_ins = 5
        else:
            max_len_ins = 4

        seed_limit = min(na, 20 if na > 30 else na)
        seeds = search_order[:seed_limit]

        for seed in seeds:
            if time.time() > stop_time or len(columns) >= max_columns:
                break
            if not is_feasible([seed]):
                continue

            route = [seed]
            add_column(route, prechecked=True)

            while len(route) < max_len_ins and time.time() < stop_time and len(columns) < max_columns:
                best_seq = None
                best_score = float("inf")
                old_total = total_distance(route)

                for c in search_order:
                    if c in route:
                        continue
                    for pos in range(len(route) + 1):
                        new_seq = route[:pos] + [c] + route[pos:]
                        if not is_feasible(new_seq):
                            continue
                        new_total = total_distance(new_seq)
                        score = new_total - old_total
                        if score < best_score - EPS:
                            best_score = score
                            best_seq = new_seq

                if best_seq is None:
                    break

                route = best_seq
                add_column(route, prechecked=True)
                if len(route) <= 5:
                    add_subsequences(route, min(4, len(route)), 20, stop_time)

    if time.time() < gen_stop and len(columns) < max_columns:
        generate_insertion_columns(gen_stop)

    def solve_with_gurobi(cols, mip_time, current_best_obj):
        if gp is None or GRB is None or not cols:
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
            y_vars = []
            q_vars = []
            all_q_vars = []

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
                y_vars.append(y)

                qd = {}
                for c in seq:
                    qv = model.addVar(
                        lb=0.0,
                        ub=float(demand[c]),
                        obj=float(col["unit"].get(c, 0.0)),
                        name=f"q_{idx}_{c}"
                    )
                    qd[c] = qv
                    all_q_vars.append(qv)
                q_vars.append(qd)

            model.update()

            if y_vars:
                model.setAttr("Start", y_vars, [0.0] * len(y_vars))
            if all_q_vars:
                model.setAttr("Start", all_q_vars, [0.0] * len(all_q_vars))

            direct_idx = {}
            for idx, col in enumerate(cols):
                if len(col["seq"]) == 1:
                    direct_idx[col["seq"][0]] = idx

            for c in active:
                idx = direct_idx.get(c)
                if idx is not None and idx < len(y_vars):
                    m = int(math.ceil(demand[c] / Q - 1e-9))
                    if m > 0:
                        y_vars[idx].Start = float(m)
                        q_vars[idx][c].Start = float(demand[c])

            cols_by_cust = {c: [] for c in active}
            for idx, col in enumerate(cols):
                for c in col["seq"]:
                    if c in cols_by_cust:
                        cols_by_cust[c].append(idx)

            for c in active:
                expr = gp.quicksum(q_vars[idx][c] for idx in cols_by_cust[c])
                model.addConstr(expr == float(demand[c]), name=f"demand_{c}")

            for idx, col in enumerate(cols):
                if idx >= len(q_vars):
                    continue
                seq = col["seq"]
                if not seq:
                    continue
                expr = gp.quicksum(q_vars[idx][c] for c in seq)
                model.addConstr(expr <= Q * y_vars[idx], name=f"cap_{idx}")

            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", max(0.1, float(mip_time)))

            last_cb = [float(current_best_obj)]

            def callback(cb_model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                        if obj < last_cb[0] - 1e-6:
                            last_cb[0] = obj
                            if logger is not None:
                                logger.log(obj)
                    except Exception:
                        pass

            if logger is not None:
                model.optimize(callback)
            else:
                model.optimize()

            if model.SolCount == 0:
                return None

            extracted_routes = []

            for idx, col in enumerate(cols):
                if idx >= len(y_vars):
                    continue
                try:
                    yv = float(y_vars[idx].X)
                except Exception:
                    continue

                m = int(round(yv)) if yv > 0.5 else 0
                if m <= 0:
                    continue

                seq = col["seq"]
                agg = {}
                total_q = 0.0

                for c in seq:
                    try:
                        val = float(q_vars[idx][c].X)
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
                if k > m:
                    k = m

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
                        extracted_routes.append((list(seq), qty))

                for c, rem in remaining_agg.items():
                    if rem > 1e-5:
                        while rem > EPS:
                            take = min(Q, rem)
                            extracted_routes.append(([c], {c: take}))
                            rem -= take

            return finalize_and_build(extracted_routes)

        except Exception:
            return None

    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    reserve = max(2.0, min(30.0, 0.1 * time_limit))
    mip_time = remaining_time - reserve

    if gp is not None and mip_time > 1.0 and len(columns) > len(active):
        mip_solution = solve_with_gurobi(columns, mip_time, best_obj)
        if mip_solution is not None:
            mip_obj = float(mip_solution["objective_value"])
            tol = 1e-6 * max(1.0, abs(best_obj))
            if mip_obj < best_obj - tol:
                best_obj = mip_obj
                best_solution = mip_solution
                if logger is not None:
                    logger.log_solution(best_obj, best_solution)

    write_solution(best_solution)


if __name__ == "__main__":
    main()