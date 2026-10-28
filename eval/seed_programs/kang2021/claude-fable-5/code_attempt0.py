import argparse
import json
import time

from solution_logger import SolutionLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--instance_path', required=True)
    p.add_argument('--solution_path', required=True)
    p.add_argument('--time_limit', type=int, default=60)
    p.add_argument('--log_path', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()
    deadline = start + max(3, args.time_limit) - 1.5

    with open(args.instance_path) as f:
        inst = json.load(f)

    N = inst['num_nodes']
    d0 = inst['depot_index']
    C = [int(c) for c in inst['customer_indices']]
    D = int(inst['parameters']['num_drones'])
    cap = inst['parameters']['drone_battery_capacities']
    t = inst['truck_travel_times']
    B = inst['drone_battery_consumption']
    T = inst['drone_delivery_time']
    sink = N

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------------- helpers ----------------
    def route_cost(route):
        c = 0
        for k in range(len(route) - 1):
            c += t[route[k]][route[k + 1]]
        c += t[route[-1]][d0]
        return c

    def build_solution(route, assignments):
        arcs = []
        for k in range(len(route) - 1):
            arcs.append([int(route[k]), int(route[k + 1])])
        arcs.append([int(route[-1]), int(sink)])
        per = {}
        node_set = set(route)
        for (d, i, c) in assignments:
            node_set.add(i)
            per.setdefault((d, i), 0.0)
            per[(d, i)] += T[d][i][c]
        wait = {}
        for i in node_set:
            wmax = 0.0
            for d in range(D):
                val = per.get((d, i), 0.0)
                if val > wmax:
                    wmax = val
            wait[i] = wmax
        obj = 0.0
        for a, b in arcs:
            obj += t[a][d0 if b == sink else b]
        obj += sum(wait.values())
        sol = {
            "objective_value": float(obj),
            "truck_route": arcs,
            "drone_assignments": [
                {"drone": int(d), "dispatch_node": int(i), "delivery_node": int(c)}
                for (d, i, c) in assignments
            ],
            "waiting_times": {str(int(i)): float(wait[i]) for i in wait},
        }
        return obj, sol

    # ---------------- heuristic ----------------
    def nearest_neighbor():
        unvis = set(C)
        route = [d0]
        cur = d0
        while unvis:
            nxt = min(unvis, key=lambda c: t[cur][c])
            route.append(nxt)
            unvis.discard(nxt)
            cur = nxt
        return route

    def improve_route(route):
        route = list(route)
        n = len(route)
        if n <= 2:
            return route
        cost = route_cost(route)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(route)
            # 2-opt (verified exactly to be safe against asymmetry)
            for i in range(1, n - 1):
                if time.time() > deadline:
                    break
                a = route[i - 1]
                for j in range(i + 1, n):
                    nxt = route[j + 1] if j + 1 < n else d0
                    delta = (t[a][route[j]] + t[route[i]][nxt]
                             - t[a][route[i]] - t[route[j]][nxt])
                    if delta < -1e-9:
                        new = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        nc = route_cost(new)
                        if nc < cost - 1e-9:
                            route = new
                            cost = nc
                            improved = True
                            a = route[i - 1]
            # Or-opt relocate single node (asymmetric-safe)
            n = len(route)
            moved = True
            while moved and time.time() < deadline:
                moved = False
                n = len(route)
                for i in range(1, n):
                    b = route[i]
                    a = route[i - 1]
                    c = route[i + 1] if i + 1 < n else d0
                    gain = t[a][b] + t[b][c] - t[a][c]
                    if gain <= 1e-9:
                        continue
                    rem = route[:i] + route[i + 1:]
                    m_ = len(rem)
                    best = None
                    for p in range(m_):
                        q = rem[p + 1] if p + 1 < m_ else d0
                        addc = t[rem[p]][b] + t[b][q] - t[rem[p]][q]
                        if addc < gain - 1e-9 and (best is None or addc < best[0]):
                            best = (addc, p)
                    if best is not None:
                        rem.insert(best[1] + 1, b)
                        route = rem
                        cost = cost - gain + best[0]
                        improved = True
                        moved = True
                        break
        return route

    def drone_assign(route, assignments):
        route = list(route)
        assignments = list(assignments)
        load = [dict() for _ in range(D)]
        wait = {}
        bat = [0.0] * D
        for (d, i, c) in assignments:
            load[d][i] = load[d].get(i, 0.0) + T[d][i][c]
            bat[d] += B[d][i][c]
            if load[d][i] > wait.get(i, 0.0):
                wait[i] = load[d][i]
        while time.time() < deadline:
            best = None
            n = len(route)
            for idx in range(1, n):
                c = route[idx]
                if wait.get(c, 0.0) > 1e-12:
                    continue  # dispatch node, cannot remove
                a = route[idx - 1]
                b = route[idx + 1] if idx + 1 < n else d0
                save = t[a][c] + t[c][b] - t[a][b]
                for d in range(D):
                    bd = bat[d]
                    Bd = B[d]
                    Td = T[d]
                    for i in route:
                        if i == c:
                            continue
                        bc = Bd[i][c]
                        if bd + bc > cap[d] + 1e-9:
                            continue
                        addw = load[d].get(i, 0.0) + Td[i][c] - wait.get(i, 0.0)
                        if addw < 0:
                            addw = 0.0
                        net = save - addw
                        if net > 1e-9 and (best is None or net > best[0]):
                            best = (net, d, i, c, idx)
            if best is None:
                break
            _, d, i, c, idx = best
            route.pop(idx)
            assignments.append((d, i, c))
            bat[d] += B[d][i][c]
            load[d][i] = load[d].get(i, 0.0) + T[d][i][c]
            if load[d][i] > wait.get(i, 0.0):
                wait[i] = load[d][i]
        return route, assignments

    route = nearest_neighbor()
    route = improve_route(route)
    assigns = []
    for it in range(4):
        route, assigns = drone_assign(route, assigns)
        new_route = improve_route(route)
        if new_route == route:
            route = new_route
            break
        route = new_route
        if time.time() > deadline:
            break

    best_obj, best_sol = build_solution(route, assigns)
    if logger:
        logger.log_solution(best_obj, best_sol)

    holder = {'obj': best_obj, 'sol': best_sol}

    # ---------------- MIP refinement with Gurobi ----------------
    remaining = deadline - time.time()
    if remaining > 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("truck_drones")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time())

            nodes = [d0] + C
            nC = len(C)

            x = {}
            for i in nodes:
                for j in C:
                    if j != i:
                        x[(i, j)] = m.addVar(vtype=GRB.BINARY, obj=t[i][j],
                                             name="x_%d_%d" % (i, j))
                x[(i, sink)] = m.addVar(vtype=GRB.BINARY, obj=t[i][d0],
                                        name="x_%d_%d" % (i, sink))
            v = {c: m.addVar(vtype=GRB.BINARY, name="v_%d" % c) for c in C}
            u = {c: m.addVar(lb=1.0, ub=max(1, nC), name="u_%d" % c) for c in C}
            y = {}
            for d in range(D):
                Bd = B[d]
                for i in nodes:
                    Bdi = Bd[i]
                    for c in C:
                        if c != i and Bdi[c] <= cap[d]:
                            y[(d, i, c)] = m.addVar(vtype=GRB.BINARY,
                                                    name="y_%d_%d_%d" % (d, i, c))
            w = {i: m.addVar(lb=0.0, obj=1.0, name="w_%d" % i) for i in nodes}
            m.ModelSense = GRB.MINIMIZE

            # flow constraints
            m.addConstr(gp.quicksum(x[(d0, j)] for j in C + [sink]
                                    if (d0, j) in x) == 1)
            m.addConstr(gp.quicksum(x[(i, sink)] for i in nodes) == 1)
            for c in C:
                m.addConstr(gp.quicksum(x[(i, c)] for i in nodes
                                        if (i, c) in x) == v[c])
                m.addConstr(gp.quicksum(x[(c, j)] for j in C + [sink]
                                        if (c, j) in x) == v[c])
            # MTZ subtour elimination
            for i in C:
                for j in C:
                    if (i, j) in x:
                        m.addConstr(u[j] >= u[i] + 1 - nC * (1 - x[(i, j)]))
            # serve each customer once
            y_by_c = {c: [] for c in C}
            y_by_d = [[] for _ in range(D)]
            y_by_di = {}
            for (d, i, c), var in y.items():
                y_by_c[c].append(var)
                y_by_d[d].append(((i, c), var))
                y_by_di.setdefault((d, i), []).append((c, var))
            for c in C:
                m.addConstr(v[c] + gp.quicksum(y_by_c[c]) == 1)
            # dispatch only from visited nodes
            for (d, i, c), var in y.items():
                if i != d0:
                    m.addConstr(var <= v[i])
            # battery
            for d in range(D):
                m.addConstr(gp.quicksum(B[d][i][c] * var
                                        for (i, c), var in y_by_d[d]) <= cap[d])
            # waiting times
            for d in range(D):
                for i in nodes:
                    lst = y_by_di.get((d, i))
                    if lst:
                        m.addConstr(w[i] >= gp.quicksum(T[d][i][c] * var
                                                        for c, var in lst))

            # warm start from heuristic
            for var in x.values():
                var.Start = 0.0
            for c in C:
                v[c].Start = 0.0
            for var in y.values():
                var.Start = 0.0
            for k in range(len(route) - 1):
                x[(route[k], route[k + 1])].Start = 1.0
            x[(route[-1], sink)].Start = 1.0
            pos = 1
            for c in route[1:]:
                v[c].Start = 1.0
                u[c].Start = float(pos)
                pos += 1
            start_wait = {}
            for (d, i, c) in assigns:
                if (d, i, c) in y:
                    y[(d, i, c)].Start = 1.0
                start_wait.setdefault((d, i), 0.0)
                start_wait[(d, i)] += T[d][i][c]
            for i in nodes:
                wv = 0.0
                for d in range(D):
                    wv = max(wv, start_wait.get((d, i), 0.0))
                w[i].Start = wv

            xk = list(x.keys())
            xv = [x[k] for k in xk]
            yk = list(y.keys())
            yv = [y[k] for k in yk]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    xs = model.cbGetSolution(xv)
                    ys = model.cbGetSolution(yv)
                    succ = {}
                    for idx, val in enumerate(xs):
                        if val > 0.5:
                            a, b = xk[idx]
                            succ[a] = b
                    r = [d0]
                    seen = {d0}
                    cur = d0
                    while cur in succ and succ[cur] != sink:
                        cur = succ[cur]
                        if cur in seen:
                            break
                        r.append(cur)
                        seen.add(cur)
                    asg = [yk[idx] for idx, val in enumerate(ys) if val > 0.5]
                    obj, sol = build_solution(r, asg)
                    if obj < holder['obj'] - 1e-9:
                        holder['obj'] = obj
                        holder['sol'] = sol
                        if logger:
                            logger.log_solution(obj, sol)

            m.optimize(cb)

            # extract final incumbent (in case callback missed it)
            if m.SolCount > 0:
                succ = {}
                for k, var in x.items():
                    if var.X > 0.5:
                        succ[k[0]] = k[1]
                r = [d0]
                seen = {d0}
                cur = d0
                while cur in succ and succ[cur] != sink:
                    cur = succ[cur]
                    if cur in seen:
                        break
                    r.append(cur)
                    seen.add(cur)
                asg = [k for k, var in y.items() if var.X > 0.5]
                obj, sol = build_solution(r, asg)
                if obj < holder['obj'] - 1e-9:
                    holder['obj'] = obj
                    holder['sol'] = sol
                    if logger:
                        logger.log_solution(obj, sol)
        except Exception:
            pass

    with open(args.solution_path, 'w') as f:
        json.dump(holder['sol'], f, indent=2)


if __name__ == '__main__':
    main()