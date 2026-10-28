import argparse
import json
import time
import sys
import traceback
from collections import defaultdict


def main():
    prog_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    num_depots = instance['num_depots']
    depots = instance['depots']
    tasks = instance['tasks']
    cp_raw = instance['compatible_pairs']
    d2t_raw = instance['depot_to_task_costs']
    t2d_raw = instance['task_to_depot_costs']

    task_ids = [t['id'] for t in tasks]
    tidx = {tid: i for i, tid in enumerate(task_ids)}
    depot_ids = [d['id'] for d in depots]
    didx = {did: i for i, did in enumerate(depot_ids)}
    capacities = [d['capacity'] for d in depots]
    total_cap = sum(capacities)

    succ = defaultdict(list)
    pred = defaultdict(list)
    pair_cost = {}
    for cp in cp_raw:
        i = tidx[cp['task_i']]
        j = tidx[cp['task_j']]
        succ[i].append(j)
        pred[j].append(i)
        pair_cost[(i, j)] = cp['cost']

    pair_list = list(pair_cost.keys())
    pidx = {p: k for k, p in enumerate(pair_list)}
    num_pairs = len(pair_list)

    d2t = {}
    for arc in d2t_raw:
        d = didx[arc['depot_id']]
        i = tidx[arc['task_id']]
        d2t[(d, i)] = arc['cost']
    t2d = {}
    for arc in t2d_raw:
        i = tidx[arc['task_id']]
        d = didx[arc['depot_id']]
        t2d[(i, d)] = arc['cost']

    BIG = 1e9
    for d in range(num_depots):
        for i in range(num_tasks):
            if (d, i) not in d2t:
                d2t[(d, i)] = BIG
            if (i, d) not in t2d:
                t2d[(i, d)] = BIG

    def make_sol(routes):
        rs = []
        tot = 0.0
        for d, ts in routes:
            if not ts:
                continue
            rc = d2t[(d, ts[0])]
            for k in range(len(ts) - 1):
                rc += pair_cost[(ts[k], ts[k + 1])]
            rc += t2d[(ts[-1], d)]
            tot += rc
            rs.append({
                "depot": depot_ids[d],
                "tasks": [task_ids[t] for t in ts],
                "cost": rc
            })
        return {"objective_value": tot, "routes": rs}

    best_routes = None
    best_obj = float('inf')
    best_sol = {"objective_value": 0.0, "routes": []}

    if num_tasks == 0:
        with open(args.solution_path, 'w') as f:
            json.dump(best_sol, f, indent=2)
        if logger:
            logger.log_solution(0.0, best_sol)
        return

    def trivial():
        if total_cap < num_tasks:
            return None
        routes = []
        dc = [0] * num_depots
        for i in range(num_tasks):
            bd, bc = -1, float('inf')
            for d in range(num_depots):
                if dc[d] < capacities[d]:
                    c = d2t[(d, i)] + t2d[(i, d)]
                    if c < bc:
                        bc, bd = c, d
            if bd < 0:
                return None
            routes.append((bd, [i]))
            dc[bd] += 1
        return routes

    def greedy(always_ext=False):
        st = sorted(range(num_tasks), key=lambda i: tasks[i]['start_time'])
        routes = []
        du = [0] * num_depots
        for i in st:
            be = None
            for ri in range(len(routes)):
                last = routes[ri][1][-1]
                if (last, i) in pair_cost:
                    c = pair_cost[(last, i)]
                    if be is None or c < be[1]:
                        be = (ri, c)
            bn = None
            for d in range(num_depots):
                if du[d] < capacities[d]:
                    c = d2t[(d, i)]
                    if bn is None or c < bn[1]:
                        bn = (d, c)
            if always_ext and be is not None:
                routes[be[0]][1].append(i)
            elif be is not None and (bn is None or be[1] <= bn[1]):
                routes[be[0]][1].append(i)
            elif bn is not None:
                routes.append([bn[0], [i]])
                du[bn[0]] += 1
            elif be is not None:
                routes[be[0]][1].append(i)
            else:
                return None
        return [(d, s) for d, s in routes]

    def merge(routes):
        if not routes:
            return routes
        bd = defaultdict(list)
        for r in routes:
            bd[r[0]].append(r)
        out = []
        for d, drs in bd.items():
            used = [False] * len(drs)
            fmap = defaultdict(list)
            for ri, r in enumerate(drs):
                fmap[r[1][0]].append(ri)
            for i in range(len(drs)):
                if used[i]:
                    continue
                m = list(drs[i][1])
                used[i] = True
                ch = True
                while ch:
                    ch = False
                    last = m[-1]
                    bj, bs = -1, 1e-6
                    for j in succ.get(last, []):
                        for ri in fmap.get(j, []):
                            if used[ri]:
                                continue
                            sv = t2d[(last, d)] + d2t[(d, j)] - pair_cost[(last, j)]
                            if sv > bs:
                                bs, bj = sv, ri
                    if bj >= 0:
                        m.extend(drs[bj][1])
                        used[bj] = True
                        ch = True
                out.append((d, m))
        return out

    for fn in [lambda: trivial(), lambda: greedy(False), lambda: greedy(True)]:
        r = fn()
        if r is not None:
            mr = merge(r)
            for rr in [r, mr]:
                s = make_sol(rr)
                if s['objective_value'] < best_obj:
                    best_obj = s['objective_value']
                    best_routes = rr
                    best_sol = s
                    if logger:
                        logger.log_solution(best_obj, best_sol)

    elapsed = time.time() - prog_start
    c_var_count = num_depots * num_pairs
    use_mip = (c_var_count <= 1_500_000 and num_tasks <= 5000
               and elapsed < args.time_limit - 5)

    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()

            m = gp.Model(env=env)
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1

            c_obj = {}
            for k in range(num_pairs):
                ck = pair_cost[pair_list[k]]
                for d in range(num_depots):
                    c_obj[(d, k)] = ck

            a = m.addVars(range(num_depots), range(num_tasks),
                          vtype=GRB.BINARY, name='a')
            b = m.addVars(range(num_tasks), range(num_depots),
                          vtype=GRB.BINARY, name='b')
            c = m.addVars(range(num_depots), range(num_pairs),
                          vtype=GRB.BINARY, name='c')
            m.update()

            try:
                a.setAttr('Obj', d2t)
                b.setAttr('Obj', t2d)
                c.setAttr('Obj', c_obj)
            except Exception:
                for (d, i), v in a.items():
                    v.Obj = d2t[(d, i)]
                for (i, d), v in b.items():
                    v.Obj = t2d[(i, d)]
                for (d, k), v in c.items():
                    v.Obj = c_obj[(d, k)]

            m.ModelSense = GRB.MINIMIZE

            for i in range(num_tasks):
                terms = [a[d, i] for d in range(num_depots)]
                for j in pred[i]:
                    k = pidx[(j, i)]
                    for d in range(num_depots):
                        terms.append(c[d, k])
                m.addConstr(gp.quicksum(terms) == 1, name=f'cv{i}')

            for i in range(num_tasks):
                for d in range(num_depots):
                    it = [a[d, i]]
                    for j in pred[i]:
                        it.append(c[d, pidx[(j, i)]])
                    ot = [b[i, d]]
                    for j in succ[i]:
                        ot.append(c[d, pidx[(i, j)]])
                    m.addConstr(
                        gp.quicksum(it) == gp.quicksum(ot),
                        name=f'fl{i}_{d}')

            for d in range(num_depots):
                m.addConstr(
                    gp.quicksum(a[d, i] for i in range(num_tasks)) <= capacities[d],
                    name=f'cp{d}')

            if best_routes is not None:
                try:
                    a.setAttr('Start', 0)
                    b.setAttr('Start', 0)
                    c.setAttr('Start', 0)
                except Exception:
                    for key in a:
                        a[key].Start = 0
                    for key in b:
                        b[key].Start = 0
                    for key in c:
                        c[key].Start = 0
                for d, ts in best_routes:
                    a[d, ts[0]].Start = 1.0
                    b[ts[-1], d].Start = 1.0
                    for k in range(len(ts) - 1):
                        c[d, pidx[(ts[k], ts[k + 1])]].Start = 1.0

            m.update()

            best_logged = [best_obj]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        ov = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if ov < best_logged[0] - 1e-6:
                            best_logged[0] = ov
                            if logger:
                                try:
                                    av = model.cbGetSolution(a)
                                    bv = model.cbGetSolution(b)
                                    cv = model.cbGetSolution(c)
                                    routes = []
                                    for d in range(num_depots):
                                        for i in range(num_tasks):
                                            if av.get((d, i), 0) > 0.5:
                                                route = [i]
                                                cur = i
                                                cnt = 0
                                                while cnt < num_tasks:
                                                    nt = None
                                                    for j in succ.get(cur, []):
                                                        k = pidx.get((cur, j))
                                                        if k is not None and cv.get((d, k), 0) > 0.5:
                                                            nt = j
                                                            break
                                                    if nt is None:
                                                        break
                                                    route.append(nt)
                                                    cur = nt
                                                    cnt += 1
                                                routes.append((d, route))
                                    sol = make_sol(routes)
                                    logger.log_solution(ov, sol)
                                except Exception:
                                    logger.log(ov)
                    except Exception:
                        pass

            el = time.time() - prog_start
            rem = max(1.0, args.time_limit - el - 3.0)
            m.Params.TimeLimit = rem

            m.optimize(cb)

            if m.SolCount > 0:
                av = {k: v.X for k, v in a.items()}
                bv = {k: v.X for k, v in b.items()}
                cv = {k: v.X for k, v in c.items()}
                routes = []
                for d in range(num_depots):
                    for i in range(num_tasks):
                        if av.get((d, i), 0) > 0.5:
                            route = [i]
                            cur = i
                            cnt = 0
                            while cnt < num_tasks:
                                nt = None
                                for j in succ.get(cur, []):
                                    k = pidx.get((cur, j))
                                    if k is not None and cv.get((d, k), 0) > 0.5:
                                        nt = j
                                        break
                                if nt is None:
                                    break
                                route.append(nt)
                                cur = nt
                                cnt += 1
                            routes.append((d, route))
                total = sum(len(r) for _, r in routes)
                if total == num_tasks:
                    sol = make_sol(routes)
                    if sol['objective_value'] < best_obj - 1e-9:
                        best_obj = sol['objective_value']
                        best_sol = sol
                        if logger:
                            logger.log_solution(best_obj, best_sol)
        except Exception as e:
            sys.stderr.write(f"MIP error: {e}\n")
            traceback.print_exc(file=sys.stderr)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)


if __name__ == '__main__':
    main()