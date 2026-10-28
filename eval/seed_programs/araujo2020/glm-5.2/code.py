#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    t_start = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    inst = load_instance(args.instance_path)
    projects = {p['project_id']: p for p in inst['projects']}
    jobs = {j['job_id']: j for j in inst['jobs']}
    ren_res = inst['resources']['renewable']
    nonren_res = inst['resources']['nonrenewable']
    num_ren = len(ren_res)
    num_nonren = len(nonren_res)
    ren_cap = [r['capacity'] for r in ren_res]
    nonren_cap = [r['capacity'] for r in nonren_res]

    # predecessors
    preds = {jid: [] for jid in jobs}
    for jid, job in jobs.items():
        for s in job['successors']:
            if s in preds:
                preds[s].append(jid)

    def min_dur(jid):
        return min(m['duration'] for m in jobs[jid]['modes'])

    # topological order (predecessors before successors)
    visited = {}
    order = []
    def dfs(jid):
        st = visited.get(jid, 0)
        if st == 2:
            return
        if st == 1:
            return  # cycle (should not happen)
        visited[jid] = 1
        for s in jobs[jid]['successors']:
            if s in jobs:
                dfs(s)
        visited[jid] = 2
        order.append(jid)
    for jid in jobs:
        dfs(jid)
    order.reverse()

    # earliest start (forward CPM pass with min-duration modes)
    es = {}
    for jid in order:
        job = jobs[jid]
        proj = projects[job['project_id']]
        if job['type'] == 'artificial_source':
            es[jid] = proj['release_date']
        else:
            val = proj['release_date']
            for p in preds[jid]:
                val = max(val, es[p] + min_dur(p))
            es[jid] = val

    # critical path duration per project
    cp = {}
    for pid, proj in projects.items():
        sink = proj['artificial_sink_job_id']
        cp[pid] = es[sink] - proj['release_date']

    # longest path from job to sink (includes own duration), min-duration modes
    dist_to_sink = {}
    for jid in reversed(order):
        job = jobs[jid]
        d = min_dur(jid)
        if job['successors']:
            ds = [dist_to_sink[s] for s in job['successors'] if s in dist_to_sink]
            if ds:
                d += max(ds)
        dist_to_sink[jid] = d

    # ---- Heuristic serial SGS for upper bound ----
    horizon_est = max(projects[p]['release_date'] + cp[p] for p in projects) \
                  + sum(min_dur(j) for j in jobs) + 100

    def run_heuristic():
        sched = {}
        order_h = sorted(jobs.keys(), key=lambda j: (es[j] + min_dur(j), j))
        ren_usage = defaultdict(int)
        nonren_used = [0] * num_nonren
        for jid in order_h:
            job = jobs[jid]
            modes_sorted = sorted(
                job['modes'],
                key=lambda m: (m['duration'], sum(m['renewable_consumption']), sum(m['nonrenewable_consumption']))
            )
            placed = False
            for m in modes_sorted:
                dur = m['duration']
                ren_c = m['renewable_consumption']
                nonren_c = m['nonrenewable_consumption']
                if any(nonren_used[r] + nonren_c[r] > nonren_cap[r] for r in range(num_nonren)):
                    continue
                est = es[jid]
                for p in preds[jid]:
                    if p in sched:
                        pm, pt = sched[p]
                        pdur = next(mm['duration'] for mm in jobs[p]['modes'] if mm['mode_id'] == pm)
                        est = max(est, pt + pdur)
                t = est
                while t <= horizon_est:
                    ok = True
                    for tau in range(t, t + dur):
                        for r in range(num_ren):
                            if ren_usage.get((tau, r), 0) + ren_c[r] > ren_cap[r]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        break
                    t += 1
                if t <= horizon_est:
                    sched[jid] = (m['mode_id'], t)
                    for tau in range(t, t + dur):
                        for r in range(num_ren):
                            if ren_c[r]:
                                ren_usage[(tau, r)] += ren_c[r]
                    for r in range(num_nonren):
                        nonren_used[r] += nonren_c[r]
                    placed = True
                    break
            if not placed:
                return None
        return sched

    heur = run_heuristic()

    # upper-bound completion per project
    ub_completion = {}
    if heur:
        for pid, proj in projects.items():
            sink = proj['artificial_sink_job_id']
            if sink in heur:
                ub_completion[pid] = heur[sink][1]
    for pid, proj in projects.items():
        if pid not in ub_completion or ub_completion[pid] is None:
            ub_completion[pid] = (proj['release_date'] + cp[pid]
                                  + sum(min_dur(j) for j in jobs if jobs[j]['project_id'] == pid) + 100)

    max_delay = sum(ub_completion[pid] - (projects[pid]['release_date'] + cp[pid]) for pid in projects)
    horizon = max(projects[pid]['release_date'] + cp[pid] + max_delay for pid in projects)

    # latest start per job
    ls = {}
    for jid, job in jobs.items():
        pid = job['project_id']
        val = projects[pid]['release_date'] + cp[pid] - dist_to_sink[jid] + max_delay
        if val > horizon:
            val = horizon
        if val < es[jid]:
            val = es[jid]
        ls[jid] = val

    # log heuristic incumbent
    best_logged_obj = None
    best_logged_mk = None
    if heur and logger:
        heur_obj = sum(heur[projects[pid]['artificial_sink_job_id']][1]
                       - (projects[pid]['release_date'] + cp[pid]) for pid in projects)
        heur_mk = max(heur[projects[pid]['artificial_sink_job_id']][1] for pid in projects)
        sol_dict = {
            'objective_value': float(heur_obj),
            'schedule': [{'job_id': jid, 'mode_id': heur[jid][0], 'start_time': int(heur[jid][1])} for jid in jobs],
            'makespan': int(heur_mk)
        }
        logger.log_solution(float(heur_obj), sol_dict)
        best_logged_obj = heur_obj
        best_logged_mk = heur_mk

    # ---- Build MIP ----
    elapsed = time.time() - t_start
    remaining = max(1.0, args.time_limit - elapsed - 2.0)

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.start()
    model = gp.Model(env=env)
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.TimeLimit = remaining
    model.Params.OutputFlag = 0

    mode_dur = {}
    mode_ren = {}
    mode_nonren = {}
    for jid, job in jobs.items():
        for m in job['modes']:
            mode_dur[(jid, m['mode_id'])] = m['duration']
            mode_ren[(jid, m['mode_id'])] = m['renewable_consumption']
            mode_nonren[(jid, m['mode_id'])] = m['nonrenewable_consumption']

    x = {}
    job_vars = defaultdict(list)
    for jid, job in jobs.items():
        if job['type'] == 'artificial_source':
            m = job['modes'][0]
            t = projects[job['project_id']]['release_date']
            v = model.addVar(vtype=GRB.BINARY, lb=1, ub=1, name=f'x_{jid}_{m["mode_id"]}_{t}')
            x[(jid, m['mode_id'], t)] = v
            job_vars[jid].append(v)
            continue
        e = es[jid]
        l = ls[jid]
        for m in job['modes']:
            for t in range(e, l + 1):
                v = model.addVar(vtype=GRB.BINARY, name=f'x_{jid}_{m["mode_id"]}_{t}')
                x[(jid, m['mode_id'], t)] = v
                job_vars[jid].append(v)

    start_expr = {jid: gp.LinExpr() for jid in jobs}
    dur_expr = {jid: gp.LinExpr() for jid in jobs}
    for (jid, mid, t), v in x.items():
        start_expr[jid].addTerms(t, v)
        dur_expr[jid].addTerms(mode_dur[(jid, mid)], v)

    # mode selection
    for jid in jobs:
        model.addConstr(gp.quicksum(job_vars[jid]) == 1)

    # precedence
    for jid, job in jobs.items():
        for p in preds[jid]:
            model.addConstr(start_expr[p] + dur_expr[p] <= start_expr[jid])

    # renewable resources
    ren_expr = defaultdict(lambda: defaultdict(gp.LinExpr))
    for (jid, mid, t), v in x.items():
        dur = mode_dur[(jid, mid)]
        rc = mode_ren[(jid, mid)]
        if dur == 0:
            continue
        for tau in range(t, t + dur):
            for r in range(num_ren):
                if rc[r]:
                    ren_expr[tau][r].addTerms(rc[r], v)
    for tau, rd in ren_expr.items():
        for r, expr in rd.items():
            model.addConstr(expr <= ren_cap[r])

    # nonrenewable resources
    if num_nonren:
        nonren_expr = [gp.LinExpr() for _ in range(num_nonren)]
        for (jid, mid, t), v in x.items():
            nc = mode_nonren[(jid, mid)]
            for r in range(num_nonren):
                if nc[r]:
                    nonren_expr[r].addTerms(nc[r], v)
        for r in range(num_nonren):
            model.addConstr(nonren_expr[r] <= nonren_cap[r])

    # objective
    makespan = model.addVar(vtype=GRB.INTEGER, name='makespan')
    delays = []
    for pid, proj in projects.items():
        sink = proj['artificial_sink_job_id']
        comp = start_expr[sink]
        delay = comp - (proj['release_date'] + cp[pid])
        delays.append(delay)
        model.addConstr(makespan >= comp)
    epsilon = 1e-5
    model.setObjective(gp.quicksum(delays) + epsilon * makespan, GRB.MINIMIZE)

    # MIP start
    if heur:
        for jid, (mid, t) in heur.items():
            v = x.get((jid, mid, t))
            if v is not None:
                v.Start = 1

    # incumbent callback for logging
    var_keys = list(x.keys())
    var_list = [x[k] for k in var_keys]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(var_list)
            except Exception:
                return
            chosen = {}
            for k, val in zip(var_keys, vals):
                if val is not None and val > 0.5:
                    jid, mid, t = k
                    chosen[jid] = (mid, t)
            # need all jobs chosen
            if len(chosen) != len(jobs):
                return
            total_delay = sum(chosen[projects[pid]['artificial_sink_job_id']][1]
                              - (projects[pid]['release_date'] + cp[pid]) for pid in projects)
            mk = max(chosen[projects[pid]['artificial_sink_job_id']][1] for pid in projects)
            nonlocal best_logged_obj, best_logged_mk
            improved = False
            if best_logged_obj is None or total_delay < best_logged_obj - 1e-9:
                improved = True
            elif abs(total_delay - best_logged_obj) < 1e-9:
                if best_logged_mk is None or mk < best_logged_mk:
                    improved = True
            if improved and logger:
                sol_dict = {
                    'objective_value': float(total_delay),
                    'schedule': [{'job_id': jid, 'mode_id': chosen[jid][0], 'start_time': int(chosen[jid][1])} for jid in jobs],
                    'makespan': int(mk)
                }
                logger.log_solution(float(total_delay), sol_dict)
                best_logged_obj = total_delay
                best_logged_mk = mk

    try:
        model.setCallback(callback)
    except Exception:
        pass

    model.optimize()

    # extract MIP solution
    best_obj = None
    best_sched = None
    best_mk = None
    if model.SolCount > 0:
        chosen = {}
        for jid in jobs:
            for m in jobs[jid]['modes']:
                e = es[jid]
                l = ls[jid]
                for t in range(e, l + 1):
                    v = x.get((jid, m['mode_id'], t))
                    if v is not None and v.X is not None and v.X > 0.5:
                        chosen[jid] = (m['mode_id'], t)
                        break
                if jid in chosen:
                    break
        if len(chosen) == len(jobs):
            total_delay = sum(chosen[projects[pid]['artificial_sink_job_id']][1]
                              - (projects[pid]['release_date'] + cp[pid]) for pid in projects)
            mk = max(chosen[projects[pid]['artificial_sink_job_id']][1] for pid in projects)
            best_obj = total_delay
            best_sched = chosen
            best_mk = mk

    # compare with heuristic
    if heur:
        heur_obj = sum(heur[projects[pid]['artificial_sink_job_id']][1]
                       - (projects[pid]['release_date'] + cp[pid]) for pid in projects)
        heur_mk = max(heur[projects[pid]['artificial_sink_job_id']][1] for pid in projects)
        if best_obj is None or heur_obj < best_obj - 1e-9 or \
           (abs(heur_obj - best_obj) < 1e-9 and heur_mk < best_mk):
            best_obj = heur_obj
            best_sched = heur
            best_mk = heur_mk

    if best_sched is None:
        # last resort: trivial schedule ignoring resources (may be infeasible but provides output)
        best_sched = {}
        for jid in order:
            job = jobs[jid]
            if job['type'] == 'artificial_source':
                t = projects[job['project_id']]['release_date']
                best_sched[jid] = (job['modes'][0]['mode_id'], t)
            else:
                est = es[jid]
                for p in preds[jid]:
                    if p in best_sched:
                        pm, pt = best_sched[p]
                        pdur = next(mm['duration'] for mm in jobs[p]['modes'] if mm['mode_id'] == pm)
                        est = max(est, pt + pdur)
                m = min(job['modes'], key=lambda mm: mm['duration'])
                best_sched[jid] = (m['mode_id'], est)
        best_obj = sum(best_sched[projects[pid]['artificial_sink_job_id']][1]
                       - (projects[pid]['release_date'] + cp[pid]) for pid in projects)
        best_mk = max(best_sched[projects[pid]['artificial_sink_job_id']][1] for pid in projects)

    schedule_list = [{'job_id': jid, 'mode_id': best_sched[jid][0], 'start_time': int(best_sched[jid][1])}
                     for jid in jobs]
    out = {
        'objective_value': float(best_obj),
        'schedule': schedule_list,
        'makespan': int(best_mk)
    }
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

    if logger:
        improved = False
        if best_logged_obj is None or best_obj < best_logged_obj - 1e-9:
            improved = True
        elif abs(best_obj - best_logged_obj) < 1e-9:
            if best_logged_mk is None or best_mk < best_logged_mk:
                improved = True
        if improved:
            logger.log_solution(float(best_obj), out)


if __name__ == '__main__':
    main()