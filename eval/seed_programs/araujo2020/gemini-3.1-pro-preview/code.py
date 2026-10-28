import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Init logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    projects = instance.get('projects', [])
    resources = instance.get('resources', {})
    jobs_list = instance.get('jobs', [])

    proj_dict = {p['project_id']: p for p in projects}
    jobs_dict = {j['job_id']: j for j in jobs_list}

    # Minimum durations
    d_min = {}
    for j in jobs_list:
        d_min[j['job_id']] = min(m['duration'] for m in j['modes'])

    # Predecessors
    preds = {j['job_id']: [] for j in jobs_list}
    for j in jobs_list:
        for succ in j['successors']:
            preds[succ].append(j['job_id'])

    # Topological sort of all jobs
    in_degree = {j: len(preds[j]) for j in preds}
    queue = [j for j in in_degree if in_degree[j] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        job_u = jobs_dict[u]
        for v in job_u['successors']:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Compute Earliest Start Times (EST_min)
    EST_min = {}
    for u in topo_order:
        job = jobs_dict[u]
        p_id = job['project_id']
        est = proj_dict[p_id]['release_date']
        for p_u in preds[u]:
            est = max(est, EST_min[p_u] + d_min[p_u])
        EST_min[u] = est

    # Compute Longest Path to sink excluding job (Tail)
    Tail = {}
    for u in reversed(topo_order):
        job = jobs_dict[u]
        if job['type'] == 'artificial_sink':
            Tail[u] = 0
        else:
            if not job['successors']:
                Tail[u] = 0
            else:
                Tail[u] = max(Tail[v] + d_min[v] for v in job['successors'])

    # Heuristic formulation to find feasible mode mapping and upper bound
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m_heu = gp.Model("HeuristicModeSelection", env=env)
    
    y = {}
    for j in jobs_list:
        for mode in j['modes']:
            y[j['job_id'], mode['mode_id']] = m_heu.addVar(vtype=GRB.BINARY)

    for j in jobs_list:
        m_heu.addConstr(gp.quicksum(y[j['job_id'], mode['mode_id']] for mode in j['modes']) == 1)

    for r_idx, r in enumerate(resources.get('nonrenewable', [])):
        m_heu.addConstr(
            gp.quicksum(
                mode['nonrenewable_consumption'][r_idx] * y[j['job_id'], mode['mode_id']]
                for j in jobs_list for mode in j['modes']
            ) <= r['capacity']
        )

    m_heu.setObjective(
        gp.quicksum(
            mode['duration'] * y[j['job_id'], mode['mode_id']]
            for j in jobs_list for mode in j['modes']
        ), GRB.MINIMIZE
    )
    
    m_heu.optimize()

    # SGS to establish initial feasible solution
    mstar = {}
    dstar = {}
    if m_heu.Status == GRB.OPTIMAL or m_heu.Status == GRB.SUBOPTIMAL:
        for j in jobs_list:
            for mode in j['modes']:
                if y[j['job_id'], mode['mode_id']].X > 0.5:
                    mstar[j['job_id']] = mode['mode_id']
                    dstar[j['job_id']] = mode['duration']
                    break
    else:
        # Fallback if infeasible just to set dummy defaults
        for j in jobs_list:
            mstar[j['job_id']] = j['modes'][0]['mode_id']
            dstar[j['job_id']] = j['modes'][0]['duration']

    start_times_heu = {}
    completion_times = {}
    active_renewables = {}
    ren_cap = {r['resource_id']: r['capacity'] for r in resources.get('renewable', [])}

    finished_count = {j['job_id']: 0 for j in jobs_list}
    # Initial eligible: jobs with strictly 0 predecessors
    eligible = set(j for j in topo_order if len(preds[j]) == 0)

    while eligible:
        best_j = None
        best_est = float('inf')
        best_idx = -1
        for j in eligible:
            p_id = jobs_dict[j]['project_id']
            est = proj_dict[p_id]['release_date']
            for p in preds[j]:
                est = max(est, completion_times[p])
            
            if est < best_est or (est == best_est and topo_order.index(j) < best_idx) or best_j is None:
                best_j = j
                best_est = est
                best_idx = topo_order.index(j)
                
        j = best_j
        eligible.remove(j)
        
        dur = dstar[j]
        m_id = mstar[j]
        mode = next(m for m in jobs_dict[j]['modes'] if m['mode_id'] == m_id)
        
        t = best_est
        if dur > 0:
            while True:
                feasible = True
                for dt in range(dur):
                    time_pt = t + dt
                    for idx, r_val in enumerate(mode['renewable_consumption']):
                        if r_val > 0:
                            r_id = resources['renewable'][idx]['resource_id']
                            if active_renewables.get(time_pt, {}).get(r_id, 0) + r_val > ren_cap[r_id]:
                                feasible = False
                                break
                    if not feasible:
                        break
                if feasible:
                    break
                t += 1
            
            for dt in range(dur):
                time_pt = t + dt
                if time_pt not in active_renewables:
                    active_renewables[time_pt] = {}
                for idx, r_val in enumerate(mode['renewable_consumption']):
                    if r_val > 0:
                        r_id = resources['renewable'][idx]['resource_id']
                        active_renewables[time_pt][r_id] = active_renewables[time_pt].get(r_id, 0) + r_val

        start_times_heu[j] = t
        completion_times[j] = t + dur
        
        for v in jobs_dict[j]['successors']:
            finished_count[v] += 1
            if finished_count[v] == len(preds[v]):
                eligible.add(v)

    # Compute MAD
    MAD = 0
    if m_heu.Status == GRB.OPTIMAL or m_heu.Status == GRB.SUBOPTIMAL:
        for p in projects:
            snk = p['artificial_sink_job_id']
            ubc = start_times_heu[snk]
            rel = p['release_date']
            cpd = EST_min[snk] - rel
            MAD += (ubc - (rel + cpd))
    else:
        MAD = sum(max(m['duration'] for m in j['modes']) for j in jobs_list) * 2

    LST = {}
    for j in topo_order:
        job = jobs_dict[j]
        p_id = job['project_id']
        p = proj_dict[p_id]
        
        rel = p['release_date']
        cpd = EST_min[p['artificial_sink_job_id']] - rel
        lct = rel + cpd + MAD - Tail[j]
        
        LST[j] = {}
        for mode in job['modes']:
            LST[j][mode['mode_id']] = lct - mode['duration']

    main_env = gp.Env(empty=True)
    main_env.start()
    model = gp.Model("MRCPSP", env=main_env)
    model.setParam('TimeLimit', max(1.0, args.time_limit - 1.5))
    model.setParam('Threads', 1)

    # Time-indexed Variables
    x = {}
    max_H = 0
    min_EST = float('inf')
    for j in topo_order:
        job = jobs_dict[j]
        for mode in job['modes']:
            m_id = mode['mode_id']
            start_t = EST_min[j]
            end_t = LST[j][m_id]
            if end_t >= start_t:
                max_H = max(max_H, end_t + mode['duration'])
                min_EST = min(min_EST, start_t)
                for t in range(start_t, end_t + 1):
                    x[j, m_id, t] = model.addVar(vtype=GRB.BINARY)

    # Exactly one mode and start time per job
    for j in topo_order:
        model.addConstr(
            gp.quicksum(x[j, m['mode_id'], t]
                        for m in jobs_dict[j]['modes']
                        for t in range(EST_min[j], LST[j][m['mode_id']] + 1)
                        if (j, m['mode_id'], t) in x
                       ) == 1
        )

    # MIP Start
    for j in topo_order:
        m_id = mstar[j]
        t_val = start_times_heu.get(j)
        if t_val is not None and (j, m_id, t_val) in x:
            x[j, m_id, t_val].Start = 1.0

    # Incremental Precedence Constraints
    for k in topo_order:
        max_t_k = -1
        for m in jobs_dict[k]['modes']:
            m_id = m['mode_id']
            if LST[k][m_id] >= EST_min[k]:
                max_t_k = max(max_t_k, LST[k][m_id])
                
        if max_t_k == -1:
            continue
            
        for j in preds[k]:
            lhs_vars_by_tau = {}
            for m in jobs_dict[k]['modes']:
                m_id = m['mode_id']
                for tau in range(EST_min[k], LST[k][m_id] + 1):
                    if (k, m_id, tau) in x:
                        lhs_vars_by_tau.setdefault(tau, []).append(x[k, m_id, tau])
                        
            rhs_vars_by_finish = {}
            for m in jobs_dict[j]['modes']:
                m_id = m['mode_id']
                for tau in range(EST_min[j], LST[j][m_id] + 1):
                    if (j, m_id, tau) in x:
                        fin = tau + m['duration']
                        rhs_vars_by_finish.setdefault(fin, []).append(x[j, m_id, tau])
                        
            t_min = min(EST_min[k], EST_min[j])
            lhs_sum = []
            rhs_sum = []
            for t in range(t_min, int(max_t_k) + 1):
                if t in lhs_vars_by_tau:
                    lhs_sum.extend(lhs_vars_by_tau[t])
                if t in rhs_vars_by_finish:
                    rhs_sum.extend(rhs_vars_by_finish[t])
                if t >= EST_min[k] and lhs_sum:
                    model.addConstr(gp.quicksum(lhs_sum) <= gp.quicksum(rhs_sum))

    # Renewable resources
    ren_capacities = resources.get('renewable', [])
    if ren_capacities and min_EST != float('inf'):
        ren_exprs = { (r_idx, t): [] for r_idx in range(len(ren_capacities)) for t in range(min_EST, max_H) }
        for j in topo_order:
            job = jobs_dict[j]
            for m in job['modes']:
                m_id = m['mode_id']
                dur = m['duration']
                if dur == 0:
                    continue
                c_rens = m['renewable_consumption']
                non_zero_rens = [r_idx for r_idx, c in enumerate(c_rens) if c > 0]
                if not non_zero_rens:
                    continue
                for tau in range(EST_min[j], LST[j][m_id] + 1):
                    if (j, m_id, tau) not in x:
                        continue
                    v = x[j, m_id, tau]
                    end_time = tau + dur - 1
                    for t in range(tau, end_time + 1):
                        if t < max_H:
                            for r_idx in non_zero_rens:
                                ren_exprs[(r_idx, t)].append(c_rens[r_idx] * v)

        for (r_idx, t), expr_list in ren_exprs.items():
            if expr_list:
                model.addConstr(gp.quicksum(expr_list) <= ren_capacities[r_idx]['capacity'])

    # Nonrenewable resources
    nonren_capacities = resources.get('nonrenewable', [])
    if nonren_capacities:
        nonren_exprs = { r_idx: [] for r_idx in range(len(nonren_capacities)) }
        for j in topo_order:
            job = jobs_dict[j]
            for m in job['modes']:
                m_id = m['mode_id']
                c_nonrens = m['nonrenewable_consumption']
                non_zero_nonrens = [r_idx for r_idx, c in enumerate(c_nonrens) if c > 0]
                if not non_zero_nonrens:
                    continue
                for tau in range(EST_min[j], LST[j][m_id] + 1):
                    if (j, m_id, tau) in x:
                        v = x[j, m_id, tau]
                        for r_idx in non_zero_nonrens:
                            nonren_exprs[r_idx].append(c_nonrens[r_idx] * v)

        for r_idx, expr_list in nonren_exprs.items():
            if expr_list:
                model.addConstr(gp.quicksum(expr_list) <= nonren_capacities[r_idx]['capacity'])

    # Makespan and Objective
    makespan = model.addVar(vtype=GRB.INTEGER, name="makespan")
    total_delay = []
    
    for p in projects:
        snk = p['artificial_sink_job_id']
        rel = p['release_date']
        cpd = EST_min[snk] - rel
        S_snk = gp.quicksum(tau * x[snk, m['mode_id'], tau]
                            for m in jobs_dict[snk]['modes']
                            for tau in range(EST_min[snk], LST[snk][m['mode_id']] + 1)
                            if (snk, m['mode_id'], tau) in x)
        model.addConstr(makespan >= S_snk)
        total_delay.append(S_snk - rel - cpd)

    eps = 1.0 / (2 * max_H + 1) if max_H > 0 else 0.001
    model.setObjective(gp.quicksum(total_delay) + eps * makespan, GRB.MINIMIZE)

    # Optimization with callback
    def cb(mod, where):
        if where == GRB.Callback.MIPSOL:
            obj = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            m_val = mod.cbGetSolution(makespan)
            delay_val = float(round(obj - eps * m_val))
            if logger:
                logger.log(delay_val)

    model.optimize(cb)

    # Extract Schedule
    schedule_out = []
    if model.SolCount > 0:
        for j in topo_order:
            for m in jobs_dict[j]['modes']:
                m_id = m['mode_id']
                if LST[j][m_id] < EST_min[j]:
                    continue
                for t in range(EST_min[j], LST[j][m_id] + 1):
                    if (j, m_id, t) in x and x[j, m_id, t].X > 0.5:
                        schedule_out.append({
                            "job_id": j,
                            "mode_id": m_id,
                            "start_time": t
                        })
    else:
        # Fallback to heuristic
        for j in topo_order:
            schedule_out.append({
                "job_id": j,
                "mode_id": mstar[j],
                "start_time": start_times_heu[j]
            })

    final_delay_val = 0.0
    makespan_val = 0
    for p in projects:
        snk = p['artificial_sink_job_id']
        s_time = next(s['start_time'] for s in schedule_out if s['job_id'] == snk)
        final_delay_val += float(s_time - p['release_date'] - (EST_min[snk] - p['release_date']))
        if s_time > makespan_val:
            makespan_val = s_time

    ans = {
        "objective_value": float(final_delay_val),
        "schedule": schedule_out,
        "makespan": int(makespan_val)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(ans, f, indent=2)

if __name__ == "__main__":
    main()