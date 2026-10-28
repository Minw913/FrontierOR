import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    projects = data['projects']
    resources = data['resources']
    jobs = data['jobs']

    renewable = resources['renewable']
    nonrenewable = resources['nonrenewable']
    R = len(renewable)
    N = len(nonrenewable)

    # Build job and mode data
    job_dict = {j['job_id']: j for j in jobs}
    project_dict = {p['project_id']: p for p in projects}

    # Compute critical path durations (ignoring resources)
    # Use longest path in DAG with min duration per job
    def get_min_duration(jid):
        j = job_dict[jid]
        if not j['modes']:
            return 0
        return min(m['duration'] for m in j['modes'])

    # Topological order for forward and backward passes
    from collections import defaultdict, deque
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for j in jobs:
        for succ in j['successors']:
            successors[j['job_id']].append(succ)
            predecessors[succ].append(j['job_id'])

    all_job_ids = [j['job_id'] for j in jobs]

    # Topological sort
    indeg = {jid: len(predecessors[jid]) for jid in all_job_ids}
    topo = []
    q = deque([jid for jid in all_job_ids if indeg[jid] == 0])
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # Earliest start (forward)
    EST = {}
    for jid in topo:
        if not predecessors[jid]:
            EST[jid] = 0
        else:
            EST[jid] = max(EST[pred] + get_min_duration(pred) for pred in predecessors[jid])

    # Critical path duration per project: max over jobs in project of (EST[j] + min_dur[j] + longest path from j to sink)
    # But for project CPD = longest path from source to sink using min durations
    cpd = {}
    for p in projects:
        source = p['artificial_source_job_id']
        sink = p['artificial_sink_job_id']
        # Longest path to sink from each node
        dist_to_sink = {}
        for jid in reversed(topo):
            if not successors[jid]:
                dist_to_sink[jid] = 0
            else:
                dist_to_sink[jid] = max(get_min_duration(jid) + dist_to_sink[s] for s in successors[jid])
        cpd[p['project_id']] = dist_to_sink[source]
        # Also update EST with release date
        rel = p['release_date']
        for jid in all_job_ids:
            if job_dict[jid]['project_id'] == p['project_id']:
                EST[jid] = max(EST[jid], rel)

    # Upper bound on completion: for now, use a simple heuristic upper bound: max EST + sum all min_durs + some slack
    # For the problem description, we need max allowable delay. We will set a reasonable T.
    max_T = 0
    for p in projects:
        pid = p['project_id']
        max_T = max(max_T, p['release_date'] + cpd[pid] * 2)  # rough upper

    # To follow description, we need latest start times.
    # Compute for each job the longest path from job to sink using min durations (actually shortest? wait)
    # The description says: "minus the total duration of the longest path from that job to the project's artificial end job computed using the shortest-duration modes"
    # Longest path with shortest-duration? It probably means using min duration for each job on the path, i.e. critical path length from that job onward.

    longest_from_job_to_sink = {}
    for jid in reversed(topo):
        j = job_dict[jid]
        if not successors[jid]:
            longest_from_job_to_sink[jid] = 0
        else:
            longest_from_job_to_sink[jid] = get_min_duration(jid) + max(longest_from_job_to_sink[s] for s in successors[jid])

    # The description gives a way to compute latest start using release + cpd - longest_from_job + max_allowable_total_delay
    # max_allowable_total_project_delay = sum (UB_p - (release_p + CPD_p))
    # We don't have known solution UB, so we must set a reasonable one.
    # Let's set max_delay_factor = 1.5 or simply make T large enough.
    total_max_delay = 0
    for p in projects:
        pid = p['project_id']
        total_max_delay += cpd[pid]  # allow up to 100% delay per project for feasibility

    T = max(EST[jid] + longest_from_job_to_sink[jid] for jid in all_job_ids) + total_max_delay + 1
    T = max(T, 200)  # safety

    # Latest start per job-mode
    LST = {}
    for j in jobs:
        jid = j['job_id']
        pid = j['project_id']
        p = project_dict[pid]
        rel = p['release_date']
        cpd_p = cpd[pid]
        L_p = rel + cpd_p + total_max_delay  # rough
        for m_idx, m in enumerate(j['modes']):
            dur = m['duration']
            lst = L_p - longest_from_job_to_sink[jid] + total_max_delay
            LST[(jid, m['mode_id'])] = min(lst, T - dur)
            # Also respect EST
            earliest = EST[jid]
            LST[(jid, m['mode_id'])] = max(earliest, LST.get((jid, m['mode_id']), earliest))

    # Now collect all possible (job, mode, start_time) variables where feasible
    possible = {}
    job_modes = defaultdict(list)
    for j in jobs:
        jid = j['job_id']
        for mdata in j['modes']:
            mid = mdata['mode_id']
            dur = mdata['duration']
            est = EST[jid]
            lst_val = LST.get((jid, mid), est)
            for t in range(est, lst_val + 1):
                possible[(jid, mid, t)] = True
                job_modes[jid].append((mid, t))

    # Build model
    model = gp.Model("MRCPSP_MultiProject_Delay")
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit - 5  # safety margin
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    # Variables: x[job, mode, t] = 1 if job starts in that mode at that time
    x = {}
    for key in possible:
        jid, mid, t = key
        x[key] = model.addVar(vtype=GRB.BINARY, name=f"x_{jid}_{mid}_{t}")

    # Exactly one mode and start time per job
    for jid in all_job_ids:
        vars_for_job = [x[k] for k in possible if k[0] == jid]
        if vars_for_job:
            model.addConstr(gp.quicksum(vars_for_job) == 1, name=f"assign_{jid}")

    # Precedence constraints
    for j in jobs:
        jid = j['job_id']
        for succ_id in j['successors']:
            # For all possible (m,t) for j and (m2,s) for succ, if t + dur > s then forbid both =1
            for k1 in possible:
                if k1[0] != jid: continue
                _, m1, t1 = k1
                dur1 = job_dict[jid]['modes'][m1]['duration'] if isinstance(m1, int) else next(mm['duration'] for mm in job_dict[jid]['modes'] if mm['mode_id']==m1)
                finish1 = t1 + dur1
                for k2 in possible:
                    if k2[0] != succ_id: continue
                    _, m2, t2 = k2
                    if finish1 > t2:
                        model.addConstr(x[k1] + x[k2] <= 1, name=f"prec_{k1}_{k2}")

    # Resource constraints
    # Renewable
    for r_idx, res in enumerate(renewable):
        cap = res['capacity']
        for tau in range(T):
            usage = []
            for key, var in x.items():
                jid, mid, tstart = key
                j = job_dict[jid]
                mode_data = next((m for m in j['modes'] if m['mode_id'] == mid), None)
                if mode_data is None:
                    continue
                dur = mode_data['duration']
                cons = mode_data['renewable_consumption'][r_idx]
                if cons > 0 and tstart <= tau < tstart + dur:
                    usage.append(cons * var)
            if usage:
                model.addConstr(gp.quicksum(usage) <= cap, name=f"renew_{r_idx}_{tau}")

    # Non-renewable
    for n_idx, nres in enumerate(nonrenewable):
        cap = nres['capacity']
        total_use = []
        for key, var in x.items():
            jid, mid, _ = key
            j = job_dict[jid]
            mode_data = next((m for m in j['modes'] if m['mode_id'] == mid), None)
            if mode_data:
                cons = mode_data['nonrenewable_consumption'][n_idx]
                total_use.append(cons * var)
        if total_use:
            model.addConstr(gp.quicksum(total_use) <= cap, name=f"nonren_{n_idx}")

    # Objective
    # For each project, completion = start time of artificial_sink
    proj_delay = {}
    sink_start = {}
    for p in projects:
        pid = p['project_id']
        sink = p['artificial_sink_job_id']
        # Get the variable for sink. Since duration=0, start_time is completion time
        sink_vars = [ (t, x[(sink, m['mode_id'], t)]) for mdata in job_dict[sink]['modes'] for t in range(T) if (sink, mdata['mode_id'], t) in x ]
        # Since exactly one, we can create expr for start time
        start_expr = gp.quicksum(t * v for t, v in sink_vars)
        sink_start[pid] = start_expr
        release = p['release_date']
        cpd_p = cpd[pid]
        proj_delay[pid] = start_expr - (release + cpd_p)

    total_delay = gp.quicksum(proj_delay.values())

    # Makespan
    makespan = model.addVar(vtype=GRB.INTEGER, name="makespan", lb=0, ub=T)
    for pid in sink_start:
        model.addConstr(makespan >= sink_start[pid])

    # Objective: total_delay + 1e-6 * makespan
    epsilon = 1e-6
    objective = total_delay + epsilon * makespan
    model.setObjective(objective, GRB.MINIMIZE)

    # Callback to log improved solutions
    best_obj = None
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and model.SolCount > 0:
        schedule = []
        for key, var in x.items():
            if var.X > 0.5:
                jid, mid, t = key
                schedule.append({
                    "job_id": int(jid),
                    "mode_id": int(mid),
                    "start_time": int(t)
                })

        # Compute true objective
        proj_completion = {}
        for p in projects:
            pid = p['project_id']
            sink = p['artificial_sink_job_id']
            for entry in schedule:
                if entry['job_id'] == sink:
                    proj_completion[pid] = entry['start_time']
                    break
        total_delay_val = 0.0
        for p in projects:
            pid = p['project_id']
            release = p['release_date']
            cpd_p = cpd[pid]
            total_delay_val += proj_completion[pid] - (release + cpd_p)

        makespan_val = max((entry['start_time'] for entry in schedule if job_dict[entry['job_id']]['type'] == 'artificial_sink'), default=0)

        result = {
            "objective_value": float(total_delay_val),
            "schedule": schedule,
            "makespan": int(makespan_val)
        }
    else:
        # No solution found - return empty with high objective (should not happen if feasible)
        result = {
            "objective_value": 1e6,
            "schedule": [],
            "makespan": 0
        }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    if logger and result["objective_value"] < 1e5:
        logger.log(result["objective_value"])

if __name__ == "__main__":
    main()