import argparse
import json
from collections import deque
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    projects = data['projects']
    resources = data['resources']
    jobs = data['jobs']
    
    # Map job_id to job
    job_dict = {job['job_id']: job for job in jobs}
    
    # Build project_jobs: project_id -> list of jobs
    project_jobs = {}
    for job in jobs:
        pid = job['project_id']
        if pid not in project_jobs:
            project_jobs[pid] = []
        project_jobs[pid].append(job)
    
    # Precompute min_d for each job
    for job in jobs:
        min_d = float('inf')
        for mode in job['modes']:
            if mode['duration'] < min_d:
                min_d = mode['duration']
        job['min_d'] = min_d
    
    # Precompute max_d for global_delay
    total_max_duration = 0
    for job in jobs:
        max_d = 0
        for mode in job['modes']:
            if mode['duration'] > max_d:
                max_d = mode['duration']
        total_max_duration += max_d
    num_projects = len(projects)
    global_delay = num_projects * total_max_duration
    
    # For each project, do forward and backward passes
    for proj in projects:
        pid = proj['project_id']
        pjobs = project_jobs.get(pid, [])
        release_date = proj['release_date']
        source_id = proj['artificial_source_job_id']
        sink_id = proj['artificial_sink_job_id']
        
        # Build graph and predecessors
        graph = {}
        pred = {}
        for job in pjobs:
            jid = job['job_id']
            graph[jid] = job['successors']
            pred[jid] = []
        for jid in graph:
            for s in graph[jid]:
                if s in pred:
                    pred[s].append(jid)
                else:
                    pred[s] = [jid]
        
        # Compute in-degrees for topological sort
        in_degree = {}
        for job in pjobs:
            jid = job['job_id']
            in_degree[jid] = len(pred.get(jid, []))
        
        # Topological sort
        q = deque()
        topo_order = []
        for job in pjobs:
            jid = job['job_id']
            if in_degree[jid] == 0:
                q.append(jid)
        while q:
            u = q.popleft()
            topo_order.append(u)
            for v in graph.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        
        # Forward pass (EST)
        EST = {}
        for jid in topo_order:
            if jid == source_id:
                EST[jid] = release_date
            else:
                max_pred = 0
                for i in pred.get(jid, []):
                    if i in EST:
                        candidate = EST[i] + job_dict[i]['min_d']
                        if candidate > max_pred:
                            max_pred = candidate
                EST[jid] = max(release_date, max_pred)
        proj['base_completion'] = EST.get(sink_id, release_date)
    
    # Compute T_max
    T_max = 0
    for proj in projects:
        deadline_p = proj['base_completion'] + global_delay
        if deadline_p > T_max:
            T_max = deadline_p
    T_max = int(T_max)
    
    # Backward pass (LST for minimal mode)
    for proj in projects:
        pid = proj['project_id']
        pjobs = project_jobs.get(pid, [])
        sink_id = proj['artificial_sink_job_id']
        deadline_p = proj['base_completion'] + global_delay
        rev_topo = list(reversed(topo_order))
        LST = {}
        for jid in rev_topo:
            if jid == sink_id:
                LST[jid] = deadline_p
            else:
                min_succ = float('inf')
                for s in graph.get(jid, []):
                    if s in LST and LST[s] < min_succ:
                        min_succ = LST[s]
                if min_succ == float('inf'):
                    min_succ = deadline_p
                LST[jid] = min_succ - job_dict[jid]['min_d']
        for jid in LST:
            job_dict[jid]['LST_minimal'] = LST[jid]
        for job in pjobs:
            jid = job['job_id']
            job['EST'] = EST[jid]
    
    # Compute time windows for each job and mode
    for job in jobs:
        min_d_j = job['min_d']
        job['modes_time_windows'] = []
        for mode in job['modes']:
            d = mode['duration']
            if 'LST_minimal' in job:
                LST_m = job['LST_minimal'] - (d - min_d_j)
                EST_j = job['EST']
                if LST_m < EST_j:
                    job['modes_time_windows'].append(None)
                else:
                    job['modes_time_windows'].append((int(EST_j), int(LST_m)))
            else:
                job['modes_time_windows'].append(None)
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build MIP model
    model = gp.Model()
    model.setParam("Threads", 1)
    x = {}
    for job in jobs:
        job_id = job['job_id']
        for idx, mode in enumerate(job['modes']):
            mode_id = mode['mode_id']
            tw = job['modes_time_windows'][idx]
            if tw is not None:
                est, lst = tw
                for t in range(est, lst + 1):
                    x[(job_id, mode_id, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{job_id}_{mode_id}_{t}")
    
    M = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T_max, name="makespan")
    
    # Assignment constraints
    for job in jobs:
        job_id = job['job_id']
        vars_job = []
        for idx, mode in enumerate(job['modes']):
            mode_id = mode['mode_id']
            tw = job['modes_time_windows'][idx]
            if tw is not None:
                est, lst = tw
                for t in range(est, lst + 1):
                    if (job_id, mode_id, t) in x:
                        vars_job.append(x[(job_id, mode_id, t)])
        if vars_job:
            model.addConstr(gp.quicksum(vars_job) == 1, name=f"assign_{job_id}")
    
    # Precedence constraints
    for job in jobs:
        job_id = job['job_id']
        for succ_id in job['successors']:
            expr = 0
            # succ_id's start time
            if succ_id in job_dict:
                succ_job = job_dict[succ_id]
                for idx_j, mode_j in enumerate(succ_job['modes']):
                    mode_id_j = mode_j['mode_id']
                    tw_j = succ_job['modes_time_windows'][idx_j]
                    if tw_j is not None:
                        est_j, lst_j = tw_j
                        for t_j in range(est_j, lst_j + 1):
                            if (succ_id, mode_id_j, t_j) in x:
                                expr += t_j * x[(succ_id, mode_id_j, t_j)]
            # job_id's start time and duration
            for idx_i, mode_i in enumerate(job['modes']):
                mode_id_i = mode_i['mode_id']
                d_i = mode_i['duration']
                tw_i = job['modes_time_windows'][idx_i]
                if tw_i is not None:
                    est_i, lst_i = tw_i
                    for t_i in range(est_i, lst_i + 1):
                        if (job_id, mode_id_i, t_i) in x:
                            expr -= t_i * x[(job_id, mode_id_i, t_i)]
                            expr -= d_i * x[(job_id, mode_id_i, t_i)]
            model.addConstr(expr >= 0, name=f"prec_{job_id}_{succ_id}")
    
    # Nonrenewable resources
    for k_idx, res in enumerate(resources['nonrenewable']):
        cap = res['capacity']
        expr = 0
        for job in jobs:
            job_id = job['job_id']
            for idx, mode in enumerate(job['modes']):
                mode_id = mode['mode_id']
                tw = job['modes_time_windows'][idx]
                if tw is not None:
                    cons = mode['nonrenewable_consumption'][k_idx]
                    est, lst = tw
                    for t in range(est, lst + 1):
                        if (job_id, mode_id, t) in x:
                            expr += cons * x[(job_id, mode_id, t)]
        model.addConstr(expr <= cap, name=f"nonrenew_{k_idx}")
    
    # Renewable resources
    for k_idx, res in enumerate(resources['renewable']):
        cap = res['capacity']
        for u in range(0, T_max):
            expr = 0
            for job in jobs:
                job_id = job['job_id']
                for idx, mode in enumerate(job['modes']):
                    mode_id = mode['mode_id']
                    d = mode['duration']
                    tw = job['modes_time_windows'][idx]
                    if tw is None:
                        continue
                    est, lst = tw
                    cons = mode['renewable_consumption'][k_idx]
                    t_low = max(est, u - d + 1)
                    t_high = min(lst, u)
                    if t_low <= t_high:
                        for t in range(t_low, t_high + 1):
                            if (job_id, mode_id, t) in x:
                                expr += cons * x[(job_id, mode_id, t)]
            if expr.size() > 0:
                model.addConstr(expr <= cap, name=f"renew_{u}_{k_idx}")
    
    # Makespan constraints
    S = 0
    for proj in projects:
        sink_id = proj['artificial_sink_job_id']
        sink_job = job_dict[sink_id]
        mode_id = sink_job['modes'][0]['mode_id']
        for idx, mode in enumerate(sink_job['modes']):
            if mode['mode_id'] == mode_id:
                tw = sink_job['modes_time_windows'][idx]
                if tw is not None:
                    est, lst = tw
                    for t in range(est, lst + 1):
                        if (sink_id, mode_id, t) in x:
                            var = x[(sink_id, mode_id, t)]
                            S += t * var
                            model.addConstr(M >= t * var)
    
    model.setObjective((T_max + 1) * S + M, GRB.MINIMIZE)
    model.setParam('TimeLimit', time_limit)
    
    # Callback for logging
    if logger:
        proj_sinks = []
        for proj in projects:
            sink_id = proj['artificial_sink_job_id']
            base_completion = proj['base_completion']
            proj_sinks.append((sink_id, base_completion))
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                total_delay = 0
                for sink_id, base_completion in proj_sinks:
                    sink_job = job_dict[sink_id]
                    mode_id = sink_job['modes'][0]['mode_id']
                    for idx, mode in enumerate(sink_job['modes']):
                        if mode['mode_id'] == mode_id:
                            tw = sink_job['modes_time_windows'][idx]
                            if tw is None:
                                continue
                            est, lst = tw
                            for t in range(est, lst + 1):
                                if (sink_id, mode_id, t) in x:
                                    val = model.cbGetSolution(x[(sink_id, mode_id, t)])
                                    if val > 0.5:
                                        total_delay += (t - base_completion)
                                        break
                logger.log(total_delay)
        model.optimize(callback)
    else:
        model.optimize()
    
    # Extract solution
    if model.status == GRB.INFEASIBLE or model.SolCount == 0:
        schedule = []
        for job in jobs:
            job_id = job['job_id']
            min_d = float('inf')
            min_mode = None
            for mode in job['modes']:
                if mode['duration'] < min_d:
                    min_d = mode['duration']
                    min_mode = mode
            mode_id = min_mode['mode_id']
            start_time = job.get('EST', 0)
            schedule.append({'job_id': job_id, 'mode_id': mode_id, 'start_time': start_time})
    else:
        schedule = []
        for job in jobs:
            job_id = job['job_id']
            scheduled = False
            for idx, mode in enumerate(job['modes']):
                mode_id = mode['mode_id']
                tw = job['modes_time_windows'][idx]
                if tw is None:
                    continue
                est, lst = tw
                for t in range(est, lst + 1):
                    key = (job_id, mode_id, t)
                    if key in x and x[key].X > 0.5:
                        schedule.append({'job_id': job_id, 'mode_id': mode_id, 'start_time': t})
                        scheduled = True
                        break
                if scheduled:
                    break
            if not scheduled:
                min_d = float('inf')
                min_mode = None
                for mode in job['modes']:
                    if mode['duration'] < min_d:
                        min_d = mode['duration']
                        min_mode = mode
                mode_id = min_mode['mode_id']
                start_time = job.get('EST', 0)
                schedule.append({'job_id': job_id, 'mode_id': mode_id, 'start_time': start_time})
    
    # Compute objective_value and makespan
    total_delay = 0
    makespan = 0
    for proj in projects:
        sink_id = proj['artificial_sink_job_id']
        for s in schedule:
            if s['job_id'] == sink_id:
                s_p = s['start_time']
                break
        base_completion = proj['base_completion']
        total_delay += (s_p - base_completion)
        if s_p > makespan:
            makespan = s_p
    
    output = {
        "objective_value": total_delay,
        "schedule": schedule,
        "makespan": makespan
    }
    
    with open(solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)