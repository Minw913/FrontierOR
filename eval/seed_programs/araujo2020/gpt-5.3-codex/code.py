import argparse
import json
import math
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def topological_order(job_ids, successors, predecessors):
    indeg = {j: len(predecessors[j]) for j in job_ids}
    q = deque([j for j in job_ids if indeg[j] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != len(job_ids):
        # Fallback if graph has cycle (should not happen in valid instances)
        return job_ids[:]
    return topo


def compute_cp_and_tail(project_jobs, source, sink, successors, predecessors, dmin):
    # Forward longest path from source (ignoring resources)
    indeg = {j: 0 for j in project_jobs}
    for j in project_jobs:
        for p in predecessors[j]:
            if p in indeg:
                indeg[j] += 1
    q = deque([j for j in project_jobs if indeg[j] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            if v in indeg:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    if len(topo) != len(project_jobs):
        topo = list(project_jobs)

    est_rel = {j: -10**12 for j in project_jobs}
    if source in est_rel:
        est_rel[source] = 0
    else:
        # If source missing unexpectedly
        for j in topo:
            est_rel[j] = 0

    for u in topo:
        if est_rel[u] < -10**11:
            est_rel[u] = 0
        finish_u = est_rel[u] + dmin[u]
        for v in successors[u]:
            if v in est_rel and finish_u > est_rel[v]:
                est_rel[v] = finish_u

    cp = est_rel.get(sink, 0)

    # Reverse longest path from job to sink (using dmin), includes own duration except sink->0
    outdeg = {j: 0 for j in project_jobs}
    for j in project_jobs:
        for v in successors[j]:
            if v in outdeg:
                outdeg[j] += 1
    rq = deque([j for j in project_jobs if outdeg[j] == 0])
    rev_topo = []
    while rq:
        u = rq.popleft()
        rev_topo.append(u)
        for p in predecessors[u]:
            if p in outdeg:
                outdeg[p] -= 1
                if outdeg[p] == 0:
                    rq.append(p)

    if len(rev_topo) != len(project_jobs):
        rev_topo = list(project_jobs)

    tail = {j: -10**12 for j in project_jobs}
    if sink in tail:
        tail[sink] = 0
    for u in rev_topo:
        if u == sink:
            continue
        best_succ = 0
        has_succ_in_proj = False
        for v in successors[u]:
            if v in tail and tail[v] > -10**11:
                has_succ_in_proj = True
                if tail[v] > best_succ:
                    best_succ = tail[v]
        if has_succ_in_proj:
            tail[u] = dmin[u] + best_succ
        else:
            # disconnected fallback
            tail[u] = dmin[u]

    return cp, est_rel, tail


def solve_mode_selection_mip(jobs_data, nonrenew_caps, time_limit_sec):
    # Multiple-choice mode selection considering only nonrenewable resources.
    model = gp.Model("mode_select")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit_sec))

    y = {}
    for j, jd in jobs_data.items():
        for m_idx, md in enumerate(jd["modes"]):
            y[(j, m_idx)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{m_idx}")

    for j, jd in jobs_data.items():
        model.addConstr(gp.quicksum(y[(j, m_idx)] for m_idx in range(len(jd["modes"]))) == 1)

    nr_count = len(nonrenew_caps)
    for r in range(nr_count):
        model.addConstr(
            gp.quicksum(
                jd["modes"][m_idx]["nonrenewable_consumption"][r] * y[(j, m_idx)]
                for j, jd in jobs_data.items()
                for m_idx in range(len(jd["modes"]))
            ) <= nonrenew_caps[r]
        )

    model.setObjective(
        gp.quicksum(
            jd["modes"][m_idx]["duration"] * y[(j, m_idx)]
            for j, jd in jobs_data.items()
            for m_idx in range(len(jd["modes"]))
        ),
        GRB.MINIMIZE
    )
    model.optimize()

    if model.SolCount > 0:
        chosen = {}
        for j, jd in jobs_data.items():
            best = 0
            for m_idx in range(len(jd["modes"])):
                if y[(j, m_idx)].X > 0.5:
                    best = m_idx
                    break
            chosen[j] = best
        return chosen
    return None


def serial_sgs(jobs_data, topo, predecessors, release_by_project, renew_caps, chosen_mode_idx):
    r_count = len(renew_caps)
    starts = {}
    usage = [defaultdict(int) for _ in range(r_count)]

    for j in topo:
        jd = jobs_data[j]
        p = jd["project_id"]
        md = jd["modes"][chosen_mode_idx[j]]
        dur = md["duration"]
        req = md["renewable_consumption"]

        est = release_by_project[p]
        for pr in predecessors[j]:
            pr_md = jobs_data[pr]["modes"][chosen_mode_idx[pr]]
            est = max(est, starts[pr] + pr_md["duration"])

        if dur == 0:
            starts[j] = est
            continue

        t = est
        while True:
            feasible = True
            for tau in range(t, t + dur):
                for k in range(r_count):
                    if usage[k][tau] + req[k] > renew_caps[k]:
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                starts[j] = t
                for tau in range(t, t + dur):
                    for k in range(r_count):
                        usage[k][tau] += req[k]
                break
            t += 1
            if t > est + 200000:
                # emergency fallback
                starts[j] = t
                break
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    projects = inst["projects"]
    jobs = inst["jobs"]
    renew_res = inst["resources"]["renewable"]
    nonrenew_res = inst["resources"]["nonrenewable"]

    renew_caps = [r["capacity"] for r in renew_res]
    nonrenew_caps = [r["capacity"] for r in nonrenew_res]

    jobs_data = {j["job_id"]: j for j in jobs}
    job_ids = list(jobs_data.keys())

    successors = {j: list(jobs_data[j]["successors"]) for j in job_ids}
    predecessors = {j: [] for j in job_ids}
    for j in job_ids:
        for s in successors[j]:
            predecessors[s].append(j)

    project_of = {j["job_id"]: j["project_id"] for j in jobs}
    release_by_project = {p["project_id"]: p["release_date"] for p in projects}
    sink_by_project = {p["project_id"]: p["artificial_sink_job_id"] for p in projects}
    source_by_project = {p["project_id"]: p["artificial_source_job_id"] for p in projects}

    # min/max durations
    dmin = {}
    dmax = {}
    for j, jd in jobs_data.items():
        ds = [m["duration"] for m in jd["modes"]]
        dmin[j] = min(ds)
        dmax[j] = max(ds)

    # Global topological order
    topo = topological_order(job_ids, successors, predecessors)

    # CP and tails per project
    jobs_by_project = defaultdict(list)
    for j in job_ids:
        jobs_by_project[project_of[j]].append(j)

    cp_by_project = {}
    est_rel_project = {}
    tail_project = {}

    for p in projects:
        pid = p["project_id"]
        pj = jobs_by_project[pid]
        cp, est_rel, tail = compute_cp_and_tail(
            set(pj),
            source_by_project[pid],
            sink_by_project[pid],
            successors,
            predecessors,
            dmin
        )
        cp_by_project[pid] = cp
        est_rel_project[pid] = est_rel
        tail_project[pid] = tail

    # global EST with precedence + release
    est_global = {j: release_by_project[project_of[j]] for j in job_ids}
    for j in topo:
        est_j = max(est_global[j], release_by_project[project_of[j]])
        for pr in predecessors[j]:
            est_j = max(est_j, est_global[pr] + dmin[pr])
        est_global[j] = est_j

    # Safe horizon
    H = max(release_by_project.values()) + sum(dmax[j] for j in job_ids)

    # Heuristic mode selection & SGS for warm start
    remain = max(1, args.time_limit - int(time.time() - t0))
    mode_choice = solve_mode_selection_mip(jobs_data, nonrenew_caps, time_limit_sec=min(20, remain))
    if mode_choice is None:
        mode_choice = {}
        for j, jd in jobs_data.items():
            best = min(range(len(jd["modes"])), key=lambda i: jd["modes"][i]["duration"])
            mode_choice[j] = best

    heur_starts = serial_sgs(
        jobs_data=jobs_data,
        topo=topo,
        predecessors=predecessors,
        release_by_project=release_by_project,
        renew_caps=renew_caps,
        chosen_mode_idx=mode_choice
    )

    # Build full MIP
    model = gp.Model("MRCMPSP")
    model.Params.TimeLimit = max(1, args.time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    x = {}
    s_var = {}

    # For renewable constraint assembly
    renew_terms = [defaultdict(list) for _ in range(len(renew_caps))]

    # Precompute "remaining downstream min duration after job", per project
    rem_after = {}
    for j in job_ids:
        pid = project_of[j]
        tval = tail_project[pid].get(j, dmin[j])
        rem_after[j] = max(0, tval - dmin[j])

    # vars
    for j in job_ids:
        s_var[j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=H, name=f"s_{j}")

    for j in job_ids:
        jd = jobs_data[j]
        pid = jd["project_id"]
        est = max(0, est_global[j], release_by_project[pid])

        for m_idx, md in enumerate(jd["modes"]):
            dur = md["duration"]
            # Safe latest start from horizon and project-tail tightening:
            lst = H - dur
            lst = min(lst, H - (dur + rem_after[j]))
            if lst < est:
                continue

            for t in range(est, lst + 1):
                v = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m_idx}_{t}")
                x[(j, m_idx, t)] = v

                # renewable occupancy
                if dur > 0:
                    req = md["renewable_consumption"]
                    for tau in range(t, t + dur):
                        for k in range(len(renew_caps)):
                            c = req[k]
                            if c:
                                renew_terms[k][tau].append((c, v))

    model.update()

    # Each job exactly once
    for j, jd in jobs_data.items():
        vars_j = [x[key] for key in x.keys() if key[0] == j]
        if not vars_j:
            # emergency: if no feasible var window due to tight bounds, add one at EST with shortest mode
            best_m = min(range(len(jd["modes"])), key=lambda i: jd["modes"][i]["duration"])
            est = max(0, est_global[j], release_by_project[jd["project_id"]])
            t = min(est, H - jd["modes"][best_m]["duration"])
            if t < 0:
                t = 0
            v = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{best_m}_{t}")
            x[(j, best_m, t)] = v
            vars_j = [v]
        model.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")

    model.update()

    # Start time definition
    for j in job_ids:
        expr = gp.quicksum(t * var for (jj, mm, t), var in x.items() if jj == j)
        model.addConstr(s_var[j] == expr, name=f"startdef_{j}")
        model.addConstr(s_var[j] >= release_by_project[project_of[j]], name=f"release_{j}")

    # Precedence
    dur_expr = {}
    for j in job_ids:
        dur_expr[j] = gp.quicksum(
            jobs_data[j]["modes"][m_idx]["duration"] * var
            for (jj, m_idx, t), var in x.items() if jj == j
        )

    for i in job_ids:
        for j in successors[i]:
            model.addConstr(s_var[i] + dur_expr[i] <= s_var[j], name=f"prec_{i}_{j}")

    # Renewable capacities
    for k in range(len(renew_caps)):
        cap = renew_caps[k]
        for tau, items in renew_terms[k].items():
            model.addConstr(gp.quicksum(c * v for c, v in items) <= cap, name=f"R_{k}_{tau}")

    # Nonrenewable capacities
    for r in range(len(nonrenew_caps)):
        cap = nonrenew_caps[r]
        expr = gp.quicksum(
            jobs_data[j]["modes"][m_idx]["nonrenewable_consumption"][r] * var
            for (j, m_idx, t), var in x.items()
        )
        model.addConstr(expr <= cap, name=f"N_{r}")

    # Project delays and makespan
    cmax = model.addVar(vtype=GRB.INTEGER, lb=0, ub=H, name="makespan")
    for p in projects:
        pid = p["project_id"]
        sink = sink_by_project[pid]
        model.addConstr(cmax >= s_var[sink], name=f"mk_{pid}")

    delay_expr = gp.quicksum(
        s_var[sink_by_project[p["project_id"]]] - (p["release_date"] + cp_by_project[p["project_id"]])
        for p in projects
    )

    eps = 1.0 / (H + 1.0)
    model.setObjective(delay_expr + eps * cmax, GRB.MINIMIZE)

    # Warm start
    for j in job_ids:
        if j in heur_starts and j in mode_choice:
            m0 = mode_choice[j]
            t0s = heur_starts[j]
            key = (j, m0, t0s)
            if key in x:
                x[key].Start = 1.0
                s_var[j].Start = t0s

    # Logger callback
    model._logger = logger
    model._best_logged = math.inf

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-9 < m._best_logged:
                m._best_logged = val
                m._logger.log(float(val))

    model.optimize(cb)

    # Extract best solution
    solution = {"objective_value": 1e18, "schedule": [], "makespan": 0}

    if model.SolCount > 0:
        schedule = []
        for j in job_ids:
            chosen = None
            for (jj, m_idx, t), var in x.items():
                if jj == j and var.X > 0.5:
                    chosen = (m_idx, t)
                    break
            if chosen is None:
                # fallback from s_var
                m_idx = 0
                t = int(round(s_var[j].X))
                chosen = (m_idx, t)
            schedule.append({
                "job_id": int(j),
                "mode_id": int(jobs_data[j]["modes"][chosen[0]]["mode_id"]),
                "start_time": int(chosen[1])
            })

        # primary objective (without epsilon tie-break)
        obj_delay = 0.0
        for p in projects:
            pid = p["project_id"]
            sink = sink_by_project[pid]
            comp = int(round(s_var[sink].X))
            obj_delay += comp - (p["release_date"] + cp_by_project[pid])

        mk = int(round(cmax.X))
        solution = {
            "objective_value": float(obj_delay),
            "schedule": schedule,
            "makespan": mk
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()