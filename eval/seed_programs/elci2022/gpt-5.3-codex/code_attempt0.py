import argparse
import json
import time
import random
import heapq
from typing import List, Dict, Tuple, Set, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

from solution_logger import SolutionLogger


def schedule_facility_best(
    jobs: List[int],
    f: int,
    s: int,
    capacity: List[int],
    req: List[List[int]],
    releases: List[int],
    proc_times: List[List[List[int]]],
) -> Tuple[int, Dict[int, Tuple[int, int]]]:
    """Heuristic scheduling for one facility and one scenario; tries multiple priority rules."""
    if not jobs:
        return 0, {}

    C = capacity[f]
    qf = req[f]
    pf = proc_times[s][f]

    # Priority rules (lower tuple is better)
    def score(rule_id: int, j: int):
        p = pf[j]
        q = qf[j]
        r = releases[j]
        if rule_id == 0:      # LPT
            return (-p, -q, r, j)
        elif rule_id == 1:    # SPT
            return (p, -q, r, j)
        elif rule_id == 2:    # area
            return (-(p * q), -p, r, j)
        else:                 # high demand first
            return (-q, -p, r, j)

    def simulate(rule_id: int) -> Tuple[int, Dict[int, Tuple[int, int]]]:
        rel_sorted = sorted(jobs, key=lambda j: releases[j])
        n_local = len(rel_sorted)
        idx = 0
        t = releases[rel_sorted[0]]
        available = []
        running = []  # (finish, job, demand)
        used = 0
        sched = {}
        started = 0

        while started < n_local or running:
            # Complete jobs
            while running and running[0][0] <= t:
                fin, jdone, dem = heapq.heappop(running)
                used -= dem

            # Release jobs
            while idx < n_local and releases[rel_sorted[idx]] <= t:
                available.append(rel_sorted[idx])
                idx += 1

            # Start as many as possible
            while True:
                free = C - used
                best = None
                best_key = None

                for j in available:
                    p = pf[j]
                    q = qf[j]
                    if q <= free or p == 0:
                        k = score(rule_id, j)
                        if best is None or k < best_key:
                            best = j
                            best_key = k

                if best is None:
                    break

                st = t
                p = pf[best]
                q = qf[best]
                fi = st + p
                sched[best] = (st, fi)
                available.remove(best)
                started += 1

                if p > 0 and q > 0:
                    heapq.heappush(running, (fi, best, q))
                    used += q

            if started >= n_local and not running:
                break

            next_t = None
            if running:
                next_t = running[0][0]
            if idx < n_local:
                nr = releases[rel_sorted[idx]]
                if next_t is None or nr < next_t:
                    next_t = nr

            if next_t is None:
                # Should not happen in feasible instances with q<=C.
                # Force-finish leftover jobs at current time to avoid infinite loop.
                for j in available:
                    st = t
                    fi = t + pf[j]
                    sched[j] = (st, fi)
                    started += 1
                break

            if next_t == t:
                next_t = t + 1
            t = next_t

        cmax = max((fi for (_, fi) in sched.values()), default=0)
        return cmax, sched

    best_cmax = None
    best_sched = None
    for rid in range(4):
        cmax, sched = simulate(rid)
        if best_cmax is None or cmax < best_cmax:
            best_cmax = cmax
            best_sched = sched

    return best_cmax if best_cmax is not None else 0, best_sched if best_sched is not None else {}


def build_initial_assignment(
    n: int,
    m: int,
    S: int,
    probs: List[float],
    capacity: List[int],
    req: List[List[int]],
    releases: List[int],
    proc_times: List[List[List[int]]],
    feasible_facilities: List[List[int]],
    time_limit_for_mip: float,
) -> List[int]:
    # Try surrogate MIP assignment first
    if GUROBI_AVAILABLE and time_limit_for_mip >= 1.0:
        try:
            model = gp.Model("surrogate_assignment")
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = time_limit_for_mip

            x = {}
            for f in range(m):
                for j in range(n):
                    ub = 1.0 if f in feasible_facilities[j] else 0.0
                    x[f, j] = model.addVar(vtype=GRB.BINARY, ub=ub, name=f"x_{f}_{j}")

            T = [model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{s}") for s in range(S)]

            for j in range(n):
                model.addConstr(gp.quicksum(x[f, j] for f in range(m)) == 1, name=f"assign_{j}")

            for s in range(S):
                for j in range(n):
                    model.addConstr(
                        T[s] >= releases[j] + gp.quicksum(proc_times[s][f][j] * x[f, j] for f in range(m)),
                        name=f"job_lb_s{s}_j{j}"
                    )

                for f in range(m):
                    if capacity[f] > 0:
                        model.addConstr(
                            T[s] >= gp.quicksum(req[f][j] * proc_times[s][f][j] * x[f, j] for j in range(n)) / float(capacity[f]),
                            name=f"energy_lb_s{s}_f{f}"
                        )

            model.setObjective(gp.quicksum(probs[s] * T[s] for s in range(S)), GRB.MINIMIZE)
            model.optimize()

            if model.SolCount > 0:
                assignment = [-1] * n
                for j in range(n):
                    best_f = None
                    best_val = -1.0
                    for f in range(m):
                        val = x[f, j].X
                        if val > best_val:
                            best_val = val
                            best_f = f
                    if best_f is None or best_f not in feasible_facilities[j]:
                        best_f = feasible_facilities[j][0]
                    assignment[j] = best_f
                return assignment
        except Exception:
            pass

    # Greedy fallback
    exp_p = [[0.0] * n for _ in range(m)]
    for f in range(m):
        for j in range(n):
            exp_p[f][j] = sum(probs[s] * proc_times[s][f][j] for s in range(S))

    loads = [0.0] * m
    jobs_order = list(range(n))
    jobs_order.sort(
        key=lambda j: max((req[f][j] * exp_p[f][j] / max(1, capacity[f]) for f in feasible_facilities[j]), default=0.0),
        reverse=True
    )

    assignment = [-1] * n
    for j in jobs_order:
        best_f = None
        best_score = None
        for f in feasible_facilities[j]:
            incr = req[f][j] * exp_p[f][j] / max(1, capacity[f])
            score = loads[f] + incr
            if best_f is None or score < best_score:
                best_f = f
                best_score = score
        assignment[j] = best_f
        loads[best_f] += req[best_f][j] * exp_p[best_f][j] / max(1, capacity[best_f])

    return assignment


def evaluate_full(
    assignment: List[int],
    n: int,
    m: int,
    S: int,
    probs: List[float],
    capacity: List[int],
    req: List[List[int]],
    releases: List[int],
    proc_times: List[List[List[int]]],
):
    jobs_by_f = [set() for _ in range(m)]
    for j, f in enumerate(assignment):
        jobs_by_f[f].add(j)

    facility_cmax = [[0] * m for _ in range(S)]
    facility_sched = [[{} for _ in range(m)] for _ in range(S)]
    scen_makespan = [0] * S

    obj = 0.0
    for s in range(S):
        smax = 0
        for f in range(m):
            cmax, sched = schedule_facility_best(
                list(jobs_by_f[f]), f, s, capacity, req, releases, proc_times
            )
            facility_cmax[s][f] = cmax
            facility_sched[s][f] = sched
            if cmax > smax:
                smax = cmax
        scen_makespan[s] = smax
        obj += probs[s] * smax

    return obj, jobs_by_f, facility_cmax, facility_sched, scen_makespan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    rng = random.Random(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    m = int(inst["m"])
    S = int(inst["num_scenarios"])

    capacity = list(map(int, inst["facility_capacity_limits"]))
    req = inst["capacity_requirements"]
    releases = list(map(int, inst["release_times"]))

    scenarios = inst["scenarios"]
    probs = [float(sc["probability"]) for sc in scenarios]
    proc_times = [sc["processing_times"] for sc in scenarios]  # [S][m][n]

    # Feasible facilities per job (non-preemptive positive-duration jobs need q<=C)
    feasible_facilities = []
    for j in range(n):
        feas = [f for f in range(m) if req[f][j] <= capacity[f]]
        if not feas:
            # Fallback to least violation facility (should not happen in valid instances)
            best_f = max(range(m), key=lambda ff: capacity[ff] - req[ff][j])
            feas = [best_f]
        feasible_facilities.append(feas)

    total_time_limit = max(1, int(args.time_limit))
    remaining = lambda: total_time_limit - (time.time() - start_time)

    mip_time = min(30.0, max(0.0, total_time_limit * 0.2))
    assignment = build_initial_assignment(
        n, m, S, probs, capacity, req, releases, proc_times, feasible_facilities, mip_time
    )

    current_obj, jobs_by_f, facility_cmax, facility_sched, scen_makespan = evaluate_full(
        assignment, n, m, S, probs, capacity, req, releases, proc_times
    )
    best_obj = current_obj
    if logger:
        logger.log(best_obj)

    # Local search: first-improvement relocation moves
    eps = 1e-9
    while remaining() > 0.5:
        improved = False
        job_order = list(range(n))
        rng.shuffle(job_order)

        for j in job_order:
            if remaining() <= 0.5:
                break

            f_old = assignment[j]
            candidates = [f for f in feasible_facilities[j] if f != f_old]
            # mild heuristic ordering
            candidates.sort(key=lambda f: req[f][j] / max(1, capacity[f]))
            rng.shuffle(candidates)

            accepted = False
            for f_new in candidates:
                if remaining() <= 0.5:
                    break

                # Prepare temporary job sets
                set_old = jobs_by_f[f_old]
                set_new = jobs_by_f[f_new]

                temp_old_jobs = list(set_old - {j})
                temp_new_jobs = list(set_new | {j})

                move_obj = 0.0
                per_s_data = []
                feasible_move = True

                for s in range(S):
                    c1, sched1 = schedule_facility_best(
                        temp_old_jobs, f_old, s, capacity, req, releases, proc_times
                    )
                    c2, sched2 = schedule_facility_best(
                        temp_new_jobs, f_new, s, capacity, req, releases, proc_times
                    )

                    other_max = 0
                    for f in range(m):
                        if f != f_old and f != f_new:
                            cf = facility_cmax[s][f]
                            if cf > other_max:
                                other_max = cf

                    new_smax = max(other_max, c1, c2)
                    move_obj += probs[s] * new_smax
                    per_s_data.append((c1, sched1, c2, sched2, new_smax))

                    if move_obj >= current_obj - eps:
                        # cannot improve anymore (nonnegative remaining contributions)
                        feasible_move = False
                        break

                if feasible_move and move_obj + eps < current_obj:
                    # Accept
                    assignment[j] = f_new
                    jobs_by_f[f_old].remove(j)
                    jobs_by_f[f_new].add(j)

                    for s in range(S):
                        c1, sched1, c2, sched2, new_smax = per_s_data[s]
                        facility_cmax[s][f_old] = c1
                        facility_sched[s][f_old] = sched1
                        facility_cmax[s][f_new] = c2
                        facility_sched[s][f_new] = sched2
                        scen_makespan[s] = new_smax

                    current_obj = move_obj
                    if current_obj + eps < best_obj:
                        best_obj = current_obj
                        if logger:
                            logger.log(best_obj)
                    improved = True
                    accepted = True
                    break

            if accepted:
                break

        if not improved:
            break

    # Build output schedule (scenario -> job)
    output_schedule = {}
    for s in range(S):
        scen_dict = {}
        for f in range(m):
            for j, (st, fi) in facility_sched[s][f].items():
                scen_dict[str(j)] = {
                    "facility": int(f),
                    "start": int(st),
                    "finish": int(fi),
                }

        # Safety fill in case any job missing
        for j in range(n):
            sj = str(j)
            if sj not in scen_dict:
                f = assignment[j]
                p = proc_times[s][f][j]
                st = releases[j]
                scen_dict[sj] = {
                    "facility": int(f),
                    "start": int(st),
                    "finish": int(st + p),
                }

        output_schedule[str(s)] = scen_dict

    output_assignment = {str(j): int(assignment[j]) for j in range(n)}

    sol = {
        "objective_value": float(current_obj),
        "assignment": output_assignment,
        "schedule": output_schedule,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()