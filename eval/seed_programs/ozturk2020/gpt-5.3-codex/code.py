import argparse
import json
import math
import random
import time
from typing import List, Dict, Any, Tuple, Optional

from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def clean_num(x: float):
    if abs(x - round(x)) <= 1e-6:
        return int(round(x))
    return float(x)


def compute_output_from_machine_batches(
    machine_batches: List[List[Dict[str, Any]]],
    job_ids: List[int],
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    batches_out = []
    job_assignments = {}
    objective = 0.0

    for m_idx, blist in enumerate(machine_batches, start=1):
        for b_idx, b in enumerate(blist, start=1):
            st = b["start_time"]
            pt = b["processing_time"]
            comp = st + pt
            ids = [job_ids[j] for j in b["jobs"]]

            batches_out.append(
                {
                    "batch_id": b_idx,
                    "machine": m_idx,
                    "start_time": clean_num(st),
                    "processing_time": clean_num(pt),
                    "jobs": ids,
                }
            )

            for jid in ids:
                job_assignments[str(jid)] = {
                    "batch": b_idx,
                    "machine": m_idx,
                    "completion_time": clean_num(comp),
                }

            objective += comp * len(ids)

    return float(objective), batches_out, job_assignments


def construct_schedule_heuristic(
    releases: List[int],
    procs: List[int],
    sizes: List[int],
    num_machines: int,
    capacity: int,
    rng: random.Random,
    mode: int = 0,
    randomized: bool = False,
) -> List[List[Dict[str, Any]]]:
    n = len(releases)
    unscheduled = set(range(n))
    machine_available = [0] * num_machines
    machine_batches = [[] for _ in range(num_machines)]

    def seed_key(j):
        if mode == 0:      # SPT-ish
            return (procs[j], releases[j], sizes[j], j)
        elif mode == 1:    # release-first
            return (releases[j], procs[j], sizes[j], j)
        else:              # size-aware
            return (procs[j] / max(1, sizes[j]), releases[j], procs[j], j)

    while unscheduled:
        m = min(range(num_machines), key=lambda mm: (machine_available[mm], len(machine_batches[mm])))
        t = machine_available[m]

        ready = [j for j in unscheduled if releases[j] <= t]
        if ready:
            pool = ready
        else:
            rmin = min(releases[j] for j in unscheduled)
            pool = [j for j in unscheduled if releases[j] == rmin]

        pool_sorted = sorted(pool, key=seed_key)
        if randomized and len(pool_sorted) > 1:
            k = min(4, len(pool_sorted))
            seed = pool_sorted[rng.randrange(k)]
        else:
            seed = pool_sorted[0]

        selected = [seed]
        rem_cap = capacity - sizes[seed]
        start = max(t, releases[seed])
        p_batch = procs[seed]

        # Add "free" jobs: no increase in start nor processing time.
        free = [j for j in unscheduled if j != seed and sizes[j] <= rem_cap and releases[j] <= start and procs[j] <= p_batch]
        free.sort(key=lambda j: (sizes[j], procs[j], releases[j], j))
        for j in free:
            if sizes[j] <= rem_cap:
                selected.append(j)
                rem_cap -= sizes[j]

        # Mildly increase processing time if cheap.
        if rem_cap > 0:
            delta_p = max(1, int(0.15 * p_batch))
            near = [
                j for j in unscheduled
                if j not in selected and sizes[j] <= rem_cap and releases[j] <= start and procs[j] <= p_batch + delta_p
            ]
            near.sort(key=lambda j: (procs[j], sizes[j], releases[j], j))
            for j in near:
                if sizes[j] > rem_cap:
                    continue
                new_p = max(p_batch, procs[j])
                # Guard against excessive delay to already selected jobs.
                extra = (new_p - p_batch) * len(selected)
                if extra <= new_p:
                    selected.append(j)
                    rem_cap -= sizes[j]
                    p_batch = new_p

        machine_batches[m].append(
            {
                "start_time": start,
                "processing_time": p_batch,
                "jobs": selected[:],
            }
        )
        machine_available[m] = start + p_batch
        unscheduled.difference_update(selected)

    return machine_batches


class MIPIncumbentLogger:
    def __init__(self, logger: Optional[SolutionLogger], best_holder: List[float]):
        self.logger = logger
        self.best_holder = best_holder

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < self.best_holder[0]:
                self.best_holder[0] = obj
                if self.logger:
                    self.logger.log(float(obj))


def try_mip_improve(
    releases: List[int],
    procs: List[int],
    sizes: List[int],
    num_machines: int,
    capacity: int,
    initial_machine_batches: List[List[Dict[str, Any]]],
    remaining_time: float,
    logger: Optional[SolutionLogger],
    global_best_obj: float,
) -> Optional[List[List[Dict[str, Any]]]]:
    n = len(releases)
    if remaining_time < 2:
        return None

    # Keep MIP moderate size.
    if n * n * num_machines > 50000:
        return None

    H = max(releases) + sum(procs) + 1
    Pmax = max(procs) if procs else 0

    model = Model("parallel_batch_schedule")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, int(remaining_time))
    model.Params.MIPFocus = 1

    J = range(n)
    M = range(num_machines)
    B = range(n)

    x = model.addVars(J, M, B, vtype=GRB.BINARY, name="x")
    y = model.addVars(M, B, vtype=GRB.BINARY, name="y")
    s = model.addVars(M, B, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    p = model.addVars(M, B, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    c = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="c")

    # Each job assigned exactly once.
    for j in J:
        model.addConstr(quicksum(x[j, m, b] for m in M for b in B) == 1)

    # Batch capacity and linking.
    for m in M:
        for b in B:
            model.addConstr(quicksum(sizes[j] * x[j, m, b] for j in J) <= capacity * y[m, b])
            model.addConstr(y[m, b] <= quicksum(x[j, m, b] for j in J))
            for j in J:
                model.addConstr(x[j, m, b] <= y[m, b])

    # Contiguous batches on each machine.
    for m in M:
        for b in range(n - 1):
            model.addConstr(y[m, b] >= y[m, b + 1])

    # Force start/proc zero when batch unused.
    for m in M:
        for b in B:
            model.addConstr(s[m, b] <= H * y[m, b])
            model.addConstr(p[m, b] <= Pmax * y[m, b])

    # Release and batch processing-time definition.
    for m in M:
        for b in B:
            for j in J:
                model.addConstr(s[m, b] >= releases[j] - H * (1 - x[j, m, b]))
                model.addConstr(p[m, b] >= procs[j] - Pmax * (1 - x[j, m, b]))

    # Sequencing on each machine (conditional on next batch used).
    for m in M:
        for b in range(n - 1):
            model.addConstr(s[m, b + 1] >= s[m, b] + p[m, b] - H * (1 - y[m, b + 1]))

    # Completion time lower bounds (objective minimizes c[j], so lower bounds are enough).
    for j in J:
        for m in M:
            for b in B:
                model.addConstr(c[j] >= s[m, b] + p[m, b] - H * (1 - x[j, m, b]))

    model.setObjective(quicksum(c[j] for j in J), GRB.MINIMIZE)

    # Warm start from heuristic.
    job_to_pos = {}
    for m, blist in enumerate(initial_machine_batches):
        for b_idx, bd in enumerate(blist):
            for j in bd["jobs"]:
                job_to_pos[j] = (m, b_idx)

    for j in J:
        if j in job_to_pos:
            m0, b0 = job_to_pos[j]
            x[j, m0, b0].Start = 1.0

    for m, blist in enumerate(initial_machine_batches):
        for b_idx, bd in enumerate(blist):
            y[m, b_idx].Start = 1.0
            s[m, b_idx].Start = float(bd["start_time"])
            p[m, b_idx].Start = float(bd["processing_time"])

    best_holder = [global_best_obj]
    cb = MIPIncumbentLogger(logger, best_holder)

    try:
        model.optimize(cb)
    except Exception:
        return None

    if model.SolCount <= 0:
        return None

    # Extract solution.
    out_batches = [[] for _ in M]
    for m in M:
        for b in B:
            if y[m, b].X > 0.5:
                jobs_mb = [j for j in J if x[j, m, b].X > 0.5]
                if jobs_mb:
                    out_batches[m].append(
                        {
                            "start_time": float(s[m, b].X),
                            "processing_time": float(p[m, b].X),
                            "jobs": jobs_mb,
                        }
                    )

    return out_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start_time = time.time()
    rng = random.Random(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["num_jobs"])
    m = int(inst["num_machines"])
    cap = int(inst["batch_capacity"])
    jobs = inst["jobs"]

    # Preserve job order as internal index.
    job_ids = [int(j["job_id"]) for j in jobs]
    releases = [int(j["release_date"]) for j in jobs]
    procs = [int(j["processing_time"]) for j in jobs]
    sizes = [int(j["size"]) for j in jobs]

    best_machine_batches = None
    best_obj = float("inf")

    # Heuristic multi-start.
    T = max(1, args.time_limit)
    heuristic_budget = T if T <= 6 else max(2.0, min(12.0, 0.35 * T))

    deterministic_modes = [0, 1, 2]
    for mode in deterministic_modes:
        if time.time() - start_time >= T:
            break
        sch = construct_schedule_heuristic(
            releases, procs, sizes, m, cap, rng, mode=mode, randomized=False
        )
        obj, _, _ = compute_output_from_machine_batches(sch, job_ids)
        if obj + 1e-9 < best_obj:
            best_obj = obj
            best_machine_batches = sch
            if logger:
                logger.log(float(best_obj))

    while time.time() - start_time < min(heuristic_budget, T):
        mode = rng.randrange(3)
        sch = construct_schedule_heuristic(
            releases, procs, sizes, m, cap, rng, mode=mode, randomized=True
        )
        obj, _, _ = compute_output_from_machine_batches(sch, job_ids)
        if obj + 1e-9 < best_obj:
            best_obj = obj
            best_machine_batches = sch
            if logger:
                logger.log(float(best_obj))

    # Optional MIP improvement if time remains.
    elapsed = time.time() - start_time
    remaining = T - elapsed
    if best_machine_batches is not None and remaining > 2:
        mip_sol = try_mip_improve(
            releases=releases,
            procs=procs,
            sizes=sizes,
            num_machines=m,
            capacity=cap,
            initial_machine_batches=best_machine_batches,
            remaining_time=remaining,
            logger=logger,
            global_best_obj=best_obj,
        )
        if mip_sol is not None:
            obj_mip, _, _ = compute_output_from_machine_batches(mip_sol, job_ids)
            if obj_mip + 1e-9 < best_obj:
                best_obj = obj_mip
                best_machine_batches = mip_sol
                if logger:
                    logger.log(float(best_obj))

    # Fallback (should not happen).
    if best_machine_batches is None:
        best_machine_batches = construct_schedule_heuristic(
            releases, procs, sizes, m, cap, rng, mode=0, randomized=False
        )
        best_obj, _, _ = compute_output_from_machine_batches(best_machine_batches, job_ids)
        if logger:
            logger.log(float(best_obj))

    objective_value, batches_out, job_assignments = compute_output_from_machine_batches(
        best_machine_batches, job_ids
    )

    solution = {
        "objective_value": float(objective_value),
        "batches": batches_out,
        "job_assignments": job_assignments,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()