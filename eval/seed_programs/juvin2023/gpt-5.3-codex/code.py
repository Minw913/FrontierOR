import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional

from solution_logger import SolutionLogger


OpKey = Tuple[int, int]


def parse_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    num_jobs = int(data["num_jobs"])
    num_machines = int(data["num_machines"])
    preemption_allowed = bool(data.get("preemption_allowed", True))
    migration_allowed = bool(data.get("migration_allowed", False))

    jobs = data["jobs"]
    n_ops = [0] * num_jobs
    eligible: Dict[OpKey, List[int]] = {}
    ptime: Dict[OpKey, Dict[int, int]] = {}
    ops: List[OpKey] = []

    for job in jobs:
        j = int(job["job_id"])
        operations = job["operations"]
        n_ops[j] = len(operations)
        for op in operations:
            k = int(op["operation_id"])
            key = (j, k)
            ops.append(key)
            em = [int(m) for m in op["eligible_machines"]]
            eligible[key] = em

            pt_map_raw = op["processing_times"]
            pt_map = {}
            for m in em:
                if str(m) in pt_map_raw:
                    pt_map[m] = int(pt_map_raw[str(m)])
                elif m in pt_map_raw:
                    pt_map[m] = int(pt_map_raw[m])
                else:
                    raise ValueError(f"Missing processing time for operation {key} on machine {m}")
            ptime[key] = pt_map

    return {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "preemption_allowed": preemption_allowed,
        "migration_allowed": migration_allowed,
        "n_ops": n_ops,
        "eligible": eligible,
        "ptime": ptime,
        "ops": ops,
    }


def make_assignment_min_pt(instance) -> Dict[OpKey, int]:
    assignment = {}
    for op in instance["ops"]:
        pt_map = instance["ptime"][op]
        best_m = min(pt_map, key=lambda m: pt_map[m])
        assignment[op] = best_m
    return assignment


def make_assignment_balanced(instance, rnd: random.Random) -> Dict[OpKey, int]:
    num_machines = instance["num_machines"]
    load = [0] * num_machines
    ops = instance["ops"][:]
    rnd.shuffle(ops)

    assignment = {}
    for op in ops:
        best_m = None
        best_score = None
        for m in instance["eligible"][op]:
            p = instance["ptime"][op][m]
            score = load[m] + p
            if best_score is None or score < best_score:
                best_score = score
                best_m = m
        assignment[op] = best_m
        load[best_m] += instance["ptime"][op][best_m]
    return assignment


def make_assignment_random_biased(instance, rnd: random.Random) -> Dict[OpKey, int]:
    assignment = {}
    for op in instance["ops"]:
        machines = instance["eligible"][op]
        weights = []
        for m in machines:
            p = instance["ptime"][op][m]
            weights.append(1.0 / max(1, p))
        s = sum(weights)
        r = rnd.random() * s
        c = 0.0
        chosen = machines[-1]
        for m, w in zip(machines, weights):
            c += w
            if r <= c:
                chosen = m
                break
        assignment[op] = chosen
    return assignment


def choose_job(
    jobs_list: List[int],
    rule: str,
    rem_total: List[int],
    rem_cur: List[int],
):
    if not jobs_list:
        return None
    if rule == "SPT":
        return min(jobs_list, key=lambda j: (rem_cur[j], rem_total[j], j))
    # default: LRT (longest remaining total)
    return max(jobs_list, key=lambda j: (rem_total[j], rem_cur[j], -j))


def simulate(
    instance,
    assignment: Dict[OpKey, int],
    rule: str = "LRT",
    deadline: Optional[float] = None,
):
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    n_ops = instance["n_ops"]
    ptime = instance["ptime"]
    preemption_allowed = instance["preemption_allowed"]

    schedules: Dict[OpKey, List[int]] = {op: [] for op in instance["ops"]}

    cur_op = [0] * num_jobs
    rem_cur = [0] * num_jobs
    rem_total = [0] * num_jobs

    unfinished = 0
    for j in range(num_jobs):
        if n_ops[j] > 0:
            unfinished += 1
            first = (j, 0)
            rem_cur[j] = ptime[first][assignment[first]]
            rem_total[j] = sum(ptime[(j, k)][assignment[(j, k)]] for k in range(n_ops[j]))

    if unfinished == 0:
        return 0.0, schedules

    t = 1
    last_proc_t = 0

    # For non-preemptive mode: machine continues running same job until operation completes.
    machine_locked_job = [None] * num_machines

    while unfinished > 0:
        if deadline is not None and (t & 127) == 0 and time.time() > deadline:
            return None, None

        available_by_machine = [[] for _ in range(num_machines)]
        for j in range(num_jobs):
            k = cur_op[j]
            if k < n_ops[j] and rem_cur[j] > 0:
                m = assignment[(j, k)]
                available_by_machine[m].append(j)

        chosen = [None] * num_machines

        for m in range(num_machines):
            if preemption_allowed:
                chosen[m] = choose_job(available_by_machine[m], rule, rem_total, rem_cur)
            else:
                lj = machine_locked_job[m]
                if lj is not None:
                    # Continue locked job if still valid/current on this machine
                    k = cur_op[lj]
                    if k < n_ops[lj] and rem_cur[lj] > 0 and assignment[(lj, k)] == m:
                        chosen[m] = lj
                    else:
                        machine_locked_job[m] = None
                        chosen[m] = choose_job(available_by_machine[m], rule, rem_total, rem_cur)
                        if chosen[m] is not None:
                            machine_locked_job[m] = chosen[m]
                else:
                    chosen[m] = choose_job(available_by_machine[m], rule, rem_total, rem_cur)
                    if chosen[m] is not None:
                        machine_locked_job[m] = chosen[m]

        progressed = False
        for m in range(num_machines):
            j = chosen[m]
            if j is None:
                continue
            k = cur_op[j]
            if k >= n_ops[j] or rem_cur[j] <= 0:
                continue

            # Process one time unit
            rem_cur[j] -= 1
            rem_total[j] -= 1
            schedules[(j, k)].append(t)
            last_proc_t = t
            progressed = True

            # Operation completed
            if rem_cur[j] == 0:
                cur_op[j] += 1
                if cur_op[j] >= n_ops[j]:
                    unfinished -= 1
                else:
                    nk = cur_op[j]
                    rem_cur[j] = ptime[(j, nk)][assignment[(j, nk)]]

                if not preemption_allowed:
                    machine_locked_job[m] = None

        # Safety: if no work progressed but unfinished remains, something is inconsistent.
        if not progressed:
            # Avoid infinite loop; return a very poor objective to reject this candidate.
            return float("inf"), schedules

        t += 1

    makespan = float(last_proc_t + 1)
    return makespan, schedules


def evaluate_assignment(instance, assignment, rules, deadline=None):
    best_obj = float("inf")
    best_sched = None
    best_rule = None
    for r in rules:
        if deadline is not None and time.time() > deadline:
            break
        obj, sched = simulate(instance, assignment, rule=r, deadline=deadline)
        if obj is None:
            continue
        if obj < best_obj:
            best_obj = obj
            best_sched = sched
            best_rule = r
    return best_obj, best_sched, best_rule


def assignment_to_output_dict(assignment: Dict[OpKey, int]):
    out = {}
    for (j, k), m in assignment.items():
        out[f"({j},{k})"] = int(m)
    return out


def schedule_to_output_dict(schedule: Dict[OpKey, List[int]]):
    out = {}
    for (j, k), periods in schedule.items():
        out[f"({j},{k})"] = [int(t) for t in periods]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    deadline = start + max(1, args.time_limit)

    rnd = random.Random(0)
    instance = parse_instance(args.instance_path)

    rules = ["LRT", "SPT"]

    # Initial candidates
    candidates = [
        make_assignment_min_pt(instance),
        make_assignment_balanced(instance, rnd),
    ]
    # A few random-biased starts
    for _ in range(3):
        candidates.append(make_assignment_random_biased(instance, rnd))

    best_obj = float("inf")
    best_assignment = None
    best_schedule = None
    best_rule = "LRT"

    # Evaluate initial solutions
    for cand in candidates:
        if time.time() > deadline - 0.05:
            break
        obj, sched, rule = evaluate_assignment(instance, cand, rules, deadline=deadline)
        if sched is not None and obj < best_obj:
            best_obj = obj
            best_assignment = cand.copy()
            best_schedule = sched
            best_rule = rule if rule is not None else "LRT"
            if logger:
                logger.log(float(best_obj))

    # Fallback (should not happen)
    if best_assignment is None:
        best_assignment = make_assignment_min_pt(instance)
        best_obj, best_schedule, best_rule = evaluate_assignment(instance, best_assignment, ["LRT"], deadline=deadline)
        if best_schedule is None:
            # Last-resort empty
            best_obj = float("inf")
            best_schedule = {op: [] for op in instance["ops"]}

    ops = instance["ops"]

    # Stochastic local improvement until time limit
    while time.time() < deadline - 0.05:
        cand = best_assignment.copy()

        # Random shake: modify 1..k operations
        k_changes = min(len(ops), 1 + rnd.randint(0, 2))
        for _ in range(k_changes):
            op = ops[rnd.randrange(len(ops))]
            em = instance["eligible"][op]
            if len(em) > 1:
                cur_m = cand[op]
                alternatives = [m for m in em if m != cur_m]
                cand[op] = alternatives[rnd.randrange(len(alternatives))]

        obj, sched, _ = evaluate_assignment(instance, cand, [best_rule], deadline=deadline)
        if sched is not None and obj < best_obj:
            best_obj = obj
            best_assignment = cand
            best_schedule = sched
            if logger:
                logger.log(float(best_obj))
            continue

        # Occasional 1-flip first-improvement
        if time.time() > deadline - 0.05:
            break
        op = ops[rnd.randrange(len(ops))]
        cur_m = best_assignment[op]
        improved = False
        for m in sorted(instance["eligible"][op], key=lambda mm: instance["ptime"][op][mm]):
            if m == cur_m:
                continue
            cand2 = best_assignment.copy()
            cand2[op] = m
            obj2, sched2, _ = evaluate_assignment(instance, cand2, [best_rule], deadline=deadline)
            if sched2 is not None and obj2 < best_obj:
                best_obj = obj2
                best_assignment = cand2
                best_schedule = sched2
                improved = True
                if logger:
                    logger.log(float(best_obj))
                break
        if improved:
            continue

    output = {
        "objective_value": float(best_obj),
        "assignment": assignment_to_output_dict(best_assignment),
        "schedule": schedule_to_output_dict(best_schedule),
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()