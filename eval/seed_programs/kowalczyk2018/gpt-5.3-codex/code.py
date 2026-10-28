import argparse
import json
import random
import time
from typing import List, Tuple

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve P||sum w_j C_j")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def ratio_key(p: int, w: int) -> float:
    if p == 0:
        return float("inf") if w > 0 else 0.0
    return w / p


def machine_schedule_and_cost(job_ids: List[int], p: List[int], w: List[int], r: List[float]):
    # Smith rule order on each machine: nonincreasing w/p
    sids = sorted(job_ids, key=lambda j: (-r[j], p[j], -w[j], j))
    t = 0
    cost = 0
    comps = []
    for j in sids:
        t += p[j]
        cost += w[j] * t
        comps.append(t)
    return sids, comps, cost


def full_eval(assignment: List[int], m: int, p: List[int], w: List[int], r: List[float]):
    machine_jobs = [[] for _ in range(m)]
    for j, k in enumerate(assignment):
        machine_jobs[k].append(j)

    machine_sorted = []
    machine_comps = []
    machine_costs = []
    total = 0
    for k in range(m):
        sids, comps, c = machine_schedule_and_cost(machine_jobs[k], p, w, r)
        machine_sorted.append(sids)
        machine_comps.append(comps)
        machine_costs.append(c)
        total += c
    return total, machine_jobs, machine_sorted, machine_comps, machine_costs


def greedy_construction(order: List[int], m: int, p: List[int], w: List[int], r: List[float], alpha: float, rng: random.Random):
    machine_jobs = [[] for _ in range(m)]
    machine_costs = [0] * m

    for j in order:
        deltas = []
        for k in range(m):
            cand = machine_jobs[k] + [j]
            _, _, new_cost = machine_schedule_and_cost(cand, p, w, r)
            delta = new_cost - machine_costs[k]
            deltas.append((delta, k))

        deltas.sort(key=lambda x: x[0])
        if alpha <= 1e-12:
            chosen_k = deltas[0][1]
        else:
            rcl_size = max(1, int(alpha * m))
            rcl_size = min(rcl_size, m)
            chosen_k = deltas[rng.randrange(rcl_size)][1]

        machine_jobs[chosen_k].append(j)
        _, _, machine_costs[chosen_k] = machine_schedule_and_cost(machine_jobs[chosen_k], p, w, r)

    assignment = [0] * len(p)
    for k in range(m):
        for j in machine_jobs[k]:
            assignment[j] = k
    return assignment


def local_search_relocate(
    assignment: List[int],
    m: int,
    p: List[int],
    w: List[int],
    r: List[float],
    start_obj: int,
    best_global_obj: int,
    logger,
    deadline: float,
):
    n = len(assignment)
    current_obj, machine_jobs, _, _, machine_costs = full_eval(assignment, m, p, w, r)

    if start_obj is not None:
        current_obj = start_obj

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for j in range(n):
            if time.time() >= deadline:
                break
            a = assignment[j]
            if len(machine_jobs[a]) <= 1:
                continue

            base_a_cost = machine_costs[a]
            jobs_a_wo = [x for x in machine_jobs[a] if x != j]
            _, _, new_a_cost = machine_schedule_and_cost(jobs_a_wo, p, w, r)

            best_move = None  # (delta, b, new_b_cost)
            for b in range(m):
                if b == a:
                    continue
                jobs_b_w = machine_jobs[b] + [j]
                _, _, new_b_cost = machine_schedule_and_cost(jobs_b_w, p, w, r)
                delta = (new_a_cost + new_b_cost) - (base_a_cost + machine_costs[b])
                if (best_move is None) or (delta < best_move[0]):
                    best_move = (delta, b, new_b_cost)

            if best_move is not None and best_move[0] < 0:
                _, b, new_b_cost = best_move
                # apply move
                machine_jobs[a].remove(j)
                machine_jobs[b].append(j)
                assignment[j] = b
                machine_costs[a] = new_a_cost
                machine_costs[b] = new_b_cost
                current_obj += best_move[0]
                improved = True

                if current_obj < best_global_obj:
                    best_global_obj = current_obj
                    if logger:
                        logger.log(float(best_global_obj))
                break  # first-improvement restart

    return assignment, int(current_obj), int(best_global_obj)


def build_output(assignment: List[int], m: int, jobs_input: List[dict], p: List[int], w: List[int], r: List[float]):
    total, machine_jobs, machine_sorted, machine_comps, machine_costs = full_eval(assignment, m, p, w, r)

    machines_out = []
    for k in range(m):
        jobs_out = []
        sids = machine_sorted[k]
        comps = machine_comps[k]
        for idx, j in enumerate(sids):
            jobs_out.append(
                {
                    "j_orig": int(jobs_input[j]["j"]),
                    "p": int(p[j]),
                    "w": int(w[j]),
                    "C_j": int(comps[idx]),
                }
            )
        machines_out.append(
            {
                "schedule_cost": int(machine_costs[k]),
                "jobs": jobs_out,
            }
        )

    return {
        "objective_value": float(total),
        "machines": machines_out,
    }


def main():
    args = parse_args()
    t0 = time.time()
    deadline = t0 + args.time_limit

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    m = int(inst["m"])
    jobs_in = inst["jobs"]

    if n == 0:
        sol = {"objective_value": 0.0, "machines": [{"schedule_cost": 0, "jobs": []} for _ in range(m)]}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    p = [int(j["p"]) for j in jobs_in]
    w = [int(j["w"]) for j in jobs_in]
    r = [ratio_key(p[i], w[i]) for i in range(n)]

    # Deterministic initial order: decreasing Smith ratio
    base_order = sorted(range(n), key=lambda j: (-r[j], p[j], -w[j], j))

    best_assignment = greedy_construction(base_order, m, p, w, r, alpha=0.0, rng=rng)
    best_obj, _, _, _, _ = full_eval(best_assignment, m, p, w, r)
    if logger:
        logger.log(float(best_obj))

    # Improve initial
    best_assignment_ls, best_obj_ls, _ = local_search_relocate(
        best_assignment[:], m, p, w, r, best_obj, best_obj, logger, deadline
    )
    if best_obj_ls < best_obj:
        best_obj = best_obj_ls
        best_assignment = best_assignment_ls[:]

    # Multi-start until time runs out
    while time.time() < deadline:
        # randomized order by slight perturbation + shuffle chunks
        if time.time() >= deadline:
            break

        perturbed = list(range(n))
        perturbed.sort(key=lambda j: (-(r[j] + rng.uniform(-1e-6, 1e-6)), p[j], -w[j], j))

        alpha = rng.choice([0.2, 0.3, 0.4, 0.5])
        cand_assignment = greedy_construction(perturbed, m, p, w, r, alpha=alpha, rng=rng)
        cand_obj, _, _, _, _ = full_eval(cand_assignment, m, p, w, r)

        if cand_obj < best_obj:
            best_obj = cand_obj
            best_assignment = cand_assignment[:]
            if logger:
                logger.log(float(best_obj))

        cand_assignment, cand_obj, _ = local_search_relocate(
            cand_assignment, m, p, w, r, cand_obj, best_obj, logger, deadline
        )

        if cand_obj < best_obj:
            best_obj = cand_obj
            best_assignment = cand_assignment[:]
            if logger:
                logger.log(float(best_obj))

    solution = build_output(best_assignment, m, jobs_in, p, w, r)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()