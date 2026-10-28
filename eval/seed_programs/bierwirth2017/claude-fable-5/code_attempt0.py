import json
import argparse
import time
import random
import math

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    nj = data["num_jobs"]
    nm = data["num_machines"]
    jobs = data["jobs"]
    # order jobs by job_id for safety
    jobs_sorted = sorted(jobs, key=lambda x: x["job_id"])
    job_ids = [jb["job_id"] for jb in jobs_sorted]

    weight = [jb["weight"] for jb in jobs_sorted]
    release = [jb["release_date"] for jb in jobs_sorted]
    due = [jb["due_date"] for jb in jobs_sorted]
    op_mach = []
    op_pt = []
    nops_per_job = []
    for jb in jobs_sorted:
        ms = [op["machine"] for op in jb["operations"]]
        ps = [op["processing_time"] for op in jb["operations"]]
        op_mach.append(ms)
        op_pt.append(ps)
        nops_per_job.append(len(ms))

    total_ops = sum(nops_per_job)
    if total_ops == 0 or nj == 0:
        sol = {"objective_value": 0.0, "schedule": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # suffix remaining work per job/op index
    rem_work = []
    for j in range(nj):
        rw = [0] * (nops_per_job[j] + 1)
        for k in range(nops_per_job[j] - 1, -1, -1):
            rw[k] = rw[k + 1] + op_pt[j][k]
        rem_work.append(rw)

    avg_p = sum(sum(ps) for ps in op_pt) / max(1, total_ops)

    rnd = random.Random(0)

    # ---------- decoder (fast objective only) ----------
    def decode_obj(seq):
        job_ready = release[:]
        op_idx = [0] * nj
        mach_ready = [0] * nm
        _om = op_mach
        _op = op_pt
        for j in seq:
            k = op_idx[j]
            op_idx[j] = k + 1
            m = _om[j][k]
            p = _op[j][k]
            jr = job_ready[j]
            mr = mach_ready[m]
            s = jr if jr > mr else mr
            e = s + p
            job_ready[j] = e
            mach_ready[m] = e
        total = 0
        for j in range(nj):
            d = job_ready[j] - due[j]
            if d > 0:
                total += weight[j] * d
        return total

    # ---------- full decoder producing schedule ----------
    def build_solution(seq):
        job_ready = release[:]
        op_idx = [0] * nj
        mach_ready = [0] * nm
        starts = [[0] * nops_per_job[j] for j in range(nj)]
        for j in seq:
            k = op_idx[j]
            op_idx[j] = k + 1
            m = op_mach[j][k]
            p = op_pt[j][k]
            s = max(job_ready[j], mach_ready[m])
            starts[j][k] = s
            e = s + p
            job_ready[j] = e
            mach_ready[m] = e
        total = 0
        schedule = []
        for j in range(nj):
            comp = job_ready[j]
            tard = max(0, comp - due[j])
            total += weight[j] * tard
            schedule.append({
                "job_id": job_ids[j],
                "completion_time": float(comp),
                "tardiness": float(tard),
                "operations": [
                    {"machine": op_mach[j][k], "start_time": float(starts[j][k])}
                    for k in range(nops_per_job[j])
                ],
            })
        return total, {"objective_value": float(total), "schedule": schedule}

    # ---------- constructive heuristics ----------
    def construct_atc(K):
        job_ready = release[:]
        op_idx = [0] * nj
        mach_ready = [0] * nm
        seq = []
        remaining = total_ops
        active = set(j for j in range(nj) if nops_per_job[j] > 0)
        while remaining > 0:
            # earliest possible start among next ops
            best_t = None
            for j in active:
                k = op_idx[j]
                m = op_mach[j][k]
                est = max(job_ready[j], mach_ready[m])
                if best_t is None or est < best_t:
                    best_t = est
            t = best_t
            # candidates that can start at t (non-delay)
            best_j = -1
            best_pr = -1.0
            for j in active:
                k = op_idx[j]
                m = op_mach[j][k]
                est = max(job_ready[j], mach_ready[m])
                if est > t:
                    continue
                p = op_pt[j][k]
                slack = due[j] - t - rem_work[j][k]
                pr = (weight[j] / max(p, 1e-9)) * math.exp(-max(0.0, slack) / max(K * avg_p, 1e-9))
                if pr > best_pr:
                    best_pr = pr
                    best_j = j
            j = best_j
            k = op_idx[j]
            m = op_mach[j][k]
            p = op_pt[j][k]
            s = max(job_ready[j], mach_ready[m])
            job_ready[j] = s + p
            mach_ready[m] = s + p
            op_idx[j] = k + 1
            if op_idx[j] >= nops_per_job[j]:
                active.discard(j)
            seq.append(j)
            remaining -= 1
        return seq

    def construct_sorted(keyfunc):
        order = sorted(range(nj), key=keyfunc)
        seq = []
        idx = [0] * nj
        rem = total_ops
        # round-robin in sorted order
        while rem > 0:
            for j in order:
                if idx[j] < nops_per_job[j]:
                    seq.append(j)
                    idx[j] += 1
                    rem -= 1
        return seq

    candidates = []
    for K in (0.5, 1.0, 2.0, 3.0):
        candidates.append(construct_atc(K))
    candidates.append(construct_sorted(lambda j: due[j]))
    candidates.append(construct_sorted(lambda j: due[j] / max(weight[j], 1e-9)))
    candidates.append(construct_sorted(lambda j: (release[j], due[j])))

    best_seq = None
    best_obj = None
    for s in candidates:
        o = decode_obj(s)
        if best_obj is None or o < best_obj:
            best_obj = o
            best_seq = s[:]

    obj_val, sol_dict = build_solution(best_seq)
    if logger:
        logger.log_solution(float(obj_val), sol_dict)
    best_sol_dict = sol_dict

    # ---------- simulated annealing ----------
    if best_obj > 0 and time.time() < deadline:
        cur = best_seq[:]
        cur_obj = best_obj
        N = total_ops
        T0 = max(1.0, 0.05 * best_obj)
        Tend = 0.5
        total_budget = max(deadline - time.time(), 1e-6)
        sa_start = time.time()
        no_improve = 0
        iters = 0
        check_mask = 255

        while True:
            iters += 1
            if (iters & check_mask) == 0:
                now = time.time()
                if now >= deadline:
                    break
                frac = min(1.0, (now - sa_start) / total_budget)
                T = T0 * ((Tend / T0) ** frac)
            elif iters == 1:
                T = T0

            # generate a move
            if rnd.random() < 0.5:
                # swap
                i = rnd.randrange(N)
                jp = rnd.randrange(N)
                if i == jp or cur[i] == cur[jp]:
                    continue
                cur[i], cur[jp] = cur[jp], cur[i]
                new_obj = decode_obj(cur)
                delta = new_obj - cur_obj
                if delta <= 0 or rnd.random() < math.exp(-delta / max(T, 1e-9)):
                    cur_obj = new_obj
                else:
                    cur[i], cur[jp] = cur[jp], cur[i]
                    new_obj = None
            else:
                # insert
                i = rnd.randrange(N)
                jp = rnd.randrange(N)
                if i == jp:
                    continue
                v = cur.pop(i)
                cur.insert(jp, v)
                new_obj = decode_obj(cur)
                delta = new_obj - cur_obj
                if delta <= 0 or rnd.random() < math.exp(-delta / max(T, 1e-9)):
                    cur_obj = new_obj
                else:
                    cur.pop(jp)
                    cur.insert(i, v)
                    new_obj = None

            if cur_obj < best_obj:
                best_obj = cur_obj
                best_seq = cur[:]
                no_improve = 0
                obj_val, best_sol_dict = build_solution(best_seq)
                if logger:
                    logger.log_solution(float(obj_val), best_sol_dict)
                if best_obj <= 0:
                    break
            else:
                no_improve += 1

            # reheat / restart from best if stuck
            if no_improve >= 20000:
                cur = best_seq[:]
                cur_obj = best_obj
                T0 = max(1.0, 0.05 * best_obj)
                sa_start = time.time()
                total_budget = max(deadline - sa_start, 1e-6)
                no_improve = 0

    # ---------- write final solution ----------
    obj_val, best_sol_dict = build_solution(best_seq)
    if logger:
        logger.log_solution(float(obj_val), best_sol_dict)
    with open(args.solution_path, "w") as f:
        json.dump(best_sol_dict, f, indent=2)


if __name__ == "__main__":
    main()