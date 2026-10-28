import argparse
import json
import math
import random
import time
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic solver for P||sum w_j T_j")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)  # accepted per spec
    return parser.parse_args()


class SafeLogger:
    """
    No-op logger to avoid environment-specific crashes (e.g., DBus issues).
    Keeps required call pattern in code while being robust.
    """
    def __init__(self, log_path=None):
        self.enabled = False
        self.log_path = log_path

    def log(self, objective_value):
        return

    def log_solution(self, objective_value, solution_dict):
        return


class ParallelWeightedTardinessSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = max(1, int(time_limit))
        self.logger = logger

        self.start_clock = time.perf_counter()
        self.rng = random.Random(0)

        self.n = int(instance["parameters"]["n"])
        self.m = int(instance["parameters"]["m"])

        jobs = sorted(instance["jobs"], key=lambda x: x["id"])
        self.job_ids = [int(j["id"]) for j in jobs]
        self.p = [int(j["processing_time"]) for j in jobs]
        self.w = [int(j["weight"]) for j in jobs]
        self.d = [int(j["due_date"]) for j in jobs]

        self.best_obj = float("inf")
        self.best_seqs = None
        self.best_solution = None

    def elapsed(self):
        return time.perf_counter() - self.start_clock

    def remaining(self):
        return self.time_limit - self.elapsed()

    def machine_cost(self, seq):
        t = 0
        c = 0.0
        for j in seq:
            t += self.p[j]
            c += self.w[j] * max(0, t - self.d[j])
        return c

    def total_cost(self, seqs):
        return sum(self.machine_cost(seq) for seq in seqs)

    def seqs_to_solution(self, seqs):
        schedule = []
        total = 0.0

        for mm in range(self.m):
            t = 0
            prev = 0
            for j in seqs[mm]:
                st = t
                ct = st + self.p[j]
                wt = self.w[j] * max(0, ct - self.d[j])
                total += wt
                schedule.append(
                    {
                        "machine_id": int(mm),
                        "prev_job": int(prev),
                        "job": int(self.job_ids[j]),
                        "start_time": int(st),
                        "completion_time": int(ct),
                        "weighted_tardiness": float(wt),
                    }
                )
                prev = self.job_ids[j]
                t = ct

        schedule.sort(key=lambda r: r["job"])
        return {"objective_value": float(total), "schedule": schedule}

    def update_best(self, seqs):
        obj = self.total_cost(seqs)
        if obj + 1e-9 < self.best_obj:
            self.best_obj = obj
            self.best_seqs = deepcopy(seqs)
            self.best_solution = self.seqs_to_solution(self.best_seqs)
            if self.logger:
                self.logger.log_solution(self.best_obj, self.best_solution)

    def construct_by_static_order(self, order):
        seqs = [[] for _ in range(self.m)]
        loads = [0] * self.m
        for j in order:
            mm = min(range(self.m), key=lambda x: (loads[x], x))
            seqs[mm].append(j)
            loads[mm] += self.p[j]
        return seqs

    def construct_atc(self, k=2.0, noise=0.0):
        uns = set(range(self.n))
        seqs = [[] for _ in range(self.m)]
        avail = [0] * self.m

        while uns:
            mm = min(range(self.m), key=lambda x: (avail[x], x))
            t = avail[mm]
            avg_p = sum(self.p[j] for j in uns) / max(1, len(uns))
            denom = max(1e-9, k * avg_p)

            best_j = None
            best_score = -1e100
            for j in uns:
                slack = max(0, self.d[j] - self.p[j] - t)
                score = (self.w[j] / self.p[j]) * math.exp(-slack / denom)
                if noise > 0:
                    score *= math.exp(noise * (self.rng.random() - 0.5))
                if score > best_score:
                    best_score = score
                    best_j = j

            seqs[mm].append(best_j)
            avail[mm] += self.p[best_j]
            uns.remove(best_j)

        return seqs

    def improve_local(self, seqs, time_budget):
        end_t = time.perf_counter() + max(0.0, time_budget)
        cur = deepcopy(seqs)
        mcost = [self.machine_cost(cur[m]) for m in range(self.m)]
        obj = sum(mcost)

        stall = 0
        while time.perf_counter() < end_t:
            if stall >= 3500:
                break

            if self.rng.random() < 0.65:
                nonempty = [m for m in range(self.m) if cur[m]]
                if not nonempty:
                    break

                i = self.rng.choice(nonempty)
                a = self.rng.randrange(len(cur[i]))
                job = cur[i][a]

                j = self.rng.randrange(self.m)
                b = self.rng.randrange(len(cur[j]) + 1)

                if i == j and (b == a or b == a + 1):
                    stall += 1
                    continue

                si = cur[i][:]
                sj = cur[j][:]
                del si[a]

                if i == j:
                    if b > a:
                        b -= 1
                    si.insert(b, job)
                    new_ci = self.machine_cost(si)
                    new_obj = obj - mcost[i] + new_ci
                    if new_obj + 1e-9 < obj:
                        cur[i] = si
                        mcost[i] = new_ci
                        obj = new_obj
                        stall = 0
                    else:
                        stall += 1
                else:
                    sj.insert(b, job)
                    new_ci = self.machine_cost(si)
                    new_cj = self.machine_cost(sj)
                    new_obj = obj - mcost[i] - mcost[j] + new_ci + new_cj
                    if new_obj + 1e-9 < obj:
                        cur[i], cur[j] = si, sj
                        mcost[i], mcost[j] = new_ci, new_cj
                        obj = new_obj
                        stall = 0
                    else:
                        stall += 1
            else:
                nonempty = [m for m in range(self.m) if cur[m]]
                if not nonempty:
                    break

                i = self.rng.choice(nonempty)
                j = self.rng.choice(nonempty)

                if i == j and len(cur[i]) < 2:
                    stall += 1
                    continue

                a = self.rng.randrange(len(cur[i]))
                b = self.rng.randrange(len(cur[j]))
                if i == j and a == b:
                    stall += 1
                    continue

                si = cur[i][:]
                sj = cur[j][:]
                si[a], sj[b] = sj[b], si[a]

                if i == j:
                    new_ci = self.machine_cost(si)
                    new_obj = obj - mcost[i] + new_ci
                    if new_obj + 1e-9 < obj:
                        cur[i] = si
                        mcost[i] = new_ci
                        obj = new_obj
                        stall = 0
                    else:
                        stall += 1
                else:
                    new_ci = self.machine_cost(si)
                    new_cj = self.machine_cost(sj)
                    new_obj = obj - mcost[i] - mcost[j] + new_ci + new_cj
                    if new_obj + 1e-9 < obj:
                        cur[i], cur[j] = si, sj
                        mcost[i], mcost[j] = new_ci, new_cj
                        obj = new_obj
                        stall = 0
                    else:
                        stall += 1

        return cur

    def solve(self):
        candidates = []

        edd = sorted(range(self.n), key=lambda j: (self.d[j], j))
        wspt = sorted(range(self.n), key=lambda j: (-self.w[j] / self.p[j], self.d[j], j))
        short = sorted(range(self.n), key=lambda j: (self.p[j], self.d[j], j))
        heavy = sorted(range(self.n), key=lambda j: (-self.w[j], self.d[j], j))
        ratio = sorted(range(self.n), key=lambda j: (self.d[j] / max(1, self.w[j]), self.p[j], j))

        for order in [edd, wspt, short, heavy, ratio]:
            candidates.append(self.construct_by_static_order(order))

        for k in [1.0, 1.5, 2.0, 3.0, 4.0]:
            candidates.append(self.construct_atc(k=k, noise=0.0))

        for seqs in candidates:
            self.update_best(seqs)

        while self.remaining() > 0.05:
            if self.best_seqs is None:
                base = self.construct_atc(k=2.0, noise=0.2)
            else:
                base = deepcopy(self.best_seqs)

            for _ in range(1 + self.rng.randint(0, 3)):
                nonempty = [m for m in range(self.m) if base[m]]
                if not nonempty:
                    break
                i = self.rng.choice(nonempty)
                a = self.rng.randrange(len(base[i]))
                job = base[i].pop(a)

                j = self.rng.randrange(self.m)
                b = self.rng.randrange(len(base[j]) + 1)
                base[j].insert(b, job)

            budget = min(0.2, max(0.01, self.remaining() * 0.25))
            improved = self.improve_local(base, budget)
            self.update_best(improved)

            if self.rng.random() < 0.25 and self.remaining() > 0.1:
                fresh = self.construct_atc(
                    k=self.rng.choice([1.0, 1.5, 2.0, 3.0, 4.0]),
                    noise=self.rng.choice([0.0, 0.1, 0.2, 0.35]),
                )
                budget2 = min(0.12, max(0.01, self.remaining() * 0.2))
                fresh = self.improve_local(fresh, budget2)
                self.update_best(fresh)

        if self.best_solution is None:
            seqs = self.construct_by_static_order(sorted(range(self.n), key=lambda j: self.d[j]))
            self.update_best(seqs)

        return self.best_solution


def main():
    args = parse_args()
    logger = SafeLogger(args.log_path)

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solver = ParallelWeightedTardinessSolver(instance, args.time_limit, logger=logger)
    solution = solver.solve()

    output = {
        "objective_value": float(solution["objective_value"]),
        "schedule": solution["schedule"],
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()