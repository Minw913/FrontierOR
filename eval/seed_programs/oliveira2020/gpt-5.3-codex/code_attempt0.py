import argparse
import json
import math
import random
import time
from copy import deepcopy

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel machine total weighted tardiness solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


class PMWTTSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()

        self.n = instance["parameters"]["n"]
        self.m = instance["parameters"]["m"]
        self.H = instance["time_horizon"]

        jobs = sorted(instance["jobs"], key=lambda x: x["id"])
        self.ids = [j["id"] for j in jobs]
        self.p = [int(j["processing_time"]) for j in jobs]
        self.w = [int(j["weight"]) for j in jobs]
        self.d = [int(j["due_date"]) for j in jobs]

        self.best_solution = None
        self.best_obj = float("inf")

    def elapsed(self):
        return time.time() - self.start_time

    def remaining(self):
        return max(0.0, self.time_limit - self.elapsed())

    def update_best(self, solution):
        if solution is None:
            return
        obj = float(solution["objective_value"])
        if obj + 1e-9 < self.best_obj:
            self.best_obj = obj
            self.best_solution = solution
            if self.logger:
                self.logger.log_solution(obj, solution)

    def build_solution_from_assignment(self, machine_of, start_of):
        if any(m is None for m in machine_of) or any(s is None for s in start_of):
            return None

        machine_jobs = [[] for _ in range(self.m)]
        schedule_rows = []
        total = 0.0

        for j in range(self.n):
            mm = int(machine_of[j])
            st = int(start_of[j])
            ct = st + self.p[j]
            wt = self.w[j] * max(0, ct - self.d[j])
            total += wt
            machine_jobs[mm].append((st, self.ids[j], j, ct, wt))

        prev_map = {}
        for mm in range(self.m):
            machine_jobs[mm].sort(key=lambda x: (x[0], x[1]))
            prev = 0
            for item in machine_jobs[mm]:
                _, jid, j, _, _ = item
                prev_map[j] = prev
                prev = jid

        for mm in range(self.m):
            for st, jid, j, ct, wt in machine_jobs[mm]:
                schedule_rows.append(
                    {
                        "machine_id": mm,
                        "prev_job": int(prev_map[j]),
                        "job": int(jid),
                        "start_time": int(st),
                        "completion_time": int(ct),
                        "weighted_tardiness": float(wt),
                    }
                )

        schedule_rows.sort(key=lambda x: x["job"])
        return {"objective_value": float(total), "schedule": schedule_rows}

    def seq_to_assignment(self, seqs):
        machine_of = [None] * self.n
        start_of = [None] * self.n
        obj = 0.0
        for mm in range(self.m):
            t = 0
            for j in seqs[mm]:
                machine_of[j] = mm
                start_of[j] = t
                t += self.p[j]
                obj += self.w[j] * max(0, t - self.d[j])
        return machine_of, start_of, obj

    def atc_construct(self, k=2.0, noise=0.0, rng=None):
        if rng is None:
            rng = random.Random(0)

        uns = set(range(self.n))
        seq = [[] for _ in range(self.m)]
        avail = [0] * self.m

        while uns:
            mm = min(range(self.m), key=lambda x: (avail[x], x))
            t = avail[mm]
            avg_p = sum(self.p[j] for j in uns) / max(1, len(uns))
            best_j, best_score = None, -1e100

            for j in uns:
                slack = max(self.d[j] - self.p[j] - t, 0)
                denom = max(1e-9, k * avg_p)
                score = (self.w[j] / self.p[j]) * math.exp(-slack / denom)
                if noise > 0:
                    score *= math.exp(noise * (rng.random() - 0.5))
                if score > best_score:
                    best_score = score
                    best_j = j

            seq[mm].append(best_j)
            avail[mm] += self.p[best_j]
            uns.remove(best_j)

        return seq

    def static_order_construct(self, order):
        seq = [[] for _ in range(self.m)]
        avail = [0] * self.m
        for j in order:
            mm = min(range(self.m), key=lambda x: (avail[x], x))
            seq[mm].append(j)
            avail[mm] += self.p[j]
        return seq

    def machine_cost(self, seq):
        t = 0
        c = 0.0
        for j in seq:
            t += self.p[j]
            c += self.w[j] * max(0, t - self.d[j])
        return c

    def local_search(self, seq, time_budget, rng):
        deadline = time.time() + max(0.0, time_budget)
        curr = deepcopy(seq)
        machine_costs = [self.machine_cost(curr[m]) for m in range(self.m)]
        curr_obj = sum(machine_costs)

        stall = 0
        max_stall = 5000

        while time.time() < deadline and stall < max_stall:
            move_type = 0 if rng.random() < 0.6 else 1

            if move_type == 0:
                nonempty = [m for m in range(self.m) if curr[m]]
                if not nonempty:
                    break
                i = rng.choice(nonempty)
                a = rng.randrange(len(curr[i]))
                job = curr[i][a]
                j = rng.randrange(self.m)
                b = rng.randrange(len(curr[j]) + 1)

                if i == j and (b == a or b == a + 1):
                    stall += 1
                    continue

                si = curr[i][:]
                sj = curr[j][:]
                del si[a]
                if i == j:
                    if b > a:
                        b -= 1
                    si.insert(b, job)
                    new_ci = self.machine_cost(si)
                    new_obj = curr_obj - machine_costs[i] + new_ci
                    if new_obj + 1e-9 < curr_obj:
                        curr[i] = si
                        machine_costs[i] = new_ci
                        curr_obj = new_obj
                        stall = 0
                    else:
                        stall += 1
                else:
                    sj.insert(b, job)
                    new_ci = self.machine_cost(si)
                    new_cj = self.machine_cost(sj)
                    new_obj = curr_obj - machine_costs[i] - machine_costs[j] + new_ci + new_cj
                    if new_obj + 1e-9 < curr_obj:
                        curr[i], curr[j] = si, sj
                        machine_costs[i], machine_costs[j] = new_ci, new_cj
                        curr_obj = new_obj
                        stall = 0
                    else:
                        stall += 1
            else:
                nonempty = [m for m in range(self.m) if curr[m]]
                if len(nonempty) == 0:
                    break
                i = rng.choice(nonempty)
                j = rng.choice(nonempty)
                if i == j and len(curr[i]) < 2:
                    stall += 1
                    continue

                a = rng.randrange(len(curr[i]))
                b = rng.randrange(len(curr[j]))

                if i == j and a == b:
                    stall += 1
                    continue

                si = curr[i][:]
                sj = curr[j][:]
                si[a], sj[b] = sj[b], si[a]

                if i == j:
                    new_ci = self.machine_cost(si)
                    new_obj = curr_obj - machine_costs[i] + new_ci
                    if new_obj + 1e-9 < curr_obj:
                        curr[i] = si
                        machine_costs[i] = new_ci
                        curr_obj = new_obj
                        stall = 0
                    else:
                        stall += 1
                else:
                    new_ci = self.machine_cost(si)
                    new_cj = self.machine_cost(sj)
                    new_obj = curr_obj - machine_costs[i] - machine_costs[j] + new_ci + new_cj
                    if new_obj + 1e-9 < curr_obj:
                        curr[i], curr[j] = si, sj
                        machine_costs[i], machine_costs[j] = new_ci, new_cj
                        curr_obj = new_obj
                        stall = 0
                    else:
                        stall += 1

        return curr

    def run_heuristics(self):
        if self.remaining() <= 0:
            return

        rng = random.Random(0)
        heur_time = min(max(1.0, 0.25 * self.time_limit), max(1.0, self.remaining() - 0.5))
        if self.time_limit <= 3:
            heur_time = max(0.5, 0.8 * self.remaining())
        t_end = time.time() + heur_time

        orders = []
        orders.append(sorted(range(self.n), key=lambda j: self.d[j]))  # EDD
        orders.append(sorted(range(self.n), key=lambda j: -(self.w[j] / self.p[j])))  # WSPT-like
        orders.append(sorted(range(self.n), key=lambda j: (self.d[j], -(self.w[j] / self.p[j]))))

        for order in orders:
            if time.time() >= t_end:
                break
            seq = self.static_order_construct(order)
            ls_budget = min(0.2, max(0.0, t_end - time.time()))
            seq = self.local_search(seq, ls_budget, rng)
            machine_of, start_of, _ = self.seq_to_assignment(seq)
            sol = self.build_solution_from_assignment(machine_of, start_of)
            self.update_best(sol)

        k_values = [1.0, 1.5, 2.0, 3.0, 4.5]
        while time.time() < t_end:
            k = rng.choice(k_values)
            noise = rng.choice([0.0, 0.1, 0.2, 0.35])
            seq = self.atc_construct(k=k, noise=noise, rng=rng)
            rem = t_end - time.time()
            ls_budget = min(0.3, max(0.0, rem))
            if ls_budget > 0.02:
                seq = self.local_search(seq, ls_budget, rng)
            machine_of, start_of, _ = self.seq_to_assignment(seq)
            sol = self.build_solution_from_assignment(machine_of, start_of)
            self.update_best(sol)

    def run_time_indexed_mip(self):
        rem = self.remaining()
        if rem <= 1.0:
            return

        total_vars = 0
        for j in range(self.n):
            total_vars += self.m * (self.H - self.p[j] + 1)
        if total_vars > 400000:
            return

        model = gp.Model("pm_wt_time_indexed")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.5, self.remaining())

        x = {}
        job_vars = [[] for _ in range(self.n)]

        # Variables and objective
        for j in range(self.n):
            latest = self.H - self.p[j]
            for mm in range(self.m):
                for s in range(latest + 1):
                    tard = max(0, s + self.p[j] - self.d[j])
                    coef = self.w[j] * tard
                    v = model.addVar(vtype=GRB.BINARY, obj=coef, name=f"x_{j}_{mm}_{s}")
                    x[(j, mm, s)] = v
                    job_vars[j].append((v, mm, s))

        # Each job once
        for j in range(self.n):
            model.addConstr(gp.quicksum(v for v, _, _ in job_vars[j]) == 1, name=f"assign_{j}")

        # Capacity constraints
        for mm in range(self.m):
            for tau in range(self.H):
                terms = []
                for j in range(self.n):
                    pj = self.p[j]
                    lo = max(0, tau - pj + 1)
                    hi = min(tau, self.H - pj)
                    if lo <= hi:
                        for s in range(lo, hi + 1):
                            terms.append(x[(j, mm, s)])
                if terms:
                    model.addConstr(gp.quicksum(terms) <= 1, name=f"cap_{mm}_{tau}")

        # Warm start from best known solution
        if self.best_solution is not None:
            row_by_jobid = {r["job"]: r for r in self.best_solution["schedule"]}
            id_to_idx = {jid: idx for idx, jid in enumerate(self.ids)}
            for jid, row in row_by_jobid.items():
                j = id_to_idx[jid]
                mm = int(row["machine_id"])
                s = int(row["start_time"])
                key = (j, mm, s)
                if key in x:
                    x[key].Start = 1.0

        model.ModelSense = GRB.MINIMIZE
        model.update()

        # Callback for incumbent logging
        model._best_obj = self.best_obj
        model._best_solution = self.best_solution
        model._job_vars = job_vars
        model._logger = self.logger
        model._n = self.n
        model._build_solution = self.build_solution_from_assignment

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-7 < m._best_obj:
                    machine_of = [None] * m._n
                    start_of = [None] * m._n
                    for j in range(m._n):
                        chosen = None
                        best_val = -1.0
                        best = None
                        for v, mm, s in m._job_vars[j]:
                            val = m.cbGetSolution(v)
                            if val > best_val:
                                best_val = val
                                best = (mm, s)
                            if val > 0.5:
                                chosen = (mm, s)
                                break
                        if chosen is None:
                            chosen = best
                        if chosen is not None:
                            machine_of[j], start_of[j] = chosen

                    sol = m._build_solution(machine_of, start_of)
                    if sol is not None and sol["objective_value"] + 1e-7 < m._best_obj:
                        m._best_obj = sol["objective_value"]
                        m._best_solution = sol
                        if m._logger:
                            m._logger.log_solution(sol["objective_value"], sol)

        model.optimize(cb)

        # pull callback best
        if model._best_solution is not None and model._best_obj + 1e-9 < self.best_obj:
            self.best_obj = model._best_obj
            self.best_solution = model._best_solution

        # ensure final incumbent extracted
        if model.SolCount > 0:
            machine_of = [None] * self.n
            start_of = [None] * self.n
            for j in range(self.n):
                found = False
                for v, mm, s in job_vars[j]:
                    if v.X > 0.5:
                        machine_of[j] = mm
                        start_of[j] = s
                        found = True
                        break
                if not found:
                    # fallback to max value
                    bestv, bestmm, bests = -1.0, None, None
                    for v, mm, s in job_vars[j]:
                        if v.X > bestv:
                            bestv, bestmm, bests = v.X, mm, s
                    machine_of[j], start_of[j] = bestmm, bests

            sol = self.build_solution_from_assignment(machine_of, start_of)
            self.update_best(sol)

    def solve(self):
        self.run_heuristics()
        self.run_time_indexed_mip()

        if self.best_solution is None:
            # emergency fallback
            order = sorted(range(self.n), key=lambda j: self.d[j])
            seq = self.static_order_construct(order)
            machine_of, start_of, _ = self.seq_to_assignment(seq)
            self.best_solution = self.build_solution_from_assignment(machine_of, start_of)
            self.best_obj = self.best_solution["objective_value"]
            if self.logger:
                self.logger.log_solution(self.best_obj, self.best_solution)

        return self.best_solution


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solver = PMWTTSolver(instance, time_limit=args.time_limit, logger=logger)
    sol = solver.solve()

    # enforce schema keys
    output = {
        "objective_value": float(sol["objective_value"]),
        "schedule": sol["schedule"],
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()