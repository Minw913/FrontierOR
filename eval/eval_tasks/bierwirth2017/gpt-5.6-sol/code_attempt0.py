import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class JobShopSolver:
    def __init__(self, instance, logger, deadline):
        self.instance = instance
        self.logger = logger
        self.deadline = deadline

        self.jobs = instance["jobs"]
        self.num_jobs = len(self.jobs)
        self.num_machines = int(instance["num_machines"])

        self.job_ops = []
        self.op_job = []
        self.op_index = []
        self.op_machine = []
        self.op_duration = []
        self.release = []
        self.due = []
        self.weight = []

        op_id = 0
        for j, job in enumerate(self.jobs):
            self.release.append(float(job["release_date"]))
            self.due.append(float(job["due_date"]))
            self.weight.append(float(job["weight"]))
            ids = []
            for k, op in enumerate(job["operations"]):
                ids.append(op_id)
                self.op_job.append(j)
                self.op_index.append(k)
                self.op_machine.append(int(op["machine"]))
                self.op_duration.append(float(op["processing_time"]))
                op_id += 1
            self.job_ops.append(ids)

        self.num_ops = op_id

        self.machine_ops = [[] for _ in range(self.num_machines)]
        for op in range(self.num_ops):
            self.machine_ops[self.op_machine[op]].append(op)

        self.remaining_processing = []
        for j in range(self.num_jobs):
            durations = [
                self.op_duration[op] for op in self.job_ops[j]
            ]
            suffix = [0.0] * len(durations)
            running = 0.0
            for k in range(len(durations) - 1, -1, -1):
                running += durations[k]
                suffix[k] = running
            self.remaining_processing.append(suffix)

        self.best_objective = float("inf")
        self.best_starts = None
        self.best_orders = None
        self.rng = random.Random(0)

    def time_left(self):
        return self.deadline - time.monotonic()

    def decode_orders(self, machine_orders):
        n = self.num_ops
        indegree = [0] * n
        successors = [[] for _ in range(n)]

        for j in range(self.num_jobs):
            ops = self.job_ops[j]
            for k in range(len(ops) - 1):
                u, v = ops[k], ops[k + 1]
                successors[u].append(v)
                indegree[v] += 1

        for order in machine_orders:
            for i in range(len(order) - 1):
                u, v = order[i], order[i + 1]
                successors[u].append(v)
                indegree[v] += 1

        starts = [0.0] * n
        for j in range(self.num_jobs):
            if self.job_ops[j]:
                starts[self.job_ops[j][0]] = self.release[j]

        queue = [op for op in range(n) if indegree[op] == 0]
        head = 0
        processed = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            processed += 1
            finish = starts[u] + self.op_duration[u]

            for v in successors[u]:
                if finish > starts[v]:
                    starts[v] = finish
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        if processed != n:
            return None

        objective = 0.0
        for j in range(self.num_jobs):
            last = self.job_ops[j][-1]
            completion = starts[last] + self.op_duration[last]
            objective += self.weight[j] * max(0.0, completion - self.due[j])

        return objective, starts

    def build_solution(self, starts):
        schedule = []
        objective = 0.0

        for j, job in enumerate(self.jobs):
            operations = []
            for op in self.job_ops[j]:
                operations.append({
                    "machine": int(self.op_machine[op]),
                    "start_time": float(starts[op]),
                })

            last = self.job_ops[j][-1]
            completion = starts[last] + self.op_duration[last]
            tardiness = max(0.0, completion - self.due[j])
            objective += self.weight[j] * tardiness

            schedule.append({
                "job_id": int(job["job_id"]),
                "completion_time": float(completion),
                "tardiness": float(tardiness),
                "operations": operations,
            })

        return {
            "objective_value": float(objective),
            "schedule": schedule,
        }

    def register_solution(self, objective, starts, machine_orders):
        if objective + 1e-7 >= self.best_objective:
            return False

        self.best_objective = float(objective)
        self.best_starts = list(starts)
        self.best_orders = [list(order) for order in machine_orders]

        if self.logger:
            solution = self.build_solution(self.best_starts)
            self.logger.log_solution(solution["objective_value"], solution)
        return True

    def dispatch_schedule(self, variant=0):
        next_index = [0] * self.num_jobs
        job_ready = list(self.release)
        machine_ready = [0.0] * self.num_machines
        machine_orders = [[] for _ in range(self.num_machines)]

        unscheduled = self.num_ops
        average_p = (
            sum(self.op_duration) / max(1, len(self.op_duration))
        )

        while unscheduled > 0:
            available = []
            pivot_completion = float("inf")
            pivot_machine = None

            for j in range(self.num_jobs):
                k = next_index[j]
                if k >= len(self.job_ops[j]):
                    continue
                op = self.job_ops[j][k]
                machine = self.op_machine[op]
                est = max(job_ready[j], machine_ready[machine])
                ect = est + self.op_duration[op]
                available.append((j, k, op, machine, est, ect))
                if ect < pivot_completion:
                    pivot_completion = ect
                    pivot_machine = machine

            conflicts = [
                item for item in available
                if item[3] == pivot_machine
                and item[4] < pivot_completion - 1e-9
            ]
            if not conflicts:
                conflicts = [
                    min(available, key=lambda x: (x[5], x[4], x[0]))
                ]

            def priority(item):
                j, k, op, machine, est, ect = item
                p = max(self.op_duration[op], 1e-9)
                remaining = self.remaining_processing[j][k]
                projected = est + remaining
                slack = self.due[j] - projected
                w = self.weight[j]

                if variant % 6 == 0:
                    kappa = 2.0
                    score = (w / p) * math.exp(
                        -max(0.0, slack) /
                        max(1e-9, kappa * average_p)
                    )
                    return (-score, self.due[j], j)
                if variant % 6 == 1:
                    return (self.due[j], -w, p, j)
                if variant % 6 == 2:
                    return (slack / max(w, 1e-9), self.due[j], j)
                if variant % 6 == 3:
                    return (p / max(w, 1e-9), self.due[j], j)
                if variant % 6 == 4:
                    return (projected - self.due[j], -w, p, j)

                noise = self.rng.random()
                urgency = slack / max(w, 1e-9)
                return (urgency + noise * average_p * 1.5,
                        self.due[j], j)

            chosen = min(conflicts, key=priority)
            j, k, op, machine, est, ect = chosen

            machine_orders[machine].append(op)
            machine_ready[machine] = ect
            job_ready[j] = ect
            next_index[j] += 1
            unscheduled -= 1

        decoded = self.decode_orders(machine_orders)
        if decoded is None:
            return None
        objective, starts = decoded
        return objective, starts, machine_orders

    def local_search(self, orders, stop_time, max_no_improve=2):
        decoded = self.decode_orders(orders)
        if decoded is None:
            return
        current_objective, current_starts = decoded
        self.register_solution(current_objective, current_starts, orders)

        no_improve = 0
        while time.monotonic() < stop_time and no_improve < max_no_improve:
            candidates = []
            for machine, order in enumerate(orders):
                for pos in range(len(order) - 1):
                    candidates.append((machine, pos))

            self.rng.shuffle(candidates)
            if self.num_ops > 600 and len(candidates) > 350:
                candidates = candidates[:350]

            improved = False
            for machine, pos in candidates:
                if time.monotonic() >= stop_time:
                    break

                order = orders[machine]
                order[pos], order[pos + 1] = order[pos + 1], order[pos]
                result = self.decode_orders(orders)

                if result is not None:
                    objective, starts = result
                    if objective + 1e-7 < current_objective:
                        current_objective = objective
                        current_starts = starts
                        improved = True
                        self.register_solution(
                            objective, starts, orders
                        )
                        break

                order[pos], order[pos + 1] = order[pos + 1], order[pos]

            if improved:
                no_improve = 0
            else:
                no_improve += 1

    def heuristic_phase(self, stop_time):
        variant = 0
        generated = 0

        while time.monotonic() < stop_time:
            result = self.dispatch_schedule(variant)
            variant += 1
            generated += 1
            if result is None:
                continue

            objective, starts, orders = result
            improved = self.register_solution(objective, starts, orders)

            if improved or generated <= 6:
                local_deadline = min(
                    stop_time,
                    time.monotonic() + max(0.02, (stop_time - time.monotonic()) * 0.25)
                )
                self.local_search(
                    [list(x) for x in orders],
                    local_deadline,
                    max_no_improve=1,
                )

            if generated >= 24 and self.best_orders is not None:
                self.local_search(
                    [list(x) for x in self.best_orders],
                    stop_time,
                    max_no_improve=3,
                )
                break

    def solve_with_gurobi(self):
        if self.best_starts is None or self.time_left() <= 0.15:
            return

        pair_count = sum(
            len(ops) * (len(ops) - 1) // 2
            for ops in self.machine_ops
        )
        if pair_count > 50000:
            return

        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return

        try:
            model = gp.Model("weighted_tardiness_job_shop")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.MIPFocus = 1

            min_release = min([0.0] + self.release)
            total_processing = sum(self.op_duration)
            max_release = max([0.0] + self.release)
            horizon = max_release + total_processing
            big_m = horizon - min_release + max(
                [0.0] + self.op_duration
            )

            start_vars = []
            for op in range(self.num_ops):
                ub = max(horizon, self.best_starts[op] +
                         self.op_duration[op])
                var = model.addVar(
                    lb=min_release,
                    ub=ub,
                    vtype=GRB.CONTINUOUS,
                    name=f"s_{op}",
                )
                var.Start = self.best_starts[op]
                start_vars.append(var)

            tardiness_vars = []
            for j in range(self.num_jobs):
                var = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"T_{j}",
                )
                last = self.job_ops[j][-1]
                completion = (
                    self.best_starts[last] + self.op_duration[last]
                )
                var.Start = max(0.0, completion - self.due[j])
                tardiness_vars.append(var)

            order_vars = {}
            for machine, ops in enumerate(self.machine_ops):
                for a_pos in range(len(ops)):
                    for b_pos in range(a_pos + 1, len(ops)):
                        a = ops[a_pos]
                        b = ops[b_pos]
                        y = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"y_{a}_{b}",
                        )
                        order_vars[(a, b)] = y

            model.update()

            for j in range(self.num_jobs):
                first = self.job_ops[j][0]
                model.addConstr(start_vars[first] >= self.release[j])

                for k in range(len(self.job_ops[j]) - 1):
                    a = self.job_ops[j][k]
                    b = self.job_ops[j][k + 1]
                    model.addConstr(
                        start_vars[b] >=
                        start_vars[a] + self.op_duration[a]
                    )

                last = self.job_ops[j][-1]
                model.addConstr(
                    tardiness_vars[j] >=
                    start_vars[last] +
                    self.op_duration[last] -
                    self.due[j]
                )

            position = {}
            if self.best_orders is not None:
                for order in self.best_orders:
                    for pos, op in enumerate(order):
                        position[op] = pos

            for machine, ops in enumerate(self.machine_ops):
                for a_pos in range(len(ops)):
                    for b_pos in range(a_pos + 1, len(ops)):
                        a = ops[a_pos]
                        b = ops[b_pos]
                        y = order_vars[(a, b)]

                        model.addConstr(
                            start_vars[a] + self.op_duration[a]
                            <= start_vars[b] + big_m * (1.0 - y)
                        )
                        model.addConstr(
                            start_vars[b] + self.op_duration[b]
                            <= start_vars[a] + big_m * y
                        )

                        if position:
                            y.Start = (
                                1.0 if position[a] < position[b] else 0.0
                            )

            model.setObjective(
                gp.quicksum(
                    self.weight[j] * tardiness_vars[j]
                    for j in range(self.num_jobs)
                ),
                GRB.MINIMIZE,
            )

            remaining = self.time_left() - 0.05
            if remaining <= 0.02:
                return
            model.Params.TimeLimit = max(0.01, remaining)

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution(start_vars)
                    machine_orders = []
                    for ops in self.machine_ops:
                        machine_orders.append(sorted(
                            ops,
                            key=lambda op: (values[op], op)
                        ))
                    decoded = self.decode_orders(machine_orders)
                    if decoded is not None:
                        objective, starts = decoded
                        self.register_solution(
                            objective, starts, machine_orders
                        )
                except Exception:
                    return

            model.optimize(incumbent_callback)

        except Exception:
            return

    def solve(self):
        # Always produce at least one feasible schedule.
        initial = self.dispatch_schedule(0)
        if initial is not None:
            objective, starts, orders = initial
            self.register_solution(objective, starts, orders)

        pair_count = sum(
            len(ops) * (len(ops) - 1) // 2
            for ops in self.machine_ops
        )

        remaining = max(0.0, self.time_left())
        if pair_count <= 50000 and remaining > 0.5:
            heuristic_budget = min(
                5.0,
                max(0.15, remaining * 0.20),
            )
        else:
            heuristic_budget = remaining

        heuristic_stop = min(
            self.deadline,
            time.monotonic() + heuristic_budget
        )
        self.heuristic_phase(heuristic_stop)

        if pair_count <= 50000 and self.time_left() > 0.15:
            self.solve_with_gurobi()
        elif self.best_orders is not None and self.time_left() > 0.02:
            self.local_search(
                [list(x) for x in self.best_orders],
                self.deadline,
                max_no_improve=4,
            )

        if self.best_starts is None:
            # Defensive fallback: process jobs in job order while respecting
            # each machine's availability.
            job_ready = list(self.release)
            machine_ready = [0.0] * self.num_machines
            starts = [0.0] * self.num_ops
            orders = [[] for _ in range(self.num_machines)]

            for j in range(self.num_jobs):
                for op in self.job_ops[j]:
                    machine = self.op_machine[op]
                    starts[op] = max(
                        job_ready[j], machine_ready[machine]
                    )
                    finish = starts[op] + self.op_duration[op]
                    job_ready[j] = finish
                    machine_ready[machine] = finish
                    orders[machine].append(op)

            solution = self.build_solution(starts)
            self.best_starts = starts
            self.best_orders = orders
            self.best_objective = solution["objective_value"]

        return self.build_solution(self.best_starts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path else None
    )

    instance = read_instance(args.instance_path)

    # Reserve a small amount of time for serialization.
    deadline = start_time + max(0, args.time_limit) - 0.05
    solver = JobShopSolver(instance, logger, deadline)
    solution = solver.solve()

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()