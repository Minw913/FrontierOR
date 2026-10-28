import argparse
import bisect
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def service_key(index):
    return str(index + 1)


def make_solution(schedule, objective):
    return {
        "objective_value": float(objective),
        "departure_schedule_minutes": {
            service_key(i): float(t) for i, t in enumerate(schedule)
        },
    }


class ProblemData:
    def __init__(self, instance):
        p = instance["parameters"]

        self.T = float(p["T_planning_horizon_minutes"])
        self.D = int(p["D_num_train_services"])
        self.NT = int(p["NT_num_demand_periods"])
        self.theta = float(p["theta_period_length_minutes"])
        self.h_min = float(p["h_min_minutes"])
        self.h_max = float(p["h_max_minutes"])
        self.capacity = float(p["C_train_capacity_passengers"])

        demand_periods = list(instance["demand_periods"])
        try:
            demand_periods.sort(key=lambda z: int(z["period"]))
        except (ValueError, TypeError):
            pass

        self.rates = [
            float(period["lambda_pax_per_min"])
            for period in demand_periods
        ]

        if len(self.rates) != self.NT:
            raise ValueError("Number of demand periods does not match NT.")

        # The schema states that equal-length periods partition the horizon.
        # Use the exact horizon as the final endpoint to avoid roundoff.
        self.breakpoints = [i * self.theta for i in range(self.NT + 1)]
        self.breakpoints[-1] = self.T

        self.cumulative = [0.0]
        for i in range(self.NT):
            length = self.breakpoints[i + 1] - self.breakpoints[i]
            self.cumulative.append(
                self.cumulative[-1] + length * self.rates[i]
            )

        self.total_demand = self.cumulative[-1]

        # Departures belong to [0,T). This small margin implements the strict
        # upper endpoint without materially changing the solution.
        self.time_epsilon = min(1e-7, max(1e-10, self.T * 1e-10))
        self.latest_departure = max(0.0, self.T - self.time_epsilon)

    def cumulative_arrivals(self, t):
        if t <= 0.0:
            return 0.0
        if t >= self.T:
            return self.total_demand

        i = bisect.bisect_right(self.breakpoints, t) - 1
        i = max(0, min(i, self.NT - 1))
        return self.cumulative[i] + (t - self.breakpoints[i]) * self.rates[i]

    def evaluate(self, schedule):
        """Evaluate a schedule using capacity-maximizing boarding."""
        previous_time = 0.0
        previous_arrivals = 0.0
        queue_after_previous_train = 0.0
        total_wait = 0.0

        for departure in schedule:
            arrivals_at_departure = self.cumulative_arrivals(departure)
            new_arrivals = arrivals_at_departure - previous_arrivals
            headway = departure - previous_time

            total_wait += headway * (
                queue_after_previous_train + 0.5 * new_arrivals
            )

            available = queue_after_previous_train + new_arrivals
            queue_after_previous_train = max(
                0.0, available - self.capacity
            )

            previous_time = departure
            previous_arrivals = arrivals_at_departure

        final_arrivals = self.total_demand - previous_arrivals
        final_headway = self.T - previous_time
        total_wait += final_headway * (
            queue_after_previous_train + 0.5 * final_arrivals
        )

        return float(total_wait)

    def feasible(self, schedule, tolerance=1e-6):
        if len(schedule) != self.D:
            return False
        if any(not math.isfinite(t) for t in schedule):
            return False
        if schedule[0] < -tolerance:
            return False
        if schedule[-1] >= self.T + tolerance:
            return False

        for i in range(1, self.D):
            gap = schedule[i] - schedule[i - 1]
            if gap < self.h_min - tolerance:
                return False
            if gap > self.h_max + tolerance:
                return False
        return True


def fixed_gap_initial_schedules(data):
    D = data.D
    T = data.T
    latest = data.latest_departure

    if D == 1:
        candidates = [[0.0], [0.5 * latest], [latest]]
        for boundary in data.breakpoints[:-1]:
            candidates.append([min(boundary, latest)])
        return candidates

    max_fitting_gap = latest / (D - 1)
    if data.h_min > max_fitting_gap + 1e-9:
        return []

    candidate_gaps = {
        data.h_min,
        min(data.h_max, max_fitting_gap),
        min(max(data.h_min, T / (D + 1)), data.h_max, max_fitting_gap),
        min(max(data.h_min, T / D), data.h_max, max_fitting_gap),
        min(max(data.h_min, T / max(1, D - 1)), data.h_max, max_fitting_gap),
    }

    schedules = []
    for gap in candidate_gaps:
        if gap < data.h_min - 1e-9 or gap > data.h_max + 1e-9:
            continue
        block = (D - 1) * gap
        slack = latest - block
        if slack < -1e-9:
            continue

        starts = {0.0, max(0.0, 0.5 * slack), max(0.0, slack)}
        for start in starts:
            schedules.append([start + i * gap for i in range(D)])

    return schedules


def coordinate_improve(data, initial, deadline, incumbent_callback):
    schedule = list(initial)
    objective = data.evaluate(schedule)
    incumbent_callback(schedule, objective)

    if data.D == 0:
        return schedule, objective

    initial_step = max(data.T / 8.0, 1e-4)
    steps = [initial_step / (2 ** k) for k in range(15)]

    for step in steps:
        if time.monotonic() >= deadline:
            break

        improved_in_round = True
        sweeps = 0

        while improved_in_round and sweeps < 3:
            improved_in_round = False
            sweeps += 1

            for j in range(data.D):
                if time.monotonic() >= deadline:
                    return schedule, objective

                if data.D == 1:
                    lower = 0.0
                    upper = data.latest_departure
                else:
                    lower = 0.0 if j == 0 else schedule[j - 1] + data.h_min
                    upper = (
                        data.latest_departure
                        if j == data.D - 1
                        else schedule[j + 1] - data.h_min
                    )

                    if j > 0:
                        upper = min(upper, schedule[j - 1] + data.h_max)
                    if j < data.D - 1:
                        lower = max(lower, schedule[j + 1] - data.h_max)

                if lower > upper + 1e-10:
                    continue

                current = schedule[j]
                candidates = {
                    current,
                    lower,
                    upper,
                    0.5 * (lower + upper),
                    max(lower, min(upper, current - step)),
                    max(lower, min(upper, current + step)),
                }

                # Include nearby demand boundaries, where the objective's
                # piecewise-quadratic expression can change.
                position = bisect.bisect_left(data.breakpoints, current)
                for k in range(max(0, position - 3),
                               min(len(data.breakpoints), position + 4)):
                    boundary = data.breakpoints[k]
                    if lower <= boundary <= upper and boundary < data.T:
                        candidates.add(boundary)

                best_value = objective
                best_time = current

                for candidate in candidates:
                    if candidate < lower - 1e-10 or candidate > upper + 1e-10:
                        continue
                    trial = list(schedule)
                    trial[j] = candidate
                    value = data.evaluate(trial)
                    if value < best_value - 1e-9:
                        best_value = value
                        best_time = candidate

                if best_time != current:
                    schedule[j] = best_time
                    objective = best_value
                    improved_in_round = True
                    incumbent_callback(schedule, objective)

    return schedule, objective


def optimize_with_gurobi(data, warm_schedule, deadline, incumbent_callback):
    remaining = deadline - time.monotonic()
    if remaining <= 0.02:
        return None

    model = gp.Model("train_departure_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.NonConvex = 2

    x = model.addVars(
        data.D,
        lb=0.0,
        ub=data.latest_departure,
        vtype=GRB.CONTINUOUS,
        name="departure",
    )
    cumulative = model.addVars(
        data.D,
        lb=0.0,
        ub=data.total_demand,
        vtype=GRB.CONTINUOUS,
        name="cumulative_arrivals",
    )

    for j in range(1, data.D):
        model.addConstr(x[j] - x[j - 1] >= data.h_min)
        model.addConstr(x[j] - x[j - 1] <= data.h_max)

    for j in range(data.D):
        model.addGenConstrPWL(
            x[j],
            cumulative[j],
            data.breakpoints,
            data.cumulative,
            name=f"arrival_pwl_{j}",
        )

    max_queue = max(1.0, data.total_demand)
    big_m = data.total_demand + data.capacity + 1.0

    available = model.addVars(
        data.D, lb=0.0, ub=max_queue, name="available"
    )
    queue = model.addVars(
        data.D, lb=0.0, ub=max_queue, name="queue_after"
    )
    overloaded = model.addVars(
        data.D, vtype=GRB.BINARY, name="overloaded"
    )

    for j in range(data.D):
        if j == 0:
            new_arrivals = cumulative[j]
            previous_queue = 0.0
        else:
            new_arrivals = cumulative[j] - cumulative[j - 1]
            previous_queue = queue[j - 1]

        model.addConstr(
            available[j] == previous_queue + new_arrivals,
            name=f"available_balance_{j}",
        )

        # queue[j] = max(available[j] - capacity, 0)
        model.addConstr(
            queue[j] >= available[j] - data.capacity,
            name=f"queue_lower_{j}",
        )
        model.addConstr(
            queue[j]
            <= available[j] - data.capacity
            + big_m * (1.0 - overloaded[j]),
            name=f"queue_overload_upper_{j}",
        )
        model.addConstr(
            queue[j] <= big_m * overloaded[j],
            name=f"queue_zero_upper_{j}",
        )

    objective = gp.QuadExpr()

    for interval in range(data.D + 1):
        if interval == 0:
            headway = x[0]
            new_arrivals = cumulative[0]
            start_queue = 0.0
        elif interval < data.D:
            headway = x[interval] - x[interval - 1]
            new_arrivals = cumulative[interval] - cumulative[interval - 1]
            start_queue = queue[interval - 1]
        else:
            headway = data.T - x[data.D - 1]
            new_arrivals = data.total_demand - cumulative[data.D - 1]
            start_queue = queue[data.D - 1]

        objective += headway * (start_queue + 0.5 * new_arrivals)

    model.setObjective(objective, GRB.MINIMIZE)

    if warm_schedule is not None:
        previous_a = 0.0
        previous_q = 0.0
        for j, departure in enumerate(warm_schedule):
            a = data.cumulative_arrivals(departure)
            arrivals = a - previous_a
            avail = previous_q + arrivals
            q = max(0.0, avail - data.capacity)

            x[j].Start = departure
            cumulative[j].Start = a
            available[j].Start = avail
            queue[j].Start = q
            overloaded[j].Start = 1.0 if avail >= data.capacity else 0.0

            previous_a = a
            previous_q = q

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            schedule = [
                float(cb_model.cbGetSolution(x[j])) for j in range(data.D)
            ]
            if data.feasible(schedule, tolerance=2e-5):
                value = data.evaluate(schedule)
                incumbent_callback(schedule, value)
        except gp.GurobiError:
            pass

    remaining = max(0.01, deadline - time.monotonic())
    model.Params.TimeLimit = remaining
    model.optimize(callback)

    if model.SolCount > 0:
        schedule = [float(x[j].X) for j in range(data.D)]
        if data.feasible(schedule, tolerance=2e-5):
            value = data.evaluate(schedule)
            incumbent_callback(schedule, value)
            return schedule, value

    return None


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    data = ProblemData(instance)

    if data.D <= 0:
        raise ValueError("D_num_train_services must be positive.")

    best_schedule = None
    best_objective = math.inf

    def register_incumbent(schedule, objective):
        nonlocal best_schedule, best_objective

        if not data.feasible(schedule, tolerance=2e-5):
            return

        if objective < best_objective - 1e-8:
            best_schedule = [float(t) for t in schedule]
            best_objective = float(objective)
            solution = make_solution(best_schedule, best_objective)
            if logger:
                logger.log_solution(best_objective, solution)

    initial_schedules = fixed_gap_initial_schedules(data)
    if not initial_schedules:
        raise ValueError(
            "No feasible schedule exists under the interior headway bounds."
        )

    for schedule in initial_schedules:
        register_incumbent(schedule, data.evaluate(schedule))

    # Reserve most of the budget for the exact nonconvex MIQCP search.
    heuristic_budget = min(
        2.0,
        max(0.0, 0.12 * args.time_limit),
    )
    heuristic_deadline = min(deadline, start_time + heuristic_budget)

    for initial in initial_schedules:
        if time.monotonic() >= heuristic_deadline:
            break
        coordinate_improve(
            data,
            initial,
            heuristic_deadline,
            register_incumbent,
        )

    if time.monotonic() < deadline:
        try:
            optimize_with_gurobi(
                data,
                best_schedule,
                deadline,
                register_incumbent,
            )
        except gp.GurobiError:
            # The heuristic incumbent remains a valid fallback.
            pass

    if best_schedule is None:
        raise RuntimeError("Failed to construct a feasible departure schedule.")

    # Re-evaluate to ensure the reported value exactly matches the output
    # schedule and the deterministic capacity-maximizing boarding policy.
    best_objective = data.evaluate(best_schedule)
    final_solution = make_solution(best_schedule, best_objective)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()