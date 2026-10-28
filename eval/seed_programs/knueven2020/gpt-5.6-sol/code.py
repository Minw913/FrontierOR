import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Thermal unit commitment solver")
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    return parser.parse_args()


def build_production_curve(generator, period_hours):
    """
    Build total-output breakpoints and cumulative production costs.

    Marginal costs are given in $/MWh. Cost ordinates are multiplied by the
    period duration so that they can be used directly in the objective.
    """
    p_max = float(generator["p_max"])
    raw_segments = generator.get("production_cost_segments", [])

    segments = []
    breakpoints = {0.0, p_max}

    for segment in raw_segments:
        left = max(
            0.0,
            min(p_max, float(segment["output_mw_start"])),
        )
        right = max(
            0.0,
            min(p_max, float(segment["output_mw_end"])),
        )
        slope = float(segment["marginal_cost_per_mwh"])

        if right > left + 1e-10:
            segments.append((left, right, slope))
            breakpoints.add(left)
            breakpoints.add(right)

    x_values = sorted(breakpoints)

    if len(x_values) == 1:
        return x_values, [0.0]

    y_values = []
    for output in x_values:
        cost = 0.0
        for left, right, slope in segments:
            covered_output = max(0.0, min(output, right) - left)
            cost += covered_output * slope
        y_values.append(period_hours * cost)

    return x_values, y_values


def get_startup_tiers(generator, period_hours):
    """
    Convert startup cost tier bounds from hours into completed time periods.

    Each returned tuple is (minimum_downtime_periods,
                            maximum_downtime_periods_or_None,
                            startup_cost).
    """
    finite_tiers = []
    unbounded_tiers = []

    for tier in generator.get("startup_costs", []):
        maximum_hours = tier.get("max_downtime_hours")
        cost = float(tier["cost"])

        if maximum_hours is None:
            unbounded_tiers.append(cost)
        else:
            maximum_periods = int(
                math.floor(float(maximum_hours) / period_hours + 1e-9)
            )
            finite_tiers.append((maximum_periods, cost))

    finite_tiers.sort(key=lambda item: item[0])

    result = []
    lower_bound = 0

    for upper_bound, cost in finite_tiers:
        if upper_bound >= lower_bound:
            result.append((lower_bound, upper_bound, cost))
            lower_bound = upper_bound + 1

    if unbounded_tiers:
        result.append((lower_bound, None, unbounded_tiers[0]))
    elif finite_tiers:
        # Be robust to an omitted final unbounded tier.
        result.append((lower_bound, None, finite_tiers[-1][1]))

    return result


def main():
    args = parse_args()
    wall_start = time.monotonic()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    generators = instance["generators"]
    num_generators = len(generators)
    num_periods = int(instance["num_time_periods"])
    period_hours = float(instance["time_period_hours"])

    demand = [float(value) for value in instance["demand"]]
    reserve_requirement = [
        float(value) for value in instance["reserve_requirement"]
    ]

    G = range(num_generators)
    T = range(num_periods)

    model = gp.Model("thermal_unit_commitment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.15

    elapsed = time.monotonic() - wall_start
    model.Params.TimeLimit = max(
        0.05,
        float(args.time_limit) - elapsed,
    )

    u = {}
    startup = {}
    shutdown = {}
    p = {}
    off_age = {}
    startup_tier = {}

    for g in G:
        generator = generators[g]
        p_max = float(generator["p_max"])
        initial_status = int(generator["initial_status"])
        initial_off_age = max(0, -initial_status)
        maximum_off_age = initial_off_age + num_periods + 1

        for t in T:
            u[g, t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"u[{g},{t}]",
            )
            startup[g, t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"startup[{g},{t}]",
            )
            shutdown[g, t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"shutdown[{g},{t}]",
            )
            p[g, t] = model.addVar(
                lb=0.0,
                ub=p_max,
                vtype=GRB.CONTINUOUS,
                name=f"p[{g},{t}]",
            )
            off_age[g, t] = model.addVar(
                lb=0.0,
                ub=maximum_off_age,
                vtype=GRB.CONTINUOUS,
                name=f"off_age[{g},{t}]",
            )

    model.update()

    linear_objective = gp.LinExpr()

    for g in G:
        generator = generators[g]

        p_min = float(generator["p_min"])
        p_max = float(generator["p_max"])
        ramp_up = float(generator["ramp_up_limit"])
        ramp_down = float(generator["ramp_down_limit"])
        startup_ramp = float(generator["startup_ramp_limit"])
        shutdown_ramp = float(generator["shutdown_ramp_limit"])
        minimum_up_time = max(0, int(generator["min_up_time"]))
        minimum_down_time = max(0, int(generator["min_down_time"]))
        no_load_cost = float(generator["no_load_cost"])

        initial_status = int(generator["initial_status"])
        initial_on = 1 if initial_status > 0 else 0
        initial_off_age = max(0, -initial_status)
        initial_power = (
            float(generator["initial_power"]) if initial_on else 0.0
        )

        maximum_off_age = initial_off_age + num_periods + 1

        # This M is sufficient for the off-age recursion only.
        age_recursion_m = maximum_off_age + 1

        tiers = get_startup_tiers(generator, period_hours)

        # Startup tier thresholds can be much larger than the planning
        # horizon. The previous implementation used maximum_off_age as M for
        # tier-selection constraints, which could make inactive long-downtime
        # tiers impose positive lower bounds and render valid instances
        # infeasible. Use a separate M covering every tier threshold.
        maximum_tier_threshold = 0
        for lower, upper, _ in tiers:
            maximum_tier_threshold = max(
                maximum_tier_threshold,
                int(lower),
            )
            if upper is not None:
                maximum_tier_threshold = max(
                    maximum_tier_threshold,
                    int(upper),
                )

        tier_m = (
            maximum_off_age
            + maximum_tier_threshold
            + 2
        )

        curve_x, curve_y = build_production_curve(
            generator,
            period_hours,
        )

        for t in T:
            linear_objective += no_load_cost * u[g, t]

            model.addConstr(
                p[g, t] >= p_min * u[g, t],
                name=f"minimum_output[{g},{t}]",
            )
            model.addConstr(
                p[g, t] <= p_max * u[g, t],
                name=f"maximum_output[{g},{t}]",
            )

            if len(curve_x) >= 2 and any(
                abs(value) > 1e-12 for value in curve_y
            ):
                model.setPWLObj(p[g, t], curve_x, curve_y)

            if t == 0:
                previous_u = initial_on
                previous_p = initial_power
                previous_off_age = initial_off_age
            else:
                previous_u = u[g, t - 1]
                previous_p = p[g, t - 1]
                previous_off_age = off_age[g, t - 1]

            model.addConstr(
                u[g, t] - previous_u
                == startup[g, t] - shutdown[g, t],
                name=f"transition[{g},{t}]",
            )
            model.addConstr(
                startup[g, t] + shutdown[g, t] <= 1,
                name=f"exclusive_transition[{g},{t}]",
            )

            model.addConstr(
                p[g, t] - previous_p
                <= ramp_up * previous_u
                + startup_ramp * startup[g, t],
                name=f"ramp_up[{g},{t}]",
            )

            model.addConstr(
                previous_p - p[g, t]
                <= ramp_down * u[g, t]
                + shutdown_ramp * shutdown[g, t],
                name=f"ramp_down[{g},{t}]",
            )

            # off_age is the number of completed consecutive off periods
            # through period t. It is zero whenever the generator is online.
            model.addConstr(
                off_age[g, t]
                <= maximum_off_age * (1 - u[g, t]),
                name=f"off_age_online_zero[{g},{t}]",
            )
            model.addConstr(
                off_age[g, t] <= previous_off_age + 1,
                name=f"off_age_upper[{g},{t}]",
            )
            model.addConstr(
                off_age[g, t]
                >= previous_off_age + 1
                - age_recursion_m * u[g, t],
                name=f"off_age_lower[{g},{t}]",
            )

        for t in T:
            if minimum_up_time > 0:
                first_period = max(
                    0,
                    t - minimum_up_time + 1,
                )
                model.addConstr(
                    gp.quicksum(
                        startup[g, k]
                        for k in range(first_period, t + 1)
                    )
                    <= u[g, t],
                    name=f"minimum_up_time[{g},{t}]",
                )

            if minimum_down_time > 0:
                first_period = max(
                    0,
                    t - minimum_down_time + 1,
                )
                model.addConstr(
                    gp.quicksum(
                        shutdown[g, k]
                        for k in range(first_period, t + 1)
                    )
                    <= 1 - u[g, t],
                    name=f"minimum_down_time[{g},{t}]",
                )

        # Enforce residual pre-horizon minimum up/down-time obligations.
        if initial_status > 0:
            remaining_up_time = max(
                0,
                minimum_up_time - initial_status,
            )
            for t in range(min(num_periods, remaining_up_time)):
                model.addConstr(
                    u[g, t] == 1,
                    name=f"initial_up_obligation[{g},{t}]",
                )
        else:
            remaining_down_time = max(
                0,
                minimum_down_time - initial_off_age,
            )
            for t in range(min(num_periods, remaining_down_time)):
                model.addConstr(
                    u[g, t] == 0,
                    name=f"initial_down_obligation[{g},{t}]",
                )

        if tiers:
            for t in T:
                prior_off_age = (
                    initial_off_age
                    if t == 0
                    else off_age[g, t - 1]
                )

                tier_variables = []

                for k, (lower, upper, cost) in enumerate(tiers):
                    z = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"startup_tier[{g},{t},{k}]",
                    )
                    startup_tier[g, t, k] = z
                    tier_variables.append(z)
                    linear_objective += cost * z

                    model.addConstr(
                        prior_off_age
                        >= lower - tier_m * (1 - z),
                        name=f"startup_tier_lower[{g},{t},{k}]",
                    )

                    if upper is not None:
                        model.addConstr(
                            prior_off_age
                            <= upper + tier_m * (1 - z),
                            name=f"startup_tier_upper[{g},{t},{k}]",
                        )

                model.addConstr(
                    gp.quicksum(tier_variables)
                    == startup[g, t],
                    name=f"startup_tier_choice[{g},{t}]",
                )

    for t in T:
        model.addConstr(
            gp.quicksum(p[g, t] for g in G) == demand[t],
            name=f"demand_balance[{t}]",
        )

        # pb can always be set to p_max*u without affecting cost. Therefore,
        # reserve feasibility can be represented directly by committed
        # capacity and pb reconstructed when producing the solution.
        model.addConstr(
            gp.quicksum(
                float(generators[g]["p_max"]) * u[g, t]
                for g in G
            )
            >= demand[t] + reserve_requirement[t],
            name=f"reserve_requirement[{t}]",
        )

    model.setObjective(linear_objective, GRB.MINIMIZE)
    model.update()

    best_logged_objective = [math.inf]

    def make_solution(value_getter, objective_value):
        schedule = {}

        for g in G:
            generator = generators[g]
            generator_id = str(generator["id"])
            p_max = float(generator["p_max"])

            u_values = []
            p_values = []
            pb_values = []

            for t in T:
                status_value = float(value_getter(u[g, t]))
                on = 1 if status_value >= 0.5 else 0

                power = float(value_getter(p[g, t]))
                if not on or abs(power) < 1e-8:
                    power = 0.0
                else:
                    power = max(0.0, min(p_max, power))

                u_values.append(on)
                p_values.append(power)
                pb_values.append(p_max if on else 0.0)

            schedule[generator_id] = {
                "u": u_values,
                "p": p_values,
                "pb": pb_values,
            }

        return {
            "objective_value": float(objective_value),
            "schedule": schedule,
        }

    def incumbent_callback(callback_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        objective_value = float(
            callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        )
        tolerance = 1e-7 * max(1.0, abs(objective_value))

        if objective_value >= best_logged_objective[0] - tolerance:
            return

        best_logged_objective[0] = objective_value

        if logger:
            solution = make_solution(
                callback_model.cbGetSolution,
                objective_value,
            )
            logger.log_solution(objective_value, solution)

    model.optimize(incumbent_callback)

    if model.SolCount == 0 and model.Status == GRB.INF_OR_UNBD:
        remaining_time = max(
            0.01,
            float(args.time_limit)
            - (time.monotonic() - wall_start),
        )
        model.reset()
        model.Params.DualReductions = 0
        model.Params.TimeLimit = remaining_time
        model.optimize(incumbent_callback)

    if model.SolCount == 0:
        raise RuntimeError(
            f"No feasible schedule found; Gurobi status={model.Status}"
        )

    final_objective = float(model.ObjVal)
    final_solution = make_solution(
        lambda variable: variable.X,
        final_objective,
    )

    tolerance = 1e-7 * max(1.0, abs(final_objective))
    if (
        logger
        and final_objective
        < best_logged_objective[0] - tolerance
    ):
        logger.log_solution(final_objective, final_solution)

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(
            final_solution,
            file,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()