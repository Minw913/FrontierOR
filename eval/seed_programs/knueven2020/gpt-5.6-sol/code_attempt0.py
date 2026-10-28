import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the thermal unit commitment problem.")
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def production_cost_points(generator):
    """
    Construct the piecewise-linear production cost function by integrating
    each segment's marginal cost over its absolute MW interval.
    """
    p_min = float(generator["p_min"])
    p_max = float(generator["p_max"])
    segments = generator.get("production_cost_segments", [])

    if p_max <= 0.0 or not segments:
        return [0.0, max(0.0, p_max)], [0.0, 0.0]

    cleaned_segments = []
    breakpoints = {0.0, p_min, p_max}

    for segment in segments:
        start = max(0.0, float(segment["output_mw_start"]))
        end = min(p_max, float(segment["output_mw_end"]))
        slope = float(segment["marginal_cost_per_mwh"])
        if end > start + 1e-12:
            cleaned_segments.append((start, end, slope))
            breakpoints.add(start)
            breakpoints.add(end)

    points = sorted(x for x in breakpoints if -1e-12 <= x <= p_max + 1e-12)

    unique_points = []
    for x in points:
        x = min(p_max, max(0.0, x))
        if not unique_points or x > unique_points[-1] + 1e-10:
            unique_points.append(x)

    if not unique_points or unique_points[0] > 1e-10:
        unique_points.insert(0, 0.0)
    if unique_points[-1] < p_max - 1e-10:
        unique_points.append(p_max)

    costs = []
    for x in unique_points:
        value = 0.0
        for start, end, slope in cleaned_segments:
            quantity = max(0.0, min(x, end) - start)
            value += slope * quantity
        costs.append(value)

    return unique_points, costs


def startup_tier_bounds(generator, period_hours):
    """
    Return (minimum prior downtime periods, maximum prior downtime periods,
    cost) for each startup tier. A None upper bound denotes the final tier.
    """
    tiers = generator.get("startup_costs", [])
    bounds = []
    previous_max_hours = None

    for index, tier in enumerate(tiers):
        if index == 0:
            lower_periods = 0
        else:
            lower_periods = int(math.floor(previous_max_hours / period_hours)) + 1

        maximum_hours = tier.get("max_downtime_hours")
        if maximum_hours is None:
            upper_periods = None
        else:
            maximum_hours = float(maximum_hours)
            upper_periods = int(math.floor(maximum_hours / period_hours + 1e-12))
            previous_max_hours = maximum_hours

        bounds.append((lower_periods, upper_periods, float(tier["cost"])))

    return bounds


def main():
    args = parse_args()
    wall_start = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    generators = instance["generators"]
    num_generators = len(generators)
    num_periods = int(instance["num_time_periods"])
    period_hours = float(instance["time_period_hours"])
    demand = [float(x) for x in instance["demand"]]
    reserve = [float(x) for x in instance["reserve_requirement"]]

    model = gp.Model("thermal_unit_commitment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    u = {}
    startup = {}
    shutdown = {}
    p = {}
    production_cost = {}
    downtime = {}
    startup_type = {}

    for g, generator in enumerate(generators):
        p_max = float(generator["p_max"])
        initial_status = int(generator["initial_status"])
        initial_off_periods = max(0, -initial_status)
        max_downtime = initial_off_periods + num_periods + 1

        for t in range(num_periods):
            u[g, t] = model.addVar(vtype=GRB.BINARY, name=f"u[{g},{t}]")
            startup[g, t] = model.addVar(vtype=GRB.BINARY, name=f"start[{g},{t}]")
            shutdown[g, t] = model.addVar(vtype=GRB.BINARY, name=f"stop[{g},{t}]")
            p[g, t] = model.addVar(lb=0.0, ub=p_max, vtype=GRB.CONTINUOUS,
                                   name=f"p[{g},{t}]")
            production_cost[g, t] = model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS, name=f"prod_cost[{g},{t}]"
            )
            downtime[g, t] = model.addVar(
                lb=0.0,
                ub=max_downtime,
                vtype=GRB.CONTINUOUS,
                name=f"downtime[{g},{t}]",
            )

    model.update()

    objective_terms = []

    for g, generator in enumerate(generators):
        p_min = float(generator["p_min"])
        p_max = float(generator["p_max"])
        ramp_up = float(generator["ramp_up_limit"])
        ramp_down = float(generator["ramp_down_limit"])
        startup_ramp = float(generator["startup_ramp_limit"])
        shutdown_ramp = float(generator["shutdown_ramp_limit"])
        min_up = int(generator["min_up_time"])
        min_down = int(generator["min_down_time"])
        no_load_cost = float(generator["no_load_cost"])
        initial_status = int(generator["initial_status"])
        initial_power = float(generator["initial_power"])
        initial_on = 1 if initial_status > 0 else 0
        initial_off_periods = max(0, -initial_status)
        max_downtime = initial_off_periods + num_periods + 1
        big_m_downtime = max_downtime + 1

        x_points, y_points = production_cost_points(generator)

        for t in range(num_periods):
            model.addConstr(p[g, t] >= p_min * u[g, t],
                            name=f"minimum_output[{g},{t}]")
            model.addConstr(p[g, t] <= p_max * u[g, t],
                            name=f"maximum_output[{g},{t}]")

            if p_max <= 1e-12 or len(x_points) < 2:
                model.addConstr(production_cost[g, t] == 0.0,
                                name=f"zero_production_cost[{g},{t}]")
            else:
                model.addGenConstrPWL(
                    p[g, t],
                    production_cost[g, t],
                    x_points,
                    y_points,
                    name=f"piecewise_cost[{g},{t}]",
                )

            objective_terms.append(no_load_cost * u[g, t])
            objective_terms.append(period_hours * production_cost[g, t])

            previous_u = initial_on if t == 0 else u[g, t - 1]
            previous_p = initial_power if t == 0 else p[g, t - 1]

            model.addConstr(
                startup[g, t] - shutdown[g, t] == u[g, t] - previous_u,
                name=f"transition[{g},{t}]",
            )
            model.addConstr(
                startup[g, t] + shutdown[g, t] <= 1,
                name=f"single_transition[{g},{t}]",
            )

            model.addConstr(
                p[g, t] - previous_p
                <= ramp_up * previous_u + startup_ramp * startup[g, t],
                name=f"ramp_up[{g},{t}]",
            )
            model.addConstr(
                previous_p - p[g, t]
                <= ramp_down * u[g, t] + shutdown_ramp * shutdown[g, t],
                name=f"ramp_down[{g},{t}]",
            )

            previous_downtime = initial_off_periods if t == 0 else downtime[g, t - 1]
            model.addConstr(
                downtime[g, t] <= max_downtime * (1 - u[g, t]),
                name=f"downtime_zero_when_on[{g},{t}]",
            )
            model.addConstr(
                downtime[g, t] <= previous_downtime + 1,
                name=f"downtime_upper[{g},{t}]",
            )
            model.addConstr(
                downtime[g, t]
                >= previous_downtime + 1 - big_m_downtime * u[g, t],
                name=f"downtime_lower[{g},{t}]",
            )

        # Minimum up-time: a startup in the preceding min_up periods implies on.
        if min_up > 0:
            for t in range(num_periods):
                first = max(0, t - min_up + 1)
                model.addConstr(
                    gp.quicksum(startup[g, k] for k in range(first, t + 1))
                    <= u[g, t],
                    name=f"minimum_up_time[{g},{t}]",
                )

        # Minimum down-time: a shutdown in the preceding min_down periods implies off.
        if min_down > 0:
            for t in range(num_periods):
                first = max(0, t - min_down + 1)
                model.addConstr(
                    gp.quicksum(shutdown[g, k] for k in range(first, t + 1))
                    <= 1 - u[g, t],
                    name=f"minimum_down_time[{g},{t}]",
                )

        # Residual initial minimum up/down obligations.
        if initial_status > 0:
            remaining_up = max(0, min_up - initial_status)
            for t in range(min(num_periods, remaining_up)):
                model.addConstr(u[g, t] == 1, name=f"initial_up_obligation[{g},{t}]")
        else:
            remaining_down = max(0, min_down - initial_off_periods)
            for t in range(min(num_periods, remaining_down)):
                model.addConstr(u[g, t] == 0,
                                name=f"initial_down_obligation[{g},{t}]")

        # Downtime-dependent startup costs.
        tier_bounds = startup_tier_bounds(generator, period_hours)
        if tier_bounds:
            for t in range(num_periods):
                previous_downtime = (
                    initial_off_periods if t == 0 else downtime[g, t - 1]
                )
                tier_variables = []

                for k, (lower, upper, cost) in enumerate(tier_bounds):
                    w = model.addVar(vtype=GRB.BINARY,
                                     name=f"startup_type[{g},{t},{k}]")
                    startup_type[g, t, k] = w
                    tier_variables.append(w)

                    # Disable tiers whose integer downtime interval is empty.
                    if upper is not None and lower > upper:
                        w.UB = 0.0
                    else:
                        model.addConstr(
                            previous_downtime
                            >= lower - big_m_downtime * (1 - w),
                            name=f"startup_tier_lower[{g},{t},{k}]",
                        )
                        if upper is not None:
                            model.addConstr(
                                previous_downtime
                                <= upper + big_m_downtime * (1 - w),
                                name=f"startup_tier_upper[{g},{t},{k}]",
                            )

                    objective_terms.append(cost * w)

                model.addConstr(
                    gp.quicksum(tier_variables) == startup[g, t],
                    name=f"choose_startup_tier[{g},{t}]",
                )

    # Demand and reserve requirements.
    for t in range(num_periods):
        model.addConstr(
            gp.quicksum(p[g, t] for g in range(num_generators)) == demand[t],
            name=f"demand_balance[{t}]",
        )
        model.addConstr(
            gp.quicksum(
                float(generators[g]["p_max"]) * u[g, t]
                for g in range(num_generators)
            ) >= demand[t] + reserve[t],
            name=f"reserve_requirement[{t}]",
        )

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    model.update()

    elapsed = time.monotonic() - wall_start
    model.Params.TimeLimit = max(0.0, float(args.time_limit) - elapsed)

    best_logged_objective = [math.inf]

    def make_solution(value_getter, objective_value):
        schedule = {}
        for g, generator in enumerate(generators):
            generator_id = str(generator["id"])
            p_max = float(generator["p_max"])
            u_values = []
            p_values = []
            pb_values = []

            for t in range(num_periods):
                on = 1 if float(value_getter(u[g, t])) >= 0.5 else 0
                power = float(value_getter(p[g, t]))
                if on == 0 or abs(power) < 1e-8:
                    power = 0.0

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

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        objective_value = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
        tolerance = 1e-7 * max(1.0, abs(objective_value))
        if objective_value >= best_logged_objective[0] - tolerance:
            return

        best_logged_objective[0] = objective_value
        if logger:
            solution = make_solution(cb_model.cbGetSolution, objective_value)
            logger.log_solution(objective_value, solution)

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

    final_objective = float(model.ObjVal)
    final_solution = make_solution(lambda var: var.X, final_objective)

    tolerance = 1e-7 * max(1.0, abs(final_objective))
    if logger and final_objective < best_logged_objective[0] - tolerance:
        logger.log_solution(final_objective, final_solution)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()