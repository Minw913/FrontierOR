import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_periods = int(data["num_time_periods"])
    periods = list(range(num_periods))
    max_displacement = int(data["maximum_displacement_delta"])

    # The narrative specifies rho, although some schema versions omit it.
    rho = float(
        data.get(
            "rho",
            data.get(
                "weight_rho",
                data.get("schedule_cost_weight", 0.5),
            ),
        )
    )
    rho = min(1.0, max(0.0, rho))

    flights = data["flights"]
    flight_ids = [int(f["id"]) for f in flights]
    flight_by_id = {int(f["id"]): f for f in flights}
    scenarios = data["scenarios"]
    airports = data["airports"]

    schedule_domains = {}
    schedule_arrival = {}
    operation_dep_domains = {}
    operation_arr_domains = {}

    for f in flights:
        fid = int(f["id"])
        requested_dep = int(f["dep_period"])
        scheduled_duration = int(f["delta_sch"])

        low = max(0, requested_dep - max_displacement)
        high = min(
            num_periods - 1,
            requested_dep + max_displacement,
            num_periods - 1 - scheduled_duration,
        )

        domain = list(range(low, high + 1))
        if not domain:
            raise RuntimeError(
                f"Flight {fid} has no feasible scheduled departure period."
            )

        schedule_domains[fid] = domain
        schedule_arrival[fid] = {
            dep: dep + scheduled_duration for dep in domain
        }

        dep_low = min(domain)
        dep_high = min(
            num_periods - 1,
            max(domain) + int(f["max_dep_delay"]),
        )
        dep_domain = list(range(dep_low, dep_high + 1))
        operation_dep_domains[fid] = dep_domain

        arr_low = max(
            0,
            dep_low + int(f["delta_min"]),
        )
        arr_high = min(
            num_periods - 1,
            max(schedule_arrival[fid].values()) + int(f["max_arr_delay"]),
            dep_high + int(f["delta_max"]),
        )
        arr_domain = list(range(arr_low, arr_high + 1))
        if not arr_domain:
            raise RuntimeError(
                f"Flight {fid} has no feasible operated arrival period."
            )
        operation_arr_domains[fid] = arr_domain

    model = gp.Model("stochastic_air_traffic_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # First-stage schedule selection: z[f,d] = 1 if scheduled departure is d.
    z = {}
    for fid in flight_ids:
        for dep in schedule_domains[fid]:
            z[fid, dep] = model.addVar(vtype=GRB.BINARY)

    # Second-stage exact operated-time variables.
    y_dep = {}
    y_arr = {}
    v_dep = {}
    v_arr = {}

    for scenario in scenarios:
        sid = int(scenario["id"])
        for fid in flight_ids:
            for t in operation_dep_domains[fid]:
                y_dep[sid, fid, t] = model.addVar(vtype=GRB.BINARY)
            for t in operation_arr_domains[fid]:
                y_arr[sid, fid, t] = model.addVar(vtype=GRB.BINARY)

            f = flight_by_id[fid]
            v_dep[sid, fid] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=int(f["max_dep_delay"]),
            )
            v_arr[sid, fid] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=int(f["max_arr_delay"]),
            )

    model.update()

    scheduled_dep_expr = {}
    scheduled_arr_expr = {}
    operated_dep_expr = {}
    operated_arr_expr = {}

    for fid in flight_ids:
        scheduled_dep_expr[fid] = gp.quicksum(
            dep * z[fid, dep] for dep in schedule_domains[fid]
        )
        scheduled_arr_expr[fid] = gp.quicksum(
            schedule_arrival[fid][dep] * z[fid, dep]
            for dep in schedule_domains[fid]
        )
        model.addConstr(
            gp.quicksum(z[fid, dep] for dep in schedule_domains[fid]) == 1
        )

    # Both aircraft and passenger connections constrain the published schedule.
    schedule_connections = (
        data.get("aircraft_connections", [])
        + data.get("passenger_connections", [])
    )
    for connection in schedule_connections:
        fi = int(connection["flight_i"])
        fj = int(connection["flight_j"])
        minimum = int(connection["min_connection_time"])
        model.addConstr(
            scheduled_dep_expr[fj] - scheduled_arr_expr[fi] >= minimum
        )

    for scenario in scenarios:
        sid = int(scenario["id"])

        for fid in flight_ids:
            f = flight_by_id[fid]

            model.addConstr(
                gp.quicksum(
                    y_dep[sid, fid, t]
                    for t in operation_dep_domains[fid]
                )
                == 1
            )
            model.addConstr(
                gp.quicksum(
                    y_arr[sid, fid, t]
                    for t in operation_arr_domains[fid]
                )
                == 1
            )

            operated_dep_expr[sid, fid] = gp.quicksum(
                t * y_dep[sid, fid, t]
                for t in operation_dep_domains[fid]
            )
            operated_arr_expr[sid, fid] = gp.quicksum(
                t * y_arr[sid, fid, t]
                for t in operation_arr_domains[fid]
            )

            # Departure delay is exactly operated departure minus schedule.
            model.addConstr(
                v_dep[sid, fid]
                == operated_dep_expr[sid, fid] - scheduled_dep_expr[fid]
            )

            # Arrival delay is the nonnegative positive part at optimum.
            model.addConstr(
                v_arr[sid, fid]
                >= operated_arr_expr[sid, fid] - scheduled_arr_expr[fid]
            )

            model.addConstr(
                operated_arr_expr[sid, fid] - operated_dep_expr[sid, fid]
                >= int(f["delta_min"])
            )
            model.addConstr(
                operated_arr_expr[sid, fid] - operated_dep_expr[sid, fid]
                <= int(f["delta_max"])
            )

        # Aircraft rotations must remain feasible in actual operations.
        for connection in data.get("aircraft_connections", []):
            fi = int(connection["flight_i"])
            fj = int(connection["flight_j"])
            minimum = int(connection["min_connection_time"])
            model.addConstr(
                operated_dep_expr[sid, fj]
                - operated_arr_expr[sid, fi]
                >= minimum
            )

        conditions = scenario["operating_conditions"]

        # Airport capacity envelopes.
        for airport in airports:
            departing_flights = [
                fid
                for fid in flight_ids
                if flight_by_id[fid]["dep_airport"] == airport
            ]
            arriving_flights = [
                fid
                for fid in flight_ids
                if flight_by_id[fid]["arr_airport"] == airport
            ]

            for t in periods:
                dep_count = gp.quicksum(
                    y_dep[sid, fid, t]
                    for fid in departing_flights
                    if t in operation_dep_domains[fid]
                )
                arr_count = gp.quicksum(
                    y_arr[sid, fid, t]
                    for fid in arriving_flights
                    if t in operation_arr_domains[fid]
                )

                condition = conditions[airport][t]
                segments = data["capacity_envelopes"][airport][condition]

                for segment in segments:
                    model.addConstr(
                        float(segment["a"]) * dep_count
                        + float(segment["b"]) * arr_count
                        <= float(segment["Q"])
                    )

    schedule_cost = gp.quicksum(
        abs(dep - int(flight_by_id[fid]["dep_period"])) * z[fid, dep]
        for fid in flight_ids
        for dep in schedule_domains[fid]
    )

    expected_delay_cost = gp.LinExpr()
    for scenario in scenarios:
        sid = int(scenario["id"])
        probability = float(scenario["probability"])
        for fid in flight_ids:
            f = flight_by_id[fid]
            expected_delay_cost += probability * (
                float(f["cost_dep_delay"]) * v_dep[sid, fid]
                + float(f["cost_arr_delay"]) * v_arr[sid, fid]
            )

    model.setObjective(
        rho * schedule_cost + (1.0 - rho) * expected_delay_cost,
        GRB.MINIMIZE,
    )

    # Supply a simple start based on requested schedules and no initial delay.
    for fid in flight_ids:
        requested = int(flight_by_id[fid]["dep_period"])
        initial_dep = min(
            schedule_domains[fid],
            key=lambda t: (abs(t - requested), t),
        )
        z[fid, initial_dep].Start = 1.0

        initial_arr = schedule_arrival[fid][initial_dep]
        for scenario in scenarios:
            sid = int(scenario["id"])
            if initial_dep in operation_dep_domains[fid]:
                y_dep[sid, fid, initial_dep].Start = 1.0
            if initial_arr in operation_arr_domains[fid]:
                y_arr[sid, fid, initial_arr].Start = 1.0
            v_dep[sid, fid].Start = 0.0
            v_arr[sid, fid].Start = 0.0

    def compute_objective(schedule_times, scenario_times):
        displacement = 0.0
        for fid in flight_ids:
            displacement += abs(
                schedule_times[fid][0]
                - int(flight_by_id[fid]["dep_period"])
            )

        expected = 0.0
        for scenario in scenarios:
            sid = int(scenario["id"])
            probability = float(scenario["probability"])
            scenario_cost = 0.0
            for fid in flight_ids:
                f = flight_by_id[fid]
                scheduled_dep, scheduled_arr = schedule_times[fid]
                operated_dep, operated_arr = scenario_times[sid][fid]
                departure_delay = operated_dep - scheduled_dep
                arrival_delay = max(0, operated_arr - scheduled_arr)
                scenario_cost += (
                    float(f["cost_dep_delay"]) * departure_delay
                    + float(f["cost_arr_delay"]) * arrival_delay
                )
            expected += probability * scenario_cost

        return rho * displacement + (1.0 - rho) * expected

    def make_solution(schedule_times, scenario_times):
        scenario_solutions = []

        for scenario in scenarios:
            sid = int(scenario["id"])
            x_dep_output = {}
            x_arr_output = {}
            v_dep_output = {}
            v_arr_output = {}

            for fid in flight_ids:
                scheduled_dep, scheduled_arr = schedule_times[fid]
                operated_dep, operated_arr = scenario_times[sid][fid]

                for t in periods:
                    x_dep_output[f"{fid}_{t}"] = (
                        1 if operated_dep >= t else 0
                    )
                    x_arr_output[f"{fid}_{t}"] = (
                        1 if operated_arr >= t else 0
                    )

                v_dep_output[str(fid)] = int(operated_dep - scheduled_dep)
                v_arr_output[str(fid)] = int(
                    max(0, operated_arr - scheduled_arr)
                )

            scenario_solutions.append(
                {
                    "scenario_id": sid,
                    "x_dep": x_dep_output,
                    "x_arr": x_arr_output,
                    "v_dep": v_dep_output,
                    "v_arr": v_arr_output,
                }
            )

        objective = compute_objective(schedule_times, scenario_times)

        return {
            "objective_value": float(objective),
            "schedule": {
                str(fid): {
                    "scheduled_dep": int(schedule_times[fid][0]),
                    "scheduled_arr": int(schedule_times[fid][1]),
                }
                for fid in flight_ids
            },
            "scenario_solutions": scenario_solutions,
        }

    def extract_times(value_getter):
        schedule_times = {}
        for fid in flight_ids:
            selected_dep = max(
                schedule_domains[fid],
                key=lambda t: value_getter(z[fid, t]),
            )
            schedule_times[fid] = (
                int(selected_dep),
                int(schedule_arrival[fid][selected_dep]),
            )

        scenario_times = {}
        for scenario in scenarios:
            sid = int(scenario["id"])
            scenario_times[sid] = {}
            for fid in flight_ids:
                selected_dep = max(
                    operation_dep_domains[fid],
                    key=lambda t: value_getter(y_dep[sid, fid, t]),
                )
                selected_arr = max(
                    operation_arr_domains[fid],
                    key=lambda t: value_getter(y_arr[sid, fid, t]),
                )
                scenario_times[sid][fid] = (
                    int(selected_dep),
                    int(selected_arr),
                )

        return schedule_times, scenario_times

    logging_enabled = [logger is not None]

    def incumbent_callback(cb_model, where):
        if (
            where == GRB.Callback.MIPSOL
            and logging_enabled[0]
            and logger is not None
        ):
            try:
                schedule_times, scenario_times = extract_times(
                    lambda variable: cb_model.cbGetSolution(variable)
                )
                solution = make_solution(schedule_times, scenario_times)
                logger.log_solution(
                    solution["objective_value"],
                    solution,
                )
            except Exception:
                # Logging failures must not terminate optimization.
                logging_enabled[0] = False

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)
    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    final_schedule_times, final_scenario_times = extract_times(
        lambda variable: variable.X
    )
    final_solution = make_solution(
        final_schedule_times,
        final_scenario_times,
    )

    if logger:
        try:
            logger.log_solution(
                final_solution["objective_value"],
                final_solution,
            )
        except Exception:
            pass

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()