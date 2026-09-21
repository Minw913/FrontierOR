import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, path)


def scheduling_cost(patient, day_id, day_label):
    costs = patient["c_sched_per_day"]
    candidates = (
        str(day_label),
        day_label,
        str(day_id),
        day_id,
    )
    for key in candidates:
        if key in costs:
            return float(costs[key])
    raise KeyError(
        f"No scheduling cost found for patient {patient['patient_id']} "
        f"on day_id={day_id}, day_label={day_label}"
    )


def make_fallback(instance, locations):
    patients = instance["patients"]
    hospitals = instance["hospitals"]
    num_ors = int(instance["num_ors_per_hospital"])

    mandatory = [i for i, p in enumerate(patients) if bool(p["is_mandatory"])]
    optional = [i for i, p in enumerate(patients) if not bool(p["is_mandatory"])]

    u_out = {}
    y_out = {}
    for hospital in hospitals:
        hid = hospital["hospital_id"]
        for day in hospital["days"]:
            did = day["day_id"]
            u_out[f"{hid},{did}"] = 0
            for r in range(int(hospital.get("num_ors", num_ors))):
                y_out[f"{hid},{did},{r}"] = 0

    w_out = {str(patients[i]["patient_id"]): 1 for i in optional}
    x_out = {}

    objective = sum(float(patients[i]["c_unsched"]) for i in optional)
    chosen_location = None

    if mandatory:
        best_value = float("inf")
        for loc_index, loc in enumerate(locations):
            value = float(loc["G"]) + float(loc["F"])
            for i in mandatory:
                value += scheduling_cost(
                    patients[i], loc["day_id"], loc["day_label"]
                )
                value += float(patients[i]["c_cancel"])
            if value < best_value:
                best_value = value
                chosen_location = loc_index

        if chosen_location is None:
            raise RuntimeError("No hospital-day-OR location exists for mandatory patients")

        loc = locations[chosen_location]
        hid = loc["hospital_id"]
        did = loc["day_id"]

        u_out[f"{hid},{did}"] = 1
        y_out[f"{hid},{did},0"] = 1

        for i in mandatory:
            pid = patients[i]["patient_id"]
            x_out[f"{hid},{did},{pid},0"] = 1

        objective += best_value

    solution = {
        "objective_value": float(objective),
        "u": u_out,
        "y": y_out,
        "x": x_out,
        "w": w_out,
    }
    return solution, chosen_location


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    patients = instance["patients"]
    hospitals = instance["hospitals"]
    scenarios = instance["scenarios"]

    num_patients = len(patients)
    num_scenarios = len(scenarios)
    default_num_ors = int(instance["num_ors_per_hospital"])

    locations = []
    location_lookup = {}
    for hospital in hospitals:
        hid = hospital["hospital_id"]
        n_or = int(hospital.get("num_ors", default_num_ors))
        for day in hospital["days"]:
            did = day["day_id"]
            day_label = day["day_label"]
            for r in range(n_or):
                loc = {
                    "hospital_id": hid,
                    "day_id": did,
                    "day_label": day_label,
                    "or_index": r,
                    "B": float(day["B_hd"]),
                    "F": float(day["F_hd"]),
                    "G": float(day["G_hd"]),
                }
                idx = len(locations)
                locations.append(loc)
                location_lookup[(hid, did, r)] = idx

    fallback, fallback_location = make_fallback(instance, locations)
    best_solution = fallback
    best_objective = float(fallback["objective_value"])

    if logger:
        try:
            logger.log_solution(best_objective, best_solution)
        except Exception:
            pass

    if time.monotonic() >= deadline:
        write_json(args.solution_path, best_solution)
        return

    mandatory_indices = [
        i for i, p in enumerate(patients) if bool(p["is_mandatory"])
    ]
    optional_indices = [
        i for i, p in enumerate(patients) if not bool(p["is_mandatory"])
    ]

    try:
        model = gp.Model("stochastic_operating_room_scheduling")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        # Hospital-day variables and maps.
        u = {}
        hospital_day_data = {}
        for hospital in hospitals:
            hid = hospital["hospital_id"]
            n_or = int(hospital.get("num_ors", default_num_ors))
            for day in hospital["days"]:
                did = day["day_id"]
                key = (hid, did)
                u[key] = model.addVar(vtype=GRB.BINARY, name=f"u[{hid},{did}]")
                hospital_day_data[key] = {
                    "day_label": day["day_label"],
                    "G": float(day["G_hd"]),
                    "F": float(day["F_hd"]),
                    "B": float(day["B_hd"]),
                    "num_ors": n_or,
                }

        num_locations = len(locations)
        x = model.addVars(
            num_locations, num_patients, vtype=GRB.BINARY, name="x"
        )
        y = model.addVars(num_locations, vtype=GRB.BINARY, name="y")

        w = {
            i: model.addVar(vtype=GRB.BINARY, name=f"w[{patients[i]['patient_id']}]")
            for i in optional_indices
        }

        z = model.addVars(
            num_scenarios,
            num_locations,
            num_patients,
            vtype=GRB.BINARY,
            name="z",
        )

        model.update()

        # Assignment constraints.
        for i in range(num_patients):
            assignment = gp.quicksum(x[l, i] for l in range(num_locations))
            if i in w:
                model.addConstr(assignment + w[i] == 1, name=f"optional[{i}]")
            else:
                model.addConstr(assignment == 1, name=f"mandatory[{i}]")

        # Opening and assignment links.
        for l, loc in enumerate(locations):
            hid = loc["hospital_id"]
            did = loc["day_id"]
            model.addConstr(x.sum(l, "*") >= y[l], name=f"used_or[{l}]")
            for i in range(num_patients):
                model.addConstr(x[l, i] <= y[l], name=f"assign_open[{l},{i}]")

        # OR ordering and suite-opening equivalence.
        for (hid, did), data in hospital_day_data.items():
            n_or = data["num_ors"]
            first_l = location_lookup[(hid, did, 0)]
            model.addConstr(u[hid, did] == y[first_l], name=f"suite[{hid},{did}]")

            for r in range(n_or - 1):
                l1 = location_lookup[(hid, did, r)]
                l2 = location_lookup[(hid, did, r + 1)]
                model.addConstr(y[l1] >= y[l2], name=f"or_order[{hid},{did},{r}]")
                model.addConstr(
                    x.sum(l1, "*") >= x.sum(l2, "*"),
                    name=f"load_order[{hid},{did},{r}]",
                )

        # Scenario recourse constraints.
        durations = []
        for scenario in scenarios:
            values = scenario["surgery_durations_minutes"]
            if len(values) != num_patients:
                raise ValueError(
                    f"Scenario {scenario['scenario_id']} has {len(values)} "
                    f"durations, expected {num_patients}"
                )
            durations.append([float(v) for v in values])

        for s in range(num_scenarios):
            for l, loc in enumerate(locations):
                model.addConstr(
                    gp.quicksum(durations[s][i] * z[s, l, i]
                                for i in range(num_patients))
                    <= loc["B"] * y[l],
                    name=f"capacity[{s},{l}]",
                )
                for i in range(num_patients):
                    model.addConstr(
                        z[s, l, i] <= x[l, i],
                        name=f"perform_assigned[{s},{l},{i}]",
                    )

        # Objective.
        objective = gp.LinExpr()

        for (hid, did), var in u.items():
            objective += hospital_day_data[(hid, did)]["G"] * var

        for l, loc in enumerate(locations):
            objective += loc["F"] * y[l]

        scenario_weight = 1.0 / num_scenarios if num_scenarios > 0 else 0.0

        for l, loc in enumerate(locations):
            for i, patient in enumerate(patients):
                sched = scheduling_cost(
                    patient, loc["day_id"], loc["day_label"]
                )
                cancel = float(patient["c_cancel"])
                objective += (sched + cancel) * x[l, i]

        for i in optional_indices:
            objective += float(patients[i]["c_unsched"]) * w[i]

        if num_scenarios > 0:
            for s in range(num_scenarios):
                for l in range(num_locations):
                    for i, patient in enumerate(patients):
                        objective += (
                            -scenario_weight
                            * float(patient["c_cancel"])
                            * z[s, l, i]
                        )

        model.setObjective(objective, GRB.MINIMIZE)

        # Complete feasible MIP start corresponding to the fallback solution.
        for (hid, did), var in u.items():
            var.Start = float(fallback["u"][f"{hid},{did}"])

        for l, loc in enumerate(locations):
            y[l].Start = float(
                fallback["y"][
                    f"{loc['hospital_id']},{loc['day_id']},{loc['or_index']}"
                ]
            )
            for i, patient in enumerate(patients):
                key = (
                    f"{loc['hospital_id']},{loc['day_id']},"
                    f"{patient['patient_id']},{loc['or_index']}"
                )
                x[l, i].Start = 1.0 if key in fallback["x"] else 0.0

        for i in optional_indices:
            w[i].Start = float(fallback["w"][str(patients[i]["patient_id"])])

        for s in range(num_scenarios):
            for l in range(num_locations):
                for i in range(num_patients):
                    z[s, l, i].Start = 0.0

        u_items = list(u.items())
        y_items = [(l, y[l]) for l in range(num_locations)]
        x_items = [
            (l, i, x[l, i])
            for l in range(num_locations)
            for i in range(num_patients)
        ]
        w_items = [(i, w[i]) for i in optional_indices]

        model._logged_best = best_objective

        def build_solution_from_values(objective_value, get_value):
            u_out = {}
            for (hid, did), var in u_items:
                u_out[f"{hid},{did}"] = int(get_value(var) > 0.5)

            y_out = {}
            for l, var in y_items:
                loc = locations[l]
                key = f"{loc['hospital_id']},{loc['day_id']},{loc['or_index']}"
                y_out[key] = int(get_value(var) > 0.5)

            x_out = {}
            for l, i, var in x_items:
                if get_value(var) > 0.5:
                    loc = locations[l]
                    pid = patients[i]["patient_id"]
                    key = (
                        f"{loc['hospital_id']},{loc['day_id']},"
                        f"{pid},{loc['or_index']}"
                    )
                    x_out[key] = 1

            w_out = {}
            for i, var in w_items:
                w_out[str(patients[i]["patient_id"])] = int(get_value(var) > 0.5)

            return {
                "objective_value": float(objective_value),
                "u": u_out,
                "y": y_out,
                "x": x_out,
                "w": w_out,
            }

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj >= cb_model._logged_best - 1e-7:
                    return

                solution = build_solution_from_values(
                    obj, lambda var: cb_model.cbGetSolution(var)
                )
                cb_model._logged_best = obj

                if logger:
                    try:
                        logger.log_solution(obj, solution)
                    except Exception:
                        pass
            except Exception:
                # Logging must never interrupt optimization.
                pass

        remaining = deadline - time.monotonic()
        if remaining > 0:
            model.Params.TimeLimit = max(0.01, remaining)
            model.optimize(incumbent_callback)

        if model.SolCount > 0:
            final_objective = float(model.ObjVal)
            final_solution = build_solution_from_values(
                final_objective, lambda var: var.X
            )
            if final_objective < best_objective - 1e-7:
                best_objective = final_objective
                best_solution = final_solution
                if logger:
                    try:
                        logger.log_solution(best_objective, best_solution)
                    except Exception:
                        pass

    except (gp.GurobiError, RuntimeError, ValueError, KeyError):
        # The precomputed fallback remains feasible, including under every
        # duration scenario because all assigned surgeries may be cancelled.
        pass

    write_json(args.solution_path, best_solution)


if __name__ == "__main__":
    main()