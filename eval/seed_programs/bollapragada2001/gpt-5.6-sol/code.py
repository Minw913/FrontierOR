import argparse
import json
import math
import os

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def finite_number(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def lower_bound(value):
    return float(value) if finite_number(value) else -GRB.INFINITY


def upper_bound(value):
    return float(value) if finite_number(value) else GRB.INFINITY


def make_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_pair(value, default_lower=None, default_upper=None):
    if not isinstance(value, dict):
        return default_lower, default_upper
    return value.get("lower", default_lower), value.get(
        "upper", default_upper
    )


def material_value(bar, material, key, default):
    if key in bar:
        return float(bar[key])

    local_material = bar.get("material_properties")
    if isinstance(local_material, dict) and key in local_material:
        return float(local_material[key])

    return float(material.get(key, default))


def group_members(group):
    if isinstance(group, list):
        values = group
    elif isinstance(group, dict):
        values = []
        for key in (
            "bar_ids",
            "bars",
            "members",
            "member_bars",
            "linked_bars",
        ):
            if key in group:
                values = group[key]
                break
    else:
        return []

    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("bar_id", value.get("id"))
        if value is not None:
            result.append(value)
    return result


def build_incidence(instance, bars, dofs):
    dimension = int(instance.get("dimension", 2))
    num_bars = len(bars)
    num_dofs = len(dofs)
    result = [[0.0 for _ in range(num_bars)] for _ in range(num_dofs)]

    dof_by_id = {dof["dof_id"]: d for d, dof in enumerate(dofs)}

    for b, bar in enumerate(bars):
        raw = bar.get("direction_cosines", [])

        if isinstance(raw, dict):
            for key, coefficient in raw.items():
                try:
                    key = int(key)
                except (TypeError, ValueError):
                    pass
                if key in dof_by_id:
                    result[dof_by_id[key]][b] = float(coefficient)
            continue

        values = [float(value) for value in raw]

        # Some data sets directly provide one compatibility/equilibrium
        # coefficient per free degree of freedom.
        if len(values) == num_dofs and num_dofs > dimension:
            for d, coefficient in enumerate(values):
                result[d][b] = coefficient
            continue

        node_i = bar.get("node_i")
        node_j = bar.get("node_j")

        for d, dof in enumerate(dofs):
            axis = AXIS_INDEX.get(str(dof.get("direction", "")).lower())
            if axis is None or axis >= len(values):
                continue

            if dof.get("node") == node_i:
                result[d][b] = -values[axis]
            elif dof.get("node") == node_j:
                result[d][b] = values[axis]

    return result


def build_load_matrix(loading_conditions, dofs):
    num_loads = len(loading_conditions)
    num_dofs = len(dofs)
    result = [[0.0 for _ in range(num_dofs)] for _ in range(num_loads)]

    by_id = {dof["dof_id"]: d for d, dof in enumerate(dofs)}
    by_node_direction = {
        (dof.get("node"), str(dof.get("direction", "")).lower()): d
        for d, dof in enumerate(dofs)
    }

    for ell, condition in enumerate(loading_conditions):
        entries = condition.get("loads", [])

        if isinstance(entries, dict):
            converted = []
            for key, value in entries.items():
                if isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("dof_id", key)
                else:
                    entry = {"dof_id": key, "force": value}
                converted.append(entry)
            entries = converted

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            index = None
            dof_id = entry.get("dof_id")

            if dof_id in by_id:
                index = by_id[dof_id]
            else:
                try:
                    converted_id = int(dof_id)
                    if converted_id in by_id:
                        index = by_id[converted_id]
                except (TypeError, ValueError):
                    pass

            if index is None:
                key = (
                    entry.get("node"),
                    str(entry.get("direction", "")).lower(),
                )
                index = by_node_direction.get(key)

            if index is not None:
                result[ell][index] += float(entry.get("force", 0.0))

    return result


def create_gurobi_model(time_limit):
    """
    Use an explicit environment. This avoids failures caused by initialization
    of Gurobi's process-global default environment in containerized runners.
    """
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.start()

    model = gp.Model("discrete_truss", env=environment)
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, float(time_limit))

    return environment, model


def solve(instance, time_limit, logger):
    bars = list(instance.get("bars", []))
    dofs = list(instance.get("degrees_of_freedom", []))
    loading_conditions = list(instance.get("loading_conditions", []))
    areas = [float(value) for value in instance.get("discrete_areas", [])]

    B = len(bars)
    D = len(dofs)
    K = len(loading_conditions)
    A = len(areas)

    if B == 0:
        raise ValueError("The instance contains no bars.")
    if A == 0:
        raise ValueError("The instance contains no candidate areas.")
    if any(area <= 0.0 for area in areas):
        raise ValueError("All discrete candidate areas must be positive.")

    material = instance.get("material_properties", {})
    if not isinstance(material, dict):
        material = {}

    lengths = [float(bar["length"]) for bar in bars]
    elasticities = [
        material_value(bar, material, "modulus_of_elasticity", 1.0)
        for bar in bars
    ]
    densities = [
        material_value(bar, material, "cost_density", 1.0)
        for bar in bars
    ]

    if any(value <= 0.0 for value in lengths):
        raise ValueError("All bar lengths must be positive.")
    if any(value <= 0.0 for value in elasticities):
        raise ValueError("All elastic moduli must be positive.")

    bar_index = {bar["bar_id"]: b for b, bar in enumerate(bars)}

    default_stress_lower, default_stress_upper = read_pair(
        instance.get("stress_bounds"),
        None,
        None,
    )
    stress_limits = {
        bar["bar_id"]: [default_stress_lower, default_stress_upper]
        for bar in bars
    }

    for item in instance.get("bar_specific_stress_bounds", []):
        bar_id = item.get("bar_id")
        if bar_id in stress_limits:
            stress_limits[bar_id][0] = item.get(
                "lower", stress_limits[bar_id][0]
            )
            stress_limits[bar_id][1] = item.get(
                "upper", stress_limits[bar_id][1]
            )

    default_elong_lower, default_elong_upper = read_pair(
        instance.get("elongation_bounds"),
        None,
        None,
    )
    elongation_limits = {
        bar["bar_id"]: [default_elong_lower, default_elong_upper]
        for bar in bars
    }

    for item in instance.get("bar_specific_elongation_bounds", []):
        bar_id = item.get("bar_id")
        if bar_id in elongation_limits:
            elongation_limits[bar_id][0] = item.get(
                "lower", elongation_limits[bar_id][0]
            )
            elongation_limits[bar_id][1] = item.get(
                "upper", elongation_limits[bar_id][1]
            )

    for bar in bars:
        bar_id = bar["bar_id"]

        local_stress = bar.get("stress_bounds")
        if isinstance(local_stress, dict):
            stress_limits[bar_id][0] = local_stress.get(
                "lower", stress_limits[bar_id][0]
            )
            stress_limits[bar_id][1] = local_stress.get(
                "upper", stress_limits[bar_id][1]
            )

        local_elongation = bar.get("elongation_bounds")
        if isinstance(local_elongation, dict):
            elongation_limits[bar_id][0] = local_elongation.get(
                "lower", elongation_limits[bar_id][0]
            )
            elongation_limits[bar_id][1] = local_elongation.get(
                "upper", elongation_limits[bar_id][1]
            )

    default_disp_lower, default_disp_upper = read_pair(
        instance.get("displacement_bounds"),
        None,
        None,
    )
    displacement_limits = {
        dof["dof_id"]: [default_disp_lower, default_disp_upper]
        for dof in dofs
    }

    for item in instance.get("dof_specific_displacement_bounds", []):
        dof_id = item.get("dof_id")
        if dof_id in displacement_limits:
            displacement_limits[dof_id][0] = item.get(
                "lower", displacement_limits[dof_id][0]
            )
            displacement_limits[dof_id][1] = item.get(
                "upper", displacement_limits[dof_id][1]
            )

    displacement_lbs = []
    displacement_ubs = []

    for dof in dofs:
        low, high = displacement_limits[dof["dof_id"]]

        local_bounds = dof.get("displacement_bounds")
        if isinstance(local_bounds, dict):
            low = local_bounds.get("lower", low)
            high = local_bounds.get("upper", high)

        low = dof.get("lower_bound", dof.get("lower", low))
        high = dof.get("upper_bound", dof.get("upper", high))

        displacement_lbs.append(lower_bound(low))
        displacement_ubs.append(upper_bound(high))

    incidence = build_incidence(instance, bars, dofs)
    external_loads = build_load_matrix(loading_conditions, dofs)

    environment, model = create_gurobi_model(time_limit)

    x = model.addVars(B, A, vtype=GRB.BINARY, name="area_selected")

    displacement = {}
    for ell in range(K):
        for d in range(D):
            displacement[d, ell] = model.addVar(
                lb=displacement_lbs[d],
                ub=displacement_ubs[d],
                vtype=GRB.CONTINUOUS,
                name="u_{}_{}".format(d, ell),
            )

    # q[b,ell,a] is the force carried by bar b for load ell if area a
    # is selected. The actual bar force is sum_a q[b,ell,a].
    q = {}
    q_limits = {}

    for b, bar in enumerate(bars):
        bar_id = bar["bar_id"]
        stress_low, stress_high = stress_limits[bar_id]
        elong_low, elong_high = elongation_limits[bar_id]

        for a, area in enumerate(areas):
            low_candidates = []
            high_candidates = []

            if finite_number(stress_low):
                low_candidates.append(float(stress_low) * area)
            if finite_number(stress_high):
                high_candidates.append(float(stress_high) * area)

            stiffness = elasticities[b] * area / lengths[b]

            if finite_number(elong_low):
                low_candidates.append(float(elong_low) * stiffness)
            if finite_number(elong_high):
                high_candidates.append(float(elong_high) * stiffness)

            force_low = (
                max(low_candidates)
                if low_candidates
                else -GRB.INFINITY
            )
            force_high = (
                min(high_candidates)
                if high_candidates
                else GRB.INFINITY
            )

            q_limits[b, a] = force_low, force_high

            if force_low > force_high + 1e-10:
                x[b, a].UB = 0.0

            variable_low = (
                min(0.0, force_low)
                if finite_number(force_low)
                else -GRB.INFINITY
            )
            variable_high = (
                max(0.0, force_high)
                if finite_number(force_high)
                else GRB.INFINITY
            )

            for ell in range(K):
                q[b, ell, a] = model.addVar(
                    lb=variable_low,
                    ub=variable_high,
                    vtype=GRB.CONTINUOUS,
                    name="q_{}_{}_{}".format(b, ell, a),
                )

    model.update()

    for b in range(B):
        model.addConstr(
            gp.quicksum(x[b, a] for a in range(A)) == 1,
            name="one_area_{}".format(b),
        )

    for group_index, group in enumerate(
        instance.get("linking_groups", [])
    ):
        member_ids = group_members(group)
        member_indices = [
            bar_index[bar_id]
            for bar_id in member_ids
            if bar_id in bar_index
        ]

        if len(member_indices) < 2:
            continue

        reference = member_indices[0]
        for b in member_indices[1:]:
            for a in range(A):
                model.addConstr(
                    x[b, a] == x[reference, a],
                    name="link_{}_{}_{}".format(group_index, b, a),
                )

    for b in range(B):
        for a in range(A):
            force_low, force_high = q_limits[b, a]

            for ell in range(K):
                variable = q[b, ell, a]

                if finite_number(force_low):
                    model.addConstr(
                        variable >= force_low * x[b, a],
                        name="q_low_{}_{}_{}".format(b, ell, a),
                    )

                if finite_number(force_high):
                    model.addConstr(
                        variable <= force_high * x[b, a],
                        name="q_high_{}_{}_{}".format(b, ell, a),
                    )

                # If either side is unbounded, finite linear bounds alone
                # cannot force q to zero when this area is not selected.
                if (
                    not finite_number(force_low)
                    or not finite_number(force_high)
                ):
                    model.addGenConstrIndicator(
                        x[b, a],
                        False,
                        variable == 0.0,
                        name="q_off_{}_{}_{}".format(b, ell, a),
                    )

    # Equilibrium at every free degree of freedom.
    for ell in range(K):
        for d in range(D):
            expression = gp.LinExpr()

            for b in range(B):
                coefficient = incidence[d][b]
                if coefficient == 0.0:
                    continue
                for a in range(A):
                    expression.add(q[b, ell, a], coefficient)

            model.addConstr(
                expression == external_loads[ell][d],
                name="equilibrium_{}_{}".format(d, ell),
            )

    # Compatibility combined with Hooke's law:
    # B_b^T u = sum_a L_b/(E_b A_a) q_ba.
    for ell in range(K):
        for b in range(B):
            left = gp.LinExpr()
            right = gp.LinExpr()

            for d in range(D):
                coefficient = incidence[d][b]
                if coefficient != 0.0:
                    left.add(displacement[d, ell], coefficient)

            for a, area in enumerate(areas):
                right.add(
                    q[b, ell, a],
                    lengths[b] / (elasticities[b] * area),
                )

            model.addConstr(
                left == right,
                name="compatibility_hooke_{}_{}".format(b, ell),
            )

    costs = {
        (b, a): densities[b] * lengths[b] * areas[a]
        for b in range(B)
        for a in range(A)
    }

    model.setObjective(
        gp.quicksum(
            costs[b, a] * x[b, a]
            for b in range(B)
            for a in range(A)
        ),
        GRB.MINIMIZE,
    )

    # A largest-area MIP start is often feasible or close to feasible.
    largest_area_index = max(range(A), key=lambda index: areas[index])
    for b in range(B):
        for a in range(A):
            x[b, a].Start = 1.0 if a == largest_area_index else 0.0

    x_variables = [x[b, a] for b in range(B) for a in range(A)]
    u_variables = [
        displacement[d, ell]
        for ell in range(K)
        for d in range(D)
    ]
    q_variables = [
        q[b, ell, a]
        for ell in range(K)
        for b in range(B)
        for a in range(A)
    ]

    def form_solution(x_values, u_values, q_values):
        selected_indices = []
        objective = 0.0

        for b in range(B):
            offset = b * A
            selected = max(
                range(A),
                key=lambda a: x_values[offset + a],
            )
            selected_indices.append(selected)
            objective += costs[b, selected]

        displacement_output = []
        for ell, condition in enumerate(loading_conditions):
            load_id = condition.get("load_id", ell)

            for d, dof in enumerate(dofs):
                value = float(u_values[ell * D + d])
                if abs(value) <= 1e-12:
                    value = 0.0

                displacement_output.append(
                    {
                        "dof_id": dof["dof_id"],
                        "load": load_id,
                        "value": value,
                    }
                )

        force_output = []
        for ell, condition in enumerate(loading_conditions):
            load_id = condition.get("load_id", ell)

            for b, bar in enumerate(bars):
                offset = ell * B * A + b * A
                force = sum(
                    float(q_values[offset + a]) for a in range(A)
                )
                if abs(force) <= 1e-10:
                    force = 0.0

                force_output.append(
                    {
                        "bar_id": bar["bar_id"],
                        "load": load_id,
                        "force": force,
                    }
                )

        return {
            "objective_value": float(objective),
            "bar_areas": [
                {
                    "bar_id": bars[b]["bar_id"],
                    "area": float(areas[selected_indices[b]]),
                    "area_index": int(selected_indices[b]),
                }
                for b in range(B)
            ],
            "displacements": displacement_output,
            "bar_forces": force_output,
        }

    best_logged = [math.inf]

    def incumbent_callback(callback_model, where):
        if logger is None or where != GRB.Callback.MIPSOL:
            return

        try:
            objective = callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objective >= best_logged[0] - 1e-8:
                return

            solution = form_solution(
                callback_model.cbGetSolution(x_variables),
                callback_model.cbGetSolution(u_variables),
                callback_model.cbGetSolution(q_variables),
            )

            if solution["objective_value"] < best_logged[0] - 1e-8:
                best_logged[0] = solution["objective_value"]
                logger.log_solution(
                    solution["objective_value"],
                    solution,
                )
        except Exception:
            # Logging must not interrupt the optimizer.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        status_names = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible or unbounded",
            GRB.UNBOUNDED: "unbounded",
            GRB.TIME_LIMIT: "time limit",
        }
        description = status_names.get(model.Status, str(model.Status))
        model.dispose()
        environment.dispose()
        raise RuntimeError(
            "No feasible solution was found; solver status: {}.".format(
                description
            )
        )

    solution = form_solution(
        [variable.X for variable in x_variables],
        [variable.X for variable in u_variables],
        [variable.X for variable in q_variables],
    )

    if logger and solution["objective_value"] < best_logged[0] - 1e-8:
        logger.log_solution(solution["objective_value"], solution)

    model.dispose()
    environment.dispose()
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    solution = solve(instance, args.time_limit, logger)

    make_parent_directory(args.solution_path)
    with open(args.solution_path, "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()