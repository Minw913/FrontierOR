import argparse
import json
import math
import os

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def finite(value):
    return value is not None and math.isfinite(float(value))


def get_group_bar_ids(group):
    if isinstance(group, list):
        return group
    for key in ("bar_ids", "bars", "members"):
        if key in group:
            values = group[key]
            result = []
            for value in values:
                if isinstance(value, dict):
                    result.append(value.get("bar_id", value.get("id")))
                else:
                    result.append(value)
            return result
    return []


def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_instance(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_and_solve(instance, time_limit, logger):
    bars = instance["bars"]
    dofs = instance["degrees_of_freedom"]
    loading_conditions = instance["loading_conditions"]
    areas = [float(a) for a in instance["discrete_areas"]]

    num_bars = len(bars)
    num_dofs = len(dofs)
    num_loads = len(loading_conditions)
    num_areas = len(areas)

    if num_areas == 0:
        raise ValueError("The instance contains no discrete cross-sectional areas.")
    if any(a <= 0.0 for a in areas):
        raise ValueError("All discrete areas must be strictly positive.")

    bar_id_to_index = {
        bar["bar_id"]: index for index, bar in enumerate(bars)
    }
    dof_id_to_index = {
        dof["dof_id"]: index for index, dof in enumerate(dofs)
    }

    # Material data may be globally specified or overridden on individual bars.
    material = instance.get("material_properties", {})
    global_elasticity = float(material.get("modulus_of_elasticity", 1.0))
    global_density = float(material.get("cost_density", 1.0))

    elasticities = []
    densities = []
    lengths = []

    for bar in bars:
        bar_material = bar.get("material_properties", {})
        elasticity = bar.get(
            "modulus_of_elasticity",
            bar_material.get("modulus_of_elasticity", global_elasticity),
        )
        density = bar.get(
            "cost_density",
            bar_material.get("cost_density", global_density),
        )
        elasticities.append(float(elasticity))
        densities.append(float(density))
        lengths.append(float(bar["length"]))

    default_stress = instance.get("stress_bounds", {})
    default_stress_lower = default_stress.get("lower", -GRB.INFINITY)
    default_stress_upper = default_stress.get("upper", GRB.INFINITY)

    stress_by_bar = {}
    for entry in instance.get("bar_specific_stress_bounds", []):
        stress_by_bar[entry["bar_id"]] = (
            entry.get("lower", default_stress_lower),
            entry.get("upper", default_stress_upper),
        )

    default_elongation = instance.get("elongation_bounds", {})
    default_elongation_lower = default_elongation.get("lower", -GRB.INFINITY)
    default_elongation_upper = default_elongation.get("upper", GRB.INFINITY)

    elongation_by_bar = {}
    for entry in instance.get("bar_specific_elongation_bounds", []):
        elongation_by_bar[entry["bar_id"]] = (
            entry.get("lower", default_elongation_lower),
            entry.get("upper", default_elongation_upper),
        )

    # Build displacement bounds, allowing optional per-DOF overrides.
    displacement_bounds = instance.get("displacement_bounds", {})
    if isinstance(displacement_bounds, dict):
        global_disp_lower = displacement_bounds.get("lower", None)
        global_disp_upper = displacement_bounds.get("upper", None)
    else:
        global_disp_lower = None
        global_disp_upper = None

    dof_specific_bounds = {}
    for entry in instance.get("dof_specific_displacement_bounds", []):
        dof_specific_bounds[entry["dof_id"]] = (
            entry.get("lower", global_disp_lower),
            entry.get("upper", global_disp_upper),
        )

    disp_lbs = []
    disp_ubs = []
    for dof in dofs:
        dof_id = dof["dof_id"]
        lower, upper = dof_specific_bounds.get(
            dof_id, (global_disp_lower, global_disp_upper)
        )
        lower = dof.get("lower_bound", dof.get("lower", lower))
        upper = dof.get("upper_bound", dof.get("upper", upper))
        disp_lbs.append(float(lower) if lower is not None else -GRB.INFINITY)
        disp_ubs.append(float(upper) if upper is not None else GRB.INFINITY)

    # B[d,b] maps positive tensile bar force to the conventional global
    # equilibrium equation B F = external load. Compatibility is B^T u = e.
    incidence = [[0.0 for _ in range(num_bars)] for _ in range(num_dofs)]

    for b, bar in enumerate(bars):
        cosines = [float(v) for v in bar["direction_cosines"]]
        node_i = bar["node_i"]
        node_j = bar["node_j"]

        # The schema normally supplies one cosine per spatial dimension.
        # A generalized one-coefficient-per-DOF representation is also accepted.
        if len(cosines) == num_dofs and len(cosines) != instance.get("dimension"):
            for d, coefficient in enumerate(cosines):
                incidence[d][b] = coefficient
        else:
            for d, dof in enumerate(dofs):
                axis = AXIS_INDEX.get(str(dof["direction"]).lower())
                if axis is None or axis >= len(cosines):
                    continue
                if dof["node"] == node_i:
                    incidence[d][b] = -cosines[axis]
                elif dof["node"] == node_j:
                    incidence[d][b] = cosines[axis]

    loads = [[0.0 for _ in range(num_dofs)] for _ in range(num_loads)]
    for ell, condition in enumerate(loading_conditions):
        for load in condition.get("loads", []):
            dof_id = load.get("dof_id")
            if dof_id in dof_id_to_index:
                d = dof_id_to_index[dof_id]
                loads[ell][d] += float(load["force"])

    model = gp.Model("discrete_truss_design")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, float(time_limit))

    # x[b,a] selects an area. q[b,l,a] is the force carried by area disjunct a;
    # it is forced to zero unless that area is selected.
    x = {}
    for b in range(num_bars):
        for a in range(num_areas):
            x[b, a] = model.addVar(vtype=GRB.BINARY, name=f"x_{b}_{a}")

    u = {}
    for ell in range(num_loads):
        for d in range(num_dofs):
            u[d, ell] = model.addVar(
                lb=disp_lbs[d],
                ub=disp_ubs[d],
                vtype=GRB.CONTINUOUS,
                name=f"u_{d}_{ell}",
            )

    q = {}
    force_bounds = {}

    for b, bar in enumerate(bars):
        bar_id = bar["bar_id"]
        stress_lower, stress_upper = stress_by_bar.get(
            bar_id, (default_stress_lower, default_stress_upper)
        )
        elong_lower, elong_upper = elongation_by_bar.get(
            bar_id, (default_elongation_lower, default_elongation_upper)
        )

        # Individual bar fields take precedence when present.
        own_stress = bar.get("stress_bounds")
        if own_stress:
            stress_lower = own_stress.get("lower", stress_lower)
            stress_upper = own_stress.get("upper", stress_upper)

        own_elongation = bar.get("elongation_bounds")
        if own_elongation:
            elong_lower = own_elongation.get("lower", elong_lower)
            elong_upper = own_elongation.get("upper", elong_upper)

        E = elasticities[b]
        L = lengths[b]

        for a, area in enumerate(areas):
            lower_candidates = []
            upper_candidates = []

            if finite(stress_lower):
                lower_candidates.append(float(stress_lower) * area)
            if finite(stress_upper):
                upper_candidates.append(float(stress_upper) * area)
            if finite(elong_lower):
                lower_candidates.append(float(elong_lower) * E * area / L)
            if finite(elong_upper):
                upper_candidates.append(float(elong_upper) * E * area / L)

            lower_force = max(lower_candidates) if lower_candidates else -GRB.INFINITY
            upper_force = min(upper_candidates) if upper_candidates else GRB.INFINITY
            force_bounds[b, a] = (lower_force, upper_force)

            if lower_force > upper_force + 1e-10:
                x[b, a].UB = 0.0

            for ell in range(num_loads):
                q[b, ell, a] = model.addVar(
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"q_{b}_{ell}_{a}",
                )

    model.update()

    for b in range(num_bars):
        model.addConstr(
            gp.quicksum(x[b, a] for a in range(num_areas)) == 1,
            name=f"one_area_{b}",
        )

    # Symmetry/linking constraints.
    for group_index, group in enumerate(instance.get("linking_groups", [])):
        group_ids = get_group_bar_ids(group)
        group_indices = [
            bar_id_to_index[bar_id]
            for bar_id in group_ids
            if bar_id in bar_id_to_index
        ]
        if len(group_indices) >= 2:
            reference = group_indices[0]
            for b in group_indices[1:]:
                for a in range(num_areas):
                    model.addConstr(
                        x[b, a] == x[reference, a],
                        name=f"link_{group_index}_{b}_{a}",
                    )

    for b in range(num_bars):
        for a in range(num_areas):
            lower_force, upper_force = force_bounds[b, a]
            for ell in range(num_loads):
                var = q[b, ell, a]

                # Bounds of the form lower*x <= q <= upper*x both enforce q=0
                # for inactive finite disjuncts.
                if finite(lower_force) and finite(upper_force):
                    model.addConstr(
                        var >= lower_force * x[b, a],
                        name=f"qlo_{b}_{ell}_{a}",
                    )
                    model.addConstr(
                        var <= upper_force * x[b, a],
                        name=f"qhi_{b}_{ell}_{a}",
                    )
                else:
                    model.addGenConstrIndicator(
                        x[b, a], False, var == 0.0,
                        name=f"qinactive_{b}_{ell}_{a}",
                    )
                    if finite(lower_force):
                        model.addGenConstrIndicator(
                            x[b, a], True, var >= lower_force,
                            name=f"qloind_{b}_{ell}_{a}",
                        )
                    if finite(upper_force):
                        model.addGenConstrIndicator(
                            x[b, a], True, var <= upper_force,
                            name=f"qhiind_{b}_{ell}_{a}",
                        )

    # Equilibrium at every free degree of freedom.
    for ell in range(num_loads):
        for d in range(num_dofs):
            expression = gp.LinExpr()
            for b in range(num_bars):
                coefficient = incidence[d][b]
                if coefficient != 0.0:
                    expression += coefficient * gp.quicksum(
                        q[b, ell, a] for a in range(num_areas)
                    )
            model.addConstr(
                expression == loads[ell][d],
                name=f"equilibrium_{d}_{ell}",
            )

    # Compatibility plus Hooke's law:
    # B^T u = sum_a q_a L/(E A_a).
    for ell in range(num_loads):
        for b in range(num_bars):
            geometric_elongation = gp.LinExpr()
            for d in range(num_dofs):
                coefficient = incidence[d][b]
                if coefficient != 0.0:
                    geometric_elongation += coefficient * u[d, ell]

            elastic_elongation = gp.quicksum(
                (lengths[b] / (elasticities[b] * areas[a])) * q[b, ell, a]
                for a in range(num_areas)
            )
            model.addConstr(
                geometric_elongation == elastic_elongation,
                name=f"compatibility_{b}_{ell}",
            )

    area_cost = {
        (b, a): densities[b] * lengths[b] * areas[a]
        for b in range(num_bars)
        for a in range(num_areas)
    }
    objective = gp.quicksum(
        area_cost[b, a] * x[b, a]
        for b in range(num_bars)
        for a in range(num_areas)
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # A largest-area partial MIP start is often feasible and gives Gurobi an
    # incumbent quickly. Continuous variables are intentionally left undefined.
    largest_area_index = max(range(num_areas), key=lambda a: areas[a])
    for b in range(num_bars):
        for a in range(num_areas):
            x[b, a].Start = 1.0 if a == largest_area_index else 0.0

    x_vars = [x[b, a] for b in range(num_bars) for a in range(num_areas)]
    u_vars = [u[d, ell] for ell in range(num_loads) for d in range(num_dofs)]
    q_vars = [
        q[b, ell, a]
        for ell in range(num_loads)
        for b in range(num_bars)
        for a in range(num_areas)
    ]

    best_logged = [math.inf]

    def make_solution(x_values, u_values, q_values):
        selected = []
        objective_value = 0.0

        for b in range(num_bars):
            start = b * num_areas
            area_index = max(
                range(num_areas),
                key=lambda a: x_values[start + a],
            )
            selected.append(area_index)
            objective_value += area_cost[b, area_index]

        displacement_output = []
        for ell, condition in enumerate(loading_conditions):
            load_id = condition["load_id"]
            for d, dof in enumerate(dofs):
                index = ell * num_dofs + d
                value = float(u_values[index])
                if abs(value) < 1e-14:
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
            load_id = condition["load_id"]
            for b, bar in enumerate(bars):
                base = (ell * num_bars + b) * num_areas
                force = sum(float(q_values[base + a]) for a in range(num_areas))
                if abs(force) < 1e-12:
                    force = 0.0
                force_output.append(
                    {
                        "bar_id": bar["bar_id"],
                        "load": load_id,
                        "force": force,
                    }
                )

        return {
            "objective_value": float(objective_value),
            "bar_areas": [
                {
                    "bar_id": bars[b]["bar_id"],
                    "area": float(areas[selected[b]]),
                    "area_index": int(selected[b]),
                }
                for b in range(num_bars)
            ],
            "displacements": displacement_output,
            "bar_forces": force_output,
        }

    def incumbent_callback(callback_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            incumbent_objective = callback_model.cbGet(
                GRB.Callback.MIPSOL_OBJ
            )
            if incumbent_objective >= best_logged[0] - 1e-8:
                return

            x_values = callback_model.cbGetSolution(x_vars)
            u_values = callback_model.cbGetSolution(u_vars)
            q_values = callback_model.cbGetSolution(q_vars)
            solution = make_solution(x_values, u_values, q_values)
            best_logged[0] = solution["objective_value"]
            logger.log_solution(solution["objective_value"], solution)
        except Exception:
            # Logging must not interrupt optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

    final_x = [var.X for var in x_vars]
    final_u = [var.X for var in u_vars]
    final_q = [var.X for var in q_vars]
    solution = make_solution(final_x, final_u, final_q)

    if logger and solution["objective_value"] < best_logged[0] - 1e-8:
        logger.log_solution(solution["objective_value"], solution)

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve a discrete cross-sectional truss design problem."
    )
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

    instance = load_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit, logger)

    ensure_parent_directory(args.solution_path)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()