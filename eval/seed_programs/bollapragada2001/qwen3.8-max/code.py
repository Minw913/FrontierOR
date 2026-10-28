import json
import argparse
import time
import numpy as np
import sys


def extract_instance_data(instance):
    """Extract all necessary data from the instance."""
    num_bars = instance['num_bars']
    num_free_dofs = instance['num_free_dofs']
    num_loading = instance['num_loading_conditions']
    num_areas = instance['num_discrete_areas']
    discrete_areas = instance['discrete_areas']
    E = instance['material_properties']['modulus_of_elasticity']
    cost_density = instance['material_properties']['cost_density']
    dimension = instance.get('dimension', 2)

    dof_list = instance['degrees_of_freedom']
    dof_to_idx = {d['dof_id']: i for i, d in enumerate(dof_list)}

    bars = instance['bars']
    bar_lengths = np.array([b['length'] for b in bars])
    bar_dir_cos = np.array([b['direction_cosines'] for b in bars])

    dir_map = {'x': 0, 'y': 1, 'z': 2}
    gamma = np.zeros((num_bars, num_free_dofs))

    for b_idx, bar in enumerate(bars):
        ni = bar['node_i']
        nj = bar['node_j']
        for d_idx, dof in enumerate(dof_list):
            node = dof['node']
            direction = dof['direction']
            d_dir = dir_map[direction]
            if node == ni:
                gamma[b_idx, d_idx] = -bar_dir_cos[b_idx, d_dir]
            elif node == nj:
                gamma[b_idx, d_idx] = bar_dir_cos[b_idx, d_dir]

    P = np.zeros((num_free_dofs, num_loading))
    for l_idx, lc in enumerate(instance['loading_conditions']):
        for load in lc['loads']:
            dof_idx = dof_to_idx[load['dof_id']]
            P[dof_idx, l_idx] = load['force']

    stress_lower = instance['stress_bounds']['lower']
    stress_upper = instance['stress_bounds']['upper']
    bar_stress_lower = np.full(num_bars, stress_lower, dtype=float)
    bar_stress_upper = np.full(num_bars, stress_upper, dtype=float)

    if 'bar_specific_stress_bounds' in instance and instance['bar_specific_stress_bounds'] is not None:
        for bsb in instance['bar_specific_stress_bounds']:
            bid = bsb['bar_id']
            if bid < num_bars:
                bar_stress_lower[bid] = bsb['lower']
                bar_stress_upper[bid] = bsb['upper']

    disp_bounds = instance.get('displacement_bounds', {})
    disp_lower = disp_bounds.get('lower', None) if disp_bounds else None
    disp_upper = disp_bounds.get('upper', None) if disp_bounds else None

    elong_lower = bar_lengths / E * bar_stress_lower
    elong_upper = bar_lengths / E * bar_stress_upper

    linking_groups = instance.get('linking_groups', [])
    if linking_groups is None:
        linking_groups = []
    bar_to_group = {}
    for g_idx, group in enumerate(linking_groups):
        bar_ids = []
        if isinstance(group, dict):
            if 'bars' in group:
                bar_ids = group['bars']
            elif 'bar_ids' in group:
                bar_ids = group['bar_ids']
            elif 'members' in group:
                bar_ids = group['members']
            elif 'bar_list' in group:
                bar_ids = group['bar_list']
        elif isinstance(group, list):
            bar_ids = group
        for bar_id in bar_ids:
            bar_to_group[bar_id] = g_idx

    return {
        'num_bars': num_bars,
        'num_free_dofs': num_free_dofs,
        'num_loading': num_loading,
        'num_areas': num_areas,
        'discrete_areas': discrete_areas,
        'E': E,
        'cost_density': cost_density,
        'gamma': gamma,
        'P': P,
        'bar_lengths': bar_lengths,
        'bar_stress_lower': bar_stress_lower,
        'bar_stress_upper': bar_stress_upper,
        'disp_lower': disp_lower,
        'disp_upper': disp_upper,
        'elong_lower': elong_lower,
        'elong_upper': elong_upper,
        'linking_groups': linking_groups,
        'bar_to_group': bar_to_group,
        'dof_list': dof_list,
        'dof_to_idx': dof_to_idx,
        'bars': bars,
    }


def build_decision_mapping(data):
    """Map bars to decision variable indices considering linking groups."""
    num_bars = data['num_bars']
    bar_to_group = data['bar_to_group']

    group_to_decision = {}
    bar_to_decision = {}
    num_decisions = 0

    for b in range(num_bars):
        if b in bar_to_group:
            g = bar_to_group[b]
            if g not in group_to_decision:
                group_to_decision[g] = num_decisions
                num_decisions += 1
            bar_to_decision[b] = group_to_decision[g]
        else:
            bar_to_decision[b] = num_decisions
            num_decisions += 1

    return bar_to_decision, num_decisions


def solve_truss(areas, data):
    """Solve truss for given areas. Returns (U, forces, elongations) or None if singular."""
    num_bars = data['num_bars']
    num_free_dofs = data['num_free_dofs']
    num_loading = data['num_loading']
    gamma = data['gamma']
    P = data['P']
    E = data['E']
    bar_lengths = data['bar_lengths']

    K = np.zeros((num_free_dofs, num_free_dofs))
    for b in range(num_bars):
        if areas[b] > 1e-12:
            stiffness = E * areas[b] / bar_lengths[b]
            gamma_b = gamma[b, :]
            K += stiffness * np.outer(gamma_b, gamma_b)

    try:
        U = np.linalg.solve(K, P)
    except np.linalg.LinAlgError:
        try:
            U = np.linalg.lstsq(K, P, rcond=None)[0]
        except Exception:
            return None, None, None

    elongations = gamma @ U
    forces = np.zeros((num_bars, num_loading))
    for b in range(num_bars):
        forces[b, :] = E * areas[b] / bar_lengths[b] * elongations[b, :]

    return U, forces, elongations


def check_feasibility(areas, U, forces, elongations, data, tol=1e-6):
    """Check all constraints. Returns (feasible, violations)."""
    num_bars = data['num_bars']
    num_loading = data['num_loading']
    num_free_dofs = data['num_free_dofs']
    E = data['E']
    bar_lengths = data['bar_lengths']
    bar_stress_lower = data['bar_stress_lower']
    bar_stress_upper = data['bar_stress_upper']
    disp_lower = data['disp_lower']
    disp_upper = data['disp_upper']

    violations = []

    for b in range(num_bars):
        for l in range(num_loading):
            stress = E / bar_lengths[b] * elongations[b, l]
            if stress < bar_stress_lower[b] - tol:
                violations.append(('stress_low', b, l, bar_stress_lower[b] - stress))
            if stress > bar_stress_upper[b] + tol:
                violations.append(('stress_high', b, l, stress - bar_stress_upper[b]))

    if disp_lower is not None:
        for d in range(num_free_dofs):
            for l in range(num_loading):
                if U[d, l] < disp_lower - tol:
                    violations.append(('disp_low', d, l, disp_lower - U[d, l]))

    if disp_upper is not None:
        for d in range(num_free_dofs):
            for l in range(num_loading):
                if U[d, l] > disp_upper + tol:
                    violations.append(('disp_high', d, l, U[d, l] - disp_upper))

    return len(violations) == 0, violations


def compute_objective(areas, data):
    """Compute total cost."""
    return sum(data['cost_density'] * data['bar_lengths'][b] * areas[b]
               for b in range(data['num_bars']))


def build_solution_dict(area_indices, bar_to_decision, instance, data, U, forces):
    """Build the solution dictionary."""
    num_bars = data['num_bars']
    num_loading = data['num_loading']
    num_free_dofs = data['num_free_dofs']
    discrete_areas = data['discrete_areas']

    areas = np.zeros(num_bars)
    for b in range(num_bars):
        areas[b] = discrete_areas[area_indices[bar_to_decision[b]]]

    obj_val = compute_objective(areas, data)

    bar_areas = []
    for b in range(num_bars):
        k = int(area_indices[bar_to_decision[b]])
        bar_areas.append({
            'bar_id': instance['bars'][b]['bar_id'],
            'area': discrete_areas[k],
            'area_index': k
        })

    displacements = []
    for dof in range(num_free_dofs):
        for l in range(num_loading):
            displacements.append({
                'dof_id': instance['degrees_of_freedom'][dof]['dof_id'],
                'load': instance['loading_conditions'][l]['load_id'],
                'value': float(U[dof, l])
            })

    bar_forces = []
    for b in range(num_bars):
        for l in range(num_loading):
            bar_forces.append({
                'bar_id': instance['bars'][b]['bar_id'],
                'load': instance['loading_conditions'][l]['load_id'],
                'force': float(forces[b, l])
            })

    return {
        'objective_value': obj_val,
        'bar_areas': bar_areas,
        'displacements': displacements,
        'bar_forces': bar_forces
    }


def solve_mip(instance, data, args, logger, start_time):
    """Solve using MIP with Gurobi."""
    from gurobipy import Model, GRB, quicksum

    num_bars = data['num_bars']
    num_free_dofs = data['num_free_dofs']
    num_loading = data['num_loading']
    num_areas = data['num_areas']
    discrete_areas = data['discrete_areas']
    E = data['E']
    cost_density = data['cost_density']
    gamma = data['gamma']
    P = data['P']
    bar_lengths = data['bar_lengths']
    elong_lower = data['elong_lower']
    elong_upper = data['elong_upper']
    bar_stress_lower = data['bar_stress_lower']
    bar_stress_upper = data['bar_stress_upper']
    disp_lower = data['disp_lower']
    disp_upper = data['disp_upper']

    bar_to_decision, num_decisions = build_decision_mapping(data)

    # Check if MIP is tractable
    estimated_vars = num_decisions * num_areas + num_bars * num_areas * num_loading + \
                     num_bars * num_loading * 2 + num_free_dofs * num_loading
    if estimated_vars > 300000:
        return None

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 3
    if remaining <= 1:
        return None

    model = Model("truss")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(remaining, 1)
    model.Params.OutputFlag = 0

    # Binary variables
    x = {}
    for d in range(num_decisions):
        for k in range(num_areas):
            x[d, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{d}_{k}")

    # Displacement variables
    u = {}
    d_lb = disp_lower if disp_lower is not None else -1e8
    d_ub = disp_upper if disp_upper is not None else 1e8
    for dof in range(num_free_dofs):
        for l in range(num_loading):
            u[dof, l] = model.addVar(lb=d_lb, ub=d_ub, name=f"u_{dof}_{l}")

    # Force variables
    F = {}
    for b in range(num_bars):
        for l in range(num_loading):
            max_area = discrete_areas[-1]
            f_bound = max(abs(bar_stress_lower[b]), abs(bar_stress_upper[b])) * max_area * 1.01
            F[b, l] = model.addVar(lb=-f_bound, ub=f_bound, name=f"F_{b}_{l}")

    # Elongation variables
    e = {}
    for b in range(num_bars):
        for l in range(num_loading):
            e[b, l] = model.addVar(lb=elong_lower[b], ub=elong_upper[b], name=f"e_{b}_{l}")

    # Auxiliary variables for McCormick linearization
    z = {}
    for b in range(num_bars):
        for k in range(num_areas):
            for l in range(num_loading):
                z[b, k, l] = model.addVar(
                    lb=min(0.0, elong_lower[b]),
                    ub=max(0.0, elong_upper[b]),
                    name=f"z_{b}_{k}_{l}"
                )

    model.update()

    # Objective
    obj = quicksum(cost_density * bar_lengths[b] * discrete_areas[k] * x[bar_to_decision[b], k]
                   for b in range(num_bars) for k in range(num_areas))
    model.setObjective(obj, GRB.MINIMIZE)

    # One area per decision
    for d in range(num_decisions):
        model.addConstr(quicksum(x[d, k] for k in range(num_areas)) == 1)

    # Equilibrium
    for dof in range(num_free_dofs):
        for l in range(num_loading):
            terms = [(gamma[b, dof], F[b, l]) for b in range(num_bars) if abs(gamma[b, dof]) > 1e-12]
            if terms:
                model.addConstr(quicksum(coeff * var for coeff, var in terms) == P[dof, l])

    # Compatibility
    for b in range(num_bars):
        for l in range(num_loading):
            terms = [(gamma[b, d], u[d, l]) for d in range(num_free_dofs) if abs(gamma[b, d]) > 1e-12]
            if terms:
                model.addConstr(e[b, l] == quicksum(coeff * var for coeff, var in terms))
            else:
                model.addConstr(e[b, l] == 0)

    # Hooke's law with McCormick
    for b in range(num_bars):
        for l in range(num_loading):
            model.addConstr(
                F[b, l] == (E / bar_lengths[b]) * quicksum(discrete_areas[k] * z[b, k, l] for k in range(num_areas))
            )

    # McCormick envelopes
    for b in range(num_bars):
        d_idx = bar_to_decision[b]
        e_lb = elong_lower[b]
        e_ub = elong_upper[b]
        for k in range(num_areas):
            for l in range(num_loading):
                model.addConstr(z[b, k, l] <= e_ub * x[d_idx, k])
                model.addConstr(z[b, k, l] >= e_lb * x[d_idx, k])
                model.addConstr(z[b, k, l] <= e[b, l] - e_lb * (1 - x[d_idx, k]))
                model.addConstr(z[b, k, l] >= e[b, l] - e_ub * (1 - x[d_idx, k]))

    model.update()
    model.optimize()

    if model.SolCount > 0:
        # Extract solution
        bar_areas = []
        for b in range(num_bars):
            d_idx = bar_to_decision[b]
            chosen_k = 0
            for k in range(num_areas):
                if x[d_idx, k].X > 0.5:
                    chosen_k = k
                    break
            bar_areas.append({
                'bar_id': instance['bars'][b]['bar_id'],
                'area': discrete_areas[chosen_k],
                'area_index': chosen_k
            })

        displacements = []
        for dof in range(num_free_dofs):
            for l in range(num_loading):
                displacements.append({
                    'dof_id': instance['degrees_of_freedom'][dof]['dof_id'],
                    'load': instance['loading_conditions'][l]['load_id'],
                    'value': u[dof, l].X
                })

        bar_forces = []
        for b in range(num_bars):
            for l in range(num_loading):
                bar_forces.append({
                    'bar_id': instance['bars'][b]['bar_id'],
                    'load': instance['loading_conditions'][l]['load_id'],
                    'force': F[b, l].X
                })

        obj_val = sum(data['cost_density'] * data['bar_lengths'][b] * bar_areas[b]['area']
                      for b in range(num_bars))

        solution = {
            'objective_value': obj_val,
            'bar_areas': bar_areas,
            'displacements': displacements,
            'bar_forces': bar_forces
        }

        if logger:
            logger.log_solution(obj_val, solution)

        return solution

    return None


def solve_heuristic_advanced(instance, data, args, logger, start_time):
    """Advanced heuristic using greedy area reduction."""
    num_bars = data['num_bars']
    num_areas = data['num_areas']
    num_loading = data['num_loading']
    discrete_areas = data['discrete_areas']
    E = data['E']
    bar_lengths = data['bar_lengths']
    gamma = data['gamma']
    P = data['P']
    bar_stress_lower = data['bar_stress_lower']
    bar_stress_upper = data['bar_stress_upper']

    bar_to_decision, num_decisions = build_decision_mapping(data)

    decision_to_bars = {}
    for b in range(num_bars):
        d = bar_to_decision[b]
        if d not in decision_to_bars:
            decision_to_bars[d] = []
        decision_to_bars[d].append(b)

    # Start with all bars at maximum area
    area_indices = np.full(num_decisions, num_areas - 1, dtype=int)

    def get_areas():
        areas = np.zeros(num_bars)
        for b in range(num_bars):
            areas[b] = discrete_areas[area_indices[bar_to_decision[b]]]
        return areas

    # Verify max area is feasible
    areas = get_areas()
    U, forces, elongations = solve_truss(areas, data)
    if U is None:
        return None

    feasible, violations = check_feasibility(areas, U, forces, elongations, data)
    if not feasible:
        return None

    # Greedy reduction: try to reduce each decision variable's area
    best_obj = compute_objective(areas, data)

    for iteration in range(10):
        if (time.time() - start_time) > args.time_limit - 1:
            break

        improved = False
        # Sort decisions by cost contribution (descending) for better greedy
        cost_per_decision = []
        for d in range(num_decisions):
            bars_in_group = decision_to_bars.get(d, [])
            cost = sum(data['cost_density'] * bar_lengths[b] for b in bars_in_group)
            cost_per_decision.append((cost, d))
        cost_per_decision.sort(reverse=True)

        for cost_val, d in cost_per_decision:
            if (time.time() - start_time) > args.time_limit - 1:
                break
            if area_indices[d] > 0:
                area_indices[d] -= 1
                areas = get_areas()
                U, forces, elongations = solve_truss(areas, data)
                if U is not None:
                    feas, _ = check_feasibility(areas, U, forces, elongations, data)
                    if feas:
                        improved = True
                        continue
                area_indices[d] += 1

        if not improved:
            break

    # Build final solution
    areas = get_areas()
    U, forces, elongations = solve_truss(areas, data)

    if U is None:
        return None

    feas, _ = check_feasibility(areas, U, forces, elongations, data)
    if not feas:
        # Revert to max
        area_indices = np.full(num_decisions, num_areas - 1, dtype=int)
        areas = get_areas()
        U, forces, elongations = solve_truss(areas, data)
        if U is None:
            return None

    solution = build_solution_dict(area_indices, bar_to_decision, instance, data, U, forces)

    if logger:
        logger.log_solution(solution['objective_value'], solution)

    return solution


def solve_heuristic_binary_search(instance, data, args, logger, start_time):
    """Heuristic using binary-search-like approach per decision variable."""
    num_bars = data['num_bars']
    num_areas = data['num_areas']
    discrete_areas = data['discrete_areas']

    bar_to_decision, num_decisions = build_decision_mapping(data)

    decision_to_bars = {}
    for b in range(num_bars):
        d = bar_to_decision[b]
        if d not in decision_to_bars:
            decision_to_bars[d] = []
        decision_to_bars[d].append(b)

    # Start from max
    area_indices = np.full(num_decisions, num_areas - 1, dtype=int)

    def get_areas():
        areas = np.zeros(num_bars)
        for b in range(num_bars):
            areas[b] = discrete_areas[area_indices[bar_to_decision[b]]]
        return areas

    def is_feasible_with_indices(indices):
        areas = np.zeros(num_bars)
        for b in range(num_bars):
            areas[b] = discrete_areas[indices[bar_to_decision[b]]]
        U, forces, elongations = solve_truss(areas, data)
        if U is None:
            return False, None, None
        feas, _ = check_feasibility(areas, U, forces, elongations, data)
        return feas, U, forces

    # Multiple passes of greedy reduction
    for pass_num in range(5):
        if (time.time() - start_time) > args.time_limit - 1:
            break

        # Randomize order for diversity
        order = np.random.permutation(num_decisions)

        for d in order:
            if (time.time() - start_time) > args.time_limit - 1:
                break
            if area_indices[d] > 0:
                # Try reducing by 1
                area_indices[d] -= 1
                feas, U, forces = is_feasible_with_indices(area_indices)
                if feas:
                    continue
                area_indices[d] += 1

    # Final solution
    areas = get_areas()
    U, forces, elongations = solve_truss(areas, data)
    if U is None:
        return None

    feas, _ = check_feasibility(areas, U, forces, elongations, data)
    if not feas:
        area_indices = np.full(num_decisions, num_areas - 1, dtype=int)
        areas = get_areas()
        U, forces, elongations = solve_truss(areas, data)
        if U is None:
            return None

    solution = build_solution_dict(area_indices, bar_to_decision, instance, data, U, forces)

    if logger:
        logger.log_solution(solution['objective_value'], solution)

    return solution


def main():
    parser = argparse.ArgumentParser(description='Truss structure optimizer')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    data = extract_instance_data(instance)

    solution = None

    # Try MIP first for small instances
    try:
        solution = solve_mip(instance, data, args, logger, start_time)
    except Exception as ex:
        print(f"MIP failed: {ex}", file=sys.stderr)

    # If MIP didn't produce solution, try heuristic
    if solution is None:
        try:
            solution = solve_heuristic_advanced(instance, data, args, logger, start_time)
        except Exception as ex:
            print(f"Advanced heuristic failed: {ex}", file=sys.stderr)

    if solution is None:
        try:
            solution = solve_heuristic_binary_search(instance, data, args, logger, start_time)
        except Exception as ex:
            print(f"Binary search heuristic failed: {ex}", file=sys.stderr)

    if solution is None:
        # Absolute fallback: all bars at max area
        num_bars = data['num_bars']
        num_areas = data['num_areas']
        discrete_areas = data['discrete_areas']
        bar_to_decision, num_decisions = build_decision_mapping(data)

        area_indices = np.full(num_decisions, num_areas - 1, dtype=int)
        areas = np.zeros(num_bars)
        for b in range(num_bars):
            areas[b] = discrete_areas[num_areas - 1]

        U, forces, elongations = solve_truss(areas, data)

        bar_areas = []
        for b in range(num_bars):
            bar_areas.append({
                'bar_id': instance['bars'][b]['bar_id'],
                'area': discrete_areas[num_areas - 1],
                'area_index': num_areas - 1
            })

        displacements = []
        bar_forces = []
        if U is not None:
            for dof in range(data['num_free_dofs']):
                for l in range(data['num_loading']):
                    displacements.append({
                        'dof_id': instance['degrees_of_freedom'][dof]['dof_id'],
                        'load': instance['loading_conditions'][l]['load_id'],
                        'value': float(U[dof, l])
                    })
            for b in range(num_bars):
                for l in range(data['num_loading']):
                    bar_forces.append({
                        'bar_id': instance['bars'][b]['bar_id'],
                        'load': instance['loading_conditions'][l]['load_id'],
                        'force': float(forces[b, l])
                    })

        obj_val = compute_objective(areas, data)
        solution = {
            'objective_value': obj_val,
            'bar_areas': bar_areas,
            'displacements': displacements,
            'bar_forces': bar_forces
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()