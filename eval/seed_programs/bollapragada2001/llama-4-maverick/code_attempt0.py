import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_truss_optimization(instance, time_limit, logger):
    num_bars = instance['num_bars']
    num_free_dofs = instance['num_free_dofs']
    num_loading_conditions = instance['num_loading_conditions']
    num_discrete_areas = instance['num_discrete_areas']
    discrete_areas = instance['discrete_areas']
    modulus_of_elasticity = instance['material_properties']['modulus_of_elasticity']
    cost_density = instance['material_properties']['cost_density']

    bar_data = instance['bars']
    dof_data = instance['degrees_of_freedom']
    loading_conditions = instance['loading_conditions']
    linking_groups = instance.get('linking_groups', [])

    # Create a Gurobi model
    model = Model('TrussOptimization')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable console logging

    # Decision variables: area selection for each bar
    bar_area_vars = model.addVars(num_bars, num_discrete_areas, vtype=GRB.BINARY, name='bar_area')
    # Linking groups: ensure bars in the same group have the same area
    for group in linking_groups:
        bars_in_group = group['bars']
        for area_idx in range(num_discrete_areas):
            for bar_id in bars_in_group[1:]:
                model.addConstr(bar_area_vars[bars_in_group[0], area_idx] == bar_area_vars[bar_id, area_idx], name=f'linking_group_{group["group_id"]}_area_{area_idx}')

    # Exactly one area per bar
    for bar_id in range(num_bars):
        model.addConstr(quicksum(bar_area_vars[bar_id, area_idx] for area_idx in range(num_discrete_areas)) == 1, name=f'one_area_per_bar_{bar_id}')

    # State variables for displacements, bar forces, and elongations under each loading condition
    displacements = model.addVars(num_free_dofs, num_loading_conditions, lb=-GRB.INFINITY, name='displacement')
    bar_forces = model.addVars(num_bars, num_loading_conditions, lb=-GRB.INFINITY, name='bar_force')
    bar_elongations = model.addVars(num_bars, num_loading_conditions, lb=-GRB.INFINITY, name='bar_elongation')

    # Objective: minimize total cost
    total_cost = quicksum(cost_density * bar['length'] * quicksum(discrete_areas[area_idx] * bar_area_vars[bar_id, area_idx] for area_idx in range(num_discrete_areas)) for bar_id, bar in enumerate(bar_data))
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints
    for load_id, loading_condition in enumerate(loading_conditions):
        # Equilibrium at each degree of freedom
        for dof_id, dof in enumerate(dof_data):
            external_force = next((load['force'] for load in loading_condition['loads'] if load['dof_id'] == dof['dof_id']), 0)
            model.addConstr(quicksum(bar_forces[bar_id, load_id] * bar['direction_cosines'][dof['direction']] for bar_id, bar in enumerate(bar_data)) == external_force, name=f'equilibrium_dof_{dof_id}_load_{load_id}')

        # Compatibility and Hooke's Law for each bar
        for bar_id, bar in enumerate(bar_data):
            bar_area = quicksum(discrete_areas[area_idx] * bar_area_vars[bar_id, area_idx] for area_idx in range(num_discrete_areas))
            model.addConstr(bar_elongation := quicksum(displacements[dof_id, load_id] * bar['direction_cosines'][dof['direction']] for dof_id, dof in enumerate(dof_data)) == bar_elongations[bar_id, load_id], name=f'compatibility_bar_{bar_id}_load_{load_id}')
            model.addConstr(bar_force := (modulus_of_elasticity / bar['length']) * bar_area * bar_elongations[bar_id, load_id] == bar_forces[bar_id, load_id], name=f'hookes_law_bar_{bar_id}_load_{load_id}')

            # Stress and elongation bounds
            stress = (modulus_of_elasticity / bar['length']) * bar_elongations[bar_id, load_id]
            specific_stress_bounds = next((bounds for bounds in instance.get('bar_specific_stress_bounds', []) if bounds['bar_id'] == bar['bar_id']), None)
            if specific_stress_bounds:
                stress_lower, stress_upper = specific_stress_bounds['lower'], specific_stress_bounds['upper']
            else:
                stress_lower, stress_upper = instance['stress_bounds']['lower'], instance['stress_bounds']['upper']
            model.addConstr(stress >= stress_lower, name=f'stress_lower_bar_{bar_id}_load_{load_id}')
            model.addConstr(stress <= stress_upper, name=f'stress_upper_bar_{bar_id}_load_{load_id}')

            elongation_lower = -GRB.INFINITY
            elongation_upper = GRB.INFINITY
            # Assuming elongation bounds are not bar-specific in the given schema
            model.addConstr(bar_elongations[bar_id, load_id] >= elongation_lower, name=f'elongation_lower_bar_{bar_id}_load_{load_id}')
            model.addConstr(bar_elongations[bar_id, load_id] <= elongation_upper, name=f'elongation_upper_bar_{bar_id}_load_{load_id}')

        # Displacement bounds
        displacement_bounds = instance.get('displacement_bounds', {})
        for dof_id in range(num_free_dofs):
            lower_bound = displacement_bounds.get('lower', -GRB.INFINITY)
            upper_bound = displacement_bounds.get('upper', GRB.INFINITY)
            if lower_bound is not None:
                model.addConstr(displacements[dof_id, load_id] >= lower_bound, name=f'displacement_lower_dof_{dof_id}_load_{load_id}')
            if upper_bound is not None:
                model.addConstr(displacements[dof_id, load_id] <= upper_bound, name=f'displacement_upper_dof_{dof_id}_load_{load_id}')

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        if logger:
            logger.log(objective_value)

        bar_areas = []
        for bar_id in range(num_bars):
            area_idx = np.argmax([bar_area_vars[bar_id, area_idx].X for area_idx in range(num_discrete_areas)])
            area = discrete_areas[area_idx]
            bar_areas.append({'bar_id': bar_data[bar_id]['bar_id'], 'area': area, 'area_index': int(area_idx)})

        displacements_sol = []
        for dof_id in range(num_free_dofs):
            for load_id in range(num_loading_conditions):
                displacements_sol.append({'dof_id': dof_data[dof_id]['dof_id'], 'load': loading_conditions[load_id]['load_id'], 'value': displacements[dof_id, load_id].X})

        bar_forces_sol = []
        for bar_id in range(num_bars):
            for load_id in range(num_loading_conditions):
                bar_forces_sol.append({'bar_id': bar_data[bar_id]['bar_id'], 'load': loading_conditions[load_id]['load_id'], 'force': bar_forces[bar_id, load_id].X})

        solution = {
            'objective_value': objective_value,
            'bar_areas': bar_areas,
            'displacements': displacements_sol,
            'bar_forces': bar_forces_sol
        }
        return solution
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

def main():
    parser = argparse.ArgumentParser(description='Truss Optimization Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_truss_optimization(instance, args.time_limit, logger)
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()