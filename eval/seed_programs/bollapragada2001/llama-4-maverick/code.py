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

    model = Model('TrussOptimization')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi logging to console

    bar_area_vars = model.addVars(num_bars, num_discrete_areas, vtype=GRB.BINARY, name='bar_area')
    for group in linking_groups:
        bars_in_group = group['bar_ids']
        for area_idx in range(num_discrete_areas):
            for bar_id in bars_in_group[1:]:
                model.addConstr(bar_area_vars[bars_in_group[0]-1, area_idx] == bar_area_vars[bar_id-1, area_idx], name=f'linking_group_{group.get("group_id", group.get("id"))}_area_{area_idx}')

    for bar_id in range(num_bars):
        model.addConstr(quicksum(bar_area_vars[bar_id, area_idx] for area_idx in range(num_discrete_areas)) == 1, name=f'one_area_per_bar_{bar_id}')

    displacements = model.addVars(num_free_dofs, num_loading_conditions, lb=-GRB.INFINITY, name='displacement')
    bar_forces = model.addVars(num_bars, num_loading_conditions, lb=-GRB.INFINITY, name='bar_force')
    bar_elongations = model.addVars(num_bars, num_loading_conditions, lb=-GRB.INFINITY, name='bar_elongation')

    total_cost = quicksum(cost_density * bar['length'] * quicksum(discrete_areas[area_idx] * bar_area_vars[bar_id, area_idx] for area_idx in range(num_discrete_areas)) for bar_id, bar in enumerate(bar_data))
    model.setObjective(total_cost, GRB.MINIMIZE)

    for load_id, loading_condition in enumerate(loading_conditions):
        external_loads = {load['dof_id']: load['force'] for load in loading_condition['loads']}
        for dof_id, dof in enumerate(dof_data):
            external_force = external_loads.get(dof['dof_id'], 0)
            bar_force_contributions = quicksum(bar_forces[bar_id, load_id] * bar['direction_cosines'][dof['direction']] for bar_id, bar in enumerate(bar_data))
            model.addConstr(bar_force_contributions == external_force, name=f'equilibrium_dof_{dof_id}_load_{load_id}')

        for bar_id, bar in enumerate(bar_data):
            bar_area = quicksum(discrete_areas[area_idx] * bar_area_vars[bar_id, area_idx] for area_idx in range(num_discrete_areas))
            elongation = quicksum(displacements[dof_id, load_id] * bar['direction_cosines'][dof_data[dof_id]['direction']] for dof_id in range(num_free_dofs))
            model.addConstr(elongation == bar_elongations[bar_id, load_id], name=f'compatibility_bar_{bar_id}_load_{load_id}')
            model.addConstr((modulus_of_elasticity / bar['length']) * bar_area * bar_elongations[bar_id, load_id] == bar_forces[bar_id, load_id], name=f'hookes_law_bar_{bar_id}_load_{load_id}')

            stress = (modulus_of_elasticity / bar['length']) * bar_elongations[bar_id, load_id]
            specific_stress_bounds = next((bounds for bounds in instance.get('bar_specific_stress_bounds', []) if bounds['bar_id'] == bar['bar_id']), None)
            if specific_stress_bounds:
                stress_lower = specific_stress_bounds.get('lower', -GRB.INFINITY)
                stress_upper = specific_stress_bounds.get('upper', GRB.INFINITY)
            else:
                stress_bounds = instance.get('stress_bounds', {})
                stress_lower = stress_bounds.get('lower', -GRB.INFINITY)
                stress_upper = stress_bounds.get('upper', GRB.INFINITY)
            if not np.isinf(stress_lower):
                model.addConstr(stress >= stress_lower, name=f'stress_lower_bar_{bar_id}_load_{load_id}')
            if not np.isinf(stress_upper):
                model.addConstr(stress <= stress_upper, name=f'stress_upper_bar_{bar_id}_load_{load_id}')

            elongation_bounds = instance.get('elongation_bounds', {})
            elongation_lower = elongation_bounds.get('lower', -GRB.INFINITY)
            elongation_upper = elongation_bounds.get('upper', GRB.INFINITY)
            if not np.isinf(elongation_lower):
                model.addConstr(bar_elongations[bar_id, load_id] >= elongation_lower, name=f'elongation_lower_bar_{bar_id}_load_{load_id}')
            if not np.isinf(elongation_upper):
                model.addConstr(bar_elongations[bar_id, load_id] <= elongation_upper, name=f'elongation_upper_bar_{bar_id}_load_{load_id}')

        displacement_bounds = instance.get('displacement_bounds', {})
        displacement_lower = displacement_bounds.get('lower', -GRB.INFINITY)
        displacement_upper = displacement_bounds.get('upper', GRB.INFINITY)
        for dof_id in range(num_free_dofs):
            if not np.isinf(displacement_lower):
                model.addConstr(displacements[dof_id, load_id] >= displacement_lower, name=f'displacement_lower_dof_{dof_id}_load_{load_id}')
            if not np.isinf(displacement_upper):
                model.addConstr(displacements[dof_id, load_id] <= displacement_upper, name=f'displacement_upper_dof_{dof_id}_load_{load_id}')

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
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
            raise Exception("No solution found.")
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
    try:
        solution = solve_truss_optimization(instance, args.time_limit, logger)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()