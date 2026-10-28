import argparse
import json
import time
import numpy as np
from heapq import heappush, heappop
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    num_bars = instance_data['num_bars']
    num_free_dofs = instance_data['num_free_dofs']
    num_loading_conditions = instance_data['num_loading_conditions']
    dimension = instance_data['dimension']
    discrete_areas = instance_data['discrete_areas']
    num_discrete_areas = len(discrete_areas)
    material_properties = instance_data['material_properties']
    E = material_properties['modulus_of_elasticity']
    cost_density = material_properties['cost_density']
    stress_bounds = instance_data['stress_bounds']
    bar_specific_stress_bounds = {item['bar_id']: {'lower': item['lower'], 'upper': item['upper']} 
                                  for item in instance_data.get('bar_specific_stress_bounds', [])}
    displacement_bounds = instance_data.get('displacement_bounds', {'lower': None, 'upper': None})
    loading_conditions = instance_data['loading_conditions']
    bars = instance_data['bars']
    nodes = instance_data['nodes']
    degrees_of_freedom = instance_data['degrees_of_freedom']
    linking_groups = instance_data.get('linking_groups', [])
    
    bars_dict = {bar['bar_id']: bar for bar in bars}
    for bar in bars:
        bar_id = bar['bar_id']
        if bar_id in bar_specific_stress_bounds:
            bar['stress_lb'] = bar_specific_stress_bounds[bar_id]['lower']
            bar['stress_ub'] = bar_specific_stress_bounds[bar_id]['upper']
        else:
            bar['stress_lb'] = stress_bounds['lower']
            bar['stress_ub'] = stress_bounds['upper']
        bar['e_lb'] = (bar['stress_lb'] * bar['length']) / E
        bar['e_ub'] = (bar['stress_ub'] * bar['length']) / E
    
    free_dofs_by_id = {dof['dof_id']: dof for dof in degrees_of_freedom}
    node_id_to_node = {node['node_id']: node for node in nodes}
    dof_id_to_index = {}
    for idx, dof in enumerate(degrees_of_freedom):
        dof_id_to_index[dof['dof_id']] = idx
    
    B = np.zeros((num_free_dofs, num_bars))
    for bar_idx, bar in enumerate(bars):
        node_i = bar['node_i']
        node_j = bar['node_j']
        dir_cos = bar['direction_cosines']
        for d in range(dimension):
            dir_str = ['x', 'y', 'z'][d] if dimension == 3 else ['x', 'y'][d]
            cos_val = dir_cos[d]
            for node_id, sign in [(node_i, -1.0), (node_j, 1.0)]:
                if node_id_to_node[node_id]['is_support']:
                    continue
                key = (node_id, dir_str)
                for dof in degrees_of_freedom:
                    if dof['node'] == node_id and dof['direction'] == dir_str:
                        dof_id = dof['dof_id']
                        dof_idx = dof_id_to_index[dof_id]
                        B[dof_idx, bar_idx] += sign * cos_val
    
    groups = []
    bar_to_group_index = {}
    bar_id_to_bar_index = {bar['bar_id']: idx for idx, bar in enumerate(bars)}
    bar_index_to_bar_id = {idx: bar['bar_id'] for idx, bar in enumerate(bars)}
    
    all_bar_ids = set(bar['bar_id'] for bar in bars)
    grouped_bar_ids = set()
    for group_dict in linking_groups:
        bar_ids_in_group = group_dict['bar_ids']
        groups.append(bar_ids_in_group)
        grouped_bar_ids.update(bar_ids_in_group)
        for bar_id in bar_ids_in_group:
            bar_to_group_index[bar_id] = len(groups) - 1
    
    ungrouped_bar_ids = all_bar_ids - grouped_bar_ids
    for bar_id in ungrouped_bar_ids:
        groups.append([bar_id])
        bar_to_group_index[bar_id] = len(groups) - 1
    
    num_groups = len(groups)
    group_length = [0.0] * num_groups
    for g in range(num_groups):
        for bar_id in groups[g]:
            bar = bars_dict[bar_id]
            group_length[g] += bar['length']
    
    def compute_cost(state):
        total_cost = 0.0
        for g in range(num_groups):
            area_val = discrete_areas[state[g]]
            total_cost += cost_density * group_length[g] * area_val
        return total_cost
    
    def check_state_feasibility(state, current_time, time_limit):
        if current_time >= time_limit:
            return False, None, None, None, None
        areas = {}
        for g in range(num_groups):
            area_val = discrete_areas[state[g]]
            for bar_id in groups[g]:
                areas[bar_id] = area_val
        
        displacements_per_load = {}
        bar_forces_per_load = {}
        feasible = True
        for lc_idx, lc in enumerate(loading_conditions):
            if time.time() - start_time >= time_limit:
                feasible = False
                break
            D_bar = np.zeros(num_bars)
            for bar_idx in range(num_bars):
                bar_id = bar_index_to_bar_id[bar_idx]
                area_val = areas[bar_id]
                bar_data = bars_dict[bar_id]
                D_bar[bar_idx] = (E * area_val) / bar_data['length']
            K = np.zeros((num_free_dofs, num_free_dofs))
            for bar_idx in range(num_bars):
                if D_bar[bar_idx] > 0:
                    col = B[:, bar_idx]
                    K += D_bar[bar_idx] * np.outer(col, col)
            P = np.zeros(num_free_dofs)
            for load in lc['loads']:
                dof_id = load['dof_id']
                if dof_id in dof_id_to_index:
                    idx = dof_id_to_index[dof_id]
                    P[idx] = load['force']
            try:
                U = np.linalg.solve(K, P)
            except np.linalg.LinAlgError:
                feasible = False
                break
            e_vec = B.T @ U
            F_vec = D_bar * e_vec
            for bar_idx in range(num_bars):
                bar_id = bar_index_to_bar_id[bar_idx]
                bar_data = bars_dict[bar_id]
                e_val = e_vec[bar_idx]
                if e_val < bar_data['e_lb'] - 1e-6 or e_val > bar_data['e_ub'] + 1e-6:
                    feasible = False
                    break
            if not feasible:
                break
            if displacement_bounds['lower'] is not None or displacement_bounds['upper'] is not None:
                lb = displacement_bounds['lower'] if displacement_bounds['lower'] is not None else -np.inf
                ub = displacement_bounds['upper'] if displacement_bounds['upper'] is not None else np.inf
                for dof_idx in range(num_free_dofs):
                    if U[dof_idx] < lb - 1e-6 or U[dof_idx] > ub + 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break
            displacements_per_load[lc_idx] = U
            bar_forces_per_load[lc_idx] = F_vec
        if feasible:
            return True, areas, displacements_per_load, bar_forces_per_load, F_vec
        else:
            return False, None, None, None, None
    
    fallback_state = tuple([num_discrete_areas - 1] * num_groups)
    fallback_cost = compute_cost(fallback_state)
    fallback_feasible, fallback_areas, fallback_displacements, fallback_bar_forces, _ = check_state_feasibility(
        fallback_state, time.time() - start_time, time_limit)
    best_solution = None
    best_cost = float('inf')
    if fallback_feasible:
        best_solution = {
            'state': fallback_state,
            'areas': fallback_areas,
            'displacements_per_load': fallback_displacements,
            'bar_forces_per_load': fallback_bar_forces
        }
        best_cost = fallback_cost
        if logger:
            logger.log(best_cost)
    
    priority_queue = []
    visited = set()
    initial_state = tuple([0] * num_groups)
    initial_cost = compute_cost(initial_state)
    heappush(priority_queue, (initial_cost, initial_state))
    visited.add(initial_state)
    
    while priority_queue and (time.time() - start_time) < time_limit:
        current_cost, current_state = heappop(priority_queue)
        if current_cost >= best_cost:
            continue
        current_time_used = time.time() - start_time
        if current_time_used >= time_limit:
            break
        (feasible, areas, displacements_per_load, bar_forces_per_load, _) = check_state_feasibility(
            current_state, current_time_used, time_limit)
        current_time_used = time.time() - start_time
        if current_time_used >= time_limit:
            break
        if feasible:
            best_solution = {
                'state': current_state,
                'areas': areas,
                'displacements_per_load': displacements_per_load,
                'bar_forces_per_load': bar_forces_per_load
            }
            best_cost = current_cost
            if logger:
                logger.log(best_cost)
            break
        for g in range(num_groups):
            if current_state[g] < num_discrete_areas - 1:
                new_state_list = list(current_state)
                new_state_list[g] += 1
                new_state = tuple(new_state_list)
                if new_state not in visited:
                    visited.add(new_state)
                    new_cost = compute_cost(new_state)
                    heappush(priority_queue, (new_cost, new_state))
    
    if best_solution is None and fallback_feasible:
        best_solution = {
            'state': fallback_state,
            'areas': fallback_areas,
            'displacements_per_load': fallback_displacements,
            'bar_forces_per_load': fallback_bar_forces
        }
        best_cost = fallback_cost
        if logger:
            logger.log(best_cost)
    elif best_solution is None:
        raise RuntimeError("No feasible solution found, even with maximum areas.")
    
    solution = {
        "objective_value": best_cost,
        "bar_areas": [],
        "displacements": [],
        "bar_forces": []
    }
    
    areas_dict = best_solution['areas']
    for bar_id, area_val in areas_dict.items():
        area_index = discrete_areas.index(area_val)
        solution["bar_areas"].append({
            "bar_id": bar_id,
            "area": area_val,
            "area_index": area_index
        })
    
    displacements_per_load = best_solution.get('displacements_per_load', {})
    for lc_idx, U in displacements_per_load.items():
        for dof_idx in range(num_free_dofs):
            dof_id = degrees_of_freedom[dof_idx]['dof_id']
            solution["displacements"].append({
                "dof_id": dof_id,
                "load": lc_idx,
                "value": U[dof_idx]
            })
    
    bar_forces_per_load = best_solution.get('bar_forces_per_load', {})
    for lc_idx, F_vec in bar_forces_per_load.items():
        for bar_idx in range(num_bars):
            bar_id = bar_index_to_bar_id[bar_idx]
            solution["bar_forces"].append({
                "bar_id": bar_id,
                "load": lc_idx,
                "force": F_vec[bar_idx]
            })
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)