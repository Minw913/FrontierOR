import argparse
import json
import numpy as np
import time
import sys

def solve_truss(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Load instance
    with open(instance_path, 'r') as f:
        inst = json.load(f)
    
    num_bars = inst['num_bars']
    num_dofs = inst['num_free_dofs']
    num_loads = inst['num_loading_conditions']
    num_linking_groups = inst['num_linking_groups']
    linking_groups = inst['linking_groups']
    bars = inst['bars']
    dofs_list = inst['degrees_of_freedom']
    material = inst['material_properties']
    discrete_areas = inst['discrete_areas']
    stress_bounds = inst['stress_bounds']
    bar_specific_stress = inst.get('bar_specific_stress_bounds', [])
    disp_bounds = inst['displacement_bounds']
    loading_conditions = inst['loading_conditions']
    
    E = material['modulus_of_elasticity']
    cost_density = material['cost_density']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build DOF id to index mapping
    dof_id_to_idx = {}
    for i, dof in enumerate(dofs_list):
        dof_id_to_idx[dof['dof_id']] = i
    
    # Build load_id to index mapping
    load_id_to_idx = {}
    load_ids_ordered = []
    for lc_idx, lc in enumerate(loading_conditions):
        load_id_to_idx[lc['load_id']] = lc_idx
        load_ids_ordered.append(lc['load_id'])
    
    # Build bar stress bounds
    bar_stress_lb = {}
    bar_stress_ub = {}
    for b in bars:
        bar_stress_lb[b['bar_id']] = stress_bounds['lower']
        bar_stress_ub[b['bar_id']] = stress_bounds['upper']
    for bsb in bar_specific_stress:
        bar_stress_lb[bsb['bar_id']] = bsb['lower']
        bar_stress_ub[bsb['bar_id']] = bsb['upper']
    
    # Map node -> list of dofs
    node_dofs = {}
    for dof in dofs_list:
        nid = dof['node']
        if nid not in node_dofs:
            node_dofs[nid] = []
        node_dofs[nid].append(dof)
    
    dimension = inst['dimension']
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    
    # For each bar, compute direction cosines w.r.t. each DOF
    bar_dof_cosines = np.zeros((num_bars, num_dofs))
    bar_lengths = np.zeros(num_bars)
    bar_ids = []
    bar_id_to_idx = {}
    
    for idx, bar in enumerate(bars):
        bar_ids.append(bar['bar_id'])
        bar_id_to_idx[bar['bar_id']] = idx
        bar_lengths[idx] = bar['length']
        dc = bar['direction_cosines']
        
        ni = bar['node_i']
        nj = bar['node_j']
        
        if ni in node_dofs:
            for dof in node_dofs[ni]:
                d_idx = dir_map[dof['direction']]
                dof_idx = dof_id_to_idx[dof['dof_id']]
                bar_dof_cosines[idx, dof_idx] = dc[d_idx]
        
        if nj in node_dofs:
            for dof in node_dofs[nj]:
                d_idx = dir_map[dof['direction']]
                dof_idx = dof_id_to_idx[dof['dof_id']]
                bar_dof_cosines[idx, dof_idx] = -dc[d_idx]
    
    # Build linking group mapping
    bar_to_group = {}
    group_bars_list = []
    if linking_groups:
        for g_idx, grp in enumerate(linking_groups):
            members = grp.get('bars', grp.get('members', []))
            group_bars_list.append(members)
            for bid in members:
                bar_to_group[bid] = g_idx
    
    # Determine independent area choices
    independent_bars = []
    for bar in bars:
        if bar['bar_id'] not in bar_to_group:
            independent_bars.append(bar['bar_id'])
    
    num_area_choices = len(discrete_areas)
    
    # Build load vectors
    load_vectors = np.zeros((num_loads, num_dofs))
    for lc in loading_conditions:
        lc_idx = load_id_to_idx[lc['load_id']]
        for ld in lc['loads']:
            dof_idx = dof_id_to_idx[ld['dof_id']]
            load_vectors[lc_idx, dof_idx] = ld['force']
    
    disp_lb_val = disp_bounds.get('lower', None)
    disp_ub_val = disp_bounds.get('upper', None)
    
    # Create choice entities
    entities = []
    entity_bars = []
    bar_to_entity = {}
    
    for g_idx, members in enumerate(group_bars_list):
        eid = len(entities)
        entities.append(('group', g_idx))
        bar_indices = [bar_id_to_idx[bid] for bid in members]
        entity_bars.append(bar_indices)
        for bi in bar_indices:
            bar_to_entity[bi] = eid
    
    for bid in independent_bars:
        eid = len(entities)
        entities.append(('bar', bid))
        bi = bar_id_to_idx[bid]
        entity_bars.append([bi])
        bar_to_entity[bi] = eid
    
    num_entities = len(entities)
    
    # Compute elongation bounds per bar from stress bounds
    elong_lb = np.zeros(num_bars)
    elong_ub = np.zeros(num_bars)
    for bi in range(num_bars):
        bid = bar_ids[bi]
        L = bar_lengths[bi]
        s_lb = bar_stress_lb[bid]
        s_ub = bar_stress_ub[bid]
        elong_lb[bi] = s_lb * L / E
        elong_ub[bi] = s_ub * L / E
    
    # Use Gurobi MIP formulation
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 1:
        remaining_time = 1
    
    model = gp.Model("truss")
    model.setParam('TimeLimit', max(1, remaining_time - 2))
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 1)
    model.setParam('MIPGap', 1e-4)
    
    # Binary variables for area selection per entity
    y = {}
    for e in range(num_entities):
        for a in range(num_area_choices):
            y[e, a] = model.addVar(vtype=GRB.BINARY, name=f"y_{e}_{a}")
    
    # Exactly one area per entity
    for e in range(num_entities):
        model.addConstr(gp.quicksum(y[e, a] for a in range(num_area_choices)) == 1)
    
    # Area variable for each bar
    area_var = {}
    for bi in range(num_bars):
        eid = bar_to_entity[bi]
        area_var[bi] = model.addVar(lb=min(discrete_areas), ub=max(discrete_areas), name=f"area_{bi}")
        model.addConstr(area_var[bi] == gp.quicksum(discrete_areas[a] * y[eid, a] for a in range(num_area_choices)))
    
    # Displacement, elongation, force variables
    disp_var = {}
    elong_var = {}
    force_var = {}
    
    dof_ids_ordered = [dofs_list[i]['dof_id'] for i in range(num_dofs)]
    
    for lc in range(num_loads):
        for d in range(num_dofs):
            lb_d = disp_lb_val if disp_lb_val is not None else -GRB.INFINITY
            ub_d = disp_ub_val if disp_ub_val is not None else GRB.INFINITY
            disp_var[lc, d] = model.addVar(lb=lb_d, ub=ub_d, name=f"disp_{lc}_{d}")
        
        for bi in range(num_bars):
            # Force bounds from stress and max area
            max_area = max(discrete_areas)
            bid = bar_ids[bi]
            f_lb = bar_stress_lb[bid] * max_area
            f_ub = bar_stress_ub[bid] * max_area
            force_var[lc, bi] = model.addVar(lb=f_lb, ub=f_ub, name=f"force_{lc}_{bi}")
            elong_var[lc, bi] = model.addVar(lb=elong_lb[bi], ub=elong_ub[bi], name=f"elong_{lc}_{bi}")
    
    # Objective
    obj_expr = gp.quicksum(cost_density * bar_lengths[bi] * area_var[bi] for bi in range(num_bars))
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints for each loading condition
    for lc in range(num_loads):
        # Equilibrium: sum_i (cosine_i_d * force_i) = external_load_d
        for d in range(num_dofs):
            nonzero_bars = [(bi, bar_dof_cosines[bi, d]) for bi in range(num_bars) if abs(bar_dof_cosines[bi, d]) > 1e-15]
            if nonzero_bars:
                expr = gp.quicksum(coeff * force_var[lc, bi] for bi, coeff in nonzero_bars)
            else:
                expr = 0
            model.addConstr(expr == load_vectors[lc, d], name=f"equil_{lc}_{d}")
        
        # Compatibility: sum_d (cosine_i_d * disp_d) = elong_i
        for bi in range(num_bars):
            nonzero_dofs = [(d, bar_dof_cosines[bi, d]) for d in range(num_dofs) if abs(bar_dof_cosines[bi, d]) > 1e-15]
            if nonzero_dofs:
                expr = gp.quicksum(coeff * disp_var[lc, d] for d, coeff in nonzero_dofs)
            else:
                expr = 0
            model.addConstr(expr == elong_var[lc, bi], name=f"compat_{lc}_{bi}")
        
        # Hooke's law: force = (E*A/L) * elong
        # Linearize: force = (E/L) * A * elong = (E/L) * sum_a (A_a * y_a * elong)
        # Use z_a = y_a * elong with McCormick
        for bi in range(num_bars):
            eid = bar_to_entity[bi]
            L = bar_lengths[bi]
            EoverL = E / L
            e_lb = elong_lb[bi]
            e_ub = elong_ub[bi]
            
            z = {}
            for a in range(num_area_choices):
                z[a] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"z_{lc}_{bi}_{a}")
                # McCormick envelopes for z_a = y[eid,a] * elong[lc,bi]
                model.addConstr(z[a] >= e_lb * y[eid, a])
                model.addConstr(z[a] <= e_ub * y[eid, a])
                model.addConstr(z[a] >= elong_var[lc, bi] - e_ub * (1 - y[eid, a]))
                model.addConstr(z[a] <= elong_var[lc, bi] - e_lb * (1 - y[eid, a]))
            
            # Hooke's law
            model.addConstr(
                force_var[lc, bi] == EoverL * gp.quicksum(discrete_areas[a] * z[a] for a in range(num_area_choices)),
                name=f"hooke_{lc}_{bi}"
            )
            
            # elong = sum of z_a
            model.addConstr(elong_var[lc, bi] == gp.quicksum(z[a] for a in range(num_area_choices)),
                          name=f"elong_link_{lc}_{bi}")
    
    model.update()
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        if logger:
            logger.log(obj_val)
        
        bar_areas_out = []
        for bi in range(num_bars):
            eid = bar_to_entity[bi]
            chosen_area = None
            chosen_idx = None
            for a in range(num_area_choices):
                if y[eid, a].X > 0.5:
                    chosen_area = discrete_areas[a]
                    chosen_idx = a
                    break
            bar_areas_out.append({
                "bar_id": bar_ids[bi],
                "area": chosen_area,
                "area_index": chosen_idx
            })
        
        displacements_out = []
        for lc in range(num_loads):
            for d in range(num_dofs):
                displacements_out.append({
                    "dof_id": dof_ids_ordered[d],
                    "load": load_ids_ordered[lc],
                    "value": float(disp_var[lc, d].X)
                })
        
        bar_forces_out = []
        for lc in range(num_loads):
            for bi in range(num_bars):
                bar_forces_out.append({
                    "bar_id": bar_ids[bi],
                    "load": load_ids_ordered[lc],
                    "force": float(force_var[lc, bi].X)
                })
        
        solution = {
            "objective_value": obj_val,
            "bar_areas": bar_areas_out,
            "displacements": displacements_out,
            "bar_forces": bar_forces_out
        }
    else:
        # Fallback: use largest area
        max_area = max(discrete_areas)
        max_area_idx = discrete_areas.index(max_area)
        areas = np.array([max_area] * num_bars)
        obj_val = sum(cost_density * bar_lengths[bi] * areas[bi] for bi in range(num_bars))
        
        if logger:
            logger.log(obj_val)
        
        bar_areas_out = []
        for bi in range(num_bars):
            bar_areas_out.append({"bar_id": bar_ids[bi], "area": max_area, "area_index": max_area_idx})
        
        displacements_out = []
        bar_forces_out = []
        
        for lc in range(num_loads):
            K = np.zeros((num_dofs, num_dofs))
            for bi in range(num_bars):
                L = bar_lengths[bi]
                A = areas[bi]
                k = E * A / L
                bc = bar_dof_cosines[bi]
                K += k * np.outer(bc, bc)
            
            try:
                u = np.linalg.solve(K, load_vectors[lc])
            except np.linalg.LinAlgError:
                u = np.zeros(num_dofs)
            
            for d in range(num_dofs):
                displacements_out.append({"dof_id": dof_ids_ordered[d], "load": load_ids_ordered[lc], "value": float(u[d])})
            
            for bi in range(num_bars):
                bc = bar_dof_cosines[bi]
                e_val = bc @ u
                f_val = E * areas[bi] / bar_lengths[bi] * e_val
                bar_forces_out.append({"bar_id": bar_ids[bi], "load": load_ids_ordered[lc], "force": float(f_val)})
        
        solution = {
            "objective_value": obj_val,
            "bar_areas": bar_areas_out,
            "displacements": displacements_out,
            "bar_forces": bar_forces_out
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_truss(args.instance_path, args.solution_path, args.time_limit, args.log_path)