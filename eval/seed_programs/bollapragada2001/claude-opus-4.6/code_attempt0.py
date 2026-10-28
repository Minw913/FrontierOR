import argparse
import json
import numpy as np
import time
from itertools import product
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
    dofs = inst['degrees_of_freedom']
    material = inst['material_properties']
    discrete_areas = inst['discrete_areas']
    stress_bounds = inst['stress_bounds']
    bar_specific_stress = inst.get('bar_specific_stress_bounds', [])
    disp_bounds = inst['displacement_bounds']
    loading_conditions = inst['loading_conditions']
    
    E = material['modulus_of_elasticity']
    cost_density = material['cost_density']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build bar stress bounds
    bar_stress_lb = {}
    bar_stress_ub = {}
    for b in bars:
        bar_stress_lb[b['bar_id']] = stress_bounds['lower']
        bar_stress_ub[b['bar_id']] = stress_bounds['upper']
    for bsb in bar_specific_stress:
        bar_stress_lb[bsb['bar_id']] = bsb['lower']
        bar_stress_ub[bsb['bar_id']] = bsb['upper']
    
    # Build direction cosine matrix: B[bar_id][dof_id] = cosine
    # For each bar, direction_cosines relate to coordinate axes
    # We need to build the actual direction cosines w.r.t. DOFs
    
    # Map node -> list of (dof_id, direction)
    node_dofs = {}
    for dof in dofs:
        nid = dof['node']
        if nid not in node_dofs:
            node_dofs[nid] = []
        node_dofs[nid].append(dof)
    
    dimension = inst['dimension']
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    
    # For each bar, compute direction cosines w.r.t. each DOF
    # bar_dof_cosines[bar_idx][dof_idx] = cosine value
    bar_dof_cosines = np.zeros((num_bars, num_dofs))
    bar_lengths = np.zeros(num_bars)
    bar_ids = []
    bar_id_to_idx = {}
    
    for idx, bar in enumerate(bars):
        bar_ids.append(bar['bar_id'])
        bar_id_to_idx[bar['bar_id']] = idx
        bar_lengths[idx] = bar['length']
        dc = bar['direction_cosines']  # direction cosines w.r.t. coordinate axes
        
        # For node_i
        ni = bar['node_i']
        nj = bar['node_j']
        
        if ni in node_dofs:
            for dof in node_dofs[ni]:
                d_idx = dir_map[dof['direction']]
                # The direction cosine for DOF at node_i is the cosine of the bar's angle
                # with that coordinate direction
                bar_dof_cosines[idx, dof['dof_id']] = dc[d_idx]
        
        if nj in node_dofs:
            for dof in node_dofs[nj]:
                d_idx = dir_map[dof['direction']]
                # For node_j, the direction cosine is negative (opposite end)
                bar_dof_cosines[idx, dof['dof_id']] = -dc[d_idx]
    
    # Build linking group mapping
    # group_bars[group_idx] = list of bar_ids in that group
    # bar_to_group[bar_id] = group_idx (or None if not in any group)
    bar_to_group = {}
    group_bars = []
    if linking_groups:
        for g_idx, grp in enumerate(linking_groups):
            members = grp.get('bars', grp.get('members', []))
            group_bars.append(members)
            for bid in members:
                bar_to_group[bid] = g_idx
    
    # Determine independent area choices
    # Each bar either belongs to a linking group or is independent
    # We need to enumerate: for each "choice entity", pick an area index
    independent_bars = []  # bar_ids not in any linking group
    for bar in bars:
        if bar['bar_id'] not in bar_to_group:
            independent_bars.append(bar['bar_id'])
    
    num_area_choices = len(discrete_areas)
    
    # Build load vectors
    # load_vector[load_idx][dof_id] = force
    load_vectors = np.zeros((num_loads, num_dofs))
    for lc in loading_conditions:
        lid = lc['load_id']
        for ld in lc['loads']:
            load_vectors[lid, ld['dof_id']] = ld['force']
    
    # Compute elongation bounds from stress bounds
    # stress = E / L * elongation => elongation = stress * L / E
    # So elongation_lb = stress_lb * L / E, elongation_ub = stress_ub * L / E
    bar_elong_lb = np.zeros(num_bars)
    bar_elong_ub = np.zeros(num_bars)
    for idx, bar in enumerate(bars):
        bid = bar['bar_id']
        L = bar['length']
        s_lb = bar_stress_lb[bid]
        s_ub = bar_stress_ub[bid]
        bar_elong_lb[idx] = s_lb * L / E
        bar_elong_ub[idx] = s_ub * L / E
    
    disp_lb = disp_bounds.get('lower', None)
    disp_ub = disp_bounds.get('upper', None)
    
    # Use Gurobi MIP formulation
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        remaining_time = 1
    
    model = gp.Model("truss")
    model.setParam('TimeLimit', remaining_time - 1)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 1)
    
    # Decision variables
    # Binary: y[bar_idx, area_idx] = 1 if bar gets that area
    # But with linking groups, bars in same group share the same selection
    
    # For linking groups: one set of binary vars per group
    # For independent bars: one set of binary vars per bar
    
    # Create "choice entities"
    # entity_id -> (type, id): type='group' or 'bar', id=group_idx or bar_id
    entities = []
    entity_bars = []  # entity_bars[e] = list of bar_indices belonging to entity e
    bar_to_entity = {}
    
    for g_idx, members in enumerate(group_bars):
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
    
    # Binary variables for area selection
    y = {}
    for e in range(num_entities):
        for a in range(num_area_choices):
            y[e, a] = model.addVar(vtype=GRB.BINARY, name=f"y_{e}_{a}")
    
    # Exactly one area per entity
    for e in range(num_entities):
        model.addConstr(gp.quicksum(y[e, a] for a in range(num_area_choices)) == 1)
    
    # Area variable for each bar (continuous, linked to binary)
    area_var = {}
    for bi in range(num_bars):
        eid = bar_to_entity[bi]
        area_var[bi] = model.addVar(lb=min(discrete_areas), ub=max(discrete_areas), name=f"area_{bi}")
        model.addConstr(area_var[bi] == gp.quicksum(discrete_areas[a] * y[eid, a] for a in range(num_area_choices)))
    
    # For each loading condition: displacement, elongation, force variables
    disp = {}
    elong = {}
    force = {}
    
    for lc in range(num_loads):
        for d in range(num_dofs):
            lb_d = disp_lb if disp_lb is not None else -GRB.INFINITY
            ub_d = disp_ub if disp_ub is not None else GRB.INFINITY
            disp[lc, d] = model.addVar(lb=lb_d, ub=ub_d, name=f"disp_{lc}_{d}")
        
        for bi in range(num_bars):
            # Force variable - bounded by stress * area
            # stress in [s_lb, s_ub], force = stress * area
            # We handle force bounds via stress constraints below
            force[lc, bi] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"force_{lc}_{bi}")
            elong[lc, bi] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"elong_{lc}_{bi}")
    
    # Objective: minimize total cost
    obj = gp.quicksum(cost_density * bar_lengths[bi] * area_var[bi] for bi in range(num_bars))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    for lc in range(num_loads):
        # Equilibrium: for each DOF d, sum over bars of B[bi,d] * force[lc,bi] = load[lc,d]
        for d in range(num_dofs):
            nonzero_bars = [(bi, bar_dof_cosines[bi, d]) for bi in range(num_bars) if abs(bar_dof_cosines[bi, d]) > 1e-15]
            if nonzero_bars:
                model.addConstr(
                    gp.quicksum(coeff * force[lc, bi] for bi, coeff in nonzero_bars) == load_vectors[lc, d],
                    name=f"equil_{lc}_{d}"
                )
            else:
                # No bars connected to this DOF - load must be zero (or this is a problem)
                pass
        
        # Compatibility: for each bar, sum over DOFs of B[bi,d] * disp[lc,d] = elong[lc,bi]
        for bi in range(num_bars):
            nonzero_dofs = [(d, bar_dof_cosines[bi, d]) for d in range(num_dofs) if abs(bar_dof_cosines[bi, d]) > 1e-15]
            model.addConstr(
                gp.quicksum(coeff * disp[lc, d] for d, coeff in nonzero_dofs) == elong[lc, bi],
                name=f"compat_{lc}_{bi}"
            )
        
        # Hooke's law: (E / L) * area * elong = force
        # This is bilinear: area * elong = force * L / E
        # We linearize using the discrete area selection
        # force[lc,bi] = (E/L) * area * elong[lc,bi]
        # = (E/L) * sum_a (discrete_areas[a] * y[e,a]) * elong[lc,bi]
        # Introduce auxiliary: z[lc,bi,a] = y[e,a] * elong[lc,bi]
        # Then force[lc,bi] = (E/L) * sum_a (discrete_areas[a] * z[lc,bi,a])
        # And z[lc,bi,a] constrained by: if y[e,a]=0 then z=0, if y[e,a]=1 then z=elong
        
        for bi in range(num_bars):
            eid = bar_to_entity[bi]
            L = bar_lengths[bi]
            EoverL = E / L
            bid = bar_ids[bi]
            s_lb = bar_stress_lb[bid]
            s_ub = bar_stress_ub[bid]
            
            # Elongation bounds based on stress
            e_lb = s_lb * L / E  # stress_lb is negative (compression)
            e_ub = s_ub * L / E  # stress_ub is positive (tension)
            
            # Create auxiliary variables for linearization
            z = {}
            for a in range(num_area_choices):
                A_a = discrete_areas[a]
                if A_a < 1e-6:
                    # Near-zero area: bar essentially absent
                    # force should be ~0, elong can be anything within bounds
                    # but force = EoverL * A_a * elong ≈ 0
                    z[a] = model.addVar(lb=e_lb, ub=e_ub, name=f"z_{lc}_{bi}_{a}")
                    # When y=0, z=0; when y=1, z=elong
                    model.addConstr(z[a] >= e_lb * y[eid, a])
                    model.addConstr(z[a] <= e_ub * y[eid, a])
                    model.addConstr(z[a] >= elong[lc, bi] - e_ub * (1 - y[eid, a]))
                    model.addConstr(z[a] <= elong[lc, bi] - e_lb * (1 - y[eid, a]))
                else:
                    z[a] = model.addVar(lb=e_lb, ub=e_ub, name=f"z_{lc}_{bi}_{a}")
                    model.addConstr(z[a] >= e_lb * y[eid, a])
                    model.addConstr(z[a] <= e_ub * y[eid, a])
                    model.addConstr(z[a] >= elong[lc, bi] - e_ub * (1 - y[eid, a]))
                    model.addConstr(z[a] <= elong[lc, bi] - e_lb * (1 - y[eid, a]))
            
            # Hooke's law
            model.addConstr(
                force[lc, bi] == EoverL * gp.quicksum(discrete_areas[a] * z[a] for a in range(num_area_choices)),
                name=f"hooke_{lc}_{bi}"
            )
            
            # Stress bounds: stress = force / area
            # force = stress * area, so s_lb * area <= force <= s_ub * area
            # force[lc,bi] >= s_lb * area_var[bi]  and  force[lc,bi] <= s_ub * area_var[bi]
            # Linearize: force >= s_lb * sum(A_a * y[e,a]) and force <= s_ub * sum(A_a * y[e,a])
            # But we already have Hooke's law which enforces this implicitly via elongation bounds
            # The elongation bounds e_lb, e_ub already encode the stress bounds
            # So we just need: e_lb <= elong <= e_ub when bar is active
            
            # Elongation constraint
            model.addConstr(elong[lc, bi] == gp.quicksum(z[a] for a in range(num_area_choices)),
                          name=f"elong_link_{lc}_{bi}")
    
    model.update()
    
    best_obj = float('inf')
    best_solution = None
    
    def callback(model, where):
        nonlocal best_obj, best_solution
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj:
                best_obj = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        # Extract areas
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
        
        # Extract displacements
        displacements_out = []
        for lc in range(num_loads):
            for d in range(num_dofs):
                displacements_out.append({
                    "dof_id": d,
                    "load": lc,
                    "value": disp[lc, d].X
                })
        
        # Extract bar forces
        bar_forces_out = []
        for lc in range(num_loads):
            for bi in range(num_bars):
                bar_forces_out.append({
                    "bar_id": bar_ids[bi],
                    "load": lc,
                    "force": force[lc, bi].X
                })
        
        solution = {
            "objective_value": obj_val,
            "bar_areas": bar_areas_out,
            "displacements": displacements_out,
            "bar_forces": bar_forces_out
        }
    else:
        # No solution found - try a heuristic fallback
        # Use the largest area for all bars
        max_area = max(discrete_areas)
        max_area_idx = discrete_areas.index(max_area)
        
        areas = np.array([max_area] * num_bars)
        obj_val = sum(cost_density * bar_lengths[bi] * areas[bi] for bi in range(num_bars))
        
        bar_areas_out = []
        for bi in range(num_bars):
            bar_areas_out.append({
                "bar_id": bar_ids[bi],
                "area": max_area,
                "area_index": max_area_idx
            })
        
        # Solve for displacements using stiffness method
        displacements_out = []
        bar_forces_out = []
        
        for lc in range(num_loads):
            # Build stiffness matrix K
            K = np.zeros((num_dofs, num_dofs))
            for bi in range(num_bars):
                L = bar_lengths[bi]
                A = areas[bi]
                k = E * A / L
                bc = bar_dof_cosines[bi]
                # K += k * bc.T @ bc (outer product)
                for d1 in range(num_dofs):
                    if abs(bc[d1]) < 1e-15:
                        continue
                    for d2 in range(num_dofs):
                        if abs(bc[d2]) < 1e-15:
                            continue
                        K[d1, d2] += k * bc[d1] * bc[d2]
            
            try:
                u = np.linalg.solve(K, load_vectors[lc])
            except np.linalg.LinAlgError:
                u = np.zeros(num_dofs)
            
            for d in range(num_dofs):
                displacements_out.append({
                    "dof_id": d,
                    "load": lc,
                    "value": float(u[d])
                })
            
            for bi in range(num_bars):
                bc = bar_dof_cosines[bi]
                e_val = bc @ u
                f_val = E * areas[bi] / bar_lengths[bi] * e_val
                bar_forces_out.append({
                    "bar_id": bar_ids[bi],
                    "load": lc,
                    "force": float(f_val)
                })
        
        solution = {
            "objective_value": obj_val,
            "bar_areas": bar_areas_out,
            "displacements": displacements_out,
            "bar_forces": bar_forces_out
        }
        
        if logger:
            logger.log(obj_val)
    
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