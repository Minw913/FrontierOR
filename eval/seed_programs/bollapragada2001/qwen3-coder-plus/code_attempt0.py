import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import time


def solve_truss_optimization(instance_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Initialize logger
    logger = None
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")

    # Prepare data structures
    num_bars = data['num_bars']
    num_loading_conditions = data['num_loading_conditions']
    num_free_dofs = data['num_free_dofs']
    discrete_areas = data['discrete_areas']
    num_discrete_areas = len(discrete_areas)
    
    # Build mapping from bar_id to index
    bar_id_to_idx = {bar['bar_id']: i for i, bar in enumerate(data['bars'])}
    dof_id_to_idx = {dof['dof_id']: i for i, dof in enumerate(data['degrees_of_freedom'])}
    
    # Direction cosine matrix (shape: num_bars x num_free_dofs)
    A_dir = np.zeros((num_bars, num_free_dofs))
    for bar in data['bars']:
        bar_idx = bar_id_to_idx[bar['bar_id']]
        for i, dof_ref in enumerate(bar['direction_cosines']):
            # Need to map from bar's nodes' DOFs to global DOF index
            # Since the bar spans two nodes, we need to know the DOFs associated with each node
            node_i = bar['node_i']
            node_j = bar['node_j']
            
            # Determine direction cosines for each DOF affected by the bar
            # For each spatial dimension, find corresponding DOFs
            dimension = data['dimension']
            for dim in range(dimension):
                dir_char = ['x', 'y', 'z'][dim] if dimension == 3 else (['x', 'y'] if dimension == 2 else ['x'])
                
                # Find DOFs for node_i and node_j in this direction
                dof_i = None
                dof_j = None
                for dof in data['degrees_of_freedom']:
                    if dof['node'] == node_i and dof['direction'] == dir_char[dim]:
                        dof_i = dof_id_to_idx[dof['dof_id']]
                        break
                for dof in data['degrees_of_freedom']:
                    if dof['node'] == node_j and dof['direction'] == dir_char[dim]:
                        dof_j = dof_id_to_idx[dof['dof_id']]
                        break
                
                if dof_i is not None:
                    A_dir[bar_idx, dof_i] += dof_ref if i == dim else 0.0
                if dof_j is not None:
                    A_dir[bar_idx, dof_j] -= dof_ref if i == dim else 0.0
    
    # Actually compute direction cosines properly based on spatial orientation
    A_dir = np.zeros((num_bars, num_free_dofs))
    for bar in data['bars']:
        bar_idx = bar_id_to_idx[bar['bar_id']]
        node_i_id = bar['node_i']
        node_j_id = bar['node_j']
        
        # Get node coordinates
        node_i = next(n for n in data['nodes'] if n['node_id'] == node_i_id)
        node_j = next(n for n in data['nodes'] if n['node_id'] == node_j_id)
        
        # Calculate direction vector
        dx = [node_j['x'] - node_i['x'], 
              node_j['y'] - node_i['y']] + ([node_j.get('z', 0) - node_i.get('z', 0)] if data['dimension'] == 3 else [])
        dx_normalized = [d / bar['length'] for d in dx]  # normalize by length
        
        # Map to DOF indices
        for dim_i, dir_char in enumerate(['x', 'y', 'z'][:data['dimension']]):
            # Find DOF for node_i in direction dir_char
            for dof in data['degrees_of_freedom']:
                if dof['node'] == node_i_id and dof['direction'] == dir_char:
                    dof_idx = dof_id_to_idx[dof['dof_id']]
                    A_dir[bar_idx, dof_idx] = dx_normalized[dim_i]
            # Find DOF for node_j in direction dir_char
            for dof in data['degrees_of_freedom']:
                if dof['node'] == node_j_id and dof['direction'] == dir_char:
                    dof_idx = dof_id_to_idx[dof['dof_id']]
                    A_dir[bar_idx, dof_idx] = -dx_normalized[dim_i]

    # Create load vectors for each loading condition
    F_loads = []
    for load_cond in data['loading_conditions']:
        F = np.zeros(num_free_dofs)
        for load in load_cond['loads']:
            dof_idx = dof_id_to_idx[load['dof_id']]
            F[dof_idx] = load['force']
        F_loads.append(F)

    # Stress bounds per bar
    stress_lower = np.full(num_bars, data['stress_bounds']['lower'])
    stress_upper = np.full(num_bars, data['stress_bounds']['upper'])
    for stress_bound in data['bar_specific_stress_bounds']:
        bar_idx = bar_id_to_idx[stress_bound['bar_id']]
        stress_lower[bar_idx] = stress_bound['lower']
        stress_upper[bar_idx] = stress_bound['upper']

    # Set up Gurobi model
    model = gp.Model("truss_design")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.NonConvex, 5)

    # Variables: for each bar, select one area from discrete options
    # x[b][a]: binary variable - 1 if bar b uses area a
    x = []
    for b in range(num_bars):
        x_bar = []
        for a in range(num_discrete_areas):
            var = model.addVar(vtype=GRB.BINARY, name=f"x_{b}_{a}")
            x_bar.append(var)
        x.append(x_bar)

    # Displacement variables
    u = []  # u[load][dof]
    for load_idx in range(num_loading_conditions):
        u_load = []
        for dof_idx in range(num_free_dofs):
            var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"u_{load_idx}_{dof_idx}")
            u_load.append(var)
        u.append(u_load)

    # Bar force variables
    force = []  # force[bar][load]
    for b in range(num_bars):
        force_bar = []
        for load_idx in range(num_loading_conditions):
            var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"f_{b}_{load_idx}")
            force_bar.append(var)
        force.append(force_bar)

    # Bar elongation variables
    elong = []  # elong[bar][load]
    for b in range(num_bars):
        elong_bar = []
        for load_idx in range(num_loading_conditions):
            var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"e_{b}_{load_idx}")
            elong_bar.append(var)
        elong.append(elong_bar)

    # Displacement bounds
    if data['displacement_bounds']['lower'] is not None:
        for load_idx in range(num_loading_conditions):
            for dof_idx in range(num_free_dofs):
                model.addConstr(u[load_idx][dof_idx] >= data['displacement_bounds']['lower'])
    if data['displacement_bounds']['upper'] is not None:
        for load_idx in range(num_loading_conditions):
            for dof_idx in range(num_free_dofs):
                model.addConstr(u[load_idx][dof_idx] <= data['displacement_bounds']['upper'])

    # Length constraints per bar
    lengths = [bar['length'] for bar in data['bars']]
    
    # E (modulus of elasticity) - common across all bars
    E = data['material_properties']['modulus_of_elasticity']
    
    # Each bar must select exactly one area
    for b in range(num_bars):
        model.addConstr(gp.quicksum(x[b][a] for a in range(num_discrete_areas)) == 1)

    # Linking groups - bars in the same group must have same area
    for group in data['linking_groups']:
        bar_indices = [bar_id_to_idx[bar_id] for bar_id in group['bar_ids']]
        # Ensure all bars in group select same area combination
        for a in range(num_discrete_areas):
            ref_bar_idx = bar_indices[0]
            for other_bar_idx in bar_indices[1:]:
                model.addConstr(x[ref_bar_idx][a] == x[other_bar_idx][a])

    # Equilibrium constraint: sum over bars of (direction cos * force) == load
    for load_idx in range(num_loading_conditions):
        F = F_loads[load_idx]
        for dof_idx in range(num_free_dofs):
            expr = gp.LinExpr()
            for b in range(num_bars):
                expr.addTerms(A_dir[b][dof_idx], force[b][load_idx])
            model.addConstr(expr == F[dof_idx])

    # Compatibility: elongation = sum over DOFs of (direction cos * displacement)
    for load_idx in range(num_loading_conditions):
        for b in range(num_bars):
            expr = gp.LinExpr()
            for dof_idx in range(num_free_dofs):
                expr.addTerms(A_dir[b][dof_idx], u[load_idx][dof_idx])
            model.addConstr(elong[b][load_idx] == expr)

    # Hooke's Law: F = (E*A/L) * e
    for load_idx in range(num_loading_conditions):
        for b in range(num_bars):
            areas = discrete_areas
            # Linearize the relationship: F[b][load_idx] = (E/L[b]) * sum_x(area[a] * x[b][a]) * elong[b][load_idx]
            # This creates bilinear terms which Gurobi can handle with appropriate settings
            area_expr = gp.LinExpr()
            for a in range(num_discrete_areas):
                area_expr.addTerms(areas[a], x[b][a])
            # Force = E/L * area_selection * elongation
            # F_{bl} = (E/L_b) * sum_a(areas[a] * x_ba) * e_bl
            # Model as F[b][l] = (E/L[b]) * s_b * e[b][l] where s_b = sum_a(areas[a]*x_ba)
            model.addGenConstrIndicator(x[b][0], True, 
                                       force[b][load_idx] == (E/lengths[b]) * areas[0] * elong[b][load_idx])
            # For linearization purposes, instead we use the following constraints:
            for a in range(num_discrete_areas):
                # If x[b][a]==1, then force = (E/L) * areas[a] * elongation
                big_M1 = max(abs(stress_upper[b]), abs(stress_lower[b])) * max(areas)  # Reasonable upper bound for force
                big_M2 = 1000  # General M (problem dependent)
                
                # Linearize F = (E/L)*A*e using indicator constraints would be complex
                # Instead we formulate as: force[b,l] = (E/L_b) * sum_a(A[a] * x[b][a] * e[b][l])
                # Let t_aba = x_ba * e_bl for each area option
                # Then force[b][l] = (E/L_b) * sum_a(A[a] * t_ba_bl)
                pass

    # To linearize, we add helper variables for each bilinear product
    # Instead of using indicators per area, linearize fully using additional variables
    # t[b][a][l] = x[b][a] * elong[b][l]
    t = []  # t[bar][area][load]
    for b in range(num_bars):
        t_bar = []
        for a in range(num_discrete_areas):
            t_area = []
            for l in range(num_loading_conditions):
                var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"t_{b}_{a}_{l}")
                t_area.append(var)
            t_bar.append(t_area)
        t.append(t_bar)

    # Add constraints for bilinear term linearization
    for b in range(num_bars):
        for a in range(num_discrete_areas):
            for l in range(num_loading_conditions):
                # We want t[b][a][l] = x[b][a] * elong[b][l]
                # Assuming bounds on elong: lb and ub
                # Without knowing exact bounds, we estimate them
                # elong limits might be implied by physical properties
                max_elong_abs = max(abs(bound) for bound in [-min(lengths)*0.01, max(lengths)*0.01])  # rough estimate
                min_val, max_val = -max_elong_abs, max_elong_abs

                # Bilinear relaxation: t <= ub*x + e*(1-x), t >= lb*x + e*(1-x), ...
                # Where e is elong[b][l]
                model.addConstr(t[b][a][l] <= max_val * x[b][a] + elong[b][l] - max_val * (1 - x[b][a]))
                model.addConstr(t[b][a][l] >= min_val * x[b][a] + elong[b][l] - min_val * (1 - x[b][a]))
                model.addConstr(t[b][a][l] <= elong[b][l] - min_val * (1 - x[b][a]))
                model.addConstr(t[b][a][l] >= elong[b][l] - max_val * (1 - x[b][a]))

    # Now Hooke's law using linearized form
    for load_idx in range(num_loading_conditions):
        for b in range(num_bars):
            areas = discrete_areas
            expr = gp.LinExpr()
            for a in range(num_discrete_areas):
                expr.addTerms((E/lengths[b]) * areas[a], t[b][a][load_idx])
            model.addConstr(force[b][load_idx] == expr)

    # Stress bounds: stress = force/area
    for load_idx in range(num_loading_conditions):
        for b in range(num_bars):
            # Define new variables for stress computation to handle division
            # This is complex because stress = force/area and area depends on x vars
            # For each area selection x[b][a], ensure stress lies within bounds
            for a in range(num_discrete_areas):
                area = discrete_areas[a]
                if abs(area) < 1e-9: continue  # Skip near-zero areas
                
                # When x[b][a]=1, stress should be in [stress_lower[b], stress_upper[b]]
                # stress = force[b][load_idx] / area
                if area > 0:
                    model.addConstr(force[b][load_idx] >= stress_lower[b] * area - (1 - x[b][a]) * 1e6)
                    model.addConstr(force[b][load_idx] <= stress_upper[b] * area + (1 - x[b][a]) * 1e6)

    # Objectives: Minimize cost
    obj_expr = gp.LinExpr()
    cost_density = data['material_properties']['cost_density']
    for b in range(num_bars):
        for a in range(num_discrete_areas):
            obj_expr.addTerms(cost_density * lengths[b] * discrete_areas[a], x[b][a])
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Define callback for logging
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    # Optimize
    if logger:
        model.optimize(mycallback)
    else:
        model.optimize()

    # Process solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract variable values
        selected_areas = {}
        for b in range(num_bars):
            for a in range(num_discrete_areas):
                if x[b][a].X > 0.5:  # Selected area index
                    selected_areas[b] = {'value': discrete_areas[a], 'index': a}
                    
        # Compute other outputs based on selected areas
        solution_bar_forces = []
        solution_displacements = []
        
        for load_idx in range(num_loading_conditions):
            for b_idx, b_data in enumerate(data['bars']):
                sol_force = force[b_idx][load_idx].X
                solution_bar_forces.append({
                    'bar_id': data['bars'][b_idx]['bar_id'],
                    'load': load_idx,
                    'force': sol_force
                })
                
            for dof_idx, dof_data in enumerate(data['degrees_of_freedom']):
                sol_disp = u[load_idx][dof_idx].X
                solution_displacements.append({
                    'dof_id': data['degrees_of_freedom'][dof_idx]['dof_id'],
                    'load': load_idx,
                    'value': sol_disp
                })

        # Format solution according to output schema
        obj_value = model.objVal if model.status == GRB.OPTIMAL else float('inf')
        best_solution = {
            'objective_value': obj_value,
            'bar_areas': [],
            'displacements': solution_displacements,
            'bar_forces': solution_bar_forces
        }
        
        for b_idx, b_data in enumerate(data['bars']):
            bar_area_info = {
                'bar_id': b_data['bar_id'],
                'area': selected_areas[b_idx]['value'],
                'area_index': selected_areas[b_idx]['index']
            }
            best_solution['bar_areas'].append(bar_area_info)

        return best_solution
    else:
        return {
            'objective_value': float('inf'),
            'bar_areas': [{'bar_id': b['bar_id'], 'area': discrete_areas[0], 'area_index': 0} for b in data['bars']],
            'displacements': [{'dof_id': dof['dof_id'], 'load': 0, 'value': 0.0} for dof in data['degrees_of_freedom']],
            'bar_forces': [{'bar_id': b['bar_id'], 'load': 0, 'force': 0.0} for b in data['bars']]
        }


def main():
    parser = argparse.ArgumentParser(description='Solve truss optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write solution JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to write log file')

    args = parser.parse_args()

    # Solve the instance
    solution = solve_truss_optimization(
        instance_path=args.instance_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )

    # Write the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()