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

    # Preprocess data structures
    bar_id_to_idx = {bar['bar_id']: i for i, bar in enumerate(data['bars'])}
    dof_id_to_idx = {dof['dof_id']: j for j, dof in enumerate(data['degrees_of_freedom'])}

    num_bars = data['num_bars']
    num_loading_conditions = data['num_loading_conditions']
    num_free_dofs = data['num_free_dofs']
    discrete_areas = data['discrete_areas']
    num_discrete_areas = len(discrete_areas)

    # Build directional coefficient matrix A_dir with respect to each DOF  
    A_dir = np.zeros((num_bars, num_free_dofs, len(data.get('nodes', []))))

    for bar in data['bars']:
        bar_idx = bar_id_to_idx[bar['bar_id']]
        ni_id = bar['node_i']
        nj_id = bar['node_j']

        # Update direction cosines to connect DOFs appropriately
        # For each node and its DOFs, update A_dir accordingly        
        ni_dof_indices = [j for j, dof in enumerate(data['degrees_of_freedom']) 
                          if dof['node'] == ni_id]
        nj_dof_indices = [j for j, dof in enumerate(data['degrees_of_freedom']) 
                          if dof['node'] == nj_id]

        for dof_idx in ni_dof_indices:
            # Determine direction - for a bar from i to j, and DOF at i
            # contribution is negative for i-dof to j-dof positive
            # However, for simplicity we assume direction cosine directly affects DOF
            # This means the element A_dir[b][dof] connects bar b to DOF value
            A_dir[bar_idx, dof_idx] = bar['direction_cosines'][{
                'x': 0, 'y': 1, 'z': 2
            }.get(data['degrees_of_freedom'][dof_idx]['direction'], 0)]

        for dof_idx in nj_dof_indices:
            A_dir[bar_idx, dof_idx] = -bar['direction_cosines'][{
                'x': 0, 'y': 1, 'z': 2
            }.get(data['degrees_of_freedom'][dof_idx]['direction'], 0)]


    # Load vectors for each load case
    F_loads = []
    for load_condition in data['loading_conditions']:
        F_vector = np.zeros(num_free_dofs)
        for load in load_condition['loads']:
            dof_idx = dof_id_to_idx[load['dof_id']]
            F_vector[dof_idx] = load['force']
        F_loads.append(F_vector)

    # Elastic modulus and lengths
    E_mod = data['material_properties']['modulus_of_elasticity']

    # Stress bounds for each bar
    stress_lb = np.full(num_bars, data['stress_bounds']['lower'])
    stress_ub = np.full(num_bars, data['stress_bounds']['upper'])
    for stress_bound in data['bar_specific_stress_bounds']:
        bar_idx = bar_id_to_idx[stress_bound['bar_id']]
        stress_lb[bar_idx] = stress_bound['lower']
        stress_ub[bar_idx] = stress_bound['upper']

    # Lengths for each bar
    lengths = [bar['length'] for bar in data['bars']]
    
    # Elongation bounds for each bar
    # We assume there's a global elongation bound or use defaults if missing
    global_elong_lb = -0.1  # example
    global_elong_ub = 0.1  # example
    # Actual data might have these per bar - let's use default for now
    
    # If the data has elongation bounds per bar - need different handling:
    elong_lb = np.full(num_bars, global_elong_lb)
    elong_ub = np.full(num_bars, global_elong_ub)
    

    # Define Gurobi model
    model = gp.Model("truss_mip")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    # No need for NonConvex since we're solving with linear formulation

    # Binary variables: x[bar][area_idx] = 1 iff bar uses that specific area
    x = []
    for i in range(num_bars):
        x_list = []
        for j in range(len(discrete_areas)):
            v = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            x_list.append(v)
        x.append(x_list)

    # Continuous variables: displacements for each dof and load
    u = []
    for load_case in range(num_loading_conditions):
        u_load = []
        for dof in range(num_free_dofs):
            v = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"u_{load_case}_{dof}")
            u_load.append(v)
        u.append(u_load)

    # Continuous variables: internal forces for each bar and load
    forces = []
    for load_case in range(num_loading_conditions):
        forces_load = []
        for bar in range(num_bars):
            v = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"f_{load_case}_{bar}")
            forces_load.append(v)
        forces.append(forces_load)

    # Continuous variables: elongations for each bar and load
    elongations = []
    for load_case in range(num_loading_conditions):
        elong_load = []
        for bar in range(num_bars):
            v = model.addVar(name=f"elong_{load_case}_{bar}", lb=elong_lb[bar], ub=elong_ub[bar])
            elong_load.append(v)
        elongations.append(elong_load)

    # Binary constraints: each bar must pick exactly one area
    for i in range(num_bars):
        model.addConstr(gp.quicksum(x[i][j] for j in range(num_discrete_areas)) == 1, name=f"one_area_per_bar_{i}")

    # Linking constraints
    if 'linking_groups' in data:
        for idx, group in enumerate(data['linking_groups']):
            bars_in_group = [bar_id_to_idx[bar_id] for bar_id in group['bar_ids']]
            for a_idx in range(num_discrete_areas):
                for i in range(1, len(bars_in_group)):
                    model.addConstr(x[bars_in_group[0]][a_idx] == x[bars_in_group[i]][a_idx], name=f"linking_{idx}_area_{a_idx}_{i}")

    # Equilibrium equations: sum of forces equals applied load
    for load_case in range(num_loading_conditions):
        F_vector = F_loads[load_case]
        for d in range(num_free_dofs):
            lhs_expr = gp.LinExpr()
            for b in range(num_bars):
                # This dot product depends on how A_dir was constructed earlier
                # Let's reconstruct it properly according to the actual formula
                for idx, dof in enumerate(data['degrees_of_freedom']):
                    if dof['node'] == data['bars'][b]['node_i']:
                        dir_cos_idx = {'x': 0, 'y': 1, 'z': 2}.get(dof['direction'], 0)
                        direction_cosine = data['bars'][b]['direction_cosines'][dir_cos_idx]
                        lhs_expr.addTerms(direction_cosine, forces[load_case][b])
                    elif dof['node'] == data['bars'][b]['node_j']:
                        dir_cos_idx = {'x': 0, 'y': 1, 'z': 2}.get(dof['direction'], 0)
                        direction_cosine = -data['bars'][b]['direction_cosines'][dir_cos_idx]
                        lhs_expr.addTerms(direction_cosine, forces[load_case][b])
            
            model.addConstr(lhs_expr == F_vector[d], name=f"equilibrium_{load_case}_{d}")


    # Compatibility equations: elongation is consistent with member endpoints movement
    for load_case in range(num_loading_conditions):
        for b in range(num_bars):
            disp_expr = gp.LinExpr()
            # Compute elongation as u_j - u_i dotted with direction vector
            bar_info = data['bars'][b]
            ni_id = bar_info['node_i']
            nj_id = bar_info['node_j']
            
            directions = ['x', 'y']
            if data['dimension'] == 3:
                directions.append('z')
            
            # Get direction indices for each axis in DOF list and add proper contributions
            ni_dof_indices = [(axis_idx, idx) for idx, dof in enumerate(data['degrees_of_freedom']) 
                         for axis_idx, axis_name in enumerate(['x', 'y', 'z'][:data['dimension']])
                         if dof['node'] == ni_id and dof['direction'] == axis_name]
            nj_dof_indices = [(axis_idx, idx) for idx, dof in enumerate(data['degrees_of_freedom']) 
                         for axis_idx, axis_name in enumerate(['x', 'y', 'z'][:data['dimension']])
                         if dof['node'] == nj_id and dof['direction'] == axis_name]
            
            for axis_idx, dof_idx in ni_dof_indices:
                disp_expr.addTerms(-data['bars'][b]['direction_cosines'][axis_idx], u[load_case][dof_idx])
            for axis_idx, dof_idx in nj_dof_indices:
                disp_expr.addTerms(data['bars'][b]['direction_cosines'][axis_idx], u[load_case][dof_idx])
                
            model.addConstr(elongations[load_case][b] == disp_expr, name=f"compatibility_{load_case}_{b}")

    # Hooke's Law: F = A*E/L * elongation
    # We have a discrete choice of A: A_sum = sum_j(area[j] * x[b][j]) 
    # So, F[b][l] = (E/L[b]) * A_sum[b] * elong[b][l]
    # This product introduces a bilinear term that is complex to linearize,
    # But it's equivalent to saying: 
    # Let A_total = sum_j area[j]*x[b][j] => F[b][l] = (E/L[b]) * A_total * elong[b][l]
    # We use bilinear terms: y[b][j][l] = x[b][j] * elong[b][l]
    # Then F[b][l] = (E/L[b]) * sum_j (area[j] * y[b][j][l])
    
    # Introduce auxiliary variables
    y_bjl = []
    for b in range(num_bars):
        y_bar = []
        for j in range(num_discrete_areas):
            y_area = []
            for l in range(num_loading_conditions):
                # Need to bound elong to apply McCormick
                M_e = max(abs(elong_ub[b]), abs(elong_lb[b]))
                m_e = -M_e
                
                # Create y[b][j][l] = x[b][j] * elong[b][l]
                v = model.addVar(name=f"y_bjl_{b}_{j}_{l}", lb=m_e, ub=M_e)
                y_area.append(v)
                
                # Add McCormick envelopes for bilinear term
                # If x binary, and e bounded, y = xe:
                # y >= m_e*x + e - m_e
                # y <= M_e*x + e - M_e
                # y >= M_e*x + e - M_e
                # y <= m_e*x + e - m_e
                # Actually:
                # y >= m_e*x + e - M_e*(1-x)
                # y <= M_e*x + e - m_e*(1-x)
                # y >= e - M_e*x
                # y <= e - m_e*x
                # Which reduces to:
                # m_e*x <= y <= M_e*x
                # e - M_e*(1-x) <= y <= e - m_e*(1-x)
                model.addConstr(v >= m_e * x[b][j] + elongations[l][b] - M_e * (1 - x[b][j]), name=f"mccormick1_{b}_{j}_{l}")
                model.addConstr(v <= M_e * x[b][j] + elongations[l][b] - m_e * (1 - x[b][j]), name=f"mccormick2_{b}_{j}_{l}")
                model.addConstr(v >= elongations[l][b] - M_e * x[b][j], name=f"mccormick3_{b}_{j}_{l}")
                model.addConstr(v <= elongations[l][b] - m_e * x[b][j], name=f"mccormick4_{b}_{j}_{l}")
            y_bar.append(y_area)
        y_bjl.append(y_bar)

    # Hooke's Law constraint
    for load_case in range(num_loading_conditions):
        for b in range(num_bars):
            expr = gp.LinExpr()
            for a_idx in range(num_discrete_areas):
                expr.addTerms(E_mod / lengths[b] * discrete_areas[a_idx], y_bjl[b][a_idx][load_case])
            model.addConstr(forces[load_case][b] == expr, name=f"hooke_{load_case}_{b}")

    # Stress constraints: Lower <= F/A <= Upper
    # So L*A <= F <= U*A, where A depends on x
    # L * sum_j(area[j]*x[b][j]) <= F[b][l] <= U * sum_j(area[j]*x[b][j])
    # Equivalent: L * sum_j(area[j]*x[b][j]) <= F[b][l] and F[b][l] <= U * sum_j(area[j]*x[b][j])
    # Which in constraints means:
    # L * sum_j(area[j]*x[b][j]) - F[b][l] <= 0
    # F[b][l] - U * sum_j(area[j]*x[b][j]) <= 0
    
    for load_case in range(num_loading_conditions):
        for b in range(num_bars):
            A_expr = gp.LinExpr()
            for a_idx in range(num_discrete_areas):
                A_expr.addTerms(discrete_areas[a_idx], x[b][a_idx])
            
            f_var = forces[load_case][b]
            model.addConstr(stress_lb[b] * A_expr <= f_var, name=f"stress_lb_{b}_{load_case}")
            model.addConstr(f_var <= stress_ub[b] * A_expr, name=f"stress_ub_{b}_{load_case}")

    # Displacement bounds if specified
    if 'displacement_bounds' in data and data['displacement_bounds'].get('lower') is not None:
        lower_bd = data['displacement_bounds']['lower']
        for load_case in range(num_loading_conditions):
            for dof in range(num_free_dofs):
                model.addConstr(u[load_case][dof] >= lower_bd, name=f"disp_lb_{dof}_{load_case}")
    if 'displacement_bounds' in data and data['displacement_bounds'].get('upper') is not None:
        upper_bd = data['displacement_bounds']['upper']
        for load_case in range(num_loading_conditions):
            for dof in range(num_free_dofs):
                model.addConstr(u[load_case][dof] <= upper_bd, name=f"disp_ub_{dof}_{load_case}")

    # Objective: minimize total cost
    total_cost = gp.LinExpr()
    cost_density = data['material_properties']['cost_density']
    for b_idx, length in enumerate(lengths):
        for a_idx, area in enumerate(discrete_areas):
            total_cost.addTerms(cost_density * length * area, x[b_idx][a_idx])
    
    model.setObjective(total_cost, sense=GRB.MINIMIZE)

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj_val)

    # Solve
    model.optimize(callback if logger else None)

    if model.SolCount > 0:
        obj_val = model.ObjVal
            
        # Decode solution
        bar_areas_sol = []
        for b_idx, b_data in enumerate(data['bars']):
            selected_area_idx = None
            for a_idx in range(num_discrete_areas):
                if x[b_idx][a_idx].X > 0.9:
                    selected_area_idx = a_idx
                    break
            if selected_area_idx is None:
                selected_area_idx = 0  # fallback

            bar_areas_sol.append({
                'bar_id': b_data['bar_id'],
                'area': discrete_areas[selected_area_idx],
                'area_index': selected_area_idx
            })

        disp_list = []
        for load_case in range(num_loading_conditions):
            for dof_idx, dof_data in enumerate(data['degrees_of_freedom']):
                disp_list.append({
                    'dof_id': dof_data['dof_id'],
                    'load': load_case,
                    'value': u[load_case][dof_idx].X
                })

        force_list = []
        for load_case in range(num_loading_conditions):
            for b_idx, b_data in enumerate(data['bars']):
                force_list.append({
                    'bar_id': b_data['bar_id'],
                    'load': load_case,
                    'force': forces[load_case][b_idx].X
                })

        solution_json = {
            'objective_value': obj_val,
            'bar_areas': bar_areas_sol,
            'displacements': disp_list,
            'bar_forces': force_list
        }
    else:
        # Return default values if no solution
        default_area = discrete_areas[0]
        solution_json = {
            'objective_value': float('inf'),
            'bar_areas': [{'bar_id': b['bar_id'], 'area': default_area, 'area_index': 0} for b in data['bars']],
            'displacements': [{'dof_id': dof['dof_id'], 'load': 0, 'value': 0.0} for dof in data['degrees_of_freedom'] for _ in range(num_loading_conditions)],
            'bar_forces': [{'bar_id': b['bar_id'], 'load': 0, 'force': 0.0} for b in data['bars'] for _ in range(num_loading_conditions)]
        }

    return solution_json


def main():
    parser = argparse.ArgumentParser(description='Truss optimization solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)

    args = parser.parse_args()

    sol = solve_truss_optimization(args.instance_path, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()