import json
import argparse
import time
import sys
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_bars = data['num_bars']
    num_free_dofs = data['num_free_dofs']
    num_loads = data['num_loading_conditions']
    num_areas = data['num_discrete_areas']
    areas = data['discrete_areas']
    E = data['material_properties']['modulus_of_elasticity']
    cost_density = data['material_properties']['cost_density']
    lengths = [b['length'] for b in data['bars']]
    direction_cosines = [b['direction_cosines'] for b in data['bars']]
    linking_groups = data.get('linking_groups', [])
    stress_lower = data['stress_bounds']['lower']
    stress_upper = data['stress_bounds']['upper']
    disp_lower = data['displacement_bounds'].get('lower')
    disp_upper = data['displacement_bounds'].get('upper')
    bar_specific_stress = {item['bar_id']: (item['lower'], item['upper']) for item in data.get('bar_specific_stress_bounds', [])}

    # Precompute mapping from bar_id to index (0-based)
    bar_id_to_idx = {b['bar_id']: i for i, b in enumerate(data['bars'])}

    # Load vectors: F[load][dof]
    F = np.zeros((num_loads, num_free_dofs))
    for lc in data['loading_conditions']:
        load_idx = lc['load_id'] - 1 if 'load_id' in lc and isinstance(lc['load_id'], int) else int(lc['load_id'].split()[-1]) - 1
        for load_entry in lc['loads']:
            dof_idx = load_entry['dof_id'] - 1
            F[load_idx, dof_idx] = load_entry['force']

    start_time = time.time()
    best_obj = float('inf')
    best_solution = None

    m = Model("TrussDesign")
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)
    m.setParam('LogToConsole', 0)
    m.setParam('MIPGap', 1e-4)

    # Binary variables: x[b, a] = 1 if bar b uses area a
    x = m.addVars(num_bars, num_areas, vtype=GRB.BINARY, name="x")

    # Force variables: f[b, l]
    f = m.addVars(num_bars, num_loads, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f")

    # Displacement variables: u[d, l]
    u = m.addVars(num_free_dofs, num_loads, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")

    # Area selection constraint
    for b in range(num_bars):
        m.addConstr(quicksum(x[b, a] for a in range(num_areas)) == 1)

    # Linking constraints
    for group in linking_groups:
        group_bars = [bar_id_to_idx[bid] for bid in group['bar_ids']]
        for a in range(num_areas):
            ref = x[group_bars[0], a]
            for other_b in group_bars[1:]:
                m.addConstr(x[other_b, a] == ref)

    # Equilibrium: sum_b (c_b,d * f_b,l) = F_l,d for each load and dof
    for l in range(num_loads):
        for d in range(num_free_dofs):
            expr = quicksum(direction_cosines[b][d % len(direction_cosines[b])] * f[b, l] for b in range(num_bars))
            m.addConstr(expr == F[l, d])

    # Compatibility and Hooke's law combined: f_b,l = (E * A_b / L_b) * (sum_d c_b,d * u_d,l)
    for b in range(num_bars):
        L_b = lengths[b]
        c_b = direction_cosines[b]
        for l in range(num_loads):
            delta = quicksum(c_b[d % len(c_b)] * u[d, l] for d in range(num_free_dofs))
            A_b = quicksum(areas[a] * x[b, a] for a in range(num_areas))
            m.addConstr(f[b, l] == (E / L_b) * A_b * delta)

    # Stress and elongation bounds
    for b_idx in range(num_bars):
        bar_id = data['bars'][b_idx]['bar_id']
        if bar_id in bar_specific_stress:
            s_l, s_u = bar_specific_stress[bar_id]
        else:
            s_l, s_u = stress_lower, stress_upper
        L_b = lengths[b_idx]
        for l in range(num_loads):
            delta = quicksum(direction_cosines[b_idx][d % len(direction_cosines[b_idx])] * u[d, l] for d in range(num_free_dofs))
            stress = (E / L_b) * delta
            m.addConstr(stress >= s_l)
            m.addConstr(stress <= s_u)

    # Displacement bounds
    if disp_lower is not None or disp_upper is not None:
        for d in range(num_free_dofs):
            for l in range(num_loads):
                if disp_lower is not None:
                    m.addConstr(u[d, l] >= disp_lower)
                if disp_upper is not None:
                    m.addConstr(u[d, l] <= disp_upper)

    # Objective: minimize sum_b cost_density * L_b * A_b
    obj_expr = quicksum(cost_density * lengths[b] * quicksum(areas[a] * x[b, a] for a in range(num_areas)) for b in range(num_bars))
    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Callback to log improved solutions
    def callback(model, where):
        nonlocal best_obj, best_solution
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    m.optimize(callback)

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        if m.SolCount > 0:
            obj_val = m.ObjVal
            if obj_val < best_obj:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

            # Extract solution
            bar_areas = []
            for b in range(num_bars):
                for a in range(num_areas):
                    if x[b, a].X > 0.5:
                        bar_areas.append({
                            "bar_id": data['bars'][b]['bar_id'],
                            "area": float(areas[a]),
                            "area_index": int(a)
                        })
                        break

            displacements = []
            for d in range(num_free_dofs):
                for l_idx in range(num_loads):
                    displacements.append({
                        "dof_id": data['degrees_of_freedom'][d]['dof_id'],
                        "load": l_idx + 1,
                        "value": float(u[d, l_idx].X)
                    })

            bar_forces = []
            for b in range(num_bars):
                for l_idx in range(num_loads):
                    bar_forces.append({
                        "bar_id": data['bars'][b]['bar_id'],
                        "load": l_idx + 1,
                        "force": float(f[b, l_idx].X)
                    })

            solution = {
                "objective_value": float(obj_val),
                "bar_areas": bar_areas,
                "displacements": displacements,
                "bar_forces": bar_forces
            }

            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
        else:
            # No feasible solution found
            solution = {
                "objective_value": 0.0,
                "bar_areas": [],
                "displacements": [],
                "bar_forces": []
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
    else:
        # Infeasible or other failure
        solution = {
            "objective_value": 0.0,
            "bar_areas": [],
            "displacements": [],
            "bar_forces": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()