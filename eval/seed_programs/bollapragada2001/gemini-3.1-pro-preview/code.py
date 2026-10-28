import argparse
import json
import sys
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

def get_dir_index(direction_str):
    d = str(direction_str).lower().strip()
    if d in ('x', '1', 'u_x', 'ux'): return 0
    if d in ('y', '2', 'u_y', 'uy'): return 1
    if d in ('z', '3', 'u_z', 'uz'): return 2
    try:
        return int(d)
    except Exception:
        pass
    raise ValueError(f"Unknown direction '{direction_str}'")

def extract_group_bars(g):
    if isinstance(g, dict):
        if 'bars' in g:
            return g['bars']
        elif 'bar_ids' in g:
            return g['bar_ids']
        else:
            for k, v in g.items():
                if isinstance(v, list):
                    return v
        return []
    elif isinstance(g, list):
        return g
    return []

def main():
    parser = argparse.ArgumentParser(description="Truss Sizing Optimization MINLP")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', type=str, help="Path to logging file")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance problem data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    bar_ids = [b['bar_id'] for b in instance.get('bars', [])]
    free_dofs = [d['dof_id'] for d in instance.get('degrees_of_freedom', [])]
    load_ids = [lc['load_id'] for lc in instance.get('loading_conditions', [])]
    discrete_areas = instance.get('discrete_areas', [])
    area_indices = list(range(len(discrete_areas)))

    # Parse structural topology & direction mappings
    # C maps a (bar, dof) combination to its directional cosine coefficient
    C = {}
    for dof in instance.get('degrees_of_freedom', []):
        d = dof['dof_id']
        n = dof['node']
        dir_idx = get_dir_index(dof['direction'])
        
        for bar in instance.get('bars', []):
            b = bar['bar_id']
            cosines = bar.get('direction_cosines', [])
            if bar['node_j'] == n:
                if dir_idx < len(cosines):
                    C[(b, d)] = cosines[dir_idx]
            elif bar['node_i'] == n:
                if dir_idx < len(cosines):
                    C[(b, d)] = -cosines[dir_idx]

    # Pre-parse given static external loads per DOF and load condition
    F_ext = {(d, l): 0.0 for d in free_dofs for l in load_ids}
    for lc in instance.get('loading_conditions', []):
        l = lc['load_id']
        for load in lc.get('loads', []):
            d = load.get('dof_id')
            if d in free_dofs:
                F_ext[(d, l)] += load.get('force', 0.0)

    # Begin Gurobi Model
    model = gp.Model("Truss_Optimization")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Variables
    x = model.addVars(bar_ids, area_indices, vtype=GRB.BINARY, name="x")
    u = model.addVars(free_dofs, load_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")
    e = model.addVars(bar_ids, load_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="e")
    f = model.addVars(bar_ids, load_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f")
    f_aux = model.addVars(bar_ids, area_indices, load_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f_aux")

    # Objective Function
    cost_density = instance.get('material_properties', {}).get('cost_density', 1.0)
    obj_expr = gp.LinExpr()
    for bar in instance.get('bars', []):
        b = bar['bar_id']
        L = bar['length']
        for k in area_indices:
            A = discrete_areas[k]
            obj_expr += cost_density * L * A * x[b, k]
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # 1. Selection and Topology Rules
    for b in bar_ids:
        model.addConstr(gp.quicksum(x[b, k] for k in area_indices) == 1, name=f"assign_{b}")

    for g in instance.get('linking_groups', []):
        gbars = [b for b in extract_group_bars(g) if b in bar_ids]
        if len(gbars) > 1:
            first_b = gbars[0]
            for b in gbars[1:]:
                for k in area_indices:
                    model.addConstr(x[first_b, k] == x[b, k], name=f"link_{first_b}_{b}_{k}")

    # 2. General Displacement Bounds
    disp_bounds = instance.get('displacement_bounds', {})
    disp_lb = disp_bounds.get('lower')
    disp_ub = disp_bounds.get('upper')
    for l in load_ids:
        for d in free_dofs:
            if disp_lb is not None:
                model.addConstr(u[d, l] >= disp_lb, name=f"disp_lb_{d}_{l}")
            if disp_ub is not None:
                model.addConstr(u[d, l] <= disp_ub, name=f"disp_ub_{d}_{l}")

    # 3. Structural Mechanics, Compatibility, Equilibrium, & Hooke's M-Constraints
    E = instance.get('material_properties', {}).get('modulus_of_elasticity', 1.0)
    
    # Compile stress bounds map
    sb_default = instance.get('stress_bounds', {})
    default_s_min = sb_default.get('lower', -1e12)
    default_s_max = sb_default.get('upper', 1e12)
    stress_min = {b: default_s_min for b in bar_ids}
    stress_max = {b: default_s_max for b in bar_ids}
    for sb in instance.get('bar_specific_stress_bounds', []):
        stress_min[sb['bar_id']] = sb['lower']
        stress_max[sb['bar_id']] = sb['upper']

    # For each local structural mechanism
    for bar in instance.get('bars', []):
        b = bar['bar_id']
        L = bar['length']
        s_min = stress_min[b]
        s_max = stress_max[b]
        
        # Determine global kinematic limit envelope
        e_min_bounds = [min(s_min, s_max) * L / E]
        e_max_bounds = [max(s_min, s_max) * L / E]
        
        eb = instance.get('elongation_bounds', {})
        if eb.get('lower') is not None: e_min_bounds.append(eb['lower'])
        if eb.get('upper') is not None: e_max_bounds.append(eb['upper'])
        
        beb = bar.get('elongation_bounds', {})
        if beb.get('lower') is not None: e_min_bounds.append(beb['lower'])
        if beb.get('upper') is not None: e_max_bounds.append(beb['upper'])
        
        e_min = max(e_min_bounds)
        e_max = min(e_max_bounds)
        if e_min > e_max:
            e_min = e_max 
            
        for l in load_ids:
            # Kinematic compatibility (Elongation = C * Displacement)
            cmp_expr = gp.LinExpr()
            for d in free_dofs:
                if (b, d) in C:
                    cmp_expr += C[(b, d)] * u[d, l]
            model.addConstr(e[b, l] == cmp_expr, name=f"compat_{b}_{l}")

            # Safety Elongation constraints
            model.addConstr(e[b, l] >= e_min, name=f"e_min_{b}_{l}")
            model.addConstr(e[b, l] <= e_max, name=f"e_max_{b}_{l}")

            # Equilibrium helper continuous forces aggregate
            model.addConstr(f[b, l] == gp.quicksum(f_aux[b, k, l] for k in area_indices), name=f"f_sum_{b}_{l}")

            # Linearized tight McCormick formulations bounding force/geometry bilinear operations
            for k in area_indices:
                A = discrete_areas[k]
                stiffness = E * A / L
                f_min = stiffness * e_min
                f_max = stiffness * e_max
                
                model.addConstr(f_aux[b, k, l] >= f_min * x[b, k], name=f"fc1_{b}_{k}_{l}")
                model.addConstr(f_aux[b, k, l] <= f_max * x[b, k], name=f"fc2_{b}_{k}_{l}")
                model.addConstr(f_aux[b, k, l] >= stiffness * e[b, l] - f_max * (1 - x[b, k]), name=f"fc3_{b}_{k}_{l}")
                model.addConstr(f_aux[b, k, l] <= stiffness * e[b, l] - f_min * (1 - x[b, k]), name=f"fc4_{b}_{k}_{l}")

    # Enforce static equilibrium across all directional DOF dimensions
    for l in load_ids:
        for d in free_dofs:
            eq_expr = gp.LinExpr()
            for b in bar_ids:
                if (b, d) in C:
                    eq_expr += C[(b, d)] * f[b, l]
            model.addConstr(eq_expr == F_ext[(d, l)], name=f"eq_{d}_{l}")

    # Define Logging Callbacks
    def gurobi_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None:
                logger.log(obj_val)

    # Perform optimization
    model.optimize(gurobi_callback)

    # Assemble formatting dictionary
    results = {
        "objective_value": 0.0,
        "bar_areas": [],
        "displacements": [],
        "bar_forces": []
    }

    if model.SolCount > 0:
        results['objective_value'] = float(model.ObjVal)
        
        for b in bar_ids:
            best_k = 0
            max_bin_val = -1.0
            for k in area_indices:
                val = x[b, k].X
                if val > max_bin_val:
                    max_bin_val = val
                    best_k = k
                    
            results['bar_areas'].append({
                "bar_id": int(b),
                "area": float(discrete_areas[best_k]),
                "area_index": int(best_k)
            })

        for d in free_dofs:
            for l in load_ids:
                results['displacements'].append({
                    "dof_id": int(d),
                    "load": int(l),
                    "value": float(u[d, l].X)
                })

        for b in bar_ids:
            for l in load_ids:
                results['bar_forces'].append({
                    "bar_id": int(b),
                    "load": int(l),
                    "force": float(f[b, l].X)
                })
    else:
        # Fallback if unresolvable configuration
        pass

    with open(args.solution_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()