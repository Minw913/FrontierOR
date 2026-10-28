import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_instance(instance, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    E = instance["material_properties"]["modulus_of_elasticity"]
    cost_density = instance["material_properties"]["cost_density"]
    
    nodes = {n["node_id"]: n for n in instance["nodes"]}
    bars = instance["bars"]
    dofs = instance["degrees_of_freedom"]
    loads = instance["loading_conditions"]
    discrete_areas = instance["discrete_areas"]
    
    # Map DOFs
    dof_map = {}
    for i, d in enumerate(dofs):
        dof_map[d["dof_id"]] = i
        
    num_bars = len(bars)
    num_dofs = len(dofs)
    num_loads = len(loads)
    num_areas = len(discrete_areas)
    
    # Compute direction cosines
    bar_coeffs = np.zeros((num_bars, num_dofs))
    for b_idx, b in enumerate(bars):
        ni = nodes[b["node_i"]]
        nj = nodes[b["node_j"]]
        L = b["length"]
        dx = (nj["x"] - ni["x"]) / L
        dy = (nj["y"] - ni["y"]) / L
        dz = (nj.get("z", 0.0) - ni.get("z", 0.0)) / L
        c = {"x": dx, "y": dy, "z": dz}
        
        for d_idx, d in enumerate(dofs):
            if d["node"] == b["node_i"]:
                bar_coeffs[b_idx, d_idx] = -c[d["direction"]]
            elif d["node"] == b["node_j"]:
                bar_coeffs[b_idx, d_idx] = c[d["direction"]]
                
    # Stress bounds
    default_stress_lower = instance["stress_bounds"]["lower"]
    default_stress_upper = instance["stress_bounds"]["upper"]
    bar_stress_lower = np.full(num_bars, default_stress_lower)
    bar_stress_upper = np.full(num_bars, default_stress_upper)
    for bsb in instance.get("bar_specific_stress_bounds", []):
        b_idx = next(i for i, b in enumerate(bars) if b["bar_id"] == bsb["bar_id"])
        bar_stress_lower[b_idx] = bsb["lower"]
        bar_stress_upper[b_idx] = bsb["upper"]
        
    L_vec = np.array([b["length"] for b in bars])
    delta_min = bar_stress_lower * L_vec / E
    delta_max = bar_stress_upper * L_vec / E
    
    # Fallback FEM solver
    def evaluate_fallback(A_vec):
        K = bar_coeffs.T @ np.diag((E / L_vec) * A_vec) @ bar_coeffs
        F_mats = []
        for l_idx, load in enumerate(loads):
            F = np.zeros(num_dofs)
            for f in load["loads"]:
                F[dof_map[f["dof_id"]]] = f["force"]
            F_mats.append(F)
            
        u_mats, delta_mats, N_mats = [], [], []
        for F in F_mats:
            u = np.linalg.solve(K, F)
            delta = bar_coeffs @ u
            N = (E / L_vec) * A_vec * delta
            u_mats.append(u)
            delta_mats.append(delta)
            N_mats.append(N)
        return u_mats, delta_mats, N_mats

    # Create Gurobi model
    model = gp.Model("truss_opt")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", max(1, time_limit))
    
    # Variables
    x = model.addVars(num_bars, num_areas, vtype=GRB.BINARY, name="x")
    u = model.addVars(num_loads, num_dofs, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")
    delta = model.addVars(num_bars, num_loads, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="delta")
    N = model.addVars(num_bars, num_loads, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="N")
    y = model.addVars(num_bars, num_areas, num_loads, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")
    
    # MIP Start: all max area
    for b_idx in range(num_bars):
        x[b_idx, num_areas - 1].Start = 1.0
        for k_idx in range(num_areas - 1):
            x[b_idx, k_idx].Start = 0.0
            
    # Constraints
    # 1. Sum x = 1
    model.addConstrs((gp.quicksum(x[b_idx, k_idx] for k_idx in range(num_areas)) == 1 for b_idx in range(num_bars)), "sum_x")
    
    # 2. Displacement bounds
    db = instance["displacement_bounds"]
    db_lower = db.get("lower")
    db_upper = db.get("upper")
    for l_idx in range(num_loads):
        for d_idx in range(num_dofs):
            if db_lower is not None:
                u[l_idx, d_idx].lb = db_lower
            if db_upper is not None:
                u[l_idx, d_idx].ub = db_upper
                
    # 3. Compatibility
    for b_idx in range(num_bars):
        for l_idx in range(num_loads):
            model.addConstr(delta[b_idx, l_idx] == gp.quicksum(bar_coeffs[b_idx, d_idx] * u[l_idx, d_idx] for d_idx in range(num_dofs)))
            
    # 4. Linearization
    for b_idx in range(num_bars):
        dmin = delta_min[b_idx]
        dmax = delta_max[b_idx]
        for k_idx in range(num_areas):
            for l_idx in range(num_loads):
                model.addConstr(y[b_idx, k_idx, l_idx] <= dmax * x[b_idx, k_idx])
                model.addConstr(y[b_idx, k_idx, l_idx] >= dmin * x[b_idx, k_idx])
                model.addConstr(y[b_idx, k_idx, l_idx] <= delta[b_idx, l_idx] - dmin * (1 - x[b_idx, k_idx]))
                model.addConstr(y[b_idx, k_idx, l_idx] >= delta[b_idx, l_idx] - dmax * (1 - x[b_idx, k_idx]))
                
    # 5. Hooke's Law
    for b_idx in range(num_bars):
        L = L_vec[b_idx]
        for l_idx in range(num_loads):
            model.addConstr(N[b_idx, l_idx] == (E / L) * gp.quicksum(discrete_areas[k_idx] * y[b_idx, k_idx, l_idx] for k_idx in range(num_areas)))
            
    # 6. Equilibrium
    for l_idx, load in enumerate(loads):
        load_map = {}
        for f in load["loads"]:
            load_map[f["dof_id"]] = f["force"]
        for d_idx, d in enumerate(dofs):
            fid = d["dof_id"]
            F_ext = load_map.get(fid, 0.0)
            model.addConstr(gp.quicksum(bar_coeffs[b_idx, d_idx] * N[b_idx, l_idx] for b_idx in range(num_bars)) == F_ext)
            
    # 7. Linking groups
    for group in instance.get("linking_groups", []):
        bar_ids = group.get("bars", group.get("bar_ids", []))
        bar_indices = [i for i, b in enumerate(bars) if b["bar_id"] in bar_ids]
        if len(bar_indices) > 1:
            for k_idx in range(num_areas):
                for b_idx in bar_indices[1:]:
                    model.addConstr(x[bar_indices[0], k_idx] == x[b_idx, k_idx])
                    
    # Objective
    model.setObjective(gp.quicksum(cost_density * L_vec[b_idx] * discrete_areas[k_idx] * x[b_idx, k_idx] 
                                   for b_idx in range(num_bars) for k_idx in range(num_areas)), GRB.MINIMIZE)
                                   
    def gurobi_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = {
                "objective_value": obj_val,
                "bar_areas": [],
                "displacements": [],
                "bar_forces": []
            }
            x_vals = {k: model.cbGet(GRB.Callback.MIPSOL_X, x[k]) for k in x.keys()}
            u_vals = {k: model.cbGet(GRB.Callback.MIPSOL_X, u[k]) for k in u.keys()}
            N_vals = {k: model.cbGet(GRB.Callback.MIPSOL_X, N[k]) for k in N.keys()}
            
            for b_idx, b in enumerate(bars):
                area = 0
                area_idx = 0
                for k_idx in range(num_areas):
                    if x_vals[b_idx, k_idx] > 0.5:
                        area = discrete_areas[k_idx]
                        area_idx = k_idx
                        break
                sol["bar_areas"].append({
                    "bar_id": b["bar_id"],
                    "area": area,
                    "area_index": area_idx
                })
                
            for l_idx, load in enumerate(loads):
                for d_idx, d in enumerate(dofs):
                    sol["displacements"].append({
                        "dof_id": d["dof_id"],
                        "load": load["load_id"],
                        "value": u_vals[l_idx, d_idx]
                    })
                    
            for l_idx, load in enumerate(loads):
                for b_idx, b in enumerate(bars):
                    sol["bar_forces"].append({
                        "bar_id": b["bar_id"],
                        "load": load["load_id"],
                        "force": N_vals[b_idx, l_idx]
                    })
            if logger:
                logger.log_solution(obj_val, sol)
                
    model.optimize(gurobi_callback)
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution = {
            "objective_value": obj_val,
            "bar_areas": [],
            "displacements": [],
            "bar_forces": []
        }
        
        for b_idx, b in enumerate(bars):
            area = 0
            area_idx = 0
            for k_idx in range(num_areas):
                if x[b_idx, k_idx].X > 0.5:
                    area = discrete_areas[k_idx]
                    area_idx = k_idx
                    break
            solution["bar_areas"].append({
                "bar_id": b["bar_id"],
                "area": area,
                "area_index": area_idx
            })
            
        for l_idx, load in enumerate(loads):
            for d_idx, d in enumerate(dofs):
                solution["displacements"].append({
                    "dof_id": d["dof_id"],
                    "load": load["load_id"],
                    "value": u[l_idx, d_idx].X
                })
                
        for l_idx, load in enumerate(loads):
            for b_idx, b in enumerate(bars):
                solution["bar_forces"].append({
                    "bar_id": b["bar_id"],
                    "load": load["load_id"],
                    "force": N[b_idx, l_idx].X
                })
    else:
        # Fallback to max area
        A_max = discrete_areas[-1]
        A_vec = np.full(num_bars, A_max)
        u_mats, delta_mats, N_mats = evaluate_fallback(A_vec)
        obj_val = sum(cost_density * L_vec[b_idx] * A_max for b_idx in range(num_bars))
        solution = {
            "objective_value": obj_val,
            "bar_areas": [{"bar_id": b["bar_id"], "area": A_max, "area_index": num_areas - 1} for b in bars],
            "displacements": [],
            "bar_forces": []
        }
        for l_idx, load in enumerate(loads):
            for d_idx, d in enumerate(dofs):
                solution["displacements"].append({
                    "dof_id": d["dof_id"],
                    "load": load["load_id"],
                    "value": u_mats[l_idx][d_idx]
                })
        for l_idx, load in enumerate(loads):
            for b_idx, b in enumerate(bars):
                solution["bar_forces"].append({
                    "bar_id": b["bar_id"],
                    "load": load["load_id"],
                    "force": N_mats[l_idx][b_idx]
                })
        if logger:
            logger.log_solution(obj_val, solution)
                
    return obj_val, solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    obj_val, solution = solve_instance(instance, args.time_limit, args.log_path)
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()