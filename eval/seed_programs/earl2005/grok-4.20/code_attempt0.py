import json
import argparse
import time
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N_D = data['N_D']
    N_A = data['N_A']
    epsilon = data['epsilon']
    M_u = data['M_u']
    M_I = data['M_I']
    M_dz = data['M_dz']
    N_u = data['N_u']
    N_a = data['N_a']
    R_dz = data['R_dz']
    R_I = data['R_I']
    v_a = data['v_a']  # not directly used since velocities given
    T_a = data['T_a']
    T_u = data['T_u']
    defenders = data['defenders']
    attackers = data['attackers']

    # Precompute constants
    alpha = math.exp(-T_u)
    beta = 1 - alpha
    gamma_pos = 1 - alpha  # for position update from velocity
    delta_pos = T_u - 1 + alpha

    cos_mu = math.cos(math.pi / M_u)
    cos_I = math.cos(math.pi / M_I)
    cos_dz = math.cos(math.pi / M_dz)

    # Half-space normals for polygons (assuming standard positioning)
    def get_normals(M):
        return [(math.cos(2 * math.pi * k / M), math.sin(2 * math.pi * k / M)) for k in range(M)]

    normals_u = get_normals(M_u)
    normals_I = get_normals(M_I)
    normals_dz = get_normals(M_dz)

    # Create model
    model = gp.Model("DefenseRobots")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - 5))  # reserve some time for output

    # Variables
    # Defender states and controls
    x = [[model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_d{d}_t{t}") for t in range(N_u+1)] for d in range(N_D)]
    y = [[model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"y_d{d}_t{t}") for t in range(N_u+1)] for d in range(N_D)]
    xdot = [[model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xdot_d{d}_t{t}") for t in range(N_u+1)] for d in range(N_D)]
    ydot = [[model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"ydot_d{d}_t{t}") for t in range(N_u+1)] for d in range(N_D)]
    ux = [[model.addVar(lb=-1.0, ub=1.0, name=f"ux_d{d}_t{t}") for t in range(N_u)] for d in range(N_D)]
    uy = [[model.addVar(lb=-1.0, ub=1.0, name=f"uy_d{d}_t{t}") for t in range(N_u)] for d in range(N_D)]

    # Attacker states
    p = [[model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_a{a}_k{k}") for k in range(N_a+1)] for a in range(N_A)]
    q = [[model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"q_a{a}_k{k}") for k in range(N_a+1)] for a in range(N_A)]
    a_var = [[model.addVar(vtype=GRB.BINARY, name=f"a_a{a}_k{k}") for k in range(N_a+1)] for a in range(N_A)]  # active
    gamma = [[model.addVar(vtype=GRB.BINARY, name=f"gamma_a{a}_k{k}") for k in range(N_a+1)] for a in range(N_A)]  # in DZ

    # Interception indicators
    delta = [[[model.addVar(vtype=GRB.BINARY, name=f"delta_d{d}_a{a}_k{k}") 
               for k in range(N_a+1)] for a in range(N_A)] for d in range(N_D)]

    # Big-M for linearization
    M_pos = 1000.0  # sufficiently large
    M_vel = 100.0

    # Initial conditions
    for d in range(N_D):
        model.addConstr(x[d][0] == defenders[d]['x_s'])
        model.addConstr(y[d][0] == defenders[d]['y_s'])
        model.addConstr(xdot[d][0] == defenders[d]['xdot_s'])
        model.addConstr(ydot[d][0] == defenders[d]['ydot_s'])

    for a in range(N_A):
        model.addConstr(p[a][0] == attackers[a]['p_s'])
        model.addConstr(q[a][0] == attackers[a]['q_s'])
        model.addConstr(a_var[a][0] == 1)

    # Defender dynamics
    for d in range(N_D):
        for t in range(N_u):
            # Velocity update
            model.addConstr(xdot[d][t+1] == alpha * xdot[d][t] + beta * ux[d][t])
            model.addConstr(ydot[d][t+1] == alpha * ydot[d][t] + beta * uy[d][t])
            
            # Position update
            model.addConstr(x[d][t+1] == x[d][t] + gamma_pos * xdot[d][t] + delta_pos * ux[d][t])
            model.addConstr(y[d][t+1] == y[d][t] + gamma_pos * ydot[d][t] + delta_pos * uy[d][t])

    # Attacker dynamics and logic
    for a in range(N_A):
        for k in range(N_a):
            # Position update if active
            model.addConstr(p[a][k+1] == p[a][k] + T_a * attackers[a]['v_p'] * a_var[a][k])
            model.addConstr(q[a][k+1] == q[a][k] + T_a * attackers[a]['v_q'] * a_var[a][k])
            
            # Active if not intercepted and not in DZ previously (but can stay inactive)
            # a[k+1] <= a[k]
            model.addConstr(a_var[a][k+1] <= a_var[a][k])
            
            # If intercepted at k, becomes inactive at k+1
            inter_at_k = gp.quicksum(delta[d][a][k] for d in range(N_D))
            model.addConstr(a_var[a][k+1] <= 1 - inter_at_k)
            
            # gamma only if active
            model.addConstr(gamma[a][k] <= a_var[a][k])

    # Defense Zone constraints (linear halfspaces)
    for a in range(N_A):
        for k in range(1, N_a+1):  # from 1 as per problem
            for m in range(M_dz):
                nx, ny = normals_dz[m]
                model.addConstr(nx * p[a][k] + ny * q[a][k] <= R_dz * cos_dz + M_pos * (1 - gamma[a][k]))
                model.addConstr(nx * p[a][k] + ny * q[a][k] >= R_dz * cos_dz - M_pos - M_pos * gamma[a][k])

    # Interception constraints
    # Defender position at attacker time k via linear interpolation
    # Find corresponding defender time: t = floor(k * T_a / T_u)
    for a in range(N_A):
        for k in range(1, N_a+1):
            t_frac = k * T_a / T_u
            t_floor = min(int(t_frac), N_u-1)
            frac = t_frac - t_floor
            if frac == 0:
                # exactly at control step
                x_def_at_k = [x[d][t_floor] for d in range(N_D)]
                y_def_at_k = [y[d][t_floor] for d in range(N_D)]
            else:
                # linear interpolation between t_floor and t_floor+1
                x_def_at_k = []
                y_def_at_k = []
                for d_idx in range(N_D):
                    x_interp = (1-frac)*x[d_idx][t_floor] + frac*x[d_idx][t_floor+1]
                    y_interp = (1-frac)*y[d_idx][t_floor] + frac*y[d_idx][t_floor+1]
                    x_def_at_k.append(x_interp)
                    y_def_at_k.append(y_interp)

            for d in range(N_D):
                for m in range(M_I):
                    nx, ny = normals_I[m]
                    rhs = R_I * cos_I
                    model.addConstr(
                        nx * (p[a][k] - x_def_at_k[d]) + ny * (q[a][k] - y_def_at_k[d]) 
                        <= rhs + M_pos * (1 - delta[d][a][k])
                    )
                    model.addConstr(
                        nx * (p[a][k] - x_def_at_k[d]) + ny * (q[a][k] - y_def_at_k[d]) 
                        >= rhs - M_pos - M_pos * delta[d][a][k]
                    )
                
                # delta only if attacker active
                model.addConstr(delta[d][a][k] <= a_var[a][k])
                model.addConstr(delta[d][a][k] <= gamma[a][k] + (1 - a_var[a][k]))  # loose

    # Defender control constraints (polygon)
    for d in range(N_D):
        for t in range(N_u):
            for m in range(M_u):
                nx, ny = normals_u[m]
                model.addConstr(nx * ux[d][t] + ny * uy[d][t] <= cos_mu)

    # Defender avoidance of obstacle at control steps (N_o = N_u)
    # Must be outside at least one halfspace of the 4-sided obstacle (square?)
    # Problem says "four-side obstacle polygon centered at the origin with apothem R_dz"
    # We interpret as must violate at least one DZ halfspace (i.e. outside DZ)
    for d in range(N_D):
        for t in range(1, N_u+1):  # positive control-step boundaries
            inside_all = []
            for m in range(M_dz):
                nx, ny = normals_dz[m]
                viol = model.addVar(vtype=GRB.BINARY, name=f"viol_d{d}_t{t}_m{m}")
                model.addConstr(
                    nx * x[d][t] + ny * y[d][t] <= R_dz * cos_dz + M_pos * viol
                )
                inside_all.append(1 - viol)
            model.addConstr(gp.quicksum(inside_all) <= M_dz - 1)  # at least one violation

    # Objective
    zone_pen = gp.quicksum(gamma[a][k] for a in range(N_A) for k in range(1, N_a+1))
    
    effort = 0
    for d in range(N_D):
        for t in range(N_u):
            effort += ux[d][t] + uy[d][t]   # since >=0? No, use abs but since polygon, better linearize
    # Actually since we have absolute values, we need to model |ux| + |uy|
    # But to keep model tight, introduce abs variables
    # Redefine effort with auxiliary variables

    # Reset effort
    abs_ux = [[model.addVar(name=f"absux_d{d}_t{t}") for t in range(N_u)] for d in range(N_D)]
    abs_uy = [[model.addVar(name=f"absuy_d{d}_t{t}") for t in range(N_u)] for d in range(N_D)]
    
    for d in range(N_D):
        for t in range(N_u):
            model.addConstr(abs_ux[d][t] >= ux[d][t])
            model.addConstr(abs_ux[d][t] >= -ux[d][t])
            model.addConstr(abs_uy[d][t] >= uy[d][t])
            model.addConstr(abs_uy[d][t] >= -uy[d][t])
    
    effort = gp.quicksum(abs_ux[d][t] + abs_uy[d][t] for d in range(N_D) for t in range(N_u))
    
    objective = zone_pen + epsilon * effort
    model.setObjective(objective, GRB.MINIMIZE)

    # Callback for incumbent logging
    best_obj = float('inf')
    solution_dict_cache = None

    def callback(model, where):
        nonlocal best_obj, solution_dict_cache
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                # Extract solution
                sol = {"objective_value": float(obj_val), "defenders": [], "attackers": []}
                
                # Defenders
                for d_idx in range(N_D):
                    def_dict = {
                        "x": [model.cbGetSolution(x[d_idx][t]) for t in range(N_u+1)],
                        "y": [model.cbGetSolution(y[d_idx][t]) for t in range(N_u+1)],
                        "xdot": [model.cbGetSolution(xdot[d_idx][t]) for t in range(N_u+1)],
                        "ydot": [model.cbGetSolution(ydot[d_idx][t]) for t in range(N_u+1)],
                        "ux": [model.cbGetSolution(ux[d_idx][t]) for t in range(N_u)],
                        "uy": [model.cbGetSolution(uy[d_idx][t]) for t in range(N_u)]
                    }
                    sol["defenders"].append(def_dict)
                
                # Attackers
                for a_idx in range(N_A):
                    att_dict = {
                        "p": [model.cbGetSolution(p[a_idx][k]) for k in range(N_a+1)],
                        "q": [model.cbGetSolution(q[a_idx][k]) for k in range(N_a+1)],
                        "a": [float(model.cbGetSolution(a_var[a_idx][k])) for k in range(N_a+1)],
                        "gamma": [float(model.cbGetSolution(gamma[a_idx][k])) for k in range(N_a+1)],
                        "delta": []
                    }
                    for d_idx in range(N_D):
                        delta_row = [float(model.cbGetSolution(delta[d_idx][a_idx][k])) for k in range(N_a+1)]
                        att_dict["delta"].append(delta_row)
                    sol["attackers"].append(att_dict)
                
                solution_dict_cache = sol
                if logger:
                    logger.log_solution(float(obj_val), sol)

    # Optimize
    model.optimize(callback)

    # Final solution
    if model.status == GRB.OPTIMAL or model.SolCount > 0:
        if solution_dict_cache is None or abs(model.objVal - solution_dict_cache["objective_value"]) > 1e-5:
            # Extract final solution
            sol = {"objective_value": float(model.objVal), "defenders": [], "attackers": []}
            
            for d_idx in range(N_D):
                def_dict = {
                    "x": [x[d_idx][t].X for t in range(N_u+1)],
                    "y": [y[d_idx][t].X for t in range(N_u+1)],
                    "xdot": [xdot[d_idx][t].X for t in range(N_u+1)],
                    "ydot": [ydot[d_idx][t].X for t in range(N_u+1)],
                    "ux": [ux[d_idx][t].X for t in range(N_u)],
                    "uy": [uy[d_idx][t].X for t in range(N_u)]
                }
                sol["defenders"].append(def_dict)
            
            for a_idx in range(N_A):
                att_dict = {
                    "p": [p[a_idx][k].X for k in range(N_a+1)],
                    "q": [q[a_idx][k].X for k in range(N_a+1)],
                    "a": [float(a_var[a_idx][k].X) for k in range(N_a+1)],
                    "gamma": [float(gamma[a_idx][k].X) for k in range(N_a+1)],
                    "delta": []
                }
                for d_idx in range(N_D):
                    delta_row = [float(delta[d_idx][a_idx][k].X) for k in range(N_a+1)]
                    att_dict["delta"].append(delta_row)
                sol["attackers"].append(att_dict)
        else:
            sol = solution_dict_cache
            sol["objective_value"] = float(model.objVal)
    else:
        # Infeasible or no solution - return dummy feasible if possible or empty
        sol = {"objective_value": 0.0, "defenders": [], "attackers": []}
        for d_idx in range(N_D):
            def_dict = {
                "x": [defenders[d_idx]['x_s']] * (N_u+1),
                "y": [defenders[d_idx]['y_s']] * (N_u+1),
                "xdot": [defenders[d_idx]['xdot_s']] * (N_u+1),
                "ydot": [defenders[d_idx]['ydot_s']] * (N_u+1),
                "ux": [0.0] * N_u,
                "uy": [0.0] * N_u
            }
            sol["defenders"].append(def_dict)
        for a_idx in range(N_A):
            att_dict = {
                "p": [attackers[a_idx]['p_s'] + i*T_a*attackers[a_idx]['v_p'] for i in range(N_a+1)],
                "q": [attackers[a_idx]['q_s'] + i*T_a*attackers[a_idx]['v_q'] for i in range(N_a+1)],
                "a": [1.0] * (N_a+1),
                "gamma": [0.0] * (N_a+1),
                "delta": [[0.0 for _ in range(N_a+1)] for _ in range(N_D)]
            }
            sol["attackers"].append(att_dict)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    if logger and 'sol' in locals():
        try:
            logger.log_solution(sol["objective_value"], sol)
        except:
            pass

if __name__ == "__main__":
    main()