import json
import argparse
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    N_D = inst['N_D']
    N_A = inst['N_A']
    epsilon = inst['epsilon']
    M_u = inst['M_u']
    M_I = inst['M_I']
    M_dz = inst['M_dz']
    N_u = inst['N_u']
    N_a = inst['N_a']
    R_dz = inst['R_dz']
    R_I = inst['R_I']
    T_a = inst['T_a']
    T_u = inst['T_u']

    defenders = inst['defenders']
    attackers = inst['attackers']

    # Precompute constants
    e_Tu = math.exp(-T_u)
    coeff_vel = 1 - e_Tu
    coeff_pos = T_u - 1 + e_Tu

    # Polygon angles
    angles_u = [2 * math.pi * m / M_u for m in range(M_u)]
    angles_I = [2 * math.pi * l / M_I for l in range(M_I)]
    angles_dz = [2 * math.pi * m / M_dz for m in range(M_dz)]

    # Big-M calculation
    max_dist = 0
    for d in defenders:
        max_dist = max(max_dist, abs(d['x_s']), abs(d['y_s']))
    for a in attackers:
        max_dist = max(max_dist, abs(a['p_s']), abs(a['q_s']))
        max_dist = max(max_dist, abs(a['p_s']) + abs(a['v_p'] * T_a * N_a), abs(a['q_s']) + abs(a['v_q'] * T_a * N_a))
    max_dist += N_u * T_u * 2  # rough upper bound for defender movement
    M_big = 2 * max_dist + 2 * R_dz + 2 * R_I + 10

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("robot_defense")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Variables
    # Defenders
    x = model.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, name="x")
    y = model.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, name="y")
    xdot = model.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, name="xdot")
    ydot = model.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, name="ydot")
    ux_p = model.addVars(N_D, N_u, lb=0, name="ux_p")
    ux_m = model.addVars(N_D, N_u, lb=0, name="ux_m")
    uy_p = model.addVars(N_D, N_u, lb=0, name="uy_p")
    uy_m = model.addVars(N_D, N_u, lb=0, name="uy_m")

    # Attackers
    p = model.addVars(N_A, N_a + 1, lb=-GRB.INFINITY, name="p")
    q = model.addVars(N_A, N_a + 1, lb=-GRB.INFINITY, name="q")
    a = model.addVars(N_A, N_a + 1, vtype=GRB.BINARY, name="a")
    gamma = model.addVars(N_A, N_a + 1, vtype=GRB.BINARY, name="gamma")
    beta_dz = model.addVars(N_A, N_a + 1, M_dz, vtype=GRB.BINARY, name="beta_dz")
    
    # Interception
    delta = model.addVars(N_D, N_A, N_a + 1, vtype=GRB.BINARY, name="delta")
    beta_I = model.addVars(N_D, N_A, N_a + 1, M_I, vtype=GRB.BINARY, name="beta_I")

    # Obstacle avoidance
    o = model.addVars(N_D, N_u + 1, 4, vtype=GRB.BINARY, name="o")

    # Objective
    obj = gp.LinExpr()
    for j in range(N_A):
        for k in range(1, N_a + 1):
            obj += gamma[j, k]
    for i in range(N_D):
        for m in range(N_u):
            obj += epsilon * (ux_p[i, m] + ux_m[i, m] + uy_p[i, m] + uy_m[i, m])
    model.setObjective(obj, GRB.MINIMIZE)

    # Defenders dynamics
    for i in range(N_D):
        model.addConstr(x[i, 0] == defenders[i]['x_s'])
        model.addConstr(y[i, 0] == defenders[i]['y_s'])
        model.addConstr(xdot[i, 0] == defenders[i]['xdot_s'])
        model.addConstr(ydot[i, 0] == defenders[i]['ydot_s'])
        
        for m in range(N_u):
            ux = ux_p[i, m] - ux_m[i, m]
            uy = uy_p[i, m] - uy_m[i, m]
            model.addConstr(xdot[i, m+1] == xdot[i, m] * e_Tu + ux * coeff_vel)
            model.addConstr(ydot[i, m+1] == ydot[i, m] * e_Tu + uy * coeff_vel)
            model.addConstr(x[i, m+1] == x[i, m] + xdot[i, m] * coeff_vel + ux * coeff_pos)
            model.addConstr(y[i, m+1] == y[i, m] + ydot[i, m] * coeff_vel + uy * coeff_pos)
            
            # Control input constraints
            for mm in range(M_u):
                theta = angles_u[mm]
                ct = math.cos(theta)
                st = math.sin(theta)
                model.addConstr(ux * ct + uy * st <= math.cos(math.pi / M_u))

    # Defender obstacle avoidance
    for i in range(N_D):
        for m in range(1, N_u + 1):
            model.addConstr(o[i, m, 0] + o[i, m, 1] + o[i, m, 2] + o[i, m, 3] >= 1)
            model.addConstr(x[i, m] >= R_dz - M_big * (1 - o[i, m, 0]))
            model.addConstr(x[i, m] <= -R_dz + M_big * (1 - o[i, m, 1]))
            model.addConstr(y[i, m] >= R_dz - M_big * (1 - o[i, m, 2]))
            model.addConstr(y[i, m] <= -R_dz + M_big * (1 - o[i, m, 3]))

    # Attackers dynamics
    for j in range(N_A):
        model.addConstr(p[j, 0] == attackers[j]['p_s'])
        model.addConstr(q[j, 0] == attackers[j]['q_s'])
        model.addConstr(a[j, 0] == 1)
        
        for k in range(N_a + 1):
            # Zone check
            for m in range(M_dz):
                theta = angles_dz[m]
                ct = math.cos(theta)
                st = math.sin(theta)
                model.addConstr(p[j, k] * ct + q[j, k] * st <= R_dz + M_big * beta_dz[j, k, m])
                model.addConstr(p[j, k] * ct + q[j, k] * st >= R_dz - M_big * (1 - beta_dz[j, k, m]))
            model.addConstrs((gamma[j, k] <= 1 - beta_dz[j, k, m] for m in range(M_dz)), name=f"gamma_ub_{j}_{k}")
            model.addConstr(gamma[j, k] >= 1 - gp.quicksum(beta_dz[j, k, m] for m in range(M_dz)))

            # Interception check
            for i in range(N_D):
                # Interpolate defender position at time t_k
                t_k = k * T_a
                if t_k >= N_u * T_u:
                    X_ik = x[i, N_u]
                    Y_ik = y[i, N_u]
                else:
                    m_int = int(math.floor(t_k / T_u))
                    dt = t_k - m_int * T_u
                    e_dt = math.exp(-dt)
                    coeff_v = 1 - e_dt
                    coeff_p = dt - 1 + e_dt
                    X_ik = x[i, m_int] + xdot[i, m_int] * coeff_v + (ux_p[i, m_int] - ux_m[i, m_int]) * coeff_p
                    Y_ik = y[i, m_int] + ydot[i, m_int] * coeff_v + (uy_p[i, m_int] - uy_m[i, m_int]) * coeff_p
                
                for l in range(M_I):
                    theta = angles_I[l]
                    ct = math.cos(theta)
                    st = math.sin(theta)
                    rx = p[j, k] - X_ik
                    ry = q[j, k] - Y_ik
                    model.addConstr(rx * ct + ry * st <= R_I + M_big * beta_I[i, j, k, l])
                    model.addConstr(rx * ct + ry * st >= R_I - M_big * (1 - beta_I[i, j, k, l]))
                model.addConstrs((delta[i, j, k] <= 1 - beta_I[i, j, k, l] for l in range(M_I)), name=f"delta_ub_{i}_{j}_{k}")
                model.addConstr(delta[i, j, k] >= 1 - gp.quicksum(beta_I[i, j, k, l] for l in range(M_I)))

            # Attacker active flag logic
            if k < N_a:
                model.addConstr(p[j, k+1] == p[j, k] + attackers[j]['v_p'] * T_a * a[j, k])
                model.addConstr(q[j, k+1] == q[j, k] + attackers[j]['v_q'] * T_a * a[j, k])
                model.addConstr(a[j, k+1] <= a[j, k])
                model.addConstr(a[j, k+1] <= 1 - gamma[j, k])
                for i in range(N_D):
                    model.addConstr(a[j, k+1] <= 1 - delta[i, j, k])
                model.addConstr(a[j, k+1] >= a[j, k] - gamma[j, k] - gp.quicksum(delta[i, j, k] for i in range(N_D)))

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Extract solution
                sol = {
                    "objective_value": obj_val,
                    "defenders": [],
                    "attackers": []
                }
                for i in range(N_D):
                    x_vals = model.cbGetSolution([x[i, m] for m in range(N_u + 1)])
                    y_vals = model.cbGetSolution([y[i, m] for m in range(N_u + 1)])
                    xd_vals = model.cbGetSolution([xdot[i, m] for m in range(N_u + 1)])
                    yd_vals = model.cbGetSolution([ydot[i, m] for m in range(N_u + 1)])
                    ux_p_vals = model.cbGetSolution([ux_p[i, m] for m in range(N_u)])
                    ux_m_vals = model.cbGetSolution([ux_m[i, m] for m in range(N_u)])
                    ux_vals = [ux_p_vals[m] - ux_m_vals[m] for m in range(N_u)]
                    uy_p_vals = model.cbGetSolution([uy_p[i, m] for m in range(N_u)])
                    uy_m_vals = model.cbGetSolution([uy_m[i, m] for m in range(N_u)])
                    uy_vals = [uy_p_vals[m] - uy_m_vals[m] for m in range(N_u)]
                    sol["defenders"].append({
                        "x": x_vals,
                        "y": y_vals,
                        "xdot": xd_vals,
                        "ydot": yd_vals,
                        "ux": ux_vals,
                        "uy": uy_vals
                    })
                for j in range(N_A):
                    p_vals = model.cbGetSolution([p[j, k] for k in range(N_a + 1)])
                    q_vals = model.cbGetSolution([q[j, k] for k in range(N_a + 1)])
                    a_vals = model.cbGetSolution([a[j, k] for k in range(N_a + 1)])
                    g_vals = model.cbGetSolution([gamma[j, k] for k in range(N_a + 1)])
                    d_vals = []
                    for i in range(N_D):
                        d_vals.append(model.cbGetSolution([delta[i, j, k] for k in range(N_a + 1)]))
                    sol["attackers"].append({
                        "p": p_vals,
                        "q": q_vals,
                        "a": a_vals,
                        "gamma": g_vals,
                        "delta": d_vals
                    })
                logger.log_solution(obj_val, sol)
    
    if logger:
        model.optimize(callback)
    else:
        model.optimize()

    # Extract final solution
    if model.SolCount == 0:
        # Output empty if no solution found
        solution = {
            "objective_value": 0.0,
            "defenders": [],
            "attackers": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    obj_val = model.ObjVal
    solution = {
        "objective_value": obj_val,
        "defenders": [],
        "attackers": []
    }

    for i in range(N_D):
        def_dict = {
            "x": [x[i, m].X for m in range(N_u + 1)],
            "y": [y[i, m].X for m in range(N_u + 1)],
            "xdot": [xdot[i, m].X for m in range(N_u + 1)],
            "ydot": [ydot[i, m].X for m in range(N_u + 1)],
            "ux": [ux_p[i, m].X - ux_m[i, m].X for m in range(N_u)],
            "uy": [uy_p[i, m].X - uy_m[i, m].X for m in range(N_u)]
        }
        solution["defenders"].append(def_dict)

    for j in range(N_A):
        delta_list = [[delta[i, j, k].X for k in range(N_a + 1)] for i in range(N_D)]
        att_dict = {
            "p": [p[j, k].X for k in range(N_a + 1)],
            "q": [q[j, k].X for k in range(N_a + 1)],
            "a": [a[j, k].X for k in range(N_a + 1)],
            "gamma": [gamma[j, k].X for k in range(N_a + 1)],
            "delta": delta_list
        }
        solution["attackers"].append(att_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()