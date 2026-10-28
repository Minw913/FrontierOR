import math
import gurobipy as gp
from gurobipy import GRB

def solve_advanced_game(params):
    model = gp.Model("AdvancedGame")
    
    # Extract parameters
    N_D = params["N_D"]
    N_A = params["N_A"]
    N_u = params["N_u"]
    N_a = params["N_a"]
    T_u = params["T_u"]
    T_a = params["T_a"]
    R_dz = params["R_dz"]
    R_I = params["R_I"]
    M_dz = params["M_dz"]
    M_I = params["M_I"]
    M_u = params["M_u"]
    epsilon = params["epsilon"]
    defenders_initial = params["defenders_initial"]
    attackers_initial = params["attackers_initial"]
    
    # Precompute constants for defender dynamics
    exp_neg_Tu = math.exp(-T_u)
    one_minus_exp_neg_Tu = 1 - exp_neg_Tu
    Tu_minus_one_plus_exp = T_u - 1 + exp_neg_Tu
    
    # Precompute polygon normals
    dz_normals = [(math.cos(2 * math.pi * m / M_dz), math.sin(2 * math.pi * m / M_dz)) for m in range(M_dz)]
    I_normals = [(math.cos(2 * math.pi * m / M_I), math.sin(2 * math.pi * m / M_I)) for m in range(M_I)]
    u_normals = [(math.cos(2 * math.pi * m / M_u), math.sin(2 * math.pi * m / M_u)) for m in range(M_u)]
    u_rhs = math.cos(math.pi / M_u)
    
    # Estimate big M (conservative bound)
    max_defender_disp = max(max(abs(x0), abs(y0)) for x0, y0, _, _ in defenders_initial) + N_u * T_u
    max_attacker_disp = 0
    for px0, py0, vx, vy in attackers_initial:
        max_attacker_disp = max(max_attacker_disp, abs(px0) + N_a * T_a * abs(vx), abs(py0) + N_a * T_a * abs(vy))
    M_big = 2 * max(max_defender_disp, max_attacker_disp) + 100
    
    # Defender variables
    x = {}
    y = {}
    vx = {}
    vy = {}
    ux = {}
    uy = {}
    for i in range(N_D):
        for j in range(N_u + 1):
            x[i, j] = model.addVar(lb=-M_big, ub=M_big, name=f"x_{i}_{j}")
            y[i, j] = model.addVar(lb=-M_big, ub=M_big, name=f"y_{i}_{j}")
            vx[i, j] = model.addVar(lb=-M_big, ub=M_big, name=f"vx_{i}_{j}")
            vy[i, j] = model.addVar(lb=-M_big, ub=M_big, name=f"vy_{i}_{j}")
        for j in range(N_u):
            ux[i, j] = model.addVar(lb=-M_big, ub=M_big, name=f"ux_{i}_{j}")
            uy[i, j] = model.addVar(lb=-M_big, ub=M_big, name=f"uy_{i}_{j}")
    
    # Obstacle avoidance binary variables
    obs_bin = {}
    for i in range(N_D):
        for j in range(1, N_u + 1):
            for m in range(4):
                obs_bin[i, j, m] = model.addVar(vtype=GRB.BINARY, name=f"obs_bin_{i}_{j}_{m}")
    
    # Attacker variables
    p = {}
    q = {}
    a_active = {}
    gamma = {}
    d_a = {}
    for a in range(N_A):
        for k in range(N_a + 1):
            p[a, k] = model.addVar(lb=-M_big, ub=M_big, name=f"p_{a}_{k}")
            q[a, k] = model.addVar(lb=-M_big, ub=M_big, name=f"q_{a}_{k}")
            a_active[a, k] = model.addVar(vtype=GRB.BINARY, name=f"a_active_{a}_{k}")
            gamma[a, k] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{a}_{k}")
        for k in range(N_a + 1):
            d_a[a, k] = model.addVar(vtype=GRB.BINARY, name=f"d_a_{a}_{k}")
    
    # Interception variables
    delta = {}
    w_gamma = {}
    w_intercept = {}
    gamma_intercept = {}
    for a in range(N_A):
        for i in range(N_D):
            for k in range(N_a + 1):
                delta[a, i, k] = model.addVar(vtype=GRB.BINARY, name=f"delta_{a}_{i}_{k}")
                gamma_intercept[a, i, k] = model.addVar(vtype=GRB.BINARY, name=f"gamma_intercept_{a}_{i}_{k}")
                for m in range(M_I):
                    w_intercept[a, i, k, m] = model.addVar(vtype=GRB.BINARY, name=f"w_intercept_{a}_{i}_{k}_{m}")
        for k in range(N_a + 1):
            for m in range(M_dz):
                w_gamma[a, k, m] = model.addVar(vtype=GRB.BINARY, name=f"w_gamma_{a}_{k}_{m}")
    
    # Defender initial conditions
    for i, (x0, y0, vx0, vy0) in enumerate(defenders_initial):
        model.addConstr(x[i, 0] == x0, f"def_init_x_{i}")
        model.addConstr(y[i, 0] == y0, f"def_init_y_{i}")
        model.addConstr(vx[i, 0] == vx0, f"def_init_vx_{i}")
        model.addConstr(vy[i, 0] == vy0, f"def_init_vy_{i}")
    
    # Defender dynamics
    for i in range(N_D):
        for j in range(N_u):
            model.addConstr(vx[i, j+1] == vx[i, j] * exp_neg_Tu + ux[i, j] * one_minus_exp_neg_Tu, f"vx_dyn_{i}_{j}")
            model.addConstr(vy[i, j+1] == vy[i, j] * exp_neg_Tu + uy[i, j] * one_minus_exp_neg_Tu, f"vy_dyn_{i}_{j}")
            model.addConstr(x[i, j+1] == x[i, j] + vx[i, j] * one_minus_exp_neg_Tu + ux[i, j] * Tu_minus_one_plus_exp, f"x_dyn_{i}_{j}")
            model.addConstr(y[i, j+1] == y[i, j] + vy[i, j] * one_minus_exp_neg_Tu + uy[i, j] * Tu_minus_one_plus_exp, f"y_dyn_{i}_{j}")
    
    # Defender control constraints
    for i in range(N_D):
        for j in range(N_u):
            for m, (nx, ny) in enumerate(u_normals):
                model.addConstr(ux[i, j] * nx + uy[i, j] * ny <= u_rhs, f"control_poly_{i}_{j}_{m}")
    
    # Defender obstacle avoidance
    for i in range(N_D):
        for j in range(1, N_u + 1):
            model.addConstr(x[i, j] <= -R_dz + M_big * (1 - obs_bin[i, j, 0]), f"obs_x_low_{i}_{j}")
            model.addConstr(x[i, j] >= R_dz - M_big * (1 - obs_bin[i, j, 1]), f"obs_x_high_{i}_{j}")
            model.addConstr(y[i, j] <= -R_dz + M_big * (1 - obs_bin[i, j, 2]), f"obs_y_low_{i}_{j}")
            model.addConstr(y[i, j] >= R_dz - M_big * (1 - obs_bin[i, j, 3]), f"obs_y_high_{i}_{j}")
            model.addConstr(obs_bin[i, j, 0] + obs_bin[i, j, 1] + obs_bin[i, j, 2] + obs_bin[i, j, 3] >= 1, f"obs_or_{i}_{j}")
    
    # Attacker initial conditions
    for a, (px0, py0, vx, vy) in enumerate(attackers_initial):
        model.addConstr(p[a, 0] == px0, f"att_init_p_{a}")
        model.addConstr(q[a, 0] == py0, f"att_init_q_{a}")
        model.addConstr(a_active[a, 0] == 1, f"att_init_active_{a}")
    
    # Attacker movement and activity
    for a in range(N_A):
        vx_a = attackers_initial[a][2]
        vy_a = attackers_initial[a][3]
        for k in range(N_a):
            model.addConstr(p[a, k+1] == p[a, k] + vx_a * T_a * a_active[a, k], f"att_move_p_{a}_{k}")
            model.addConstr(q[a, k+1] == q[a, k] + vy_a * T_a * a_active[a, k], f"att_move_q_{a}_{k}")
            model.addConstr(a_active[a, k+1] <= a_active[a, k], f"active_cont1_{a}_{k}")
            model.addConstr(a_active[a, k+1] <= 1 - gamma[a, k], f"active_cont2_{a}_{k}")
            model.addConstr(a_active[a, k+1] <= 1 - d_a[a, k], f"active_cont3_{a}_{k}")
            model.addConstr(a_active[a, k+1] >= a_active[a, k] + (1 - gamma[a, k]) + (1 - d_a[a, k]) - 2, f"active_cont4_{a}_{k}")
    
    # d_a constraints (OR over defenders for interception)
    for a in range(N_A):
        for k in range(N_a + 1):
            for i in range(N_D):
                model.addConstr(d_a[a, k] >= delta[a, i, k], f"d_a_lb_{a}_{k}_{i}")
            model.addConstr(d_a[a, k] <= gp.quicksum(delta[a, i, k] for i in range(N_D)), f"d_a_ub_{a}_{k}")
    
    # Defense zone condition (gamma)
    for a in range(N_A):
        for k in range(N_a + 1):
            for m in range(M_dz):
                nx, ny = dz_normals[m]
                s = p[a, k] * nx + q[a, k] * ny
                model.addConstr(s <= R_dz + M_big * (1 - w_gamma[a, k, m]), f"dz_ub_{a}_{k}_{m}")
                model.addConstr(s >= R_dz - M_big * w_gamma[a, k, m], f"dz_lb_{a}_{k}_{m}")
            for m in range(M_dz):
                model.addConstr(gamma[a, k] <= w_gamma[a, k, m], f"gamma_ub_{a}_{k}_{m}")
            model.addConstr(gamma[a, k] >= 1 - M_dz + gp.quicksum(w_gamma[a, k, m] for m in range(M_dz)), f"gamma_lb_{a}_{k}")
    
    # Interception condition
    for a in range(N_A):
        for i in range(N_D):
            for k in range(N_a + 1):
                # Find control step index j for interpolation
                t_k = k * T_a
                j = int(t_k // T_u)
                h = t_k - j * T_u
                exp_neg_h = math.exp(-h)
                one_minus_exp_neg_h = 1 - exp_neg_h
                h_minus_one_plus_exp = h - 1 + exp_neg_h
                
                # Interpolated defender position
                x_interp = x[i, j] + vx[i, j] * one_minus_exp_neg_h + ux[i, j] * h_minus_one_plus_exp
                y_interp = y[i, j] + vy[i, j] * one_minus_exp_neg_h + uy[i, j] * h_minus_one_plus_exp
                
                # Intercept polygon constraints
                for m in range(M_I):
                    nx, ny = I_normals[m]
                    s = (p[a, k] - x_interp) * nx + (q[a, k] - y_interp) * ny
                    model.addConstr(s <= R_I + M_big * (1 - w_intercept[a, i, k, m]), f"intercept_ub_{a}_{i}_{k}_{m}")
                    model.addConstr(s >= R_I - M_big * w_intercept[a, i, k, m], f"intercept_lb_{a}_{i}_{k}_{m}")
                
                # AND over sides
                for m in range(M_I):
                    model.addConstr(gamma_intercept[a, i, k] <= w_intercept[a, i, k, m], f"intercept_and_ub_{a}_{i}_{k}_{m}")
                model.addConstr(gamma_intercept[a, i, k] >= 1 - M_I + gp.quicksum(w_intercept[a, i, k, m] for m in range(M_I)), f"intercept_and_lb_{a}_{i}_{k}")
                
                # Link to delta
                model.addConstr(delta[a, i, k] <= a_active[a, k], f"delta_act_{a}_{i}_{k}")
                model.addConstr(delta[a, i, k] <= gamma_intercept[a, i, k], f"delta_int_{a}_{i}_{k}")
                model.addConstr(delta[a, i, k] >= a_active[a, k] + gamma_intercept[a, i, k] - 1, f"delta_lb_{a}_{i}_{k}")
    
    # Objective
    total_penetration = gp.quicksum(gamma[a, k] for a in range(N_A) for k in range(N_a + 1))
    total_control_effort = gp.quicksum(gp.abs_(ux[i, j]) + gp.abs_(uy[i, j]) for i in range(N_D) for j in range(N_u))
    model.setObjective(total_penetration + epsilon * total_control_effort, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    # Extract solution
    solution = {
        "defenders": [],
        "attackers": [],
        "objective": model.objVal
    }
    
    # Defender trajectories
    for i in range(N_D):
        traj = []
        for j in range(N_u + 1):
            traj.append([x[i, j].X, y[i, j].X, vx[i, j].X, vy[i, j].X])
        solution["defenders"].append(traj)
    
    # Attacker trajectories and indicators
    for a in range(N_A):
        attacker_sol = {
            "trajectory": [],
            "active": [],
            "gamma": [],
            "delta": [[] for _ in range(N_D)]
        }
        for k in range(N_a + 1):
            attacker_sol["trajectory"].append([p[a, k].X, q[a, k].X])
            attacker_sol["active"].append(a_active[a, k].X)
            attacker_sol["gamma"].append(gamma[a, k].X)
            for i in range(N_D):
                attacker_sol["delta"][i].append(delta[a, i, k].X)
        solution["attackers"].append(attacker_sol)
    
    return solution