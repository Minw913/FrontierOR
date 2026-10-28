import json
import argparse
import numpy as np
import math
import time
from itertools import product

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

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
    defenders_data = inst['defenders']
    attackers_data = inst['attackers']

    # Precompute defense zone half-spaces
    # Regular polygon with M_dz sides, apothem R_dz
    # Half-space k: n_k . p <= R_dz where n_k = (cos(2*pi*k/M_dz), sin(2*pi*k/M_dz))
    dz_normals = []
    for k in range(M_dz):
        angle = 2 * math.pi * k / M_dz
        dz_normals.append((math.cos(angle), math.sin(angle)))

    def in_defense_zone(px, py):
        for nx, ny in dz_normals:
            if nx * px + ny * py > R_dz + 1e-9:
                return False
        return True

    # Intercept polygon half-spaces
    int_normals = []
    for k in range(M_I):
        angle = 2 * math.pi * k / M_I
        int_normals.append((math.cos(angle), math.sin(angle)))

    def in_intercept_zone(dx, dy):
        for nx, ny in int_normals:
            if nx * dx + ny * dy > R_I + 1e-9:
                return False
        return True

    # Control input polygon: M_u sides, inscribed in unit disk, apothem cos(pi/M_u)
    ctrl_normals = []
    ctrl_rhs = math.cos(math.pi / M_u)
    for k in range(M_u):
        angle = 2 * math.pi * k / M_u
        ctrl_normals.append((math.cos(angle), math.sin(angle)))

    def control_feasible(ux, uy):
        for nx, ny in ctrl_normals:
            if nx * ux + ny * uy > ctrl_rhs + 1e-9:
                return False
        return True

    # Obstacle avoidance: 4-side polygon centered at origin, apothem R_dz
    # Defender must be OUTSIDE at least one half-space
    obs_normals = []
    for k in range(4):
        angle = 2 * math.pi * k / 4
        obs_normals.append((math.cos(angle), math.sin(angle)))

    def outside_obstacle(px, py):
        # Outside if at least one half-space is violated
        for nx, ny in obs_normals:
            if nx * px + ny * py > R_dz + 1e-9:
                return True
        return False

    # Dynamics
    exp_Tu = math.exp(-T_u)
    A_pos = 1 - exp_Tu  # coefficient of v in position update
    B_pos = T_u - 1 + exp_Tu  # coefficient of u in position update
    A_vel = exp_Tu  # coefficient of v in velocity update
    B_vel = 1 - exp_Tu  # coefficient of u in velocity update

    def simulate_defender(x0, y0, vx0, vy0, ux_list, uy_list):
        """Simulate defender trajectory given control inputs."""
        xs = [x0]
        ys = [y0]
        vxs = [vx0]
        vys = [vy0]
        for k in range(N_u):
            x_new = xs[-1] + A_pos * vxs[-1] + B_pos * ux_list[k]
            y_new = ys[-1] + A_pos * vys[-1] + B_pos * uy_list[k]
            vx_new = A_vel * vxs[-1] + B_vel * ux_list[k]
            vy_new = A_vel * vys[-1] + B_vel * uy_list[k]
            xs.append(x_new)
            ys.append(y_new)
            vxs.append(vx_new)
            vys.append(vy_new)
        return xs, ys, vxs, vys

    def interpolate_defender_at_time(t, xs, ys, vxs, vys, ux_list, uy_list):
        """Get defender position at arbitrary time t by finding which control interval it's in
        and using continuous dynamics."""
        # Control step boundaries are at times 0, T_u, 2*T_u, ..., N_u*T_u
        k = int(t / T_u)
        if k >= N_u:
            k = N_u - 1
        tau = t - k * T_u
        if tau < -1e-12:
            tau = 0
        if tau > T_u + 1e-12:
            tau = T_u

        # Within interval [k*T_u, (k+1)*T_u], control is ux_list[k], uy_list[k]
        # State at start of interval: xs[k], ys[k], vxs[k], vys[k]
        exp_tau = math.exp(-tau)
        x_t = xs[k] + (1 - exp_tau) * vxs[k] + (tau - 1 + exp_tau) * ux_list[k]
        y_t = ys[k] + (1 - exp_tau) * vys[k] + (tau - 1 + exp_tau) * uy_list[k]
        return x_t, y_t

    # Compute attacker trajectories (without interception)
    attacker_trajs = []
    for j in range(N_A):
        a = attackers_data[j]
        ps = [a['p_s']]
        qs = [a['q_s']]
        for k in range(N_a):
            ps.append(ps[-1] + a['v_p'] * T_a)
            qs.append(qs[-1] + a['v_q'] * T_a)
        attacker_trajs.append((ps, qs))

    # Determine when each attacker enters defense zone (if no interception)
    attacker_enters_dz = []
    for j in range(N_A):
        ps, qs = attacker_trajs[j]
        enter_step = None
        for k in range(N_a + 1):
            if in_defense_zone(ps[k], qs[k]):
                enter_step = k
                break
        attacker_enters_dz.append(enter_step)

    # Count steps in DZ if never intercepted
    attacker_dz_steps_no_intercept = []
    for j in range(N_A):
        ps, qs = attacker_trajs[j]
        count = 0
        for k in range(1, N_a + 1):
            if in_defense_zone(ps[k], qs[k]):
                count += 1
        attacker_dz_steps_no_intercept.append(count)

    # For each attacker, compute the "value" of intercepting it before step k
    # = number of DZ steps from k to N_a
    def dz_steps_from(j, k_intercept):
        """Number of DZ steps saved if attacker j is intercepted at step k_intercept."""
        ps, qs = attacker_trajs[j]
        count = 0
        for k in range(k_intercept, N_a + 1):
            if in_defense_zone(ps[k], qs[k]):
                count += 1
        return count

    # Try to use Gurobi MIP
    try:
        import gurobipy as gp
        from gurobipy import GRB
        use_gurobi = True
    except:
        use_gurobi = False

    remaining_time = args.time_limit - (time.time() - start_time) - 2

    if use_gurobi and remaining_time > 5:
        sol = solve_with_gurobi(inst, args, start_time, logger,
                                N_D, N_A, epsilon, M_u, M_I, M_dz, N_u, N_a,
                                R_dz, R_I, T_a, T_u, defenders_data, attackers_data,
                                dz_normals, int_normals, ctrl_normals, ctrl_rhs,
                                obs_normals, exp_Tu, A_pos, B_pos, A_vel, B_vel,
                                attacker_trajs, attacker_enters_dz, attacker_dz_steps_no_intercept,
                                remaining_time)
        if sol is not None:
            with open(args.solution_path, 'w') as f:
                json.dump(sol, f)
            return

    # Fallback: greedy heuristic
    sol = solve_greedy(inst, args, start_time, logger,
                       N_D, N_A, epsilon, M_u, M_I, M_dz, N_u, N_a,
                       R_dz, R_I, T_a, T_u, defenders_data, attackers_data,
                       dz_normals, int_normals, ctrl_normals, ctrl_rhs,
                       obs_normals, exp_Tu, A_pos, B_pos, A_vel, B_vel,
                       attacker_trajs, attacker_enters_dz, attacker_dz_steps_no_intercept)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


def solve_with_gurobi(inst, args, start_time, logger,
                      N_D, N_A, epsilon, M_u, M_I, M_dz, N_u, N_a,
                      R_dz, R_I, T_a, T_u, defenders_data, attackers_data,
                      dz_normals, int_normals, ctrl_normals, ctrl_rhs,
                      obs_normals, exp_Tu, A_pos, B_pos, A_vel, B_vel,
                      attacker_trajs, attacker_enters_dz, attacker_dz_steps_no_intercept,
                      time_limit):
    import gurobipy as gp
    from gurobipy import GRB

    M_big = 1e4  # Big-M constant

    try:
        model = gp.Model("defender_attacker")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(time_limit - 5, 10)

        # Defender variables
        # x[i][k], y[i][k] for k=0..N_u
        # vx[i][k], vy[i][k] for k=0..N_u
        # ux[i][k], uy[i][k] for k=0..N_u-1
        x = {}
        y = {}
        vx = {}
        vy = {}
        ux = {}
        uy = {}
        ux_abs = {}
        uy_abs = {}

        for i in range(N_D):
            d = defenders_data[i]
            for k in range(N_u + 1):
                x[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"x_{i}_{k}")
                y[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"y_{i}_{k}")
                vx[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"vx_{i}_{k}")
                vy[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"vy_{i}_{k}")
            for k in range(N_u):
                ux[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"ux_{i}_{k}")
                uy[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"uy_{i}_{k}")
                if epsilon > 0:
                    ux_abs[i, k] = model.addVar(lb=0, name=f"ux_abs_{i}_{k}")
                    uy_abs[i, k] = model.addVar(lb=0, name=f"uy_abs_{i}_{k}")

            # Initial conditions
            model.addConstr(x[i, 0] == d['x_s'])
            model.addConstr(y[i, 0] == d['y_s'])
            model.addConstr(vx[i, 0] == d['xdot_s'])
            model.addConstr(vy[i, 0] == d['ydot_s'])

            # Dynamics constraints
            for k in range(N_u):
                model.addConstr(x[i, k+1] == x[i, k] + A_pos * vx[i, k] + B_pos * ux[i, k])
                model.addConstr(y[i, k+1] == y[i, k] + A_pos * vy[i, k] + B_pos * uy[i, k])
                model.addConstr(vx[i, k+1] == A_vel * vx[i, k] + B_vel * ux[i, k])
                model.addConstr(vy[i, k+1] == A_vel * vy[i, k] + B_vel * uy[i, k])

            # Control constraints
            for k in range(N_u):
                for nx, ny in ctrl_normals:
                    model.addConstr(nx * ux[i, k] + ny * uy[i, k] <= ctrl_rhs)

                if epsilon > 0:
                    model.addConstr(ux_abs[i, k] >= ux[i, k])
                    model.addConstr(ux_abs[i, k] >= -ux[i, k])
                    model.addConstr(uy_abs[i, k] >= uy[i, k])
                    model.addConstr(uy_abs[i, k] >= -uy[i, k])

            # Obstacle avoidance: at steps 1..N_u, defender must be outside the obstacle
            # 4-sided polygon, must violate at least one half-space
            for k in range(1, N_u + 1):
                # Binary: z[i,k,s] = 1 means half-space s is violated (defender outside on this side)
                z_obs = {}
                for s in range(4):
                    z_obs[s] = model.addVar(vtype=GRB.BINARY, name=f"zobs_{i}_{k}_{s}")
                # At least one must be 1
                model.addConstr(gp.quicksum(z_obs[s] for s in range(4)) >= 1)
                for s in range(4):
                    nx, ny = obs_normals[s]
                    # If z_obs[s] = 1, then nx*x + ny*y > R_dz
                    # nx*x + ny*y >= R_dz + eps - M*(1 - z_obs[s])
                    # Use a small epsilon to enforce strict inequality
                    model.addConstr(nx * x[i, k] + ny * y[i, k] >= R_dz + 1e-6 - M_big * (1 - z_obs[s]))

        # Attacker variables
        # a[j][k] = 1 if attacker j is active at step k (0..N_a)
        # gamma[j][k] = 1 if attacker j is in DZ at step k
        # delta[i][j][k] = 1 if defender i intercepts attacker j at step k

        a_var = {}
        gamma_var = {}
        delta_var = {}

        for j in range(N_A):
            atk = attackers_data[j]
            ps, qs = attacker_trajs[j]

            for k in range(N_a + 1):
                a_var[j, k] = model.addVar(vtype=GRB.BINARY, name=f"a_{j}_{k}")
                gamma_var[j, k] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{j}_{k}")
                for i in range(N_D):
                    delta_var[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"delta_{i}_{j}_{k}")

            # Initially active
            model.addConstr(a_var[j, 0] == 1)

            # Attacker deactivation: once deactivated, stays deactivated
            for k in range(N_a):
                model.addConstr(a_var[j, k+1] <= a_var[j, k])

            # Attacker deactivation conditions:
            # Attacker becomes inactive if it enters DZ or is intercepted
            # a[j,k+1] = 0 if gamma[j,k] = 1 or any delta[i,j,k] = 1
            for k in range(N_a):
                model.addConstr(a_var[j, k+1] <= 1 - gamma_var[j, k])
                for i in range(N_D):
                    model.addConstr(a_var[j, k+1] <= 1 - delta_var[i, j, k])

            # gamma[j,k]: attacker in DZ at step k
            # gamma can only be 1 if attacker is active
            for k in range(N_a + 1):
                model.addConstr(gamma_var[j, k] <= a_var[j, k])

            # Position of attacker at step k (if active) is ps[k], qs[k]
            # gamma[j,k] = 1 iff attacker is active AND in DZ
            # For gamma to be 1, all DZ half-spaces must be satisfied
            # For gamma to be 0, at least one must be violated (or attacker inactive)
            for k in range(N_a + 1):
                if in_defense_zone_check(ps[k], qs[k], dz_normals, R_dz):
                    # Attacker position IS in DZ if active
                    # gamma[j,k] = a[j,k]
                    model.addConstr(gamma_var[j, k] == a_var[j, k])
                else:
                    # Attacker position is NOT in DZ
                    model.addConstr(gamma_var[j, k] == 0)

            # Interception: delta[i,j,k] = 1 iff defender i intercepts attacker j at step k
            # Only if attacker is active
            for k in range(N_a + 1):
                for i in range(N_D):
                    model.addConstr(delta_var[i, j, k] <= a_var[j, k])

                    # Interception requires defender position (interpolated at attacker time k*T_a)
                    # to be within R_I of attacker position
                    t_atk = k * T_a
                    # Find which control interval this falls in
                    ctrl_k = int(t_atk / T_u)
                    if ctrl_k >= N_u:
                        ctrl_k = N_u - 1
                    tau = t_atk - ctrl_k * T_u

                    if tau < 1e-12:
                        # At control boundary
                        def_x_expr = x[i, ctrl_k]
                        def_y_expr = y[i, ctrl_k]
                    elif abs(tau - T_u) < 1e-12:
                        def_x_expr = x[i, ctrl_k + 1]
                        def_y_expr = y[i, ctrl_k + 1]
                    else:
                        exp_tau = math.exp(-tau)
                        c_x = (1 - exp_tau)
                        c_u_x = (tau - 1 + exp_tau)
                        def_x_expr = x[i, ctrl_k] + c_x * vx[i, ctrl_k] + c_u_x * ux[i, ctrl_k]
                        def_y_expr = y[i, ctrl_k] + c_x * vy[i, ctrl_k] + c_u_x * uy[i, ctrl_k]

                    # Relative position
                    rel_x = ps[k] - def_x_expr
                    rel_y = qs[k] - def_y_expr

                    # For delta[i,j,k] = 1, all intercept half-spaces must be satisfied
                    for s, (nx, ny) in enumerate(int_normals):
                        # nx * rel_x + ny * rel_y <= R_I + M_big * (1 - delta[i,j,k])
                        model.addConstr(nx * rel_x + ny * rel_y <= R_I + M_big * (1 - delta_var[i, j, k]))

        # Objective: minimize sum of gamma[j,k] for k=1..N_a, j=0..N_A-1
        # plus epsilon * sum of |ux| + |uy|
        obj = gp.quicksum(gamma_var[j, k] for j in range(N_A) for k in range(1, N_a + 1))
        if epsilon > 0:
            obj += epsilon * gp.quicksum(ux_abs[i, k] + uy_abs[i, k]
                                          for i in range(N_D) for k in range(N_u))

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()

        # Callback for logging
        best_obj = [float('inf')]

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj[0] - 1e-6:
                    best_obj[0] = obj_val
                    if logger:
                        logger.log(obj_val)

        model.optimize(callback)

        if model.SolCount > 0:
            # Extract solution
            sol = extract_gurobi_solution(model, N_D, N_A, N_u, N_a, epsilon,
                                          x, y, vx, vy, ux, uy,
                                          a_var, gamma_var, delta_var,
                                          attacker_trajs, defenders_data, attackers_data,
                                          T_a, T_u, dz_normals, R_dz, int_normals, R_I,
                                          A_pos, B_pos, A_vel, B_vel, exp_Tu,
                                          ux_abs if epsilon > 0 else None,
                                          uy_abs if epsilon > 0 else None)

            if logger:
                logger.log_solution(sol['objective_value'], sol)
            return sol
        else:
            return None
    except Exception as e:
        print(f"Gurobi error: {e}")
        return None


def in_defense_zone_check(px, py, dz_normals, R_dz):
    for nx, ny in dz_normals:
        if nx * px + ny * py > R_dz + 1e-9:
            return False
    return True


def extract_gurobi_solution(model, N_D, N_A, N_u, N_a, epsilon,
                            x, y, vx, vy, ux, uy,
                            a_var, gamma_var, delta_var,
                            attacker_trajs, defenders_data, attackers_data,
                            T_a, T_u, dz_normals, R_dz, int_normals, R_I,
                            A_pos, B_pos, A_vel, B_vel, exp_Tu,
                            ux_abs, uy_abs):
    sol_defenders = []
    for i in range(N_D):
        d = {
            'x': [x[i, k].X for k in range(N_u + 1)],
            'y': [y[i, k].X for k in range(N_u + 1)],
            'xdot': [vx[i, k].X for k in range(N_u + 1)],
            'ydot': [vy[i, k].X for k in range(N_u + 1)],
            'ux': [ux[i, k].X for k in range(N_u)],
            'uy': [uy[i, k].X for k in range(N_u)]
        }
        sol_defenders.append(d)

    sol_attackers = []
    for j in range(N_A):
        ps, qs = attacker_trajs[j]
        a_vals = [round(a_var[j, k].X) for k in range(N_a + 1)]
        gamma_vals = [round(gamma_var[j, k].X) for k in range(N_a + 1)]

        # Compute actual attacker positions considering deactivation
        actual_p = [ps[0]]
        actual_q = [qs[0]]
        active = True
        for k in range(1, N_a + 1):
            if a_vals[k] > 0.5 and active:
                actual_p.append(ps[k])
                actual_q.append(qs[k])
            else:
                active = False
                actual_p.append(actual_p[-1])
                actual_q.append(actual_q[-1])

        delta_vals = []
        for i in range(N_D):
            delta_vals.append([round(delta_var[i, j, k].X) for k in range(N_a + 1)])

        sol_attackers.append({
            'p': actual_p,
            'q': actual_q,
            'a': [float(v) for v in a_vals],
            'gamma': [float(v) for v in gamma_vals],
            'delta': [[float(v) for v in delta_vals[i]] for i in range(N_D)]
        })

    obj_val = model.ObjVal
    return {
        'objective_value': obj_val,
        'defenders': sol_defenders,
        'attackers': sol_attackers
    }


def solve_greedy(inst, args, start_time, logger,
                 N_D, N_A, epsilon, M_u, M_I, M_dz, N_u, N_a,
                 R_dz, R_I, T_a, T_u, defenders_data, attackers_data,
                 dz_normals, int_normals, ctrl_normals, ctrl_rhs,
                 obs_normals, exp_Tu, A_pos, B_pos, A_vel, B_vel,
                 attacker_trajs, attacker_enters_dz, attacker_dz_steps_no_intercept):
    import math

    # Simple greedy: each defender does nothing (zero control inputs)
    # Then try to improve by assigning defenders to intercept attackers

    def simulate_defender(x0, y0, vx0, vy0, ux_list, uy_list):
        xs = [x0]
        ys = [y0]
        vxs = [vx0]
        vys = [vy0]
        for k in range(N_u):
            x_new = xs[-1] + A_pos * vxs[-1] + B_pos * ux_list[k]
            y_new = ys[-1] + A_pos * vys[-1] + B_pos * uy_list[k]
            vx_new = A_vel * vxs[-1] + B_vel * ux_list[k]
            vy_new = A_vel * vys[-1] + B_vel * uy_list[k]
            xs.append(x_new)
            ys.append(y_new)
            vxs.append(vx_new)
            vys.append(vy_new)
        return xs, ys, vxs, vys

    def interpolate_defender_pos(t_atk, xs, ys, vxs, vys, ux_list, uy_list):
        ctrl_k = int(t_atk / T_u)
        if ctrl_k >= N_u:
            ctrl_k = N_u - 1
        tau = t_atk - ctrl_k * T_u
        if tau < 1e-12:
            return xs[ctrl_k], ys[ctrl_k]
        if abs(tau - T_u) < 1e-12:
            return xs[ctrl_k + 1], ys[ctrl_k + 1]
        exp_tau = math.exp(-tau)
        c_v = 1 - exp_tau
        c_u = tau - 1 + exp_tau
        def_x = xs[ctrl_k] + c_v * vxs[ctrl_k] + c_u * ux_list[ctrl_k]
        def_y = ys[ctrl_k] + c_v * vys[ctrl_k] + c_u * uy_list[ctrl_k]
        return def_x, def_y

    def in_intercept_zone(dx, dy):
        for nx, ny in int_normals:
            if nx * dx + ny * dy > R_I + 1e-9:
                return False
        return True

    def in_dz(px, py):
        for nx, ny in dz_normals:
            if nx * px + ny * py > R_dz + 1e-9:
                return False
        return True

    def outside_obs(px, py):
        for nx, ny in obs_normals:
            if nx * px + ny * py > R_dz + 1e-9:
                return True
        return False

    def project_control(ux_val, uy_val):
        """Project control input onto feasible set."""
        # Check if already feasible
        feasible = True
        for nx, ny in ctrl_normals:
            if nx * ux_val + ny * uy_val > ctrl_rhs + 1e-9:
                feasible = False
                break
        if feasible:
            return ux_val, uy_val

        # Scale down
        if abs(ux_val) < 1e-12 and abs(uy_val) < 1e-12:
            return 0.0, 0.0

        # Binary search for largest scale factor
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            ok = True
            for nx, ny in ctrl_normals:
                if nx * (mid * ux_val) + ny * (mid * uy_val) > ctrl_rhs + 1e-9:
                    ok = False
                    break
            if ok:
                lo = mid
            else:
                hi = mid
        return lo * ux_val, lo * uy_val

    # Start with zero controls for all defenders
    defender_controls = []
    for i in range(N_D):
        defender_controls.append(([0.0] * N_u, [0.0] * N_u))

    # Compute trajectories
    def compute_all():
        trajs = []
        for i in range(N_D):
            d = defenders_data[i]
            ux_list, uy_list = defender_controls[i]
            xs, ys, vxs, vys = simulate_defender(d['x_s'], d['y_s'], d['xdot_s'], d['ydot_s'],
                                                  ux_list, uy_list)
            trajs.append((xs, ys, vxs, vys, ux_list, uy_list))
        return trajs

    def evaluate_solution(def_trajs):
        """Compute objective value and solution dict."""
        total_gamma = 0
        total_effort = 0

        sol_attackers = []
        for j in range(N_A):
            ps_orig, qs_orig = attacker_trajs[j]
            atk = attackers_data[j]

            a_vals = [1.0]
            gamma_vals = []
            delta_all = [[0.0] * (N_a + 1) for _ in range(N_D)]

            # Step 0
            in_dz_0 = in_dz(ps_orig[0], qs_orig[0])
            gamma_vals.append(1.0 if in_dz_0 else 0.0)

            # Check interception at step 0
            intercepted_0 = False
            for i in range(N_D):
                xs, ys, vxs, vys, ux_list, uy_list = def_trajs[i]
                dx = ps_orig[0] - xs[0]
                dy = qs_orig[0] - ys[0]
                if in_intercept_zone(dx, dy):
                    delta_all[i][0] = 1.0
                    intercepted_0 = True

            active = True
            if in_dz_0 or intercepted_0:
                active = False

            actual_p = [ps_orig[0]]
            actual_q = [qs_orig[0]]

            for k in range(1, N_a + 1):
                if active:
                    a_vals.append(1.0)
                    px, py = ps_orig[k], qs_orig[k]
                    actual_p.append(px)
                    actual_q.append(py)

                    in_dz_k = in_dz(px, py)
                    gamma_vals.append(1.0 if in_dz_k else 0.0)
                    if in_dz_k:
                        total_gamma += 1

                    # Check interception
                    intercepted_k = False
                    t_atk = k * T_a
                    for i in range(N_D):
                        xs, ys, vxs, vys, ux_list, uy_list = def_trajs[i]
                        def_x, def_y = interpolate_defender_pos(t_atk, xs, ys, vxs, vys, ux_list, uy_list)
                        dx = px - def_x
                        dy = py - def_y
                        if in_intercept_zone(dx, dy):
                            delta_all[i][k] = 1.0
                            intercepted_k = True

                    if in_dz_k or intercepted_k:
                        active = False
                else:
                    a_vals.append(0.0)
                    actual_p.append(actual_p[-1])
                    actual_q.append(actual_q[-1])
                    gamma_vals.append(0.0)

            sol_attackers.append({
                'p': actual_p,
                'q': actual_q,
                'a': a_vals,
                'gamma': gamma_vals,
                'delta': delta_all
            })

        if epsilon > 0:
            for i in range(N_D):
                ux_list, uy_list = defender_controls[i]
                for k in range(N_u):
                    total_effort += abs(ux_list[k]) + abs(uy_list[k])

        obj = total_gamma + epsilon * total_effort

        sol_defenders = []
        for i in range(N_D):
            xs, ys, vxs, vys, ux_list, uy_list = def_trajs[i]
            sol_defenders.append({
                'x': list(xs),
                'y': list(ys),
                'xdot': list(vxs),
                'ydot': list(vys),
                'ux': list(ux_list),
                'uy': list(uy_list)
            })

        return obj, {
            'objective_value': obj,
            'defenders': sol_defenders,
            'attackers': sol_attackers
        }

    def_trajs = compute_all()
    best_obj, best_sol = evaluate_solution(def_trajs)
    if logger:
        logger.log_solution(best_obj, best_sol)

    # Try proportional navigation approach for each defender-attacker pair
    # For each attacker that has DZ steps, try to compute intercept trajectory

    def compute_intercept_controls(def_idx, atk_idx):
        """Try to compute control inputs for defender def_idx to intercept attacker atk_idx."""
        d = defenders_data[def_idx]
        atk = attackers_data[atk_idx]
        ps, qs = attacker_trajs[atk_idx]

        # Find earliest attacker time step where we could intercept
        # Try each attacker time step as target
        best_controls = None
        best_step = None

        enter = attacker_enters_dz[atk_idx]
        if enter is None:
            return None, None  # Attacker never enters DZ

        # Try to intercept before it enters DZ, or at entry step
        max_step = min(enter, N_a)

        for target_k in range(max_step + 1):
            target_time = target_k * T_a
            target_px = ps[target_k]
            target_py = qs[target_k]

            # Use a simple approach: compute desired trajectory to reach target
            # at target_time using proportional control

            # How many control steps before target_time?
            n_ctrl = int(target_time / T_u)
            if n_ctrl > N_u:
                n_ctrl = N_u

            if n_ctrl == 0:
                # Check if we can intercept at step 0
                dx = ps[0] - d['x_s']
                dy = qs[0] - d['y_s']
                if in_intercept_zone(dx, dy):
                    return [0.0] * N_u, [0.0] * N_u
                continue

            # Simple approach: constant control to reach target
            # We need to solve for constant u such that position at target_time matches target

            # For constant control u over all steps:
            # x(n) = x0 + sum_{k=0}^{n-1} [A_pos * vx(k) + B_pos * u_x]
            # This is complex, let's just use iterative approach

            # Try: aim control at target position with proportional gain
            ux_list = [0.0] * N_u
            uy_list = [0.0] * N_u

            # Simple: use constant velocity toward target
            # Approximate: desired velocity = (target - current) / time
            if target_time > 1e-6:
                desired_vx = (target_px - d['x_s']) / target_time
                desired_vy = (target_py - d['y_s']) / target_time
            else:
                continue

            for k in range(N_u):
                ux_list[k], uy_list[k] = project_control(desired_vx, desired_vy)

            # Simulate and check if we intercept
            xs, ys, vxs, vys = simulate_defender(d['x_s'], d['y_s'], d['xdot_s'], d['ydot_s'],
                                                  ux_list, uy_list)

            # Check interception at target step
            def_x, def_y = interpolate_defender_pos(target_time, xs, ys, vxs, vys, ux_list, uy_list)
            dx = target_px - def_x
            dy = target_py - def_y

            if in_intercept_zone(dx, dy):
                # Check obstacle avoidance
                obs_ok = True
                for k in range(1, N_u + 1):
                    if not outside_obs(xs[k], ys[k]):
                        obs_ok = False
                        break
                if obs_ok:
                    return ux_list, uy_list

        # More sophisticated: iterative refinement
        # Try different gain values
        for gain in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            for target_k in range(max(0, enter - 3), min(enter + 1, N_a + 1)):
                if time.time() - start_time > args.time_limit - 2:
                    return None, None

                target_time = target_k * T_a
                target_px = ps[target_k]
                target_py = qs[target_k]

                ux_list = [0.0] * N_u
                uy_list = [0.0] * N_u

                # Iterative: at each control step, aim toward where attacker will be
                cur_x, cur_y = d['x_s'], d['y_s']
                cur_vx, cur_vy = d['xdot_s'], d['ydot_s']

                for k in range(N_u):
                    t_now = k * T_u
                    t_remaining = target_time - t_now
                    if t_remaining < T_u:
                        t_remaining = T_u

                    # Desired velocity
                    dvx = gain * (target_px - cur_x) / t_remaining
                    dvy = gain * (target_py - cur_y) / t_remaining

                    ux_val, uy_val = project_control(dvx, dvy)
                    ux_list[k] = ux_val
                    uy_list[k] = uy_val

                    # Update state
                    new_x = cur_x + A_pos * cur_vx + B_pos * ux_val
                    new_y = cur_y + A_pos * cur_vy + B_pos * uy_val
                    new_vx = A_vel * cur_vx + B_vel * ux_val
                    new_vy = A_vel * cur_vy + B_vel * uy_val
                    cur_x, cur_y = new_x, new_y
                    cur_vx, cur_vy = new_vx, new_vy

                xs, ys, vxs, vys = simulate_defender(d['x_s'], d['y_s'], d['xdot_s'], d['ydot_s'],
                                                      ux_list, uy_list)

                # Check interception at all attacker steps up to and including target
                for check_k in range(N_a + 1):
                    t_atk = check_k * T_a
                    def_x, def_y = interpolate_defender_pos(t_atk, xs, ys, vxs, vys, ux_list, uy_list)
                    dx = ps[check_k] - def_x
                    dy = qs[check_k] - def_y
                    if in_intercept_zone(dx, dy):
                        obs_ok = True
                        for kk in range(1, N_u + 1):
                            if not outside_obs(xs[kk], ys[kk]):
                                obs_ok = False
                                break
                        if obs_ok:
                            return ux_list, uy_list
                        break

        return None, None

    # Try assigning defenders to attackers
    # Sort attackers by DZ steps (most impactful first)
    atk_value = []
    for j in range(N_A):
        atk_value.append((attacker_dz_steps_no_intercept[j], j))
    atk_value.sort(reverse=True)

    assigned = set()
    for _, j in atk_value:
        if attacker_dz_steps_no_intercept[j] == 0:
            continue
        if time.time() - start_time > args.time_limit - 5:
            break

        # Try each unassigned defender
        best_def = None
        best_ctrl = None
        for i in range(N_D):
            if i in assigned:
                continue
            if time.time() - start_time > args.time_limit - 3:
                break
            ux_list, uy_list = compute_intercept_controls(i, j)
            if ux_list is not None:
                best_def = i
                best_ctrl = (ux_list, uy_list)
                break  # Take first feasible

        if best_def is not None:
            assigned.add(best_def)
            defender_controls[best_def] = best_ctrl

    def_trajs = compute_all()
    obj, sol = evaluate_solution(def_trajs)
    if obj < best_obj:
        best_obj = obj
        best_sol = sol
        if logger:
            logger.log_solution(best_obj, best_sol)

    return best_sol


if __name__ == '__main__':
    main()