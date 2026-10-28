import json
import argparse
import math
import numpy as np
import sys
import time as time_module
from gurobipy import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time_module.time()

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
    T_a = data['T_a']
    T_u = data['T_u']
    defenders_data = data['defenders']
    attackers_data = data['attackers']

    # Precompute constants
    exp_Tu = math.exp(-T_u)
    one_minus_exp_Tu = 1.0 - exp_Tu
    Tu_minus_1_plus_exp_Tu = T_u - 1.0 + exp_Tu

    # Polygon normals
    dz_cos = [math.cos(2 * math.pi * m / M_dz) for m in range(M_dz)]
    dz_sin = [math.sin(2 * math.pi * m / M_dz) for m in range(M_dz)]
    I_cos = [math.cos(2 * math.pi * m / M_I) for m in range(M_I)]
    I_sin = [math.sin(2 * math.pi * m / M_I) for m in range(M_I)]
    u_cos = [math.cos(2 * math.pi * m / M_u) for m in range(M_u)]
    u_sin = [math.sin(2 * math.pi * m / M_u) for m in range(M_u)]
    u_rhs = math.cos(math.pi / M_u)
    obs_cos = [math.cos(2 * math.pi * m / 4) for m in range(4)]
    obs_sin = [math.sin(2 * math.pi * m / 4) for m in range(4)]

    # Free trajectories
    p_free = np.zeros((N_A, N_a + 1))
    q_free = np.zeros((N_A, N_a + 1))
    for a in range(N_A):
        for k in range(N_a + 1):
            p_free[a, k] = attackers_data[a]['p_s'] + attackers_data[a]['v_p'] * k * T_a
            q_free[a, k] = attackers_data[a]['q_s'] + attackers_data[a]['v_q'] * k * T_a

    # Zone entry step for each attacker
    k_zone = []
    for a in range(N_A):
        kz = N_a + 1
        for k in range(N_a + 1):
            inside = True
            for m in range(M_dz):
                if dz_cos[m] * p_free[a, k] + dz_sin[m] * q_free[a, k] > R_dz + 1e-9:
                    inside = False
                    break
            if inside:
                kz = k
                break
        k_zone.append(kz)

    # Zone cost
    zone_cost = []
    for a in range(N_A):
        if k_zone[a] <= N_a:
            zone_cost.append(float(N_a - max(k_zone[a], 1) + 1))
        else:
            zone_cost.append(0.0)

    # Interpolation info for each attacker step
    interp_j = []
    interp_s = []
    for k in range(N_a + 1):
        t = k * T_a
        j = int(t / T_u) if T_u > 0 else 0
        if j >= N_u:
            j = N_u
            s = 0.0
        else:
            s = t - j * T_u
        interp_j.append(j)
        interp_s.append(s)

    # Big-M values
    max_coord = 0.0
    for a in range(N_A):
        for k in range(N_a + 1):
            max_coord = max(max_coord, abs(p_free[a, k]), abs(q_free[a, k]))
    for d in range(N_D):
        max_coord = max(max_coord, abs(defenders_data[d]['x_s']), abs(defenders_data[d]['y_s']))
    max_coord += N_u * T_u + 10.0
    M_intercept = 2.0 * max_coord + R_I + 10.0
    M_obs = max_coord + R_dz + 10.0

    def build_solution_from_values(x_val, y_val, xd_val, yd_val, ux_val, uy_val, delta_val, intercept_val):
        """Build the output solution dictionary from variable values."""
        obj_val = 0.0
        for a in range(N_A):
            if a in intercept_val and intercept_val[a] > 0.5:
                pass  # no zone cost
            elif k_zone[a] <= N_a:
                obj_val += zone_cost[a]

        if epsilon > 0:
            for d in range(N_D):
                for k in range(N_u):
                    obj_val += epsilon * (abs(ux_val[d][k]) + abs(uy_val[d][k]))

        sol_defenders = []
        for d in range(N_D):
            sol_defenders.append({
                "x": [float(x_val[d][k]) for k in range(N_u + 1)],
                "y": [float(y_val[d][k]) for k in range(N_u + 1)],
                "xdot": [float(xd_val[d][k]) for k in range(N_u + 1)],
                "ydot": [float(yd_val[d][k]) for k in range(N_u + 1)],
                "ux": [float(ux_val[d][k]) for k in range(N_u)],
                "uy": [float(uy_val[d][k]) for k in range(N_u)]
            })

        sol_attackers = []
        for a in range(N_A):
            # Determine interception step
            k_int = None
            if a in intercept_val and intercept_val[a] > 0.5:
                for k in range(min(k_zone[a], N_a + 1)):
                    for d in range(N_D):
                        if (d, a, k) in delta_val and delta_val[d, a, k] > 0.5:
                            if k_int is None or k < k_int:
                                k_int = k
                            break
                    if k_int is not None and k == k_int:
                        break

            # Compute positions, active, gamma, delta
            p_out = [0.0] * (N_a + 1)
            q_out = [0.0] * (N_a + 1)
            a_out = [0.0] * (N_a + 1)
            gamma_out = [0.0] * (N_a + 1)
            delta_out = [[0.0] * (N_a + 1) for _ in range(N_D)]

            if k_int is not None:
                # Intercepted at k_int
                for k in range(N_a + 1):
                    if k <= k_int:
                        p_out[k] = float(p_free[a, k])
                        q_out[k] = float(q_free[a, k])
                        a_out[k] = 1.0
                    else:
                        p_out[k] = float(p_free[a, k_int])
                        q_out[k] = float(q_free[a, k_int])
                        a_out[k] = 0.0
                    gamma_out[k] = 0.0
                for d in range(N_D):
                    for k in range(N_a + 1):
                        if (d, a, k) in delta_val and delta_val[d, a, k] > 0.5:
                            delta_out[d][k] = 1.0
            elif k_zone[a] <= N_a:
                # Enters zone at k_zone[a]
                for k in range(N_a + 1):
                    if k <= k_zone[a]:
                        p_out[k] = float(p_free[a, k])
                        q_out[k] = float(q_free[a, k])
                        a_out[k] = 1.0
                    else:
                        p_out[k] = float(p_free[a, k_zone[a]])
                        q_out[k] = float(q_free[a, k_zone[a]])
                        a_out[k] = 0.0
                    if k >= max(k_zone[a], 1):
                        gamma_out[k] = 1.0
                    else:
                        gamma_out[k] = 0.0
                # gamma at k_zone if k_zone >= 1
                if k_zone[a] >= 1:
                    gamma_out[k_zone[a]] = 1.0
                elif k_zone[a] == 0:
                    # In zone from step 0, gamma counts from step 1
                    for k in range(1, N_a + 1):
                        gamma_out[k] = 1.0
            else:
                # Never enters zone
                for k in range(N_a + 1):
                    p_out[k] = float(p_free[a, k])
                    q_out[k] = float(q_free[a, k])
                    a_out[k] = 1.0
                    gamma_out[k] = 0.0

            sol_attackers.append({
                "p": p_out,
                "q": q_out,
                "a": a_out,
                "gamma": gamma_out,
                "delta": delta_out
            })

        return {"objective_value": float(obj_val), "defenders": sol_defenders, "attackers": sol_attackers}

    def build_fallback():
        """Build a solution with zero control."""
        x_val = [[0.0] * (N_u + 1) for _ in range(N_D)]
        y_val = [[0.0] * (N_u + 1) for _ in range(N_D)]
        xd_val = [[0.0] * (N_u + 1) for _ in range(N_D)]
        yd_val = [[0.0] * (N_u + 1) for _ in range(N_D)]
        ux_val = [[0.0] * N_u for _ in range(N_D)]
        uy_val = [[0.0] * N_u for _ in range(N_D)]

        for d in range(N_D):
            x_val[d][0] = defenders_data[d]['x_s']
            y_val[d][0] = defenders_data[d]['y_s']
            xd_val[d][0] = defenders_data[d]['xdot_s']
            yd_val[d][0] = defenders_data[d]['ydot_s']
            for k in range(N_u):
                xd_val[d][k + 1] = xd_val[d][k] * exp_Tu
                yd_val[d][k + 1] = yd_val[d][k] * exp_Tu
                x_val[d][k + 1] = x_val[d][k] + xd_val[d][k] * one_minus_exp_Tu
                y_val[d][k + 1] = y_val[d][k] + yd_val[d][k] * one_minus_exp_Tu

        return build_solution_from_values(x_val, y_val, xd_val, yd_val, ux_val, uy_val, {}, {})

    # Check if we have enough time for MILP
    elapsed = time_module.time() - start_time
    remaining = args.time_limit - elapsed

    if remaining < 2:
        solution = build_fallback()
        if logger:
            logger.log_solution(solution['objective_value'], solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Build MILP
    try:
        model = Model("defense")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, remaining - 1))
        model.setParam('OutputFlag', 0)

        # Defender state variables
        x = [[None] * (N_u + 1) for _ in range(N_D)]
        y = [[None] * (N_u + 1) for _ in range(N_D)]
        xdv = [[None] * (N_u + 1) for _ in range(N_D)]
        ydv = [[None] * (N_u + 1) for _ in range(N_D)]
        uxv = [[None] * N_u for _ in range(N_D)]
        uyv = [[None] * N_u for _ in range(N_D)]

        for d in range(N_D):
            for k in range(N_u + 1):
                x[d][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{d}_{k}")
                y[d][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"y_{d}_{k}")
                xdv[d][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xd_{d}_{k}")
                ydv[d][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"yd_{d}_{k}")
            for k in range(N_u):
                uxv[d][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"ux_{d}_{k}")
                uyv[d][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"uy_{d}_{k}")

        # Interception binary variables
        delta_vars = {}
        intercept_vars = {}
        interceptable = []

        for a in range(N_A):
            if 1 <= k_zone[a] <= N_a:
                interceptable.append(a)
                intercept_vars[a] = model.addVar(vtype=GRB.BINARY, name=f"inter_{a}")
                for d in range(N_D):
                    for k in range(k_zone[a]):
                        # Geometric filter
                        dist = math.sqrt((p_free[a, k] - defenders_data[d]['x_s']) ** 2 +
                                         (q_free[a, k] - defenders_data[d]['y_s']) ** 2)
                        init_speed = math.sqrt(defenders_data[d]['xdot_s'] ** 2 + defenders_data[d]['ydot_s'] ** 2)
                        max_reach = N_u * T_u + init_speed * (1 - math.exp(-N_u * T_u)) + 5.0
                        if dist <= max_reach + R_I:
                            delta_vars[d, a, k] = model.addVar(vtype=GRB.BINARY, name=f"dl_{d}_{a}_{k}")

        # Obstacle avoidance binaries
        b_obs = {}
        for d in range(N_D):
            for k in range(1, N_u + 1):
                for m in range(4):
                    b_obs[d, k, m] = model.addVar(vtype=GRB.BINARY, name=f"bo_{d}_{k}_{m}")

        # Control effort auxiliaries
        ux_abs = [[None] * N_u for _ in range(N_D)]
        uy_abs = [[None] * N_u for _ in range(N_D)]
        for d in range(N_D):
            for k in range(N_u):
                ux_abs[d][k] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"uxa_{d}_{k}")
                uy_abs[d][k] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"uya_{d}_{k}")

        model.update()

        # Initial conditions
        for d in range(N_D):
            model.addConstr(x[d][0] == defenders_data[d]['x_s'])
            model.addConstr(y[d][0] == defenders_data[d]['y_s'])
            model.addConstr(xdv[d][0] == defenders_data[d]['xdot_s'])
            model.addConstr(ydv[d][0] == defenders_data[d]['ydot_s'])

        # Dynamics
        for d in range(N_D):
            for k in range(N_u):
                model.addConstr(xdv[d][k + 1] == xdv[d][k] * exp_Tu + uxv[d][k] * one_minus_exp_Tu)
                model.addConstr(ydv[d][k + 1] == ydv[d][k] * exp_Tu + uyv[d][k] * one_minus_exp_Tu)
                model.addConstr(x[d][k + 1] == x[d][k] + xdv[d][k] * one_minus_exp_Tu + uxv[d][k] * Tu_minus_1_plus_exp_Tu)
                model.addConstr(y[d][k + 1] == y[d][k] + ydv[d][k] * one_minus_exp_Tu + uyv[d][k] * Tu_minus_1_plus_exp_Tu)

        # Control polygon constraints
        for d in range(N_D):
            for k in range(N_u):
                for m in range(M_u):
                    model.addConstr(u_cos[m] * uxv[d][k] + u_sin[m] * uyv[d][k] <= u_rhs)

        # Obstacle avoidance
        for d in range(N_D):
            for k in range(1, N_u + 1):
                model.addConstr(quicksum(b_obs[d, k, m] for m in range(4)) >= 1)
                for m in range(4):
                    model.addConstr(obs_cos[m] * x[d][k] + obs_sin[m] * y[d][k] >= R_dz - M_obs * (1 - b_obs[d, k, m]))

        # Interception constraints
        for a in interceptable:
            relevant = [(d, k) for (d, aa, k) in delta_vars if aa == a]
            if relevant:
                model.addConstr(intercept_vars[a] <= quicksum(delta_vars[d, a, k] for d, k in relevant))
                for d, k in relevant:
                    model.addConstr(intercept_vars[a] >= delta_vars[d, a, k])

            for (d, aa, k), dv in delta_vars.items():
                if aa != a:
                    continue
                j = interp_j[k]
                s = interp_s[k]

                if j < N_u:
                    exp_s = math.exp(-s)
                    oms = 1.0 - exp_s
                    sm1pes = s - 1.0 + exp_s
                    xi = x[d][j] + xdv[d][j] * oms + uxv[d][j] * sm1pes
                    yi = y[d][j] + ydv[d][j] * oms + uyv[d][j] * sm1pes
                else:
                    xi = x[d][N_u] + xdv[d][N_u] * (1.0 - math.exp(-s)) if s > 0 else x[d][N_u]
                    yi = y[d][N_u] + ydv[d][N_u] * (1.0 - math.exp(-s)) if s > 0 else y[d][N_u]

                for m in range(M_I):
                    lhs = I_cos[m] * (p_free[a, k] - xi) + I_sin[m] * (q_free[a, k] - yi)
                    model.addConstr(lhs <= R_I + M_intercept * (1 - dv))

        # Absolute value constraints
        for d in range(N_D):
            for k in range(N_u):
                model.addConstr(ux_abs[d][k] >= uxv[d][k])
                model.addConstr(ux_abs[d][k] >= -uxv[d][k])
                model.addConstr(uy_abs[d][k] >= uyv[d][k])
                model.addConstr(uy_abs[d][k] >= -uyv[d][k])

        # Objective
        obj = LinExpr()
        for a in range(N_A):
            if a in intercept_vars:
                obj += zone_cost[a] * (1 - intercept_vars[a])
            elif k_zone[a] <= N_a:
                obj += zone_cost[a]

        if epsilon > 0:
            for d in range(N_D):
                for k in range(N_u):
                    obj += epsilon * (ux_abs[d][k] + uy_abs[d][k])

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()

        # Solve with callback for logging
        best_solution = [None]
        best_obj = [GRB.INFINITY]

        def my_callback(cb_model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    cb_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if cb_obj < best_obj[0]:
                        # Extract values
                        x_val = [[cb_model.cbGetSolution(x[d][k]) for k in range(N_u + 1)] for d in range(N_D)]
                        y_val = [[cb_model.cbGetSolution(y[d][k]) for k in range(N_u + 1)] for d in range(N_D)]
                        xd_val = [[cb_model.cbGetSolution(xdv[d][k]) for k in range(N_u + 1)] for d in range(N_D)]
                        yd_val = [[cb_model.cbGetSolution(ydv[d][k]) for k in range(N_u + 1)] for d in range(N_D)]
                        ux_val = [[cb_model.cbGetSolution(uxv[d][k]) for k in range(N_u)] for d in range(N_D)]
                        uy_val = [[cb_model.cbGetSolution(uyv[d][k]) for k in range(N_u)] for d in range(N_D)]
                        delta_val = {}
                        for key, var in delta_vars.items():
                            delta_val[key] = cb_model.cbGetSolution(var)
                        inter_val = {}
                        for a in interceptable:
                            inter_val[a] = cb_model.cbGetSolution(intercept_vars[a])

                        sol = build_solution_from_values(x_val, y_val, xd_val, yd_val, ux_val, uy_val, delta_val, inter_val)
                        best_obj[0] = cb_obj
                        best_solution[0] = sol
                        if logger:
                            logger.log_solution(cb_obj, sol)
                except Exception:
                    pass

        model.optimize(my_callback)

        # Extract final solution
        if model.SolCount > 0:
            x_val = [[x[d][k].X for k in range(N_u + 1)] for d in range(N_D)]
            y_val = [[y[d][k].X for k in range(N_u + 1)] for d in range(N_D)]
            xd_val = [[xdv[d][k].X for k in range(N_u + 1)] for d in range(N_D)]
            yd_val = [[ydv[d][k].X for k in range(N_u + 1)] for d in range(N_D)]
            ux_val = [[uxv[d][k].X for k in range(N_u)] for d in range(N_D)]
            uy_val = [[uyv[d][k].X for k in range(N_u)] for d in range(N_D)]
            delta_val = {}
            for key, var in delta_vars.items():
                delta_val[key] = var.X
            inter_val = {}
            for a in interceptable:
                inter_val[a] = intercept_vars[a].X

            solution = build_solution_from_values(x_val, y_val, xd_val, yd_val, ux_val, uy_val, delta_val, inter_val)
        elif best_solution[0] is not None:
            solution = best_solution[0]
        else:
            solution = build_fallback()

    except Exception as e:
        solution = build_fallback()

    if logger:
        logger.log_solution(solution['objective_value'], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()