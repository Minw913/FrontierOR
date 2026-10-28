import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Defense-zone interception MILP solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    N_D = int(inst['N_D'])
    N_A = int(inst['N_A'])
    eps = float(inst.get('epsilon', 0.0))
    M_u = int(inst['M_u'])
    M_I = int(inst['M_I'])
    M_dz = int(inst['M_dz'])
    N_u = int(inst['N_u'])
    N_a = int(inst['N_a'])
    R_dz = float(inst['R_dz'])
    R_I = float(inst['R_I'])
    T_a = float(inst['T_a'])
    T_u = float(inst['T_u'])
    defenders = inst.get('defenders', [])
    attackers = inst.get('attackers', [])

    E = math.exp(-T_u)
    Bv = 1.0 - E
    Bx = T_u - 1.0 + E

    def normals(M):
        return [(math.cos(2.0 * math.pi * m / M), math.sin(2.0 * math.pi * m / M))
                for m in range(M)]

    u_norm = normals(M_u)
    I_norm = normals(M_I)
    dz_norm = normals(M_dz)
    obs_norm = normals(4)
    u_rhs = math.cos(math.pi / M_u)
    R_I_circ = R_I / math.cos(math.pi / M_I)

    # Nominal attacker trajectories (positions while active are deterministic)
    P = []
    for j in range(N_A):
        aj = attackers[j]
        ps = float(aj['p_s']); qs = float(aj['q_s'])
        vp = float(aj['v_p']); vq = float(aj['v_q'])
        P.append([(ps + vp * T_a * k, qs + vq * T_a * k) for k in range(N_a + 1)])

    def inside_dz(pt):
        px, py = pt
        for (c, s) in dz_norm:
            if c * px + s * py > R_dz + 1e-9:
                return False
        return True

    kDZ = []
    for j in range(N_A):
        kd = None
        for k in range(N_a + 1):
            if inside_dz(P[j][k]):
                kd = k
                break
        kDZ.append(kd)

    # Attackers that can still be intercepted before entering the DZ
    var_att = [j for j in range(N_A) if kDZ[j] is not None and 1 <= kDZ[j] <= N_a]
    W = {j: float(N_a - kDZ[j] + 1) for j in var_att}

    # Defender position interpolation mapping for each attacker sample step
    interp = []
    for k in range(N_a + 1):
        t = k * T_a
        if N_u <= 0 or t >= N_u * T_u - 1e-12:
            interp.append((N_u, 0.0))
        else:
            c = int(math.floor(t / T_u + 1e-9))
            if c > N_u - 1:
                c = N_u - 1
            f = (t - c * T_u) / T_u
            f = max(0.0, min(1.0, f))
            interp.append((c, f))

    # Big-M estimates
    maxP = 1.0
    for j in range(N_A):
        for k in range(N_a + 1):
            nrm = math.hypot(P[j][k][0], P[j][k][1])
            if nrm > maxP:
                maxP = nrm
    maxv0 = 0.0
    maxx0 = 0.0
    for d in defenders:
        maxv0 = max(maxv0, math.hypot(float(d['xdot_s']), float(d['ydot_s'])))
        maxx0 = max(maxx0, math.hypot(float(d['x_s']), float(d['y_s'])))
    vmax = max(1.0, maxv0)
    reach = maxx0 + N_u * T_u * vmax
    M_delta = maxP + reach + R_I + 1.0
    M_obs = reach + R_dz + 1.0

    # Necessary reachability condition used to prune interception binaries
    def can_reach(i, j, k):
        d0 = defenders[i]
        x0n = math.hypot(float(d0['x_s']), float(d0['y_s']))
        v0n = math.hypot(float(d0['xdot_s']), float(d0['ydot_s']))
        vmax_i = max(1.0, v0n)
        t_move = min(k * T_a, N_u * T_u)
        return math.hypot(P[j][k][0], P[j][k][1]) <= x0n + vmax_i * t_move + R_I_circ + 1e-9

    def attacker_arrays(j, t_stop, delta_vals):
        kd = kDZ[j]
        p_arr = []; q_arr = []; a_arr = []; g_arr = []
        d_arr = [[0.0] * (N_a + 1) for _ in range(N_D)]
        if kd is not None and 1 <= kd <= N_a:
            for k in range(N_a + 1):
                kk = k if k < t_stop else t_stop
                p_arr.append(P[j][kk][0]); q_arr.append(P[j][kk][1])
                a_arr.append(1.0 if k <= t_stop else 0.0)
                g_arr.append(1.0 if (k >= kd and t_stop == kd) else 0.0)
            for (i, k), v in delta_vals.items():
                if 0 <= i < N_D and 0 <= k <= N_a:
                    d_arr[i][k] = float(v)
        elif kd == 0:
            for k in range(N_a + 1):
                p_arr.append(P[j][0][0]); q_arr.append(P[j][0][1])
                a_arr.append(1.0 if k == 0 else 0.0)
                g_arr.append(1.0)
        else:
            for k in range(N_a + 1):
                p_arr.append(P[j][k][0]); q_arr.append(P[j][k][1])
                a_arr.append(1.0); g_arr.append(0.0)
        return p_arr, q_arr, a_arr, g_arr, d_arr

    def fallback_solution():
        sol = {'objective_value': 0.0, 'defenders': [], 'attackers': []}
        for i in range(N_D):
            d0 = defenders[i]
            xs = [float(d0['x_s'])]; ys = [float(d0['y_s'])]
            xds = [float(d0['xdot_s'])]; yds = [float(d0['ydot_s'])]
            for _ in range(N_u):
                xs.append(xs[-1] + Bv * xds[-1])
                ys.append(ys[-1] + Bv * yds[-1])
                xds.append(E * xds[-1])
                yds.append(E * yds[-1])
            sol['defenders'].append({'x': xs, 'y': ys, 'xdot': xds, 'ydot': yds,
                                     'ux': [0.0] * N_u, 'uy': [0.0] * N_u})
        total_gamma = 0.0
        for j in range(N_A):
            kd = kDZ[j]
            if kd is not None and 1 <= kd <= N_a:
                t_stop = kd
            elif kd == 0:
                t_stop = 0
            else:
                t_stop = N_a
            p_arr, q_arr, a_arr, g_arr, d_arr = attacker_arrays(j, t_stop, {})
            total_gamma += sum(g_arr[1:])
            sol['attackers'].append({'p': p_arr, 'q': q_arr, 'a': a_arr,
                                     'gamma': g_arr, 'delta': d_arr})
        sol['objective_value'] = float(total_gamma)
        return sol

    final_sol = None
    try:
        model = gp.Model('defense_zone')
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        INF = GRB.INFINITY
        x = {}; y = {}; xd = {}; yd = {}; ux = {}; uy = {}
        for i in range(N_D):
            for k in range(N_u + 1):
                x[i, k] = model.addVar(lb=-INF, name=f'x_{i}_{k}')
                y[i, k] = model.addVar(lb=-INF, name=f'y_{i}_{k}')
                xd[i, k] = model.addVar(lb=-INF, name=f'xd_{i}_{k}')
                yd[i, k] = model.addVar(lb=-INF, name=f'yd_{i}_{k}')
            for k in range(N_u):
                ux[i, k] = model.addVar(lb=-1.0, ub=1.0, name=f'ux_{i}_{k}')
                uy[i, k] = model.addVar(lb=-1.0, ub=1.0, name=f'uy_{i}_{k}')

        ex = {}; ey = {}
        if eps > 0.0:
            for i in range(N_D):
                for k in range(N_u):
                    ex[i, k] = model.addVar(lb=0.0, name=f'ex_{i}_{k}')
                    ey[i, k] = model.addVar(lb=0.0, name=f'ey_{i}_{k}')

        ovar = {}
        for i in range(N_D):
            for k in range(1, N_u + 1):
                for m in range(4):
                    ovar[i, k, m] = model.addVar(vtype=GRB.BINARY, name=f'o_{i}_{k}_{m}')

        avar = {}; dvar = {}
        for j in var_att:
            kd = kDZ[j]
            for k in range(kd + 1):
                avar[j, k] = model.addVar(vtype=GRB.BINARY, name=f'a_{j}_{k}')
            for i in range(N_D):
                for k in range(kd):
                    if can_reach(i, j, k):
                        dvar[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'd_{i}_{j}_{k}')

        dvar_list = {j: [] for j in var_att}
        for (i, j, k), dv in dvar.items():
            dvar_list[j].append((i, k, dv))

        # Initial conditions and linear defender dynamics
        for i in range(N_D):
            d0 = defenders[i]
            model.addConstr(x[i, 0] == float(d0['x_s']))
            model.addConstr(y[i, 0] == float(d0['y_s']))
            model.addConstr(xd[i, 0] == float(d0['xdot_s']))
            model.addConstr(yd[i, 0] == float(d0['ydot_s']))
            for k in range(N_u):
                model.addConstr(xd[i, k + 1] == E * xd[i, k] + Bv * ux[i, k])
                model.addConstr(yd[i, k + 1] == E * yd[i, k] + Bv * uy[i, k])
                model.addConstr(x[i, k + 1] == x[i, k] + Bv * xd[i, k] + Bx * ux[i, k])
                model.addConstr(y[i, k + 1] == y[i, k] + Bv * yd[i, k] + Bx * uy[i, k])

        # Control input polygon (inscribed in unit disk)
        for i in range(N_D):
            for k in range(N_u):
                for (c, s) in u_norm:
                    model.addConstr(c * ux[i, k] + s * uy[i, k] <= u_rhs)

        # Control effort absolute values
        if eps > 0.0:
            for i in range(N_D):
                for k in range(N_u):
                    model.addConstr(ex[i, k] >= ux[i, k])
                    model.addConstr(ex[i, k] >= -ux[i, k])
                    model.addConstr(ey[i, k] >= uy[i, k])
                    model.addConstr(ey[i, k] >= -uy[i, k])

        # Obstacle avoidance: outside at least one half-space of the 4-sided polygon
        for i in range(N_D):
            for k in range(1, N_u + 1):
                model.addConstr(gp.quicksum(ovar[i, k, m] for m in range(4)) >= 1)
                for m in range(4):
                    c, s = obs_norm[m]
                    model.addConstr(c * x[i, k] + s * y[i, k]
                                    >= R_dz - M_obs * (1.0 - ovar[i, k, m]))

        # Attacker activity / interception logic
        for j in var_att:
            kd = kDZ[j]
            model.addConstr(avar[j, 0] == 1.0)
            for k in range(kd):
                dks = [dvar[i, j, k] for i in range(N_D) if (i, j, k) in dvar]
                model.addConstr(avar[j, k + 1] <= avar[j, k])
                for dv in dks:
                    model.addConstr(avar[j, k + 1] <= 1.0 - dv)
                if dks:
                    model.addConstr(avar[j, k + 1] >= avar[j, k] - gp.quicksum(dks))
                else:
                    model.addConstr(avar[j, k + 1] >= avar[j, k])
            for i in range(N_D):
                for k in range(kd):
                    if (i, j, k) not in dvar:
                        continue
                    dv = dvar[i, j, k]
                    model.addConstr(dv <= avar[j, k])
                    cst, f = interp[k]
                    Px, Py = P[j][k]
                    if cst >= N_u:
                        Dx = x[i, N_u]; Dy = y[i, N_u]
                    elif f <= 1e-12:
                        Dx = x[i, cst]; Dy = y[i, cst]
                    else:
                        Dx = (1.0 - f) * x[i, cst] + f * x[i, cst + 1]
                        Dy = (1.0 - f) * y[i, cst] + f * y[i, cst + 1]
                    for (c, s) in I_norm:
                        model.addConstr(c * Dx + s * Dy
                                        >= c * Px + s * Py - R_I - M_delta * (1.0 - dv))

        # Objective: DZ occupation steps + weighted control effort
        obj = gp.LinExpr()
        for j in var_att:
            obj += W[j] * avar[j, kDZ[j]]
        if eps > 0.0:
            for i in range(N_D):
                for k in range(N_u):
                    obj += eps * ex[i, k] + eps * ey[i, k]
        model.setObjective(obj, GRB.MINIMIZE)

        def build_solution(getval):
            sol = {'objective_value': 0.0, 'defenders': [], 'attackers': []}
            total_effort = 0.0
            for i in range(N_D):
                xs = [getval(x[i, k]) for k in range(N_u + 1)]
                ys = [getval(y[i, k]) for k in range(N_u + 1)]
                xds = [getval(xd[i, k]) for k in range(N_u + 1)]
                yds = [getval(yd[i, k]) for k in range(N_u + 1)]
                uxs = [getval(ux[i, k]) for k in range(N_u)]
                uys = [getval(uy[i, k]) for k in range(N_u)]
                total_effort += sum(abs(v) for v in uxs) + sum(abs(v) for v in uys)
                sol['defenders'].append({'x': xs, 'y': ys, 'xdot': xds, 'ydot': yds,
                                         'ux': uxs, 'uy': uys})
            total_gamma = 0.0
            for j in range(N_A):
                kd = kDZ[j]
                if kd is not None and 1 <= kd <= N_a:
                    t_stop = 0
                    for k in range(kd + 1):
                        if getval(avar[j, k]) > 0.5:
                            t_stop = k
                    delta_vals = {}
                    for (i, k, dv) in dvar_list[j]:
                        if getval(dv) > 0.5:
                            delta_vals[(i, k)] = 1.0
                    p_arr, q_arr, a_arr, g_arr, d_arr = attacker_arrays(j, t_stop, delta_vals)
                else:
                    p_arr, q_arr, a_arr, g_arr, d_arr = attacker_arrays(j, 0, {})
                total_gamma += sum(g_arr[1:])
                sol['attackers'].append({'p': p_arr, 'q': q_arr, 'a': a_arr,
                                         'gamma': g_arr, 'delta': d_arr})
            sol['objective_value'] = float(total_gamma + eps * total_effort)
            return sol

        allvars = (list(x.values()) + list(y.values()) + list(xd.values()) + list(yd.values())
                   + list(ux.values()) + list(uy.values())
                   + list(avar.values()) + list(dvar.values()))

        def incumbent_cb(m, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    vals = m.cbGetSolution(allvars)
                    vmap = dict(zip(allvars, vals))
                    sol = build_solution(lambda v: vmap[v])
                    logger.log_solution(sol['objective_value'], sol)
                except Exception:
                    pass

        remaining = args.time_limit - (time.time() - t0)
        model.Params.TimeLimit = max(1.0, remaining - 0.5)
        if logger is not None:
            model.optimize(incumbent_cb)
        else:
            model.optimize()
        if model.SolCount > 0:
            final_sol = build_solution(lambda v: v.X)
    except Exception:
        final_sol = None

    if final_sol is None:
        final_sol = fallback_solution()

    if logger is not None:
        try:
            logger.log_solution(final_sol['objective_value'], final_sol)
        except Exception:
            pass

    with open(args.solution_path, 'w') as f:
        json.dump(final_sol, f)


if __name__ == '__main__':
    main()