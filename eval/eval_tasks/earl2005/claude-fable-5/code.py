import argparse
import json
import math
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t0 = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    N_D = inst['N_D']
    N_A = inst['N_A']
    eps = float(inst['epsilon'])
    M_u = inst['M_u']
    M_I = inst['M_I']
    M_dz = inst['M_dz']
    N_u = inst['N_u']
    N_a = inst['N_a']
    R_dz = float(inst['R_dz'])
    R_I = float(inst['R_I'])
    T_a = float(inst['T_a'])
    T_u = float(inst['T_u'])
    defenders = inst['defenders']
    attackers = inst['attackers']

    A = math.exp(-T_u)
    Bv = 1.0 - A
    Bx = T_u - 1.0 + A

    def normals(M):
        return [(math.cos(2.0 * math.pi * m / M), math.sin(2.0 * math.pi * m / M))
                for m in range(M)]

    n_dz = normals(M_dz)
    n_I = normals(M_I)
    n_u = normals(M_u)
    n_ob = normals(4)
    rhs_u = math.cos(math.pi / M_u)
    TOL = 1e-9

    # ---------- nominal attacker paths (constant velocity, never frozen) ----------
    paths = []
    INlist = []
    Kent = []
    for at in attackers:
        p = at['p_s']
        q = at['q_s']
        P = [(p, q)]
        for _ in range(N_a):
            p = p + at['v_p'] * T_a
            q = q + at['v_q'] * T_a
            P.append((p, q))
        paths.append(P)
        inl = [1 if all(cx * px + cy * qy <= R_dz + TOL for cx, cy in n_dz) else 0
               for (px, qy) in P]
        INlist.append(inl)
        K = None
        for k, v in enumerate(inl):
            if v:
                K = k
                break
        Kent.append(K)

    # ---------- interpolation coefficients at attacker sample times ----------
    def interp(t):
        j = int(math.floor(t / T_u + 1e-9))
        if j >= N_u:
            return (N_u, 0.0, 0.0)
        tau = t - j * T_u
        if tau < 1e-12:
            return (j, 0.0, 0.0)
        e = math.exp(-tau)
        return (j, 1.0 - e, tau - 1.0 + e)

    samp = [interp(k * T_a) for k in range(N_a + 1)]

    # ---------- exact simulation given control inputs ----------
    def simulate(u_all):
        st = []
        for i, d in enumerate(defenders):
            x = [d['x_s']]; y = [d['y_s']]
            vx = [d['xdot_s']]; vy = [d['ydot_s']]
            for js in range(N_u):
                ux_, uy_ = u_all[i][js]
                x.append(x[-1] + Bv * vx[-1] + Bx * ux_)
                y.append(y[-1] + Bv * vy[-1] + Bx * uy_)
                vx.append(A * vx[-1] + Bv * ux_)
                vy.append(A * vy[-1] + Bv * uy_)
            st.append((x, y, vx, vy))
        dpos = []
        for i in range(N_D):
            x, y, vx, vy = st[i]
            pos = []
            for k in range(N_a + 1):
                j, a1, a2 = samp[k]
                if j >= N_u:
                    pos.append((x[N_u], y[N_u]))
                else:
                    ux_, uy_ = u_all[i][j]
                    pos.append((x[j] + a1 * vx[j] + a2 * ux_,
                                y[j] + a1 * vy[j] + a2 * uy_))
            dpos.append(pos)
        pen = 0.0
        att_out = []
        for ja, at in enumerate(attackers):
            p = [at['p_s']]; q = [at['q_s']]
            avec = []; gvec = []
            dvec = [[] for _ in range(N_D)]
            active = 1
            for k in range(N_a + 1):
                if k > 0:
                    p.append(p[k - 1] + at['v_p'] * T_a * avec[k - 1])
                    q.append(q[k - 1] + at['v_q'] * T_a * avec[k - 1])
                inside = 1 if all(cx * p[k] + cy * q[k] <= R_dz + TOL
                                  for cx, cy in n_dz) else 0
                anyd = 0
                for i in range(N_D):
                    dx = p[k] - dpos[i][k][0]
                    dy = q[k] - dpos[i][k][1]
                    dl = 1 if all(cx * dx + cy * dy <= R_I + TOL
                                  for cx, cy in n_I) else 0
                    dvec[i].append(float(dl))
                    if dl:
                        anyd = 1
                gvec.append(float(inside))
                act = 1 if (active and not inside and not anyd) else 0
                avec.append(float(act))
                active = act
            pen += sum(gvec[1:])
            att_out.append({'p': [float(v) for v in p],
                            'q': [float(v) for v in q],
                            'a': avec, 'gamma': gvec, 'delta': dvec})
        effort = sum(abs(u_all[i][js][c]) for i in range(N_D)
                     for js in range(N_u) for c in (0, 1))
        obj = pen + eps * effort
        def_out = []
        for i in range(N_D):
            x, y, vx, vy = st[i]
            def_out.append({'x': [float(v) for v in x],
                            'y': [float(v) for v in y],
                            'xdot': [float(v) for v in vx],
                            'ydot': [float(v) for v in vy],
                            'ux': [float(u_all[i][js][0]) for js in range(N_u)],
                            'uy': [float(u_all[i][js][1]) for js in range(N_u)]})
        return obj, {'objective_value': float(obj),
                     'defenders': def_out, 'attackers': att_out}

    def clip_u(ux_, uy_):
        mval = max(cx * ux_ + cy * uy_ for cx, cy in n_u)
        if mval > rhs_u and mval > 0.0:
            s = rhs_u / mval
            ux_ *= s
            uy_ *= s
        ux_ = max(-1.0, min(1.0, ux_))
        uy_ = max(-1.0, min(1.0, uy_))
        return ux_, uy_

    def obstacle_ok(sol):
        for d in sol['defenders']:
            for j in range(1, N_u + 1):
                if not any(cx * d['x'][j] + cy * d['y'][j] >= R_dz - 1e-9
                           for cx, cy in n_ob):
                    return False
        return True

    state = {'obj': float('inf'), 'sol': None}

    def consider(obj, sol):
        if obj < state['obj'] - 1e-12:
            state['obj'] = obj
            state['sol'] = sol
            if logger:
                try:
                    logger.log_solution(float(obj), sol)
                except Exception:
                    pass

    # fallback: zero control
    u0 = [[(0.0, 0.0) for _ in range(N_u)] for _ in range(N_D)]
    fb_obj, fb_sol = simulate(u0)
    fb_feasible = obstacle_ok(fb_sol)
    if fb_feasible:
        consider(fb_obj, fb_sol)

    # ---------- build MILP ----------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model('defense')
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.IntFeasTol = 1e-7
        m.Params.FeasibilityTol = 1e-8

        Vb = 1.0
        Pinit = 1.0
        for d in defenders:
            Vb = max(Vb, abs(d['xdot_s']), abs(d['ydot_s']))
            Pinit = max(Pinit, abs(d['x_s']), abs(d['y_s']))
        Pb = Pinit + N_u * T_u * Vb + 1.0
        maxA = 1.0
        for P in paths:
            for (px, qy) in P:
                maxA = max(maxA, abs(px), abs(qy))
        Mob = R_dz + Pb + 1.0
        Mint = maxA + Pb + R_I + 1.0

        X = {}; Y = {}; VX = {}; VY = {}; UX = {}; UY = {}
        for i in range(N_D):
            for j in range(N_u + 1):
                X[i, j] = m.addVar(lb=-Pb, ub=Pb)
                Y[i, j] = m.addVar(lb=-Pb, ub=Pb)
                VX[i, j] = m.addVar(lb=-Vb, ub=Vb)
                VY[i, j] = m.addVar(lb=-Vb, ub=Vb)
            for j in range(N_u):
                UX[i, j] = m.addVar(lb=-1.0, ub=1.0)
                UY[i, j] = m.addVar(lb=-1.0, ub=1.0)

        obj_expr = gp.LinExpr()
        for i, d in enumerate(defenders):
            m.addConstr(X[i, 0] == d['x_s'])
            m.addConstr(Y[i, 0] == d['y_s'])
            m.addConstr(VX[i, 0] == d['xdot_s'])
            m.addConstr(VY[i, 0] == d['ydot_s'])
            for j in range(N_u):
                m.addConstr(X[i, j + 1] == X[i, j] + Bv * VX[i, j] + Bx * UX[i, j])
                m.addConstr(Y[i, j + 1] == Y[i, j] + Bv * VY[i, j] + Bx * UY[i, j])
                m.addConstr(VX[i, j + 1] == A * VX[i, j] + Bv * UX[i, j])
                m.addConstr(VY[i, j + 1] == A * VY[i, j] + Bv * UY[i, j])
                for cx, cy in n_u:
                    m.addConstr(cx * UX[i, j] + cy * UY[i, j] <= rhs_u)
                if eps > 0:
                    sx = m.addVar(lb=0.0)
                    sy = m.addVar(lb=0.0)
                    m.addConstr(sx >= UX[i, j]); m.addConstr(sx >= -UX[i, j])
                    m.addConstr(sy >= UY[i, j]); m.addConstr(sy >= -UY[i, j])
                    obj_expr += eps * (sx + sy)
            # obstacle avoidance at positive control-step boundaries
            for j in range(1, N_u + 1):
                obins = []
                for cx, cy in n_ob:
                    o = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(cx * X[i, j] + cy * Y[i, j] >=
                                R_dz + 1e-4 - Mob * (1 - o))
                    obins.append(o)
                m.addConstr(gp.quicksum(obins) >= 1)

        def reach_bound(i, t):
            v0 = math.hypot(defenders[i]['xdot_s'], defenders[i]['ydot_s'])
            e = math.exp(-t)
            return v0 * (1.0 - e) + (t - 1.0 + e)

        Rcirc = R_I / math.cos(math.pi / M_I) if M_I >= 3 else R_I

        const_cost = 0.0
        for ja in range(N_A):
            K = Kent[ja]
            if K is None:
                continue
            if K == 0:
                const_cost += float(N_a)
                continue
            # already intercepted at step 0 by initial defender positions?
            P0 = paths[ja][0]
            int0 = False
            for i in range(N_D):
                dx = P0[0] - defenders[i]['x_s']
                dy = P0[1] - defenders[i]['y_s']
                if all(cx * dx + cy * dy <= R_I + TOL for cx, cy in n_I):
                    int0 = True
                    break
            if int0:
                continue
            w = float(N_a - K + 1)
            cand = []
            for k in range(1, K):
                t = k * T_a
                Pk = paths[ja][k]
                for i in range(N_D):
                    d0 = math.hypot(Pk[0] - defenders[i]['x_s'],
                                    Pk[1] - defenders[i]['y_s'])
                    if d0 <= reach_bound(i, t) + Rcirc + 0.05:
                        cand.append((i, k))
            if not cand:
                const_cost += w
                continue
            z = m.addVar(vtype=GRB.BINARY)
            dts = []
            for (i, k) in cand:
                dv = m.addVar(vtype=GRB.BINARY)
                j, a1, a2 = samp[k]
                if j >= N_u:
                    ex = X[i, N_u]
                    ey = Y[i, N_u]
                else:
                    ex = X[i, j] + a1 * VX[i, j] + a2 * UX[i, j]
                    ey = Y[i, j] + a1 * VY[i, j] + a2 * UY[i, j]
                Pk = paths[ja][k]
                for cx, cy in n_I:
                    m.addConstr(cx * Pk[0] + cy * Pk[1] - (cx * ex + cy * ey)
                                <= R_I - 1e-4 + Mint * (1 - dv))
                dts.append(dv)
            m.addConstr(z <= gp.quicksum(dts))
            obj_expr += w * (1 - z)
        obj_expr += const_cost
        m.setObjective(obj_expr, GRB.MINIMIZE)

        flat_u = []
        for i in range(N_D):
            for j in range(N_u):
                flat_u.append(UX[i, j])
                flat_u.append(UY[i, j])

        def extract_u(vals):
            u_all = []
            idx = 0
            for i in range(N_D):
                row = []
                for j in range(N_u):
                    ux_ = vals[idx]; uy_ = vals[idx + 1]
                    idx += 2
                    row.append(clip_u(ux_, uy_))
                u_all.append(row)
            return u_all

        def cbfun(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = model.cbGetSolution(flat_u)
                    u_all = extract_u(vals)
                    obj, sol = simulate(u_all)
                    consider(obj, sol)
                except Exception:
                    pass

        tl = args.time_limit - (time.time() - t0) - 3.0
        m.Params.TimeLimit = max(1.0, tl)
        m.optimize(cbfun)

        if m.SolCount > 0:
            vals = [v.X for v in flat_u]
            u_all = extract_u(vals)
            obj, sol = simulate(u_all)
            consider(obj, sol)
    except Exception:
        pass

    if state['sol'] is None:
        # last resort (may violate obstacle constraints, but output something)
        state['obj'] = fb_obj
        state['sol'] = fb_sol
        if logger:
            try:
                logger.log_solution(float(fb_obj), fb_sol)
            except Exception:
                pass

    with open(args.solution_path, 'w') as f:
        json.dump(state['sol'], f)


if __name__ == '__main__':
    main()