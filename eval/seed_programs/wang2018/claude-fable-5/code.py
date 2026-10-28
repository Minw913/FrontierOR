import argparse
import json
import time
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path) as f:
        data = json.load(f)

    V = data['num_vessels']
    B = data['num_berths']
    S = data['num_subblocks']
    H = data['H']
    E = data['E']
    co = data['penalty_cost_coefficient_co']
    cap = data['QC_capacity_per_timestep']
    DL = data['loading_route_lengths_D_L']
    DU = data['average_unloading_route_lengths_D_U']
    vessels = data['vessels']
    T = H + E

    # ------------------------------------------------------------------
    # Enumerate candidate (berth, profile, start) combos per vessel.
    # Each combo: (b, profile_idx, alpha, handling_time, qc_list, base_cost)
    # base_cost = penalty deviation cost + unloading transport cost
    # ------------------------------------------------------------------
    combos = []
    for v in range(V):
        vd = vessels[v]
        ftw = vd['feasible_time_window']
        etw = vd['expected_time_window']
        pen = vd['penalty_cost_coefficient']
        U = vd['unloading_containers']
        lst = []
        for pi, prof in enumerate(vd['qc_profiles']):
            h = prof['handling_time']
            q = prof['qc_per_timestep']
            tlo = max(1, ftw[0])
            thi = min(H, ftw[1] - h + 1, T - h + 1)
            for t in range(tlo, thi + 1):
                beta = t + h - 1
                penc = pen * (max(0, etw[0] - t) + max(0, beta - etw[1]))
                for b in range(B):
                    base = penc + co * DU[b] * U
                    lst.append((b, pi, t, h, q, base))
        combos.append(lst)

    for v in range(V):
        if not combos[v]:
            raise RuntimeError("Vessel %d has no feasible (berth,profile,time) option" % v)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def qc_check_full_ok(usage):
        for tau in range(1, H + 1):
            if tau <= E:
                if usage[tau] + usage[tau + H] > cap[tau - 1]:
                    return False
            else:
                if usage[tau] > cap[tau - 1]:
                    return False
        return True

    def build_solution(chosen, subs):
        """chosen: v -> combo tuple ; subs: v -> list of subblocks"""
        usage = [0] * (T + 2)
        obj = 0.0
        vout = []
        berth_group = {}
        for v in range(V):
            b, pi, t, h, q, base = chosen[v]
            beta = t + h - 1
            obj += base
            m_v = vessels[v]['reserved_subblocks']
            L = vessels[v]['loading_containers']
            sblist = sorted(subs.get(v, []))
            if m_v > 0:
                for s in sblist:
                    obj += co * DL[s][b] * L / m_v
            for i in range(h):
                usage[t + i] += q[i]
            vout.append({
                'vessel_id': vessels[v]['vessel_id'],
                'alpha': t,
                'beta': beta,
                'berth': b,
                'profile': pi,
                'subblocks': sblist,
                'handling_time': h,
                'qc_per_timestep': list(q),
            })
            berth_group.setdefault(b, []).append((t, beta))
        bout = []
        for b in sorted(berth_group.keys()):
            iv = berth_group[b]
            bout.append({
                'berth': b,
                'rho': min(a for a, _ in iv),
                'varsigma': max(e for _, e in iv),
            })
        qc = {str(tau): usage[tau] for tau in range(1, T + 1) if usage[tau] > 0}
        sol = {
            'objective_value': obj,
            'vessels': vout,
            'berths': bout,
            'qc_usage': qc,
        }
        return obj, sol

    # ------------------------------------------------------------------
    # Greedy heuristic (feasible start / fallback)
    # ------------------------------------------------------------------
    def try_greedy():
        order = sorted(range(V), key=lambda v: (vessels[v]['feasible_time_window'][0],
                                                -vessels[v]['penalty_cost_coefficient']))
        usage = [0] * (T + 2)
        berth_iv = {b: [] for b in range(B)}
        berth_span = {b: [None, None] for b in range(B)}
        chosen = {}
        if S > 0:
            meanDL = [sum(DL[s][b] for s in range(S)) / S for b in range(B)]
        else:
            meanDL = [0.0] * B
        for v in order:
            L = vessels[v]['loading_containers']
            cand = sorted(combos[v], key=lambda c: c[5] + co * meanDL[c[0]] * L)
            picked = None
            for (b, pi, t, h, q, base) in cand:
                beta = t + h - 1
                ok = True
                for (s2, e2) in berth_iv[b]:
                    if not (beta < s2 or t > e2):
                        ok = False
                        break
                if not ok:
                    continue
                mn, mx = berth_span[b]
                nmn = t if mn is None else min(mn, t)
                nmx = beta if mx is None else max(mx, beta)
                if nmx - nmn > H - 1:
                    continue
                for i in range(h):
                    usage[t + i] += q[i]
                feas = True
                for tau in range(t, beta + 1):
                    if tau <= E:
                        if usage[tau] + usage[tau + H] > cap[tau - 1]:
                            feas = False
                            break
                    elif tau <= H:
                        if usage[tau] > cap[tau - 1]:
                            feas = False
                            break
                    else:
                        t0_ = tau - H
                        if usage[t0_] + usage[tau] > cap[t0_ - 1]:
                            feas = False
                            break
                if not feas:
                    for i in range(h):
                        usage[t + i] -= q[i]
                    continue
                picked = (b, pi, t, h, q, base)
                berth_iv[b].append((t, beta))
                berth_span[b] = [nmn, nmx]
                break
            if picked is None:
                return None
            chosen[v] = picked
        # subblocks
        free = set(range(S))
        subs = {}
        vorder = sorted(range(V),
                        key=lambda v: -(vessels[v]['loading_containers'] /
                                        max(1, vessels[v]['reserved_subblocks'])))
        for v in vorder:
            m_v = vessels[v]['reserved_subblocks']
            if m_v <= 0:
                subs[v] = []
                continue
            b = chosen[v][0]
            avail = sorted(free, key=lambda s: DL[s][b])
            if len(avail) < m_v:
                return None
            pick = avail[:m_v]
            for s in pick:
                free.discard(s)
            subs[v] = pick
        return chosen, subs

    best = {'obj': float('inf'), 'sol': None}

    greedy_res = try_greedy()
    if greedy_res is not None:
        g_obj, g_sol = build_solution(greedy_res[0], greedy_res[1])
        best['obj'] = g_obj
        best['sol'] = g_sol
        if logger:
            logger.log_solution(g_obj, g_sol)

    # ------------------------------------------------------------------
    # MIP model
    # ------------------------------------------------------------------
    model = gp.Model("berth_yard")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = args.time_limit - (time.time() - t_start) - 3.0
    model.Params.TimeLimit = max(1.0, remaining)

    x = {}
    for v in range(V):
        for k, c in enumerate(combos[v]):
            x[(v, k)] = model.addVar(vtype=GRB.BINARY, obj=c[5], name="x_%d_%d" % (v, k))

    z = {}
    cvar = {}
    for v in range(V):
        m_v = vessels[v]['reserved_subblocks']
        if m_v > 0:
            for s in range(S):
                z[(v, s)] = model.addVar(vtype=GRB.BINARY, name="z_%d_%d" % (v, s))
                cvar[(v, s)] = model.addVar(lb=0.0, obj=1.0, name="c_%d_%d" % (v, s))

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # assignment
    for v in range(V):
        model.addConstr(gp.quicksum(x[(v, k)] for k in range(len(combos[v]))) == 1)

    # subblock count / capacity
    for v in range(V):
        m_v = vessels[v]['reserved_subblocks']
        if m_v > 0:
            model.addConstr(gp.quicksum(z[(v, s)] for s in range(S)) == m_v)
    for s in range(S):
        terms = [z[(v, s)] for v in range(V) if (v, s) in z]
        if terms:
            model.addConstr(gp.quicksum(terms) <= 1)

    # y expressions per (v, b)
    yexpr = {}
    berths_of = {v: set() for v in range(V)}
    for v in range(V):
        for k, c in enumerate(combos[v]):
            b = c[0]
            berths_of[v].add(b)
            if (v, b) not in yexpr:
                yexpr[(v, b)] = gp.LinExpr()
            yexpr[(v, b)].add(x[(v, k)])

    # loading cost linearization: c[v,s] >= coef*(z[v,s] + y[v,b] - 1)
    for v in range(V):
        m_v = vessels[v]['reserved_subblocks']
        if m_v <= 0:
            continue
        L = vessels[v]['loading_containers']
        for s in range(S):
            for b in berths_of[v]:
                coef = co * DL[s][b] * L / m_v
                if coef > 0:
                    model.addConstr(cvar[(v, s)] >= coef * (z[(v, s)] + yexpr[(v, b)] - 1))

    # berth occupancy (non-overlap) and QC usage expressions
    occ = {}
    use = {tau: gp.LinExpr() for tau in range(1, T + 1)}
    for v in range(V):
        for k, (b, pi, t, h, q, base) in enumerate(combos[v]):
            var = x[(v, k)]
            for i in range(h):
                tau = t + i
                occ.setdefault((b, tau), []).append(var)
                if q[i] > 0:
                    use[tau].add(var, q[i])
    for (b, tau), lst in occ.items():
        if len(lst) > 1:
            model.addConstr(gp.quicksum(lst) <= 1)

    # periodic QC capacity
    for tau in range(1, H + 1):
        if tau <= E:
            model.addConstr(use[tau] + use[tau + H] <= cap[tau - 1])
        else:
            model.addConstr(use[tau] <= cap[tau - 1])

    # berth span (wrap-around) constraints
    Mbig = T + 1
    rho = {}
    varsig = {}
    for b in range(B):
        rho[b] = model.addVar(lb=1, ub=T, name="rho_%d" % b)
        varsig[b] = model.addVar(lb=1, ub=T, name="vs_%d" % b)
        model.addConstr(varsig[b] - rho[b] <= H - 1)
    for v in range(V):
        alpha_expr = gp.quicksum(combos[v][k][2] * x[(v, k)] for k in range(len(combos[v])))
        beta_expr = gp.quicksum((combos[v][k][2] + combos[v][k][3] - 1) * x[(v, k)]
                                for k in range(len(combos[v])))
        for b in berths_of[v]:
            model.addConstr(rho[b] <= alpha_expr + Mbig * (1 - yexpr[(v, b)]))
            model.addConstr(varsig[b] >= beta_expr - Mbig * (1 - yexpr[(v, b)]))

    # warm start from greedy
    if greedy_res is not None:
        g_chosen, g_subs = greedy_res
        for v in range(V):
            gb, gpi, gt, gh, gq, gbase = g_chosen[v]
            for k, c in enumerate(combos[v]):
                x[(v, k)].Start = 1.0 if (c[0] == gb and c[1] == gpi and c[2] == gt) else 0.0
            m_v = vessels[v]['reserved_subblocks']
            if m_v > 0:
                chosen_set = set(g_subs[v])
                for s in range(S):
                    z[(v, s)].Start = 1.0 if s in chosen_set else 0.0

    xkeys = list(x.keys())
    xlist = [x[k] for k in xkeys]
    zkeys = list(z.keys())
    zlist = [z[k] for k in zkeys]

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = mdl.cbGetSolution(xlist)
                chosen = {}
                for (vk, val) in zip(xkeys, xv):
                    if val > 0.5:
                        chosen[vk[0]] = combos[vk[0]][vk[1]]
                if len(chosen) < V:
                    return
                subs = {v: [] for v in range(V)}
                if zlist:
                    zv = mdl.cbGetSolution(zlist)
                    for (vs_, val) in zip(zkeys, zv):
                        if val > 0.5:
                            subs[vs_[0]].append(vs_[1])
                obj, sol = build_solution(chosen, subs)
                if obj < best['obj'] - 1e-9:
                    best['obj'] = obj
                    best['sol'] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    try:
        model.optimize(callback)
    except Exception:
        pass

    # Extract final incumbent from model (in case callback missed anything)
    try:
        if model.SolCount > 0:
            chosen = {}
            for (vk, var) in zip(xkeys, xlist):
                if var.X > 0.5:
                    chosen[vk[0]] = combos[vk[0]][vk[1]]
            subs = {v: [] for v in range(V)}
            for (vs_, var) in zip(zkeys, zlist):
                if var.X > 0.5:
                    subs[vs_[0]].append(vs_[1])
            if len(chosen) == V:
                obj, sol = build_solution(chosen, subs)
                if obj < best['obj'] - 1e-9:
                    best['obj'] = obj
                    best['sol'] = sol
                    if logger:
                        logger.log_solution(obj, sol)
    except Exception:
        pass

    if best['sol'] is None:
        # No feasible solution found at all; emit an empty placeholder
        best['sol'] = {
            'objective_value': float('inf'),
            'vessels': [],
            'berths': [],
            'qc_usage': {},
        }
        sys.stderr.write("WARNING: no feasible solution found\n")

    with open(args.solution_path, 'w') as f:
        json.dump(best['sol'], f, indent=2)


if __name__ == '__main__':
    main()