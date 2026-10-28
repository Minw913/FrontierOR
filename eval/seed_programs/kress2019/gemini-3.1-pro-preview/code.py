import json
import logging
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    S = data['S']
    n = data['n']
    m = data['m']
    p = data['p']
    sigma_w = data['sigma_w']
    sigma_l = data['sigma_l']

    seaside_containers = data.get('seaside_containers', [])
    landside_containers = data.get('landside_containers', [])

    deadline_groups = defaultdict(list)
    for c in landside_containers:
        deadline_groups[c['deadline']].append(c['id'])

    sorted_deadlines = sorted(deadline_groups.keys())
    g = len(sorted_deadlines)

    D_v = sorted_deadlines
    G_v = [deadline_groups[d] for d in sorted_deadlines]

    ls_src = {c['id']: c['source_slot'] for c in landside_containers}
    ls_eft = {c['id']: c['earliest_finish_time'] for c in landside_containers}
    ls_dl = {c['id']: c['deadline'] for c in landside_containers}

    target = [c['target_slot'] for c in seaside_containers]
    landside_cid_to_idx = {c['id']: idx for idx, c in enumerate(landside_containers)}

    t_w, t_l = 0, 0
    pos_w, pos_l = sigma_w, sigma_l
    for c in seaside_containers:
        t_w += abs(pos_w - 0) + p
        pos_w = 0
        tgt = c['target_slot']
        drop_pos = tgt if tgt <= S else S
        t_w += abs(pos_w - drop_pos) + p
        pos_w = drop_pos
        if tgt > S:
            t_l = max(t_w, t_l + abs(pos_l - S)) + p
            pos_l = S
            t_l += abs(pos_l - (S+1)) + p
            pos_l = S+1

    heur_T = max(t_w, t_l)
    T_bound = int(heur_T)
    for c in landside_containers:
        T_bound += 2 * S + 2 * p + 10
    actual_max_EFT = max([c['earliest_finish_time'] for c in landside_containers] + [0])
    T_bound = max(T_bound, actual_max_EFT + 50)
    T_bound = min(T_bound, 3000)

    model = gp.Model()
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('MIPGap', 1e-4)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj)
            except Exception:
                pass

    w = model.addVars(T_bound, vtype=GRB.INTEGER, lb=0, ub=S, name="w")
    l = model.addVars(T_bound, vtype=GRB.INTEGER, lb=1, ub=S+1, name="l")

    w_lift = model.addVars(n, T_bound-p+1, vtype=GRB.BINARY, name="w_lift")
    w_drop = model.addVars(n, T_bound-p+1, vtype=GRB.BINARY, name="w_drop")
    drop_slot_w = model.addVars(n, vtype=GRB.INTEGER, lb=1, ub=S, name="drop_slot_w")

    U = model.addVars(n, vtype=GRB.BINARY, name="U")
    l_ss_lift = model.addVars(n, T_bound-p+1, vtype=GRB.BINARY, name="l_ss_lift")
    l_ss_drop = model.addVars(n, T_bound-p+1, vtype=GRB.BINARY, name="l_ss_drop")

    Z = model.addVars(m, vtype=GRB.BINARY, name="Z")
    l_ls_lift = model.addVars(m, T_bound-p+1, vtype=GRB.BINARY, name="l_ls_lift")
    l_ls_drop = model.addVars(m, T_bound-p+1, vtype=GRB.BINARY, name="l_ls_drop")

    C_max = model.addVar(vtype=GRB.INTEGER, lb=0, name="C_max")
    B = model.addVars(g, vtype=GRB.BINARY, name="B")

    M = S + 5
    M_T = max(T_bound + 100, actual_max_EFT + 100)

    model.addConstr(w[0] == sigma_w)
    model.addConstr(l[0] == sigma_l)
    for t in range(T_bound):
        model.addConstr(w[t] <= l[t] - 1)
        if t < T_bound - 1:
            model.addConstr(w[t+1] - w[t] <= 1)
            model.addConstr(w[t+1] - w[t] >= -1)
            model.addConstr(l[t+1] - l[t] <= 1)
            model.addConstr(l[t+1] - l[t] >= -1)

    for i in range(n):
        model.addConstr(w_lift.sum(i, '*') == 1)
        model.addConstr(w_drop.sum(i, '*') == 1)
        
        start_w_l = gp.quicksum(t * w_lift[i, t] for t in range(T_bound-p+1))
        start_w_d = gp.quicksum(t * w_drop[i, t] for t in range(T_bound-p+1))
        model.addConstr(start_w_d >= start_w_l + p)
        if i < n - 1:
            start_w_l_next = gp.quicksum(t * w_lift[i+1, t] for t in range(T_bound-p+1))
            model.addConstr(start_w_l_next >= start_w_d + p)

        for t in range(T_bound - p + 1):
            for tau in range(p):
                model.addConstr(w[t+tau] <= M * (1 - w_lift[i, t]))
                model.addConstr(w[t+tau] >= drop_slot_w[i] - M * (1 - w_drop[i, t]))
                model.addConstr(w[t+tau] <= drop_slot_w[i] + M * (1 - w_drop[i, t]))

        tgt = target[i]
        if tgt <= S:
            model.addConstr(drop_slot_w[i] >= tgt - M * U[i])
            model.addConstr(drop_slot_w[i] <= tgt + M * U[i])
        else:
            model.addConstr(U[i] == 1)

    for i in range(n):
        model.addConstr(l_ss_lift.sum(i, '*') == U[i])
        model.addConstr(l_ss_drop.sum(i, '*') == U[i])
        
        start_l_l_ss = gp.quicksum(t * l_ss_lift[i, t] for t in range(T_bound-p+1))
        start_l_d_ss = gp.quicksum(t * l_ss_drop[i, t] for t in range(T_bound-p+1))
        start_w_d_ss = gp.quicksum(t * w_drop[i, t] for t in range(T_bound-p+1))
        
        model.addConstr(start_l_l_ss >= start_w_d_ss - M_T * (1 - U[i]))
        model.addConstr(start_l_d_ss >= start_l_l_ss + p - M_T * (1 - U[i]))

        for t in range(T_bound - p + 1):
            for tau in range(p):
                model.addConstr(l[t+tau] >= drop_slot_w[i] - M * (1 - l_ss_lift[i, t]))
                model.addConstr(l[t+tau] <= drop_slot_w[i] + M * (1 - l_ss_lift[i, t]))
                model.addConstr(l[t+tau] >= target[i] - M * (1 - l_ss_drop[i, t]))
                model.addConstr(l[t+tau] <= target[i] + M * (1 - l_ss_drop[i, t]))

    for idx, c in enumerate(landside_containers):
        cid = c['id']
        model.addConstr(l_ls_lift.sum(idx, '*') == Z[idx])
        model.addConstr(l_ls_drop.sum(idx, '*') == Z[idx])
        
        start_ll = gp.quicksum(t * l_ls_lift[idx, t] for t in range(T_bound-p+1))
        start_ld = gp.quicksum(t * l_ls_drop[idx, t] for t in range(T_bound-p+1))
        finish_ld = gp.quicksum((t+p) * l_ls_drop[idx, t] for t in range(T_bound-p+1))
        
        model.addConstr(start_ld >= start_ll + p - M_T * (1 - Z[idx]))
        model.addConstr(finish_ld <= ls_dl[cid] + M_T * (1 - Z[idx]))
        model.addConstr(finish_ld >= ls_eft[cid] - M_T * (1 - Z[idx]))
        
        src = ls_src[cid]
        for t in range(T_bound - p + 1):
            for tau in range(p):
                model.addConstr(l[t+tau] >= src - M * (1 - l_ls_lift[idx, t]))
                model.addConstr(l[t+tau] <= src + M * (1 - l_ls_lift[idx, t]))
                model.addConstr(l[t+tau] >= (S+1) - M * (1 - l_ls_drop[idx, t]))

    for t in range(T_bound):
        actives = []
        for i in range(n):
            actives.append(
                gp.quicksum(l_ss_lift[i, tau] for tau in range(min(t, T_bound-p)+1)) -
                gp.quicksum(l_ss_drop[i, tau] for tau in range(min(t-p, T_bound-p)+1) if tau >= 0)
            )
        for idx in range(m):
            actives.append(
                gp.quicksum(l_ls_lift[idx, tau] for tau in range(min(t, T_bound-p)+1)) -
                gp.quicksum(l_ls_drop[idx, tau] for tau in range(min(t-p, T_bound-p)+1) if tau >= 0)
            )
        model.addConstr(gp.quicksum(actives) <= 1)

    for i in range(n):
        end_w = gp.quicksum((t+p) * w_drop[i, t] for t in range(T_bound-p+1))
        end_l = gp.quicksum((t+p) * l_ss_drop[i, t] for t in range(T_bound-p+1))
        model.addConstr(C_max >= end_w - M_T * U[i])
        model.addConstr(C_max >= end_l - M_T * (1 - U[i]))

    for v in range(g):
        model.addConstr(C_max >= D_v[v] - M_T * (1 - B[v]))
        model.addConstr(C_max <= D_v[v] - 1 + M_T * B[v])
        
        group_indices = [landside_cid_to_idx[cid] for cid in G_v[v]]
        for idx in group_indices:
            model.addConstr(Z[idx] >= B[v])
        
        rhs_expr = 1 - B[v]
        for u in range(v):
            rhs_expr -= (1 - B[u])
        model.addConstr(gp.quicksum(Z[idx] for idx in group_indices) >= rhs_expr)

    model.setObjective(C_max, GRB.MINIMIZE)
    model.optimize(cb)

    if model.SolCount > 0:
        w_pos = {str(t): int(round(w[t].X)) for t in range(T_bound)}
        l_pos = {str(t): int(round(l[t].X)) for t in range(T_bound)}

        max_t = int(round(C_max.X))
        for idx in range(m):
            if Z[idx].X > 0.5:
                ft = int(round(sum((t+p)*l_ls_drop[idx, t].X for t in range(T_bound-p+1))))
                max_t = max(max_t, ft)
        for i in range(n):
            if U[i].X > 0.5:
                ft = int(round(sum((t+p)*l_ss_drop[i, t].X for t in range(T_bound-p+1))))
                max_t = max(max_t, ft)

        w_pos_out = {str(t): w_pos[str(t)] for t in range(max_t+1)}
        l_pos_out = {str(t): l_pos[str(t)] for t in range(max_t+1)}

        seaside_events = []
        for i in range(n):
            c_id = seaside_containers[i]['id']
            tgt = seaside_containers[i]['target_slot']
            w_d_val = int(round(sum(t * w_drop[i, t].X for t in range(T_bound-p+1))))
            d_s_val = int(round(drop_slot_w[i].X))
            
            event = {
                "container_id": c_id,
                "target_slot": tgt,
                "seaside_drop_time": w_d_val,
                "seaside_drop_slot": d_s_val
            }
            if U[i].X > 0.5:
                l_s_l_val = int(round(sum(t * l_ss_lift[i, t].X for t in range(T_bound-p+1))))
                l_s_d_val = int(round(sum(t * l_ss_drop[i, t].X for t in range(T_bound-p+1))))
                event["landside_lift_time"] = l_s_l_val
                event["landside_lift_slot"] = d_s_val
                event["landside_drop_time"] = l_s_d_val
            seaside_events.append(event)

        landside_events = []
        for idx, c in enumerate(landside_containers):
            if Z[idx].X > 0.5:
                c_id = c['id']
                src = c['source_slot']
                l_l_val = int(round(sum(t * l_ls_lift[idx, t].X for t in range(T_bound-p+1))))
                l_d_val = int(round(sum(t * l_ls_drop[idx, t].X for t in range(T_bound-p+1))))
                event = {
                    "container_id": c_id,
                    "source_slot": src,
                    "lift_time": l_l_val,
                    "drop_time": l_d_val
                }
                landside_events.append(event)

        out_data = {
            "objective_value": float(C_max.X),
            "crane_w_positions": w_pos_out,
            "crane_l_positions": l_pos_out,
            "seaside_events": seaside_events,
            "landside_events": landside_events
        }

        with open(args.solution_path, 'w') as f:
            json.dump(out_data, f, indent=4)

if __name__ == '__main__':
    main()