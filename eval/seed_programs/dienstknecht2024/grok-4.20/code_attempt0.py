import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    params = instance['problem_parameters']
    containers = instance['containers']
    trades = instance['trades']

    T = params['num_periods']
    C = params['num_containers']
    Trades = len(trades)
    d_max = params['d_max']

    # Map ids to indices
    cont_id_to_idx = {cont['id']: i for i, cont in enumerate(containers)}
    trade_id_to_idx = {trade['id']: i for i, trade in enumerate(trades)}

    # Precompute data
    avail = [[False] * (T + 2) for _ in range(C)]  # 1-based periods
    for c_idx, cont in enumerate(containers):
        s = cont['availability_start']
        e = cont['availability_end']
        for t in range(s, e + 1):
            avail[c_idx][t] = True

    demand = []
    start_p = []
    end_p = []
    max_disp = []
    for tr in trades:
        demand.append(tr['container_demand'])
        start_p.append(tr['start_period'])
        end_p.append(tr['end_period'])
        max_disp.append(tr['max_dispersion'])

    # Adjacency list (indices)
    adj = [[] for _ in range(C)]
    for c_idx, cont in enumerate(containers):
        for aid in cont['adjacent_containers']:
            adj[c_idx].append(cont_id_to_idx[aid])

    # Build MIP
    m = gp.Model("ContainerAssignment")
    m.setParam(GRB.Param.TimeLimit, args.time_limit - 5)
    m.setParam(GRB.Param.Threads, 1)
    m.setParam(GRB.Param.MIPGap, 0.01)
    m.setParam(GRB.Param.LogToConsole, 0)

    # x[t,c,k] = 1 if container c assigned to trade k in period t
    x = {}
    for t in range(1, T + 1):
        for c in range(C):
            if not avail[c][t]:
                continue
            for k in range(Trades):
                if start_p[k] <= t <= end_p[k]:
                    x[t, c, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{t}_{c}_{k}")

    # y[t,k,g,c] = 1 if container c is used as a "group leader" for trade k in t, group g
    y = {}
    for t in range(1, T + 1):
        for k in range(Trades):
            if not (start_p[k] <= t <= end_p[k]):
                continue
            for g in range(max_disp[k]):
                for c in range(C):
                    if avail[c][t]:
                        y[t, k, g, c] = m.addVar(vtype=GRB.BINARY, name=f"y_{t}_{k}_{g}_{c}")

    # z[t,k,c] = 1 if container c is assigned to trade k at t AND is a group root
    # We use y for that

    # Objective: minimize sum of changes
    changes = {}
    for k in range(Trades):
        s = start_p[k]
        e = end_p[k]
        for t in range(s + 1, e + 1):
            for c in range(C):
                if avail[c][t] and avail[c][t - 1]:
                    changes[t, k, c] = m.addVar(vtype=GRB.BINARY, name=f"ch_{t}_{k}_{c}")
                elif avail[c][t] and not avail[c][t - 1]:
                    # Must count if assigned and new availability
                    pass

    obj_expr = gp.LinExpr()

    for k in range(Trades):
        s = start_p[k]
        e = end_p[k]
        for t in range(s + 1, e + 1):
            for c in range(C):
                if avail[c][t]:
                    if avail[c][t - 1]:
                        var = changes[t, k, c]
                        obj_expr.add(var)
                    else:
                        # If newly available and assigned, it counts as change
                        if (t, c, k) in x:
                            obj_expr.add(x[t, c, k])

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints

    # 1. Demand satisfaction
    for t in range(1, T + 1):
        for k in range(Trades):
            if not (start_p[k] <= t <= end_p[k]):
                continue
            lhs = gp.LinExpr()
            for c in range(C):
                if avail[c][t] and (t, c, k) in x:
                    lhs.add(x[t, c, k])
            m.addConstr(lhs == demand[k], name=f"demand_{t}_{k}")

    # 2. Container at most one trade
    for t in range(1, T + 1):
        for c in range(C):
            if not avail[c][t]:
                continue
            lhs = gp.LinExpr()
            for k in range(Trades):
                if (t, c, k) in x:
                    lhs.add(x[t, c, k])
            m.addConstr(lhs <= 1, name=f"atmostone_{t}_{c}")

    # 3. Dispersion constraints using y variables (group leaders)
    for t in range(1, T + 1):
        for k in range(Trades):
            if not (start_p[k] <= t <= end_p[k]):
                continue
            Dk = max_disp[k]
            # Each used container must be covered by exactly one leader
            for c in range(C):
                if not avail[c][t]:
                    continue
                if (t, c, k) not in x:
                    continue
                lhs = gp.LinExpr()
                for g in range(Dk):
                    lhs.add(y[t, k, g, c])
                m.addConstr(lhs == x[t, c, k], name=f"cover_{t}_{k}_{c}")

            # Each leader must be assigned to the trade
            for g in range(Dk):
                for c in range(C):
                    if avail[c][t]:
                        m.addConstr(y[t, k, g, c] <= x.get((t, c, k), 0), name=f"leader_assigned_{t}_{k}_{g}_{c}")

            # Flow / connectivity constraints to ensure at most Dk components
            # We use multi-commodity like or simple potential based method.
            # Here we use a simple but effective: each non-leader must have a "parent" toward a leader.
            # This requires additional variables.

    # Add parent variables for connectivity
    # parent[t,k,c1,c2] = 1 if c2 is parent of c1 for trade k at t
    parent = {}
    for t in range(1, T + 1):
        for k_idx in range(Trades):
            if not (start_p[k_idx] <= t <= end_p[k_idx]):
                continue
            for c in range(C):
                if not avail[c][t]:
                    continue
                for a in adj[c]:
                    if avail[a][t]:
                        parent[t, k_idx, c, a] = m.addVar(vtype=GRB.BINARY, name=f"p_{t}_{k_idx}_{c}_{a}")

    # Link parent to assignment and y
    for t in range(1, T + 1):
        for k in range(Trades):
            if not (start_p[k] <= t <= end_p[k]):
                continue
            Dk = max_disp[k]
            for c in range(C):
                if not avail[c][t] or (t, c, k) not in x:
                    continue
                # If assigned and not a leader, must have exactly 1 parent
                is_leader_sum = gp.LinExpr()
                for g in range(Dk):
                    is_leader_sum.add(y[t, k, g, c])
                m.addConstr(
                    gp.quicksum(parent.get((t, k, c, a), 0) for a in adj[c] if avail[a][t]) == 
                    (1 - is_leader_sum) * x[t, c, k],
                    name=f"parent_count_{t}_{k}_{c}"
                )
                # Cannot have parent if not assigned
                for a in adj[c]:
                    if avail[a][t] and (t, k, c, a) in parent:
                        m.addConstr(parent[t, k, c, a] <= x[t, c, k])
                        m.addConstr(parent[t, k, c, a] <= x[t, a, k])

    # No cycles: we use potential / ordering on containers
    # Assign each container an integer level
    level = {}
    for t in range(1, T + 1):
        for k in range(Trades):
            if not (start_p[k] <= t <= end_p[k]):
                continue
            for c in range(C):
                if avail[c][t]:
                    level[t, k, c] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=C, name=f"lev_{t}_{k}_{c}")

    for t in range(1, T + 1):
        for k in range(Trades):
            if not (start_p[k] <= t <= end_p[k]):
                continue
            for c in range(C):
                if not avail[c][t] or (t, c, k) not in x:
                    continue
                for a_idx, a in enumerate(adj[c]):
                    if avail[a][t] and (t, k, c, a) in parent:
                        pvar = parent[t, k, c, a]
                        m.addConstr(
                            level[t, k, c] >= level[t, k, a] + 1 - C * (1 - pvar),
                            name=f"level_inc_{t}_{k}_{c}_{a}"
                        )
                        m.addConstr(
                            level[t, k, c] <= level[t, k, a] + C + C * (1 - pvar),
                            name=f"level_bound_{t}_{k}_{c}_{a}"
                        )

    # Change detection
    for k in range(Trades):
        s = start_p[k]
        e = end_p[k]
        for t in range(s + 1, e + 1):
            for c in range(C):
                if avail[c][t] and avail[c][t - 1] and (t, c, k) in x and (t - 1, c, k) in x:
                    m.addConstr(
                        changes[t, k, c] >= x[t, c, k] - x[t - 1, c, k],
                        name=f"change_pos_{t}_{k}_{c}"
                    )
                    m.addConstr(
                        changes[t, k, c] >= x[t - 1, c, k] - x[t, c, k],
                        name=f"change_neg_{t}_{k}_{c}"
                    )

    # For new availability, we already added x itself to objective if newly available

    m.update()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Solve with callback to log improving solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if logger and obj < model._best_obj:
                model._best_obj = obj
                logger.log(float(obj))

    m._best_obj = float('inf')
    m.optimize(callback)

    # Extract solution
    if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # If infeasible or error, try to provide empty but valid
        assignment_dict = {}
        for tr in trades:
            tid = str(tr['id'])
            assignment_dict[tid] = {}
            s = tr['start_period']
            e = tr['end_period']
            for p in range(s, e + 1):
                assignment_dict[tid][str(p)] = []
        obj_val = 0.0
    else:
        assignment_dict = {}
        for tr_idx, tr in enumerate(trades):
            tid = str(tr['id'])
            assignment_dict[tid] = {}
            s = tr['start_period']
            e = tr['end_period']
            for t in range(s, e + 1):
                assigned = []
                for c_idx in range(C):
                    if avail[c_idx][t] and (t, c_idx, tr_idx) in x:
                        if abs(x[t, c_idx, tr_idx].X) > 0.5:
                            assigned.append(containers[c_idx]['id'])
                assignment_dict[tid][str(t)] = assigned
        obj_val = m.objVal if m.objVal is not None else 0.0

    solution = {
        "objective_value": float(obj_val),
        "assignments": assignment_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()