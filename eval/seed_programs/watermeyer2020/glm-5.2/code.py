import json
import argparse
import sys
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    num_activities = instance['num_activities']
    num_resources = instance['num_resources']
    d_bar = instance['d_bar']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    N = num_activities
    end = N - 1

    # Build graph for bounds
    fwd_adj = [[] for _ in range(N)]
    rev_adj = [[] for _ in range(N)]
    for tc in temporal_constraints:
        i, j, delta = tc['i'], tc['j'], tc['delta']
        fwd_adj[i].append((j, delta))
        rev_adj[j].append((i, delta))
    # Add max duration constraint: S_0 - S_end >= -d_bar  => edge end -> 0 weight -d_bar
    fwd_adj[end].append((0, -d_bar))
    rev_adj[0].append((end, -d_bar))

    INF = 10**12
    # Forward Bellman-Ford from 0
    dist_fwd = [INF] * N
    dist_fwd[0] = 0
    for _ in range(N - 1):
        updated = False
        for u in range(N):
            if dist_fwd[u] == INF:
                continue
            for v, w in fwd_adj[u]:
                if dist_fwd[u] + w < dist_fwd[v] - 1e-9:
                    dist_fwd[v] = dist_fwd[u] + w
                    updated = True
        if not updated:
            break
    # Check negative cycle
    for u in range(N):
        if dist_fwd[u] == INF:
            continue
        for v, w in fwd_adj[u]:
            if dist_fwd[u] + w < dist_fwd[v] - 1e-9:
                print("Negative cycle detected (forward), infeasible", file=sys.stderr)
                sol = {"objective_value": d_bar, "start_times": {str(i): 0 for i in range(N)}}
                with open(args.solution_path, 'w') as f:
                    json.dump(sol, f)
                return

    LB = [max(0, int(round(dist_fwd[i]))) if dist_fwd[i] != INF else 0 for i in range(N)]
    LB[0] = 0

    # Reverse Bellman-Ford from end
    dist_rev = [INF] * N
    dist_rev[end] = 0
    for _ in range(N - 1):
        updated = False
        for u in range(N):
            if dist_rev[u] == INF:
                continue
            for v, w in rev_adj[u]:
                if dist_rev[u] + w < dist_rev[v] - 1e-9:
                    dist_rev[v] = dist_rev[u] + w
                    updated = True
        if not updated:
            break
    # Check negative cycle
    for u in range(N):
        if dist_rev[u] == INF:
            continue
        for v, w in rev_adj[u]:
            if dist_rev[u] + w < dist_rev[v] - 1e-9:
                print("Negative cycle detected (reverse), infeasible", file=sys.stderr)
                sol = {"objective_value": d_bar, "start_times": {str(i): 0 for i in range(N)}}
                with open(args.solution_path, 'w') as f:
                    json.dump(sol, f)
                return

    UB = [int(round(d_bar - dist_rev[i])) if dist_rev[i] != INF else int(d_bar) for i in range(N)]
    for i in range(N):
        if UB[i] < LB[i]:
            print("Infeasible bounds", file=sys.stderr)
            sol = {"objective_value": d_bar, "start_times": {str(i): int(LB[i]) for i in range(N)}}
            with open(args.solution_path, 'w') as f:
                json.dump(sol, f)
            return

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("project_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 1)
    model.Params.OutputFlag = 0

    x = {}
    act_times = {}
    for i in range(N):
        lb = LB[i]
        ub = UB[i]
        if lb > ub:
            continue
        times = list(range(lb, ub + 1))
        act_times[i] = times
        for t in times:
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
    model.update()

    # Each activity starts exactly once
    for i in range(N):
        if i not in act_times:
            print(f"Activity {i} has no feasible start times", file=sys.stderr)
            sol = {"objective_value": d_bar, "start_times": {str(j): int(LB[j]) for j in range(N)}}
            with open(args.solution_path, 'w') as f:
                json.dump(sol, f)
            return
        model.addConstr(gp.quicksum(x[i, t] for t in act_times[i]) == 1)

    # S_0 = 0
    model.addConstr(x[0, 0] == 1)

    # Temporal constraints
    for tc in temporal_constraints:
        i, j, delta = tc['i'], tc['j'], tc['delta']
        S_i = gp.quicksum(t * x[i, t] for t in act_times[i])
        S_j = gp.quicksum(t * x[j, t] for t in act_times[j])
        model.addConstr(S_j - S_i >= delta)

    # Resource constraints
    max_t = max(UB) if UB else 0
    max_p = 0
    for i in range(N):
        act = activities.get(str(i), {})
        p_i = act.get('processing_time', 0)
        if p_i > max_p:
            max_p = p_i
    T_prefix = max_t + max_p + 1

    for res_id, res in resources.items():
        capacity = res['capacity']
        periods = res['periods']
        if capacity < 0:
            continue
        pref = [0] * (T_prefix + 1)
        for tau in periods:
            if 1 <= tau <= T_prefix:
                pref[tau] = 1
        for tau in range(1, T_prefix + 1):
            pref[tau] += pref[tau - 1]
        
        expr = gp.LinExpr()
        for i in range(N):
            act = activities.get(str(i), {})
            p_i = act.get('processing_time', 0)
            res_demands = act.get('resource_demands', {})
            d_ik = res_demands.get(res_id, 0)
            if d_ik == 0 or p_i == 0:
                continue
            for t in act_times[i]:
                upper = min(t + p_i, T_prefix)
                count = pref[upper] - pref[t]
                if count > 0:
                    expr += d_ik * count * x[i, t]
        model.addConstr(expr <= capacity)

    # Objective
    S_end = gp.quicksum(t * x[end, t] for t in act_times[end])
    model.setObjective(S_end, GRB.MINIMIZE)

    # Callback
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = {}
            for i in range(model._N):
                if i not in model._act_times:
                    sol[str(i)] = int(model._LB[i])
                    continue
                vals = model.cbGetSolution([model._x[i, t] for t in model._act_times[i]])
                for t, v in zip(model._act_times[i], vals):
                    if v > 0.5:
                        sol[str(i)] = int(t)
                        break
                else:
                    sol[str(i)] = int(model._LB[i])
            if logger:
                logger.log_solution(obj, sol)

    model._x = x
    model._act_times = act_times
    model._N = N
    model._LB = LB

    # Warm start
    for i in range(N):
        for t in act_times[i]:
            x[i, t].Start = 1 if t == LB[i] else 0

    model.optimize(callback)

    if model.SolCount == 0:
        print("No solution found by Gurobi, using earliest start heuristic", file=sys.stderr)
        start_times = {str(i): int(LB[i]) for i in range(N)}
        obj_val = start_times[str(end)]
        if logger:
            logger.log_solution(obj_val, start_times)
    else:
        start_times = {}
        for i in range(N):
            for t in act_times[i]:
                if x[i, t].X > 0.5:
                    start_times[str(i)] = int(t)
                    break
            else:
                start_times[str(i)] = int(LB[i])
        obj_val = int(round(model.ObjVal))
        if logger:
            logger.log_solution(obj_val, start_times)

    sol_output = {
        "objective_value": obj_val,
        "start_times": start_times
    }
    with open(args.solution_path, 'w') as f:
        json.dump(sol_output, f)

if __name__ == "__main__":
    solve()