import argparse
import json
import time
from collections import deque
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_time_windows(N, edges, deadline):
    """
    Compute earliest start (ES) and latest start (LS) times using longest paths.
    edges: list of (i, j, d) meaning s_j >= s_i + d
    Returns (ES, LS, dist_to_end) or (None, None, None) if positive cycle exists.
    """
    # ES: longest path from node 0
    ES = [0] * N
    adj = [[] for _ in range(N)]
    for i, j, d in edges:
        adj[i].append((j, d))

    in_q = [False] * N
    cnt = [0] * N
    q = deque([0])
    in_q[0] = True
    while q:
        u = q.popleft()
        in_q[u] = False
        for v, d in adj[u]:
            if ES[u] + d > ES[v]:
                ES[v] = ES[u] + d
                cnt[v] += 1
                if cnt[v] > N:
                    return None, None, None
                if not in_q[v]:
                    q.append(v)
                    in_q[v] = True

    # dist_to_end: longest path from i to N-1
    rev_adj = [[] for _ in range(N)]
    for i, j, d in edges:
        rev_adj[j].append((i, d))
    dist_to_end = [-10**9] * N
    dist_to_end[N - 1] = 0
    q = deque([N - 1])
    in_q = [False] * N
    in_q[N - 1] = True
    cnt = [0] * N
    while q:
        u = q.popleft()
        in_q[u] = False
        for v, d in rev_adj[u]:
            if dist_to_end[u] + d > dist_to_end[v]:
                dist_to_end[v] = dist_to_end[u] + d
                cnt[v] += 1
                if cnt[v] > N:
                    return None, None, None
                if not in_q[v]:
                    q.append(v)
                    in_q[v] = True

    LS = [deadline] * N
    for i in range(N):
        if dist_to_end[i] > -10**8:
            LS[i] = deadline - dist_to_end[i]
        LS[i] = min(LS[i], deadline)
        LS[i] = max(LS[i], ES[i])
    LS[0] = 0
    return ES, LS, dist_to_end


def build_model(N, p, edges, resources, deadline, ES, LS, start_x=None):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    x = {}
    var_list = []
    var_info = []
    for i in range(N):
        for s in range(ES[i], LS[i] + 1):
            v = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")
            x[i, s] = v
            var_list.append(v)
            var_info.append((i, s))
            if start_x is not None:
                v.Start = 1.0 if (i, s) in start_x else 0.0

    s_var = model.addVars(N, lb=0, ub=deadline, name="s")
    for i in range(N):
        model.addConstr(
            s_var[i] == gp.quicksum(s * x[i, s] for s in range(ES[i], LS[i] + 1))
        )

    model.addConstr(s_var[0] == 0)
    model.addConstr(s_var[N - 1] <= deadline)

    for (i, j, d) in edges:
        model.addConstr(s_var[j] >= s_var[i] + d)

    for resource in resources:
        time_periods = resource['time_periods']
        capacity = resource['capacity']
        demands = resource['demands']
        tp_set = set(time_periods)
        prefix = [0] * (deadline + 1)
        for t in range(1, deadline + 1):
            prefix[t] = prefix[t - 1] + (1 if t in tp_set else 0)

        terms = []
        for i in range(N):
            if demands[i] == 0 or p[i] == 0:
                continue
            for s in range(ES[i], LS[i] + 1):
                end = min(s + p[i], deadline)
                if end > s:
                    c = prefix[end] - prefix[s]
                    if c > 0:
                        terms.append(demands[i] * c * x[i, s])
        if terms:
            model.addConstr(gp.quicksum(terms) <= capacity)

    model.setObjective(s_var[N - 1], GRB.MINIMIZE)
    return model, x, s_var, var_list, var_info


def make_callback(var_info, num_x_vars, N, logger, best_obj):
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                sol = model.cbGet(GRB.Callback.MIPSOL_SOL)
                start_times = [0] * N
                for idx in range(num_x_vars):
                    if sol[idx] > 0.5:
                        i, s = var_info[idx]
                        start_times[i] = s
                solution_dict = {
                    "objective_value": int(round(obj)),
                    "start_times": start_times,
                }
                if logger:
                    logger.log_solution(int(round(obj)), solution_dict)
    return callback


def extract_start_times(model, x, ES, LS, N):
    start_times = [0] * N
    for i in range(N):
        found = False
        for s in range(ES[i], LS[i] + 1):
            if x[i, s].X > 0.5:
                start_times[i] = s
                found = True
                break
        if not found:
            start_times[i] = ES[i]
    return start_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['num_nodes']
    p = instance['processing_times']
    deadline = instance['deadline']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    # Unify all temporal constraints to the form s_j >= s_i + d
    edges = []
    for tc in temporal_constraints:
        i = tc['from']
        j = tc['to']
        delta = tc['delta']
        if tc['type'] == 'min_lag':
            edges.append((i, j, delta))
        elif tc['type'] == 'max_lag':
            edges.append((j, i, -delta))
        else:
            edges.append((i, j, delta))

    ES, LS, dist_to_end = compute_time_windows(N, edges, deadline)
    if ES is None:
        # Positive cycle detected -> infeasible
        output = {"objective_value": deadline, "start_times": [0] * N}
        with open(args.solution_path, 'w') as f:
            json.dump(output, f)
        if logger:
            logger.log_solution(deadline, output)
        return

    # Trivial infeasibility check
    for i in range(N):
        if ES[i] > LS[i]:
            output = {"objective_value": deadline, "start_times": ES}
            with open(args.solution_path, 'w') as f:
                json.dump(output, f)
            if logger:
                logger.log_solution(deadline, output)
            return

    best_obj = [float('inf')]
    best_start = None

    # Phase 1: solve with original time windows
    model, x, s_var, var_list, var_info = build_model(N, p, edges, resources, deadline, ES, LS)
    cb = make_callback(var_info, len(var_list), N, logger, best_obj)

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining <= 1:
        model.setParam('TimeLimit', max(1, remaining))
        model.optimize(cb)
    else:
        phase1_time = min(remaining * 0.3, 10.0)
        model.setParam('TimeLimit', phase1_time)
        model.optimize(cb)

    if model.SolCount > 0:
        best_obj[0] = model.ObjVal
        best_start = extract_start_times(model, x, ES, LS, N)

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    # Phase 2: tighten windows and resolve if we have an incumbent
    if remaining > 1 and best_obj[0] < float('inf') and best_obj[0] < deadline:
        new_deadline = int(round(best_obj[0]))
        new_LS = [new_deadline] * N
        for i in range(N):
            if dist_to_end[i] > -10**8:
                new_LS[i] = new_deadline - dist_to_end[i]
            new_LS[i] = min(new_LS[i], new_deadline)
            new_LS[i] = max(new_LS[i], ES[i])
        new_LS[0] = 0
        new_LS[N - 1] = min(new_LS[N - 1], new_deadline)

        tighter = any(new_LS[i] < LS[i] for i in range(N))
        if tighter:
            start_x = {}
            if best_start:
                for i in range(N):
                    start_x[(i, best_start[i])] = 1.0
            model2, x2, s_var2, var_list2, var_info2 = build_model(
                N, p, edges, resources, deadline, ES, new_LS, start_x
            )
            cb2 = make_callback(var_info2, len(var_list2), N, logger, best_obj)
            model2.setParam('TimeLimit', remaining)
            model2.optimize(cb2)
            if model2.SolCount > 0 and model2.ObjVal < best_obj[0]:
                best_obj[0] = model2.ObjVal
                best_start = extract_start_times(model2, x2, ES, new_LS, N)
            model2.dispose()
    elif remaining > 1 and best_obj[0] == float('inf'):
        # No solution yet: continue with the same model
        model.setParam('TimeLimit', remaining)
        model.optimize(cb)
        if model.SolCount > 0:
            best_obj[0] = model.ObjVal
            best_start = extract_start_times(model, x, ES, LS, N)

    # Write final solution
    if best_start is not None:
        output = {
            "objective_value": int(round(best_obj[0])),
            "start_times": best_start,
        }
    else:
        # Fallback to earliest start times if no feasible solution found
        output = {
            "objective_value": ES[N - 1],
            "start_times": ES,
        }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

    model.dispose()


if __name__ == "__main__":
    main()