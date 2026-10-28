import json
import argparse
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Import the provided logger
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ------------------------------------------------------------------
    # 1. Read instance
    # ------------------------------------------------------------------
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_real = data['num_real_activities']
    num_total = data['num_activities_total']
    num_res = data['num_resources']
    T = data['max_project_duration']

    # processing times
    proc_time = {}
    for act in data['activities']:
        proc_time[act['id']] = act['processing_time']
    N = num_real + 1          # project-end activity id
    assert proc_time[0] == 0
    assert proc_time[N] == 0

    # temporal constraints
    edges = []
    for tc in data['temporal_constraints']:
        edges.append((tc['from_activity'], tc['to_activity'], tc['time_lag']))

    # resources
    resources = []
    for res in data['resources']:
        Pi = set(res['Pi_k'])
        cap = res['capacity']
        demands = res['demands']
        # keys may be strings in JSON; convert to int
        demands = {int(k): v for k, v in demands.items()}
        resources.append({'Pi': Pi, 'cap': cap, 'demands': demands})

    # ------------------------------------------------------------------
    # 2. Compute earliest (ES) and latest (LS) start times
    # ------------------------------------------------------------------
    INF = float('inf')
    ES = [-INF] * (N + 1)
    ES[0] = 0
    for _ in range(N + 1):
        updated = False
        for i, j, lag in edges:
            if ES[i] != -INF and ES[j] < ES[i] + lag:
                ES[j] = ES[i] + lag
                updated = True
        if not updated:
            break
    if updated:
        print("Warning: positive cycle detected in temporal graph. Instance may be infeasible.")
    for i in range(N + 1):
        if ES[i] > T:
            ES[i] = T
        if ES[i] < 0:
            ES[i] = 0

    LS = [T] * (N + 1)
    LS[0] = 0
    LS[N] = T
    for _ in range(N + 1):
        updated = False
        for i, j, lag in edges:
            if LS[j] != INF and LS[i] > LS[j] - lag:
                LS[i] = LS[j] - lag
                updated = True
        if not updated:
            break
    for i in range(N + 1):
        if LS[i] < 0:
            LS[i] = 0
        if LS[i] > T:
            LS[i] = T
        if LS[i] < ES[i]:
            LS[i] = ES[i]   # domain becomes empty -> Gurobi will declare infeasible

    # ------------------------------------------------------------------
    # 3. Build MILP
    # ------------------------------------------------------------------
    model = gp.Model("project_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    x = {}                 # x[i][t] = binary variable for start time t
    fixed_start = {}      # activities with a single possible start time
    for i in range(N + 1):
        if ES[i] == LS[i]:
            fixed_start[i] = ES[i]
        else:
            x[i] = {}
            for t in range(ES[i], LS[i] + 1):
                x[i][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # each activity starts exactly once
    for i in range(N + 1):
        if i not in fixed_start:
            model.addConstr(gp.quicksum(x[i].values()) == 1, name=f"start_{i}")

    # helper to get start time expression
    def get_S_expr(i):
        if i in fixed_start:
            return fixed_start[i]
        return gp.quicksum(t * var for t, var in x[i].items())

    # temporal constraints
    for i, j, lag in edges:
        S_i = get_S_expr(i)
        S_j = get_S_expr(j)
        model.addConstr(S_j - S_i >= lag, name=f"tc_{i}_{j}_{lag}")

    # partially renewable resource constraints
    for k, res in enumerate(resources):
        Pi = res['Pi']
        cap = res['cap']
        demands = res['demands']
        expr = gp.LinExpr()
        for i in range(N + 1):
            d = demands.get(i, 0)
            if d == 0:
                continue
            p_i = proc_time[i]
            if p_i == 0:
                continue
            if i in fixed_start:
                s = fixed_start[i]
                coeff = 0
                for t in range(s + 1, min(s + p_i, T) + 1):
                    if t in Pi:
                        coeff += 1
                if coeff > 0:
                    expr += d * coeff
            else:
                for s, var in x[i].items():
                    coeff = 0
                    for t in range(s + 1, min(s + p_i, T) + 1):
                        if t in Pi:
                            coeff += 1
                    if coeff > 0:
                        expr += d * coeff * var
        if expr.size() > 0:
            model.addConstr(expr <= cap, name=f"res_{k}")

    # objective: minimise start time of project-end activity
    S_E = get_S_expr(N)
    model.setObjective(S_E, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # 4. Callback for logging incumbents
    # ------------------------------------------------------------------
    best_obj = INF
    best_schedule = {}

    # collect all variables for the callback
    x_vars_list = []
    for i in x:
        for t, var in x[i].items():
            x_vars_list.append(var)

    def build_schedule_from_vals(var_to_val):
        schedule = {}
        for i in range(N + 1):
            if i in fixed_start:
                schedule[str(i)] = fixed_start[i]
            else:
                s_val = None
                for t, var in x[i].items():
                    if var_to_val[var] > 0.5:
                        s_val = t
                        break
                if s_val is None:
                    s_val = ES[i]
                schedule[str(i)] = s_val
        return schedule

    def callback(model, where):
        nonlocal best_obj, best_schedule
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGetSolution(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj:
                best_obj = obj
                vals = model.cbGetSolution(x_vars_list)
                var_to_val = {v: val for v, val in zip(x_vars_list, vals)}
                schedule = build_schedule_from_vals(var_to_val)
                best_schedule = schedule
                if logger:
                    logger.log_solution(obj, schedule)

    model.optimize(callback)

    # ------------------------------------------------------------------
    # 5. Retrieve and write final solution
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        # use the incumbent stored in the model
        var_to_val = {v: v.X for v in x_vars_list}
        schedule = build_schedule_from_vals(var_to_val)
        obj_val = model.ObjVal
        if obj_val < best_obj:
            best_obj = obj_val
            best_schedule = schedule
            if logger:
                logger.log_solution(obj_val, schedule)
    elif best_schedule:
        # use best schedule found during callbacks
        obj_val = best_obj
        schedule = best_schedule
    else:
        # fallback: ES schedule (may violate resource constraints, but respects temporal ones)
        schedule = {str(i): ES[i] for i in range(N + 1)}
        obj_val = ES[N]

    output = {
        "objective_value": float(obj_val),
        "schedule": {str(i): int(schedule[str(i)]) for i in range(N + 1)}
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()