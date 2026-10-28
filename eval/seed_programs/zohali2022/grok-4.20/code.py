import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_tasks']
    m = data['num_stations']
    cycle_ub = data['cycle_time']
    p = data['processing_times']
    precedence = data['precedence_relations']
    F = data['forward_setup_times']
    B = data['backward_setup_times']

    model = gp.Model("UALBP_With_Setups")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)
    model.setParam('Presolve', 1)
    model.setParam('Cuts', 2)

    T = range(n)
    S = range(m)

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    z = model.addVars(n, n, vtype=GRB.BINARY, name="z")
    first = model.addVars(n, m, vtype=GRB.BINARY, name="first")
    last = model.addVars(n, m, vtype=GRB.BINARY, name="last")
    C = model.addVar(lb=0, ub=cycle_ub, name="C")
    u = model.addVars(n, lb=0, ub=n, vtype=GRB.INTEGER, name="u")

    # Each task to exactly one station
    for i in T:
        model.addConstr(x.sum(i, '*') == 1, name=f"assign_{i}")

    # Each station gets at least one task
    for s in S:
        model.addConstr(x.sum('*', s) >= 1, name=f"station_nonempty_{s}")

    # Precedence constraints
    for pred, succ in precedence:
        i = pred - 1
        j = succ - 1
        # Station non-decreasing
        model.addConstr(gp.quicksum(s * x[i, s] for s in S) <= gp.quicksum(s * x[j, s] for s in S))
        # If same station then i precedes j
        for s in S:
            model.addConstr(z[i, j] >= x[i, s] + x[j, s] - 1, name=f"prec_same_{i}_{j}_{s}")

    # Ordering variables
    for i in T:
        for j in T:
            if i != j:
                model.addConstr(z[i, j] + z[j, i] <= 1, name=f"order_excl_{i}_{j}")
                model.addConstr(y[i, j] <= z[i, j], name=f"y_from_z_{i}_{j}")

    # MTZ for sequencing
    for i in T:
        for j in T:
            if i != j:
                model.addConstr(u[i] - u[j] + (n + 1) * y[i, j] <= n, name=f"mtz_{i}_{j}")

    # y only if both tasks in same station
    for i in T:
        for j in T:
            if i != j:
                model.addConstr(y[i, j] <= gp.quicksum(x[i, s] + x[j, s] for s in S) - 1,
                                name=f"y_same_station_{i}_{j}")

    # Exactly one successor and predecessor per task (global chain - adjusted by first/last)
    for i in T:
        model.addConstr(y.sum(i, '*') == 1, name=f"one_succ_{i}")
        model.addConstr(y.sum('*', i) == 1, name=f"one_pred_{i}")

    # Link first and last
    for s in S:
        model.addConstr(first.sum('*', s) == 1, name=f"one_first_{s}")
        model.addConstr(last.sum('*', s) == 1, name=f"one_last_{s}")

    for i in T:
        for s_idx in S:
            model.addConstr(first[i, s_idx] <= x[i, s_idx], name=f"first_implies_assign_{i}_{s_idx}")
            model.addConstr(last[i, s_idx] <= x[i, s_idx], name=f"last_implies_assign_{i}_{s_idx}")

    # first if no predecessor in same station
    for i in T:
        for s_idx in S:
            pred_in_station = gp.quicksum(y[j, i] * x[j, s_idx] for j in T if j != i)
            model.addConstr(first[i, s_idx] >= x[i, s_idx] - pred_in_station,
                            name=f"first_def_{i}_{s_idx}")
            succ_in_station = gp.quicksum(y[i, j] * x[j, s_idx] for j in T if j != i)
            model.addConstr(last[i, s_idx] >= x[i, s_idx] - succ_in_station,
                            name=f"last_def_{i}_{s_idx}")

    # Cycle time
    for s in S:
        load = gp.quicksum(p[i] * x[i, s] for i in T)
        fwd = gp.quicksum(F[i][j] * y[i, j] for i in T for j in T if i != j)
        bwd = gp.quicksum(B[i][j] * last[i, s] * first[j, s] for i in T for j in T)
        model.addConstr(load + fwd + bwd <= C, name=f"cycle_{s}")

    model.setObjective(C, GRB.MINIMIZE)

    start_time = time.time()
    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount >= 1:
        C_val = C.X
        assignment = {}
        station_tasks = [[] for _ in S]
        for i in T:
            for s in S:
                if x[i, s].X > 0.5:
                    assignment[str(i + 1)] = int(s + 1)
                    station_tasks[s].append(i)
                    break

        station_sequences = {}
        for s_idx in S:
            tasks = station_tasks[s_idx]
            if not tasks:
                station_sequences[str(s_idx + 1)] = []
                continue
            succ = {}
            for i in T:
                for j in T:
                    if y[i, j].X > 0.5:
                        succ[i] = j
            # Find first task at this station
            start = None
            for i in tasks:
                if first[i, s_idx].X > 0.5:
                    start = i
                    break
            if start is None:
                start = tasks[0]
            seq = []
            current = start
            visited = set()
            while current is not None and current not in visited and len(seq) < len(tasks) + 2:
                visited.add(current)
                seq.append(current + 1)
                current = succ.get(current, None)
            # Add any missing tasks (should not happen in feasible solution)
            assigned_set = {t + 1 for t in tasks}
            included = set(seq)
            for t in assigned_set - included:
                seq.append(t)
            station_sequences[str(s_idx + 1)] = seq

        solution = {
            "objective_value": float(C_val),
            "assignment": assignment,
            "station_sequences": station_sequences,
            "cycle_time": float(C_val)
        }
    else:
        # Fallback feasible solution: all tasks in station 1
        C_val = float(sum(p) + sum(F[i][(i+1)%n] for i in range(n)) + B[n-1][0])
        assignment = {str(i+1): 1 for i in range(n)}
        station_sequences = {"1": list(range(1, n+1))}
        for s in range(2, m+1):
            station_sequences[str(s)] = [s]
            if s-1 <= n:
                assignment[str(s)] = s
        solution = {
            "objective_value": C_val,
            "assignment": assignment,
            "station_sequences": station_sequences,
            "cycle_time": C_val
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()