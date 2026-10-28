import argparse
import json
import math
from collections import defaultdict, deque
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_tasks']
    cycle_time_ub = data['cycle_time']
    m = data['num_stations']
    processing_times = data['processing_times']
    precedence_relations = data['precedence_relations']
    forward_setup_times = data['forward_setup_times']
    backward_setup_times = data['backward_setup_times']

    precedence_relations = [prec for prec in precedence_relations 
                            if 0 <= prec[0] < n and 0 <= prec[1] < n]

    model = gp.Model("U-shaped_assembly_line")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    x = {}
    for i in range(n):
        for s in range(m):
            x[i, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")

    y = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for s in range(m):
                    y[i, j, s] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{s}")

    f = {}
    l = {}
    for i in range(n):
        for s in range(m):
            f[i, s] = model.addVar(vtype=GRB.BINARY, name=f"f_{i}_{s}")
            l[i, s] = model.addVar(vtype=GRB.BINARY, name=f"l_{i}_{s}")

    w = {}
    for i in range(n):
        for j in range(n):
            for s in range(m):
                w[i, j, s] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{s}")

    p = {}
    for i in range(n):
        p[i] = model.addVar(lb=1.0, ub=float(n), vtype=GRB.CONTINUOUS, name=f"p_{i}")

    T = {}
    for s in range(m):
        T[s] = model.addVar(lb=0.0, ub=cycle_time_ub, vtype=GRB.CONTINUOUS, name=f"T_{s}")

    M_var = model.addVar(lb=0.0, ub=cycle_time_ub, vtype=GRB.CONTINUOUS, name="M")

    model.update()

    model.setObjective(M_var, GRB.MINIMIZE)

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, s] for s in range(m)) == 1, f"assign_task_{i}")

    for s in range(m):
        model.addConstr(gp.quicksum(x[i, s] for i in range(n)) >= 1, f"station_{s}_nonempty")

    for prec in precedence_relations:
        i, j = prec
        for k in range(m):
            model.addConstr(gp.quicksum(x[i, s] for s in range(0, k+1)) >= x[j, k], 
                           f"precedence_{i}_{j}_station_{k}")

    for s in range(m):
        for i in range(n):
            model.addConstr(gp.quicksum(y[i, j, s] for j in range(n) if j != i) + l[i, s] == x[i, s],
                           f"station_{s}_task_{i}_successor_or_last")
            model.addConstr(gp.quicksum(y[j, i, s] for j in range(n) if j != i) + f[i, s] == x[i, s],
                           f"station_{s}_task_{i}_predecessor_or_first")

    for s in range(m):
        model.addConstr(gp.quicksum(f[i, s] for i in range(n)) == 1, f"station_{s}_exactly_one_first")
        model.addConstr(gp.quicksum(l[i, s] for i in range(n)) == 1, f"station_{s}_exactly_one_last")

    for s in range(m):
        for i in range(n):
            model.addConstr(f[i, s] <= x[i, s], f"first_in_station_{s}_task_{i}")
            model.addConstr(l[i, s] <= x[i, s], f"last_in_station_{s}_task_{i}")

    for s in range(m):
        for i in range(n):
            for j in range(n):
                model.addConstr(w[i, j, s] <= l[i, s], f"w_upper1_{i}_{j}_{s}")
                model.addConstr(w[i, j, s] <= f[j, s], f"w_upper2_{i}_{j}_{s}")
                model.addConstr(w[i, j, s] >= l[i, s] + f[j, s] - 1, f"w_lower_{i}_{j}_{s}")

    for s in range(m):
        expr = gp.quicksum(processing_times[i] * x[i, s] for i in range(n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    expr += forward_setup_times[i][j] * y[i, j, s]
        for i in range(n):
            for j in range(n):
                expr += backward_setup_times[i][j] * w[i, j, s]
        model.addConstr(T[s] == expr, f"station_time_expr_{s}")
        model.addConstr(T[s] <= M_var, f"station_{s}_time_leq_M")

    for s in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(y[i, j, s] <= x[i, s], f"y_implies_x_i_{i}_{j}_{s}")
                    model.addConstr(y[i, j, s] <= x[j, s], f"y_implies_x_j_{i}_{j}_{s}")

    bigM_pos = n + 1
    for s in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(p[j] >= p[i] + 1 - bigM_pos * (1 - y[i, j, s]), 
                                   f"position_succ_ge_{i}_{j}_{s}")
                    model.addConstr(p[j] <= p[i] + 1 + bigM_pos * (1 - y[i, j, s]),
                                   f"position_succ_le_{i}_{j}_{s}")

    for prec in precedence_relations:
        i, j = prec
        for s in range(m):
            model.addConstr(p[j] >= p[i] + 1 - bigM_pos * (2 - x[i, s] - x[j, s]),
                           f"precedence_sequence_{i}_{j}_{s}")

    if logger:
        def grb_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        model.optimize(grb_callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        assignment = {}
        for i in range(n):
            for s in range(m):
                if x[i, s].x > 0.5:
                    assignment[str(i)] = s + 1
                    break

        station_sequences = {}
        for s in range(m):
            first_task = None
            for i in range(n):
                if f[i, s].x > 0.5:
                    first_task = i
                    break
            if first_task is None:
                station_sequences[str(s + 1)] = []
                continue
            seq = [first_task + 1]
            current = first_task
            while True:
                next_task = None
                for j in range(n):
                    if j == current:
                        continue
                    if y[current, j, s].x > 0.5:
                        next_task = j
                        break
                if next_task is None:
                    break
                seq.append(next_task + 1)
                current = next_task
            station_sequences[str(s + 1)] = seq

        cycle_time_val = M_var.x
        solution = {
            "objective_value": cycle_time_val,
            "cycle_time": cycle_time_val,
            "assignment": assignment,
            "station_sequences": station_sequences
        }
    else:
        graph = defaultdict(list)
        in_degree = [0] * n
        for i, j in precedence_relations:
            graph[i].append(j)
            in_degree[j] += 1

        q = deque()
        for i in range(n):
            if in_degree[i] == 0:
                q.append(i)

        topo_order = []
        while q:
            node = q.popleft()
            topo_order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    q.append(neighbor)

        if len(topo_order) != n:
            topo_order = list(range(n))

        per_station = (n + m - 1) // m
        assignment = {}
        station_tasks = defaultdict(list)
        for idx, task in enumerate(topo_order):
            s = idx // per_station
            if s >= m:
                s = m - 1
            assignment[str(task)] = s + 1
            station_tasks[s].append(task)

        station_sequences = {}
        max_station_time = 0
        for s in range(m):
            tasks_here = station_tasks[s]
            station_sequences[str(s + 1)] = [t + 1 for t in tasks_here]
            if not tasks_here:
                continue
            total_time = 0
            for t in tasks_here:
                total_time += processing_times[t]
            for idx in range(len(tasks_here) - 1):
                i = tasks_here[idx]
                j = tasks_here[idx + 1]
                total_time += forward_setup_times[i][j]
            last = tasks_here[-1]
            first = tasks_here[0]
            total_time += backward_setup_times[last][first]
            if total_time > max_station_time:
                max_station_time = total_time

        solution = {
            "objective_value": max_station_time,
            "cycle_time": max_station_time,
            "assignment": assignment,
            "station_sequences": station_sequences
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()