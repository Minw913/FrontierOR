import json
import argparse
import time
import sys
import numpy as np
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def compute_travel_time(departure_time, i, j, data, period_boundaries, speeds, distances, max_speeds):
    dist = distances[i][j]
    if dist <= 1e-6:
        return 0.0
    H = len(period_boundaries) - 1
    t = departure_time
    remaining = dist
    current_period = 0
    while current_period < H and t >= period_boundaries[current_period + 1]:
        current_period += 1
    while remaining > 1e-9 and current_period < H:
        speed = speeds[i][j][current_period]
        if speed < 1e-9:
            return float('inf')
        period_end = period_boundaries[current_period + 1]
        time_to_end = period_end - t
        tentative_time = remaining / speed
        if tentative_time <= time_to_end + 1e-9:
            t += tentative_time
            remaining = 0.0
        else:
            dist_covered = speed * time_to_end
            remaining -= dist_covered
            t = period_end
            current_period += 1
    if remaining > 1e-6:
        return float('inf')
    return t - departure_time

def solve_tsp(data, time_limit):
    n = data['parameters']['n_customers']
    N = data['parameters']['total_vertices']
    H = data['parameters']['H']
    period_boundaries = data['parameters']['period_boundaries']
    distances = data['arcs']['distances']
    max_speeds = data['arcs']['max_speeds_u_ij']
    speeds = data['arcs']['speeds_v_ijh']
    depot = 0

    try:
        model = gp.Model("TDTSP")
        model.setParam(GRB.Param.TimeLimit, time_limit * 0.95)
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.MIPGap, 1e-4)
        model.setParam(GRB.Param.LogToConsole, 0)

        x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
        u = model.addVars(N, lb=0, ub=N, vtype=GRB.CONTINUOUS, name="u")
        t = model.addVars(N, lb=0, name="t")

        for i in range(N):
            x[i, i].UB = 0

        for i in range(N):
            model.addConstr(gp.quicksum(x[i, j] for j in range(N) if i != j) == 1, name=f"out_{i}")
            model.addConstr(gp.quicksum(x[j, i] for j in range(N) if i != j) == 1, name=f"in_{i}")

        for i in range(1, N):
            model.addConstr(u[i] >= 1, name=f"u_lower_{i}")
            model.addConstr(u[i] <= n + 1, name=f"u_upper_{i}")

        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    model.addConstr(u[i] - u[j] + 1 <= (n + 1) * (1 - x[i, j]), name=f"mtz_{i}_{j}")

        M = 10000.0
        bigM_time = 1.5 * sum(max(distances[i][j] / max(1e-6, min(speeds[i][j])) for j in range(N)) for i in range(N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    travel_time_expr = gp.LinExpr()
                    for h in range(H):
                        speed_h = speeds[i][j][h]
                        if speed_h > 1e-9:
                            delta_t_h = min(period_boundaries[h+1], bigM_time) - period_boundaries[h]
                            travel_time_expr.addTerms(distances[i][j] / speed_h, x[i, j])
                    model.addConstr(t[j] >= t[i] + travel_time_expr - bigM_time * (1 - x[i, j]), name=f"time_{i}_{j}")

        model.addConstr(t[depot] == 0, name="start_time")

        obj = t[depot] + gp.quicksum(distances[i][j] / max(1e-6, max(speeds[i][j])) * x[i, j] for i in range(N) for j in range(N))
        model.setObjective(obj, GRB.MINIMIZE)

        logger = None
        if hasattr(solve_tsp, 'logger') and solve_tsp.logger:
            logger = solve_tsp.logger

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if objval < float('inf'):
                    if logger:
                        logger.log(float(objval))
                    if model._best_obj is None or objval < model._best_obj:
                        model._best_obj = objval
                        try:
                            x_sol = model.cbGetSolution(x)
                            tour = [0]
                            current = 0
                            visited = set([0])
                            for _ in range(N):
                                for next_node in range(N):
                                    if next_node != current and x_sol[current, next_node] > 0.5:
                                        if next_node in visited and next_node != 0:
                                            continue
                                        tour.append(next_node)
                                        visited.add(next_node)
                                        current = next_node
                                        break
                            if current != 0:
                                tour.append(0)
                            if len(tour) == N + 2 and tour[-1] == 0:
                                model._best_tour = tour
                                model._best_obj = objval
                        except:
                            pass

        model._best_obj = None
        model._best_tour = None
        model.optimize(callback)

        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
            return None, None

        best_obj = None
        best_tour = None

        if model._best_obj is not None:
            best_obj = float(model._best_obj)
            best_tour = model._best_tour
        elif model.SolCount > 0:
            best_obj = float(model.ObjVal)
            x_sol = model.getAttr('X', x)
            tour = [0]
            current = 0
            for _ in range(N):
                for j in range(N):
                    if j != current and x_sol[current, j] > 0.5:
                        tour.append(j)
                        current = j
                        break
            tour.append(0)
            best_tour = tour
        else:
            print("No feasible solution found.")
            return None, None

        if best_obj is None or best_tour is None:
            return None, None

        if best_obj > data['parameters']['time_horizon_T'] + 1e-3:
            print("Warning: Solution exceeds time horizon.")

        return best_obj, best_tour

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solve_tsp.logger = logger

    start_time = time.time()
    data = load_instance(args.instance_path)

    objective, tour = solve_tsp(data, args.time_limit)

    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - int(elapsed))
    if objective is None or tour is None:
        print("Failed to find a feasible solution.")
        obj_val = 999999.0
        tour = list(range(data['parameters']['total_vertices'] + 1))
    else:
        obj_val = objective

    output = {
        "objective_value": float(obj_val),
        "tour": [int(x) for x in tour]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()