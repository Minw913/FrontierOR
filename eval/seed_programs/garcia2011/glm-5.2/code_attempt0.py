import argparse
import json
import time
import random
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def evaluate_solution(opened_facilities, n, cost_matrix):
    total_cost = 0
    assignments = []
    for i in range(n):
        best_fac = -1
        best_cost = float('inf')
        for j in opened_facilities:
            c = cost_matrix[i][j]
            if c < best_cost:
                best_cost = c
                best_fac = j
        total_cost += best_cost
        assignments.append({"customer": i, "facility": best_fac, "cost": best_cost})
    return total_cost, assignments

def solve_p_median():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance["n"]
    p = instance["p"]
    cost_matrix = instance["cost_matrix"]

    best_sol = None
    best_cost = float('inf')

    def log_and_update(cost, opened, assignments):
        nonlocal best_sol, best_cost
        if cost < best_cost:
            best_cost = cost
            best_sol = {
                "objective_value": cost,
                "opened_facilities": sorted(opened),
                "assignments": assignments
            }
            if logger:
                logger.log_solution(best_cost, best_sol)

    def quick_local_search(time_budget):
        nonlocal best_sol, best_cost
        random.seed(0)
        opened = random.sample(range(n), p)
        cost, assignments = evaluate_solution(opened, n, cost_matrix)
        log_and_update(cost, opened, assignments)
        
        local_search_iterations = 0
        while time.time() - start_time < time_budget:
            if local_search_iterations > 0:
                opened = random.sample(range(n), p)
                cost, assignments = evaluate_solution(opened, n, cost_matrix)
                log_and_update(cost, opened, assignments)
            local_search_iterations += 1
            
            improved = True
            while improved and time.time() - start_time < time_budget:
                improved = False
                opened_set = set(opened)
                closed = [j for j in range(n) if j not in opened_set]
                for f_out in list(opened):
                    for f_in in closed:
                        if time.time() - start_time > time_budget:
                            break
                        new_opened = [f for f in opened if f != f_out]
                        new_opened.append(f_in)
                        new_cost, new_assignments = evaluate_solution(new_opened, n, cost_matrix)
                        if new_cost < cost:
                            opened = new_opened
                            cost = new_cost
                            assignments = new_assignments
                            improved = True
                            log_and_update(cost, opened, assignments)
                            break
                    if improved:
                        break

    if n <= 400:
        quick_local_search(args.time_limit * 0.1)
        if time.time() - start_time < args.time_limit:
            init_sol = best_sol
            m = gp.Model("p_median")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1, args.time_limit - (time.time() - start_time))
            m.Params.OutputFlag = 0

            obj_dict = {(i, j): cost_matrix[i][j] for i in range(n) for j in range(n)}
            y = m.addVars(n, n, vtype=GRB.BINARY, obj=obj_dict, name="y")
            x = m.addVars(n, vtype=GRB.BINARY, name="x")
            m.ModelSense = GRB.MINIMIZE

            m.addConstrs((y.sum(i, '*') == 1 for i in range(n)), name="assign")
            m.addConstr(x.sum() == p, name="p_meds")
            m.addConstrs((y[i, j] <= x[j] for i in range(n) for j in range(n)), name="link")

            if init_sol:
                for j in range(n):
                    x[j].Start = 1 if j in init_sol["opened_facilities"] else 0
                for i in range(n):
                    for j in range(n):
                        y[i, j].Start = 1 if init_sol["assignments"][i]["facility"] == j else 0

            m._vars_x = x
            m._vars_y = y
            m._logger = logger
            m._n = n
            m._cost_matrix = cost_matrix

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if model._logger:
                        x_vals = model.cbGetSolution(model._vars_x)
                        y_vals = model.cbGetSolution(model._vars_y)
                        opened_facilities = [j for j in range(model._n) if x_vals[j] > 0.5]
                        assignments = []
                        for i in range(model._n):
                            for j in range(model._n):
                                if y_vals[i, j] > 0.5:
                                    assignments.append({
                                        "customer": i,
                                        "facility": j,
                                        "cost": model._cost_matrix[i][j]
                                    })
                                    break
                        solution_dict = {
                            "objective_value": int(round(obj)),
                            "opened_facilities": opened_facilities,
                            "assignments": assignments
                        }
                        model._logger.log_solution(int(round(obj)), solution_dict)

            m.optimize(callback)

            if m.SolCount > 0:
                x_vals = m.getAttr("X", x)
                y_vals = m.getAttr("X", y)
                opened_facilities = [j for j in range(n) if x_vals[j] > 0.5]
                assignments = []
                obj_val = 0
                for i in range(n):
                    for j in range(n):
                        if y_vals[i, j] > 0.5:
                            assignments.append({
                                "customer": i,
                                "facility": j,
                                "cost": cost_matrix[i][j]
                            })
                            obj_val += cost_matrix[i][j]
                            break
                gurobi_sol = {
                    "objective_value": int(round(obj_val)),
                    "opened_facilities": opened_facilities,
                    "assignments": assignments
                }
                if gurobi_sol["objective_value"] < best_cost:
                    best_sol = gurobi_sol
    else:
        quick_local_search(args.time_limit)

    if not best_sol:
        random.seed(0)
        opened = random.sample(range(n), p)
        cost, assignments = evaluate_solution(opened, n, cost_matrix)
        best_sol = {
            "objective_value": cost,
            "opened_facilities": sorted(opened),
            "assignments": assignments
        }
        if logger:
            logger.log_solution(cost, best_sol)

    if best_sol:
        with open(args.solution_path, 'w') as f:
            json.dump(best_sol, f, indent=2)

if __name__ == "__main__":
    solve_p_median()