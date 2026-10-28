import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def compute_total_cost(assignment, linear_costs, quadratic_dict, n_clients):
    total = 0.0
    for i in range(n_clients):
        j = assignment[i]
        total += linear_costs[i][j]
    
    for key, cost_val in quadratic_dict.items():
        assignments_list = list(key)
        a1 = assignments_list[0]
        a2 = assignments_list[1]
        i1, j1 = a1
        i2, j2 = a2
        if assignment[i1] == j1 and assignment[i2] == j2:
            total += cost_val
    return total

def greedy_initial_assignment(linear_costs, quad_index, n_servers, n_clients):
    assignment = [0] * n_clients
    for i in range(n_clients):
        best_j = None
        best_cost = None
        for j in range(n_servers):
            cost_here = linear_costs[i][j]
            a_ij = (i, j)
            if a_ij in quad_index:
                for k in range(i):
                    a_k = (k, assignment[k])
                    if a_k in quad_index[a_ij]:
                        cost_here += quad_index[a_ij][a_k]
            if best_cost is None or cost_here < best_cost:
                best_cost = cost_here
                best_j = j
        assignment[i] = best_j
    return assignment

def compute_delta(i, old_j, new_j, assignment, linear_costs, quad_index, n_clients):
    delta_linear = linear_costs[i][new_j] - linear_costs[i][old_j]
    delta_quad = 0.0
    a_i_old = (i, old_j)
    a_i_new = (i, new_j)
    for k in range(n_clients):
        if k == i:
            continue
        a_k = (k, assignment[k])
        cost_old = 0.0
        if a_i_old in quad_index and a_k in quad_index[a_i_old]:
            cost_old = quad_index[a_i_old][a_k]
        cost_new = 0.0
        if a_i_new in quad_index and a_k in quad_index[a_i_new]:
            cost_new = quad_index[a_i_new][a_k]
        delta_quad += cost_new - cost_old
    return delta_linear + delta_quad

def local_search(assignment, start_total_cost, linear_costs, quad_index, n_servers, n_clients, time_limit, start_time, logger):
    current_cost = start_total_cost
    improved = True
    while improved:
        if time.time() - start_time >= time_limit:
            break
        improved = False
        for i in range(n_clients):
            if time.time() - start_time >= time_limit:
                break
            current_j = assignment[i]
            for j in range(n_servers):
                if j == current_j:
                    continue
                if time.time() - start_time >= time_limit:
                    break
                delta = compute_delta(i, current_j, j, assignment, linear_costs, quad_index, n_clients)
                if delta < -1e-5:
                    assignment[i] = j
                    current_cost += delta
                    if logger:
                        logger.log(current_cost)
                    improved = True
                    break
    return assignment, current_cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_clients = instance['n_clients']
    n_servers = instance['n_servers']
    demands = instance['demands']
    processing_rates = instance['processing_rates']
    linear_costs = instance['linear_costs']
    quadratic_costs_list = instance['quadratic_costs']

    quadratic_dict = {}
    quad_index = {}
    for entry in quadratic_costs_list:
        c1, s1, c2, s2, cost_val = entry
        if c1 == c2:
            continue
        a1 = (c1, s1)
        a2 = (c2, s2)
        key = frozenset({a1, a2})
        quadratic_dict[key] = quadratic_dict.get(key, 0.0) + cost_val

        if a1 not in quad_index:
            quad_index[a1] = {}
        quad_index[a1][a2] = quad_index[a1].get(a2, 0.0) + cost_val

        if a2 not in quad_index:
            quad_index[a2] = {}
        quad_index[a2][a1] = quad_index[a2].get(a1, 0.0) + cost_val

    total_quad_terms = len(quadratic_dict)
    time_remaining = max(1, args.time_limit - (time.time() - start_time))

    if total_quad_terms <= 500000:
        try:
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', time_remaining)

            x = {}
            for i in range(n_clients):
                for j in range(n_servers):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

            for i in range(n_clients):
                model.addConstr(gp.quicksum(x[i, j] for j in range(n_servers)) == 1)

            linear_expr = gp.quicksum(linear_costs[i][j] * x[i, j] for i in range(n_clients) for j in range(n_servers))
            quad_expr = gp.QuadExpr()
            for key, cost_val in quadratic_dict.items():
                a1, a2 = list(key)
                i1, j1 = a1
                i2, j2 = a2
                quad_expr += cost_val * x[i1, j1] * x[i2, j2]

            model.setObjective(linear_expr + quad_expr, GRB.MINIMIZE)

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj_val)

            model.optimize(callback)

            if model.SolCount > 0:
                assignment_sol = [0] * n_clients
                for i in range(n_clients):
                    for j in range(n_servers):
                        if abs(x[i, j].X - 1.0) < 1e-5:
                            assignment_sol[i] = j
                obj_val = model.ObjVal
            else:
                raise Exception("Gurobi found no solution, falling back to heuristic")
        except Exception as e:
            print(f"Gurobi failed: {e}, falling back to heuristic")
            assignment_sol = greedy_initial_assignment(linear_costs, quad_index, n_servers, n_clients)
            obj_val = compute_total_cost(assignment_sol, linear_costs, quadratic_dict, n_clients)
            if logger:
                logger.log(obj_val)
            time_remaining = max(1, args.time_limit - (time.time() - start_time))
            assignment_sol, obj_val = local_search(assignment_sol, obj_val, linear_costs, quad_index, n_servers, n_clients, time_remaining, start_time, logger)
    else:
        assignment_sol = greedy_initial_assignment(linear_costs, quad_index, n_servers, n_clients)
        obj_val = compute_total_cost(assignment_sol, linear_costs, quadratic_dict, n_clients)
        if logger:
            logger.log(obj_val)
        time_remaining = max(1, args.time_limit - (time.time() - start_time))
        assignment_sol, obj_val = local_search(assignment_sol, obj_val, linear_costs, quad_index, n_servers, n_clients, time_remaining, start_time, logger)

    assignment_dict = {str(i): assignment_sol[i] for i in range(n_clients)}
    solution = {
        "objective_value": obj_val,
        "assignment": assignment_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()