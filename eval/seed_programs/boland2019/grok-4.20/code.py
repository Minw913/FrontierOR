import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_facilities']
    m = data['num_customers']
    demands = data['demands']
    capacities = data['capacities']
    f1 = data['fixed_costs_obj1']
    f2 = data['fixed_costs_obj2']
    c1 = data['assignment_costs_obj1']
    c2 = data['assignment_costs_obj2']

    total_demand = sum(demands)
    min_cap_needed = max(demands)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    best_combined = float('inf')
    best_z1 = None
    best_z2 = None
    best_open = None
    best_assign = None
    pareto_points = []
    pareto_solutions = []

    # Precompute for each facility the assignment costs sorted (for greedy)
    assign_savings = []
    for i in range(n):
        savings = []
        for j in range(m):
            savings.append((c1[i][j] + c2[i][j], j))
        savings.sort()
        assign_savings.append(savings)

    def is_feasible(opened):
        cap_sum = sum(capacities[i] for i in opened)
        return cap_sum >= total_demand and max(capacities[i] for i in opened) >= min_cap_needed

    def solve_for_weights(w1, w2, time_remaining):
        nonlocal best_combined, best_z1, best_z2, best_open, best_assign
        try:
            model = gp.Model("bi_cap")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam('TimeLimit', max(1, time_remaining - 5))
            model.setParam('LogToConsole', 0)

            open_var = model.addVars(n, vtype=GRB.BINARY, name="open")
            assign1 = model.addVars(n, m, vtype=GRB.BINARY, name="assign1")

            obj = gp.quicksum(open_var[i] * (w1 * f1[i] + w2 * f2[i]) for i in range(n))
            obj += gp.quicksum(assign1[i, j] * (w1 * c1[i][j] + w2 * c2[i][j]) for i in range(n) for j in range(m))
            model.setObjective(obj, GRB.MINIMIZE)

            for j in range(m):
                model.addConstr(gp.quicksum(assign1[i, j] for i in range(n)) == 1)

            for i in range(n):
                model.addConstr(gp.quicksum(assign1[i, j] * demands[j] for j in range(m)) <= capacities[i] * open_var[i])

            for i in range(n):
                for j in range(m):
                    model.addConstr(assign1[i, j] <= open_var[i])

            model.optimize()

            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
                z1_val = sum(f1[i] for i in range(n) if open_var[i].X > 0.5)
                z1_val += sum(c1[i][j] for i in range(n) for j in range(m) if assign1[i, j].X > 0.5)
                z2_val = sum(f2[i] for i in range(n) if open_var[i].X > 0.5)
                z2_val += sum(c2[i][j] for i in range(n) for j in range(m) if assign1[i, j].X > 0.5)
                combined = 0.5 * z1_val + 0.5 * z2_val

                if combined < best_combined - 1e-6:
                    best_combined = combined
                    best_z1 = int(z1_val)
                    best_z2 = int(z2_val)
                    best_open = [i for i in range(n) if open_var[i].X > 0.5]
                    best_assign = {}
                    for j in range(m):
                        for i in range(n):
                            if assign1[i, j].X > 0.5:
                                best_assign[str(j)] = i
                                break
                    if logger:
                        logger.log(best_combined)
                    update_pareto(z1_val, z2_val, best_open[:], dict(best_assign))

                return z1_val, z2_val
        except:
            pass
        return None, None

    def update_pareto(z1v, z2v, open_list, assign_dict):
        nonlocal pareto_points, pareto_solutions
        for idx, (pz1, pz2) in enumerate(pareto_points):
            if pz1 <= z1v and pz2 <= z2v:
                return
            if pz1 <= z1v and pz2 < z2v:
                return
            if pz1 < z1v and pz2 <= z2v:
                return
        to_remove = []
        for idx, (pz1, pz2) in enumerate(pareto_points):
            if z1v <= pz1 and z2v <= pz2:
                to_remove.append(idx)
        for idx in sorted(to_remove, reverse=True):
            del pareto_points[idx]
            del pareto_solutions[idx]
        pareto_points.append([int(z1v), int(z2v)])
        pareto_solutions.append({
            "z1": int(z1v),
            "z2": int(z2v),
            "open_facilities": open_list,
            "assignments": assign_dict
        })

    # Initial heuristic with varied weights
    remaining_time = args.time_limit
    weights_to_try = [(1,0), (0,1), (1,1), (2,1), (1,2), (3,1), (1,3)]
    for w1, w2 in weights_to_try:
        if time.time() - start_time > args.time_limit * 0.85:
            break
        solve_for_weights(w1, w2, args.time_limit - (time.time() - start_time))

    # Epsilon-constraint like: optimize z1 with constraint on z2
    if time.time() - start_time < args.time_limit * 0.9:
        try:
            model = gp.Model("eps_z1")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam('TimeLimit', max(5, int(args.time_limit - (time.time() - start_time))//2))
            model.setParam('LogToConsole', 0)

            open_var = model.addVars(n, vtype=GRB.BINARY, name="open")
            assign1 = model.addVars(n, m, vtype=GRB.BINARY, name="assign")

            z1_expr = gp.quicksum(open_var[i] * f1[i] for i in range(n)) + \
                      gp.quicksum(assign1[i,j] * c1[i][j] for i in range(n) for j in range(m))
            z2_expr = gp.quicksum(open_var[i] * f2[i] for i in range(n)) + \
                      gp.quicksum(assign1[i,j] * c2[i][j] for i in range(n) for j in range(m))

            model.setObjective(z1_expr, GRB.MINIMIZE)

            for j in range(m):
                model.addConstr(gp.quicksum(assign1[i,j] for i in range(n)) == 1)
            for i in range(n):
                model.addConstr(gp.quicksum(assign1[i,j]*demands[j] for j in range(m)) <= capacities[i] * open_var[i])
                for j in range(m):
                    model.addConstr(assign1[i,j] <= open_var[i])

            # Add some upper bound on z2 based on previous solutions
            if best_z2 is not None:
                model.addConstr(z2_expr <= best_z2 * 1.1)

            model.optimize()

            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
                z1v = int(z1_expr.getValue())
                z2v = int(z2_expr.getValue())
                combined = 0.5 * z1v + 0.5 * z2v
                if combined < best_combined:
                    best_combined = combined
                    best_z1 = z1v
                    best_z2 = z2v
                    best_open = [i for i in range(n) if open_var[i].X > 0.5]
                    best_assign = {}
                    for j_idx in range(m):
                        for i_idx in range(n):
                            if assign1[i_idx, j_idx].X > 0.5:
                                best_assign[str(j_idx)] = i_idx
                                break
                    if logger:
                        logger.log(best_combined)
                    update_pareto(z1v, z2v, best_open[:], dict(best_assign))
        except:
            pass

    # Final attempt with equal weights if needed
    if best_combined == float('inf') or time.time() - start_time < args.time_limit * 0.95:
        solve_for_weights(1, 1, args.time_limit - (time.time() - start_time))

    # If we still have no solution, run a fast greedy heuristic
    if best_combined == float('inf'):
        # Try to open all facilities with sufficient capacity
        opened = list(range(n))
        opened.sort(key=lambda i: (f1[i] + f2[i]) / capacities[i])
        cum_cap = 0
        selected = []
        for i in opened:
            selected.append(i)
            cum_cap += capacities[i]
            if cum_cap >= total_demand:
                break
        if cum_cap >= total_demand:
            # Assign using combined costs
            combined_cost = [ [c1[i][j] + c2[i][j] for j in range(m)] for i in range(n)]
            assignment = [-1] * m
            assigned_demand = [0] * n
            customer_list = list(range(m))
            customer_list.sort(key=lambda j: demands[j], reverse=True)
            for j in customer_list:
                best_i = -1
                best_c = float('inf')
                for i in selected:
                    if assigned_demand[i] + demands[j] <= capacities[i]:
                        if combined_cost[i][j] < best_c:
                            best_c = combined_cost[i][j]
                            best_i = i
                if best_i != -1:
                    assignment[j] = best_i
                    assigned_demand[best_i] += demands[j]
            if -1 not in assignment:
                z1_val = sum(f1[i] for i in selected)
                z2_val = sum(f2[i] for i in selected)
                for j in range(m):
                    i = assignment[j]
                    z1_val += c1[i][j]
                    z2_val += c2[i][j]
                combined = 0.5 * z1_val + 0.5 * z2_val
                if combined < best_combined:
                    best_combined = combined
                    best_z1 = int(z1_val)
                    best_z2 = int(z2_val)
                    best_open = sorted(selected)
                    best_assign = {str(j): assignment[j] for j in range(m)}
                    update_pareto(best_z1, best_z2, best_open, best_assign)
                    if logger:
                        logger.log(best_combined)

    # Final Pareto cleanup
    if pareto_points:
        pareto_points.sort(key=lambda x: x[0])
        final_pareto = []
        final_sols = []
        for i, pt in enumerate(pareto_points):
            dominated = False
            for k, other in enumerate(pareto_points):
                if k == i:
                    continue
                if other[0] <= pt[0] and other[1] <= pt[1] and (other[0] < pt[0] or other[1] < pt[1]):
                    dominated = True
                    break
            if not dominated:
                final_pareto.append(pt)
                final_sols.append(pareto_solutions[i])
        pareto_points = final_pareto
        pareto_solutions = final_sols

    # Build output
    output = {
        "objective_value": float(best_combined),
        "pareto_front": pareto_points if pareto_points else [[best_z1, best_z2]] if best_z1 is not None else [[0, 0]],
        "solutions": pareto_solutions if pareto_solutions else [{
            "z1": best_z1 if best_z1 is not None else 0,
            "z2": best_z2 if best_z2 is not None else 0,
            "open_facilities": best_open if best_open is not None else [],
            "assignments": best_assign if best_assign is not None else {}
        }]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()