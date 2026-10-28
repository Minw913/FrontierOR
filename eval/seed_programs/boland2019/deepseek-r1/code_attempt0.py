import gurobipy as gp
from gurobipy import GRB
import json
import time
import argparse
from solution_logger import SolutionLogger

def build_model(data):
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    demands = data['demands']
    capacities = data['capacities']
    fixed_costs_obj1 = data['fixed_costs_obj1']
    fixed_costs_obj2 = data['fixed_costs_obj2']
    assignment_costs_obj1 = data['assignment_costs_obj1']
    assignment_costs_obj2 = data['assignment_costs_obj2']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_facilities, num_customers, vtype=GRB.BINARY, name="y")
    
    for j in range(num_customers):
        model.addConstr(gp.quicksum(y[i, j] for i in range(num_facilities)) == 1, name=f"assign_{j}")
    
    for i in range(num_facilities):
        for j in range(num_customers):
            model.addConstr(y[i, j] <= x[i], name=f"open_{i}_{j}")
    
    for i in range(num_facilities):
        model.addConstr(gp.quicksum(demands[j] * y[i, j] for j in range(num_customers)) <= capacities[i] * x[i], name=f"cap_{i}")
    
    z1 = gp.quicksum(fixed_costs_obj1[i] * x[i] for i in range(num_facilities)) + \
         gp.quicksum(assignment_costs_obj1[i][j] * y[i, j] for i in range(num_facilities) for j in range(num_customers))
    
    z2 = gp.quicksum(fixed_costs_obj2[i] * x[i] for i in range(num_facilities)) + \
         gp.quicksum(assignment_costs_obj2[i][j] * y[i, j] for i in range(num_facilities) for j in range(num_customers))
    
    return model, x, y, z1, z2

def extract_solution(model, x, y, num_facilities, num_customers):
    open_facilities = []
    for i in range(num_facilities):
        if x[i].X > 0.5:
            open_facilities.append(i)
    
    assignments = {}
    for j in range(num_customers):
        for i in range(num_facilities):
            if y[i, j].X > 0.5:
                assignments[str(j)] = i
                break
    return open_facilities, assignments

def update_pareto_front(pareto_front, solutions, z1_val, z2_val, solution_record):
    non_dominated = True
    new_pareto_front = []
    new_solutions = []
    for i, (z1, z2) in enumerate(pareto_front):
        if z1 <= z1_val and z2 <= z2_val and (z1 < z1_val or z2 < z2_val):
            non_dominated = False
        if z1_val <= z1 and z2_val <= z2 and (z1_val < z1 or z2_val < z2):
            continue
        new_pareto_front.append((z1, z2))
        new_solutions.append(solutions[i])
    if non_dominated:
        new_pareto_front.append((z1_val, z2_val))
        new_solutions.append(solution_record)
    return new_pareto_front, new_solutions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    
    model, x, y, z1_expr, z2_expr = build_model(data)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    pareto_front = []
    solutions = []
    best_combined = float('inf')
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        print("Time limit exceeded before starting optimization.")
        output_solution(pareto_front, solutions, args.solution_path)
        return
    
    model.setParam('TimeLimit', max(1, remaining_time))
    
    # Lex min for z1 then z2
    model.setObjective(z1_expr, GRB.MINIMIZE)
    model.optimize()
    lex1_point = None
    if model.status == GRB.OPTIMAL:
        z1_val = model.ObjVal
        constr_z1 = model.addConstr(z1_expr == z1_val, "constr_z1")
        model.setObjective(z2_expr, GRB.MINIMIZE)
        model.optimize()
        if model.status == GRB.OPTIMAL:
            z2_val = z2_expr.getValue()
            open_facilities, assignments = extract_solution(model, x, y, num_facilities, num_customers)
            sol_dict = {
                "z1": int(z1_val),
                "z2": int(z2_val),
                "open_facilities": open_facilities,
                "assignments": assignments
            }
            combined = 0.5 * z1_val + 0.5 * z2_val
            if combined < best_combined:
                best_combined = combined
                if logger:
                    logger.log_solution(combined, sol_dict)
            pareto_front, solutions = update_pareto_front(pareto_front, solutions, z1_val, z2_val, sol_dict)
            lex1_point = (z1_val, z2_val)
        model.remove(constr_z1)
    
    # Lex min for z2 then z1
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 0:
        model.setParam('TimeLimit', max(1, remaining_time))
        model.setObjective(z2_expr, GRB.MINIMIZE)
        model.optimize()
        lex2_point = None
        if model.status == GRB.OPTIMAL:
            z2_val = model.ObjVal
            constr_z2 = model.addConstr(z2_expr == z2_val, "constr_z2")
            model.setObjective(z1_expr, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                z1_val = z1_expr.getValue()
                open_facilities, assignments = extract_solution(model, x, y, num_facilities, num_customers)
                sol_dict = {
                    "z1": int(z1_val),
                    "z2": int(z2_val),
                    "open_facilities": open_facilities,
                    "assignments": assignments
                }
                combined = 0.5 * z1_val + 0.5 * z2_val
                if combined < best_combined:
                    best_combined = combined
                    if logger:
                        logger.log_solution(combined, sol_dict)
                pareto_front, solutions = update_pareto_front(pareto_front, solutions, z1_val, z2_val, sol_dict)
                lex2_point = (z1_val, z2_val)
            model.remove(constr_z2)
    
    # Epsilon-constraint method
    if lex1_point and lex2_point:
        lex1_z1, lex1_z2 = lex1_point
        lex2_z1, lex2_z2 = lex2_point
        epsilon = lex1_z2 - 1
        if epsilon >= lex2_z2:
            constr_eps = model.addConstr(z2_expr <= epsilon, "epsilon_constraint")
            while epsilon >= lex2_z2 and (time.time() - start_time) < args.time_limit:
                remaining_time = args.time_limit - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                model.setParam('TimeLimit', max(1, remaining_time))
                model.setObjective(z1_expr, GRB.MINIMIZE)
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    z1_val = model.ObjVal
                    z2_val = z2_expr.getValue()
                    open_facilities, assignments = extract_solution(model, x, y, num_facilities, num_customers)
                    sol_dict = {
                        "z1": int(z1_val),
                        "z2": int(z2_val),
                        "open_facilities": open_facilities,
                        "assignments": assignments
                    }
                    combined = 0.5 * z1_val + 0.5 * z2_val
                    if combined < best_combined:
                        best_combined = combined
                        if logger:
                            logger.log_solution(combined, sol_dict)
                    pareto_front, solutions = update_pareto_front(pareto_front, solutions, z1_val, z2_val, sol_dict)
                    epsilon = z2_val - 1
                    constr_eps.setAttr('rhs', epsilon)
                else:
                    break
            model.remove(constr_eps)
    
    # Find solution minimizing combined cost
    min_combined_sol = None
    min_combined_val = float('inf')
    for sol in solutions:
        combined = 0.5 * sol['z1'] + 0.5 * sol['z2']
        if combined < min_combined_val:
            min_combined_val = combined
            min_combined_sol = sol
    
    # Prepare output
    output_dict = {
        "objective_value": min_combined_val,
        "pareto_front": [[int(p[0]), int(p[1])] for p in pareto_front],
        "solutions": solutions
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_dict, f, indent=2)

if __name__ == "__main__":
    main()