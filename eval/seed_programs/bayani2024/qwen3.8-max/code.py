import argparse
import json
import time
import sys

def compute_objective(assignment, n_clients, linear_costs, quadratic_costs):
    """Compute the total objective value for a given assignment."""
    obj = 0.0
    for i in range(n_clients):
        obj += linear_costs[i][assignment[i]]
    for entry in quadratic_costs:
        c1, s1, c2, s2, cost = entry
        c1, s1, c2, s2 = int(c1), int(s1), int(c2), int(s2)
        if assignment[c1] == s1 and assignment[c2] == s2:
            obj += cost
    return obj

def local_search(assignment, n_clients, n_servers, linear_costs, quadratic_costs, 
                 time_limit, logger, start_time):
    """Improve solution via local search (single moves)."""
    quad_by_client = {i: [] for i in range(n_clients)}
    for entry in quadratic_costs:
        c1, s1, c2, s2, cost = entry
        c1, s1, c2, s2 = int(c1), int(s1), int(c2), int(s2)
        quad_by_client[c1].append((c2, s2, s1, cost))
        quad_by_client[c2].append((c1, s1, s2, cost))
    
    current_obj = compute_objective(assignment, n_clients, linear_costs, quadratic_costs)
    best_obj = current_obj
    improved = True
    
    while improved and (time.time() - start_time) < time_limit - 1:
        improved = False
        for i in range(n_clients):
            if time.time() - start_time > time_limit - 0.5:
                break
            old_server = assignment[i]
            best_new_server = old_server
            best_delta = 0.0
            
            for new_server in range(n_servers):
                if new_server == old_server:
                    continue
                delta = linear_costs[i][new_server] - linear_costs[i][old_server]
                
                for (other_c, other_s, my_s, cost) in quad_by_client[i]:
                    if assignment[other_c] == other_s:
                        if my_s == old_server:
                            delta -= cost
                        if my_s == new_server:
                            delta += cost
                
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_new_server = new_server
            
            if best_new_server != old_server:
                assignment[i] = best_new_server
                current_obj += best_delta
                improved = True
                
                if current_obj < best_obj - 1e-12:
                    best_obj = current_obj
                    if logger:
                        sol_dict = {
                            "objective_value": best_obj,
                            "assignment": {str(k): v for k, v in assignment.items()}
                        }
                        logger.log_solution(best_obj, sol_dict)
    
    return assignment, best_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n_clients = instance['n_clients']
    n_servers = instance['n_servers']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']
    
    remaining_time = args.time_limit - (time.time() - start_time)
    
    assignment = None
    objective_value = None
    
    # Try Gurobi
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        gurobi_time = max(1, int(remaining_time * 0.7))
        
        model = gp.Model()
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', gurobi_time)
        model.setParam('OutputFlag', 0)
        
        x = {}
        for i in range(n_clients):
            for j in range(n_servers):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        model.update()
        
        for i in range(n_clients):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n_servers)) == 1)
        
        obj = gp.LinExpr()
        for i in range(n_clients):
            for j in range(n_servers):
                obj += linear_costs[i][j] * x[i, j]
        
        qobj = gp.QuadExpr()
        for entry in quadratic_costs:
            c1, s1, c2, s2, cost = entry
            c1, s1, c2, s2 = int(c1), int(s1), int(c2), int(s2)
            qobj += cost * x[c1, s1] * x[c2, s2]
        
        model.setObjective(obj + qobj, GRB.MINIMIZE)
        
        best_gurobi_obj = float('inf')
        
        def callback(model, where):
            nonlocal best_gurobi_obj
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_gurobi_obj:
                    best_gurobi_obj = obj_val
                    if logger:
                        assign_dict = {}
                        for i in range(n_clients):
                            for j in range(n_servers):
                                if model.cbGetSolution(x[i, j]) > 0.5:
                                    assign_dict[str(i)] = j
                                    break
                        sol = {"objective_value": obj_val, "assignment": assign_dict}
                        logger.log_solution(obj_val, sol)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            assignment = {}
            for i in range(n_clients):
                for j in range(n_servers):
                    if x[i, j].X > 0.5:
                        assignment[i] = j
                        break
            objective_value = model.ObjVal
        else:
            assignment = None
            
    except Exception as e:
        assignment = None
    
    # If no Gurobi solution, use greedy
    if assignment is None:
        assignment = {}
        for i in range(n_clients):
            best_j = min(range(n_servers), key=lambda j: linear_costs[i][j])
            assignment[i] = best_j
        objective_value = compute_objective(assignment, n_clients, linear_costs, quadratic_costs)
        if logger:
            sol_dict = {
                "objective_value": objective_value,
                "assignment": {str(k): v for k, v in assignment.items()}
            }
            logger.log_solution(objective_value, sol_dict)
    
    # Local search improvement
    elapsed = time.time() - start_time
    if elapsed < args.time_limit - 2:
        assignment, objective_value = local_search(
            assignment, n_clients, n_servers, linear_costs, quadratic_costs,
            args.time_limit, logger, start_time
        )
    
    # Final objective verification
    objective_value = compute_objective(assignment, n_clients, linear_costs, quadratic_costs)
    
    solution = {
        "objective_value": objective_value,
        "assignment": {str(k): v for k, v in assignment.items()}
    }
    
    if logger:
        logger.log_solution(objective_value, solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()