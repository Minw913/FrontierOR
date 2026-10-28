import json
import argparse
import time
import sys
from gurobipy import *


def evaluate_solution(y, num_facilities, num_customers, demands, fixed_costs,
                      trans_cost, penalty_cost, disruption_probs, sorted_facilities):
    """Evaluate the objective value for a given facility opening decision."""
    total = sum(fixed_costs[j] * y[j] for j in range(num_facilities))
    for i in range(num_customers):
        prob_unavail = 1.0
        exp_cost = 0.0
        for j in sorted_facilities[i]:
            if y[j] == 1:
                prob_serves = prob_unavail * (1.0 - disruption_probs[j])
                exp_cost += trans_cost[i][j] * prob_serves
                prob_unavail *= disruption_probs[j]
        exp_cost += penalty_cost[i] * prob_unavail
        total += demands[i] * exp_cost
    return total


def greedy_heuristic(num_facilities, num_customers, demands, fixed_costs,
                     trans_cost, penalty_cost, disruption_probs, sorted_facilities,
                     time_limit, logger, start_time):
    """Greedy construction + local search heuristic for large instances."""
    
    def evaluate(y):
        return evaluate_solution(y, num_facilities, num_customers, demands,
                                 fixed_costs, trans_cost, penalty_cost,
                                 disruption_probs, sorted_facilities)
    
    y = [0] * num_facilities
    current_cost = evaluate(y)
    
    best_y = y[:]
    best_cost = current_cost
    
    if logger:
        logger.log_solution(best_cost, {"objective_value": best_cost, "facility_locations": best_y[:]})
    
    # Greedy construction: add facilities one at a time
    while time.time() - start_time < time_limit * 0.6:
        best_j = -1
        best_new_cost = current_cost
        
        for j in range(num_facilities):
            if y[j] == 0:
                y[j] = 1
                new_cost = evaluate(y)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_j = j
                y[j] = 0
        
        if best_j >= 0 and best_new_cost < current_cost:
            y[best_j] = 1
            current_cost = best_new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_y = y[:]
                if logger:
                    logger.log_solution(best_cost, {"objective_value": best_cost, "facility_locations": best_y[:]})
        else:
            break
    
    # Local search: try closing each open facility
    improved = True
    while improved and time.time() - start_time < time_limit * 0.9:
        improved = False
        for j in range(num_facilities):
            if y[j] == 1:
                y[j] = 0
                new_cost = evaluate(y)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_y = y[:]
                        if logger:
                            logger.log_solution(best_cost, {"objective_value": best_cost, "facility_locations": best_y[:]})
                else:
                    y[j] = 1
    
    # Local search: try opening each closed facility
    improved = True
    while improved and time.time() - start_time < time_limit * 0.95:
        improved = False
        for j in range(num_facilities):
            if y[j] == 0:
                y[j] = 1
                new_cost = evaluate(y)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_y = y[:]
                        if logger:
                            logger.log_solution(best_cost, {"objective_value": best_cost, "facility_locations": best_y[:]})
                else:
                    y[j] = 0
    
    return best_y, best_cost


def solve_mip(num_facilities, num_customers, demands, fixed_costs,
              trans_cost, penalty_cost, disruption_probs, sorted_facilities,
              time_limit, logger, initial_y=None):
    """Solve using MIP formulation with Gurobi."""
    
    model = Model("RFLP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(time_limit, 1)
    model.Params.OutputFlag = 0
    
    # Decision variables
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
    
    # u[i,k] = probability that first k cheapest facilities for customer i are all unavailable
    u = model.addVars(num_customers, num_facilities + 1, lb=0.0, ub=1.0, name="u")
    
    # v[i,k] = u[i,k-1] * y[j_k(i)] (linearized product)
    v = model.addVars(num_customers, num_facilities, lb=0.0, ub=1.0, name="v")
    
    # Constraints
    for i in range(num_customers):
        model.addConstr(u[i, 0] == 1.0)
        
        for k in range(num_facilities):
            j_k = sorted_facilities[i][k]
            q_j = disruption_probs[j_k]
            
            # Linearization: v[i,k] = u[i,k-1] * y[j_k]
            model.addConstr(v[i, k] <= u[i, k - 1])
            model.addConstr(v[i, k] <= y[j_k])
            model.addConstr(v[i, k] >= u[i, k - 1] + y[j_k] - 1)
            
            # Recurrence: u[i,k+1] = u[i,k] - v[i,k] * (1 - q_j)
            model.addConstr(u[i, k + 1] == u[i, k] - v[i, k] * (1.0 - q_j))
    
    # Objective
    obj = quicksum(fixed_costs[j] * y[j] for j in range(num_facilities))
    for i in range(num_customers):
        for k in range(num_facilities):
            j_k = sorted_facilities[i][k]
            q_j = disruption_probs[j_k]
            obj += demands[i] * trans_cost[i][j_k] * (1.0 - q_j) * v[i, k]
        obj += demands[i] * penalty_cost[i] * u[i, num_facilities]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Warm start
    if initial_y is not None:
        for j in range(num_facilities):
            y[j].Start = initial_y[j]
    
    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                y_vals = model.cbGetSolution([y[j] for j in range(num_facilities)])
                facility_locations = [int(round(y_vals[j])) for j in range(num_facilities)]
                if logger:
                    logger.log_solution(obj_val, {
                        "objective_value": obj_val,
                        "facility_locations": facility_locations
                    })
            except Exception:
                pass
    
    model.optimize(callback)
    
    if model.SolCount > 0:
        facility_locations = [int(round(y[j].X)) for j in range(num_facilities)]
        obj_val = model.ObjVal
        return facility_locations, obj_val
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Reliable Facility Location Problem")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    start_time = time.time()
    
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    nodes = instance['nodes']
    trans_cost = instance['transportation_cost_matrix']
    penalty_cost = instance['penalty_cost']
    
    demands = [nodes[i]['demand'] for i in range(num_customers)]
    fixed_costs = [nodes[j]['fixed_setup_cost'] for j in range(num_facilities)]
    disruption_probs = [nodes[j]['disruption_probability'] for j in range(num_facilities)]
    
    # For each customer, sort facilities by transportation cost
    sorted_facilities = []
    for i in range(num_customers):
        order = sorted(range(num_facilities), key=lambda j: trans_cost[i][j])
        sorted_facilities.append(order)
    
    # Compute initial feasible solution (open cheapest facility for each customer)
    initial_open = set()
    for i in range(num_customers):
        initial_open.add(sorted_facilities[i][0])
    initial_y = [1 if j in initial_open else 0 for j in range(num_facilities)]
    initial_cost = evaluate_solution(initial_y, num_facilities, num_customers,
                                     demands, fixed_costs, trans_cost,
                                     penalty_cost, disruption_probs, sorted_facilities)
    
    if logger:
        logger.log_solution(initial_cost, {
            "objective_value": initial_cost,
            "facility_locations": initial_y[:]
        })
    
    best_y = initial_y[:]
    best_cost = initial_cost
    
    # Decide approach based on problem size
    problem_size = num_facilities * num_customers
    
    if problem_size <= 250000:
        # Use MIP approach
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time > 1:
            try:
                mip_y, mip_cost = solve_mip(
                    num_facilities, num_customers, demands, fixed_costs,
                    trans_cost, penalty_cost, disruption_probs, sorted_facilities,
                    remaining_time, logger, initial_y
                )
                if mip_y is not None and mip_cost < best_cost:
                    best_y = mip_y
                    best_cost = mip_cost
            except Exception as e:
                print(f"MIP solve error: {e}", file=sys.stderr)
    else:
        # Use heuristic approach for large instances
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time > 1:
            try:
                heur_y, heur_cost = greedy_heuristic(
                    num_facilities, num_customers, demands, fixed_costs,
                    trans_cost, penalty_cost, disruption_probs, sorted_facilities,
                    remaining_time, logger, start_time
                )
                if heur_cost < best_cost:
                    best_y = heur_y
                    best_cost = heur_cost
            except Exception as e:
                print(f"Heuristic error: {e}", file=sys.stderr)
    
    # Write final solution
    solution = {
        "objective_value": best_cost,
        "facility_locations": best_y
    }
    
    if logger:
        logger.log_solution(best_cost, solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()