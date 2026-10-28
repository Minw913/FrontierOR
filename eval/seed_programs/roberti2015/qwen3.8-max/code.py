import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def greedy_heuristic(m, n, supply, demand, fixed_cost, variable_cost, capacity):
    """
    Greedy heuristic: iteratively assign flow to cheapest effective cost routes.
    Effective cost considers both variable and amortized fixed cost.
    """
    remaining_supply = supply[:]
    remaining_demand = demand[:]
    flow = [[0.0] * n for _ in range(m)]
    
    # Build list of all routes sorted by variable cost (primary) then fixed cost (secondary)
    routes = []
    for i in range(m):
        for j in range(n):
            routes.append((variable_cost[i][j], fixed_cost[i][j], i, j))
    routes.sort()
    
    # Greedily assign
    for _, _, i, j in routes:
        if remaining_supply[i] <= 0 or remaining_demand[j] <= 0:
            continue
        amt = min(remaining_supply[i], remaining_demand[j], capacity[i][j])
        if amt > 0:
            flow[i][j] = amt
            remaining_supply[i] -= amt
            remaining_demand[j] -= amt
    
    # Check feasibility
    for i in range(m):
        if remaining_supply[i] > 0:
            return None  # Infeasible (shouldn't happen with complete bipartite)
    
    # Compute objective
    obj = 0.0
    arcs_used = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if flow[i][j] > 0.5:
                arcs_used[i][j] = 1
                obj += fixed_cost[i][j] + variable_cost[i][j] * flow[i][j]
    
    return obj, flow, arcs_used


def solve_with_gurobi(m, n, supply, demand, fixed_cost, variable_cost, capacity, time_limit, logger):
    """Solve the fixed-charge transportation problem using Gurobi MIP."""
    
    model = gp.Model("FCTP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    # Decision variables
    x = {}
    y = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(lb=0, ub=capacity[i][j], name=f"x_{i}_{j}")
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    
    model.update()
    
    # Objective
    obj = gp.LinExpr()
    for i in range(m):
        for j in range(n):
            obj += variable_cost[i][j] * x[i, j] + fixed_cost[i][j] * y[i, j]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Supply constraints
    for i in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n)) == supply[i],
            name=f"supply_{i}"
        )
    
    # Demand constraints
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(m)) == demand[j],
            name=f"demand_{j}"
        )
    
    # Linking constraints
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] <= capacity[i][j] * y[i, j], name=f"link_{i}_{j}")
    
    # Add valid inequalities: if y[i,j]=1 then at least 1 unit must flow
    # This helps tighten the LP relaxation
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] >= y[i, j], name=f"minflow_{i}_{j}")
    
    model.update()
    
    # Set warm start from greedy heuristic if available
    greedy_result = greedy_heuristic(m, n, supply, demand, fixed_cost, variable_cost, capacity)
    if greedy_result is not None:
        g_obj, g_flow, g_arcs = greedy_result
        for i in range(m):
            for j in range(n):
                x[i, j].Start = g_flow[i][j]
                y[i, j].Start = g_arcs[i][j]
    
    # Callback to log incumbent solutions
    best_logged = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_logged[0]:
                best_logged[0] = obj_val
                flow_sol = [[model.cbGetSolution(x[i, j]) for j in range(n)] for i in range(m)]
                arcs_sol = [[int(round(model.cbGetSolution(y[i, j]))) for j in range(n)] for i in range(m)]
                solution_dict = {
                    "objective_value": obj_val,
                    "flow": flow_sol,
                    "arcs_used": arcs_sol
                }
                if logger:
                    logger.log_solution(obj_val, solution_dict)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        flow_sol = [[x[i, j].X for j in range(n)] for i in range(m)]
        arcs_sol = [[int(round(y[i, j].X)) for j in range(n)] for i in range(m)]
        
        # Recompute objective from solution to avoid numerical issues
        obj_val = 0.0
        for i in range(m):
            for j in range(n):
                if arcs_sol[i][j] == 1:
                    obj_val += fixed_cost[i][j]
                obj_val += variable_cost[i][j] * flow_sol[i][j]
        
        return obj_val, flow_sol, arcs_sol
    elif greedy_result is not None:
        return greedy_result
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fixed-Charge Transportation Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions.')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    m = instance['num_sources']
    n = instance['num_sinks']
    supply = instance['supply']
    demand = instance['demand']
    fixed_cost = instance['fixed_cost']
    variable_cost = instance['variable_cost']
    capacity = instance['capacity']

    start_time = time.time()

    # First get a greedy solution quickly
    greedy_result = greedy_heuristic(m, n, supply, demand, fixed_cost, variable_cost, capacity)
    if greedy_result is not None and logger:
        logger.log_solution(greedy_result[0], {
            "objective_value": greedy_result[0],
            "flow": greedy_result[1],
            "arcs_used": greedy_result[2]
        })

    # Solve with Gurobi
    remaining_time = max(1, args.time_limit - int(time.time() - start_time))
    result = solve_with_gurobi(m, n, supply, demand, fixed_cost, variable_cost, capacity,
                               remaining_time, logger)

    if result is not None:
        obj_val, flow_sol, arcs_sol = result
        solution = {
            "objective_value": obj_val,
            "flow": flow_sol,
            "arcs_used": arcs_sol
        }
    else:
        # Fallback: zero solution (should not happen for valid instances)
        solution = {
            "objective_value": 0.0,
            "flow": [[0.0] * n for _ in range(m)],
            "arcs_used": [[0] * n for _ in range(m)]
        }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()