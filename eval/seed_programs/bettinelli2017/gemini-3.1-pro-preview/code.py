import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_items = data['num_items']
    capacity = data['capacity']
    profits = data['items']['profits']
    weights = data['items']['weights']
    edges = data['conflict_graph']['edges']
    
    # Initialize logger
    logger = None
    if log_path and SolutionLogger is not None:
        logger = SolutionLogger(log_path, sense="maximize")
    
    # Pre-process for initial heuristic
    adj = {i: [] for i in range(num_items)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def greedy(sort_key):
        order = sorted(range(num_items), key=sort_key, reverse=True)
        current_weight = 0
        selected = set()
        blocked = [False] * num_items
        for i in order:
            if not blocked[i] and current_weight + weights[i] <= capacity:
                selected.add(i)
                current_weight += weights[i]
                for v in adj[i]:
                    blocked[v] = True
        return selected

    # Evaluate multiple greedy strategies to get a strong initial bound
    s1 = greedy(lambda i: profits[i] / weights[i])
    s2 = greedy(lambda i: profits[i])
    
    val1 = sum(profits[i] for i in s1)
    val2 = sum(profits[i] for i in s2)
    
    if val1 > val2:
        best_init, best_val = s1, val1
    else:
        best_init, best_val = s2, val2
        
    best_obj_so_far = float(best_val)
    if logger:
        logger.log(best_obj_so_far)
        
    # Setup Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('LogToConsole', 0)
    env.start()
    
    model = gp.Model("KnapsackWithConflicts", env=env)
    
    # Restrict to a single CPU core as per requirements
    model.setParam('Threads', 1)
    
    if time_limit is not None and time_limit > 0:
        model.setParam('TimeLimit', time_limit)
        
    # Decision Variables
    x = model.addVars(num_items, vtype=GRB.BINARY, name="x")
    
    # Warm start with the best greedy solution discovered
    for i in range(num_items):
        x[i].Start = 1 if i in best_init else 0
        
    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity, name="Capacity")
    
    # Conflict constraints (at most one item from any conflict edge)
    model.addConstrs((x[u] + x[v] <= 1 for u, v in edges), name="Conflict")
    
    # Objective function
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(num_items)), GRB.MAXIMIZE)
    
    # Callback to log any improved feasible solutions found by the solver
    def callback(m, where):
        nonlocal best_obj_so_far
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > best_obj_so_far + 1e-5:
                best_obj_so_far = obj
                if logger:
                    logger.log(obj)
                    
    model.optimize(callback)
    
    # Extract final optimal or best found solution
    if model.SolCount > 0:
        objective_value = float(round(model.ObjVal))
        selected_items = [i for i in range(num_items) if x[i].X > 0.5]
    else:
        # Fallback mechanism in case Gurobi does not capture a solution internally
        objective_value = float(best_val)
        selected_items = list(best_init)
        
    solution = {
        "objective_value": objective_value,
        "selected_items": selected_items
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the problem instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions")
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)