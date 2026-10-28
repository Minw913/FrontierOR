import json
import argparse
import time
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def greedy_solution(num_trips, num_duties, duties, costs):
    """Find a greedy feasible solution using cost-per-new-trip heuristic."""
    covered = set()
    selected = []
    remaining = set(range(num_duties))
    
    while len(covered) < num_trips and remaining:
        best_j = None
        best_ratio = float('inf')
        
        for j in remaining:
            new_trips = set(duties[j]) - covered
            if len(new_trips) > 0:
                ratio = costs[j] / len(new_trips)
                if ratio < best_ratio or (ratio == best_ratio and len(new_trips) > len(set(duties[best_j]) - covered) if best_j is not None else True):
                    best_ratio = ratio
                    best_j = j
        
        if best_j is not None:
            selected.append(best_j)
            covered.update(duties[best_j])
            remaining.discard(best_j)
        else:
            break
    
    if len(covered) == num_trips:
        return selected, sum(costs[j] for j in selected)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_trips = instance['parameters']['num_trips_M']
    num_duties = instance['parameters']['num_duties_N']
    duties = instance['duties']
    costs = instance['costs']
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Build trip-to-duty mapping
    trip_to_duties = [[] for _ in range(num_trips)]
    for j in range(num_duties):
        for trip_id in duties[j]:
            trip_to_duties[trip_id].append(j)
    
    # Try greedy solution first as initial incumbent
    best_selected = None
    best_obj = float('inf')
    
    greedy_sel, greedy_obj = greedy_solution(num_trips, num_duties, duties, costs)
    if greedy_sel is not None:
        best_selected = greedy_sel
        best_obj = greedy_obj
        if logger:
            logger.log_solution(best_obj, {"objective_value": best_obj, "selected_duties": best_selected})
    
    # Check remaining time
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    
    if remaining_time > 0.5:
        # Create Gurobi model
        model = Model("set_partitioning")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(remaining_time - 0.5, 0.1))
        model.setParam('OutputFlag', 0)
        
        # Variables
        x = model.addVars(num_duties, vtype=GRB.BINARY, name="x")
        
        # Objective: minimize total cost
        model.setObjective(quicksum(costs[j] * x[j] for j in range(num_duties)), GRB.MINIMIZE)
        
        # Constraints: each trip covered exactly once
        for i in range(num_trips):
            model.addConstr(
                quicksum(x[j] for j in trip_to_duties[i]) == 1,
                name=f"cover_trip_{i}"
            )
        
        # Provide warm start from greedy solution
        if best_selected is not None:
            selected_set = set(best_selected)
            for j in range(num_duties):
                x[j].Start = 1.0 if j in selected_set else 0.0
        
        # Callback to log incumbent solutions
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                selected = [j for j in range(num_duties) if model.cbGetSolution(x[j]) > 0.5]
                solution_dict = {
                    "objective_value": obj_val,
                    "selected_duties": selected
                }
                if logger:
                    logger.log_solution(obj_val, solution_dict)
        
        model.setCallback(callback)
        
        # Solve
        model.optimize()
        
        # Extract best solution from Gurobi
        if model.SolCount > 0:
            selected = [j for j in range(num_duties) if x[j].X > 0.5]
            obj_val = sum(costs[j] for j in selected)
            if obj_val < best_obj:
                best_selected = selected
                best_obj = obj_val
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "selected_duties": best_selected})
    
    # Write final solution
    if best_selected is None:
        # Last resort: try a simple sequential greedy
        best_selected = []
        covered = set()
        for j in range(num_duties):
            duty_trips = set(duties[j])
            if not duty_trips.intersection(covered):
                best_selected.append(j)
                covered.update(duty_trips)
                if len(covered) == num_trips:
                    break
        best_obj = sum(costs[j] for j in best_selected)
    
    solution = {
        "objective_value": best_obj,
        "selected_duties": best_selected
    }
    
    if logger:
        logger.log_solution(best_obj, solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()