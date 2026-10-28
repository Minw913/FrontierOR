import sys
import json
import argparse
import time

def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_items = data['num_items']
    num_objectives = data['num_objectives']
    weights = data['weights']
    capacity = data['capacity']
    objective_coefficients = data['objective_coefficients']
    
    logger = None
    if log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(log_path, sense="minimize")
        except ImportError:
            pass

    # Dynamic Programming state variable.
    # dp tracks the non-dominated states in the space of (weight, obj1, ..., objM)
    # State representation: (weight, obj_tuple, mask)
    # Start with the empty knapsack: weight 0, all objectives 0, nothing selected
    dp = [ (0, tuple([0] * num_objectives), 0) ]
    
    for i in range(num_items):
        # Time limit check with a tiny buffer to gracefully terminate
        if time.time() - start_time > time_limit - 0.1:
            break
            
        w_i = weights[i]
        obj_i = tuple(objective_coefficients[o][i] for o in range(num_objectives))
        
        new_states = dp.copy()
        for w, obj, mask in dp:
            new_w = w + w_i
            if new_w <= capacity:
                new_obj = tuple(obj[o] + obj_i[o] for o in range(num_objectives))
                new_mask = mask | (1 << i)
                new_states.append((new_w, new_obj, new_mask))
                
        # Filter combined states to keep only mutually non-dominated states
        # By sorting first by weight and then lexicographically by objectives,
        # we only need to compare any new state against previously accepted ones.
        new_states.sort(key=lambda x: (x[0], x[1]))
        filtered = []
        for state in new_states:
            w_state, obj_state, _ = state
            is_dominated = False
            for f_w, f_obj, _ in filtered:
                # Since sorted, f_w <= w_state is guaranteed.
                # If f_obj is <= obj_state in all dimensions, f dominates or is functionally identical to the current state.
                if all(f_obj[o] <= obj_state[o] for o in range(num_objectives)):
                    is_dominated = True
                    break
            if not is_dominated:
                filtered.append(state)
        
        dp = filtered
        
        if logger:
            # Recompute current actual Pareto front size on the achievable objectives only
            temp_dp = sorted(dp, key=lambda x: x[1])
            curr_pareto_len = 0
            curr_pareto_objs = []
            for state in temp_dp:
                _, obj_state, _ = state
                is_dom = False
                for p_obj in curr_pareto_objs:
                    if all(p_obj[o] <= obj_state[o] for o in range(num_objectives)):
                        is_dom = True
                        break
                if not is_dom:
                    curr_pareto_objs.append(obj_state)
                    curr_pareto_len += 1
            logger.log(curr_pareto_len)
            
    # Final projection and filter to obtain the true Pareto front on objectives alone
    dp.sort(key=lambda x: x[1])
    pareto = []
    for state in dp:
        w_state, obj_state, mask = state
        is_dominated = False
        for p_obj, p_mask in pareto:
            if all(p_obj[o] <= obj_state[o] for o in range(num_objectives)):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append((obj_state, mask))
            
    # Formatting output according strictly to required solution_schema.json format
    objective_value = len(pareto)
    non_dominated_objective_vectors = []
    solutions = []
    
    for obj_tuple, mask in pareto:
        non_dominated_objective_vectors.append(list(obj_tuple))
        sol = [(mask >> i) & 1 for i in range(num_items)]
        solutions.append(sol)
        
    output = {
        "objective_value": objective_value,
        "non_dominated_objective_vectors": non_dominated_objective_vectors,
        "solutions": solutions
    }
    
    with open(solution_path, 'w') as f:
        json.dump(output, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Multi-Objective Knapsack Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, help="Path to log JSONL file")
    
    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == "__main__":
    main()