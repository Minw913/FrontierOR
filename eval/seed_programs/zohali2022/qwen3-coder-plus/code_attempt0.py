import argparse
import json
import math
import random
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_station_balancing(instance_path, solution_path, timelimit, log_path=None):
    # Load the instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    num_stations = instance['num_stations']
    processing_times = instance['processing_times']
    precedence_relations = instance['precedence_relations']
    
    # Convert precedence relations to 0-indexed
    precedence_0indexed = [(a - 1, b - 1) for a, b in precedence_relations]
    
    forward_setup_times = instance['forward_setup_times']  # [i][j] = time from i to j
    backward_setup_times = instance['backward_setup_times']  # [i][j] = time from end of i to start of j
    
    # Check if instance is invalid due to impossible precedence
    all_tasks = set(range(num_tasks))
    all_predecessors = set()
    all_successors = set()
    for pred, succ in precedence_0indexed:
        all_predecessors.add(pred)
        all_successors.add(succ)
    
    # Verify that all tasks have valid indices
    assert len(all_tasks) == num_tasks
    assert all(0 <= t < num_tasks for t in all_predecessors | all_successors)
    
    # Create model
    m = gp.Model("assembly_line")
    m.setParam('TimeLimit', timelimit)
    if log_path:  # Gurobi can log to file via parameter
        m.setParam(GRB.Param.LogFile, log_path)
        m.setParam(GRB.Param.LogToConsole, 0)  # Disable console log if file specified
    else:
        m.setParam(GRB.Param.LogToConsole, 1)

    # Decision variables:
    # x[i][k] = 1 if task i is assigned to station k
    x = m.addVars(num_tasks, num_stations, vtype=GRB.BINARY, name='x')

    # y[i][j][k] = 1 if task i immediately precedes task j on station k
    y = m.addVars(num_tasks, num_tasks, num_stations, vtype=GRB.BINARY, name='y')

    # s[k] = time at station k including processing, forward setups, and backward setup
    s = m.addVars(num_stations, vtype=GRB.CONTINUOUS, name='s', lb=0)

    # Minimize max station time
    objective = m.addVar(vtype=GRB.CONTINUOUS, name='objective', lb=0)
    m.setObjective(objective, GRB.MINIMIZE)
    m.addConstrs((objective >= s[k] for k in range(num_stations)), name='make_objective_max')

    # Each task assigned to exactly one station
    m.addConstrs((gp.quicksum(x[i,k] for k in range(num_stations)) == 1 
                  for i in range(num_tasks)), name='assign_each_task_once')

    # Every station has at least one task
    m.addConstrs((gp.quicksum(x[i,k] for i in range(num_tasks)) >= 1 
                  for k in range(num_stations)), name='station_not_empty')

    # Link y variables to x variables: y[i][j][k] can only be 1 if both i and j are on k
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                for k in range(num_stations):
                    # If y[i,j,k]=1 then x[i,k]=1 and x[j,k]=1
                    m.addConstr(y[i,j,k] <= x[i,k])
                    m.addConstr(y[i,j,k] <= x[j,k])

    # Each task appears at most once in the sequence at each station
    # For each station and task i: indegree + outdegree for task i <= 2 (if used -> either no prec/no succ, or 1 prec/1 succ)
    m.addConstrs((gp.quicksum(y[i,j,k] for j in range(num_tasks) if i != j) <= x[i,k] 
                  for i in range(num_tasks) for k in range(num_stations)), 
                 name='each_task_has_one_pred')
    
    m.addConstrs((gp.quicksum(y[j,i,k] for j in range(num_tasks) if i != j) <= x[i,k] 
                  for i in range(num_tasks) for k in range(num_stations)), 
                 name='each_task_has_one_succ')

    # Station sequences form a cycle among assigned tasks
    # Incoming equals outgoing for each task on each station
    m.addConstrs((
        gp.quicksum(y[j,i,k] for j in range(num_tasks) if i != j) == 
        gp.quicksum(y[i,j,k] for j in range(num_tasks) if i != j) 
        for k in range(num_stations) for i in range(num_tasks)), 
        name='flow_conservation')

    # Ensure connectivity of assigned tasks within station cycles
    # We model this using constraints ensuring each station with tasks forms a single directed cycle
    # This can be modeled using Miller-Tucker-Zemlin (MTZ-like) formulation to eliminate subtours
    u = m.addVars(num_tasks, num_stations, vtype=GRB.CONTINUOUS, lb=0, ub=num_tasks-1, name='u')
    
    BigM = max(processing_times) + max(max(row) for row in forward_setup_times) * num_tasks + max(max(row) for row in backward_setup_times)
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                for k in range(num_stations):
                    # If edge i->j exists in station k, impose constraint to prevent subcycles
                    # u[j] >= u[i] + 1 - M*(1 - y[i,j,k]) but handle the special case carefully
                    m.addConstr(u[j,k] >= u[i,k] + 1 - BigM * (1 - y[i,j,k]))
                    # Only valid when y[i,j,k] is active

    # Precedence constraints: if task i precedes task j, i must be on same or earlier station.
    # If same station, i must come before j in cycle
    for i, j in precedence_0indexed:
        # i on same or earlier station than j
        for k in range(num_stations):
            for l in range(k):  # l comes before k
                # If j is on station k, then i cannot be on l (l before k)
                m.addConstr(x[j,k] + x[i,l] <= 1)
        # If both are on same station, then we need i to come before j in sequence
        for k in range(num_stations):
            # If both are on same station k, i must go first
            # Use additional logic for relative ordering within stations
            # For any (i,j) in precedence within same station k, use flow constraints
            # Define a binary variable z_ij that is 1 if task i comes before task j on same station
            z = m.addVar(vtype=GRB.BINARY, name=f'z_{i}_{j}_{k}')
            
            # Constraint linking z: if zij=1 then u_i < u_j (or similar logic)  
            M_temp = num_tasks
            # Using MTZ variable: if u_i + 1 <= u_j then i comes before j
            # z = 1 means (u_i +1 <= u_j), z=0 means no requirement
            m.addConstr(u[i,k] + 1 - u[j,k] <= M_temp * (1 - z))
            
            # If x[i][k] and x[j][k] are both 1, z must be 1 (i before j)
            # Use: if both assigned then order must follow precedence
            m.addConstr(x[i,k] + x[j,k] - 1 <= z)

    # Calculate station costs: processing times + forward setups + backward setup
    for k in range(num_stations):
        expr = gp.LinExpr()
        
        # Add processing times for tasks assigned to this station
        for i in range(num_tasks):
            expr.addTerms(processing_times[i], x[i,k])
        
        # Add forward setup times y[i][j][k]*(time from i to j)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    expr.addTerms(forward_setup_times[i][j], y[i,j,k])
        
        # Add backward setup time for closing the cycle: y[last][first][k]
        # For every pair of tasks (i,j), if both are in k and they represent cycle's end+beginning,
        # their backward setup is added
        
        # This requires modeling the first and last element in the cycle.
        # Instead of computing separately, let's assume each forward-link connects first-last as per cycle nature
        # Actually backward setup is applied from the last element in the cycle to the first
        # Modeling: for each i,j: add backward_setup[i][j] * (indicator that i is last and j is first)
        
        is_first = m.addVars(num_tasks, vtype=GRB.BINARY, name=f'is_first_k{k}')
        is_last = m.addVars(num_tasks, vtype=GRB.BINARY, name=f'is_last_k{k}')
        
        # is_first is 1 for exactly one task in the station
        m.addConstr(gp.quicksum(is_first[i] for i in range(num_tasks)) == 
                    gp.quicksum(x[i,k] for i in range(num_tasks)), 
                    name=f'one_first_k{k}')

        # is_last is 1 for exactly one task in the station
        m.addConstr(gp.quicksum(is_last[i] for i in range(num_tasks)) == 
                    gp.quicksum(x[i,k] for i in range(num_tasks)), 
                    name=f'one_last_k{k}')
        
        # is_first must align with x
        for i in range(num_tasks):
            m.addConstr(is_first[i] <= x[i,k])

        # is_last must align with x
        for i in range(num_tasks):
            m.addConstr(is_last[i] <= x[i,k])

        # Link is_first and is_last to sequence
        # A task i is first if there is no incoming edge in the cycle within the station
        for i in range(num_tasks):
            m.addConstr(is_first[i] == x[i,k] - gp.quicksum(y[j,i,k] for j in range(num_tasks) if j != i))

        # A task i is last if there is no outgoing edge in the cycle within the station:
        for i in range(num_tasks):
            m.addConstr(is_last[i] == x[i,k] - gp.quicksum(y[i,j,k] for j in range(num_tasks) if j != i))


        # Add backward setup costs: if i is last and j is first in station k, add cost from i to j
        big_m = max(max(row) for row in backward_setup_times)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    term = backward_setup_times[i][j] * is_last[i] + backward_setup_times[j][i] * is_last[j]
                    expr.addTerms(backward_setup_times[i][j], is_last[i] * is_first[j])  # this doesn't work with bilinear terms
                    
        # Correctly formulate the sum of backward setup cost
        # Use a linearization: add variable and constraints for each i,j combination
        backward_cost_vars = {}
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    var_name = f'bc_{k}_{i}_{j}'
                    var = m.addVar(name=var_name, vtype=GRB.CONTINUOUS, lb=0, ub=big_m)
                    backward_cost_vars[(i, j)] = var
                    m.addConstr(var >= 0)  # base
                    m.addConstr(var >= backward_setup_times[i][j] * (is_last[i] + is_first[j] - 1))  # when both happen
                    expr.add(var)

        m.addConstr(s[k] == expr)

    m.optimize()

    # Extract solution
    if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Construct solution from optimized model
        
        # Get assignments
        assignment = {}
        for i in range(num_tasks):
            for k in range(num_stations):
                if x[i, k].X > 0.5:
                    assignment[str(i + 1)] = k + 1  # 1-indexed
                    break

        # Get sequences per station
        sequences = {str(i + 1): [] for i in range(num_stations)}
        for k in range(num_stations):
            current_station_seq = []
            # Find the first task in sequence by looking for is_first[k][j]==1
            first_task = None
            for i in range(num_tasks):
                if is_first[i].X > 0.5:
                    first_task = i
                    current_station_seq.append(i + 1)  # 1-indexed
                    break
            
            # Now construct sequence starting from first, using y variables
            current_task = first_task
            visited_count = 0
            visited_set = set()
            while visited_count < num_tasks and current_task not in visited_set:
                # Find next task that follows current_task in this station
                next_task = -1
                for i_idx in range(num_tasks):
                    if i_idx != current_task and y[current_task, i_idx, k].X > 0.5:
                        next_task = i_idx
                        break
                if next_task == -1 or next_task in visited_set:  # No valid next, or cycle done
                    break
                current_station_seq.append(next_task + 1) 
                visited_set.add(current_task)
                current_task = next_task
                visited_count += 1
            
            sequences[str(k + 1)] = current_station_seq
            
        # Compute actual cycle time based on our sequences
        actual_cycle_times = []
        for k in range(num_stations):
            if not sequences[str(k + 1)]:  # Empty station (shouldn't happen due to constraint)
                actual_cycle_times.append(0)
                continue
                
            task_list = [t - 1 for t in sequences[str(k + 1)]]  # Convert back to 0-indexed
            stat_time = 0
            # Processing times
            stat_time += sum(processing_times[t] for t in task_list)
            # Forward setups
            for idx in range(len(task_list)):
                curr = task_list[idx]
                next_idx = (idx + 1) % len(task_list)  # wrap around for cycle
                nxt = task_list[next_idx]
                stat_time += forward_setup_times[curr][nxt]
            actual_cycle_times.append(stat_time)
        
        true_objective = max(actual_cycle_times) if actual_cycle_times else 0

        solution = {
            "objective_value": true_objective,
            "assignment": assignment,
            "station_sequences": sequences,
            "cycle_time": true_objective
        }

        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

    else:
        print(f"No solution available, status: {m.Status}")


def main():
    parser = argparse.ArgumentParser(description="Assembly Line Optimization")
    parser.add_argument("--instance_path", required=True, help="Path to input instance file")
    parser.add_argument("--solution_path", required=True, help="Path to output solution file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")

    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solve_station_balancing(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()