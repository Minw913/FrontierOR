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
    forward_setup_times = instance['forward_setup_times']  
    backward_setup_times = instance['backward_setup_times']

    # Check types and lengths match
    assert len(processing_times) == num_tasks
    assert len(forward_setup_times) == num_tasks
    for row in forward_setup_times:
        assert len(row) == num_tasks
    assert len(backward_setup_times) == num_tasks
    for row in backward_setup_times:
        assert len(row) == num_tasks
        
    # Convert to 0-indexed
    precedence_0indexed = [(a - 1, b - 1) for a, b in precedence_relations]

    # Create model
    m = gp.Model("assembly_line")
    m.setParam("Threads", 1)
    m.setParam('TimeLimit', timelimit)

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj_val)

    m.optimize(callback)

    # Decision Variables
    # x[i][k] = 1 if task i assigned to station k
    x = m.addVars(num_tasks, num_stations, vtype=GRB.BINARY, name='x')

    # y[i][j][k] = 1 if task i is immediately before task j at station k
    y = m.addVars(num_tasks, num_tasks, num_stations, vtype=GRB.BINARY, name='y')

    # Continuous vars for stage times
    station_times = m.addVars(num_stations, vtype=GRB.CONTINUOUS, lb=0, name='station_time')

    # Objective: minimize max station time
    z = m.addVar(vtype=GRB.CONTINUOUS, name='max_time', lb=0)
    m.setObjective(z, GRB.MINIMIZE)
    m.addConstrs((z >= station_times[k] for k in range(num_stations)), name='min_max')

    # Constraints
    # 1. Each task assigned to exactly one station
    m.addConstrs((gp.quicksum(x[i, k] for k in range(num_stations)) == 1 
                  for i in range(num_tasks)), name='task_assign')

    # 2. Each station has at least one task
    m.addConstrs((gp.quicksum(x[i, k] for i in range(num_tasks)) >= 1 
                  for k in range(num_stations)), name='station_load')

    # 3. Connect y variables to x variables
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                for k in range(num_stations):
                    # If y[i,j,k] = 1, then both x[i,k] and x[j,k] must be 1
                    m.addConstr(y[i, j, k] <= x[i, k])
                    m.addConstr(y[i, j, k] <= x[j, k])

    # 4. In-degree and out-degree for sequence at station
    for k in range(num_stations):
        for i in range(num_tasks):
            m.addConstr(gp.quicksum(y[j, i, k] for j in range(num_tasks) if j != i) == 
                        gp.quicksum(y[i, j, k] for j in range(num_tasks) if j != i),
                        name=f'flow_balance_{i}_{k}')
            # If x[i,k]=1, then in and out degree must be 1
            m.addConstr(gp.quicksum(y[j, i, k] for j in range(num_tasks) if j != i) <= x[i, k],
                        name=f'degree_in_{i}_{k}')
            m.addConstr(gp.quicksum(y[i, j, k] for j in range(num_tasks) if j != i) <= x[i, k],
                        name=f'degree_out_{i}_{k}')

    # Eliminate subtours using MTZ-style constraints
    # u[i,k] indicates the sequence position for task i at station k
    u = m.addVars(num_tasks, num_stations, vtype=GRB.CONTINUOUS, lb=0, ub=num_tasks-1, name='u')

    # MTZ for subtour elimination
    big_M = num_tasks
    for k in range(num_stations):
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # If task i precedes j in station k, then u[j,k] should be greater than u[i,k]
                    m.addConstr(u[j, k] >= u[i, k] + 1 - big_M * (1 - y[i, j, k]),
                                name=f'mtz_{i}_{j}_{k}')

    # Precedence constraints
    for prec, succ in precedence_0indexed:
        for k in range(num_stations):
            # Ensure succ not on earlier station than prec
            for earlier_k in range(k):  # stations before k
                # If succ is on station k, prec cannot be on earlier_k
                m.addConstr(x[succ, k] + x[prec, earlier_k] <= 1,
                            name=f'precedence_{prec}_{succ}_{k}_{earlier_k}')
            # Same-station order: if both on k, prec must come before succ
            # Using MTZ vars: ensure u[prec,k] < u[succ,k]
            m.addConstr(u[succ, k] >= u[prec, k] + 1 - big_M * (2 - x[prec, k] - x[succ, k]),
                        name=f'same_station_order_{prec}_{succ}_{k}')


    # Set up station times (processing + forward setup + backward setup)
    # Need indicator variables for first and last tasks at each station
    is_first = m.addVars(num_tasks, num_stations, vtype=GRB.BINARY, name='is_first')
    is_last = m.addVars(num_tasks, num_stations, vtype=GRB.BINARY, name='is_last')

    for k in range(num_stations):
        # Exactly one 'first' task per station that is actually in the station
        m.addConstr(gp.quicksum(is_first[i, k] for i in range(num_tasks)) == 
                    gp.quicksum(x[i, k] for i in range(num_tasks)),
                    name=f'one_first_per_station_{k}')

        # Exactly one 'last' task per station that is actually in the station
        m.addConstr(gp.quicksum(is_last[i, k] for i in range(num_tasks)) == 
                    gp.quicksum(x[i, k] for i in range(num_tasks)),
                    name=f'one_last_per_station_{k}')

        for i in range(num_tasks):
            # is_first[i,k] == 1 iff x[i,k] == 1 and no task precedes i at this station
            m.addConstr(is_first[i, k] <= x[i, k], name=f'first_in_station_{i}_{k}')
            m.addConstr(is_first[i, k] >= x[i, k] - gp.quicksum(y[j, i, k] for j in range(num_tasks) if j != i),
                        name=f'def_is_first_{i}_{k}')

            # is_last[i,k] == 1 iff x[i,k] == 1 and no task follows i at this station
            m.addConstr(is_last[i, k] <= x[i, k], name=f'last_in_station_{i}_{k}')
            m.addConstr(is_last[i, k] >= x[i, k] - gp.quicksum(y[i, j, k] for j in range(num_tasks) if j != i),
                        name=f'def_is_last_{i}_{k}')

    # Station time calculation
    for k in range(num_stations):
        expr = gp.LinExpr()

        # Add processing times
        for i in range(num_tasks):
            expr.addTerms(processing_times[i], x[i, k])

        # Add forward setup times
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    expr.addTerms(forward_setup_times[i][j], y[i, j, k])

        # Add backward setup times (from last task back to first task in cycle)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Add backward time only if task i is the last one and task j is the first one in the sequence
                    # This is is_last[i,k] AND is_first[j,k]
                    temp_var = m.addVar(vtype=GRB.BINARY, name=f'BWTEMP_{i}_{j}_{k}')
                    
                    # Linearize multiplication using big-M constraints: temp = is_last[i,k] * is_first[j,k]
                    m.addConstr(temp_var <= is_last[i, k])
                    m.addConstr(temp_var <= is_first[j, k])
                    m.addConstr(temp_var >= is_last[i, k] + is_first[j, k] - 1)
                    
                    expr.addTerms(backward_setup_times[i][j], temp_var)

        m.addConstr(station_times[k] == expr)

    m.optimize(callback)
    
    # Logging after optimization
    if m.SolCount > 0 and logger:
        obj_val = m.ObjVal
        logger.log(obj_val)
    
    # Extract solution if available
    if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        # Build result
        assignment = {}
        for i in range(num_tasks):
            for k in range(num_stations):
                if x[i, k].X > 0.5:
                    assignment[str(i + 1)] = k + 1  # To 1-indexed
                    break

        # Get sequences for each station: traverse the sequence graph defined by y variables
        station_sequences = {}
        for k in range(num_stations):
            seq = []
            # Find the starting task
            start_task = None
            for i in range(num_tasks):
                if is_first[i, k].X > 0.5:
                    start_task = i
                    break
            
            if start_task is not None:
                # Build sequence starting from start_task
                current = start_task
                visited = set()
                while True:
                    seq.append(current + 1)  # to 1-indexed
                    visited.add(current)
                    # Find next task in cycle
                    next_found = False
                    for j in range(num_tasks):
                        if j not in visited and y[current, j, k].X > 0.5:
                            current = j
                            next_found = True
                            break
                    if not next_found:  # End of cycle
                        break
                    if current == start_task:  # We came back to start - break only when we have seen all
                        if len(visited) >= sum(round(x[t, k].X) for t in range(num_tasks)):  # All assigned to this station were visited
                            break
                        
            station_sequences[str(k + 1)] = seq

        # Final cycle time is the maximum station time
        cycle_time = max([station_times[k].X for k in range(num_stations)]) if num_stations > 0 else 0

        result = {
            "objective_value": cycle_time,
            "assignment": assignment,
            "station_sequences": station_sequences,
            "cycle_time": cycle_time
        }
        
        # Write result to file
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        # Return empty solution to meet schema requirements
        result = {
            "objective_value": float('inf'),
            "assignment": {},
            "station_sequences": {str(k+1):[] for k in range(num_stations)},
            "cycle_time": float('inf')
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Assembly Line Optimization")
    parser.add_argument("--instance_path", required=True, help="Path to input instance file")
    parser.add_argument("--solution_path", required=True, help="Path to output solution file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")

    args = parser.parse_args()

    solve_station_balancing(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()