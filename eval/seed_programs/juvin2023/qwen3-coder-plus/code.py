import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_job_shop_scheduling(instance_path, solution_path, time_limit, log_path):
    # Load the instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    preemption_allowed = instance['preemption_allowed']
    migration_allowed = instance['migration_allowed']
    jobs_data = instance['jobs']

    # Extract the data structures
    jobs_operations = []  # List of lists, where each job has a list of operations
    op_to_data_map = {}  # (job_idx, op_idx) -> operation data

    for j_id, job in enumerate(jobs_data):
        ops_list = []
        for o_id, op in enumerate(job['operations']):
            ops_list.append({
                'id': o_id,
                'eligible_machines': op['eligible_machines'],
                'processing_times': {int(m_id): op['processing_times'][str(m_id)] 
                                   for m_id in op['eligible_machines']}
            })
            op_to_data_map[(j_id, o_id)] = {
                'eligible_machines': op['eligible_machines'],
                'processing_times': {int(m_id): op['processing_times'][str(m_id)] 
                                   for m_id in op['eligible_machines']}
            }
        jobs_operations.append(ops_list)
    
    # Calculate upper bound for time horizon
    # We'll calculate worst case processing time for each job and take the max
    max_time = 0
    for job_ops in jobs_operations:
        job_time = 0
        for op_data in job_ops:
            min_p = min(op_data['processing_times'].values()) if op_data['processing_times'] else 0
            job_time += min_p
        max_time = max(max_time, job_time)
    HORIZON = max_time + sum(len(job['operations']) * max(
        [op['processing_times'].get(mid, 999999) for mid in op['eligible_machines']] + [0]
        for job in jobs_data for op in job['operations']
    ))  # Heuristic large enough value

    # Create model
    model = gp.Model("JobShopScheduling")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    model.setParam('IntegralityFocus', 1)
    
    # Precompute variables domains
    valid_ops = []  # List of (job_idx, op_idx) pairs
    for j_id in range(num_jobs):
        for o_id in range(len(jobs_operations[j_id])):
            valid_ops.append((j_id, o_id))
    
    # Decision variables
    # x[j,o,m] : 1 if operation (j,o) is scheduled on machine m
    x = {}
    for (j, o) in valid_ops:
        for m in op_to_data_map[(j,o)]['eligible_machines']:
            x[(j, o, m)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{o}_{m}")
    
    # y[j,o,t] : 1 if operation (j,o) is processed at time t
    y = {}
    for (j, o) in valid_ops:
        for t in range(HORIZON):
            y[(j, o, t)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{o}_{t}")

    # c[j, o] - completion time of operation (j, o)
    c = {}
    for (j, o) in valid_ops:
        c[(j, o)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"c_{j}_{o}")
        
    # Makespan variable
    makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="makespan")

    # Constraints
    
    # 1. Each operation has to be assigned to exactly one machine
    for (j, o) in valid_ops:
        model.addConstr(gp.quicksum(x[(j, o, m)] for m in op_to_data_map[(j, o)]['eligible_machines']) == 1)

    # 2. Processing time requirements
    for (j, o) in valid_ops:
        p_times = op_to_data_map[(j, o)]['processing_times']
        for m in op_to_data_map[(j, o)]['eligible_machines']:
            p_time = op_to_data_map[(j, o)]['processing_times'][m]
            model.addConstr(gp.quicksum(y[(j, o, t)] for t in range(HORIZON)) >= p_time * x[(j, o, m)])
            
            # Define completion time constraints properly
            # If (j,o) is assigned to machine m, its completion time must be at least 
            # the start of its last time unit of processing + processing time
            for t in range(HORIZON):
                model.addConstr(c[(j, o)] >= (t + p_time) * y[(j, o, t)] - (HORIZON * (1 - x[(j, o, m)])))

    # 3. Sequential ordering within jobs (each operation must finish before the next one starts)
    for j in range(num_jobs):
        job_ops_list = jobs_operations[j]
        num_ops = len(job_ops_list)
        for k in range(num_ops - 1):
            prev_op_id = k
            next_op_id = k + 1
            
            # Previous operation must completely finish before next operation starts
            for t in range(HORIZON):
                # If the next operation is active at time t, that time unit corresponds to its first occurrence
                # and all previous operations must have been completed 
                
                # Sum of y terms from beginning of time indicates when the current operation is being processed
                # This will check that previous operation finishes before the next one begins
                # Model: if y[(j, next_op_id, s)] = 1 for some s starting <= t, 
                # then sum of y[(j, prev_op_id, r)] for r < s should equal processing requirement
                
                # Alternative approach: for each time unit t of the next op id, check previous comp time is less than t
                next_op_sum_before_t = gp.LinExpr()
                for s in range(t+1):  # s includes t to make sure we handle ongoing ops at time t
                    next_op_sum_before_t.addTerms(1.0, y[(j, next_op_id, s)])
                    
                # Now ensure prev_op finishes before time period t where next_op could start
                # This requires that prev_op has completed all its processing before the start of time period where next starts
                model.addConstr(next_op_sum_before_t <= 1)  # Next operation can't be processed at multiple times simultaneously in context of this constraint
                
                # The actual sequence: prev completion < next start
                # We track completion time c[(j, prev_op_id)] and ensure no y for next starts until that is reached
                for s in range(t+1):
                    model.addConstr(s >= c[(j, prev_op_id)] - t*(1-y[(j, next_op_id, s)]) - HORIZON*(y[(j, next_op_id, s)]))

    # A proper way to represent job precedence
    for j in range(num_jobs):
        job_ops_list = jobs_operations[j]
        num_ops = len(job_ops_list)
        for k in range(num_ops - 1):
            prev_op_id = k
            next_op_id = k + 1
            model.addConstr(c[(j, prev_op_id)] <= gp.quicksum(t * y[(j, next_op_id, t)] for t in range(HORIZON)) - 
                           (1 - gp.quicksum(x[(j, next_op_id, m)] for m in op_to_data_map[(j, next_op_id)]['eligible_machines'])) * HORIZON)

    # 4. Machine capacity constraints
    for m in range(num_machines):
        for t in range(HORIZON):
            # Only add term if this job-op combination is eligible on this machine
            machine_usage = gp.LinExpr()
            for (j, o) in valid_ops:
                if m in op_to_data_map[(j, o)]['eligible_machines']:
                    machine_usage.addTerms(1.0, y[(j, o, t)] * x[(j, o, m)])
            model.addConstr(machine_usage <= 1, name=f"machine_cap_{m}_{t}")

    # 5. Linking y and x - y should only be 1 if x is 1 for that (j,o) and possible m
    for (j, o) in valid_ops:
        for t in range(HORIZON):
            # For y[j,o,t] to be 1, x[j,o,m] must be 1 for some m eligible with j,o
            model.addConstr(y[(j, o, t)] <= gp.quicksum(x[(j, o, m)] for m in op_to_data_map[(j, o)]['eligible_machines']))

    # 6. Makespan definition
    for (j, o) in valid_ops:
        model.addConstr(makespan >= c[(j, o)]) 

    # Set objective
    model.setObjective(makespan, GRB.MINIMIZE)

    # Set log callback to capture intermediate solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if log_path:
                logger.log(obj)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Optimize
    model.optimize(mycallback)
    
    # Process results
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract solution values
        assignment = {}
        schedule = {}
        
        for (j, o, m) in x:
            if x[(j, o, m)].x > 0.5:  # Binary variable is true
                assignment[f"({j},{o})"] = m
        
        for (j, o, t) in y:
            if y[(j, o, t)].x > 0.5:
                key = f"({j},{o})"
                if key not in schedule:
                    schedule[key] = []
                schedule[key].append(t + 1)  # Convert to 1-indexed time
        
        obj_value = makespan.x
        
        solution = {
            "objective_value": obj_value,
            "assignment": assignment,
            "schedule": schedule
        }
        
        # Write the result
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Return empty solution if infeasible or no solution found
        solution = {
            "objective_value": float('inf'),
            "assignment": {},
            "schedule": {}
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve Job Shop Scheduling Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, default=300, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file')

    args = parser.parse_args()

    solve_job_shop_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)