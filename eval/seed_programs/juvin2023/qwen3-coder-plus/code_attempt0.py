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
        c[(j, o)] = model.addVar(vtype=GRB.INTEGER, name=f"c_{j}_{o}")
        
    # Makespan variable
    makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")

    # Constraints
    
    # 1. Each operation has to be assigned to exactly one machine
    for (j, o) in valid_ops:
        model.addConstr(gp.quicksum(x[(j, o, m)] for m in op_to_data_map[(j, o)]['eligible_machines']) == 1)

    # 2. Processing time requirements
    for (j, o) in valid_ops:
        for m in op_to_data_map[(j, o)]['eligible_machines']:
            p_time = op_to_data_map[(j, o)]['processing_times'][m]
            model.addConstr(gp.quicksum(y[(j, o, t)] for t in range(HORIZON)) >= p_time*x[(j, o, m)])
            model.addConstr(c[(j, o)] >= gp.quicksum(t*y[(j, o, t-tmp+1)] for t in range(tmp, HORIZON) for tmp in range(p_time)) + 
                            p_time*x[(j, o, m)], name=f"comp_time_{j}_{o}_on_{m}")

    # 3. Sequential ordering within jobs
    for j in range(num_jobs):
        job_ops = jobs_operations[j]
        for k in range(len(job_ops)-1):
            prev_op = k
            next_op = k+1
            for m in op_to_data_map[(j, prev_op)]['eligible_machines']:
                # Completion time of previous operation is when its last processing interval ends
                model.addConstr(c[(j, prev_op)] <= gp.quicksum(t*y[(j, prev_op, t)] for t in range(HORIZON)) + 
                                1 - x[(j, prev_op, m)], name=f"seq_{j}_{prev_op}_{next_op}_{m}")
            
            # Make sure next operation starts only after the previous one completes
            for t in range(HORIZON):
                model.addConstr((gp.quicksum(y[(j, prev_op, t_prime) for t_prime in range(min(HORIZON, t+1))])) >= 
                                y[(j, next_op, t)] * gp.quicksum(x[(j, prev_op, m)] for m in op_to_data_map[(j, prev_op)]['eligible_machines']),
                                name=f"seq_order_{j}_{prev_op}_{next_op}_{t}")

    # 4. Machine capacity constraints
    for m in range(num_machines):
        for t in range(HORIZON):
            machine_usage_expr = gp.LinExpr()
            for (j, o) in valid_ops:
                if m in [mid for mid in op_to_data_map[(j, o)]['eligible_machines']]:
                    machine_usage_expr.addTerms(1, y[(j, o, t)] * x[(j, o, m)])
            model.addConstr(machine_usage_expr <= 1, name=f"machine_cap_{m}_{t}")

    # 5. Linking variables: an operation processes at time t only if assigned to a machine
    for (j, o) in valid_ops:
        for t in range(HORIZON):
            model.addConstr(y[(j, o, t)] <= gp.quicksum(x[(j, o, m)] for m in op_to_data_map[(j, o)]['eligible_machines']))

    # 6. Definition of makespan
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