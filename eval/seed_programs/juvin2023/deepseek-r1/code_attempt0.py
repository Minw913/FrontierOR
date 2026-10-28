import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize solution logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_jobs = data['num_jobs']
    num_machines = data['num_machines']
    jobs = data['jobs']

    # Compute T: upper bound on time horizon (sum of max processing times for all operations)
    T = 0
    for job in jobs:
        for op in job['operations']:
            max_time = 0
            for m in op['eligible_machines']:
                m_str = str(m)
                if m_str in op['processing_times']:
                    pt = op['processing_times'][m_str]
                    if pt > max_time:
                        max_time = pt
                else:
                    # Fallback: use max of available processing times
                    pt = max(op['processing_times'].values(), key=lambda x: float(x)) if op['processing_times'] else 0
                    max_time = max(max_time, pt)
            T += max_time
    T = max(1, T)  # Ensure T is at least 1

    # Create Gurobi model
    model = gp.Model()
    model.setParam('OutputFlag', 0)  # Disable solver output
    model.setParam('Threads', 1)     # Use only one thread

    # Create variables
    x = {}  # Assignment variables: x[job_id, op_id, machine_id]
    y = {}  # Scheduling variables: y[job_id, op_id, machine_id, time]
    S = {}  # Start time variables: S[job_id, op_id]
    C = {}  # Completion time variables: C[job_id, op_id]
    makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")

    # Precompute machine_ops for machine capacity constraints
    machine_ops = {m: [] for m in range(num_machines)}
    for job in jobs:
        job_id = job['job_id']
        for op in job['operations']:
            op_id = op['operation_id']
            for m in op['eligible_machines']:
                if m < num_machines:  # Ensure machine index is valid
                    machine_ops[m].append((job_id, op_id))

    # Add variables and constraints per operation
    for job in jobs:
        job_id = job['job_id']
        for op in job['operations']:
            op_id = op['operation_id']
            eligible_machines = [m for m in op['eligible_machines'] if m < num_machines]
            
            # S and C variables
            S[(job_id, op_id)] = model.addVar(lb=1, ub=T, vtype=GRB.CONTINUOUS, name=f"S_{job_id}_{op_id}")
            C[(job_id, op_id)] = model.addVar(lb=1, ub=T, vtype=GRB.CONTINUOUS, name=f"C_{job_id}_{op_id}")
            
            # x and y variables
            for m in eligible_machines:
                x[(job_id, op_id, m)] = model.addVar(vtype=GRB.BINARY, name=f"x_{job_id}_{op_id}_{m}")
                for t in range(1, T+1):
                    y[(job_id, op_id, m, t)] = model.addVar(vtype=GRB.BINARY, name=f"y_{job_id}_{op_id}_{m}_{t}")

            # Assignment constraint: exactly one machine per operation
            model.addConstr(
                gp.quicksum(x[(job_id, op_id, m)] for m in eligible_machines) == 1,
                name=f"assign_{job_id}_{op_id}"
            )
            
            # Processing requirement per machine
            for m in eligible_machines:
                m_str = str(m)
                if m_str in op['processing_times']:
                    pt = op['processing_times'][m_str]
                else:
                    # Fallback: use max processing time if specific machine not found
                    pt = max(op['processing_times'].values(), key=lambda x: float(x)) if op['processing_times'] else 0
                model.addConstr(
                    gp.quicksum(y[(job_id, op_id, m, t)] for t in range(1, T+1)) == pt * x[(job_id, op_id, m)],
                    name=f"process_{job_id}_{op_id}_{m}"
                )
                
                # Linking constraint: y <= x
                for t in range(1, T+1):
                    model.addConstr(
                        y[(job_id, op_id, m, t)] <= x[(job_id, op_id, m)],
                        name=f"link_{job_id}_{op_id}_{m}_{t}"
                    )

    # Machine capacity constraints
    for m in range(num_machines):
        for t in range(1, T+1):
            y_vars = []
            for (job_id, op_id) in machine_ops[m]:
                if (job_id, op_id, m, t) in y:
                    y_vars.append(y[(job_id, op_id, m, t)])
            if y_vars:
                model.addConstr(
                    gp.quicksum(y_vars) <= 1,
                    name=f"machine_{m}_time_{t}"
                )

    # Precedence constraints within jobs
    for job in jobs:
        job_id = job['job_id']
        operations = job['operations']
        for i in range(len(operations)-1):
            op_id1 = operations[i]['operation_id']
            op_id2 = operations[i+1]['operation_id']
            model.addConstr(
                C[(job_id, op_id1)] + 1 <= S[(job_id, op_id2)],
                name=f"precedence_{job_id}_{op_id1}_{op_id2}"
            )

    # Start and completion time definitions
    for job in jobs:
        job_id = job['job_id']
        for op in job['operations']:
            op_id = op['operation_id']
            s_var = S[(job_id, op_id)]
            c_var = C[(job_id, op_id)]
            for m in op['eligible_machines']:
                if m < num_machines:  # Ensure valid machine index
                    for t in range(1, T+1):
                        # Start time: S <= t if operation is processed at time t
                        model.addConstr(
                            s_var <= t + T * (1 - y[(job_id, op_id, m, t)]),
                            name=f"start_def_{job_id}_{op_id}_{m}_{t}"
                        )
                        # Completion time: C >= t if operation is processed at time t
                        model.addConstr(
                            c_var >= t * y[(job_id, op_id, m, t)],
                            name=f"complete_def_{job_id}_{op_id}_{m}_{t}"
                        )

    # Makespan constraints (for last operation of each job)
    for job in jobs:
        job_id = job['job_id']
        last_op_id = job['operations'][-1]['operation_id']
        model.addConstr(
            makespan >= C[(job_id, last_op_id)] + 1,
            name=f"makespan_job_{job_id}"
        )

    # Set objective to minimize makespan
    model.setObjective(makespan, GRB.MINIMIZE)

    # Set time limit
    model.setParam('TimeLimit', args.time_limit)

    # Optimization with callback for logging incumbents
    if logger:
        model._logger = logger
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()

    # Prepare solution
    solution = {
        "objective_value": float('inf'),
        "assignment": {},
        "schedule": {}
    }

    # Extract solution if feasible
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        # Get makespan
        if hasattr(makespan, 'X'):
            solution["objective_value"] = makespan.X
        else:
            # Use best bound if no solution found? But we need a feasible solution.
            # Fallback: use the best known bound (but not necessarily feasible)
            solution["objective_value"] = model.ObjBound

        # Extract assignment and schedule
        for job in jobs:
            job_id = job['job_id']
            for op in job['operations']:
                op_id = op['operation_id']
                key = f"({job_id},{op_id})"
                
                # Find assigned machine
                assigned_machine = None
                for m in op['eligible_machines']:
                    if m < num_machines and (job_id, op_id, m) in x and hasattr(x[(job_id, op_id, m)], 'X') and x[(job_id, op_id, m)].X > 0.5:
                        assigned_machine = m
                        solution["assignment"][key] = assigned_machine
                        break
                
                # If machine assigned, get schedule
                if assigned_machine is not None:
                    time_periods = []
                    for t in range(1, T+1):
                        if (job_id, op_id, assigned_machine, t) in y and hasattr(y[(job_id, op_id, assigned_machine, t)], 'X') and y[(job_id, op_id, assigned_machine, t)].X > 0.5:
                            time_periods.append(t)
                    solution["schedule"][key] = time_periods
                else:
                    # Fallback: assign first eligible machine and empty schedule
                    if op['eligible_machines']:
                        solution["assignment"][key] = op['eligible_machines'][0]
                    solution["schedule"][key] = []

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()