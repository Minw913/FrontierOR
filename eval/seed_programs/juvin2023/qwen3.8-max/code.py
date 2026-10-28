import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def greedy_heuristic(jobs, num_machines):
    """Greedy list scheduling heuristic for fallback/warm start."""
    machine_free = [1] * num_machines
    assignment = {}
    schedule = {}

    for job in jobs:
        jid = job['job_id']
        prev_end = 1
        for op in job['operations']:
            oid = op['operation_id']
            eligible = op['eligible_machines']
            ptimes = {int(k): int(v) for k, v in op['processing_times'].items()}

            best_m = None
            best_end = float('inf')
            for m in eligible:
                start = max(machine_free[m], prev_end)
                end = start + ptimes[m]
                if end < best_end:
                    best_end = end
                    best_m = m

            start = max(machine_free[best_m], prev_end)
            end = start + ptimes[best_m]
            machine_free[best_m] = end
            prev_end = end

            assignment[f"({jid},{oid})"] = best_m
            schedule[f"({jid},{oid})"] = list(range(start, end))

    makespan = max(machine_free) if machine_free else 0
    return makespan, assignment, schedule


def main():
    parser = argparse.ArgumentParser(description="FJSP with Preemption Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']

    # Parse operations
    all_ops = []
    for job in jobs:
        jid = job['job_id']
        for op in job['operations']:
            oid = op['operation_id']
            eligible = op['eligible_machines']
            ptimes = {int(k): int(v) for k, v in op['processing_times'].items()}
            all_ops.append((jid, oid, eligible, ptimes))

    # Compute time horizon
    T = 0
    for (jid, oid, eligible, ptimes) in all_ops:
        if ptimes:
            T += max(ptimes.values())

    # Edge case: no processing needed
    if T == 0 or len(all_ops) == 0:
        solution = {
            "objective_value": 0.0,
            "assignment": {f"({jid},{oid})": eligible[0] for (jid, oid, eligible, ptimes) in all_ops},
            "schedule": {f"({jid},{oid})": [] for (jid, oid, eligible, ptimes) in all_ops}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0.0, solution)
        return

    # Get heuristic solution
    heur_makespan, heur_assignment, heur_schedule = greedy_heuristic(jobs, num_machines)

    if logger:
        heur_sol = {
            "objective_value": float(heur_makespan),
            "assignment": heur_assignment,
            "schedule": heur_schedule
        }
        logger.log_solution(float(heur_makespan), heur_sol)

    # Try MIP approach
    try:
        model = gp.Model("FJSP_Preemptive")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = args.time_limit
        model.Params.OutputFlag = 0

        # Decision variables
        x = {}
        z = {}
        u = {}

        for (jid, oid, eligible, ptimes) in all_ops:
            for m in eligible:
                x[jid, oid, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{jid}_{oid}_{m}")
            for m in eligible:
                for t in range(1, T + 1):
                    z[jid, oid, m, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{jid}_{oid}_{m}_{t}")
            for t in range(1, T + 2):
                u[jid, oid, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{jid}_{oid}_{t}")

        C = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="makespan")
        model.update()

        # Objective: minimize makespan
        model.setObjective(C, GRB.MINIMIZE)

        # Constraint 1: Each operation assigned to exactly one eligible machine
        for (jid, oid, eligible, ptimes) in all_ops:
            model.addConstr(
                gp.quicksum(x[jid, oid, m] for m in eligible) == 1,
                name=f"assign_{jid}_{oid}"
            )

        # Constraint 2: Can only process on assigned machine
        for (jid, oid, eligible, ptimes) in all_ops:
            for m in eligible:
                for t in range(1, T + 1):
                    model.addConstr(z[jid, oid, m, t] <= x[jid, oid, m])

        # Constraint 3: Total processing time requirement
        for (jid, oid, eligible, ptimes) in all_ops:
            model.addConstr(
                gp.quicksum(z[jid, oid, m, t] for m in eligible for t in range(1, T + 1)) >=
                gp.quicksum(ptimes[m] * x[jid, oid, m] for m in eligible),
                name=f"proc_{jid}_{oid}"
            )

        # Constraint 3b: Upper bound per machine (strengthens LP relaxation)
        for (jid, oid, eligible, ptimes) in all_ops:
            for m in eligible:
                model.addConstr(
                    gp.quicksum(z[jid, oid, m, t] for t in range(1, T + 1)) <=
                    ptimes[m] * x[jid, oid, m],
                    name=f"proc_ub_{jid}_{oid}_{m}"
                )

        # Constraint 4: Machine capacity (at most one operation per machine per period)
        ops_by_machine = {m: [] for m in range(num_machines)}
        for (jid, oid, eligible, ptimes) in all_ops:
            for m in eligible:
                ops_by_machine[m].append((jid, oid))

        for m in range(num_machines):
            if ops_by_machine[m]:
                for t in range(1, T + 1):
                    model.addConstr(
                        gp.quicksum(z[jid, oid, m, t] for (jid, oid) in ops_by_machine[m]) <= 1,
                        name=f"cap_{m}_{t}"
                    )

        # Constraint 5: Can't process if operation already completed
        for (jid, oid, eligible, ptimes) in all_ops:
            for m in eligible:
                for t in range(1, T + 1):
                    model.addConstr(z[jid, oid, m, t] + u[jid, oid, t] <= 1)

        # Constraint 6: Monotonicity of completion indicator
        for (jid, oid, eligible, ptimes) in all_ops:
            for t in range(1, T + 1):
                model.addConstr(u[jid, oid, t] <= u[jid, oid, t + 1])

        # Constraint 7: All operations must complete by end of horizon
        for (jid, oid, eligible, ptimes) in all_ops:
            model.addConstr(u[jid, oid, T + 1] == 1)

        # Constraint 8: Precedence constraints
        for job in jobs:
            jid = job['job_id']
            ops_list = job['operations']
            for i in range(len(ops_list) - 1):
                oid = ops_list[i]['operation_id']
                next_oid = ops_list[i + 1]['operation_id']
                next_eligible = ops_list[i + 1]['eligible_machines']
                for t in range(1, T + 1):
                    model.addConstr(
                        u[jid, oid, t] >= gp.quicksum(
                            z[jid, next_oid, m, t] for m in next_eligible
                        ),
                        name=f"prec_{jid}_{oid}_{next_oid}_{t}"
                    )

        # Constraint 9: Makespan definition
        for job in jobs:
            jid = job['job_id']
            last_op = job['operations'][-1]
            last_oid = last_op['operation_id']
            last_eligible = last_op['eligible_machines']
            for t in range(1, T + 1):
                model.addConstr(
                    C >= (t + 1) * gp.quicksum(
                        z[jid, last_oid, m, t] for m in last_eligible
                    ),
                    name=f"mkspn_{jid}_{t}"
                )

        model.update()

        # Set heuristic solution as warm start
        for (jid, oid, eligible, ptimes) in all_ops:
            key = f"({jid},{oid})"
            hm = heur_assignment.get(key)
            if hm is not None and hm in eligible:
                x[jid, oid, hm].Start = 1.0
                for m in eligible:
                    if m != hm:
                        x[jid, oid, m].Start = 0.0
                periods_set = set(heur_schedule.get(key, []))
                for m in eligible:
                    for t in range(1, T + 1):
                        if m == hm and t in periods_set:
                            z[jid, oid, m, t].Start = 1.0
                        else:
                            z[jid, oid, m, t].Start = 0.0
                if periods_set:
                    last_period = max(periods_set)
                    for t in range(1, T + 2):
                        u[jid, oid, t].Start = 0.0 if t <= last_period else 1.0
                else:
                    for t in range(1, T + 2):
                        u[jid, oid, t].Start = 1.0

        C.Start = float(heur_makespan)

        # Callback for logging incumbent solutions
        best_logged = [float('inf')]

        def my_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_logged[0] - 1e-9:
                    best_logged[0] = obj
                    sol_assignment = {}
                    sol_schedule = {}
                    for (jid, oid, eligible, ptimes) in all_ops:
                        am = None
                        for m in eligible:
                            if model.cbGetSolution(x[jid, oid, m]) > 0.5:
                                am = m
                                break
                        if am is None:
                            am = eligible[0]
                        sol_assignment[f"({jid},{oid})"] = am
                        periods = []
                        for t in range(1, T + 1):
                            if model.cbGetSolution(z[jid, oid, am, t]) > 0.5:
                                periods.append(t)
                        sol_schedule[f"({jid},{oid})"] = periods

                    solution = {
                        "objective_value": float(obj),
                        "assignment": sol_assignment,
                        "schedule": sol_schedule
                    }
                    if logger:
                        logger.log_solution(float(obj), solution)

        # Solve
        model.optimize(my_callback)

        # Extract best solution
        if model.SolCount > 0:
            obj_val = model.ObjVal
            assignment = {}
            schedule = {}
            for (jid, oid, eligible, ptimes) in all_ops:
                am = None
                for m in eligible:
                    if x[jid, oid, m].X > 0.5:
                        am = m
                        break
                if am is None:
                    am = eligible[0]
                assignment[f"({jid},{oid})"] = am
                periods = []
                for t in range(1, T + 1):
                    if z[jid, oid, am, t].X > 0.5:
                        periods.append(t)
                schedule[f"({jid},{oid})"] = periods

            solution = {
                "objective_value": float(obj_val),
                "assignment": assignment,
                "schedule": schedule
            }

            if logger and obj_val < best_logged[0] - 1e-9:
                logger.log_solution(float(obj_val), solution)
        else:
            # Fall back to heuristic
            solution = {
                "objective_value": float(heur_makespan),
                "assignment": heur_assignment,
                "schedule": heur_schedule
            }

    except Exception as e:
        # If MIP fails, use heuristic solution
        solution = {
            "objective_value": float(heur_makespan),
            "assignment": heur_assignment,
            "schedule": heur_schedule
        }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()