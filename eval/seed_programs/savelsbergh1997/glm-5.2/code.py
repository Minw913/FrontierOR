import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB


def greedy_solution(num_agents, num_jobs, profits, weights, capacities):
    """Construct a feasible solution using a greedy heuristic.
    Returns a list assignment[j] = agent, or None if infeasible."""
    assignment = [-1] * num_jobs
    remaining_cap = list(capacities)

    # Order jobs by descending maximum profit (most valuable jobs first)
    job_order = sorted(
        range(num_jobs),
        key=lambda j: max(profits[a][j] for a in range(num_agents)),
        reverse=True,
    )

    for j in job_order:
        # Try agents in descending order of profit for this job
        agents_sorted = sorted(
            range(num_agents),
            key=lambda a: profits[a][j],
            reverse=True,
        )
        assigned = False
        for a in agents_sorted:
            if remaining_cap[a] >= weights[a][j]:
                assignment[j] = a
                remaining_cap[a] -= weights[a][j]
                assigned = True
                break
        if not assigned:
            # Could not place this job feasibly
            return None
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    start_time = time.time()

    # ------------------------------------------------------------------
    # Greedy warm start
    # ------------------------------------------------------------------
    best_obj = 0.0
    best_assignment = {}

    greedy_assign = greedy_solution(num_agents, num_jobs, profits, weights, capacities)
    if greedy_assign is not None:
        best_obj = float(sum(profits[greedy_assign[j]][j] for j in range(num_jobs)))
        best_assignment = {str(j): greedy_assign[j] for j in range(num_jobs)}
        if logger:
            logger.log_solution(best_obj, best_assignment)

    # ------------------------------------------------------------------
    # Gurobi model
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)

    # x[a, j] = 1 if agent a performs job j
    x = model.addVars(num_agents, num_jobs, vtype=GRB.BINARY, name='x')

    # Objective: maximize total profit
    model.setObjective(
        gp.quicksum(
            profits[a][j] * x[a, j]
            for a in range(num_agents)
            for j in range(num_jobs)
        ),
        GRB.MAXIMIZE,
    )

    # Each job assigned exactly once
    model.addConstrs(
        (x.sum('*', j) == 1 for j in range(num_jobs)),
        name='job_assignment',
    )

    # Agent capacity constraints
    model.addConstrs(
        (
            gp.quicksum(weights[a][j] * x[a, j] for j in range(num_jobs)) <= capacities[a]
            for a in range(num_agents)
        ),
        name='capacity',
    )

    # Provide MIP start
    if greedy_assign is not None:
        for a in range(num_agents):
            for j in range(num_jobs):
                x[a, j].Start = 1.0 if greedy_assign[j] == a else 0.0

    # Set Gurobi parameters as required
    remaining_time = max(1.0, args.time_limit - (time.time() - start_time))
    model.setParam('TimeLimit', remaining_time)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    # Store a flat variable list for the callback
    var_list = [x[a, j] for a in range(num_agents) for j in range(num_jobs)]
    model._var_list = var_list
    model._logger = logger

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = model.cbGetSolution(model._var_list)
            assignment = {}
            for j in range(num_jobs):
                for a in range(num_agents):
                    if sol[a * num_jobs + j] > 0.5:
                        assignment[str(j)] = a
                        break
            if model._logger:
                model._logger.log_solution(obj, assignment)

    model.optimize(callback)

    # ------------------------------------------------------------------
    # Retrieve best solution found
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        obj_val = model.ObjVal
        assignment = {}
        for j in range(num_jobs):
            for a in range(num_agents):
                if x[a, j].X > 0.5:
                    assignment[str(j)] = a
                    break
        if obj_val > best_obj:
            best_obj = float(obj_val)
            best_assignment = assignment

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output = {
        "objective_value": best_obj,
        "assignment": best_assignment,
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()