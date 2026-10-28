import json
import argparse
import time
from solution_logger import SolutionLogger


def greedy_heuristic(num_agents, num_jobs, profits, weights, capacities):
    """Greedy heuristic for GAP based on regret."""
    remaining_cap = list(capacities)
    assignment = {}

    # Compute priority for each job: difference between best and second-best profit
    job_priorities = []
    for j in range(num_jobs):
        job_profits = sorted([(profits[i][j], i) for i in range(num_agents)], reverse=True)
        if len(job_profits) >= 2:
            regret = job_profits[0][0] - job_profits[1][0]
        else:
            regret = job_profits[0][0]
        job_priorities.append((regret, j))

    job_priorities.sort(reverse=True)

    total_profit = 0
    for _, j in job_priorities:
        # Find best feasible agent for job j
        best_agent = -1
        best_profit = -1
        for i in range(num_agents):
            if weights[i][j] <= remaining_cap[i] and profits[i][j] > best_profit:
                best_profit = profits[i][j]
                best_agent = i

        if best_agent == -1:
            # Fallback: assign to agent with most remaining capacity
            best_agent = max(range(num_agents), key=lambda i: remaining_cap[i])

        assignment[j] = best_agent
        remaining_cap[best_agent] -= weights[best_agent][j]
        total_profit += profits[best_agent][j]

    return assignment, total_profit


def local_search(num_agents, num_jobs, profits, weights, capacities, assignment, time_limit, logger=None):
    """Local search improvement via swap and move operations."""
    start_time = time.time()
    best_assignment = dict(assignment)
    best_profit = sum(profits[best_assignment[j]][j] for j in range(num_jobs))

    # Compute current loads
    loads = [0] * num_agents
    for j in range(num_jobs):
        loads[best_assignment[j]] += weights[best_assignment[j]][j]

    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.3:
        improved = False
        for j in range(num_jobs):
            current_agent = best_assignment[j]
            current_profit = profits[current_agent][j]

            # Try moving job j to another agent
            for i in range(num_agents):
                if i == current_agent:
                    continue
                if loads[i] + weights[i][j] <= capacities[i]:
                    new_profit = profits[i][j]
                    if new_profit > current_profit:
                        # Move job j from current_agent to i
                        loads[current_agent] -= weights[current_agent][j]
                        loads[i] += weights[i][j]
                        best_assignment[j] = i
                        best_profit += new_profit - current_profit
                        improved = True
                        break

            if improved:
                break

        # Try swap: exchange jobs between two agents
        if not improved:
            for j1 in range(num_jobs):
                if improved:
                    break
                a1 = best_assignment[j1]
                for j2 in range(j1 + 1, num_jobs):
                    a2 = best_assignment[j2]
                    if a1 == a2:
                        continue
                    # Swap j1 and j2
                    new_load_a1 = loads[a1] - weights[a1][j1] + weights[a1][j2]
                    new_load_a2 = loads[a2] - weights[a2][j2] + weights[a2][j1]
                    if new_load_a1 <= capacities[a1] and new_load_a2 <= capacities[a2]:
                        old_profit = profits[a1][j1] + profits[a2][j2]
                        new_profit = profits[a1][j2] + profits[a2][j1]
                        if new_profit > old_profit:
                            loads[a1] = new_load_a1
                            loads[a2] = new_load_a2
                            best_assignment[j1] = a2
                            best_assignment[j2] = a1
                            best_profit += new_profit - old_profit
                            improved = True
                            break

    return best_assignment, best_profit


def solve_with_gurobi(num_agents, num_jobs, profits, weights, capacities, time_limit, 
                      greedy_assignment=None, logger=None):
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("GAP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for i in range(num_agents):
        for j in range(num_jobs):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Objective: maximize total profit
    model.setObjective(
        gp.quicksum(profits[i][j] * x[i, j] for i in range(num_agents) for j in range(num_jobs)),
        GRB.MAXIMIZE
    )

    # Each job assigned to exactly one agent
    for j in range(num_jobs):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_agents)) == 1,
            name=f"assign_{j}"
        )

    # Capacity constraints
    for i in range(num_agents):
        model.addConstr(
            gp.quicksum(weights[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i],
            name=f"cap_{i}"
        )

    # Set MIP start from greedy solution
    if greedy_assignment is not None:
        for j in range(num_jobs):
            for i in range(num_agents):
                x[i, j].Start = 1.0 if greedy_assignment[j] == i else 0.0

    # Callback to log intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                assignment = {}
                for j in range(num_jobs):
                    for i in range(num_agents):
                        if model.cbGetSolution(x[i, j]) > 0.5:
                            assignment[j] = i
                            break
                solution_dict = {
                    "objective_value": float(obj_val),
                    "assignment": {str(j): assignment[j] for j in range(num_jobs)}
                }
                logger.log_solution(obj_val, solution_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        assignment = {}
        total_profit = 0
        for j in range(num_jobs):
            for i in range(num_agents):
                if x[i, j].X > 0.5:
                    assignment[j] = i
                    total_profit += profits[i][j]
                    break
        return assignment, total_profit

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    start_time = time.time()

    # Phase 1: Greedy heuristic for initial solution
    assignment, total_profit = greedy_heuristic(num_agents, num_jobs, profits, weights, capacities)

    if logger:
        solution_dict = {
            "objective_value": float(total_profit),
            "assignment": {str(j): assignment[j] for j in range(num_jobs)}
        }
        logger.log_solution(total_profit, solution_dict)

    # Phase 2: Local search improvement
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 2:
        assignment, total_profit = local_search(
            num_agents, num_jobs, profits, weights, capacities, 
            assignment, remaining, logger
        )
        if logger:
            solution_dict = {
                "objective_value": float(total_profit),
                "assignment": {str(j): assignment[j] for j in range(num_jobs)}
            }
            logger.log_solution(total_profit, solution_dict)

    # Phase 3: Gurobi MIP
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 1:
        gurobi_assignment, gurobi_profit = solve_with_gurobi(
            num_agents, num_jobs, profits, weights, capacities,
            remaining, assignment, logger
        )
        if gurobi_assignment is not None and gurobi_profit > total_profit:
            assignment = gurobi_assignment
            total_profit = gurobi_profit

    # Write final solution
    solution = {
        "objective_value": float(total_profit),
        "assignment": {str(j): assignment[j] for j in range(num_jobs)}
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()