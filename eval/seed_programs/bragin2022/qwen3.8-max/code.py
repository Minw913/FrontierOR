import argparse
import json
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    cost_matrix = instance['cost_matrix']
    resource_matrix = instance['resource_matrix']
    capacities = instance['capacities']

    start_time = time.time()

    best_solution = None
    best_objective = float('inf')

    # --- Greedy heuristic for initial feasible solution ---
    def greedy_heuristic():
        remaining_cap = capacities[:]
        assignment = {}
        total_cost = 0

        # Compute regret for each job: difference between 2nd best and best cost
        job_info = []
        for j in range(num_jobs):
            costs_sorted = sorted(range(num_machines), key=lambda m: cost_matrix[m][j])
            best_cost = cost_matrix[costs_sorted[0]][j]
            second_cost = cost_matrix[costs_sorted[1]][j] if num_machines > 1 else best_cost
            regret = second_cost - best_cost
            job_info.append((regret, j))

        # Assign jobs in decreasing order of regret
        job_info.sort(key=lambda x: -x[0])

        for _, j in job_info:
            best_m = None
            best_c = float('inf')
            for m in range(num_machines):
                if remaining_cap[m] >= resource_matrix[m][j] and cost_matrix[m][j] < best_c:
                    best_c = cost_matrix[m][j]
                    best_m = m
            if best_m is not None:
                assignment[str(j)] = best_m
                remaining_cap[best_m] -= resource_matrix[best_m][j]
                total_cost += best_c
            else:
                return None, float('inf')
        return assignment, total_cost

    # Try multiple greedy strategies
    # Strategy 1: Regret-based
    sol1, obj1 = greedy_heuristic()
    if sol1 is not None and obj1 < best_objective:
        best_objective = obj1
        best_solution = sol1
        if logger:
            logger.log_solution(best_objective, {"objective_value": best_objective, "assignments": best_solution})

    # Strategy 2: Assign by lowest cost first (job order by min cost ascending)
    def greedy_by_cost():
        remaining_cap = capacities[:]
        assignment = {}
        total_cost = 0
        job_order = sorted(range(num_jobs), key=lambda j: min(cost_matrix[m][j] for m in range(num_machines)))
        for j in job_order:
            best_m = None
            best_c = float('inf')
            for m in range(num_machines):
                if remaining_cap[m] >= resource_matrix[m][j] and cost_matrix[m][j] < best_c:
                    best_c = cost_matrix[m][j]
                    best_m = m
            if best_m is not None:
                assignment[str(j)] = best_m
                remaining_cap[best_m] -= resource_matrix[best_m][j]
                total_cost += best_c
            else:
                return None, float('inf')
        return assignment, total_cost

    sol2, obj2 = greedy_by_cost()
    if sol2 is not None and obj2 < best_objective:
        best_objective = obj2
        best_solution = sol2
        if logger:
            logger.log_solution(best_objective, {"objective_value": best_objective, "assignments": best_solution})

    # Strategy 3: Assign by resource efficiency (cost/resource ratio)
    def greedy_by_ratio():
        remaining_cap = capacities[:]
        assignment = {}
        total_cost = 0
        # Sort jobs by max resource consumption (descending) - hardest to place first
        job_order = sorted(range(num_jobs), key=lambda j: max(resource_matrix[m][j] for m in range(num_machines)), reverse=True)
        for j in job_order:
            best_m = None
            best_c = float('inf')
            for m in range(num_machines):
                if remaining_cap[m] >= resource_matrix[m][j] and cost_matrix[m][j] < best_c:
                    best_c = cost_matrix[m][j]
                    best_m = m
            if best_m is not None:
                assignment[str(j)] = best_m
                remaining_cap[best_m] -= resource_matrix[best_m][j]
                total_cost += best_c
            else:
                return None, float('inf')
        return assignment, total_cost

    sol3, obj3 = greedy_by_ratio()
    if sol3 is not None and obj3 < best_objective:
        best_objective = obj3
        best_solution = sol3
        if logger:
            logger.log_solution(best_objective, {"objective_value": best_objective, "assignments": best_solution})

    # --- Solve with Gurobi ---
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    if remaining_time > 0.5:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("GAP")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(remaining_time - 1.0, 1.0))
            model.setParam('OutputFlag', 0)

            # Decision variables
            x = {}
            for m in range(num_machines):
                for j in range(num_jobs):
                    x[m, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{m}_{j}")

            model.update()

            # Objective: minimize total cost
            model.setObjective(
                gp.quicksum(cost_matrix[m][j] * x[m, j] for m in range(num_machines) for j in range(num_jobs)),
                GRB.MINIMIZE
            )

            # Assignment constraints: each job to exactly one machine
            for j in range(num_jobs):
                model.addConstr(
                    gp.quicksum(x[m, j] for m in range(num_machines)) == 1,
                    name=f"assign_{j}"
                )

            # Capacity constraints
            for m in range(num_machines):
                model.addConstr(
                    gp.quicksum(resource_matrix[m][j] * x[m, j] for j in range(num_jobs)) <= capacities[m],
                    name=f"cap_{m}"
                )

            # Warm start with best heuristic solution
            if best_solution is not None:
                for j in range(num_jobs):
                    m_assigned = best_solution[str(j)]
                    for m in range(num_machines):
                        x[m, j].Start = 1.0 if m == m_assigned else 0.0

            # Callback to log intermediate solutions
            def callback(model, where):
                nonlocal best_objective, best_solution
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < best_objective:
                        assignments = {}
                        for j in range(num_jobs):
                            for m in range(num_machines):
                                if model.cbGetSolution(x[m, j]) > 0.5:
                                    assignments[str(j)] = m
                                    break
                        best_objective = obj_val
                        best_solution = assignments
                        if logger:
                            logger.log_solution(best_objective, {"objective_value": best_objective, "assignments": best_solution})

            model.optimize(callback)

            # Extract final solution
            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val < best_objective:
                    assignments = {}
                    for j in range(num_jobs):
                        for m in range(num_machines):
                            if x[m, j].X > 0.5:
                                assignments[str(j)] = m
                                break
                    best_objective = obj_val
                    best_solution = assignments
                    if logger:
                        logger.log_solution(best_objective, {"objective_value": best_objective, "assignments": best_solution})

        except Exception as e:
            # If Gurobi fails, rely on heuristic solution
            pass

    # Write final solution
    if best_solution is not None:
        solution = {
            "objective_value": float(best_objective),
            "assignments": best_solution
        }
    else:
        # Fallback: assign all jobs to machine 0 (may be infeasible but must output something)
        solution = {
            "objective_value": float(sum(cost_matrix[0][j] for j in range(num_jobs))),
            "assignments": {str(j): 0 for j in range(num_jobs)}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()