import json
import argparse
import time
import sys
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['num_scenarios_N']
    n = instance['problem_dimension_n']
    K_values = instance['K_values']
    scenarios = np.array(instance['scenarios'], dtype=float)
    first_stage_costs = np.array(instance.get('first_stage_costs', [0.0]*n), dtype=float)

    # Precompute for efficiency
    max_per_dim = np.max(scenarios, axis=0)
    min_per_dim = np.min(scenarios, axis=0)

    solution = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }

    best_t_global = 0.0
    best_solution_dict = None

    def log_if_better(t_val, sol_dict):
        nonlocal best_t_global, best_solution_dict
        if t_val > best_t_global + 1e-8:
            best_t_global = t_val
            best_solution_dict = sol_dict.copy()
            solution["objective_value"] = float(best_t_global)
            if logger:
                logger.log_solution(float(t_val), sol_dict)

    remaining_time = lambda: max(1, args.time_limit - (time.time() - start_time))

    # Solve for each K in order, starting from smallest for faster warm-up
    for K in sorted(K_values):
        if time.time() - start_time > args.time_limit * 0.95:
            break

        # ------------------- One-stage IP-mu (with convex combinations) -------------------
        m = Model("one_stage_IP_mu")
        m.setParam('Seed', 0)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NumericFocus', 0)
        m.setParam('Threads', 1)
        m.setParam('TimeLimit', remaining_time())

        # Binary assignment: mu[i][k] = 1 if scenario i assigned to aggregate k
        mu = {}
        for i in range(N):
            for k in range(K):
                mu[i, k] = m.addVar(vtype=GRB.BINARY, name=f"mu_{i}_{k}")

        # lambda weights: lambda[k][j] weight of scenario j in aggregate k
        lam = {}
        for k in range(K):
            for j in range(N):
                lam[k, j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"lam_{k}_{j}")

        t = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="t")

        # Constraints
        # Each scenario assigned to exactly one aggregate
        for i in range(N):
            m.addConstr(quicksum(mu[i, k] for k in range(K)) == 1)

        # Lambda only used if aggregate is "active" - but since K is fixed, we allow all
        # Convex combination: sum lambda = 1 for each aggregate
        for k in range(K):
            m.addConstr(quicksum(lam[k, j] for j in range(N)) == 1)

        # Linking: lambda can only be positive for scenarios assigned to that aggregate (big-M style via t)
        # But more importantly, the scaling constraint:
        for i in range(N):
            for d in range(n):
                # For every original scenario i and dimension d:
                # t * (sum_j lambda[assigned_k][j] * scenarios[j][d])  <= scenarios[i][d]
                # Since assignment is variable, we linearize using mu
                lhs = t * quicksum(lam[k, j] * scenarios[j, d] for j in range(N) for k in range(K) if True)
                # Correct linearization:
                # sum_k mu[i,k] * (sum_j lam[k,j] * s[j,d]) >= t * s[i,d]   wait no:
                # We want: for the assigned aggregate's convex combo c,  c_d >= t * s[i,d] for all d? No.

                # From problem: "the ratio between its cost and the cost of the best-representing scenario from the selected set"
                # "scaling-factor guarantee: for any original scenario, the ratio between its cost (in any dimension)
                # and the cost of the best-representing scenario from the selected set is at least the scaling factor"

                # This means: for each original i, there exists a representative r (from reduced set) such that
                # for all dimensions d: s[i,d] <= (1/t) * r_d    i.e. r_d >= t * s[i,d]
                # So representative dominates i after scaling by t.

                # Since representative is a convex combination of original scenarios (in one-stage_IP_mu),
                # we have:
                # sum_j lambda[k,j] * scenarios[j,d]  >= t * scenarios[i,d]    for the k that i is assigned to.

                # Linearized:
                # For each i, k, d:   sum_j lambda[k,j] * s[j,d]  >= t * s[i,d] - M*(1-mu[i,k])

        M = 2.0 * max_per_dim.max()  # safe big-M
        for i in range(N):
            for k in range(K):
                for d in range(n):
                    m.addConstr(
                        quicksum(lam[k, j] * scenarios[j, d] for j in range(N)) >=
                        t * scenarios[i, d] - M * (1 - mu[i, k])
                    )

        # Each aggregate must be used at least once? Not strictly necessary but helps
        for k in range(K):
            m.addConstr(quicksum(mu[i, k] for i in range(N)) >= 1)

        m.setObjective(t, GRB.MAXIMIZE)
        m.optimize()

        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            t_val = t.X
            if m.status == GRB.OPTIMAL:
                t_val = t.X

            # Extract assignment and weights
            assignment = {}
            lambda_weights = {}
            for k in range(K):
                lambda_weights[k] = {}
                for j in range(N):
                    if lam[k, j].X > 1e-8:
                        lambda_weights[k][j] = float(lam[k, j].X)

            for i in range(N):
                for k in range(K):
                    if mu[i, k].X > 0.5:
                        assignment[i] = k
                        break

            one_stage_result = {
                "t": float(t_val),
                "approximation_ratio": float(1.0 / t_val) if t_val > 1e-9 else 1.0,
                "assignment": {int(k): int(v) for k, v in assignment.items()},
                "lambda_weights": {int(k): {int(jj): float(vv) for jj, vv in w.items()} for k, w in lambda_weights.items()},
                "solver_status": int(m.status),
                "best_bound": float(m.ObjBound),
                "mip_gap": float(m.MIPGap) if m.MIPGap is not None else 0.0
            }

            solution["one_stage_IP_mu"][str(K)] = one_stage_result

            # Update global best
            current_sol = {
                "objective_value": float(t_val),
                "one_stage_IP_mu": {str(K): one_stage_result},
                "two_stage_IP": {}
            }
            log_if_better(t_val, current_sol)

        # ------------------- Two-stage IP (scenario selection) -------------------
        if time.time() - start_time > args.time_limit * 0.9:
            continue

        m2 = Model("two_stage_IP")
        m2.setParam('Seed', 0)
        m2.setParam('MIPGap', 1e-4)
        m2.setParam('NumericFocus', 0)
        m2.setParam('Threads', 1)
        m2.setParam('TimeLimit', remaining_time())

        x = {}  # x[j] = 1 if scenario j is selected
        for j in range(N):
            x[j] = m2.addVar(vtype=GRB.BINARY, name=f"x_{j}")

        t2 = m2.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="t2")

        # Exactly K scenarios selected
        m2.addConstr(quicksum(x[j] for j in range(N)) == K)

        # For each original scenario i, there must be a selected j such that s[j,d] >= t * s[i,d] for all d
        for i in range(N):
            for d in range(n):
                m2.addConstr(
                    quicksum(x[j] * scenarios[j, d] for j in range(N)) >= t2 * scenarios[i, d]
                )

        m2.setObjective(t2, GRB.MAXIMIZE)
        m2.optimize()

        if m2.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            t2_val = t2.X
            selected = [j for j in range(N) if x[j].X > 0.5]

            two_stage_result = {
                "t": float(t2_val),
                "approximation_ratio": float(1.0 / t2_val) if t2_val > 1e-9 else 1.0,
                "selected_scenarios": [int(j) for j in selected]
            }

            solution["two_stage_IP"][str(K)] = two_stage_result

            # Update global best if better
            current_sol2 = {
                "objective_value": float(t2_val),
                "one_stage_IP_mu": {},
                "two_stage_IP": {str(K): two_stage_result}
            }
            log_if_better(t2_val, current_sol2)

    # Final objective is the best t found across all
    solution["objective_value"] = float(best_t_global)

    # Fill in any missing K with worst-case defaults if needed
    for K in K_values:
        kstr = str(K)
        if kstr not in solution["one_stage_IP_mu"]:
            solution["one_stage_IP_mu"][kstr] = {
                "t": 0.0,
                "approximation_ratio": 1.0,
                "assignment": {},
                "lambda_weights": {},
                "solver_status": -1,
                "best_bound": 0.0,
                "mip_gap": 0.0
            }
        if kstr not in solution["two_stage_IP"]:
            solution["two_stage_IP"][kstr] = {
                "t": 0.0,
                "approximation_ratio": 1.0,
                "selected_scenarios": list(range(K))
            }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path} with objective {solution['objective_value']}")

if __name__ == "__main__":
    main()