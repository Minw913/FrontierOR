import json
import argparse
import time
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    facility_cost = instance['facility_cost']
    customer_demands = instance['customer_demands']
    facility_to_customers = instance['facility_to_customers_coverage']
    customer_to_facilities = instance['customer_to_facilities_coverage']
    covering_demand_D = instance['covering_demand_D']
    budget_B = instance['budget_B']
    total_demand = instance['total_demand_D_bar']

    # Precompute coverage as lists of ints
    F2C = [facility_to_customers.get(str(i), []) for i in range(num_facilities)]
    C2F = [customer_to_facilities.get(str(j), []) for j in range(num_customers)]

    # Determine which variant to prioritize
    # We will solve both, but report the one with better relative quality within time
    # However, to respect time limit strictly, we solve PSCLP first (min cost), then MCLP if time remains

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    best_psclp_cost = float('inf')
    best_psclp_open = []
    best_psclp_covered = 0.0

    best_mclp_demand = 0.0
    best_mclp_open = []
    best_mclp_covered = 0.0

    # Solve PSCLP (Partial Set Covering Location Problem) - Minimize cost
    try:
        model_ps = gp.Model("PSCLP")
        model_ps.setParam('Seed', 0)
        model_ps.setParam('MIPGap', 1e-4)
        model_ps.setParam('NumericFocus', 0)
        model_ps.setParam('Threads', 1)
        model_ps.setParam('TimeLimit', max(1, args.time_limit - 5))

        x = model_ps.addVars(num_facilities, vtype=GRB.BINARY, name="x")
        y = model_ps.addVars(num_customers, vtype=GRB.BINARY, name="y")

        # Objective: minimize total opening cost
        model_ps.setObjective(gp.quicksum(facility_cost[i] * x[i] for i in range(num_facilities)), GRB.MINIMIZE)

        # Covering constraints
        for j in range(num_customers):
            if C2F[j]:
                model_ps.addConstr(y[j] <= gp.quicksum(x[i] for i in C2F[j]), f"cover_{j}")
            else:
                model_ps.addConstr(y[j] == 0, f"cover_{j}_impossible")

        # Demand requirement
        model_ps.addConstr(gp.quicksum(customer_demands[j] * y[j] for j in range(num_customers)) >= covering_demand_D,
                           "demand_threshold")

        model_ps.optimize()

        if model_ps.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            if model_ps.SolCount > 0:
                current_cost = model_ps.ObjVal
                open_fac = [i for i in range(num_facilities) if x[i].X > 0.5]
                covered_dem = sum(customer_demands[j] * y[j].X for j in range(num_customers))

                if current_cost < best_psclp_cost:
                    best_psclp_cost = current_cost
                    best_psclp_open = open_fac
                    best_psclp_covered = covered_dem
                    if logger:
                        sol_dict = {
                            "PSCLP": {
                                "objective_value": float(current_cost),
                                "open_facilities": [int(i) for i in open_fac],
                                "covered_demand": float(covered_dem)
                            }
                        }
                        logger.log_solution(float(current_cost), sol_dict)
    except Exception as e:
        print("PSCLP solver error:", e, file=sys.stderr)

    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time < 2:
        # Not enough time for MCLP, skip or use trivial
        pass
    else:
        # Solve MCLP (Maximal Covering Location Problem) - Maximize covered demand
        try:
            model_mc = gp.Model("MCLP")
            model_mc.setParam('Seed', 0)
            model_mc.setParam('MIPGap', 1e-4)
            model_mc.setParam('NumericFocus', 0)
            model_mc.setParam('Threads', 1)
            model_mc.setParam('TimeLimit', int(remaining_time - 1))

            x = model_mc.addVars(num_facilities, vtype=GRB.BINARY, name="x")
            y = model_mc.addVars(num_customers, vtype=GRB.BINARY, name="y")

            # Objective: maximize covered demand
            model_mc.setObjective(gp.quicksum(customer_demands[j] * y[j] for j in range(num_customers)), GRB.MAXIMIZE)

            # Budget constraint
            model_mc.addConstr(gp.quicksum(facility_cost[i] * x[i] for i in range(num_facilities)) <= budget_B,
                               "budget")

            # Covering constraints
            for j in range(num_customers):
                if C2F[j]:
                    model_mc.addConstr(y[j] <= gp.quicksum(x[i] for i in C2F[j]), f"cover_{j}")
                else:
                    model_mc.addConstr(y[j] == 0, f"cover_{j}_impossible")

            model_mc.optimize()

            if model_mc.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
                if model_mc.SolCount > 0:
                    current_demand = model_mc.ObjVal
                    open_fac = [i for i in range(num_facilities) if x[i].X > 0.5]
                    covered_dem = current_demand

                    if current_demand > best_mclp_demand + 1e-6:
                        best_mclp_demand = current_demand
                        best_mclp_open = open_fac
                        best_mclp_covered = covered_dem
                        if logger:
                            sol_dict = {
                                "MCLP": {
                                    "objective_value": float(current_demand),
                                    "open_facilities": [int(i) for i in open_fac],
                                    "covered_demand": float(covered_dem)
                                }
                            }
                            logger.log_solution(float(current_demand), sol_dict)
        except Exception as e:
            print("MCLP solver error:", e, file=sys.stderr)

    # Decide which primary problem to report
    # Prefer PSCLP if we have a feasible solution, otherwise MCLP
    if best_psclp_cost < float('inf'):
        primary_type = "PSCLP"
        objective_value = float(best_psclp_cost)
        results = {
            "PSCLP": {
                "objective_value": float(best_psclp_cost),
                "open_facilities": [int(i) for i in best_psclp_open],
                "covered_demand": float(best_psclp_covered)
            },
            "MCLP": {
                "objective_value": float(best_mclp_demand),
                "open_facilities": [int(i) for i in best_mclp_open],
                "covered_demand": float(best_mclp_covered)
            }
        }
    else:
        primary_type = "MCLP"
        objective_value = float(best_mclp_demand)
        results = {
            "PSCLP": {
                "objective_value": float(best_psclp_cost if best_psclp_cost < float('inf') else 0.0),
                "open_facilities": [],
                "covered_demand": 0.0
            },
            "MCLP": {
                "objective_value": float(best_mclp_demand),
                "open_facilities": [int(i) for i in best_mclp_open],
                "covered_demand": float(best_mclp_covered)
            }
        }

    solution = {
        "objective_value": objective_value,
        "primary_problem_type": primary_type,
        "results": results
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path} with primary {primary_type}", file=sys.stderr)


if __name__ == "__main__":
    main()