import argparse
import json
import time
import gurobipy as gp
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    budget_B = instance['budget_B']
    covering_demand_D = instance['covering_demand_D']
    facility_costs = instance['facility_cost']
    customer_demands = instance['customer_demands']
    customer_to_facilities_coverage = instance['customer_to_facilities_coverage']
    total_demand_D_bar = instance['total_demand_D_bar']

    F = list(range(num_facilities))
    C = list(range(num_customers))

    customer_coverage_sets = []
    for i in range(num_customers):
        key = str(i)
        if key in customer_to_facilities_coverage:
            customer_coverage_sets.append(set(customer_to_facilities_coverage[key]))
        else:
            customer_coverage_sets.append(set())

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def psclp_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(model._x)
            open_facilities = [j for j in F if x_vals[j] > 0.5]
            open_set = set(open_facilities)
            covered_demand = 0
            for i in C:
                if customer_coverage_sets[i] and customer_coverage_sets[i] & open_set:
                    covered_demand += customer_demands[i]
            cost = sum(facility_costs[j] for j in open_facilities)
            solution_dict = {
                "objective_value": cost,
                "primary_problem_type": "PSCLP",
                "results": {
                    "PSCLP": {
                        "objective_value": cost,
                        "open_facilities": sorted(open_facilities),
                        "covered_demand": covered_demand
                    },
                    "MCLP": {
                        "objective_value": 0,
                        "open_facilities": [],
                        "covered_demand": 0
                    }
                }
            }
            logger.log_solution(cost, solution_dict)

    model_psclp = gp.Model()
    x_psclp = model_psclp.addVars(num_facilities, vtype=gp.GRB.BINARY, name="x_psclp")
    z_psclp = model_psclp.addVars(num_customers, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="z_psclp")

    model_psclp.setObjective(gp.quicksum(facility_costs[j] * x_psclp[j] for j in F), gp.GRB.MINIMIZE)

    for i in C:
        if customer_coverage_sets[i]:
            model_psclp.addConstr(z_psclp[i] <= gp.quicksum(x_psclp[j] for j in customer_coverage_sets[i]), f"cov_psclp_{i}")
        else:
            model_psclp.addConstr(z_psclp[i] == 0, f"cov_psclp_{i}")

    model_psclp.addConstr(gp.quicksum(customer_demands[i] * z_psclp[i] for i in C) >= covering_demand_D, "demand_threshold_psclp")

    model_psclp.setParam('Seed', 0)
    model_psclp.setParam('MIPGap', 1e-4)
    model_psclp.setParam('NumericFocus', 0)
    model_psclp.setParam('Threads', 1)
    model_psclp.setParam('OutputFlag', 0)

    time_left_psclp = max(0, args.time_limit - (time.time() - start_time))
    model_psclp.setParam('TimeLimit', time_left_psclp)

    if logger:
        model_psclp._x = x_psclp
        model_psclp.optimize(psclp_callback)
    else:
        model_psclp.optimize()

    if model_psclp.SolCount > 0:
        open_facilities_psclp = [j for j in F if x_psclp[j].X > 0.5]
        open_set_psclp = set(open_facilities_psclp)
        covered_demand_psclp = 0
        for i in C:
            if customer_coverage_sets[i] and customer_coverage_sets[i] & open_set_psclp:
                covered_demand_psclp += customer_demands[i]
        cost_psclp = sum(facility_costs[j] for j in open_facilities_psclp)
    else:
        open_facilities_psclp = F
        open_set_psclp = set(open_facilities_psclp)
        covered_demand_psclp = 0
        for i in C:
            if customer_coverage_sets[i] and customer_coverage_sets[i] & open_set_psclp:
                covered_demand_psclp += customer_demands[i]
        cost_psclp = sum(facility_costs)

    time_left_mclp = max(0, args.time_limit - (time.time() - start_time))
    if time_left_mclp > 0:
        model_mclp = gp.Model()
        x_mclp = model_mclp.addVars(num_facilities, vtype=gp.GRB.BINARY, name="x_mclp")
        z_mclp = model_mclp.addVars(num_customers, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="z_mclp")

        model_mclp.setObjective(gp.quicksum(customer_demands[i] * z_mclp[i] for i in C), gp.GRB.MAXIMIZE)

        for i in C:
            if customer_coverage_sets[i]:
                model_mclp.addConstr(z_mclp[i] <= gp.quicksum(x_mclp[j] for j in customer_coverage_sets[i]), f"cov_mclp_{i}")
            else:
                model_mclp.addConstr(z_mclp[i] == 0, f"cov_mclp_{i}")

        model_mclp.addConstr(gp.quicksum(facility_costs[j] * x_mclp[j] for j in F) <= budget_B, "budget_constraint_mclp")

        model_mclp.setParam('Seed', 0)
        model_mclp.setParam('MIPGap', 1e-4)
        model_mclp.setParam('NumericFocus', 0)
        model_mclp.setParam('Threads', 1)
        model_mclp.setParam('OutputFlag', 0)
        model_mclp.setParam('TimeLimit', time_left_mclp)
        model_mclp.optimize()

        if model_mclp.SolCount > 0:
            open_facilities_mclp = [j for j in F if x_mclp[j].X > 0.5]
            open_set_mclp = set(open_facilities_mclp)
            covered_demand_mclp = 0
            for i in C:
                if customer_coverage_sets[i] and customer_coverage_sets[i] & open_set_mclp:
                    covered_demand_mclp += customer_demands[i]
        else:
            open_facilities_mclp = []
            covered_demand_mclp = 0
    else:
        open_facilities_mclp = []
        covered_demand_mclp = 0

    solution = {
        "objective_value": cost_psclp,
        "primary_problem_type": "PSCLP",
        "results": {
            "PSCLP": {
                "objective_value": cost_psclp,
                "open_facilities": sorted(open_facilities_psclp),
                "covered_demand": covered_demand_psclp
            },
            "MCLP": {
                "objective_value": covered_demand_mclp,
                "open_facilities": sorted(open_facilities_mclp),
                "covered_demand": covered_demand_mclp
            }
        }
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()