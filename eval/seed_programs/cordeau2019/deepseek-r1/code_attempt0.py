import json
import sys
import time
import gurobipy

def main():
    if len(sys.argv) != 4:
        print("Usage: python facility_location.py <instance_file> <solution_file> <time_limit>")
        sys.exit(1)

    instance_file = sys.argv[1]
    solution_file = sys.argv[2]
    time_limit = float(sys.argv[3])

    start_time = time.time()

    with open(instance_file) as f:
        instance = json.load(f)

    facilities = instance["facilities"]
    customers = instance["customers"]
    facility_costs = [f["opening_cost"] for f in facilities]
    customer_demands = [c["demand"] for c in customers]
    customer_to_facilities_coverage = instance["customer_to_facilities_coverage"]
    D = instance["covering_demand_D"]
    B = instance["budget_B"]

    num_facilities = len(facilities)
    num_customers = len(customers)
    F = list(range(num_facilities))
    C = list(range(num_customers))

    # Solve PSCLP
    model_psclp = gurobipy.Model()
    x_psclp = {}
    for j in F:
        x_psclp[j] = model_psclp.addVar(vtype=gurobipy.GRB.BINARY, name=f"x_psclp_{j}")

    z_psclp = {}
    for i in C:
        z_psclp[i] = model_psclp.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=1, name=f"z_psclp_{i}")

    model_psclp.setObjective(gurobipy.quicksum(facility_costs[j] * x_psclp[j] for j in F), gurobipy.GRB.MINIMIZE)

    for i in C:
        key = str(i)
        if key in customer_to_facilities_coverage:
            F_i = customer_to_facilities_coverage[key]
            model_psclp.addConstr(z_psclp[i] <= gurobipy.quicksum(x_psclp[j] for j in F_i), name=f"cov_psclp_{i}")
        else:
            model_psclp.addConstr(z_psclp[i] == 0, name=f"cov_psclp_{i}")

    model_psclp.addConstr(gurobipy.quicksum(customer_demands[i] * z_psclp[i] for i in C) >= D, name="demand_threshold_psclp")

    model_psclp.setParam('Seed', 0)
    model_psclp.setParam('MIPGap', 1e-4)
    model_psclp.setParam('NumericFocus', 0)
    model_psclp.setParam('Threads', 1)
    model_psclp.setParam('OutputFlag', 0)

    time_left_psclp = max(0, time_limit - (time.time() - start_time))
    model_psclp.setParam('TimeLimit', time_left_psclp)
    model_psclp.optimize()

    if model_psclp.SolCount > 0:
        open_facilities_psclp = [j for j in F if x_psclp[j].X > 0.5]
        covered_demand_psclp = 0
        for i in C:
            key = str(i)
            if key in customer_to_facilities_coverage:
                F_i = customer_to_facilities_coverage[key]
                if any(j in open_facilities_psclp for j in F_i):
                    covered_demand_psclp += customer_demands[i]
        cost_psclp = sum(facility_costs[j] for j in open_facilities_psclp)
    else:
        open_facilities_psclp = F
        covered_demand_psclp = 0
        for i in C:
            key = str(i)
            if key in customer_to_facilities_coverage and customer_to_facilities_coverage[key]:
                covered_demand_psclp += customer_demands[i]
        cost_psclp = sum(facility_costs[j] for j in F)

    # Solve MCLP
    time_left_mclp = max(0, time_limit - (time.time() - start_time))
    if time_left_mclp > 0:
        model_mclp = gurobipy.Model()
        x_mclp = {}
        for j in F:
            x_mclp[j] = model_mclp.addVar(vtype=gurobipy.GRB.BINARY, name=f"x_mclp_{j}")

        z_mclp = {}
        for i in C:
            z_mclp[i] = model_mclp.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=1, name=f"z_mclp_{i}")

        model_mclp.setObjective(gurobipy.quicksum(customer_demands[i] * z_mclp[i] for i in C), gurobipy.GRB.MAXIMIZE)

        for i in C:
            key = str(i)
            if key in customer_to_facilities_coverage:
                F_i = customer_to_facilities_coverage[key]
                model_mclp.addConstr(z_mclp[i] <= gurobipy.quicksum(x_mclp[j] for j in F_i), name=f"cov_mclp_{i}")
            else:
                model_mclp.addConstr(z_mclp[i] == 0, name=f"cov_mclp_{i}")

        model_mclp.addConstr(gurobipy.quicksum(facility_costs[j] * x_mclp[j] for j in F) <= B, name="budget_constraint_mclp")

        model_mclp.setParam('Seed', 0)
        model_mclp.setParam('MIPGap', 1e-4)
        model_mclp.setParam('NumericFocus', 0)
        model_mclp.setParam('Threads', 1)
        model_mclp.setParam('OutputFlag', 0)
        model_mclp.setParam('TimeLimit', time_left_mclp)
        model_mclp.optimize()

        if model_mclp.SolCount > 0:
            open_facilities_mclp = [j for j in F if x_mclp[j].X > 0.5]
            covered_demand_mclp = 0
            for i in C:
                key = str(i)
                if key in customer_to_facilities_coverage:
                    F_i = customer_to_facilities_coverage[key]
                    if any(j in open_facilities_mclp for j in F_i):
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

    with open(solution_file, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()