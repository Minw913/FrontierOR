import argparse
import json
import time
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_instance(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    start_time = time.time()

    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    budget_B = data['budget_B']
    covering_demand_D = data['covering_demand_D']
    total_demand = data['total_demand_D_bar']
    facility_cost = data['facility_cost']
    customer_demands = data['customer_demands']

    # Coverage relationships
    facility_to_customers = {}
    for k, v in data['facility_to_customers_coverage'].items():
        facility_to_customers[int(k)] = v

    customer_to_facilities = {}
    for k, v in data['customer_to_facilities_coverage'].items():
        customer_to_facilities[int(k)] = v

    # Time allocation: split remaining time between PSCLP and MCLP
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    time_per_problem = remaining / 2.0

    # ==================== PSCLP ====================
    # Minimize total opening cost s.t. covered demand >= D
    psclp_result = None
    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        m = gp.Model("PSCLP", env=env)
        m.setParam('Seed', 0)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NumericFocus', 0)
        m.setParam('Threads', 1)
        m.setParam('TimeLimit', max(1, time_per_problem))

        # Variables
        y = m.addVars(num_facilities, vtype=GRB.BINARY, name="y")
        z = m.addVars(num_customers, vtype=GRB.BINARY, name="z")

        # Objective: minimize total cost
        m.setObjective(gp.quicksum(facility_cost[j] * y[j] for j in range(num_facilities)), GRB.MINIMIZE)

        # Coverage constraint: z[i] <= sum of y[j] for j in coverage of i
        for i in range(num_customers):
            facs = customer_to_facilities.get(i, [])
            if len(facs) == 0:
                m.addConstr(z[i] == 0)
            else:
                m.addConstr(z[i] <= gp.quicksum(y[j] for j in facs))

        # Demand threshold
        m.addConstr(gp.quicksum(customer_demands[i] * z[i] for i in range(num_customers)) >= covering_demand_D)

        # Callback for logging
        psclp_logger = SolutionLogger(log_path, sense="minimize") if log_path else None
        
        def psclp_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if psclp_logger:
                    psclp_logger.log(obj)

        m.optimize(psclp_callback)

        if m.SolCount > 0:
            open_facs = [j for j in range(num_facilities) if y[j].X > 0.5]
            # Compute covered customers
            covered_customers = set()
            for j in open_facs:
                for i in facility_to_customers.get(j, []):
                    covered_customers.add(i)
            covered_dem = sum(customer_demands[i] for i in covered_customers)
            psclp_result = {
                "objective_value": float(m.ObjVal),
                "open_facilities": open_facs,
                "covered_demand": float(covered_dem)
            }
        
        m.dispose()
        env.dispose()
    except Exception as e:
        print(f"PSCLP error: {e}")

    # ==================== MCLP ====================
    # Maximize covered demand s.t. total cost <= budget
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    mclp_result = None
    
    try:
        env2 = gp.Env(empty=True)
        env2.setParam('OutputFlag', 0)
        env2.start()
        
        m2 = gp.Model("MCLP", env=env2)
        m2.setParam('Seed', 0)
        m2.setParam('MIPGap', 1e-4)
        m2.setParam('NumericFocus', 0)
        m2.setParam('Threads', 1)
        m2.setParam('TimeLimit', max(1, remaining - 1))

        # Variables
        y2 = m2.addVars(num_facilities, vtype=GRB.BINARY, name="y")
        z2 = m2.addVars(num_customers, vtype=GRB.BINARY, name="z")

        # Objective: maximize covered demand
        m2.setObjective(gp.quicksum(customer_demands[i] * z2[i] for i in range(num_customers)), GRB.MAXIMIZE)

        # Coverage constraint
        for i in range(num_customers):
            facs = customer_to_facilities.get(i, [])
            if len(facs) == 0:
                m2.addConstr(z2[i] == 0)
            else:
                m2.addConstr(z2[i] <= gp.quicksum(y2[j] for j in facs))

        # Budget constraint
        m2.addConstr(gp.quicksum(facility_cost[j] * y2[j] for j in range(num_facilities)) <= budget_B)

        mclp_logger = SolutionLogger(log_path, sense="maximize") if log_path else None

        def mclp_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if mclp_logger:
                    mclp_logger.log(obj)

        m2.optimize(mclp_callback)

        if m2.SolCount > 0:
            open_facs2 = [j for j in range(num_facilities) if y2[j].X > 0.5]
            covered_customers2 = set()
            for j in open_facs2:
                for i in facility_to_customers.get(j, []):
                    covered_customers2.add(i)
            covered_dem2 = sum(customer_demands[i] for i in covered_customers2)
            mclp_result = {
                "objective_value": float(m2.ObjVal),
                "open_facilities": open_facs2,
                "covered_demand": float(covered_dem2)
            }
        
        m2.dispose()
        env2.dispose()
    except Exception as e:
        print(f"MCLP error: {e}")

    # Build solution
    # Determine primary problem type - use PSCLP if we have it, else MCLP
    if psclp_result is not None:
        primary = "PSCLP"
        obj_val = psclp_result["objective_value"]
    elif mclp_result is not None:
        primary = "MCLP"
        obj_val = mclp_result["objective_value"]
    else:
        primary = "PSCLP"
        obj_val = 0.0

    # Default results if solver failed
    if psclp_result is None:
        psclp_result = {
            "objective_value": 0.0,
            "open_facilities": [],
            "covered_demand": 0.0
        }
    if mclp_result is None:
        mclp_result = {
            "objective_value": 0.0,
            "open_facilities": [],
            "covered_demand": 0.0
        }

    solution = {
        "objective_value": obj_val,
        "primary_problem_type": primary,
        "results": {
            "PSCLP": psclp_result,
            "MCLP": mclp_result
        }
    }

    # Final logging
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    if logger:
        logger.log_solution(obj_val, solution)

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve_instance(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()