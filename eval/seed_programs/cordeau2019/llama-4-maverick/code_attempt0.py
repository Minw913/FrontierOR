import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def read_instance_data(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve_psclp(instance_data, model, logger, time_limit):
    num_facilities = instance_data['num_facilities']
    num_customers = instance_data['num_customers']
    facility_cost = instance_data['facility_cost']
    customer_demands = instance_data['customer_demands']
    customer_to_facilities_coverage = instance_data['customer_to_facilities_coverage']
    covering_demand_D = instance_data.get('covering_demand_D', None)
    if covering_demand_D is None:
        total_demand_D_bar = instance_data['total_demand_D_bar']
        covering_demand_fraction = instance_data['covering_demand_fraction']
        covering_demand_D = total_demand_D_bar * covering_demand_fraction

    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    facility_open = model.addVars(num_facilities, vtype=GRB.BINARY, name='facility_open')
    customer_covered = model.addVars(num_customers, vtype=GRB.BINARY, name='customer_covered')

    model.setObjective(gp.quicksum(facility_cost[i] * facility_open[i] for i in range(num_facilities)), GRB.MINIMIZE)

    model.addConstrs((customer_covered[j] <= gp.quicksum(facility_open[i] for i in customer_to_facilities_coverage[str(j)]) for j in range(num_customers)), name='customer_coverage')
    model.addConstr(gp.quicksum(customer_demands[j] * customer_covered[j] for j in range(num_customers)) >= covering_demand_D, name='demand_coverage')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        open_facilities = [i for i in range(num_facilities) if facility_open[i].x > 0.5]
        covered_demand = sum(customer_demands[j] for j in range(num_customers) if customer_covered[j].x > 0.5)
        psclp_solution = {
            'objective_value': model.getObjective().getValue(),
            'open_facilities': open_facilities,
            'covered_demand': covered_demand
        }
        if logger:
            logger.log_solution(model.getObjective().getValue(), psclp_solution)
        return psclp_solution
    else:
        return None

def solve_mclp(instance_data, model, logger, time_limit):
    num_facilities = instance_data['num_facilities']
    num_customers = instance_data['num_customers']
    facility_cost = instance_data['facility_cost']
    customer_demands = instance_data['customer_demands']
    customer_to_facilities_coverage = instance_data['customer_to_facilities_coverage']
    budget_B = instance_data['budget_B']

    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    facility_open = model.addVars(num_facilities, vtype=GRB.BINARY, name='facility_open')
    customer_covered = model.addVars(num_customers, vtype=GRB.BINARY, name='customer_covered')

    model.setObjective(gp.quicksum(customer_demands[j] * customer_covered[j] for j in range(num_customers)), GRB.MAXIMIZE)

    model.addConstrs((customer_covered[j] <= gp.quicksum(facility_open[i] for i in customer_to_facilities_coverage[str(j)]) for j in range(num_customers)), name='customer_coverage')
    model.addConstr(gp.quicksum(facility_cost[i] * facility_open[i] for i in range(num_facilities)) <= budget_B, name='budget_constraint')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        open_facilities = [i for i in range(num_facilities) if facility_open[i].x > 0.5]
        covered_demand = sum(customer_demands[j] for j in range(num_customers) if customer_covered[j].x > 0.5)
        mclp_solution = {
            'objective_value': model.getObjective().getValue(),
            'open_facilities': open_facilities,
            'covered_demand': covered_demand
        }
        if logger:
            logger.log_solution(model.getObjective().getValue(), mclp_solution)
        return mclp_solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance_data = read_instance_data(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model_psclp = gp.Model('PSCLP')
    model_mclp = gp.Model('MCLP')

    psclp_solution = solve_psclp(instance_data, model_psclp, logger, args.time_limit)
    mclp_solution = solve_mclp(instance_data, model_mclp, logger, args.time_limit)

    solution = {
        'objective_value': psclp_solution['objective_value'],
        'primary_problem_type': 'PSCLP',
        'results': {
            'PSCLP': psclp_solution,
            'MCLP': mclp_solution
        }
    }

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()