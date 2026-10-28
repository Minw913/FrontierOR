import argparse
import json
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def calculate_distances(instance):
    # Distances are precomputed in the instance data
    return instance['distances']

def build_model(instance, distances):
    gp = instance['global_parameters']
    ip = instance['instance_parameters']
    params = instance['interdiction_parameters']
    customers = instance['customers']
    level1_facilities = instance['level1_facilities']
    level2_facilities = instance['level2_facilities']
    num_level1_facilities = ip['num_level1_facilities']
    num_level2_facilities = ip['num_level2_facilities']
    num_customers = ip['num_customers']
    num_interdiction_levels_K = ip['num_interdiction_levels_K']
    k_max = ip['k_max']
    theta = gp['theta']
    sigma = gp['sigma']
    alpha = gp['alpha']
    beta = gp['beta']
    gamma = gp['gamma']
    alpha_prime = gp['alpha_prime']
    beta_prime = gp['beta_prime']
    gamma_prime = gp['gamma_prime']
    total_interdiction_budget = ip['total_interdiction_budget']
    interdiction_cost_level1_h1 = params['interdiction_cost_level1_h1']
    interdiction_cost_level2_h2 = params['interdiction_cost_level2_h2']
    capacity_reduction_level1_d1 = params['capacity_reduction_level1_d1']
    capacity_reduction_level2_d2 = params['capacity_reduction_level2_d2']

    model = Model('interdiction_model')
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)

    # Interdiction variables
    z1 = model.addVars(num_level1_facilities, num_interdiction_levels_K, vtype=GRB.BINARY, name='z1')
    z2 = model.addVars(num_level2_facilities, num_interdiction_levels_K, vtype=GRB.BINARY, name='z2')

    # Customer demand allocation variables
    x1_I = model.addVars(num_customers, num_level1_facilities, vtype=GRB.CONTINUOUS, name='x1_I', lb=0)
    x1_II = model.addVars(num_customers, num_level2_facilities, vtype=GRB.CONTINUOUS, name='x1_II', lb=0)
    x2_II = model.addVars(num_customers, num_level2_facilities, vtype=GRB.CONTINUOUS, name='x2_II', lb=0)
    o1 = model.addVars(num_customers, vtype=GRB.CONTINUOUS, name='o1', lb=0)
    o2 = model.addVars(num_customers, vtype=GRB.CONTINUOUS, name='o2', lb=0)

    # Referral demand variables
    r_I_II = model.addVars(num_level1_facilities, num_level2_facilities, vtype=GRB.CONTINUOUS, name='r_I_II', lb=0)
    o_r_I = model.addVars(num_level1_facilities, vtype=GRB.CONTINUOUS, name='o_r_I', lb=0)
    o_r_o1 = model.addVars(num_customers, vtype=GRB.CONTINUOUS, name='o_r_o1', lb=0)

    # Constraints for interdiction variables
    model.addConstrs((quicksum(z1[i, k] for k in range(num_interdiction_levels_K)) == 1 for i in range(num_level1_facilities)), name='z1_sum')
    model.addConstrs((quicksum(z2[j, k] for k in range(num_interdiction_levels_K)) == 1 for j in range(num_level2_facilities)), name='z2_sum')
    model.addConstr(quicksum(interdiction_cost_level1_h1.get(str(k), 0) * z1[i, k] for i in range(num_level1_facilities) for k in range(num_interdiction_levels_K)) + 
                    quicksum(interdiction_cost_level2_h2.get(str(k), 0) * z2[j, k] for j in range(num_level2_facilities) for k in range(num_interdiction_levels_K)) <= total_interdiction_budget, name='interdiction_budget')

    # Demand satisfaction constraints
    model.addConstrs((quicksum(x1_I[i, j] for j in range(num_level1_facilities)) + quicksum(x1_II[i, j] for j in range(num_level2_facilities)) + o1[i] == theta * customers['demands'][i] for i in range(num_customers)), name='type1_demand')
    model.addConstrs((quicksum(x2_II[i, j] for j in range(num_level2_facilities)) + o2[i] == (1 - theta) * customers['demands'][i] for i in range(num_customers)), name='type2_demand')
    model.addConstrs((quicksum(r_I_II[j, k] for k in range(num_level2_facilities)) + o_r_I[j] == sigma * quicksum(x1_I[i, j] for i in range(num_customers)) for j in range(num_level1_facilities)), name='referral_demand_I')
    model.addConstrs((o_r_o1[i] == sigma * o1[i] for i in range(num_customers)), name='referral_demand_o1')

    # Capacity constraints
    model.addConstrs((quicksum(x1_I[i, j] for i in range(num_customers)) <= level1_facilities['capacity_type1'][j] * (1 - quicksum(capacity_reduction_level1_d1.get(str(k), 0) * z1[j, k] for k in range(num_interdiction_levels_K))) for j in range(num_level1_facilities)), name='level1_capacity')
    model.addConstrs((quicksum(x1_II[i, j] for i in range(num_customers)) <= level2_facilities['capacity_type1'][j] * (1 - quicksum(capacity_reduction_level2_d2.get(str(k), 0) * z2[j, k] for k in range(num_interdiction_levels_K))) for j in range(num_level2_facilities)), name='level2_type1_capacity')
    model.addConstrs((quicksum(x2_II[i, j] for i in range(num_customers)) + quicksum(r_I_II[i, j] for i in range(num_level1_facilities)) <= level2_facilities['capacity_type2'][j] * (1 - quicksum(capacity_reduction_level2_d2.get(str(k), 0) * z2[j, k] for k in range(num_interdiction_levels_K))) for j in range(num_level2_facilities)), name='level2_type2_capacity')

    # Calculate defender's cost
    defender_cost = quicksum(alpha * distances['customer_to_level1_facility'][i][j] * x1_I[i, j] for i in range(num_customers) for j in range(num_level1_facilities)) + \
                    quicksum(beta * distances['customer_to_level2_facility'][i][j] * x1_II[i, j] for i in range(num_customers) for j in range(num_level2_facilities)) + \
                    quicksum(beta * distances['customer_to_level2_facility'][i][j] * x2_II[i, j] for i in range(num_customers) for j in range(num_level2_facilities)) + \
                    quicksum(gamma * distances['level1_to_level2_facility'][j][k] * r_I_II[j, k] for j in range(num_level1_facilities) for k in range(num_level2_facilities)) + \
                    quicksum(alpha_prime * o1[i] for i in range(num_customers)) + \
                    quicksum(beta_prime * o2[i] for i in range(num_customers)) + \
                    quicksum(gamma_prime * o_r_I[j] for j in range(num_level1_facilities)) + \
                    quicksum(gamma_prime * o_r_o1[i] for i in range(num_customers)) + \
                    quicksum(sigma * gamma_prime * o1[i] for i in range(num_customers))

    model.setObjective(defender_cost, GRB.MAXIMIZE)

    return model, z1, z2

def solve_instance(instance_path, time_limit, log_path=None):
    instance = read_instance(instance_path)
    distances = calculate_distances(instance)
    model, z1, z2 = build_model(instance, distances)

    if log_path:
        logger = SolutionLogger(log_path, sense="maximize")
    else:
        logger = None

    model.Params.TimeLimit = time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        interdiction_pattern = {}
        num_level1_facilities = instance['instance_parameters']['num_level1_facilities']
        num_level2_facilities = instance['instance_parameters']['num_level2_facilities']
        num_interdiction_levels_K = instance['instance_parameters']['num_interdiction_levels_K']
        for i in range(num_level1_facilities):
            for k in range(num_interdiction_levels_K):
                if z1[i, k].x > 0.5:
                    interdiction_pattern[f'S1_{i}'] = k
        for j in range(num_level2_facilities):
            for k in range(num_interdiction_levels_K):
                if z2[j, k].x > 0.5:
                    interdiction_pattern[f'S2_{j}'] = k

        solution = {
            'objective_value': objective_value,
            'interdiction_pattern': interdiction_pattern
        }
        return solution
    else:
        raise Exception(f"Model status: {model.status}")

def main():
    parser = argparse.ArgumentParser(description='Solve the interdiction problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    solution = solve_instance(args.instance_path, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()