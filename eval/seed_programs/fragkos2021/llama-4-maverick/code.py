import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_nodes = instance['num_nodes']
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    nodes = instance['nodes']
    transportation_cost_matrix = instance['transportation_cost_matrix']
    penalty_cost = instance['penalty_cost']

    demand = [node['demand'] for node in nodes[:num_customers]]
    fixed_setup_cost = [node['fixed_setup_cost'] for node in nodes[:num_facilities]]
    survival_probability = [node['survival_probability'] for node in nodes[:num_facilities]]

    try:
        model = gp.Model("facility_location")
        model.setParam("Threads", 1)
        facility_open = model.addVars(num_facilities, vtype=GRB.BINARY, name="facility_open")

        total_fixed_cost = gp.quicksum(fixed_setup_cost[i] * facility_open[i] for i in range(num_facilities))

        # Linearize the expected cost calculation
        customer_cost = model.addVars(num_customers, name="customer_cost")
        for c in range(num_customers):
            for f in range(num_facilities):
                model.addConstr(customer_cost[c] <= transportation_cost_matrix[c][f] + (1 - facility_open[f]) * (penalty_cost[c] - transportation_cost_matrix[c][f]))
            model.addConstr(customer_cost[c] <= penalty_cost[c])

        expected_cost = gp.quicksum(demand[c] * customer_cost[c] for c in range(num_customers))

        model.setObjective(total_fixed_cost + expected_cost, GRB.MINIMIZE)

        model.Params.TimeLimit = time_limit
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            facility_locations = [int(facility_open[i].X) for i in range(num_facilities)]
            objective_value = model.ObjVal

            if logger:
                logger.log(objective_value)

            return objective_value, facility_locations
        else:
            print("No solution found.")
            return None, None

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Solve facility location problem under disruption risk.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the JSON solution file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, facility_locations = solve(instance, args.time_limit, logger)

    if objective_value is not None and facility_locations is not None:
        solution = {
            "objective_value": objective_value,
            "facility_locations": facility_locations
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()