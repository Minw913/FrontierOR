import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit):
    T_planning_horizon_minutes = instance['parameters']['T_planning_horizon_minutes']
    D_num_train_services = instance['parameters']['D_num_train_services']
    NT_num_demand_periods = instance['parameters']['NT_num_demand_periods']
    theta_period_length_minutes = instance['parameters']['theta_period_length_minutes']
    h_min_minutes = instance['parameters']['h_min_minutes']
    h_max_minutes = instance['parameters']['h_max_minutes']
    C_train_capacity_passengers = instance['parameters']['C_train_capacity_passengers']
    demand_periods = instance['demand_periods']

    arrival_rates = [period['lambda_pax_per_min'] for period in demand_periods]

    model = Model("TrainScheduling")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Decision Variables
    z = model.addVars(D_num_train_services, NT_num_demand_periods, vtype=GRB.BINARY, name='z')
    t = model.addVars(D_num_train_services, vtype=GRB.CONTINUOUS, name='t')
    h = model.addVars(D_num_train_services + 1, vtype=GRB.CONTINUOUS, name='h')
    w = model.addVars(D_num_train_services + 1, vtype=GRB.CONTINUOUS, name='w')
    b = model.addVars(D_num_train_services, vtype=GRB.CONTINUOUS, name='b')
    A = model.addVars(D_num_train_services + 1, vtype=GRB.CONTINUOUS, name='A')

    # Objective Function
    objective = quicksum(h[d] * (w[d] + 0.5 * A[d]) for d in range(D_num_train_services + 1))
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    model.addConstr(quicksum(z[d, p] for p in range(NT_num_demand_periods)) == 1 for d in range(D_num_train_services))
    model.addConstr(t[d] >= p * theta_period_length_minutes for d in range(D_num_train_services) for p in range(NT_num_demand_periods) if z[d, p].x == 1)
    model.addConstr(t[d] < (p + 1) * theta_period_length_minutes for d in range(D_num_train_services) for p in range(NT_num_demand_periods) if z[d, p].x == 1)
    model.addConstr(z[d, p] <= quicksum(z[d-1, p_prime] for p_prime in range(p+1)) for d in range(1, D_num_train_services) for p in range(NT_num_demand_periods))
    model.addConstr(h[0] + quicksum(h[d] for d in range(1, D_num_train_services + 1)) == T_planning_horizon_minutes)
    model.addConstr(t[d] == t[d-1] + h[d] for d in range(1, D_num_train_services + 1))
    model.addConstr(h[d] >= h_min_minutes for d in range(1, D_num_train_services))
    model.addConstr(h[d] <= h_max_minutes for d in range(1, D_num_train_services))
    model.addConstr(A[0] == quicksum(arrival_rates[p] * min(theta_period_length_minutes, t[0]) for p in range(NT_num_demand_periods) if t[0] >= p * theta_period_length_minutes))
    model.addConstr(A[d] == quicksum(arrival_rates[p] * theta_period_length_minutes for p in range(NT_num_demand_periods) if t[d-1] < p * theta_period_length_minutes and t[d] >= p * theta_period_length_minutes) + 
                    arrival_rates[[p for p in range(NT_num_demand_periods) if t[d-1] < (p+1) * theta_period_length_minutes and t[d] >= p * theta_period_length_minutes][0]] * (t[d] - max(t[d-1], [p for p in range(NT_num_demand_periods) if t[d-1] < (p+1) * theta_period_length_minutes and t[d] >= p * theta_period_length_minutes][0] * theta_period_length_minutes)) for d in range(1, D_num_train_services + 1))
    model.addConstr(w[0] == 0)
    model.addConstr(w[d] == w[d-1] + A[d-1] - b[d-1] for d in range(1, D_num_train_services + 1))
    model.addConstr(b[d] <= C_train_capacity_passengers for d in range(D_num_train_services))
    model.addConstr(b[d] <= w[d] + A[d] for d in range(D_num_train_services))

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        departure_schedule_minutes = {f'train_{d}': t[d].x for d in range(D_num_train_services)}
        objective_value = model.objVal
        return objective_value, departure_schedule_minutes
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, departure_schedule_minutes = solve(instance, args.time_limit)

    if objective_value is not None and departure_schedule_minutes is not None:
        solution = {
            "objective_value": objective_value,
            "departure_schedule_minutes": departure_schedule_minutes
        }
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
        if logger:
            logger.log(objective_value)

if __name__ == "__main__":
    main()