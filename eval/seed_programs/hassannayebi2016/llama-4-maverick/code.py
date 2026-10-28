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
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    z = model.addVars(D_num_train_services, NT_num_demand_periods, vtype=GRB.BINARY, name='z')
    t = model.addVars(D_num_train_services, vtype=GRB.CONTINUOUS, name='t')
    h = model.addVars(D_num_train_services + 1, vtype=GRB.CONTINUOUS, name='h')
    w = model.addVars(D_num_train_services + 1, vtype=GRB.CONTINUOUS, name='w')
    b = model.addVars(D_num_train_services, vtype=GRB.CONTINUOUS, name='b')
    A = model.addVars(D_num_train_services + 1, vtype=GRB.CONTINUOUS, name='A')

    model.addConstrs((quicksum(z[d, p] for p in range(NT_num_demand_periods)) == 1 for d in range(D_num_train_services)), name='c1')
    model.addConstrs((t[d] >= quicksum(p * theta_period_length_minutes * z[d, p] for p in range(NT_num_demand_periods)) for d in range(D_num_train_services)), name='c2_lb')
    model.addConstrs((t[d] < quicksum((p + 1) * theta_period_length_minutes * z[d, p] for p in range(NT_num_demand_periods)) for d in range(D_num_train_services)), name='c2_ub')
    for d in range(1, D_num_train_services):
        for p in range(NT_num_demand_periods):
            model.addConstr(quicksum(z[d-1, p_prime] for p_prime in range(p+1)) >= z[d, p], name='c3')
    model.addConstr(h[0] == t[0], name='c4_initial')
    model.addConstr(quicksum(h[d] for d in range(D_num_train_services + 1)) == T_planning_horizon_minutes, name='c4_rest')
    model.addConstrs((h[d] == t[d] - t[d-1] for d in range(1, D_num_train_services + 1)), name='c5')
    model.addConstrs((h[d] >= h_min_minutes for d in range(1, D_num_train_services)), name='c6_lb')
    model.addConstrs((h[d] <= h_max_minutes for d in range(1, D_num_train_services)), name='c6_ub')

    def compute_A(d):
        A_expr = 0
        for p in range(NT_num_demand_periods):
            start_time = p * theta_period_length_minutes
            end_time = (p + 1) * theta_period_length_minutes
            if d == 0:
                A_expr += arrival_rates[p] * min(theta_period_length_minutes, max(0, t[d] - start_time))
            elif d == D_num_train_services:
                A_expr += arrival_rates[p] * max(0, end_time - max(t[d-1], start_time))
            else:
                A_expr += arrival_rates[p] * max(0, min(t[d], end_time) - max(t[d-1], start_time))
        return A_expr

    model.addConstr(A[0] == compute_A(0), name='c7_initial')
    for d in range(1, D_num_train_services + 1):
        model.addConstr(A[d] == compute_A(d), name=f'c7_interior_{d}')

    model.addConstr(w[0] == 0, name='c8_initial')
    model.addConstrs((w[d] == w[d-1] + A[d-1] - b[d-1] for d in range(1, D_num_train_services + 1)), name='c8_dynamic')
    model.addConstrs((b[d] <= C_train_capacity_passengers for d in range(D_num_train_services)), name='c9_capacity')
    model.addConstrs((b[d] <= w[d] + A[d] for d in range(D_num_train_services)), name='c10_demand')

    objective = quicksum(h[d] * (w[d] + 0.5 * A[d]) for d in range(D_num_train_services + 1))
    model.setObjective(objective, GRB.MINIMIZE)

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