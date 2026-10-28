import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, log_path):
    n_generators = instance['n_generators']
    n_scenarios = instance['n_scenarios']
    T = instance['T']
    generators = instance['generators']
    demand_scenarios = instance['demand_scenarios']
    reserve_scenarios = instance['reserve_scenarios']

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    try:
        model = gp.Model('UnitCommitment')
        model.setParam("Threads", 1)
        model.Params.LogToConsole = 1
        model.Params.TimeLimit = time_limit

        # Variables
        x = [[model.addVar(vtype=GRB.BINARY, name=f'x_{g}_{t}') for t in range(T)] for g in range(n_generators)]
        y = [[model.addVar(vtype=GRB.BINARY, name=f'y_{g}_{t}') for t in range(T)] for g in range(n_generators)]
        y_bar = [[model.addVar(vtype=GRB.BINARY, name=f'y_bar_{g}_{t}') for t in range(T)] for g in range(n_generators)]
        q = [[model.addVar(lb=0, name=f'q_{g}_{t}') for t in range(T)] for g in range(n_generators)]
        p = [[[model.addVar(lb=0, name=f'p_{g}_{s}_{t}') for t in range(T)] for s in range(n_scenarios)] for g in range(n_generators)]
        p_bar = [[[model.addVar(lb=0, name=f'p_bar_{g}_{s}_{t}') for t in range(T)] for s in range(n_scenarios)] for g in range(n_generators)]

        # Constraints
        for g in range(n_generators):
            M, m, L, ell, RU, SU, RD, SD, c_g, c_f, startup_costs_K = (
                generators[g]['M'], generators[g]['m'], generators[g]['L'], generators[g]['ell'],
                generators[g]['RU'], generators[g]['SU'], generators[g]['RD'], generators[g]['SD'],
                generators[g]['c_g'], generators[g]['c_f'], generators[g]['startup_costs_K']
            )
            for t in range(T):
                # Start-up, shut-down, and commitment status linkage
                if t > 0:
                    model.addConstr(y[g][t] - y_bar[g][t] == x[g][t] - x[g][t-1], name=f'start_shutdown_link_{g}_{t}')
                else:
                    model.addConstr(y[g][t] - y_bar[g][t] == x[g][t], name=f'start_shutdown_link_init_{g}_{t}')

                # Minimum up-time and down-time constraints
                for tau in range(max(0, t-L+1), t+1):
                    model.addConstr(y[g][tau] <= x[g][t], name=f'min_up_time_{g}_{t}_{tau}')
                for tau in range(max(0, t-ell+1), t+1):
                    model.addConstr(y_bar[g][tau] <= 1 - x[g][t], name=f'min_down_time_{g}_{t}_{tau}')

                # Start-up cost computation
                for k in range(1, t+2):
                    if k <= t+1:
                        model.addConstr(q[g][t] >= startup_costs_K[k-1] * (x[g][t] - sum(x[g][tau] for tau in range(max(0, t-k), t))), name=f'startup_cost_{g}_{t}_{k}')

                # Production and available capacity constraints when generator is on or off
                for s in range(n_scenarios):
                    model.addConstr(p[g][s][t] <= p_bar[g][s][t], name=f'prod_cap_{g}_{s}_{t}')
                    model.addConstr(p[g][s][t] >= x[g][t] * m, name=f'prod_min_{g}_{s}_{t}')
                    model.addConstr(p_bar[g][s][t] <= x[g][t] * M, name=f'cap_max_{g}_{s}_{t}')

                    # Ramping constraints
                    if t > 0:
                        model.addConstr(p[g][s][t] - p[g][s][t-1] <= RU * x[g][t-1] + SU * y[g][t], name=f'ramp_up_{g}_{s}_{t}')
                        model.addConstr(p[g][s][t-1] - p[g][s][t] <= RD * x[g][t] + SD * y_bar[g][t], name=f'ramp_down_{g}_{s}_{t}')

            # Demand and reserve requirements
            for s in range(n_scenarios):
                for t in range(T):
                    model.addConstr(gp.quicksum(p[g][s][t] for g in range(n_generators)) >= demand_scenarios[s][t], name=f'demand_{s}_{t}')
                    model.addConstr(gp.quicksum(p_bar[g][s][t] for g in range(n_generators)) >= demand_scenarios[s][t] + reserve_scenarios[s][t], name=f'reserve_{s}_{t}')

        # Objective: Minimize total expected cost
        total_fixed_operating_cost = gp.quicksum(c_f * x[g][t] for g in range(n_generators) for t in range(T))
        total_startup_cost = gp.quicksum(q[g][t] for g in range(n_generators) for t in range(T))
        total_production_cost = gp.quicksum((1/n_scenarios) * c_g * p[g][s][t] for g in range(n_generators) for s in range(n_scenarios) for t in range(T))
        model.setObjective(total_fixed_operating_cost + total_startup_cost + total_production_cost, GRB.MINIMIZE)

        model.optimize()
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if logger and model.SolCount > 0:
                logger.log(model.ObjVal)
            return {
                'objective_value': model.ObjVal if model.SolCount > 0 else float('inf'),
                'x': [[int(x[g][t].X) for t in range(T)] for g in range(n_generators)],
                'y': [[int(y[g][t].X) for t in range(T)] for g in range(n_generators)],
                'y_bar': [[int(y_bar[g][t].X) for t in range(T)] for g in range(n_generators)],
                'q': [[q[g][t].X for t in range(T)] for g in range(n_generators)],
                'p': [[[p[g][s][t].X for t in range(T)] for s in range(n_scenarios)] for g in range(n_generators)],
                'p_bar': [[[p_bar[g][s][t].X for t in range(T)] for s in range(n_scenarios)] for g in range(n_generators)],
            }
        else:
            return None

    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')
        return None

def main():
    parser = argparse.ArgumentParser(description='Unit Commitment Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    if solution is not None:
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()