import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def build_model(instance):
    num_demand_sites = instance['num_demand_sites']
    num_production_sites = instance['num_production_sites']
    depot_id = num_production_sites
    num_scenarios = instance['num_scenarios']
    num_time_periods = instance['num_time_periods']
    num_module_sizes = instance['num_module_sizes']
    units_per_module_size = instance['units_per_module_size']
    u_max = instance['u_max']
    scenario_probabilities = instance['scenario_probabilities']
    demands = [scenario['demands'] for scenario in instance['scenarios']]
    production_costs = instance['production_costs']
    relocation_costs = instance['relocation_costs']
    penalty_costs = instance['penalty_costs']
    module_sizes = instance['module_sizes']
    module_capacities = [size['capacity'] for size in module_sizes]
    module_capital_costs = [size['capital_cost_per_unit'] for size in module_sizes]

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # First-stage decision: number of units of each module size to purchase
    y = model.addVars(num_module_sizes, units_per_module_size, vtype=GRB.BINARY, name='y')

    # Second-stage decisions
    x = model.addVars(num_demand_sites, num_production_sites, num_time_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='x', lb=0, ub=1)
    q = model.addVars(num_demand_sites, num_time_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='q', lb=0, ub=1)
    v = model.addVars(num_production_sites+1, num_module_sizes, num_time_periods, num_scenarios, vtype=GRB.INTEGER, name='v', lb=0)
    w = model.addVars(num_production_sites+1, num_production_sites+1, num_module_sizes, num_time_periods, num_scenarios, vtype=GRB.INTEGER, name='w', lb=0)

    # Objective: minimize total expected cost
    capital_cost = gp.quicksum(module_capital_costs[m] * y.sum(m, '*') for m in range(num_module_sizes))
    expected_operating_cost = gp.quicksum(scenario_probabilities[s] * (
        gp.quicksum(production_costs[i][j] * demands[s][i][t] * x[i, j, t, s] for i in range(num_demand_sites) for j in range(num_production_sites) for t in range(num_time_periods)) +
        gp.quicksum(relocation_costs[j1][j2][m] * w[j1, j2, m, t, s] for j1 in range(num_production_sites) for j2 in range(num_production_sites) if j1 != j2 for m in range(num_module_sizes) for t in range(num_time_periods)) +
        gp.quicksum(relocation_costs[j][depot_id][m] * w[j, depot_id, m, t, s] for j in range(num_production_sites) for m in range(num_module_sizes) for t in range(num_time_periods)) +
        gp.quicksum(relocation_costs[depot_id][j][m] * w[depot_id, j, m, t, s] for j in range(num_production_sites) for m in range(num_module_sizes) for t in range(num_time_periods)) +
        gp.quicksum(penalty_costs[i][t] * demands[s][i][t] * q[i, t, s] for i in range(num_demand_sites) for t in range(num_time_periods))
    ) for s in range(num_scenarios))
    model.setObjective(capital_cost + expected_operating_cost, GRB.MINIMIZE)

    # Constraints
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for i in range(num_demand_sites):
                model.addConstr(gp.quicksum(x[i, j, t, s] for j in range(num_production_sites)) + q[i, t, s] == 1, name=f'demand_coverage_{i}_{t}_{s}')
            for j in range(num_production_sites):
                production = gp.quicksum(demands[s][i][t] * x[i, j, t, s] for i in range(num_demand_sites))
                capacity = gp.quicksum(module_capacities[m] * v[j, m, t, s] for m in range(num_module_sizes))
                model.addConstr(production <= capacity, name=f'production_capacity_{j}_{t}_{s}')
                model.addConstr(production <= u_max, name=f'global_production_limit_{j}_{t}_{s}')
            for m in range(num_module_sizes):
                for j in range(num_production_sites+1):
                    inflow = gp.quicksum(w[jp, j, m, t, s] for jp in range(num_production_sites+1) if jp != j)
                    outflow = gp.quicksum(w[j, jp, m, t, s] for jp in range(num_production_sites+1) if jp != j)
                    if t == 0:
                        if j == depot_id:  
                            model.addConstr(v[j, m, t, s] == y.sum(m, '*') - outflow, name=f'depot_balance_{m}_{t}_{s}')
                        else:
                            model.addConstr(v[j, m, t, s] == inflow, name=f'production_site_balance_{j}_{m}_{t}_{s}')
                    else:
                        model.addConstr(v[j, m, t, s] == v[j, m, t-1, s] + inflow - outflow, name=f'site_balance_{j}_{m}_{t}_{s}')

    return model, y, x, q, v, w

def extract_solution(model, y, x, q, v, w, instance):
    num_module_sizes = instance['num_module_sizes']
    units_per_module_size = instance['units_per_module_size']
    num_demand_sites = instance['num_demand_sites']
    num_production_sites = instance['num_production_sites']
    num_time_periods = instance['num_time_periods']
    num_scenarios = instance['num_scenarios']

    solution = {
        'objective_value': model.ObjVal,
        'model_variables': {},
        'y': {f'y_{m}_{k}': y[m, k].X for m in range(num_module_sizes) for k in range(units_per_module_size)},
        'model_variable_name_patterns': {
            "y[m,k]": f"m=0..{num_module_sizes-1}, k=0..{units_per_module_size-1}",
            "x[i,j,t,s]": f"i=0..{num_demand_sites-1}, j=0..{num_production_sites-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
            "q[i,t,s]": f"i=0..{num_demand_sites-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
            "v[j,m,t,s]": f"j=0..{num_production_sites-1}, m=0..{num_module_sizes-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
            "v_depot[m,t,s]": f"m=0..{num_module_sizes-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
            "w[j,jp,m,t,s]": f"j=0..{num_production_sites-1}, jp=0..{num_production_sites-1}, m=0..{num_module_sizes-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
            "w_fd[j,m,t,s]": f"j=0..{num_production_sites-1}, m=0..{num_module_sizes-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
            "w_td[j,m,t,s]": f"j=0..{num_production_sites-1}, m=0..{num_module_sizes-1}, t=0..{num_time_periods-1}, s=0..{num_scenarios-1}",
        }
    }

    for i in range(num_demand_sites):
        for j in range(num_production_sites):
            for t in range(num_time_periods):
                for s in range(num_scenarios):
                    if x[i, j, t, s].X > 1e-6:
                        solution['model_variables'][f'x[{i},{j},{t},{s}]'] = x[i, j, t, s].X
    for i in range(num_demand_sites):
        for t in range(num_time_periods):
            for s in range(num_scenarios):
                if q[i, t, s].X > 1e-6:
                    solution['model_variables'][f'q[{i},{t},{s}]'] = q[i, t, s].X
    for j in range(num_production_sites):
        for m in range(num_module_sizes):
            for t in range(num_time_periods):
                for s in range(num_scenarios):
                    if v[j, m, t, s].X > 0.5:
                        solution['model_variables'][f'v[{j},{m},{t},{s}]'] = v[j, m, t, s].X
    for m in range(num_module_sizes):
        for t in range(num_time_periods):
            for s in range(num_scenarios):
                if v[num_production_sites, m, t, s].X > 0.5:
                    solution['model_variables'][f'v_depot[{m},{t},{s}]'] = v[num_production_sites, m, t, s].X
    for j1 in range(num_production_sites+1):
        for j2 in range(num_production_sites+1):
            if j1 == j2:
                continue
            for m in range(num_module_sizes):
                for t in range(num_time_periods):
                    for s in range(num_scenarios):
                        if w[j1, j2, m, t, s].X > 0.5:
                            if j1 == num_production_sites:
                                solution['model_variables'][f'w_fd[{j2},{m},{t},{s}]'] = w[j1, j2, m, t, s].X
                            elif j2 == num_production_sites:
                                solution['model_variables'][f'w_td[{j1]},{m},{t},{s}]'] = w[j1, j2, m, t, s].X
                            else:
                                solution['model_variables'][f'w[{j1},{j2},{m},{t},{s}]'] = w[j1, j2, m, t, s].X

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, y, x, q, v, w = build_model(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(model, y, x, q, v, w, instance)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=2)
        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        print(f"Optimization ended with status {model.status}")

if __name__ == "__main__":
    main()