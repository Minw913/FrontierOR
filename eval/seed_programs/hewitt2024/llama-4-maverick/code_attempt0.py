import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance_data(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def create_model(instance_data):
    num_products = instance_data['num_products']
    num_facilities = instance_data['num_facilities']
    products = instance_data['products']
    facilities = instance_data['facilities']
    product_facility_data = instance_data['product_facility_data']
    products_distributions = instance_data['products_distributions']

    model = gp.Model("Production_Planning")
    model.modelSense = GRB.MAXIMIZE

    # Create variables
    x = model.addVars([(p['id'], f['id']) for p in products for f in facilities], name='x', lb=0)
    y = model.addVars([(p['id'], f['id'], pl['level_id']) for p in products for f in facilities for pl in next(pfd for pfd in product_facility_data if pfd['product_id'] == p['id'] and pfd['facility_id'] == f['id'])['production_levels']], vtype=GRB.BINARY, name='y')
    delta = model.addVars([(p['id'], d['distribution_id']) for p in products for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions']], vtype=GRB.BINARY, name='delta')
    z = model.addVars([(p['id'], d['distribution_id'], s['scenario_id']) for p in products for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions'] for s in d['scenarios']], name='z', lb=0)
    w = model.addVars([(p['id'], d['distribution_id'], s['scenario_id']) for p in products for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions'] for s in d['scenarios']], name='w', lb=0)
    o = model.addVars([(p['id'], d['distribution_id'], s['scenario_id']) for p in products for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions'] for s in d['scenarios']], name='o', lb=0)

    # Objective function
    model.setObjective(
        gp.quicksum(
            p['sale_price'] * w[p['id'], d['distribution_id'], s['scenario_id']] + p['discounted_price'] * o[p['id'], d['distribution_id'], s['scenario_id']]
            for p in products
            for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions']
            for s in d['scenarios']
        ) * s['probability']
        - gp.quicksum(
            next(pfd for pfd in product_facility_data if pfd['product_id'] == p['id'] and pfd['facility_id'] == f['id'])['manufacturing_cost'] * x[p['id'], f['id']]
            for p in products
            for f in facilities
        ),
        GRB.MAXIMIZE
    )

    # Constraints
    for p in products:
        for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions']:
            for s in d['scenarios']:
                model.addConstr(w[p['id'], d['distribution_id'], s['scenario_id']] <= s['demand'], name=f"demand_limit_{p['id']}_{d['distribution_id']}_{s['scenario_id']}")
                model.addConstr(w[p['id'], d['distribution_id'], s['scenario_id']] + o[p['id'], d['distribution_id'], s['scenario_id']] == z[p['id'], d['distribution_id'], s['scenario_id']], name=f"inventory_balance_{p['id']}_{d['distribution_id']}_{s['scenario_id']}")

    for p in products:
        for f in facilities:
            pfd = next(pfd for pfd in product_facility_data if pfd['product_id'] == p['id'] and pfd['facility_id'] == f['id'])
            model.addConstr(x[p['id'], f['id']] >= gp.quicksum(pl['lower_bound'] * y[p['id'], f['id'], pl['level_id']] for pl in pfd['production_levels']), name=f"production_lower_bound_{p['id']}_{f['id']}")
            model.addConstr(x[p['id'], f['id']] <= gp.quicksum(pl['upper_bound'] * y[p['id'], f['id'], pl['level_id']] for pl in pfd['production_levels']), name=f"production_upper_bound_{p['id']}_{f['id']}")
            model.addConstr(gp.quicksum(y[p['id'], f['id'], pl['level_id']] for pl in pfd['production_levels']) == 1, name=f"production_level_selection_{p['id']}_{f['id']}")

    for p in products:
        model.addConstr(gp.quicksum(delta[p['id'], d['distribution_id']] for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions']) == 1, name=f"distribution_selection_{p['id']}")

    for p in products:
        for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions']:
            if d['type'] == 'yield_and_demand':
                for f in facilities:
                    pfd = next(pfd for pfd in product_facility_data if pfd['product_id'] == p['id'] and pfd['facility_id'] == f['id'])
                    level_id = d['level_combination'][f['id'] - 1]
                    model.addConstr(y[p['id'], f['id'], level_id] >= delta[p['id'], d['distribution_id']], name=f"distribution_level_match_{p['id']}_{d['distribution_id']}_{f['id']}")

    for f in facilities:
        model.addConstr(gp.quicksum(x[p['id'], f['id']] for p in products) <= f['capacity'], name=f"facility_capacity_{f['id']}")

    for p in products:
        for d in next(pd for pd in products_distributions if pd['product_id'] == p['id'])['distributions']:
            for s in d['scenarios']:
                if d['type'] == 'yield_and_demand':
                    model.addConstr(z[p['id'], d['distribution_id'], s['scenario_id']] == gp.quicksum(s['yields'][str(f['id'])] * x[p['id'], f['id']] for f in facilities), name=f"available_inventory_{p['id']}_{d['distribution_id']}_{s['scenario_id']}")

    return model, x, y, delta, z, w, o

def write_solution(model, x, y, delta, z, w, o, solution_path):
    solution = {
        'objective_value': model.ObjVal,
        'solution': {
            'x': {f"{p}_{f}": x[p, f].X for p, f in x},
            'y': {f"{p}_{f}_{l}": y[p, f, l].X for p, f, l in y},
            'delta': {f"{p}_{d}": delta[p, d].X for p, d in delta},
            'z': {f"{p}_{d}_{s}": z[p, d, s].X for p, d, s in z},
            'w': {f"{p}_{d}_{s}": w[p, d, s].X for p, d, s in w},
            'o': {f"{p}_{d}_{s}": o[p, d, s].X for p, d, s in o}
        }
    }
    with open(solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Production Planning Optimization')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    instance_data = read_instance_data(args.instance_path)
    model, x, y, delta, z, w, o = create_model(instance_data)

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 1

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        write_solution(model, x, y, delta, z, w, o, args.solution_path)

if __name__ == "__main__":
    main()