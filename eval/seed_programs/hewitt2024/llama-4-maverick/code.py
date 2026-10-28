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
    model.setParam("Threads", 1)
    model.modelSense = GRB.MAXIMIZE

    product_ids = [p['id'] for p in products]
    facility_ids = [f['id'] for f in facilities]

    x = model.addVars([(prod_id, facil_id) for prod_id in product_ids for facil_id in facility_ids], name='x', lb=0)
    y = model.addVars([(prod_id, facil_id, pl['level_id']) for prod_id in product_ids for facil_id in facility_ids for pl in next((pfd['production_levels'] for pfd in product_facility_data if pfd['product_id'] == prod_id and pfd['facility_id'] == facil_id), [])], vtype=GRB.BINARY, name='y')
    delta = model.addVars([(prod_id, d['distribution_id']) for prod_id in product_ids for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), [])], vtype=GRB.BINARY, name='delta')
    z = model.addVars([(prod_id, d['distribution_id'], s['scenario_id']) for prod_id in product_ids for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []) for s in d['scenarios']], name='z', lb=0)
    w = model.addVars([(prod_id, d['distribution_id'], s['scenario_id']) for prod_id in product_ids for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []) for s in d['scenarios']], name='w', lb=0)
    o = model.addVars([(prod_id, d['distribution_id'], s['scenario_id']) for prod_id in product_ids for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []) for s in d['scenarios']], name='o', lb=0)

    # Objective function
    obj = gp.QuadExpr()
    for prod_id in product_ids:
        product = next(p for p in products if p['id'] == prod_id)
        for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []):
            for s in d['scenarios']:
                obj += s['probability'] * (product['sale_price'] * w[prod_id, d['distribution_id'], s['scenario_id']] + product['discounted_price'] * o[prod_id, d['distribution_id'], s['scenario_id']])
    obj -= gp.quicksum(next(pfd['manufacturing_cost'] for pfd in product_facility_data if pfd['product_id'] == prod_id and pfd['facility_id'] == facil_id) * x[prod_id, facil_id] for prod_id in product_ids for facil_id in facility_ids)
    model.setObjective(obj)

    # Constraints
    for prod_id in product_ids:
        for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []):
            for s in d['scenarios']:
                model.addConstr(w[prod_id, d['distribution_id'], s['scenario_id']] <= s['demand'])
                model.addConstr(w[prod_id, d['distribution_id'], s['scenario_id']] + o[prod_id, d['distribution_id'], s['scenario_id']] == z[prod_id, d['distribution_id'], s['scenario_id']])

    for prod_id in product_ids:
        for facil_id in facility_ids:
            pfd = next((pfd for pfd in product_facility_data if pfd['product_id'] == prod_id and pfd['facility_id'] == facil_id), None)
            if pfd:
                model.addConstr(x[prod_id, facil_id] >= gp.quicksum(pl['lower_bound'] * y[prod_id, facil_id, pl['level_id']] for pl in pfd['production_levels']))
                model.addConstr(x[prod_id, facil_id] <= gp.quicksum(pl['upper_bound'] * y[prod_id, facil_id, pl['level_id']] for pl in pfd['production_levels']))
                model.addConstr(gp.quicksum(y[prod_id, facil_id, pl['level_id']] for pl in pfd['production_levels']) == 1)

    for prod_id in product_ids:
        model.addConstr(gp.quicksum(delta[prod_id, d['distribution_id']] for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), [])) == 1)

    for prod_id in product_ids:
        for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []):
            if d['type'] == 'yield_and_demand' and d['level_combination'] is not None:
                for facil_id in facility_ids:
                    pfd = next((pfd for pfd in product_facility_data if pfd['product_id'] == prod_id and pfd['facility_id'] == facil_id), None)
                    if pfd:
                        level_id = d['level_combination'][facil_id - 1]
                        model.addConstr(y[prod_id, facil_id, level_id] >= delta[prod_id, d['distribution_id']])

    for facil_id in facility_ids:
        facility = next(f for f in facilities if f['id'] == facil_id)
        model.addConstr(gp.quicksum(x[prod_id, facil_id] for prod_id in product_ids) <= facility['capacity'])

    for prod_id in product_ids:
        for d in next((pd['distributions'] for pd in products_distributions if pd['product_id'] == prod_id), []):
            for s in d['scenarios']:
                if d['type'] == 'yield_and_demand':
                    model.addConstr(z[prod_id, d['distribution_id'], s['scenario_id']] == gp.quicksum(s['yields'][str(facil_id)] * x[prod_id, facil_id] for facil_id in facility_ids))

    return model, x, y, delta, z, w, o

def write_solution(model, x, y, delta, z, w, o, solution_path):
    solution = {
        'objective_value': model.ObjVal if model.status == GRB.OPTIMAL else None,
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

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.cbGet(GRB.Callback.MIPSOL_OBJ))

    model.optimize(callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        write_solution(model, x, y, delta, z, w, o, args.solution_path)

if __name__ == "__main__":
    main()