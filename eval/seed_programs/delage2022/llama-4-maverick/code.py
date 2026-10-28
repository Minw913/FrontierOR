import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_instance(instance):
    problem_type = instance['problem_type']
    machine_length = instance['machine_length']
    num_products = instance['num_products']
    num_patterns = instance['num_patterns']
    cost_setup_pattern = instance['cost_setup_pattern']
    cost_repetition = instance['cost_repetition']
    M = instance['M']
    products = instance['products']
    product_ids = [product['id'] for product in products]
    product_demands = {product['id']: product['demand'] for product in products}
    product_lengths = {product['id']: product['length'] for product in products}
    product_N_i = {product['id']: product['N_i'] for product in products}

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Variables
    used = model.addVars(num_patterns, vtype=GRB.BINARY, name='used')
    repetitions = model.addVars(num_patterns, vtype=GRB.INTEGER, name='repetitions')
    pieces = model.addVars(num_patterns, product_ids, vtype=GRB.INTEGER, name='pieces')
    if problem_type != 'closed_ended_cscp':
        extra_pieces = model.addVars(product_ids, vtype=GRB.INTEGER, name='extra_pieces', lb=0)
    else:
        extra_pieces = None
    if problem_type == 'cscp_with_knife_dependent_setups':
        knife_count = model.addVars(num_patterns, vtype=GRB.INTEGER, name='knife_count')
    else:
        knife_count = None

    # Constraints
    for p in range(num_patterns):
        model.addConstr(gp.quicksum(product_lengths[i] * pieces[p, i] for i in product_ids) <= machine_length * used[p], name=f'pattern_length_{p}')
        for i in product_ids:
            model.addConstr(pieces[p, i] <= product_N_i[i] * used[p], name=f'piece_limit_{p}_{i}')
        model.addConstr(repetitions[p] <= M * used[p], name=f'repetition_limit_{p}')

    for i in product_ids:
        total_production = gp.quicksum(pieces[p, i] * repetitions[p] for p in range(num_patterns))
        if problem_type != 'closed_ended_cscp':
            model.addConstr(total_production + extra_pieces[i] >= product_demands[i], name=f'demand_{i}')
        else:
            model.addConstr(total_production >= product_demands[i], name=f'demand_{i}')

    if problem_type != 'closed_ended_cscp':
        model.addConstr(gp.quicksum(extra_pieces[i] for i in product_ids) <= gp.quicksum(repetitions[p] for p in range(num_patterns)), name='extra_pieces_limit')

    for p in range(num_patterns - 1):
        model.addConstr(repetitions[p] >= repetitions[p + 1], name=f'repetition_order_{p}')

    if problem_type == 'cscp_with_knife_dependent_setups':
        for p in range(num_patterns):
            model.addConstr(knife_count[p] == gp.quicksum(pieces[p, i] for i in product_ids), name=f'knife_count_{p}')

    # Objective
    objective = gp.quicksum(cost_setup_pattern * used[p] for p in range(num_patterns)) + gp.quicksum(cost_repetition * repetitions[p] for p in range(num_patterns))
    if problem_type == 'cscp_with_knife_dependent_setups':
        cost_knife_setup = instance['cost_knife_setup']
        objective += gp.quicksum(cost_knife_setup * knife_count[p] for p in range(num_patterns))
    model.setObjective(objective, GRB.MINIMIZE)

    return model, used, repetitions, pieces, extra_pieces, knife_count

def construct_solution(instance, used, repetitions, pieces, extra_pieces=None, knife_count=None):
    problem_type = instance['problem_type']
    products = instance['products']
    product_ids = [product['id'] for product in products]

    solution = {}
    solution['patterns'] = {}
    for p in range(instance['num_patterns']):
        pattern_solution = {
            'used': used[p].X,
            'repetitions': int(repetitions[p].X),
            'pieces': {i: int(pieces[p, i].X) for i in product_ids}
        }
        if problem_type == 'cscp_with_knife_dependent_setups' and knife_count is not None:
            pattern_solution['knife_count'] = int(knife_count[p].X)
        solution['patterns'][str(p)] = pattern_solution

    if extra_pieces is not None:
        solution['extra_pieces'] = {i: int(extra_pieces[i].X) for i in product_ids}
    else:
        solution['extra_pieces'] = {i: 0 for i in product_ids}

    solution['production_quantities'] = {}
    for p in range(instance['num_patterns']):
        for i in product_ids:
            key = f'product_{i}_pattern_{p}'
            solution['production_quantities'][key] = int(pieces[p, i].X * repetitions[p].X)

    solution['objective_value'] = instance['cost_setup_pattern'] * sum(used[p].X for p in range(instance['num_patterns'])) + instance['cost_repetition'] * sum(repetitions[p].X for p in range(instance['num_patterns']))
    if problem_type == 'cscp_with_knife_dependent_setups' and knife_count is not None:
        solution['objective_value'] += instance['cost_knife_setup'] * sum(knife_count[p].X for p in range(instance['num_patterns']))

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    model, used, repetitions, pieces, extra_pieces, knife_count = solve_instance(instance)

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = construct_solution(instance, used, repetitions, pieces, extra_pieces, knife_count)
        solution['objective_value'] = model.ObjVal
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()