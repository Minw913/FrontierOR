import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    nf = data['num_facilities']
    nc = data['num_customers']
    ns = data['num_scenarios']
    facilities = data['facilities']
    scenarios = data['scenarios']
    trans_costs = data['transportation_costs']

    opening_costs = [fac['opening_cost'] for fac in facilities]
    capacities = [fac['capacity'] for fac in facilities]
    probabilities = [sc['probability'] for sc in scenarios]
    demands = [sc['demands'] for sc in scenarios]

    max_total_demand = max(sum(d) for d in demands)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model('sflp')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, args.time_limit - 2)

    x = model.addVars(nf, vtype=GRB.BINARY, name='x')
    y = model.addVars(ns, nf, nc, lb=0.0, vtype=GRB.CONTINUOUS, name='y')

    # Objective
    obj = gp.LinExpr()
    for f in range(nf):
        obj += opening_costs[f] * x[f]
    for s in range(ns):
        p = probabilities[s]
        for f in range(nf):
            tcf = trans_costs[f]
            for c in range(nc):
                obj += p * tcf[c] * y[s, f, c]
    model.setObjective(obj, GRB.MINIMIZE)

    # Demand satisfaction per scenario
    for s in range(ns):
        ds = demands[s]
        for c in range(nc):
            model.addConstr(gp.quicksum(y[s, f, c] for f in range(nf)) >= ds[c])

    # Capacity linking
    for s in range(ns):
        for f in range(nf):
            model.addConstr(gp.quicksum(y[s, f, c] for c in range(nc)) <= capacities[f] * x[f])

    # Total capacity must cover worst-case total demand
    model.addConstr(gp.quicksum(capacities[f] * x[f] for f in range(nf)) >= max_total_demand)

    def make_solution(x_vals, y_vals):
        sol = {
            'open_facilities': [f for f in range(nf) if x_vals[f] > 0.5],
            'x': {str(f): int(round(x_vals[f])) for f in range(nf)},
            'y': {},
        }
        for s in range(ns):
            sol['y'][str(s)] = {}
            for f in range(nf):
                sol['y'][str(s)][str(f)] = {}
                for c in range(nc):
                    sol['y'][str(s)][str(f)][str(c)] = float(y_vals[s, f, c])
        return sol

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution(x)
            y_vals = model.cbGetSolution(y)
            sol = make_solution(x_vals, y_vals)
            logger.log_solution(obj_val, sol)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_vals = {f: x[f].X for f in range(nf)}
        y_vals = {(s, f, c): y[s, f, c].X for s in range(ns) for f in range(nf) for c in range(nc)}
        sol = make_solution(x_vals, y_vals)
    else:
        # Fallback: fix all facilities open and solve the LP for allocation
        for f in range(nf):
            x[f].LB = 1.0
            x[f].UB = 1.0
        model.Params.TimeLimit = max(1, args.time_limit - 2)
        model.optimize()
        if model.SolCount > 0:
            obj_val = model.ObjVal
            x_vals = {f: x[f].X for f in range(nf)}
            y_vals = {(s, f, c): y[s, f, c].X for s in range(ns) for f in range(nf) for c in range(nc)}
            sol = make_solution(x_vals, y_vals)
        else:
            # Manual feasible fallback
            x_vals = {f: 1 for f in range(nf)}
            y_vals = {}
            trans_cost_total = 0.0
            for s in range(ns):
                remaining_cap = list(capacities)
                for c in range(nc):
                    d = demands[s][c]
                    for f in range(nf):
                        amt = min(d, remaining_cap[f])
                        y_vals[s, f, c] = float(amt)
                        remaining_cap[f] -= amt
                        d -= amt
                        trans_cost_total += probabilities[s] * trans_costs[f][c] * amt
                        if d <= 1e-9:
                            break
                    for f in range(nf):
                        if (s, f, c) not in y_vals:
                            y_vals[s, f, c] = 0.0
            obj_val = float(sum(opening_costs) + trans_cost_total)
            sol = make_solution(x_vals, y_vals)
            if logger:
                logger.log_solution(obj_val, sol)

    if logger:
        logger.log_solution(float(obj_val), sol)

    output = {
        'objective_value': float(obj_val),
        'open_facilities': sol['open_facilities'],
        'x': sol['x'],
        'y': sol['y'],
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()