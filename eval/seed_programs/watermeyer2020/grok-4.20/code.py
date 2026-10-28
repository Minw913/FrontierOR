import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n_real = data['n']
    num_activities = data['num_activities']
    num_resources = data['num_resources']
    d_bar = data['d_bar']
    activities = data['activities']
    temporal_constraints = data['temporal_constraints']
    resources = data['resources']

    # Build processing times and demands
    p = [0] * num_activities
    demand = [[0] * num_resources for _ in range(num_activities)]
    for aid_str, act in activities.items():
        aid = int(aid_str)
        p[aid] = act['processing_time']
        for rid_str, dem in act.get('resource_demands', {}).items():
            rid = int(rid_str)
            demand[aid][rid] = dem

    # Build temporal constraints: list of (i, j, delta)
    temps = []
    for tc in temporal_constraints:
        i = tc['i']
        j = tc['j']
        delta = tc['delta']
        temps.append((i, j, delta))

    # Resource data
    capacities = [0] * num_resources
    period_sets = [[] for _ in range(num_resources)]
    for rid_str, res in resources.items():
        rid = int(rid_str)
        capacities[rid] = res['capacity']
        period_sets[rid] = res['periods']

    # T is d_bar; periods assumed from 1 to d_bar
    T = d_bar

    # Precompute overlap coefficients: coeff[a][r][s]
    coeff = [[[0 for _ in range(T + 1)] for _ in range(num_resources)] for _ in range(num_activities)]

    for a in range(num_activities):
        if p[a] == 0:
            continue
        for r in range(num_resources):
            periods = set(period_sets[r])
            if not periods:
                continue
            for s in range(T + 1):
                overlap = 0
                for t in range(s + 1, s + p[a] + 1):
                    if t in periods:
                        overlap += 1
                coeff[a][r][s] = overlap

    # Create model
    model = gp.Model("PARTIALLY_RENEWABLE_RCPSP")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.2

    # Start time variables
    S = {}
    for a in range(num_activities):
        ub = T if a != 0 else 0
        S[a] = model.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f"S_{a}")

    # Objective: minimize project end
    end_act = num_activities - 1
    model.setObjective(S[end_act], GRB.MINIMIZE)

    # Temporal constraints
    for i, j, delta in temps:
        model.addConstr(S[j] - S[i] >= delta, name=f"temp_{i}_{j}_{delta}")

    # Project start fixed
    model.addConstr(S[0] == 0, name="start_at_zero")

    # Resource constraints using indicator variables
    for r in range(num_resources):
        if not period_sets[r] or capacities[r] == 0:
            continue
        total_cons = gp.LinExpr()
        cap = capacities[r]
        for a in range(num_activities):
            dval = demand[a][r]
            if dval == 0:
                continue
            for s in range(T + 1):
                c = coeff[a][r][s]
                if c == 0:
                    continue
                consumption = dval * c
                if consumption == 0:
                    continue
                delta_var = model.addVar(vtype=GRB.BINARY, name=f"delta_{a}_{r}_{s}")
                model.addConstr(S[a] == s, name=f"link_{a}_{r}_{s}")
                total_cons += consumption * delta_var
        if total_cons.size() > 0:
            model.addConstr(total_cons <= cap, name=f"res_{r}")

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(int(round(obj)))

    model.optimize(callback)

    if model.SolCount > 0:
        start_times_dict = {}
        for a in range(num_activities):
            start_times_dict[str(a)] = int(round(S[a].X))
        objective = int(round(S[end_act].X))

        solution = {
            "objective_value": objective,
            "start_times": start_times_dict
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if logger:
            logger.log(objective)
    else:
        solution = {
            "objective_value": -1,
            "start_times": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()