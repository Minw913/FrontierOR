import argparse
import json
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
        instance = json.load(f)

    num_fac = instance['num_facilities']
    num_cus = instance['num_customers']
    num_scen = instance['num_scenarios']
    facilities = instance['facilities']
    customers = instance['customers']
    trans_costs = instance['transportation_costs']
    scenarios = instance['scenarios']

    open_cost = [fac['opening_cost'] for fac in facilities]
    capacity = [fac['capacity'] for fac in facilities]
    probs = [scen['probability'] for scen in scenarios]
    demands = [scen['demands'] for scen in scenarios]

    # Precompute total demand per scenario for constraint
    total_dem_scen = [sum(demands[s]) for s in range(num_scen)]

    model = gp.Model("StochasticFacilityLocation")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)  # Allow 1% gap for speed-quality tradeoff
    model.setParam('LogToConsole', 0)

    # Variables
    x = model.addVars(num_fac, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_scen, num_fac, num_cus, vtype=GRB.CONTINUOUS, lb=0, name="y")

    # Objective: opening costs + expected transportation costs
    obj_open = gp.quicksum(open_cost[i] * x[i] for i in range(num_fac))
    obj_trans = gp.quicksum(
        probs[s] * trans_costs[i][j] * y[s, i, j]
        for s in range(num_scen)
        for i in range(num_fac)
        for j in range(num_cus)
    )
    model.setObjective(obj_open + obj_trans, GRB.MINIMIZE)

    # Constraints
    # 1. Demand satisfaction
    for s in range(num_scen):
        for j in range(num_cus):
            model.addConstr(
                gp.quicksum(y[s, i, j] for i in range(num_fac)) >= demands[s][j],
                name=f"demand_s{s}_c{j}"
            )

    # 2. Capacity constraints
    for s in range(num_scen):
        for i in range(num_fac):
            model.addConstr(
                gp.quicksum(y[s, i, j] for j in range(num_cus)) <= capacity[i] * x[i],
                name=f"cap_s{s}_f{i}"
            )

    # 3. Sufficient total capacity for worst-case scenario
    max_total_dem = max(total_dem_scen)
    model.addConstr(
        gp.quicksum(capacity[i] * x[i] for i in range(num_fac)) >= max_total_dem,
        name="total_capacity"
    )

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        open_facilities = [i for i in range(num_fac) if x[i].X > 0.5]
        
        x_dict = {str(i): 1 if x[i].X > 0.5 else 0 for i in range(num_fac)}
        
        y_dict = {}
        for s in range(num_scen):
            s_key = str(s)
            y_dict[s_key] = {}
            for i in range(num_fac):
                i_key = str(i)
                y_dict[s_key][i_key] = {}
                for j in range(num_cus):
                    j_key = str(j)
                    y_dict[s_key][i_key][j_key] = float(y[s, i, j].X)
        
        solution = {
            "objective_value": float(objective_value),
            "open_facilities": open_facilities,
            "x": x_dict,
            "y": y_dict
        }
    else:
        # Infeasible - should not happen with this formulation
        solution = {
            "objective_value": 0.0,
            "open_facilities": [],
            "x": {},
            "y": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()