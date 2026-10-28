import gurobipy as gp
from gurobipy import GRB
import json
import argparse
from solution_logger import SolutionLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve driver duty set partitioning problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Optional path to log intermediate solutions.')
    
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    params = data['parameters']
    M = params['num_trips_M']
    N = params['num_duties_N']
    trips = data['trips']
    duties = data['duties']
    costs = data['costs']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    
    x = model.addVars(N, vtype=GRB.BINARY, name="x")
    
    model.setObjective(gp.quicksum(costs[j] * x[j] for j in range(N)), GRB.MINIMIZE)
    
    trip_duties = [[] for _ in range(M)]
    for j in range(N):
        for trip_id in duties[j]:
            trip_duties[trip_id].append(j)
    
    for i in range(M):
        model.addConstr(gp.quicksum(x[j] for j in trip_duties[i]) == 1, name=f"trip_{i}")
    
    model.setParam('TimeLimit', args.time_limit)
    
    if logger:
        def callback(model_param, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model_param.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        objective_value = model.ObjVal
        selected_duties = [j for j in range(N) if x[j].X > 0.5]
    else:
        objective_value = 0.0
        selected_duties = []
    
    solution = {
        "objective_value": objective_value,
        "selected_duties": selected_duties
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)