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
        instance = json.load(f)

    num_trips = instance['parameters']['num_trips_M']
    num_duties = instance['parameters']['num_duties_N']
    duties = instance['duties']
    costs = instance['costs']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("BusDutyCovering")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)

    x = model.addVars(num_duties, vtype=GRB.BINARY, name="x")

    for t in range(num_trips):
        covering_duties = [d for d in range(num_duties) if t in duties[d]]
        model.addConstr(gp.quicksum(x[d] for d in covering_duties) == 1, name=f"cover_{t}")

    model.setObjective(gp.quicksum(costs[d] * x[d] for d in range(num_duties)), GRB.MINIMIZE)

    # Callback to log improving solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    start_time = time.time()
    model.optimize(callback)

    selected = [d for d in range(num_duties) if x[d].X > 0.5]
    objective = float(model.ObjVal)

    solution = {
        "objective_value": objective,
        "selected_duties": selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()