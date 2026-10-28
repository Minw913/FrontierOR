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

    n = instance['num_vertices']
    edges = instance['edges']
    adj = instance.get('adjacency_list', [[] for _ in range(n)])

    # Build adjacency matrix for quick lookup
    adj_set = [set(adj[i]) for i in range(n)]

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Gurobi model for Maximum Clique
    model = gp.Model("MaxClique")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.MIPGap, 0.0)
    model.setParam(GRB.Param.LogToConsole, 0)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Edge constraints: for every pair of non-adjacent vertices, at most one can be selected
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj_set[i]:
                model.addConstr(x[i] + x[j] <= 1, f"no_edge_{i}_{j}")

    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)

    # Callback to log improving solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if logger:
                logger.log(obj)

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        clique = [i for i in range(n) if x[i].X > 0.5]
        objective = len(clique)
    else:
        # Fallback: return empty solution
        clique = []
        objective = 0

    # Final log
    if logger and objective > 0:
        logger.log(objective)

    solution = {
        "objective_value": int(objective),
        "clique_vertices": [int(v) for v in sorted(clique)]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path} with objective {objective}")

if __name__ == "__main__":
    main()