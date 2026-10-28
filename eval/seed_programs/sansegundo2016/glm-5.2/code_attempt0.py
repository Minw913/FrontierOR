import json
import argparse
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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    adj_raw = graph['adjacency_list']
    
    # Vertices are expected to be integers according to the output schema
    V = sorted([int(v) for v in adj_raw.keys()])
    
    adj_set = {}
    for v in V:
        adj_set[v] = set([int(u) for u in adj_raw[str(v)]])

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.start()

    model = gp.Model(env=env)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)

    x = model.addVars(V, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[i] for i in V), GRB.MAXIMIZE)

    # Add constraints for non-edges: x_i + x_j <= 1
    for i in range(len(V)):
        u = V[i]
        for j in range(i + 1, len(V)):
            v = V[j]
            if v not in adj_set[u]:
                model.addConstr(x[u] + x[v] <= 1)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = model.cbGetSolution(model._vars)
            selected = [i for i in V if sol[i] > 0.5]
            if logger:
                logger.log_solution(int(obj), selected)

    model._vars = x
    model.optimize(callback)

    obj_val = 0
    selected = []
    
    if model.SolCount > 0:
        obj_val = int(round(model.ObjVal))
        for i in V:
            if x[i].X > 0.5:
                selected.append(i)
    
    # Fallback if no solution was found (e.g., empty graph)
    if obj_val == 0 and len(V) > 0:
        obj_val = 1
        selected = [V[0]]
        if logger:
            logger.log_solution(1, [V[0]])

    solution = {
        "objective_value": obj_val,
        "clique_vertices": selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()