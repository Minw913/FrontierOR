import argparse
import json
import random
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_nodes = data['num_nodes']
    num_edges = data['num_edges']
    F = data['bisection_capacity_F']
    W = data['total_node_weight']
    node_weights = data['node_weights']
    edges = data['edges']
    edge_weights = data['edge_weights']

    def calculate_objective(S):
        s_set = set(S)
        obj = 0.0
        for j, (u, v) in enumerate(edges):
            if (u in s_set) != (v in s_set):
                obj += edge_weights[j]
        return obj

    best_obj = None
    best_S = None
    best_comp = None

    # Simple Greedy Heuristic for warm start
    random.seed(0)
    nodes = list(range(1, num_nodes))
    random.shuffle(nodes)

    S_list = [0]
    s_W = node_weights[0]
    comp_list = []
    c_W = 0

    for i in nodes:
        if s_W <= c_W:
            if s_W + node_weights[i] <= F:
                S_list.append(i)
                s_W += node_weights[i]
            else:
                comp_list.append(i)
                c_W += node_weights[i]
        else:
            if c_W + node_weights[i] <= F:
                comp_list.append(i)
                c_W += node_weights[i]
            else:
                S_list.append(i)
                s_W += node_weights[i]

    if s_W <= F and c_W <= F:
        best_obj = calculate_objective(S_list)
        best_S = S_list
        best_comp = comp_list
        if logger:
            logger.log_solution(best_obj, {
                "objective_value": float(best_obj),
                "partition_S": best_S,
                "partition_complement": best_comp
            })

    # MIP Model
    model = gp.Model("bisection")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_edges, vtype=GRB.BINARY, name="y")

    # Node 0 must be in partition_S
    model.addConstr(x[0] == 1, name="fix_node_0")

    # Weight constraints
    model.addConstr(gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes)) <= F, name="upper_weight")
    model.addConstr(gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes)) >= W - F, name="lower_weight")

    # Edge cut constraints
    for j, (u, v) in enumerate(edges):
        model.addConstr(y[j] >= x[u] - x[v], name=f"edge_cut_pos_{j}")
        model.addConstr(y[j] >= x[v] - x[u], name=f"edge_cut_neg_{j}")

    model.setObjective(gp.quicksum(edge_weights[j] * y[j] for j in range(num_edges)), GRB.MINIMIZE)

    # Set initial solution if found
    if best_S is not None:
        for i in range(num_nodes):
            x[i].start = 1.0 if i in best_S else 0.0
        for j, (u, v) in enumerate(edges):
            y[j].start = 1.0 if (u in best_S) != (v in best_S) else 0.0

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution(x)
            
            S = [i for i in range(num_nodes) if x_vals[i] > 0.5]
            comp = [i for i in range(num_nodes) if x_vals[i] < 0.5]
            
            solution_dict = {
                "objective_value": float(obj),
                "partition_S": S,
                "partition_complement": comp
            }
            if logger:
                logger.log_solution(float(obj), solution_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        x_vals = model.getAttr('X', x)
        S = [i for i in range(num_nodes) if x_vals[i] > 0.5]
        comp = [i for i in range(num_nodes) if x_vals[i] < 0.5]
        obj = model.ObjVal
        
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_S = S
            best_comp = comp

    # Absolute fallback if no solution was found at all
    if best_obj is None:
        best_obj = 0.0
        best_S = [0]
        best_comp = [i for i in range(1, num_nodes)]

    output = {
        "objective_value": float(best_obj),
        "partition_S": best_S,
        "partition_complement": best_comp
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()