import gurobipy as gp
from gurobipy import GRB
import json
import argparse
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
    
    num_nodes = data['num_nodes']
    num_edges = data['num_edges']
    F = data['bisection_capacity_F']
    total_node_weight = data['total_node_weight']
    node_weights = data['node_weights']
    edges = data['edges']
    edge_weights = data['edge_weights']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    
    total_weight_expr = gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes))
    model.addConstr(total_weight_expr >= total_node_weight - F, "weight_lb")
    model.addConstr(total_weight_expr <= F, "weight_ub")
    
    obj = gp.QuadExpr()
    for idx in range(num_edges):
        u, v = edges[idx]
        cost = edge_weights[idx]
        y = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y_{u}_{v}")
        model.addConstr(y >= x[u] - x[v], f"edge_{idx}_diff1")
        model.addConstr(y >= x[v] - x[u], f"edge_{idx}_diff2")
        model.addConstr(y <= x[u] + x[v], f"edge_{idx}_ub1")
        model.addConstr(y <= 2 - x[u] - x[v], f"edge_{idx}_ub2")
        obj += cost * y
        
    model.setObjective(obj, GRB.MINIMIZE)
    
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
        
    if logger:
        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
        
    if model.SolCount > 0:
        x_vals = model.getAttr('X', x)
        ref_val = x_vals[0]
        group0 = [i for i in range(num_nodes) if x_vals[i] == ref_val]
        group1 = [i for i in range(num_nodes) if x_vals[i] != ref_val]
    else:
        group0 = [0]
        group1 = [i for i in range(1, num_nodes)]
        
    cut_sum = 0.0
    for idx in range(num_edges):
        u, v = edges[idx]
        if (u in group0 and v in group1) or (u in group1 and v in group0):
            cut_sum += edge_weights[idx]
            
    solution = {
        "objective_value": cut_sum,
        "partition_S": group0,
        "partition_complement": group1
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()