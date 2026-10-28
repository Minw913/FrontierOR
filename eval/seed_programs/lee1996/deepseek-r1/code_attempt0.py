import gurobipy as gp
from gurobipy import GRB
import json
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve the Steiner Network Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    m = data['parameters']['m']
    n_steiner = data['parameters']['n']
    node_weights = data['steiner_nodes']['node_weights']
    assignment_costs = data['cost_matrices']['assignment_costs']['matrix']
    interconnection_costs = data['cost_matrices']['interconnection_costs']['matrix']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    y = model.addVars(n_steiner, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, n_steiner, vtype=GRB.BINARY, name="x")
    z_root = model.addVars(n_steiner, vtype=GRB.BINARY, name="z_root")
    
    edges_steiner = []
    for j in range(n_steiner):
        for k in range(j+1, n_steiner):
            edges_steiner.append((j, k))
    z_steiner = model.addVars(edges_steiner, vtype=GRB.BINARY, name="z_steiner")
    
    nodes_extended = list(range(n_steiner + 1))
    dir_edges = []
    for i in nodes_extended:
        for j in nodes_extended:
            if i != j:
                dir_edges.append((i, j))
    flow = model.addVars(dir_edges, vtype=GRB.CONTINUOUS, name="flow")
    
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_steiner)) == 1, f"assign_target_{i}")
    
    for i in range(m):
        for j in range(n_steiner):
            model.addConstr(x[i, j] <= y[j], f"active_assign_{i}_{j}")
    
    for j in range(n_steiner):
        model.addConstr(z_root[j] <= y[j], f"root_active_{j}")
    
    for (j, k) in edges_steiner:
        model.addConstr(z_steiner[j, k] <= y[j], f"steiner_active_{j}_{k}_j")
        model.addConstr(z_steiner[j, k] <= y[k], f"steiner_active_{j}_{k}_k")
    
    model.addConstr(
        gp.quicksum(flow[n_steiner, j] for j in range(n_steiner)) == gp.quicksum(y[j] for j in range(n_steiner)),
        "flow_root"
    )
    
    for j in range(n_steiner):
        inflow = gp.quicksum(flow[i, j] for i in nodes_extended if i != j)
        outflow = gp.quicksum(flow[j, i] for i in nodes_extended if i != j)
        model.addConstr(inflow - outflow == y[j], f"flow_conserv_{j}")
    
    edge_var_dict = {}
    for j in range(n_steiner):
        edge_var_dict[(j, n_steiner)] = z_root[j]
    for (j, k) in edges_steiner:
        edge_var_dict[(j, k)] = z_steiner[j, k]
    
    for (i, j) in dir_edges:
        u, v = (i, j) if i < j else (j, i)
        model.addConstr(flow[i, j] <= n_steiner * edge_var_dict[(u, v)], f"flow_cap_{i}_{j}")
    
    total_edges = gp.quicksum(z_root[j] for j in range(n_steiner)) + gp.quicksum(z_steiner[j, k] for (j, k) in edges_steiner)
    model.addConstr(total_edges == gp.quicksum(y[j] for j in range(n_steiner)), "total_edges_constraint")
    
    activation_cost = gp.quicksum(node_weights[j] * y[j] for j in range(n_steiner))
    assignment_cost = gp.quicksum(assignment_costs[i][j] * x[i, j] for i in range(m) for j in range(n_steiner))
    interconnection_cost = gp.quicksum(interconnection_costs[j][k] * z_steiner[j, k] for (j, k) in edges_steiner)
    model.setObjective(activation_cost + assignment_cost + interconnection_cost, GRB.MINIMIZE)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
    
    try:
        if logger:
            model.optimize(callback)
        else:
            model.optimize()
    except Exception as e:
        print(f"Optimization error: {e}")
    
    if model.SolCount > 0:
        active_steiner_nodes = [j for j in range(n_steiner) if y[j].X > 0.5]
        assignments = {}
        for i in range(m):
            for j in range(n_steiner):
                if x[i, j].X > 0.5:
                    assignments[str(i)] = j
                    break
        steiner_tree_edges = [[j, k] for (j, k) in edges_steiner if z_steiner[j, k].X > 0.5]
        objective_value = model.ObjVal
    else:
        best_j = 0
        best_cost = None
        for j in range(n_steiner):
            cost = node_weights[j]
            for i in range(m):
                cost += assignment_costs[i][j]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_j = j
        active_steiner_nodes = [best_j]
        assignments = {str(i): best_j for i in range(m)}
        steiner_tree_edges = []
        objective_value = best_cost
    
    solution = {
        "objective_value": float(objective_value),
        "active_steiner_nodes": active_steiner_nodes,
        "assignments": assignments,
        "steiner_tree_edges": steiner_tree_edges
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()