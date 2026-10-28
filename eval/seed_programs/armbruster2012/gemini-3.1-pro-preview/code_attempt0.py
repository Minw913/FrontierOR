import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser(description="Graph Bisection Optimization using Gurobi.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Maximum execution time in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions.")
    args = parser.parse_args()

    # Initialize Logger
    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_nodes = data['num_nodes']
    num_edges = data['num_edges']
    capacity = data['bisection_capacity_F']
    total_wt = data['total_node_weight']
    node_weights = data['node_weights']
    edges = data['edges']
    edge_weights = data['edge_weights']

    # Environment setup
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)     # Suppress standard output
    env.setParam('Threads', 1)        # Hard limit to single CPU core
    env.start()

    # Model definition
    model = gp.Model("GraphBisection", env=env)
    
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
        
    model.setParam('MIPFocus', 1)     # Focus on finding high-quality feasible solutions

    # Decision variables
    # x[i]: Indicator if node i is contained in partition_S
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    # y[j]: Indicator if edge j is cut
    y = model.addVars(num_edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

    # Objective logic
    # To reduce the number of constraints, we selectively bind y[j] based on the weight's sign.
    obj_expr = gp.LinExpr()
    
    for idx, (u, v) in enumerate(edges):
        w = edge_weights[idx]
        if w > 0:
            obj_expr.add(y[idx], w)
            model.addConstr(y[idx] >= x[u] - x[v])
            model.addConstr(y[idx] >= x[v] - x[u])
        elif w < 0:
            obj_expr.add(y[idx], w)
            model.addConstr(y[idx] <= x[u] + x[v])
            model.addConstr(y[idx] <= 2 - x[u] - x[v])

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Capacity constraints
    expr_w = gp.LinExpr()
    for i in range(num_nodes):
        expr_w.add(x[i], node_weights[i])
        
    model.addConstr(expr_w <= capacity, name="Capacity_S")
    model.addConstr(expr_w >= total_wt - capacity, name="Capacity_Complement")
    
    # Symmetry breaking - Set reference node (0) always in partition_S
    model.addConstr(x[0] == 1, name="Symmetry_breaking")

    # Provide a constructive greedy starting solution to kickstart the solver
    sorted_nodes = sorted(range(num_nodes), key=lambda i: node_weights[i], reverse=True)
    assign = {0: 1}
    bin_wt = {1: node_weights[0], 0: 0}
    
    for i in sorted_nodes:
        if i == 0:
            continue
        # Allocate to the currently lighter bin to ideally respect constraints natively
        target = 1 if bin_wt[1] <= bin_wt[0] else 0
        if bin_wt[target] + node_weights[i] <= capacity:
            assign[i] = target
            bin_wt[target] += node_weights[i]
        else:
            alt = 1 - target
            assign[i] = alt
            bin_wt[alt] += node_weights[i]

    for i in range(num_nodes):
        x[i].Start = assign[i]

    # Solution interception callback
    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Execute optimization
    model.optimize(callback)

    # Assess result and format the required payload
    if model.SolCount > 0:
        partition_S = [i for i in range(num_nodes) if x[i].X > 0.5]
        partition_comp = [i for i in range(num_nodes) if x[i].X <= 0.5]
        obj_val = model.ObjVal
    else:
        # Fallback to the initial greedy heuristic if no feasible configuration converges strictly within the timeout
        partition_S = [i for i, val in assign.items() if val == 1]
        partition_comp = [i for i, val in assign.items() if val == 0]
        obj_val = sum(edge_weights[idx] for idx, (u, v) in enumerate(edges) if assign[u] != assign[v])

    solution = {
        "objective_value": float(obj_val),
        "partition_S": partition_S,
        "partition_complement": partition_comp
    }

    # Transmit to schema JSON output
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()