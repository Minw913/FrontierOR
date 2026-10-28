import argparse
import json
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class MipLoggingCallback:
    def __init__(self, logger):
        self.logger = logger
        
    def __call__(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if self.logger:
                    self.logger.log(obj)
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Telecommunications Minimum-Cost Network Design")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the final solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, default=0, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default="", help="Path to a JSONL file for logging intermediate solutions.")
    args = parser.parse_args()

    # Initialize SolutionLogger
    logger = None
    if args.log_path:
        # SolutionLogger initializes its file and manages formatting
        if SolutionLogger is not None:
            logger = SolutionLogger(args.log_path, sense="minimize")
        else:
            print("Warning: solution_logger module not found. Proceeding without logging.")

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    m = data['parameters']['m']
    n = data['parameters']['n']

    steiner_weights = data['steiner_nodes']['node_weights']
    assign_costs = data['cost_matrices']['assignment_costs']['matrix']
    inter_costs = data['cost_matrices']['interconnection_costs']['matrix']

    # Initialize Gurobi model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    
    model = gp.Model("SteinerTree_Network", env=env)
    model.setParam('Threads', 1)

    if args.time_limit > 0:
        model.setParam('TimeLimit', args.time_limit)

    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    r = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="r")
    
    # Directed edges for spanning tree calculation
    # Using specific tuple iterators avoids self-loops out of the box
    g = model.addVars(((i, j) for i in range(n) for j in range(n) if i != j), vtype=GRB.BINARY, name="g")
    
    # Single-commodity flow capacities
    f = model.addVars(((i, j) for i in range(n) for j in range(n) if i != j), vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name="f")

    # Objective Function
    obj_y = gp.quicksum(steiner_weights[j] * y[j] for j in range(n))
    obj_x = gp.quicksum(assign_costs[i][j] * x[i, j] for i in range(m) for j in range(n))
    obj_g = gp.quicksum(inter_costs[i][j] * g[i, j] for i in range(n) for j in range(n) if i != j)
    model.setObjective(obj_y + obj_x + obj_g, GRB.MINIMIZE)

    # Target assignment constraints
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(m)), name="assign_sum")
    model.addConstrs((x[i, j] <= y[j] for i in range(m) for j in range(n)), name="assign_bound")

    # One and only one root node across the activated nodes
    model.addConstr(gp.quicksum(r[j] for j in range(n)) == 1, name="one_root")
    model.addConstrs((r[j] <= y[j] for j in range(n)), name="root_bound")

    # Edge logic ensuring exactly 1 incoming connection per non-root active node
    model.addConstrs((gp.quicksum(g[i, j] for i in range(n) if i != j) == y[j] - r[j] for j in range(n)), name="in_degree")
    model.addConstrs((g[i, j] <= y[i] for i in range(n) for j in range(n) if i != j), name="active_orig")

    # Flow bounding within edges
    model.addConstrs((f[i, j] <= (n - 1) * g[i, j] for i in range(n) for j in range(n) if i != j), name="flow_bound")

    # Tree flow constraints simulating consumption and establishing connected components strictly up to the root
    for j in range(n):
        P_j = gp.quicksum(f[j, k] for k in range(n) if k != j) - gp.quicksum(f[i, j] for i in range(n) if i != j)
        model.addConstr(P_j >= -y[j], name=f"flow_lb_{j}")
        model.addConstr(P_j <= n * r[j] - y[j], name=f"flow_ub_{j}")

    # Attach Logging Callback if requested
    if logger:
        cb = MipLoggingCallback(logger)
        model.optimize(cb)
    else:
        model.optimize()

    # Reconstruct explicitly and strictly using pure Python evaluations
    if model.SolCount > 0:
        active_steiner = [j for j in range(n) if y[j].X > 0.5]
        
        assignments = {}
        target_cost_sum = 0.0
        for i in range(m):
            # Select definitively identical node mapped by objective optimization argmin logic
            best_j = min(active_steiner, key=lambda j: assign_costs[i][j])
            assignments[str(i)] = best_j
            target_cost_sum += assign_costs[i][best_j]
            
        tree_edges = []
        tree_cost_sum = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and g[i, j].X > 0.5:
                    u, v = min(i, j), max(i, j)
                    if [u, v] not in tree_edges:
                        tree_edges.append([u, v])
                        tree_cost_sum += inter_costs[u][v]
        
        final_steiner_cost = sum(steiner_weights[j] for j in active_steiner)
        true_objective = float(final_steiner_cost + target_cost_sum + tree_cost_sum)

        solution = {
            "objective_value": true_objective,
            "active_steiner_nodes": active_steiner,
            "assignments": assignments,
            "steiner_tree_edges": tree_edges
        }

        with open(args.solution_path, 'w') as out_f:
            json.dump(solution, out_f, indent=2)
            
    else:
        # Fallback or strict error indicating no feasible solution was identified 
        # Limits typically prevent this assuming simple M >= 1 initial bounds.
        print("No feasible solution was found within the designated limits.", file=sys.stderr)

if __name__ == '__main__':
    main()