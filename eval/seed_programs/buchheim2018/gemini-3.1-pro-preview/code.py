import argparse
import json
import logging
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

class QSPCallback:
    def __init__(self, logger, x_vars, arcs, source, target):
        self.logger = logger
        self.x_vars = x_vars
        self.arcs = arcs
        self.source = source
        self.target = target
        self.best_obj = float('inf')

    def __call__(self, model, where):
        # We process lazy constraints whenever Gurobi finds a new Mixed Integer Solution.
        if where == gp.GRB.Callback.MIPSOL:
            # 1. Fetch the proposed binary solution
            x_vals = model.cbGetSolution(self.x_vars)
            
            # Map selected edges
            selected = [i for i, val in enumerate(x_vals) if val > 0.5]
            
            next_node = {} 
            edge_of = {} 
            for i in selected:
                u = self.arcs[i]['from_node']
                v = self.arcs[i]['to_node']
                next_node[u] = v
                edge_of[u] = i
                
            visited = set()
            curr = self.source
            
            # Trace the main path originating from the source
            while curr in next_node:
                visited.add(curr)
                nxt = next_node[curr]
                if nxt in visited:
                    break
                curr = nxt
            visited.add(self.target)
            
            has_cycle = False
            # Find and eliminate any disconnected sub-tours
            for u in list(next_node.keys()):
                if u not in visited:
                    cycle_edges = []
                    curr_cycle = u
                    cycle_visited = set()
                    
                    while curr_cycle not in cycle_visited and curr_cycle in next_node:
                        cycle_visited.add(curr_cycle)
                        e = edge_of[curr_cycle]
                        cycle_edges.append(e)
                        curr_cycle = next_node[curr_cycle]
                        
                    if len(cycle_edges) > 0:
                        # Append Subtour Elimination Constraint (SEC)
                        model.cbLazy(gp.quicksum(self.x_vars[e] for e in cycle_edges) <= len(cycle_edges) - 1)
                        has_cycle = True
                        for node in cycle_visited:
                            visited.add(node)
                            
            # 2. Proceed to log the objective value only if the solution doesn't contain any sub-tours
            if not has_cycle:
                obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if obj < self.best_obj - 1e-5:
                    self.best_obj = obj
                    if self.logger:
                        self.logger.log(float(obj))

def solve_qsp(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_nodes = data['num_nodes']
    num_arcs = data['num_arcs']
    source = data['source_node']
    target = data['target_node']
    arcs = data['arcs']
    linear_costs = data['linear_costs']
    quadratic_costs = data['quadratic_costs']
    
    # Initialize JSONL logger for convergence
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Instantiate the MIQP model
    model = gp.Model("QuadraticShortestPath")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)              # Restricted to a single CPU core requirement
    model.setParam('LazyConstraints', 1)      # Required to use SEC callbacks
    model.setParam('OutputFlag', 1)
    
    # Define binary decision variables for each edge natively (improves integration performance)
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(num_arcs)]
    
    # Efficiently structure bulk lists for Quad expression modeling in C backend
    q_coeffs, q_vars1, q_vars2 = [], [], []
    for i in range(num_arcs):
        row_q = quadratic_costs[i]
        for j in range(num_arcs):
            if row_q[j] != 0:
                q_coeffs.append(row_q[j])
                q_vars1.append(x[i])
                q_vars2.append(x[j])

    obj = gp.QuadExpr()
    if q_coeffs:
        obj.addTerms(q_coeffs, q_vars1, q_vars2)

    l_coeffs, l_vars = [], []
    for i in range(num_arcs):
        if linear_costs[i] != 0:
            l_coeffs.append(linear_costs[i])
            l_vars.append(x[i])
            
    if l_coeffs:
        obj.addTerms(l_coeffs, l_vars)

    model.setObjective(obj, GRB.MINIMIZE)
    
    # Mapping in/out arcs for topology construction
    in_arcs = {i: [] for i in range(num_nodes)}
    out_arcs = {i: [] for i in range(num_nodes)}
    
    for i, a in enumerate(arcs):
        in_arcs[a['to_node']].append(i)
        out_arcs[a['from_node']].append(i)
        
    # Introduce Network flow equations
    for i in range(num_nodes):
        in_sum = gp.quicksum(x[j] for j in in_arcs[i])
        out_sum = gp.quicksum(x[j] for j in out_arcs[i])
        
        if i == source:
            model.addConstr(out_sum == 1, name=f"flow_source_out")
            model.addConstr(in_sum == 0, name=f"flow_source_in")
        elif i == target:
            model.addConstr(in_sum == 1, name=f"flow_target_in")
            model.addConstr(out_sum == 0, name=f"flow_target_out")
        else:
            model.addConstr(in_sum == out_sum, name=f"flow_{i}_cons")
            model.addConstr(out_sum <= 1, name=f"max_out_degree_{i}") # Strong logical cut: limits path node recurrences
            
    cb = QSPCallback(logger, x, arcs, source, target)
    
    try:
        model.optimize(cb)
    except gp.GurobiError as e:
        # Fallback setting for handling potentially mathematically irregular/non-PSD Q matrices
        if e.errno == 10020:
            model.setParam('NonConvex', 2)
            model.optimize(cb)
        else:
            raise
            
    # Extract decision outputs from the globally optimal or BKS incumbent
    if model.solCount > 0:
        obj_val = float(model.objVal)
        selected_arcs = []
        for i, a in enumerate(arcs):
            if x[i].X > 0.5:
                selected_arcs.append({
                    "id": a['id'],
                    "from_node": a['from_node'],
                    "to_node": a['to_node']
                })
    else:
        obj_val = float('inf')
        selected_arcs = []
        logging.warning("Time limit expired without reaching a continuous integer bounds solution.")

    # Cast to precisely adhere to the required output schema constraint
    solution = {
        "objective_value": obj_val,
        "solution_arcs": selected_arcs
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance data")
    parser.add_argument("--solution_path", type=str, required=True, help="Path for exporting solving solution")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit constraint in sec")
    parser.add_argument("--log_path", type=str, default=None, help="JSONL filepath for iterative incumbent solving progress")
    args = parser.parse_args()
    
    solve_qsp(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == "__main__":
    main()