import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def compute_euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class NetworkDesignProblem:
    def __init__(self, instance_data):
        # Initialize parameters
        self.m = instance_data["parameters"]["m"]
        self.n = instance_data["parameters"]["n"]
        
        # Target and Steiner nodes
        target_coords = instance_data["target_nodes"]["coordinates"]
        steiner_coords = instance_data["steiner_nodes"]["coordinates"]
        self.steiner_node_weights = instance_data["steiner_nodes"]["node_weights"]
        
        # Assignment costs matrix
        self.assignment_costs = instance_data["cost_matrices"]["assignment_costs"]
        
        # Interconnection costs matrix
        self.interconnection_costs = instance_data["cost_matrices"]["interconnection_costs"]
        
    def solve(self, time_limit, log_path=None):
        # Create a Gurobi model
        model = gp.Model("NetworkDesignProblem")
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = time_limit
        model.setParam('MIPFocus', 1)
        
        # Variables
        activate = model.addVars(self.n, vtype=GRB.BINARY, name='activate')
        assign = model.addVars(self.m, self.n, vtype=GRB.BINARY, name='assign')
        edge_active = model.addVars(self.n, self.n, vtype=GRB.BINARY, name='edge_active')
        
        # Objective function - minimizing total cost
        obj_expr = (
            sum(activate[i] * self.steiner_node_weights[i] for i in range(self.n)) +
            sum(assign[t, s] * self.assignment_costs[t][s] for t in range(self.m) for s in range(self.n)) +
            sum(edge_active[i, j] * self.interconnection_costs[i][j] for i in range(self.n) for j in range(i+1, self.n))
        )
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Each target node is assigned to exactly one Steiner node
        for t in range(self.m):
            model.addConstr(sum(assign[t, s] for s in range(self.n)) == 1)
            
        # Target node can only be assigned to active Steiner node
        for t in range(self.m):
            for s in range(self.n):
                model.addConstr(assign[t, s] <= activate[s])
                
        # Edges can only exist between active Steiner nodes
        for i in range(self.n):
            for j in range(i+1, self.n):
                model.addConstr(edge_active[i, j] <= activate[i])
                model.addConstr(edge_active[i, j] <= activate[j])
        
        # Spanning tree constraint: |edges| = |vertices| - 1 if at least one vertex is active
        total_activate = sum(activate[i] for i in range(self.n))
        total_edges = sum(edge_active[i, j] for i in range(self.n) for j in range(i+1, self.n))
        model.addConstr(total_edges == total_activate - gp.min_(total_activate, 1))
        
        # Subtour elimination (MTZ-style) but we can use connectivity constraints
        # For each subset S of nodes to avoid cycles: |E(S)| <= |S| - 1
    
        # Connectivity - ensure all activated Steiner nodes form a connected component
        # MTZ-like subtour elimination
        
        # Alternative approach: flow-based connectivity (for large graphs, MTZ might work better)
        # Define flow variables
        flow = model.addVars(self.n, self.n, lb=0, ub=self.n-1, vtype=GRB.CONTINUOUS, name='flow')
        
        # Ensure all activated nodes form a tree with one "root"
        root_flow_vars = model.addVars(self.n, vtype=GRB.BINARY, name='root_indicator')
        
        # Exactly one root
        model.addConstr(sum(root_flow_vars[i] for i in range(self.n)) == 1)
        
        # Ensure flows go out from roots
        for i in range(self.n):
            model.addConstr(sum(flow[j, i] for j in range(self.n) if j != i) - 
                           sum(flow[i, j] for j in range(self.n) if j != i) ==
                          activate[i] - self.n * root_flow_vars[i])
        
        # Link flows to edges
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    model.addConstr(flow[i, j] <= self.n * edge_active[i, j])
        
        # Set callback to track solution updates if logging is enabled
        if log_path:
            logger = SolutionLogger(log_path, sense="minimize")
            
            def mycallback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
                    
            model.optimize(mycallback)
        else:
            model.optimize()
        
        # Extract the solution
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            # Get active Steiner nodes
            active_steiner_nodes = [i for i in range(self.n) if activate[i].X > 0.5]
            
            # Get assignments
            assignments = {}
            for t in range(self.m):
                for s in range(self.n):
                    if assign[t, s].X > 0.5:
                        assignments[str(t)] = s
                        break
            
            # Get Steiner tree edges
            steiner_tree_edges = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if edge_active[i, j].X > 0.5:
                        steiner_tree_edges.append([i, j])
                        
            obj_val = model.ObjVal if model.solCount > 0 else float('inf')
            
            # Prepare output
            solution = {
                'objective_value': obj_val,
                'active_steiner_nodes': active_steiner_nodes,
                'assignments': assignments,
                'steiner_tree_edges': steiner_tree_edges
            }
            
            return solution
        else:
            # Return empty solution in case of infeasible/failed status
            return {
                'objective_value': float('inf'),
                'active_steiner_nodes': [],
                'assignments': {},
                'steiner_tree_edges': []
            }


def main():
    parser = argparse.ArgumentParser(description='Solve network design problem.')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit for solving in seconds')
    parser.add_argument('--log_path', help='Path for logging intermediate solutions (optional)')
    
    args = parser.parse_args()

    # Initialize the solution logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    problem = NetworkDesignProblem(instance_data)
    solution = problem.solve(args.time_limit, args.log_path)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
        

if __name__ == "__main__":
    main()