import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB


def compute_euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class NetworkDesignProblem:
    def __init__(self, instance_data):
        # Initialize parameters
        self.m = instance_data["parameters"]["m"]
        self.n = instance_data["parameters"]["n"]
        
        # Target and Steiner nodes
        self.steiner_node_weights = instance_data["steiner_nodes"]["node_weights"]
        
        # Assignment costs matrix
        self.assignment_costs = instance_data["cost_matrices"]["assignment_costs"]
        
        # Interconnection costs matrix
        self.interconnection_costs = instance_data["cost_matrices"]["interconnection_costs"]
        
    def solve(self, time_limit, log_path=None):
        # Create a Gurobi model
        model = gp.Model("NetworkDesignProblem")
        model.setParam("Threads", 1)
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = time_limit
        model.setParam('MIPFocus', 1)
        
        # Variables
        activate = model.addVars(self.n, vtype=GRB.BINARY, name='activate')
        assign = model.addVars(self.m, self.n, vtype=GRB.BINARY, name='assign')
        edge_active = model.addVars([(i, j) for i in range(self.n) for j in range(i+1, self.n)],
                                     vtype=GRB.BINARY, name='edge_active')
        
        # Build objective expression
        total_activation_cost = gp.quicksum(activate[i] * self.steiner_node_weights[i] for i in range(self.n))
        total_assignment_cost = gp.quicksum(
            assign[t, s] * self.assignment_costs[t][s] 
            for t in range(self.m) 
            for s in range(self.n)
        )
        total_edge_cost = gp.quicksum(
            edge_active[i, j] * self.interconnection_costs[i][j] 
            for i in range(self.n) 
            for j in range(i+1, self.n)
        )

        # Objective function - minimizing total cost
        model.setObjective(total_activation_cost + total_assignment_cost + total_edge_cost, GRB.MINIMIZE)
        
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
        
        # Number of edges in a spanning tree is equal to the number of active Steiner nodes minus 1,
        # provided that there is at least one active node.
        total_nodes_active = sum(activate[i] for i in range(self.n))
        total_edges_active = sum(edge_active[i, j] for i in range(self.n) for j in range(i+1, self.n))
        
        # Ensure the number of edges forms a correct forest (spanning tree for each component)
        # If there are active nodes, we need a proper spanning structure
        model.addConstr(total_nodes_active >= total_edges_active)
        model.addGenConstrIndicator(activate.sum(), 1, total_edges_active == total_nodes_active - 1)

        # Subtour elimination constraint to ensure that active Steiner nodes are connected
        # Create a multi-commodity flow based subtour elimination
        if self.n > 1:
            # Variables for the spanning tree (flow variables to enforce connectivity)
            flow = model.addVars([(i, j, k) for i in range(self.n) for j in range(i+1, self.n) for k in range(self.n)],
                                 lb=0, ub=self.n-1, vtype=GRB.CONTINUOUS, name='flow')

            # Conservation of flow: each Steiner node k receives unit flow from root (node 0) via spanning tree
            # We use the flow-based MCF constraints to enforce connectivity
            for k in range(1, self.n):  # Node 0 is treated as the root
                for i in range(self.n):
                    if i == k:
                        continue
                        
                    incoming = sum(flow[j, i, k] if j < i else flow[i, j, k] for j in range(self.n) if i != j)
                    outgoing = sum(flow[i, j, k] if i < j else flow[j, i, k] for j in range(self.n) if i != j)
                    
                    # Calculate flow into node i for commodity k
                    if i == 0:  # Root sends out flow
                        model.addConstr(outgoing - incoming == (activate[k] if i == 0 and k != 0 else 0))
                    elif i == k:  # Destination receives flow
                        model.addConstr(outgoing - incoming == activate[k] * (-1))
                    else:  # Intermediate nodes conserve flow
                        model.addConstr(outgoing - incoming == activate[k] * 0)

                    # Bound flows by existence of edge
                    for j in range(i+1, self.n):
                        if (i==k or j==k) and (not (i==0 or j==0) if k != 0 else True):
                            # Adjust for the direction
                            model.addConstr(flow[min(i,j), max(i,j), k] <= (self.n - 1) * edge_active[min(i,j), max(i,j)])

                # Flow conservation constraints
                for j_idx in range(len([j for j in range(i+1, self.n)])):
                    j = [jj for jj in range(i+1, self.n)][j_idx]
                    if k != i and k != j:
                        # Each non-root, non-k Steiner node has net flow of 0 for commodity k
                        pass 

        # Set up solution logging callback if needed
        if log_path:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(log_path, sense="minimize")
            
            def log_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
                    
            model.optimize(log_callback)
        else:
            model.optimize()
        
        # Extract the solution
        if model.SolCount > 0:
            # Get active Steiner nodes
            active_steiner_nodes = [i for i in range(self.n) if activate[i].x > 0.5]
            
            # Get assignments
            assignments = {}
            for t in range(self.m):
                for s in range(self.n):
                    if assign[t, s].x > 0.5:
                        assignments[str(t)] = s
                        break
            
            # Get Steiner tree edges
            steiner_tree_edges = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if edge_active[i, j].x > 0.5:
                        steiner_tree_edges.append([i, j])
                        
            obj_val = model.ObjVal
            
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