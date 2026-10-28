import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_bisection(instance_data, time_limit, log_path):
    """
    Solves the bisection problem using a Mixed Integer Programming approach.
    
    The problem is to partition the nodes into two groups such that:
    1. Each group has a total weight no more than the bisection_capacity_F
    2. Number of edges across between the two groups is minimized
    """
    # Extract instance data
    n = instance_data["num_nodes"]
    m = instance_data["num_edges"]
    F = instance_data["bisection_capacity_F"]
    weights = instance_data["node_weights"]
    edges = instance_data["edges"]
    edge_costs = instance_data["edge_weights"]
    
    total_weight = sum(weights)
    
    # Initialize logging
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create model
    model = gp.Model("graph_bisection")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Binary variables: x[i] = 1 means node i is in partition S, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Binary variables: y[j] = 1 means edge j is cut
    y = model.addVars(m, vtype=GRB.BINARY, name="y")
    
    # Constraint: Group S (nodes where x[i]=1) has weight <= F
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= F, name="capacity_S")
    
    # Constraint: Group T (nodes where x[i]=0) has weight <= F
    model.addConstr(gp.quicksum(weights[i] * (1 - x[i]) for i in range(n)) <= F, name="capacity_T")
    
    # For each edge, determine if it's cut (endpoints in different partitions)
    for j in range(m):
        u, v = edges[j]
        cost = edge_costs[j]
        
        # If u is in S and v is in T, or u is in T and v is in S, then edge is cut
        # This can be expressed as y[j] >= |x[u] - x[v]| which translates to:
        # y[j] >= x[u] - x[v] 
        # y[j] >= x[v] - x[u]
        model.addConstr(y[j] >= x[u] - x[v], name=f"edge_cut_{j}_1")
        model.addConstr(y[j] >= x[v] - x[u], name=f"edge_cut_{j}_2")
    
    # Objective: minimize sum of costs of cut edges
    obj = gp.quicksum(edge_costs[j] * y[j] for j in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    # Track best solution found
    best_solution = {
        "objective_value": float('inf'),
        "partition_S": [],
        "partition_complement": []
    }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current solution values
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            
            # Get the assignments
            x_vals = model.cbGetSolution(x)
            partition_s = [i for i in range(n) if x_vals[i] > 0.5]
            partition_t = [i for i in range(n) if x_vals[i] < 0.5]

            # Update best solution if this is better
            if obj_val < best_solution["objective_value"]:
                best_solution.update({
                    "objective_value": obj_val,
                    "partition_S": partition_s,
                    "partition_complement": partition_t
                })
                
                if logger:
                    logger.log(obj_val)

    # Optimize with callback to track solution progress
    model.optimize(callback)
    
    # Return best found solution, or construct from final model values if needed
    if model.solCount > 0 and model.objVal < best_solution["objective_value"]:
        x_vals = model.getAttr('X', x)
        partition_s = [i for i in range(n) if x_vals[i] > 0.5]
        partition_t = [i for i in range(n) if x_vals[i] < 0.5]
        best_solution.update({
            "objective_value": model.objVal,
            "partition_S": partition_s,
            "partition_complement": partition_t
        })

    return best_solution


def main():
    parser = argparse.ArgumentParser(description='Solve graph bisection problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write solution')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file')

    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_bisection(instance_data, args.time_limit, args.log_path)

    # Ensure capacity constraints are satisfied (just double check calculations here)
    node_weights = instance_data["node_weights"]
    partition_s_weight = sum(node_weights[i] for i in solution["partition_S"])
    partition_t_weight = sum(node_weights[i] for i in solution["partition_complement"])
    F = instance_data["bisection_capacity_F"]

    # Validate the constraint satisfaction manually
    assert partition_s_weight <= F, f"Partition S exceeds capacity: {partition_s_weight} > {F}"
    assert partition_t_weight <= F, f"Partition T exceeds capacity: {partition_t_weight} > {F}"

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()