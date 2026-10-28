import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_client_server_assignment(instance_data, time_limit, log_path):
    """
    Solve the distributed processing system optimization problem using Gurobi.
    This is a quadratic assignment problem with both linear and quadratic costs.
    """
    # Extract data from instance
    n_clients = instance_data["n_clients"]
    n_servers = instance_data["n_servers"]
    demands = instance_data["demands"]
    processing_rates = instance_data["processing_rates"]
    linear_costs = instance_data["linear_costs"]
    quadratic_costs_list = instance_data["quadratic_costs"]

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("client_server_assignment")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables: x[c][s] = 1 if client c is assigned to server s
    x = []
    for c in range(n_clients):
        x_row = []
        for s in range(n_servers):
            var = model.addVar(vtype=GRB.BINARY, name=f"x_{c}_{s}")
            x_row.append(var)
        x.append(x_row)

    # Each client must be assigned to exactly one server
    for c in range(n_clients):
        model.addConstr(gp.quicksum(x[c][s] for s in range(n_servers)) == 1)

    # Prepare data structures for quadratic costs
    # Map from (c1, s1, c2, s2) -> cost
    quad_cost_map = {}
    for qc in quadratic_costs_list:
        c1, s1, c2, s2, cost = int(qc[0]), int(qc[1]), int(qc[2]), int(qc[3]), qc[4]
        quad_cost_map[(c1, s1, c2, s2)] = cost
        quad_cost_map[(c2, s2, c1, s1)] = cost  # Since it's symmetric

    # Compute linear costs part
    linear_cost_expr = gp.QuadExpr(0)
    for c in range(n_clients):
        for s in range(n_servers):
            linear_cost_expr += linear_costs[c][s] * x[c][s]

    # Build quadratic part of the objective function
    quad_cost_expr = gp.QuadExpr(0)
    # Adjacent interactions (same server)
    for s in range(n_servers):
        for c1 in range(n_clients):
            for c2 in range(c1 + 1, n_clients):
                # Cost when c1 and c2 are both assigned to s
                if (c1, s, c2, s) in quad_cost_map:
                    cost = quad_cost_map[(c1, s, c2, s)]
                else:
                    # As per description, adjacent interaction cost = sum of linear costs
                    # when they share the same server
                    cost = linear_costs[c1][s] + linear_costs[c2][s]
                
                quad_cost_expr += cost * x[c1][s] * x[c2][s]
    
    # Out-of-star interactions (different servers)
    for (c1, s1, c2, s2), cost in quad_cost_map.items():
        if s1 != s2:  # Only add out-of-star interactions
            c1, s1, c2, s2 = int(c1), int(s1), int(c2), int(s2)
            if c1 < c2 or (c1 > c2 and (c2, s2, c1, s1) not in quad_cost_map):  # Avoid duplicate
                quad_cost_expr += cost * x[c1][s1] * x[c2][s2]

    # Set objective
    total_obj = linear_cost_expr + quad_cost_expr
    model.setObjective(total_obj, GRB.MINIMIZE)

    def solution_callback(model, where):
        """Callback for logging intermediate solutions"""
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    try:
        # Optimize the model
        model.optimize(solution_callback)
        
        # Extract solution
        if model.SolCount > 0:
            assignment = {}
            total_cost = 0
            
            # Get the assignment
            for c in range(n_clients):
                for s in range(n_servers):
                    if x[c][s].X > 0.5:  # Binary variable check
                        assignment[str(c)] = s
                        break
            
            # Calculate objective value by evaluating the expression explicitly
            total_linear_cost = 0
            assignment_pairs = []
            
            for c in range(n_clients):
                assigned_s = assignment[str(c)]
                total_linear_cost += linear_costs[c][assigned_s]
                assignment_pairs.append((c, assigned_s))
            
            total_quad_cost = 0
            for i in range(len(assignment_pairs)):
                for j in range(i + 1, len(assignment_pairs)):
                    c1, s1 = assignment_pairs[i]
                    c2, s2 = assignment_pairs[j]
                    
                    if (c1, s1, c2, s2) in quad_cost_map:
                        total_quad_cost += quad_cost_map[(c1, s1, c2, s2)]
                    elif s1 == s2:  # Adjacent interactions when not explicitly listed
                        total_quad_cost += linear_costs[c1][s1] + linear_costs[c2][s2]
                        
            total_cost = total_linear_cost + total_quad_cost
        else:
            # No solution found - this shouldn't happen with a valid relaxation, return dummy
            assignment = {str(c): 0 for c in range(n_clients)}
            total_cost = float('inf')

    except Exception as e:
        # Failed to solve - return empty solution
        print(f"Exception occurred: {e}")
        assignment = {str(c): 0 for c in range(n_clients)}
        total_cost = float('inf')

    # Final solution result
    if model.status == GRB.OPTIMAL:
        # Calculate final true objective value since MIPSOL values might not be accurate
        true_assignment = {}
        for c in range(n_clients):
            for s in range(n_servers):
                if x[c][s].X > 0.5:  
                    true_assignment[str(c)] = s
                    break
        
        total_lin_cost = 0
        actual_assignments = []
        for c in range(n_clients):
            assigned_s = true_assignment[str(c)]
            total_lin_cost += linear_costs[c][assigned_s]
            actual_assignments.append((c, assigned_s))
        
        total_quad_cost = 0
        for i in range(len(actual_assignments)):
            for j in range(i + 1, len(actual_assignments)):
                c1, s1 = actual_assignments[i]
                c2, s2 = actual_assignments[j]
                
                if (c1, s1, c2, s2) in quad_cost_map:
                    total_quad_cost += quad_cost_map[(c1, s1, c2, s2)]
                elif s1 == s2:  
                    total_quad_cost += linear_costs[c1][s1] + linear_costs[c2][s2]

        result = {
            "objective_value": total_lin_cost + total_quad_cost,
            "assignment": true_assignment
        }
    else:
        # Return last valid solution or best found
        true_assignment = {}
        for c in range(n_clients):
            for s in range(n_servers):
                value = x[c][s].X if hasattr(x[c][s], 'X') else 0
                if value > 0.5:  
                    true_assignment[str(c)] = s
                    break
        # If no assignments were made in partial solution, default to first server
        if len(true_assignment) == 0:
            true_assignment = {str(c): 0 for c in range(n_clients)}

        # Recalculate objective value
        total_lin_cost = 0
        for c in range(n_clients):
            assigned_s = true_assignment[str(c)] 
            total_lin_cost += linear_costs[c][assigned_s]

        assignment_pairs = [(int(c), s) for c, s in true_assignment.items()]
        total_quad_cost = 0
        for i in range(len(assignment_pairs)):
            for j in range(i + 1, len(assignment_pairs)):
                c1, s1 = assignment_pairs[i]
                c2, s2 = assignment_pairs[j]

                if (c1, s1, c2, s2) in quad_cost_map:
                    total_quad_cost += quad_cost_map[(c1, s1, c2, s2)]
                elif s1 == s2:
                    total_quad_cost += linear_costs[c1][s1] + linear_costs[c2][s2]

        result = {
            "objective_value": total_lin_cost + total_quad_cost,
            "assignment": true_assignment
        }

    return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, required=False,
                        help="Path to a JSONL file for logging intermediate solutions.")

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_client_server_assignment(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()