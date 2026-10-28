import argparse
import json
import math
import random
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_pickup_and_delivery(instance_data, timelimit):
    # Extract instance data
    num_stacks_s = instance_data["parameters"]["num_stacks_s"]
    stack_height_l = instance_data["parameters"]["stack_height_l"]
    num_requests_n = instance_data["parameters"]["num_requests_n"]
    num_pickup_vertices = instance_data["parameters"]["num_pickup_vertices"]
    num_delivery_vertices = instance_data["parameters"]["num_delivery_vertices"]

    # Extract pickup region info
    pickup_depot_index = instance_data["pickup_region"]["depot_index"]
    pickup_distance_matrix = instance_data["pickup_region"]["distance_matrix"]
    
    # Extract delivery region info
    delivery_depot_index = instance_data["delivery_region"]["depot_index"]
    delivery_distance_matrix = instance_data["delivery_region"]["distance_matrix"]
    
    # Extract request info
    requests = instance_data["requests"]
    
    # Create mapping from request_id to index
    request_id_to_idx = {req["request_id"]: i for i, req in enumerate(requests)}
    
    # Extract pickup and delivery vertices for each request
    pickup_vertices_map = {i: req["pickup_vertex"] for i, req in enumerate(requests)}  # i -> pickup_vertex
    delivery_vertices_map = {i: req["delivery_vertex"] for i, req in enumerate(requests)}  # i -> delivery_vertex
    
    # Invert the mapping (vertex id -> request index)
    pickup_req_id_map = {req["pickup_vertex"]: i for i, req in enumerate(requests)}
    delivery_req_id_map = {req["delivery_vertex"]: i for i, req in enumerate(requests)}

    # Initialize Gurobi model
    model = gp.Model("stack_constrained_pickup_delivery")
    model.setParam(GRB.Param.TimeLimit, timelimit)
    model.setParam(GRB.Param.MIPFocus, 1)  # focus on finding feasible solutions

    # Create binary variables for the arcs in pickup region
    x_pickup = {}
    for i in range(num_pickup_vertices):
        for j in range(num_pickup_vertices):
            if i != j:
                x_pickup[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_pickup_{i}_{j}")

    # Create binary variables for the arcs in delivery region
    x_delivery = {}
    for i in range(num_delivery_vertices):
        for j in range(num_delivery_vertices):
            if i != j:
                x_delivery[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_delivery_{i}_{j}")

    # Subtour elimination using MTZ (Miller-Tucker-Zemlin) method
    u_pickup = {}
    for i in range(1, num_pickup_vertices):
        u_pickup[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_pickup_vertices-1, name=f"u_pickup_{i}")

    u_delivery = {}
    for i in range(1, num_delivery_vertices):
        u_delivery[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_delivery_vertices-1, name=f"u_delivery_{i}")

    # Constraint: degree constraints for pickup region
    for i in range(num_pickup_vertices):
        # Outgoing arcs
        model.addConstr(gp.quicksum(x_pickup[i, j] for j in range(num_pickup_vertices) if i != j) == 1)
        # Incoming arcs
        model.addConstr(gp.quicksum(x_pickup[j, i] for j in range(num_pickup_vertices) if i != j) == 1)

    # Constraint: degree constraints for delivery region
    for i in range(num_delivery_vertices):
        # Outgoing arcs
        model.addConstr(gp.quicksum(x_delivery[i, j] for j in range(num_delivery_vertices) if i != j) == 1)
        # Incoming arcs
        model.addConstr(gp.quicksum(x_delivery[j, i] for j in range(num_delivery_vertices) if i != j) == 1)

    # Subtour elimination constraints for pickup region
    for i in range(1, num_pickup_vertices):
        for j in range(1, num_pickup_vertices):
            if i != j:
                model.addConstr(u_pickup[i] - u_pickup[j] + 1 <= (num_pickup_vertices-1) * (1 - x_pickup[i, j]))

    # Subtour elimination constraints for delivery region
    for i in range(1, num_delivery_vertices):
        for j in range(1, num_delivery_vertices):
            if i != j:
                model.addConstr(u_delivery[i] - u_delivery[j] + 1 <= (num_delivery_vertices-1) * (1 - x_delivery[i, j]))

    # Objective function: minimize total travel cost
    pickup_cost_expr = gp.quicksum(pickup_distance_matrix[i][j] * x_pickup[i, j]
                                   for i in range(num_pickup_vertices)
                                   for j in range(num_pickup_vertices) if i != j)
    delivery_cost_expr = gp.quicksum(delivery_distance_matrix[i][j] * x_delivery[i, j]
                                     for i in range(num_delivery_vertices)
                                     for j in range(num_delivery_vertices) if i != j)
    
    total_cost = pickup_cost_expr + delivery_cost_expr
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Now we need to handle the LIFO stack constraints
    # This requires tracking when pickup request i happens before pickup request j,
    # and when delivery request k happens before delivery request l.
    
    # We introduce binary variables to track whether request i is picked up before request j in the pick up tour
    pickup_before = {}
    for i in range(num_requests_n):
        for j in range(num_requests_n):
            if i != j:
                pickup_before[i, j] = model.addVar(vtype=GRB.BINARY, name=f"pickup_before_{i}_{j}")
    
    # Also binary variables to track whether request k is delivered before request l in the delivery tour
    delivery_before = {}
    for k in range(num_requests_n):
        for l in range(num_requests_n):
            if k != l:
                delivery_before[k, l] = model.addVar(vtype=GRB.BINARY, name=f"delivery_before_{k}_{l}")

    # For pickup tour, the precedence of visit order corresponds to ordering of positions
    # We'll use MTZ-like variables again to get the position in each tour
    pos_pickup = {}  # pos_pickup[req_idx] gives relative position in pickup tour of request i
    for i in range(num_requests_n):
        # position of request i in pickup tour among requests (not counting depot)
        pickup_vertex_idx = pickup_vertices_map[i]
        if pickup_vertex_idx != pickup_depot_index:
            pos_pickup[i] = u_pickup[pickup_vertex_idx]
        else:
            pos_pickup[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"pos_pickup_{i}")

    # Similarly for deliveries
    pos_delivery = {}  # pos_delivery[req_idx] gives relative position in delivery tour of request i
    for i in range(num_requests_n):
        # position of request i in delivery tour among requests
        delivery_vertex_idx = delivery_vertices_map[i]
        if delivery_vertex_idx != delivery_depot_index:
            pos_delivery[i] = u_delivery[delivery_vertex_idx]
        else:
            pos_delivery[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"pos_delivery_{i}")

    # Now add logic constraints to make pickup_before and delivery_before consistent with actual visit orders
    # Use big-M formulation to connect these
    M = max(
        len([x for req_list in pickup_distance_matrix for x in req_list]),
        len([x for req_list in delivery_distance_matrix for x in req_list])
    ) * 2  # a safe upper bound on differences in positions

    for i in range(num_requests_n):
        for j in range(num_requests_n):
            if i != j:
                # pickup_before[i,j] is true iff pos_pickup[i] < pos_pickup[j]
                model.addConstr(pos_pickup[i] <= pos_pickup[j] - 0.5 + M*(1-pickup_before[i,j]))
                model.addConstr(pos_pickup[i] >= pos_pickup[j] + 0.5 - M*pickup_before[i,j])

    for k in range(num_requests_n):
        for l in range(num_requests_n):
            if k != l:
                # delivery_before[k,l] is true iff pos_delivery[k] < pos_delivery[l]
                model.addConstr(pos_delivery[k] <= pos_delivery[l] - 0.5 + M*(1-delivery_before[k,l]))
                model.addConstr(pos_delivery[k] >= pos_delivery[l] + 0.5 - M*delivery_before[k,l])

    # Add stack capacity and LIFO loading constraints
    # If request_i is picked up before request_j AND request_k is delivered before request_j 
    # and they are assigned to same stack, then we must deliver i before k.
    # More explicitly, if i is picked up before j (i<j in pick order), and k is delivered before j (k<j in del order),
    # and i and k share the same stack, then i must be delivered before k (so that after taking j off the stack
    # and k off the stack, we can access j before i, but for k to be delivered first implies i was below k).
    # Let's reframe the LIFO logic:
    # If pi < pj (request i picked before j) and dj < di (j delivered before i), then i and j must be in different stacks.
    
    # To implement stack constraints, we'll need additional variables for item-stack assignment,
    # but the problem doesn't require explicit tracking of the assignment in the output.
    # Instead, we apply lifted ordering inequalities.
    
    # Add the LIFO-stack compatibility constraint
    for i in range(num_requests_n):
        for j in range(num_requests_n):
            if i != j:
                for k in range(num_requests_n):
                    if k != i and k != j:
                        # if i before j in pickup AND j before i in delivery,
                        # then i and k cannot be scheduled as k picked before j but j before k in delivery
                        # The basic incompatibility: if A picked before B => A delivered after B (under LIFO within same stack)
                        # So if we want to check consistency of an overall ordering scenario:
                        # we must satisfy: forall requests w.picked before x & y.delivered before z,
                        # if w,x in same stack then x should be delivered before w (contradiction unless w=x)
                        # OR, more directly:
                        # If A picked before B, and C delivered before D, and A,C in same stack and B,D in same stack,
                        # then (A picked after than C OR B delivered before D) [or something like that]
                        
                        # Actually let's implement it as follows:
                        # For any three requests i, j, k such that i picked before j AND (i or j) goes into same stack as k:
                        # Then j cannot be delivered before i UNLESS i & k are in different stacks OR j & k in different stacks
                        # This gets complex with binary stack assignments

                        # Instead, we'll use an indirect approach by imposing that:
                        # If pi < pj and dj < di (where p is pickup time, d is delivery time),
                        # then request i and j cannot be in same stack.
                        # This can be represented as: pickup_before[i,j] + delivery_before[j,i] <= 1 + stack_diff[i,j]
                        # where stack_diff = 1 if in different stacks, but we don't know stacking
                        # Since all requests are on the same vehicle with fixed stack numbers,
                        # we might need to create an explicit stack assignment constraint.
                        # However, the real constraint is between pairs.
                        
                        # Better approach: For any pairs i,j:
                        # If i picked before j AND j delivered before i, these orderings are incompatible with LIFO in same stack
                        # So we just need inequality of pickup_before[i,j] + delivery_before[j,i] <= 1
                        model.addConstr(pickup_before[i, j] + delivery_before[j, i] <= 1)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Optimize
    global logger  # assuming logger is in the scope of the callback
    
    try:
        model.optimize(callback)
        
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Extract the tours
            pickup_order = extract_tour_from_arcs(x_pickup, num_pickup_vertices)
            delivery_order = extract_tour_from_arcs(x_delivery, num_delivery_vertices)
            
            obj_val = model.objVal
            
            solution_obj = {
                "objective_value": obj_val,
                "pickup_tour": pickup_order,
                "delivery_tour": delivery_order
            }
            return solution_obj
        
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            return None
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Return the best found solution so far if exists
        if model.SolCount > 0:
            pickup_order = extract_tour_from_arcs(x_pickup, num_pickup_vertices)
            delivery_order = extract_tour_from_arcs(x_delivery, num_delivery_vertices)
            
            solution_obj = {
                "objective_value": model.objVal if model.objVal is not None else float('inf'),
                "pickup_tour": pickup_order,
                "delivery_tour": delivery_order
            }
            return solution_obj
        else:
            return None
    

def extract_tour_from_arcs(arc_vars, n_vertices):
    """
    Extract a tour from binary arc variables.
    Returns list of integers representing the sequence of vertices visited.
    """
    # Create a dictionary from (i, j) to their current value
    arcs_used = []
    for i in range(n_vertices):
        for j in range(n_vertices):
            if i != j and arc_vars[i, j].X > 0.9:
                arcs_used.append((i, j))
    
    # Build adjacency dict for tour reconstruction
    next_node = {}
    prev_node = {}
    
    for i, j in arcs_used:
        next_node[i] = j
        prev_node[j] = i
    
    # Start from node 0 (the depot)
    tour = [0]
    curr = next_node[0]
    
    while curr != 0:
        tour.append(curr)
        curr = next_node[curr]
    
    return tour


def main():
    parser = argparse.ArgumentParser(description='Solve Stack-Constrained Pickup and Delivery Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions')

    args = parser.parse_args()

    global logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_pickup_and_delivery(instance_data, args.time_limit)

    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution found.")
        # Write empty/placeholder solution to avoid output errors
        with open(args.solution_path, 'w') as f:
            json.dump({
                "objective_value": float('inf'),
                "pickup_tour": [],
                "delivery_tour": []
            }, f)


if __name__ == "__main__":
    main()