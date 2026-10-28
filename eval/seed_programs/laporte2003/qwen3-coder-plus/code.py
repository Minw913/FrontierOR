import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_purchase_tour(instance_path, solution_path, timelimit, log_path=None):
    # Load the instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_vertices = data['num_vertices']
    num_markets = data['num_markets']
    num_products = data['num_products']
    depot = data['depot']
    # Coordinates might be None for random instances, so we handle that separately if needed
    tau = data.get('tau', None)
    travel_costs = data['travel_costs']
    product_demands = data['product_demands']
    product_markets = {int(k): v for k, v in data['product_markets'].items()}
    purchase_costs = {tuple(map(int, k.split(','))): v 
                      for k, v in data['purchase_costs'].items()}
    supplies = {tuple(map(int, k.split(','))): v 
                for k, v in data['supplies'].items()}

    # Determine mandatory markets
    mandatory_markets = set()
    for p_idx in range(num_products):
        prod_supply_at_other_markets = {}
        for m_idx in range(num_markets):
            if m_idx != depot and p_idx in [int(k.split(',')[0]) for k in supplies.keys() if f"{p_idx},{m_idx}" in supplies]:
                prod_supply_at_other_markets[m_idx] = supplies[(p_idx, m_idx)]

        # Check for each market whether its absence makes the total supply less than demand
        for m_idx in list(prod_supply_at_other_markets.keys()):
            total_supply_without_m = sum(
                s for m, s in prod_supply_at_other_markets.items() if m != m_idx
            )
            if total_supply_without_m < product_demands[p_idx]:
                mandatory_markets.add(m_idx)

    # Create model
    model = gp.Model("PurchaseTour")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, timelimit)

    # Sets
    markets = list(range(num_markets))
    visited_markets = [m for m in markets if m != depot]
    products = list(range(num_products))
    vertices = list(range(num_vertices))

    # Variables
    # Binary variable: x[i,j] = 1 if edge (i,j) is in the tour
    x = {}
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Continuous variable: Purchase quantity of each product at each market (where it's available)
    # We need to map these purchases to markets that have these products
    q = {}
    for (p, m) in purchase_costs.keys():
        if (m != depot):  # Only non-depot markets
            q[p, m] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=supplies.get((p, m), 0),
                                   name=f"q_{p}_{m}")

    # Variable to order nodes to avoid sub-tours explicitly
    u = {}  # Position variables for ordering markets in tour
    for i in range(1, num_vertices):
        u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"u_{i}")

    # Constraints

    # Constraint: Mandatory markets must be visited
    # For each mandatory market, there must be at least 2 edges incident to it (in and out)
    for m in mandatory_markets:
        expr = gp.LinExpr(0)
        # Sum of edges connecting to market m (edges to depot and other markets)
        for v in vertices:
            if v != m:
                if v < m:
                    expr += x[v, m]
                else:
                    expr += x[m, v]
        model.addConstr(expr == 2, f"mandatory_visit_{m}")

    # Each non-depot vertex should have degree 2 if visited, 0 otherwise
    # Since we don't know which markets to visit yet, we'll enforce exact degree 2 for each non-depot vertex implicitly
    for i in visited_markets:
        # Sum over all adjacent edges
        deg_expr = gp.LinExpr(0)
        for j in vertices:
            if i < j:
                deg_expr += x[i, j]
            elif i > j:
                deg_expr += x[j, i]
        model.addConstr(deg_expr == 2, f"degree_{i}")

    # The depot must connect to exactly 2 other nodes
    depot_degree = gp.LinExpr(0)
    for j in [v for v in vertices if v != depot]:
        a, b = min(depot, j), max(depot, j)
        depot_degree += x[a, b]
    model.addConstr(depot_degree == 2, "depot_degree")

    # Subtour elimination by adding ordering constraints like for TSP
    # We'll use Miller-Tucker-Zemlin (MTZ) constraints
    # u[j] >= u[i] + 1 - n*(1 - x[i][j])
    for i in range(1, num_vertices):
        for j in range(1, num_vertices):
            if i != j and depot not in [i, j]:  # Avoid adding constraint to depot
                a, b = min(i, j), max(i, j)
                x_edge = x[a, b] if a < b else x[b, a]
                # Using MTZ subtour elimination: u_i - u_j + 1 <= (n - 1)*(1 - x_ij)
                model.addConstr(u[i] - u[j] + (num_markets +1) * x_edge <= (num_markets +1) - 1)

    # Demand satisfaction constraints
    for p in products:
        market_indices_for_p = []
        for m in visited_markets:
            if (p, m) in q:
                market_indices_for_p.append(m)
        
        expr = gp.LinExpr(0)
        for m in market_indices_for_p:
            expr += q[p, m]
        model.addConstr(expr == product_demands[p], f"demand_{p}")

    # Objective function: Minimize (Travel Cost + Purchase Cost)

    # Travel cost expression
    travel_cost_expr = gp.LinExpr(0)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            edge_key = f"{min(i, j)},{max(i, j)}"
            c = travel_costs[edge_key]
            travel_cost_expr += c * x[i, j]

    # Purchase cost expression
    purchase_cost_expr = gp.LinExpr(0)
    for (p, m), quantity in q.items():
        purchase_cost_expr += purchase_costs.get((p, m), 999999999) * quantity

    # Combine costs
    total_cost_expr = travel_cost_expr + purchase_cost_expr

    # Set objective
    model.setObjective(total_cost_expr, GRB.MINIMIZE)

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGetSolution(model.ObjVal)
            if logger:
                logger.log(obj_val)

    # Solve
    model.optimize(mycallback)

    # Post-processing: Extract solution
    if model.SolCount > 0:
        # Find visited markets
        visited_m = []
        for m in visited_markets:
            total_edges_here = 0
            for j in vertices:
                if m < j and x[m, j].x > 0.5:
                    total_edges_here += 1
                elif m > j and x[j, m].x > 0.5:
                    total_edges_here += 1
            if total_edges_here == 2:
                visited_m.append(m)

        # Build tour by traversing connected components
        tour_edges_list = []
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if x[i, j].x > 0.5:
                    tour_edges_list.append([i, j])

        # Reconstruct tour order from edges and depot
        # Start with edges involving depot
        current_node = depot
        tour = [current_node]
        next_edges = {frozenset(e) for e in tour_edges_list}
        
        while len(tour) <= len(visited_m) + 1:
            # Find next node connected to current node that isn't previous one
            next_vertex = None
            for e in next_edges:
                if current_node in e:
                    # Get other vertex in that edge
                    if e != frozenset([tour[-2] if len(tour) >= 2 else -1, current_node]):
                        next_vertex = list(e)[0] if list(e)[0] != current_node else list(e)[1]
                        break
            
            if next_vertex is None:
                break
            
            tour.append(next_vertex)
            current_node = next_vertex
            
            # Remove the used edge
            edge_to_remove = set()
            for e in next_edges:
                if current_node in e and tour[-2] in e:
                    edge_to_remove.add(e)
                    
            for e in edge_to_remove:
                next_edges.remove(e)
                
            if current_node == depot and len(tour) > 1:
                break  # Completed the tour back to depot
        
        # Ensure the tour starts and ends at depot and includes the return journey
        if tour[0] == depot and tour[-1] != depot:
            tour.append(depot)
        elif tour[0] != depot:
            raise Exception("Invalid tour reconstruction")

        # Construct purchases dict
        purchases = {}
        for p in products:
            purchases[p] = {}
            for m in visited_markets:
                if (p, m) in q:
                    purchases[p][m] = round(q[p, m].x)

        solution = {
            "objective_value": model.objVal,
            "visited_markets": sorted(visited_m),
            "tour": tour,
            "tour_edges": tour_edges_list,
            "purchases": purchases
        }

        # Write solution
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found!")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve purchase tour optimization problem.')
    parser.add_argument('--instance_path', required=True, help='Path to the input instance JSON file.')
    parser.add_argument('--solution_path', required=True, help='Path where the solution JSON file should be written.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', default=None, help='Path to log intermediate solutions (optional)')
    
    args = parser.parse_args()

    solve_purchase_tour(args.instance_path, args.solution_path, args.time_limit, args.log_path)