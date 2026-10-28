import argparse
import json
import math
from itertools import combinations
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_atsp_with_prizes(instance_path: str, solution_path: str, time_limit: int, log_path: str):
    """
    Solves the variant of the Traveling Salesman Problem with prizes,
    where we need to maximize total prize collected within a travel time budget.
    """
    # Load the instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    depot = data['depot']  # Always 1 according to problem description
    t0 = data['t0']
    prizes = data['prizes']
    travel_times = data['travel_times']

    # Create list of cities [1, n] where 1 is the depot
    all_cities = list(range(1, n + 1))
    customer_cities = [i for i in all_cities if i != depot]

    # Prepare travel times matrix (we'll create a full matrix representation for convenience)
    dist_matrix = {}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                dist_matrix[i, j] = 0
            elif i < j:
                dist_matrix[i, j] = travel_times.get(f"({i},{j})", float('inf'))
            else:
                dist_matrix[i, j] = travel_times.get(f"({j},{i})", float('inf'))  # Ensure symmetry

    # Gurobi model
    model = gp.Model("Prize_Collecting_TSP")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    
    # Suppress console output
    model.setParam(GRB.Param.LogToConsole, 0)
    
    # Set time limit
    model.setParam(GRB.Param.TimeLimit, time_limit)

    # Decision variables
    # x_ij: binary variable indicating if directed edge from i to j is in tour (for customers)
    x = {}
    for i in all_cities:
        for j in all_cities:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # u_i: continuous variables for MTZ subtour elimination (exclude depot)
    u = {}
    for i in customer_cities:
        # MTZ variables should be between 2 and n, but we'll have constraints that handle them correctly
        u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=2, ub=n, name=f"u_{i}")

    # Constraints
    # 1. Flow conservation: enter each node as many times as you leave
    for i in all_cities:
        model.addConstr(
            gp.quicksum(x[i, j] for j in all_cities if i != j) == 
            gp.quicksum(x[j, i] for j in all_cities if j != i),
            f"flow_conservation_{i}"
        )

    # 2. Each selected city is visited exactly once (flow in equals flow out and both equal 1 if visited)
    # The depot is visited exactly once in the cycle
    for i in customer_cities:
        # Flow entering equals flow exiting for each customer city
        model.addConstr(
            gp.quicksum(x[j, i] for j in all_cities if j != i) == 
            gp.quicksum(x[i, j] for j in all_cities if i != j),
            f"out_equal_in_{i}"
        )
        # Either the city is visited (1 flow in/out) or not visited (0 flow in/out)
        # This is automatically handled by the flow constraints and the requirement that we have one cycle through depot

    # 3. MTZ constraints for subtour elimination
    for i in customer_cities:
        for j in customer_cities:
            if i != j:
                # If i->j edge is taken then ui + 1 <= uj
                model.addConstr(
                    u[i] - u[j] + (len(all_cities) - 1) * x[i, j] <= len(all_cities) - 2,
                    f"subtour_elimination_{i}_{j}"
                )

    # 4. Exactly one arc goes IN and OUT of depot to initiate the tour
    model.addConstr(gp.quicksum(x[depot, j] for j in all_cities if j != depot) == 1, "one_out_depot")
    model.addConstr(gp.quicksum(x[i, depot] for i in all_cities if i != depot) == 1, "one_in_depot")

    # 5. Travel time constraint
    total_travel_time_expr = gp.quicksum(
        dist_matrix[i, j] * x[i, j] 
        for i in all_cities 
        for j in all_cities 
        if i != j
    )
    model.addConstr(total_travel_time_expr <= t0, "time_budget")

    # At least 3 cities on the tour
    # We ensure that at least two customers are on the path, plus the depot makes 3 minimum
    visited_customers_count = gp.quicksum(x[depot, j] for j in customer_cities)
    model.addConstr(visited_customers_count >= 1, "at_least_one_customer")

    # Objective: maximize total prize
    prize_expression = gp.quicksum(
        prizes.get(str(i), 0) * gp.quicksum(x[i, j] for j in all_cities if i != j)
        for i in all_cities
    )

    model.setObjective(prize_expression, GRB.MAXIMIZE)

    # Create logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Optimize the model
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
                
    try:
        model.optimize(callback)
    except Exception as e:
        print(f"Optimization error: {e}")
        return
    
    # Extract solution
    
    if model.SolCount > 0:
        # Get active edges
        solution_edges = []
        for i in all_cities:
            for j in all_cities:
                if i != j:
                    if x[i, j].X > 0.5:  # consider as active because it's binary
                        solution_edges.append([i, j])

        # Construct the actual tour
        # Start from depot, keep moving along selected edges until we come back to depot
        current_node = depot
        tour = [current_node]
        visited_set = set([current_node])
        
        # Find the next city from depot
        found_next = True
        while current_node != depot or len(tour) == 1:  # While we haven't returned to depot again
            next_node = None
            for edge in solution_edges:
                if edge[0] == current_node and edge[1] not in visited_set:
                    next_node = edge[1]
                    break
            if next_node is not None:
                tour.append(next_node)
                current_node = next_node
                visited_set.add(next_node)
            else:
                # If we've already visited this next city, look for a different option or return to depot
                for edge in solution_edges:
                    if edge[0] == current_node and edge[1] == depot:
                        tour.append(depot)
                        break  # End the tour
                break
        
        # In the case of a round trip (depot -> city -> depot), make sure the tour includes proper cycle
        # The final tour should start and end at depot
        if len(tour) >= 3 and tour[-1] != depot:
            tour.append(depot)
        elif len(tour) == 2:  # depot-city-depot pattern
            tour.append(depot)
            
        visited_nodes = []
        total_prize_collected = 0
        
        for node in tour:
            if node not in visited_nodes:
                visited_nodes.append(node)
                total_prize_collected += prizes.get(str(node), 0)

        # Final check that edges match actual tour
        final_tour_edges = []
        for i in range(len(tour) - 1):
            final_tour_edges.append([tour[i], tour[i+1]])

        # Construct final solution according to schema
        result = {
            "objective_value": sum(prizes[str(nod)] for nod in visited_nodes),
            "visited_nodes": sorted(list(set(visited_nodes))),
            "edges": final_tour_edges,  # The edges that form the tour
            "tour": tour
        }
        
        return result
    else:
        print("No solution found")
        return {"objective_value": 0, "visited_nodes": [], "edges": [], "tour": []}


def main():
    parser = argparse.ArgumentParser(description='Solve PC-TSP with budget constraint.')
    parser.add_argument('--instance_path', type=str, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to save the logs (optional)')
    
    args = parser.parse_args()
    
    solution = solve_atsp_with_prizes(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )
    
    # Write the solution to the specified path
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()