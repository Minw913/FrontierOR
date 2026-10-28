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
        u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n-1, name=f"u_{i}")

    # Constraints
    # 1. Flow conservation: enter each node as many times as you leave
    for i in all_cities:
        model.addConstr(
            gp.quicksum(x[i, j] for j in all_cities if i != j) == 
            gp.quicksum(x[j, i] for j in all_cities if j != i),
            f"flow_conservation_{i}"
        )

    # 2. Visit the depot exactly twice (enter and exit once)
    model.addConstr(
        gp.quicksum(x[depot, j] for j in all_cities if j != depot) == 1,
        "out_depot"
    )
    model.addConstr(
        gp.quicksum(x[j, depot] for j in all_cities if j != depot) == 1,
        "in_depot"
    )

    # 3. For each customer city, decide whether to visit it (in=out=1) or not (in=out=0)
    for i in customer_cities:
        model.addConstr(
            gp.quicksum(x[i, j] for j in all_cities if i != j) == 
            gp.quicksum(x[j, i] for j in all_cities if j != i),
            f"visit_rule_{i}"
        )
        
        # MTZ constraints to eliminate subtours and enforce connectivity
        # If arc (i,j) is used (for i,j non-depot), then u_i + 1 <= u_j <= n-1 - (n-2)*x_ij
        # The simplified version is: u[i] - u[j] + (n-1) * x[i,j] <= n - 2
        for j in customer_cities:
            if i != j:
                model.addConstr(
                    u[i] - u[j] + (len(customer_cities)) * x[i, j] <= len(customer_cities) - 1,
                    f"MTZ_{i}_{j}"
                )
    
    # 4. Tour must be connected through depot - single cycle constraint handled via MTZ
    # 5. Travel time constraint
    total_travel_time_expr = gp.quicksum(
        dist_matrix[i, j] * x[i, j] 
        for i in all_cities 
        for j in all_cities 
        if i != j
    )
    model.addConstr(total_travel_time_expr <= t0, "time_budget")

    # At least 3 cities - at least one customer needs to be visited
    model.addConstr(
        gp.quicksum(x[depot, j] for j in customer_cities) >= 1,
        "min_three_cities"
    )

    # Objective: maximize total prize
    prize_expression = gp.quicksum(
        prizes[str(j)] * x[depot, j]  # Visiting a city means going OUT from depot to that city
        for j in customer_cities
    )
    
    # We also need to add up the prizes for the visited customer cities
    # A city i is visited IF flow exits it to some other city (or enters from some other city)
    for i in customer_cities:
        prize_expression += prizes[str(i)] * gp.quicksum(x[i, j] for j in all_cities if i != j)

    model.setObjective(prize_expression, GRB.MAXIMIZE)

    # Create logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Optimize the model
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
                
    model.optimize(callback)
    
    # Extract solution
    status = model.Status
    
    if status == GRB.INF_OR_UNBD or status == GRB.UNBOUNDA:
        print("Model is infeasible or unbounded")
        return {"status": "INFEASIBLE_OR_UNBOUNDED"}
    elif status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return {"status": "INFEASIBLE"}
    elif status == GRB.CUTOFF:
        print("Solution is worse than cutoff")
        return {"status": "CUTOFF"}

    # Build the tour
    if model.SolCount > 0:
        solution_edges = []
        for i in all_cities:
            for j in all_cities:
                if i != j:
                    if x[i, j].X > 0.99:  # consider as active if greater than 0.5 due to binary
                        solution_edges.append((i, j))

        # Build the tour by traversing the active edges
        tour = []
        current_city = depot
        tour.append(current_city)
        visited_set = {depot}
        
        # Since tour is a simple cycle, keep moving till we return to depot
        while True:
            next_city_found = False
            for edge in solution_edges:
                if edge[0] == current_city and edge[1] not in visited_set:
                    current_city = edge[1]
                    tour.append(current_city)
                    visited_set.add(current_city)
                    next_city_found = True
                    break
            
            if not next_city_found:
                # Should come back to depot to complete the cycle
                for edge in solution_edges:
                    if edge[0] == current_city and edge[1] == depot:
                        tour.append(depot)  # Complete the cycle
                        break
                break
        
        # Calculate total prize actually collected
        visited_nodes = []
        total_prize_collected = 0
        for i, node in enumerate(tour):
            if node not in visited_nodes:  # First occurrence of a node
                visited_nodes.append(node)
                # Add prize only the first time we include it (in case of multiple visits due to error in logic, though shouldn't happen for TSP)
                total_prize_collected += prizes.get(str(node), 0)

        # Construct final solution according to schema
        result = {
            "objective_value": total_prize_collected,
            "visited_nodes": sorted(visited_nodes),
            "edges": [[edge[0], edge[1]] for edge in solution_edges],
            "tour": tour
        }
        
        return result
    else:
        print("No solution found")
        return {"status": "NO_SOLUTION_FOUND"}


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