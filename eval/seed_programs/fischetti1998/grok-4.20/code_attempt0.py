import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    depot = instance['depot']
    t0 = instance['t0']
    prizes = instance['prizes']
    travel_times = instance['travel_times']

    cities = list(range(1, n+1))
    nodes = [i for i in cities if i != depot]
    K = len(nodes)

    # Build symmetric travel time matrix
    tt = {}
    for i in cities:
        for j in cities:
            if i == j:
                tt[i,j] = 0
            elif i < j:
                key = f"({i},{j})"
                tt[i,j] = travel_times.get(key, travel_times.get(f"({j},{i})", 0))
            else:
                key = f"({j},{i})"
                tt[i,j] = travel_times.get(key, travel_times.get(f"({i},{j})", 0))

    # Create Gurobi model
    model = gp.Model("OP_TSP")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 0.01
    model.Params.LazyConstraints = 1

    # Variables
    x = {}
    for i in cities:
        for j in cities:
            if i != j:
                x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    y = {}
    for i in nodes:
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    u = {}
    for i in cities:
        u[i] = model.addVar(lb=0, ub=K+1, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    # Objective: maximize total prize
    obj = gp.quicksum(prizes.get(str(i), 0) * y[i-1 if i > depot else i] wait, better way:
    prize_dict = {int(k): v for k,v in prizes.items()}
    objective = gp.quicksum(prize_dict.get(i, 0) * y[i] for i in nodes)
    model.setObjective(objective, GRB.MAXIMIZE)

    # At least 3 cities: depot + at least 2 others
    model.addConstr(gp.quicksum(y[i] for i in nodes) >= 2, "min_cities")

    # Degree constraints
    for i in nodes:
        model.addConstr(gp.quicksum(x[i,j] for j in cities if j != i) == 2 * y[i], f"degree_in_{i}")
        model.addConstr(gp.quicksum(x[j,i] for j in cities if j != i) == 2 * y[i], f"degree_out_{i}")

    # Depot is always visited
    model.addConstr(gp.quicksum(x[depot,j] for j in cities if j != depot) == 2, "depot_degree")
    model.addConstr(gp.quicksum(x[j,depot] for j in cities if j != depot) == 2, "depot_degree2")

    # Time budget
    total_time = gp.quicksum(tt[i,j] * x[i,j] for i in cities for j in cities if i != j)
    model.addConstr(total_time <= t0, "time_budget")

    # MTZ subtour elimination
    for i in nodes:
        for j in nodes:
            if i != j:
                model.addConstr(u[i] - u[j] + (K+1)*x[i,j] <= K, f"mtz_{i}_{j}")

    for i in nodes:
        model.addConstr(u[i] >= 1, f"u_lower_{i}")
        model.addConstr(u[i] <= K, f"u_upper_{i}")

    # Link y and u (if visited then u between 1 and K)
    for i in nodes:
        model.addConstr(u[i] <= (K+1) * y[i], f"u_link_{i}")

    # Callback for lazy constraints and logging
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    best_obj = -1
    incumbent_solution = None

    def callback(model, where):
        nonlocal best_obj, incumbent_solution
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                # Get solution
                x_sol = model.cbGetSolution(x)
                y_sol = model.cbGetSolution(y)
                
                visited = [depot]
                for i in nodes:
                    if y_sol[i] > 0.5:
                        visited.append(i)
                
                # Reconstruct tour using greedy following
                tour_edges = []
                for i in cities:
                    for j in cities:
                        if i != j and x_sol[i,j] > 0.5:
                            tour_edges.append([i,j])
                
                # Build tour sequence
                tour = [depot]
                current = depot
                visited_set = set(visited)
                remaining = set(visited) - {depot}
                
                while remaining:
                    found = False
                    for j in cities:
                        if j in remaining and x_sol[current, j] > 0.5:
                            tour.append(j)
                            remaining.remove(j)
                            current = j
                            found = True
                            break
                    if not found:
                        # Try reverse direction if needed
                        for j in cities:
                            if j in remaining and x_sol[j, current] > 0.5:
                                # This would be going backward, need different reconstruction
                                break
                tour.append(depot)
                
                # Validate tour length
                if len(tour) != len(visited) + 1:
                    # Fallback: use a simple cycle from x_sol
                    adj = {i: [] for i in visited}
                    for i in visited:
                        for j in visited:
                            if i != j and x_sol.get((i,j), 0) > 0.5:
                                adj[i].append(j)
                    
                    # Start from depot and follow
                    tour = [depot]
                    current = depot
                    prev = None
                    while True:
                        next_city = None
                        for neigh in adj[current]:
                            if neigh != prev:
                                next_city = neigh
                                break
                        if next_city is None or next_city == depot:
                            break
                        tour.append(next_city)
                        prev = current
                        current = next_city
                    tour.append(depot)
                
                # Update best
                best_obj = obj_val
                incumbent_solution = {
                    "objective_value": float(obj_val),
                    "visited_nodes": sorted([int(v) for v in visited]),
                    "edges": sorted([sorted([int(a),int(b)]) for a,b in tour_edges if x_sol[a,b]>0.5]),
                    "tour": [int(v) for v in tour]
                }
                
                if logger:
                    logger.log(float(obj_val))
                
                # Provide solution to MIP
                if incumbent_solution:
                    model.cbSetSolution([y[i] for i in nodes] + [x[i,j] for i in cities for j in cities if i!=j], 
                                      [y_sol[i] for i in nodes] + [x_sol[i,j] for i in cities for j in cities if i!=j])
                    model.cbUseSolution()

    # Set callback
    model.optimize(callback)

    # After optimization, get best solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and model.SolCount > 0:
        if incumbent_solution is None:
            # Extract final solution if callback didn't catch best
            obj_val = model.objVal
            x_sol = model.getAttr('X', x)
            y_sol = model.getAttr('X', y)
            
            visited = [depot]
            for i in nodes:
                if y_sol[i] > 0.5:
                    visited.append(i)
            
            # Build edges and tour
            edges = []
            adj = {i:[] for i in visited}
            for i in visited:
                for j in visited:
                    if i < j and x_sol.get((i,j),0) > 0.5:
                        edges.append([int(i),int(j)])
                        adj[i].append(j)
                        adj[j].append(i)
                    elif j < i and x_sol.get((j,i),0) > 0.5:
                        edges.append([int(j),int(i)])
                        adj[i].append(j)
                        adj[j].append(i)
            
            # Build Euler tour from adjacency
            tour = [depot]
            current = depot
            prev = None
            visited_copy = set(visited)
            while len(tour) < len(visited) + 1:
                moved = False
                for nxt in adj[current]:
                    if nxt != prev:
                        tour.append(nxt)
                        prev = current
                        current = nxt
                        moved = True
                        break
                if not moved:
                    break
            if tour[-1] != depot:
                tour.append(depot)
            
            incumbent_solution = {
                "objective_value": float(obj_val),
                "visited_nodes": sorted([int(v) for v in visited]),
                "edges": sorted(edges),
                "tour": [int(v) for v in tour]
            }
    else:
        # No solution found, return minimal feasible tour if possible
        # Find two closest cities to depot within time
        min_tour = None
        min_prize = 0
        best_two = None
        min_cost = float('inf')
        
        for i in nodes:
            for j in nodes:
                if i < j:
                    cost = tt[depot,i] + tt[i,j] + tt[j,depot]
                    if cost <= t0 and cost < min_cost:
                        min_cost = cost
                        best_two = (i,j)
                        min_prize = prize_dict.get(i,0) + prize_dict.get(j,0)
        
        if best_two:
            i, j = best_two
            incumbent_solution = {
                "objective_value": float(min_prize),
                "visited_nodes": sorted([int(depot), int(i), int(j)]),
                "edges": sorted([[int(depot),int(i)],[int(i),int(j)],[int(j),int(depot)]]),
                "tour": [int(depot), int(i), int(j), int(depot)]
            }
        else:
            # Trivial case - just depot (but problem requires at least 3 cities)
            # If not possible, return empty but valid structure
            incumbent_solution = {
                "objective_value": 0.0,
                "visited_nodes": [int(depot)],
                "edges": [],
                "tour": [int(depot), int(depot)]
            }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(incumbent_solution, f, indent=2)

if __name__ == "__main__":
    main()