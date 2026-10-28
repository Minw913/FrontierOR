import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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
    model.Params.MIPGap = 1e-4
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
    prize_dict = {int(k): v for k, v in prizes.items()}
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
                model.addConstr(u[i] - u[j] + (K + 1) * x[i, j] <= K, f"mtz_{i}_{j}")

    for i in nodes:
        model.addConstr(u[i] >= 1, f"u_lower_{i}")
        model.addConstr(u[i] <= K, f"u_upper_{i}")

    # Link y and u
    for i in nodes:
        model.addConstr(u[i] <= (K + 1) * y[i], f"u_link_{i}")

    # Callback for lazy constraints and logging
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    best_obj = -1.0
    incumbent_solution = None

    def callback(model, where):
        nonlocal best_obj, incumbent_solution
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                x_sol = model.cbGetSolution(x)
                y_sol = model.cbGetSolution(y)
                
                visited = [depot]
                for i in nodes:
                    if y_sol[i] > 0.5:
                        visited.append(i)
                
                # Build adjacency list from solution
                adj = {i: [] for i in visited}
                edges = []
                for i in visited:
                    for j in visited:
                        if i != j and x_sol.get((i, j), 0.0) > 0.5:
                            adj[i].append(j)
                            if i < j:
                                edges.append([int(i), int(j)])
                
                # Reconstruct tour starting and ending at depot
                tour = [depot]
                current = depot
                prev = None
                steps = 0
                max_steps = len(visited) * 2
                while steps < max_steps:
                    steps += 1
                    moved = False
                    for nxt in adj[current]:
                        if nxt != prev:
                            tour.append(nxt)
                            prev = current
                            current = nxt
                            moved = True
                            break
                    if not moved or current == depot:
                        break
                if tour[-1] != depot:
                    tour.append(depot)
                
                # Ensure we have a valid cycle
                if len(tour) < 3:
                    # Fallback: pick any cycle
                    tour = [depot]
                    if len(visited) >= 2:
                        visited_sorted = sorted([v for v in visited if v != depot])
                        tour.append(visited_sorted[0])
                        if len(visited_sorted) > 1:
                            tour.append(visited_sorted[1])
                        else:
                            tour.append(visited_sorted[0])
                    tour.append(depot)
                
                incumbent_solution = {
                    "objective_value": float(obj_val),
                    "visited_nodes": sorted([int(v) for v in visited]),
                    "edges": sorted(edges),
                    "tour": [int(v) for v in tour]
                }
                
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract best solution found
    if incumbent_solution is None and model.SolCount > 0:
        try:
            obj_val = model.objVal
            x_sol = model.getAttr('X', x)
            y_sol = model.getAttr('X', y)
            
            visited = [depot]
            for i in nodes:
                if y_sol[i] > 0.5:
                    visited.append(i)
            
            adj = {i: [] for i in visited}
            edges = []
            for i in visited:
                for j in visited:
                    if i != j:
                        val = x_sol[i, j] if (i, j) in x_sol else x_sol.get((j, i), 0.0)
                        if val > 0.5:
                            adj[i].append(j)
                            if i < j:
                                edges.append([int(i), int(j)])
            
            tour = [depot]
            current = depot
            prev = None
            steps = 0
            max_steps = len(visited) * 2
            while steps < max_steps:
                steps += 1
                moved = False
                for nxt in adj.get(current, []):
                    if nxt != prev:
                        tour.append(nxt)
                        prev = current
                        current = nxt
                        moved = True
                        break
                if not moved or current == depot:
                    break
            if tour[-1] != depot:
                tour.append(depot)
            
            incumbent_solution = {
                "objective_value": float(obj_val),
                "visited_nodes": sorted([int(v) for v in visited]),
                "edges": sorted(edges),
                "tour": [int(v) for v in tour]
            }
        except:
            pass

    # Fallback: find a feasible small tour if no solution found
    if incumbent_solution is None or incumbent_solution["objective_value"] < 1e-6:
        prize_dict = {int(k): v for k, v in prizes.items()}
        min_cost = float('inf')
        best_i, best_j = None, None
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                i = nodes[i_idx]
                j = nodes[j_idx]
                cost = tt[depot, i] + tt[i, j] + tt[j, depot]
                if cost <= t0 and cost < min_cost:
                    min_cost = cost
                    best_i = i
                    best_j = j
        
        if best_i is not None and best_j is not None:
            p = prize_dict.get(best_i, 0) + prize_dict.get(best_j, 0)
            incumbent_solution = {
                "objective_value": float(p),
                "visited_nodes": sorted([int(depot), int(best_i), int(best_j)]),
                "edges": sorted([[int(depot), int(best_i)], [int(best_i), int(best_j)], [int(best_j), int(depot)]]),
                "tour": [int(depot), int(best_i), int(best_j), int(depot)]
            }
        else:
            # No feasible tour with 3 cities
            incumbent_solution = {
                "objective_value": 0.0,
                "visited_nodes": [int(depot)],
                "edges": [],
                "tour": [int(depot), int(depot)]
            }

    # Ensure output matches schema
    final_solution = {
        "objective_value": float(incumbent_solution["objective_value"]),
        "visited_nodes": [int(v) for v in incumbent_solution["visited_nodes"]],
        "edges": [[int(a), int(b)] for a, b in incumbent_solution["edges"]],
        "tour": [int(v) for v in incumbent_solution["tour"]]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()