import argparse
import json
import sys
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    n = inst['n']
    depot = inst['depot']
    t0 = inst['t0']
    
    prizes = {int(k): float(v) for k, v in inst['prizes'].items()}
    if depot not in prizes:
        prizes[depot] = 0.0

    max_node = max(prizes.keys()) if prizes else depot
    for k in inst['travel_times']:
        parts = k.strip('()').split(',')
        i, j = int(parts[0].strip()), int(parts[1].strip())
        max_node = max(max_node, i, j)
    N = max_node + 1

    dist = [[0] * N for _ in range(N)]
    for k, v in inst['travel_times'].items():
        parts = k.strip('()').split(',')
        i, j = int(parts[0].strip()), int(parts[1].strip())
        dist[i][j] = v
        dist[j][i] = v

    try:
        import gurobipy as gp
        from gurobipy import GRB
        GUROBI_AVAILABLE = True
    except ImportError:
        GUROBI_AVAILABLE = False

    if GUROBI_AVAILABLE:
        env = gp.Env(empty=True)
        env.setParam('Seed', 0)
        env.setParam('Threads', 1)
        env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model(env=env)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('TimeLimit', max(1, args.time_limit - 1))

        cities = list(range(N))
        nondepot = [i for i in cities if i != depot]

        y = {i: model.addVar(vtype=GRB.BINARY, name=f'y_{i}') for i in cities}
        x = {}
        for i in cities:
            for j in cities:
                if i != j and dist[i][j] > 0: # Add variables only for valid edges
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        f = {}
        for i in cities:
            for j in cities:
                if i != j and dist[i][j] > 0:
                    f[i, j] = model.addVar(lb=0, name=f'f_{i}_{j}')

        model.setObjective(gp.quicksum(prizes[i] * y[i] for i in cities), GRB.MAXIMIZE)

        model.addConstr(y[depot] == 1)

        for i in cities:
            model.addConstr(gp.quicksum(x[i, j] for j in cities if j != i and (i, j) in x) == y[i])
            model.addConstr(gp.quicksum(x[j, i] for j in cities if j != i and (j, i) in x) == y[i])

        model.addConstr(
            gp.quicksum(dist[i][j] * x[i, j] for i in cities for j in cities if i != j and (i, j) in x) <= t0
        )
        model.addConstr(gp.quicksum(y[i] for i in cities) >= 3)

        model.addConstr(
            gp.quicksum(f[depot, j] for j in cities if j != depot and (depot, j) in f)
            - gp.quicksum(f[j, depot] for j in cities if j != depot and (j, depot) in f)
            == gp.quicksum(y[i] for i in nondepot)
        )
        for i in nondepot:
            model.addConstr(
                gp.quicksum(f[j, i] for j in cities if j != i and (j, i) in f)
                - gp.quicksum(f[i, j] for j in cities if j != i and (i, j) in f)
                == y[i]
            )

        for i in cities:
            for j in cities:
                if i != j and (i, j) in f:
                    model.addConstr(f[i, j] <= (N - 1) * x[i, j])

        best_obj = [0.0]
        best_sol = [{
            "objective_value": 0.0,
            "visited_nodes": [depot],
            "edges": [],
            "tour": [depot, depot]
        }]

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val > best_obj[0] + 1e-9:
                    best_obj[0] = obj_val
                    yvals = model.cbGetSolution([y[i] for i in cities])
                    xvals = {}
                    for key in x:
                        xvals[key] = model.cbGetSolution(x[key])
                    visited = [cities[k] for k in range(len(cities)) if yvals[k] > 0.5]
                    edges = []
                    succ = {}
                    for i in cities:
                        for j in cities:
                            if i != j and (i, j) in xvals and xvals[(i, j)] > 0.5:
                                edges.append([i, j])
                                succ[i] = j
                    tour = [depot]
                    nxt = succ.get(depot)
                    guard = 0
                    while nxt is not None and nxt != depot and guard < N + 2:
                        tour.append(nxt)
                        nxt = succ.get(nxt)
                        guard += 1
                    tour.append(depot)
                    sol = {
                        "objective_value": float(obj_val),
                        "visited_nodes": visited,
                        "edges": edges,
                        "tour": tour
                    }
                    best_sol[0] = sol
                    if logger:
                        logger.log_solution(obj_val, sol)

        model.optimize(callback)

        if model.SolCount > 0 and model.ObjVal > best_obj[0] + 1e-9:
            visited = [i for i in cities if y[i].X > 0.5]
            edges = []
            succ = {}
            for i in cities:
                for j in cities:
                    if i != j and (i, j) in x and x[i, j].X > 0.5:
                        edges.append([i, j])
                        succ[i] = j
            tour = [depot]
            cur = depot
            nxt = succ.get(depot)
            guard = 0
            while nxt is not None and nxt != depot and guard < N + 2:
                tour.append(nxt)
                nxt = succ.get(nxt)
                guard += 1
            tour.append(depot)
            obj = sum(prizes[i] for i in visited)
            best_sol[0] = {
                "objective_value": float(obj),
                "visited_nodes": visited,
                "edges": edges,
                "tour": tour
            }

        with open(args.solution_path, 'w') as fout:
            json.dump(best_sol[0], fout, indent=2)

    else:
        start_time = time.time()
        
        def calc_time(tour):
            return sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))
            
        def calc_prize(tour):
            return sum(prizes.get(i, 0.0) for i in tour[1:-1])
            
        def log_sol(tour):
            obj = calc_prize(tour)
            sol = {
                "objective_value": float(obj),
                "visited_nodes": tour[1:-1],
                "edges": [[tour[i], tour[i+1]] for i in range(len(tour)-1)],
                "tour": tour
            }
            if logger:
                logger.log_solution(obj, sol)
            return obj, sol

        best_tour = [depot, depot]
        best_prize = 0.0
        
        for i in range(1, N):
            if i == depot: continue
            if i not in prizes: continue
            for j in range(i+1, N):
                if j == depot: continue
                if j not in prizes: continue
                t = dist[depot][i] + dist[i][j] + dist[j][depot]
                if t <= t0:
                    p = prizes[i] + prizes[j]
                    if p > best_prize:
                        best_prize = p
                        best_tour = [depot, i, j, depot]
                        
        current_best_obj, current_best_sol = log_sol(best_tour)
        
        while time.time() - start_time < args.time_limit - 1:
            tour = best_tour[:]
            current_time = calc_time(tour)
            current_prize = calc_prize(tour)
            unvisited = [i for i in range(1, N) if i not in tour and i in prizes]
            
            improved = True
            while improved and time.time() - start_time < args.time_limit - 1:
                improved = False
                
                best_delta_prize = -1
                best_u = -1
                best_pos = -1
                for u in unvisited:
                    for i in range(len(tour)-1):
                        delta_time = dist[tour[i]][u] + dist[u][tour[i+1]] - dist[tour[i]][tour[i+1]]
                        if current_time + delta_time <= t0:
                            if prizes[u] > best_delta_prize:
                                best_delta_prize = prizes[u]
                                best_u = u
                                best_pos = i
                if best_u != -1:
                    delta_time = dist[tour[best_pos]][best_u] + dist[best_u][tour[best_pos+1]] - dist[tour[best_pos]][tour[best_pos+1]]
                    tour.insert(best_pos+1, best_u)
                    current_time += delta_time
                    current_prize += prizes[best_u]
                    unvisited.remove(best_u)
                    improved = True
                    continue
                    
                for i in range(1, len(tour)-1):
                    for u in unvisited:
                        delta_time = dist[tour[i-1]][u] + dist[u][tour[i+1]] - dist[tour[i-1]][tour[i]] - dist[tour[i]][tour[i+1]]
                        if current_time + delta_time <= t0:
                            if prizes[u] > prizes[tour[i]] + 1e-6:
                                old_node = tour[i]
                                tour[i] = u
                                current_time += delta_time
                                current_prize += prizes[u] - prizes[old_node]
                                unvisited.remove(u)
                                unvisited.append(old_node)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue
                    
                for i in range(1, len(tour)-2):
                    for j in range(i+1, len(tour)-1):
                        delta = dist[tour[i-1]][tour[j]] + dist[tour[i]][tour[j+1]] - dist[tour[i-1]][tour[i]] - dist[tour[j]][tour[j+1]]
                        if delta < -1e-6:
                            tour[i:j+1] = reversed(tour[i:j+1])
                            current_time += delta
                            improved = True
                if improved:
                    continue
                    
            if current_prize > best_prize + 1e-6:
                best_prize = current_prize
                best_tour = tour[:]
                current_best_obj, current_best_sol = log_sol(best_tour)
                
            if len(tour) > 3:
                k = random.randint(1, min(3, len(tour)-3))
                for _ in range(k):
                    idx = random.randint(1, len(tour)-2)
                    removed = tour.pop(idx)
                    unvisited.append(removed)
            else:
                tour = [depot, depot]
                unvisited = [i for i in range(1, N) if i != depot and i in prizes]
                random.shuffle(unvisited)
                for u in list(unvisited):
                    for i in range(len(tour)-1):
                        delta_time = dist[tour[i]][u] + dist[u][tour[i+1]] - dist[tour[i]][tour[i+1]]
                        if current_time + delta_time <= t0:
                            tour.insert(i+1, u)
                            unvisited.remove(u)
                            break

        with open(args.solution_path, 'w') as fout:
            json.dump(current_best_sol, fout, indent=2)

if __name__ == '__main__':
    main()