import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    n = inst['n']
    depot = inst['depot']
    t0 = inst['t0']
    prizes = {int(k): float(v) for k, v in inst['prizes'].items()}
    travel_times = {}
    for k, v in inst['travel_times'].items():
        i, j = k.strip('()').split(',')
        i, j = int(i), int(j)
        travel_times[(i, j)] = v
        travel_times[(j, i)] = v

    def t(i, j):
        if i == j:
            return 0
        return travel_times[(i, j)]

    cities = list(prizes.keys())
    nondepot = [i for i in cities if i != depot]

    # Fallback feasible solution: shortest 3-city tour (depot + 2 cities).
    def build_fallback():
        best = None
        for i in nondepot:
            for j in nondepot:
                if i < j:
                    tt = t(depot, i) + t(i, j) + t(j, depot)
                    if tt <= t0:
                        if best is None or prizes[i] + prizes[j] > best[0]:
                            best = (prizes[i] + prizes[j], i, j)
        if best is not None:
            obj, ci, cj = best
            return {
                "objective_value": float(obj),
                "visited_nodes": [depot, ci, cj],
                "edges": [[depot, ci], [ci, cj], [cj, depot]],
                "tour": [depot, ci, cj, depot],
            }
        return {
            "objective_value": 0.0,
            "visited_nodes": [depot],
            "edges": [],
            "tour": [depot],
        }

    best_solution = build_fallback()
    best_obj = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_obj, best_solution)

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('TimeLimit', max(1, args.time_limit - 1))

    y = {i: model.addVar(vtype=GRB.BINARY, name=f'y_{i}') for i in cities}
    x = {}
    for i in cities:
        for j in cities:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
    f = {}
    for i in cities:
        for j in cities:
            if i != j:
                f[i, j] = model.addVar(lb=0, name=f'f_{i}_{j}')

    model.setObjective(gp.quicksum(prizes[i] * y[i] for i in cities), GRB.MAXIMIZE)

    model.addConstr(y[depot] == 1)

    for i in cities:
        model.addConstr(gp.quicksum(x[i, j] for j in cities if j != i) == y[i])
        model.addConstr(gp.quicksum(x[j, i] for j in cities if j != i) == y[i])

    model.addConstr(
        gp.quicksum(t(i, j) * x[i, j] for i in cities for j in cities if i != j) <= t0
    )
    model.addConstr(gp.quicksum(y[i] for i in cities) >= 3)

    model.addConstr(
        gp.quicksum(f[depot, j] for j in cities if j != depot)
        - gp.quicksum(f[j, depot] for j in cities if j != depot)
        == gp.quicksum(y[i] for i in nondepot)
    )
    for i in nondepot:
        model.addConstr(
            gp.quicksum(f[j, i] for j in cities if j != i)
            - gp.quicksum(f[i, j] for j in cities if j != i)
            == y[i]
        )

    for i in cities:
        for j in cities:
            if i != j:
                model.addConstr(f[i, j] <= (n - 1) * x[i, j])

    # Lower bound from fallback
    model.addConstr(gp.quicksum(prizes[i] * y[i] for i in cities) >= best_obj)

    def extract_sol():
        visited = [i for i in cities if y[i].X > 0.5]
        edges = []
        succ = {}
        for i in cities:
            for j in cities:
                if i != j and x[i, j].X > 0.5:
                    edges.append([i, j])
                    succ[i] = j
        tour = [depot]
        cur = depot
        nxt = succ.get(depot)
        guard = 0
        while nxt is not None and nxt != depot and guard < n + 2:
            tour.append(nxt)
            nxt = succ.get(nxt)
            guard += 1
        tour.append(depot)
        obj = sum(prizes[i] for i in visited)
        return {
            "objective_value": float(obj),
            "visited_nodes": visited,
            "edges": edges,
            "tour": tour,
        }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_obj + 1e-9:
                nonlocal_best[0] = obj_val
                yvals = model.cbGetSolution([y[i] for i in cities])
                xvals = {}
                for i in cities:
                    for j in cities:
                        if i != j:
                            xvals[(i, j)] = model.cbGetSolution(x[i, j])
                visited = [cities[k] for k in range(len(cities)) if yvals[k] > 0.5]
                edges = []
                succ = {}
                for i in cities:
                    for j in cities:
                        if i != j and xvals[(i, j)] > 0.5:
                            edges.append([i, j])
                            succ[i] = j
                tour = [depot]
                nxt = succ.get(depot)
                guard = 0
                while nxt is not None and nxt != depot and guard < n + 2:
                    tour.append(nxt)
                    nxt = succ.get(nxt)
                    guard += 1
                tour.append(depot)
                sol = {
                    "objective_value": float(obj_val),
                    "visited_nodes": visited,
                    "edges": edges,
                    "tour": tour,
                }
                nonlocal_sol[0] = sol
                if logger:
                    logger.log_solution(obj_val, sol)

    nonlocal_best = [best_obj]
    nonlocal_sol = [best_solution]

    model.optimize(callback)

    if model.SolCount > 0 and model.ObjVal > best_obj + 1e-9:
        best_solution = extract_sol()
        best_obj = best_solution["objective_value"]
    elif nonlocal_sol[0]["objective_value"] > best_obj + 1e-9:
        best_solution = nonlocal_sol[0]
        best_obj = best_solution["objective_value"]

    with open(args.solution_path, 'w') as fout:
        json.dump(best_solution, fout, indent=2)


if __name__ == '__main__':
    main()