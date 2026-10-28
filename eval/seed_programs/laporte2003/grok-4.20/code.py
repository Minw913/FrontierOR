import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_vertices = instance['num_vertices']
    num_markets = instance['num_markets']
    num_products = instance['num_products']
    depot = instance['depot']
    travel_costs_raw = instance['travel_costs']
    product_demands = instance['product_demands']
    product_markets = instance['product_markets']
    purchase_costs_raw = instance['purchase_costs']
    supplies_raw = instance['supplies']

    # Build cost matrix
    cost = [[0.0] * num_vertices for _ in range(num_vertices)]
    for key, val in travel_costs_raw.items():
        i, j = map(int, key.split(','))
        cost[i][j] = float(val)
        cost[j][i] = float(val)

    markets = [i for i in range(num_vertices) if i != depot]
    assert len(markets) == num_markets

    avail = [set() for _ in range(num_products)]
    for p_str, mlist in product_markets.items():
        p = int(p_str)
        for m in mlist:
            avail[p].add(int(m))

    purchase_cost = [[0.0] * num_vertices for _ in range(num_products)]
    supply = [[0] * num_vertices for _ in range(num_products)]
    for key, val in purchase_costs_raw.items():
        p_str, m_str = key.split(',')
        p = int(p_str)
        m = int(m_str)
        purchase_cost[p][m] = float(val)
    for key, val in supplies_raw.items():
        p_str, m_str = key.split(',')
        p = int(p_str)
        m = int(m_str)
        supply[p][m] = int(val)

    demands = [int(d) for d in product_demands]

    # Identify mandatory markets
    mandatory = set()
    for p in range(num_products):
        total_without = sum(supply[p][m] for m in avail[p])
        for m in avail[p]:
            if total_without - supply[p][m] < demands[p]:
                mandatory.add(m)

    start_time = time.time()
    best_obj = float('inf')
    best_solution = None

    try:
        model = gp.Model("MarketTour")
        model.setParam(GRB.Param.TimeLimit, max(5, args.time_limit - 10))
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.MIPFocus, 1)
        model.setParam(GRB.Param.Heuristics, 0.8)
        model.setParam(GRB.Param.Cuts, 2)
        model.setParam(GRB.Param.LogToConsole, 0)

        # Variables
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in markets}
        x = {}
        for i_idx, i in enumerate(range(num_vertices)):
            for j in range(i + 1, num_vertices):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        z = {}
        for p in range(num_products):
            for m in avail[p]:
                z[p, m] = model.addVar(lb=0, ub=supply[p][m], vtype=GRB.INTEGER, name=f"z_{p}_{m}")

        u = {i: model.addVar(lb=0, ub=num_markets + 1, vtype=GRB.CONTINUOUS, name=f"u_{i}") for i in markets}

        # Objective
        travel_expr = gp.quicksum(cost[i][j] * x[i, j] for i, j in x.keys())
        purchase_expr = gp.quicksum(purchase_cost[p][m] * z[p, m] for p, m in z.keys())
        model.setObjective(travel_expr + purchase_expr, GRB.MINIMIZE)

        # Degree constraints
        for i in markets:
            incident = gp.quicksum(x[min(i, j), max(i, j)] for j in range(num_vertices) if j != i)
            model.addConstr(incident == 2 * y[i], name=f"deg_{i}")

        depot_inc = gp.quicksum(x[min(depot, j), max(depot, j)] for j in range(num_vertices) if j != depot)
        model.addConstr(depot_inc == 2, name="depot_deg")

        for m in mandatory:
            model.addConstr(y[m] == 1, name=f"mand_{m}")

        # MTZ
        M_big = num_markets + 1
        for i in markets:
            for j in markets:
                if i != j:
                    e = x[min(i, j), max(i, j)]
                    model.addConstr(u[i] - u[j] + M_big * e <= M_big - 1, name=f"mtz_{i}_{j}")

        # Demand and supply linking
        for p in range(num_products):
            model.addConstr(gp.quicksum(z[p, m] for m in avail[p] if (p, m) in z) == demands[p], name=f"dem_{p}")
            for m in avail[p]:
                if (p, m) in z:
                    model.addConstr(z[p, m] <= supply[p][m] * y[m], name=f"sup_{p}_{m}")

        # At least 2 markets in total for cycle (n>=4 but visited can be small)
        model.addConstr(gp.quicksum(y[i] for i in markets) >= 2, name="min_visit")

        def callback(model, where):
            nonlocal best_obj, best_solution, logger, start_time
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj < best_obj - 1e-4:
                    best_obj = obj
                    if logger:
                        logger.log(float(obj))

        model.optimize(callback)

        if model.SolCount >= 1:
            obj_val = model.objVal
            visited = [i for i in markets if y[i].X > 0.5]

            # Extract used edges
            edge_dict = {i: [] for i in range(num_vertices)}
            for (i, j), var in x.items():
                if var.X > 0.5:
                    edge_dict[i].append(j)
                    edge_dict[j].append(i)

            # Build tour from depot
            tour = [depot]
            current = depot
            prev = None
            visited_count = len(visited) + 1
            for _ in range(visited_count):
                next_node = None
                for neigh in edge_dict[current]:
                    if neigh != prev:
                        next_node = neigh
                        break
                if next_node is None:
                    break
                tour.append(next_node)
                prev = current
                current = next_node
            if tour[-1] != depot:
                tour.append(depot)

            # If tour is invalid, build a simple cycle
            if len(tour) != len(visited) + 2 or tour[-1] != depot:
                tour = [depot] + sorted(visited) + [depot]

            tour_edges = [[min(tour[k], tour[k+1]), max(tour[k], tour[k+1])] for k in range(len(tour)-1)]

            # Purchases
            purchases = {}
            for p in range(num_products):
                purchases[p] = {}
                for m in avail[p]:
                    if (p, m) in z:
                        val = z[p, m].X
                        if val > 0.1:
                            purchases[p][m] = int(round(val))

            best_solution = {
                "objective_value": float(obj_val),
                "visited_markets": sorted([int(i) for i in visited]),
                "tour": [int(v) for v in tour],
                "tour_edges": tour_edges,
                "purchases": purchases
            }

        # Fallback heuristic if MIP failed to find solution
        if best_solution is None:
            visited_set = set(mandatory)
            for p in range(num_products):
                candidates = sorted([m for m in avail[p]], key=lambda m: purchase_cost[p][m])
                for c in candidates:
                    if c not in visited_set:
                        visited_set.add(c)
                        break
            visited_list = sorted(list(visited_set))
            if len(visited_list) < 2 and markets:
                visited_list = sorted(markets[:2])

            tour = [depot] + visited_list + [depot]
            tour_edges = [[min(tour[k], tour[k+1]), max(tour[k], tour[k+1])] for k in range(len(tour)-1)]

            purchases = {}
            for p in range(num_products):
                purchases[p] = {}
                rem = demands[p]
                candidates = sorted([m for m in avail[p] if m in visited_set], key=lambda m: purchase_cost[p][m])
                for m in candidates:
                    if rem <= 0:
                        break
                    take = min(rem, supply[p][m])
                    if take > 0:
                        purchases[p][m] = int(take)
                        rem -= take

            travel_total = sum(cost[tour[k]][tour[k+1]] for k in range(len(tour)-1))
            purch_total = sum(purchase_cost[p][m] * q for p, pdict in purchases.items() for m, q in pdict.items())
            total_obj = travel_total + purch_total

            best_solution = {
                "objective_value": float(total_obj),
                "visited_markets": [int(v) for v in visited_list],
                "tour": [int(v) for v in tour],
                "tour_edges": tour_edges,
                "purchases": purchases
            }
            if logger:
                logger.log(float(total_obj))

        with open(args.solution_path, 'w') as f:
            json.dump(best_solution, f, indent=2)

    except Exception as e:
        print("Solver error:", str(e), file=sys.stderr)
        # Minimal valid output
        fallback = {
            "objective_value": 0.0,
            "visited_markets": [],
            "tour": [depot, depot],
            "tour_edges": [[depot, depot]],
            "purchases": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()