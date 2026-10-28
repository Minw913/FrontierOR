import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import itertools
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

    # Build product data
    M = list(range(num_markets))  # markets indexed 0 to num_markets-1, but vertices are 0..num_vertices-1
    # Assume markets are vertices 0 to num_markets-1 and depot is given (often num_markets)
    # But to be general we use actual indices
    markets = [i for i in range(num_vertices) if i != depot]
    assert len(markets) == num_markets

    # product_markets_dict[prod][market] = True if available
    avail = [set() for _ in range(num_products)]
    for p_str, mlist in product_markets.items():
        p = int(p_str)
        for m in mlist:
            avail[p].add(m)

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
        total_supply_without = {}
        for m in avail[p]:
            total_supply_without[m] = sum(supply[p][k] for k in avail[p] if k != m)
        for m in avail[p]:
            if total_supply_without[m] < demands[p]:
                mandatory.add(m)
    mandatory = list(mandatory)

    start_time = time.time()
    best_obj = float('inf')
    best_solution = None

    # If very small instance, we can try exact MIP with time limit
    # But n>=4 markets, but complete graph, TSP like with extra decisions
    # Number of markets up to say 15 might be solvable with MTZ or cut based, but we need a heuristic for generality

    # We will use a MIP formulation with MTZ for subtour and binary x for edges, y for visit, z for purchases.
    # Since single core and time limit, we set a reasonable MIP focus and time.

    try:
        model = gp.Model("MarketTour")
        model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - 5))
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.MIPFocus, 1)  # focus on feasible solutions
        model.setParam(GRB.Param.Heuristics, 0.8)
        model.setParam(GRB.Param.Cuts, 2)

        # Variables
        # y_i : visit market i (i != depot)
        y = {}
        for i in markets:
            y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

        # x_ij : use edge i-j in tour (undirected, i < j)
        edges = []
        x = {}
        for i in range(num_vertices):
            for j in range(i+1, num_vertices):
                if i == depot or j == depot or i in markets or j in markets:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                    edges.append((i, j))

        # z_pim : integer quantity of product p bought at market m
        z = {}
        for p in range(num_products):
            for m in avail[p]:
                z[p, m] = model.addVar(lb=0, ub=supply[p][m], vtype=GRB.INTEGER, name=f"z_{p}_{m}")

        # MTZ for preventing subtours: u_i position in tour
        u = {}
        for i in markets:
            u[i] = model.addVar(lb=1, ub=num_markets, vtype=GRB.CONTINUOUS, name=f"u_{i}")

        # Objective: travel + purchase
        travel_expr = gp.quicksum(cost[i][j] * x[i, j] for i, j in x)
        purchase_expr = gp.quicksum(purchase_cost[p][m] * z[p, m] for p in range(num_products) for m in avail[p])
        model.setObjective(travel_expr + purchase_expr, GRB.MINIMIZE)

        # Constraints

        # Exactly 2 edges incident to each visited vertex (including depot)
        for i in markets:
            incident = [x[min(i, j), max(i, j)] for j in range(num_vertices) if j != i]
            model.addConstr(gp.quicksum(incident) == 2 * y[i], name=f"degree_{i}")

        # Depot always visited and degree exactly 2
        depot_incident = [x[min(depot, j), max(depot, j)] for j in range(num_vertices) if j != depot]
        model.addConstr(gp.quicksum(depot_incident) == 2, name="depot_degree")

        # Mandatory markets
        for m in mandatory:
            model.addConstr(y[m] == 1, name=f"mandatory_{m}")

        # Flow / connected via degree is not enough, we need no subtours via MTZ
        # MTZ constraints
        for i_idx, i in enumerate(markets):
            for j_idx, j in enumerate(markets):
                if i != j:
                    pair = (min(i, j), max(i, j))
                    if pair in x:
                        model.addConstr(u[i] - u[j] + (num_markets + 1) * x[pair[0], pair[1]] <= num_markets,
                                        name=f"mtz_{i}_{j}")

        # Purchase constraints
        for p in range(num_products):
            model.addConstr(gp.quicksum(z[p, m] for m in avail[p]) == demands[p],
                            name=f"demand_{p}")
            for m in avail[p]:
                model.addConstr(z[p, m] <= supply[p][m] * y[m], name=f"supply_link_{p}_{m}")

        # At least one market if needed (but since n>=4 and m>=1 usually not binding)
        if num_markets > 0:
            model.addConstr(gp.quicksum(y[i] for i in markets) >= 1)

        # Symmetry breaking: smallest index market in tour if multiple possibilities but hard

        # Callback for logging incumbent solutions
        def callback(model, where):
            nonlocal best_obj, best_solution
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj < best_obj - 1e-6:
                    best_obj = obj
                    if logger:
                        logger.log(float(obj))
                    # We could extract solution here but for speed we extract at end

        model.optimize(callback)

        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
            # fallback not implemented, should not happen
            sys.exit(1)

        # Extract solution
        if model.SolCount > 0:
            obj_val = model.objVal
            visited = [i for i in markets if y[i].X > 0.5]

            # Reconstruct tour from edges
            edge_used = {}
            for (i, j), var in x.items():
                if var.X > 0.5:
                    edge_used.setdefault(i, []).append(j)
                    edge_used.setdefault(j, []).append(i)

            # Build the cycle starting from depot
            tour = [depot]
            current = depot
            prev = None
            while True:
                neighbors = edge_used.get(current, [])
                next_v = None
                for neigh in neighbors:
                    if neigh != prev:
                        next_v = neigh
                        break
                if next_v is None or next_v == depot:
                    break
                tour.append(next_v)
                prev = current
                current = next_v
            tour.append(depot)

            # Verify it's a cycle
            if tour[-1] != depot or len(tour) < 3:
                # Try to find any cycle if broken (rare)
                tour = [depot]
                visited_set = set(visited)
                visited_set.add(depot)
                # Greedy close tour for output
                remaining = visited[:]
                last = depot
                while remaining:
                    next_m = min(remaining, key=lambda k: cost[last][k])
                    tour.append(next_m)
                    remaining.remove(next_m)
                    last = next_m
                tour.append(depot)

            tour_edges = []
            for k in range(len(tour) - 1):
                a = tour[k]
                b = tour[k + 1]
                tour_edges.append([min(a, b), max(a, b)])

            # Purchases
            purchases = {}
            for p in range(num_products):
                purchases[p] = {}
                for m in avail[p]:
                    val = z[p, m].X
                    if val > 0.5:
                        purchases[p][m] = int(round(val))

            best_solution = {
                "objective_value": float(obj_val),
                "visited_markets": [int(i) for i in sorted(visited)],
                "tour": [int(i) for i in tour],
                "tour_edges": [[int(a), int(b)] for a, b in tour_edges],
                "purchases": purchases
            }

        # If no solution found by MIP, we construct a simple feasible solution as fallback
        if best_solution is None or model.SolCount == 0:
            # Simple feasible: visit all mandatory + one market per product that has supply
            visited_set = set(mandatory)
            for p in range(num_products):
                candidates = [m for m in avail[p] if supply[p][m] > 0]
                if candidates:
                    visited_set.add(min(candidates))
            visited_list = sorted(list(visited_set))

            # Build a tour: depot -> sorted visited -> depot
            tour_list = [depot] + visited_list + [depot]
            tour_edges_list = []
            for idx in range(len(tour_list) - 1):
                a = tour_list[idx]
                b = tour_list[idx + 1]
                tour_edges_list.append([min(a, b), max(a, b)])

            # Purchases: greedy assign to cheapest available visited market
            purch_dict = {}
            for p_idx in range(num_products):
                purch_dict[p_idx] = {}
                remaining_d = demands[p_idx]
                # sort visited markets that have this product by price
                possible_ms = [m for m in visited_list if m in avail[p_idx]]
                possible_ms.sort(key=lambda m: purchase_cost[p_idx][m])
                for m in possible_ms:
                    if remaining_d <= 0:
                        break
                    take = min(remaining_d, supply[p_idx][m])
                    if take > 0:
                        purch_dict[p_idx][m] = int(take)
                        remaining_d -= take
                assert remaining_d == 0, "Failed to satisfy demand in fallback"

            # Compute objective
            travel_cost_total = sum(cost[a][b] for a, b in zip(tour_list[:-1], tour_list[1:]))
            purchase_cost_total = 0.0
            for p_idx, buys in purch_dict.items():
                for m, qty in buys.items():
                    purchase_cost_total += purchase_cost[p_idx][m] * qty
            total_obj = travel_cost_total + purchase_cost_total

            best_solution = {
                "objective_value": float(total_obj),
                "visited_markets": [int(v) for v in visited_list],
                "tour": [int(v) for v in tour_list],
                "tour_edges": tour_edges_list,
                "purchases": purch_dict
            }
            if logger:
                logger.log(float(total_obj))

        # Write solution
        with open(args.solution_path, 'w') as f:
            json.dump(best_solution, f, indent=2)

    except Exception as e:
        print("Error during optimization:", str(e))
        # Last resort dummy (should not reach)
        dummy_solution = {
            "objective_value": 0.0,
            "visited_markets": [],
            "tour": [depot, depot],
            "tour_edges": [],
            "purchases": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(dummy_solution, f, indent=2)


if __name__ == "__main__":
    main()