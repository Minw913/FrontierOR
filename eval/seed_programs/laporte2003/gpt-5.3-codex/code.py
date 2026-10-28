import argparse
import json
import math
import time
from collections import defaultdict, deque
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_pair_key(key):
    a, b = key.split(",")
    return int(a), int(b)


def get_dict_value(dct, *keys, default=None):
    """Try multiple key formats for robustness."""
    for k in keys:
        if k in dct:
            return dct[k]
    return default


def build_data(instance):
    num_vertices = int(instance["num_vertices"])
    depot = int(instance["depot"])
    vertices = list(range(num_vertices))
    markets = [v for v in vertices if v != depot]

    num_products = int(instance["num_products"])
    demands = [float(x) for x in instance["product_demands"]]

    # Travel costs
    travel_costs = {}
    for k, v in instance["travel_costs"].items():
        i, j = parse_pair_key(k)
        if i > j:
            i, j = j, i
        travel_costs[(i, j)] = float(v)

    # Product markets availability
    product_markets_raw = instance["product_markets"]
    product_markets = {}
    for p in range(num_products):
        vals = get_dict_value(product_markets_raw, str(p), p, default=[])
        product_markets[p] = [int(x) for x in vals]

    # Purchase costs and supplies
    purchase_costs_raw = instance["purchase_costs"]
    supplies_raw = instance["supplies"]

    purchase_costs = {}
    supplies = {}
    for p in range(num_products):
        for i in product_markets[p]:
            c = get_dict_value(
                purchase_costs_raw,
                f"{p},{i}",
                f"{int(p)},{int(i)}",
                default=None
            )
            s = get_dict_value(
                supplies_raw,
                f"{p},{i}",
                f"{int(p)},{int(i)}",
                default=None
            )
            if c is None or s is None:
                raise ValueError(f"Missing purchase_cost or supply for product {p}, market {i}")
            purchase_costs[(p, i)] = float(c)
            supplies[(p, i)] = float(s)

    return {
        "num_vertices": num_vertices,
        "depot": depot,
        "vertices": vertices,
        "markets": markets,
        "num_products": num_products,
        "demands": demands,
        "travel_costs": travel_costs,
        "product_markets": product_markets,
        "purchase_costs": purchase_costs,
        "supplies": supplies,
    }


def compute_mandatory_markets(data):
    mandatory = set()
    demands = data["demands"]
    product_markets = data["product_markets"]
    supplies = data["supplies"]

    for p, d in enumerate(demands):
        mkts = product_markets[p]
        total_supply = sum(supplies[(p, i)] for i in mkts)
        # Feasibility should hold by assumption, but guard anyway
        if total_supply + 1e-9 < d:
            continue
        for i in mkts:
            others = total_supply - supplies[(p, i)]
            if others + 1e-9 < d:
                mandatory.add(i)
    return mandatory


def connected_components(selected_vertices, selected_edges):
    adj = {v: [] for v in selected_vertices}
    for i, j in selected_edges:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    comps = []
    seen = set()
    for v in selected_vertices:
        if v in seen:
            continue
        stack = [v]
        seen.add(v)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    return comps


def solve_instance(instance, time_limit, logger=None):
    data = build_data(instance)
    vertices = data["vertices"]
    markets = data["markets"]
    depot = data["depot"]
    num_products = data["num_products"]

    mandatory_markets = compute_mandatory_markets(data)

    model = gp.Model("PurchasingTour")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.LazyConstraints = 1

    # Edge variables
    edge_list = [(i, j) for i in vertices for j in vertices if i < j]
    x = model.addVars(edge_list, vtype=GRB.BINARY, name="x")

    # Visit variables for markets
    y = model.addVars(markets, vtype=GRB.BINARY, name="y")

    # Purchase variables
    q = {}
    for p in range(num_products):
        for i in data["product_markets"][p]:
            q[(p, i)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{p}_{i}")

    model.update()

    # Mandatory markets
    for i in mandatory_markets:
        if i in y:
            y[i].LB = 1.0

    # Degree constraints for markets
    for i in markets:
        inc = []
        for j in vertices:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            inc.append(x[(a, b)])
        model.addConstr(gp.quicksum(inc) == 2 * y[i], name=f"deg_{i}")

    # Degree for depot (always in tour)
    inc_depot = []
    for j in vertices:
        if j == depot:
            continue
        a, b = (depot, j) if depot < j else (j, depot)
        inc_depot.append(x[(a, b)])
    model.addConstr(gp.quicksum(inc_depot) == 2, name="deg_depot")

    # Product demand satisfaction
    for p in range(num_products):
        model.addConstr(
            gp.quicksum(q[(p, i)] for i in data["product_markets"][p]) == data["demands"][p],
            name=f"demand_{p}"
        )

    # Supply & activation
    for (p, i), var in q.items():
        model.addConstr(var <= data["supplies"][(p, i)] * y[i], name=f"supply_{p}_{i}")

    # Objective
    travel_obj = gp.quicksum(data["travel_costs"][(i, j)] * x[(i, j)] for (i, j) in edge_list)
    purchase_obj = gp.quicksum(data["purchase_costs"][(p, i)] * q[(p, i)] for (p, i) in q.keys())
    model.setObjective(travel_obj + purchase_obj, GRB.MINIMIZE)

    # Attach for callback
    model._x = x
    model._y = y
    model._vertices = vertices
    model._markets = markets
    model._depot = depot
    model._logger = logger
    model._best_logged = float("inf")

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            xsol = m.cbGetSolution(m._x)
            ysol = m.cbGetSolution(m._y)

            selected_vertices = [m._depot] + [i for i in m._markets if ysol[i] > 0.5]
            selected_set = set(selected_vertices)

            selected_edges = []
            for (i, j), val in xsol.items():
                if val > 0.5 and i in selected_set and j in selected_set:
                    selected_edges.append((i, j))

            comps = connected_components(selected_vertices, selected_edges)

            if len(comps) > 1:
                for comp in comps:
                    if m._depot in comp:
                        continue
                    if len(comp) == 0:
                        continue
                    comp_set = set(comp)
                    lhs_terms = []
                    comp_list = sorted(comp_set)
                    for idx_a in range(len(comp_list)):
                        a = comp_list[idx_a]
                        for idx_b in range(idx_a + 1, len(comp_list)):
                            b = comp_list[idx_b]
                            aa, bb = (a, b) if a < b else (b, a)
                            lhs_terms.append(m._x[(aa, bb)])
                    # eliminate subtour not containing depot
                    m.cbLazy(gp.quicksum(lhs_terms) <= gp.quicksum(m._y[i] for i in comp_list) - 1)
                return

            # Log only connected incumbent (feasible for lazy SECs)
            if m._logger is not None:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < m._best_logged:
                    m._best_logged = obj
                    m._logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount == 0:
        return {
            "objective_value": float("inf"),
            "visited_markets": [],
            "tour": [depot, depot],
            "tour_edges": [],
            "purchases": {str(p): {} for p in range(num_products)},
        }

    # Extract solution
    x_val = model.getAttr("X", x)
    y_val = model.getAttr("X", y)
    q_val = model.getAttr("X", q)

    visited_markets = sorted([i for i in markets if y_val[i] > 0.5])

    tour_edges = []
    for (i, j), v in x_val.items():
        if v > 0.5:
            tour_edges.append([int(i), int(j)])

    # Reconstruct ordered tour
    selected_vertices = [depot] + visited_markets
    adj = {v: [] for v in selected_vertices}
    for i, j in tour_edges:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    tour = [depot]
    prev = None
    cur = depot
    max_steps = len(selected_vertices) + 5
    for _ in range(max_steps):
        nbrs = adj[cur]
        if not nbrs:
            break
        if prev is None:
            nxt = nbrs[0]
        else:
            nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
            if nxt is None:
                break
        tour.append(nxt)
        if nxt == depot:
            break
        prev, cur = cur, nxt

    if tour[-1] != depot:
        tour.append(depot)

    purchases = {}
    for p in range(num_products):
        inner = {}
        for i in data["product_markets"][p]:
            val = q_val[(p, i)]
            if val > 1e-6:
                inner[str(i)] = float(val)
        purchases[str(p)] = inner

    objective_value = float(model.ObjVal)

    return {
        "objective_value": objective_value,
        "visited_markets": [int(i) for i in visited_markets],
        "tour": [int(v) for v in tour],
        "tour_edges": [[int(a), int(b)] for a, b in tour_edges],
        "purchases": purchases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()