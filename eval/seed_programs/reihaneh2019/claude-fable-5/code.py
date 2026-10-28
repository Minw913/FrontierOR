import json
import math
import time
import argparse

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", type=str, required=True)
    p.add_argument("--solution_path", type=str, required=True)
    p.add_argument("--time_limit", type=int, default=300)
    p.add_argument("--log_path", type=str, default=None)
    return p.parse_args()


def build_solution(arcs, assign, data, C, A, node_ids, cust_ids):
    """
    arcs: iterable of (i, j) matrix-index arcs (0 = depot)
    assign: dict customer_index -> site matrix index (1..n)
    Returns full solution dict following the schema.
    """
    succ = {}
    depot_outs = []
    for (i, j) in arcs:
        if i == 0:
            depot_outs.append(j)
        else:
            succ[i] = j
    depot_outs.sort()

    routes = {}
    site_tour = {}
    n = len(node_ids) - 1
    for v, start in enumerate(depot_outs):
        seq_idx = [0]
        cur = start
        steps = 0
        while cur != 0 and steps <= n + 2:
            seq_idx.append(cur)
            site_tour[cur] = v
            cur = succ.get(cur, 0)
            steps += 1
        seq_idx.append(0)
        arc_list = []
        for a in range(len(seq_idx) - 1):
            arc_list.append([node_ids[seq_idx[a]], node_ids[seq_idx[a + 1]]])
        routes[str(v)] = {
            "arcs": arc_list,
            "sequence": [node_ids[k] for k in seq_idx],
        }

    routing_cost = 0.0
    for (i, j) in arcs:
        routing_cost += C[i][j]

    assign_cost = 0.0
    assignments = {}
    customer_tours = {}
    for c, s in assign.items():
        assign_cost += A[c][s - 1]
        assignments[str(cust_ids[c])] = node_ids[s]
        customer_tours[str(cust_ids[c])] = {
            "vehicle": site_tour.get(s, 0),
            "site": node_ids[s],
        }

    obj = routing_cost + assign_cost
    return {
        "objective_value": obj,
        "routes": routes,
        "assignments": assignments,
        "customer_tours": customer_tours,
    }


def greedy_heuristic(n, m, Q, K, C, A, dem):
    """Construct a feasible solution: pack customers into tours (FFD), pick
    sites per tour, route with nearest neighbor. Returns (arcs, assign) or None."""
    order = sorted(range(m), key=lambda c: -dem[c])
    bins = []      # list of lists of customer indices
    bin_load = []
    for c in order:
        placed = False
        for b in range(len(bins)):
            if bin_load[b] + dem[c] <= Q:
                bins[b].append(c)
                bin_load[b] += dem[c]
                placed = True
                break
        if not placed:
            bins.append([c])
            bin_load.append(dem[c])
    if len(bins) > K:
        return None
    if any(l > Q for l in bin_load):
        return None

    used_sites = set()
    assign = {}
    tour_sites = []
    for b in range(len(bins)):
        my_sites = set()
        for c in bins[b]:
            best_s = None
            best_cost = None
            for s in range(1, n + 1):
                if s in used_sites and s not in my_sites:
                    continue
                cost = A[c][s - 1]
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_s = s
            if best_s is None:
                return None
            assign[c] = best_s
            my_sites.add(best_s)
            used_sites.add(best_s)
        tour_sites.append(my_sites)

    arcs = []
    for sites in tour_sites:
        if not sites:
            continue
        remaining = set(sites)
        cur = 0
        while remaining:
            nxt = min(remaining, key=lambda s: C[cur][s])
            arcs.append((cur, nxt))
            remaining.discard(nxt)
            cur = nxt
        arcs.append((cur, 0))
    return arcs, assign


def main():
    args = parse_args()
    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["num_delivery_sites"]
    m = data["num_customers"]
    Q = data["vehicle_capacity"]
    K = data["num_vehicles"]
    C = data["routing_cost_matrix"]
    A = data["assignment_cost_matrix"]
    dem = [c["demand"] for c in data["customers"]]
    cust_ids = [c["id"] for c in data["customers"]]
    node_ids = [data["depot"]["id"]] + [s["id"] for s in data["delivery_sites"]]

    # ---------- heuristic warm start ----------
    heur = greedy_heuristic(n, m, Q, K, C, A, dem)
    heur_sol = None
    if heur is not None:
        heur_arcs, heur_assign = heur
        heur_sol = build_solution(heur_arcs, heur_assign, data, C, A, node_ids, cust_ids)
        if logger:
            logger.log_solution(heur_sol["objective_value"], heur_sol)

    # ---------- MIP model ----------
    model = gp.Model("clrp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    nodes = list(range(n + 1))
    arcs_all = [(i, j) for i in nodes for j in nodes if i != j]

    x = model.addVars(arcs_all, vtype=GRB.BINARY, name="x")
    y = model.addVars(range(1, n + 1), vtype=GRB.BINARY, name="y")
    z = model.addVars(range(m), range(1, n + 1), vtype=GRB.BINARY, name="z")
    # flow variables (no flow returns to depot)
    flow_arcs = [(i, j) for (i, j) in arcs_all if j != 0]
    fl = model.addVars(flow_arcs, lb=0.0, ub=Q, name="f")

    # degree constraints for sites
    for s in range(1, n + 1):
        model.addConstr(gp.quicksum(x[s, j] for j in nodes if j != s) == y[s])
        model.addConstr(gp.quicksum(x[i, s] for i in nodes if i != s) == y[s])
    # depot balance and fleet size
    model.addConstr(gp.quicksum(x[0, j] for j in range(1, n + 1)) ==
                    gp.quicksum(x[j, 0] for j in range(1, n + 1)))
    model.addConstr(gp.quicksum(x[0, j] for j in range(1, n + 1)) <= K)

    # assignment constraints
    for c in range(m):
        model.addConstr(gp.quicksum(z[c, s] for s in range(1, n + 1)) == 1)
    for c in range(m):
        for s in range(1, n + 1):
            model.addConstr(z[c, s] <= y[s])
    for s in range(1, n + 1):
        model.addConstr(y[s] <= gp.quicksum(z[c, s] for c in range(m)))

    # flow (capacity + connectivity)
    for (i, j) in flow_arcs:
        model.addConstr(fl[i, j] <= Q * x[i, j])
    for s in range(1, n + 1):
        inflow = gp.quicksum(fl[i, s] for i in nodes if i != s)
        outflow = gp.quicksum(fl[s, j] for j in nodes if j != s and j != 0)
        model.addConstr(inflow - outflow ==
                        gp.quicksum(dem[c] * z[c, s] for c in range(m)))

    model.setObjective(
        gp.quicksum(C[i][j] * x[i, j] for (i, j) in arcs_all) +
        gp.quicksum(A[c][s - 1] * z[c, s] for c in range(m) for s in range(1, n + 1)),
        GRB.MINIMIZE,
    )

    # warm start
    if heur is not None:
        heur_arc_set = set(heur_arcs)
        for (i, j) in arcs_all:
            x[i, j].Start = 1.0 if (i, j) in heur_arc_set else 0.0
        used = set(heur_assign.values())
        for s in range(1, n + 1):
            y[s].Start = 1.0 if s in used else 0.0
        for c in range(m):
            for s in range(1, n + 1):
                z[c, s].Start = 1.0 if heur_assign.get(c) == s else 0.0

    elapsed = time.time() - t0
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    xvars = list(x.values())
    xkeys = list(x.keys())
    zvars = list(z.values())
    zkeys = list(z.keys())

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = mdl.cbGetSolution(xvars)
                zv = mdl.cbGetSolution(zvars)
                arcs = [xkeys[k] for k in range(len(xkeys)) if xv[k] > 0.5]
                assign = {}
                best = {}
                for k in range(len(zkeys)):
                    if zv[k] > 0.5:
                        c, s = zkeys[k]
                        if c not in best or zv[k] > best[c]:
                            best[c] = zv[k]
                            assign[c] = s
                sol = build_solution(arcs, assign, data, C, A, node_ids, cust_ids)
                if logger:
                    logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass

    try:
        model.optimize(callback)
    except Exception:
        pass

    final_sol = None
    if model.SolCount > 0:
        arcs = [(i, j) for (i, j) in arcs_all if x[i, j].X > 0.5]
        assign = {}
        for c in range(m):
            best_s, best_v = None, -1.0
            for s in range(1, n + 1):
                v = z[c, s].X
                if v > best_v:
                    best_v = v
                    best_s = s
            assign[c] = best_s
        final_sol = build_solution(arcs, assign, data, C, A, node_ids, cust_ids)
        if heur_sol is not None and heur_sol["objective_value"] < final_sol["objective_value"]:
            final_sol = heur_sol
    elif heur_sol is not None:
        final_sol = heur_sol
    else:
        final_sol = {
            "objective_value": float("inf"),
            "routes": {},
            "assignments": {},
            "customer_tours": {},
        }

    if logger and final_sol["objective_value"] < float("inf"):
        logger.log_solution(final_sol["objective_value"], final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()