import argparse
import json
import time
from typing import List, Dict, Tuple, Optional

from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve VRP with Linehaul-Backhaul constraints")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def route_cost(route, dist, depot):
    lh = route["lh"]
    bh = route["bh"]
    if not lh:
        if bh:
            return float("inf")
        return dist[depot][depot]

    c = 0
    c += dist[depot][lh[0]]
    for i in range(len(lh) - 1):
        c += dist[lh[i]][lh[i + 1]]

    if bh:
        c += dist[lh[-1]][bh[0]]
        for i in range(len(bh) - 1):
            c += dist[bh[i]][bh[i + 1]]
        c += dist[bh[-1]][depot]
    else:
        c += dist[lh[-1]][depot]
    return c


def total_cost(routes, dist, depot):
    return sum(route_cost(r, dist, depot) for r in routes)


def nearest_neighbor_order(nodes: List[int], dist, start: int) -> List[int]:
    if not nodes:
        return []
    unvisited = set(nodes)
    order = []
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist[cur][x])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def lh_remove_delta(route, pos, dist, depot):
    lh, bh = route["lh"], route["bh"]
    node = lh[pos]
    prev_node = depot if pos == 0 else lh[pos - 1]
    if pos == len(lh) - 1:
        next_node = bh[0] if bh else depot
    else:
        next_node = lh[pos + 1]
    return dist[prev_node][next_node] - dist[prev_node][node] - dist[node][next_node]


def lh_insert_delta(route, node, pos, dist, depot):
    lh, bh = route["lh"], route["bh"]
    prev_node = depot if pos == 0 else lh[pos - 1]
    if pos == len(lh):
        next_node = bh[0] if bh else depot
    else:
        next_node = lh[pos]
    return dist[prev_node][node] + dist[node][next_node] - dist[prev_node][next_node]


def bh_remove_delta(route, pos, dist, depot):
    lh, bh = route["lh"], route["bh"]
    node = bh[pos]
    prev_node = lh[-1] if pos == 0 else bh[pos - 1]
    next_node = depot if pos == len(bh) - 1 else bh[pos + 1]
    return dist[prev_node][next_node] - dist[prev_node][node] - dist[node][next_node]


def bh_insert_delta(route, node, pos, dist, depot):
    lh, bh = route["lh"], route["bh"]
    prev_node = lh[-1] if pos == 0 else bh[pos - 1]
    next_node = depot if pos == len(bh) else bh[pos]
    return dist[prev_node][node] + dist[node][next_node] - dist[prev_node][next_node]


def improve_route_by_relocate(route, part, dist, depot, time_check):
    # First-improvement relocate within one sequence
    while True:
        if time_check():
            return False
        seq = route[part]
        n = len(seq)
        if n <= 2:
            return False

        base = route_cost(route, dist, depot)
        improved = False
        for i in range(n):
            if time_check():
                return False
            node = seq[i]
            reduced = seq[:i] + seq[i + 1:]
            for j in range(n):
                if j == i:
                    continue
                cand = reduced[:j] + [node] + reduced[j:]
                new_route = {"lh": route["lh"][:], "bh": route["bh"][:]}
                new_route[part] = cand
                c = route_cost(new_route, dist, depot)
                if c + 1e-9 < base:
                    route[part] = cand
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return False


def assign_linehauls_mip(
    lh_nodes, lh_demand, m, Q, remaining_time
) -> Optional[List[List[int]]]:
    if not lh_nodes:
        return [[] for _ in range(m)]

    model = Model("linehaul_assign")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, min(30.0, remaining_time))

    x = {}
    for i in lh_nodes:
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
    a = {k: model.addVar(vtype=GRB.BINARY, name=f"a_{k}") for k in range(m)}
    model.update()

    for i in lh_nodes:
        model.addConstr(quicksum(x[i, k] for k in range(m)) == 1)

    for k in range(m):
        model.addConstr(quicksum(lh_demand[i] * x[i, k] for i in lh_nodes) <= Q)
        model.addConstr(a[k] <= quicksum(x[i, k] for i in lh_nodes))
        model.addConstr(quicksum(x[i, k] for i in lh_nodes) <= len(lh_nodes) * a[k])

    # Encourage maximum number of active linehaul routes (helps backhaul feasibility)
    BIG = 10_000
    model.setObjective(BIG * quicksum(a[k] for k in range(m)), GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None

    routes_lh = [[] for _ in range(m)]
    for i in lh_nodes:
        for k in range(m):
            if x[i, k].X > 0.5:
                routes_lh[k].append(i)
                break
    return routes_lh


def assign_linehauls_greedy(lh_nodes, lh_demand, m, Q):
    routes = [[] for _ in range(m)]
    load = [0] * m
    # seed one per route as much as possible
    nodes_sorted = sorted(lh_nodes, key=lambda i: lh_demand[i], reverse=True)
    seed = min(m, len(nodes_sorted))
    for k in range(seed):
        i = nodes_sorted[k]
        if lh_demand[i] > Q:
            return None
        routes[k].append(i)
        load[k] += lh_demand[i]

    for i in nodes_sorted[seed:]:
        candidates = [k for k in range(m) if load[k] + lh_demand[i] <= Q]
        if not candidates:
            return None
        k = min(candidates, key=lambda kk: load[kk])  # balance
        routes[k].append(i)
        load[k] += lh_demand[i]
    return routes


def assign_backhauls_mip(
    bh_nodes, bh_demand, active_routes, routes, m, Q, dist, depot, remaining_time
):
    if not bh_nodes:
        return [[] for _ in range(m)]

    active = [k for k in active_routes if routes[k]["lh"]]
    if not active:
        return None

    model = Model("backhaul_assign")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, min(30.0, remaining_time))

    y = {}
    for j in bh_nodes:
        for k in active:
            y[j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")
    model.update()

    for j in bh_nodes:
        model.addConstr(quicksum(y[j, k] for k in active) == 1)

    for k in active:
        model.addConstr(quicksum(bh_demand[j] * y[j, k] for j in bh_nodes) <= Q)

    # proxy objective based on each route's current linehaul end
    obj_terms = []
    for j in bh_nodes:
        for k in active:
            lh_last = routes[k]["lh"][-1]
            proxy = dist[lh_last][j] + dist[j][depot] - dist[lh_last][depot]
            obj_terms.append(proxy * y[j, k])
    model.setObjective(quicksum(obj_terms), GRB.MINIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None

    routes_bh = [[] for _ in range(m)]
    for j in bh_nodes:
        for k in active:
            if y[j, k].X > 0.5:
                routes_bh[k].append(j)
                break
    return routes_bh


def assign_backhauls_greedy(bh_nodes, bh_demand, routes, Q):
    m = len(routes)
    routes_bh = [[] for _ in range(m)]
    cap_used = [0] * m
    active = [k for k in range(m) if routes[k]["lh"]]

    if bh_nodes and not active:
        return None

    for j in sorted(bh_nodes, key=lambda x: bh_demand[x], reverse=True):
        candidates = [k for k in active if cap_used[k] + bh_demand[j] <= Q]
        if not candidates:
            return None
        k = min(candidates, key=lambda kk: cap_used[kk])
        routes_bh[k].append(j)
        cap_used[k] += bh_demand[j]
    return routes_bh


def solve(instance, time_limit, logger=None):
    start = time.time()

    def remaining():
        return max(0.0, time_limit - (time.time() - start))

    def timeout():
        return remaining() <= 0.0

    dist = instance["distance_matrix"]
    depot = instance["depot"]["index"]

    lh_nodes = [c["index"] for c in instance["linehaul_customers"]]
    bh_nodes = [c["index"] for c in instance["backhaul_customers"]]
    lh_demand = {c["index"]: int(c["demand"]) for c in instance["linehaul_customers"]}
    bh_demand = {c["index"]: int(c["demand"]) for c in instance["backhaul_customers"]}

    m = int(instance["num_vehicles"])
    Q = int(instance["vehicle_capacity"])

    # ---------- Linehaul assignment ----------
    lh_assignment = None
    if not timeout():
        lh_assignment = assign_linehauls_mip(
            lh_nodes, lh_demand, m, Q, remaining_time=min(remaining(), max(1.0, time_limit * 0.2))
        )
    if lh_assignment is None:
        lh_assignment = assign_linehauls_greedy(lh_nodes, lh_demand, m, Q)

    if lh_assignment is None:
        # fallback trivial infeasible output
        routes_out = [[depot, depot] for _ in range(m)]
        return {"objective_value": float("inf"), "routes": routes_out}

    routes = [{"lh": lh_assignment[k][:], "bh": []} for k in range(m)]

    # Build initial LH order
    for k in range(m):
        if routes[k]["lh"]:
            routes[k]["lh"] = nearest_neighbor_order(routes[k]["lh"], dist, depot)

    # ---------- Backhaul assignment ----------
    active_routes = [k for k in range(m) if routes[k]["lh"]]
    bh_assignment = None
    if not timeout():
        bh_assignment = assign_backhauls_mip(
            bh_nodes,
            bh_demand,
            active_routes,
            routes,
            m,
            Q,
            dist,
            depot,
            remaining_time=min(remaining(), max(1.0, time_limit * 0.2)),
        )
    if bh_assignment is None:
        bh_assignment = assign_backhauls_greedy(bh_nodes, bh_demand, routes, Q)

    if bh_assignment is None:
        # fallback infeasible output
        routes_out = [[depot] + r["lh"] + [depot] for r in routes]
        obj = total_cost(routes, dist, depot)
        return {"objective_value": float(obj), "routes": routes_out}

    for k in range(m):
        routes[k]["bh"] = bh_assignment[k][:]

    # Build initial BH order
    for k in range(m):
        if routes[k]["bh"]:
            routes[k]["bh"] = nearest_neighbor_order(routes[k]["bh"], dist, routes[k]["lh"][-1])

    # Local intra-route improvement
    if not timeout():
        for k in range(m):
            improve_route_by_relocate(routes[k], "lh", dist, depot, timeout)
            improve_route_by_relocate(routes[k], "bh", dist, depot, timeout)

    # Loads for fast feasibility in inter-route moves
    lh_load = [sum(lh_demand[i] for i in routes[k]["lh"]) for k in range(m)]
    bh_load = [sum(bh_demand[j] for j in routes[k]["bh"]) for k in range(m)]

    current_obj = total_cost(routes, dist, depot)
    best_obj = current_obj
    best_routes = [{"lh": r["lh"][:], "bh": r["bh"][:]} for r in routes]
    if logger:
        logger.log(float(best_obj))

    # ---------- Inter-route relocate local search ----------
    improved = True
    while improved and not timeout():
        improved = False

        # Try LH relocate (first improvement)
        moved = False
        for a in range(m):
            if moved or timeout():
                break
            for i_pos in range(len(routes[a]["lh"])):
                if timeout():
                    break
                node = routes[a]["lh"][i_pos]
                if len(routes[a]["lh"]) == 1 and len(routes[a]["bh"]) > 0:
                    continue  # would create backhaul-only route
                delta_rem = lh_remove_delta(routes[a], i_pos, dist, depot)

                for b in range(m):
                    if a == b:
                        continue
                    if lh_load[b] + lh_demand[node] > Q:
                        continue
                    for ins_pos in range(len(routes[b]["lh"]) + 1):
                        delta_ins = lh_insert_delta(routes[b], node, ins_pos, dist, depot)
                        delta = delta_rem + delta_ins
                        if delta < -1e-9:
                            # apply
                            routes[a]["lh"].pop(i_pos)
                            routes[b]["lh"].insert(ins_pos, node)
                            lh_load[a] -= lh_demand[node]
                            lh_load[b] += lh_demand[node]
                            current_obj += delta
                            moved = True
                            improved = True
                            break
                    if moved:
                        break
                if moved:
                    break

        if moved:
            # quick route re-optimization for changed routes
            for k in range(m):
                improve_route_by_relocate(routes[k], "lh", dist, depot, timeout)
                improve_route_by_relocate(routes[k], "bh", dist, depot, timeout)
            current_obj = total_cost(routes, dist, depot)
            if current_obj + 1e-9 < best_obj:
                best_obj = current_obj
                best_routes = [{"lh": r["lh"][:], "bh": r["bh"][:]} for r in routes]
                if logger:
                    logger.log(float(best_obj))
            continue

        # Try BH relocate
        moved = False
        for a in range(m):
            if moved or timeout():
                break
            for i_pos in range(len(routes[a]["bh"])):
                if timeout():
                    break
                node = routes[a]["bh"][i_pos]
                delta_rem = bh_remove_delta(routes[a], i_pos, dist, depot)

                for b in range(m):
                    if a == b:
                        continue
                    if not routes[b]["lh"]:
                        continue
                    if bh_load[b] + bh_demand[node] > Q:
                        continue
                    for ins_pos in range(len(routes[b]["bh"]) + 1):
                        delta_ins = bh_insert_delta(routes[b], node, ins_pos, dist, depot)
                        delta = delta_rem + delta_ins
                        if delta < -1e-9:
                            routes[a]["bh"].pop(i_pos)
                            routes[b]["bh"].insert(ins_pos, node)
                            bh_load[a] -= bh_demand[node]
                            bh_load[b] += bh_demand[node]
                            current_obj += delta
                            moved = True
                            improved = True
                            break
                    if moved:
                        break
                if moved:
                    break

        if moved:
            for k in range(m):
                improve_route_by_relocate(routes[k], "lh", dist, depot, timeout)
                improve_route_by_relocate(routes[k], "bh", dist, depot, timeout)
            current_obj = total_cost(routes, dist, depot)
            if current_obj + 1e-9 < best_obj:
                best_obj = current_obj
                best_routes = [{"lh": r["lh"][:], "bh": r["bh"][:]} for r in routes]
                if logger:
                    logger.log(float(best_obj))

    # Build output
    routes_out = []
    for r in best_routes:
        if not r["lh"] and not r["bh"]:
            routes_out.append([depot, depot])
        else:
            routes_out.append([depot] + r["lh"] + r["bh"] + [depot])

    return {"objective_value": float(best_obj), "routes": routes_out}


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()