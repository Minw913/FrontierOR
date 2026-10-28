import argparse
import json
import time
import math
import heapq
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from solution_logger import SolutionLogger


INF = 10**18


def dijkstra_all_pairs(num_nodes, adj):
    dist = [[INF] * num_nodes for _ in range(num_nodes)]
    parent_node = [[-1] * num_nodes for _ in range(num_nodes)]
    parent_edge = [[-1] * num_nodes for _ in range(num_nodes)]

    for s in range(num_nodes):
        d = dist[s]
        d[s] = 0
        pq = [(0, s)]
        while pq:
            cd, u = heapq.heappop(pq)
            if cd != d[u]:
                continue
            for v, w, eid in adj[u]:
                nd = cd + w
                if nd < d[v]:
                    d[v] = nd
                    parent_node[s][v] = u
                    parent_edge[s][v] = eid
                    heapq.heappush(pq, (nd, v))
                elif nd == d[v]:
                    # deterministic tie-break
                    if parent_edge[s][v] == -1 or eid < parent_edge[s][v]:
                        parent_node[s][v] = u
                        parent_edge[s][v] = eid
    return dist, parent_node, parent_edge


def reconstruct_path_edges(src, dst, parent_node, parent_edge):
    if src == dst:
        return []
    if parent_node[src][dst] == -1:
        return None
    cur = dst
    path = []
    while cur != src:
        pe = parent_edge[src][cur]
        pn = parent_node[src][cur]
        if pe == -1 or pn == -1:
            return None
        path.append(pe)
        cur = pn
    path.reverse()
    return path


def eval_route_deadhead(order, tasks, depot_idx, dist, recover=False):
    m = len(order)
    if m == 0:
        return 0.0, []

    # dp[i][o]: min deadhead up to i-th task serviced with orientation o
    # o=0 => task traversed u->v, end=v, start=u
    # o=1 => task traversed v->u, end=u, start=v
    dp = [[INF, INF] for _ in range(m)]
    par = [[-1, -1] for _ in range(m)] if recover else None

    t0 = tasks[order[0]]
    u0, v0 = t0["u"], t0["v"]
    if dist[depot_idx][u0] < INF:
        dp[0][0] = dist[depot_idx][u0]
    if dist[depot_idx][v0] < INF:
        dp[0][1] = dist[depot_idx][v0]

    for i in range(1, m):
        ti = tasks[order[i]]
        ui, vi = ti["u"], ti["v"]

        tprev = tasks[order[i - 1]]
        up, vp = tprev["u"], tprev["v"]

        prev_end = [vp, up]  # if prev orient 0 end=vp, orient1 end=up
        curr_start = [ui, vi]

        for o in (0, 1):
            best_val = INF
            best_po = -1
            s_node = curr_start[o]
            for po in (0, 1):
                e_node = prev_end[po]
                if dp[i - 1][po] >= INF or dist[e_node][s_node] >= INF:
                    continue
                val = dp[i - 1][po] + dist[e_node][s_node]
                if val < best_val:
                    best_val = val
                    best_po = po
            dp[i][o] = best_val
            if recover:
                par[i][o] = best_po

    last_task = tasks[order[-1]]
    ul, vl = last_task["u"], last_task["v"]
    last_end = [vl, ul]

    best_total = INF
    best_last_o = -1
    for o in (0, 1):
        if dp[m - 1][o] >= INF or dist[last_end[o]][depot_idx] >= INF:
            continue
        val = dp[m - 1][o] + dist[last_end[o]][depot_idx]
        if val < best_total:
            best_total = val
            best_last_o = o

    if not recover:
        return best_total, None

    if best_last_o == -1:
        return INF, None

    orient = [0] * m
    orient[m - 1] = best_last_o
    for i in range(m - 1, 0, -1):
        orient[i - 1] = par[i][orient[i]]
    return best_total, orient


def build_initial_assignment_greedy(tasks, K, Q):
    # Best-fit decreasing by demand
    idxs = list(range(len(tasks)))
    idxs.sort(key=lambda t: tasks[t]["demand"], reverse=True)

    routes = [[] for _ in range(K)]
    loads = [0] * K

    for t in idxs:
        d = tasks[t]["demand"]
        best_k = -1
        best_rem = None
        for k in range(K):
            if loads[k] + d <= Q:
                rem = Q - (loads[k] + d)
                if best_rem is None or rem < best_rem:
                    best_rem = rem
                    best_k = k
        if best_k == -1:
            return None, None
        routes[best_k].append(t)
        loads[best_k] += d
    return routes, loads


def assignment_mip(tasks, K, Q, time_limit):
    if gp is None:
        return None, None
    R = len(tasks)
    try:
        model = gp.Model("assign")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1.0, time_limit)
        x = model.addVars(R, K, vtype=GRB.BINARY, name="x")
        for t in range(R):
            model.addConstr(gp.quicksum(x[t, k] for k in range(K)) == 1)
        for k in range(K):
            model.addConstr(gp.quicksum(tasks[t]["demand"] * x[t, k] for t in range(R)) <= Q)
        model.setObjective(0.0, GRB.MINIMIZE)
        model.optimize()

        if model.SolCount == 0:
            return None, None

        routes = [[] for _ in range(K)]
        loads = [0] * K
        for t in range(R):
            assigned = None
            for k in range(K):
                if x[t, k].X > 0.5:
                    assigned = k
                    break
            if assigned is None:
                assigned = 0
            routes[assigned].append(t)
            loads[assigned] += tasks[t]["demand"]
        return routes, loads
    except Exception:
        return None, None


def construct_orders_from_assignment(routes_tasks, tasks, depot_idx, dist, deadline):
    K = len(routes_tasks)
    routes = [[] for _ in range(K)]

    # insertion order heuristic: larger demand first, tie farther from depot
    for k in range(K):
        tset = routes_tasks[k]
        tset_sorted = sorted(
            tset,
            key=lambda t: (tasks[t]["demand"], max(dist[depot_idx][tasks[t]["u"]], dist[depot_idx][tasks[t]["v"]])),
            reverse=True,
        )
        order = []
        for t in tset_sorted:
            if time.time() > deadline:
                order.append(t)
                continue
            best_pos = 0
            best_cost = INF
            for pos in range(len(order) + 1):
                cand = order[:pos] + [t] + order[pos:]
                c, _ = eval_route_deadhead(cand, tasks, depot_idx, dist, recover=False)
                if c < best_cost:
                    best_cost = c
                    best_pos = pos
            order.insert(best_pos, t)
        routes[k] = order
    return routes


def total_deadhead_cost(routes, tasks, depot_idx, dist):
    costs = []
    total = 0.0
    for r in routes:
        c, _ = eval_route_deadhead(r, tasks, depot_idx, dist, recover=False)
        costs.append(c)
        total += c
    return total, costs


def local_search(routes, loads, tasks, K, Q, depot_idx, dist, deadline, logger, service_const):
    total_dead, route_costs = total_deadhead_cost(routes, tasks, depot_idx, dist)
    best_obj = service_const + total_dead
    if logger:
        logger.log(best_obj)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Relocate (including intra-route reposition)
        for a in range(K):
            if time.time() > deadline:
                break
            ra = routes[a]
            for i in range(len(ra)):
                if time.time() > deadline:
                    break
                t = ra[i]
                dt = tasks[t]["demand"]

                move_done = False
                for b in range(K):
                    if time.time() > deadline:
                        break
                    if a != b and loads[b] + dt > Q:
                        continue
                    rb = routes[b]

                    positions = range(len(rb) + 1)
                    for j in positions:
                        if time.time() > deadline:
                            break
                        if a == b:
                            if j == i or j == i + 1:
                                continue
                            new_r = list(ra)
                            item = new_r.pop(i)
                            j2 = j - 1 if j > i else j
                            new_r.insert(j2, item)
                            new_cost, _ = eval_route_deadhead(new_r, tasks, depot_idx, dist, recover=False)
                            delta = new_cost - route_costs[a]
                            if delta < -1e-9:
                                routes[a] = new_r
                                route_costs[a] = new_cost
                                total_dead += delta
                                improved = True
                                move_done = True
                        else:
                            new_ra = ra[:i] + ra[i + 1 :]
                            new_rb = rb[:j] + [t] + rb[j:]
                            new_ca, _ = eval_route_deadhead(new_ra, tasks, depot_idx, dist, recover=False)
                            new_cb, _ = eval_route_deadhead(new_rb, tasks, depot_idx, dist, recover=False)
                            delta = (new_ca + new_cb) - (route_costs[a] + route_costs[b])
                            if delta < -1e-9:
                                routes[a] = new_ra
                                routes[b] = new_rb
                                route_costs[a] = new_ca
                                route_costs[b] = new_cb
                                loads[a] -= dt
                                loads[b] += dt
                                total_dead += delta
                                improved = True
                                move_done = True

                        if move_done:
                            obj = service_const + total_dead
                            if obj + 1e-9 < best_obj:
                                best_obj = obj
                                if logger:
                                    logger.log(best_obj)
                            break
                    if move_done:
                        break
                if move_done:
                    break
            if improved:
                break

        if improved:
            continue

        # Swap between routes
        for a in range(K):
            if time.time() > deadline:
                break
            for b in range(a + 1, K):
                if time.time() > deadline:
                    break
                ra = routes[a]
                rb = routes[b]
                swap_done = False
                for i in range(len(ra)):
                    if time.time() > deadline:
                        break
                    t1 = ra[i]
                    d1 = tasks[t1]["demand"]
                    for j in range(len(rb)):
                        if time.time() > deadline:
                            break
                        t2 = rb[j]
                        d2 = tasks[t2]["demand"]

                        if loads[a] - d1 + d2 > Q or loads[b] - d2 + d1 > Q:
                            continue

                        new_ra = list(ra)
                        new_rb = list(rb)
                        new_ra[i] = t2
                        new_rb[j] = t1

                        new_ca, _ = eval_route_deadhead(new_ra, tasks, depot_idx, dist, recover=False)
                        new_cb, _ = eval_route_deadhead(new_rb, tasks, depot_idx, dist, recover=False)
                        delta = (new_ca + new_cb) - (route_costs[a] + route_costs[b])

                        if delta < -1e-9:
                            routes[a] = new_ra
                            routes[b] = new_rb
                            route_costs[a] = new_ca
                            route_costs[b] = new_cb
                            loads[a] = loads[a] - d1 + d2
                            loads[b] = loads[b] - d2 + d1
                            total_dead += delta
                            improved = True
                            swap_done = True
                            obj = service_const + total_dead
                            if obj + 1e-9 < best_obj:
                                best_obj = obj
                                if logger:
                                    logger.log(best_obj)
                            break
                    if swap_done:
                        break
                if swap_done:
                    break
            if improved:
                break

    return routes


def solve(instance, time_limit, logger=None):
    start = time.time()
    deadline = start + max(1, time_limit) - 1e-3

    depot_raw = instance["depot"]
    K = int(instance["fleet"]["num_vehicles"])
    Q = int(instance["fleet"]["vehicle_capacity"])
    edges = instance["edges"]

    # Node mapping
    node_set = {depot_raw}
    for e in edges:
        u, v = e["endpoints"]
        node_set.add(u)
        node_set.add(v)
    nodes = sorted(node_set)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    N = len(nodes)
    depot_idx = node_to_idx[depot_raw]

    # Graph
    adj = [[] for _ in range(N)]
    edge_cost = {}
    edge_endpoints_idx = {}
    for e in edges:
        eid = int(e["edge_id"])
        u_raw, v_raw = e["endpoints"]
        u = node_to_idx[u_raw]
        v = node_to_idx[v_raw]
        c = float(e["cost"])
        edge_cost[eid] = c
        edge_endpoints_idx[eid] = (u, v)
        adj[u].append((v, c, eid))
        adj[v].append((u, c, eid))

    # Required tasks
    tasks = []
    service_const = 0.0
    for e in edges:
        if e["is_required"]:
            eid = int(e["edge_id"])
            u = node_to_idx[e["endpoints"][0]]
            v = node_to_idx[e["endpoints"][1]]
            d = int(e["demand"])
            sc = float(e["service_cost"])
            tasks.append({"edge_id": eid, "u": u, "v": v, "demand": d, "service_cost": sc})
            service_const += sc

    # Precompute shortest paths
    dist, parent_node, parent_edge = dijkstra_all_pairs(N, adj)

    # Quick infeasibility check
    for t in tasks:
        if t["demand"] > Q:
            # impossible
            routes_out = []
            for k in range(K):
                routes_out.append({"vehicle": k, "serviced_edges": [], "deadheaded_edges": []})
            return {"objective_value": float(1e18), "routes": routes_out}

    # Assignment
    routes_tasks, loads = build_initial_assignment_greedy(tasks, K, Q)
    if routes_tasks is None:
        remaining = max(1.0, deadline - time.time())
        routes_tasks, loads = assignment_mip(tasks, K, Q, time_limit=remaining)
    if routes_tasks is None:
        routes_out = []
        for k in range(K):
            routes_out.append({"vehicle": k, "serviced_edges": [], "deadheaded_edges": []})
        return {"objective_value": float(1e18), "routes": routes_out}

    # Build initial ordered routes
    routes = construct_orders_from_assignment(routes_tasks, tasks, depot_idx, dist, deadline)

    # Improve
    routes = local_search(routes, loads, tasks, K, Q, depot_idx, dist, deadline, logger, service_const)

    # Build output with deadheaded edge counts
    output_routes = []
    total_deadheading = 0.0
    for k in range(K):
        order = routes[k]
        dead_cost, orient = eval_route_deadhead(order, tasks, depot_idx, dist, recover=True)
        if dead_cost >= INF or orient is None:
            # fallback if somehow unreachable
            orient = [0] * len(order)
            dead_counts = {}
        else:
            dead_counts = defaultdict(int)
            cur = depot_idx
            for pos, t_idx in enumerate(order):
                t = tasks[t_idx]
                u, v = t["u"], t["v"]
                o = orient[pos]
                s_node = u if o == 0 else v
                e_node = v if o == 0 else u

                p = reconstruct_path_edges(cur, s_node, parent_node, parent_edge)
                if p is None:
                    p = []
                for eid in p:
                    dead_counts[eid] += 1
                cur = e_node

            p = reconstruct_path_edges(cur, depot_idx, parent_node, parent_edge)
            if p is None:
                p = []
            for eid in p:
                dead_counts[eid] += 1

        serviced_edges = [tasks[t]["edge_id"] for t in order]
        deadheaded_edges = [{"edge_id": int(eid), "times": int(times)} for eid, times in sorted(dead_counts.items()) if times > 0]

        # recompute deadheading cost exactly from counts
        route_dead_cost = 0.0
        for item in deadheaded_edges:
            route_dead_cost += edge_cost[item["edge_id"]] * item["times"]
        total_deadheading += route_dead_cost

        output_routes.append(
            {
                "vehicle": k,
                "serviced_edges": serviced_edges,
                "deadheaded_edges": deadheaded_edges,
            }
        )

    objective_value = float(service_const + total_deadheading)
    return {"objective_value": objective_value, "routes": output_routes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()