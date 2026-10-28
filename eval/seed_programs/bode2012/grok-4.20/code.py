import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    depot = instance['depot']
    K = instance['fleet']['num_vehicles']
    Q = instance['fleet']['vehicle_capacity']
    edges = instance['edges']
    num_nodes = instance['graph']['num_nodes']

    # Build graph data
    edge_list = []  # (u, v, cost, is_req, demand, serv_cost, edge_id)
    required = []   # list of (idx_in_edge_list, edge_id, demand, serv_cost)
    total_service_cost = 0
    edge_id_to_idx = {}
    adj = defaultdict(list)

    for e in edges:
        u, v = e['endpoints']
        eid = e['edge_id']
        cost = e['cost']
        is_req = e['is_required']
        dem = e['demand']
        sc = e['service_cost']
        idx = len(edge_list)
        edge_list.append((u, v, cost, is_req, dem, sc, eid))
        edge_id_to_idx[eid] = idx
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        if is_req:
            required.append((idx, eid, dem, sc))
            total_service_cost += sc

    R = len(required)
    if R == 0:
        # Trivial case
        obj = 0.0
        routes = []
        for k in range(K):
            routes.append({
                "vehicle": k,
                "serviced_edges": [],
                "deadheaded_edges": []
            })
        output = {
            "objective_value": float(obj),
            "routes": routes
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        if logger:
            logger.log(obj)
        return

    # Precompute shortest path distances
    INF = 10**9
    dist = [[INF] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        dist[i][i] = 0
    for u, v, c, _, _, _, _ in edge_list:
        dist[u][v] = min(dist[u][v], c)
        dist[v][u] = min(dist[v][u], c)
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Predecessor for path reconstruction
    pred = [[-1] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and dist[i][j] < INF:
                for neigh, e_idx in adj[i]:
                    if neigh == j and edge_list[e_idx][2] == dist[i][j]:
                        pred[i][j] = neigh
                        break
                else:
                    for mid in range(num_nodes):
                        if dist[i][mid] + dist[mid][j] == dist[i][j] and dist[i][mid] < INF:
                            pred[i][j] = mid
                            break

    def get_path(start, end):
        if dist[start][end] >= INF or start == end:
            return []
        path = []
        current = start
        visited = set()
        while current != end and current not in visited:
            visited.add(current)
            next_node = pred[current][end]
            if next_node == -1:
                return []
            found = False
            for neigh, e_idx in adj[current]:
                if neigh == next_node:
                    path.append(e_idx)
                    found = True
                    break
            if not found:
                return []
            current = next_node
        return path

    # MIP for assigning required edges to vehicles
    m = gp.Model("CARP_Assignment")
    m.setParam('TimeLimit', max(1, args.time_limit - 10))
    m.setParam('Threads', 1)
    m.setParam('LogToConsole', 0)

    x = {}
    for r in range(R):
        for k in range(K):
            x[r, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{r}_{k}")

    for k in range(K):
        m.addConstr(gp.quicksum(required[r][2] * x[r, k] for r in range(R)) <= Q, name=f"cap_{k}")

    for r in range(R):
        m.addConstr(gp.quicksum(x[r, k] for k in range(K)) == 1, name=f"assign_{r}")

    # Objective is arbitrary since all assignments have same estimated cost in this heuristic
    m.setObjective(0, GRB.MINIMIZE)

    m.optimize()

    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # Fallback: round robin assignment by demand
        sorted_req_idx = sorted(range(R), key=lambda i: -required[i][2])
        vehicle_load = [0] * K
        assignment = [-1] * R
        for r in sorted_req_idx:
            dem = required[r][2]
            best_v = -1
            best_load = 10**9
            for v in range(K):
                if vehicle_load[v] + dem <= Q and vehicle_load[v] < best_load:
                    best_v = v
                    best_load = vehicle_load[v]
            if best_v == -1:
                for v in range(K):
                    if vehicle_load[v] + dem <= Q:
                        best_v = v
                        break
                else:
                    best_v = 0
            assignment[r] = best_v
            vehicle_load[best_v] += dem
    else:
        assignment = [-1] * R
        for r in range(R):
            for k in range(K):
                if x[r, k].X > 0.5:
                    assignment[r] = k
                    break

    # Build serviced lists
    vehicle_serviced = [[] for _ in range(K)]
    for r in range(R):
        k = assignment[r]
        vehicle_serviced[k].append(required[r][1])

    # Build deadheading
    vehicle_dead = [defaultdict(int) for _ in range(K)]
    total_dead_cost = 0.0

    for r in range(R):
        k = assignment[r]
        e_idx, eid, dem, sc = required[r]
        u, v, _, _, _, _, _ = edge_list[e_idx]
        
        # Option 1: depot -> u, service to v, v -> depot
        path_to1 = get_path(depot, u)
        path_from1 = get_path(v, depot)
        cost1 = sum(edge_list[e][2] for e in path_to1) + sum(edge_list[e][2] for e in path_from1)
        
        # Option 2: depot -> v, service to u, u -> depot
        path_to2 = get_path(depot, v)
        path_from2 = get_path(u, depot)
        cost2 = sum(edge_list[e][2] for e in path_to2) + sum(edge_list[e][2] for e in path_from2)
        
        if cost1 <= cost2:
            for ee in path_to1:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]
            for ee in path_from1:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]
        else:
            for ee in path_to2:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]
            for ee in path_from2:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]

    objective = total_service_cost + total_dead_cost

    # Build output routes
    routes = []
    for k in range(K):
        dead_list = []
        for ee, cnt in sorted(vehicle_dead[k].items()):
            if cnt > 0:
                dead_list.append({
                    "edge_id": edge_list[ee][6],
                    "times": int(cnt)
                })
        routes.append({
            "vehicle": k,
            "serviced_edges": sorted(vehicle_serviced[k]),
            "deadheaded_edges": dead_list
        })

    if logger:
        logger.log(objective)

    output = {
        "objective_value": float(objective),
        "routes": routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()