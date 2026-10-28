import json
import argparse
import heapq
import time
from collections import defaultdict
import math

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def build_adjacency_list(edges, num_nodes):
    graph = {i: [] for i in range(num_nodes)}
    for edge in edges:
        u, v = edge['endpoints']
        eid = edge['edge_id']
        cost = edge['cost']
        graph[u].append((v, eid, cost))
        graph[v].append((u, eid, cost))
    return graph

def dijkstra_distances(graph, source, num_nodes):
    dist = [float('inf')] * num_nodes
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist[node]:
            continue
        for neighbor, eid, cost in graph[node]:
            new_dist = d + cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return dist

def compute_all_pairs_distances(graph, num_nodes):
    dist_matrix = []
    for i in range(num_nodes):
        dist_matrix.append(dijkstra_distances(graph, i, num_nodes))
    return dist_matrix

def dijkstra_path_edges(graph, source, target):
    if source == target:
        return []
    dist = defaultdict(lambda: float('inf'))
    prev_node = {}
    prev_edge = {}
    dist[source] = 0
    heap = [(0, source, None, None)]
    while heap:
        d, node, parent, edge_id = heapq.heappop(heap)
        if node == target:
            break
        if d != dist[node]:
            continue
        for neighbor, eid, cost in graph[node]:
            new_d = d + cost
            if new_d < dist[neighbor]:
                dist[neighbor] = new_d
                prev_node[neighbor] = node
                prev_edge[neighbor] = eid
                heapq.heappush(heap, (new_d, neighbor, node, eid))
    if dist[target] == float('inf'):
        return []
    path_edges = []
    current = target
    while current != source:
        eid = prev_edge[current]
        path_edges.append(eid)
        current = prev_node[current]
    path_edges.reverse()
    return path_edges

def resequence_route(edge_ids, dist_matrix, depot, edges_dict):
    if not edge_ids:
        return []
    unassigned = set(edge_ids)
    current = depot
    route = []
    while unassigned:
        best_cost = float('inf')
        best_eid = None
        best_direction = None
        for eid in unassigned:
            edge_data = edges_dict[eid]
            u, v = edge_data['endpoints']
            cost1 = dist_matrix[current][u]
            cost2 = dist_matrix[current][v]
            if cost1 < best_cost:
                best_cost = cost1
                best_eid = eid
                best_direction = (u, v)
            if cost2 < best_cost:
                best_cost = cost2
                best_eid = eid
                best_direction = (v, u)
        if best_eid is None:
            break
        unassigned.remove(best_eid)
        route.append((best_eid, best_direction[0], best_direction[1]))
        current = best_direction[1]
    return route

def compute_route_cost(route, dist_matrix, depot):
    if not route:
        return 0
    cost = dist_matrix[depot][route[0][1]]
    for i in range(1, len(route)):
        prev_end = route[i-1][2]
        next_start = route[i][1]
        cost += dist_matrix[prev_end][next_start]
    last_end = route[-1][2]
    cost += dist_matrix[last_end][depot]
    return cost

def reverse_segment(segment):
    new_segment = []
    for edge in reversed(segment):
        eid, start, end = edge
        new_segment.append((eid, end, start))
    return new_segment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    depot = instance['depot']
    num_vehicles = instance['fleet']['num_vehicles']
    vehicle_capacity = instance['fleet']['vehicle_capacity']
    num_nodes = instance['graph']['num_nodes']
    edges_data = instance['edges']
    required_edges = [edge for edge in edges_data if edge['is_required']]

    edges_dict = {edge['edge_id']: edge for edge in edges_data}
    service_cost_total = sum(edge['service_cost'] for edge in required_edges)

    graph = build_adjacency_list(edges_data, num_nodes)
    dist_matrix = compute_all_pairs_distances(graph, num_nodes)

    unserviced = set(edge['edge_id'] for edge in required_edges)
    current_solution = []
    current_loads = []

    for k in range(num_vehicles):
        if not unserviced:
            break
        current_node = depot
        current_load = 0
        route = []
        while unserviced and current_load <= vehicle_capacity:
            candidates = []
            for eid in unserviced:
                edge = edges_dict[eid]
                u, v = edge['endpoints']
                demand = edge['demand']
                if current_load + demand > vehicle_capacity:
                    continue
                cost1 = dist_matrix[current_node][u]
                cost2 = dist_matrix[current_node][v]
                min_cost = min(cost1, cost2)
                candidates.append((min_cost, eid, u, v, cost1, cost2))
            if not candidates:
                break
            candidate = min(candidates, key=lambda x: x[0])
            _, eid, u, v, cost1, cost2 = candidate
            if cost1 <= cost2:
                start, end = u, v
            else:
                start, end = v, u
            route.append((eid, start, end))
            current_load += edges_dict[eid]['demand']
            unserviced.remove(eid)
            current_node = end
        current_solution.append(route)
        current_loads.append(current_load)

    while unserviced:
        assigned = False
        for k in range(num_vehicles):
            if not unserviced:
                break
            eid = next(iter(unserviced))
            edge = edges_dict[eid]
            demand = edge['demand']
            if current_loads[k] + demand <= vehicle_capacity:
                unserviced.remove(eid)
                current_solution[k].append((eid, None, None))
                current_loads[k] += demand
                assigned = True
        if not assigned:
            break

    for k in range(num_vehicles):
        edge_ids = [edge[0] for edge in current_solution[k]]
        if edge_ids:
            current_solution[k] = resequence_route(edge_ids, dist_matrix, depot, edges_dict)

    total_deadheading_cost = 0
    for route in current_solution:
        total_deadheading_cost += compute_route_cost(route, dist_matrix, depot)
    total_cost_initial = service_cost_total + total_deadheading_cost
    best_solution = current_solution
    best_total_deadheading_cost = total_deadheading_cost
    best_loads = current_loads.copy()

    if logger:
        logger.log(total_cost_initial)

    improvement = True
    while improvement and time.time() - start_time < args.time_limit:
        improvement = False

        if time.time() - start_time >= args.time_limit:
            break

        found = False
        for i in range(num_vehicles):
            if time.time() - start_time >= args.time_limit:
                break
            if found:
                break
            for j in range(num_vehicles):
                if i == j:
                    continue
                if time.time() - start_time >= args.time_limit:
                    break
                if found:
                    break
                for edge_tuple in current_solution[i]:
                    eid = edge_tuple[0]
                    edge_data = edges_dict[eid]
                    demand = edge_data['demand']
                    if best_loads[j] + demand > vehicle_capacity:
                        continue

                    edge_ids_i = [e[0] for e in current_solution[i] if e[0] != eid]
                    edge_ids_j = [e[0] for e in current_solution[j]] + [eid]

                    new_route_i = resequence_route(edge_ids_i, dist_matrix, depot, edges_dict)
                    new_route_j = resequence_route(edge_ids_j, dist_matrix, depot, edges_dict)

                    old_cost_i = compute_route_cost(current_solution[i], dist_matrix, depot)
                    old_cost_j = compute_route_cost(current_solution[j], dist_matrix, depot)
                    new_cost_i = compute_route_cost(new_route_i, dist_matrix, depot)
                    new_cost_j = compute_route_cost(new_route_j, dist_matrix, depot)

                    delta = (new_cost_i + new_cost_j) - (old_cost_i + old_cost_j)

                    if delta < -1e-5:
                        current_solution[i] = new_route_i
                        current_solution[j] = new_route_j
                        current_loads[i] -= demand
                        current_loads[j] += demand
                        total_deadheading_cost = best_total_deadheading_cost + delta
                        if total_deadheading_cost < best_total_deadheading_cost:
                            best_solution = [list(route) for route in current_solution]
                            best_total_deadheading_cost = total_deadheading_cost
                            best_loads = current_loads.copy()
                        improvement = True
                        found = True
                        if logger:
                            logger.log(service_cost_total + total_deadheading_cost)
                        break
        if improvement:
            continue

        if time.time() - start_time >= args.time_limit:
            break

        found = False
        for i in range(num_vehicles):
            if time.time() - start_time >= args.time_limit:
                break
            if found:
                break
            for j in range(num_vehicles):
                if i == j:
                    continue
                if time.time() - start_time >= args.time_limit:
                    break
                if found:
                    break
                for edge_tuple_i in current_solution[i]:
                    eid_i = edge_tuple_i[0]
                    edge_i_data = edges_dict[eid_i]
                    demand_i = edge_i_data['demand']
                    if found:
                        break
                    for edge_tuple_j in current_solution[j]:
                        eid_j = edge_tuple_j[0]
                        edge_j_data = edges_dict[eid_j]
                        demand_j = edge_j_data['demand']
                        if current_loads[i] - demand_i + demand_j <= vehicle_capacity and \
                           current_loads[j] - demand_j + demand_i <= vehicle_capacity:

                            edge_ids_i = [e[0] for e in current_solution[i] if e[0] != eid_i] + [eid_j]
                            edge_ids_j = [e[0] for e in current_solution[j] if e[0] != eid_j] + [eid_i]

                            new_route_i = resequence_route(edge_ids_i, dist_matrix, depot, edges_dict)
                            new_route_j = resequence_route(edge_ids_j, dist_matrix, depot, edges_dict)

                            old_cost_i = compute_route_cost(current_solution[i], dist_matrix, depot)
                            old_cost_j = compute_route_cost(current_solution[j], dist_matrix, depot)
                            new_cost_i = compute_route_cost(new_route_i, dist_matrix, depot)
                            new_cost_j = compute_route_cost(new_route_j, dist_matrix, depot)

                            delta = (new_cost_i + new_cost_j) - (old_cost_i + old_cost_j)

                            if delta < -1e-5:
                                current_solution[i] = new_route_i
                                current_solution[j] = new_route_j
                                current_loads[i] = current_loads[i] - demand_i + demand_j
                                current_loads[j] = current_loads[j] - demand_j + demand_i
                                total_deadheading_cost = best_total_deadheading_cost + delta
                                if total_deadheading_cost < best_total_deadheading_cost:
                                    best_solution = [list(route) for route in current_solution]
                                    best_total_deadheading_cost = total_deadheading_cost
                                    best_loads = current_loads.copy()
                                improvement = True
                                found = True
                                if logger:
                                    logger.log(service_cost_total + total_deadheading_cost)
                                break
        if improvement:
            continue

        if time.time() - start_time >= args.time_limit:
            break

        found = False
        for k in range(num_vehicles):
            if time.time() - start_time >= args.time_limit:
                break
            if found:
                break
            route = current_solution[k]
            n = len(route)
            for i in range(n):
                if time.time() - start_time >= args.time_limit:
                    break
                if found:
                    break
                for j in range(i+1, n):
                    new_route = route[:i] + reverse_segment(route[i:j+1]) + route[j+1:]
                    new_cost = compute_route_cost(new_route, dist_matrix, depot)
                    old_cost = compute_route_cost(route, dist_matrix, depot)
                    if new_cost < old_cost - 1e-5:
                        current_solution[k] = new_route
                        total_deadheading_cost = best_total_deadheading_cost + (new_cost - old_cost)
                        if total_deadheading_cost < best_total_deadheading_cost:
                            best_solution = [list(route) for route in current_solution]
                            best_total_deadheading_cost = total_deadheading_cost
                            best_loads = current_loads.copy()
                        improvement = True
                        found = True
                        if logger:
                            logger.log(service_cost_total + total_deadheading_cost)
                        break

    total_cost = service_cost_total + best_total_deadheading_cost
    routes_output = []
    for k in range(num_vehicles):
        route = best_solution[k]
        if not route:
            routes_output.append({
                "vehicle": k,
                "serviced_edges": [],
                "deadheaded_edges": []
            })
            continue
        serviced_edges = [edge[0] for edge in route]
        segments = []
        segments.append((depot, route[0][1]))
        for idx in range(len(route)-1):
            segments.append((route[idx][2], route[idx+1][1]))
        segments.append((route[-1][2], depot))
        deadheading_counts = defaultdict(int)
        for u, v in segments:
            if u == v:
                continue
            path_edges = dijkstra_path_edges(graph, u, v)
            for eid in path_edges:
                deadheading_counts[eid] += 1
        deadheaded_list = [{"edge_id": eid, "times": count} for eid, count in deadheading_counts.items()]
        routes_output.append({
            "vehicle": k,
            "serviced_edges": serviced_edges,
            "deadheaded_edges": deadheaded_list
        })
    output = {
        "objective_value": total_cost,
        "routes": routes_output
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()