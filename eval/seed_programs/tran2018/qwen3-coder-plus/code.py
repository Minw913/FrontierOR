import argparse
import json
import math
from collections import defaultdict
from solution_logger import SolutionLogger
import itertools


def solve_station_location(input_data, time_limit):
    n = input_data['parameters']['n']
    R = input_data['parameters']['R']  # Vehicle driving range
    p = input_data['parameters']['p']  # Number of stations to open
    od_pairs = input_data['od_pairs']
    nodes = input_data['nodes']
    
    # Create mapping of node id to coordinates
    node_coords = {}
    for node in nodes:
        node_coords[node['id']] = (node['x'], node['y'])
    
    # Create adjacency graph to compute pair-wise distances between nodes
    adj = defaultdict(list)
    for arc in input_data['arcs']:
        u, v = arc['from'], arc['to']
        dist = arc['distance']
        adj[u].append((v, dist))
        adj[v].append((u, dist))

    # Function to compute shortest distance between two nodes - Dijkstra algorithm
    def dijkstra(start, end, graph):
        import heapq
        dist = {i: float('inf') for i in range(n)}
        dist[start] = 0
        pq = [(0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if u == end:
                return d
            if d > dist[u]:
                continue
                
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
                    
        return dist[end]

    # Precompute shortest distances between all pairs of nodes
    all_pair_distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = dijkstra(i, j, adj)
            all_pair_distances[(i, j)] = dist_ij
            all_pair_distances[(j, i)] = dist_ij

    # Process each OD trip by expanding it to round trip
    expanded_od_trips = []
    for idx, od in enumerate(od_pairs):
        origin = od['origin']
        dest = od['destination']
        
        # We have the shortest path from origin to destination
        forward_path = od['shortest_path']
        
        # Backward trip is the same path in reverse
        backward_path = list(reversed(forward_path))
        
        trip_info = {
            'id': idx,
            'flow': od['flow'],
            'forward_path': forward_path,
            'backward_path': backward_path,
            'origin': origin,
            'dest': dest
        }
        expanded_od_trips.append(trip_info)

    # Find for each trip which arcs require coverage by refueling stations
    trip_requirements = []

    for trip in expanded_od_trips:
        # Full round trip: forward path then reversed for backward path
        round_trip_path = trip['forward_path'] + trip['backward_path'][1:]  # Skip first node to avoid duplicate
        trip_id = trip['id']
        flow = trip['flow']

        arcs_needing_coverage = []
        
        # Compute cumulative distances along the round trip
        cumul_dists = [0.0]
        for i in range(1, len(round_trip_path)):
            prev_node = round_trip_path[i-1]
            curr_node = round_trip_path[i]
            edge_dist = all_pair_distances.get((prev_node, curr_node), float('inf'))  # Edge length
            cumul_dists.append(cumul_dists[-1] + edge_dist)
        
        # Determine which arcs need to be serviced by refueling stations
        # Arc i is the connection from node i to i+1
        for i in range(len(round_trip_path) - 1):
            dist_to_arc_start = cumul_dists[i]  # Distance from origin to the start of this arc
            if dist_to_arc_start > R:
                # This arc needs to be covered by a refueling station
                arcs_needing_coverage.append({
                    'from_node': round_trip_path[i],
                    'to_node': round_trip_path[i+1],
                    'dist_from_origin': dist_to_arc_start
                })
        
        # For each arc that needs coverage, find all potential stations that can service it
        potential_stations = set(range(n))  # Potential everywhere
        if len(arcs_needing_coverage) > 0:
            valid_station_candidates = []
            for arc in arcs_needing_coverage:
                # To service this arc, station must be reachable from the arc start within the range
                # This means distance from station to arc-start-node + length of arc <= R
                possible_stations = []
                for st_idx in range(n):
                    dist_st_to_arc_start = all_pair_distances.get((st_idx, arc['from_node']), float('inf'))
                    if dist_st_to_arc_start <= R:
                        possible_stations.append(st_idx)
                
                if not possible_stations:
                    # If we already can't service this arc, this is problematic
                    pass
                
                # For this trip to be fully covered, we need AT LEAST ONE station
                # from the intersection of options for each required arc.
                # But since that's complex for a greedy, let's just use union (meaning any of them)
                for s in possible_stations:
                    potential_stations.add(s)

            # Finally, a station is beneficial if choosing it helps satisfy a requirement for this trip
            # We'll calculate the set of stations that can potentially serve some arc of this trip
            final_st_list = list(potential_stations)
        else:
            final_st_list = list(range(n))  # Trip doesn't need stations, any station is valid

        trip_requirements.append({
            'trip_id': trip_id,
            'flow': flow,
            'potential_stations': final_st_list,
            'arcs_needing_coverage': arcs_needing_coverage
        })

    # Apply greedy algorithm to solve the max coverage problem
    chosen_stations = []
    covered_trips_indices = set()

    for _ in range(p):
        # Find the next best station to add
        best_station = None
        best_marginal_gain = -1

        # Get all station candidates across non-covered trips
        non_covered_trips_reqs = []
        for req in trip_requirements:
            if req['trip_id'] not in covered_trips_indices:
                non_covered_trips_reqs.append(req)

        potential_stations = set()
        for req in non_covered_trips_reqs:
            potential_stations.update(req['potential_stations'])

        for st in potential_stations:
            if st in chosen_stations:
                continue

            # How many currently uncovered trips would be covered by adding station 'st'?
            newly_covered = set()
            for req in non_covered_trips_reqs:
                # If this station is in the valid set of stations for this trip, 
                # the trip gets covered if it wasn't already
                if st in req['potential_stations']:
                    newly_covered.add(req['trip_id'])

            marginal_flow = sum(req['flow'] for req in trip_requirements if req['trip_id'] in newly_covered)
            
            if marginal_flow > best_marginal_gain:
                best_marginal_gain = marginal_flow
                best_station = st
                best_newly_covered = newly_covered

        if best_station is not None:
            chosen_stations.append(best_station)
            covered_trips_indices.update(best_newly_covered)
        else:
            break  # No more improvement available

    # Calculate final objective value by checking which trips are fully served by chosen stations
    final_covered_trips = []
    for req in trip_requirements:
        req_covered = False
        for arc_needs_cover in req['arcs_needing_coverage']:
            # Need at least one station that can service this critical arc
            arc_covered = False
            for st in chosen_stations:
                # Check if station st can service this arc
                dist_st_to_arc_start = all_pair_distances.get((st, arc_needs_cover['from_node']), float('inf'))
                if dist_st_to_arc_start <= R:
                    # Station can reach the arc start in time - we assume it covers the arc
                    arc_covered = True
                    break
            if not arc_covered:
                req_covered = False  # If any arc isn't covered, the trip isn't satisfied
                break
        else:
            req_covered = True  # All arcs are covered
        
        if req_covered:
            final_covered_trips.append(req['trip_id'])

    total_flow = sum(trip['flow'] for i, trip in enumerate(expanded_od_trips) if i in final_covered_trips)

    solution = {
        "objective_value": total_flow,
        "stations": chosen_stations,
        "refueled_od_pairs": final_covered_trips
    }

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)

    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Solve the problem
    solution = solve_station_location(instance_data, args.time_limit)
    
    # Log final solution if logger exists
    if logger:
        logger.log(solution["objective_value"])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()