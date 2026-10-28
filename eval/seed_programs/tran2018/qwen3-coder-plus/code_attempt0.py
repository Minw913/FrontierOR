import argparse
import json
import math
from collections import defaultdict
import random
from solution_logger import SolutionLogger


def calculate_path_distances(path):
    """Calculate cumulative distances along path."""
    cumul_distances = [0]  # Start with 0 for the origin
    for i in range(1, len(path)):
        # Calculate distance between adjacent nodes in path
        dist = math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
        cumul_distances.append(cumul_distances[-1] + dist)
    return cumul_distances


def solve_station_location(input_data, time_limit):
    n = input_data['parameters']['n']
    R = input_data['parameters']['R']  # Vehicle driving range
    p = input_data['parameters']['p']  # Number of stations to open
    od_pairs = input_data['od_pairs']
    
    # Create mapping of node id to coordinates
    node_coords = {}
    for node in input_data['nodes']:
        node_coords[node['id']] = (node['x'], node['y'])
    
    # Expand each OD pair to cover both legs (o->d and d->o)
    expanded_od_trips = []
    for idx, od in enumerate(od_pairs):
        origin = od['origin']
        dest = od['destination']
        
        # Forward leg: from origin to destination
        shortest_path_ids = od['shortest_path']
        
        # Expand the path to be a list of nodes with their coordinates
        forward_path_with_coords = [(node_coords[node_id]['x'], node_coords[node_id]['y']) for node_id in shortest_path_ids]
        
        # Backward leg: from destination to origin
        backward_path_with_coords = list(reversed(forward_path_with_coords))
        backward_path_ids = list(reversed(shortest_path_ids))
        
        expanded_od_trips.append({
            'id': idx,
            'flow': od['flow'],
            'forward_path_ids': shortest_path_ids,
            'forward_path_coords': forward_path_with_coords,
            'backward_path_ids': backward_path_ids,
            'backward_path_coords': backward_path_with_coords,
            'origin': origin,
            'destination': dest
        })
    
    # Now we need to determine what arcs on each trip can be served by each potential station
    # Arcs are consecutive pairs of nodes in the round trip
    
    def get_path_cumulative_distances(node_path, node_coords):
        """Get cumulative distances along the path"""
        dists = [0.0]
        for i in range(1, len(node_path)):
            x1, y1 = node_coords[node_path[i-1]]['x'], node_coords[node_path[i-1]]['y']
            x2, y2 = node_coords[node_path[i]]['x'], node_coords[node_path[i]]['y']
            edge_dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            dists.append(dists[-1] + edge_dist)
        return dists
    
    # Precompute for each OD trip the list of directed arcs and when they become unreachable
    trip_requirements = []  # For each trip, which stations could serve it
    
    for trip_idx, trip in enumerate(expanded_od_trips):
        forward_dists = get_path_cumulative_distances(trip['forward_path_ids'], node_coords)
        backward_dists = get_path_cumulative_distances(trip['backward_path_ids'], node_coords)
        
        # Full round trip is forward followed by backward (with offset because backward starts after forward leg)
        total_forward_dist = forward_dists[-1]
        round_trip_ids = trip['forward_path_ids'] + trip['backward_path_ids'][1:]  # Skip first to avoid duplicate
        round_trip_dists = forward_dists + [d + total_forward_dist for d in backward_dists[1:]]
        
        # List of arcs in the round trip
        round_trip_arcs = []
        for i in range(len(round_trip_ids) - 1):
            # An arc is from round_trip_ids[i] to round_trip_ids[i+1], 
            # but what matters is the distance from the origin of the whole trip
            # At start of arc from point i to point i+1, 
            # the vehicle has traveled round_trip_dists[i] distance since the beginning 
            arc_info = {
                'arc': (round_trip_ids[i], round_trip_ids[i+1]), 
                'start_from_origin_dist': round_trip_dists[i],
                'end_reached_dist': round_trip_dists[i+1],
                'from_node': round_trip_ids[i],
                'to_node': round_trip_ids[i+1],
                'arc_length': round_trip_dists[i+1] - round_trip_dists[i]
            }
            round_trip_arcs.append(arc_info)
        
        # Calculate cumulative trip distance starting from origin (first leg and back)
        max_range_at_arc_start = R
        arcs_need_coverage = []
        for a_idx, arc_info in enumerate(round_trip_arcs):
            start_dist = arc_info['start_from_origin_dist']
            
            # For first leg - if we can reach the arc start from origin with initial fuel, it's fine
            if start_dist <= R:
                continue  # This arc doesn't need special station coverage

            arcs_need_coverage.append(a_idx)  # Need station to help serve this arc
        
        # Find what stations can serve these critical arcs
        valid_stations_for_trip = set()
        for c_idx in arcs_need_coverage:
            arc_info = round_trip_arcs[c_idx]
            
            # A station at 'station_node_id' can serve arc if:
            # vehicle can drive from station to arc start and across arc itself within range R
            for st_node_id in range(n):  # Possible station locations could be all nodes
                # Distance from station to start of arc (need to trace backwards)
                x1, y1 = node_coords[st_node_id]['x'], node_coords[st_node_id]['y']
                x2, y2 = node_coords[arc_info['from_node']]['x'], node_coords[arc_info['from_node']]['y']
                dist_from_station_to_arc_start = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # The vehicle leaving station has full tank R
                # So it can reach an arc if distance to arc start <= R AND it can traverse the arc
                # Since arc length <= remaining range when reaching start
                if dist_from_station_to_arc_start + arc_info['arc_length'] <= R:
                    valid_stations_for_trip.add(st_node_id)
        
        trip_requirements.append({
            'trip_id': trip_idx,
            'valid_stations': list(valid_stations_for_trip),
            'arcs_need_coverage': arcs_need_coverage  # Arc IDs in the round trip that require station coverage
        })

    # We have reduced this to a set cover-like problem
    # Maximize weighted set covering with constraint of using at most p stations
    # This is a maximum coverage problem variant
    
    # Greedy heuristic for max coverage with budget p
    eligible_stations = set()
    for req in trip_requirements:
        for st_id in req['valid_stations']:
            eligible_stations.add(st_id)
    
    chosen_stations = set()
    covered_trips = set()
    
    # In greedy algorithm we select stations iteratively that covers most additional uncovered demand
    for iter_num in range(p):
        best_station = None
        best_additional_flow = -1
        best_newly_covered_trips = set()
        
        for st_candidate in eligible_stations:
            if st_candidate in chosen_stations:
                continue
            
            newly_covered_here = set()
            for req in trip_requirements:
                # Check if this station together with already chosen ones covers this trip
                required_stations_set = set(req['valid_stations'])
                
                # If adding this candidate station satisfies the requirements AND the trip isn't already covered
                # Then this trip will be newly satisfied
                if req['trip_id'] in covered_trips:
                    # Already covered with current stations, nothing new
                    continue
                
                if st_candidate in required_stations_set:
                    newly_covered_here.add(req['trip_id'])
            
            # Calculate total additional flow if we add this station
            additional_flow = sum(expanded_od_trips[tid]['flow'] for tid in newly_covered_here)
            
            if additional_flow > best_additional_flow:
                best_additional_flow = additional_flow
                best_station = st_candidate
                best_newly_covered_trips = newly_covered_here.copy()
        
        if best_station is not None:
            chosen_stations.add(best_station)
            covered_trips.update(best_newly_covered_trips)
        else:
            # No station provides benefit, break early
            break
    
    # Final check to make sure we didn't select too many: trim to p if needed
    chosen_stations = list(chosen_stations)[:p]
    
    # Double check which trips are actually satisfied by our chosen stations
    final_covered_trips = set()
    station_set = set(chosen_stations)
    for req in trip_requirements:
        # A trip is covered if it has at least one of its valid stations in the chosen set
        if any(s in station_set for s in req['valid_stations']):
            final_covered_trips.add(req['trip_id'])
    
    # Calculate the final objective value
    obj_val = sum(expanded_od_trips[trip_id]['flow'] for trip_id in final_covered_trips)
    
    solution = {
        "objective_value": obj_val,
        "stations": chosen_stations,
        "refueled_od_pairs": sorted(list(final_covered_trips))
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
    
    # Log final solution if logger exist
    if logger:
        logger.log(solution["objective_value"])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()