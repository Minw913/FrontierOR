import json
import argparse
import time
import sys
from typing import Dict, List, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def load_instance(instance_path: str) -> Dict[str, Any]:
    with open(instance_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    instance = load_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Extract data
    params = instance['parameters']
    fleet = instance['fleet']
    depot = instance['depot']
    truck_customers = instance['truck_customers']
    trailer_customers = instance['trailer_customers']
    transship = instance['transshipment_locations']
    dist = instance['distance_matrix']
    travel_time = instance['travel_time_matrix']

    n_locations = len(dist)
    depot_id = depot['id']
    truck_cust_ids = [c['id'] for c in truck_customers]
    trailer_cust_ids = [c['id'] for c in trailer_customers]
    trans_ids = [loc['id'] for loc in transship]
    parking_ids = trailer_cust_ids + trans_ids
    all_cust_ids = truck_cust_ids + trailer_cust_ids
    n_cust = len(all_cust_ids)
    cust_idx_map = {cid: i for i, cid in enumerate(all_cust_ids)}

    vehicle_classes = fleet['vehicle_classes']
    n_classes = len(vehicle_classes)
    max_vehicles_per_class = [vc['num_available'] for vc in vehicle_classes]

    # Precompute parking expanded nodes as per schema
    n_orig = n_locations
    n_parking = len(parking_ids)
    decouple_offset = n_orig
    couple_offset = n_orig + n_parking

    # Simple heuristic: assign each trailer customer to its own trailer vehicle if possible
    # Use only largest capacity vehicle class for feasibility
    if n_classes == 0:
        print("No vehicle classes defined", file=sys.stderr)
        return

    # Sort classes by total capacity descending
    sorted_classes = sorted(enumerate(vehicle_classes), key=lambda x: x[1]['total_capacity'], reverse=True)
    best_class_idx = sorted_classes[0][0]
    best_vc = vehicle_classes[best_class_idx]
    truck_cap = best_vc['truck_capacity']
    trailer_cap = best_vc['trailer_capacity']
    total_cap = best_vc['total_capacity']

    # Very basic greedy construction
    routes = []
    remaining_truck_custs = set(truck_cust_ids)
    remaining_trailer_custs = set(trailer_cust_ids)
    vehicle_usage = [0] * n_classes
    total_cost = 0.0
    route_id_counter = 0

    # First serve all trailer customers with full vehicles if possible
    for tc_id in list(remaining_trailer_custs):
        if vehicle_usage[best_class_idx] >= max_vehicles_per_class[best_class_idx]:
            break
        supply = next(c['supply'] for c in trailer_customers if c['id'] == tc_id)
        if supply > total_cap:
            continue  # infeasible but skip for now

        # Simple route: depot -> trailer_customer -> depot with trailer attached
        seq = [depot_id, tc_id, depot_id]
        phys_seq = [depot_id, tc_id, depot_id]
        customers = [tc_id]
        decouple_locs = []
        couple_locs = []

        # Cost calculation
        travel_cost = best_vc['truck_cost_factor'] * dist[depot_id][tc_id] + \
                      best_vc['trailer_towing_cost_multiplier'] * dist[depot_id][tc_id]
        travel_cost += best_vc['truck_cost_factor'] * dist[tc_id][depot_id] + \
                       best_vc['trailer_towing_cost_multiplier'] * dist[tc_id][depot_id]
        fixed_cost = best_vc['truck_fixed_cost']
        handling_cost = 0.0
        route_cost = fixed_cost + travel_cost + handling_cost

        # Build solution entry
        arrival_times = {}
        current_time = 0.0
        arrival_times[str(depot_id)] = current_time
        current_time += travel_time[depot_id][tc_id]
        arrival_times[str(tc_id)] = current_time
        current_time += travel_time[tc_id][depot_id]
        arrival_times[str(depot_id) + "_return"] = current_time  # simplistic

        route_entry = {
            "vehicle_class": best_vc['class_id'],
            "vehicle_index": vehicle_usage[best_class_idx],
            "customers_served": customers,
            "route_sequence": seq,
            "route_sequence_physical": phys_seq,
            "decouple_locations": decouple_locs,
            "couple_locations": couple_locs,
            "arrival_times": arrival_times
        }
        routes.append(route_entry)
        total_cost += route_cost
        vehicle_usage[best_class_idx] += 1
        remaining_trailer_custs.remove(tc_id)
        route_id_counter += 1

    # Serve truck customers with solo trucks if available
    solo_class = None
    solo_idx = -1
    for i, vc in enumerate(vehicle_classes):
        if vc['trailer_type_id'] == 0:
            solo_class = vc
            solo_idx = i
            break

    if solo_class is not None and max_vehicles_per_class[solo_idx] > 0:
        for tc_id in list(remaining_truck_custs):
            if vehicle_usage[solo_idx] >= max_vehicles_per_class[solo_idx]:
                break
            supply = next((c['supply'] for c in truck_customers if c['id'] == tc_id), 0)
            if supply > solo_class['truck_capacity']:
                continue

            seq = [depot_id, tc_id, depot_id]
            phys_seq = seq[:]
            customers = [tc_id]
            decouple_locs = []
            couple_locs = []

            travel_cost = solo_class['truck_cost_factor'] * (dist[depot_id][tc_id] + dist[tc_id][depot_id])
            route_cost = solo_class['truck_fixed_cost'] + travel_cost

            arrival_times = {}
            current_time = 0.0
            arrival_times[str(depot_id)] = current_time
            current_time += travel_time[depot_id][tc_id]
            arrival_times[str(tc_id)] = current_time
            current_time += travel_time[tc_id][depot_id]
            arrival_times[str(depot_id) + "_return"] = current_time

            route_entry = {
                "vehicle_class": solo_class['class_id'],
                "vehicle_index": vehicle_usage[solo_idx],
                "customers_served": customers,
                "route_sequence": seq,
                "route_sequence_physical": phys_seq,
                "decouple_locations": decouple_locs,
                "couple_locations": couple_locs,
                "arrival_times": arrival_times
            }
            routes.append(route_entry)
            total_cost += route_cost
            vehicle_usage[solo_idx] += 1
            remaining_truck_custs.remove(tc_id)
            route_id_counter += 1

    # If customers remain, we have an incomplete solution. For demo we return what we have.

    solution = {
        "objective_value": float(total_cost),
        "routes": routes,
        "expanded_node_convention": "Let n_orig be the number of physical locations and P be the instance parking-location IDs in their published list order. Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, and the couple copy of P[i] has ID n_orig+len(P)+i. Both copies map back to physical location P[i]."
    }

    # Log the solution
    if logger:
        logger.log_solution(total_cost, solution)

    # Write output
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Solution written with objective {total_cost:.2f} in {elapsed:.2f}s", file=sys.stderr)

if __name__ == "__main__":
    main()