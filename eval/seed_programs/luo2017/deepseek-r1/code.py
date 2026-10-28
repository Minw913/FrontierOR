import json
import math
import time
import argparse
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()

    logger = SolutionLogger(log_path, sense='minimize') if log_path else None

    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['num_customers']
    customers = instance['customers']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    a_param = instance['cost_function']['a']
    b_param = instance['cost_function']['b']
    dist_mat = instance['distance_matrix']
    
    for row in dist_mat:
        row.append(row[0])

    depot_end_row = dist_mat[0] + [0.0]
    dist_mat.append(depot_end_row)
    
    virtual_depot_id = n + 1

    all_routes = []
    for customer in customers:
        c_id = customer['id']
        c_demand = customer['demand']
        num_trips = math.ceil(c_demand / vehicle_capacity)
        remaining = c_demand
        for _ in range(num_trips):
            q = min(vehicle_capacity, remaining)
            route = [0, c_id, virtual_depot_id]
            all_routes.append((route, q, c_id))
            remaining -= q

    out_routes_list = []
    total_cost = 0.0
    for idx, (route_nodes, q, c_id) in enumerate(all_routes):
        dist_0_to_c = dist_mat[0][c_id]
        dist_c_to_depot = dist_mat[c_id][virtual_depot_id]
        cost_edge1 = dist_0_to_c * (a_param * q + b_param)
        cost_edge2 = dist_c_to_depot * (a_param * 0 + b_param)
        route_cost = cost_edge1 + cost_edge2
        total_cost += route_cost

        deliveries_dict = {str(c_id): float(q)}
        edges_list = [
            {"from": 0, "to": c_id, "load_weight": float(q)},
            {"from": c_id, "to": virtual_depot_id, "load_weight": 0.0}
        ]

        out_route = {
            "vehicle": idx,
            "route": route_nodes,
            "deliveries": deliveries_dict,
            "edges": edges_list
        }
        out_routes_list.append(out_route)

    solution_dict = {
        "objective_value": total_cost,
        "routes": out_routes_list
    }

    elapsed = time.time() - start_time
    if elapsed > time_limit:
        print(f"Warning: solution reached time limit of {time_limit} seconds, elapsed {elapsed}")

    if logger:
        logger.log_solution(total_cost, solution_dict)

    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)