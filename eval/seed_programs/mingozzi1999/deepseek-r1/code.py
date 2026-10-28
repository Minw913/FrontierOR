import argparse
import json
import time
from math import sqrt
from solution_logger import SolutionLogger

def build_route(linehaul_list, backhaul_list, dist, depot_index=0):
    if not linehaul_list and not backhaul_list:
        return [depot_index, depot_index], 0
    if not backhaul_list:
        nodes = [depot_index] + linehaul_list
        tour = [depot_index]
        unvisited = set(linehaul_list)
        current = depot_index
        while unvisited:
            next_node = min(unvisited, key=lambda x: dist[current][x])
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        tour.append(depot_index)
        cost = 0
        for i in range(len(tour)-1):
            cost += dist[tour[i]][tour[i+1]]
        return tour, cost
    else:
        tour_linehaul = [depot_index]
        unvisited_linehaul = set(linehaul_list)
        current = depot_index
        while unvisited_linehaul:
            next_node = min(unvisited_linehaul, key=lambda x: dist[current][x])
            tour_linehaul.append(next_node)
            unvisited_linehaul.remove(next_node)
            current = next_node
        last_linehaul = current
        if len(backhaul_list) <= 5:
            candidates = backhaul_list
        else:
            candidates = sorted(backhaul_list, key=lambda x: dist[last_linehaul][x])[:5]
        best_tour = None
        best_cost = float('inf')
        for first_backhaul in candidates:
            tour_backhaul = [first_backhaul]
            unvisited_backhaul = set(backhaul_list) - {first_backhaul}
            cur = first_backhaul
            while unvisited_backhaul:
                next_node = min(unvisited_backhaul, key=lambda x: dist[cur][x])
                tour_backhaul.append(next_node)
                unvisited_backhaul.remove(next_node)
                cur = next_node
            tour_candidate = tour_linehaul + tour_backhaul + [depot_index]
            cost = 0
            for i in range(len(tour_candidate)-1):
                cost += dist[tour_candidate[i]][tour_candidate[i+1]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour_candidate
        return best_tour, best_cost

def main():
    parser = argparse.ArgumentParser(description='Solve Vehicle Routing Problem with Backhauls.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions.')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    depot_index = instance['depot']['index']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    linehaul_customers = instance['linehaul_customers']
    backhaul_customers = instance['backhaul_customers']
    dist = instance['distance_matrix']

    demands = {}
    for cust in linehaul_customers:
        demands[cust['index']] = cust['demand']
    for cust in backhaul_customers:
        demands[cust['index']] = cust['demand']

    nL = len(linehaul_customers)
    if nL < num_vehicles:
        raise ValueError(f"Number of linehaul customers ({nL}) is less than number of vehicles ({num_vehicles}).")

    start_time = time.time()

    linehaul_sorted = sorted(linehaul_customers, key=lambda x: x['demand'], reverse=True)
    vehicles = [{'linehaul': [], 'backhaul': []} for _ in range(num_vehicles)]
    linehaul_load = [0] * num_vehicles

    for i in range(num_vehicles):
        cust = linehaul_sorted[i]
        vehicles[i]['linehaul'].append(cust['index'])
        linehaul_load[i] = cust['demand']

    for i in range(num_vehicles, len(linehaul_sorted)):
        cust = linehaul_sorted[i]
        assigned = False
        for k in range(num_vehicles):
            if linehaul_load[k] + cust['demand'] <= vehicle_capacity:
                vehicles[k]['linehaul'].append(cust['index'])
                linehaul_load[k] += cust['demand']
                assigned = True
                break
        if not assigned:
            min_load = min(linehaul_load)
            for k in range(num_vehicles):
                if linehaul_load[k] == min_load:
                    vehicles[k]['linehaul'].append(cust['index'])
                    linehaul_load[k] += cust['demand']
                    break

    backhaul_sorted = sorted(backhaul_customers, key=lambda x: x['demand'], reverse=True)
    backhaul_load = [0] * num_vehicles

    for cust in backhaul_sorted:
        centroids = []
        for k in range(num_vehicles):
            if vehicles[k]['linehaul']:
                coords = []
                for lh_index in vehicles[k]['linehaul']:
                    for cust_info in linehaul_customers:
                        if cust_info['index'] == lh_index:
                            coords.append(cust_info['coordinates'])
                            break
                cx = sum(coord[0] for coord in coords) / len(coords)
                cy = sum(coord[1] for coord in coords) / len(coords)
                centroids.append((cx, cy))
            else:
                centroids.append(tuple(instance['depot']['coordinates']))
        
        xc, yc = cust['coordinates']
        dists = []
        for k in range(num_vehicles):
            cx, cy = centroids[k]
            dx = xc - cx
            dy = yc - cy
            d_val = sqrt(dx**2 + dy**2)
            dists.append((d_val, k))
        dists.sort(key=lambda x: x[0])
        assigned = False
        for d_val, k in dists:
            if backhaul_load[k] + cust['demand'] <= vehicle_capacity:
                vehicles[k]['backhaul'].append(cust['index'])
                backhaul_load[k] += cust['demand']
                assigned = True
                break
        if not assigned:
            min_load = min(backhaul_load)
            for k in range(num_vehicles):
                if backhaul_load[k] == min_load:
                    vehicles[k]['backhaul'].append(cust['index'])
                    backhaul_load[k] += cust['demand']
                    break

    routes = []
    total_cost = 0
    for k in range(num_vehicles):
        linehaul_list = vehicles[k]['linehaul']
        backhaul_list = vehicles[k]['backhaul']
        route, cost = build_route(linehaul_list, backhaul_list, dist, depot_index)
        routes.append(route)
        total_cost += cost

    best_solution = routes
    best_cost = total_cost
    if logger:
        logger.log(best_cost)

    max_time = start_time + args.time_limit
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 1:
        current_solution = routes
        current_cost = best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for cust in linehaul_customers:
                if time.time() >= max_time:
                    break
                cust_index = cust['index']
                current_vehicle = None
                for k in range(num_vehicles):
                    if cust_index in vehicles[k]['linehaul']:
                        current_vehicle = k
                        break
                for new_vehicle in range(num_vehicles):
                    if time.time() >= max_time:
                        break
                    if new_vehicle == current_vehicle:
                        continue
                    if linehaul_load[new_vehicle] + cust['demand'] <= vehicle_capacity:
                        old_linehaul_current = vehicles[current_vehicle]['linehaul'][:]
                        old_linehaul_new = vehicles[new_vehicle]['linehaul'][:]
                        if cust_index not in old_linehaul_current:
                            continue
                        old_linehaul_current.remove(cust_index)
                        old_linehaul_new.append(cust_index)
                        old_backhaul_current = vehicles[current_vehicle]['backhaul'][:]
                        old_backhaul_new = vehicles[new_vehicle]['backhaul'][:]
                        route_current, cost_current = build_route(old_linehaul_current, old_backhaul_current, dist, depot_index)
                        route_new, cost_new = build_route(old_linehaul_new, old_backhaul_new, dist, depot_index)
                        old_route_current = current_solution[current_vehicle]
                        old_route_new = current_solution[new_vehicle]
                        old_cost_current = 0
                        for i in range(len(old_route_current)-1):
                            old_cost_current += dist[old_route_current[i]][old_route_current[i+1]]
                        old_cost_new = 0
                        for i in range(len(old_route_new)-1):
                            old_cost_new += dist[old_route_new[i]][old_route_new[i+1]]
                        old_total_two = old_cost_current + old_cost_new
                        new_total_two = cost_current + cost_new
                        if new_total_two < old_total_two:
                            new_total_cost = current_cost - old_total_two + new_total_two
                            vehicles[current_vehicle]['linehaul'] = old_linehaul_current
                            vehicles[new_vehicle]['linehaul'] = old_linehaul_new
                            linehaul_load[current_vehicle] -= cust['demand']
                            linehaul_load[new_vehicle] += cust['demand']
                            new_solution = current_solution[:]
                            new_solution[current_vehicle] = route_current
                            new_solution[new_vehicle] = route_new
                            current_solution = new_solution
                            current_cost = new_total_cost
                            if current_cost < best_cost:
                                best_solution = current_solution
                                best_cost = current_cost
                                if logger:
                                    logger.log(best_cost)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            for cust in backhaul_customers:
                if time.time() >= max_time:
                    break
                cust_index = cust['index']
                current_vehicle = None
                for k in range(num_vehicles):
                    if cust_index in vehicles[k]['backhaul']:
                        current_vehicle = k
                        break
                for new_vehicle in range(num_vehicles):
                    if time.time() >= max_time:
                        break
                    if new_vehicle == current_vehicle:
                        continue
                    if backhaul_load[new_vehicle] + cust['demand'] <= vehicle_capacity:
                        old_backhaul_current = vehicles[current_vehicle]['backhaul'][:]
                        old_backhaul_new = vehicles[new_vehicle]['backhaul'][:]
                        if cust_index not in old_backhaul_current:
                            continue
                        old_backhaul_current.remove(cust_index)
                        old_backhaul_new.append(cust_index)
                        old_linehaul_current = vehicles[current_vehicle]['linehaul'][:]
                        old_linehaul_new = vehicles[new_vehicle]['linehaul'][:]
                        route_current, cost_current = build_route(old_linehaul_current, old_backhaul_current, dist, depot_index)
                        route_new, cost_new = build_route(old_linehaul_new, old_backhaul_new, dist, depot_index)
                        old_route_current = current_solution[current_vehicle]
                        old_route_new = current_solution[new_vehicle]
                        old_cost_current = 0
                        for i in range(len(old_route_current)-1):
                            old_cost_current += dist[old_route_current[i]][old_route_current[i+1]]
                        old_cost_new = 0
                        for i in range(len(old_route_new)-1):
                            old_cost_new += dist[old_route_new[i]][old_route_new[i+1]]
                        old_total_two = old_cost_current + old_cost_new
                        new_total_two = cost_current + cost_new
                        if new_total_two < old_total_two:
                            new_total_cost = current_cost - old_total_two + new_total_two
                            vehicles[current_vehicle]['backhaul'] = old_backhaul_current
                            vehicles[new_vehicle]['backhaul'] = old_backhaul_new
                            backhaul_load[current_vehicle] -= cust['demand']
                            backhaul_load[new_vehicle] += cust['demand']
                            new_solution = current_solution[:]
                            new_solution[current_vehicle] = route_current
                            new_solution[new_vehicle] = route_new
                            current_solution = new_solution
                            current_cost = new_total_cost
                            if current_cost < best_cost:
                                best_solution = current_solution
                                best_cost = current_cost
                                if logger:
                                    logger.log(best_cost)
                            improved = True
                            break
                if improved:
                    break

    solution = {
        "objective_value": best_cost,
        "routes": best_solution
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()