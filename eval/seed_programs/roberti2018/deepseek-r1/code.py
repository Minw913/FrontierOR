import json
import math
import argparse
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def euclidean_distance(a, b):
    return math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)

def precompute_distances(depot, satellites, customers):
    nodes = []
    nodes.append({'id': depot['id'], 'x': depot['x'], 'y': depot['y']})
    for s in satellites:
        nodes.append(s)
    for c in customers:
        nodes.append(c)
    
    dist = {}
    for i in nodes:
        for j in nodes:
            dist[(i['id'], j['id'])] = euclidean_distance(i, j)
    return dist

def compute_group_data(customers, groups):
    group_info = {}
    for g in groups:
        group_id = g['group_id']
        customer_ids = g['customer_ids']
        total_demand = 0
        xs = []
        ys = []
        for cid in customer_ids:
            customer = next(c for c in customers if c['id'] == cid)
            total_demand += customer['demand']
            xs.append(customer['x'])
            ys.append(customer['y'])
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        group_info[group_id] = {
            'total_demand': total_demand,
            'centroid': (centroid_x, centroid_y),
            'customer_ids': customer_ids
        }
    return group_info

def assign_groups_with_mip(group_info, satellites, T2, time_limit, dist):
    model = gp.Model('GroupAssignment')
    model.setParam("Threads", 1)
    groups = list(group_info.keys())
    sats = [s['id'] for s in satellites]
    
    x = model.addVars(groups, sats, vtype=GRB.BINARY, name='x')
    
    obj = 0
    for g in groups:
        for s in sats:
            sat = next(sat for sat in satellites if sat['id'] == s)
            centroid = group_info[g]['centroid']
            sat_loc = {'x': sat['x'], 'y': sat['y']}
            d = dist[(sat['id'], sat['id'])]
            cost_second = euclidean_distance(sat_loc, {'x': centroid[0], 'y': centroid[1]}) * group_info[g]['total_demand']
            cost_handling = sat['handling_cost'] * group_info[g]['total_demand']
            obj += (cost_second + cost_handling) * x[g, s]
    model.setObjective(obj, GRB.MINIMIZE)
    
    for g in groups:
        model.addConstr(gp.quicksum(x[g, s] for s in sats) == 1, f'assign_{g}')
    
    for s in sats:
        sat = next(sat for sat in satellites if sat['id'] == s)
        model.addConstr(gp.quicksum(group_info[g]['total_demand'] * x[g, s] for g in groups) <= sat['capacity'], f'cap_{s}')
        min_vehicles = math.ceil(group_info[g]['total_demand'] / Q2) if Q2 > 0 else 0
        model.addConstr(gp.quicksum(min_vehicles * x[g, s] for g in groups) <= sat['max_vehicles'], f'maxv_{s}')
    
    total_min_vehicles = gp.quicksum(math.ceil(group_info[g]['total_demand'] / Q2) * x[g, s] for g in groups for s in sats)
    model.addConstr(total_min_vehicles <= T2, 'total_vehicles')
    
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignment = {}
        for g in groups:
            for s in sats:
                if x[g, s].X > 0.5:
                    assignment[g] = s
        return assignment, model.objVal
    else:
        return None, None

def assign_groups_greedy(group_info, satellites, T2, dist):
    groups_sorted = sorted(group_info.keys(), key=lambda g: group_info[g]['total_demand'], reverse=True)
    remaining_cap = {s['id']: s['capacity'] for s in satellites}
    remaining_veh = {s['id']: s['max_vehicles'] for s in satellites}
    remaining_T2 = T2
    assignment = {}
    total_handling_cost = 0
    total_second_cost = 0
    
    for g in groups_sorted:
        total_demand = group_info[g]['total_demand']
        min_vehicles = math.ceil(total_demand / Q2)
        centroid = group_info[g]['centroid']
        best_sat = None
        best_cost = float('inf')
        for s in satellites:
            if total_demand <= remaining_cap[s['id']] and min_vehicles <= remaining_veh[s['id']] and min_vehicles <= remaining_T2:
                cost_second = euclidean_distance(s, {'x': centroid[0], 'y': centroid[1]}) * total_demand
                cost_handling = s['handling_cost'] * total_demand
                cost = cost_second + cost_handling
                if cost < best_cost:
                    best_cost = cost
                    best_sat = s['id']
        if best_sat is None:
            best_cost = float('inf')
            for s in satellites:
                violation_cap = max(0, total_demand - remaining_cap[s['id']])
                violation_veh = max(0, min_vehicles - remaining_veh[s['id']])
                violation_T2 = max(0, min_vehicles - remaining_T2)
                violation = violation_cap + violation_veh + violation_T2
                if violation < best_cost:
                    best_cost = violation
                    best_sat = s['id']
        assignment[g] = best_sat
        if total_demand <= remaining_cap[best_sat] and min_vehicles <= remaining_veh[best_sat] and min_vehicles <= remaining_T2:
            remaining_cap[best_sat] -= total_demand
            remaining_veh[best_sat] -= min_vehicles
            remaining_T2 -= min_vehicles
            centroid = group_info[g]['centroid']
            s_obj = next(s for s in satellites if s['id'] == best_sat)
            total_second_cost += euclidean_distance(s_obj, {'x': centroid[0], 'y': centroid[1]}) * total_demand
            total_handling_cost += s_obj['handling_cost'] * total_demand
    total_estimated_cost = total_second_cost + total_handling_cost
    return assignment, total_estimated_cost

def savings_algorithm(customers, depot_id, vehicle_capacity, dist):
    if not customers:
        return []
    demand = {c['id']: c['demand'] for c in customers}
    nodes = [c['id'] for c in customers]
    routes = [[depot_id, node, depot_id] for node in nodes]
    route_demands = [demand[node] for node in nodes]
    savings = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            ni = nodes[i]
            nj = nodes[j]
            s = dist[(depot_id, ni)] + dist[(depot_id, nj)] - dist[(ni, nj)]
            savings.append((s, i, j, ni, nj))
    savings.sort(key=lambda x: x[0], reverse=True)
    merged = [False] * len(routes)
    for s, i, j, ni, nj in savings:
        if merged[i] or merged[j]:
            continue
        route_i = routes[i]
        route_j = routes[j]
        total_demand = route_demands[i] + route_demands[j]
        if total_demand > vehicle_capacity:
            continue
        if route_i[-2] == depot_id and route_j[1] == depot_id:
            new_route = route_i[:-1] + route_j[1:]
            routes[i] = new_route
            route_demands[i] = total_demand
            merged[j] = True
            routes[j] = None
        elif route_j[-2] == depot_id and route_i[1] == depot_id:
            new_route = route_j[:-1] + route_i[1:]
            routes[j] = new_route
            route_demands[j] = total_demand
            merged[i] = True
            routes[i] = None
    final_routes = [r for r in routes if r is not None and len(r) > 2]
    return final_routes

def nearest_neighbor_tsp(nodes, start_id, dist):
    if not nodes:
        return [start_id, start_id]
    unvisited = set(nodes)
    current = start_id
    route = [start_id]
    unvisited.remove(current)
    while unvisited:
        next_node = min(unvisited, key=lambda n: dist[(current, n)])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    route.append(start_id)
    return route

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    global Q2, Q1
    depot = data['depot']
    satellites = data['satellites']
    customers = data['customers']
    groups = data['groups']
    num_satellites = data['num_satellites']
    num_customers = data['num_customers']
    num_groups = data['num_groups']
    Q1 = data['Q1']
    Q2 = data['Q2']
    T1 = data['T1']
    T2 = data['T2']
    
    dist = precompute_distances(depot, satellites, customers)
    group_info = compute_group_data(customers, groups)
    
    time_used = time.time() - start_time
    time_left = args.time_limit - time_used
    if time_left <= 1:
        time_left = 1
    
    if num_groups * num_satellites <= 2000 and num_groups <= 200 and num_satellites <= 50:
        assignment, _ = assign_groups_with_mip(group_info, satellites, T2, min(10, time_left), dist)
    else:
        assignment, _ = assign_groups_greedy(group_info, satellites, T2, dist)
    
    if assignment is None:
        assignment, _ = assign_groups_greedy(group_info, satellites, T2, dist)
    
    group_assignments_output = {gid: sat_id for gid, sat_id in assignment.items()}
    
    satellite_to_customers = defaultdict(list)
    for gid, sat_id in assignment.items():
        for cid in group_info[gid]['customer_ids']:
            customer = next(c for c in customers if c['id'] == cid)
            satellite_to_customers[sat_id].append(customer)
    
    second_echelon_routes = []
    satellite_vehicle_count = defaultdict(int)
    total_vehicles_used = 0
    for sat_id, cust_list in satellite_to_customers.items():
        if not cust_list:
            continue
        sat_obj = next(s for s in satellites if s['id'] == sat_id)
        routes = savings_algorithm(cust_list, sat_id, Q2, dist)
        for route in routes:
            second_echelon_routes.append({
                'satellite': sat_id,
                'route': route
            })
        num_vehicles = len(routes)
        satellite_vehicle_count[sat_id] = num_vehicles
        total_vehicles_used += num_vehicles
    
    if total_vehicles_used > T2:
        pass
    
    satellite_total_demand = defaultdict(int)
    for sat_id, cust_list in satellite_to_customers.items():
        satellite_total_demand[sat_id] = sum(c['demand'] for c in cust_list)
    
    chunks = []
    for sat_id, total_demand in satellite_total_demand.items():
        if total_demand == 0:
            continue
        num_chunks = math.ceil(total_demand / Q1)
        for i in range(num_chunks):
            if i < num_chunks - 1:
                chunk_size = Q1
            else:
                chunk_size = total_demand - (num_chunks-1)*Q1
            chunks.append((sat_id, chunk_size))
    
    chunks.sort(key=lambda x: x[1], reverse=True)
    bins = []
    for (sat_id, chunk_size) in chunks:
        placed = False
        for bin in bins:
            if sat_id not in bin['sats'] and bin['total'] + chunk_size <= Q1:
                bin['total'] += chunk_size
                bin['sats'].append(sat_id)
                bin['deliveries'][sat_id] = bin['deliveries'].get(sat_id, 0) + chunk_size
                placed = True
                break
        if not placed:
            new_bin = {'total': chunk_size, 'sats': [sat_id], 'deliveries': {sat_id: chunk_size}}
            bins.append(new_bin)
    
    first_echelon_routes = []
    for i, bin in enumerate(bins):
        if not bin['sats']:
            continue
        route_nodes = [depot['id']] + bin['sats'] + [depot['id']]
        if len(route_nodes) > 3:
            tsp_route = nearest_neighbor_tsp(bin['sats'], depot['id'], dist)
        else:
            tsp_route = route_nodes
        first_echelon_routes.append({
            'vehicle': i,
            'route': tsp_route,
            'deliveries': bin['deliveries']
        })
    
    first_echelon_travel_cost = 0
    for route_data in first_echelon_routes:
        route = route_data['route']
        for i in range(len(route)-1):
            first_echelon_travel_cost += dist[(route[i], route[i+1])]
    
    handling_cost_total = 0
    for sat_id, total_demand in satellite_total_demand.items():
        sat = next(s for s in satellites if s['id'] == sat_id)
        handling_cost_total += sat['handling_cost'] * total_demand
    
    second_echelon_travel_cost = 0
    for route_data in second_echelon_routes:
        route = route_data['route']
        for i in range(len(route)-1):
            second_echelon_travel_cost += dist[(route[i], route[i+1])]
    
    objective_value = first_echelon_travel_cost + handling_cost_total + second_echelon_travel_cost
    
    solution = {
        "objective_value": objective_value,
        "first_echelon_routes": first_echelon_routes,
        "second_echelon_routes": second_echelon_routes,
        "group_assignments": group_assignments_output
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    if logger:
        logger.log(objective_value)

if __name__ == '__main__':
    main()