import argparse
import json
import time
import math
from gurobipy import Model, GRB, quicksum

def first_fit_decreasing(items, bin_capacity):
    items_sorted = sorted(items, reverse=True)
    bins = []
    for item in items_sorted:
        placed = False
        for i in range(len(bins)):
            if bins[i] + item <= bin_capacity:
                bins[i] += item
                placed = True
                break
        if not placed:
            bins.append(item)
    return len(bins)

def savings_algorithm(depot_id, customers, demands, vehicle_capacity, distance_matrix):
    if not customers:
        return []
    
    routes = []
    for c in customers:
        routes.append([depot_id, c, depot_id])
    loads = [demands[c] for c in customers]
    route_details = []
    for i in range(len(customers)):
        route = routes[i]
        load = loads[i]
        route_details.append({
            'route': route,
            'load': load,
            'end0': customers[i],
            'end1': customers[i]
        })
    
    in_route = {}
    for idx, c in enumerate(customers):
        in_route[c] = idx

    savings_list = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s = distance_matrix[depot_id][c1] + distance_matrix[depot_id][c2] - distance_matrix[c1][c2]
            savings_list.append((c1, c2, s))
    
    savings_list.sort(key=lambda x: x[2], reverse=True)
    
    for (c1, c2, s_val) in savings_list:
        idx1 = in_route.get(c1)
        idx2 = in_route.get(c2)
        if idx1 is None or idx2 is None or idx1 == idx2:
            continue
        
        route1 = route_details[idx1]
        route2 = route_details[idx2]
        
        merged = False
        new_route = None
        new_load = None
        new_end0 = None
        new_end1 = None
        
        if route1['end1'] == c1 and route2['end0'] == c2 and route1['load'] + route2['load'] <= vehicle_capacity:
            new_route = route1['route'][:-1] + route2['route'][1:]
            new_load = route1['load'] + route2['load']
            new_end0 = route1['end0']
            new_end1 = route2['end1']
            merged = True
        elif route1['end1'] == c1 and route2['end1'] == c2 and route1['load'] + route2['load'] <= vehicle_capacity:
            reversed_route2 = [depot_id] + list(reversed(route2['route'][1:-1])) + [depot_id]
            new_route = route1['route'][:-1] + reversed_route2[1:]
            new_load = route1['load'] + route2['load']
            new_end0 = route1['end0']
            new_end1 = route2['route'][1] 
            merged = True
        elif route1['end0'] == c1 and route2['end0'] == c2 and route1['load'] + route2['load'] <= vehicle_capacity:
            reversed_route1 = [depot_id] + list(reversed(route1['route'][1:-1])) + [depot_id]
            new_route = reversed_route1[:-1] + route2['route'][1:]
            new_load = route1['load'] + route2['load']
            new_end0 = route1['route'][-2] 
            new_end1 = route2['end1']
            merged = True
        elif route1['end0'] == c1 and route2['end1'] == c2 and route1['load'] + route2['load'] <= vehicle_capacity:
            new_route = route2['route'][:-1] + route1['route'][1:]
            new_load = route1['load'] + route2['load']
            new_end0 = route2['end0']
            new_end1 = route1['end1']
            merged = True
        
        if merged:
            route_details[idx1] = {
                'route': new_route,
                'load': new_load,
                'end0': new_end0,
                'end1': new_end1
            }
            for c in route2['route'][1:-1]:
                in_route[c] = idx1
            route_details[idx2] = None
            in_route.pop(c2, None)
            in_route.pop(c1, None)
            for node in new_route[1:-1]:
                in_route[node] = idx1
    
    final_routes = []
    for rd in route_details:
        if rd is not None:
            final_routes.append(rd['route'])
    return final_routes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers_list = instance['customers']
    distance_matrix = instance['distance_matrix']

    customers_by_id = {c['id']: c for c in customers_list}
    customer_ids = sorted(customers_by_id.keys())
    customer_ids_set = set(customer_ids)

    model = Model('inventory_routing')
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(1, args.time_limit - (time.time() - start_time)))

    T = num_periods
    C = num_customers

    depot_inv = [None] * (T+1)
    customer_inv = {}
    delivery = {}

    depot_inv[0] = depot['initial_inventory']

    for cid in customer_ids:
        customer_inv[cid, 0] = customers_by_id[cid]['initial_inventory']

    for t in range(1, T+1):
        depot_inv[t] = model.addVar(lb=0, ub=depot['max_inventory'], name=f'depot_inv_{t}')
        for cid in customer_ids:
            customer_inv[cid, t] = model.addVar(lb=customers_by_id[cid]['min_inventory'], 
                                               ub=customers_by_id[cid]['max_inventory'], 
                                               name=f'cust{cid}_inv_{t}')
            delivery[t, cid] = model.addVar(lb=0, name=f'delivery_{t}_{cid}')

    model.update()

    total_inventory_cost = 0
    for t in range(1, T+1):
        total_inventory_cost += depot['holding_cost'] * depot_inv[t]
        for cid in customer_ids:
            total_inventory_cost += customers_by_id[cid]['holding_cost'] * customer_inv[cid, t]

    model.setObjective(total_inventory_cost, GRB.MINIMIZE)

    for t in range(1, T+1):
        model.addConstr(
            depot_inv[t] == depot_inv[t-1] + depot['production_per_period'] - 
            quicksum(delivery[t, cid] for cid in customer_ids),
            name=f'depot_balance_{t}'
        )
        for cid in customer_ids:
            c = customers_by_id[cid]
            model.addConstr(
                customer_inv[cid, t] == customer_inv[cid, t-1] + delivery[t, cid] - c['demand_per_period'],
                name=f'cust_balance_{t}_{cid}'
            )
            model.addConstr(
                delivery[t, cid] <= vehicle_capacity,
                name=f'vehicle_capacity_{t}_{cid}'
            )
            if t == 1:
                model.addConstr(
                    delivery[t, cid] <= c['max_inventory'] - customer_inv[cid, 0],
                    name=f'max_inv_{t}_{cid}'
                )
            else:
                model.addConstr(
                    delivery[t, cid] <= c['max_inventory'] - customer_inv[cid, t-1],
                    name=f'max_inv_{t}_{cid}'
                )
        model.addConstr(
            quicksum(delivery[t, cid] for cid in customer_ids) <= num_vehicles * vehicle_capacity,
            name=f'total_delivery_capacity_{t}'
        )

    model.optimize()

    delivery_quantities = {}
    customer_inv_values = {}
    depot_inv_values = [depot['initial_inventory']]
    total_inventory_cost_value = 0

    if model.SolCount > 0:
        for t in range(1, T+1):
            depot_inv_values.append(depot_inv[t].X)
            for cid in customer_ids:
                delivery_quantities[t, cid] = delivery[t, cid].X
                customer_inv_values[cid, t] = customer_inv[cid, t].X
        total_inventory_cost_value = model.objVal
    else:
        for t in range(1, T+1):
            total_delivery = 0
            for cid in customer_ids:
                c = customers_by_id[cid]
                delivery_quantities[t, cid] = c['demand_per_period']
                total_delivery += c['demand_per_period']
            
            depot_inv_t = depot_inv_values[t-1] + depot['production_per_period'] - total_delivery
            if depot_inv_t < 0:
                available = depot_inv_values[t-1] + depot['production_per_period']
                if available < 0:
                    available = 0
                for cid in customer_ids:
                    delivery_quantities[t, cid] = (customers_by_id[cid]['demand_per_period'] / total_delivery) * available
                depot_inv_t = 0
            elif depot_inv_t > depot['max_inventory']:
                depot_inv_t = depot['max_inventory']
            depot_inv_values.append(depot_inv_t)
            
            for cid in customer_ids:
                c = customers_by_id[cid]
                if t == 1:
                    prev_inv = c['initial_inventory']
                else:
                    prev_inv = customer_inv_values[cid, t-1]
                new_inv = prev_inv + delivery_quantities[t, cid] - c['demand_per_period']
                if new_inv < c['min_inventory']:
                    new_inv = c['min_inventory']
                if new_inv > c['max_inventory']:
                    new_inv = c['max_inventory']
                customer_inv_values[cid, t] = new_inv
        
        for t in range(1, T+1):
            total_inventory_cost_value += depot['holding_cost'] * depot_inv_values[t]
            for cid in customer_ids:
                total_inventory_cost_value += customers_by_id[cid]['holding_cost'] * customer_inv_values[cid, t]

    total_travel_cost = 0
    solution_routes = {}
    solution_deliveries = {}
    solution_inventories = {}

    for t in range(1, T+1):
        period_deliveries = {}
        customers_this_period = []
        demands_this_period = {}
        for cid in customer_ids:
            q = delivery_quantities.get((t, cid), 0)
            if q > 1e-5:
                period_deliveries[str(cid)] = q
                customers_this_period.append(cid)
                demands_this_period[cid] = q
        solution_deliveries[t] = period_deliveries

        routes_t = []
        if customers_this_period:
            routes_t = savings_algorithm(0, customers_this_period, demands_this_period, vehicle_capacity, distance_matrix)
            for route in routes_t:
                cost = 0
                for k in range(len(route)-1):
                    i = route[k]
                    j = route[k+1]
                    cost += distance_matrix[i][j]
                total_travel_cost += cost
        solution_routes[t] = routes_t

        inventories_t = {
            'depot': depot_inv_values[t]
        }
        for cid in customer_ids:
            inv_val = customer_inv_values.get((cid, t), 0)
            inventories_t[str(cid)] = inv_val
        solution_inventories[t] = inventories_t

    total_cost = total_inventory_cost_value + total_travel_cost

    solution_details = {}
    for t in range(1, T+1):
        solution_details[str(t)] = {
            'routes': solution_routes[t],
            'deliveries': solution_deliveries[t],
            'inventories': solution_inventories[t]
        }

    output = {
        "objective_value": total_cost,
        "solution_details": {
            "periods": solution_details
        }
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()