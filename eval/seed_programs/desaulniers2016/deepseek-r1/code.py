import argparse
import json
import time
import math
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data

def build_stage1_model(data):
    num_customers = data['num_customers']
    num_periods = data['num_periods']
    num_vehicles = data['num_vehicles']
    vehicle_capacity = data['vehicle_capacity']
    num_nodes = data['num_nodes']
    holding_costs = data['holding_costs']
    customer_demands = data['customer_demands']
    supplier_production = data['supplier_production']
    inventory_capacities = data['inventory_capacities']
    initial_inventories = data['initial_inventories']
    distance_matrix = data['distance_matrix']

    model = gp.Model("Stage1_Inventory")
    model.setParam("Threads", 1)

    I = {}
    x = {}
    y = {}

    for i in range(num_nodes):
        for t in range(num_periods):
            I[(i, t)] = model.addVar(lb=0, ub=inventory_capacities[i], name=f"I_{i}_{t}")

    for i in range(1, num_nodes):
        for t in range(num_periods):
            x[(i, t)] = model.addVar(lb=0, ub=min(vehicle_capacity, inventory_capacities[i]), name=f"x_{i}_{t}")
            y[(i, t)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")

    for i in range(1, num_nodes):
        for t in range(num_periods):
            model.addConstr(x[(i, t)] <= min(vehicle_capacity, inventory_capacities[i]) * y[(i, t)], name=f"delivery_indicator_{i}_{t}")

    for t in range(num_periods):
        total_delivery = gp.quicksum(x[(i, t)] for i in range(1, num_nodes))
        model.addConstr(total_delivery <= vehicle_capacity * num_vehicles, name=f"fleet_capacity_{t}")

    for i in range(num_nodes):
        if i == 0:
            model.addConstr(I[(0, 0)] == initial_inventories[0] + supplier_production[0] - gp.quicksum(x[(i, 0)] for i in range(1, num_nodes)), name=f"depot_balance_0")
        else:
            model.addConstr(I[(i, 0)] == initial_inventories[i] + x[(i, 0)] - customer_demands[i-1][0], name=f"cust_balance_{i}_0")

    for t in range(1, num_periods):
        model.addConstr(I[(0, t)] == I[(0, t-1)] + supplier_production[t] - gp.quicksum(x[(i, t)] for i in range(1, num_nodes)), name=f"depot_balance_{t}")
        for i in range(1, num_nodes):
            model.addConstr(I[(i, t)] == I[(i, t-1)] + x[(i, t)] - customer_demands[i-1][t], name=f"cust_balance_{i}_{t}")

    for i in range(1, num_nodes):
        model.addConstr(initial_inventories[i] + x[(i, 0)] <= inventory_capacities[i], name=f"delivery_capacity_{i}_0")
        for t in range(1, num_periods):
            model.addConstr(I[(i, t-1)] + x[(i, t)] <= inventory_capacities[i], name=f"delivery_capacity_{i}_{t}")

    holding_cost = 0
    for i in range(num_nodes):
        for t in range(num_periods):
            holding_cost += holding_costs[i] * I[(i, t)]

    routing_approx = 0
    for i in range(1, num_nodes):
        for t in range(num_periods):
            routing_approx += 2 * distance_matrix[0][i] * y[(i, t)]

    model.setObjective(holding_cost + routing_approx, GRB.MINIMIZE)
    model.update()

    return model, I, x, y

def savings_algorithm(customers, dist_matrix, vehicle_cap, num_vehicles):
    if not customers:
        return []

    routes = [[0, i, 0] for i, d in customers]
    demands = {i: d for i, d in customers}
    route_demands = [demands[i] for i, d in customers]
    route_info = []
    for idx, route in enumerate(routes):
        first = route[1]
        last = route[1]
        route_info.append({
            'nodes': route,
            'demand': route_demands[idx],
            'first': first,
            'last': last
        })

    savings_list = []
    n = len(route_info)
    for i in range(n):
        for j in range(i+1, n):
            last_i = route_info[i]['last']
            first_j = route_info[j]['first']
            last_j = route_info[j]['last']
            saving1 = dist_matrix[last_i][0] + dist_matrix[0][first_j] - dist_matrix[last_i][first_j]
            saving2 = dist_matrix[last_i][0] + dist_matrix[0][last_j] - dist_matrix[last_i][last_j]
            saving = max(saving1, saving2)
            reverse_j = 0 if saving1 >= saving2 else 1
            savings_list.append((saving, i, j, reverse_j))

    savings_list.sort(key=lambda x: x[0], reverse=True)
    active_routes = list(range(n))
    merged_routes = [False] * n

    while savings_list:
        s, i_idx, j_idx, rev = savings_list.pop(0)
        if merged_routes[i_idx] or merged_routes[j_idx]:
            continue
        if route_info[i_idx]['demand'] + route_info[j_idx]['demand'] > vehicle_cap:
            continue
        if rev == 0:
            new_route_nodes = route_info[i_idx]['nodes'][:-1] + route_info[j_idx]['nodes'][1:]
        else:
            rev_j = [0] + list(reversed(route_info[j_idx]['nodes'][1:-1])) + [0]
            new_route_nodes = route_info[i_idx]['nodes'][:-1] + rev_j[1:]
        new_demand = route_info[i_idx]['demand'] + route_info[j_idx]['demand']
        first_cust = new_route_nodes[1]
        last_cust = new_route_nodes[-2]
        new_route_info = {
            'nodes': new_route_nodes,
            'demand': new_demand,
            'first': first_cust,
            'last': last_cust
        }
        route_info.append(new_route_info)
        new_idx = len(route_info) - 1
        merged_routes[i_idx] = True
        merged_routes[j_idx] = True
        merged_routes.append(False)
        active_routes = [idx for idx in active_routes if not merged_routes[idx]]
        active_routes.append(new_idx)
        savings_list = [item for item in savings_list if not (merged_routes[item[1]] or merged_routes[item[2]])]
        for k in active_routes:
            if k == new_idx:
                continue
            last_i = route_info[new_idx]['last']
            first_k = route_info[k]['first']
            last_k = route_info[k]['last']
            saving1 = dist_matrix[last_i][0] + dist_matrix[0][first_k] - dist_matrix[last_i][first_k]
            saving2 = dist_matrix[last_i][0] + dist_matrix[0][last_k] - dist_matrix[last_i][last_k]
            saving = max(saving1, saving2)
            rev_flag = 0 if saving1 >= saving2 else 1
            savings_list.append((saving, new_idx, k, rev_flag))
        savings_list.sort(key=lambda x: x[0], reverse=True)

    final_routes = []
    for idx in active_routes:
        if not merged_routes[idx]:
            final_routes.append(route_info[idx]['nodes'])
    return final_routes

def ffd_bin_packing(customers, vehicle_cap, num_vehicles):
    items = sorted([(i, d) for i, d in customers], key=lambda x: x[1], reverse=True)
    bins = [[] for _ in range(num_vehicles)]
    bin_sums = [0] * num_vehicles
    for (node, demand) in items:
        assigned = False
        for b in range(num_vehicles):
            if bin_sums[b] + demand <= vehicle_cap:
                bins[b].append(node)
                bin_sums[b] += demand
                assigned = True
                break
        if not assigned:
            bins[0].append(node)
            bin_sums[0] += demand
    routes = []
    for bin_customers in bins:
        if bin_customers:
            routes.append([0] + bin_customers + [0])
    return routes

def compute_route_cost(route, dist_matrix):
    cost = 0
    for i in range(len(route)-1):
        cost += dist_matrix[route[i]][route[i+1]]
    return cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = read_instance(args.instance_path)
    time_left = args.time_limit - (time.time() - start_time)
    if time_left <= 1:
        raise Exception("Time limit too tight.")

    model, I, x, y = build_stage1_model(data)
    model.setParam('TimeLimit', max(1, time_left * 0.8))
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        raise Exception("Stage1 model is infeasible.")
    if model.status == GRB.TIME_LIMIT and model.SolCount == 0:
        raise Exception("Stage1 model timed out without a solution.")

    num_periods = data['num_periods']
    num_nodes = data['num_nodes']
    vehicle_capacity = data['vehicle_capacity']
    num_vehicles = data['num_vehicles']
    distance_matrix = data['distance_matrix']
    customer_demands = data['customer_demands']

    deliveries = {}
    for i in range(1, num_nodes):
        for t in range(num_periods):
            if y.get((i, t)) is not None and y[(i, t)].X > 0.5:
                if t not in deliveries:
                    deliveries[t] = []
                deliveries[t].append((i, x[(i, t)].X))

    inventories = {}
    for i in range(num_nodes):
        for t in range(num_periods):
            if i not in inventories:
                inventories[i] = {}
            inventories[i][t] = I[(i, t)].X

    routes_output = {}
    total_travel_cost = 0

    for t in range(num_periods):
        routes_output[str(t)] = {}
        if t not in deliveries:
            continue
        customers_t = deliveries[t]
        routes_savings = savings_algorithm(customers_t, distance_matrix, vehicle_capacity, num_vehicles)
        if len(routes_savings) > num_vehicles:
            routes_t = ffd_bin_packing(customers_t, vehicle_capacity, num_vehicles)
        else:
            routes_t = routes_savings
        for vidx, route in enumerate(routes_t[:num_vehicles]):
            if len(route) < 2:
                continue
            route_key = str(vidx)
            routes_output[str(t)][route_key] = {
                'route': route,
                'deliveries': {}
            }
            for node in route[1:-1]:
                for (i, d) in customers_t:
                    if i == node:
                        routes_output[str(t)][route_key]['deliveries'][str(node)] = d
            total_travel_cost += compute_route_cost(route, distance_matrix)

    total_holding_cost = 0
    for i in range(num_nodes):
        for t in range(num_periods):
            total_holding_cost += data['holding_costs'][i] * inventories[i][t]

    objective_value = total_travel_cost + total_holding_cost
    if logger:
        logger.log(objective_value)

    solution = {
        "objective_value": objective_value,
        "routes": {},
        "inventories": {}
    }

    for period, vehicles in routes_output.items():
        solution["routes"][period] = {}
        for vehicle, route_data in vehicles.items():
            solution["routes"][period][vehicle] = {
                "route": route_data['route'],
                "deliveries": {str(cust): qty for cust, qty in route_data['deliveries'].items()}
            }

    for i in range(num_nodes):
        solution["inventories"][str(i)] = {str(t): float(inventories[i][t]) for t in range(num_periods)}

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()