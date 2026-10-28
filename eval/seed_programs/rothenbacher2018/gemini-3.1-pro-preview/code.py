import argparse
import json
import time
import math
import random
import collections

class Route:
    def __init__(self, vc, stops, evaluate_fn):
        self.vc = vc
        self.stops = stops
        self.cost, self.is_feasible = evaluate_fn(vc, stops)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        data = json.load(f)

    params = data['parameters']
    dist_matrix = data['distance_matrix']
    time_matrix = data['travel_time_matrix']
    depot_id = data['depot']['id']

    customers = {}
    truck_customers = set()
    trailer_customers = set()

    for c in data.get('truck_customers', []):
        customers[c['id']] = c
        truck_customers.add(c['id'])

    for c in data.get('trailer_customers', []):
        customers[c['id']] = c
        trailer_customers.add(c['id'])

    transshipment_ids = [t['id'] for t in data.get('transshipment_locations', [])]

    P_list = []
    for c in data.get('trailer_customers', []):
        P_list.append(c['id'])
    for t in data.get('transshipment_locations', []):
        P_list.append(t['id'])

    all_locs = [depot_id] + list(customers.keys()) + transshipment_ids
    n_orig = max(all_locs) + 1 if all_locs else 0

    truck_avail = {t['id']: t['num_available'] for t in data['fleet']['truck_types']}
    trailer_avail = {t['id']: t['num_available'] for t in data['fleet']['trailer_types']}
    class_avail = {c['class_id']: c['num_available'] for c in data['fleet']['vehicle_classes']}
    vehicle_classes = data['fleet']['vehicle_classes']

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    def check_time():
        if time.time() - start_time > args.time_limit - 1.5:
            raise TimeoutError

    def evaluate_route(vc, stops):
        time_val = 0.0
        cost = vc['truck_fixed_cost']
        truck_load = 0
        trailer_load = 0
        trailer_attached = vc['trailer_capacity'] > 0
        curr_loc = depot_id

        for stop_type, loc_id in stops:
            dist = dist_matrix[curr_loc][loc_id]
            t_time = time_matrix[curr_loc][loc_id]
            cost += dist * (vc['truck_cost_factor'] + (vc['trailer_towing_cost_multiplier'] if trailer_attached else 0))
            time_val += t_time
            curr_loc = loc_id

            if stop_type in ('service_attached', 'service_unattached'):
                time_val += customers[loc_id]['service_time']
                load = customers[loc_id]['supply']
                if trailer_attached:
                    space = vc['trailer_capacity'] - trailer_load
                    if load <= space:
                        trailer_load += load
                    else:
                        trailer_load = vc['trailer_capacity']
                        truck_load += (load - space)
                else:
                    truck_load += load
                if truck_load > vc['truck_capacity'] or trailer_load > vc['trailer_capacity']:
                    return float('inf'), False

            elif stop_type == 'decouple':
                time_val += params['decoupling_time']
                cost += params['decoupling_cost']
                trailer_attached = False

            elif stop_type == 'couple':
                time_val += params['coupling_time']
                cost += params['coupling_cost']
                trailer_attached = True
                space = vc['trailer_capacity'] - trailer_load
                transferred = min(truck_load, space)
                if transferred > 0:
                    trailer_load += transferred
                    truck_load -= transferred
                    time_val += params.get('transfer_time', 0.0)
                    cost += params.get('transfer_cost', 0.0)

        dist = dist_matrix[curr_loc][depot_id]
        t_time = time_matrix[curr_loc][depot_id]
        cost += dist * (vc['truck_cost_factor'] + (vc['trailer_towing_cost_multiplier'] if trailer_attached else 0))
        time_val += t_time

        if time_val > params['max_route_duration'] or time_val > params['planning_horizon'][1] - params['planning_horizon'][0]:
            return float('inf'), False
        if trailer_attached != (vc['trailer_capacity'] > 0):
            return float('inf'), False

        return cost, True

    def get_available_classes(routes):
        t_used = collections.defaultdict(int)
        tr_used = collections.defaultdict(int)
        c_used = collections.defaultdict(int)
        for r in routes:
            t_used[r.vc['truck_type_id']] += 1
            if r.vc['trailer_type_id'] != 0:
                tr_used[r.vc['trailer_type_id']] += 1
            c_used[r.vc['class_id']] += 1

        avail = []
        for vc in vehicle_classes:
            if c_used[vc['class_id']] < class_avail[vc['class_id']]:
                if t_used[vc['truck_type_id']] < truck_avail[vc['truck_type_id']]:
                    if vc['trailer_type_id'] == 0 or tr_used[vc['trailer_type_id']] < trailer_avail[vc['trailer_type_id']]:
                        avail.append(vc)
        return avail

    def get_all_insertions(c, routes, avail_vehicles):
        insertions = []
        for r_idx, r in enumerate(routes):
            vc = r.vc
            stops = r.stops
            is_attached_list = [vc['trailer_capacity'] > 0]
            for s in stops:
                if s[0] == 'decouple': is_attached_list.append(False)
                elif s[0] == 'couple': is_attached_list.append(True)
                else: is_attached_list.append(is_attached_list[-1])

            for i in range(len(stops) + 1):
                is_attached = is_attached_list[i]
                if c in truck_customers:
                    if not is_attached:
                        possible = stops[:i] + [('service_unattached', c)] + stops[i:]
                        cost, feasible = evaluate_route(vc, possible)
                        if feasible: insertions.append((cost - r.cost, r_idx, possible, None))
                    else:
                        if vc['trailer_capacity'] > 0:
                            for p in P_list:
                                possible = stops[:i] + [('decouple', p), ('service_unattached', c), ('couple', p)] + stops[i:]
                                cost, feasible = evaluate_route(vc, possible)
                                if feasible: insertions.append((cost - r.cost, r_idx, possible, None))
                else:
                    if is_attached:
                        possible = stops[:i] + [('service_attached', c)] + stops[i:]
                        cost, feasible = evaluate_route(vc, possible)
                        if feasible: insertions.append((cost - r.cost, r_idx, possible, None))
                        if vc['trailer_capacity'] > 0:
                            for p in P_list:
                                possible = stops[:i] + [('decouple', p), ('service_unattached', c), ('couple', p)] + stops[i:]
                                cost, feasible = evaluate_route(vc, possible)
                                if feasible: insertions.append((cost - r.cost, r_idx, possible, None))
                    else:
                        possible = stops[:i] + [('service_unattached', c)] + stops[i:]
                        cost, feasible = evaluate_route(vc, possible)
                        if feasible: insertions.append((cost - r.cost, r_idx, possible, None))

        for vc in avail_vehicles:
            if c in truck_customers:
                if vc['trailer_capacity'] == 0:
                    possible = [('service_unattached', c)]
                    cost, feasible = evaluate_route(vc, possible)
                    if feasible: insertions.append((cost, -1, possible, vc))
                else:
                    for p in P_list:
                        possible = [('decouple', p), ('service_unattached', c), ('couple', p)]
                        cost, feasible = evaluate_route(vc, possible)
                        if feasible: insertions.append((cost, -1, possible, vc))
            else:
                if vc['trailer_capacity'] == 0:
                    possible = [('service_unattached', c)]
                    cost, feasible = evaluate_route(vc, possible)
                    if feasible: insertions.append((cost, -1, possible, vc))
                else:
                    possible = [('service_attached', c)]
                    cost, feasible = evaluate_route(vc, possible)
                    if feasible: insertions.append((cost, -1, possible, vc))
                    for p in P_list:
                        possible = [('decouple', p), ('service_unattached', c), ('couple', p)]
                        cost, feasible = evaluate_route(vc, possible)
                        if feasible: insertions.append((cost, -1, possible, vc))
        return insertions

    def greedy_repair(routes, removed_customers):
        unrouted = list(removed_customers)
        unrouted.sort(key=lambda x: customers[x]['supply'] + random.uniform(0, 50), reverse=True)
        current_routes = [Route(r.vc, list(r.stops), evaluate_route) for r in routes]

        for c in unrouted:
            check_time()
            avail = get_available_classes(current_routes)
            insertions = get_all_insertions(c, current_routes, avail)
            if not insertions:
                return None
            insertions.sort(key=lambda x: x[0])
            best = insertions[0]
            delta, r_idx, new_stops, vc = best
            if r_idx == -1:
                current_routes.append(Route(vc, new_stops, evaluate_route))
            else:
                current_routes[r_idx] = Route(current_routes[r_idx].vc, new_stops, evaluate_route)
        return current_routes

    def repair(routes, removed_customers):
        unrouted = list(removed_customers)
        current_routes = [Route(r.vc, list(r.stops), evaluate_route) for r in routes]

        while unrouted:
            check_time()
            best_c = None
            best_insertion = None
            max_regret = -float('inf')
            avail = get_available_classes(current_routes)

            for c in unrouted:
                insertions = get_all_insertions(c, current_routes, avail)
                if not insertions:
                    return None
                insertions.sort(key=lambda x: x[0])
                c1 = insertions[0][0]
                c2 = insertions[1][0] if len(insertions) > 1 else c1 + 10000.0
                regret = c2 - c1 + random.random() * 0.1
                if regret > max_regret:
                    max_regret = regret
                    best_c = c
                    best_insertion = insertions[0]

            c = best_c
            unrouted.remove(c)
            delta, r_idx, new_stops, vc = best_insertion
            if r_idx == -1:
                current_routes.append(Route(vc, new_stops, evaluate_route))
            else:
                current_routes[r_idx] = Route(current_routes[r_idx].vc, new_stops, evaluate_route)
        return current_routes

    def destroy(routes, k):
        new_routes = [Route(r.vc, list(r.stops), evaluate_route) for r in routes]
        all_custs = []
        for r_idx, r in enumerate(new_routes):
            for i, stop in enumerate(r.stops):
                if stop[0] in ('service_attached', 'service_unattached'):
                    all_custs.append((r_idx, i, stop[1]))
        
        removed = set()
        if not all_custs: return new_routes, removed
        
        k = min(k, len(all_custs))
        to_remove = random.sample(all_custs, k)
        to_remove.sort(key=lambda x: x[1], reverse=True)
        
        for r_idx, i, cust_id in to_remove:
            del new_routes[r_idx].stops[i]
            removed.add(cust_id)
            
        cleaned_routes = []
        for r in new_routes:
            stops = r.stops
            changed = True
            while changed:
                changed = False
                for i in range(len(stops)-1):
                    if stops[i][0] == 'decouple' and stops[i+1][0] == 'couple' and stops[i][1] == stops[i+1][1]:
                        del stops[i:i+2]
                        changed = True
                        break
            if any(s[0] in ('service_attached', 'service_unattached') for s in stops):
                r.cost, _ = evaluate_route(r.vc, stops)
                cleaned_routes.append(r)
        return cleaned_routes, removed

    def build_output(routes):
        out_routes = []
        for r in routes:
            vc = r.vc
            seq = [depot_id]
            seq_phys = [depot_id]
            arr_times = {}
            curr_time = params['planning_horizon'][0]
            curr_loc = depot_id
            trailer_attached = vc['trailer_capacity'] > 0
            truck_load = 0
            trailer_load = 0
            decouple_locs = []
            couple_locs = []
            custs_served = []
            arr_times[str(depot_id)] = curr_time

            for stop_type, loc_id in r.stops:
                dist = dist_matrix[curr_loc][loc_id]
                t_time = time_matrix[curr_loc][loc_id]
                curr_time += t_time
                curr_loc = loc_id

                if stop_type in ('service_attached', 'service_unattached'):
                    seq.append(loc_id)
                    seq_phys.append(loc_id)
                    arr_times[str(loc_id)] = curr_time
                    custs_served.append(loc_id)
                    curr_time += customers[loc_id]['service_time']
                    load = customers[loc_id]['supply']
                    if trailer_attached:
                        space = vc['trailer_capacity'] - trailer_load
                        if load <= space: trailer_load += load
                        else:
                            trailer_load = vc['trailer_capacity']
                            truck_load += (load - space)
                    else:
                        truck_load += load

                elif stop_type == 'decouple':
                    p_idx = P_list.index(loc_id)
                    exp_id = n_orig + p_idx
                    seq.append(exp_id)
                    seq_phys.append(loc_id)
                    arr_times[str(exp_id)] = curr_time
                    decouple_locs.append(loc_id)
                    curr_time += params['decoupling_time']
                    trailer_attached = False

                elif stop_type == 'couple':
                    p_idx = P_list.index(loc_id)
                    exp_id = n_orig + len(P_list) + p_idx
                    seq.append(exp_id)
                    seq_phys.append(loc_id)
                    arr_times[str(exp_id)] = curr_time
                    couple_locs.append(loc_id)
                    curr_time += params['coupling_time']
                    trailer_attached = True
                    space = vc['trailer_capacity'] - trailer_load
                    transferred = min(truck_load, space)
                    if transferred > 0:
                        trailer_load += transferred
                        truck_load -= transferred
                        curr_time += params.get('transfer_time', 0.0)

            t_time = time_matrix[curr_loc][depot_id]
            curr_time += t_time
            seq.append(depot_id)
            seq_phys.append(depot_id)
            arr_times[str(depot_id)] = curr_time

            out_routes.append({
                "vehicle_class": vc['class_id'],
                "vehicle_index": 0,
                "customers_served": custs_served,
                "route_sequence": seq,
                "route_sequence_physical": seq_phys,
                "decouple_locations": decouple_locs,
                "couple_locations": couple_locs,
                "arrival_times": arr_times
            })

        class_counts = collections.defaultdict(int)
        for r in out_routes:
            r["vehicle_index"] = class_counts[r["vehicle_class"]]
            class_counts[r["vehicle_class"]] += 1
            
        cost = sum(r.cost for r in routes)
        return {
            "objective_value": cost,
            "routes": out_routes
        }

    random.seed(42)
    all_cust_ids = list(customers.keys())
    
    best_routes = None
    try:
        for _ in range(5):
            best_routes = greedy_repair([], all_cust_ids)
            if best_routes is not None:
                break
    except TimeoutError:
        pass

    if best_routes is None:
        return

    best_cost = sum(r.cost for r in best_routes)
    current_routes = [Route(r.vc, list(r.stops), evaluate_route) for r in best_routes]
    current_cost = best_cost

    if logger:
        logger.log_solution(best_cost, build_output(best_routes))

    initial_temp = 100.0
    iteration = 0

    try:
        while True:
            check_time()
            iteration += 1
            k = random.randint(min(4, len(all_cust_ids)), min(15, len(all_cust_ids)))
            destroyed, removed = destroy(current_routes, k)
            
            if random.random() < 0.5:
                repaired = greedy_repair(destroyed, removed)
            else:
                repaired = repair(destroyed, removed)
                
            if repaired is not None:
                cost = sum(r.cost for r in repaired)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = repaired
                    if logger:
                        logger.log_solution(best_cost, build_output(best_routes))
                    current_routes = repaired
                    current_cost = cost
                else:
                    temp = initial_temp * (0.99 ** iteration)
                    if random.random() < math.exp((current_cost - cost) / max(temp, 1e-5)):
                        current_routes = repaired
                        current_cost = cost
    except TimeoutError:
        pass

    final_output = build_output(best_routes)
    with open(args.solution_path, "w") as f:
        json.dump(final_output, f, indent=2)

if __name__ == "__main__":
    main()