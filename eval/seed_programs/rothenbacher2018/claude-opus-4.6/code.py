import json
import argparse
import time
import math
import copy
import random
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    random.seed(42)
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    params = data['parameters']
    fleet = data['fleet']
    depot = data['depot']
    truck_customers = data.get('truck_customers', [])
    trailer_customers = data.get('trailer_customers', [])
    transshipment_locations = data.get('transshipment_locations', [])
    dist_matrix = data['distance_matrix']
    time_matrix = data['travel_time_matrix']
    
    max_route_duration = params['max_route_duration']
    service_time_param = params['service_time']
    coupling_time = params['coupling_time']
    decoupling_time = params['decoupling_time']
    transfer_time = params['transfer_time']
    coupling_cost = params['coupling_cost']
    decoupling_cost = params['decoupling_cost']
    transfer_cost = params['transfer_cost']
    planning_horizon = params['planning_horizon']
    horizon_start = planning_horizon[0]
    horizon_end = planning_horizon[1]
    
    depot_id = depot['id']
    
    tc_ids = set(c['id'] for c in truck_customers)
    trc_ids = set(c['id'] for c in trailer_customers)
    all_customer_ids = tc_ids | trc_ids
    
    customer_supply = {}
    customer_service_time = {}
    customer_coords = {}
    for c in truck_customers:
        customer_supply[c['id']] = c['supply']
        customer_service_time[c['id']] = c.get('service_time', service_time_param)
        customer_coords[c['id']] = (c['x'], c['y'])
    for c in trailer_customers:
        customer_supply[c['id']] = c['supply']
        customer_service_time[c['id']] = c.get('service_time', service_time_param)
        customer_coords[c['id']] = (c['x'], c['y'])
    
    trans_ids = set(t['id'] for t in transshipment_locations)
    # Parking locations: trailer customers + transshipment locations
    parking_ids = trc_ids | trans_ids
    
    vehicle_classes = fleet['vehicle_classes']
    truck_types = {t['id']: t for t in fleet['truck_types']}
    trailer_types = {t['id']: t for t in fleet.get('trailer_types', [])}
    
    # Time windows for customers - check if they exist in data
    # The problem says each customer has a time window but the schema for customers
    # doesn't explicitly list them. We'll assume planning_horizon as default.
    customer_tw = {}
    for c in truck_customers:
        tw_start = c.get('earliest', horizon_start)
        tw_end = c.get('latest', horizon_end)
        customer_tw[c['id']] = (tw_start, tw_end)
    for c in trailer_customers:
        tw_start = c.get('earliest', horizon_start)
        tw_end = c.get('latest', horizon_end)
        customer_tw[c['id']] = (tw_start, tw_end)
    
    def dist(i, j):
        return dist_matrix[i][j]
    
    def ttime(i, j):
        return time_matrix[i][j]
    
    n_orig = len(dist_matrix)
    parking_list = sorted(parking_ids)
    
    # A route is represented as a sequence of actions
    # For simplicity, we'll use a representation where:
    # - A solo truck route is just a sequence of customers
    # - A complete vehicle route has segments: 
    #   with-trailer segments and without-trailer subtours from parking locations
    
    # Route representation:
    # {
    #   'vehicle_class_id': int,
    #   'segments': list of segments
    # }
    # A segment is either:
    #   {'type': 'with_trailer', 'customers': [list of trailer customer ids]}
    #   {'type': 'subtour', 'park_loc': int, 'customers': [list of customer ids]}
    
    # For solo trucks (no trailer), the route is just a list of customers
    # For complete vehicles, we have a more complex structure
    
    # Let me simplify: represent a route as an ordered list of "stops"
    # Each stop has: location_id, action (serve, decouple, couple, transfer)
    
    # Actually let me use an even simpler representation for the heuristic:
    # A route is a list of customer IDs in visit order, plus metadata about
    # where decoupling/coupling happens.
    
    # For a solo truck: just visit customers in order
    # For a complete vehicle: identify parking locations for subtours
    
    # SIMPLIFIED APPROACH:
    # Phase 1: Assign customers to routes using nearest-neighbor heuristic
    # Phase 2: Try to improve with local search
    
    class Route:
        def __init__(self, vc_id):
            self.vc_id = vc_id
            vc = None
            for v in vehicle_classes:
                if v['class_id'] == vc_id:
                    vc = v
                    break
            self.vc = vc
            self.has_trailer = vc['trailer_type_id'] != 0
            self.truck_cap = vc['truck_capacity']
            self.trailer_cap = vc['trailer_capacity']
            self.total_cap = vc['total_capacity']
            self.truck_cost_factor = vc['truck_cost_factor']
            self.trailer_towing_mult = vc['trailer_towing_cost_multiplier']
            self.fixed_cost = vc['truck_fixed_cost']
            
            # Simple representation: ordered list of visits
            # For complete vehicles, we track parking/subtour structure
            # Format: list of (location_id, 'serve'|'park'|'unpark')
            # Actually, let's keep it really simple:
            # For solo: just a list of customer IDs to visit in order
            # For complete vehicle with trailer:
            #   main_sequence: list of items, each is either:
            #     ('customer', cust_id) - visited with trailer
            #     ('subtour', park_loc, [cust_ids]) - decouple at park_loc, visit custs, return
            self.customers = []  # for solo trucks
            self.main_sequence = []  # for complete vehicles
        
        def get_all_customers(self):
            if not self.has_trailer:
                return list(self.customers)
            else:
                result = []
                for item in self.main_sequence:
                    if item[0] == 'customer':
                        result.append(item[1])
                    elif item[0] == 'subtour':
                        # If park_loc is a trailer customer, it's served when visited with trailer
                        # Actually, parking at a location doesn't mean serving it
                        result.extend(item[2])
                return result
    
    # Let's try a simpler approach: treat everything as a giant TSP-like problem
    # and use insertion heuristics
    
    # For each vehicle class, determine which customers it can serve
    # Solo trucks: can serve both truck and trailer customers
    # Complete vehicles: can serve trailer customers with trailer attached,
    #   and truck customers via subtours (decouple trailer, visit truck customers, return)
    
    # Strategy:
    # 1. Sort vehicle classes by cost efficiency
    # 2. Use cheapest insertion to build routes
    # 3. For truck customers, if using complete vehicle, need subtour from parking location
    
    # Let me implement a straightforward greedy nearest-neighbor construction
    
    def evaluate_solo_route(customers, vc):
        """Evaluate a solo truck route visiting customers in given order."""
        if not customers:
            return None
        
        truck_cap = vc['truck_capacity']
        cost_factor = vc['truck_cost_factor']
        fixed = vc['truck_fixed_cost']
        
        total_load = 0
        total_dist_cost = 0.0
        current_time = horizon_start
        current_loc = depot_id
        
        arrival_times = {depot_id: horizon_start}
        
        for cid in customers:
            travel_t = ttime(current_loc, cid)
            travel_d = dist(current_loc, cid)
            
            arrive = current_time + travel_t
            # Check time window
            tw_start, tw_end = customer_tw.get(cid, (horizon_start, horizon_end))
            if arrive < tw_start:
                arrive = tw_start
            if arrive > tw_end:
                return None  # infeasible
            
            total_load += customer_supply[cid]
            if total_load > truck_cap:
                return None  # capacity exceeded
            
            total_dist_cost += cost_factor * travel_d
            stime = customer_service_time.get(cid, service_time_param)
            current_time = arrive + stime
            current_loc = cid
        
        # Return to depot
        travel_t = ttime(current_loc, depot_id)
        travel_d = dist(current_loc, depot_id)
        total_dist_cost += cost_factor * travel_d
        end_time = current_time + travel_t
        
        if end_time > horizon_end:
            return None
        if end_time - horizon_start > max_route_duration:
            return None
        
        total_cost = fixed + total_dist_cost
        return total_cost
    
    def evaluate_complete_vehicle_route_simple(trailer_custs, subtours, vc):
        """
        Evaluate a complete vehicle route.
        trailer_custs: list of trailer customer IDs visited with trailer attached (in order)
        subtours: list of (park_loc_index_in_main_route, park_loc, [truck_customer_ids])
          park_loc_index: after which trailer_cust index to do the subtour
                          -1 means before any trailer customer (right from depot)
                          len(trailer_custs) means after all trailer customers
        """
        truck_cap = vc['truck_capacity']
        trailer_cap = vc['trailer_capacity']
        cost_factor = vc['truck_cost_factor']
        tow_mult = vc['trailer_towing_cost_multiplier']
        fixed = vc['truck_fixed_cost']
        
        # Build the full sequence of actions
        # Main route: depot -> [trailer_custs with possible subtours interspersed] -> depot
        
        # Sort subtours by their position
        subtour_map = {}  # position -> list of (park_loc, [customers])
        for (pos, ploc, custs) in subtours:
            if pos not in subtour_map:
                subtour_map[pos] = []
            subtour_map[pos].append((ploc, custs))
        
        truck_load = 0
        trailer_load = 0
        total_cost = fixed
        current_time = horizon_start
        current_loc = depot_id
        trailer_attached = True
        
        def travel_with_trailer(from_loc, to_loc):
            nonlocal current_time, total_cost, current_loc
            d = dist(from_loc, to_loc)
            t = ttime(from_loc, to_loc)
            total_cost += (cost_factor + tow_mult) * d
            current_time += t
            current_loc = to_loc
        
        def travel_without_trailer(from_loc, to_loc):
            nonlocal current_time, total_cost, current_loc
            d = dist(from_loc, to_loc)
            t = ttime(from_loc, to_loc)
            total_cost += cost_factor * d
            current_time += t
            current_loc = to_loc
        
        def do_subtour_at(park_loc, cust_list):
            nonlocal truck_load, trailer_load, current_time, total_cost, current_loc
            
            # First travel to park location with trailer
            if current_loc != park_loc:
                travel_with_trailer(current_loc, park_loc)
            
            # If park_loc is a trailer customer, serve it now (with trailer attached)
            if park_loc in trc_ids and park_loc in all_customer_ids:
                # Check if park_loc should be served - only if it's in our customer list
                pass  # We handle serving separately
            
            # Decouple trailer
            current_time += decoupling_time
            total_cost += decoupling_cost
            
            # Do subtour(s) visiting cust_list
            if cust_list:
                sub_loc = park_loc
                for cid in cust_list:
                    travel_without_trailer(sub_loc, cid)
                    
                    tw_start, tw_end = customer_tw.get(cid, (horizon_start, horizon_end))
                    if current_time < tw_start:
                        current_time = tw_start
                    if current_time > tw_end:
                        return False
                    
                    truck_load += customer_supply[cid]
                    if truck_load > truck_cap:
                        return False
                    
                    stime = customer_service_time.get(cid, service_time_param)
                    current_time += stime
                    sub_loc = cid
                
                # Return to park location
                travel_without_trailer(sub_loc, park_loc)
            
            # Transfer load from truck to trailer
            if truck_load > 0 and trailer_load < trailer_cap:
                transfer_amount = min(truck_load, trailer_cap - trailer_load)
                trailer_load += transfer_amount
                truck_load -= transfer_amount
                current_time += transfer_time
                total_cost += transfer_cost
            
            # Couple trailer
            current_time += coupling_time
            total_cost += coupling_cost
            
            return True
        
        # Process subtours at position -1 (before any trailer customer)
        if -1 in subtour_map:
            for (ploc, custs) in subtour_map[-1]:
                if not do_subtour_at(ploc, custs):
                    return None
        
        # Visit trailer customers in order
        for idx, cid in enumerate(trailer_custs):
            travel_with_trailer(current_loc, cid)
            
            tw_start, tw_end = customer_tw.get(cid, (horizon_start, horizon_end))
            if current_time < tw_start:
                current_time = tw_start
            if current_time > tw_end:
                return None
            
            supply = customer_supply[cid]
            # Load into trailer first, then truck
            trailer_space = trailer_cap - trailer_load
            if trailer_space >= supply:
                trailer_load += supply
            else:
                trailer_load += trailer_space
                truck_load += supply - trailer_space
            
            if truck_load > truck_cap:
                return None
            
            stime = customer_service_time.get(cid, service_time_param)
            current_time += stime
            
            # Process subtours at this position
            if idx in subtour_map:
                for (ploc, custs) in subtour_map[idx]:
                    if ploc == cid:
                        # Decouple at current location
                        current_time += decoupling_time
                        total_cost += decoupling_cost
                        
                        if custs:
                            sub_loc = ploc
                            for scid in custs:
                                travel_without_trailer(sub_loc, scid)
                                tw_s, tw_e = customer_tw.get(scid, (horizon_start, horizon_end))
                                if current_time < tw_s:
                                    current_time = tw_s
                                if current_time > tw_e:
                                    return None
                                truck_load += customer_supply[scid]
                                if truck_load > truck_cap:
                                    return None
                                st = customer_service_time.get(scid, service_time_param)
                                current_time += st
                                sub_loc = scid
                            travel_without_trailer(sub_loc, ploc)
                        
                        # Transfer
                        if truck_load > 0 and trailer_load < trailer_cap:
                            tr_amt = min(truck_load, trailer_cap - trailer_load)
                            trailer_load += tr_amt
                            truck_load -= tr_amt
                            current_time += transfer_time
                            total_cost += transfer_cost
                        
                        current_time += coupling_time
                        total_cost += coupling_cost
                    else:
                        if not do_subtour_at(ploc, custs):
                            return None
        
        # Process subtours at position len(trailer_custs)
        pos_end = len(trailer_custs)
        if pos_end in subtour_map:
            for (ploc, custs) in subtour_map[pos_end]:
                if not do_subtour_at(ploc, custs):
                    return None
        
        # Return to depot
        travel_with_trailer(current_loc, depot_id)
        
        if current_time > horizon_end:
            return None
        if current_time - horizon_start > max_route_duration:
            return None
        
        return total_cost
    
    # =========================================================================
    # MAIN HEURISTIC: Greedy construction
    # =========================================================================
    
    # Available vehicles per class
    vc_avail = {}
    for vc in vehicle_classes:
        vc_avail[vc['class_id']] = vc['num_available']
    
    # Track truck/trailer type usage
    truck_type_avail = {}
    for tt in fleet['truck_types']:
        truck_type_avail[tt['id']] = tt['num_available']
    
    trailer_type_avail = {}
    for tt in fleet.get('trailer_types', []):
        trailer_type_avail[tt['id']] = tt['num_available']
    
    # Sort vehicle classes by cost efficiency (cheapest first)
    # Prefer classes with trailers for larger capacity
    vc_sorted = sorted(vehicle_classes, key=lambda v: v['truck_fixed_cost'])
    
    unserved = set(all_customer_ids)
    routes = []
    
    def build_solo_route(vc, available_customers):
        """Build a route for a solo truck using nearest neighbor."""
        truck_cap = vc['truck_capacity']
        cost_factor = vc['truck_cost_factor']
        
        route_customers = []
        current_loc = depot_id
        current_time = horizon_start
        current_load = 0
        remaining = set(available_customers)
        
        while remaining:
            best_cid = None
            best_cost = float('inf')
            
            for cid in remaining:
                supply = customer_supply[cid]
                if current_load + supply > truck_cap:
                    continue
                
                travel_t = ttime(current_loc, cid)
                arrive = current_time + travel_t
                tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
                if arrive < tw_s:
                    arrive = tw_s
                if arrive > tw_e:
                    continue
                
                stime = customer_service_time.get(cid, service_time_param)
                depart = arrive + stime
                # Check if can return to depot in time
                return_time = depart + ttime(cid, depot_id)
                if return_time > horizon_end or return_time - horizon_start > max_route_duration:
                    continue
                
                d = dist(current_loc, cid)
                insertion_cost = cost_factor * d
                
                if insertion_cost < best_cost:
                    best_cost = insertion_cost
                    best_cid = cid
            
            if best_cid is None:
                break
            
            travel_t = ttime(current_loc, best_cid)
            arrive = current_time + travel_t
            tw_s, tw_e = customer_tw.get(best_cid, (horizon_start, horizon_end))
            if arrive < tw_s:
                arrive = tw_s
            stime = customer_service_time.get(best_cid, service_time_param)
            
            route_customers.append(best_cid)
            current_load += customer_supply[best_cid]
            current_time = arrive + stime
            current_loc = best_cid
            remaining.remove(best_cid)
        
        return route_customers
    
    def build_complete_vehicle_route(vc, available_customers):
        """Build a route for a complete vehicle (truck + trailer)."""
        truck_cap = vc['truck_capacity']
        trailer_cap = vc['trailer_capacity']
        cost_factor = vc['truck_cost_factor']
        tow_mult = vc['trailer_towing_cost_multiplier']
        
        # Strategy: visit trailer customers with trailer, 
        # do subtours from parking locations for truck customers
        avail_trailer_custs = [c for c in available_customers if c in trc_ids]
        avail_truck_custs = [c for c in available_customers if c in tc_ids]
        
        route_trailer_custs = []
        route_subtours = []  # (position, park_loc, [truck_custs])
        
        current_loc = depot_id
        current_time = horizon_start
        truck_load = 0
        trailer_load = 0
        
        remaining_trailer = set(avail_trailer_custs)
        remaining_truck = set(avail_truck_custs)
        all_served = []
        
        # First, greedily add trailer customers
        while remaining_trailer:
            best_cid = None
            best_cost = float('inf')
            
            for cid in remaining_trailer:
                supply = customer_supply[cid]
                new_trailer_load = trailer_load
                new_truck_load = truck_load
                t_space = trailer_cap - new_trailer_load
                if t_space >= supply:
                    new_trailer_load += supply
                else:
                    new_trailer_load += t_space
                    new_truck_load += supply - t_space
                
                if new_truck_load > truck_cap:
                    continue
                
                travel_t = ttime(current_loc, cid)
                d = dist(current_loc, cid)
                arrive = current_time + travel_t
                tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
                if arrive < tw_s:
                    arrive = tw_s
                if arrive > tw_e:
                    continue
                
                stime = customer_service_time.get(cid, service_time_param)
                depart = arrive + stime
                return_time = depart + ttime(cid, depot_id)
                if return_time > horizon_end or return_time - horizon_start > max_route_duration:
                    continue
                
                cost = (cost_factor + tow_mult) * d
                if cost < best_cost:
                    best_cost = cost
                    best_cid = cid
            
            if best_cid is None:
                break
            
            supply = customer_supply[best_cid]
            t_space = trailer_cap - trailer_load
            if t_space >= supply:
                trailer_load += supply
            else:
                trailer_load += t_space
                truck_load += supply - t_space
            
            travel_t = ttime(current_loc, best_cid)
            arrive = current_time + travel_t
            tw_s, tw_e = customer_tw.get(best_cid, (horizon_start, horizon_end))
            if arrive < tw_s:
                arrive = tw_s
            stime = customer_service_time.get(best_cid, service_time_param)
            
            route_trailer_custs.append(best_cid)
            all_served.append(best_cid)
            current_time = arrive + stime
            current_loc = best_cid
            remaining_trailer.remove(best_cid)
        
        # Now try to add truck customers via subtours
        # Find best parking location for subtours
        if remaining_truck:
            # Try to do subtour from last trailer customer or a transshipment location
            possible_parks = set()
            for cid in route_trailer_custs:
                possible_parks.add(cid)
            for tid in trans_ids:
                possible_parks.add(tid)
            
            if not possible_parks and not route_trailer_custs:
                # No parking location available, can't serve truck customers with this vehicle
                pass
            else:
                # Try subtour from end of route
                for park_loc in possible_parks:
                    if not remaining_truck:
                        break
                    
                    # Save state
                    saved_time = current_time
                    saved_loc = current_loc
                    saved_truck_load = truck_load
                    saved_trailer_load = trailer_load
                    
                    # Travel to park location
                    if current_loc != park_loc:
                        t = ttime(current_loc, park_loc)
                        d_travel = dist(current_loc, park_loc)
                    else:
                        t = 0
                        d_travel = 0
                    
                    time_at_park = current_time + t + decoupling_time
                    subtour_truck_load = 0
                    
                    # Transfer before decoupling? No - "No load transfer occurs at decoupling operations"
                    # After decoupling, do subtour
                    subtour_custs = []
                    sub_loc = park_loc
                    sub_time = time_at_park
                    sub_truck_load = truck_load  # truck keeps its current load
                    
                    sub_remaining = set(remaining_truck)
                    while sub_remaining:
                        best_sc = None
                        best_sc_cost = float('inf')
                        
                        for scid in sub_remaining:
                            new_load = sub_truck_load + customer_supply[scid]
                            if new_load > truck_cap:
                                continue
                            
                            st = ttime(sub_loc, scid)
                            sd = dist(sub_loc, scid)
                            sarrive = sub_time + st
                            stw_s, stw_e = customer_tw.get(scid, (horizon_start, horizon_end))
                            if sarrive < stw_s:
                                sarrive = stw_s
                            if sarrive > stw_e:
                                continue
                            
                            sstime = customer_service_time.get(scid, service_time_param)
                            sdepart = sarrive + sstime
                            # Check can return to park, transfer, couple, return to depot
                            ret_park = sdepart + ttime(scid, park_loc)
                            ret_time = ret_park + transfer_time + coupling_time + ttime(park_loc, depot_id)
                            if ret_time > horizon_end or ret_time - horizon_start > max_route_duration:
                                continue
                            
                            sc = cost_factor * sd
                            if sc < best_sc_cost:
                                best_sc_cost = sc
                                best_sc = scid
                        
                        if best_sc is None:
                            break
                        
                        st = ttime(sub_loc, best_sc)
                        sarrive = sub_time + st
                        stw_s, stw_e = customer_tw.get(best_sc, (horizon_start, horizon_end))
                        if sarrive < stw_s:
                            sarrive = stw_s
                        sstime = customer_service_time.get(best_sc, service_time_param)
                        
                        subtour_custs.append(best_sc)
                        sub_truck_load += customer_supply[best_sc]
                        sub_time = sarrive + sstime
                        sub_loc = best_sc
                        sub_remaining.remove(best_sc)
                    
                    if subtour_custs:
                        # Return to park
                        sub_time += ttime(sub_loc, park_loc)
                        # Transfer
                        new_truck_load = sub_truck_load
                        new_trailer_load = trailer_load
                        if new_truck_load > 0 and new_trailer_load < trailer_cap:
                            tr_amt = min(new_truck_load, trailer_cap - new_trailer_load)
                            new_trailer_load += tr_amt
                            new_truck_load -= tr_amt
                            sub_time += transfer_time
                        # Couple
                        sub_time += coupling_time
                        # Return to depot
                        ret_time = sub_time + ttime(park_loc, depot_id)
                        
                        if ret_time <= horizon_end and ret_time - horizon_start <= max_route_duration:
                            # Accept this subtour
                            pos = len(route_trailer_custs) - 1
                            if park_loc in route_trailer_custs:
                                pos = route_trailer_custs.index(park_loc)
                            route_subtours.append((pos, park_loc, subtour_custs))
                            for sc in subtour_custs:
                                remaining_truck.discard(sc)
                                all_served.append(sc)
                            current_time = sub_time
                            current_loc = park_loc
                            truck_load = new_truck_load
                            trailer_load = new_trailer_load
                            break  # Only one subtour for now
        
        return route_trailer_custs, route_subtours, all_served
    
    def try_use_vehicle_class(vc, unserved_customers):
        """Try to build a route using this vehicle class."""
        vc_id = vc['class_id']
        tt_id = vc['truck_type_id']
        tr_id = vc['trailer_type_id']
        
        if truck_type_avail.get(tt_id, 0) <= 0:
            return None, []
        if tr_id != 0 and trailer_type_avail.get(tr_id, 0) <= 0:
            return None, []
        
        has_trailer = tr_id != 0
        
        if not has_trailer:
            # Solo truck - can visit all customers
            avail = list(unserved_customers)
            custs = build_solo_route(vc, avail)
            if custs:
                cost = evaluate_solo_route(custs, vc)
                if cost is not None:
                    return cost, custs
            return None, []
        else:
            # Complete vehicle
            avail = list(unserved_customers)
            trailer_custs, subtours, all_served = build_complete_vehicle_route(vc, avail)
            if all_served:
                return 0, all_served  # placeholder cost
            return None, []
    
    # =========================================================================
    # Better approach: build all routes greedily
    # =========================================================================
    
    # Simple approach: use solo trucks for everything first (simpler to implement correctly)
    # Then try to optimize
    
    def compute_route_details_solo(customers, vc):
        """Compute full details of a solo truck route."""
        truck_cap = vc['truck_capacity']
        cost_factor = vc['truck_cost_factor']
        fixed = vc['truck_fixed_cost']
        
        total_cost = fixed
        current_time = horizon_start
        current_loc = depot_id
        current_load = 0
        arrival_times = {}
        arrival_times[depot_id] = horizon_start
        
        for cid in customers:
            travel_t = ttime(current_loc, cid)
            d = dist(current_loc, cid)
            arrive = current_time + travel_t
            tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
            if arrive < tw_s:
                arrive = tw_s
            
            current_load += customer_supply[cid]
            total_cost += cost_factor * d
            stime = customer_service_time.get(cid, service_time_param)
            current_time = arrive + stime
            current_loc = cid
            arrival_times[cid] = arrive
        
        # Return to depot
        d = dist(current_loc, depot_id)
        travel_t = ttime(current_loc, depot_id)
        total_cost += cost_factor * d
        end_time = current_time + travel_t
        
        return total_cost, arrival_times, end_time
    
    def is_feasible_solo(customers, vc):
        """Check if a solo route is feasible."""
        truck_cap = vc['truck_capacity']
        current_time = horizon_start
        current_loc = depot_id
        current_load = 0
        
        for cid in customers:
            travel_t = ttime(current_loc, cid)
            arrive = current_time + travel_t
            tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
            if arrive < tw_s:
                arrive = tw_s
            if arrive > tw_e:
                return False
            
            current_load += customer_supply[cid]
            if current_load > truck_cap:
                return False
            
            stime = customer_service_time.get(cid, service_time_param)
            current_time = arrive + stime
            current_loc = cid
        
        travel_t = ttime(current_loc, depot_id)
        end_time = current_time + travel_t
        if end_time > horizon_end:
            return False
        if end_time - horizon_start > max_route_duration:
            return False
        
        return True
    
    # =========================================================================
    # Complete vehicle route evaluator with full detail
    # =========================================================================
    
    def evaluate_and_detail_complete_route(main_seq, vc):
        """
        main_seq: list of items, each is:
          ('with_trailer', cust_id) - visit trailer customer with trailer
          ('subtour', park_loc, [cust_ids]) - decouple, do subtour, couple
        
        Returns (cost, arrival_times, physical_sequence, decouple_locs, couple_locs) or None
        """
        truck_cap = vc['truck_capacity']
        trailer_cap = vc['trailer_capacity']
        cost_factor = vc['truck_cost_factor']
        tow_mult = vc['trailer_towing_cost_multiplier']
        fixed = vc['truck_fixed_cost']
        
        total_cost = fixed
        current_time = horizon_start
        current_loc = depot_id
        truck_load = 0
        trailer_load = 0
        trailer_attached = True
        
        physical_seq = [depot_id]
        decouple_locs = []
        couple_locs = []
        arrival_times = {}
        arrival_times[depot_id] = horizon_start
        
        for item in main_seq:
            if item[0] == 'with_trailer':
                cid = item[1]
                if not trailer_attached:
                    return None  # shouldn't happen
                
                d = dist(current_loc, cid)
                t = ttime(current_loc, cid)
                total_cost += (cost_factor + tow_mult) * d
                arrive = current_time + t
                
                tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
                if arrive < tw_s:
                    arrive = tw_s
                if arrive > tw_e:
                    return None
                
                supply = customer_supply[cid]
                t_space = trailer_cap - trailer_load
                if t_space >= supply:
                    trailer_load += supply
                else:
                    trailer_load += t_space
                    truck_load += supply - t_space
                
                if truck_load > truck_cap:
                    return None
                
                stime = customer_service_time.get(cid, service_time_param)
                current_time = arrive + stime
                current_loc = cid
                physical_seq.append(cid)
                arrival_times[cid] = arrive
                
            elif item[0] == 'subtour':
                park_loc = item[1]
                subtour_custs = item[2]
                
                # Travel to park location with trailer (if not already there)
                if current_loc != park_loc:
                    d = dist(current_loc, park_loc)
                    t = ttime(current_loc, park_loc)
                    total_cost += (cost_factor + tow_mult) * d
                    current_time += t
                    current_loc = park_loc
                    physical_seq.append(park_loc)
                
                # Decouple
                current_time += decoupling_time
                total_cost += decoupling_cost
                decouple_locs.append(park_loc)
                trailer_attached = False
                
                # Subtour
                sub_loc = park_loc
                for scid in subtour_custs:
                    d = dist(sub_loc, scid)
                    t = ttime(sub_loc, scid)
                    total_cost += cost_factor * d
                    arrive = current_time + t
                    
                    tw_s, tw_e = customer_tw.get(scid, (horizon_start, horizon_end))
                    if arrive < tw_s:
                        arrive = tw_s
                    if arrive > tw_e:
                        return None
                    
                    truck_load += customer_supply[scid]
                    if truck_load > truck_cap:
                        return None
                    
                    stime = customer_service_time.get(scid, service_time_param)
                    current_time = arrive + stime
                    sub_loc = scid
                    physical_seq.append(scid)
                    arrival_times[scid] = arrive
                
                # Return to park location
                if sub_loc != park_loc:
                    d = dist(sub_loc, park_loc)
                    t = ttime(sub_loc, park_loc)
                    total_cost += cost_factor * d
                    current_time += t
                    physical_seq.append(park_loc)
                
                # Transfer load from truck to trailer
                if truck_load > 0 and trailer_load < trailer_cap:
                    tr_amt = min(truck_load, trailer_cap - trailer_load)
                    trailer_load += tr_amt
                    truck_load -= tr_amt
                    current_time += transfer_time
                    total_cost += transfer_cost
                
                # Couple
                current_time += coupling_time
                total_cost += coupling_cost
                couple_locs.append(park_loc)
                trailer_attached = True
        
        # Return to depot
        d = dist(current_loc, depot_id)
        t = ttime(current_loc, depot_id)
        total_cost += (cost_factor + tow_mult) * d
        current_time += t
        physical_seq.append(depot_id)
        
        if current_time > horizon_end:
            return None
        if current_time - horizon_start > max_route_duration:
            return None
        
        return total_cost, arrival_times, physical_seq, decouple_locs, couple_locs
    
    # =========================================================================
    # Main construction heuristic
    # =========================================================================
    
    def construct_solution():
        """Construct an initial feasible solution."""
        unserved = set(all_customer_ids)
        solution_routes = []
        
        # Track available vehicles
        tt_used = {tt['id']: 0 for tt in fleet['truck_types']}
        tr_used = {tt['id']: 0 for tt in fleet.get('trailer_types', [])}
        
        # Sort customers by distance from depot (farthest first for better packing)
        sorted_customers = sorted(all_customer_ids, 
                                  key=lambda c: dist(depot_id, c), reverse=True)
        
        while unserved:
            if time.time() - start_time > time_limit * 0.8:
                break
            
            best_route = None
            best_cost = float('inf')
            best_served = []
            best_vc = None
            
            for vc in vehicle_classes:
                tt_id = vc['truck_type_id']
                tr_id = vc['trailer_type_id']
                
                if tt_used.get(tt_id, 0) >= truck_type_avail.get(tt_id, 0):
                    continue
                if tr_id != 0 and tr_used.get(tr_id, 0) >= trailer_type_avail.get(tr_id, 0):
                    continue
                
                has_trailer = tr_id != 0
                
                if not has_trailer:
                    # Solo truck - nearest neighbor
                    avail = [c for c in sorted_customers if c in unserved]
                    custs = build_solo_route(vc, avail)
                    if custs:
                        cost = evaluate_solo_route(custs, vc)
                        if cost is not None:
                            # Cost per customer
                            efficiency = cost / len(custs) if custs else float('inf')
                            if len(custs) > len(best_served) or (len(custs) == len(best_served) and cost < best_cost):
                                best_route = ('solo', custs)
                                best_cost = cost
                                best_served = custs
                                best_vc = vc
                else:
                    # Complete vehicle
                    avail = [c for c in sorted_customers if c in unserved]
                    avail_trailer = [c for c in avail if c in trc_ids]
                    avail_truck = [c for c in avail if c in tc_ids]
                    
                    # Build: first visit trailer customers with trailer,
                    # then find a good parking spot for truck customer subtours
                    
                    # Nearest neighbor for trailer customers
                    route_tc = []
                    current_loc_nn = depot_id
                    current_time_nn = horizon_start
                    truck_load_nn = 0
                    trailer_load_nn = 0
                    remaining_tc = set(avail_trailer)
                    
                    while remaining_tc:
                        best_next = None
                        best_next_cost = float('inf')
                        
                        for cid in remaining_tc:
                            supply = customer_supply[cid]
                            new_tl = trailer_load_nn
                            new_tkl = truck_load_nn
                            ts = vc['trailer_capacity'] - new_tl
                            if ts >= supply:
                                new_tl += supply
                            else:
                                new_tl += ts
                                new_tkl += supply - ts
                            
                            if new_tkl > vc['truck_capacity']:
                                continue
                            
                            t = ttime(current_loc_nn, cid)
                            d = dist(current_loc_nn, cid)
                            arrive = current_time_nn + t
                            tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
                            if arrive < tw_s:
                                arrive = tw_s
                            if arrive > tw_e:
                                continue
                            
                            stime = customer_service_time.get(cid, service_time_param)
                            depart = arrive + stime
                            ret = depart + ttime(cid, depot_id)
                            if ret > horizon_end or ret - horizon_start > max_route_duration:
                                continue
                            
                            c = (vc['truck_cost_factor'] + vc['trailer_towing_cost_multiplier']) * d
                            if c < best_next_cost:
                                best_next_cost = c
                                best_next = cid
                        
                        if best_next is None:
                            break
                        
                        supply = customer_supply[best_next]
                        ts = vc['trailer_capacity'] - trailer_load_nn
                        if ts >= supply:
                            trailer_load_nn += supply
                        else:
                            trailer_load_nn += ts
                            truck_load_nn += supply - ts
                        
                        t = ttime(current_loc_nn, best_next)
                        arrive = current_time_nn + t
                        tw_s, tw_e = customer_tw.get(best_next, (horizon_start, horizon_end))
                        if arrive < tw_s:
                            arrive = tw_s
                        stime = customer_service_time.get(best_next, service_time_param)
                        
                        route_tc.append(best_next)
                        current_time_nn = arrive + stime
                        current_loc_nn = best_next
                        remaining_tc.remove(best_next)
                    
                    # Now try subtours for truck customers
                    main_seq = [('with_trailer', c) for c in route_tc]
                    all_served_here = list(route_tc)
                    
                    # Find best parking location for subtour
                    if avail_truck:
                        best_park = None
                        best_subtour_custs = []
                        best_subtour_cost = float('inf')
                        
                        # Try each possible parking location
                        candidate_parks = set()
                        for c in route_tc:
                            candidate_parks.add(c)
                        for t_id in trans_ids:
                            candidate_parks.add(t_id)
                        
                        if not candidate_parks:
                            # Use transshipment or go to one specifically
                            for t_id in trans_ids:
                                candidate_parks.add(t_id)
                        
                        for park in candidate_parks:
                            # Try building subtour from this park location
                            test_seq = list(main_seq)
                            
                            # If park is in route_tc, insert subtour after visiting it
                            # Otherwise, add travel to park location
                            subtour_custs_test = []
                            sub_loc = park
                            sub_time_est = current_time_nn
                            
                            # Rough estimate: can we do any subtour from here?
                            sub_truck_load = truck_load_nn
                            sub_remaining = set(avail_truck)
                            
                            while sub_remaining:
                                best_sc = None
                                best_sc_d = float('inf')
                                
                                for scid in sub_remaining:
                                    if sub_truck_load + customer_supply[scid] > vc['truck_capacity']:
                                        continue
                                    d_sc = dist(sub_loc, scid)
                                    if d_sc < best_sc_d:
                                        best_sc_d = d_sc
                                        best_sc = scid
                                
                                if best_sc is None:
                                    break
                                
                                subtour_custs_test.append(best_sc)
                                sub_truck_load += customer_supply[best_sc]
                                sub_loc = best_sc
                                sub_remaining.remove(best_sc)
                            
                            if subtour_custs_test:
                                # Insert subtour at end of main sequence
                                test_full_seq = list(main_seq) + [('subtour', park, subtour_custs_test)]
                                result = evaluate_and_detail_complete_route(test_full_seq, vc)
                                if result is not None:
                                    cost_test = result[0]
                                    if len(subtour_custs_test) + len(route_tc) > len(best_served) or \
                                       (len(subtour_custs_test) + len(route_tc) == len(best_served) and cost_test < best_cost):
                                        # Try to validate and use
                                        best_park = park
                                        best_subtour_custs = subtour_custs_test
                                        all_served_complete = list(route_tc) + list(subtour_custs_test)
                                        
                                        if len(all_served_complete) > len(best_served) or \
                                           (len(all_served_complete) == len(best_served) and cost_test < best_cost):
                                            best_route = ('complete', test_full_seq)
                                            best_cost = cost_test
                                            best_served = all_served_complete
                                            best_vc = vc
                        
                        if not best_park and route_tc:
                            # No subtour, just trailer customers
                            result = evaluate_and_detail_complete_route(main_seq, vc)
                            if result is not None:
                                cost_test = result[0]
                                if len(route_tc) > len(best_served) or \
                                   (len(route_tc) == len(best_served) and cost_test < best_cost):
                                    best_route = ('complete', main_seq)
                                    best_cost = cost_test
                                    best_served = list(route_tc)
                                    best_vc = vc
                    elif route_tc:
                        result = evaluate_and_detail_complete_route(main_seq, vc)
                        if result is not None:
                            cost_test = result[0]
                            if len(route_tc) > len(best_served) or \
                               (len(route_tc) == len(best_served) and cost_test < best_cost):
                                best_route = ('complete', main_seq)
                                best_cost = cost_test
                                best_served = list(route_tc)
                                best_vc = vc
            
            if best_route is None or not best_served:
                # Try individual customers with any available vehicle
                # Pick the first unserved customer and assign to cheapest vehicle
                if not unserved:
                    break
                
                target = next(iter(unserved))
                found = False
                for vc in vc_sorted:
                    tt_id = vc['truck_type_id']
                    tr_id = vc['trailer_type_id']
                    if tt_used.get(tt_id, 0) >= truck_type_avail.get(tt_id, 0):
                        continue
                    if tr_id != 0 and tr_used.get(tr_id, 0) >= trailer_type_avail.get(tr_id, 0):
                        continue
                    
                    has_trailer = tr_id != 0
                    
                    if target in tc_ids and has_trailer:
                        # Need subtour for truck customer with complete vehicle
                        # Find parking location
                        parks = list(trans_ids)
                        if not parks:
                            # No transshipment, try without trailer
                            continue
                        
                        # Find closest park to both depot and customer
                        best_p = None
                        best_p_cost = float('inf')
                        for p in parks:
                            c = dist(depot_id, p) + dist(p, target) + dist(target, p) + dist(p, depot_id)
                            if c < best_p_cost:
                                best_p_cost = c
                                best_p = p
                        
                        if best_p is None:
                            continue
                        
                        seq = [('subtour', best_p, [target])]
                        result = evaluate_and_detail_complete_route(seq, vc)
                        if result is not None:
                            solution_routes.append(('complete', seq, vc, [target]))
                            tt_used[tt_id] = tt_used.get(tt_id, 0) + 1
                            if tr_id != 0:
                                tr_used[tr_id] = tr_used.get(tr_id, 0) + 1
                            unserved.remove(target)
                            found = True
                            break
                    elif target in tc_ids and not has_trailer:
                        # Solo truck can serve truck customer
                        if is_feasible_solo([target], vc):
                            solution_routes.append(('solo', [target], vc, [target]))
                            tt_used[tt_id] = tt_used.get(tt_id, 0) + 1
                            unserved.remove(target)
                            found = True
                            break
                    elif target in trc_ids:
                        if not has_trailer:
                            if is_feasible_solo([target], vc):
                                solution_routes.append(('solo', [target], vc, [target]))
                                tt_used[tt_id] = tt_used.get(tt_id, 0) + 1
                                unserved.remove(target)
                                found = True
                                break
                        else:
                            seq = [('with_trailer', target)]
                            result = evaluate_and_detail_complete_route(seq, vc)
                            if result is not None:
                                solution_routes.append(('complete', seq, vc, [target]))
                                tt_used[tt_id] = tt_used.get(tt_id, 0) + 1
                                tr_used[tr_id] = tr_used.get(tr_id, 0) + 1
                                unserved.remove(target)
                                found = True
                                break
                
                if not found:
                    # Cannot serve this customer - shouldn't happen with valid instance
                    break
                continue
            
            # Add best route to solution
            tt_id = best_vc['truck_type_id']
            tr_id = best_vc['trailer_type_id']
            
            if best_route[0] == 'solo':
                solution_routes.append(('solo', best_route[1], best_vc, best_served))
            else:
                solution_routes.append(('complete', best_route[1], best_vc, best_served))
            
            tt_used[tt_id] = tt_used.get(tt_id, 0) + 1
            if tr_id != 0:
                tr_used[tr_id] = tr_used.get(tr_id, 0) + 1
            
            for c in best_served:
                unserved.discard(c)
        
        return solution_routes, unserved
    
    solution_routes, remaining_unserved = construct_solution()
    
    if remaining_unserved:
        # Try harder to serve remaining customers
        # Use individual routes for each remaining customer
        tt_used = {}
        tr_used = {}
        for tt in fleet['truck_types']:
            tt_used[tt['id']] = 0
        for tt in fleet.get('trailer_types', []):
            tr_used[tt['id']] = 0
        
        for r in solution_routes:
            vc = r[2]
            tt_id = vc['truck_type_id']
            tr_id = vc['trailer_type_id']
            tt_used[tt_id] += 1
            if tr_id != 0:
                tr_used[tr_id] += 1
        
        for cid in list(remaining_unserved):
            for vc in vc_sorted:
                tt_id = vc['truck_type_id']
                tr_id = vc['trailer_type_id']
                if tt_used.get(tt_id, 0) >= truck_type_avail.get(tt_id, 0):
                    continue
                if tr_id != 0 and tr_used.get(tr_id, 0) >= trailer_type_avail.get(tr_id, 0):
                    continue
                
                has_trailer = tr_id != 0
                
                if cid in tc_ids:
                    if not has_trailer:
                        if customer_supply[cid] <= vc['truck_capacity'] and is_feasible_solo([cid], vc):
                            solution_routes.append(('solo', [cid], vc, [cid]))
                            tt_used[tt_id] += 1
                            remaining_unserved.discard(cid)
                            break
                    else:
                        # Need parking location
                        parks = sorted(trans_ids | trc_ids, key=lambda p: dist(depot_id, p) + dist(p, cid))
                        for park in parks:
                            if park in all_customer_ids and park not in remaining_unserved:
                                # This customer is already served, can't serve again
                                # But we can still use the location as parking
                                pass
                            seq = [('subtour', park, [cid])]
                            result = evaluate_and_detail_complete_route(seq, vc)
                            if result is not None:
                                solution_routes.append(('complete', seq, vc, [cid]))
                                tt_used[tt_id] += 1
                                if tr_id != 0:
                                    tr_used[tr_id] += 1
                                remaining_unserved.discard(cid)
                                break
                        if cid not in remaining_unserved:
                            break
                else:
                    if not has_trailer:
                        if customer_supply[cid] <= vc['truck_capacity'] and is_feasible_solo([cid], vc):
                            solution_routes.append(('solo', [cid], vc, [cid]))
                            tt_used[tt_id] += 1
                            remaining_unserved.discard(cid)
                            break
                    else:
                        seq = [('with_trailer', cid)]
                        result = evaluate_and_detail_complete_route(seq, vc)
                        if result is not None:
                            solution_routes.append(('complete', seq, vc, [cid]))
                            tt_used[tt_id] += 1
                            if tr_id != 0:
                                tr_used[tr_id] += 1
                            remaining_unserved.discard(cid)
                            break
    
    # =========================================================================
    # Local search improvement
    # =========================================================================
    
    def try_2opt_solo(customers, vc):
        """Try 2-opt improvement on a solo route."""
        if len(customers) < 2:
            return customers, False
        
        improved = False
        best_custs = list(customers)
        best_cost = evaluate_solo_route(best_custs, vc)
        if best_cost is None:
            return customers, False
        
        for i in range(len(best_custs) - 1):
            for j in range(i + 1, len(best_custs)):
                new_custs = best_custs[:i] + list(reversed(best_custs[i:j+1])) + best_custs[j+1:]
                new_cost = evaluate_solo_route(new_custs, vc)
                if new_cost is not None and new_cost < best_cost - 1e-6:
                    best_custs = new_custs
                    best_cost = new_cost
                    improved = True
        
        return best_custs, improved
    
    def improve_solution(solution_routes, time_budget):
        """Apply local search improvements."""
        imp_start = time.time()
        improved = True
        
        while improved and time.time() - imp_start < time_budget:
            improved = False
            
            # 2-opt on solo routes
            for i, route in enumerate(solution_routes):
                if time.time() - imp_start > time_budget:
                    break
                if route[0] == 'solo':
                    custs = route[1]
                    vc = route[2]
                    new_custs, imp = try_2opt_solo(custs, vc)
                    if imp:
                        solution_routes[i] = ('solo', new_custs, vc, new_custs)
                        improved = True
            
            # Try relocate: move a customer from one route to another
            for i in range(len(solution_routes)):
                if time.time() - imp_start > time_budget:
                    break
                if solution_routes[i][0] != 'solo':
                    continue
                
                custs_i = list(solution_routes[i][1])
                vc_i = solution_routes[i][2]
                
                for ci_idx in range(len(custs_i)):
                    if time.time() - imp_start > time_budget:
                        break
                    
                    cid = custs_i[ci_idx]
                    old_cost_i = evaluate_solo_route(custs_i, vc_i)
                    if old_cost_i is None:
                        continue
                    
                    new_custs_i = custs_i[:ci_idx] + custs_i[ci_idx+1:]
                    if new_custs_i:
                        new_cost_i = evaluate_solo_route(new_custs_i, vc_i)
                    else:
                        new_cost_i = 0  # Route becomes empty, save fixed cost
                    
                    if new_cost_i is None:
                        continue
                    
                    for j in range(len(solution_routes)):
                        if i == j:
                            continue
                        if solution_routes[j][0] != 'solo':
                            continue
                        
                        custs_j = list(solution_routes[j][1])
                        vc_j = solution_routes[j][2]
                        old_cost_j = evaluate_solo_route(custs_j, vc_j)
                        if old_cost_j is None:
                            continue
                        
                        # Check if customer type is compatible
                        if cid in tc_ids:
                            # Truck customer - can go to any solo truck
                            pass
                        
                        # Try inserting at each position
                        best_pos = None
                        best_new_cost_j = float('inf')
                        
                        for pos in range(len(custs_j) + 1):
                            new_custs_j = custs_j[:pos] + [cid] + custs_j[pos:]
                            nc = evaluate_solo_route(new_custs_j, vc_j)
                            if nc is not None and nc < best_new_cost_j:
                                best_new_cost_j = nc
                                best_pos = pos
                        
                        if best_pos is not None:
                            saving = (old_cost_i + old_cost_j) - (new_cost_i + best_new_cost_j)
                            if not new_custs_i:
                                saving += vc_i['truck_fixed_cost']  # Save fixed cost of eliminated route
                            
                            if saving > 1e-6:
                                new_custs_j = custs_j[:best_pos] + [cid] + custs_j[best_pos:]
                                solution_routes[j] = ('solo', new_custs_j, vc_j, new_custs_j)
                                if new_custs_i:
                                    solution_routes[i] = ('solo', new_custs_i, vc_i, new_custs_i)
                                else:
                                    solution_routes[i] = None
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # Remove None routes
            solution_routes = [r for r in solution_routes if r is not None]
        
        return solution_routes
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2:
        solution_routes = improve_solution(solution_routes, remaining_time - 1)
    
    # =========================================================================
    # Format output
    # =========================================================================
    
    parking_list = sorted(parking_ids)
    
    def format_solution(solution_routes):
        total_cost = 0.0
        output_routes = []
        vc_counters = {}
        
        for route in solution_routes:
            rtype = route[0]
            vc = route[2]
            vc_id = vc['class_id']
            
            if vc_id not in vc_counters:
                vc_counters[vc_id] = 0
            v_idx = vc_counters[vc_id]
            vc_counters[vc_id] += 1
            
            if rtype == 'solo':
                custs = route[1]
                cost_details = compute_route_details_solo(custs, vc)
                cost, arrival_times, end_time = cost_details
                total_cost += cost
                
                physical_seq = [depot_id] + list(custs) + [depot_id]
                
                # Build route_sequence (same as physical for solo)
                route_seq = list(physical_seq)
                
                arrival_dict = {}
                arrival_dict[str(depot_id)] = horizon_start
                for c in custs:
                    arrival_dict[str(c)] = arrival_times[c]
                
                output_routes.append({
                    'vehicle_class': vc_id,
                    'vehicle_index': v_idx,
                    'customers_served': list(custs),
                    'route_sequence': route_seq,
                    'route_sequence_physical': physical_seq,
                    'decouple_locations': [],
                    'couple_locations': [],
                    'arrival_times': arrival_dict
                })
                
            elif rtype == 'complete':
                main_seq = route[1]
                result = evaluate_and_detail_complete_route(main_seq, vc)
                
                if result is None:
                    # Shouldn't happen
                    continue
                
                cost, arrival_times, physical_seq, decouple_locs, couple_locs = result
                total_cost += cost
                
                all_custs = route[3]
                
                # Build route_sequence with expanded nodes
                # Expanded node convention:
                # Physical nodes: 0..n_orig-1
                # Decouple copy of P[i]: n_orig + i
                # Couple copy of P[i]: n_orig + len(P) + i
                
                route_seq = []
                route_phys = list(physical_seq)
                
                # Track decouple/couple visits to create expanded sequence
                decouple_idx = 0
                couple_idx = 0
                
                # Build expanded sequence
                # We need to walk through the main_seq and physical_seq together
                route_seq_expanded = [depot_id]
                
                for item in main_seq:
                    if item[0] == 'with_trailer':
                        route_seq_expanded.append(item[1])
                    elif item[0] == 'subtour':
                        park = item[1]
                        custs = item[2]
                        
                        # Travel to park (if not already there)
                        p_idx = parking_list.index(park) if park in parking_list else -1
                        
                        if p_idx >= 0:
                            decouple_node = n_orig + p_idx
                            couple_node = n_orig + len(parking_list) + p_idx
                        else:
                            decouple_node = park
                            couple_node = park
                        
                        # Add decouple node
                        route_seq_expanded.append(decouple_node)
                        
                        # Add subtour customers
                        for sc in custs:
                            route_seq_expanded.append(sc)
                        
                        # Add couple node (return to park)
                        route_seq_expanded.append(couple_node)
                
                route_seq_expanded.append(depot_id)
                
                arrival_dict = {}
                arrival_dict[str(depot_id)] = horizon_start
                for c, t in arrival_times.items():
                    arrival_dict[str(c)] = t
                
                # Add arrival times for expanded nodes
                # We need to compute these properly
                # Re-walk the route
                curr_time = horizon_start
                curr_loc = depot_id
                trailer_attached = True
                
                expanded_arrivals = {}
                expanded_arrivals[str(depot_id)] = horizon_start
                
                for item in main_seq:
                    if item[0] == 'with_trailer':
                        cid = item[1]
                        t = ttime(curr_loc, cid)
                        arrive = curr_time + t
                        tw_s, tw_e = customer_tw.get(cid, (horizon_start, horizon_end))
                        if arrive < tw_s:
                            arrive = tw_s
                        stime = customer_service_time.get(cid, service_time_param)
                        
                        expanded_arrivals[str(cid)] = arrive
                        curr_time = arrive + stime
                        curr_loc = cid
                        
                    elif item[0] == 'subtour':
                        park = item[1]
                        custs = item[2]
                        p_idx = parking_list.index(park) if park in parking_list else -1
                        
                        # Travel to park
                        if curr_loc != park:
                            t = ttime(curr_loc, park)
                            arrive_park = curr_time + t
                            curr_loc = park
                        else:
                            arrive_park = curr_time
                        
                        if p_idx >= 0:
                            decouple_node = n_orig + p_idx
                            couple_node = n_orig + len(parking_list) + p_idx
                        else:
                            decouple_node = park
                            couple_node = park
                        
                        expanded_arrivals[str(decouple_node)] = arrive_park
                        
                        curr_time = arrive_park + decoupling_time
                        
                        sub_loc = park
                        for scid in custs:
                            t = ttime(sub_loc, scid)
                            arrive = curr_time + t
                            tw_s, tw_e = customer_tw.get(scid, (horizon_start, horizon_end))
                            if arrive < tw_s:
                                arrive = tw_s
                            stime = customer_service_time.get(scid, service_time_param)
                            expanded_arrivals[str(scid)] = arrive
                            curr_time = arrive + stime
                            sub_loc = scid
                        
                        # Return to park
                        if sub_loc != park:
                            t = ttime(sub_loc, park)
                            curr_time += t
                        
                        # Transfer
                        # Check if transfer happens
                        # We already computed this in evaluate, just add time
                        curr_time += transfer_time  # simplified
                        
                        expanded_arrivals[str(couple_node)] = curr_time
                        curr_time += coupling_time
                        curr_loc = park
                
                output_routes.append({
                    'vehicle_class': vc_id,
                    'vehicle_index': v_idx,
                    'customers_served': all_custs,
                    'route_sequence': route_seq_expanded,
                    'route_sequence_physical': route_phys,
                    'decouple_locations': decouple_locs,
                    'couple_locations': couple_locs,
                    'arrival_times': expanded_arrivals
                })
        
        return total_cost, output_routes
    
    total_cost, output_routes = format_solution(solution_routes)
    
    # Build expanded node convention description
    solution = {
        'objective_value': total_cost,
        'routes': output_routes,
        'expanded_node_convention': f"Let n_orig be the number of physical locations and P be the instance parking-location IDs in their published list order. Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, and the couple copy of P[i] has ID n_orig+len(P)+i. Both copies map back to physical location P[i]."
    }
    
    if logger:
        logger.log_solution(total_cost, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    return solution

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)