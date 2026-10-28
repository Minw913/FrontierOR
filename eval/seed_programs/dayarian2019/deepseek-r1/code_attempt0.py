import math
import json

def main():
    # Read input from stdin
    data = json.load(sys.stdin)
    
    # Extract global parameters
    global_parameters = data["global_parameters"]
    orders = data["orders"]
    travel_time_matrix = data["travel_time_matrix"]
    production_windows = data["production_windows"]
    subperiods = data["subperiods"]
    periods = data["periods"]
    products = data["products"]
    num_vehicles = global_parameters["num_vehicles"]
    vehicle_capacity = global_parameters["vehicle_capacity"]
    horizon_start = global_parameters["horizon_start_minutes"]
    period_length = global_parameters["period_length_minutes"]
    num_periods = global_parameters["num_periods"]
    num_subperiods = global_parameters["num_subperiods"]
    setup_length_subperiods = global_parameters["setup_length_subperiods"]
    num_production_employees = global_parameters["num_production_employees"]
    num_workstations = global_parameters["num_workstations"]
    production_rates = {p["product_id"]: p["production_rate_per_subperiod"] for p in products}
    
    # Create order_id to index mapping
    order_id_to_index = {order["order_id"]: i for i, order in enumerate(orders)}
    
    # Precompute target departure period for each order
    for order in orders:
        order_id = order["order_id"]
        idx = order_id_to_index[order_id] + 1
        direct_travel_time = travel_time_matrix[0][idx]
        target_departure_time = (order["time_window_start"] + order["time_window_end"]) / 2 - direct_travel_time - order["service_time"]
        target_departure_period = int((target_departure_time - horizon_start) // period_length)
        
        # Get production windows for this order
        order_pw = production_windows.get(str(order_id), {})
        release_times = []
        due_times = []
        for p_id, qty in order["demands"].items():
            if p_id in order_pw:
                release_times.append(order_pw[p_id]["release_time"])
                due_times.append(order_pw[p_id]["due_time"])
        
        if release_times and due_times:
            earliest_period = max(0, int((min(release_times) - horizon_start) // period_length))
            latest_period = min(num_periods - 1, int((min(due_times) - horizon_start) // period_length))
            target_departure_period = max(earliest_period, min(target_departure_period, latest_period))
        
        order["target_departure_period"] = max(0, min(num_periods - 1, target_departure_period))
    
    # Group orders by target departure period
    period_orders = [[] for _ in range(num_periods)]
    for order in orders:
        period_orders[order["target_departure_period"]].append(order)
    
    # Form trips per period (bin packing by space_requirement)
    trips_by_period = [[] for _ in range(num_periods)]
    for period in range(num_periods):
        orders_in_period = sorted(period_orders[period], key=lambda o: o["space_requirement"], reverse=True)
        current_trip = []
        current_space = 0
        for order in orders_in_period:
            if current_space + order["space_requirement"] <= vehicle_capacity:
                current_trip.append(order)
                current_space += order["space_requirement"]
            else:
                if current_trip:
                    trips_by_period[period].append(current_trip)
                current_trip = [order]
                current_space = order["space_requirement"]
        if current_trip:
            trips_by_period[period].append(current_trip)
    
    # Assign vehicles to trips (round-robin per period)
    vehicle_trips = [[] for _ in range(num_vehicles)]
    vehicle_usage = [0] * num_vehicles
    for period in range(num_periods):
        for trip in trips_by_period[period]:
            vehicle_idx = min(range(num_vehicles), key=lambda i: vehicle_usage[i])
            vehicle_usage[vehicle_idx] += 1
            vehicle_trips[vehicle_idx].append({
                "period": period,
                "orders": trip
            })
    
    # Build routes output
    routes = []
    for vehicle_idx, trips in enumerate(vehicle_trips):
        if not trips:
            continue
        vehicle_route = {
            "vehicle": vehicle_idx,
            "trips": []
        }
        for trip_idx, trip in enumerate(trips):
            # Sort orders by time_window_start, then x-coordinate, then order_id
            sorted_orders = sorted(trip["orders"], key=lambda o: (o["time_window_start"], o.get("x", 0), o["order_id"]))
            vehicle_route["trips"].append({
                "trip": trip_idx,
                "customers": [o["order_id"] for o in sorted_orders],
                "start_period": trip["period"]
            })
        routes.append(vehicle_route)
    
    # Production scheduling
    production_schedule = []
    assigned = {}
    for product in products:
        p_id = product["product_id"]
        assigned[p_id] = [0.0] * num_subperiods
    
    for order in orders:
        order_id = order["order_id"]
        for p_id, demand in order["demands"].items():
            p_id = str(p_id)
            valid_subperiods = production_windows.get(str(order_id), {}).get(p_id, {}).get("valid_subperiod_ids", [])
            remaining = demand
            for subp in sorted(valid_subperiods):
                capacity = num_production_employees * production_rates[int(p_id)] * 0.8
                available = max(0, capacity - assigned[p_id][subp])
                assign_amt = min(remaining, available)
                if assign_amt > 0:
                    production_schedule.append({
                        "order_id": order_id,
                        "product_id": int(p_id),
                        "subperiod_id": subp,
                        "quantity": assign_amt
                    })
                    assigned[p_id][subp] += assign_amt
                    remaining -= assign_amt
                    if remaining <= 0:
                        break
    
    # Compute staffing levels
    employees = {}
    for product in products:
        p_id = product["product_id"]
        employees[p_id] = [0] * num_subperiods
    
    for prod_sched in production_schedule:
        p_id = prod_sched["product_id"]
        subp = prod_sched["subperiod_id"]
        qty = prod_sched["quantity"]
        rate = production_rates[p_id]
        employees[p_id][subp] += qty / rate
    
    # Round up employees and compute setups
    setups = {}
    for p_id in employees:
        setups[p_id] = [0] * num_subperiods
        for subp in range(num_subperiods):
            employees[p_id][subp] = math.ceil(employees[p_id][subp])
        if num_subperiods > 0:
            setups[p_id][0] = employees[p_id][0]
            for subp in range(1, num_subperiods):
                diff = employees[p_id][subp] - employees[p_id][subp-1]
                setups[p_id][subp] = max(0, diff)
    
    # Compute active setups per subperiod
    setup_emp = [0] * num_subperiods
    for subp in range(num_subperiods):
        for p_id in setups:
            start_subp = max(0, subp - setup_length_subperiods + 1)
            for s in range(start_subp, subp + 1):
                setup_emp[subp] += setups[p_id][s]
    
    # Compute total employees per subperiod
    total_emp = [0] * num_subperiods
    for subp in range(num_subperiods):
        for p_id in employees:
            total_emp[subp] += employees[p_id][subp]
        total_emp[subp] += setup_emp[subp]
    
    # Compute workstations_by_period and shift_starts_by_period
    workstations_by_period = [0] * num_periods
    for period in range(num_periods):
        start_subp = period * global_parameters["subperiods_per_period"]
        end_subp = min(start_subp + global_parameters["subperiods_per_period"], num_subperiods)
        workstations_by_period[period] = max(total_emp[start_subp:end_subp], default=0)
    
    shift_starts_by_period = [0] * num_periods
    shift_starts_by_period[0] = workstations_by_period[0]
    for period in range(1, num_periods):
        shift_starts_by_period[period] = max(0, workstations_by_period[period] - workstations_by_period[period-1])
    
    # Prepare output
    output = {
        "routes": routes,
        "production_schedule": production_schedule,
        "workstations_by_period": workstations_by_period,
        "shift_starts_by_period": shift_starts_by_period
    }
    
    # Write output to stdout
    json.dump(output, sys.stdout)

if __name__ == "__main__":
    import sys
    main()