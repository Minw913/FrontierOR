import argparse
import json
import math
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    global_parameters = data["global_parameters"]
    orders = data["orders"]
    travel_time_matrix = data["travel_time_matrix"]
    production_windows = data["production_windows"]
    subperiods_data = data["subperiods"]
    periods = data["periods"]
    products = data["products"]
    depot = data["depot"]
    num_vehicles = data["num_vehicles"]
    num_drivers = data["num_drivers"]
    vehicle_capacity = data["vehicle_capacity"]
    num_workstations = data["num_workstations"]
    num_production_employees = data["num_production_employees"]
    
    horizon_start = global_parameters["horizon_start_minutes"]
    period_length = global_parameters["period_length_minutes"]
    num_periods = global_parameters["num_periods"]
    num_subperiods = global_parameters["num_subperiods"]
    setup_length_subperiods = global_parameters["setup_length_subperiods"]
    preparation_time_minutes = global_parameters["preparation_time_minutes"]
    service_time_minutes = global_parameters["service_time_minutes"]
    min_paid_time_minutes = global_parameters["min_paid_time_minutes"]
    driver_salary_per_minute = global_parameters["driver_salary_per_minute"]
    vehicle_operating_cost_per_minute = global_parameters["vehicle_operating_cost_per_minute"]
    employee_salary_per_subperiod = global_parameters["employee_salary_per_subperiod"]
    setup_cost = global_parameters["setup_cost"]
    max_shift_route_length_minutes = global_parameters["max_shift_route_length_minutes"]
    max_trips_per_route = global_parameters["max_trips_per_route"]

    production_rates = {p["product_id"]: p["production_rate_per_subperiod"] for p in products}
    order_id_to_index = {order["order_id"]: i for i, order in enumerate(orders)}
    matrix_index_to_order_id = {0: "depot"}
    for idx, order in enumerate(orders):
        matrix_index_to_order_id[idx+1] = order["order_id"]

    if time.time() - start_time > args.time_limit:
        return

    for order in orders:
        order_id = order["order_id"]
        idx = order_id_to_index[order_id] + 1
        direct_travel_time = travel_time_matrix[0][idx]
        target_delivery_time = (order["time_window_start"] + order["time_window_end"]) / 2
        target_departure_time = target_delivery_time - direct_travel_time - order["service_time"]
        target_departure_period = max(0, min(num_periods-1, int((target_departure_time - horizon_start) // period_length)))
        
        order_pw = production_windows.get(str(order_id), {})
        if order_pw:
            max_due_time = -1
            for p_id, pw_info in order_pw.items():
                if "due_time" in pw_info:
                    max_due_time = max(max_due_time, pw_info["due_time"])
            if max_due_time > 0:
                min_departure_period = max(0, min(num_periods-1, int((max_due_time - horizon_start) // period_length)))
                target_departure_period = max(target_departure_period, min_departure_period)
        order["target_departure_period"] = target_departure_period

    if time.time() - start_time > args.time_limit:
        return

    period_orders = [[] for _ in range(num_periods)]
    for order in orders:
        period_orders[order["target_departure_period"]].append(order)

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

    if time.time() - start_time > args.time_limit:
        return

    max_vehicles_to_assign = min(num_vehicles, num_drivers)
    vehicle_trips = [[] for _ in range(max_vehicles_to_assign)]
    vehicle_trip_counts = [0] * max_vehicles_to_assign
    for period in range(num_periods):
        for trip in trips_by_period[period]:
            candidate_vehicles = []
            for vehicle_idx in range(max_vehicles_to_assign):
                if vehicle_trip_counts[vehicle_idx] < max_trips_per_route:
                    candidate_vehicles.append(vehicle_idx)
            if candidate_vehicles:
                vehicle_idx = min(candidate_vehicles, key=lambda i: vehicle_trip_counts[i])
            else:
                vehicle_idx = min(range(max_vehicles_to_assign), key=lambda i: vehicle_trip_counts[i])
            vehicle_trip_counts[vehicle_idx] += 1
            vehicle_trips[vehicle_idx].append({
                "period": period,
                "orders": trip
            })

    if time.time() - start_time > args.time_limit:
        return

    routes = []
    total_vehicle_operating_cost = 0.0
    total_driver_cost = 0.0
    order_id_to_matrix_index = {order["order_id"]: i+1 for i, order in enumerate(orders)}
    for vehicle_idx, trips in enumerate(vehicle_trips):
        if not trips:
            continue
        vehicle_route = {
            "vehicle": vehicle_idx,
            "trips": []
        }
        sorted_trips = sorted(trips, key=lambda t: t["period"])
        first_period = sorted_trips[0]["period"]
        last_period = sorted_trips[-1]["period"]
        first_departure_time = horizon_start + first_period * period_length
        last_departure_time = horizon_start + last_period * period_length
        last_return_time = 0.0
        total_travel_time_route = 0.0
        for trip_idx, trip in enumerate(sorted_trips):
            orders_in_trip = trip["orders"]
            sorted_orders = sorted(orders_in_trip, key=lambda o: (o["time_window_start"], o.get("x", 0), o["order_id"]))
            customer_ids = [o["order_id"] for o in sorted_orders]
            vehicle_route["trips"].append({
                "trip": trip_idx,
                "customers": customer_ids,
                "start_period": trip["period"]
            })
            travel_time = 0.0
            service_time_total = 0.0
            prev_index = 0
            for order in sorted_orders:
                matrix_index = order_id_to_matrix_index[order["order_id"]]
                travel_time += travel_time_matrix[prev_index][matrix_index]
                service_time_total += order["service_time"]
                prev_index = matrix_index
            travel_time += travel_time_matrix[prev_index][0]
            total_travel_time_route += travel_time
            trip_return_time = horizon_start + trip["period"] * period_length + preparation_time_minutes + travel_time + service_time_total
            last_return_time = max(last_return_time, trip_return_time)
        total_vehicle_operating_cost += total_travel_time_route * vehicle_operating_cost_per_minute
        paid_time_minutes = last_return_time - first_departure_time
        paid_time_minutes = max(paid_time_minutes, min_paid_time_minutes)
        paid_time_minutes = min(paid_time_minutes, max_shift_route_length_minutes)
        driver_cost = paid_time_minutes * driver_salary_per_minute
        total_driver_cost += driver_cost
        routes.append(vehicle_route)

    if time.time() - start_time > args.time_limit:
        return

    production_schedule = []
    assigned = {}
    for product in products:
        p_id = product["product_id"]
        assigned[p_id] = [0.0] * num_subperiods

    for order in orders:
        order_id = order["order_id"]
        if time.time() - start_time > args.time_limit:
            for p_id, demand in order["demands"].items():
                p_id_int = int(p_id)
                if demand > 0:
                    valid_subperiods = production_windows.get(str(order_id), {}).get(p_id, {}).get("valid_subperiod_ids", [])
                    if valid_subperiods:
                        last_subp = max(valid_subperiods)
                        production_schedule.append({
                            "order": order_id,
                            "product": p_id_int,
                            "subperiod": last_subp,
                            "quantity": demand
                        })
                        assigned[p_id_int][last_subp] += demand
            continue
        for p_id, demand in order["demands"].items():
            p_id_int = int(p_id)
            valid_subperiods = production_windows.get(str(order_id), {}).get(p_id, {}).get("valid_subperiod_ids", [])
            remaining = demand
            for subp in sorted(valid_subperiods):
                capacity = num_production_employees * production_rates[p_id_int] - assigned[p_id_int][subp]
                assign_amt = min(remaining, max(0, capacity))
                if assign_amt > 0:
                    production_schedule.append({
                        "order": order_id,
                        "product": p_id_int,
                        "subperiod": subp,
                        "quantity": assign_amt
                    })
                    assigned[p_id_int][subp] += assign_amt
                    remaining -= assign_amt
                    if remaining <= 0:
                        break
            if remaining > 0 and time.time() - start_time <= args.time_limit:
                for subp in sorted(valid_subperiods):
                    assign_amt = min(remaining, num_production_employees * production_rates[p_id_int])
                    if assign_amt > 0:
                        production_schedule.append({
                            "order": order_id,
                            "product": p_id_int,
                            "subperiod": subp,
                            "quantity": assign_amt
                        })
                        assigned[p_id_int][subp] += assign_amt
                        remaining -= assign_amt
                        if remaining <= 0:
                            break
            if remaining > 0:
                if valid_subperiods:
                    last_subp = max(valid_subperiods)
                    production_schedule.append({
                        "order": order_id,
                        "product": p_id_int,
                        "subperiod": last_subp,
                        "quantity": remaining
                    })
                    assigned[p_id_int][last_subp] += remaining

    if time.time() - start_time > args.time_limit:
        pass

    employees_by_product_subperiod = {}
    setups_by_product_subperiod = {}
    for product in products:
        p_id = product["product_id"]
        employees_by_product_subperiod[p_id] = [0] * num_subperiods
        setups_by_product_subperiod[p_id] = [0] * num_subperiods

    for prod_sched in production_schedule:
        p_id = prod_sched["product"]
        subp = prod_sched["subperiod"]
        qty = prod_sched["quantity"]
        rate = production_rates[p_id]
        employees_by_product_subperiod[p_id][subp] += qty / rate

    for p_id in employees_by_product_subperiod:
        for subp in range(num_subperiods):
            employees_by_product_subperiod[p_id][subp] = math.ceil(employees_by_product_subperiod[p_id][subp])
        if num_subperiods > 0:
            setups_by_product_subperiod[p_id][0] = employees_by_product_subperiod[p_id][0]
            for subp in range(1, num_subperiods):
                diff = employees_by_product_subperiod[p_id][subp] - employees_by_product_subperiod[p_id][subp-1]
                setups_by_product_subperiod[p_id][subp] = max(0, diff)

    setup_emp = [0] * num_subperiods
    for subp in range(num_subperiods):
        for p_id in setups_by_product_subperiod:
            start_subp = max(0, subp - setup_length_subperiods + 1)
            for s in range(start_subp, subp+1):
                setup_emp[subp] += setups_by_product_subperiod[p_id][s]

    total_emp = [0] * num_subperiods
    for subp in range(num_subperiods):
        for p_id in employees_by_product_subperiod:
            total_emp[subp] += employees_by_product_subperiod[p_id][subp]
        total_emp[subp] += setup_emp[subp]

    subperiods_per_period = global_parameters["subperiods_per_period"]
    workstations_by_period = [0] * num_periods
    for period in range(num_periods):
        start_subp = period * subperiods_per_period
        end_subp = min(start_subp + subperiods_per_period, num_subperiods)
        workstations_by_period[period] = max(total_emp[start_subp:end_subp]) if start_subp < end_subp else 0

    shift_starts_by_period = [0] * num_periods
    shift_starts_by_period[0] = workstations_by_period[0]
    for period in range(1, num_periods):
        shift_starts_by_period[period] = max(0, workstations_by_period[period] - workstations_by_period[period-1])

    total_employee_cost = 0.0
    for period in range(num_periods):
        total_employee_cost += workstations_by_period[period] * employee_salary_per_subperiod * subperiods_per_period

    total_setup_cost = 0.0
    for p_id in setups_by_product_subperiod:
        for subp in range(num_subperiods):
            total_setup_cost += setups_by_product_subperiod[p_id][subp] * setup_cost

    objective_value = total_driver_cost + total_vehicle_operating_cost + total_employee_cost + total_setup_cost

    production_staffing = {
        "employees_by_product_subperiod": [
            {"product": p_id, "subperiod": subp, "employees": int(employees_by_product_subperiod[p_id][subp])}
            for p_id in employees_by_product_subperiod
            for subp in range(num_subperiods)
        ],
        "setups_by_product_subperiod": [
            {"product": p_id, "subperiod": subp, "setups": int(setups_by_product_subperiod[p_id][subp])}
            for p_id in setups_by_product_subperiod
            for subp in range(num_subperiods)
        ],
        "workstations_by_period": workstations_by_period,
        "shift_starts_by_period": shift_starts_by_period
    }

    solution_dict = {
        "objective_value": objective_value,
        "routes": routes,
        "production_schedule": [
            {
                "order": s["order"],
                "product": s["product"],
                "subperiod": s["subperiod"],
                "quantity": s["quantity"]
            } for s in production_schedule
        ],
        "production_staffing": production_staffing
    }

    if logger:
        logger.log_solution(objective_value, solution_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()