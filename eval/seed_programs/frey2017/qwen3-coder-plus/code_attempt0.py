import argparse
import json
import math
from itertools import product
import gurobipy as gp
from gurobipy import GRB

def solve_airport_baggage(instance_data):
    # Parse input data
    num_periods = instance_data["planning_horizon"]["num_periods"]
    
    target_utilization = instance_data["objective"]["target_utilization"]
    threshold_values = instance_data["objective"]["threshold_values"]
    penalties = instance_data["objective"]["penalties"]
    num_thresholds = len(threshold_values)
    
    peak_intervals = instance_data["peak_intervals"]
    
    central_storage_capacity = instance_data["central_storage"]["capacity_bags"]
    depletion_rate_per_period = instance_data["central_storage"]["depletion_rate_bags_per_period"]
    depletion_offset = instance_data["central_storage"]["depletion_offset_periods"]
    
    working_rate_bags_per_period = instance_data["working_rate_bags_per_period"]
    baggage_transfer_time = instance_data["baggage_transfer_time_periods"]
    flight_end_offset = instance_data["flight_end_offset_periods"]
    quantile_used = instance_data["quantile_used"]
    
    num_flights = instance_data["num_flights"]
    num_carousels = instance_data["num_carousels"]
    
    flights_data = {f["flight_id"]: f for f in instance_data["flights"]}
    flights_ids = sorted(flights_data.keys())
    
    carousel_data = {c["carousel_id"]: c for c in instance_data["carousels"]}
    carousels_ids = sorted(carousel_data.keys())
    
    carousel_types_data = instance_data["carousel_types"]
    
    # Create model
    model = gp.Model("airport_baggage")
    model.setParam('OutputFlag', 0)
    
    # Variables
    x = {}  # x[flight_id, carousel_id] = 1 if flight is assigned to carousel
    for flight_id in flights_ids:
        for carousel_id in carousels_ids:
            x[flight_id, carousel_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{flight_id}_{carousel_id}")
    
    y = {}  # y[flight_id, carousel_id, ws_count] = 1 if flight is assigned to carousel with ws_count working stations
    for flight_id in flights_ids:
        for carousel_id in carousels_ids:
            carousel_type = carousel_data[carousel_id]["type_r"]
            bounds = flights_data[flight_id].get("working_station_bounds_by_carousel_type", {}).get(str(carousel_type), {})
            Wmin = bounds.get("Wmin", 0)
            Wmax = bounds.get("Wmax", 0)
            for ws_count in range(Wmin, Wmax + 1):
                y[flight_id, carousel_id, ws_count] = model.addVar(vtype=GRB.BINARY, name=f"y_{flight_id}_{carousel_id}_{ws_count}")
    
    z = {}  # z[flight_id, hs, ds] = 1 if flight starts at handling start hs and depletion start ds
    for flight_id in flights_ids:
        flight_info = flights_data[flight_id]
        earliest_start = flight_info["earliest_start_period"]
        latest_start = flight_info["latest_start_period"]
        end_period = flight_info["end_period"]
        
        # Determine possible start times for handling and depletion
        for hs in range(earliest_start, min(latest_start + 1, num_periods)):  # Handling start
            max_ds = hs  # Depletion must start before or at handling start
            min_ds = hs - depletion_offset  # Depletion must start in time to finish
            min_ds = max(0, min_ds)
            
            # Ensure end is realistic
            max_processing_time_needed = math.ceil(sum(flight_info["arrival_profile"]) / (working_rate_bags_per_period * 2))  # Rough estimate
            
            for ds in range(min_ds, max_ds + 1):
                z[flight_id, hs, ds] = model.addVar(vtype=GRB.BINARY, name=f"z_{flight_id}_{hs}_{ds}")
    
    # Violation variables for each carousel, peak interval, and threshold level
    v = {}
    for carousel_id in carousels_ids:
        for peak_interval in peak_intervals:
            for t_idx, _ in enumerate(threshold_values):
                v[carousel_id, peak_interval["peak_id"], t_idx] = model.addVar(vtype=GRB.BINARY,
                    name=f"v_{carousel_id}_{peak_interval['peak_id']}_{t_idx}")
    
    # Max utilization excess for each carousel, each peak interval
    u_max = {}
    for carousel_id in carousels_ids:
        for peak_interval in peak_intervals:
            u_max[carousel_id, peak_interval["peak_id"]] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_max_{carousel_id}_{peak_interval['peak_id']}"
            )
    
    # Workload contribution of each flight at each time period if assigned
    def calculate_flight_workload(flight_id, carousel_id, handling_start, ws_count):
        """Calculate workload of flight on carousel at each time period"""
        flight_info = flights_data[flight_id]
        workload_per_period = [0.0] * num_periods
        
        for rel_period in range(len(flight_info["arrival_profile"])):
            abs_period = handling_start + rel_period
            if 0 <= abs_period < num_periods:
                # Estimate how many bags remain on the belt at this time
                num_stations = ws_count
                rate_per_station = working_rate_bags_per_period
                total_load_rate = num_stations * rate_per_station
                
                # Approximation: assume bags arrive and are loaded over the course of the handling duration
                # Baggage stays on the belt until the end of handling
                if abs_period >= handling_start:
                    if flight_info["arrival_profile"][rel_period] > total_load_rate:
                        # Some bags might stay longer than one period
                        workload_per_period[abs_period] = min(flight_info["arrival_profile"][rel_period],
                                                              carousel_data[carousel_id]["Kcb"])
                        # Note: actual calculation would need more detailed tracking
                    else:
                        workload_per_period[abs_period] = flight_info["arrival_profile"][rel_period]
                    # Cap to carousel belt capacity
                    workload_per_period[abs_period] = min(workload_per_period[abs_period], 
                                                        carousel_data[carousel_id]["Kcb"])
        
        return workload_per_period
    
    # Storage contribution - simplified
    def calculate_flight_storage(flight_id, handling_start, depletion_start):
        """Calculate storage requirements of flight over time"""
        flight_info = flights_data[flight_id]
        storage_per_period = [0] * num_periods
        total_bags = sum(flight_info["arrival_profile"])
        
        # All arrival periods contribute to storage until depletion starts
        for rel_period in range(len(flight_info["arrival_profile"])):
            abs_period = handling_start + rel_period
            if 0 <= abs_period < num_periods:
                for store_period in range(depletion_start, abs_period):  # Before handling starts
                    if 0 <= store_period < num_periods:
                        # Add bags that arrived but weren't processed yet
                        if store_period < handling_start:
                            storage_per_period[store_period] += flight_info["arrival_profile"][rel_period]
        
        # Once handling starts, storage is reduced at depletion rate
        bags_remaining_in_storage = sum(storage_per_period[:handling_start]) if handling_start > 0 else 0
        for p in range(handling_start, num_periods):
            if bags_remaining_in_storage > 0:
                reduction = min(bags_remaining_in_storage, depletion_rate_per_period)
                for prev_p in range(max(0, p-num_periods+1), p+1):
                    if storage_per_period[prev_p] > 0:
                        reduction_now = min(storage_per_period[prev_p], reduction)
                        storage_per_period[prev_p] -= reduction_now
                        storage_per_period[p] = max(0, storage_per_period[p] + reduction_now)
                        reduction -= reduction_now
                        if reduction <= 0:
                            break
                
        return storage_per_period
        
    # Add constraints
    # Constraint: Each flight must be assigned to exactly one carousel
    for flight_id in flights_ids:
        model.addConstr(gp.quicksum(x[flight_id, carousel_id] for carousel_id in carousels_ids) == 1,
                       name=f"assign_flight_{flight_id}")

    # More constraints are needed for the full implementation...
    # This is simplified due to complexity; here's the outline
    
    # For now, we'll implement a more direct version focusing on core assignments
    # because complete mathematical formulation would be extremely large
    
    # Alternative approach: Use heuristic methods directly
    # We'll implement a more compact ILP after understanding essential relationships
    
    model.update()
    
    # Since creating a full Gurobi model may be intractable due to number of variables,
    # Let's implement a greedy heuristic instead:
    pass

class SimpleHeuristicSolver:
    def __init__(self, instance_data):
        self.instance_data = instance_data
        self.flights = {f["flight_id"]: f for f in instance_data["flights"]}
        self.carousels = {c["carousel_id"]: c for c in instance_data["carousels"]}
        self.num_periods = instance_data["planning_horizon"]["num_periods"]
        self.target_utilization = instance_data["objective"]["target_utilization"]
        self.threshold_values = instance_data["objective"]["threshold_values"]
        self.penalties = instance_data["objective"]["penalties"]
        self.peak_intervals = instance_data["peak_intervals"]
        
        # Precompute workload contributions for a given assignment
        self.workloads = {}  # Will compute as needed
        
    def calculate_workload(self, flight_id, carousel_id, handling_start, ws_count):
        """Calculate workload of flight on carousel at each time period"""
        flight_info = self.flights[flight_id]
        carousel_info = self.carousels[carousel_id]
        workload = [0] * self.num_periods
        total_bags = sum(flight_info["arrival_profile"])
        
        # For simplicity, assume uniform load over the handling period
        handling_duration = len(flight_info["arrival_profile"])
        for i in range(handling_duration):
            period = handling_start + i
            if period < self.num_periods:
                # Approximate workload - in detail this would track bag movement
                # For first pass, just put arrival bags on the belt and keep them there during handling
                workload[period] = min(flight_info["arrival_profile"][i], carousel_info["Kcb"])
                
        # Actually, need more accurate tracking: bags arrive and get loaded gradually
        # Simpler model: spread arrival over time using working rate
        total_loading_capacity = ws_count * self.instance_data["working_rate_bags_per_period"]
        remaining_to_handle = total_bags
        
        for i in range(handling_duration):
            period = handling_start + i
            if period >= self.num_periods:
                continue
            arrival_now = flight_info["arrival_profile"][i]
            can_process = min(total_loading_capacity, remaining_to_handle)
            # On-belt workload is bags that haven't been processed yet + newly arrived
            belt_load = max(remaining_to_handle - can_process, 0) + arrival_now
            workload[period] = min(belt_load, carousel_info["Kcb"])
            remaining_to_handle -= can_process
            if remaining_to_handle <= 0:
                break
                
        return workload

    def solve(self):
        # Greedy assignment approach
        assignments = []
        
        # Sort flights by urgency (earliest end period first)
        sorted_flights = sorted(self.flights.items(), key=lambda x: x[1]["end_period"])
        
        # Initialize current utilizations per carousel type
        carousel_type_usage = {str(ty): {p: [] for p in range(self.num_periods)} 
                               for ty in self.instance_data["carousel_types"].keys()}
        
        # Track assignment details in a simple form
        schedule = {}
        
        for flight_id, fdata in sorted_flights:
            # Try different carousels and start times to find best fit
            best_score = float('inf')
            best_assignment = None
            
            for carousel_id, cdata in self.carousels.items():
                # Check if carousel type supports flight
                carousel_type = str(cdata["type_r"])
                ws_bounds = fdata["working_station_bounds_by_carousel_type"].get(carousel_type, {})
                
                if not ws_bounds:
                    continue  # This type carousel not compatible
                
                Wmin = ws_bounds["Wmin"]
                Wmax = ws_bounds["Wmax"]
                
                # Try different start times
                for start_time in range(fdata["earliest_start_period"], fdata["latest_start_period"] + 1):
                    # Try different working station counts
                    for ws_count in range(Wmin, min(Wmax + 1, cdata["Kws"] + 1)):
                        # Calculate workload
                        workload = self.calculate_workload(flight_id, carousel_id, start_time, ws_count)
                        
                        # Calculate max utilization during peak intervals if assigned
                        violation_penalty = 0
                        
                        for peak in self.peak_intervals:
                            peak_max_util = 0
                            for p in range(peak["start_period"], min(peak["end_period"] + 1, self.num_periods)):
                                if p >= len(workload):
                                    continue
                                total_wl = sum(assigned_data["workload"][p] for fid, assigned_data in schedule.items() 
                                              if carousel_id in [assign["carousel_id"] for assign in assignments if assign["flight_id"]==fid])
                                # Add this flight's workload temporarily
                                total_wl += workload[p]  
                                belt_cap = cdata["Kcb"] 
                                
                                if belt_cap > 0:
                                    ut = total_wl / belt_cap
                                    peak_max_util = max(peak_max_util, ut)
                            
                            # Calculate violation penalties for this peak
                            if peak_max_util > self.target_utilization:
                                excess = peak_max_util - self.target_utilization
                                for thr_idx, thresh_val in enumerate(self.threshold_values):
                                    if excess > thresh_val:
                                        violation_penalty += self.penalties[thr_idx]
                        
                        # Simple scoring - penalize violations
                        score = violation_penalty
                        
                        if score < best_score:
                            best_score = score
                            best_assignment = {
                                "flight_id": flight_id,
                                "carousel_id": carousel_id,
                                "working_stations": ws_count,
                                "handling_start_period": start_time,
                                "depletion_start_period": max(0, start_time - 5),  # Simplified
                                "end_period": fdata["end_period"]
                            }
                    
                    if best_assignment is not None:  # Found valid assignment for this carousel
                        break
                if best_assignment is not None:  # Found valid assignment for this flight
                    break
            
            if best_assignment:
                assignments.append(best_assignment)
                schedule[flight_id] = {
                    "workload": self.calculate_workload(
                        best_assignment["flight_id"],
                        best_assignment["carousel_id"],
                        best_assignment["handling_start_period"],
                        best_assignment["working_stations"]),
                    "assigned": True
                }
        
        # Now compute final penalties
        total_penalty = 0
        violations = []
        
        for carousel_id in self.carousels:
            cdata = self.carousels[carousel_id] 
            carousel_type = str(cdata["type_r"])
            
            # Aggregate workloads across all flights assigned to this carousel
            carousel_workloads = [0] * self.num_periods
            for assign in assignments:
                if assign["carousel_id"] == carousel_id:
                    workload = self.calculate_workload(assign["flight_id"], assign["carousel_id"],
                                                    assign["handling_start_period"], assign["working_stations"])
                    for period in range(len(carousel_workloads)):
                        if period < len(workload):
                            carousel_workloads[period] += workload[period]
            
            for peak in self.peak_intervals:
                peak_max_util = 0
                violation_found = False
                for p in range(peak["start_period"], min(peak["end_period"] + 1, self.num_periods)):
                    belt_load = carousel_workloads[p]
                    cap = cdata["Kcb"]
                    if cap > 0:
                        util = belt_load / cap
                        peak_max_util = max(peak_max_util, util)
                
                # Track threshold violations
                if peak_max_util > self.target_utilization:
                    excess = peak_max_util - self.target_utilization
                    
                    for thr_idx, thresh_val in enumerate(self.threshold_values):
                        if excess > thresh_val:
                            total_penalty += self.penalties[thr_idx]
                            violations.append({
                                "carousel_id": carousel_id,
                                "peak_interval": peak["peak_id"],
                                "threshold_level": thr_idx,
                                "threshold_value": self.threshold_values[thr_idx]
                            })
                            # Only count the highest violated threshold per carousel/peak
                            break
        
        return {
            "objective_value": total_penalty,
            "assignments": assignments,
            "utilization_violations": violations
        }

def main():
    parser = argparse.ArgumentParser(description='Solve airport baggage handling problem.')
    parser.add_argument('--instance_path', type=str, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
        
    # Initialize logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    # Solve the problem
    solver = SimpleHeuristicSolver(instance_data)
    
    solution = solver.solve()

    # Log objective if using logger
    if logger:
        logger.log(solution["objective_value"])

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()