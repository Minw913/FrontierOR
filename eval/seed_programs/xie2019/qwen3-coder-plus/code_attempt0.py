import argparse
import json
import math
import random
from itertools import product
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def generate_qc_profiles(vessel_duration_hours, time_steps_per_shift, qc_profile_config, is_large):
    """
    Generate all possible QC profiles for a vessel based on its duration and QC profile configuration.
    """
    min_handling_shifts = qc_profile_config['min_handling_shifts']
    max_handling_shifts = qc_profile_config['max_handling_shifts']
    min_qc = qc_profile_config['min_qc']
    max_qc = qc_profile_config['max_qc']

    # Calculate the number of time steps needed for the vessel based on its duration
    # We need to convert duration in hours to time steps
    # Since we don't know the exact conversion factor from hours to time steps,
    # we'll generate profiles based on shifts and then map to time steps
    
    profiles = {}
    
    # Generate all possible shift-level QC sequences
    for num_shifts in range(min_handling_shifts, max_handling_shifts + 1):
        # For each starting within-shift position (0 to time_steps_per_shift-1)
        for start_within_shift in range(time_steps_per_shift):
            # Generate possible QC assignments per shift
            # We'll generate a few sample profiles based on the constraints
            qc_sequences = []
            
            # Generate all combinations of QC assignments per shift within limits
            for qcs_in_shift in product(range(min_qc, max_qc + 1), repeat=num_shifts):
                qc_sequences.append(list(qcs_in_shift))
            
            # Convert these to time-step level profiles
            for i, qcs in enumerate(qc_sequences):
                # Create a profile that covers num_shifts * time_steps_per_shift time steps
                profile = [0] * (num_shifts * time_steps_per_shift)
                for shift_idx, qc_count in enumerate(qcs):
                    for ts_idx in range(time_steps_per_shift):
                        profile[shift_idx * time_steps_per_shift + ts_idx] = qc_count
                
                # Group profiles by their starting within-shift index
                if start_within_shift not in profiles:
                    profiles[start_within_shift] = []
                
                # Only add the profile if it makes sense for the vessel's duration
                # This is a simplified version - in practice, we would match more closely to vessel needs
                profiles[start_within_shift].append({
                    'duration': len(profile),
                    'qc_sequence': profile
                })
    
    # Limit the number of profiles based on big/small config
    max_profiles = qc_profile_config.get('max_profiles', float('inf'))
    for start_pos in profiles:
        if len(profiles[start_pos]) > max_profiles:
            profiles[start_pos] = profiles[start_pos][:max_profiles]
    
    return profiles


def get_available_profiles_for_vessel(vessel, all_profiles, start_time_step):
    """
    Get the list of available QC profiles for a vessel given a start time step.
    """
    start_within_shift = start_time_step % vessel['time_steps_per_shift']
    if start_within_shift in all_profiles:
        return all_profiles[start_within_shift]
    else:
        return []


def solve_berth_allocation(instance_data, time_limit):
    """
    Solve the berth allocation problem using Gurobi.
    """
    num_vessels = instance_data['num_vessels']
    num_berths = instance_data['num_berths']
    total_qc = instance_data['total_qc']
    time_steps_per_shift = instance_data['time_steps_per_shift']
    total_time_steps = instance_data['total_time_steps']
    c1 = instance_data['c1']
    c2 = instance_data['c2']
    
    vessels = instance_data['vessels']
    berths = instance_data['berths']
    
    # Determine which vessels are large vs small based on duration
    # This is somewhat arbitrary since we don't have explicit size indicators
    avg_duration = sum(v['duration_hours'] for v in vessels) / len(vessels) if vessels else 0
    large_vessels = {i for i, v in enumerate(vessels) if v['duration_hours'] >= avg_duration}
    small_vessels = {i for i, v in enumerate(vessels)} - large_vessels
    
    # Generate QC profiles for each vessel
    all_vessel_profiles = {}
    for i, vessel in enumerate(vessels):
        if i in large_vessels:
            profiles = generate_qc_profiles(
                vessel['duration_hours'],
                time_steps_per_shift,
                instance_data['qc_profiles']['big'],
                True
            )
        else:
            profiles = generate_qc_profiles(
                vessel['duration_hours'],
                time_steps_per_shift,
                instance_data['qc_profiles']['small'],
                False
            )
        
        all_vessel_profiles[i] = profiles
    
    # Create model
    model = gp.Model("BerthAllocation")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # x[i][j][t] = 1 if vessel i is assigned to berth j and starts at time t
    x = {}
    for i in range(num_vessels):
        x[i] = {}
        for j in range(num_berths):
            x[i][j] = {}
            # Vessel can only start within its feasible window and the berth's availability window
            start_earliest = max(vessels[i]['a'], berths[j]['a_k'])
            start_latest = min(vessels[i]['b'], berths[j]['b_k'])
            
            for t in range(start_earliest, start_latest + 1):
                # Get available profiles for this vessel starting at time t
                available_profiles = get_available_profiles_for_vessel(
                    vessels[i], 
                    all_vessel_profiles[i], 
                    t
                )
                
                for p_idx, profile in enumerate(available_profiles):
                    duration = profile['duration']
                    end_t = t + duration - 1
                    
                    # Check if the service interval fits within berth availability
                    if end_t <= berths[j]['b_k']:
                        x[i][j][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}_{p_idx}")
    
    # Objective function: minimize total weighted deviation cost
    obj_expr = gp.LinExpr()
    
    for i in range(num_vessels):
        for j in range(num_berths):
            start_earliest = max(vessels[i]['a'], berths[j]['a_k'])
            start_latest = min(vessels[i]['b'], berths[j]['b_k'])
            
            for t in range(start_earliest, start_latest + 1):
                available_profiles = get_available_profiles_for_vessel(
                    vessels[i], 
                    all_vessel_profiles[i], 
                    t
                )
                
                for p_idx, profile in enumerate(available_profiles):
                    duration = profile['duration']
                    end_t = t + duration - 1
                    
                    if end_t <= berths[j]['b_k']:
                        # Add deviation costs to objective
                        berth_dev_cost = c1 * abs(j - vessels[i]['k_bar'])
                        time_dev_cost = c2 * abs(t - vessels[i]['t_bar'])
                        
                        obj_expr += (berth_dev_cost + time_dev_cost) * x[i][j][t]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Each vessel is assigned exactly one berth and one start time
    for i in range(num_vessels):
        model.addConstr(gp.quicksum(
            x[i][j][t] 
            for j in range(num_berths) 
            for t in range(max(vessels[i]['a'], berths[j]['a_k']), 
                          min(vessels[i]['b'], berths[j]['b_k']) + 1)
            if any(get_available_profiles_for_vessel(vessels[i], all_vessel_profiles[i], t))
        ) == 1, f"vessel_assignment_{i}")
    
    # 2. Berth capacity: no two vessels can occupy the same berth at the same time
    for j in range(num_berths):
        for t_global in range(total_time_steps):
            overlapping_assignments = []
            
            for i in range(num_vessels):
                start_earliest = max(vessels[i]['a'], berths[j]['a_k'])
                start_latest = min(vessels[i]['b'], berths[j]['b_k'])
                
                for t_start in range(start_earliest, start_latest + 1):
                    available_profiles = get_available_profiles_for_vessel(
                        vessels[i], 
                        all_vessel_profiles[i], 
                        t_start
                    )
                    
                    for p_idx, profile in enumerate(available_profiles):
                        duration = profile['duration']
                        end_t = t_start + duration - 1
                        
                        if end_t <= berths[j]['b_k'] and t_start <= t_global <= end_t:
                            overlapping_assignments.append(x[i][j][t_start])
            
            if overlapping_assignments:
                model.addConstr(gp.quicksum(overlapping_assignments) <= 1, f"berth_capacity_{j}_{t_global}")
    
    # 3. QC capacity: total QCs used at each time step cannot exceed available QCs
    for t_global in range(total_time_steps):
        qc_usage = gp.LinExpr()
        
        for i in range(num_vessels):
            for j in range(num_berths):
                start_earliest = max(vessels[i]['a'], berths[j]['a_k'])
                start_latest = min(vessels[i]['b'], berths[j]['b_k'])
                
                for t_start in range(start_earliest, start_latest + 1):
                    available_profiles = get_available_profiles_for_vessel(
                        vessels[i], 
                        all_vessel_profiles[i], 
                        t_start
                    )
                    
                    for p_idx, profile in enumerate(available_profiles):
                        duration = profile['duration']
                        end_t = t_start + duration - 1
                        
                        if end_t <= berths[j]['b_k'] and t_start <= t_global <= end_t:
                            # Add QC usage for this vessel at this time step
                            rel_time = t_global - t_start
                            if rel_time < len(profile['qc_sequence']):
                                qc_usage += profile['qc_sequence'][rel_time] * x[i][j][t_start]
        
        model.addConstr(qc_usage <= total_qc, f"qc_capacity_{t_global}")
    
    # Optimize the model
    model.optimize()
    
    # Extract solution
    solution = {"objective_value": None, "vessels": []}
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution["objective_value"] = model.objVal
        
        # Initialize vessel solutions
        for i in range(num_vessels):
            solution["vessels"].append({"id": i, "berth": None, "start_time": None, "end_time": None, "profile": None})
        
        # Find the assigned values
        for i in range(num_vessels):
            for j in range(num_berths):
                start_earliest = max(vessels[i]['a'], berths[j]['a_k'])
                start_latest = min(vessels[i]['b'], berths[j]['b_k'])
                
                for t in range(start_earliest, start_latest + 1):
                    available_profiles = get_available_profiles_for_vessel(
                        vessels[i], 
                        all_vessel_profiles[i], 
                        t
                    )
                    
                    for p_idx, profile in enumerate(available_profiles):
                        duration = profile['duration']
                        end_t = t + duration - 1
                        
                        if end_t <= berths[j]['b_k'] and x[i][j][t].x > 0.9:
                            solution["vessels"][i]["berth"] = j
                            solution["vessels"][i]["start_time"] = t
                            solution["vessels"][i]["end_time"] = end_t
                            solution["vessels"][i]["profile"] = p_idx
                            break
                    if solution["vessels"][i]["start_time"] is not None:
                        break
                if solution["vessels"][i]["start_time"] is not None:
                    break
    
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve berth allocation problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_berth_allocation(instance_data, args.time_limit)
    
    # Log the final solution if logger is available
    if logger and solution["objective_value"] is not None:
        logger.log_solution(solution["objective_value"], solution)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()