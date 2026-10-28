import argparse
import json
import math
import random
from itertools import product
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def generate_qc_profiles_for_vessel(vessel_id, vessel_data, time_steps_per_shift, qc_profile_config, is_large, total_time_steps):
    """
    Generate QC profiles for a specific vessel based on its requirements.
    """
    # Determine if vessel is large or small based on duration
    min_handling_shifts = qc_profile_config['min_handling_shifts']
    max_handling_shifts = qc_profile_config['max_handling_shifts']
    min_qc = qc_profile_config['min_qc']
    max_qc = qc_profile_config['max_qc']
    
    # Calculate how many time steps the vessel needs based on its duration
    # For now, we'll generate all possible combinations within the bounds
    profiles = {}  # Grouped by starting within-shift position
    
    for start_within_shift in range(time_steps_per_shift):
        profiles[start_within_shift] = []
        
        # Generate profiles for different durations
        for num_shifts in range(min_handling_shifts, max_handling_shifts + 1):
            if num_shifts * time_steps_per_shift > total_time_steps:
                continue
                
            # Generate all possible QC assignments for this duration
            for qcs_in_shifts in product(range(min_qc, max_qc + 1), repeat=num_shifts):
                # Create a profile with the sequence of QCs over time steps
                qc_sequence = []
                for qc_count in qcs_in_shifts:
                    for _ in range(time_steps_per_shift):
                        qc_sequence.append(qc_count)
                
                # Add this profile
                profile_info = {
                    'duration': len(qc_sequence),
                    'qc_sequence': qc_sequence[:]
                }
                profiles[start_within_shift].append(profile_info)
    
    return profiles


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
    avg_duration = sum(v['duration_hours'] for v in vessels) / len(vessels) if vessels else 0
    large_vessels = {i for i, v in enumerate(vessels) if v['duration_hours'] >= avg_duration}
    small_vessels = {i for i, v in enumerate(vessels)} - large_vessels
    
    # Generate QC profiles for each vessel
    all_vessel_profiles = {}
    for i, vessel in enumerate(vessels):
        if i in large_vessels:
            profiles = generate_qc_profiles_for_vessel(
                i, vessel, time_steps_per_shift,
                instance_data['qc_profiles']['big'], True, total_time_steps
            )
        else:
            profiles = generate_qc_profiles_for_vessel(
                i, vessel, time_steps_per_shift,
                instance_data['qc_profiles']['small'], False, total_time_steps
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
    # x[i,j,t,p] = 1 if vessel i is assigned to berth j, starts at time t, and uses profile p
    x = {}
    valid_assignments = []
    
    for i in range(num_vessels):
        for j in range(num_berths):
            # Vessel can only start within its feasible window and the berth's availability window
            start_earliest = max(vessels[i]['a'], berths[j]['a_k'])
            start_latest = min(vessels[i]['b'], berths[j]['b_k'])
            
            for t in range(start_earliest, start_latest + 1):
                # Determine which profile set to use based on the starting within-shift position
                start_within_shift = t % time_steps_per_shift
                if start_within_shift in all_vessel_profiles[i]:
                    available_profiles = all_vessel_profiles[i][start_within_shift]
                else:
                    available_profiles = []
                
                for p_idx, profile in enumerate(available_profiles):
                    duration = profile['duration']
                    end_t = t + duration - 1
                    
                    # Check if the service interval fits within berth availability
                    if end_t <= berths[j]['b_k']:
                        var_name = f"x_{i}_{j}_{t}_{p_idx}"
                        x[(i, j, t, p_idx)] = model.addVar(vtype=GRB.BINARY, name=var_name)
                        valid_assignments.append((i, j, t, p_idx, duration))
    
    # Objective function: minimize total weighted deviation cost
    obj_expr = gp.LinExpr()
    
    for i, j, t, p_idx, duration in valid_assignments:
        # Add deviation costs to objective
        berth_dev_cost = c1 * abs(j - vessels[i]['k_bar'])
        time_dev_cost = c2 * abs(t - vessels[i]['t_bar'])
        
        obj_expr += (berth_dev_cost + time_dev_cost) * x[(i, j, t, p_idx)]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Each vessel is assigned exactly one berth, start time, and profile
    for i in range(num_vessels):
        model.addConstr(
            gp.quicksum(x[(i, j, t, p_idx)] for i2, j, t, p_idx, duration in valid_assignments if i2 == i) == 1,
            f"vessel_assignment_{i}"
        )
    
    # 2. Berth capacity: no two vessels can occupy the same berth at the same time
    for j in range(num_berths):
        for t_global in range(total_time_steps):
            overlapping_vars = []
            
            for i, j_assign, t_start, p_idx, duration in valid_assignments:
                if j_assign == j:
                    end_t = t_start + duration - 1
                    if t_start <= t_global <= end_t:
                        overlapping_vars.append(x[(i, j_assign, t_start, p_idx)])
            
            if len(overlapping_vars) > 1:
                model.addConstr(gp.quicksum(overlapping_vars) <= 1, f"berth_capacity_{j}_{t_global}")
    
    # 3. QC capacity: total QCs used at each time step cannot exceed available QCs
    for t_global in range(total_time_steps):
        qc_usage = gp.LinExpr()
        
        for i, j_assign, t_start, p_idx, duration in valid_assignments:
            if t_start <= t_global <= (t_start + duration - 1):
                # Find the profile info to get QC usage at this time step
                start_within_shift = t_start % time_steps_per_shift
                if start_within_shift in all_vessel_profiles[i]:
                    profile_list = all_vessel_profiles[i][start_within_shift]
                    if p_idx < len(profile_list):
                        profile = profile_list[p_idx]
                        rel_time = t_global - t_start
                        if rel_time < len(profile['qc_sequence']):
                            qc_usage += profile['qc_sequence'][rel_time] * x[(i, j_assign, t_start, p_idx)]
        
        model.addConstr(qc_usage <= total_qc, f"qc_capacity_{t_global}")
    
    # Optimize the model
    model.optimize()
    
    # Extract solution
    solution = {"objective_value": None, "vessels": []}
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            solution["objective_value"] = model.objVal
            
            # Initialize vessel solutions
            for i in range(num_vessels):
                solution["vessels"].append({"id": i, "berth": None, "start_time": None, "end_time": None, "profile": None})
            
            # Find the assigned values
            for i, j, t, p_idx, duration in valid_assignments:
                if x[(i, j, t, p_idx)].x > 0.9:
                    solution["vessels"][i]["berth"] = j
                    solution["vessels"][i]["start_time"] = t
                    solution["vessels"][i]["end_time"] = t + duration - 1
                    solution["vessels"][i]["profile"] = p_idx
    
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