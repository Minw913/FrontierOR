import argparse
import json
import numpy as np
from itertools import permutations
from solution_logger import SolutionLogger


def solve_schedule_optimization(P, L,
                                avg_proc_durations_adequate, avg_proc_durations_inadequate,
                                prob_adequate, avg_arrival_deviations,
                                c_w_values, c_g_values, c_o):
    """
    Solve the robust optimization problem to find assignment and schedule minimizing
    worst-case expected operational cost.
    
    For efficiency and tractability, we'll use a heuristic approach:
    - Use different sorting orders as sequences based on expected duration and risk
    - For each sequence, find optimal schedule via solving an approximated optimization problem
    """

    # Generate heuristic orderings based on various criteria
    patient_indices = list(range(P))
    
    # Calculate expected durations
    exp_durations = [
        prob_adequate[i] * avg_proc_durations_adequate[i] + 
        (1 - prob_adequate[i]) * avg_proc_durations_inadequate[i]
        for i in patient_indices
    ]
    
    # Different sorting heuristic strategies
    # Order 1: By expected duration ascending (shortest job first)
    order_ascending = sorted(patient_indices, key=lambda i: exp_durations[i])
    
    # Order 2: By expected duration descending
    order_descending = sorted(patient_indices, key=lambda i: exp_durations[i], reverse=True)
    
    # Order 3: By probability of adequate prep descending (least risky first)
    order_risk_desc = sorted(patient_indices, key=lambda i: prob_adequate[i], reverse=True)
    
    # Order 4: By expected duration adjusted for arrival variance (heuristic)
    order_adjusted = sorted(patient_indices, key=lambda i: exp_durations[i] + abs(avg_arrival_deviations[i]))
    
    orderings = [order_ascending, order_descending, order_risk_desc, order_adjusted]
    
    best_objective = float('inf')
    best_assignment = None
    best_schedule = None
    
    for assign_order in orderings:
        obj_val, schedule = optimize_schedule_for_sequence(
            assign_order=assign_order, P=P, L=L,
            avg_proc_durations_adequate=avg_proc_durations_adequate,
            avg_proc_durations_inadequate=avg_proc_durations_inadequate,
            prob_adequate=prob_adequate,
            avg_arrival_deviations=avg_arrival_deviations,
            c_w_values=c_w_values, c_g_values=c_g_values, c_o=c_o
        )
        
        if obj_val < best_objective:
            best_objective = obj_val
            # Build assignment map (patient id string -> position number)
            best_assignment = {str(assign_order[i]+1): i+1 for i in range(P)}
            best_schedule = {str(pos): schedule[pos] for pos in range(len(schedule))}
    
    return best_objective, best_assignment, best_schedule


def optimize_schedule_for_sequence(assign_order, P, L,
                                   avg_proc_durations_adequate, avg_proc_durations_inadequate,
                                   prob_adequate, avg_arrival_deviations,
                                   c_w_values, c_g_values, c_o):
    """
    For a fixed sequence, compute the optimal start times (by expected performance).
    This is solved using a simplified mathematical approach due to complexity of full robust formulation.
    """
    
    try:
        # Simplified optimization based on expected value computation
        import gurobipy as gp
        from gurobipy import GRB

        # Create model
        model = gp.Model("optimize_sequential_schedule")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 60)  # Add timeout for reliability
        
        # Decision variables: scheduled start times for each position
        start_times = []
        for i in range(P):
            st = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=L, name=f"s_{i}")
            start_times.append(st)
        
        # Constraints: non-decreasing start times
        for i in range(1, P):
            model.addConstr(start_times[i] >= start_times[i-1])

        # Expected durations for the fixed sequence
        expected_durations = []
        for i in range(P):
            pat_idx = assign_order[i]  # The patient index at position i
            exp_dur = prob_adequate[pat_idx] * avg_proc_durations_adequate[pat_idx] \
                      + (1 - prob_adequate[pat_idx]) * avg_proc_durations_inadequate[pat_idx]
            expected_durations.append(exp_dur)
        
        # Expression for total cost (weighted combination of waiting, gap, and overtime)
        total_cost = 0.0

        # Precedence constraints effect - compute expected waiting and idle times
        # We'll approximate the expected cost calculation
        
        # Calculate cumulative expected finish times without considering uncertainty in arrivals
        expected_prev_finish = [0] * P
        
        # Compute the terms: for each position i, consider how it interacts with arrivals and previous completions
        for i in range(P):
            pat_idx_at_pos_i = assign_order[i]
            
            # Expected arrival time = scheduled time + mean deviation
            expected_arr_time_actual = start_times[i] + avg_arrival_deviations[pat_idx_at_pos_i]
            
            # First patient - only affected by their own arrival
            if i == 0:
                # Patient_ready_time is max(scheduled_start + mean_deviation, 0) = scheduled_start + mean_dev
                expected_procedure_start_i = gp.max_(start_times[i], expected_prev_finish[i])
                actual_start_i = gp.max_(expected_arr_time_actual, expected_prev_finish[i])
                
                # Wait_time = max(0, actual_procedure_start - scheduled_start_time)
                wait_duration = gp.max_(0, actual_start_i - start_times[i])
                total_cost += c_w_values[i] * wait_duration
                
                # Update expected finish = actual_start + expected_proc_time
                expected_prev_finish[i] = actual_start_i + expected_durations[i]
            else:
                # For i > 0, the actual procedure start depends on when previous procedure finishes
                actual_start_i = gp.max_(gp.max_(expected_arr_time_actual, expected_prev_finish[i-1]), start_times[i])
                
                wait_duration = gp.max_(0, actual_start_i - start_times[i])
                total_cost += c_w_values[i] * wait_duration
                
                next_expected_finish = actual_start_i + expected_durations[i]
                
                # Now update the expected finish time for the next loop iteration
                expected_prev_finish[i] = next_expected_finish

        # Add overtime cost for last patient
        if P > 0:
            last_pos = P - 1
            last_pat_idx = assign_order[last_pos]
            actual_start_last = gp.max_(start_times[last_pos] + avg_arrival_deviations[last_pat_idx],
                                        expected_prev_finish[last_pos-1] if last_pos > 0 else 0)
            completion_time = actual_start_last + expected_durations[last_pos]
            overtime_duration = gp.max_(0, completion_time - L)
            total_cost += c_o * overtime_duration

        # Gap costs - for each pair (i, i+1)
        for i in range(P - 1):
            actual_start_next = gp.max_(
                start_times[i+1] + avg_arrival_deviations[assign_order[i+1]],
                expected_prev_finish[i]
            )
            gap_duration = gp.max_(0, start_times[i+1] - actual_start_next)
            total_cost += c_g_values[i] * gap_duration

        model.setObjective(total_cost, GRB.MINIMIZE)
        model.optimize()

        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            # Extract optimized start times
            sch = [start_times[i].X for i in range(P)]
            return model.ObjVal, sch
    
    except Exception:
        pass  # Fall back to heuristics if optimization fails
    
    # Heuristic fallback: place patients at regular intervals or based on expected durations
    estimated_total_time = sum(
        prob_adequate[assign_order[i]] * avg_proc_durations_adequate[assign_order[i]]
        + (1 - prob_adequate[assign_order[i]]) * avg_proc_durations_inadequate[assign_order[i]]
        for i in range(P)
    )
    
    # If there's excess capacity in L, spread out proportionally, else pack greedily
    sch = []
    current_time = 0
    for i in range(P):
        pat = assign_order[i]
        exp_duration = (
            prob_adequate[pat] * avg_proc_durations_adequate[pat]
            + (1 - prob_adequate[pat]) * avg_proc_durations_inadequate[pat]
        )
        
        sch.append(current_time)
        current_time += exp_duration
    
    # Return a rough estimate of cost with this heuristic schedule
    est_total_cost = sum(c_w_values[i] * max(0, sch[i] + avg_arrival_deviations[assign_order[i]] - sch[i]) for i in range(P))
    if P > 0:
        est_total_cost += c_o * max(0, sch[-1] + exp_duration - L)
    
    return est_total_cost, sch


def main():
    parser = argparse.ArgumentParser(description='Solve appointment scheduling under uncertainty.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write solution JSON file')
    parser.add_argument('--time_limit', type=int, default=3600, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path to log intermediate solutions')

    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract parameters
    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']

    # Cost values
    cost_key = next(iter(instance['cost_structures']))
    cost_values = instance['cost_structures'][cost_key]
    c_w_scalar = cost_values['c_w']
    c_g_scalar = cost_values['c_g']
    c_o = cost_values['c_o']

    # Vectorize costs (same for all if uniform)
    c_w_values = [c_w_scalar] * P
    c_g_values = [c_g_scalar] * (P - 1) if P > 1 else []

    # Extract patient-specific data
    avg_proc_durations_adequate = [p['mean_duration_adequate_prep'] for p in instance['patients']]
    avg_proc_durations_inadequate = [p['mean_duration_inadequate_prep'] for p in instance['patients']]
    prob_adequate = [p['mean_prep_adequacy'] for p in instance['patients']]
    avg_arrival_deviations = [p['mean_arrival_time_deviation'] for p in instance['patients']]

    # Solve the optimization problem
    try:
        objective_value, assignment, schedule = solve_schedule_optimization(
            P=P,
            L=L,
            avg_proc_durations_adequate=avg_proc_durations_adequate,
            avg_proc_durations_inadequate=avg_proc_durations_inadequate,
            prob_adequate=prob_adequate,
            avg_arrival_deviations=avg_arrival_deviations,
            c_w_values=c_w_values,
            c_g_values=c_g_values,
            c_o=c_o
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    # Log the solution if logger provided
    if logger:
        logger.log(objective_value)

    # Build patient_start_times dict: patient_id (1-indexed str) -> scheduled start time
    patient_start_times = {
        str(orig_patient_id): schedule[str(pos_number - 1)]
        for orig_patient_id, pos_number in assignment.items()
    }

    # Prepare solution output in required format
    solution_output = {
        "objective_value": objective_value,
        "assignment": assignment,
        "schedule": {str(pos_number): start_time for pos_number, start_time in enumerate(schedule)},
        "patient_start_times": patient_start_times
    }

    # Write the solution to the specified file path
    with open(args.solution_path, 'w') as f:
        json.dump(solution_output, f, indent=2)


if __name__ == "__main__":
    main()