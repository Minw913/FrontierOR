import argparse
import json
import math
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_berth_allocation(instance_data, time_limit):
    # Extract instance data
    num_vessels = instance_data["num_vessels"]
    num_berths = instance_data["num_berths"]
    num_QCs = instance_data["num_QCs"]
    num_subblocks = instance_data["num_subblocks"]
    H = instance_data["H"]
    E = instance_data["E"]
    penalty_cost_coefficient_co = instance_data["penalty_cost_coefficient_co"]
    QC_capacity_per_timestep = instance_data["QC_capacity_per_timestep"]
    loading_route_lengths_D_L = instance_data["loading_route_lengths_D_L"]
    average_unloading_route_lengths_D_U = instance_data["average_unloading_route_lengths_D_U"]
    vessels_info = instance_data["vessels"]

    # Compute max handling time across all vessels and profiles
    max_handling_time = max(
        profile["handling_time"] 
        for v in vessels_info 
        for profile in v["qc_profiles"]
    )

    # Create model
    model = gp.Model("BerthAllocation")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    # Decision variables
    x = {}  # x[v][b] = 1 if vessel v is assigned to berth b
    y = {}  # y[v][p] = 1 if vessel v selects profile p
    z = {}  # z[v][t] = 1 if vessel v starts handling at time t
    w = {}  # w[v][s] = 1 if vessel v reserves subblock s

    for v_idx in range(num_vessels):
        v_data = vessels_info[v_idx]
        feasible_start = v_data["feasible_time_window"][0]
        feasible_end = v_data["feasible_time_window"][1]

        # Berth assignment
        for b in range(num_berths):
            x[v_idx, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{v_idx}_{b}")

        # Profile selection
        for p_idx in range(len(v_data["qc_profiles"])):
            y[v_idx, p_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{v_idx}_{p_idx}")

        # Start time assignment
        for t in range(feasible_start, feasible_end + 1):
            z[v_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{v_idx}_{t}")

        # Subblock reservation
        for s in range(num_subblocks):
            w[v_idx, s] = model.addVar(vtype=GRB.BINARY, name=f"w_{v_idx}_{s}")

    # Start time variables (continuous representation)
    alpha = {}  # Start time for vessel v
    beta = {}   # End time for vessel v
    for v_idx in range(num_vessels):
        alpha[v_idx] = model.addVar(vtype=GRB.INTEGER, lb=vessels_info[v_idx]["feasible_time_window"][0],
                                    ub=vessels_info[v_idx]["feasible_time_window"][1], name=f"alpha_{v_idx}")
        beta[v_idx] = model.addVar(vtype=GRB.INTEGER, name=f"beta_{v_idx}")

    # Big-M constant
    M = H + E + 100  # Sufficiently large number

    # Constraints
    # Each vessel is assigned exactly one berth
    for v_idx in range(num_vessels):
        model.addConstr(gp.quicksum(x[v_idx, b] for b in range(num_berths)) == 1)

    # Each vessel selects exactly one profile
    for v_idx in range(num_vessels):
        model.addConstr(gp.quicksum(y[v_idx, p_idx] for p_idx in range(len(vessels_info[v_idx]["qc_profiles"]))) == 1)

    # Each vessel starts at exactly one time step
    for v_idx in range(num_vessels):
        feasible_start = vessels_info[v_idx]["feasible_time_window"][0]
        feasible_end = vessels_info[v_idx]["feasible_time_window"][1]
        model.addConstr(gp.quicksum(z[v_idx, t] for t in range(feasible_start, feasible_end + 1)) == 1)

    # Each subblock is reserved by at most one vessel
    for s in range(num_subblocks):
        model.addConstr(gp.quicksum(w[v_idx, s] for v_idx in range(num_vessels)) <= 1)

    # Each vessel reserves exactly the required number of subblocks
    for v_idx in range(num_vessels):
        model.addConstr(gp.quicksum(w[v_idx, s] for s in range(num_subblocks)) == vessels_info[v_idx]["reserved_subblocks"])

    # Link alpha to z variables
    for v_idx in range(num_vessels):
        feasible_start = vessels_info[v_idx]["feasible_time_window"][0]
        feasible_end = vessels_info[v_idx]["feasible_time_window"][1]
        model.addConstr(alpha[v_idx] == gp.quicksum(t * z[v_idx, t] for t in range(feasible_start, feasible_end + 1)))

    # Define beta based on alpha and handling time
    for v_idx in range(num_vessels):
        expr = alpha[v_idx]
        for p_idx, profile in enumerate(vessels_info[v_idx]["qc_profiles"]):
            expr += (profile["handling_time"] - 1) * y[v_idx, p_idx]
        model.addConstr(beta[v_idx] == expr)

    # No overlap on the same berth
    for v1_idx in range(num_vessels):
        for v2_idx in range(v1_idx + 1, num_vessels):
            for b in range(num_berths):
                # If both vessels are assigned to the same berth, they cannot overlap
                # Either v1 finishes before v2 starts, or v2 finishes before v1 starts
                # Using big-M constraints
                
                # Introduce binary variable to indicate order
                o12 = model.addVar(vtype=GRB.BINARY, name=f"o12_{v1_idx}_{v2_idx}_{b}")  # 1 if v1 before v2
                o21 = 1 - o12  # 1 if v2 before v1
                
                # If both assigned to same berth, then either v1 finishes before v2 starts or vice versa
                model.addGenConstrIndicator(x[v1_idx, b], True, 
                    x[v2_idx, b], True,
                    (beta[v1_idx] + 1 <= alpha[v2_idx]) | (beta[v2_idx] + 1 <= alpha[v1_idx]))

    # QC capacity constraints with periodicity
    for t in range(1, H + E + 1):
        qc_usage_expr = gp.LinExpr()
        for v_idx in range(num_vessels):
            for p_idx, profile in enumerate(vessels_info[v_idx]["qc_profiles"]):
                profile_handling_time = profile["handling_time"]
                profile_qc_usage = profile["qc_per_timestep"]
                
                # Sum over all possible start times that would affect time t
                for start_time in range(max(1, t - profile_handling_time + 1), min(t + 1, H + E + 2)):
                    if (start_time >= vessels_info[v_idx]["feasible_time_window"][0] and 
                        start_time <= vessels_info[v_idx]["feasible_time_window"][1]):
                        
                        # Check if this vessel is active at time t when starting at start_time
                        if start_time <= t < start_time + profile_handling_time:
                            offset_in_profile = t - start_time
                            if offset_in_profile < len(profile_qc_usage):
                                qc_at_time_t = profile_qc_usage[offset_in_profile]
                                qc_usage_expr.addTerms(qc_at_time_t, z[v_idx, start_time] * y[v_idx, p_idx])
        
        # Apply periodic constraint
        if t > E and t <= H:
            # Regular time step: just check current week
            model.addConstr(qc_usage_expr <= QC_capacity_per_timestep[t - 1])
        elif t <= E:
            # Early time step: check current week + week after (periodic)
            qc_usage_next_week_expr = gp.LinExpr()
            for v_idx in range(num_vessels):
                for p_idx, profile in enumerate(vessels_info[v_idx]["qc_profiles"]):
                    profile_handling_time = profile["handling_time"]
                    profile_qc_usage = profile["qc_per_timestep"]
                    
                    # Time t corresponds to time t+H in next week
                    next_week_t = t + H
                    for start_time in range(max(1, next_week_t - profile_handling_time + 1), min(next_week_t + 1, H + E + 2)):
                        if (start_time >= vessels_info[v_idx]["feasible_time_window"][0] and 
                            start_time <= vessels_info[v_idx]["feasible_time_window"][1]):
                            
                            # Check if this vessel is active at time next_week_t when starting at start_time
                            if start_time <= next_week_t < start_time + profile_handling_time:
                                offset_in_profile = next_week_t - start_time
                                if offset_in_profile < len(profile_qc_usage):
                                    qc_at_time_next_week = profile_qc_usage[offset_in_profile]
                                    qc_usage_next_week_expr.addTerms(qc_at_time_next_week, z[v_idx, start_time - H] * y[v_idx, p_idx])
            
            model.addConstr(qc_usage_expr + qc_usage_next_week_expr <= QC_capacity_per_timestep[t - 1])

    # Berth periodicity constraint
    for b in range(num_berths):
        # For each berth, compute earliest start and latest end among assigned vessels
        # rho_b = min of alpha[v] for v assigned to b
        # varsigma_b = max of beta[v] for v assigned to b
        # Constraint: varsigma_b - rho_b <= H - 1
        
        # We'll use indicator constraints to enforce this
        rho_b = model.addVar(vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"rho_{b}")
        varsigma_b = model.addVar(vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"varsigma_{b}")
        
        # rho_b should be the minimum start time of vessels assigned to b
        for v_idx in range(num_vessels):
            # If vessel v is assigned to b, then rho_b <= alpha[v]
            model.addGenConstrIndicator(x[v_idx, b], True, rho_b <= alpha[v_idx])
            
            # Also ensure that if there is a vessel on this berth, we properly capture min
            # Add big-M constraints to enforce that if v is assigned to b, alpha[v] >= rho_b
            model.addGenConstrIndicator(x[v_idx, b], True, alpha[v_idx] >= rho_b)
        
        # For max calculation
        for v_idx in range(num_vessels):
            # If vessel v is assigned to b, then varsigma_b >= beta[v]
            model.addGenConstrIndicator(x[v_idx, b], True, varsigma_b >= beta[v_idx])
            
            # Also ensure that if v is assigned to b, beta[v] <= varsigma_b
            model.addGenConstrIndicator(x[v_idx, b], True, beta[v_idx] <= varsigma_b)
        
        # Periodicity constraint
        model.addConstr(varsigma_b - rho_b <= H - 1)

    # Objective function
    total_cost = gp.LinExpr()

    # Penalty costs for time deviations
    for v_idx in range(num_vessels):
        v_data = vessels_info[v_idx]
        exp_start = v_data["expected_time_window"][0]
        exp_end = v_data["expected_time_window"][1]
        
        # Deviation from expected start time (if early)
        early_dev = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"early_dev_{v_idx}")
        model.addConstr(early_dev >= exp_start - alpha[v_idx])
        
        # Deviation from expected end time (if late)
        late_dev = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"late_dev_{v_idx}")
        model.addConstr(late_dev >= beta[v_idx] - exp_end)
        
        total_cost.addTerms(v_data["penalty_cost_coefficient"], early_dev + late_dev)

    # Loading transportation costs
    for v_idx in range(num_vessels):
        v_data = vessels_info[v_idx]
        loading_cost_expr = gp.LinExpr()
        
        for s in range(num_subblocks):
            for b in range(num_berths):
                # If vessel v reserves subblock s and is assigned to berth b
                loading_cost_expr.addTerms(
                    penalty_cost_coefficient_co * loading_route_lengths_D_L[s][b] * 
                    (v_data["loading_containers"] / v_data["reserved_subblocks"]),
                    w[v_idx, s] * x[v_idx, b]
                )
        
        total_cost.add(loading_cost_expr)

    # Unloading transportation costs
    for v_idx in range(num_vessels):
        v_data = vessels_info[v_idx]
        unloading_cost_expr = gp.LinExpr()
        
        for b in range(num_berths):
            # If vessel v is assigned to berth b
            unloading_cost_expr.addTerms(
                penalty_cost_coefficient_co * average_unloading_route_lengths_D_U[b] * 
                v_data["unloading_containers"],
                x[v_idx, b]
            )
        
        total_cost.add(unloading_cost_expr)

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Prepare solution
    solution = {}
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        solution["objective_value"] = model.objVal if model.solCount > 0 else float('inf')
        
        vessel_solutions = []
        for v_idx in range(num_vessels):
            # Find assigned berth
            assigned_berth = -1
            for b in range(num_berths):
                if x[v_idx, b].x > 0.5:
                    assigned_berth = b
                    break
            
            # Find selected profile
            selected_profile_idx = -1
            for p_idx in range(len(vessels_info[v_idx]["qc_profiles"])):
                if y[v_idx, p_idx].x > 0.5:
                    selected_profile_idx = p_idx
                    break
            
            # Find start time
            start_time = -1
            feasible_start = vessels_info[v_idx]["feasible_time_window"][0]
            feasible_end = vessels_info[v_idx]["feasible_time_window"][1]
            for t in range(feasible_start, feasible_end + 1):
                if z[v_idx, t].x > 0.5:
                    start_time = t
                    break
            
            # Find reserved subblocks
            reserved_subblocks = []
            for s in range(num_subblocks):
                if w[v_idx, s].x > 0.5:
                    reserved_subblocks.append(s)
            
            profile_data = vessels_info[v_idx]["qc_profiles"][selected_profile_idx]
            end_time = start_time + profile_data["handling_time"] - 1
            
            vessel_sol = {
                "vessel_id": v_idx,
                "alpha": start_time,
                "beta": end_time,
                "berth": assigned_berth,
                "profile": selected_profile_idx,
                "subblocks": reserved_subblocks,
                "handling_time": profile_data["handling_time"],
                "qc_per_timestep": profile_data["qc_per_timestep"]
            }
            vessel_solutions.append(vessel_sol)
        
        solution["vessels"] = vessel_solutions
        
        # Calculate berth information
        berth_solutions = []
        for b in range(num_berths):
            assigned_vessels = [v_idx for v_idx in range(num_vessels) if x[v_idx, b].x > 0.5]
            if assigned_vessels:
                rho_b = min(vessel_solutions[v_idx]["alpha"] for v_idx in assigned_vessels)
                varsigma_b = max(vessel_solutions[v_idx]["beta"] for v_idx in assigned_vessels)
            else:
                rho_b = 0
                varsigma_b = 0
                
            berth_solutions.append({
                "berth": b,
                "rho": rho_b,
                "varsigma": varsigma_b
            })
        
        solution["berths"] = berth_solutions
        
        # Calculate QC usage
        qc_usage = {}
        for t in range(1, H + E + 1):
            total_qc_at_t = 0
            for v_idx in range(num_vessels):
                for p_idx, profile in enumerate(vessels_info[v_idx]["qc_profiles"]):
                    if y[v_idx, p_idx].x > 0.5:  # If this profile is selected
                        start_time = vessel_solutions[v_idx]["alpha"]
                        profile_handling_time = profile["handling_time"]
                        profile_qc_usage = profile["qc_per_timestep"]
                        
                        # Check if vessel is active at time t
                        if start_time <= t < start_time + profile_handling_time:
                            offset_in_profile = t - start_time
                            if offset_in_profile < len(profile_qc_usage):
                                total_qc_at_t += profile_qc_usage[offset_in_profile]
            
            if total_qc_at_t > 0:
                qc_usage[str(t)] = total_qc_at_t
        
        solution["qc_usage"] = qc_usage
    else:
        solution["objective_value"] = float('inf')
        solution["vessels"] = []
        solution["berths"] = []
        solution["qc_usage"] = {}

    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve berth allocation problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')

    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_berth_allocation(instance_data, args.time_limit)

    # Log the final solution if logger exists
    if logger and solution["objective_value"] != float('inf'):
        logger.log_solution(solution["objective_value"], solution)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()