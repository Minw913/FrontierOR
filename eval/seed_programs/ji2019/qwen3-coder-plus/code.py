import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def solve_shipping_problem(instance_data, time_limit):
    # Parse input data
    num_ships = instance_data["num_ships"]
    num_berths = instance_data["num_berths"]
    num_chambers = instance_data["num_chambers"]
    W = instance_data["chamber_width_W"]  # Chamber width
    L = instance_data["chamber_length_L"]  # Chamber length
    Du = instance_data["lockage_duration_Du"]  # Lockage duration
    MT = instance_data["min_time_between_lockages_MT"]  # Min time between lockages
    sc = instance_data["lock_start_availability_sc"]  # Chamber start availability
    berth_end_time = instance_data["berth_end_availability_ek"]  # Berth end availability
    C_max = instance_data["C_max"]  # Upper bound constant
    
    ships = instance_data["ships"]
    berths = instance_data["berths"]
    
    # Extract ship properties
    widths = [ship["width"] for ship in ships]
    lengths = [ship["length"] for ship in ships]
    arrival_times = [ship["arrival_time"] for ship in ships]
    tp = [ship["transshipment_penalty"] for ship in ships]  # transshipment penalties
    mr_sets = [ship["mooring_set_MR"] for ship in ships]  # mooring sets
    eligible_berths = [ship["eligible_berths_B"] for ship in ships]  # eligible berths
    mr_matrix = [[j in mooring_list for j in range(num_ships)] for mooring_list in mr_sets]
    
    # Handling times
    H_ik_matrix = instance_data["handling_times"]  # [ship_idx][berth_idx]

    # Initialize model
    m = gp.Model("lock_scheduling")
    m.setParam("Threads", 1)
    m.setParam('TimeLimit', time_limit)
    m.setParam('LogToConsole', 0)

    # Binary variables: mode selection for each ship (0=lock, 1=transshipment)
    y = [m.addVar(vtype=GRB.BINARY, name=f"mode_{i}") for i in range(num_ships)]

    # Binary variables: assign lock ships to lockages
    max_lockages = num_ships  # As stated in problem description
    x_lock = {}  # x_lock[(i, l)] = 1 if ship i goes in lockage l
    for i in range(num_ships):
        for l in range(max_lockages):
            x_lock[(i, l)] = m.addVar(vtype=GRB.BINARY, name=f"x_lock_{i}_{l}")

    # Binary variables: assign transshipment ships to berths in specific order
    x_trans = {}  # x_trans[(i, k, o)] = 1 if ship i goes to berth k in order o
    for i in range(num_ships):
        for k in range(num_berths):
            for o in range(num_ships):  # Order at berth - enough for worst case
                is_eligible = k in eligible_berths[i]
                if is_eligible:
                    x_trans[(i, k, o)] = m.addVar(vtype=GRB.BINARY, name=f"x_trans_{i}_{k}_{o}")
                else:
                    x_trans[(i, k, o)] = 0

    # Binary variables: use specific lockages / chambers
    z_l = [m.addVar(vtype=GRB.BINARY, name=f"use_lockage_{l}") for l in range(max_lockages)]
    
    # Variables: positioning ships in lockage chamber (integer grid assumed but relax as continuous)
    pos_x = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=W, name=f"pos_x_{i}") for i in range(num_ships)]
    pos_y = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=L, name=f"pos_y_{i}") for i in range(num_ships)]
    
    # Binary variables: spatial relations for non-overlapping (for MIP)
    r_ijx = {}  # Ship i to the left of j
    r_ijy = {}  # Ship i behind j
    for i in range(num_ships):
        for j in range(num_ships):
            if i != j:
                r_ijx[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"r_ijx_{i}_{j}")
                r_ijy[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"r_ijy_{i}_{j}")
    
    # Variable: Completion time of each lockage
    c_l = [m.addVar(vtype=GRB.CONTINUOUS, lb=sc + Du, name=f"c_l_{l}") for l in range(max_lockages)]

    # Variable: Departure time of each ship
    d_i = [m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"d_i_{i}") for i in range(num_ships)]
    
    # Variables for transshipment at berths
    s_ik = {}  # Start time for ship i at berth k
    for i in range(num_ships):
        for k in eligible_berths[i]:  # Only consider eligible berths
            s_ik[(i, k)] = m.addVar(vtype=GRB.CONTINUOUS, lb=berths[k]["start_availability_sk"], name=f"s_ik_{i}_{k}")

    # Constraints
    # Mode assignments -> Each ship chooses lock or transshipment
    for i in range(num_ships):
        total_assignment = gp.quicksum(x_lock[(i, l)] for l in range(max_lockages))
        m.addConstr(total_assignment + y[i] == 1)
        
    # Lockage assignment constraints
    for i in range(num_ships):
        for l in range(max_lockages):
            # If ship is in transshipment, it cannot be assigned to any lockage
            m.addConstr(x_lock[(i, l)] <= 1 - y[i])

    # If any ship is assigned to lockage l, mark it as used
    for l in range(max_lockages):
        for i in range(num_ships):
            m.addConstr(x_lock[(i, l)] <= z_l[l])
        # Use lockage if any ship assigned
        m.addConstr(gp.quicksum(x_lock[(i, l)] for i in range(num_ships)) <= num_ships * z_l[l])
        m.addConstr(gp.quicksum(x_lock[(i, l)] for i in range(num_ships)) >= z_l[l])

    # Only transshipment ships can go to berths
    for i in range(num_ships):
        # Each transship-mode ship goes to exactly one berth-order combination
        eligible_assignments = gp.quicksum(
            x_trans[(i, k, o)] 
            for k in eligible_berths[i] 
            for o in range(num_ships)
            if k in eligible_berths[i]
        )
        m.addConstr(eligible_assignments == y[i])

    # Ensure at most one ship per berth-order slot
    for k in range(num_berths):
        for o in range(num_ships):
            m.addConstr(gp.quicksum(
                x_trans.get((i, k, o), 0) 
                for i in range(num_ships) 
                if k in eligible_berths[i]
            ) <= 1)

    # Spatial non-overlapping constraints
    for i in range(num_ships):
        for j in range(num_ships):
            if i != j:
                # pos_x[j] >= pos_x[i] + width_i - M * (1 - r_ijx[i,j])
                M_pos = W
                m.addConstr(pos_x[j] >= pos_x[i] + widths[i] - M_pos * (1 - r_ijx[(i, j)]))
                
                # pos_y[j] >= pos_y[i] + lengths[i] - M * (1 - r_ijy[i,j])
                M_pos = L  
                m.addConstr(pos_y[j] >= pos_y[i] + lengths[i] - M_pos * (1 - r_ijy[(i, j)]))

                # Fit in chamber constraints
                m.addConstr(pos_x[i] + widths[i] <= W) 
                m.addConstr(pos_y[i] + lengths[i] <= L)

    # For each lockage, enforce non-overlap condition if both ships are in same lockage
    for l in range(max_lockages):
        for i in range(num_ships):
            for j in range(i+1, num_ships):
                # If both i and j are in lockage l, they must be non-overlapping
                m.addConstr(
                    x_lock[(i, l)] + x_lock[(j, l)] <= 1 + r_ijx[(i, j)] + r_ijx[(j, i)] + r_ijy[(i, j)] + r_ijy[(j, i)]
                )
    
    # Mooring constraints
    for i in range(num_ships):
        mooring_terms = []

        # Mooring possibilities:
        # 1. Moored to left of another ship in same lockage
        for j in range(num_ships):
            if mr_matrix[j][i]:  # Ship j is allowed to moor to left of i
                for l in range(max_lockages):
                    moored_to_j_left_of_i = m.addVar(vtype=GRB.BINARY, name=f"m_{j}_left_of_{i}_l{l}")
                    
                    # Link this with both being in same lockage l
                    m.addConstr(moored_to_j_left_of_i <= x_lock[(i, l)])
                    m.addConstr(moored_to_j_left_of_i <= x_lock[(j, l)])
                    
                    # Enforce positioning: j is adjacent and on the left of i
                    m.addConstr(pos_x[i] <= pos_x[j] + widths[j] + C_max*(1 - moored_to_j_left_of_i))
                    m.addConstr(pos_x[i] >= pos_x[j] + widths[j] - C_max*(1 - moored_to_j_left_of_i))
                    
                    # Y alignment constraint: j's y-span contains i's y-span, or vice versa
                    within1 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[i] >= pos_y[j] - C_max * (1 - within1))
                    m.addConstr(pos_y[i] + lengths[i] <= pos_y[j] + lengths[j] + C_max * (1 - within1))
                    
                    within2 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[j] >= pos_y[i] - C_max * (1 - within2))
                    m.addConstr(pos_y[j] + lengths[j] <= pos_y[i] + lengths[i] + C_max * (1 - within2))
                    
                    m.addConstr(within1 + within2 >= moored_to_j_left_of_i)
                    
                    mooring_terms.append(moored_to_j_left_of_i)
        
        # Similarly for j to the right of i (i moored to the left of j)
        for j in range(num_ships):
            if mr_matrix[i][j]:  # i is allowed to go to the left of j (so j moors right of i)
                for l in range(max_lockages):
                    moored_to_i_right_of_j = m.addVar(vtype=GRB.BINARY, name=f"m_{i}_right_of_{j}_l{l}")
                    m.addConstr(moored_to_i_right_of_j <= x_lock[(i, l)])
                    m.addConstr(moored_to_i_right_of_j <= x_lock[(j, l)])

                    m.addConstr(pos_x[j] + widths[j] <= pos_x[i] + C_max*(1 - moored_to_i_right_of_j))
                    m.addConstr(pos_x[j] + widths[j] >= pos_x[i] - C_max*(1 - moored_to_i_right_of_j))

                    within1 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[j] >= pos_y[i] - C_max * (1 - within1))
                    m.addConstr(pos_y[j] + lengths[j] <= pos_y[i] + lengths[i] + C_max * (1 - within1))
                    
                    within2 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[i] >= pos_y[j] - C_max * (1 - within2))
                    m.addConstr(pos_y[i] + lengths[i] <= pos_y[j] + lengths[j] + C_max * (1 - within2))
                    
                    m.addConstr(within1 + within2 >= moored_to_i_right_of_j)

                    mooring_terms.append(moored_to_i_right_of_j)
        
        # Mooring to chamber walls
        # To left wall: pos_x[i] = 0
        left_wall_moored = m.addVar(vtype=GRB.BINARY, name=f"wall_left_m_{i}")
        for l in range(max_lockages):
            moored_to_left_wall_in_l = m.addVar(vtype=GRB.BINARY, name=f"m_leftwall_{i}_l{l}")
            m.addConstr(moored_to_left_wall_in_l <= x_lock[(i, l)])
            m.addConstr(pos_x[i] <= C_max * (1 - moored_to_left_wall_in_l))
            m.addConstr(pos_x[i] >= -C_max * (1 - moored_to_left_wall_in_l))
            mooring_terms.append(moored_to_left_wall_in_l)
        
        # To right wall: pos_x[i] + widths[i] = W
        right_wall_moored = m.addVar(vtype=GRB.BINARY, name=f"wall_right_m_{i}")
        for l in range(max_lockages):
            moored_to_right_wall_in_l = m.addVar(vtype=GRB.BINARY, name=f"m_rightwall_{i}_l{l}")
            m.addConstr(moored_to_right_wall_in_l <= x_lock[(i, l)])
            m.addConstr(pos_x[i] + widths[i] <= W + C_max * (1 - moored_to_right_wall_in_l))
            m.addConstr(pos_x[i] + widths[i] >= W - C_max * (1 - moored_to_right_wall_in_l))
            mooring_terms.append(moored_to_right_wall_in_l)
        
        # Finally, each ship in a lock must be moored
        m.addConstr(gp.quicksum(mooring_terms) >= gp.quicksum(x_lock[(i, l)] for l in range(max_lockages)))

    # Lockage completion time constraints
    for l in range(max_lockages):
        # Must wait for all assigned ships arrival and chamber availability
        for i in range(num_ships):
            m.addConstr(c_l[l] >= (arrival_times[i] + Du) - C_max * (1 - x_lock[(i, l)]))
        m.addConstr(c_l[l] >= (sc + Du) - C_max * (1 - z_l[l]))

    # Minimum time between lockages
    for l in range(1, max_lockages):
        m.addConstr(c_l[l-1] + MT <= c_l[l] + C_max * (1 - z_l[l]))
        m.addConstr(c_l[l-1] <= c_l[l] + C_max * (2 - z_l[l-1] - z_l[l]))  # Monotonicity if both used

    # FCFS constraint for lock-mode ships
    for i in range(num_ships):
        for j in range(num_ships):
            if arrival_times[i] < arrival_times[j]:
                # If both are lock mode, i must be in same or earlier lockage than j
                # We can express this with position values
                lockage_pos_i = gp.quicksum(l * x_lock[(i, l)] for l in range(max_lockages))
                lockage_pos_j = gp.quicksum(l * x_lock[(j, l)] for l in range(max_lockages))
                
                # Apply only if both are lock-mode
                mode_i = gp.quicksum(x_lock[(i, l)] for l in range(max_lockages))
                mode_j = gp.quicksum(x_lock[(j, l)] for l in range(max_lockages))
                
                m.addConstr(
                    lockage_pos_i <= lockage_pos_j + max_lockages * (2 - mode_i - mode_j)
                )

    # Set departure times accordingly
    for i in range(num_ships):
        # For lock mode ships: when their lockage completes
        for l in range(max_lockages):
            m.addConstr(d_i[i] <= c_l[l] + C_max * (1 - x_lock[(i, l)]))
            m.addConstr(d_i[i] >= c_l[l] - C_max * (1 - x_lock[(i, l)]))
        
        # For trans mode: departure after service completes
        for k in eligible_berths[i]:
            # Service start + handling time = departure time if assigned to this berth
            expr = s_ik[(i, k)] + H_ik_matrix[i][k]
            for o in range(num_ships):
                m.addConstr(d_i[i] <= expr + C_max * (1 - x_trans.get((i, k, o), 0)))
                m.addConstr(d_i[i] >= expr - C_max * (1 - x_trans.get((i, k, o), 0)))

    # Schedule at berths: order ships according to their assigned order
    for k in range(num_berths):
        sk_start = berths[k]["start_availability_sk"]
        
        # The service start must be at least as late as the berth availability
        for i in range(num_ships):
            if k in eligible_berths[i]:
                # Ensure that service start respects berth start time
                m.addConstr(s_ik[(i, k)] >= sk_start - C_max * (1 - gp.quicksum(x_trans.get((i, k, o), 0) for o in range(num_ships))))
                m.addConstr(s_ik[(i, k)] >= arrival_times[i] - C_max * (1 - gp.quicksum(x_trans.get((i, k, o), 0) for o in range(num_ships))))
        
        # Sequence the ship starts in chronological order by their assignment
        for o in range(1, num_ships):
            # Find each ship assigned to this berth in this order and make sure it starts after previous
            # The cumulative approach: sum all processing times for orders before o
            total_previous_time = 0
            
            for prev_order in range(o):
                for ip in range(num_ships):
                    if k in eligible_berths[ip]:
                        total_previous_time += x_trans.get((ip, k, prev_order), 0) * H_ik_matrix[ip][k]
            
            # For all candidates i in order o: start >= total_previous_time
            for i in range(num_ships):
                if k in eligible_berths[i]:
                    m.addConstr(
                        s_ik[(i, k)] >= total_previous_time - C_max * (1 - x_trans.get((i, k, o), 0))
                    )

    # Objective function components
    obj_parts = []

    # For lockmode: delay = departure_time - arrival_time
    for i in range(num_ships):
        obj_parts.append((1 - y[i]) * (d_i[i] - arrival_times[i]))

    # Add transshipment penalties for those choosing transshipment
    for i in range(num_ships):
        obj_parts.append(tp[i] * y[i])

    # Add transshipment delays: departure time - arrival time
    for i in range(num_ships):
        obj_parts.append(y[i] * (d_i[i] - arrival_times[i]))

    m.setObjective(gp.quicksum(obj_parts), GRB.MINIMIZE)

    # Solve
    m.optimize()

    # Extract and return solution
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        result = {
            "objective_value": m.objVal if m.objVal != float('inf') else float('inf'),
            "modes": {},
            "completion_times": {},
            "lockage_assignments": {},
            "lockage_completion_times": {},
            "berth_assignments": {}
        }

        for i in range(num_ships):
            val = round(y[i].x) # mode: 1=transship, 0=lock
            result["modes"][str(i)] = val
            result["completion_times"][str(i)] = d_i[i].x

        for i in range(num_ships):
            if result["modes"][str(i)] == 0:  # lock mode
                for l in range(max_lockages):
                    if x_lock[(i, l)].x > 0.5:
                        result["lockage_assignments"][str(i)] = l
                        break

        for l in range(max_lockages):
            if z_l[l].x > 0.9 and any(x_lock[(i, l)].x > 0.5 for i in range(num_ships)):
                result["lockage_completion_times"][str(l)] = c_l[l].x

        for i in range(num_ships):
            if result["modes"][str(i)] == 1:  # trans mode
                for k in eligible_berths[i]:
                    for o in range(num_ships):
                        var = x_trans.get((i, k, o), 0)
                        if hasattr(var, 'x') and var.x > 0.9:
                            result["berth_assignments"][str(i)] = {"berth": k, "order": o}
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

        return result
    else:
        return {"objective_value": float('inf'), "modes": {}, "completion_times": {}, "lockage_assignments": {}, "lockage_completion_times": {}, "berth_assignments": {}}


def main():
    parser = argparse.ArgumentParser(description='Solve shipping lock scheduling problem.')
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the input instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where to write the output solution JSON")
    parser.add_argument("--time_limit", type=int, default=600, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, help="Path to log intermediate solutions")

    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve
    solution = solve_shipping_problem(instance_data, args.time_limit)

    # Log if we got a valid solution
    if solution and solution['objective_value'] != float('inf'):
        if logger:
            logger.log(solution['objective_value'])

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()