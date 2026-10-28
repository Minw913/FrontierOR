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
    m.setParam('TimeLimit', time_limit)
    m.setParam('LogToConsole', 1)

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
                x_trans[(i, k, o)] = m.addVar(vtype=GRB.BINARY, name=f"x_trans_{i}_{k}_{o}")

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
    
    # Variable: Transshipment time with idle time considerations
    # Arrival of ship i at berth k in order o is calculated dynamically (no fixed var) 
    s_ik = {}  # Start service of ship i at berth k
    w_ik = {}  # Total waiting/idle time up to serving ship i at berth k (from its start)
    
    # First, compute all service variables per berth/order pair since we might have more orders than ships
    for i in range(num_ships):
        for k in eligible_berths[i]:  # Only consider eligible berths
            s_ik[(i, k)] = m.addVar(vtype=GRB.CONTINUOUS, lb=berths[k]["start_availability_sk"], name=f"s_ik_{i}_{k}")
            w_ik[(i, k)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_ik_{i}_{k}")
    
    # Constraints
    # Mode assignments
    for i in range(num_ships):
        m.addConstr(y[i] + gp.quicksum(x_lock[(i, l)] for l in range(max_lockages)) == 1)
    
    # Lockage assignment constraints
    for i in range(num_ships):
        for l in range(max_lockages):
            m.addConstr(x_lock[(i, l)] <= 1 - y[i])  # Can only be in lockage if in lock mode
    
    for l in range(max_lockages):
        # If anything in this lockage, then z_l = 1; otherwise z_l can be 0
        for i in range(num_ships):
            m.addConstr(x_lock[(i, l)] <= z_l[l])
        
        # Use lockage if any ship assigned
        m.addConstr(gp.quicksum(x_lock[(i, l)] for i in range(num_ships)) >= z_l[l] * 0.5)  # At least one ships if used
        
    # Berth assignment - only transship ships to berths
    for i in range(num_ships):
        berth_assignment_sum = gp.quicksum(
            x_trans[(i, k, o)] 
            for k in eligible_berths[i] 
            for o in range(num_ships)
        )
        m.addConstr(berth_assignment_sum <= y[i])  # Only if transshipping
        
        # Each transship-mode ship goes to exactly one berth-order combination
        m.addConstr(berth_assignment_sum == y[i])

    # Ensure at most one ship per berth-order slot
    for k in range(num_berths):
        for o in range(num_ships):
            m.addConstr(gp.quicksum(x_trans[(i, k, o)] for i in range(num_ships) if k in eligible_berths[i]) <= 1)

    # Spatial non-overlapping constraints
    for i in range(num_ships):
        for j in range(num_ships):
            if i != j:
                # Position constraints with binary relation vars
                # Constraint: (left of) or (behind) for non-overlap
                # (pos_x[j] >= pos_x[i] + width_i) OR (pos_y[j] >= pos_y[i] + length_i)
                
                # pos_x[j] >= pos_x[i] + width_i - M * (1 - r_ijx[i,j])
                M_pos = W
                m.addConstr(pos_x[j] >= pos_x[i] + widths[i] - M_pos * (1 - r_ijx[(i, j)]))
                
                # pos_y[j] >= pos_y[i] + lengths[i] - M * (1 - r_ijy[(i,j)])
                M_pos = L  
                m.addConstr(pos_y[j] >= pos_y[i] + lengths[i] - M_pos * (1 - r_ijy[(i, j)]))
                
                # Fit in chamber constraints
                m.addConstr(pos_x[i] + widths[i] <= W) 
                m.addConstr(pos_y[i] + lengths[i] <= L)

    # For each lockage, ship overlap constraint applied conditionally
    for l in range(max_lockages):
        for i in range(num_ships):
            for j in range(i+1, num_ships):
                m.addConstr(
                    r_ijx[(i, j)] + r_ijx[(j, i)] +
                    r_ijy[(i, j)] + r_ijy[(j, i)] 
                    >= x_lock[(i, l)] + x_lock[(j, l)] - 1
                )

    # Mooring constraints
    for i in range(num_ships):
        # Ship must be moored to something: 
        # left of another allowed ship in same lockage;
        # right of another allowed ship in same lockage;
        # or left/right walls of chamber.
        
        # Define mooring constraints - not straightforward with the definition provided.
        # This will be tricky: require that the ship is directly adjacent to a side of a permitted object
        
        m_constraint_terms = []
        
        # Adjacency to another ship to the left of 'i' (i is to the right of ship j)
        for j in range(num_ships):
            if mr_matrix[j][i]:  # j is allowed to moor on i's left side (ie, ship j moors to left of i)
                # This means: j is to the left of i, with no gap
                # pos_x[i] == pos_x[j] + widths[j] 
                # For each possible lockage
                for l in range(max_lockages):
                    # Create an aux var to link these conditions
                    moored_to_j_left_of_i = m.addVar(vtype=GRB.BINARY, name=f"m_{j}_left_of_{i}_l{l}")
                    
                    # If both in same lockage & j is immediately to left of i & properly aligned
                    m.addConstr(moored_to_j_left_of_i <= x_lock[(i, l)])
                    m.addConstr(moored_to_j_left_of_i <= x_lock[(j, l)])
                    
                    # Enforce proximity/alignment constraint indirectly with big-M logic
                    # pos_x[i] = pos_x[j] + width[j], and proper vertical alignment
                    
                    # Length alignment constraint (i fully fits in j's vertical span or vice versa)
                    # This is complex to express with linear constraints. For now we'll simplify:
                    # We only need to encourage mooring without enforcing it completely.
                    # Let's define a mooring constraint that ensures ships align properly:
                    
                    m.addConstr(pos_x[i] <= pos_x[j] + widths[j] + C_max*(1 - moored_to_j_left_of_i))
                    m.addConstr(pos_x[i] >= pos_x[j] + widths[j] - C_max*(1 - moored_to_j_left_of_i))
                    
                    # Y alignment: i fully within j's y span or vice versa
                    M_for_overlap = C_max
                    # i_y_start >= j_y_start, i_y_end <= j_y_end (i is within j's y-range)
                    within1 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[i] >= pos_y[j] - M_for_overlap * (1 - within1))
                    m.addConstr(pos_y[i] + lengths[i] <= pos_y[j] + lengths[j] + M_for_overlap * (1 - within1))
                    
                    # j_y_start >= i_y_start, j_y_end <= i_y_end (j is within i's y-range)
                    within2 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[j] >= pos_y[i] - M_for_overlap * (1 - within2))
                    m.addConstr(pos_y[j] + lengths[j] <= pos_y[i] + lengths[i] + M_for_overlap * (1 - within2))
                    
                    # Either i is within j, or j is within i, or both if they perfectly overlap in y
                    m.addConstr(within1 + within2 >= moored_to_j_left_of_i)
                    
                    m_constraint_terms.append(moored_to_j_left_of_i)
        
        # Similar for i to the left of another ship j (j moors to right of i)
        for j in range(num_ships):
            if mr_matrix[i][j]:  # i is allowed to go to j's left side (j moors to right of i)
                for l in range(max_lockages):
                    moored_to_i_right_of_j = m.addVar(vtype=GRB.BINARY, name=f"m_{i}_right_of_{j}_l{l}")
                    m.addConstr(moored_to_i_right_of_j <= x_lock[(i, l)])
                    m.addConstr(moored_to_i_right_of_j <= x_lock[(j, l)])

                    # pos_x[j] + widths[j] == pos_x[i] 
                    m.addConstr(pos_x[j] + widths[j] <= pos_x[i] + C_max*(1 - moored_to_i_right_of_j))
                    m.addConstr(pos_x[j] + widths[j] >= pos_x[i] - C_max*(1 - moored_to_i_right_of_j))

                    # Y alignment: j fully within i's span or vice versa
                    M_for_overlap = C_max
                    within1 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[j] >= pos_y[i] - M_for_overlap * (1 - within1))
                    m.addConstr(pos_y[j] + lengths[j] <= pos_y[i] + lengths[i] + M_for_overlap * (1 - within1))
                    
                    within2 = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(pos_y[i] >= pos_y[j] - M_for_overlap * (1 - within2))
                    m.addConstr(pos_y[i] + lengths[i] <= pos_y[j] + lengths[j] + M_for_overlap * (1 - within2))
                    
                    m.addConstr(within1 + within2 >= moored_to_i_right_of_j)

                    m_constraint_terms.append(moored_to_i_right_of_j)
        
        # Mooring to chamber walls requires more complex logic and probably custom constraints
        # Simpler approach: enforce that each ship must be adjacent to something else (another ship or wall)
        # For left wall: pos_x[i] = 0
        # For right wall: pos_x[i] + widths[i] = W
        
        # Left wall mooring: add terms to enforce pos_x[i] = 0 if assigned to a lock
        m_wall_left = m.addVar(vtype=GRB.BINARY, name=f"m_wall_left_{i}")
        # Check if moored to left wall in any lockage
        for l in range(max_lockages):
            moored_to_left_wall_in_l = m.addVar(vtype=GRB.BINARY, name=f"m_leftwall_{i}_l{l}")
            
            m.addConstr(moored_to_left_wall_in_l <= x_lock[(i, l)])  # Only if assigned
            # pos_x[i] == 0 means touching left wall
            m.addConstr(pos_x[i] <= C_max * (1 - moored_to_left_wall_in_l))
            m.addConstr(pos_x[i] >= -C_max * (1 - moored_to_left_wall_in_l))
        
            m_constraint_terms.append(moored_to_left_wall_in_l)
        
        # Right wall mooring
        m_wall_right = m.addVar(vtype=GRB.BINARY, name=f"m_wall_right_{i}")
        for l in range(max_lockages):
            moored_to_right_wall_in_l = m.addVar(vtype=GRB.BINARY, name=f"m_rightwall_{i}_l{l}")
            m.addConstr(moored_to_right_wall_in_l <= x_lock[(i, l)])
            # pos_x[i] + widths[i] == W means touching right wall
            m.addConstr((pos_x[i] + widths[i]) <= W + C_max * (1 - moored_to_right_wall_in_l))
            m.addConstr((pos_x[i] + widths[i]) >= W - C_max * (1 - moored_to_right_wall_in_l))

            m_constraint_terms.append(moored_to_right_wall_in_l)
        
        # Final mooring constraint: i must be moored to at least one entity in its lockage
        m.addConstr(gp.quicksum(m_constraint_terms) >= gp.quicksum(x_lock[(i, l)] for l in range(max_lockages)))

    # Lockage completion time constraints
    for l in range(max_lockages):
        # Must wait for all assigned ships and chamber to be ready
        for i in range(num_ships):
            # Completion time of lockage l >= arrival of i + Du (if i is assigned to this lockage)
            m.addConstr(c_l[l] >= (arrival_times[i] + Du) - C_max * (1 - x_lock[(i, l)]))
        
        # Also ensure c_l[l] >= sc + Du if used
        m.addConstr(c_l[l] >= (sc + Du) - C_max * (1 - z_l[l]))

    # Minimum time between lockages
    for l in range(1, max_lockages):
        m.addConstr(c_l[l-1] + MT <= c_l[l] + C_max * (1 - z_l[l]))
        m.addConstr(c_l[l-1] <= c_l[l] + C_max * (1 - z_l[l-1]))  # Monotonicity

    # FCFS constraint for lockships: if ship i arrives before j then lockage i <= lockage j
    for i in range(num_ships):
        for j in range(num_ships):
            if arrival_times[i] < arrival_times[j]:
                # Both must be in lock mode
                lock_mode_i = gp.quicksum(x_lock[(i, l)] for l in range(max_lockages))
                lock_mode_j = gp.quicksum(x_lock[(j, l)] for l in range(max_lockages))
                
                # For each possible lockage assignment
                for l1 in range(max_lockages):
                    for l2 in range(l1 + 1, max_lockages):  # l1 < l2 implies we're okay
                        pass
                    # For l1 > l2, we must ensure that doesn't happen if both are lock
                    # x_lock[i,l1] and x_lock[j,l2] cannot both be 1 if l1 > l2
                    if l1 > 0:  # because we want order constraint when in lockmode
                        m.addConstr(
                            x_lock[(i, l1)] <= 
                            1 - x_lock[(j, l1 - 1)]  # If j is at l1-1, i can’t be at l1 > l1-1
                            if l1 > 0
                            else GRB.INFINITY  # Placeholder for loop logic refactoring
                        )
                        
                # Better constraint using total ordering
                lockage_order_i = gp.quicksum(l * x_lock[(i, l)] for l in range(max_lockages))
                lockage_order_j = gp.quicksum(l * x_lock[(j, l)] for l in range(max_lockages))
                
                # Only apply if both are lock-mode
                m.addConstr(lockage_order_i <= lockage_order_j + max_lockages * (
                    2 - lock_mode_i - lock_mode_j
                ))

    # Departure time settings
    for i in range(num_ships):
        # For lock mode ships
        for l in range(max_lockages):
            m.addConstr(d_i[i] <= c_l[l] + C_max * (1 - x_lock[(i, l)]))
            m.addConstr(d_i[i] >= c_l[l] - C_max * (1 - x_lock[(i, l)]))
        
        # For transshipment ships: d_i = arrival time (in this simplified logic,
        # more accurately it depends on berth processing, which will be below)
        # Update: for transship mode, let's reconsider this
        
        # Actually, departure time for transship is when its service ends at berth
        for k in range(num_berths):
            for o in range(num_ships):
                # If assigned to this berth and order, its departure equals service start + handling time
                m.addConstr(
                    d_i[i] <= 
                    (s_ik.get((i, k), 0) + H_ik_matrix[i][k])
                    + C_max * (1 - x_trans.get((i, k, o), 0))
                )
                m.addConstr(
                    d_i[i] >= 
                    (s_ik.get((i, k), 0) + H_ik_matrix[i][k])
                    - C_max * (1 - x_trans.get((i, k, o), 0))
                )

    # For transship mode, service start at berth constraints
    for k in range(num_berths):
        # We need to order services on the same berth based on assigned order
        # For each order o > 0, the start of service o >= finish of order (o-1)
        assigned_ship_orders = []  # List of (ship, order) pairs for berth k
        for order in range(num_ships):
            for i in range(num_ships):
                if k in eligible_berths[i]:
                    # This is complex because we do not know ahead which ship will be in which order
                    # Instead, we'll make start-of-service constraints based on all possible combinations
                    s_ik_var = s_ik.get((i, k), None)
                    if s_ik_var is not None:
                        # Calculate earliest start: max of berth's start and arrival_time + idles + previous processing
                        # We approximate by enforcing sequential processing:
                        for prev_o in range(order):
                            for prev_i in range(num_ships):
                                if k in eligible_berths[prev_i]:
                                    # If i at order o and prev_i at prev_o, start of i >= start + proc of prev_i  
                                    # This is complex, so we’ll use a simpler approach:
                                    # Define a sequence explicitly based on x_trans
                                    
                                    # But it's easier to create variables linking the service order:
                                    m.addConstr(
                                        s_ik.get((i, k), 0) >=
                                        s_ik.get((prev_i, k), 0) + H_ik_matrix[prev_i][k]
                                        - C_max * (2 - x_trans.get((i, k, order), 0) - x_trans.get((prev_i, k, prev_o), 0))
                                    )

    # Now for transshipment scheduling: a ship starts after its turn in order and after berth availability
    for i in range(num_ships):
        for k in eligible_berths[i]:
            h_ik = H_ik_matrix[i][k]
            
            # Service start time depends on sum of handling+idle time of all preceding orders on this berth
            # Create helper variables for each possible precedence
            
            # This is getting complex. Let's reformulate with a standard flow-shop pattern
            
            # For each pair (i,k,o): if ship i is scheduled in order o at berth k
            for o in range(num_ships):
                if o == 0:  # First in sequence
                    m.addConstr(
                        s_ik[(i, k)] >= berth_start[berth_idx_to_k[k]] - C_max *(1 - x_trans[(i, k, o)])
                    )
                else:  # Later in sequence
                    # Find the actual previous ship in order (unknown ahead of time)
                    # We need logical implications: if (i1, k, o-1) and (i2, k, o) =>
                    # s[i2,k] >= s[i1,k] + h[i1,k] (with appropriate big-M)
    
    # Let's create binary vars and constraints to represent the "preceding" relationship
    
    # Precedence vars: p[i1, k, i2, order] = 1 means i1 precedes i2 in order at k
    p_order = {}  # p[(i1, k, i2, o)] is 1 if both assigned and i2 is in order after i1
    for i1 in range(num_ships):
        for k in eligible_berths[i1]:
            for i2 in range(i1 + 1, num_ships):  # Symmetry break
                for o in range(1, num_ships):  # Non-first orders
                    for ord_i1 in range(o):
                        for ord_i2 in range(ord_i1 + 1, o + 1):
                            # This is getting too complex. Let’s use a simpler version:
                            # For each (i, k, ord), calculate cumulative time till that position
    
    # Better approach: Define cumulative time up to each order
    cum_t = {}
    for k in range(num_berths):
        for o in range(num_ships):
            cum_t[(k, o)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"cum_t_{k}_{o}")
            
            # Cumulative time = sum of proc times of all ships in orders 0 to o-1
            m.addConstr(
                cum_t[(k, o)] == 
                gp.quicksum(
                    x_trans[(i, k, ord)] * H_ik_matrix[i][k] 
                    for i in range(num_ships) if k in eligible_berths[i] for ord in range(o)
                ) 
            )
    
    for k in range(num_berths):
        sk_b = berths[k]["start_availability_sk"]
        for i in range(num_ships):
            if k in eligible_berths[i]:
                # Service start of ship i at k = max(sk_b, sum of procs at k of ships before i)
                for o in range(num_ships):
                    # If ship i is in order o at k
                    # Its start time s_ik[i,k] >= sk_b + cum_t[k][o-1] (if valid), and also based on arrival
                    if o == 0:
                        m.addConstr(s_ik[(i, k)] >= sk_b - C_max * (1 - gp.quicksum(x_trans[(i, k, 0)])))
                        m.addConstr(s_ik[(i, k)] >= arrival_times[i] - C_max * (1 - gp.quicksum(x_trans[(i, k, 0)])))
                    else:
                        # Sum of h[i1][k] for all i1 in orders 0 to o-1
                        total_prev_proc = gp.quicksum(
                            x_trans[(ip, k, op)] * H_ik_matrix[ip][k] 
                            for ip in range(num_ships) if k in eligible_berths[ip] 
                            for op in range(o)
                        )
                        m.addConstr(s_ik[(i, k)] >= sk_b + total_prev_proc - C_max * (1 - x_trans[(i, k, o)]))
                        m.addConstr(s_ik[(i, k)] >= arrival_times[i])
    
    # Simplified version: just enforce correct sequence via cumulative sums
    for k in range(num_berths):
        sk_b = berths[k]["start_availability_sk"]
        m.addConstr(cum_t[(k, 0)] == 0)  # No cum proc time before any
        
        for o in range(1, num_ships):
            current_proc_sum = gp.quicksum(
                x_trans[(i, k, o-1)] * H_ik_matrix[i][k] 
                for i in range(num_ships) if k in eligible_berths[i]
            )
            m.addConstr(cum_t[(k, o)] == cum_t[(k, o-1)] + current_proc_sum)

        # For each ship i at k, if in order o, its start time is cum_t + delay due to arrival
        for i in range(num_ships):
            if k in eligible_berths[i]:
                # s_ik[i,k] = max(sk_b + cum_time_till_before_i_at_k, arrival_i)
                # This needs more variables for conditional time due to arrival
                
                # So s_ik[i,k] >= sk_b + cum of prev orders
                # s_ik[i,k] >= arrival_i
                for o in range(num_ships):
                    m_cum = cum_t[(k, o)] if o == 0 else gp.quicksum(
                        x_trans[(ip, k, op)] * H_ik_matrix[ip][k] 
                        for ip in range(num_ships) if k in eligible_berths[ip]
                        for op in range(o)
                    )
                    
                    # More carefully:
                    start_time_expr = sk_b + gp.quicksum(
                        x_trans[(ip, k, op)] * H_ik_matrix[ip][k] 
                        for ip in range(num_ships) if k in eligible_berths[ip]
                        for op in range(o)  
                    ) 
                    m.addConstr(s_ik[(i, k)] >= start_time_expr - C_max * (1 - x_trans[(i, k, o)]))
                    m.addConstr(s_ik[(i, k)] >= arrival_times[i] - C_max * (1 - x_trans[(i, k, o)]))

    # Objective function
    obj_parts = []

    # Sum of lock delays: total departure time minus arrival time for lock-mode ships
    for i in range(num_ships):
        # d_i[i] already captures departure time
        # Total delay for lock-mode ships = sum of (d_i[i] - arrival_times[i]) for lock ships
        lock_delay = d_i[i] - arrival_times[i]
        obj_parts.append(y[i] * 0)  # Don't double count here; use x_lock sum
        # We want = sum of (d_i - arrival_i) for lock mode
        obj_parts.append(x_lock[(i, 0)] * (d_i[i] - arrival_times[i]))  # Example for l=0
        for l in range(1, max_lockages):
            obj_parts.append(x_lock[(i, l)] * (d_i[i] - arrival_times[i]))

    # Sum of transshipment penalties (penalties if choosing trans mode)
    for i in range(num_ships):
        obj_parts.append(y[i] * tp[i])

    # Transship delays? Yes - this includes the additional time spent waiting for service vs. just passing through
    # d_i[i] = end of service time for transship; arrival is always arrival_time
    # Transship delay component = sum of (d - arrival) for transship mode
    for i in range(num_ships):
        obj_parts.append(y[i] * (d_i[i] - arrival_times[i]))
    
    m.setObjective(gp.quicksum(obj_parts), GRB.MINIMIZE)

    # Solve
    m.optimize()

    # Extract solution
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        from collections import defaultdict

        # Build result dictionary
        result = {
            "objective_value": m.objVal if m.status == GRB.OPTIMAL else float('inf'),
            "modes": {},
            "completion_times": {},
            "lockage_assignments": {},
            "lockage_completion_times": {},
            "berth_assignments": {}
        }

        # Mode assignment
        for i in range(num_ships):
            val = round(y[i].x)
            result["modes"][str(i)] = val
            result["completion_times"][str(i)] = d_i[i].x

        # Lockage assignments
        for i in range(num_ships):
            if result["modes"][str(i)] == 0:  # Is lock mode
                for l in range(max_lockages):
                    if x_lock[(i, l)].x > 0.9:
                        result["lockage_assignments"][str(i)] = l
                        break

        # Lock completion times
        for l in range(max_lockages):
            if z_l[l].x > 0.9:
                result["lockage_completion_times"][str(l)] = c_l[l].x

        # Berth assignments
        for i in range(num_ships):
            if result["modes"][str(i)] == 1:  # Trans mode
                for k in eligible_berths[i]:
                    for o in range(num_ships):
                        if x_trans[(i, k, o)].x > 0.9:
                            result["berth_assignments"][str(i)] = {"berth": k, "order": o}
                            break
                    else:
                        continue
                    break

        return result
    else:
        return None


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