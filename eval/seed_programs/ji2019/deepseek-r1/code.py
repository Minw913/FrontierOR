import argparse
import json
from typing import Dict, List, Tuple, Any, Set, Optional
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from solution_logger import SolutionLogger

def main(instance_path: str, solution_path: str, time_limit: int, log_path: Optional[str]) -> None:
    # Load instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_ships = instance['num_ships']
    num_berths = instance['num_berths']
    num_chambers = instance['num_chambers']
    W = instance['chamber_width_W']
    L = instance['chamber_length_L']
    Du = instance['lockage_duration_Du']
    MT = instance['min_time_between_lockages_MT']
    lock_start = instance['lock_start_availability_sc']
    berth_end = instance['berth_end_availability_ek']
    C_max = instance['C_max']
    ships_data = instance['ships']
    berths_data = instance['berths']
    handling_times = instance['handling_times']
    
    # Preprocess ships: sort by arrival time (and by id if tie)
    ships_data_sorted = sorted(ships_data, key=lambda x: (x['arrival_time'], x['id']))
    orig_id_to_sorted_idx = {}
    for idx, ship in enumerate(ships_data_sorted):
        orig_id_to_sorted_idx[ship['id']] = idx
    
    # Create sorted lists of ship attributes
    arrival_times = [ship['arrival_time'] for ship in ships_data_sorted]
    ship_widths = [ship['width'] for ship in ships_data_sorted]
    ship_lengths = [ship['length'] for ship in ships_data_sorted]
    penalties = [ship['transshipment_penalty'] for ship in ships_data_sorted]
    mooring_sets = []
    for ship in ships_data_sorted:
        mooring_list = ship['mooring_set_MR']
        mooring_sets.append(set(mooring_list))
    eligible_berths = []
    for ship in ships_data_sorted:
        eligible_berths.append(ship['eligible_berths_B'])
    
    # Berth start times
    berth_starts = [berth['start_availability_sk'] for berth in berths_data]
    
    # Create model
    model = gp.Model('InlandWaterway')
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision Variables
    # 1. Mode selection: m[i] = 1 if ship i chooses transshipment, else 0
    m = model.addVars(num_ships, vtype=GRB.BINARY, name="mode")
    
    # 2. Lock-mode ships: assignment to lockages
    x = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="x")
    u = model.addVars(num_ships, vtype=GRB.BINARY, name="u")
    lockage_completion_times = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="lockage_completion")
    
    # Position variables for lock-mode ships
    posx = model.addVars(num_ships, lb=0, ub=W, vtype=GRB.CONTINUOUS, name="posx")
    posy = model.addVars(num_ships, lb=0, ub=L, vtype=GRB.CONTINUOUS, name="posy")
    
    # Mooring variables for lock-mode ships
    # For each ship i: left_wall, right_wall, and for each j in mooring_sets[i]: left and right
    moor_left_wall = model.addVars(num_ships, vtype=GRB.BINARY, name="moor_left_wall")
    moor_right_wall = model.addVars(num_ships, vtype=GRB.BINARY, name="moor_right_wall")
    moor_left_ship = {}  # (i,j) for j in mooring_sets[i]
    moor_right_ship = {}
    for i in range(num_ships):
        for j in mooring_sets[i]:
            if j < num_ships:  # Ensure j is a valid ship index
                moor_left_ship[i, j] = model.addVar(vtype=GRB.BINARY, name=f"moor_left_ship_{i}_{j}")
                moor_right_ship[i, j] = model.addVar(vtype=GRB.BINARY, name=f"moor_right_ship_{i}_{j}")
    
    # 3. Transshipment-mode ships: assignment to berths and positions
    z = model.addVars(num_ships, num_berths, num_ships, vtype=GRB.BINARY, name="z")
    # Start time at each berth for each position
    S = model.addVars(num_berths, num_ships, lb=0, ub=berth_end, vtype=GRB.CONTINUOUS, name="S")
    # Handling time at each berth for each position
    H = model.addVars(num_berths, num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="H")
    assigned = model.addVars(num_berths, num_ships, vtype=GRB.BINARY, name="assigned")
    
    # Constraints
    # 1. Mode selection: Each ship is either lock or transshipment
    #    No additional constraint needed beyond the variable definitions
    
    # 2. Lock-mode ships: assignment to lockages
    for i in range(num_ships):
        model.addConstr(gp.quicksum(x[i, l] for l in range(num_ships)) == 1 - m[i], name=f"lock_assign_{i}")
    
    for l in range(num_ships):
        # u_l = 1 if any ship is in lockage l
        model.addConstr(u[l] <= gp.quicksum(x[i, l] for i in range(num_ships)), name=f"u_lb_{l}")
        model.addConstr(u[l] * num_ships >= gp.quicksum(x[i, l] for i in range(num_ships)), name=f"u_ub_{l}")
        # Lockage l must have u_{l-1} = 1 if u_l = 1 (for l>=1)
        if l >= 1:
            model.addConstr(u[l] <= u[l-1], name=f"contiguous_{l}")
    
    # Lockage completion time constraints
    M_big = C_max
    for l in range(num_ships):
        # If u[l]=1, then lockage_completion_times[l] >= lock_start + Du
        model.addConstr(lockage_completion_times[l] >= lock_start + Du - M_big * (1 - u[l]), name=f"lock_start_{l}")
        # lockage_completion_times[l] >= max arrival time of ships in lockage l + Du
        for i in range(num_ships):
            model.addConstr(lockage_completion_times[l] >= arrival_times[i] + Du - M_big * (1 - x[i, l]), name=f"lock_arrival_{l}_{i}")
        # Minimum time between consecutive lockages
        if l >= 1:
            model.addConstr(lockage_completion_times[l] >= lockage_completion_times[l-1] + MT - M_big * (1 - u[l]), name=f"lock_interval_{l}")
    
    # FCFS for lock-mode ships: for i < j (in sorted order) and both lock-mode, the lockage index of i <= that of j
    for i in range(num_ships):
        for j in range(i+1, num_ships):
            # Only if both are lock-mode
            model.addConstr(
                gp.quicksum(l * x[i, l] for l in range(num_ships)) <= 
                gp.quicksum(l * x[j, l] for l in range(num_ships)) + M_big * (m[i] + m[j]),
                name=f"FCFS_{i}_{j}"
            )
    
    # Non-overlapping constraints for ships in the same lockage
    for i in range(num_ships):
        for j in range(i+1, num_ships):
            # Only consider if both are lock-mode
            # L_ij = 1 if both i and j are lock-mode and in the same lockage
            L_ij = model.addVar(vtype=GRB.BINARY, name=f"L_{i}_{j}")
            model.addConstr(L_ij <= 1 - m[i], name=f"L_ij_lock_i_{i}_{j}")
            model.addConstr(L_ij <= 1 - m[j], name=f"L_ij_lock_j_{i}_{j}")
            # L_ij = sum_l x[i,l] * x[j,l]
            sum_z = 0
            for l in range(num_ships):
                z_ijl = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_{l}")
                model.addConstr(z_ijl <= x[i, l], name=f"z_ijl_1_{i}_{j}_{l}")
                model.addConstr(z_ijl <= x[j, l], name=f"z_ijl_2_{i}_{j}_{l}")
                model.addConstr(z_ijl >= x[i, l] + x[j, l] - 1, name=f"z_ijl_3_{i}_{j}_{l}")
                sum_z += z_ijl
            model.addConstr(L_ij == sum_z, name=f"L_ij_eq_{i}_{j}")
            
            # Non-overlapping: use two binary variables for horizontal/vertical separation
            d_ij = model.addVar(vtype=GRB.BINARY, name=f"d_{i}_{j}")
            e_ij = model.addVar(vtype=GRB.BINARY, name=f"e_{i}_{j}")
            # Horizontal: d_ij=0; vertical: d_ij=1
            # Horizontal separation: e_ij=0 -> i left of j; e_ij=1 -> j left of i
            # Vertical separation: e_ij=0 -> i below j; e_ij=1 -> j below i
            model.addConstr(
                posx[i] + ship_widths[i] <= posx[j] + 
                M_big * (d_ij + e_ij) + 
                M_big * (1 - L_ij),
                name=f"nonoverlap1_{i}_{j}"
            )
            model.addConstr(
                posx[j] + ship_widths[j] <= posx[i] + 
                M_big * (d_ij + (1 - e_ij)) + 
                M_big * (1 - L_ij),
                name=f"nonoverlap2_{i}_{j}"
            )
            model.addConstr(
                posy[i] + ship_lengths[i] <= posy[j] + 
                M_big * ((1 - d_ij) + e_ij) + 
                M_big * (1 - L_ij),
                name=f"nonoverlap3_{i}_{j}"
            )
            model.addConstr(
                posy[j] + ship_lengths[j] <= posy[i] + 
                M_big * ((1 - d_ij) + (1 - e_ij)) + 
                M_big * (1 - L_ij),
                name=f"nonoverlap4_{i}_{j}"
            )
    
    # Mooring constraints
    for i in range(num_ships):
        # Only for lock-mode ships
        # Sum of mooring options >= 1
        sum_moor = moor_left_wall[i] + moor_right_wall[i]
        for j in mooring_sets[i]:
            if j < num_ships:  # Valid ship index
                sum_moor += moor_left_ship[i, j] + moor_right_ship[i, j]
        model.addConstr(sum_moor >= 1 - m[i], name=f"moor_cover_{i}")
        
        # Geometric constraints for mooring
        model.addConstr(posx[i] <= M_big * (1 - moor_left_wall[i]), name=f"moor_left_wall_x_{i}")
        model.addConstr(posx[i] == 0, name=f"moor_left_wall_x_eq_{i}").setAttr("rhs", 0).setAttr("sense", GRB.EQUAL).setAttr("constrname", f"moor_left_wall_x_eq_{i}")
        model.addConstr(posx[i] + ship_widths[i] >= W - M_big * (1 - moor_right_wall[i]), name=f"moor_right_wall_x_{i}")
        model.addConstr(posx[i] + ship_widths[i] == W, name=f"moor_right_wall_x_eq_{i}").setAttr("rhs", W).setAttr("sense", GRB.EQUAL).setAttr("constrname", f"moor_right_wall_x_eq_{i}")
        
        for j in mooring_sets[i]:
            if j < num_ships:
                # Left of j: i is to the left of j, and contained in j's length
                model.addConstr(
                    posx[i] + ship_widths[i] == posx[j] + 
                    M_big * (1 - moor_left_ship[i, j]) + 
                    M_big * (m[i] + m[j]),
                    name=f"moor_left_ship_x_{i}_{j}"
                )
                model.addConstr(posy[i] >= posy[j] - M_big * (1 - moor_left_ship[i, j]), name=f"moor_left_ship_y1_{i}_{j}")
                model.addConstr(posy[i] + ship_lengths[i] <= posy[j] + ship_lengths[j] + M_big * (1 - moor_left_ship[i, j]), name=f"moor_left_ship_y2_{i}_{j}")
                
                # Right of j: j is to the left of i
                model.addConstr(
                    posx[j] + ship_widths[j] == posx[i] + 
                    M_big * (1 - moor_right_ship[i, j]) + 
                    M_big * (m[i] + m[j]),
                    name=f"moor_right_ship_x_{i}_{j}"
                )
                model.addConstr(posy[i] >= posy[j] - M_big * (1 - moor_right_ship[i, j]), name=f"moor_right_ship_y1_{i}_{j}")
                model.addConstr(posy[i] + ship_lengths[i] <= posy[j] + ship_lengths[j] + M_big * (1 - moor_right_ship[i, j]), name=f"moor_right_ship_y2_{i}_{j}")
                
                # Reciprocal mooring constraints
                if i in mooring_sets[j] and j < num_ships and i < num_ships:
                    model.addConstr(moor_left_ship[i, j] + moor_right_ship[j, i] <= 1, name=f"moor_reciprocal1_{i}_{j}")
                    model.addConstr(moor_right_ship[i, j] + moor_left_ship[j, i] <= 1, name=f"moor_reciprocal2_{i}_{j}")
    
    # Transshipment-mode ships: assignment to berths and positions
    for i in range(num_ships):
        # Each transshipment-mode ship assigned to exactly one (berth, position)
        model.addConstr(
            gp.quicksum(z[i, k, p] for k in range(num_berths) for p in range(num_ships)) == m[i],
            name=f"trans_assign_{i}"
        )
    
    for k in range(num_berths):
        for p in range(num_ships):
            # Each berth position has at most one ship
            model.addConstr(
                gp.quicksum(z[i, k, p] for i in range(num_ships)) <= 1,
                name=f"berth_pos_{k}_{p}"
            )
            # assigned[k,p] = 1 if there is a ship at berth k, position p
            model.addConstr(
                assigned[k, p] == gp.quicksum(z[i, k, p] for i in range(num_ships)),
                name=f"assigned_{k}_{p}"
            )
            # Handling time at berth k, position p
            model.addConstr(
                H[k, p] == gp.quicksum(handling_times[i][k] * z[i, k, p] for i in range(num_ships)),
                name=f"H_{k}_{p}"
            )
    
    # Berth scheduling constraints
    for k in range(num_berths):
        # First position
        for i in range(num_ships):
            model.addConstr(
                S[k, 0] >= arrival_times[i] - M_big * (1 - z[i, k, 0]),
                name=f"S_k0_arrival_{k}_{i}"
            )
        model.addConstr(S[k, 0] >= berth_starts[k], name=f"S_k0_berth_start_{k}")
        
        # Subsequent positions
        for p in range(1, num_ships):
            model.addConstr(
                S[k, p] >= S[k, p-1] + H[k, p-1],
                name=f"S_kp_prev_{k}_{p}"
            )
            for i in range(num_ships):
                model.addConstr(
                    S[k, p] >= arrival_times[i] - M_big * (1 - z[i, k, p]),
                    name=f"S_kp_arrival_{k}_{p}_{i}"
                )
        
        # Berth available until berth_end
        for p in range(num_ships):
            model.addConstr(
                S[k, p] + H[k, p] <= berth_end + M_big * (1 - assigned[k, p]),
                name=f"berth_end_{k}_{p}"
            )
    
    # Objective: minimize total cost = delays + penalties
    # Delay for lock-mode ship i: lockage_completion_times[l] - arrival_times[i] (where l is the lockage of i)
    lock_delay = gp.quicksum(
        (gp.quicksum(lockage_completion_times[l] * x[i, l] for l in range(num_ships)) - arrival_times[i]) * (1 - m[i])
        for i in range(num_ships)
    )
    # Delay for transshipment-mode ship i: (S[k,p] + H[k,p]) - arrival_times[i] (for its assigned k,p)
    trans_delay = gp.quicksum(
        gp.quicksum((S[k, p] + H[k, p] - arrival_times[i]) * z[i, k, p] for k in range(num_berths) for p in range(num_ships))
        for i in range(num_ships)
    )
    trans_penalty = gp.quicksum(penalties[i] * m[i] for i in range(num_ships))
    
    total_cost = lock_delay + trans_delay + trans_penalty
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Initialize solution logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Callback function to log intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    # Optimize
    model.optimize(callback if logger else None)
    
    # Extract solution
    solution = {
        "objective_value": model.objVal,
        "modes": {},
        "completion_times": {},
        "lockage_assignments": {},
        "lockage_completion_times": {},
        "berth_assignments": {}
    }
    
    # Modes and completion times
    for i in range(num_ships):
        orig_id = ships_data_sorted[i]['id']
        solution["modes"][str(orig_id)] = int(m[i].X)
        if m[i].X < 0.5:  # Lock mode
            # Departure time = lockage completion time
            lockage = None
            for l in range(num_ships):
                if x[i, l].X > 0.5:
                    lockage = l
                    break
            if lockage is not None:
                solution["lockage_assignments"][str(orig_id)] = lockage
                solution["completion_times"][str(orig_id)] = lockage_completion_times[lockage].X
        else:  # Transshipment mode
            # Departure time = arrival time
            solution["completion_times"][str(orig_id)] = arrival_times[i]
            # Find berth and position
            for k in range(num_berths):
                for p in range(num_ships):
                    if z[i, k, p].X > 0.5:
                        solution["berth_assignments"][str(orig_id)] = {
                            "berth": k,
                            "order": p
                        }
                        break
    
    # Lockage completion times for used lockages
    for l in range(num_ships):
        if u[l].X > 0.5:
            solution["lockage_completion_times"][str(l)] = lockage_completion_times[l].X
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)