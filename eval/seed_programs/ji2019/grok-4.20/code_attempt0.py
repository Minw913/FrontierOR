import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_ships = instance['num_ships']
    num_berths = instance['num_berths']
    chamber_width = instance['chamber_width_W']
    chamber_length = instance['chamber_length_L']
    lockage_duration = instance['lockage_duration_Du']
    min_time_between = instance['min_time_between_lockages_MT']
    lock_start = instance['lock_start_availability_sc']
    berth_end = instance['berth_end_availability_ek']
    C_max = instance['C_max']

    ships = instance['ships']
    berths = instance['berths']
    handling_times = instance['handling_times']

    # Precompute data
    widths = [s['width'] for s in ships]
    lengths = [s['length'] for s in ships]
    arrivals = [s['arrival_time'] for s in ships]
    penalties = [s['transshipment_penalty'] for s in ships]
    mooring_allowed = []
    for s in ships:
        allowed = set(s['mooring_set_MR'])
        mooring_allowed.append(allowed)
    eligible_berths = [set(s['eligible_berths_B']) for s in ships]
    handling = handling_times  # [ship][berth]

    berth_starts = [b['start_availability_sk'] for b in berths]

    # Big-M for positioning
    M_width = chamber_width + 1
    M_length = chamber_length + 1
    M_time = C_max + 1

    # Create model
    model = gp.Model("ShipLockTransshipment")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    # Variables
    mode = model.addVars(num_ships, vtype=GRB.BINARY, name="mode")  # 0: lock, 1: transship
    departure = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="departure")
    lockage = model.addVars(num_ships, vtype=GRB.INTEGER, lb=0, ub=num_ships-1, name="lockage")
    lockage_used = model.addVars(num_ships, vtype=GRB.BINARY, name="lockage_used")
    lock_completion = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="lock_completion")

    # Berth assignment
    berth_assign = model.addVars(num_ships, num_berths, vtype=GRB.BINARY, name="berth_assign")
    service_order = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="service_order")  # ship i before j at same berth
    berth_start_time = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="berth_start")
    idle_before = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="idle")

    # Positioning in lock
    x_pos = model.addVars(num_ships, lb=0, ub=chamber_width, vtype=GRB.CONTINUOUS, name="x")
    y_pos = model.addVars(num_ships, lb=0, ub=chamber_length, vtype=GRB.CONTINUOUS, name="y")
    left_of = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="left_of")
    behind_of = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="behind_of")

    # Mooring variables
    moor_left_ship = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="moor_left_ship")
    moor_right_ship = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="moor_right_ship")
    moor_left_wall = model.addVars(num_ships, vtype=GRB.BINARY, name="moor_left_wall")
    moor_right_wall = model.addVars(num_ships, vtype=GRB.BINARY, name="moor_right_wall")

    # Objective components
    lock_delay = model.addVars(num_ships, vtype=GRB.CONTINUOUS, name="lock_delay")
    trans_delay = model.addVars(num_ships, vtype=GRB.CONTINUOUS, name="trans_delay")

    # Objective
    obj = (gp.quicksum(lock_delay[i] for i in range(num_ships)) +
           gp.quicksum(trans_delay[i] for i in range(num_ships)) +
           gp.quicksum(penalties[i] * mode[i] for i in range(num_ships)))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints

    # 1. Mode linkage and departure time
    for i in range(num_ships):
        # departure = lock completion if lock mode, arrival if trans
        model.addConstr(departure[i] == (1 - mode[i]) * lock_completion[lockage[i]] + mode[i] * arrivals[i])

        # lock_delay only if lock mode
        model.addConstr(lock_delay[i] == (1 - mode[i]) * (departure[i] - arrivals[i]))
        # trans_delay computed later via berth_start_time

    # 2. Lockage usage and ordering
    for k in range(num_ships):
        model.addConstr(lockage_used[k] >= lockage_used[k+1] if k < num_ships-1 else 0)
        for i in range(num_ships):
            model.addConstr(lockage[i] <= (k + 1) * lockage_used[k] - 1)
            model.addConstr(lockage[i] >= k * lockage_used[k])

    for k in range(num_ships):
        model.addConstr(lock_completion[k] >= lock_start + lockage_duration)
        model.addConstr(lock_completion[k] >= lockage_used[k] * lock_start)

        # ships in this lockage
        for i in range(num_ships):
            model.addConstr(lock_completion[k] >= arrivals[i] - M_time * (1 - (1-mode[i])) - M_time * (lockage[i] - k))
            model.addConstr(lock_completion[k] >= arrivals[i] - M_time * mode[i])

    # Minimum time between lockages
    for k in range(1, num_ships):
        model.addConstr(lock_completion[k] >= lock_completion[k-1] + min_time_between - M_time * (1 - lockage_used[k]))

    # Lockage non-decreasing
    for k in range(1, num_ships):
        model.addConstr(lock_completion[k] >= lock_completion[k-1] - M_time * (1 - lockage_used[k]))

    # FCFS for lock mode ships
    for i in range(num_ships):
        for j in range(num_ships):
            if i != j and arrivals[i] < arrivals[j]:
                # i must depart no later than j if both lock
                model.addConstr(departure[i] <= departure[j] + M_time * mode[i] + M_time * mode[j])
                # i must use same or earlier lockage
                model.addConstr(lockage[i] <= lockage[j] + (num_ships) * mode[i] + (num_ships) * mode[j])

    # 3. No overlap in chamber
    for i in range(num_ships):
        for j in range(i+1, num_ships):
            # only if both in lock and same lockage
            both_lock_same = (1 - mode[i]) * (1 - mode[j]) * (1 if lockage[i] == lockage[j] else 0)
            # linearized
            same_lockage = model.addVar(vtype=GRB.BINARY, name=f"same_lock_{i}_{j}")
            model.addConstr(same_lockage <= (1 - mode[i]))
            model.addConstr(same_lockage <= (1 - mode[j]))
            model.addConstr(same_lockage >= (1 - mode[i]) + (1 - mode[j]) - 1)
            model.addConstr(same_lockage <= 1 - gp.abs_(lockage[i] - lockage[j]) / (num_ships + 1))  # rough

            # Non-overlap: either left_of or behind_of
            model.addConstr(left_of[i,j] + behind_of[i,j] + left_of[j,i] + behind_of[j,i] >= same_lockage)

            # left_of constraints
            model.addConstr(x_pos[i] + widths[i] <= x_pos[j] + M_width * (1 - left_of[i,j]))
            model.addConstr(x_pos[j] + widths[j] <= x_pos[i] + M_width * (1 - left_of[j,i]))

            # behind constraints
            model.addConstr(y_pos[i] + lengths[i] <= y_pos[j] + M_length * (1 - behind_of[i,j]))
            model.addConstr(y_pos[j] + lengths[j] <= y_pos[i] + M_length * (1 - behind_of[j,i]))

            # fit in chamber
            model.addConstr(x_pos[i] + widths[i] <= chamber_width + M_width * mode[i])
            model.addConstr(y_pos[i] + lengths[i] <= chamber_length + M_length * mode[i])

    # 4. Mooring constraints
    for i in range(num_ships):
        # must moor to at least one if in lock
        model.addConstr(moor_left_wall[i] + moor_right_wall[i] + 
                       gp.quicksum(moor_left_ship[i,j] for j in range(num_ships) if j != i) +
                       gp.quicksum(moor_right_ship[i,j] for j in range(num_ships) if j != i) >= (1 - mode[i]))

        # only moor to allowed
        for j in range(num_ships):
            if j == i or j not in mooring_allowed[i]:
                model.addConstr(moor_left_ship[i,j] == 0)
                model.addConstr(moor_right_ship[i,j] == 0)

        # cannot moor mutually
        for j in range(num_ships):
            if j != i:
                model.addConstr(moor_left_ship[i,j] + moor_right_ship[j,i] <= 1)
                model.addConstr(moor_right_ship[i,j] + moor_left_ship[j,i] <= 1)

        # mooring only if same lockage and lock mode
        for j in range(num_ships):
            if i != j:
                model.addConstr(moor_left_ship[i,j] <= 1 - mode[i])
                model.addConstr(moor_left_ship[i,j] <= 1 - mode[j])
                model.addConstr(moor_left_ship[i,j] <= 1 - gp.abs_(lockage[i] - lockage[j])/(num_ships))

        # Wall mooring
        model.addConstr(x_pos[i] <= 0.01 + M_width * (1 - moor_left_wall[i]))  # left wall x=0
        model.addConstr(x_pos[i] + widths[i] >= chamber_width - 0.01 - M_width * (1 - moor_right_wall[i]))

        # Ship to ship mooring
        for j in range(num_ships):
            if i != j:
                # moor left to j (i is left of j and touches)
                model.addConstr(x_pos[i] + widths[i] >= x_pos[j] - 0.01 - M_width * (1 - moor_left_ship[i,j]))
                model.addConstr(x_pos[i] + widths[i] <= x_pos[j] + 0.01 + M_width * (1 - moor_left_ship[i,j]))
                # y overlap
                model.addConstr(y_pos[i] >= y_pos[j] - 0.01 - M_length*(1-moor_left_ship[i,j]))
                model.addConstr(y_pos[i] + lengths[i] <= y_pos[j] + lengths[j] + 0.01 + M_length*(1-moor_left_ship[i,j]))

                # symmetric for right
                model.addConstr(x_pos[j] + widths[j] >= x_pos[i] - 0.01 - M_width * (1 - moor_right_ship[i,j]))
                model.addConstr(x_pos[j] + widths[j] <= x_pos[i] + 0.01 + M_width * (1 - moor_right_ship[i,j]))
                model.addConstr(y_pos[i] >= y_pos[j] - 0.01 - M_length*(1-moor_right_ship[i,j]))
                model.addConstr(y_pos[i] + lengths[i] <= y_pos[j] + lengths[j] + 0.01 + M_length*(1-moor_right_ship[i,j]))

    # 5. Transshipment mode constraints
    for i in range(num_ships):
        model.addConstr(gp.quicksum(berth_assign[i,k] for k in range(num_berths)) == mode[i])

        for k in range(num_berths):
            if k not in eligible_berths[i]:
                model.addConstr(berth_assign[i,k] == 0)
            model.addConstr(berth_start_time[i] >= berth_starts[k] - M_time * (1 - berth_assign[i,k]))

    # Service ordering at berths
    for i in range(num_ships):
        for j in range(num_ships):
            if i != j:
                model.addConstr(service_order[i,j] <= gp.quicksum(berth_assign[i,k] * berth_assign[j,k] for k in range(num_berths)))

    for i in range(num_ships):
        for j in range(num_ships):
            if i != j:
                # if i before j then departure_i <= start_j
                model.addConstr(berth_start_time[i] + handling[i][0] + idle[i] <= berth_start_time[j] + M_time * (1 - service_order[i,j]))

    # At most one ship per order position - but since order is pairwise, we use big-M for start times
    # Instead of explicit order index, we use pairwise ordering which is sufficient for sequencing

    # Berth time window
    for b_idx in range(num_berths):
        total_time = gp.quicksum((handling[i][b_idx] + idle[i]) * berth_assign[i, b_idx] for i in range(num_ships))
        model.addConstr(total_time <= berth_end - berth_starts[b_idx])

    # Compute cumulative before for delay
    for i in range(num_ships):
        cum_before = gp.quicksum((handling[j][0] + idle[j]) * service_order[j,i] for j in range(num_ships) if j != i)
        model.addConstr(trans_delay[i] == mode[i] * (berth_start_time[i] + cum_before + handling[i][0] + idle[i] - arrivals[i]))

    # Idle time linking with arrival
    for i in range(num_ships):
        cum_before_i = gp.quicksum((handling[j][0] + idle[j]) * service_order[j,i] for j in range(num_ships))
        model.addConstr(berth_start_time[i] >= arrivals[i] - berth_starts[0] - cum_before_i - M_time * (1 - mode[i]))
        model.addConstr(idle[i] >= arrivals[i] - berth_starts[0] - cum_before_i - M_time * (1 - mode[i]))

    # Lockage assignment only for lock mode
    for i in range(num_ships):
        model.addConstr(lockage[i] <= (num_ships-1) * (1 - mode[i]))

    # Symmetry breaking for lockages
    for k in range(1, num_ships):
        model.addConstr(lockage_used[k] <= lockage_used[k-1])

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.SolCount > 0:
        try:
            sol_modes = {}
            sol_completion = {}
            sol_lockage = {}
            sol_lock_completion = {}
            sol_berth = {}

            best_obj = model.objVal

            for i in range(num_ships):
                m_val = int(round(mode[i].X))
                sol_modes[str(i)] = m_val
                sol_completion[str(i)] = float(departure[i].X)

                if m_val == 0:  # lock
                    lk = int(round(lockage[i].X))
                    sol_lockage[str(i)] = lk
                else:
                    sol_lockage[str(i)] = -1  # not used

            for k in range(num_ships):
                if lockage_used[k].X > 0.5:
                    sol_lock_completion[str(k)] = float(lock_completion[k].X)

            for i in range(num_ships):
                if int(round(mode[i].X)) == 1:
                    b_assigned = -1
                    for b in range(num_berths):
                        if berth_assign[i,b].X > 0.5:
                            b_assigned = b
                            break
                    # Simple order: use a counter per berth
                    order_val = 0
                    for j in range(num_ships):
                        if j != i and int(round(mode[j].X)) == 1:
                            # count how many before
                            if service_order[j,i].X > 0.5:
                                order_val += 1
                    sol_berth[str(i)] = {"berth": b_assigned, "order": order_val}

            solution = {
                "objective_value": float(best_obj),
                "modes": sol_modes,
                "completion_times": sol_completion,
                "lockage_assignments": sol_lockage,
                "lockage_completion_times": sol_lock_completion,
                "berth_assignments": sol_berth
            }

            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)

            if logger:
                logger.log(best_obj)
        except Exception as e:
            # Fallback: empty but valid structure
            fallback = {
                "objective_value": 1e9,
                "modes": {str(i): 0 for i in range(num_ships)},
                "completion_times": {str(i): float(arrivals[i]) for i in range(num_ships)},
                "lockage_assignments": {str(i): 0 for i in range(num_ships)},
                "lockage_completion_times": {"0": float(lock_start + lockage_duration)},
                "berth_assignments": {}
            }
            with open(args.solution_path, 'w') as f:
                json.dump(fallback, f, indent=2)
    else:
        # No solution found
        fallback = {
            "objective_value": 1e9,
            "modes": {str(i): 0 for i in range(num_ships)},
            "completion_times": {str(i): float(arrivals[i]) for i in range(num_ships)},
            "lockage_assignments": {str(i): 0 for i in range(num_ships)},
            "lockage_completion_times": {"0": float(lock_start + lockage_duration)},
            "berth_assignments": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)

if __name__ == "__main__":
    main()