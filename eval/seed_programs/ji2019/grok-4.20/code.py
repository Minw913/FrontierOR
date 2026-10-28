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
    berths_data = instance['berths']
    handling_times = instance['handling_times']

    widths = [s['width'] for s in ships]
    lengths = [s['length'] for s in ships]
    arrivals = [s['arrival_time'] for s in ships]
    penalties = [s['transshipment_penalty'] for s in ships]
    mooring_allowed = []
    for s in ships:
        mr = s.get('mooring_set_MR', [])
        if isinstance(mr, list):
            mooring_allowed.append(set(int(x) for x in mr))
        else:
            mooring_allowed.append(set())
    eligible_berths_list = []
    for s in ships:
        eb = s.get('eligible_berths_B', list(range(num_berths)))
        if isinstance(eb, list):
            eligible_berths_list.append([int(x) for x in eb])
        else:
            eligible_berths_list.append(list(range(num_berths)))
    handling = handling_times

    berth_starts = [b['start_availability_sk'] for b in berths_data]

    M = C_max + 1000.0
    M_pos = max(chamber_width, chamber_length) + 10.0

    model = gp.Model("ShipLock")
    model.Params.TimeLimit = max(5, args.time_limit - 10)
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0
    model.Params.Presolve = 1
    model.Params.Method = 3
    model.Params.Heuristics = 0.2

    mode = model.addVars(num_ships, vtype=GRB.BINARY, name="mode")
    departure = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="depart")
    lockage_idx = model.addVars(num_ships, vtype=GRB.INTEGER, lb=0, ub=num_ships-1, name="lockage_idx")
    lock_used = model.addVars(num_ships, vtype=GRB.BINARY, name="lock_used")
    lock_comp = model.addVars(num_ships, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="lock_comp")

    berth_assign = model.addVars(num_ships, num_berths, vtype=GRB.BINARY, name="b_assign")
    order_var = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="order")
    b_start = model.addVars(num_ships, lb=0, ub=C_max, name="b_start")
    b_idle = model.addVars(num_ships, lb=0, ub=C_max, name="b_idle")

    x = model.addVars(num_ships, lb=0, ub=chamber_width, name="x")
    y = model.addVars(num_ships, lb=0, ub=chamber_length, name="y")
    left_of = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="left")
    behind_of = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="behind")

    moor_l_ship = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="mls")
    moor_r_ship = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name="mrs")
    moor_l_wall = model.addVars(num_ships, vtype=GRB.BINARY, name="mlw")
    moor_r_wall = model.addVars(num_ships, vtype=GRB.BINARY, name="mrw")

    lock_delay = model.addVars(num_ships, lb=0, ub=C_max, name="ldelay")
    trans_delay = model.addVars(num_ships, lb=0, ub=C_max, name="tdelay")

    obj = (gp.quicksum(lock_delay[i] for i in range(num_ships)) +
           gp.quicksum(trans_delay[i] for i in range(num_ships)) +
           gp.quicksum(penalties[i] * mode[i] for i in range(num_ships)))
    model.setObjective(obj, GRB.MINIMIZE)

    for i in range(num_ships):
        model.addConstr(lock_delay[i] == (1 - mode[i]) * (departure[i] - arrivals[i]))
        model.addConstr(trans_delay[i] == mode[i] * (b_start[i] + handling[i][0] - arrivals[i] + b_idle[i]))

    for i in range(num_ships):
        model.addConstr(departure[i] <= arrivals[i] + (1 - mode[i]) * (C_max - arrivals[i]))
        model.addConstr(departure[i] >= arrivals[i])

    for k in range(num_ships):
        model.addConstr(lock_comp[k] <= C_max * lock_used[k])

    for k in range(1, num_ships):
        model.addConstr(lock_used[k] <= lock_used[k-1])
    model.addConstr(lock_used[0] == 1)

    for i in range(num_ships):
        model.addConstr(lockage_idx[i] <= (num_ships - 1) * (1 - mode[i]))
        for k in range(num_ships):
            model.addConstr((lockage_idx[i] - k) <= (num_ships) * (mode[i] + (1 - lock_used[k])))
            model.addConstr((k - lockage_idx[i]) <= (num_ships) * (mode[i] + (1 - lock_used[k])))

    for k in range(num_ships):
        model.addConstr(lock_comp[k] >= lock_start + lockage_duration - M * (1 - lock_used[k]))
        for i in range(num_ships):
            model.addConstr(lock_comp[k] >= arrivals[i] - M * (mode[i] + (1 if k != 0 else 0)) - M * (1 - (1 - gp.abs_(lockage_idx[i] - k) / float(num_ships + 1))))

    for k in range(1, num_ships):
        model.addConstr(lock_comp[k] >= lock_comp[k-1] + min_time_between - M * (1 - lock_used[k]))

    for i in range(num_ships):
        for j in range(i+1, num_ships):
            if arrivals[i] < arrivals[j]:
                model.addConstr(departure[i] <= departure[j] + M * (mode[i] + mode[j]))
                model.addConstr(lockage_idx[i] <= lockage_idx[j] + num_ships * (mode[i] + mode[j]))

    for i in range(num_ships):
        model.addConstr(x[i] + widths[i] <= chamber_width + M_pos * mode[i])
        model.addConstr(y[i] + lengths[i] <= chamber_length + M_pos * mode[i])

    for i in range(num_ships):
        for j in range(i+1, num_ships):
            diff_idx = gp.abs_(lockage_idx[i] - lockage_idx[j])
            same_lock_bin = model.addVar(vtype=GRB.BINARY, name=f"same_{i}_{j}")
            model.addConstr(same_lock_bin <= 2 - mode[i] - mode[j])
            model.addConstr(same_lock_bin <= 1 - diff_idx / float(num_ships))
            model.addConstr(same_lock_bin >= 0)

            model.addConstr(left_of[i,j] + behind_of[i,j] + left_of[j,i] + behind_of[j,i] >= same_lock_bin)

            model.addConstr(x[i] + widths[i] <= x[j] + M_pos * (1 - left_of[i,j]))
            model.addConstr(x[j] + widths[j] <= x[i] + M_pos * (1 - left_of[j,i]))
            model.addConstr(y[i] + lengths[i] <= y[j] + M_pos * (1 - behind_of[i,j]))
            model.addConstr(y[j] + lengths[j] <= y[i] + M_pos * (1 - behind_of[j,i]))

    for i in range(num_ships):
        sum_moor = moor_l_wall[i] + moor_r_wall[i]
        for j_idx in mooring_allowed[i]:
            j = int(j_idx)
            sum_moor += moor_l_ship[i,j] + moor_r_ship[i,j]
        model.addConstr(sum_moor >= 1 - mode[i])

        model.addConstr(x[i] <= M_pos * (1 - moor_l_wall[i]))
        model.addConstr(x[i] + widths[i] >= chamber_width - M_pos * (1 - moor_r_wall[i]))

        for j_idx in mooring_allowed[i]:
            j = int(j_idx)
            if i == j: continue
            model.addConstr(moor_l_ship[i,j] <= 2 - mode[i] - mode[j])
            model.addConstr(moor_l_ship[i,j] <= 1 - gp.abs_(lockage_idx[i] - lockage_idx[j])/float(num_ships))
            model.addConstr(x[i] + widths[i] >= x[j] - 0.01 - M_pos*(1-moor_l_ship[i,j]))
            model.addConstr(x[i] + widths[i] <= x[j] + 0.01 + M_pos*(1-moor_l_ship[i,j]))
            model.addConstr(y[i] >= y[j] - M_pos*(1-moor_l_ship[i,j]))
            model.addConstr(y[i] + lengths[i] <= y[j] + lengths[j] + M_pos*(1-moor_l_ship[i,j]))

            model.addConstr(moor_r_ship[i,j] <= 2 - mode[i] - mode[j])
            model.addConstr(moor_r_ship[i,j] <= 1 - gp.abs_(lockage_idx[i] - lockage_idx[j])/float(num_ships))
            model.addConstr(x[j] + widths[j] >= x[i] - 0.01 - M_pos*(1-moor_r_ship[i,j]))
            model.addConstr(x[j] + widths[j] <= x[i] + 0.01 + M_pos*(1-moor_r_ship[i,j]))
            model.addConstr(y[i] >= y[j] - M_pos*(1-moor_r_ship[i,j]))
            model.addConstr(y[i] + lengths[i] <= y[j] + lengths[j] + M_pos*(1-moor_r_ship[i,j]))

    for i in range(num_ships):
        for j in range(i+1, num_ships):
            model.addConstr(moor_l_ship[i,j] + moor_r_ship[j,i] <= 1)
            model.addConstr(moor_r_ship[i,j] + moor_l_ship[j,i] <= 1)

    for i in range(num_ships):
        model.addConstr(gp.quicksum(berth_assign[i, b] for b in range(num_berths)) == mode[i])
        for b_idx in range(num_berths):
            if b_idx not in eligible_berths_list[i]:
                model.addConstr(berth_assign[i, b_idx] == 0)
            else:
                model.addConstr(b_start[i] >= berth_starts[b_idx] - M * (1 - berth_assign[i, b_idx]))

    for i in range(num_ships):
        for j in range(num_ships):
            if i != j:
                same_berth = gp.quicksum(berth_assign[i, b] * berth_assign[j, b] for b in range(num_berths))
                model.addConstr(order_var[i, j] <= same_berth)
                model.addConstr(b_start[i] + handling[i][0] + b_idle[i] <= b_start[j] + M * (1 - order_var[i, j]))

    for i in range(num_ships):
        cum_before = gp.quicksum((handling[j][0] + b_idle[j]) * order_var[j, i] for j in range(num_ships) if j != i)
        model.addConstr(b_start[i] >= arrivals[i] - M * (1 - mode[i]))
        model.addConstr(b_idle[i] >= arrivals[i] - b_start[i] - cum_before - M * (1 - mode[i]))
        model.addConstr(b_idle[i] >= 0)

    for b in range(num_berths):
        total_time = gp.quicksum((handling[i][b] + b_idle[i]) * berth_assign[i, b] for i in range(num_ships))
        model.addConstr(total_time <= berth_end - berth_starts[b] + 0.1)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objv)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.objVal
        mode_vals = [int(round(mode[i].X)) for i in range(num_ships)]
        dep_vals = [float(departure[i].X) for i in range(num_ships)]
        lidx_vals = [int(round(lockage_idx[i].X)) for i in range(num_ships)]
        lc_vals = [float(lock_comp[k].X) for k in range(num_ships)]

        modes_out = {str(i): mode_vals[i] for i in range(num_ships)}
        comp_out = {}
        for i in range(num_ships):
            if mode_vals[i] == 0:
                comp_out[str(i)] = max(dep_vals[i], float(arrivals[i]))
            else:
                comp_out[str(i)] = float(arrivals[i])

        lock_assign_out = {}
        used_l = set()
        for i in range(num_ships):
            if mode_vals[i] == 0:
                lidx = lidx_vals[i]
                lock_assign_out[str(i)] = int(lidx)
                used_l.add(lidx)
            else:
                lock_assign_out[str(i)] = 0

        lock_comp_out = {}
        for k_idx, k in enumerate(sorted(used_l)):
            lock_comp_out[str(k)] = lc_vals[k]

        berth_out = {}
        for i in range(num_ships):
            if mode_vals[i] == 1:
                b_assigned = 0
                for b_val in range(num_berths):
                    if berth_assign[i, b_val].X > 0.4:
                        b_assigned = b_val
                        break
                ord_pos = 0
                for j_val in range(num_ships):
                    if j_val != i and mode_vals[j_val] == 1 and order_var[j_val, i].X > 0.4:
                        ord_pos += 1
                berth_out[str(i)] = {"berth": int(b_assigned), "order": int(ord_pos)}

        solution = {
            "objective_value": float(obj_val),
            "modes": modes_out,
            "completion_times": comp_out,
            "lockage_assignments": lock_assign_out,
            "lockage_completion_times": lock_comp_out,
            "berth_assignments": berth_out
        }
    else:
        solution = {
            "objective_value": 999999.0,
            "modes": {str(i): 0 for i in range(num_ships)},
            "completion_times": {str(i): float(arrivals[i]) for i in range(num_ships)},
            "lockage_assignments": {str(i): 0 for i in range(num_ships)},
            "lockage_completion_times": {"0": float(lock_start + lockage_duration)},
            "berth_assignments": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()