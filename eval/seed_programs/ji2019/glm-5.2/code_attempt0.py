import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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

    N = instance["num_ships"]
    B = instance["num_berths"]
    W = instance["chamber_width_W"]
    L = instance["chamber_length_L"]
    Du = instance["lockage_duration_Du"]
    MT = instance["min_time_between_lockages_MT"]
    Sc = instance["lock_start_availability_sc"]
    Ek = instance["berth_end_availability_ek"]
    C_max = instance["C_max"]

    ships = instance["ships"]
    berths = instance["berths"]
    handling_times = instance["handling_times"]

    ship_id_to_idx = {ship["id"]: idx for idx, ship in enumerate(ships)}

    M_x = W + max(s["width"] for s in ships) + 1
    M_y = L + max(s["length"] for s in ships) + 1

    model = gp.Model("ships")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    mode = model.addVars(N, vtype=GRB.BINARY, name="mode")
    lock_assign = model.addVars(N, N, vtype=GRB.BINARY, name="lock_assign")
    berth_assign = model.addVars(N, B, N, vtype=GRB.BINARY, name="berth_assign")
    lockage_used = model.addVars(N, vtype=GRB.BINARY, name="lockage_used")
    lockage_time = model.addVars(N, vtype=GRB.CONTINUOUS, lb=Sc+Du, ub=C_max, name="lockage_time")

    x = model.addVars(N, vtype=GRB.INTEGER, lb=0, ub=W, name="x")
    y = model.addVars(N, vtype=GRB.INTEGER, lb=0, ub=L, name="y")

    left = model.addVars(N, N, vtype=GRB.BINARY, name="left")
    behind = model.addVars(N, N, vtype=GRB.BINARY, name="behind")
    moor_left = model.addVars(N, N, vtype=GRB.BINARY, name="moor_left")
    moor_right = model.addVars(N, N, vtype=GRB.BINARY, name="moor_right")
    wall_left = model.addVars(N, vtype=GRB.BINARY, name="wall_left")
    wall_right = model.addVars(N, vtype=GRB.BINARY, name="wall_right")
    same_lockage = model.addVars(N, N, vtype=GRB.BINARY, name="same_lockage")

    cum_idle = model.addVars(B, N, vtype=GRB.CONTINUOUS, lb=0, ub=C_max, name="cum_idle")
    berth_start = model.addVars(B, N, vtype=GRB.CONTINUOUS, lb=0, ub=Ek, name="berth_start")
    dep = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=C_max, name="dep")

    for i in range(N):
        model.addConstr(mode[i] == sum(lock_assign[i, k] for k in range(N)))
        model.addConstr(mode[i] == sum(berth_assign[i, j, o] for j in range(B) for o in range(N)))
        
        model.addConstr(x[i] + ships[i]["width"] <= W)
        model.addConstr(y[i] + ships[i]["length"] <= L)
        
        model.addConstr(wall_left[i] <= 1 - mode[i])
        model.addConstr(wall_right[i] <= 1 - mode[i])
        model.addConstr(wall_left[i] + wall_right[i] + sum(moor_left[i, j] for j in range(N)) + sum(moor_right[i, j] for j in range(N)) >= 1 - mode[i])
        
        model.addConstr(x[i] <= M_x * (1 - wall_left[i]))
        model.addConstr(x[i] + ships[i]["width"] >= W - M_x * (1 - wall_right[i]))
        model.addConstr(x[i] + ships[i]["width"] <= W + M_x * (1 - wall_right[i]))
        
        MR_i_ids = ships[i]["mooring_set_MR"]
        MR_i_indices = [ship_id_to_idx[id] for id in MR_i_ids if id in ship_id_to_idx]
        
        for j in range(N):
            model.addConstr(moor_left[i, j] <= 1 - mode[i])
            model.addConstr(moor_left[i, j] <= 1 - mode[j])
            model.addConstr(moor_right[i, j] <= 1 - mode[i])
            model.addConstr(moor_right[i, j] <= 1 - mode[j])
            model.addConstr(moor_left[i, j] + moor_right[j, i] <= 1)
            
            if j not in MR_i_indices:
                model.addConstr(moor_left[i, j] == 0)
                model.addConstr(moor_right[i, j] == 0)
            else:
                for k in range(N):
                    model.addConstr(moor_left[i, j] <= 1 - lock_assign[i, k] + lock_assign[j, k])
                    model.addConstr(moor_left[i, j] <= 1 + lock_assign[i, k] - lock_assign[j, k])
                    model.addConstr(moor_right[i, j] <= 1 - lock_assign[i, k] + lock_assign[j, k])
                    model.addConstr(moor_right[i, j] <= 1 + lock_assign[i, k] - lock_assign[j, k])
                
                model.addConstr(x[i] + ships[i]["width"] - x[j] <= M_x * (1 - moor_left[i, j]))
                model.addConstr(x[i] + ships[i]["width"] - x[j] >= -M_x * (1 - moor_left[i, j]))
                model.addConstr(y[i] - y[j] >= -M_y * (1 - moor_left[i, j]))
                model.addConstr(y[i] + ships[i]["length"] - y[j] - ships[j]["length"] <= M_y * (1 - moor_left[i, j]))
                
                model.addConstr(x[j] + ships[j]["width"] - x[i] <= M_x * (1 - moor_right[i, j]))
                model.addConstr(x[j] + ships[j]["width"] - x[i] >= -M_x * (1 - moor_right[i, j]))
                model.addConstr(y[i] - y[j] >= -M_y * (1 - moor_right[i, j]))
                model.addConstr(y[i] + ships[i]["length"] - y[j] - ships[j]["length"] <= M_y * (1 - moor_right[i, j]))

    for i in range(N):
        for j in range(i+1, N):
            for k in range(N):
                model.addConstr(same_lockage[i, j] <= 1 - lock_assign[i, k] + lock_assign[j, k])
                model.addConstr(same_lockage[i, j] <= 1 + lock_assign[i, k] - lock_assign[j, k])
                model.addConstr(same_lockage[i, j] >= lock_assign[i, k] + lock_assign[j, k] - 1)
            model.addConstr(same_lockage[i, j] <= 1 - mode[i])
            model.addConstr(same_lockage[i, j] <= 1 - mode[j])
            
            model.addConstr(x[i] + ships[i]["width"] <= x[j] + M_x * (2 - same_lockage[i, j] - left[i, j]))
            model.addConstr(x[j] + ships[j]["width"] <= x[i] + M_x * (2 - same_lockage[i, j] - left[j, i]))
            model.addConstr(y[i] + ships[i]["length"] <= y[j] + M_y * (2 - same_lockage[i, j] - behind[i, j]))
            model.addConstr(y[j] + ships[j]["length"] <= y[i] + M_y * (2 - same_lockage[i, j] - behind[j, i]))
            model.addConstr(left[i, j] + left[j, i] + behind[i, j] + behind[j, i] >= same_lockage[i, j])

    for k in range(N):
        for i in range(N):
            model.addConstr(lockage_used[k] >= lock_assign[i, k])
        if k > 0:
            model.addConstr(lockage_used[k] <= lockage_used[k-1])
            model.addConstr(lockage_time[k] >= lockage_time[k-1] + MT * lockage_used[k])
        model.addConstr(lockage_time[k] <= C_max * lockage_used[k] + Sc + Du)

    for i in range(N):
        for k in range(N):
            model.addConstr(lockage_time[k] >= ships[i]["arrival_time"] + Du - C_max * (1 - lock_assign[i, k]))
            model.addConstr(dep[i] >= lockage_time[k] - C_max * (1 - lock_assign[i, k]))
            model.addConstr(dep[i] <= lockage_time[k] + C_max * (1 - lock_assign[i, k]))

    for i in range(N):
        for j in range(N):
            if ships[i]["arrival_time"] < ships[j]["arrival_time"]:
                model.addConstr(dep[i] <= dep[j] + C_max * (mode[i] + mode[j]))
                for k in range(N):
                    model.addConstr(lock_assign[i, k] <= sum(lock_assign[j, kp] for kp in range(k+1)) + mode[i] + mode[j])

    for j in range(B):
        sk = berths[j]["start_availability_sk"]
        model.addConstr(cum_idle[j, 0] <= C_max * sum(berth_assign[i, j, 0] for i in range(N)))
        for o in range(N):
            if o > 0:
                model.addConstr(cum_idle[j, o] >= cum_idle[j, o-1])
                model.addConstr(cum_idle[j, o] <= cum_idle[j, o-1] + C_max * sum(berth_assign[i, j, o] for i in range(N)))
            
            model.addConstr(berth_start[j, o] == sk + 
                            sum(handling_times[i][j] * berth_assign[i, j, op] for i in range(N) for op in range(o)) + 
                            cum_idle[j, o])
            
            for i in range(N):
                model.addConstr(berth_start[j, o] >= ships[i]["arrival_time"] - C_max * (1 - berth_assign[i, j, o]))
                model.addConstr(dep[i] >= berth_start[j, o] + handling_times[i][j] - C_max * (1 - berth_assign[i, j, o]))
                model.addConstr(dep[i] <= berth_start[j, o] + handling_times[i][j] + C_max * (1 - berth_assign[i, j, o]))
            
            model.addConstr(sum(berth_assign[i, j, o] for i in range(N)) <= 1)
            if o < N - 1:
                model.addConstr(sum(berth_assign[i, j, o+1] for i in range(N)) <= sum(berth_assign[i, j, o] for i in range(N)))
        
        model.addConstr(sum(handling_times[i][j] * berth_assign[i, j, o] for i in range(N) for o in range(N)) + cum_idle[j, N-1] <= Ek - sk)

    for i in range(N):
        eligible = ships[i]["eligible_berths_B"]
        for j in range(B):
            if berths[j]["id"] not in eligible:
                model.addConstr(sum(berth_assign[i, j, o] for o in range(N)) == 0)

    obj = sum(dep[i] - ships[i]["arrival_time"] for i in range(N)) + sum(mode[i] * ships[i]["transshipment_penalty"] for i in range(N))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.SolCount > 0:
        obj_val = model.ObjVal
        modes = {}
        completion_times = {}
        lockage_assignments = {}
        lockage_completion_times = {}
        berth_assignments = {}
        
        for i in range(N):
            m = round(mode[i].X)
            modes[str(ships[i]["id"])] = m
            completion_times[str(ships[i]["id"])] = dep[i].X
            if m == 0:
                for k in range(N):
                    if round(lock_assign[i, k].X) == 1:
                        lockage_assignments[str(ships[i]["id"])] = k
                        break
            else:
                for j in range(B):
                    for o in range(N):
                        if round(berth_assign[i, j, o].X) == 1:
                            berth_assignments[str(ships[i]["id"])] = {
                                "berth": berths[j]["id"],
                                "order": o
                            }
                            break
                    if str(ships[i]["id"]) in berth_assignments:
                        break
                        
        for k in range(N):
            if round(lockage_used[k].X) == 1:
                lockage_completion_times[str(k)] = lockage_time[k].X
                
        solution = {
            "objective_value": obj_val,
            "modes": modes,
            "completion_times": completion_times,
            "lockage_assignments": lockage_assignments,
            "lockage_completion_times": lockage_completion_times,
            "berth_assignments": berth_assignments
        }
        
        if logger:
            logger.log_solution(obj_val, solution)
            
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()