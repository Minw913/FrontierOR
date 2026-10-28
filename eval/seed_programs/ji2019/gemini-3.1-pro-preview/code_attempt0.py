import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    
    args = parser.parse_args()
    
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # 1. Parse parameters
    ships = data["ships"]
    berths = data["berths"]
    num_ships = data["num_ships"]
    W = data["chamber_width_W"]
    L = data["chamber_length_L"]
    Du = data["lockage_duration_Du"]
    MT = data["min_time_between_lockages_MT"]
    lock_start = data["lock_start_availability_sc"]
    berth_end = data["berth_end_availability_ek"]
    C_max = data["C_max"]

    ship_ids = [s["id"] for s in ships]
    berth_ids = [b["id"] for b in berths]

    H = {}
    for i, s in enumerate(ships):
        H[s["id"]] = {}
        for j, b in enumerate(berths):
            H[s["id"]][b["id"]] = data["handling_times"][i][j]

    MaxW = max([s["width"] for s in ships] + [0])
    MaxL = max([s["length"] for s in ships] + [0])
    MW = W + MaxW
    MLen = L + MaxL

    m_idx = list(range(1, num_ships + 1))
    p_idx = list(range(1, num_ships + 1))

    # 2. Build model
    model = gp.Model("LockAndBerthScheduling")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Mode choice
    z = {s["id"]: model.addVar(vtype=GRB.BINARY, name=f"z_{s['id']}") for s in ships}

    # Lock variables
    y_lock = {m: model.addVar(vtype=GRB.BINARY, name=f"y_lock_{m}") for m in m_idx}
    c_lock = {m: model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=C_max, name=f"c_lock_{m}") for m in m_idx}

    u = {}
    for s in ships:
        for m in m_idx:
            u[s["id"], m] = model.addVar(vtype=GRB.BINARY, name=f"u_{s['id']}_{m}")

    x = {s["id"]: model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=W, name=f"x_{s['id']}") for s in ships}
    y = {s["id"]: model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=L, name=f"y_{s['id']}") for s in ships}

    left = {}
    behind = {}
    for i in range(num_ships):
        si = ships[i]["id"]
        for j in range(i + 1, num_ships):
            sj = ships[j]["id"]
            left[si, sj] = model.addVar(vtype=GRB.BINARY, name=f"left_{si}_{sj}")
            left[sj, si] = model.addVar(vtype=GRB.BINARY, name=f"left_{sj}_{si}")
            behind[si, sj] = model.addVar(vtype=GRB.BINARY, name=f"behind_{si}_{sj}")
            behind[sj, si] = model.addVar(vtype=GRB.BINARY, name=f"behind_{sj}_{si}")

    moor_LW = {s["id"]: model.addVar(vtype=GRB.BINARY, name=f"mLW_{s['id']}") for s in ships}
    moor_RW = {s["id"]: model.addVar(vtype=GRB.BINARY, name=f"mRW_{s['id']}") for s in ships}
    ML = {}
    MR = {}
    for s in ships:
        si = s["id"]
        for sj in s["mooring_set_MR"]:
            ML[si, sj] = model.addVar(vtype=GRB.BINARY, name=f"ML_{si}_{sj}")
            MR[si, sj] = model.addVar(vtype=GRB.BINARY, name=f"MR_{si}_{sj}")

    # Berth variables
    v = {}
    for s in ships:
        for k in berth_ids:
            for p in p_idx:
                v[s["id"], k, p] = model.addVar(vtype=GRB.BINARY, name=f"v_{s['id']}_{k}_{p}")

    BS = {}
    BC = {}
    for k in berth_ids:
        for p in p_idx:
            BS[k, p] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=C_max, name=f"BS_{k}_{p}")
            BC[k, p] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=C_max, name=f"BC_{k}_{p}")

    # Delays
    D_lock = {s["id"]: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"D_lock_{s['id']}") for s in ships}
    D_trans = {s["id"]: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"D_trans_{s['id']}") for s in ships}

    model.update()

    # 3. Constraints
    for s in ships:
        si = s["id"]
        W_i, L_i, A_i = s["width"], s["length"], s["arrival_time"]
        
        # Mode distribution mapping
        model.addConstr(gp.quicksum(v[si, k, p] for k in berth_ids for p in p_idx) == z[si])
        model.addConstr(gp.quicksum(u[si, m] for m in m_idx) == 1 - z[si])
        
        if len(s["eligible_berths_B"]) == 0:
            model.addConstr(z[si] == 0)
        else:
            for k in berth_ids:
                if k not in s["eligible_berths_B"]:
                    for p in p_idx:
                        model.addConstr(v[si, k, p] == 0)
                        
        # Chamber fitting
        model.addConstr(x[si] + W_i <= W + MW * z[si])
        model.addConstr(y[si] + L_i <= L + MLen * z[si])

        # Mooring limits bounds
        model.addConstr(x[si] <= MW * (1 - moor_LW[si]))
        model.addConstr(x[si] + W_i >= W - MW * (1 - moor_RW[si]))
        
        model.addConstr(moor_LW[si] <= 1 - z[si])
        model.addConstr(moor_RW[si] <= 1 - z[si])

        valid_ML = [ML[si, sj] for sj in s["mooring_set_MR"]]
        valid_MR = [MR[si, sj] for sj in s["mooring_set_MR"]]
        model.addConstr(moor_LW[si] + moor_RW[si] + gp.quicksum(valid_ML) + gp.quicksum(valid_MR) >= 1 - z[si])
        
        for sj in s["mooring_set_MR"]:
            model.addConstr(ML[si, sj] <= 1 - z[si])
            model.addConstr(ML[si, sj] <= 1 - z[sj])
            model.addConstr(MR[si, sj] <= 1 - z[si])
            model.addConstr(MR[si, sj] <= 1 - z[sj])
            
            s_j = next(sx for sx in ships if sx["id"] == sj)
            W_j, L_j = s_j["width"], s_j["length"]
            
            # Left mooring logic
            model.addConstr(x[si] + W_i >= x[sj] - MW * (1 - ML[si, sj]))
            model.addConstr(x[si] + W_i <= x[sj] + MW * (1 - ML[si, sj]))
            model.addConstr(y[si] >= y[sj] - MLen * (1 - ML[si, sj]))
            model.addConstr(y[si] + L_i <= y[sj] + L_j + MLen * (1 - ML[si, sj]))
            for m in m_idx:
                model.addConstr(u[si, m] - u[sj, m] + ML[si, sj] <= 1)
                model.addConstr(u[sj, m] - u[si, m] + ML[si, sj] <= 1)
                
            # Right mooring logic
            model.addConstr(x[si] <= x[sj] + W_j + MW * (1 - MR[si, sj]))
            model.addConstr(x[si] >= x[sj] + W_j - MW * (1 - MR[si, sj]))
            model.addConstr(y[si] >= y[sj] - MLen * (1 - MR[si, sj]))
            model.addConstr(y[si] + L_i <= y[sj] + L_j + MLen * (1 - MR[si, sj]))
            for m in m_idx:
                model.addConstr(u[si, m] - u[sj, m] + MR[si, sj] <= 1)
                model.addConstr(u[sj, m] - u[si, m] + MR[si, sj] <= 1)
                
    for i in range(num_ships):
        si = ships[i]["id"]
        for j in range(i + 1, num_ships):
            sj = ships[j]["id"]
            if sj in ships[i]["mooring_set_MR"] and si in ships[j]["mooring_set_MR"]:
                model.addConstr(ML[si, sj] + MR[sj, si] <= 1)
                model.addConstr(MR[si, sj] + ML[sj, si] <= 1)

    for i in range(num_ships):
        si = ships[i]["id"]
        W_i, L_i = ships[i]["width"], ships[i]["length"]
        for j in range(i + 1, num_ships):
            sj = ships[j]["id"]
            W_j, L_j = ships[j]["width"], ships[j]["length"]
            
            for m in m_idx:
                model.addConstr(left[si, sj] + left[sj, si] + behind[si, sj] + behind[sj, si] >= u[si, m] + u[sj, m] - 1)
                
            model.addConstr(x[si] + W_i <= x[sj] + MW * (1 - left[si, sj]))
            model.addConstr(x[sj] + W_j <= x[si] + MW * (1 - left[sj, si]))
            model.addConstr(y[si] + L_i <= y[sj] + MLen * (1 - behind[si, sj]))
            model.addConstr(y[sj] + L_j <= y[si] + MLen * (1 - behind[sj, si]))

    for i in range(num_ships):
        si = ships[i]["id"]
        A_i = ships[i]["arrival_time"]
        for j in range(num_ships):
            if i == j: continue
            sj = ships[j]["id"]
            A_j = ships[j]["arrival_time"]
            if A_i < A_j:
                expr_mi = gp.quicksum(m * u[si, m] for m in m_idx)
                expr_mj = gp.quicksum(m * u[sj, m] for m in m_idx)
                model.addConstr(expr_mi <= expr_mj + num_ships * (z[si] + z[sj]))

    model.addConstr(y_lock[1] <= num_ships)
    for m in m_idx:
        model.addConstr(y_lock[m] <= gp.quicksum(u[s["id"], m] for s in ships))
        for s in ships:
            model.addConstr(u[s["id"], m] <= y_lock[m])
            
        if m < num_ships:
            model.addConstr(y_lock[m] >= y_lock[m+1])
            model.addConstr(c_lock[m] <= c_lock[m+1])

    for m in m_idx:
        if m > 1:
            model.addConstr(c_lock[m] >= c_lock[m-1] + MT * y_lock[m])
        else:
            model.addConstr(c_lock[m] >= lock_start + Du * y_lock[m])
            
        for s in ships:
            model.addConstr(c_lock[m] >= s["arrival_time"] + Du - C_max * (1 - u[s["id"], m]))

    for s in ships:
        si = s["id"]
        for m in m_idx:
            model.addConstr(D_lock[si] >= c_lock[m] - s["arrival_time"] - C_max * (1 - u[si, m]))

    b_starts = {b["id"]: b["start_availability_sk"] for b in berths}
    for k in berth_ids:
        model.addConstr(BS[k, 1] >= b_starts[k])
        for p in p_idx:
            model.addConstr(gp.quicksum(v[s["id"], k, p] for s in ships) <= 1)
            if p > 1:
                model.addConstr(gp.quicksum(v[s["id"], k, p] for s in ships) <= gp.quicksum(v[s["id"], k, p-1] for s in ships))
                model.addConstr(BS[k, p] >= BC[k, p-1])
                
            model.addConstr(BC[k, p] == BS[k, p] + gp.quicksum(H[s["id"]][k] * v[s["id"], k, p] for s in ships))
            model.addConstr(BC[k, p] <= berth_end)
            
            for s in ships:
                si = s["id"]
                model.addConstr(BS[k, p] >= s["arrival_time"] - C_max * (1 - v[si, k, p]))
                model.addConstr(D_trans[si] >= BC[k, p] - s["arrival_time"] - C_max * (1 - v[si, k, p]))

    objective = gp.quicksum(D_lock[s["id"]] + D_trans[s["id"]] + s["transshipment_penalty"] * z[s["id"]] for s in ships)
    model.setObjective(objective, GRB.MINIMIZE)

    # 4. Solution Logging Callback
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(cb)

    # 5. Extract results
    if model.SolCount > 0:
        modes = {}
        completion_times = {}
        lockage_assignments = {}
        lockage_completion_times = {}
        berth_assignments = {}

        for s in ships:
            si = s["id"]
            is_trans = (z[si].X > 0.5)
            modes[str(si)] = 1 if is_trans else 0
            if is_trans:
                completion_times[str(si)] = float(s["arrival_time"])
                for k in berth_ids:
                    for p in p_idx:
                        if v[si, k, p].X > 0.5:
                            berth_assignments[str(si)] = {"berth": k, "order": p}
                            break
            else:
                selected_m = next(m for m in m_idx if u[si, m].X > 0.5)
                lockage_assignments[str(si)] = selected_m
                completion_times[str(si)] = float(c_lock[selected_m].X)

        for m in m_idx:
            if y_lock[m].X > 0.5:
                lockage_completion_times[str(m)] = float(c_lock[m].X)

        output = {
            "objective_value": float(model.ObjVal),
            "modes": modes,
            "completion_times": completion_times,
            "lockage_assignments": lockage_assignments,
            "lockage_completion_times": lockage_completion_times,
            "berth_assignments": berth_assignments
        }

        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()