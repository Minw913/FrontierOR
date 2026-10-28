import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Optimize Surgery Scheduling")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path for the output JSON solution")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the log file")
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # 1. Load Data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_scenarios = max(1, data.get('num_scenarios', 1))
    num_ors = data['num_ors_per_hospital']
    ors = list(range(num_ors))
    
    hospital_ids = []
    days = []
    day_label_to_id = {}
    B_hd = {}
    F_hd = {}
    G_hd = {}

    for h_info in data['hospitals']:
        h_id = h_info['hospital_id']
        hospital_ids.append(h_id)
        for d_info in h_info['days']:
            d_id = d_info['day_id']
            if d_id not in days:
                days.append(d_id)
            B_hd[(h_id, d_id)] = d_info['B_hd']
            F_hd[(h_id, d_id)] = d_info['F_hd']
            G_hd[(h_id, d_id)] = d_info['G_hd']
            day_label_to_id[str(d_info['day_label'])] = d_info['day_id']

    mandatory_patients = []
    non_mandatory_patients = []
    patient_ids = []
    c_sched = defaultdict(dict)
    c_unsched = {}
    c_cancel = {}

    for p in data['patients']:
        p_id = p['patient_id']
        patient_ids.append(p_id)
        if p.get('is_mandatory', False):
            mandatory_patients.append(p_id)
        else:
            non_mandatory_patients.append(p_id)
            
        c_unsched[p_id] = p.get('c_unsched', 0.0)
        c_cancel[p_id] = p.get('c_cancel', 0.0)
        
        for d_lbl, val in p.get('c_sched_per_day', {}).items():
            if str(d_lbl) in day_label_to_id:
                d_id = day_label_to_id[str(d_lbl)]
                c_sched[p_id][d_id] = val
            else:
                try:
                    c_sched[p_id][int(d_lbl)] = val
                except ValueError:
                    pass

    scenario_ids = []
    durations = {}
    for s in data['scenarios']:
        s_id = s['scenario_id']
        scenario_ids.append(s_id)
        durations[s_id] = {}
        # Assumes scenario list order strictly matches patient list order
        for i, dur in enumerate(s['surgery_durations_minutes']):
            p_id = data['patients'][i]['patient_id']
            durations[s_id][p_id] = dur

    # 2. Setup Gurobi Model
    # Suppress console logging to stay quiet unless needed, environment automatically uses 1 thread limits if containerized.
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    model = gp.Model("SurgeryScheduling", env=env)
    
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1) 

    # 3. Variables & Objective Constants construction
    u_obj = {}
    for h in hospital_ids:
        for d in days:
            u_obj[(h, d)] = G_hd[(h, d)]
    u = model.addVars(hospital_ids, days, vtype=GRB.BINARY, obj=u_obj)

    y_obj = {}
    for h in hospital_ids:
        for d in days:
            for o in ors:
                y_obj[(h, d, o)] = F_hd[(h, d)]
    y = model.addVars(hospital_ids, days, ors, vtype=GRB.BINARY, obj=y_obj)

    w_obj = {p: c_unsched[p] for p in non_mandatory_patients}
    w = model.addVars(non_mandatory_patients, vtype=GRB.BINARY, obj=w_obj)

    # Pre-select valid combos to radically reduce dimensionality 
    x_obj_dict = {}
    valid_x_tuples = []
    for h_id in hospital_ids:
        for d_id in days:
            for o_id in ors:
                for p_id in patient_ids:
                    if d_id in c_sched[p_id]:
                        tup = (h_id, d_id, o_id, p_id)
                        valid_x_tuples.append(tup)
                        x_obj_dict[tup] = c_sched[p_id][d_id] + c_cancel[p_id]

    x = model.addVars(valid_x_tuples, vtype=GRB.BINARY, obj=x_obj_dict)

    valid_x_set = set(valid_x_tuples)
    P_obj_dict = {}
    valid_P_tuples = []

    for s_id in scenario_ids:
        for h_id in hospital_ids:
            for d_id in days:
                B = B_hd[(h_id, d_id)]
                for p_id in patient_ids:
                    # Ignore patients directly filtered because of impossible runtime guarantees under capacities
                    if durations[s_id].get(p_id, 0) <= B:
                        # Use symmetry logic: if it's feasible for OR 0, it operates for all identical ones
                        if (h_id, d_id, ors[0], p_id) in valid_x_set:
                            val = -c_cancel[p_id] / num_scenarios
                            for o_id in ors:
                                tup = (s_id, h_id, d_id, o_id, p_id)
                                valid_P_tuples.append(tup)
                                P_obj_dict[tup] = val

    P = model.addVars(valid_P_tuples, vtype=GRB.BINARY, obj=P_obj_dict)

    # 4. Constraints Synthesis
    # Mandatory patient allocation
    x_by_p = defaultdict(list)
    for t in valid_x_tuples:
        x_by_p[t[3]].append(t)

    for p_id in mandatory_patients:
        model.addConstr(gp.quicksum(x[t] for t in x_by_p[p_id]) == 1)

    for p_id in non_mandatory_patients:
        model.addConstr(gp.quicksum(x[t] for t in x_by_p[p_id]) + w[p_id] == 1)

    # Facility hierarchy relationships
    model.addConstrs((x[t] <= y[t[0], t[1], t[2]] for t in valid_x_tuples))
    model.addConstrs((y[h, d, o] <= u[h, d] for h in hospital_ids for d in days for o in ors))

    # Break OR identity symmetry natively
    for h in hospital_ids:
        for d in days:
            for i in range(1, len(ors)):
                model.addConstr(y[h, d, ors[i]] <= y[h, d, ors[i-1]])

    # Scenario sub-problem ties
    model.addConstrs((P[t] <= x[t[1], t[2], t[3], t[4]] for t in valid_P_tuples))

    P_by_shdo = defaultdict(list)
    for t in valid_P_tuples:
        P_by_shdo[(t[0], t[1], t[2], t[3])].append(t)

    for (s_id, h_id, d_id, o_id), t_list in P_by_shdo.items():
        B = B_hd[(h_id, d_id)]
        model.addConstr(
            gp.quicksum(durations[s_id][t[4]] * P[t] for t in t_list) <= B * y[h_id, d_id, o_id]
        )

    # 5. Execute 
    model.optimize(log_callback)

    if model.SolCount == 0:
        return

    # 6. Parse Output 
    sol = {
        "objective_value": float(model.ObjVal),
        "u": {},
        "y": {},
        "x": {},
        "w": {}
    }

    for h in hospital_ids:
        for d in days:
            sol["u"][f"{h},{d}"] = int(round(u[h, d].X))
            for o in ors:
                sol["y"][f"{h},{d},{o}"] = int(round(y[h, d, o].X))
                for p in patient_ids:
                    if (h, d, o, p) in x:
                        val = int(round(x[h, d, o, p].X))
                        if val == 1:
                            sol["x"][f"{h},{d},{p},{o}"] = 1

    for p in non_mandatory_patients:
        sol["w"][f"{p}"] = int(round(w[p].X))

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=4)


if __name__ == "__main__":
    main()