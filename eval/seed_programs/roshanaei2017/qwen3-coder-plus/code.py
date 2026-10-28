import argparse
import json
import math
from gurobipy import *
import itertools
from solution_logger import SolutionLogger


def solve_surgical_scheduling(instance_data, time_limit):
    # Extract instance data
    num_patients = instance_data["num_patients"]
    num_surgeons = instance_data["num_surgeons"]
    num_hospitals = instance_data["num_hospitals"]
    ORs_per_hospital = instance_data["ORs_per_hospital"]
    num_days = instance_data["num_days"]
    alpha = instance_data["alpha"]

    patients = {p['patient_id']: p for p in instance_data["patients"]}
    surgeons = {s['surgeon_id']: s for s in instance_data["surgeons"]}
    hospitals = {}
    for hosp in instance_data["hospitals"]:
        hosp_id = hosp["hospital_id"]
        hospitals[hosp_id] = hosp

        # Process daily OR parameters
        for or_data in hosp["ORs"]:
            or_data["daily_regular_time"] = {}
            or_data["daily_max_overtime"] = {}
            if "daily" in or_data and or_data["daily"]:
                for day_str, d_info in or_data["daily"].items():
                    day = int(day_str)
                    or_data["daily_regular_time"][day] = d_info["regular_time"]
                    or_data["daily_max_overtime"][day] = d_info["max_overtime"]
            else:
                for day in range(num_days):
                    or_data["daily_regular_time"][day] = or_data["regular_time"]
                    or_data["daily_max_overtime"][day] = or_data["max_overtime"]

    # Create model
    model = Model("Surgical Scheduling")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    model.setParam('Heuristics', 0.05)

    # Decision Variables
    # Assignment variables
    x = {}  # x[p, s, h, d, r] = 1 if patient p assigned to surgeon s, hospital h, day d, OR r
    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            for h_id in hospitals:
                if h_id not in p.get("eligible_ORs_by_hospital", {}):
                    continue
                for d in range(num_days):
                    # Only consider days up to due date for mandatory patients
                    if d > p["due_date"] and p["is_mandatory"]:
                        continue
                    for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital)):
                        x[p_id, s_id, h_id, d, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{p_id}_{s_id}_{h_id}_{d}_{r}")

    # OR open decision variables
    y = {}  # y[h, d, r] = 1 if OR r at hospital h is opened on day d
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                y[h_id, d, r_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{h_id}_{d}_{r_idx}")

    # Surgeon-hospital assignment variables
    z = {}  # z[s, h, d] = 1 if surgeon s is assigned to hospital h on day d
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:  # Only considering days when surgeon is available
                z[s_id, h_id, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{s_id}_{h_id}_{d}")

    # Variables for finish time for each patient assignment
    F = {}  # F[p, s, h, d, r] = finish time of patient p assigned to s, h, d, r
    M = 10000  # Large enough number for big-M constraints

    for p_id, s_id, h_id, d, r in x:
        F[p_id, s_id, h_id, d, r] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"F_{p_id}_{s_id}_{h_id}_{d}_{r}")

    # Overtime used in each OR
    o = {}  # o[h, d, r] = overtime used (in minutes) in OR r at hospital h on day d
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                or_obj = hospitals[h_id]["ORs"][r_idx]
                max_ot_val = or_obj["daily_max_overtime"].get(d, or_obj["max_overtime"])
                o[h_id, d, r_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=max_ot_val, name=f"o_{h_id}_{d}_{r_idx}")

    # OR completion times
    completion_time = {}
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                completion_time[h_id, d, r_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"CT_{h_id}_{d}_{r_idx}")

    # Surgeon start/end times at each hospital-day
    start_time = {}
    end_time = {}
    for s_id in surgeons:
        for h_id in hospitals:
            for d in surgeons[s_id]['operating_days']:
                start_time[s_id, h_id, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ST_{s_id}_{h_id}_{d}")
                end_time[s_id, h_id, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ET_{s_id}_{h_id}_{d}")

    # Variables for ordering of patients in same OR
    u_or = {}  # u_or[p1, p2, s1, s2, h, d, r] = 1 if patient p1 is before patient p2 in same OR
    for p1_key in patients.keys():
        for p2_key in patients.keys():
            if p1_key >= p2_key:  # Avoid duplicate pairs
                continue
            p1 = patients[p1_key]
            p2 = patients[p2_key]
            s1_list = p1["eligible_surgeons"]
            s2_list = p2["eligible_surgeons"]

            for s1 in s1_list:
                for h in hospitals:
                    for d in range(num_days):
                        # Check if p1, p2 can be assigned in same hospital, same day
                        if d > p1["due_date"] and p1["is_mandatory"]:
                            continue
                        if d > p2["due_date"] and p2["is_mandatory"]:
                            continue

                        r1_list = p1.get("eligible_ORs_by_hospital", {}).get(h, range(ORs_per_hospital))
                        r2_list = p2.get("eligible_ORs_by_hospital", {}).get(h, range(ORs_per_hospital))
                        common_r = set(r1_list).intersection(set(r2_list))

                        for r in common_r:
                            if s1 in s2_list:  # Both lists can use same surgeon
                              for s2 in s2_list:
                                 # We only add the variable if both configurations are valid in the x assignment variables
                                if (p1_key, s1, h, d, r) in x and (p2_key, s2, h, d, r) in x:
                                  u_or[p1_key, p2_key, s1, s2, h, d, r] = model.addVar(vtype=GRB.BINARY, name=f"uor_{p1_key}_{p2_key}_{s1}_{s2}_{h}_{d}_{r}")
                            elif s1 not in s2_list:  # Different surgeons, just check the existence of their assignment
                               # Add only based on shared configuration possibilities
                               if (p1_key, s1, h, d, r) in x:
                                 for s2 in s2_list:
                                   if (p2_key, s2, h, d, r) in x:
                                      u_or[p1_key, p2_key, s1, s2, h, d, r] = model.addVar(vtype=GRB.BINARY, name=f"uor_{p1_key}_{p2_key}_{s1}_{s2}_{h}_{d}_{r}")


    # Variables for ordering of patients requiring same surgeon
    u_surg = {}  # u_surg[p1, p2, s, h1, h2, d] = 1 if patient p1's surgery is before p2's by same surgeon
    for p1_key in patients.keys():
        for p2_key in patients.keys():
            if p1_key >= p2_key:  # Avoid duplicate pairs
                continue
            p1 = patients[p1_key]
            p2 = patients[p2_key]

            # Get common surgeons
            common_surgeons = set(p1["eligible_surgeons"]) & set(p2["eligible_surgeons"])
            for s in common_surgeons:
                for d in range(num_days):
                    if d > p1["due_date"] and p1["is_mandatory"]:
                        continue
                    if d > p2["due_date"] and p2["is_mandatory"]:
                        continue
                    
                    # Check all possible hospital/day combinations where a surgeon can be
                    for h1 in hospitals:
                        if h1 not in p1.get("eligible_ORs_by_hospital", {}):
                            continue
                        for h2 in hospitals:
                            if h2 not in p2.get("eligible_ORs_by_hospital", {}):
                                continue
                            u_surg[p1_key, p2_key, s, h1, h2, d] = model.addVar(vtype=GRB.BINARY, name=f"usurg_{p1_key}_{p2_key}_{s}_{h1}_{h2}_{d}")

    model.update()

    # Constraint 1: Mandatory patients must be scheduled exactly once
    for p_id in patients:
        p = patients[p_id]
        if p["is_mandatory"]:
            model.addConstr(quicksum(x[p_id, s_id, h_id, d, r] 
                           for s_id in p["eligible_surgeons"] 
                           for h_id in p.get("eligible_ORs_by_hospital", {})  
                           for d in range(p["due_date"] + 1)
                           for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                           if (p_id, s_id, h_id, d, r) in x) == 1)

    # Constraint 2: Optional patients scheduled at most once
    for p_id in patients:
        p = patients[p_id]
        if not p["is_mandatory"]:
            model.addConstr(quicksum(x[p_id, s_id, h_id, d, r] 
                           for s_id in p["eligible_surgeons"] 
                           for h_id in p.get("eligible_ORs_by_hospital", {})  
                           for d in range(num_days)
                           for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                           if (p_id, s_id, h_id, d, r) in x) <= 1)

    # Constraint 3: If patient assigned, surgeon must be available at that hospital-day
    for p_id, s_id, h_id, d, r in x:
        model.addConstr(x[p_id, s_id, h_id, d, r] <= z[s_id, h_id, d])

    # Constraint 4: If patient assigned, OR must be opened
    for p_id, s_id, h_id, d, r in x:
        model.addConstr(x[p_id, s_id, h_id, d, r] <= y[h_id, d, r])

    # Constraint 5: OR time capacity - total load cannot exceed regular time + overtime
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                reg_time = hospitals[h_id]["ORs"][r_idx]["daily_regular_time"].get(d, 
                    hospitals[h_id]["ORs"][r_idx]["regular_time"])
                
                expr = LinExpr()
                for p_id in patients:
                    p = patients[p_id]
                    if r_idx not in p.get("eligible_ORs_by_hospital", {}).get(h_id, []):
                        continue
                    eligible_surgeons = p["eligible_surgeons"]
                    for s_id in eligible_surgeons:
                        if (p_id, s_id, h_id, d, r_idx) in x:
                            total_time = p["preparation_time"] + p["surgeon_specific_times"][s_id] + p["cleaning_time"]
                            expr += total_time * x[p_id, s_id, h_id, d, r_idx]
                
                model.addConstr(expr <= reg_time + o[h_id, d, r_idx])

    # Constraint 6: Surgeon time capacity using alpha parameter
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:
                max_time = s["availability_by_day"].get(d, float('inf'))
                
                expr = LinExpr()
                for p_id in patients:
                    p = patients[p_id]
                    if s_id not in p["eligible_surgeons"]:
                        continue
                    p_ors = p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                    for r in p_ors:
                        if (p_id, s_id, h_id, d, r) in x:
                            proc_time = p["surgeon_specific_times"][s_id]
                            total_occ_time = p["preparation_time"] + proc_time + p["cleaning_time"]
                            weighted_time = alpha * proc_time + (1 - alpha) * total_occ_time
                            expr += weighted_time * x[p_id, s_id, h_id, d, r]
                        
                model.addConstr(expr <= max_time)

    # Constraint 7: Finish time calculations
    for p_id, s_id, h_id, d, r in x:
        p = patients[p_id]
        proc_time = p["surgeon_specific_times"][s_id]
        total_occ_time = p["preparation_time"] + proc_time
        model.addConstr(F[p_id, s_id, h_id, d, r] >= total_occ_time - M * (1 - x[p_id, s_id, h_id, d, r]))

    # Constraint 8: OR completion time must account for each patient in that OR
    for p_id, s_id, h_id, d, r in x:
        p = patients[p_id]
        model.addConstr(completion_time[h_id, d, r] >= F[p_id, s_id, h_id, d, r] + p["cleaning_time"] - M * (1 - x[p_id, s_id, h_id, d, r]))

    # Constraint 9: If a patient is treated by surgeon s at hospital h on day d, the surgeon's end time must cover it
    for p_id, s_id, h_id, d, r in x:
        model.addConstr(end_time[s_id, h_id, d] >= F[p_id, s_id, h_id, d, r] - M * (1 - x[p_id, s_id, h_id, d, r]))

    # Constraint 10: No overlapping in OR - either p1 before p2 or p2 before p1
    for (p1key, p2key, s1, s2, h, d, r), u_var in u_or.items():
        p1 = patients[p1key]
        p2 = patients[p2key]
        
        # p1 finishes before p2 starts
        finish_p1 = F.get((p1key, s1, h, d, r), None)
        finish_p2 = F.get((p2key, s2, h, d, r), None)
        
        if finish_p1 is None or finish_p2 is None:
            continue
        
        proc_time_p2 = p2["surgeon_specific_times"][s2]
        
        # When u_or = 1: p1 finishes before p2 starts
        model.addConstr(
            finish_p1 + p1["cleaning_time"] + p2["preparation_time"] <=
            finish_p2 - proc_time_p2 + M * (3 - x.get((p1key, s1, h, d, r), 0) - x.get((p2key, s2, h, d, r), 0) - u_var)
        )
        
        # When u_or = 0: p2 finishes before p1 starts
        proc_time_p1 = p1["surgeon_specific_times"][s1]
        model.addConstr(
            finish_p2 + p2["cleaning_time"] + p1["preparation_time"] <=
            finish_p1 - proc_time_p1 + M * (2 - x.get((p1key, s1, h, d, r), 0) - x.get((p2key, s2, h, d, r), 0) + u_var)
        )

    # Constraint 11: No overlapping surgeries for same surgeon - ensure sequential scheduling
    for (p1key, p2key, s, h1, h2, d), u_var in u_surg.items():
        p1 = patients[p1key]
        p2 = patients[p2key]

        # Find the actual assignments - we need actual finish times for both patients under same surgeon
        # This becomes complex - for any assignment of p1 and p2 both to same surgeon s on same day d
        finish_p1_possible = [(pp, ss, hh, dd, rr) for (pp, ss, hh, dd, rr) in F if pp == p1key and ss == s and dd == d and hh == h1]
        finish_p2_possible = [(pp, ss, hh, dd, rr) for (pp, ss, hh, dd, rr) in F if pp == p2key and ss == s and dd == d and hh == h2]

        for (p1_ass, s1_ass, h1_ass, d1_ass, r1_ass) in finish_p1_possible:
            for (p2_ass, s2_ass, h2_ass, d2_ass, r2_ass) in finish_p2_possible:
                finish1 = F.get((p1_ass, s1_ass, h1_ass, d1_ass, r1_ass))
                finish2 = F.get((p2_ass, s2_ass, h2_ass, d2_ass, r2_ass))
                
                if finish1 is None or finish2 is None:
                    continue
                
                proc_time1 = p1["surgeon_specific_times"][s1_ass]
                proc_time2 = p2["surgeon_specific_times"][s2_ass]
                
                # When u_surg = 1: p1 finishes before p2 surgery starts
                model.addConstr(
                    finish1 + p2["preparation_time"] <=
                    finish2 - proc_time2 + M * (3 - x.get((p1_ass, s1_ass, h1_ass, d1_ass, r1_ass), 0) - 
                       x.get((p2_ass, s2_ass, h2_ass, d2_ass, r2_ass), 0) - u_var)
                )
                
                # When u_surg = 0: p2 finishes before p1 surgery starts
                model.addConstr(
                    finish2 + p1["preparation_time"] <=
                    finish1 - proc_time1 + M * (2 - x.get((p1_ass, s1_ass, h1_ass, d1_ass, r1_ass), 0) - 
                       x.get((p2_ass, s2_ass, h2_ass, d2_ass, r2_ass), 0) + u_var)
                )


    # Objective: minimize total costs minus rewards
    total_cost = LinExpr()

    # Add fixed OR opening costs
    for h_id in hospitals:
        h = hospitals[h_id]
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                or_obj = h["ORs"][r_idx]
                cost_per_min = or_obj["fixed_cost_per_hour"] / 60.0  # Convert hourly rate to per minute
                total_cost.addTerms(cost_per_min, y[h_id, d, r_idx])

    # Add surgeon fixed costs
    for s_id, h_id, d in z:
        if str(s_id) in hospitals[h_id]["surgeon_fixed_costs_per_day"]:
            cost = hospitals[h_id]["surgeon_fixed_costs_per_day"][str(s_id)]
            total_cost.addTerms(cost, z[s_id, h_id, d])

    # Add overtime costs
    for h_id in hospitals:
        h = hospitals[h_id]
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                or_obj = h["ORs"][r_idx]
                ot_cost_per_min = or_obj["overtime_cost_per_hour"] / 60.0
                total_cost.addTerms(ot_cost_per_min, o[h_id, d, r_idx])

    # Subtract rewards for scheduled optional patients
    for p_id in patients:
        p = patients[p_id]
        if p["is_mandatory"]:
            continue  # rewards only for optional patients
        
        reward = p["reward"]
        for s_id, h_id, d, r in [(_s, _h, _d, _r) for _s in p["eligible_surgeons"] 
                                         for _h in p.get("eligible_ORs_by_hospital", {})
                                         for _d in range(num_days)
                                         for _r in p.get("eligible_ORs_by_hospital", {}).get(_h, range(ORs_per_hospital))
                                         if (_p := p_id, _s, _h, _d, _r) in x]:
            if (p_id, s_id, h_id, d, r) in x:
                total_cost.add(-reward, x[p_id, s_id, h_id, d, r])

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Optimize
    model.optimize()
    
    # Handle solution
    if model.status == GRB.INF_OR_UNBD or model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return {
            "objective_value": float('inf'),
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": []
        }

    assignments = []
    opened_ors = []
    surgeon_assignments = []
    or_sequences = []
    surgeon_sequences = []

    # Extract assignments
    for p_id, s_id, h_id, d, r in x:
        if x[p_id, s_id, h_id, d, r].x > 0.5:
            p = patients[p_id]
            surgery_duration = p["surgeon_specific_times"][s_id]
            total_OCC = p["preparation_time"] + surgery_duration + p["cleaning_time"]
            
            assignments.append({
                "patient_id": int(p_id),
                "surgeon_id": int(s_id),
                "hospital_id": int(h_id),
                "day": int(d),
                "or_id": int(r),
                "is_mandatory": bool(p["is_mandatory"]),
                "T_ps": float(total_OCC),
                "finish_time": float(F[p_id, s_id, h_id, d, r].x),
                "surgery_start_time": float(F[p_id, s_id, h_id, d, r].x - surgery_duration),
                "room_entry_time": float(F[p_id, s_id, h_id, d, r].x - surgery_duration - p["preparation_time"]),
                "room_exit_time": float(F[p_id, s_id, h_id, d, r].x + p["cleaning_time"])
            })

    # Extract opened ORs
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                if y[h_id, d, r_idx].x > 0.5:
                    or_obj = hospitals[h_id]["ORs"][r_idx]
                    overtime_used = o[h_id, d, r_idx].x
                    regular_time = or_obj["daily_regular_time"].get(d, or_obj["regular_time"])
                    completion_t = completion_time[h_id, d, r_idx].x if completion_time[h_id, d, r_idx].x > 0 else 0
                    opened_ors.append({
                        "hospital_id": int(h_id),
                        "day": int(d),
                        "or_id": int(r_idx),
                        "regular_time": int(regular_time),
                        "completion_time": float(completion_t),
                        "overtime": float(overtime_used)
                    })

    # Extract surgeon assignments
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:
                if z[s_id, h_id, d].x > 0.5:
                    start_t = start_time.get((s_id, h_id, d)).x if start_time.get((s_id, h_id, d)) else 0
                    end_t = end_time.get((s_id, h_id, d)).x if end_time.get((s_id, h_id, d)) else 0
                    surgeon_assignments.append({
                        "surgeon_id": int(s_id),
                        "hospital_id": int(h_id),
                        "day": int(d),
                        "start_time": float(start_t),
                        "end_time": float(end_t)
                    })

    # Extract OR sequences
    for (p1, p2, s1, s2, h, d, r), u_var in u_or.items():
        if u_var.x > 0.5:
            # Make sure both are actually assigned
            if ((p1, s1, h, d, r) in x and 
                (p2, s2, h, d, r) in x and 
                x[p1, s1, h, d, r].x > 0.5 and 
                x[p2, s2, h, d, r].x > 0.5):
                or_sequences.append({
                    "hospital_id": int(h),
                    "day": int(d),
                    "or_id": int(r),
                    "patient_p": int(p1),
                    "patient_k": int(p2),
                    "p_after_k": int(1)
                })

    # Extract surgeon sequences
    for (p1, p2, s, h1, h2, d), u_var in u_surg.items():
        # Extract only for real assignments
        found_assignments = False
        for r1 in patients[p1].get("eligible_ORs_by_hospital", {}).get(h1, []):
            for r2 in patients[p2].get("eligible_ORs_by_hospital", {}).get(h2, []):
                if ((p1, s, h1, d, r1) in x and 
                     (p2, s, h2, d, r2) in x and
                     x[p1, s, h1, d, r1].x > 0.5 and 
                     x[p2, s, h2, d, r2].x > 0.5):
                    found_assignments = True
                    break
            if found_assignments:
                break
        
        if found_assignments and u_var.x > 0.5:
            surgeon_sequences.append({
                "hospital_id": int(h1),  # Since surgeon moves between hosps, we might need convention, use h1
                "day": int(d),
                "surgeon_id": int(s),
                "patient_p": int(p1),
                "patient_k": int(p2),
                "p_after_k": int(1)
            })

    objective_value = model.objVal if model.SolCount > 0 else float('inf')

    return {
        "objective_value": float(objective_value),
        "assignments": assignments,
        "opened_ors": opened_ors,
        "surgeon_assignments": surgeon_assignments,
        "or_sequences": or_sequences,
        "surgeon_sequences": surgeon_sequences
    }


def main():
    parser = argparse.ArgumentParser(description='Surgical Scheduling Optimizer')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')

    args = parser.parse_args()

    # Initialize logger if path provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_surgical_scheduling(instance_data, args.time_limit)

    # Log the final objective value using the internal logger
    if logger and 'objective_value' in solution and not math.isinf(solution['objective_value']):
        logger.log(solution['objective_value'])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()