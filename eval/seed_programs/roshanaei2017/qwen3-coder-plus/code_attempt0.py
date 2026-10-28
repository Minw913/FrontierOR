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
        
        # Flatten the daily information into OR properties for the relevant days
        for or_data in hosp["ORs"]:
            or_data["daily_regular_time"] = {}
            or_data["daily_max_overtime"] = {}
            if "daily" in or_data:
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
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    model.setParam('Heuristics', 0.05)
    
    # Initialize logger inside the function
    logger = SolutionLogger(None, sense="minimize")  # We'll handle calls manually via provided logger interface

    # Decision Variables
    # Binary variables: assignment of patient p to surgeon s, hospital h, day d, OR r
    x = {}  # x[p, s, h, d, r] = 1 if patient p assigned to surgeon s, hospital h, day d, OR r
    
    # Binary variables: OR open decision
    y = {}  # y[h, d, r] = 1 if OR r at hospital h is opened on day d
    z = {}  # z[s, h, d] = 1 if surgeon s is assigned to hospital h on day d

    # Overtime usage variables
    o = {}  # o[h, d, r] = overtime used (in minutes) in OR r at hospital h on day d

    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            for h_id in hospitals:  # Check what we can do with ORs - for now assume patient eligible for all ORs at some hospital
                for d in range(num_days):
                    # Only consider days up to due date for scheduling
                    if d > p["due_date"] and p["is_mandatory"]:
                        continue
                    # OR selection based on eligibility
                    eligible_ors = p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                    
                    for r in eligible_ors:
                        x[p_id, s_id, h_id, d, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{p_id}_{s_id}_{h_id}_{d}_{r}")

    # Open OR variables
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                y[h_id, d, r_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{h_id}_{d}_{r_idx}")

    # Surgeon-hospital assignment variables
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:  # Only for days surgeon is available
                z[s_id, h_id, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{s_id}_{h_id}_{d}")
                
    # Overtime variables
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                o[h_id, d, r_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"o_{h_id}_{d}_{r_idx}")
    
    # Timing variables for sequences
    # Finish time of patient p
    finish_times = {}
    for p_id in patients:
        for s_id in patients[p_id]["eligible_surgeons"]:
            for h_id in hospitals:
                for d in range(num_days):
                    eligible_ors = patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                    for r in eligible_ors:
                        finish_times[p_id, s_id, h_id, d, r] = model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0.0, name=f"ft_{p_id}_{s_id}_{h_id}_{d}_{r}"
                        )
                        
    # Completion times of ORs and surgeons
    or_completion_times = {}
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                or_completion_times[h_id, d, r_idx] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name=f"ct_{h_id}_{d}_{r_idx}"
                )

    surgeon_start_times = {}
    surgeon_end_times = {}
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:
                surgeon_start_times[s_id, h_id, d] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name=f"sst_{s_id}_{h_id}_{d}"
                )
                surgeon_end_times[s_id, h_id, d] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name=f"set_{s_id}_{h_id}_{d}"
                )
                
    # Sequencing variables
    # For OR precedence
    u_or = {}
    for p_id in patients:
        p = patients[p_id]
        for s_ids in p["eligible_surgeons"]:
            for h_id in hospitals:
                for d in range(num_days):  # Consider constraints with due date if needed
                    if d > p["due_date"] and p["is_mandatory"]:
                        continue
                    eligible_ors = p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                    
                    for r in eligible_ors:
                        for k_id in patients:
                            if k_id == p_id:
                                continue
                            k = patients[k_id]
                            if d > k["due_date"] and k["is_mandatory"]:
                                continue
                            
                            if s_id in k["eligible_surgeons"]:
                                eligible_ors_k = k.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                                if r in eligible_ors_k:
                                    if (p_id, s_id, h_id, d, r) in x and (k_id, s_id, h_id, d, r) in x:
                                        u_or[p_id, k_id, s_id, h_id, d, r] = model.addVar(vtype=GRB.BINARY,
                                                                                         name=f"u_or_{p_id}_{k_id}_{s_id}_{h_id}_{d}_{r}")

    # For surgeon precedence
    u_surg = {}
    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            for h_id in hospitals:
                for d in range(num_days):
                    if d > p["due_date"] and p["is_mandatory"]:
                        continue
                    for k_id in patients:
                        if k_id == p_id:
                            continue
                        k = patients[k_id]
                        if s_id in k["eligible_surgeons"]:
                            if d > k["due_date"] and k["is_mandatory"]:
                                continue
                            # Check if both could potentially be at same hospital
                            # This is more complex - need to track if surgeon visits this hospital on day d
                            u_surg[p_id, k_id, s_id, h_id, d] = model.addVar(vtype=GRB.BINARY,
                                                                             name=f"u_surg_{p_id}_{k_id}_{s_id}_{h_id}_{d}")

    model.update()

    # Constraints
    
    # Each mandatory patient must be assigned exactly once
    for p_id in patients:
        p = patients[p_id]
        if p["is_mandatory"]:
            model.addConstr(quicksum(x[p_id, s_id, h_id, d, r] 
                                     for s_id in p["eligible_surgeons"] 
                                     for h_id in hospitals
                                     for d in range(p["due_date"]+1)
                                     for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                                     if (p_id, s_id, h_id, d, r) in x) == 1)

    # Each optional patient assigned at most once
    for p_id in patients:
        p = patients[p_id]
        if not p["is_mandatory"]:
            model.addConstr(quicksum(x[p_id, s_id, h_id, d, r] 
                                     for s_id in p["eligible_surgeons"] 
                                     for h_id in hospitals
                                     for d in range(num_days)
                                     for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                                     if (p_id, s_id, h_id, d, r) in x) <= 1)

    # Patient assignment requires both surgeon assignment and OR availability
    for p_id, s_id, h_id, d, r in x:
        # Surgeon must be assigned to hospital on this day
        model.addConstr(x[p_id, s_id, h_id, d, r] <= z[s_id, h_id, d])
        
        # OR must be opened
        model.addConstr(x[p_id, s_id, h_id, d, r] <= y[h_id, d, r])
        
    # Calculate finish times and ensure scheduling feasibility

    # Each OR's total time must fit in its time slot (reg_time + overtime)
    for h_id in hospitals:
        h = hospitals[h_id]
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                total_load_in_OR = LinExpr()
                reg_time = h["ORs"][r_idx]["daily_regular_time"].get(d, h["ORs"][r_idx]["regular_time"])
                max_ot = h["ORs"][r_idx]["daily_max_overtime"].get(d, h["ORs"][r_idx]["max_overtime"])
                
                for p_id in patients:
                    p = patients[p_id]
                    if r_idx in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital)):
                        for s_id in p["eligible_surgeons"]:
                            if (p_id, s_id, h_id, d, r_idx) in x:
                                surg_time = p["surgeon_specific_times"][s_id]
                                total_time = p["preparation_time"] + surg_time + p["cleaning_time"]
                                total_load_in_OR += total_time * x[p_id, s_id, h_id, d, r_idx]
                
                # Constraint related to total load vs time available
                model.addConstr(total_load_in_OR <= reg_time + o[h_id, d, r_idx])
                
                # Max overtime constraint
                model.addConstr(o[h_id, d, r_idx] <= max_ot)

    # Surgeon capacity constraint using alpha (schedule tightness)
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:
                # Alpha-weighted load constraint
                total_alpha_weighted_time = LinExpr()
                max_avail_time = s["availability_by_day"].get(d, float('inf'))
                
                # For each patient assigned to surgeon s, hospital h, day d
                for p_id in patients:
                    p = patients[p_id]
                    if s_id in p["eligible_surgeons"]:
                        # Check if they can be assigned to hospital h with that surgeon and on day d
                        for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital)):
                            if (p_id, s_id, h_id, d, r) in x:
                                surg_time = p["surgeon_specific_times"][s_id]
                                total_time = p["preparation_time"] + surg_time + p["cleaning_time"]
                                # Alpha-weighted load: alpha * surgery_time + (1-alpha) * total_time
                                alpha_weighted = alpha * surg_time + (1 - alpha) * total_time
                                total_alpha_weighted_time += alpha_weighted * x[p_id, s_id, h_id, d, r]
                
                model.addConstr(total_alpha_weighted_time <= max_avail_time)

    # Precedence constraints for same OR - if both patients scheduled in same day/OR, one must go after another
    # We only do binary precedence where both patients assigned to same day, same OR
    for p_id, k_id in itertools.combinations(patients.keys(), 2):
        p = patients[p_id]
        k = patients[k_id]
        
        for s1_id in p["eligible_surgeons"]:
            for s2_id in k["eligible_surgeons"]:
                for h_id in hospitals:
                    for d in range(min(num_days, max(p["due_date"], k["due_date"]) + 1)):  # Adjust to common scope of due dates
                        if d > p["due_date"] and p["is_mandatory"]:
                            continue
                        if d > k["due_date"] and k["is_mandatory"]:
                            continue
                        
                        eligible_ors_p = p.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                        eligible_ors_k = k.get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                        
                        for r in set(eligible_ors_p) & set(eligible_ors_k):  # Only shared ORs
                            if (p_id, s1_id, h_id, d, r) in x and (k_id, s2_id, h_id, d, r) in x:
                                # Define ordering variables and constraints
                                # Use a Big M formulation to ensure non-overlap
                                
                                # Define M as maximum possible clock time - say double shift (e.g., 12 hours = 720 min)
                                M = 1440  # 24 hours in minutes, safe upper bound

                                # Define the relationship: if both patients are in same OR on same day
                                # then either p comes before k, or k comes before p
                                
                                # We introduce binary variables u_or to represent precedence
                                # If u_or[p,k,s,h,d,r] = 1 means p finishes before k starts (considering setup for k)
                                
                                if (p_id, k_id, s1_id, h_id, d, r) not in u_or:
                                    if (k_id, p_id, s1_id, h_id, d, r) in u_or:
                                        # Use symmetric variable if exists
                                        continue
                                    u_or[p_id, k_id, s1_id, h_id, d, r] = model.addVar(vtype=GRB.BINARY,
                                                                                      name=f"u_or_{p_id}_{k_id}_{s1_id}_{h_id}_{d}_{r}")

                                # Finish time for p must be <= start time of k (with required setup) or vice versa
                                # Using Big M method
                                
                                # If u_or[p,k,...] = 1: then p finishes before k starts
                                # finish_time_p <= start_time_k - epsilon => finish_time_p + p["cleaning_time"] (for transition) + 
                                # k["preparation_time"] <= start_time_k 
                                # So finish_time_p + p["cleaning_time"] + k["preparation_time"] <= start_time_k
                                model.addGenConstrIndicator(
                                    u_or[p_id, k_id, s1_id, h_id, d, r], 1,
                                    finish_times[p_id, s1_id, h_id, d, r] + p["cleaning_time"] + k["preparation_time"] <= 
                                        finish_times[k_id, s2_id, h_id, d, r] - k["surgeon_specific_times"][s2_id]
                                )
                                
                                # If u_or[p,k,...] = 0: then k finishes before p starts
                                model.addGenConstrIndicator(
                                    u_or[p_id, k_id, s1_id, h_id, d, r], 0,
                                    finish_times[k_id, s2_id, h_id, d, r] + k["cleaning_time"] + p["preparation_time"] <= 
                                        finish_times[p_id, s1_id, h_id, d, r] - p["surgeon_specific_times"][s1_id]
                                )

    # Precedence constraints for same surgeon across any ORs in the same day and hospital
    for p_id, k_id in itertools.combinations(patients.keys(), 2):
        # Both patients must be assigned to same surgeon, day, and hospital
        # Both may be in same hospital-day and same surgeon (regardless of OR) – need to ensure surgeon isn't scheduled in parallel
        # Same logic – but precedence is based on surgeon's start times
        
        for s_id in set(patients[p_id]["eligible_surgeons"]) & set(patients[k_id]["eligible_surgeons"]):
            
            for h_id in hospitals:
                for d in range(min(patients[p_id]["due_date"]+1 if patients[p_id]["is_mandatory"] else num_days,
                                   patients[k_id]["due_date"]+1 if patients[k_id]["is_mandatory"] else num_days)):
                    
                    # Define if p and k are both assigned surgeon s at hospital h on day d
                    # We use a similar binary indicator
    
                    p_assigned_to_hs = quicksum(x[p_id, s_id, h_id, d, r] 
                                                for r in patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                                                if (p_id, s_id, h_id, d, r) in x)
                    k_assigned_to_hs = quicksum(x[k_id, s_id, h_id, d, r] 
                                                for r in patients[k_id].get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital))
                                                if (k_id, s_id, h_id, d, r) in x)
    
                    # Add constraint: if both are assigned surgeon s and hospital h on day d, exactly one precedes another
                    # We use binary u_surg[p,k,s,h,d] = 1 means p finishes before k starts
                    # Define the relationship: if p is assigned to s@h@day and k also assigned to s@h@day
                    model.addGenConstrIndicator(
                        u_surg[p_id, k_id, s_id, h_id, d], 1,
                        finish_times[p_id, s_id, h_id, d, [r for r in patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital)) if (p_id, s_id, h_id, d, r) in x][0]] +
                        (patients[k_id]["preparation_time"]) <=
                        finish_times[k_id, s_id, h_id, d, [r for r in patients[k_id].get("eligible_ORs_by_hospital", {}).get(h_id, range(ORs_per_hospital)) if (k_id, s_id, h_id, d, r) in x][0]] -
                        patients[k_id]["surgeon_specific_times"][s_id]
                        # Above is rough – we need to find proper OR assignment for the p/k pairs
                    )
                    # To make code cleaner, let's restructure the constraint with the right OR selected
                    # Use conditional or create helper vars
                    
    # For each patient assigned to a (surgeon, day, hospital, OR), we set the finish time appropriately
    # We need to linearize the OR assignment for timing constraints
    
    # This complex scheduling requires us to fix the model structure first:
    
    # Let's define more carefully:
    # - For each assigned patient, calculate their finish time as part of their assignment
    
    # Calculate finish time for each assigned patient
    # For each combination (p_id, s_id, h_id, d, r):
    for p_id, s_id, h_id, d, r in x:
        p = patients[p_id]
        proc_time = p["surgeon_specific_times"][s_id]
        
        # Set finish time constraints
        model.addConstr(finish_times[p_id, s_id, h_id, d, r] >= 
                p["preparation_time"] + proc_time)  # Minimum finish time
        
        # Also connect all actual assignments
        model.addConstr(x[p_id, s_id, h_id, d, r] == 0)
        
    # Rebuild model based on correct structure

    # Reset and rebuild with better variable handling
    vars_to_remove = []
    for var in model.getVars():
        vars_to_remove.append(var)

    # Remove the old variables since we're restructuring heavily
    model = Model("Surgical Scheduling")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.05)

    x = {}

    # Rebuild x with correct keys
    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            for h_id in hospitals:
                if h_id not in p.get("eligible_ORs_by_hospital", {}):
                    continue
                for d in range(num_days):
                    if d > p["due_date"] and p["is_mandatory"]:
                        continue
                    for r in p.get("eligible_ORs_by_hospital").get(h_id, []):
                        x[p_id, s_id, h_id, d, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{p_id}_{s_id}_{h_id}_{d}_{r}")

    # Same for y, z, o
    y = {}
    z = {}
    o = {}

    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                y[h_id, d, r_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{h_id}_{d}_{r_idx}")

    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:
                z[s_id, h_id, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{s_id}_{h_id}_{d}")

    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                or_obj = hospitals[h_id]["ORs"][r_idx]
                max_ot_val = or_obj["daily_max_overtime"].get(d, or_obj["max_overtime"])
                o[h_id, d, r_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=max_ot_val, name=f"o_{h_id}_{d}_{r_idx}")

    # Timing variables for sequences
    F_pshr = {}  # Finish time of patient p assigned to surgeon s in hospital h on day d in OR r
    for (p_id, s_id, h_id, d, r) in x:
        F_pshr[p_id, s_id, h_id, d, r] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"F_{p_id}_{s_id}_{h_id}_{d}_{r}")

    # OR completion times
    C_hdr = {}
    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                C_hdr[h_id, d, r_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{h_id}_{d}_{r_idx}")

    # Surgeon end times
    S_shd = {}
    for s_id in surgeons:
        for h_id in hospitals:
            for d in surgeons[s_id]['operating_days']:
                S_shd[s_id, h_id, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{s_id}_{h_id}_{d}")

    # Order variables for OR
    u_pkhdr = {}
    for p_id in patients:
        for k_id in patients:
            if k_id <= p_id: continue
            for s_id in patients[p_id]["eligible_surgeons"]:
                for s2_id in patients[k_id]["eligible_surgeons"]:
                    for h_id in hospitals:
                        for d in range(num_days):
                            eligible_r_p = patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, [])
                            eligible_r_k = patients[k_id].get("eligible_ORs_by_hospital", {}).get(h_id, [])
                            for r in set(eligible_r_p) & set(eligible_r_k):
                                # Both are eligible in the same OR
                                if (p_id, s_id, h_id, d, r) in x and (k_id, s2_id, h_id, d, r) in x:
                                    u_pkhdr[p_id, k_id, s_id, s2_id, h_id, d, r] = model.addVar(
                                        vtype=GRB.BINARY, name=f"u_pk{p_id}_{k_id}_{s_id}_{s2_id}_{h_id}_{d}_{r}")

    # Order variables for surgeon
    v_pkhshd = {}
    for p_id in patients:
        for k_id in patients:
            if k_id <= p_id: continue
            for s_id in set(patients[p_id]["eligible_surgeons"]) & set(patients[k_id]["eligible_surgeons"]):
                for h_id in hospitals:
                    for d in range(num_days):
                        # Both can be assigned to same surgeon, hospital, day
                        p_eligible_r = patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, [])
                        k_eligible_r = patients[k_id].get("eligible_ORs_by_hospital", {}).get(h_id, [])
                        if p_eligible_r and k_eligible_r:  # They can be at this hospital
                            v_pkhshd[p_id, k_id, s_id, h_id, d] = model.addVar(
                                vtype=GRB.BINARY, name=f"v_pk{p_id}_{k_id}_{s_id}_{h_id}_{d}")

    model.update()

    # Constraint C1: Mandatory patients assigned exactly once
    for p_id in patients:
        p = patients[p_id]
        if p['is_mandatory']:
            model.addConstr(LinExpr([(1, x[p, s, h, day, r]) 
                                  for s in p["eligible_surgeons"] 
                                  for h in p.get("eligible_ORs_by_hospital", {}) 
                                  for day in range(p["due_date"] + 1)
                                  for r in p.get("eligible_ORs_by_hospital", {}).get(h, []) if (p, s, h, day, r) in x]) == 1)

    # Constraint C2: Optional patients assigned at most once
    for p_id in patients:
        p = patients[p_id]
        if not p['is_mandatory']:
            model.addConstr(LinExpr([(1, x[p, s, h, day, r]) 
                                  for s in p["eligible_surgeons"] 
                                  for h in p.get("eligible_ORs_by_hospital", {}) 
                                  for day in range(num_days)
                                  for r in p.get("eligible_ORs_by_hospital", {}).get(h, []) if (p, s, h, day, r) in x]) <= 1)

    # Constraint C3: Patient assigned to OR requires OR open
    for p_id, s_id, h_id, d, r in x:
        model.addConstr(x[p_id, s_id, h_id, d, r] <= y[h_id, d, r])

    # Constraint C4: Patient assigned to surgeon requires surgeon at hospital
    for p_id, s_id, h_id, d, r in x:
        model.addConstr(x[p_id, s_id, h_id, d, r] <= z[s_id, h_id, d])

    # Total load on OR <= regular time + overtime
    M_bp = 10000  # A big number
    for h_id in hospitals:
        h = hospitals[h_id]
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                total_dur = LinExpr()
                reg_time = h["ORs"][r_idx]["daily_regular_time"].get(d, h["ORs"][r_idx]["regular_time"])
                
                for p_id in patients:
                    p = patients[p_id]
                    if r_idx not in p.get("eligible_ORs_by_hospital", {}).get(h_id, []):
                        continue
                    s_list = p["eligible_surgeons"]
                    for s_id in s_list:
                        if (p_id, s_id, h_id, d, r_idx) in x:
                            dur = p["preparation_time"] + p["surgeon_specific_times"][s_id] + p["cleaning_time"]
                            total_dur.addTerms(dur, x[p_id, s_id, h_id, d, r_idx])
                
                # Add constraint for OR capacity
                model.addConstr(total_dur <= reg_time + o[h_id, d, r_idx])
                # Max overtime constraint is already handled in variable definition

    # Surgeon load constraint with alpha
    for s_id in surgeons:
        s = surgeons[s_id]
        for h_id in hospitals:
            for d in s['operating_days']:
                load_exp = LinExpr()
                avail = s["availability_by_day"].get(d, float('inf'))
                
                for p_id in patients:
                    p = patients[p_id]
                    if s_id not in p["eligible_surgeons"]:
                        continue
                    # Check if this surgeon is available for same hospital-day
                    for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, []):
                        if (p_id, s_id, h_id, d, r) in x:
                            base_time = p["preparation_time"] + p["surgeon_specific_times"][s_id] + p["cleaning_time"]
                            surg_time = p["surgeon_specific_times"][s_id]
                            adjusted_time = alpha * surg_time + (1 - alpha) * base_time
                            load_exp.addTerms(adjusted_time, x[p_id, s_id, h_id, d, r])
                
                model.addConstr(load_exp <= avail)

    # Finish time is at least proc time + prep time if assigned
    T_pr = {}
    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            T_pr[(p_id, s_id)] = p["preparation_time"] + p["surgeon_specific_times"][s_id]

    for p_id, s_id, h_id, d, r in x:
        model.addConstr(F_pshr[p_id, s_id, h_id, d, r] >= T_pr[(p_id, s_id)] - M_bp*(1 - x[p_id, s_id, h_id, d, r]))

    # OR completion time is at least finish time if assigned to this OR
    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            for h_id in hospitals:
                for d in range(num_days):
                    if d > p["due_date"] and p["is_mandatory"]: continue
                    for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, []):
                        if (p_id, s_id, h_id, d, r) in x:
                            model.addConstr(C_hdr[h_id, d, r] >= F_pshr[p_id, s_id, h_id, d, r] + p["cleaning_time"] - M_bp * (1 - x[p_id, s_id, h_id, d, r]))

    # Surgeon end time calculation
    for p_id in patients:
        p = patients[p_id]
        for s_id in p["eligible_surgeons"]:
            for h_id in hospitals:
                for d in range(num_days):
                    if d > p["due_date"] and p["is_mandatory"]: continue
                    for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, []):
                        if (p_id, s_id, h_id, d, r) in x:
                            model.addConstr(S_shd[s_id, h_id, d] >= F_pshr[p_id, s_id, h_id, d, r] - M_bp * (1 - x[p_id, s_id, h_id, d, r]))

    # OR non-overlapping constraints
    for p_id, k_id in itertools.combinations(patients.keys(), 2):
        for h_id in hospitals:
            for d in range(num_days):
                common_ors = set(patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, [])) & \
                             set(patients[k_id].get("eligible_ORs_by_hospital", {}).get(h_id, []))
                
                for r in common_ors:
                    s_list_p = patients[p_id]["eligible_surgeons"]
                    s_list_k = patients[k_id]["eligible_surgeons"]
                    
                    for s1 in s_list_p:
                        for s2 in s_list_k:
                            if (p_id, s1, h_id, d, r) not in x or (k_id, s2, h_id, d, r) not in x:
                                continue
                                
                            # Add binary for precedence
                            if (p_id, k_id, s1, s2, h_id, d, r) in u_pkhdr:
                                u_var = u_pkhdr[p_id, k_id, s1, s2, h_id, d, r]
                            else:
                                continue

                            # p finishes before k can begin (considering k's prep_time and cleanup of p)
                            model.addConstr(
                                F_pshr[p_id, s1, h_id, d, r] + patients[p_id]["cleaning_time"] + patients[k_id]["preparation_time"] 
                                <= F_pshr[k_id, s2, h_id, d, r] - patients[k_id]["surgeon_specific_times"][s2] + 
                                M_bp * (3 - x[p_id, s1, h_id, d, r] - x[k_id, s2, h_id, d, r] - u_var))

                            # k finishes before p can begin
                            model.addConstr(
                                F_pshr[k_id, s2, h_id, d, r] + patients[k_id]["cleaning_time"] + patients[p_id]["preparation_time"] 
                                <= F_pshr[p_id, s1, h_id, d, r] - patients[p_id]["surgeon_specific_times"][s1] + 
                                M_bp * (2 - x[p_id, s1, h_id, d, r] - x[k_id, s2, h_id, d, r] + u_var))

    # Surgeon sequential constraints
    s_id_lists = {}
    for p_id in patients:
        for s_id in patients[p_id]["eligible_surgeons"]:
            if s_id not in s_id_lists:
                s_id_lists[s_id] = []
            s_id_lists[s_id].append(p_id)

    for s_id, plist in s_id_lists.items():
        for i, p_id in enumerate(plist):
            for j, k_id in enumerate(plist):
                if i >= j: continue
                for h_id in hospitals:
                    for d in range(num_days):
                        if d > patients[p_id]["due_date"] and patients[p_id]["is_mandatory"]: continue
                        if d > patients[k_id]["due_date"] and patients[k_id]["is_mandatory"]: continue

                        # Check if both can be assigned to same surgeon and hospital-day
                        p_available_at_h = h_id in patients[p_id].get("eligible_ORs_by_hospital", {})
                        k_available_at_h = h_id in patients[k_id].get("eligible_ORs_by_hospital", {})
                        
                        if not p_available_at_h or not k_available_at_h:
                            continue

                        if (p_id, k_id, s_id, h_id, d) not in v_pkhshd:
                            continue

                        v_binary = v_pkhshd[p_id, k_id, s_id, h_id, d]

                        # Find an assignment of p and k where surgeon does both at same hospital, different ORs
                        possible_assignments = []
                        for r_p in patients[p_id].get("eligible_ORs_by_hospital", {}).get(h_id, []):
                            for r_k in patients[k_id].get("eligible_ORs_by_hospital", {}).get(h_id, []):
                                if (p_id, s_id, h_id, d, r_p) in x and (k_id, s_id, h_id, d, r_k) in x:
                                    possible_assignments.append((r_p, r_k))

                        if possible_assignments:
                            r1, r2 = possible_assignments[0]  # Just pick first for simplicity in constraint

                            model.addConstr(
                                F_pshr[p_id, s_id, h_id, d, r1] + patients[p_id]["preparation_time"] <=
                                F_pshr[k_id, s_id, h_id, d, r2] - patients[k_id]["surgeon_specific_times"][s_id] +
                                M_bp * (3 - x[p_id, s_id, h_id, d, r1] - x[k_id, s_id, h_id, d, r2] - v_binary))

                            model.addConstr(
                                F_pshr[k_id, s_id, h_id, d, r2] + patients[k_id]["preparation_time"] <=
                                F_pshr[p_id, s_id, h_id, d, r1] - patients[p_id]["surgeon_specific_times"][s_id] +
                                M_bp * (2 - x[p_id, s_id, h_id, d, r1] - x[k_id, s_id, h_id, d, r2] + v_binary))

    # Objective function: minimize sum of costs minus rewards
    obj_expr = LinExpr()

    # OR opening costs
    for h_id in hospitals:
        h = hospitals[h_id]
        for d in range(num_days):
            for r in range(ORs_per_hospital):
                or_obj = h["ORs"][r]
                cost_per_min = or_obj["fixed_cost_per_hour"] / 60.0  # Convert per hour to per minute
                obj_expr.addTerms(cost_per_min, y[h_id, d, r])

    # Surgeon assignment costs
    for s_id in surgeons:
        for h_id in hospitals:
            if str(s_id) in hospitals[h_id]["surgeon_fixed_costs_per_day"]:
                cost = hospitals[h_id]["surgeon_fixed_costs_per_day"][str(s_id)]
                for d in surgeons[s_id]['operating_days']:
                    obj_expr.addTerms(cost, z[s_id, h_id, d])
            else:
                for d in surgeons[s_id]['operating_days']:
                    obj_expr.addTerms(0.0, z[s_id, h_id, d])

    # Overtime costs
    for h_id in hospitals:
        h = hospitals[h_id]
        for d in range(num_days):
            for r in range(ORs_per_hospital):
                or_obj = h["ORs"][r]
                ot_cost_per_min = or_obj["overtime_cost_per_hour"] / 60.0
                obj_expr.addTerms(ot_cost_per_min, o[h_id, d, r])

    # Rewards for optional patients (subtract)
    for p_id in patients:
        p = patients[p_id]
        if not p["is_mandatory"]:
            reward = p["reward"]
            p_assignment_sum = LinExpr()
            for s_id in p["eligible_surgeons"]:
                for h_id in p.get("eligible_ORs_by_hospital", {}):
                    for d in range(num_days):
                        for r in p.get("eligible_ORs_by_hospital", {}).get(h_id, []):
                            if (p_id, s_id, h_id, d, r) in x:
                                p_assignment_sum.addTerms(reward, x[p_id, s_id, h_id, d, r])
            obj_expr.add(-1.0, p_assignment_sum)

    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Optimization
    best_obj_value = float('inf')
    def callback(model, where):
        nonlocal best_obj_value
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj_value:
                best_obj_value = obj

    model.optimize(callback)

    # Extract and build solution from model
    if model.status == GRB.INF_OR_UNBD or model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED:
        print("Model is infeasible or unbounded")
        return {"objective_value": float('inf'), "assignments": [], "opened_ors": [], "surgeon_assignments": [],
                "or_sequences": [], "surgeon_sequences": []}
                
    # Return current best if we reached time limit
    if model.SolCount == 0:
        return {"objective_value": float('inf'), "assignments": [], "opened_ors": [], "surgeon_assignments": [],
                "or_sequences": [], "surgeon_sequences": []}

    sol_obj = model.objVal

    assignments = []
    opened_ors = []
    surgeon_assignments = []
    or_sequences = []
    surgeon_sequences = []

    # Build solution data structures
    for p_id, s_id, h_id, d, r in x:
        if x[p_id, s_id, h_id, d, r].x > 0.9:
            # Patient is assigned
            p = patients[p_id]
            surgeon_proc_time = p["surgeon_specific_times"][s_id]
            total_time = p["preparation_time"] + surgeon_proc_time + p["cleaning_time"]
            finish_t = F_pshr.get((p_id, s_id, h_id, d, r), 0).x
            start_surg_t = finish_t - surgeon_proc_time
            
            assignments.append({
                "patient_id": int(p_id),
                "surgeon_id": int(s_id),
                "hospital_id": int(h_id),
                "day": int(d),
                "or_id": int(r),
                "is_mandatory": p["is_mandatory"],
                "T_ps": float(total_time),
                "finish_time": float(finish_t),
                "surgery_start_time": float(start_surg_t),
                "room_entry_time": float(start_surg_t - p["preparation_time"]),
                "room_exit_time": float(finish_t + p["cleaning_time"])
            })

    for h_id in hospitals:
        for d in range(num_days):
            for r_idx in range(ORs_per_hospital):
                if y[h_id, d, r_idx].x > 0.9:
                    ot_used = o[h_id, d, r_idx].x
                    reg_time = hospitals[h_id]["ORs"][r_idx]["daily_regular_time"].get(d, 
                        hospitals[h_id]["ORs"][r_idx]["regular_time"])
                    opened_ors.append({
                        "hospital_id": int(h_id),
                        "day": int(d),
                        "or_id": int(r_idx),
                        "regular_time": int(reg_time),
                        "completion_time": float(C_hdr.get((h_id, d, r_idx), 0).x),
                        "overtime": float(ot_used)
                    })
                        
    for s_id in surgeons:
        for h_id in hospitals:
            for d in surgeons[s_id]["operating_days"]:
                if z[s_id, h_id, d].x > 0.9:
                    surgeon_assignments.append({
                        "surgeon_id": int(s_id),
                        "hospital_id": int(h_id),
                        "day": int(d),
                        "start_time": float(0.),  # Not computed explicitly
                        "end_time": float(S_shd.get((s_id, h_id, d), 0).x)
                    })

    for pk, var in u_pkhdr.items():
        if var.x > 0.9:
            p, k, s1, s2, h, d, r = pk
            or_sequences.append({
                "hospital_id": int(h),
                "day": int(d),
                "or_id": int(r),
                "patient_p": int(p),
                "patient_k": int(k),
                "p_after_k": int(1)
            })

    for pk, var in v_pkhshd.items():
        if var.x > 0.9:
            p, k, s, h, d = pk
            surgeon_sequences.append({
                "hospital_id": int(h),
                "day": int(d),
                "surgeon_id": int(s),
                "patient_p": int(p),
                "patient_k": int(k),
                "p_after_k": int(1)
            })

    return {
        "objective_value": sol_obj,
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