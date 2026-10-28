import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_surgery_scheduling(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Extract instance parameters
    num_patients = instance_data['num_patients']
    num_hospitals = instance_data['num_hospitals']
    num_days = instance_data['num_days']
    num_ors_per_hospital = instance_data['num_ors_per_hospital']
    num_scenarios = instance_data['num_scenarios']
    
    global_params = instance_data['global_parameters']
    
    kappa_1 = global_params['kappa_1']
    kappa_2 = global_params['kappa_2'] 
    kappa_3 = global_params['kappa_3']
    kappa_4 = global_params['kappa_4']
    Gamma = global_params['Gamma']
    
    # Parse patient data
    patients = {p['patient_id']: p for p in instance_data['patients']}
    patient_ids = list(patients.keys())
    
    # Parse hospital data
    hospitals_data = {}
    for h_data in instance_data['hospitals']:
        h_id = h_data['hospital_id']
        hospitals_data[h_id] = h_data
    
    B_hd = {}  # Operating time limit per OR
    F_hd = {}  # Cost of opening each OR
    G_hd = {}  # Cost of opening hospital suite
    for h_data in instance_data['hospitals']:
        h_id = h_data['hospital_id']
        for day_data in h_data['days']:
            d_id = day_data['day_id']
            B_hd[(h_id, d_id)] = day_data['B_hd']
            F_hd[(h_id, d_id)] = day_data['F_hd']
            G_hd[(h_id, d_id)] = day_data['G_hd']

    # Parse scenario data
    scenarios = {s['scenario_id']: s for s in instance_data['scenarios']}
    scenario_ids = list(scenarios.keys())

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Build model
    model = gp.Model("SurgeryScheduling")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)  # Single core as required

    # Decision variables
    # u: Hospital-suite open on day (binary)
    u = {}
    for h_id in range(num_hospitals):
        for d_id in range(num_days):
            u[(h_id, d_id)] = model.addVar(vtype=GRB.BINARY, name=f"u_{h_id}_{d_id}")
    
    # y: OR open on day (binary)
    y = {}
    for h_id in range(num_hospitals):
        for d_id in range(num_days):
            for or_idx in range(num_ors_per_hospital):
                y[(h_id, d_id, or_idx)] = model.addVar(vtype=GRB.BINARY, name=f"y_{h_id}_{d_id}_{or_idx}")
    
    # x: Patient assigned to specific OR at hospital on day (binary)
    x = {}
    for h_id in range(num_hospitals):
        for d_id in range(num_days):
            for p_id in patient_ids:
                for or_idx in range(num_ors_per_hospital):
                    x[(h_id, d_id, p_id, or_idx)] = model.addVar(vtype=GRB.BINARY, name=f"x_{h_id}_{d_id}_{p_id}_{or_idx}")
                    
    # w: Postponed non-mandatory patients (binary)
    w = {}
    non_mandatory_patients = [p_id for p_id in patient_ids if not patients[p_id]['is_mandatory']]
    for p_id in non_mandatory_patients:
        w[p_id] = model.addVar(vtype=GRB.BINARY, name=f"w_{p_id}")

    # Vars for second-stage decisions (scenario-dependent cancellations)
    v = {}  # v[s, h, d, p, o] = 1 if patient p's surgery is performed (not cancelled) in scenario s
    for s_id in scenario_ids:
        for h_id in range(num_hospitals):
            for d_id in range(num_days):
                for p_id in patient_ids:
                    for or_idx in range(num_ors_per_hospital):
                        v[(s_id, h_id, d_id, p_id, or_idx)] = model.addVar(
                            vtype=GRB.BINARY, name=f"v_{s_id}_{h_id}_{d_id}_{p_id}_{or_idx}"
                        )

    # Constraints
    # Constraint: Mandatory patients must be scheduled
    for p_id in patient_ids:
        if patients[p_id]['is_mandatory']:
            # Each mandatory patient must be assigned to exactly one OR on some day
            model.addConstr(
                gp.quicksum([
                    x[(h_id, d_id, p_id, or_idx)]
                    for h_id in range(num_hospitals)
                    for d_id in range(num_days)
                    for or_idx in range(num_ors_per_hospital)
                ]) == 1,
                name=f"mandatory_patient_{p_id}"
            )
    
    # Constraint: Non-mandatory patients either scheduled or postponed
    for p_id in non_mandatory_patients:
        model.addConstr(
            w[p_id] + gp.quicksum([
                x[(h_id, d_id, p_id, or_idx)]
                for h_id in range(num_hospitals)
                for d_id in range(num_days)
                for or_idx in range(num_ors_per_hospital)
            ]) == 1,
            name=f"non_mandatory_patient_{p_id}"
        )

    # Constraint: An OR can only be open if the hospital suite is open
    for h_id in range(num_hospitals):
        for d_id in range(num_days):
            for or_idx in range(num_ors_per_hospital):
                model.addConstr(y[(h_id, d_id, or_idx)] <= u[(h_id, d_id)], 
                                name=f"hospital_suite_open_{h_id}_{d_id}_{or_idx}")
                
    # Constraint: Patient can only be scheduled if the OR is open
    for h_id in range(num_hospitals):
        for d_id in range(num_days):
            for p_id in patient_ids:
                for or_idx in range(num_ors_per_hospital):
                    model.addConstr(x[(h_id, d_id, p_id, or_idx)] <= y[(h_id, d_id, or_idx)],
                                    name=f"or_open_for_patient_{h_id}_{d_id}_{p_id}_{or_idx}")

    # Get surgery durations from scenarios
    surgery_durations = {}  # surgery_durations[scenario_id, patient_id]
    for s_id in scenario_ids:
        durations = scenarios[s_id]['surgery_durations_minutes']
        for p_id_idx, dur in enumerate(durations):
            p_id = patient_ids[p_id_idx]  # Assumes order matches
            surgery_durations[(s_id, p_id)] = dur

    # Second-stage constraints for each scenario
    for s_id in scenario_ids:
        # Link first-stage and second-stage variables (must be done per scenario)
        for h_id in range(num_hospitals):
            for d_id in range(num_days):
                for p_id in patient_ids:
                    for or_idx in range(num_ors_per_hospital):
                        # If a patient is not scheduled to this OR, we can't perform
                        model.addConstr(v[(s_id, h_id, d_id, p_id, or_idx)] <= x[(h_id, d_id, p_id, or_idx)],
                                        name=f"perform_vs_schedule_{s_id}_{h_id}_{d_id}_{p_id}_{or_idx}")

        # Capacity constraint per OR per scenario
        for h_id in range(num_hospitals):
            for d_id in range(num_days):
                for or_idx in range(num_ors_per_hospital):
                    model.addConstr(
                        gp.quicksum([
                            surgery_durations[(s_id, p_id)] * v[(s_id, h_id, d_id, p_id, or_idx)]
                            for p_id in patient_ids
                        ]) <= B_hd[(h_id, d_id)],
                        name=f"capacity_{s_id}_{h_id}_{d_id}_{or_idx}"
                    )

    # Objective function components
    # 1. Opening hospital suite costs sum_{h,d} G_{h,d} * u_{h,d}
    suite_cost = gp.quicksum(G_hd[(h_id, d_id)] * u[(h_id, d_id)] 
                             for h_id in range(num_hospitals) for d_id in range(num_days))
    
    # 2. Opening OR costs sum_{h,d,o} F_{h,d} * y_{h,d,o}
    or_cost = gp.quicksum(F_hd[(h_id, d_id)] * y[(h_id, d_id, or_idx)] 
                          for h_id in range(num_hospitals) 
                          for d_id in range(num_days) 
                          for or_idx in range(num_ors_per_hospital))
    
    # 3. Scheduling benefits (negative, subtracted from total cost)
    sched_benefits = gp.quicksum([
        patients[p_id].get('c_sched_per_day', {}).get(str(d_id+1), 0) * x[(h_id, d_id, p_id, or_idx)]
        for h_id in range(num_hospitals)
        for d_id in range(num_days)
        for p_id in patient_ids
        for or_idx in range(num_ors_per_hospital)
    ])
    
    # 4. Postponement penalty for non-mandatory patients
    postponement_penalty = gp.quicksum(
        patients[p_id]['c_unsched'] * w[p_id] 
        for p_id in non_mandatory_patients
    )
    
    # 5. Expected cancellation cost over scenarios
    cancellation_costs = []
    for s_id in scenario_ids:
        # Calculate cancellation cost for this scenario
        scenario_cancellation_cost = gp.quicksum([
            patients[p_id]['c_cancel'] * 
            (x[(h_id, d_id, p_id, or_idx)] - v[(s_id, h_id, d_id, p_id, or_idx)])
            for h_id in range(num_hospitals)
            for d_id in range(num_days)
            for p_id in patient_ids
            for or_idx in range(num_ors_per_hospital)
        ])
        cancellation_costs.append(scenario_cancellation_cost)
        
    # Expected cancellation cost over all scenarios
    expected_cancellation_cost = gp.quicksum(cancellation_costs) / num_scenarios

    # Full objective: minimize total cost
    total_cost = (suite_cost + 
                  or_cost - 
                  sched_benefits + 
                  postponement_penalty + 
                  expected_cancellation_cost)
    
    model.setObjective(total_cost, GRB.MINIMIZE)

    def mycallback(model, where):
        if where == GRB.Callback.MIP:
            obj_val = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj_val)

    # Optimize with callback to log intermediate solutions
    model.optimize(mycallback)

    # Extract solution
    solution = {
        "objective_value": model.objVal,
        "u": {f"{h_id},{d_id}": int(u[(h_id, d_id)].x) 
              for h_id in range(num_hospitals) for d_id in range(num_days)},
        "y": {f"{h_id},{d_id},{or_idx}": int(y[(h_id, d_id, or_idx)].x)
              for h_id in range(num_hospitals) 
              for d_id in range(num_days) 
              for or_idx in range(num_ors_per_hospital)},
        "x": {f"{h_id},{d_id},{p_id},{or_idx}": int(x[(h_id, d_id, p_id, or_idx)].x)
              for h_id in range(num_hospitals)
              for d_id in range(num_days)
              for p_id in patient_ids
              for or_idx in range(num_ors_per_hospital)
              if x[(h_id, d_id, p_id, or_idx)].x > 0.5},
        "w": {f"{p_id}": int(w[p_id].x) for p_id in non_mandatory_patients 
              if w[p_id].x > 0.5}
    }

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write solution JSON file')
    parser.add_argument('--time_limit', type=int, default=600, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to write solution logs')
    
    args = parser.parse_args()
    
    solve_surgery_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)