import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def optimize(instance):
    num_patients = instance['num_patients']
    num_surgeons = instance['num_surgeons']
    num_hospitals = instance['num_hospitals']
    num_days = instance['num_days']
    alpha = instance['alpha']
    patients = instance['patients']
    surgeons = instance['surgeons']
    hospitals = instance['hospitals']

    model = Model('SurgicalScheduling')
    model.modelSense = GRB.MINIMIZE

    # Decision Variables
    x = {}  # Patient assignment
    y = {}  # OR opening status
    z = {}  # Surgeon assignment to hospitals
    overtime = {}  # Overtime used in each OR

    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient['eligible_ORs_by_hospital']:
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)]:
                                x[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(vtype=GRB.BINARY, name=f'x_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                y[(hospital_id, day, or_id)] = model.addVar(vtype=GRB.BINARY, name=f'y_{hospital_id}_{day}_{or_id}')
                overtime[(hospital_id, day, or_id)] = model.addVar(lb=0, name=f'overtime_{hospital_id}_{day}_{or_id}')

    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days']:
                    z[(surgeon_id, hospital_id, day)] = model.addVar(vtype=GRB.BINARY, name=f'z_{surgeon_id}_{hospital_id}_{day}')

    # Auxiliary Variables
    finish_time = {}  # Finish time of each patient's surgery
    surgery_start_time = {}  # Start time of each patient's surgery
    room_entry_time = {}  # Time patient enters OR
    room_exit_time = {}  # Time patient exits OR
    or_completion_time = {}  # Completion time of each OR
    surgeon_start_time = {}  # Start time of surgeon's work at hospital-day
    surgeon_end_time = {}  # End time of surgeon's work at hospital-day

    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient['eligible_ORs_by_hospital']:
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)]:
                                finish_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'finish_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')
                                surgery_start_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'surgery_start_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')
                                room_entry_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'room_entry_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')
                                room_exit_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'room_exit_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                or_completion_time[(hospital_id, day, or_id)] = model.addVar(lb=0, name=f'or_completion_time_{hospital_id}_{day}_{or_id}')

    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days']:
                    surgeon_start_time[(surgeon_id, hospital_id, day)] = model.addVar(lb=0, name=f'surgeon_start_time_{surgeon_id}_{hospital_id}_{day}')
                    surgeon_end_time[(surgeon_id, hospital_id, day)] = model.addVar(lb=0, name=f'surgeon_end_time_{surgeon_id}_{hospital_id}_{day}')

    # Objective Function
    obj = quicksum(hospitals[hospital_id]['ORs'][or_id]['fixed_cost_per_hour'] * 60 * y[(hospital_id, day, or_id)] for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(instance['ORs_per_hospital'])) + \
          quicksum(hospitals[hospital_id]['surgeon_fixed_costs_per_day'][str(surgeon_id)] * z[(surgeon_id, hospital_id, day)] for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days']) + \
          quicksum(hospitals[hospital_id]['ORs'][or_id]['overtime_cost_per_hour'] * 60 * overtime[(hospital_id, day, or_id)] for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(instance['ORs_per_hospital'])) - \
          quicksum(patient['reward'] * quicksum(x[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)] for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x) for patient in patients if not patient['is_mandatory'])
    model.setObjective(obj)

    # Constraints
    # 1. Mandatory patients must be scheduled exactly once
    for patient in patients:
        if patient['is_mandatory']:
            model.addConstr(quicksum(x[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)] for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x) == 1)

    # 2. Optional patients can be scheduled at most once
    for patient in patients:
        if not patient['is_mandatory']:
            model.addConstr(quicksum(x[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)] for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x) <= 1)

    # 3. Patient can only be assigned if OR is open and surgeon is available
    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient['eligible_ORs_by_hospital']:
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)]:
                                if (patient_id, surgeon_id, hospital_id, day, or_id) in x:
                                    model.addConstr(x[(patient_id, surgeon_id, hospital_id, day, or_id)] <= y[(hospital_id, day, or_id)])
                                    model.addConstr(x[(patient_id, surgeon_id, hospital_id, day, or_id)] <= z[(surgeon_id, hospital_id, day)])

    # 4. OR Capacity Constraint
    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                regular_time = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['regular_time'] if str(day) in hospitals[hospital_id]['ORs'][or_id]['daily'] else hospitals[hospital_id]['ORs'][or_id]['regular_time']
                model.addConstr(quicksum((patient['preparation_time'] + patient['surgeon_specific_times'][str(surgeon_id)] + patient['cleaning_time']) * x[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)] for patient in patients for surgeon_id in patient['eligible_surgeons'] if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x) <= regular_time + overtime[(hospital_id, day, or_id)])

    # 5. Overtime Limit
    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                max_overtime = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['max_overtime'] if str(day) in hospitals[hospital_id]['ORs'][or_id]['daily'] else hospitals[hospital_id]['ORs'][or_id]['max_overtime']
                model.addConstr(overtime[(hospital_id, day, or_id)] <= max_overtime * y[(hospital_id, day, or_id)])

    # 6. Surgeon Availability
    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days']:
                    available_time = surgeons[surgeon_id]['availability_by_day'][str(day)]
                    model.addConstr(quicksum((alpha * patient['surgeon_specific_times'][str(surgeon_id)] + (1 - alpha) * (patient['preparation_time'] + patient['surgeon_specific_times'][str(surgeon_id)] + patient['cleaning_time'])) * x[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)] for patient in patients if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x) <= available_time * z[(surgeon_id, hospital_id, day)])

    # 7. Temporal Constraints
    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                patients_in_or = [(patient['patient_id'], surgeon_id, hospital_id, day, or_id) for patient in patients for surgeon_id in patient['eligible_surgeons'] if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x]
                for i in range(len(patients_in_or)):
                    for j in range(i + 1, len(patients_in_or)):
                        patient_i_id, surgeon_i_id, _, _, _ = patients_in_or[i]
                        patient_j_id, surgeon_j_id, _, _, _ = patients_in_or[j]
                        patient_i = next(patient for patient in patients if patient['patient_id'] == patient_i_id)
                        patient_j = next(patient for patient in patients if patient['patient_id'] == patient_j_id)
                        M = 1e6
                        model.addConstr(room_exit_time[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] + room_entry_time[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] <= M * (1 + x[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] - x[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)]))
                        model.addConstr(room_exit_time[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] + room_entry_time[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] <= M * (1 + x[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] - x[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)]))
                        model.addConstr(finish_time[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] + surgery_start_time[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] <= M * (2 - x[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] - x[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] + (1 if surgeon_i_id == surgeon_j_id else 0)))
                        model.addConstr(finish_time[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] + surgery_start_time[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] <= M * (2 - x[(patient_i_id, surgeon_i_id, hospital_id, day, or_id)] - x[(patient_j_id, surgeon_j_id, hospital_id, day, or_id)] + (1 if surgeon_i_id == surgeon_j_id else 0)))

    # 8. OR Completion Time and Overtime
    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                for patient in patients:
                    for surgeon_id in patient['eligible_surgeons']:
                        if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x:
                            model.addConstr(or_completion_time[(hospital_id, day, or_id)] >= room_exit_time[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)])
                regular_time = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['regular_time'] if str(day) in hospitals[hospital_id]['ORs'][or_id]['daily'] else hospitals[hospital_id]['ORs'][or_id]['regular_time']
                model.addConstr(overtime[(hospital_id, day, or_id)] >= or_completion_time[(hospital_id, day, or_id)] - regular_time)

    # 9. Surgeon Start and End Time
    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days']:
                    for patient in patients:
                        for or_id in range(instance['ORs_per_hospital']):
                            if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x:
                                model.addConstr(surgeon_start_time[(surgeon_id, hospital_id, day)] <= surgery_start_time[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)])
                                model.addConstr(surgeon_end_time[(surgeon_id, hospital_id, day)] >= finish_time[(patient['patient_id'], surgeon_id, hospital_id, day, or_id)])
                    model.addConstr(surgeon_end_time[(surgeon_id, hospital_id, day)] - surgeon_start_time[(surgeon_id, hospital_id, day)] <= surgeons[surgeon_id]['availability_by_day'][str(day)])

    # 10. Room Entry, Surgery Start, Finish, and Room Exit Times
    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient['eligible_ORs_by_hospital']:
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)]:
                                if (patient_id, surgeon_id, hospital_id, day, or_id) in x:
                                    model.addConstr(room_entry_time[(patient_id, surgeon_id, hospital_id, day, or_id)] + patient['preparation_time'] <= surgery_start_time[(patient_id, surgeon_id, hospital_id, day, or_id)])
                                    model.addConstr(surgery_start_time[(patient_id, surgeon_id, hospital_id, day, or_id)] + patient['surgeon_specific_times'][str(surgeon_id)] <= finish_time[(patient_id, surgeon_id, hospital_id, day, or_id)])
                                    model.addConstr(finish_time[(patient_id, surgeon_id, hospital_id, day, or_id)] + patient['cleaning_time'] <= room_exit_time[(patient_id, surgeon_id, hospital_id, day, or_id)])

    model.optimize()
    return model

def construct_solution(instance, model):
    num_patients = instance['num_patients']
    num_surgeons = instance['num_surgeons']
    num_hospitals = instance['num_hospitals']
    num_days = instance['num_days']
    patients = instance['patients']
    surgeons = instance['surgeons']
    hospitals = instance['hospitals']

    solution = {
        "objective_value": model.getObjective().getValue(),
        "assignments": [],
        "opened_ors": [],
        "surgeon_assignments": [],
        "or_sequences": [],
        "surgeon_sequences": []
    }

    x = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'x_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if f'x_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    y = {(hospital_id, day, or_id): model.getVarByName(f'y_{hospital_id}_{day}_{or_id}') for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(instance['ORs_per_hospital']) if f'y_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    z = {(surgeon_id, hospital_id, day): model.getVarByName(f'z_{surgeon_id}_{hospital_id}_{day}') for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days'] if f'z_{surgeon_id}_{hospital_id}_{day}' in [var.VarName for var in model.getVars()]}
    finish_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'finish_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if f'finish_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    surgery_start_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'surgery_start_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if f'surgery_start_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    room_entry_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'room_entry_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if f'room_entry_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    room_exit_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'room_exit_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient['eligible_ORs_by_hospital'] for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)] if f'room_exit_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    or_completion_time = {(hospital_id, day, or_id): model.getVarByName(f'or_completion_time_{hospital_id}_{day}_{or_id}') for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(instance['ORs_per_hospital']) if f'or_completion_time_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    surgeon_start_time = {(surgeon_id, hospital_id, day): model.getVarByName(f'surgeon_start_time_{surgeon_id}_{hospital_id}_{day}') for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days'] if f'surgeon_start_time_{surgeon_id}_{hospital_id}_{day}' in [var.VarName for var in model.getVars()]}
    surgeon_end_time = {(surgeon_id, hospital_id, day): model.getVarByName(f'surgeon_end_time_{surgeon_id}_{hospital_id}_{day}') for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days'] if f'surgeon_end_time_{surgeon_id}_{hospital_id}_{day}' in [var.VarName for var in model.getVars()]}
    overtime = {(hospital_id, day, or_id): model.getVarByName(f'overtime_{hospital_id}_{day}_{or_id}') for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(instance['ORs_per_hospital']) if f'overtime_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}

    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient['eligible_ORs_by_hospital']:
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'][str(hospital_id)]:
                                if (patient_id, surgeon_id, hospital_id, day, or_id) in x and x[(patient_id, surgeon_id, hospital_id, day, or_id)].X > 0.5:
                                    T_ps = patient['preparation_time'] + patient['surgeon_specific_times'][str(surgeon_id)] + patient['cleaning_time']
                                    solution['assignments'].append({
                                        "patient_id": patient_id,
                                        "surgeon_id": surgeon_id,
                                        "hospital_id": hospital_id,
                                        "day": day,
                                        "or_id": or_id,
                                        "is_mandatory": patient['is_mandatory'],
                                        "T_ps": T_ps,
                                        "finish_time": finish_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X,
                                        "surgery_start_time": surgery_start_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X,
                                        "room_entry_time": room_entry_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X,
                                        "room_exit_time": room_exit_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X
                                    })

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                if y[(hospital_id, day, or_id)].X > 0.5:
                    regular_time = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['regular_time'] if str(day) in hospitals[hospital_id]['ORs'][or_id]['daily'] else hospitals[hospital_id]['ORs'][or_id]['regular_time']
                    solution['opened_ors'].append({
                        "hospital_id": hospital_id,
                        "day": day,
                        "or_id": or_id,
                        "regular_time": regular_time,
                        "completion_time": or_completion_time[(hospital_id, day, or_id)].X,
                        "overtime": overtime[(hospital_id, day, or_id)].X
                    })

    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days'] and z[(surgeon_id, hospital_id, day)].X > 0.5:
                    solution['surgeon_assignments'].append({
                        "surgeon_id": surgeon_id,
                        "hospital_id": hospital_id,
                        "day": day,
                        "start_time": surgeon_start_time[(surgeon_id, hospital_id, day)].X,
                        "end_time": surgeon_end_time[(surgeon_id, hospital_id, day)].X
                    })

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(instance['ORs_per_hospital']):
                patients_in_or = [assignment for assignment in solution['assignments'] if assignment['hospital_id'] == hospital_id and assignment['day'] == day and assignment['or_id'] == or_id]
                for i in range(len(patients_in_or)):
                    for j in range(i + 1, len(patients_in_or)):
                        patient_p_id = patients_in_or[i]['patient_id']
                        patient_k_id = patients_in_or[j]['patient_id']
                        # Assuming a specific order based on finish time for simplicity
                        p_after_k = 1 if patients_in_or[i]['finish_time'] > patients_in_or[j]['finish_time'] else 0
                        solution['or_sequences'].append({
                            "hospital_id": hospital_id,
                            "day": day,
                            "or_id": or_id,
                            "patient_p": patient_p_id,
                            "patient_k": patient_k_id,
                            "p_after_k": p_after_k
                        })

    for surgeon in solution['surgeon_assignments']:
        surgeon_id = surgeon['surgeon_id']
        hospital_id = surgeon['hospital_id']
        day = surgeon['day']
        patients_for_surgeon = [assignment for assignment in solution['assignments'] if assignment['surgeon_id'] == surgeon_id and assignment['hospital_id'] == hospital_id and assignment['day'] == day]
        for i in range(len(patients_for_surgeon)):
            for j in range(i + 1, len(patients_for_surgeon)):
                patient_p_id = patients_for_surgeon[i]['patient_id']
                patient_k_id = patients_for_surgeon[j]['patient_id']
                p_after_k = 1 if patients_for_surgeon[i]['finish_time'] > patients_for_surgeon[j]['finish_time'] else 0
                solution['surgeon_sequences'].append({
                    "hospital_id": hospital_id,
                    "day": day,
                    "surgeon_id": surgeon_id,
                    "patient_p": patient_p_id,
                    "patient_k": patient_k_id,
                    "p_after_k": p_after_k
                })

    return solution

def main():
    parser = argparse.ArgumentParser(description='Surgical Scheduling Optimization')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)

    model = Model('SurgicalScheduling')
    model.Params.TimeLimit = args.time_limit
    model = optimize(instance)

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
        logger.log(model.getObjective().getValue())

    solution = construct_solution(instance, model)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()