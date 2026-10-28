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
    ORs_per_hospital = instance['ORs_per_hospital']

    model = Model('SurgicalScheduling')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    # Decision Variables
    x = {}  
    y = {}  
    z = {}  
    overtime = {}  

    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}):
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'].get(str(hospital_id), []):
                                x[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(vtype=GRB.BINARY, name=f'x_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(ORs_per_hospital):
                y[(hospital_id, day, or_id)] = model.addVar(vtype=GRB.BINARY, name=f'y_{hospital_id}_{day}_{or_id}')
                overtime[(hospital_id, day, or_id)] = model.addVar(lb=0, name=f'overtime_{hospital_id}_{day}_{or_id}')

    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days']:
                    z[(surgeon_id, hospital_id, day)] = model.addVar(vtype=GRB.BINARY, name=f'z_{surgeon_id}_{hospital_id}_{day}')

    # Auxiliary Variables
    finish_time = {}  
    surgery_start_time = {}  
    room_entry_time = {}  
    room_exit_time = {}  
    or_completion_time = {}  
    surgeon_start_time = {}  
    surgeon_end_time = {}  

    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}):
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'].get(str(hospital_id), []):
                                if (patient_id, surgeon_id, hospital_id, day, or_id) in x:
                                    finish_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'finish_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')
                                    surgery_start_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'surgery_start_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')
                                    room_entry_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'room_entry_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')
                                    room_exit_time[(patient_id, surgeon_id, hospital_id, day, or_id)] = model.addVar(lb=0, name=f'room_exit_time_{patient_id}_{surgeon_id}_{hospital_id}_{day}_{or_id}')

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(ORs_per_hospital):
                or_completion_time[(hospital_id, day, or_id)] = model.addVar(lb=0, name=f'or_completion_time_{hospital_id}_{day}_{or_id}')

    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if day in surgeons[surgeon_id]['operating_days']:
                    surgeon_start_time[(surgeon_id, hospital_id, day)] = model.addVar(lb=0, name=f'surgeon_start_time_{surgeon_id}_{hospital_id}_{day}')
                    surgeon_end_time[(surgeon_id, hospital_id, day)] = model.addVar(lb=0, name=f'surgeon_end_time_{surgeon_id}_{hospital_id}_{day}')

    # Objective Function
    obj = quicksum(hospitals[hospital_id]['ORs'][or_id].get('fixed_cost_per_hour', 0) * y[(hospital_id, day, or_id)] for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(ORs_per_hospital) if (hospital_id, day, or_id) in y) + \
          quicksum(hospitals[hospital_id].get('surgeon_fixed_costs_per_day', {}).get(str(surgeon_id), 0) * z[(surgeon_id, hospital_id, day)] for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if (surgeon_id, hospital_id, day) in z) + \
          quicksum(hospitals[hospital_id]['ORs'][or_id].get('overtime_cost_per_hour', 0) / 60 * overtime[(hospital_id, day, or_id)] for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(ORs_per_hospital) if (hospital_id, day, or_id) in overtime) - \
          quicksum(patient.get('reward', 0) * quicksum(x.get((patient['patient_id'], surgeon_id, hospital_id, day, or_id), 0) for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), [])) for patient in patients if not patient['is_mandatory'])
    model.setObjective(obj)

    # Constraints
    model.update()
    # 1. Mandatory patients must be scheduled exactly once
    for patient in patients:
        if patient['is_mandatory']:
            model.addConstr(quicksum([x.get((patient['patient_id'], surgeon_id, hospital_id, day, or_id), 0) for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), [])]) == 1)

    # 2. Optional patients can be scheduled at most once
    for patient in patients:
        if not patient['is_mandatory']:
            model.addConstr(quicksum([x.get((patient['patient_id'], surgeon_id, hospital_id, day, or_id), 0) for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), [])]) <= 1)

    # 3. Patient can only be assigned if OR is open and surgeon is available
    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}):
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'].get(str(hospital_id), []):
                                if (patient_id, surgeon_id, hospital_id, day, or_id) in x:
                                    model.addConstr(x[(patient_id, surgeon_id, hospital_id, day, or_id)] <= y.get((hospital_id, day, or_id), 0))
                                    model.addConstr(x[(patient_id, surgeon_id, hospital_id, day, or_id)] <= z.get((surgeon_id, hospital_id, day), 0))

    # 4. OR Capacity Constraint
    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(ORs_per_hospital):
                if (hospital_id, day, or_id) in y:
                    regular_time = hospitals[hospital_id]['ORs'][or_id].get('regular_time', 480)  
                    if str(day) in hospitals[hospital_id]['ORs'][or_id].get('daily', {}):
                        regular_time = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['regular_time']
                    model.addConstr(quicksum([(patient['preparation_time'] + patient['surgeon_specific_times'][str(surgeon_id)] + patient['cleaning_time']) * x.get((patient['patient_id'], surgeon_id, hospital_id, day, or_id), 0) for patient in patients for surgeon_id in patient['eligible_surgeons'] if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x]) <= regular_time + overtime[(hospital_id, day, or_id)])

    # 5. Overtime Limit
    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(ORs_per_hospital):
                if (hospital_id, day, or_id) in y:
                    max_overtime = hospitals[hospital_id]['ORs'][or_id].get('max_overtime', 120)  
                    if str(day) in hospitals[hospital_id]['ORs'][or_id].get('daily', {}):
                        max_overtime = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['max_overtime']
                    model.addConstr(overtime[(hospital_id, day, or_id)] <= max_overtime * y[(hospital_id, day, or_id)])

    # 6. Surgeon Availability
    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if (surgeon_id, hospital_id, day) in z:
                    available_time = surgeons[surgeon_id]['availability_by_day'][str(day)]
                    model.addConstr(quicksum([(alpha * patient['surgeon_specific_times'][str(surgeon_id)] + (1 - alpha) * (patient['preparation_time'] + patient['surgeon_specific_times'][str(surgeon_id)] + patient['cleaning_time'])) * x.get((patient['patient_id'], surgeon_id, hospital_id, day, or_id), 0) for patient in patients for or_id in range(ORs_per_hospital) if (patient['patient_id'], surgeon_id, hospital_id, day, or_id) in x]) <= available_time * z[(surgeon_id, hospital_id, day)])

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
    ORs_per_hospital = instance['ORs_per_hospital']

    solution = {
        "objective_value": model.getObjective().getValue(),
        "assignments": [],
        "opened_ors": [],
        "surgeon_assignments": [],
        "or_sequences": [],
        "surgeon_sequences": []
    }

    x = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'x_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), []) if f'x_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    y = {(hospital_id, day, or_id): model.getVarByName(f'y_{hospital_id}_{day}_{or_id}') for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(ORs_per_hospital) if f'y_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    z = {(surgeon_id, hospital_id, day): model.getVarByName(f'z_{surgeon_id}_{hospital_id}_{day}') for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days'] if f'z_{surgeon_id}_{hospital_id}_{day}' in [var.VarName for var in model.getVars()]}
    finish_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'finish_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), []) if f'finish_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    surgery_start_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'surgery_start_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), []) if f'surgery_start_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    room_entry_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'room_entry_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), []) if f'room_entry_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    room_exit_time = {(patient['patient_id'], surgeon_id, hospital_id, day, or_id): model.getVarByName(f'room_exit_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}') for patient in patients for surgeon_id in patient['eligible_surgeons'] for hospital_id in range(num_hospitals) if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}) for day in range(num_days) if day <= patient['due_date'] - 1 for or_id in patient.get('eligible_ORs_by_hospital', {}).get(str(hospital_id), []) if f'room_exit_time_{patient["patient_id"]}_{surgeon_id}_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    or_completion_time = {(hospital_id, day, or_id): model.getVarByName(f'or_completion_time_{hospital_id}_{day}_{or_id}') for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(ORs_per_hospital) if f'or_completion_time_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}
    surgeon_start_time = {(surgeon_id, hospital_id, day): model.getVarByName(f'surgeon_start_time_{surgeon_id}_{hospital_id}_{day}') for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days'] if f'surgeon_start_time_{surgeon_id}_{hospital_id}_{day}' in [var.VarName for var in model.getVars()]}
    surgeon_end_time = {(surgeon_id, hospital_id, day): model.getVarByName(f'surgeon_end_time_{surgeon_id}_{hospital_id}_{day}') for surgeon_id in range(num_surgeons) for hospital_id in range(num_hospitals) for day in range(num_days) if day in surgeons[surgeon_id]['operating_days'] if f'surgeon_end_time_{surgeon_id}_{hospital_id}_{day}' in [var.VarName for var in model.getVars()]}
    overtime = {(hospital_id, day, or_id): model.getVarByName(f'overtime_{hospital_id}_{day}_{or_id}') for hospital_id in range(num_hospitals) for day in range(num_days) for or_id in range(ORs_per_hospital) if f'overtime_{hospital_id}_{day}_{or_id}' in [var.VarName for var in model.getVars()]}

    for patient in patients:
        patient_id = patient['patient_id']
        for surgeon_id in patient['eligible_surgeons']:
            for hospital_id in range(num_hospitals):
                if str(hospital_id) in patient.get('eligible_ORs_by_hospital', {}):
                    for day in range(num_days):
                        if day <= patient['due_date'] - 1:
                            for or_id in patient['eligible_ORs_by_hospital'].get(str(hospital_id), []):
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
                                        "finish_time": finish_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X if (patient_id, surgeon_id, hospital_id, day, or_id) in finish_time else 0,
                                        "surgery_start_time": surgery_start_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X if (patient_id, surgeon_id, hospital_id, day, or_id) in surgery_start_time else 0,
                                        "room_entry_time": room_entry_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X if (patient_id, surgeon_id, hospital_id, day, or_id) in room_entry_time else 0,
                                        "room_exit_time": room_exit_time[(patient_id, surgeon_id, hospital_id, day, or_id)].X if (patient_id, surgeon_id, hospital_id, day, or_id) in room_exit_time else 0
                                    })

    for hospital_id in range(num_hospitals):
        for day in range(num_days):
            for or_id in range(ORs_per_hospital):
                if (hospital_id, day, or_id) in y and y[(hospital_id, day, or_id)].X > 0.5:
                    regular_time = hospitals[hospital_id]['ORs'][or_id].get('regular_time', 480)  
                    if str(day) in hospitals[hospital_id]['ORs'][or_id].get('daily', {}):
                        regular_time = hospitals[hospital_id]['ORs'][or_id]['daily'][str(day)]['regular_time']
                    solution['opened_ors'].append({
                        "hospital_id": hospital_id,
                        "day": day,
                        "or_id": or_id,
                        "regular_time": regular_time,
                        "completion_time": or_completion_time[(hospital_id, day, or_id)].X if (hospital_id, day, or_id) in or_completion_time else 0,
                        "overtime": overtime[(hospital_id, day, or_id)].X if (hospital_id, day, or_id) in overtime else 0
                    })

    for surgeon_id in range(num_surgeons):
        for hospital_id in range(num_hospitals):
            for day in range(num_days):
                if (surgeon_id, hospital_id, day) in z and z[(surgeon_id, hospital_id, day)].X > 0.5:
                    solution['surgeon_assignments'].append({
                        "surgeon_id": surgeon_id,
                        "hospital_id": hospital_id,
                        "day": day,
                        "start_time": surgeon_start_time[(surgeon_id, hospital_id, day)].X if (surgeon_id, hospital_id, day) in surgeon_start_time else 0,
                        "end_time": surgeon_end_time[(surgeon_id, hospital_id, day)].X if (surgeon_id, hospital_id, day) in surgeon_end_time else 0
                    })

    # Sequences are not directly available from the variables, so they are not filled here

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
    model.setParam("Threads", 1)
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