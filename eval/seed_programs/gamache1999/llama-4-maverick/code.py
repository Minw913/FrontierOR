import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    model = Model("CrewRostering")
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  
    
    num_crew_members = instance['num_crew_members']
    num_pairings = instance['num_mother_pairings']
    pairings = instance['pairings']
    crew_members = instance['crew_members']
    
    CrewMembers = range(num_crew_members)
    Pairings = range(num_pairings)
    
    pairing_start = {i: pairings[i]['start_day'] * 24 + pairings[i]['departure_hour'] for i in Pairings}
    pairing_end = {i: (pairings[i]['start_day'] + pairings[i]['duration_days']) * 24 + pairings[i]['return_hour'] for i in Pairings}
    crew_qualifications = {i: set(crew_members[i]['qualifications']) for i in CrewMembers}
    pre_assigned_activities = {i: crew_members[i]['pre_assigned_activities'] for i in CrewMembers}
    
    assignment = model.addVars(num_crew_members, num_pairings, vtype=GRB.BINARY, name='assignment')
    uncovered_pairings = model.addVars(num_pairings, vtype=GRB.INTEGER, name='uncovered_pairings', lb=0)
    
    model.setObjective(quicksum(uncovered_pairings[i] * pairings[i]['duration_minutes'] for i in Pairings), GRB.MINIMIZE)
    
    for i in Pairings:
        model.addConstr(quicksum(assignment[c, i] for c in CrewMembers) + uncovered_pairings[i] == pairings[i]['coverage_requirement'], name=f'coverage_{i}')
        
        for c in CrewMembers:
            if pairings[i]['aircraft_type'] not in crew_qualifications[c]:
                model.addConstr(assignment[c, i] == 0, name=f'qualification_{c}_{i}')
            for activity in pre_assigned_activities[c]:
                if (pairing_start[i] < (activity['end_day'] + 1) * 24 and 
                    pairing_end[i] > activity['start_day'] * 24):
                    model.addConstr(assignment[c, i] == 0, name=f'overlap_{c}_{i}_{activity}')
    
    model.update()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    model = solve(instance)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model.Params.TimeLimit = args.time_limit
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            "objective_value": model.getObjective().getValue(),
            "rosters": {},
            "uncovered_pairings": []
        }
        
        assignment_vars = model.getVars()
        assignment_dict = {}
        uncovered_pairings_dict = {}
        
        for v in assignment_vars:
            if v.varName.startswith('assignment'):
                parts = v.varName.split('[')
                indices = parts[1].split(']')[0].split(',')
                c = int(indices[0])
                i = int(indices[1])
                assignment_dict[(c, i)] = v.X
            elif v.varName.startswith('uncovered_pairings'):
                index = int(v.varName.split('[')[1].split(']')[0])
                uncovered_pairings_dict[index] = v.X
        
        for c in range(instance['num_crew_members']):
            crew_pairings = [i for i in range(instance['num_mother_pairings']) if (c, i) in assignment_dict and assignment_dict[(c, i)] > 0.5]
            solution['rosters'][str(c)] = {
                "roster_index": 0,  
                "pairings": crew_pairings,
                "unproductive_time": 0  
            }
        
        for i in range(instance['num_mother_pairings']):
            if i in uncovered_pairings_dict and uncovered_pairings_dict[i] > 0:
                solution['uncovered_pairings'].append({
                    "pairing_id": i,
                    "uncovered_count": int(uncovered_pairings_dict[i]),
                    "duration_minutes": instance['pairings'][i]['duration_minutes']
                })
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        
        if logger:
            logger.log(model.getObjective().getValue())

if __name__ == "__main__":
    main()