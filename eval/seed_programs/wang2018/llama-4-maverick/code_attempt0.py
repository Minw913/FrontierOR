import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_vessels = instance['num_vessels']
    num_berths = instance['num_berths']
    num_QCs = instance['num_QCs']
    num_subblocks = instance['num_subblocks']
    H = instance['H']
    E = instance['E']
    penalty_cost_coefficient_co = instance['penalty_cost_coefficient_co']
    QC_capacity_per_timestep = instance['QC_capacity_per_timestep']
    loading_route_lengths_D_L = instance['loading_route_lengths_D_L']
    average_unloading_route_lengths_D_U = instance['average_unloading_route_lengths_D_U']
    vessels = instance['vessels']

    model = gp.Model("BerthAllocation")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Decision Variables
    x = model.addVars(num_vessels, num_berths, vtype=GRB.BINARY, name='x')  # Berth assignment
    y = model.addVars(num_vessels, num_vessels, num_berths, vtype=GRB.BINARY, name='y')  # Vessel precedence at berth
    z = model.addVars(num_vessels, [qc_profile['handling_time'] for vessel in vessels for qc_profile in vessel['qc_profiles']], vtype=GRB.BINARY, name='z')  # QC profile selection
    alpha = model.addVars(num_vessels, vtype=GRB.INTEGER, name='alpha')  # Handling start time step
    subblock_allocation = model.addVars(num_vessels, num_subblocks, vtype=GRB.BINARY, name='subblock_allocation')  # Subblock allocation

    # Auxiliary Variables
    beta = model.addVars(num_vessels, vtype=GRB.INTEGER, name='beta')  # Handling end time step
    rho = model.addVars(num_berths, vtype=GRB.INTEGER, name='rho')  # Earliest start time among vessels at berth
    varsigma = model.addVars(num_berths, vtype=GRB.INTEGER, name='varsigma')  # Latest end time among vessels at berth
    qc_usage = model.addVars(H + E, vtype=GRB.INTEGER, name='qc_usage')  # QC usage at each time step

    # Objective Function
    penalty_cost = gp.quicksum(
        vessels[v]['penalty_cost_coefficient'] * (
            gp.max_(0, vessels[v]['expected_time_window'][0] - alpha[v]) + 
            gp.max_(0, beta[v] - vessels[v]['expected_time_window'][1])
        ) for v in range(num_vessels)
    )
    loading_cost = gp.quicksum(
        x[v, b] * subblock_allocation[v, s] * loading_route_lengths_D_L[s][b] * 
        vessels[v]['loading_containers'] / vessels[v]['reserved_subblocks'] * penalty_cost_coefficient_co 
        for v in range(num_vessels) for b in range(num_berths) for s in range(num_subblocks)
    )
    unloading_cost = gp.quicksum(
        x[v, b] * average_unloading_route_lengths_D_U[b] * vessels[v]['unloading_containers'] * penalty_cost_coefficient_co 
        for v in range(num_vessels) for b in range(num_berths)
    )
    model.setObjective(penalty_cost + loading_cost + unloading_cost, GRB.MINIMIZE)

    # Constraints
    # 1. Each vessel is assigned to exactly one berth
    model.addConstrs((x.sum(v, '*') == 1 for v in range(num_vessels)), name='BerthAssignment')

    # 2. Each vessel selects exactly one QC profile
    offset = 0
    for v in range(num_vessels):
        num_profiles = len(vessels[v]['qc_profiles'])
        model.addConstr(gp.quicksum(z[v, p] for p in range(offset, offset + num_profiles)) == 1, name=f'QCProfileSelection_{v}')
        offset += num_profiles

    # 3. Handling start and end times are within feasible window
    for v in range(num_vessels):
        feasible_start = vessels[v]['feasible_time_window'][0]
        feasible_end = vessels[v]['feasible_time_window'][1]
        model.addConstr(alpha[v] >= feasible_start, name=f'FeasibleStart_{v}')
        model.addConstr(beta[v] <= feasible_end, name=f'FeasibleEnd_{v}')

    # 4. Handling end time is determined by start time and chosen QC profile
    offset = 0
    for v in range(num_vessels):
        num_profiles = len(vessels[v]['qc_profiles'])
        for p in range(num_profiles):
            handling_time = vessels[v]['qc_profiles'][p]['handling_time']
            model.addConstr(beta[v] >= alpha[v] + handling_time - 1 - (1 - z[v, offset + p]) * (H + E), name=f'HandlingTime_{v}_{p}')
        offset += num_profiles

    # 5. Vessels at the same berth are handled sequentially
    for v1 in range(num_vessels):
        for v2 in range(v1 + 1, num_vessels):
            for b in range(num_berths):
                model.addConstr(beta[v1] <= alpha[v2] + (1 - y[v1, v2, b]) * (H + E), name=f'VesselPrecedence_{v1}_{v2}_{b}_1')
                model.addConstr(beta[v2] <= alpha[v1] + y[v1, v2, b] * (H + E), name=f'VesselPrecedence_{v1}_{v2}_{b}_2')
                model.addConstr(y[v1, v2, b] + y[v2, v1, b] >= x[v1, b] + x[v2, b] - 1, name=f'VesselPrecedence_{v1}_{v2}_{b}_3')

    # 6. QC usage does not exceed capacity at any time step
    for t in range(H + E):
        qc_usage_expr = gp.quicksum(
            z[v, offset + p] * vessels[v]['qc_profiles'][p]['qc_per_timestep'][t - alpha[v]] 
            for v in range(num_vessels) for p in range(len(vessels[v]['qc_profiles'])) 
            for offset in [sum(len(vessels[v_]['qc_profiles']) for v_ in range(v))] 
            if 0 <= t - alpha[v] < vessels[v]['qc_profiles'][p]['handling_time']
        )
        model.addConstr(qc_usage[t] == qc_usage_expr, name=f'QCUsage_{t}')
        if t <= E:
            model.addConstr(qc_usage[t] + qc_usage[t + H] <= QC_capacity_per_timestep[t], name=f'QCcapacity_{t}')
        else:
            model.addConstr(qc_usage[t] <= QC_capacity_per_timestep[t], name=f'QCcapacity_{t}')

    # 7. Berth schedule is cyclically consistent
    for b in range(num_berths):
        model.addConstr(rho[b] == gp.min_(alpha[v] for v in range(num_vessels) if x[v, b].x > 0.5), name=f'Rho_{b}')
        model.addConstr(varsigma[b] == gp.max_(beta[v] for v in range(num_vessels) if x[v, b].x > 0.5), name=f'Varsigma_{b}')
        model.addConstr(varsigma[b] - rho[b] <= H - 1, name=f'CyclicConsistency_{b}')

    # 8. Subblock allocation
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(subblock_allocation[v, s] for s in range(num_subblocks)) == vessels[v]['reserved_subblocks'], name=f'SubblockReservation_{v}')
    for s in range(num_subblocks):
        model.addConstr(gp.quicksum(subblock_allocation[v, s] for v in range(num_vessels)) <= 1, name=f'SubblockUniqueness_{s}')

    return model, x, y, z, alpha, beta, subblock_allocation, rho, varsigma, qc_usage

def construct_solution(instance, x, y, z, alpha, beta, subblock_allocation, rho, varsigma, qc_usage):
    vessels = instance['vessels']
    num_vessels = instance['num_vessels']
    num_berths = instance['num_berths']
    num_subblocks = instance['num_subblocks']
    H = instance['H']
    E = instance['E']

    solution = {'vessels': [], 'berths': [], 'qc_usage': {}}
    for v in range(num_vessels):
        vessel_solution = {
            'vessel_id': vessels[v]['vessel_id'],
            'alpha': int(alpha[v].x),
            'beta': int(beta[v].x),
            'berth': int([b for b in range(num_berths) if x[v, b].x > 0.5][0]),
            'profile': int([p for p in range(len(vessels[v]['qc_profiles'])) if z[v, sum(len(vessels[v_]['qc_profiles']) for v_ in range(v)) + p].x > 0.5][0]),
            'subblocks': sorted([s for s in range(num_subblocks) if subblock_allocation[v, s].x > 0.5]),
            'handling_time': vessels[v]['qc_profiles'][int([p for p in range(len(vessels[v]['qc_profiles'])) if z[v, sum(len(vessels[v_]['qc_profiles']) for v_ in range(v)) + p].x > 0.5][0])]['handling_time'],
            'qc_per_timestep': vessels[v]['qc_profiles'][int([p for p in range(len(vessels[v]['qc_profiles'])) if z[v, sum(len(vessels[v_]['qc_profiles']) for v_ in range(v)) + p].x > 0.5][0])]['qc_per_timestep']
        }
        solution['vessels'].append(vessel_solution)

    for b in range(num_berths):
        berth_solution = {
            'berth': b,
            'rho': int(rho[b].x),
            'varsigma': int(varsigma[b].x)
        }
        solution['berths'].append(berth_solution)

    for t in range(H + E):
        if qc_usage[t].x > 0:
            solution['qc_usage'][str(t)] = int(qc_usage[t].x)

    solution['objective_value'] = float(instance['vessels'][0]['penalty_cost_coefficient'] * (max(0, instance['vessels'][0]['expected_time_window'][0] - alpha[0].x) + max(0, beta[0].x - instance['vessels'][0]['expected_time_window'][1])) + 
                                        sum([x[v, b].x * subblock_allocation[v, s].x * instance['loading_route_lengths_D_L'][s][b] * instance['vessels'][v]['loading_containers'] / instance['vessels'][v]['reserved_subblocks'] * instance['penalty_cost_coefficient_co'] for v in range(num_vessels) for b in range(num_berths) for s in range(num_subblocks)]) + 
                                        sum([x[v, b].x * instance['average_unloading_route_lengths_D_U'][b] * instance['vessels'][v]['unloading_containers'] * instance['penalty_cost_coefficient_co'] for v in range(num_vessels) for b in range(num_berths)]))

    return solution

def main():
    parser = argparse.ArgumentParser(description='Berth Allocation Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model, x, y, z, alpha, beta, subblock_allocation, rho, varsigma, qc_usage = solve(instance)
    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if logger:
        if model.SolCount > 0:
            logger.log_solution(model.ObjVal, construct_solution(instance, x, y, z, alpha, beta, subblock_allocation, rho, varsigma, qc_usage))

    if model.SolCount > 0:
        solution = construct_solution(instance, x, y, z, alpha, beta, subblock_allocation, rho, varsigma, qc_usage)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()