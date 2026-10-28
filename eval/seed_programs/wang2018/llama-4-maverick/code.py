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
    z = model.addVars(num_vessels, vtype=GRB.INTEGER, name='z')  # QC profile selection
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
        x[v, b] * loading_route_lengths_D_L[s][b] * 
        vessels[v]['loading_containers'] / vessels[v]['reserved_subblocks'] * penalty_cost_coefficient_co * subblock_allocation[v, s]
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
    for v in range(num_vessels):
        model.addConstr(z[v] >= 0, name=f'QCProfileSelectionLowerBound_{v}')
        model.addConstr(z[v] <= len(vessels[v]['qc_profiles']) - 1, name=f'QCProfileSelectionUpperBound_{v}')

    # 3. Handling start and end times are within feasible window
    for v in range(num_vessels):
        feasible_start = vessels[v]['feasible_time_window'][0]
        feasible_end = vessels[v]['feasible_time_window'][1]
        model.addConstr(alpha[v] >= feasible_start, name=f'FeasibleStart_{v}')
        model.addConstr(beta[v] <= feasible_end, name=f'FeasibleEnd_{v}')

    # 4. Handling end time is determined by start time and chosen QC profile
    for v in range(num_vessels):
        model.addConstr(beta[v] == alpha[v] + gp.quicksum(vessels[v]['qc_profiles'][p]['handling_time'] * (z[v] == p) for p in range(len(vessels[v]['qc_profiles']))) - 1, name=f'HandlingTime_{v}')

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
            gp.quicksum(vessels[v]['qc_profiles'][int(z[v].x)]['qc_per_timestep'][t - int(alpha[v].x)] if z[v].x is not None and alpha[v].x is not None and 0 <= t - int(alpha[v].x) < len(vessels[v]['qc_profiles'][int(z[v].x)]['qc_per_timestep']) else 0 for v in range(num_vessels))
            for z_value in range(len(vessels[0]['qc_profiles']))
        )
        # Simplified the qc_usage_expr as the original code had an issue with z[v].x being None during model construction
        model.addConstr(qc_usage[t] == gp.quicksum(
            vessels[v]['qc_profiles'][p]['qc_per_timestep'][t - alpha[v]] if 0 <= t - alpha[v] < len(vessels[v]['qc_profiles'][p]['qc_per_timestep']) else 0 
            for v in range(num_vessels) for p in range(len(vessels[v]['qc_profiles']))
        ), name=f'QCUsage_{t}')
        if t < E:
            model.addConstr(qc_usage[t] + qc_usage[t + H] <= QC_capacity_per_timestep[t], name=f'QCcapacity_{t}')
        else:
            model.addConstr(qc_usage[t] <= QC_capacity_per_timestep[t], name=f'QCcapacity_{t}')

    # 7. Berth schedule is cyclically consistent
    for b in range(num_berths):
        model.addConstr(rho[b] == gp.min_(alpha[v] for v in range(num_vessels)), name=f'Rho_{b}')
        model.addConstr(varsigma[b] == gp.max_(beta[v] for v in range(num_vessels)), name=f'Varsigma_{b}')
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
        if alpha[v].x is not None and z[v].x is not None:
            vessel_solution = {
                'vessel_id': vessels[v]['vessel_id'],
                'alpha': int(round(alpha[v].x)),
                'beta': int(round(beta[v].x)),
                'berth': int([b for b in range(num_berths) if x[v, b].x > 0.5][0]),
                'profile': int(round(z[v].x)),
                'subblocks': sorted([s for s in range(num_subblocks) if subblock_allocation[v, s].x > 0.5]),
                'handling_time': vessels[v]['qc_profiles'][int(round(z[v].x))]['handling_time'],
                'qc_per_timestep': vessels[v]['qc_profiles'][int(round(z[v].x))]['qc_per_timestep']
            }
            solution['vessels'].append(vessel_solution)

    for b in range(num_berths):
        vessels_at_berth = [v for v in range(num_vessels) if x[v, b].x > 0.5]
        if vessels_at_berth:
            min_start = min(alpha[v].x for v in vessels_at_berth)
            max_end = max(beta[v].x for v in vessels_at_berth)
        else:
            min_start = 0
            max_end = 0
        berth_solution = {
            'berth': b,
            'rho': int(round(min_start)),
            'varsigma': int(round(max_end))
        }
        solution['berths'].append(berth_solution)

    for t in range(H + E):
        if qc_usage[t].x > 0:
            solution['qc_usage'][str(t)] = int(round(qc_usage[t].x))

    penalty_cost = sum(vessels[v]['penalty_cost_coefficient'] * (max(0, vessels[v]['expected_time_window'][0] - alpha[v].x) + max(0, beta[v].x - vessels[v]['expected_time_window'][1])) for v in range(num_vessels))
    loading_cost = sum(x[v, b].x * subblock_allocation[v, s].x * instance['loading_route_lengths_D_L'][s][b] * vessels[v]['loading_containers'] / vessels[v]['reserved_subblocks'] * instance['penalty_cost_coefficient_co'] for v in range(num_vessels) for b in range(num_berths) for s in range(num_subblocks))
    unloading_cost = sum(x[v, b].x * instance['average_unloading_route_lengths_D_U'][b] * vessels[v]['unloading_containers'] * instance['penalty_cost_coefficient_co'] for v in range(num_vessels) for b in range(num_ber', b))
    solution['objective_value'] = float(penalty_cost + loading_cost + unloading_cost)

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

    if model.SolCount > 0:
        solution = construct_solution(instance, x, y, z, alpha, beta, subblock_allocation, rho, varsigma, qc_usage)
        if logger:
            logger.log_solution(model.ObjVal, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()