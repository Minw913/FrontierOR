import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance, time_limit, logger):
    num_generators = instance['num_generators']
    num_time_periods = instance['num_time_periods']
    time_period_hours = instance['time_period_hours']
    generators = instance['generators']
    demand = instance['demand']
    reserve_requirement = instance['reserve_requirement']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Variables
    u = {(g['id'], t): model.addVar(vtype=GRB.BINARY, name=f"u_{g['id']}_{t}") for g in generators for t in range(num_time_periods)}
    p = {(g['id'], t): model.addVar(lb=0, name=f"p_{g['id']}_{t}") for g in generators for t in range(num_time_periods)}
    pb = {(g['id'], t): model.addVar(lb=0, name=f"pb_{g['id']}_{t}") for g in generators for t in range(num_time_periods)}
    startup_cost = {(g['id'], t): model.addVar(lb=0, name=f"startup_cost_{g['id']}_{t}") for g in generators for t in range(num_time_periods)}

    # Objective
    production_cost = gp.quicksum(
        g['no_load_cost'] * u[g['id'], t] 
        for g in generators for t in range(num_time_periods)
    )
    for g in generators:
        for t in range(num_time_periods):
            cost = 0
            p_gt = p[g['id'], t]
            for segment in g['production_cost_segments']:
                segment_output = gp.min_(p_gt, segment['output_mw_end'] - segment['output_mw_start'])
                cost += segment['marginal_cost_per_mwh'] * time_period_hours * segment_output
                p_gt = gp.max_(p_gt - (segment['output_mw_end'] - segment['output_mw_start']), 0)
            production_cost += cost
    model.setObjective(
        production_cost + 
        gp.quicksum(startup_cost[g['id'], t] for g in generators for t in range(num_time_periods)), 
        GRB.MINIMIZE
    )

    # Constraints
    for t in range(num_time_periods):
        model.addConstr(gp.quicksum(p[g['id'], t] for g in generators) == demand[t], name=f"demand_{t}")
        model.addConstr(gp.quicksum(pb[g['id'], t] for g in generators) >= demand[t] + reserve_requirement[t], name=f"reserve_{t}")

    for g in generators:
        g_id = g['id']
        for t in range(num_time_periods):
            model.addConstr(p[g_id, t] >= g['p_min'] * u[g_id, t], name=f"p_min_{g_id}_{t}")
            model.addConstr(p[g_id, t] <= g['p_max'] * u[g_id, t], name=f"p_max_{g_id}_{t}")
            model.addConstr(pb[g_id, t] >= p[g_id, t], name=f"pb_min_{g_id}_{t}")
            model.addConstr(pb[g_id, t] <= g['p_max'] * u[g_id, t], name=f"pb_max_{g_id}_{t}")

            if t > 0:
                model.addConstr(u[g_id, t] - u[g_id, t-1] <= 1, name=f"startup_{g_id}_{t}")
                model.addConstr(u[g_id, t-1] - u[g_id, t] <= 1, name=f"shutdown_{g_id}_{t}")
                model.addConstr(p[g_id, t] - p[g_id, t-1] <= g['ramp_up_limit'] * u[g_id, t-1] + g['startup_ramp_limit'] * (u[g_id, t] - u[g_id, t-1]), name=f"ramp_up_{g_id}_{t}")
                model.addConstr(p[g_id, t-1] - p[g_id, t] <= g['ramp_down_limit'] * u[g_id, t] + g['shutdown_ramp_limit'] * (u[g_id, t-1] - u[g_id, t]), name=f"ramp_down_{g_id}_{t}")

            # Startup cost
            if t == 0:
                if g['initial_status'] < 0:
                    downtime = -g['initial_status'] * time_period_hours
                    for i, sc in enumerate(g['startup_costs']):
                        if downtime <= sc['max_downtime_hours'] or sc['max_downtime_hours'] is None:
                            model.addConstr(startup_cost[g_id, t] >= sc['cost'] * (u[g_id, t]), name=f"startup_cost_{g_id}_{t}_{i}")
                            break
            else:
                model.addConstr(startup_cost[g_id, t] >= g['startup_costs'][0]['cost'] * (u[g_id, t] - u[g_id, t-1]), name=f"startup_cost_logic_{g_id}_{t}")

        # Minimum up/down time constraints
        if g['min_up_time'] > 1:
            for t in range(num_time_periods - g['min_up_time'] + 1):
                model.addConstr(gp.quicksum(u[g_id, k] for k in range(t, t + g['min_up_time'])) >= g['min_up_time'] * (u[g_id, t + g['min_up_time'] - 1] - u[g_id, t - 1] if t > 0 else u[g_id, t + g['min_up_time'] - 1]), name=f"min_up_time_{g_id}_{t}")
        if g['min_down_time'] > 1:
            for t in range(num_time_periods - g['min_down_time'] + 1):
                model.addConstr(gp.quicksum(1 - u[g_id, k] for k in range(t, t + g['min_down_time'])) >= g['min_down_time'] * (u[g_id, t - 1] - u[g_id, t + g['min_down_time'] - 1] if t > 0 else 1 - u[g_id, t + g['min_down_time'] - 1]), name=f"min_down_time_{g_id}_{t}")

        # Initial conditions
        if g['initial_status'] > 0:
            model.addConstr(u[g_id, 0] == 1, name=f"initial_on_{g_id}")
            if g['initial_status'] < g['min_up_time']:
                for t in range(1, min(g['min_up_time'] - g['initial_status'], num_time_periods)):
                    model.addConstr(u[g_id, t] == 1, name=f"initial_min_up_{g_id}_{t}")
        else:
            model.addConstr(u[g_id, 0] == 0, name=f"initial_off_{g_id}")
            if -g['initial_status'] < g['min_down_time']:
                for t in range(1, min(g['min_down_time'] + g['initial_status'], num_time_periods)):
                    model.addConstr(u[g_id, t] == 0, name=f"initial_min_down_{g_id}_{t}")

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.getObjective().getValue())

    model.optimize(callback)

    return model

def construct_solution(instance, model):
    generators = instance['generators']
    num_time_periods = instance['num_time_periods']
    solution = {'schedule': {}}
    for g in generators:
        g_id = g['id']
        solution['schedule'][g_id] = {
            'u': [int(model.getVarByName(f"u_{g_id}_{t}").x) for t in range(num_time_periods)],
            'p': [model.getVarByName(f"p_{g_id}_{t}").x for t in range(num_time_periods)],
            'pb': [model.getVarByName(f"pb_{g_id}_{t}").x for t in range(num_time_periods)]
        }
    solution['objective_value'] = model.getObjective().getValue() if model.SolCount > 0 else float('inf')
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    model = solve(instance, args.time_limit, logger)

    solution = construct_solution(instance, model)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()