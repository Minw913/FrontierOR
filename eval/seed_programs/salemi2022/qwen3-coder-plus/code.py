import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_power_system(instance_data, time_limit):
    # Extract instance data
    n_generators = instance_data["n_generators"]
    n_scenarios = instance_data["n_scenarios"]
    T = instance_data["T"]
    generators = instance_data["generators"]
    demand_scenarios = np.array(instance_data["demand_scenarios"])
    reserve_scenarios = np.array(instance_data["reserve_scenarios"])

    # Create model
    model = gp.Model("PowerSystem")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Decision Variables
    x = {}  # Commitment status
    y = {}  # Start-up indicators
    y_bar = {}  # Shut-down indicators
    q = {}  # Start-up costs
    p = {}  # Production levels
    p_bar = {}  # Available capacity

    for g in range(n_generators):
        for t in range(T):
            x[g,t] = model.addVar(vtype=GRB.BINARY, name=f"x_{g}_{t}")
            y[g,t] = model.addVar(vtype=GRB.BINARY, name=f"y_{g}_{t}")
            y_bar[g,t] = model.addVar(vtype=GRB.BINARY, name=f"y_bar_{g}_{t}")
            q[g,t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"q_{g}_{t}")
            
            for s in range(n_scenarios):
                p[g,s,t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{g}_{s}_{t}")
                p_bar[g,s,t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_bar_{g}_{s}_{t}")

    # Constraint 1: Linking start-up and shut-down indicators to commitment status
    for g in range(n_generators):
        model.addConstr(x[g,0] == y[g,0] - y_bar[g,0])
        for t in range(1, T):
            model.addConstr(x[g,t] - x[g,t-1] == y[g,t] - y_bar[g,t])

    # Constraint 2: Minimum up-time requirement
    for g in range(n_generators):
        L = generators[g]["L"]
        for t in range(T):
            if t >= L - 1:
                # Sum of start-ups from period t-L+1 to t should not exceed x[g,t]
                start_period = t - L + 1
                model.addConstr(
                    gp.quicksum(y[g,k] for k in range(start_period, t+1)) <= x[g,t]
                )

    # Constraint 3: Minimum down-time requirement
    for g in range(n_generators):
        ell = generators[g]["ell"]
        for t in range(T):
            if t >= ell - 1:
                # Sum of shut-downs from period t-ell+1 to t should not exceed 1 - x[g,t]
                shutdown_period = t - ell + 1
                model.addConstr(
                    gp.quicksum(y_bar[g,k] for k in range(shutdown_period, t+1)) <= 1 - x[g,t]
                )

    # Constraint 4: Start-up cost constraint
    for g in range(n_generators):
        c_g_start = generators[g]["startup_costs_K"]
        for t in range(T):
            for k in range(1, min(t + 2, T + 1)):  # k represents consecutive inactive periods
                # Need to sum up the commitment statuses k periods back
                if t - k >= 0:
                    # The actual expression is x[g,t] - sum[x[g,t-k+1]..x[g,t-1]]
                    # Rearranging to ensure that if a start-up occurs after k inactive periods,
                    # the associated startup cost is satisfied
                    model.addConstr(
                        q[g,t] >= c_g_start[k-1] * (x[g,t] - gp.quicksum(x[g,i] for i in range(t-k+1, t)))
                    )
                else:
                    # For case where we go back to beginning and consider what was committed before
                    model.addConstr(q[g,t] >= c_g_start[k-1] * (x[g,t]))  # Since x[g,-1] is not defined, assuming generator is off initially

    # Handle initial startup properly
    # At t=0, the only way to incur startup cost is if we were off and now starting
    for g in range(n_generators):
        for t in range(T):
            # For any k, we need to check all possible inactive counts leading to current t
            for k in range(1, t + 2):
                if k <= t + 1:
                    # If we're considering a sequence of k-1 inactive periods followed by start at t
                    # Then we look at x[g,t-i] from i in [k-1, k-2, ..., 1] - none true means k-1 inactives
                    inactive_sum = 0
                    indices = []
                    for i in range(k-1):
                        period_idx = t - i - 1
                        if period_idx >= 0:
                            indices.append(period_idx)
                    
                    if len(indices) > 0:
                        inactive_sum = gp.quicksum(x[g,idx] for idx in indices)
                    else:
                        # Before T=0, assumes all generators are off
                        inactive_sum = 0
                    
                    model.addConstr(q[g,t] >= c_g_start[min(k-1, len(c_g_start)-1)] * (x[g,t] - inactive_sum))

    # Simpler implementation for start-up costs based on the condition interpretation:
    # For each k = 1..T, the cost q[g,t] >= u_k * (x[g,t] - sum of previous consecutive k commitments)
    # Where u_k is the startup cost after k consecutive inactive periods
    for g in range(n_generators):
        startup_costs = generators[g]["startup_costs_K"]
        for t in range(T):
            for k in range(1, min(t + 2, T + 1)):
                past_sum = 0
                valid_periods_to_sum = max(0, t - k + 1)
                if valid_periods_to_sum <= t: 
                    past_range = range(max(0, t - k + 1), t)  # Only sum from previous periods
                    if len(past_range) > 0:
                        past_sum = gp.quicksum(x[g,i] for i in past_range)
                    else:
                        # If t-k+1 >= t, then no previous periods to sum
                        past_sum = 0
                
                # Apply constraint for k consecutive inactive periods
                # x[g,t] - past_sum will be positive only if x[g,t]=1 and the k past periods were mostly 0
                # This models the proper relationship
                model.addConstr(
                    q[g,t] >= startup_costs[min(k-1, len(startup_costs)-1)] * 
                    (x[g,t] - past_sum) 
                )

    # Constraint 5: Production bounds when generator is off
    for g in range(n_generators):
        m_g = generators[g]["m"]  # minimum production
        M_g = generators[g]["M"]  # maximum production
        for s in range(n_scenarios):
            for t in range(T):
                # When x[g,t] = 0, p[g,s,t] and p_bar[g,s,t] = 0
                model.addConstr(p[g,s,t] <= x[g,t] * M_g)  # If off, production is 0
                model.addConstr(p_bar[g,s,t] <= x[g,t] * M_g)  # If off, available capacity is 0
                
                # When x[g,t] = 1, production bounds apply
                model.addConstr(p[g,s,t] >= x[g,t] * m_g)
                model.addConstr(p_bar[g,s,t] <= x[g,t] * M_g)
                # Available capacity bounds
                model.addConstr(p_bar[g,s,t] >= x[g,t] * m_g)
                model.addConstr(p_bar[g,s,t] <= M_g)

    # Constraint 6: Ramp limits
    for g in range(n_generators):
        RU = generators[g]["RU"]
        SU = generators[g]["SU"]
        RD = generators[g]["RD"]
        SD = generators[g]["SD"]
        for s in range(n_scenarios):
            # At t=0, we assume the generator was off previously
            for t in range(T):
                if t == 0:
                    # Increase constraint for t=0: p[g,s,0] <= SU * y[g,0] + RU * x[g,0]
                    model.addConstr(p[g,s,t] <= SU * y[g,t])  # Since initially off, production from startup only limited by SU
                else:
                    # Increase: p[g,s,t] - p[g,s,t-1] <= RU * x[g,t-1] + SU * y[g,t]
                    model.addConstr(
                        p[g,s,t] - p[g,s,t-1] <= RU * x[g,t-1] + SU * y[g,t]
                    )
                
                # Decrease limitation: p[g,s,t-1] - p[g,s,t] <= RD * x[g,t] + SD * y_bar[g,t]
                if t > 0:
                    model.addConstr(
                        p[g,s,t-1] - p[g,s,t] <= RD * x[g,t] + SD * y_bar[g,t]
                    )
                else:
                    # For t=0, we only need to consider if the generator would be shutting down from previous
                    # Since x[g,-1] = 0 by assumption, there's no previous operation to shut down from
                    pass

    # Constraint 7: Demand satisfaction
    for s in range(n_scenarios):
        for t in range(T):
            model.addConstr(
                gp.quicksum(p[g,s,t] for g in range(n_generators)) >= demand_scenarios[s][t]
            )

    # Constraint 8: Reserve requirements
    for s in range(n_scenarios):
        for t in range(T):
            model.addConstr(
                gp.quicksum(p_bar[g,s,t] for g in range(n_generators)) >= demand_scenarios[s][t] + reserve_scenarios[s][t]
            )

    # Objective function
    # Fixed operational costs: \sum_g \sum_t c_f * x[g,t]
    # Startup costs: \sum_g \sum_t q[g,t]
    # Production costs: \frac{1}{n_scenarios} * \sum_s \sum_g \sum_t c_g * p[g,s,t]
    
    fixed_cost_expr = gp.quicksum(
        generators[g]["c_f"] * x[g,t] 
        for g in range(n_generators) 
        for t in range(T)
    )
    
    startup_cost_expr = gp.quicksum(
        q[g,t]
        for g in range(n_generators)
        for t in range(T)
    )
    
    avg_prod_cost_expr = (1.0 / float(n_scenarios)) * gp.quicksum(
        generators[g]["c_g"] * p[g,s,t]
        for g in range(n_generators)
        for s in range(n_scenarios)
        for t in range(T)
    )
    
    objective = fixed_cost_expr + startup_cost_expr + avg_prod_cost_expr
    
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()

    # Build solution dictionary
    x_values = [[int(x[g,t].X) for t in range(T)] for g in range(n_generators)]
    y_values = [[int(y[g,t].X) for t in range(T)] for g in range(n_generators)]
    y_bar_values = [[int(y_bar[g,t].X) for t in range(T)] for g in range(n_generators)]
    q_values = [[float(q[g,t].X) for t in range(T)] for g in range(n_generators)]
    
    p_values = [[[float(p[g,s,t].X) for t in range(T)] for s in range(n_scenarios)] for g in range(n_generators)]
    p_bar_values = [[[float(p_bar[g,s,t].X) for t in range(T)] for s in range(n_scenarios)] for g in range(n_generators)]

    solution = {
        "objective_value": model.objVal if model.solCount > 0 else float('inf'),
        "x": x_values,
        "y": y_values,
        "y_bar": y_bar_values,
        "q": q_values,
        "p": p_values,
        "p_bar": p_bar_values
    }

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize logger if log path provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve
    solution = solve_power_system(instance_data, args.time_limit)
    
    # Check if we got a feasible solution and inform logger
    if solution["objective_value"] != float('inf') and logger:
        logger.log(solution["objective_value"])

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()