import argparse
import json
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    N = instance['N']
    first_stage = instance['first_stage']
    second_stage = instance['second_stage']
    
    lb = first_stage['variable_lower_bounds']
    h = second_stage['holding_costs']
    p = second_stage['penalty_costs']
    tc_mean = second_stage['transshipment_costs_mean']
    tc_std = second_stage['transshipment_costs_std']
    random_cost_links = second_stage['random_cost_links']
    random_cost_std_fraction = second_stage['random_cost_std_fraction']
    mean_demands = second_stage['mean_demands']
    
    demand_dist = second_stage['demand_distribution']
    num_demand_scenarios = demand_dist['num_scenarios']
    demand_scenarios = demand_dist['scenarios']
    demand_probs = demand_dist['probabilities']
    
    # Generate cost scenarios using the random seed approach
    # We need to handle random transshipment costs
    # The problem says costs are drawn from Normal(mean, std) where std = fraction * mean
    # and a random seed is used to generate cost realizations
    # Since no explicit random seed is given in the schema fields we can see,
    # let's check if there's a seed in the instance
    random_seed = instance.get('random_seed', None)
    if random_seed is None:
        random_seed = second_stage.get('random_seed', None)
    if random_seed is None:
        random_seed = instance.get('seed', None)
    
    # Number of cost scenarios - for random cost links
    # If there are no random cost links, we just use deterministic costs
    num_random_links = len(random_cost_links)
    
    if num_random_links > 0 and random_seed is not None:
        # Generate cost scenarios - use same number as demand scenarios for simplicity
        # Actually, cost and demand are independent, so we can use a moderate number of cost scenarios
        # and combine with demand scenarios
        rng = np.random.RandomState(random_seed)
        num_cost_scenarios = num_demand_scenarios  # Match demand scenarios
        
        # Generate cost realizations for random links
        cost_realizations = {}
        for link in random_cost_links:
            i, j = link[0], link[1]
            mean = tc_mean[i][j]
            std = tc_std[i][j]
            if std == 0:
                std = random_cost_std_fraction * mean
            samples = rng.normal(mean, std, num_cost_scenarios)
            # Costs should be non-negative
            samples = np.maximum(samples, 0.0)
            cost_realizations[(i, j)] = samples
        
        # Total scenarios = demand_scenarios x cost_scenarios
        # But this could be huge, so we'll sample combined scenarios
        # Actually, let's keep them separate and combine
        # For tractability, use a moderate approach
        
        # If too many combined scenarios, sample
        total_combined = num_demand_scenarios * num_cost_scenarios
        
        if total_combined <= 500:
            # Use all combinations
            scenarios = []
            for ds in range(num_demand_scenarios):
                for cs in range(num_cost_scenarios):
                    prob = demand_probs[ds] / num_cost_scenarios
                    demands = demand_scenarios[ds]
                    costs = [[tc_mean[i][j] for j in range(N)] for i in range(N)]
                    for (i, j), samples in cost_realizations.items():
                        costs[i][j] = samples[cs]
                    scenarios.append((demands, costs, prob))
        else:
            # Sample to keep manageable
            max_scenarios = 500
            scenarios = []
            # Use all demand scenarios, sample cost scenarios
            num_cost_samples = max(1, max_scenarios // num_demand_scenarios)
            for ds in range(num_demand_scenarios):
                for cs in range(num_cost_samples):
                    prob = demand_probs[ds] / num_cost_samples
                    demands = demand_scenarios[ds]
                    costs = [[tc_mean[i][j] for j in range(N)] for i in range(N)]
                    for (i, j), samples in cost_realizations.items():
                        idx = cs % num_cost_scenarios
                        costs[i][j] = samples[idx]
                    scenarios.append((demands, costs, prob))
    else:
        # No random cost links or no seed - deterministic costs
        scenarios = []
        for ds in range(num_demand_scenarios):
            prob = demand_probs[ds]
            demands = demand_scenarios[ds]
            costs = tc_mean
            scenarios.append((demands, costs, prob))
    
    num_scenarios = len(scenarios)
    
    elapsed = time.time() - start_time
    remaining_time = max(time_limit - elapsed - 5, 10)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build Gurobi model
    model = gp.Model("inventory_transshipment")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', remaining_time)
    model.setParam('Threads', 1)
    
    # First stage variables: order-up-to levels
    s = model.addVars(N, lb=[lb[i] for i in range(N)], name="s")
    
    # Second stage variables per scenario
    e = {}  # ending inventory
    f = {}  # local fulfillment
    q = {}  # replenishment
    r = {}  # backlog
    t = {}  # transshipment
    
    for sc in range(num_scenarios):
        for i in range(N):
            e[sc, i] = model.addVar(lb=0.0, name=f"e_{sc}_{i}")
            f[sc, i] = model.addVar(lb=0.0, name=f"f_{sc}_{i}")
            q[sc, i] = model.addVar(lb=0.0, name=f"q_{sc}_{i}")
            r[sc, i] = model.addVar(lb=0.0, name=f"r_{sc}_{i}")
        for i in range(N):
            for j in range(N):
                if i != j:
                    t[sc, i, j] = model.addVar(lb=0.0, name=f"t_{sc}_{i}_{j}")
    
    model.update()
    
    # Constraints per scenario
    for sc in range(num_scenarios):
        demands_sc, costs_sc, prob_sc = scenarios[sc]
        
        for i in range(N):
            # Constraint 1: f[i] + sum_j t[i,j] + e[i] = s[i]
            outbound = gp.quicksum(t[sc, i, j] for j in range(N) if j != i)
            model.addConstr(f[sc, i] + outbound + e[sc, i] == s[i],
                          name=f"balance1_{sc}_{i}")
            
            # Constraint 2: f[i] + sum_j t[j,i] + r[i] = d[i]
            inbound = gp.quicksum(t[sc, j, i] for j in range(N) if j != i)
            model.addConstr(f[sc, i] + inbound + r[sc, i] == demands_sc[i],
                          name=f"balance2_{sc}_{i}")
            
            # Constraint 3: e[i] + q[i] = s[i]
            model.addConstr(e[sc, i] + q[sc, i] == s[i],
                          name=f"balance3_{sc}_{i}")
        
        # Constraint 4: sum_i (r[i] + q[i]) = sum_i d[i]
        model.addConstr(
            gp.quicksum(r[sc, i] + q[sc, i] for i in range(N)) == sum(demands_sc),
            name=f"balance4_{sc}")
    
    # Objective: minimize expected cost
    obj = gp.LinExpr()
    for sc in range(num_scenarios):
        demands_sc, costs_sc, prob_sc = scenarios[sc]
        
        for i in range(N):
            obj += prob_sc * h[i] * e[sc, i]
            obj += prob_sc * p[i] * r[sc, i]
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    if isinstance(costs_sc, list):
                        c_ij = costs_sc[i][j]
                    else:
                        c_ij = costs_sc[i][j]
                    obj += prob_sc * c_ij * t[sc, i, j]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Use callback to log solutions
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    # This is an LP, so solve directly
    model.optimize()
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL or model.SolCount > 0:
        obj_val = model.ObjVal
        if logger:
            logger.log(obj_val)
        
        # Extract solution
        s_val = [s[i].X for i in range(N)]
        
        scenario_solutions = []
        for sc in range(num_scenarios):
            e_val = [e[sc, i].X for i in range(N)]
            f_val = [f[sc, i].X for i in range(N)]
            q_val = [q[sc, i].X for i in range(N)]
            r_val = [r[sc, i].X for i in range(N)]
            t_val = [[0.0] * N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        t_val[i][j] = t[sc, i, j].X
            
            scenario_solutions.append({
                'e': e_val,
                'f': f_val,
                'q': q_val,
                'r': r_val,
                't': t_val
            })
        
        # If we used combined scenarios but output expects per-demand-scenario,
        # we need to map back. The output schema says length = num_scenarios from demand_distribution.
        # Let's check: if we have combined scenarios, we need to aggregate or restructure.
        
        # Actually, re-reading the problem: the output expects one entry per demand scenario.
        # But we solved with combined (demand x cost) scenarios.
        # For the output, we need to provide one operational plan per scenario.
        # The question is what "scenario" means in the output.
        
        # Looking at the schema: "scenarios": "<list[dict], length num_scenarios>"
        # where num_scenarios comes from demand_distribution.num_scenarios
        
        # So we need exactly num_demand_scenarios entries.
        # If we had cost uncertainty, we need to re-solve or average.
        
        # Best approach: re-solve per demand scenario with expected costs for output,
        # or restructure. Let's restructure by averaging over cost scenarios.
        
        if num_random_links > 0 and random_seed is not None and num_scenarios > num_demand_scenarios:
            # We have combined scenarios. Need to produce per-demand-scenario output.
            # For each demand scenario, pick the first cost scenario's solution (or average).
            # Actually, let's re-solve with just the demand scenarios using mean costs
            # for the output, using the s_val we found.
            
            output_scenarios = []
            for ds in range(num_demand_scenarios):
                demands_ds = demand_scenarios[ds]
                # Solve a small LP for this demand scenario with mean costs
                m2 = gp.Model("recourse")
                m2.setParam('OutputFlag', 0)
                m2.setParam('Threads', 1)
                
                e2 = m2.addVars(N, lb=0.0)
                f2 = m2.addVars(N, lb=0.0)
                q2 = m2.addVars(N, lb=0.0)
                r2 = m2.addVars(N, lb=0.0)
                t2 = {}
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            t2[i, j] = m2.addVar(lb=0.0)
                
                m2.update()
                
                for i in range(N):
                    outbound = gp.quicksum(t2[i, j] for j in range(N) if j != i)
                    m2.addConstr(f2[i] + outbound + e2[i] == s_val[i])
                    
                    inbound = gp.quicksum(t2[j, i] for j in range(N) if j != i)
                    m2.addConstr(f2[i] + inbound + r2[i] == demands_ds[i])
                    
                    m2.addConstr(e2[i] + q2[i] == s_val[i])
                
                m2.addConstr(gp.quicksum(r2[i] + q2[i] for i in range(N)) == sum(demands_ds))
                
                obj2 = gp.LinExpr()
                for i in range(N):
                    obj2 += h[i] * e2[i]
                    obj2 += p[i] * r2[i]
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            obj2 += tc_mean[i][j] * t2[i, j]
                
                m2.setObjective(obj2, GRB.MINIMIZE)
                m2.optimize()
                
                e_out = [e2[i].X for i in range(N)]
                f_out = [f2[i].X for i in range(N)]
                q_out = [q2[i].X for i in range(N)]
                r_out = [r2[i].X for i in range(N)]
                t_out = [[0.0] * N for _ in range(N)]
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            t_out[i][j] = t2[i, j].X
                
                output_scenarios.append({
                    'e': e_out,
                    'f': f_out,
                    'q': q_out,
                    'r': r_out,
                    't': t_out
                })
                
                m2.dispose()
            
            # Recompute objective with mean costs for output
            total_obj = 0.0
            for ds in range(num_demand_scenarios):
                sc_sol = output_scenarios[ds]
                sc_cost = 0.0
                for i in range(N):
                    sc_cost += h[i] * sc_sol['e'][i]
                    sc_cost += p[i] * sc_sol['r'][i]
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            sc_cost += tc_mean[i][j] * sc_sol['t'][i][j]
                total_obj += demand_probs[ds] * sc_cost
            
            solution = {
                'objective_value': obj_val,  # Use the stochastic objective
                'scenarios': output_scenarios,
                's': s_val
            }
        else:
            solution = {
                'objective_value': obj_val,
                'scenarios': scenario_solutions,
                's': s_val
            }
    else:
        # Fallback: use mean demands as order-up-to levels
        s_val = [max(lb[i], mean_demands[i]) for i in range(N)]
        
        output_scenarios = []
        for ds in range(num_demand_scenarios):
            demands_ds = demand_scenarios[ds]
            e_val = [0.0] * N
            f_val = [0.0] * N
            q_val = [0.0] * N
            r_val = [0.0] * N
            t_val = [[0.0] * N for _ in range(N)]
            
            for i in range(N):
                f_val[i] = min(s_val[i], demands_ds[i])
                e_val[i] = s_val[i] - f_val[i]
                r_val[i] = demands_ds[i] - f_val[i]
                q_val[i] = s_val[i] - e_val[i]
            
            output_scenarios.append({
                'e': e_val,
                'f': f_val,
                'q': q_val,
                'r': r_val,
                't': t_val
            })
        
        total_obj = 0.0
        for ds in range(num_demand_scenarios):
            sc_sol = output_scenarios[ds]
            sc_cost = 0.0
            for i in range(N):
                sc_cost += h[i] * sc_sol['e'][i]
                sc_cost += p[i] * sc_sol['r'][i]
            total_obj += demand_probs[ds] * sc_cost
        
        if logger:
            logger.log(total_obj)
        
        solution = {
            'objective_value': total_obj,
            'scenarios': output_scenarios,
            's': s_val
        }
    
    model.dispose()
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()