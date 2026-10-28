import argparse
import json
import numpy as np
from itertools import product
import gurobipy as gp
from gurobipy import GRB
def solve(retailer_data):
    """
    Solve the two-stage stochastic inventory optimization problem with transshipments.

    :param retailer_data: Instance data loaded from the instance JSON
    :return: dict representing the solution, with the schema as in solution_schema.json
    """
    # Extract problem parameters
    N = retailer_data["N"]
    variable_lower_bounds = np.array(retailer_data["first_stage"]["variable_lower_bounds"])

    holding_costs = np.array(retailer_data["second_stage"]["holding_costs"])
    penalty_costs = np.array(retailer_data["second_stage"]["penalty_costs"])
    transp_costs_mean = np.array(retailer_data["second_stage"]["transshipment_costs_mean"])
    transp_costs_std = np.array(retailer_data["second_stage"]["transshipment_costs_std"])
    random_cost_links = retailer_data["second_stage"].get("random_cost_links", [])
    
    mean_demands = np.array(retailer_data["second_stage"]["mean_demands"])
    num_scenarios = retailer_data["second_stage"]["demand_distribution"]["num_scenarios"]
    scenarios = np.array(retailer_data["second_stage"]["demand_distribution"]["scenarios"])
    probabilities = np.array(retailer_data["second_stage"]["demand_distribution"]["probabilities"])

    # Generate random transshipment costs for each scenario based on given random_seed
    # Though seed isn't in instance schema, we assume fixed generation here
    rng = np.random.RandomState(42)  # Default seed to reproduce results
    transp_per_scenario = []
    for scenario_idx in range(num_scenarios):
        scenario_matrix = transp_costs_mean.copy()
        for i, j in random_cost_links:
            # Get mean cost
            mu = transp_costs_mean[i, j]
            sigma = transp_costs_std[i, j]
            
            # Draw value; use provided fraction method if std is not specified per link directly
            if abs(sigma) < 1e-9:  # If not set explicitly, infer from mean and fraction
                fraction = retailer_data["second_stage"].get("random_cost_std_fraction", 0.0)
                sigma = fraction * mu
            
            cost = rng.normal(mu, sigma)
            scenario_matrix[i, j] = max(0.0, cost)  # Ensure non-negative cost
        transp_per_scenario.append(scenario_matrix)
    
    # Model creation
    model = gp.Model("stochastic_inventory")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)  # Suppress output if needed

    # First stage: Determine the order-up-to levels s[j]
    s = model.addVars(N, lb=variable_lower_bounds, name='s')

    # Second stage: Define variables for each scenario
    e = {}  # Ending inventories
    f = {}  # Stock used locally (fulfilled on-site)
    q = {}  # Replenishment amounts (inventory needed to fill order-up-to after demand)
    r = {}  # Local backlogs / shortages
    t = {}  # Transshipment quantities

    obj_expr = 0.0
    
    # For each scenario
    transp_costs_scenario = []
    for s_idx in range(num_scenarios):
        prob_s = probabilities[s_idx]
        d_s = scenarios[s_idx]  # Demand vector for this scenario
        c_s = transp_per_scenario[s_idx]  # Cost matrix for this scenario

        # Vars for scenario s_idx
        e_i = model.addVars(N, lb=0, name=f'e_{s_idx}')
        f_i = model.addVars(N, lb=0, name=f'f_{s_idx}')
        q_i = model.addVars(N, lb=0, name=f'q_{s_idx}')
        r_i = model.addVars(N, lb=0, name=f'r_{s_idx}')
        t_ij = model.addVars([(i, j) for i in range(N) for j in range(N) if i != j],
                             lb=0, name=f't_{s_idx}')

        # Update mappings
        e[s_idx] = e_i
        f[s_idx] = f_i
        q[s_idx] = q_i
        r[s_idx] = r_i
        t[s_idx] = t_ij

        # Constraints
        
        # Balance 1: Stock allocation at each retailer j:
        # s[j] = sum_{i\neq j} t_{j,i} + f_j + e_j \forall j
        for j in range(N):
            outbound_sum = gp.quicksum(t_ij[(j, i)] if (j, i) in t_ij.keys() else 0 for i in range(N))
            inbound_sum_from_f_and_e = f_i[j] + e_i[j]
            model.addConstr(s[j] == outbound_sum + inbound_sum_from_f_and_e,
                            name=f'balance1_scenario{s_idx}_retailer{j}')
        
        # Balance 2: Demand fulfillment at each retailer j:
        # d[j] = sum_{i\neq j} t_{i,j} + f_j + r_j \forall j
        for j in range(N):
            inbound_sum = gp.quicksum(t_ij[(i, j)] if (i, j) in t_ij.keys() else 0 for i in range(N))
            direct_and_backlog = f_i[j] + r_i[j]
            model.addConstr(d_s[j] == inbound_sum + direct_and_backlog,
                            name=f'balance2_scenario{s_idx}_retailer{j}')
        
        # Balance 3: Replenish to order-up-to level at each retailer j:
        # s[j] = e_j + q_j \forall j
        for j in range(N):
            model.addConstr(s[j] == e_i[j] + q_i[j],
                            name=f'balance3_scenario{s_idx}_retailer{j}')
        
        # Objective contribution for scenario s_idx
        # sum_j (h[j]*e[j] + p[j]*r[j]) + sum_{i,j} c_{ij}*t_{ij}
        scen_holding_cost = gp.quicksum(holding_costs[j]*e_i[j] for j in range(N))
        scen_penalty_cost = gp.quicksum(penalty_costs[j]*r_i[j] for j in range(N))

        scen_transp_cost = gp.LinExpr()
        for (i, j), t_var in t_ij.items():
            scen_transp_cost += c_s[i, j] * t_var

        total_scenario_cost = scen_holding_cost + scen_penalty_cost + scen_transp_cost
        obj_expr += prob_s * total_scenario_cost   # Sum of prob * scenario cost

    # Set objective
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Model solved now
    model.optimize()

    # Prepare output
    s_vals = [s[j].X for j in range(N)]

    scenario_results = []
    for s_idx in range(num_scenarios):
        e_row = [e[s_idx][j].X for j in range(N)]
        f_row = [[0]*N for _ in range(N)]  # Create full N x N matrix, with 0 diagonal
        for i, j in t[s_idx].keys():
            f_row[i][j] = t[s_idx][(i, j)].X

        scenario_results.append({
            "e": [round(e_val, 6) for e_val in e_row],
            "f": [round(f[s_idx][j].X, 6) for j in range(N)],
            "q": [round(q[s_idx][j].X, 6) for j in range(N)],
            "r": [round(r[s_idx][j].X, 6) for j in range(N)],
            "t": [[round(f_row[i][j], 6) for j in range(N)] for i in range(N)]
        })

    result = {
        "objective_value": round(model.ObjVal, 6),
        "s": [round(s_val, 6) for s_val in s_vals],
        "scenarios": scenario_results
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Solve stochastic inventory optimization with transshipments.")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, default=500,
                        help="Time limit for solving in seconds (default: 86400)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to the log file for intermediate results")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as infile:
        instance = json.load(infile)

    # Initialize the logger (using "minimize" because we are minimizing cost)
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Set time limit for the model
    try:
        solution = solve(instance)

        # Log the final objective if we have a logger
        if logger:
            logger.log(solution['objective_value'])

        with open(args.solution_path, 'w') as outfile:
            json.dump(solution, outfile)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()