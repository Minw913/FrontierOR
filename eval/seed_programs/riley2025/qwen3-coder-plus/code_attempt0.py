import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def solve_supply_chain(instance_path, solution_path, time_limit, log_path=None):
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Extract parameters
    num_demand_sites = instance["num_demand_sites"]
    num_production_sites = instance["num_production_sites"]
    num_scenarios = instance["num_scenarios"]
    num_time_periods = instance["num_time_periods"]
    num_module_sizes = instance["num_module_sizes"]
    units_per_module_size = instance["units_per_module_size"]
    u_max = instance["u_max"]
    scenario_probabilities = instance["scenario_probabilities"]

    # Extract coordinates and costs
    production_costs = np.array(instance["production_costs"])  # [i][j]
    relocation_costs = np.array(instance["relocation_costs"])  # [j][jp][m]
    penalty_costs = np.array(instance["penalty_costs"])  # [i][t]

    # Extract module information
    module_capacities = [ms["capacity"] for ms in instance["module_sizes"]]
    module_capital_costs = [ms["capital_cost_per_unit"] for ms in instance["module_sizes"]]

    # Extract scenario demands
    scenario_demands = []
    for scenario in instance["scenarios"]:
        scenario_demands.append(np.array(scenario["demands"]))

    # Create model
    model = gp.Model("SupplyChain")
    
    # Set Gurobi parameters
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    # y[m,k]: whether module unit k of size m is purchased
    y = {}
    for m in range(num_module_sizes):
        for k in range(units_per_module_size):
            y[m, k] = model.addVar(vtype=GRB.BINARY, name=f"y[{m},{k}]")

    # x[i,j,t,s]: fraction of demand at site i served from production site j in period t under scenario s
    x = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for i in range(num_demand_sites):
                for j in range(num_production_sites):
                    x[i, j, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x[{i},{j},{t},{s}]")

    # q[i,t,s]: fraction of demand at site i that remains unsatisfied in period t under scenario s
    q = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for i in range(num_demand_sites):
                q[i, t, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"q[{i},{t},{s}]")

    # v[j,m,t,s]: number of modules of size m at production site j in period t under scenario s
    v = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for j in range(num_production_sites):
                for m in range(num_module_sizes):
                    v[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"v[{j},{m},{t},{s}]")

    # v_depot[m,t,s]: number of modules of size m at depot in period t under scenario s
    v_depot = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for m in range(num_module_sizes):
                v_depot[m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"v_depot[{m},{t},{s}]")

    # w[j,jp,m,t,s]: number of modules of size m relocated from site j to site jp in period t under scenario s
    w = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for j in range(num_production_sites):
                for jp in range(num_production_sites):
                    if j != jp:  # Only consider relocations between different sites
                        for m in range(num_module_sizes):
                            w[j, jp, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"w[{j},{jp},{m},{t},{s}]")

    # w_fd[j,m,t,s]: number of modules of size m moved from depot to production site j in period t under scenario s
    w_fd = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for j in range(num_production_sites):
                for m in range(num_module_sizes):
                    w_fd[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"w_fd[{j},{m},{t},{s}]")

    # w_td[j,m,t,s]: number of modules of size m moved from production site j to depot in period t under scenario s
    w_td = {}
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for j in range(num_production_sites):
                for m in range(num_module_sizes):
                    w_td[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"w_td[{j},{m},{t},{s}]")

    model.update()

    # Constraints
    
    # Link y variables to total number of modules at depot in period 0
    for s in range(num_scenarios):
        for m in range(num_module_sizes):
            # Initial modules at depot = total purchased modules of size m
            model.addConstr(
                v_depot[m, 0, s] == gp.quicksum(y[m, k] for k in range(units_per_module_size)),
                name=f"initial_depot_inventory_{m}_{s}"
            )

    # Flow balance at depot for each module size, period, and scenario
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for m in range(num_module_sizes):
                if t == 0:
                    # Already handled by initial inventory constraint
                    continue
                else:
                    # Inventory at depot at time t = previous inventory + received from sites - sent to sites
                    model.addConstr(
                        v_depot[m, t, s] == 
                        v_depot[m, t-1, s] + 
                        gp.quicksum(w_td[j, m, t-1, s] for j in range(num_production_sites)) -
                        gp.quicksum(w_fd[j, m, t-1, s] for j in range(num_production_sites)),
                        name=f"depot_flow_balance_{m}_{t}_{s}"
                    )

    # Flow balance at production sites for each module size, period, and scenario
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for j in range(num_production_sites):
                for m in range(num_module_sizes):
                    if t == 0:
                        # Initial inventory at site j = modules moved from depot in period 0
                        model.addConstr(
                            v[j, m, t, s] == w_fd[j, m, t, s],
                            name=f"site_initial_inventory_{j}_{m}_{t}_{s}"
                        )
                    else:
                        # Inventory at site j at time t = previous inventory + received from other sites/depot - sent to other sites/depot
                        model.addConstr(
                            v[j, m, t, s] == 
                            v[j, m, t-1, s] +
                            w_td[j, m, t-1, s] +  # received from depot
                            gp.quicksum(w[jj, j, m, t-1, s] for jj in range(num_production_sites) if jj != j) -  # received from other sites
                            w_fd[j, m, t-1, s] -  # sent to site from depot
                            gp.quicksum(w[j, jj, m, t-1, s] for jj in range(num_production_sites) if jj != j),  # sent to other sites
                            name=f"site_flow_balance_{j}_{m}_{t}_{s}"
                        )

    # Demand satisfaction constraints
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for i in range(num_demand_sites):
                model.addConstr(
                    gp.quicksum(x[i, j, t, s] for j in range(num_production_sites)) + q[i, t, s] == 1.0,
                    name=f"demand_satisfaction_{i}_{t}_{s}"
                )

    # Capacity constraints at production sites
    for s in range(num_scenarios):
        for t in range(num_time_periods):
            for j in range(num_production_sites):
                # Total demand served from site j cannot exceed capacity of modules at site j
                model.addConstr(
                    gp.quicksum(
                        scenario_demands[s][i, t] * x[i, j, t, s] 
                        for i in range(num_demand_sites)
                    ) <= 
                    gp.quicksum(v[j, m, t, s] * module_capacities[m] for m in range(num_module_sizes)),
                    name=f"capacity_constraint_site_{j}_period_{t}_scenario_{s}"
                )
                
                # Total demand served from site j cannot exceed global max production
                model.addConstr(
                    gp.quicksum(
                        scenario_demands[s][i, t] * x[i, j, t, s] 
                        for i in range(num_demand_sites)
                    ) <= u_max,
                    name=f"global_capacity_constraint_site_{j}_period_{t}_scenario_{s}"
                )

    # Objective function
    # Capital cost: sum over all module sizes and units of capital cost * y[m,k]
    capital_cost = gp.quicksum(
        module_capacities[m] * y[m, k] 
        for m in range(num_module_sizes) 
        for k in range(units_per_module_size)
    )

    # Expected operational cost
    expected_operational_cost = 0

    for s in range(num_scenarios):
        scenario_operational_cost = 0
        
        # Production and transportation cost
        prod_trans_cost = gp.quicksum(
            production_costs[i, j] * scenario_demands[s][i, t] * x[i, j, t, s]
            for i in range(num_demand_sites)
            for j in range(num_production_sites)
            for t in range(num_time_periods)
        )
        
        # Relocation cost
        relocation_cost = gp.quicksum(
            relocation_costs[j, jp, m] * w[j, jp, m, t, s]
            for j in range(num_production_sites)
            for jp in range(num_production_sites)
            if j != jp
            for m in range(num_module_sizes)
            for t in range(num_time_periods)
        )
        
        # Additional relocation costs: depot to site and site to depot
        depot_relocation_cost = gp.quicksum(
            relocation_costs[num_production_sites, j, m] * w_fd[j, m, t, s]  # Assuming depot is indexed as num_production_sites
            for j in range(num_production_sites)
            for m in range(num_module_sizes)
            for t in range(num_time_periods)
        ) + gp.quicksum(
            relocation_costs[j, num_production_sites, m] * w_td[j, m, t, s]  # Depot as destination
            for j in range(num_production_sites)
            for m in range(num_module_sizes)
            for t in range(num_time_periods)
        )
        
        # Unmet demand penalty
        unmet_penalty = gp.quicksum(
            penalty_costs[i, t] * scenario_demands[s][i, t] * q[i, t, s]
            for i in range(num_demand_sites)
            for t in range(num_time_periods)
        )
        
        scenario_operational_cost = prod_trans_cost + relocation_cost + depot_relocation_cost + unmet_penalty
        expected_operational_cost += scenario_probabilities[s] * scenario_operational_cost

    total_cost = capital_cost + expected_operational_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Solve the model
    start_time = time.time()
    
    # Setup callback for logging
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if log_path:
                # Build partial solution dict with current values
                sol_dict = {
                    "objective_value": obj,
                    "model_variables": {},
                    "y": {}
                }
                
                # Add y variables to solution dict
                for m in range(num_module_sizes):
                    for k in range(units_per_module_size):
                        val = model.cbGetSolution(y[m, k])
                        if abs(val) > 1e-6:  # Only add non-zero values
                            sol_dict["y"][f"y_{m}_{k}"] = round(val)
                
                if logger:
                    logger.log_solution(obj, sol_dict)

    # Initialize logger if path provided
    logger = None
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
    
    model.optimize(mycallback)

    # Prepare solution output
    solution = {
        "objective_value": model.objVal if model.solCount > 0 else float('inf'),
        "model_variables": {},
        "y": {},
        "model_variable_name_patterns": {
            "y[m,k]": "m=0..num_module_sizes-1, k=0..units_per_module_size-1",
            "z[m,s]": "module size m, scenario s",
            "x[i,j,t,s]": "demand site i, production site j, period t, scenario s",
            "q[i,t,s]": "unmet-demand fraction",
            "v[j,m,t,s]": "units at production site",
            "v_depot[m,t,s]": "units at depot",
            "w[j,jp,m,t,s]": "relocation between distinct production sites j and jp",
            "w_fd[j,m,t,s]": "depot-to-production-site relocation",
            "w_td[j,m,t,s]": "production-site-to-depot relocation"
        }
    }

    # Populate y variables
    for m in range(num_module_sizes):
        for k in range(units_per_module_size):
            val = y[m, k].x
            if abs(val) > 1e-8:
                solution["y"][f"y_{m}_{k}"] = round(val)

    # Store all model variables in sparse format
    vars_to_store = [
        ("x", x),
        ("q", q),
        ("v", v),
        ("v_depot", v_depot),
        ("w", w),
        ("w_fd", w_fd),
        ("w_td", w_td)
    ]

    for var_name, var_dict in vars_to_store:
        for idx, var in var_dict.items():
            val = var.x
            if abs(val) > 1e-8:
                # Convert tuple indices to string format matching schema
                idx_str = "_".join(map(str, idx))
                solution["model_variables"][f"{var_name}[{idx_str.replace('_', ',')}]"] = round(val) if var.VType == 'I' else val

    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve supply chain optimization problem')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to the log JSONL file (optional)')
    
    args = parser.parse_args()
    
    solve_supply_chain(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()