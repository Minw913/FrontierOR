import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger


def solve_hub_location(instance_data):
    """
    Solves the stochastic hub location problem using Gurobi.
    
    :param instance_data: Dictionary containing the problem instance data
    :return: Solution dictionary
    """
    
    # Unpack the instance data
    n = instance_data["n"]
    num_scenarios = instance_data["num_scenarios"]
    chi = instance_data["cost_parameters"]["chi"]
    alpha = instance_data["cost_parameters"]["alpha"]
    delta = instance_data["cost_parameters"]["delta"]
    p_hubs = instance_data["p_hubs"]  # None means uncapacitated p-hub median
    distances = instance_data["distances"]
    fixed_costs = instance_data["fixed_costs"]
    hub_capacities = instance_data["hub_capacities"]  # Can be None for uncapacitated
    scenarios = instance_data["scenarios"]
    
    # Create model
    model = gp.Model("StochasticHubLocation")

    # Add a time limit if available through parameters passed later
    model.setParam(GRB.Param.TimeLimit, instance_data.get("time_limit", 3600))
    # Set random seed if needed to ensure reproducibility
    # model.setParam(GRB.Param.Seed, 123)

    # Define the set of scenarios
    S = range(num_scenarios)
    P = range(n)

    # Decision Variables:
    # z_i: Binary, 1 if hub i is located, 0 otherwise
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # x[origin][hub][s]: Binary, 1 if origin node 'origin' is assigned to hub 'hub'
    #                    in scenario s. Note: origin != hub by our construction below
    x = {}
    for s in S:
        x[s] = {}
        for i in P:
            x[s][i] = model.addVars([j for j in P if i != j], vtype=GRB.BINARY, name=f"x_{i}_{s}")
            
    # y[i][j][k][l][s]: Continuous, flow from origin i to destination j via hubs k and l 
    #                   in scenario s. y variables represent the flow from (i,j) routed through
    #                   the hub k for origin and l for destination.
    y = {}
    for s in S:
        prob = scenarios[s]["probability"]
        D = scenarios[s]["demands"]
        y[s] = {}
        for i in P:
            y[s][i] = {}
            for j in P:
                y[s][i][j] = {}
                for k in P:
                    y[s][i][j][k] = {}
                    for l in P:
                        y[s][i][j][k][l] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                        obj=prob * alpha * distances[k][l],
                                                        name=f"y_{i}_{j}_{k}_{l}_{s}")

    # Objective function components will be added as lazy constraints or
    # via auxiliary variables if necessary. For now, we just defined y vars with correct objective coefficients
    # Let's restructure the way we handle the objective:
    # Minimize: Sum of (fixed costs * z) + sum_over_scenarios(p_s * (
    #              sum_over_nodes(sum_over_assigned_hub(dist_to_hub * (
    #                   chi * outflow + delta * inflow))) +
    #              sum_over_od_pairs(dist_hub_to_hub * alpha * flow)
    #           ))
    fixed_cost_expr = gp.quicksum(fixed_costs[i] * z[i] for i in P)
    
    # We need to add terms related to collection/distribution
    col_dist_cost = gp.QuadExpr(0)
    for s in S:
        scenario_prob = scenarios[s]["probability"]
        scenario_demands = scenarios[s]["demands"]
        
        # Calculate total flow from and to each node in this scenario
        out_flow = [sum(scenario_demands[i][j] for j in P) for i in P]
        in_flow = [sum(scenario_demands[i][j] for i in P) for j in P]
        
        for i in P:
            total_out_flow = out_flow[i]
            total_in_flow = in_flow[i]
            
            # The cost for node i to send/receive flow through its allocated hub
            # depends on the allocation x[i][k][s] in scenario s
            for k in P:
                if i != k:
                    term_cost = scenario_prob * distances[i][k] * (chi * total_out_flow + delta * total_in_flow)
                    col_dist_cost += x[s][i][k] * term_cost
                else:  # i == k -> If node i itself is used as a hub, there's no cost from that perspective.
                    term_cost = scenario_prob * 0 * (chi * total_out_flow + delta * total_in_flow)
                    # This case is implicitly 0, handled by no assignment
        
    # Since we already included inter-hub transfer costs in the definition of y variables,
    # we construct a total obj expression
    obj_expr = fixed_cost_expr + col_dist_cost
    # The y variables' obj coefficients are already properly set
    
    model.setObjective(obj_expr, GRB.MINIMIZE)


    # Constraints:
    
    # 1. At least one hub must be selected
    model.addConstr(gp.quicksum(z) >= 1, "at_least_one_hub")
    
    # If p_hubs is specified, enforce exactly p_hubs to be open
    if p_hubs is not None:
        model.addConstr(gp.quicksum(z) == p_hubs, f"exactly_{p_hubs}_hubs")

    # 2. Each non-hub node is assigned to exactly one hub in each scenario
    for s in S:
        for i in P:
            # Node i can be assigned to a hub that is either:
            # - itself if hub i is opened: then x_s[i][i] could be 1, or 
            # - another hub j if z[j]=1
            # We must enforce assignment constraint:
            # If z[i] = 1 (i is a hub), then x[s][i][k]==0 for all k!=i
            # If z[i] = 0 (i is not a hub), then sum{x[s][i][k]*z[k]} = 1
            
            # Simplified: sum of x[s][i][k] for all k!=i should equal 1-z[i] (if i not hub) or 0 (if i is hub)
            model.addConstr(gp.quicksum(x[s][i][k] for k in P if i != k) == 1 - z[i], f"assignment_single_hub_{i}_{s}")


    # 3. A node can only be assigned to an open hub
    for s in S:
        for i in P:
             for k in P:
                 if i != k:
                    # x_s[i][k] <= z[k]
                    model.addConstr(x[s][i][k] <= z[k], f"assign_to_open_hub_{i}_{k}_{s}")

    # 4. Flow conservation: ensure y variables reflect assignments correctly
    # For each origin i and scenario s, the flow y[i][*][assigned_hub_for_i][*][s]
    # should match the flow from i to all destinations j
    # And y[*][j][*][assigned_hub_for_j][s] should match flow from all origins i to j
    for s in S:
        D = scenarios[s]["demands"]
        for i in P:
            for j in P:
                if D[i][j] > 0:
                    # Sum of y[i][j][k][l][s] over all possible hub combinations should equal D[i][j]
                    model.addConstr(
                        gp.quicksum(
                            y[s][i][j][k][l] for k in P for l in P
                        ) == D[i][j],
                        f"demand_satisfaction_{i}_{j}_{s}"
                    )
                    
                    # Now link the allocation variables to y vars
                    for k in P:
                        # If node i is assigned to hub k, then all y[i][j][ki][*][s] have ki=k
                        # i.e. y[i][j][k2][l][s] = 0 for all k2!=k, so we use sum trick:
                        # For all k:
                        model.addConstr(
                              gp.quicksum(y[s][i][j][k2][l] for l in P for k2 in P if k2 != k) 
                              <= (1 - (x[s][i][k] if i != k else z[i])) * D[i][j],  # if assigning to k, then k2!=k part = 0
                          f"flow_assignment_from_{i}_to_hub_{k}_dest_{j}_{s}")
                          
                    for l in P:
                         model.addConstr(
                              gp.quicksum(y[s][i][j][k][l2] for k in P for l2 in P if l2 != l) 
                              <= (1 - (x[s][j][l] if j != l else z[j])) * D[i][j], 
                          f"flow_assignment_to_dest_{j}_thru_hub_{l}_from_{i}_{s}")


    # 5. Capacity constraints (if capacitated)
    if hub_capacities is not None:
        for s in S:
            out_flows = [sum(scenarios[s]["demands"][i][j] for j in P) for i in P]
            for k in P:
                total_hub_out_flows = gp.LinExpr()
                
                # Add contributions from non-hub nodes assigned to hub k
                for i in P:
                    if i != k:
                        # If x[s][i][k] = 1, add out_flows[i]
                        total_hub_out_flows.addTerms(out_flows[i], x[s][i][k])
                    else:
                        # If z[i] = 1, hub is active and its flows count
                        # Check if the hub i=k has outgoing flow
                        total_hub_out_flows.addConstant(out_flows[i] * z[i])  # Note: Not completely accurate since z[i] binary but flows exist even if not hub unless demand is 0
                
                model.addConstr(total_hub_out_flows <= hub_capacities[k], 
                                f"capacity_constraint_hub_{k}_scenario_{s}")

    # Optimize model
    model.optimize()

    # Prepare solution
    hubs_list = []
    for i in range(n):
        if z[i].X > 0.99:
            hubs_list.append(i)

    z_vals = [int(z[i].X > 0.99) for i in range(n)]

    allocations = {f"{s}": {} for s in S}
    x_final = {f"{s}": {} for s in S}
    
    for s in S:
        for i in P:
            # Determine to which hub node i is assigned in scenario s
            if z[i].X > 0.99:
                allocations[f"{s}"][f"{i}"] = i  # Hub assigned to itself
                x_final[f"{s}"][f"{i}"] = {}  # No actual assignment since it's a hub
            else:
                for j in P:
                    if i != j:
                        if x[s][i][j].X > 0.99:
                            allocations[f"{s}"][f"{i}"] = j
                            
                            # Record in x_values as per schema: x_vals[scenario][origin_node][candidate_hub]
                            x_final[f"{s}"] = x_final.get(f"{s}", {})
                            inner_dict = x_final[f"{s}"].get(f"{i}", {})
                            inner_dict[f"{j}"] = int(x[s][i][j].X > 0.99)
                            x_final[f"{s}"][f"{i}"] = inner_dict
                            break  # There should be exactly one assignment


    total_obj_val = model.ObjVal

    solution = {
        "objective_value": total_obj_val,
        "hubs": hubs_list,
        "z_values": z_vals,
        "allocations": allocations,
        "x_values": x_final
    }

    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Stochastic Hub Location Problem')
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to write solution JSON file")
    parser.add_argument("--time_limit", required=True, type=int, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", required=False, help="Path to write log file (JSONL format)")

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Add time limit to instance data for convenience in function
    instance_data["time_limit"] = args.time_limit
    
    # Initialize logger if specified
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Solve the problem
    try:
        solution = solve_hub_location(instance_data)
        
        # Log the final best objective if logging is enabled
        if logger:
            logger.log(solution['objective_value'])
        
    except Exception as e:
        print(f"Error solving instance: {e}")
        raise e

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print("Solution saved to:", args.solution_path)


if __name__ == "__main__":
    main()