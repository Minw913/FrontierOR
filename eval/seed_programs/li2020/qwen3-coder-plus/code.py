import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger
def solve_interdiction_problem(instance_data, time_limit, log_path):
    try:
        # Extract global parameters
        theta = instance_data["global_parameters"]["theta"]
        sigma = instance_data["global_parameters"]["sigma"]
        alpha = instance_data["global_parameters"]["alpha"]
        beta = instance_data["global_parameters"]["beta"]
        gamma = instance_data["global_parameters"]["gamma"]
        alpha_prime = instance_data["global_parameters"]["alpha_prime"]
        beta_prime = instance_data["global_parameters"]["beta_prime"]
        gamma_prime = instance_data["global_parameters"]["gamma_prime"]
        total_interdiction_budget = instance_data["instance_parameters"]["total_interdiction_budget"]
    
        # Extract dimensional information
        num_level1_facilities = instance_data["instance_parameters"]["num_level1_facilities"]
        num_level2_facilities = instance_data["instance_parameters"]["num_level2_facilities"]
        num_customers = instance_data["instance_parameters"]["num_customers"]
        K_levels = instance_data["instance_parameters"]["num_interdiction_levels_K"]
        k_max = instance_data["instance_parameters"]["k_max"]
    
        # Extract demand data
        demands = instance_data["customers"]["demands"]
    
        # Extract level 1 and level 2 facility data
        initial_cap_l1 = instance_data["level1_facilities"]["capacity_type1"]
        initial_cap_l2_t1 = instance_data["level2_facilities"]["capacity_type1"]
        initial_cap_l2_t2 = instance_data["level2_facilities"]["capacity_type2"]
    
        # Extract cost and reduction data
        h1 = instance_data["interdiction_parameters"]["interdiction_cost_level1_h1"]
        h2 = instance_data["interdiction_parameters"]["interdiction_cost_level2_h2"]
        d1 = instance_data["interdiction_parameters"]["capacity_reduction_level1_d1"]
        d2 = instance_data["interdiction_parameters"]["capacity_reduction_level2_d2"]
    
        # Extract distances
        dist_c_l1 = instance_data["distances"]["customer_to_level1_facility"] 
        dist_c_l2 = instance_data["distances"]["customer_to_level2_facility"]
        dist_l1_l2 = instance_data["distances"]["level1_to_level2_facility"]
    
        # Calculate demand splits
        demand_t1 = [theta * d for d in demands]  # Type I demand for each customer
        demand_t2 = [(1 - theta) * d for d in demands]  # Type II demand for each customer

        # Create the model
        model = gp.Model("interdiction_bilevel")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPFocus", 1)  # Focus on finding feasible solutions
        model.setParam("OutputFlag", 0)

        # Decision variables for interdictor - interdiction level for each facility
        # We use binary variables to indicate which level is chosen for each facility
        interdict_levels_l1 = {}  # interdict_levels_l1[j][l] = 1 if level 1 facility j is attacked at level l
        interdict_levels_l2 = {}  # interdict_levels_l2[k][l] = 1 if level 2 facility k is attacked at level l

        # Create variables for level 1 facilities
        for j in range(num_level1_facilities):
            interdict_levels_l1[j] = {}
            for l in range(K_levels):
                interdict_levels_l1[j][l] = model.addVar(vtype=GRB.BINARY, 
                                                        name=f"interdict_l1_{j}_level_{l}")
        
        # Create variables for level 2 facilities                            
        for k in range(num_level2_facilities):
            interdict_levels_l2[k] = {}
            for l in range(K_levels):
                interdict_levels_l2[k][l] = model.addVar(vtype=GRB.BINARY, 
                                                         name=f"interdict_l2_{k}_level_{l}")
    
        # Constraint: Exactly one interdiction level per facility
        for j in range(num_level1_facilities):
            model.addConstr(gp.quicksum(interdict_levels_l1[j][l] for l in range(K_levels)) == 1,
                           name=f"one_level_l1_{j}")
    
        for k in range(num_level2_facilities):
            model.addConstr(gp.quicksum(interdict_levels_l2[k][l] for l in range(K_levels)) == 1,
                           name=f"one_level_l2_{k}")
    
        # Budget constraint for interdiction
        budget_expr = gp.LinExpr()
    
        # Add interdiction costs for level 1 facilities
        for j in range(num_level1_facilities):
            for l in range(1, K_levels):  # Level 0 has zero cost
                level_cost = float(h1[str(l)]) if str(l) in h1 else 0.0
                budget_expr.addTerms(level_cost, interdict_levels_l1[j][l])
    
        # Add interdiction costs for level 2 facilities
        for k in range(num_level2_facilities):
            for l in range(1, K_levels):
                level_cost = float(h2[str(l)]) if str(l) in h2 else 0.0
                budget_expr.addTerms(level_cost, interdict_levels_l2[k][l])
    
        model.addConstr(budget_expr <= total_interdiction_budget, name="budget")
    
        # Compute effective capacities after interdiction (for use later)
        effective_cap_l1 = {}  # Expression for effective capacity after interdiction
        effective_cap_l2_t1 = {}
        effective_cap_l2_t2 = {}
    
        for j in range(num_level1_facilities):
            cap_expr = gp.LinExpr()
            initial_cap = float(initial_cap_l1[j])
            for l in range(K_levels):
                reduction_factor = float(d1[str(l)]) if str(l) in d1 else 0.0
                actual_cap_contribution = initial_cap * (1.0 - reduction_factor)
                cap_expr.addTerms(actual_cap_contribution, interdict_levels_l1[j][l])
            effective_cap_l1[j] = cap_expr
    
        for k in range(num_level2_facilities):
            cap_expr_t1 = gp.LinExpr()
            cap_expr_t2 = gp.LinExpr()
            initial_cap_t1 = float(initial_cap_l2_t1[k])
            initial_cap_t2 = float(initial_cap_l2_t2[k])
            for l in range(K_levels):
                reduction_factor = float(d2[str(l)]) if str(l) in d2 else 0.0
                cap_contribution_t1 = initial_cap_t1 * (1.0 - reduction_factor)
                cap_contribution_t2 = initial_cap_t2 * (1.0 - reduction_factor)
                cap_expr_t1.addTerms(cap_contribution_t1, interdict_levels_l2[k][l])
                cap_expr_t2.addTerms(cap_contribution_t2, interdict_levels_l2[k][l])
            effective_cap_l2_t1[k] = cap_expr_t1
            effective_cap_l2_t2[k] = cap_expr_t2
    
        # Now define defender decision variables
        # Amount of type I demand from customer i to level 1 facility j
        x1 = {}
        for i in range(num_customers):
            for j in range(num_level1_facilities):
                x1[i, j] = model.addVar(lb=0.0, name=f"x1_{i}_{j}")
        
        # Amount of type I demand from customer i to level 2 facility k
        x2_t1 = {}
        for i in range(num_customers):
            for k in range(num_level2_facilities):
                x2_t1[i, k] = model.addVar(lb=0.0, name=f"x2_t1_{i}_{k}")
        
        # Amount of type II demand from customer i to level 2 facility k
        x2_t2 = {}
        for i in range(num_customers):
            for k in range(num_level2_facilities):
                x2_t2[i, k] = model.addVar(lb=0.0, name=f"x2_t2_{i}_{k}")
    
        # Forwarded referral demand from level 1 facility j to level 2 facility k
        z1 = {}
        for j in range(num_level1_facilities):
            for k in range(num_level2_facilities):
                z1[j, k] = model.addVar(lb=0.0, name=f"z1_{j}_{k}")
                
        # Outsourced demand variables
        out_t1 = {}  # Outsourced type I demand from customer i
        out_t2 = {}  # Outsourced type II demand from customer i
        out_r1 = {}  # Outsourced referral demand from level 1 facility j
        
        for i in range(num_customers):
            out_t1[i] = model.addVar(lb=0.0, name=f"out_t1_{i}")
            out_t2[i] = model.addVar(lb=0.0, name=f"out_t2_{i}")
            
        for j in range(num_level1_facilities):
            out_r1[j] = model.addVar(lb=0.0, name=f"out_r1_{j}")
    
        # Constraints for demand satisfaction
        # Type I demand
        for i in range(num_customers):
            model.addConstr(
                gp.quicksum(x1[i, j] for j in range(num_level1_facilities)) +
                gp.quicksum(x2_t1[i, k] for k in range(num_level2_facilities)) +
                out_t1[i] == demand_t1[i],
                name=f"typeI_demand_satisfaction_c{i}"
            )
        
        # Type II demand
        for i in range(num_customers):
            model.addConstr(
                gp.quicksum(x2_t2[i, k] for k in range(num_level2_facilities)) +
                out_t2[i] == demand_t2[i],
                name=f"typeII_demand_satisfaction_c{i}"
            )
    
        # Capacity constraints for facilities
        # Level 1 facilities: only type I capacity
        for j in range(num_level1_facilities):
            model.addConstr(
                gp.quicksum(x1[i, j] for i in range(num_customers)) <= effective_cap_l1[j],
                name=f"capacity_l1_f{j}"
            )
    
        # Level 2 facilities: type I capacity constraint
        for k in range(num_level2_facilities):
            model.addConstr(
                gp.quicksum(x2_t1[i, k] for i in range(num_customers)) <= effective_cap_l2_t1[k],
                name=f"capacity_l2_t1_f{k}"
            )
    
        # Level 2 facilities: total type II + referral demand constraint
        for k in range(num_level2_facilities):
            model.addConstr(
                gp.quicksum(x2_t2[i, k] for i in range(num_customers)) +
                gp.quicksum(z1[j, k] for j in range(num_level1_facilities)) <= effective_cap_l2_t2[k],
                name=f"capacity_l2_combined_f{k}"
            )
    
        # Referral generation and allocation constraints
        # At each level 1 facility j: generated refers to type I demand served
        for j in range(num_level1_facilities):
            type_i_served_at_j = gp.quicksum(x1[i, j] for i in range(num_customers))
            generated_referral = sigma * type_i_served_at_j
            model.addConstr(
                gp.quicksum(z1[j, k] for k in range(num_level2_facilities)) + out_r1[j] == generated_referral,
                name=f"referral_balance_l1_{j}"
            )
    
        # Finally, define the defender's objective function (the defender minimizes this)
        # This is what the interdictor tries to maximize
        def_total_cost = gp.LinExpr()
        
        # Customer-to-facility allocation costs
        for i in range(num_customers):
            for j in range(num_level1_facilities):
                def_total_cost.addTerms(alpha * dist_c_l1[i][j], x1[i, j])
            for k in range(num_level2_facilities):
                def_total_cost.addTerms(beta * dist_c_l2[i][k], x2_t1[i, k])
                def_total_cost.addTerms(beta * dist_c_l2[i][k], x2_t2[i, k])
    
        # Referral forwarding costs
        for j in range(num_level1_facilities):
            for k in range(num_level2_facilities):
                def_total_cost.addTerms(gamma * dist_l1_l2[j][k], z1[j, k])
    
        # Outsourcing costs
        for i in range(num_customers):
            def_total_cost.addTerms(alpha_prime, out_t1[i])
            def_total_cost.addTerms(beta_prime, out_t2[i])
            
        for j in range(num_level1_facilities):
            def_total_cost.addTerms(gamma_prime, out_r1[j])
    
        # The interdictor wants to maximize the defender's minimum cost.
        # So we set the defender's cost as the objective to be maximized
        # (since defender will choose the allocations to minimize it for any given attack)
        model.setObjective(def_total_cost, GRB.MAXIMIZE)
    
        # Solve model
        model.optimize()
    
        # Extract solution
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            solution = {
                "objective_value": model.ObjVal if model.SolCount > 0 else 0.0,
                "interdiction_pattern": {}
            }
            
            # Extract interdiction pattern
            for j in range(num_level1_facilities):
                for l in range(K_levels):
                    if abs(interdict_levels_l1[j][l].X - 1) < 0.001:
                        solution["interdiction_pattern"][f"S1_{j}"] = l
                        break
            
            for k in range(num_level2_facilities):
                for l in range(K_levels):
                    if abs(interdict_levels_l2[k][l].X - 1) < 0.001:
                        solution["interdiction_pattern"][f"S2_{k}"] = l
                        break
    
            return solution
        else:
            # Infeasible, or unknown status
            return {
                "objective_value": 0.0,
                "interdiction_pattern": {}
            }
    
    except Exception as e:
        # Return error-safe solution
        print(f"Error solving problem: {e}")
        return {
            "objective_value": 0.0,
            "interdiction_pattern": {}
        }


def main():
    parser = argparse.ArgumentParser(description='Solve the nested hierarchical service network interdiction problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to save log file.')
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if specified
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Solve the instance
    solution = solve_interdiction_problem(instance_data, args.time_limit, args.log_path)

    if logger and solution["objective_value"] > 0:  # Valid solution found
        logger.log(solution["objective_value"])
    
    # Write solution
    output_dict = {
        "objective_value": solution["objective_value"],
        "interdiction_pattern": solution["interdiction_pattern"]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    main()