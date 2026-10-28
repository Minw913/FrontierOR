import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger


def solve_interdiction_problem(instance_data, time_limit, log_path):
    # Extract global parameters
    theta = instance_data["global_parameters"]["theta"]
    sigma = instance_data["global_parameters"]["sigma"]
    alpha = instance_data["global_parameters"]["alpha"]
    beta = instance_data["global_parameters"]["beta"]
    gamma = instance_data["global_parameters"]["gamma"]
    alpha_prime = instance_data["global_parameters"]["alpha_prime"]
    beta_prime = instance_data["global_parameters"]["beta_prime"]
    gamma_prime = instance_data["gamma_prime"]
    total_interdiction_budget = instance_data["instance_parameters"]["total_interdiction_budget"]
    
    # Extract dimensional information
    num_level1_facilities = instance_data["instance_parameters"]["num_level1_facilities"]
    num_level2_facilities = instance_data["instance_parameters"]["num_level2_facilities"]
    num_customers = instance_data["instance_parameters"]["num_customers"]
    K_levels = instance_data["instance_parameters"]["num_interdiction_levels_K"]
    k_max = instance_data["instance_parameters"]["k_max"]
    
    # Extract demand data
    demands = instance_data["customers"]["demands"]
    x_coords_customers = instance_data["customers"]["x_coordinates"]
    y_coords_customers = instance_data["customers"]["y_coordinates"]
    
    # Extract level 1 facility data
    x_coords_l1 = instance_data["level1_facilities"]["x_coordinates"]
    y_coords_l1 = instance_data["level1_facilities"]["y_coordinates"]
    initial_cap_l1 = instance_data["level1_facilities"]["capacity_type1"]
    
    # Extract level 2 facility data
    x_coords_l2 = instance_data["level2_facilities"]["x_coordinates"]
    y_coords_l2 = instance_data["level2_facilities"]["y_coordinates"]
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
    
    # Create the model that simulates the defender's response for a given interdiction pattern
    def create_defender_model(interdict_values):
        m = gp.Model("defender_model")
        m.setParam("OutputFlag", 0)
        
        # Variables
        # Flow from customer i to level 1 facility j for type I demand
        x1 = m.addVars(num_customers, num_level1_facilities, lb=0, name="x1")
        # Flow from customer i to level 2 facility k for type I demand
        x2_1 = m.addVars(num_customers, num_level2_facilities, lb=0, name="x2_1")
        # Flow from customer i to level 2 facility k for type II demand
        x2_2 = m.addVars(num_customers, num_level2_facilities, lb=0, name="x2_2")
        # Referral flow from level 1 facility j to level 2 facility k
        z1 = m.addVars(num_level1_facilities, num_level2_facilities, lb=0, name="z1")
        # Outsourced type I demand for customer i
        out1 = m.addVars(num_customers, lb=0, name="out1")
        # Outsourced type II demand for customer i
        out2 = m.addVars(num_customers, lb=0, name="out2")
        # Outsourced referral demand from level 1 facility j
        out_r1 = m.addVars(num_level1_facilities, lb=0, name="out_r1")
        # Outsourced referral demand from outsourced type I for customer i
        out_r_from_out1 = m.addVars(num_customers, lb=0, name="out_r_from_out1")
        
        # Calculate residual capacities after interdiction
        cap_l1 = []
        for j in range(num_level1_facilities):
            k_val = int(interdict_values[f'S1_{j}'])
            red_factor = d1.get(str(k_val), 0.0)
            cap_l1.append(initial_cap_l1[j] * (1 - red_factor))
            
        cap_l2_t1 = []
        cap_l2_t2 = []
        for k in range(num_level2_facilities):
            k_val = int(interdict_values[f'S2_{k}'])
            red_factor = d2.get(str(k_val), 0.0)
            cap_l2_t1.append(initial_cap_l2_t1[k] * (1 - red_factor))
            cap_l2_t2.append(initial_cap_l2_t2[k] * (1 - red_factor))
        
        # Constraints
        # Demand satisfaction for type I
        for i in range(num_customers):
            m.addConstr(
                gp.quicksum(x1[i, j] for j in range(num_level1_facilities)) +
                gp.quicksum(x2_1[i, k] for k in range(num_level2_facilities)) +
                out1[i] == demand_t1[i],
                name=f"demand_satisfaction_t1_{i}"
            )
        
        # Demand satisfaction for type II  
        for i in range(num_customers):
            m.addConstr(
                gp.quicksum(x2_2[i, k] for k in range(num_level2_facilities)) +
                out2[i] == demand_t2[i],
                name=f"demand_satisfaction_t2_{i}"
            )
        
        # Capacity constraint for level I facilities (type I only)
        for j in range(num_level1_facilities):
            m.addConstr(
                gp.quicksum(x1[i, j] for i in range(num_customers)) <= cap_l1[j],
                name=f"capacity_l1_{j}"
            )
            
        # Capacity constraint for level II facilities type I demand
        for k in range(num_level2_facilities):
            m.addConstr(
                gp.quicksum(x2_1[i, k] for i in range(num_customers)) <= cap_l2_t1[k],
                name=f"capacity_l2_t1_{k}"
            )
            
        # Capacity constraint for level II facilities (type II + referrals)
        for k in range(num_level2_facilities):
            referral_incoming = (
                gp.quicksum(z1[j, k] for j in range(num_level1_facilities)) +
                sigma * out_r_from_out1.sum()  # Actually, outsourced direct type I doesn't generate referrals in the same way as in-facility service
            )
            m.addConstr(
                gp.quicksum(x2_2[i, k] for i in range(num_customers)) +
                referral_incoming <= cap_l2_t2[k],
                name=f"capacity_l2_combined_{k}"
            )
            
        # Referral from l1 facilities constraint
        for j in range(num_level1_facilities):
            total_t1_served_at_j = gp.quicksum(x1[i, j] for i in range(num_customers))
            actual_referral = sigma * total_t1_served_at_j
            m.addConstr(
                gp.quicksum(z1[j, k] for k in range(num_level2_facilities)) + out_r1[j] == actual_referral,
                name=f"referral_balance_l1_{j}"
            )
          
        # Referral from outsourced T1 demand also gets outsourced
        for i in range(num_customers):
             m.addConstr(out_r_from_out1[i] == sigma * out1[i], name=f"referral_from_out1_{i}")
        
        # Objective
        obj_lin = gp.LinExpr()
        
        # Costs for transportation to L1
        for i in range(num_customers):
            for j in range(num_level1_facilities):
                obj_lin.addTerms(alpha * dist_c_l1[i][j], x1[i, j])
                
        # Costs for transportation to L2 for type I and II
        for i in range(num_customers):
            for k in range(num_level2_facilities):
                obj_lin.addTerms(beta * dist_c_l2[i][k], x2_1[i, k])
                obj_lin.addTerms(beta * dist_c_l2[i][k], x2_2[i, k])
        
        # Costs for referrals from L1 to L2
        for j in range(num_level1_facilities):
            for k in range(num_level2_facilities):
                obj_lin.addTerms(gamma * dist_l1_l2[j][k], z1[j, k])
        
        # Outsourcing costs
        for i in range(num_customers):
            obj_lin.addTerms(alpha_prime, out1[i])
            obj_lin.addTerms(beta_prime, out2[i])
            # Referential cost for outsourced T1
            obj_lin.addTerms(sigma * gamma_prime, out1[i])  # This is captured with out1[i]
        
        # Outsourcing for referral from L1 facilities
        for j in range(num_level1_facilities):
            obj_lin.addTerms(gamma_prime, out_r1[j])
        
        m.setObjective(obj_lin, GRB.MINIMIZE)
        
        return m, x1, x2_1, x2_2, z1, out1, out2, out_r1, out_r_from_out1

    # Bilevel problem as MILP using KKT conditions or big-M reformulation
    # Here we will use a direct approach with enumeration or MIP formulation if possible

    # For solving this with a modern bi-level approach (MIP reformulation), we would need
    # a more complex model incorporating the defender's dual variables.
    # Instead we implement a grid search approach limited by time and a heuristic approach if time too short
    
    model = gp.Model("interdiction_bilevel")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPFocus", 1)  # Focus on finding feasible solutions
    
    # Decision variables for interdictor
    interdict_x1 = {}  # S1_j
    interdict_x2 = {}  # S2_k 
    
    for j in range(num_level1_facilities):
        interdict_x1[j] = model.addVars(list(range(K_levels)), vtype=GRB.BINARY, name=f"interdict_l1_{j}")
        
    for k in range(num_level2_facilities):
        interdict_x2[k] = model.addVars(list(range(K_levels)), vtype=GRB.BINARY, name=f"interdict_l2_{k}")
    
    # Each facility gets exactly one interdiction level
    for j in range(num_level1_facilities):
        model.addConstr(gp.quicksum(interdict_x1[j][l] for l in range(K_levels)) == 1)
        
    for k in range(num_level2_facilities):
        model.addConstr(gp.quicksum(interdict_x2[k][l] for l in range(K_levels)) == 1)
    
    # Budget constraints
    budget_expr = gp.LinExpr()
    
    for j in range(num_level1_facilities):
        for l in range(1, K_levels):  # Skip l=0 because cost is 0
            budget_expr.addTerms(h1[str(l)], interdict_x1[j][l])
    
    for k in range(num_level2_facilities):
        for l in range(1, K_levels):  # Skip l=0 because cost is 0
            budget_expr.addTerms(h2[str(l)], interdict_x2[k][l])
    
    model.addConstr(budget_expr <= total_interdiction_budget)
    
    # We will now transform this into a MIP-friendly bi-level form
    # using Fortuny-Amat and McCarl method or similar
    # but for efficiency, we'll implement a custom solution loop

    # To handle the bilevel nature, we:
    # 1. Relax it to assume we know the optimal defender response
    # 2. Formulate a single-level equivalent using KKT conditions + strong duality
    
    # Create dual variables and constraints for the inner problem
    # This approach becomes quite complex. Let's try a more practical approach:
    # Sample-based or enumeration with limited depth

    # Initialize values to track best interdiction
    best_interdict_pattern = {}
    best_cost = float('-inf')
    
    # Use Gurobi's multi-objective or iterative approach
    # This is a max-min problem: max over x {min over y of defender cost}
    # For small enough instances, enumerate, for larger use solver
    # Since this is NP-hard in general, we use exact solver approach with proper reformulation
    
    # Reformulate as single-level using KKT conditions
    # Introduce primal and dual variables for defender problem + complementarity
    # But complementarity makes it non-linear. We linearize with binary variables using big-M method.
    
    # Variables
    # Interdiction decisions (already done)
    
    # Defender problem primal variables (we'll duplicate them symbolically)
    # This will become a large single level problem with duals
    
    # Create dual variables for all defender constraints
    mu_c1 = model.addVars(num_customers, lb=-GRB.INFINITY, name="mu_c1")  # demand I
    mu_c2 = model.addVars(num_customers, lb=-GRB.INFINITY, name="mu_c2")  # demand II
    mu_l1 = model.addVars(num_level1_facilities, lb=-GRB.INFINITY, ub=0, name="mu_l1")  # l1 capacity
    mu_l2t1 = model.addVars(num_level2_facilities, lb=-GRB.INFINITY, ub=0, name="mu_l2t1")  # l2 cap t1
    mu_l2cmb = model.addVars(num_level2_facilities, lb=-GRB.INFINITY, ub=0, name="mu_l2cmb")  # l2 cap comb
    mu_r1 = model.addVars(num_level1_facilities, lb=-GRB.INFINITY, name="mu_r1")  # ref constraint
    mu_out1 = model.addVars(num_customers, lb=-GRB.INFINITY, ub=0, name="mu_out1")  # referral from out1 constraint

    # Primal variables for defender (these are part of bilinear terms)
    x1 = model.addVars(num_customers, num_level1_facilities, lb=0, name="x1")
    x2_1 = model.addVars(num_customers, num_level2_facilities, lb=0, name="x2_1")
    x2_2 = model.addVars(num_customers, num_level2_facilities, lb=0, name="x2_2")
    z1 = model.addVars(num_level1_facilities, num_level2_facilities, lb=0, name="z1")
    out1 = model.addVars(num_customers, lb=0, name="out1")
    out2 = model.addVars(num_customers, lb=0, name="out2")
    out_r1 = model.addVars(num_level1_facilities, lb=0, name="out_r1")
    out_r_from_out1 = model.addVars(num_customers, lb=0, name="out_r_from_out1")
    
    # Binary variables for complementarity in KKT
    # These indicate if a primal or dual constraint is active at zero
    delta_demand1 = model.addVars(num_customers, num_level1_facilities, vtype=GRB.BINARY, name="delta_d1")
    delta_demand2_1 = model.addVars(num_customers, num_level2_facilities, vtype=GRB.BINARY, name="delta_d2_1")
    delta_demand2_2 = model.addVars(num_customers, num_level2_facilities, vtype=GRB.BINARY, name="delta_d2_2")
    delta_ref = model.addVars(num_level1_facilities, num_level2_facilities, vtype=GRB.BINARY, name="delta_ref")
    delta_out1 = model.addVars(num_customers, vtype=GRB.BINARY, name="delta_out1")
    delta_out2 = model.addVars(num_customers, vtype=GRB.BINARY, name="delta_out2")
    delta_outr1 = model.addVars(num_level1_facilities, vtype=GRB.BINARY, name="delta_outr1")
    delta_outr_o1 = model.addVars(num_customers, vtype=GRB.BINARY, name="delta_outr_o1")
    
    # Calculate effective capacities as bilinear terms depending on interdiction decision
    cap_l1_var = {}
    cap_l2t1_var = {}
    cap_l2cmb_var = {}

    for j in range(num_level1_facilities):
        expr = 0
        for l in range(K_levels):
            red_factor = d1.get(str(l), 0.0)
            expr += initial_cap_l1[j] * (1 - red_factor) * interdict_x1[j][l]
        cap_l1_var[j] = expr
        
    for k in range(num_level2_facilities):
        expr1 = 0
        expr2 = 0
        for l in range(K_levels):
            red_factor = d2.get(str(l), 0.0)
            expr1 += initial_cap_l2_t1[k] * (1 - red_factor) * interdict_x2[k][l]
            expr2 += initial_cap_l2_t2[k] * (1 - red_factor) * interdict_x2[k][l]
        cap_l2t1_var[k] = expr1
        cap_l2cmb_var[k] = expr2

    # Add constraints representing defender problem (stationarity, primal feasibility)
    # Demand satisfaction constraints
    for i in range(num_customers):
        model.addConstr(
            gp.quicksum(x1[i, j] for j in range(num_level1_facilities)) +
            gp.quicksum(x2_1[i, k] for k in range(num_level2_facilities)) +
            out1[i] == demand_t1[i],
            name=f"def_primal_d1_{i}"
        )

        model.addConstr(
            gp.quicksum(x2_2[i, k] for k in range(num_level2_facilities)) +
            out2[i] == demand_t2[i],
            name=f"def_primal_d2_{i}"
        )

    # Capacity constraints
    for j in range(num_level1_facilities):
        model.addConstr(
            gp.quicksum(x1[i, j] for i in range(num_customers)) <= cap_l1_var[j],
            name=f"def_primal_cap_l1_{j}"
        )

    for k in range(num_level2_facilities):
        model.addConstr(
            gp.quicksum(x2_1[i, k] for i in range(num_customers)) <= cap_l2t1_var[k],
            name=f"def_primal_cap_l2_t1_{k}"
        )
        
        # For referral capacity constraint in level 2, we need to be careful 
        # about the referral term for now we only add the z1 part explicitly
        total_used = gp.quicksum(x2_2[i, k] for i in range(num_customers)) + \
                    gp.quicksum(z1[j, k] for j in range(num_level1_facilities))
        model.addConstr(total_used <= cap_l2cmb_var[k], name=f"def_primal_cap_l2_cmb_{k}")

    # Referral balance
    for j in range(num_level1_facilities):
        total_t1_served = gp.quicksum(x1[i, j] for i in range(num_customers))
        actual_referral = sigma * total_t1_served
        ref_satisfied = gp.quicksum(z1[j, k] for k in range(num_level2_facilities)) + out_r1[j]
        model.addConstr(ref_satisfied == actual_referral, name=f"def_primal_ref_{j}")

    # Out1 referral constraint
    for i in range(num_customers):
        model.addConstr(out_r_from_out1[i] == sigma * out1[i], name=f"def_out_r_out1_{i}")

    # Stationarity conditions for defender (KKT conditions for LP)
    # \nabla f - \sum \lambda_i \nabla g_i = 0
    # For x1[i,j]: grad = alpha * c_ij - pi1_c[i] - pi1_l1[j] 
    M = 1e6  # Big-M constant

    # Add stationarity constraints
    # For each primal variable, we have partial derivatives = 0 with Lagrange multipliers
    
    # For x1[i,j]: coefficient in minimization obj = alpha * dist_c_l1[i][j]
    # In KKT: alpha * dist_c_l1[i][j] - mu_c1[i] - mu_l1[j] >= 0
    # This gives: mu_c1[i] + mu_l1[j] <= alpha * dist_c_l1[i][j] 
    # Using complementary slackness:
    for i in range(num_customers):
        for j in range(num_level1_facilities):
            model.addConstr(mu_c1[i] + mu_l1[j] <= alpha * dist_c_l1[i][j],
                            name=f"stat_x1_constr_{i}_{j}")
            # Complementarity: x1[i,j] * (lhs_rhs - obj_coeff) = 0
            # Linearized: x1[i,j] * (alpha*dist - mu_c1[i] - mu_l1[j]) = 0
            # Using Fortuny-Amat:
            # x1[i,j] <= M * delta_demand1[i,j]
            # (alpha * dist_c_l1[i][j] - mu_c1[i] - mu_l1[j]) <= M * (1 - delta_demand1[i,j])
            model.addConstr(x1[i, j] <= M * delta_demand1[i, j])
            model.addConstr(
                alpha * dist_c_l1[i][j] - mu_c1[i] - mu_l1[j] <= M * (1 - delta_demand1[i, j]),
                name=f"comp_x1_{i}_{j}"
            )

    # For x2_1[i,k]: beta * dist_c_l2[i][k] - mu_c1[i] - mu_l2t1[k] >= 0 -> <= by sign
    for i in range(num_customers):
        for k in range(num_level2_facilities):
            model.addConstr(mu_c1[i] + mu_l2t1[k] <= beta * dist_c_l2[i][k],
                            name=f"stat_x2_1_constr_{i}_{k}")
            model.addConstr(x2_1[i, k] <= M * delta_demand2_1[i, k])
            model.addConstr(
                beta * dist_c_l2[i][k] - mu_c1[i] - mu_l2t1[k] <= M * (1 - delta_demand2_1[i, k]),
                name=f"comp_x2_1_{i}_{k}"
            )

    # For x2_2[i,k]: beta * dist_c_l2[i][k] - mu_c2[i] - mu_l2cmb[k] >= 0
    for i in range(num_customers):
        for k in range(num_level2_facilities):
            model.addConstr(mu_c2[i] + mu_l2cmb[k] <= beta * dist_c_l2[i][k],
                            name=f"stat_x2_2_constr_{i}_{k}")
            model.addConstr(x2_2[i, k] <= M * delta_demand2_2[i, k])
            model.addConstr(
                beta * dist_c_l2[i][k] - mu_c2[i] - mu_l2cmb[k] <= M * (1 - delta_demand2_2[i, k]),
                name=f"comp_x2_2_{i}_{k}"
            )

    # For z1[j,k]: gamma * dist_l1_l2[j][k] - mu_r1[j] - (-mu_l2cmb[k]) >= 0 => mu_r1[j] + mu_l2cmb[k] <= gamma * dist
    for j in range(num_level1_facilities):
        for k in range(num_level2_facilities):
            model.addConstr(mu_r1[j] + mu_l2cmb[k] <= gamma * dist_l1_l2[j][k],
                            name=f"stat_z1_constr_{j}_{k}")
            model.addConstr(z1[j, k] <= M * delta_ref[j, k])
            model.addConstr(
                gamma * dist_l1_l2[j][k] - mu_r1[j] - mu_l2cmb[k] <= M * (1 - delta_ref[j, k]),
                name=f"comp_z1_{j}_{k}"
            )

    # For out_k vars: cost coefficients - multipliers
    # out1: (alpha_prime + sigma * gamma_prime) - (-mu_c1[i]) - (-mu_out1[i]) = 0
    # So: (alpha_prime + sigma * gamma_prime) + mu_c1[i] + mu_out1[i] = 0
    # Which means: mu_c1[i] + mu_out1[i] = -(alpha_prime + sigma * gamma_prime)
    for i in range(num_customers):
        model.addConstr(-mu_c1[i] - mu_out1[i] <= alpha_prime + sigma * gamma_prime,
                        name=f"stat_out1_constr_{i}")
        model.addConstr(out1[i] <= M * delta_out1[i])
        model.addConstr(
            (alpha_prime + sigma * gamma_prime) + mu_c1[i] + mu_out1[i] <= M * (1 - delta_out1[i]),
            name=f"comp_out1_{i}"
        )

    # out2: beta_prime + mu_c2[i] + mu_out2[i] = 0
    # mu_c2[i] + mu_out2[i] = -beta_prime
    for i in range(num_customers):
        model.addConstr(-mu_c2[i] - mu_out2[i] <= beta_prime, name=f"stat_out2_constr_{i}")
        model.addConstr(out2[i] <= M * delta_out2[i])
        model.addConstr(
            beta_prime + mu_c2[i] + mu_out2[i] <= M * (1 - delta_out2[i]), name=f"comp_out2_{i}"
        )

    # out_r1: gamma_prime - (-mu_r1[j]) = -gamma_prime - mu_r1[j] >= 0 => mu_r1[j] <= -gamma_prime
    # So we have: gamma_prime + mu_r1[j] + mu_our1[j] = 0 but no mu_our1 since out_r1 not in demand eqn
    # Actually gamma_prime + mu_r1[j] = 0 =>  mu_r1[j] = -gamma_prime
    for j in range(num_level1_facilities):
        model.addConstr(mu_r1[j] <= -gamma_prime, name=f"stat_outr1_constr_{j}")
        model.addConstr(out_r1[j] <= M * delta_outr1[j])
        model.addConstr(
            gamma_prime + mu_r1[j] <= M * (1 - delta_outr1[j]), name=f"comp_outr1_{j}"
        )

    # out_r_from_out1: sigma*gamma_prime + mu_out1[i] = 0 ? No this is not an independent flow var
    # Its cost is part of out1, and the constraint is out_r_from_out1[i] = sigma*out1[i]
    # Cost handled already in out1 coefficient. No dual needed for this link const

    # Objective: Maximize defender's minimum cost
    # Due to strong duality, min primal cost = max dual cost
    # So we can either use primal objective value as a variable and constraint or directly use dual obj
    # Let DefCost be defender's cost (minimize it) and our obj is -DefCost (maximize)
    # Strong duality: primal_obj = dual_obj
    def_cost_expr = gp.LinExpr()

    for i in range(num_customers):
        for j in range(num_level1_facilities):
            def_cost_expr.addTerms(alpha * dist_c_l1[i][j], x1[i, j])

    for i in range(num_customers):
        for k in range(num_level2_facilities):
            def_cost_expr.addTerms(beta * dist_c_l2[i][k], x2_1[i, k])
            def_cost_expr.addTerms(beta * dist_c_l2[i][k], x2_2[i, k])

    for j in range(num_level1_facilities):
        for k in range(num_level2_facilities):
            def_cost_expr.addTerms(gamma * dist_l1_l2[j][k], z1[j, k])

    for i in range(num_customers):
        def_cost_expr.addTerms(alpha_prime + sigma * gamma_prime, out1[i])
        def_cost_expr.addTerms(beta_prime, out2[i])

    for j in range(num_level1_facilities):
        def_cost_expr.addTerms(gamma_prime, out_r1[j])

    # Dual objective (for strong duality check)
    dual_obj = gp.LinExpr()
    for i in range(num_customers):
        dual_obj.addConstant(-demand_t1[i] * mu_c1[i])
        dual_obj.addConstant(-demand_t2[i] * mu_c2[i])
        
    for j in range(num_level1_facilities):
        dual_obj.addTerms(-1, mu_l1[j] * cap_l1_var[j])  # This is now bilinear!
    
    for k in range(num_level2_facilities):
        dual_obj.addTerms(-1, mu_l2t1[k] * cap_l2t1_var[k])  # Also bilinear!
        dual_obj.addTerms(-1, mu_l2cmb[k] * cap_l2cmb_var[k])  # Also bilinear!

    # Add def_cost_expr to maximize (negative in min prob)
    model.setObjective(def_cost_expr, GRB.MAXIMIZE)
    
    # Solve the model
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        # Extract solution
        interdict_solution = {}
        for j in range(num_level1_facilities):
            for l in range(K_levels):
                if interdict_x1[j][l].X > 0.99:
                    interdict_solution[f"S1_{j}"] = l
        
        for k in range(num_level2_facilities):
            for l in range(K_levels):
                if interdict_x2[k][l].X > 0.99:
                    interdict_solution[f"S2_{k}"] = l

        return {
            "objective_value": model.ObjVal,
            "interdiction_pattern": interdict_solution
        }
    elif model.Status in [GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # Try to get best solution found
        if model.SolCount > 0:
            interdict_solution = {}
            for j in range(num_level1_facilities):
                for l in range(K_levels):
                    if interdict_x1[j][l].X > 0.99:
                        interdict_solution[f"S1_{j}"] = l
            
            for k in range(num_level2_facilities):
                for l in range(K_levels):
                    if interdict_x2[k][l].X > 0.99:
                        interdict_solution[f"S2_{k}"] = l

            return {
                "objective_value": model.ObjVal,
                "interdiction_pattern": interdict_solution
            }

    return {
        "objective_value": float('inf'),
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

    if solution['objective_value'] != float('inf'):
        # Log the solution if logger is provided
        if logger:
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