import json
import math
import time
import gurobipy as gp
from gurobipy import GRB

def main(instance_path, solution_path, time_limit):
    start_time = time.time()
    with open(instance_path) as f:
        data = json.load(f)
    
    # Extract parameters
    theta = data["theta"]
    sigma = data["sigma"]
    total_interdiction_budget = data["total_interdiction_budget"]
    interdiction_parameters = data["interdiction_parameters"]
    level1_facilities = data["level1_facilities"]
    level2_facilities = data["level2_facilities"]
    customers = data["customers"]
    distances = data["distances"]
    
    n1 = len(level1_facilities["id"])
    n2 = len(level2_facilities["id"])
    n_customers = len(customers["id"])
    K = data["num_interdiction_levels_K"]
    
    # Precompute level2 to level2 distances
    dist_level2_level2 = []
    for j2 in range(n2):
        dist_row = []
        x1, y1 = level2_facilities["x_coordinate"][j2], level2_facilities["y_coordinate"][j2]
        for j2p in range(n2):
            x2 = level2_facilities["x_coordinate"][j2p]
            y2 = level2_facilities["y_coordinate"][j2p]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            dist_row.append(dist)
        dist_level2_level2.append(dist_row)
    
    # Initialize master problem
    m = gp.Model()
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('OutputFlag', 0)
    
    # Add interdiction variables
    x1 = {}
    for j1 in range(n1):
        for k in range(K):
            cost = interdiction_parameters["interdiction_cost_level1_h1"].get(str(k), 0.0)
            x1[j1, k] = m.addVar(vtype=GRB.BINARY, name=f"x1_{j1}_{k}")
    
    x2 = {}
    for j2 in range(n2):
        for k in range(K):
            cost = interdiction_parameters["interdiction_cost_level2_h2"].get(str(k), 0.0)
            x2[j2, k] = m.addVar(vtype=GRB.BINARY, name=f"x2_{j2}_{k}")
    
    eta = m.addVar(lb=0, name="eta")
    
    # Facility selection constraints
    for j1 in range(n1):
        m.addConstr(gp.quicksum(x1[j1, k] for k in range(K)) == 1)
    for j2 in range(n2):
        m.addConstr(gp.quicksum(x2[j2, k] for k in range(K)) == 1)
    
    # Budget constraint
    budget_expr = gp.quicksum(
        interdiction_parameters["interdiction_cost_level1_h1"].get(str(k), 0.0) * x1[j1, k]
        for j1 in range(n1) for k in range(K)
    ) + gp.quicksum(
        interdiction_parameters["interdiction_cost_level2_h2"].get(str(k), 0.0) * x2[j2, k]
        for j2 in range(n2) for k in range(K)
    )
    m.addConstr(budget_expr <= total_interdiction_budget)
    m.setObjective(eta, GRB.MAXIMIZE)
    
    # Benders loop
    best_defender_cost = -10**15
    best_solution = None
    iteration = 0
    
    while time.time() - start_time < time_limit:
        time_remaining = time_limit - (time.time() - start_time)
        m.setParam('TimeLimit', max(0, time_remaining))
        m.optimize()
        
        if m.SolCount == 0:
            break
        
        # Extract interdiction pattern
        x1_vals = {}
        for j1 in range(n1):
            for k in range(K):
                if x1[j1, k].X > 0.5:
                    x1_vals[j1] = k
                    break
        
        x2_vals = {}
        for j2 in range(n2):
            for k in range(K):
                if x2[j2, k].X > 0.5:
                    x2_vals[j2] = k
                    break
        
        # Compute capacities
        cap1_level1 = [
            level1_facilities["capacity_type1"][j1] * 
            (1 - interdiction_parameters["capacity_reduction_level1_d1"].get(str(x1_vals[j1]), 0.0))
            for j1 in range(n1)
        ]
        
        cap1_level2 = [
            level2_facilities["capacity_type1"][j2] * 
            (1 - interdiction_parameters["capacity_reduction_level2_d2"].get(str(x2_vals[j2]), 0.0))
            for j2 in range(n2)
        ]
        
        cap2_level2 = [
            level2_facilities["capacity_type2"][j2] * 
            (1 - interdiction_parameters["capacity_reduction_level2_d2"].get(str(x2_vals[j2]), 0.0))
            for j2 in range(n2)
        ]
        
        # Solve defender's problem
        d_model = gp.Model()
        d_model.setParam('OutputFlag', 0)
        
        # Add variables
        y = {}
        for i in range(n_customers):
            for j1 in range(n1):
                y[i, j1] = d_model.addVar(name=f"y_{i}_{j1}")
        
        z = {}
        for i in range(n_customers):
            for j2 in range(n2):
                z[i, j2] = d_model.addVar(name=f"z_{i}_{j2}")
        
        w = {}
        for i in range(n_customers):
            for j2 in range(n2):
                w[i, j2] = d_model.addVar(name=f"w_{i}_{j2}")
        
        o1 = [d_model.addVar(name=f"o1_{i}") for i in range(n_customers)]
        o2 = [d_model.addVar(name=f"o2_{i}") for i in range(n_customers)]
        
        r1 = {}
        for j1 in range(n1):
            for j2 in range(n2):
                r1[j1, j2] = d_model.addVar(name=f"r1_{j1}_{j2}")
        
        r2 = {}
        for j2 in range(n2):
            for j2p in range(n2):
                r2[j2, j2p] = d_model.addVar(name=f"r2_{j2}_{j2p}")
        
        o_r1 = [d_model.addVar(name=f"o_r1_{j1}") for j1 in range(n1)]
        o_r2 = [d_model.addVar(name=f"o_r2_{j2}") for j2 in range(n2)]
        
        # Demand constraints
        for i in range(n_customers):
            d_i = customers["demand"][i]
            d_model.addConstr(
                gp.quicksum(y[i, j1] for j1 in range(n1)) +
                gp.quicksum(z[i, j2] for j2 in range(n2)) +
                o1[i] == theta * d_i,
                name=f"type1_demand_{i}"
            )
            d_model.addConstr(
                gp.quicksum(w[i, j2] for j2 in range(n2)) +
                o2[i] == (1 - theta) * d_i,
                name=f"type2_demand_{i}"
            )
        
        # Level I capacity constraints
        cap_constr_level1 = []
        for j1 in range(n1):
            constr = d_model.addConstr(
                gp.quicksum(y[i, j1] for i in range(n_customers)) <= cap1_level1[j1],
                name=f"cap1_{j1}"
            )
            cap_constr_level1.append(constr)
        
        # Level II capacity constraints
        cap_constr_level2_type1 = []
        cap_constr_level2_type2 = []
        for j2 in range(n2):
            cap_constr_level2_type1.append(
                d_model.addConstr(
                    gp.quicksum(z[i, j2] for i in range(n_customers)) <= cap1_level2[j2],
                    name=f"cap2_type1_{j2}"
                )
            )
            cap_constr_level2_type2.append(
                d_model.addConstr(
                    gp.quicksum(w[i, j2] for i in range(n_customers)) +
                    gp.quicksum(r1[j1, j2] for j1 in range(n1)) +
                    gp.quicksum(r2[j2p, j2] for j2p in range(n2)) <= cap2_level2[j2],
                    name=f"cap2_type2_{j2}"
                )
            )
        
        # Referral balance constraints
        for j1 in range(n1):
            d_model.addConstr(
                gp.quicksum(r1[j1, j2] for j2 in range(n2)) + o_r1[j1] ==
                sigma * gp.quicksum(y[i, j1] for i in range(n_customers)),
                name=f"ref1_{j1}"
            )
        
        for j2 in range(n2):
            d_model.addConstr(
                gp.quicksum(r2[j2, j2p] for j2p in range(n2)) + o_r2[j2] ==
                sigma * gp.quicksum(z[i, j2] for i in range(n_customers)),
                name=f"ref2_{j2}"
            )
        
        # Objective
        transport_cost = 0
        for i in range(n_customers):
            for j1 in range(n1):
                transport_cost += data["alpha"] * y[i, j1] * distances["customer_to_level1_facility"][i][j1]
            for j2 in range(n2):
                transport_cost += data["beta"] * z[i, j2] * distances["customer_to_level2_facility"][i][j2]
                transport_cost += data["beta"] * w[i, j2] * distances["customer_to_level2_facility"][i][j2]
        
        for j1 in range(n1):
            for j2 in range(n2):
                transport_cost += data["gamma"] * r1[j1, j2] * distances["level1_to_level2_facility"][j1][j2]
        
        for j2 in range(n2):
            for j2p in range(n2):
                transport_cost += data["gamma"] * r2[j2, j2p] * dist_level2_level2[j2][j2p]
        
        outsourcing_cost = (
            (data["alpha_prime"] + sigma * data["gamma_prime"]) * gp.quicksum(o1) +
            data["beta_prime"] * gp.quicksum(o2) +
            data["gamma_prime"] * (gp.quicksum(o_r1) + gp.quicksum(o_r2))
        )
        
        d_model.setObjective(transport_cost + outsourcing_cost)
        
        # Solve
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining <= 0:
            break
        d_model.setParam('TimeLimit', max(0, time_remaining))
        d_model.optimize()
        
        if d_model.Status == GRB.OPTIMAL:
            defender_cost = d_model.ObjVal
            
            # Update best solution
            if defender_cost > best_defender_cost:
                best_defender_cost = defender_cost
                best_solution = {
                    "interdiction_levels_level1": [x1_vals[j1] for j1 in range(n1)],
                    "interdiction_levels_level2": [x2_vals[j2] for j2 in range(n2)]
                }
            
            # Get duals for capacity constraints
            mu1 = [constr.Pi for constr in cap_constr_level1]
            mu2 = [constr.Pi for constr in cap_constr_level2_type1]
            mu3 = [constr.Pi for constr in cap_constr_level2_type2]
            
            # Generate Benders cut
            coeff_expr = 0
            constant_term = defender_cost
            
            for j1 in range(n1):
                k_current = x1_vals[j1]
                reduction = interdiction_parameters["capacity_reduction_level1_d1"].get(str(k_current), 0.0)
                coeff_j1k = level1_facilities["capacity_type1"][j1] * reduction * (-mu1[j1])
                constant_term -= coeff_j1k
                for k in range(K):
                    reduction_k = interdiction_parameters["capacity_reduction_level1_d1"].get(str(k), 0.0)
                    coeff = level1_facilities["capacity_type1"][j1] * reduction_k * (-mu1[j1])
                    coeff_expr += coeff * x1[j1, k]
            
            for j2 in range(n2):
                k_current = x2_vals[j2]
                reduction = interdiction_parameters["capacity_reduction_level2_d2"].get(str(k_current), 0.0)
                coeff_j2k = (
                    level2_facilities["capacity_type1"][j2] * reduction * (-mu2[j2]) +
                    level2_facilities["capacity_type2"][j2] * reduction * (-mu3[j2])
                )
                constant_term -= coeff_j2k
                for k in range(K):
                    reduction_k = interdiction_parameters["capacity_reduction_level2_d2"].get(str(k), 0.0)
                    coeff = (
                        level2_facilities["capacity_type1"][j2] * reduction_k * (-mu2[j2]) +
                        level2_facilities["capacity_type2"][j2] * reduction_k * (-mu3[j2])
                    )
                    coeff_expr += coeff * x2[j2, k]
            
            m.addConstr(eta <= constant_term + coeff_expr)
            iteration += 1
        else:
            break
    
    # Save best solution
    if best_solution:
        with open(solution_path, 'w') as f:
            json.dump(best_solution, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_path', type=str)
    parser.add_argument('solution_path', type=str)
    parser.add_argument('time_limit', type=float)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit)