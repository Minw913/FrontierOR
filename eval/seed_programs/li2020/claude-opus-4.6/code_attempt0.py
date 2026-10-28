import argparse
import json
import math
import time
import itertools
import numpy as np
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Parse global parameters
    gp = data['global_parameters']
    theta = gp['theta']
    sigma = gp['sigma']
    alpha = gp['alpha']
    beta = gp['beta']
    gamma = gp['gamma']
    alpha_prime = gp['alpha_prime']
    beta_prime = gp['beta_prime']
    gamma_prime = gp['gamma_prime']
    
    # Parse instance parameters
    ip = data['instance_parameters']
    n1 = ip['num_level1_facilities']
    n2 = ip['num_level2_facilities']
    nc = ip['num_customers']
    K = ip['num_interdiction_levels_K']
    k_max = ip['k_max']
    budget = ip['total_interdiction_budget']
    
    # Parse interdiction parameters
    interdict = data['interdiction_parameters']
    h1 = {}
    for k, v in interdict['interdiction_cost_level1_h1'].items():
        h1[int(k)] = v
    h2 = {}
    for k, v in interdict['interdiction_cost_level2_h2'].items():
        h2[int(k)] = v
    d1 = {0: 0.0}
    for k, v in interdict['capacity_reduction_level1_d1'].items():
        d1[int(k)] = v
    d2 = {0: 0.0}
    for k, v in interdict['capacity_reduction_level2_d2'].items():
        d2[int(k)] = v
    
    # Parse customers
    cust = data['customers']
    cx = cust['x_coordinates']
    cy = cust['y_coordinates']
    demands = cust['demands']
    
    # Parse level 1 facilities
    l1 = data['level1_facilities']
    l1x = l1['x_coordinates']
    l1y = l1['y_coordinates']
    cap1 = l1['capacity_type1']
    
    # Parse level 2 facilities
    l2 = data['level2_facilities']
    l2x = l2['x_coordinates']
    l2y = l2['y_coordinates']
    cap2_t1 = l2['capacity_type1']
    cap2_t2 = l2['capacity_type2']
    
    # Parse distances
    dist = data['distances']
    dist_c_l1 = dist['customer_to_level1_facility']
    dist_c_l2 = dist['customer_to_level2_facility']
    dist_l1_l2 = dist['level1_to_level2_facility']
    
    # We also need level2-to-level2 distances for referrals from level2 facilities
    dist_l2_l2 = [[0.0]*n2 for _ in range(n2)]
    for j1 in range(n2):
        for j2 in range(n2):
            dist_l2_l2[j1][j2] = math.sqrt((l2x[j1]-l2x[j2])**2 + (l2y[j1]-l2y[j2])**2)
    
    # The intensity levels are 0, 1, ..., k_max
    intensity_levels = list(range(k_max + 1))
    
    def solve_defender(interdict_l1, interdict_l2):
        """
        Given interdiction levels for each facility, solve the defender's LP.
        interdict_l1: list of length n1, intensity level for each level 1 facility
        interdict_l2: list of length n2, intensity level for each level 2 facility
        Returns the optimal cost or None if infeasible.
        """
        import gurobipy as gp_solver
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.5:
            return None
        
        # Compute remaining capacities
        rem_cap1 = [cap1[i] * (1 - d1[interdict_l1[i]]) for i in range(n1)]
        rem_cap2_t1 = [cap2_t1[j] * (1 - d2[interdict_l2[j]]) for j in range(n2)]
        rem_cap2_t2 = [cap2_t2[j] * (1 - d2[interdict_l2[j]]) for j in range(n2)]
        
        try:
            m = gp_solver.Model()
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', max(remaining_time - 0.5, 1.0))
            m.setParam('Method', 2)  # Barrier for LP
            m.setParam('Threads', 1)
            
            # Decision variables:
            # x1[c][i]: type I demand from customer c to level 1 facility i
            x1 = {}
            for c in range(nc):
                for i in range(n1):
                    x1[c, i] = m.addVar(lb=0.0, name=f"x1_{c}_{i}")
            
            # x2[c][j]: type I demand from customer c to level 2 facility j
            x2 = {}
            for c in range(nc):
                for j in range(n2):
                    x2[c, j] = m.addVar(lb=0.0, name=f"x2_{c}_{j}")
            
            # y[c][j]: type II demand from customer c to level 2 facility j
            y = {}
            for c in range(nc):
                for j in range(n2):
                    y[c, j] = m.addVar(lb=0.0, name=f"y_{c}_{j}")
            
            # r1[i][j]: referral from level 1 facility i to level 2 facility j
            r1 = {}
            for i in range(n1):
                for j in range(n2):
                    r1[i, j] = m.addVar(lb=0.0, name=f"r1_{i}_{j}")
            
            # r2[j1][j2]: referral from level 2 facility j1 to level 2 facility j2
            r2 = {}
            for j1 in range(n2):
                for j2 in range(n2):
                    r2[j1, j2] = m.addVar(lb=0.0, name=f"r2_{j1}_{j2}")
            
            # Outsourcing variables
            # os1[c]: outsourced type I demand for customer c
            os1 = {}
            for c in range(nc):
                os1[c] = m.addVar(lb=0.0, name=f"os1_{c}")
            
            # os2[c]: outsourced type II demand for customer c
            os2 = {}
            for c in range(nc):
                os2[c] = m.addVar(lb=0.0, name=f"os2_{c}")
            
            # osr1[i]: outsourced referral from level 1 facility i
            osr1 = {}
            for i in range(n1):
                osr1[i] = m.addVar(lb=0.0, name=f"osr1_{i}")
            
            # osr2[j]: outsourced referral from level 2 facility j
            osr2 = {}
            for j in range(n2):
                osr2[j] = m.addVar(lb=0.0, name=f"osr2_{j}")
            
            m.update()
            
            # Constraints:
            # 1. Type I demand coverage for each customer
            for c in range(nc):
                dem1 = theta * demands[c]
                m.addConstr(
                    gp_solver.quicksum(x1[c, i] for i in range(n1)) +
                    gp_solver.quicksum(x2[c, j] for j in range(n2)) +
                    os1[c] == dem1,
                    name=f"cov1_{c}"
                )
            
            # 2. Type II demand coverage for each customer
            for c in range(nc):
                dem2 = (1 - theta) * demands[c]
                m.addConstr(
                    gp_solver.quicksum(y[c, j] for j in range(n2)) +
                    os2[c] == dem2,
                    name=f"cov2_{c}"
                )
            
            # 3. Referral balance for level 1 facilities
            for i in range(n1):
                total_t1_served = gp_solver.quicksum(x1[c, i] for c in range(nc))
                m.addConstr(
                    sigma * total_t1_served ==
                    gp_solver.quicksum(r1[i, j] for j in range(n2)) + osr1[i],
                    name=f"ref1_{i}"
                )
            
            # 4. Referral balance for level 2 facilities
            for j1 in range(n2):
                total_t1_served = gp_solver.quicksum(x2[c, j1] for c in range(nc))
                m.addConstr(
                    sigma * total_t1_served ==
                    gp_solver.quicksum(r2[j1, j2] for j2 in range(n2)) + osr2[j1],
                    name=f"ref2_{j1}"
                )
            
            # 5. Capacity constraints for level 1 facilities (type I)
            for i in range(n1):
                m.addConstr(
                    gp_solver.quicksum(x1[c, i] for c in range(nc)) <= rem_cap1[i],
                    name=f"cap1_{i}"
                )
            
            # 6. Capacity constraints for level 2 facilities (type I)
            for j in range(n2):
                m.addConstr(
                    gp_solver.quicksum(x2[c, j] for c in range(nc)) <= rem_cap2_t1[j],
                    name=f"cap2t1_{j}"
                )
            
            # 7. Capacity constraints for level 2 facilities (type II + referral)
            for j in range(n2):
                m.addConstr(
                    gp_solver.quicksum(y[c, j] for c in range(nc)) +
                    gp_solver.quicksum(r1[i, j] for i in range(n1)) +
                    gp_solver.quicksum(r2[j1, j] for j1 in range(n2)) <= rem_cap2_t2[j],
                    name=f"cap2t2_{j}"
                )
            
            # Objective: minimize total cost
            obj = gp_solver.LinExpr()
            
            # Type I demand from customer to level 1: alpha * distance * demand
            for c in range(nc):
                for i in range(n1):
                    obj += alpha * dist_c_l1[c][i] * x1[c, i]
            
            # Type I demand from customer to level 2: beta * distance * demand
            for c in range(nc):
                for j in range(n2):
                    obj += beta * dist_c_l2[c][j] * x2[c, j]
            
            # Type II demand from customer to level 2: beta * distance * demand
            for c in range(nc):
                for j in range(n2):
                    obj += beta * dist_c_l2[c][j] * y[c, j]
            
            # Referral from level 1 to level 2: gamma * distance
            for i in range(n1):
                for j in range(n2):
                    obj += gamma * dist_l1_l2[i][j] * r1[i, j]
            
            # Referral from level 2 to level 2: gamma * distance
            for j1 in range(n2):
                for j2 in range(n2):
                    obj += gamma * dist_l2_l2[j1][j2] * r2[j1, j2]
            
            # Outsourcing costs
            for c in range(nc):
                obj += alpha_prime * os1[c]
                obj += beta_prime * os2[c]
                # Outsourced type I also generates referral outsourcing
                obj += sigma * gamma_prime * os1[c]
            
            for i in range(n1):
                obj += gamma_prime * osr1[i]
            
            for j in range(n2):
                obj += gamma_prime * osr2[j]
            
            m.setObjective(obj, gp_solver.GRB.MINIMIZE)
            m.optimize()
            
            if m.status == gp_solver.GRB.OPTIMAL or m.status == gp_solver.GRB.SUBOPTIMAL:
                return m.ObjVal
            else:
                return None
                
        except Exception as e:
            return None
    
    def solve_bilevel_enumeration():
        """
        For small instances, enumerate all feasible interdiction patterns.
        """
        best_obj = -1e30
        best_pattern_l1 = [0] * n1
        best_pattern_l2 = [0] * n2
        
        # Generate all possible intensity assignments
        # For each facility, intensity in {0, 1, ..., k_max}
        # Subject to budget constraint
        
        # Use recursive enumeration with pruning
        pattern_l1 = [0] * n1
        pattern_l2 = [0] * n2
        
        def get_cost(pattern_l1, pattern_l2):
            total = 0.0
            for i in range(n1):
                if pattern_l1[i] > 0:
                    total += h1[pattern_l1[i]]
            for j in range(n2):
                if pattern_l2[j] > 0:
                    total += h2[pattern_l2[j]]
            return total
        
        # For very small instances, just enumerate
        total_facilities = n1 + n2
        total_combos = (k_max + 1) ** total_facilities
        
        if total_combos > 100000:
            return None  # Too many to enumerate
        
        count = 0
        for combo in itertools.product(intensity_levels, repeat=total_facilities):
            if time.time() - start_time > time_limit - 2:
                break
            
            pl1 = list(combo[:n1])
            pl2 = list(combo[n1:])
            
            cost = 0.0
            for i in range(n1):
                if pl1[i] > 0:
                    cost += h1[pl1[i]]
            for j in range(n2):
                if pl2[j] > 0:
                    cost += h2[pl2[j]]
            
            if cost > budget + 1e-9:
                continue
            
            obj = solve_defender(pl1, pl2)
            if obj is not None and obj > best_obj:
                best_obj = obj
                best_pattern_l1 = pl1[:]
                best_pattern_l2 = pl2[:]
                if logger:
                    logger.log(best_obj)
            
            count += 1
        
        return best_obj, best_pattern_l1, best_pattern_l2
    
    def solve_bilevel_mip():
        """
        Solve using MIP formulation with big-M or direct bilevel approach.
        We'll use an iterative approach: solve the MIP master problem for interdiction,
        then solve defender LP for each candidate.
        
        Actually, let's use a direct MIP formulation of the bilevel problem using
        LP duality for the inner problem.
        """
        import gurobipy as gp_solver
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 2:
            return None
        
        try:
            m = gp_solver.Model()
            m.setParam('OutputFlag', 1)
            m.setParam('TimeLimit', max(remaining_time - 2, 1.0))
            m.setParam('Threads', 1)
            m.setParam('MIPGap', 0.01)
            
            # Interdiction binary variables
            # z1[i][k] = 1 if level 1 facility i is attacked at intensity k
            z1 = {}
            for i in range(n1):
                for k in intensity_levels:
                    z1[i, k] = m.addVar(vtype=gp_solver.GRB.BINARY, name=f"z1_{i}_{k}")
            
            z2 = {}
            for j in range(n2):
                for k in intensity_levels:
                    z2[j, k] = m.addVar(vtype=gp_solver.GRB.BINARY, name=f"z2_{j}_{k}")
            
            # Exactly one intensity per facility
            for i in range(n1):
                m.addConstr(gp_solver.quicksum(z1[i, k] for k in intensity_levels) == 1)
            for j in range(n2):
                m.addConstr(gp_solver.quicksum(z2[j, k] for k in intensity_levels) == 1)
            
            # Budget constraint
            budget_expr = gp_solver.LinExpr()
            for i in range(n1):
                for k in intensity_levels:
                    if k > 0:
                        budget_expr += h1[k] * z1[i, k]
            for j in range(n2):
                for k in intensity_levels:
                    if k > 0:
                        budget_expr += h2[k] * z2[j, k]
            m.addConstr(budget_expr <= budget)
            
            # Now the defender's problem (inner LP) - we take its dual
            # The inner LP is a min LP, so its dual is a max LP.
            # The interdictor wants to maximize the defender's cost.
            # max-min = max of dual of inner LP
            
            # Let's define the inner LP precisely:
            # Primal variables:
            #   x1[c,i] >= 0: type I from customer c to level 1 facility i
            #   x2[c,j] >= 0: type I from customer c to level 2 facility j
            #   y[c,j] >= 0: type II from customer c to level 2 facility j
            #   r1[i,j] >= 0: referral from level 1 facility i to level 2 facility j
            #   r2[j1,j2] >= 0: referral from level 2 facility j1 to level 2 facility j2
            #   os1[c] >= 0: outsourced type I for customer c
            #   os2[c] >= 0: outsourced type II for customer c
            #   osr1[i] >= 0: outsourced referral from level 1 facility i
            #   osr2[j] >= 0: outsourced referral from level 2 facility j
            
            # Primal constraints (all equalities):
            #   (A) For each c: sum_i x1[c,i] + sum_j x2[c,j] + os1[c] = theta*d[c]  [dual: lambda1[c], free]
            #   (B) For each c: sum_j y[c,j] + os2[c] = (1-theta)*d[c]  [dual: lambda2[c], free]
            #   (C) For each i: sigma * sum_c x1[c,i] - sum_j r1[i,j] - osr1[i] = 0  [dual: mu1[i], free]
            #   (D) For each j: sigma * sum_c x2[c,j] - sum_j2 r2[j,j2] - osr2[j] = 0  [dual: mu2[j], free]
            
            # Primal constraints (inequalities):
            #   (E) For each i: sum_c x1[c,i] <= rem_cap1[i]  [dual: pi1[i] >= 0]
            #   (F) For each j: sum_c x2[c,j] <= rem_cap2_t1[j]  [dual: pi2[j] >= 0]
            #   (G) For each j: sum_c y[c,j] + sum_i r1[i,j] + sum_j1 r2[j1,j] <= rem_cap2_t2[j]  [dual: pi3[j] >= 0]
            
            # rem_cap1[i] = cap1[i] * (1 - sum_k d1[k]*z1[i,k])
            # rem_cap2_t1[j] = cap2_t1[j] * (1 - sum_k d2[k]*z2[j,k])
            # rem_cap2_t2[j] = cap2_t2[j] * (1 - sum_k d2[k]*z2[j,k])
            
            # Dual of the inner LP (maximize):
            # max sum_c lambda1[c]*theta*d[c] + sum_c lambda2[c]*(1-theta)*d[c] 
            #     - sum_i pi1[i]*rem_cap1[i] - sum_j pi2[j]*rem_cap2_t1[j] - sum_j pi3[j]*rem_cap2_t2[j]
            # s.t.
            #   For x1[c,i]: lambda1[c] + sigma*mu1[i] - pi1[i] <= alpha*dist_c_l1[c][i]
            #   For x2[c,j]: lambda1[c] + sigma*mu2[j] - pi2[j] <= beta*dist_c_l2[c][j]
            #   For y[c,j]: lambda2[c] - pi3[j] <= beta*dist_c_l2[c][j]
            #   For r1[i,j]: -mu1[i] - pi3[j] <= gamma*dist_l1_l2[i][j]
            #   For r2[j1,j2]: -mu2[j1] - pi3[j2] <= gamma*dist_l2_l2[j1][j2]
            #   For os1[c]: lambda1[c] <= alpha_prime + sigma*gamma_prime
            #   For os2[c]: lambda2[c] <= beta_prime
            #   For osr1[i]: -mu1[i] <= gamma_prime
            #   For osr2[j]: -mu2[j] <= gamma_prime
            #   pi1[i], pi2[j], pi3[j] >= 0; lambda1[c], lambda2[c], mu1[i], mu2[j] free
            
            # The objective involves bilinear terms: pi1[i] * rem_cap1[i] where rem_cap1[i] depends on z1[i,k].
            # We need to linearize these.
            
            # Dual variables
            lam1 = {}
            for c in range(nc):
                lam1[c] = m.addVar(lb=-gp_solver.GRB.INFINITY, name=f"lam1_{c}")
            
            lam2 = {}
            for c in range(nc):
                lam2[c] = m.addVar(lb=-gp_solver.GRB.INFINITY, name=f"lam2_{c}")
            
            mu1 = {}
            for i in range(n1):
                mu1[i] = m.addVar(lb=-gp_solver.GRB.INFINITY, name=f"mu1_{i}")
            
            mu2 = {}
            for j in range(n2):
                mu2[j] = m.addVar(lb=-gp_solver.GRB.INFINITY, name=f"mu2_{j}")
            
            pi1 = {}
            for i in range(n1):
                pi1[i] = m.addVar(lb=0.0, name=f"pi1_{i}")
            
            pi2 = {}
            for j in range(n2):
                pi2[j] = m.addVar(lb=0.0, name=f"pi2_{j}")
            
            pi3 = {}
            for j in range(n2):
                pi3[j] = m.addVar(lb=0.0, name=f"pi3_{j}")
            
            m.update()
            
            # Dual constraints (these are linear, no issue)
            # For x1[c,i]:
            for c in range(nc):
                for i in range(n1):
                    m.addConstr(lam1[c] + sigma * mu1[i] - pi1[i] <= alpha * dist_c_l1[c][i])
            
            # For x2[c,j]:
            for c in range(nc):
                for j in range(n2):
                    m.addConstr(lam1[c] + sigma * mu2[j] - pi2[j] <= beta * dist_c_l2[c][j])
            
            # For y[c,j]:
            for c in range(nc):
                for j in range(n2):
                    m.addConstr(lam2[c] - pi3[j] <= beta * dist_c_l2[c][j])
            
            # For r1[i,j]:
            for i in range(n1):
                for j in range(n2):
                    m.addConstr(-mu1[i] - pi3[j] <= gamma * dist_l1_l2[i][j])
            
            # For r2[j1,j2]:
            for j1 in range(n2):
                for j2 in range(n2):
                    m.addConstr(-mu2[j1] - pi3[j2] <= gamma * dist_l2_l2[j1][j2])
            
            # For os1[c]:
            for c in range(nc):
                m.addConstr(lam1[c] <= alpha_prime + sigma * gamma_prime)
            
            # For os2[c]:
            for c in range(nc):
                m.addConstr(lam2[c] <= beta_prime)
            
            # For osr1[i]:
            for i in range(n1):
                m.addConstr(-mu1[i] <= gamma_prime)
            
            # For osr2[j]:
            for j in range(n2):
                m.addConstr(-mu2[j] <= gamma_prime)
            
            # Now the objective: maximize
            # sum_c lam1[c]*theta*d[c] + sum_c lam2[c]*(1-theta)*d[c]
            # - sum_i pi1[i] * cap1[i] * (1 - sum_k d1[k]*z1[i,k])
            # - sum_j pi2[j] * cap2_t1[j] * (1 - sum_k d2[k]*z2[j,k])
            # - sum_j pi3[j] * cap2_t2[j] * (1 - sum_k d2[k]*z2[j,k])
            
            # The bilinear terms are: pi1[i] * z1[i,k], pi2[j] * z2[j,k], pi3[j] * z2[j,k]
            # We need to linearize these products of a continuous and binary variable.
            
            # For pi1[i] * z1[i,k]:
            # We know pi1[i] >= 0. We need an upper bound on pi1[i].
            # From dual constraints for x1[c,i]: pi1[i] >= lam1[c] + sigma*mu1[i] - alpha*dist_c_l1[c][i]
            # lam1[c] <= alpha_prime + sigma*gamma_prime
            # mu1[i] >= -gamma_prime (from osr1 constraint)
            # So pi1[i] <= alpha_prime + sigma*gamma_prime + sigma*gamma_prime + alpha*max_dist + ...
            # Let's compute a reasonable upper bound.
            
            # Upper bound on lam1: alpha_prime + sigma*gamma_prime
            # mu1[i] range: -gamma_prime <= mu1[i] (from osr1)
            # From r1[i,j]: -mu1[i] <= gamma_prime + pi3[j], so mu1[i] >= -(gamma_prime + pi3[j])
            # Actually mu1[i] >= -gamma_prime from osr1
            # Upper bound on mu1[i]: from x1 constraint: mu1[i] <= (alpha*dist + pi1[i] - lam1[c])/sigma
            # This gets complex. Let's use a big M approach with a reasonable bound.
            
            max_dist = 0
            for c in range(nc):
                for i in range(n1):
                    max_dist = max(max_dist, dist_c_l1[c][i])
                for j in range(n2):
                    max_dist = max(max_dist, dist_c_l2[c][j])
            for i in range(n1):
                for j in range(n2):
                    max_dist = max(max_dist, dist_l1_l2[i][j])
            for j1 in range(n2):
                for j2 in range(n2):
                    max_dist = max(max_dist, dist_l2_l2[j1][j2])
            
            # Upper bound on pi values
            M_pi = alpha_prime + sigma * gamma_prime + beta_prime + gamma_prime + max(alpha, beta, gamma) * max_dist * 2
            M_pi = max(M_pi, 100)  # safety
            
            # Linearization: for each bilinear term pi * z (continuous * binary)
            # Introduce w = pi * z, with 0 <= w <= M*z, 0 <= w <= pi, w >= pi - M*(1-z)
            
            # w1[i,k] = pi1[i] * z1[i,k]
            w1 = {}
            for i in range(n1):
                for k in intensity_levels:
                    if d1[k] > 0:  # only needed when capacity is reduced
                        w1[i, k] = m.addVar(lb=0.0, name=f"w1_{i}_{k}")
            
            # w2[j,k] = pi2[j] * z2[j,k]
            w2 = {}
            for j in range(n2):
                for k in intensity_levels:
                    if d2[k] > 0:
                        w2[j, k] = m.addVar(lb=0.0, name=f"w2_{j}_{k}")
            
            # w3[j,k] = pi3[j] * z2[j,k]
            w3 = {}
            for j in range(n2):
                for k in intensity_levels:
                    if d2[k] > 0:
                        w3[j, k] = m.addVar(lb=0.0, name=f"w3_{j}_{k}")
            
            m.update()
            
            # Linearization constraints
            for i in range(n1):
                for k in intensity_levels:
                    if d1[k] > 0:
                        m.addConstr(w1[i, k] <= M_pi * z1[i, k])
                        m.addConstr(w1[i, k] <= pi1[i])
                        m.addConstr(w1[i, k] >= pi1[i] - M_pi * (1 - z1[i, k]))
            
            for j in range(n2):
                for k in intensity_levels:
                    if d2[k] > 0:
                        m.addConstr(w2[j, k] <= M_pi * z2[j, k])
                        m.addConstr(w2[j, k] <= pi2[j])
                        m.addConstr(w2[j, k] >= pi2[j] - M_pi * (1 - z2[j, k]))
                        
                        m.addConstr(w3[j, k] <= M_pi * z2[j, k])
                        m.addConstr(w3[j, k] <= pi3[j])
                        m.addConstr(w3[j, k] >= pi3[j] - M_pi * (1 - z2[j, k]))
            
            # Objective
            obj = gp_solver.LinExpr()
            
            # Fixed part
            for c in range(nc):
                obj += lam1[c] * theta * demands[c]
                obj += lam2[c] * (1 - theta) * demands[c]
            
            # -pi1[i] * cap1[i] + pi1[i] * cap1[i] * sum_k d1[k]*z1[i,k]
            # = -pi1[i]*cap1[i] + cap1[i]*sum_k d1[k] * w1[i,k]
            for i in range(n1):
                obj -= pi1[i] * cap1[i]
                for k in intensity_levels:
                    if d1[k] > 0:
                        obj += cap1[i] * d1[k] * w1[i, k]
            
            # -pi2[j] * cap2_t1[j] * (1 - sum_k d2[k]*z2[j,k])
            for j in range(n2):
                obj -= pi2[j] * cap2_t1[j]
                for k in intensity_levels:
                    if d2[k] > 0:
                        obj += cap2_t1[j] * d2[k] * w2[j, k]
            
            # -pi3[j] * cap2_t2[j] * (1 - sum_k d2[k]*z2[j,k])
            for j in range(n2):
                obj -= pi3[j] * cap2_t2[j]
                for k in intensity_levels:
                    if d2[k] > 0:
                        obj += cap2_t2[j] * d2[k] * w3[j, k]
            
            m.setObjective(obj, gp_solver.GRB.MAXIMIZE)
            
            # Add callback to log solutions
            best_sol = {'obj': -1e30, 'l1': [0]*n1, 'l2': [0]*n2}
            
            def callback(model, where):
                if where == gp_solver.GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(gp_solver.GRB.Callback.MIPSOL_OBJ)
                    if obj_val > best_sol['obj']:
                        # Extract interdiction pattern
                        pl1 = [0] * n1
                        pl2 = [0] * n2
                        for i in range(n1):
                            for k in intensity_levels:
                                if model.cbGetSolution(z1[i, k]) > 0.5:
                                    pl1[i] = k
                        for j in range(n2):
                            for k in intensity_levels:
                                if model.cbGetSolution(z2[j, k]) > 0.5:
                                    pl2[j] = k
                        best_sol['obj'] = obj_val
                        best_sol['l1'] = pl1
                        best_sol['l2'] = pl2
                        if logger:
                            logger.log(obj_val)
            
            m.optimize(callback)
            
            if m.status in [gp_solver.GRB.OPTIMAL, gp_solver.GRB.SUBOPTIMAL, gp_solver.GRB.TIME_LIMIT]:
                if m.SolCount > 0:
                    pl1 = [0] * n1
                    pl2 = [0] * n2
                    for i in range(n1):
                        for k in intensity_levels:
                            if z1[i, k].X > 0.5:
                                pl1[i] = k
                    for j in range(n2):
                        for k in intensity_levels:
                            if z2[j, k].X > 0.5:
                                pl2[j] = k
                    
                    # Verify by solving defender LP
                    actual_obj = solve_defender(pl1, pl2)
                    if actual_obj is not None:
                        return actual_obj, pl1, pl2
                    else:
                        return m.ObjVal, pl1, pl2
            
            # If MIP didn't find a solution, return best from callback
            if best_sol['obj'] > -1e29:
                actual_obj = solve_defender(best_sol['l1'], best_sol['l2'])
                if actual_obj is not None:
                    return actual_obj, best_sol['l1'], best_sol['l2']
            
            return None
            
        except Exception as e:
            print(f"MIP error: {e}")
            return None
    
    # First, compute the no-interdiction baseline
    base_pattern_l1 = [0] * n1
    base_pattern_l2 = [0] * n2
    base_obj = solve_defender(base_pattern_l1, base_pattern_l2)
    
    best_obj = base_obj if base_obj is not None else 0.0
    best_l1 = base_pattern_l1[:]
    best_l2 = base_pattern_l2[:]
    
    if logger and base_obj is not None:
        logger.log(best_obj)
    
    # Decide approach based on problem size
    total_facilities = n1 + n2
    total_combos = (k_max + 1) ** total_facilities
    
    remaining_time = time_limit - (time.time() - start_time)
    
    if total_combos <= 50000 and remaining_time > 5:
        # Enumeration approach
        result = solve_bilevel_enumeration()
        if result is not None:
            obj, pl1, pl2 = result
            if obj > best_obj:
                best_obj = obj
                best_l1 = pl1
                best_l2 = pl2
    else:
        # Use MIP approach
        if remaining_time > 5:
            result = solve_bilevel_mip()
            if result is not None:
                obj, pl1, pl2 = result
                # Verify with actual defender LP
                actual_obj = solve_defender(pl1, pl2)
                if actual_obj is not None:
                    if actual_obj > best_obj:
                        best_obj = actual_obj
                        best_l1 = pl1
                        best_l2 = pl2
                        if logger:
                            logger.log(best_obj)
                elif obj > best_obj:
                    best_obj = obj
                    best_l1 = pl1
                    best_l2 = pl2
        
        # Also try some greedy heuristic patterns if time allows
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            # Try attacking each facility individually at max intensity
            for i in range(n1):
                if time.time() - start_time > time_limit - 3:
                    break
                if k_max > 0 and h1[k_max] <= budget:
                    pl1 = [0] * n1
                    pl2 = [0] * n2
                    pl1[i] = k_max
                    obj = solve_defender(pl1, pl2)
                    if obj is not None and obj > best_obj:
                        best_obj = obj
                        best_l1 = pl1[:]
                        best_l2 = pl2[:]
                        if logger:
                            logger.log(best_obj)
            
            for j in range(n2):
                if time.time() - start_time > time_limit - 3:
                    break
                if k_max > 0 and h2[k_max] <= budget:
                    pl1 = [0] * n1
                    pl2 = [0] * n2
                    pl2[j] = k_max
                    obj = solve_defender(pl1, pl2)
                    if obj is not None and obj > best_obj:
                        best_obj = obj
                        best_l1 = pl1[:]
                        best_l2 = pl2[:]
                        if logger:
                            logger.log(best_obj)
            
            # Greedy: iteratively add attacks to the facility that increases cost most
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 10:
                cur_l1 = [0] * n1
                cur_l2 = [0] * n2
                cur_budget_used = 0.0
                
                improved = True
                while improved and time.time() - start_time < time_limit - 5:
                    improved = False
                    best_increment_obj = best_obj
                    best_fac = None
                    
                    for i in range(n1):
                        if time.time() - start_time > time_limit - 3:
                            break
                        cur_k = cur_l1[i]
                        if cur_k >= k_max:
                            continue
                        new_k = cur_k + 1
                        add_cost = h1[new_k] - (h1[cur_k] if cur_k > 0 else 0)
                        if cur_budget_used + add_cost > budget + 1e-9:
                            continue
                        pl1 = cur_l1[:]
                        pl1[i] = new_k
                        obj = solve_defender(pl1, cur_l2)
                        if obj is not None and obj > best_increment_obj:
                            best_increment_obj = obj
                            best_fac = ('l1', i, new_k, add_cost)
                    
                    for j in range(n2):
                        if time.time() - start_time > time_limit - 3:
                            break
                        cur_k = cur_l2[j]
                        if cur_k >= k_max:
                            continue
                        new_k = cur_k + 1
                        add_cost = h2[new_k] - (h2[cur_k] if cur_k > 0 else 0)
                        if cur_budget_used + add_cost > budget + 1e-9:
                            continue
                        pl2 = cur_l2[:]
                        pl2[j] = new_k
                        obj = solve_defender(cur_l1, pl2)
                        if obj is not None and obj > best_increment_obj:
                            best_increment_obj = obj
                            best_fac = ('l2', j, new_k, add_cost)
                    
                    if best_fac is not None:
                        improved = True
                        if best_fac[0] == 'l1':
                            old_k = cur_l1[best_fac[1]]
                            cur_l1[best_fac[1]] = best_fac[2]
                            cur_budget_used += best_fac[3]
                        else:
                            old_k = cur_l2[best_fac[1]]
                            cur_l2[best_fac[1]] = best_fac[2]
                            cur_budget_used += best_fac[3]
                        
                        if best_increment_obj > best_obj:
                            best_obj = best_increment_obj
                            best_l1 = cur_l1[:]
                            best_l2 = cur_l2[:]
                            if logger:
                                logger.log(best_obj)
    
    # Build solution
    solution = {
        "objective_value": best_obj,
        "interdiction_pattern": {}
    }
    
    for i in range(n1):
        solution["interdiction_pattern"][f"S1_{i}"] = best_l1[i]
    for j in range(n2):
        solution["interdiction_pattern"][f"S2_{j}"] = best_l2[j]
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)