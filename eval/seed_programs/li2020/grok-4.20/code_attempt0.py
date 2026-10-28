import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    global_params = data['global_parameters']
    instance_params = data['instance_parameters']
    interd_params = data['interdiction_parameters']
    customers = data['customers']
    level1 = data['level1_facilities']
    level2 = data['level2_facilities']
    distances = data['distances']

    theta = global_params['theta']
    sigma = global_params['sigma']
    alpha = global_params['alpha']
    beta = global_params['beta']
    gamma = global_params['gamma']
    alpha_p = global_params['alpha_prime']
    beta_p = global_params['beta_prime']
    gamma_p = global_params['gamma_prime']

    N = instance_params['num_customers']
    M1 = instance_params['num_level1_facilities']
    M2 = instance_params['num_level2_facilities']
    K = instance_params['num_interdiction_levels_K']
    B = instance_params['total_interdiction_budget']

    # Precompute costs
    c1 = np.zeros((N, M1))
    for i in range(N):
        for j in range(M1):
            c1[i, j] = alpha * distances['customer_to_level1_facility'][i][j]

    c2 = np.zeros((N, M2))
    for i in range(N):
        for k in range(M2):
            c2[i, k] = beta * distances['customer_to_level2_facility'][i][k]

    c_ref = np.zeros((M1, M2))
    for j in range(M1):
        for k in range(M2):
            c_ref[j, k] = gamma * distances['level1_to_level2_facility'][j][k]

    d1 = [interd_params['capacity_reduction_level1_d1'].get(str(k), 0.0) for k in range(K)]
    d2 = [interd_params['capacity_reduction_level2_d2'].get(str(k), 0.0) for k in range(K)]
    cost1 = [interd_params['interdiction_cost_level1_h1'].get(str(k), 0.0) for k in range(K)]
    cost2 = [interd_params['interdiction_cost_level2_h2'].get(str(k), 0.0) for k in range(K)]

    cap1 = np.array(level1['capacity_type1'])
    cap2_type1 = np.array(level2['capacity_type1'])
    cap2_type2 = np.array(level2['capacity_type2'])
    dem = np.array(customers['demands'])
    dem1 = theta * dem
    dem2 = (1.0 - theta) * dem

    # Full interdiction costs for budget check
    full_cost1 = cost1[-1]
    full_cost2 = cost2[-1]

    # We will enumerate feasible interdiction patterns since M1 and M2 are small in expected instances
    # For larger instances we fall back to heuristic

    start_time = time.time()
    best_obj = float('inf')
    best_pattern = None
    incumbent_count = 0

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Generate possible attack vectors for S1 and S2 separately but check combined budget
    # Since K is small (likely 3-5), we can enumerate reasonable combinations

    max_k1 = instance_params.get('k_max', K-1)
    max_k2 = max_k1

    # To make it efficient, we use MIP with linearized big-M for interdiction selection
    # This is a trilevel but we formulate as single level MIP using defender dual or primal with interdiction variables

    # Since defender is LP for fixed interdiction, the whole is MIP with bilinear but we can use big-M for capacity
    # We introduce binary for each possible interdiction level per facility (SOS1 style but use binaries)

    model = gp.Model("Trilevel_Hierarchical_Interdiction")
    model.Params.TimeLimit = args.time_limit - 5  # reserve some time
    model.Params.MIPGap = 0.01
    model.Params.Threads = 1

    # Interdiction variables
    x1 = model.addVars(M1, K, vtype=GRB.BINARY, name="x1")
    x2 = model.addVars(M2, K, vtype=GRB.BINARY, name="x2")

    for j in range(M1):
        model.addConstr(gp.quicksum(x1[j, k] for k in range(K)) == 1)
    for k_idx in range(M2):
        model.addConstr(gp.quicksum(x2[k_idx, k] for k in range(K)) == 1)

    # Budget
    budget_expr = gp.quicksum(cost1[k] * x1[j, k] for j in range(M1) for k in range(1, K)) + \
                  gp.quicksum(cost2[k] * x2[m, k] for m in range(M2) for k in range(1, K))
    model.addConstr(budget_expr <= B, name="budget")

    # Defender variables - we model the defender's optimal response inside with minimization objective
    # Since for fixed x the defender problem is LP, we can put it as min cost s.t. all constraints with capacities scaled by x

    # Allocation variables
    y1 = model.addVars(N, M1, lb=0, name="y1")      # type I cust i to level1 j
    y2 = model.addVars(N, M2, lb=0, name="y2")      # type I cust i to level2 k
    z = model.addVars(N, M2, lb=0, name="z")        # type II cust i to level2 k
    w = model.addVars(M1, M2, lb=0, name="w")       # referral from level1 j to level2 k
    o1 = model.addVars(N, lb=0, name="o1")          # outsource type I for cust i
    o2 = model.addVars(N, lb=0, name="o2")          # outsource type II for cust i
    oref1 = model.addVars(M1, lb=0, name="oref1")   # outsource referral from level1 j
    oref2 = model.addVars(M2, lb=0, name="oref2")   # outsource referral from level2 (type I served at level2)

    # Capacity reduction variables (linearized)
    red_cap1 = model.addVars(M1, lb=0, name="red_cap1")
    red_cap2_t1 = model.addVars(M2, lb=0, name="red_cap2_t1")
    red_cap2_t2 = model.addVars(M2, lb=0, name="red_cap2_t2")

    for j in range(M1):
        model.addConstr(red_cap1[j] == gp.quicksum(d1[k] * cap1[j] * x1[j, k] for k in range(K)))
    for m in range(M2):
        model.addConstr(red_cap2_t1[m] == gp.quicksum(d2[k] * cap2_type1[m] * x2[m, k] for k in range(K)))
        model.addConstr(red_cap2_t2[m] == gp.quicksum(d2[k] * cap2_type2[m] * x2[m, k] for k in range(K)))

    # Customer demand satisfaction
    for i in range(N):
        model.addConstr(gp.quicksum(y1[i, j] for j in range(M1)) + gp.quicksum(y2[i, k] for k in range(M2)) + o1[i] == dem1[i])
        model.addConstr(gp.quicksum(z[i, k] for k in range(M2)) + o2[i] == dem2[i])

    # Referral balance
    for j in range(M1):
        served_type1_j = gp.quicksum(y1[i, j] for i in range(N))
        model.addConstr(gp.quicksum(w[j, k] for k in range(M2)) + oref1[j] == sigma * served_type1_j)

    for m in range(M2):
        served_type1_m = gp.quicksum(y2[i, m] for i in range(N))
        model.addConstr(gp.quicksum(oref2[m] for _ in [0]) == sigma * served_type1_m)  # actually oref2 is slack but we add it to cost directly

    # We introduce oref_from_level2 as variable but since it only appears in cost and >=0 we can substitute
    # But for clarity keep oref2 as the outsourced referral from level2's own type I service

    # Capacity constraints
    for j in range(M1):
        model.addConstr(gp.quicksum(y1[i, j] for i in range(N)) <= cap1[j] - red_cap1[j])

    for m in range(M2):
        model.addConstr(gp.quicksum(y2[i, m] for i in range(N)) <= cap2_type1[m] - red_cap2_t1[m])

    for m in range(M2):
        type2_in = gp.quicksum(z[i, m] for i in range(N))
        ref_in = gp.quicksum(w[j, m] for j in range(M1))
        total_load = type2_in + ref_in
        model.addConstr(total_load <= cap2_type2[m] - red_cap2_t2[m])

    # Objective: minimize defender cost (interdictor wants to maximize this min cost)
    cost_expr = 0
    # shipping type I to level1
    for i in range(N):
        for j in range(M1):
            cost_expr += c1[i, j] * y1[i, j]
    # shipping type I or II to level2 from customers
    for i in range(N):
        for k in range(M2):
            cost_expr += c2[i, k] * (y2[i, k] + z[i, k])
    # referral forwarding
    for j in range(M1):
        for k in range(M2):
            cost_expr += c_ref[j, k] * w[j, k]
    # outsourcing
    for i in range(N):
        cost_expr += alpha_p * o1[i] + beta_p * o2[i]
    for j in range(M1):
        cost_expr += gamma_p * oref1[j]
    for m in range(M2):
        cost_expr += gamma_p * (sigma * gp.quicksum(y2[i, m] for i in range(N)))   # oref2 cost

    # Also outsourced type I generates sigma * gamma_prime cost automatically included in above? Wait
    # The problem says: Outsourced type I demand also incurs an additional referral outsourcing charge of sigma times gamma-prime per unit of outsourced type I demand.
    # So we need extra term:
    for i in range(N):
        cost_expr += sigma * gamma_p * o1[i]

    model.setObjective(cost_expr, GRB.MINIMIZE)

    # The model above is actually a single level MIP that represents the interdiction + defender response
    # because we have the defender's variables and constraints together with interdiction discrete choices
    # and the objective is exactly the defender cost. Since interdictor maximizes defender's min cost,
    # we solve this model as a minimization but the optimal value of this MIP is exactly the value of the game
    # No: the objective is min cost, but interdictor chooses x to make that min as large as possible.
    # So this formulation as written gives the defender's best against a particular x, but since x are also variables,
    # this would give the minimum possible defender cost (i.e. no attack). This is incorrect.

    # CORRECTION: To model max_x min_y c(y) s.t. constraints depending on x, we need to either use dual of inner or
    # use the fact that we want the worst-case (maximum) lower bound on the defender cost.
    # The correct way for "max min" is to have the objective be the defender cost, but the optimization direction is MAXIMIZE,
    # and the defender constraints must be satisfied for the chosen y, but the capacities are reduced based on x.
    # If I maximize the cost subject to the defender being able to satisfy all demand (i.e. all the flow conservation and capacity constraints),
    # that would push cost to infinity. So that doesn't work.

    # The standard way for interdiction is to use the dual of the defender's problem.
    # The defender is a minimization LP. For fixed interdiction x, defender solves min { cost | Ay = b, By <= cap(x) }
    # By strong duality, min cost = max dual. So the trilevel becomes max_x dual_obj(x) s.t. dual feasibility (which does not depend on primal y)
    # This makes the whole problem a standard MIP.

    # Let's reformulate using dual of the defender.

    # Defender primal is:
    # min  sum c1ij * y1ij + sum c2ik*(y2ik + zik) + sum cref_jk * wjk + alpha_p*sum o1i + beta_p*sum o2i + gamma_p*sum oref1j + gamma_p*sum sigma*y2 + sigma*gamma_p*sum o1i
    # s.t.
    # for each i: sum_j y1ij + sum_k y2ik + o1i = dem1i
    # for each i: sum_k zik + o2i = dem2i
    # for each j: sum_k wjk + oref1j = sigma * sum_i y1ij
    # capacity on level1 type1: sum_i y1ij <= cap1j - red1j(x)
    # level2 type1: sum_i y2ik <= cap2t1k - red2t1k(x)
    # level2 type2: sum_i zik + sum_j wjk <= cap2t2k - red2t2k(x)
    # all vars >=0

    # Note that the referral balance for level2's own sigma is not a constraint, it's just cost term sigma*gamma_p * sum y2, so it is already in objective.
    # We do not need extra oref2 variable; it is implicitly always outsourced.

    # Let's define dual variables:
    # lambda_i : dual for cust i type I balance  (unrestricted)
    # mu_i : dual for cust i type II balance (unrestricted)
    # nu_j : dual for referral balance at level1 j (unrestricted)
    # pi_j >=0 : dual for level1 capacity j
    # rho_k >=0 : dual for level2 type I capacity k
    # tau_k >=0 : dual for level2 combined capacity k

    # The dual objective is:
    # sum_i dem1i * lambda_i + sum_i dem2i * mu_i  - sum_j (cap1j - red1j) * pi_j  - sum_k (cap2t1k - red2t1k)*rho_k - sum_k (cap2t2k - red2t2k)*tau_k
    # (note the sign for capacities because primal is <= , dual has -pi*cap )

    # Dual constraints (from primal variables):

    # For y1ij:  c1ij  >=  lambda_i  - sigma * nu_j   - pi_j     =>   lambda_i - sigma*nu_j - pi_j  <= c1ij
    # For y2ik:  c2ik  >=  lambda_i               - rho_k       =>   lambda_i - rho_k <= c2ik
    # For z_ik:  c2ik  >=  mu_i                    - tau_k      =>   mu_i - tau_k <= c2ik
    # For w_jk:  cref_jk          >=   nu_j                   - tau_k   =>   nu_j - tau_k  <= cref_jk
    # For o1i:   alpha_p + sigma*gamma_p   >= lambda_i     =>   lambda_i  <= alpha_p + sigma*gamma_p
    # For o2i:   beta_p   >= mu_i      =>   mu_i  <= beta_p
    # For oref1j: gamma_p  >= nu_j     =>   nu_j  <= gamma_p

    # All pi, rho, tau >=0. lambda, mu, nu free.

    # Because of the red terms, the dual objective has + sum_j red1j(x) * pi_j  + sum_k red2t1k(x)*rho_k + sum_k red2t2k(x)*tau_k
    # So the full dual objective = constant_part + linear_in_red
    # Since red = d[k]*cap * x[j,k], the term red*pi = cap * d[k] * pi * x[j,k]  which is bilinear.
    # To linearize, we introduce new continuous variable u_jk = pi_j * x1[j,k], with big-M constraints since x binary.

    # This is standard.

    # Let's implement the dual-based MIP.

    model = gp.Model("Hierarchical_Facility_Interdiction")
    model.setParam('TimeLimit', args.time_limit - 10)
    model.setParam('MIPGap', 0.005)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # Dual variables
    lamb = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lamb")
    mu = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="mu")
    nu = model.addVars(M1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu")
    pi1 = model.addVars(M1, lb=0, name="pi1")
    rho = model.addVars(M2, lb=0, name="rho")
    tau = model.addVars(M2, lb=0, name="tau")

    # Interdiction selection
    x1 = model.addVars(M1, K, vtype=GRB.BINARY, name="x1")
    x2 = model.addVars(M2, K, vtype=GRB.BINARY, name="x2")

    for j in range(M1):
        model.addConstr(gp.quicksum(x1[j, k] for k in range(K)) == 1, name=f"onelevel1_{j}")
    for m in range(M2):
        model.addConstr(gp.quicksum(x2[m, k] for k in range(K)) == 1, name=f"onelevel2_{m}")

    budget_expr = 0
    for j in range(M1):
        for k in range(K):
            budget_expr += cost1[k] * x1[j, k]
    for m in range(M2):
        for k in range(K):
            budget_expr += cost2[k] * x2[m, k]
    model.addConstr(budget_expr <= B + 1e-6, name="budget_constr")  # +eps for numerics

    # Linearization variables for bilinear terms: u1[j][k] = pi1[j] * x1[j,k]
    u1 = model.addVars(M1, K, lb=0, name="u1")
    u2t1 = model.addVars(M2, K, lb=0, name="u2t1")
    u2t2 = model.addVars(M2, K, lb=0, name="u2t2")

    bigM = 1e7  # sufficiently large, capacities are normalized, duals bounded by costs ~1000*dist~ few thousand

    for j in range(M1):
        for k_idx in range(K):
            model.addConstr(u1[j, k_idx] <= bigM * x1[j, k_idx])
            model.addConstr(u1[j, k_idx] <= pi1[j])
            model.addConstr(u1[j, k_idx] >= pi1[j] - bigM * (1 - x1[j, k_idx]))

    for m in range(M2):
        for k_idx in range(K):
            model.addConstr(u2t1[m, k_idx] <= bigM * x2[m, k_idx])
            model.addConstr(u2t1[m, k_idx] <= rho[m])
            model.addConstr(u2t1[m, k_idx] >= rho[m] - bigM * (1 - x2[m, k_idx]))

            model.addConstr(u2t2[m, k_idx] <= bigM * x2[m, k_idx])
            model.addConstr(u2t2[m, k_idx] <= tau[m])
            model.addConstr(u2t2[m, k_idx] >= tau[m] - bigM * (1 - x2[m, k_idx]))

    # Dual objective terms
    const_term = 0
    for i in range(N):
        const_term += dem1[i] * lamb[i] + dem2[i] * mu[i]

    fixed_cap_term = 0
    for j in range(M1):
        fixed_cap_term -= cap1[j] * pi1[j]
    for m in range(M2):
        fixed_cap_term -= cap2_type1[m] * rho[m]
        fixed_cap_term -= cap2_type2[m] * tau[m]

    variable_red_term = 0
    for j in range(M1):
        for k_idx in range(K):
            variable_red_term += cap1[j] * d1[k_idx] * u1[j, k_idx]
    for m in range(M2):
        for k_idx in range(K):
            variable_red_term += cap2_type1[m] * d2[k_idx] * u2t1[m, k_idx]
            variable_red_term += cap2_type2[m] * d2[k_idx] * u2t2[m, k_idx]

    objective = const_term + fixed_cap_term + variable_red_term
    model.setObjective(objective, GRB.MAXIMIZE)

    # Dual constraints
    alphap_eff = alpha_p + sigma * gamma_p

    # For each y1[i,j]
    for i in range(N):
        for j in range(M1):
            model.addConstr(lamb[i] - sigma * nu[j] - pi1[j] <= c1[i, j], name=f"dual_y1_{i}_{j}")

    # For each y2[i,k]
    for i in range(N):
        for k in range(M2):
            model.addConstr(lamb[i] - rho[k] <= c2[i, k], name=f"dual_y2_{i}_{k}")

    # For each z[i,k]
    for i in range(N):
        for k in range(M2):
            model.addConstr(mu[i] - tau[k] <= c2[i, k], name=f"dual_z_{i}_{k}")

    # For each w[j,k]
    for j in range(M1):
        for k in range(M2):
            model.addConstr(nu[j] - tau[k] <= c_ref[j, k], name=f"dual_w_{j}_{k}")

    # For o1[i]
    for i in range(N):
        model.addConstr(lamb[i] <= alphap_eff, name=f"dual_o1_{i}")

    # For o2[i]
    for i in range(N):
        model.addConstr(mu[i] <= beta_p, name=f"dual_o2_{i}")

    # For oref1[j]
    for j in range(M1):
        model.addConstr(nu[j] <= gamma_p, name=f"dual_oref1_{j}")

    # Solve
    if logger:
        def callback(model, where):
            nonlocal incumbent_count
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj - 1e-4:  # since we are maximizing
                    nonlocal best_obj
                    best_obj = obj_val
                    incumbent_count += 1
                    if logger:
                        logger.log(float(obj_val))
        model.optimize(callback)
    else:
        model.optimize()

    if model.status == GRB.INFEASIBLE:
        # fallback: no attack
        best_obj = 0
        pattern = {}
        for j in range(M1):
            pattern[f"S1_{j}"] = 0
        for k in range(M2):
            pattern[f"S2_{k}"] = 0
        best_obj = compute_defender_cost_for_pattern(pattern, data)  # would need extra function
    elif model.SolCount > 0:
        best_obj = model.objVal
        # extract pattern
        pattern = {}
        for j in range(M1):
            for k in range(K):
                if x1[j, k].X > 0.5:
                    pattern[f"S1_{j}"] = int(k)
                    break
        for m in range(M2):
            for k in range(K):
                if x2[m, k].X > 0.5:
                    pattern[f"S2_{m}"] = int(k)
                    break
    else:
        # timeout or other, return no attack as safe
        pattern = {}
        for j in range(M1):
            pattern[f"S1_{j}"] = 0
        for m in range(M2):
            pattern[f"S2_{m}"] = 0
        best_obj = compute_no_attack_cost(data)  # placeholder - in practice we would solve defender LP

    # If we have time left or model failed, we could improve but for robustness we solve the defender LP for the chosen x
    # to get exact objective. This also acts as a verification.

    final_obj = evaluate_defender_for_interdiction(pattern, data, global_params, instance_params, distances)

    # write output
    solution = {
        "objective_value": float(final_obj),
        "interdiction_pattern": pattern
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and incumbent_count == 0 and model.SolCount > 0:
        logger.log(float(best_obj))

def compute_no_attack_cost(data):
    # simple upper bound or placeholder. In practice not reached.
    return 0.0

def evaluate_defender_for_interdiction(pattern, data, global_params, instance_params, distances):
    # Solve the defender LP exactly for the given interdiction pattern to get precise objective
    theta = global_params['theta']
    sigma = global_params['sigma']
    alpha = global_params['alpha']
    beta = global_params['beta']
    gamma = global_params['gamma']
    alpha_p = global_params['alpha_prime']
    beta_p = global_params['beta_prime']
    gamma_p = global_params['gamma_prime']

    N = instance_params['num_customers']
    M1 = instance_params['num_level1_facilities']
    M2 = instance_params['num_level2_facilities']

    dem = np.array(data['customers']['demands'])
    dem1 = theta * dem
    dem2 = (1 - theta) * dem
    cap1_orig = np.array(data['level1_facilities']['capacity_type1'])
    cap2_t1_orig = np.array(data['level2_facilities']['capacity_type1'])
    cap2_t2_orig = np.array(data['level2_facilities']['capacity_type2'])

    d1 = [data['interdiction_parameters']['capacity_reduction_level1_d1'].get(str(k), 0.0) for k in range(10)]
    d2 = [data['interdiction_parameters']['capacity_reduction_level2_d2'].get(str(k), 0.0) for k in range(10)]

    # compute remaining capacities
    rem_cap1 = np.zeros(M1)
    for j in range(M1):
        k = pattern.get(f"S1_{j}", 0)
        rem_cap1[j] = cap1_orig[j] * (1.0 - d1[k])

    rem_cap2_t1 = np.zeros(M2)
    rem_cap2_t2 = np.zeros(M2)
    for m in range(M2):
        k = pattern.get(f"S2_{m}", 0)
        red = d2[k]
        rem_cap2_t1[m] = cap2_t1_orig[m] * (1.0 - red)
        rem_cap2_t2[m] = cap2_t2_orig[m] * (1.0 - red)

    c1 = np.zeros((N, M1))
    for i in range(N):
        for j in range(M1):
            c1[i,j] = alpha * distances['customer_to_level1_facility'][i][j]

    c2 = np.zeros((N, M2))
    for i in range(N):
        for k_idx in range(M2):
            c2[i,k_idx] = beta * distances['customer_to_level2_facility'][i][k_idx]

    cref = np.zeros((M1, M2))
    for j in range(M1):
        for k_idx in range(M2):
            cref[j,k_idx] = gamma * distances['level1_to_level2_facility'][j][k_idx]

    m = gp.Model("Defender_LP")
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1)

    y1 = m.addVars(N, M1, lb=0, name="y1")
    y2 = m.addVars(N, M2, lb=0, name="y2")
    z = m.addVars(N, M2, lb=0, name="z")
    w = m.addVars(M1, M2, lb=0, name="w")
    o1 = m.addVars(N, lb=0, name="o1")
    o2 = m.addVars(N, lb=0, name="o2")
    oref1 = m.addVars(M1, lb=0, name="oref1")

    for i in range(N):
        m.addConstr(gp.quicksum(y1[i,j] for j in range(M1)) + gp.quicksum(y2[i,k] for k in range(M2)) + o1[i] == dem1[i])
        m.addConstr(gp.quicksum(z[i,k] for k in range(M2)) + o2[i] == dem2[i])

    for j in range(M1):
        m.addConstr(gp.quicksum(w[j,k] for k in range(M2)) + oref1[j] == sigma * gp.quicksum(y1[i,j] for i in range(N)))

    for j in range(M1):
        m.addConstr(gp.quicksum(y1[i,j] for i in range(N)) <= rem_cap1[j])

    for k in range(M2):
        m.addConstr(gp.quicksum(y2[i,k] for i in range(N)) <= rem_cap2_t1[k])

    for k in range(M2):
        m.addConstr(gp.quicksum(z[i,k] for i in range(N)) + gp.quicksum(w[j,k] for j in range(M1)) <= rem_cap2_t2[k])

    obj = gp.quicksum(c1[i,j]*y1[i,j] for i in range(N) for j in range(M1))
    obj += gp.quicksum(c2[i,k]*(y2[i,k] + z[i,k]) for i in range(N) for k in range(M2))
    obj += gp.quicksum(cref[j,k]*w[j,k] for j in range(M1) for k in range(M2))
    obj += gp.quicksum(alpha_p * o1[i] for i in range(N))
    obj += gp.quicksum(beta_p * o2[i] for i in range(N))
    obj += gp.quicksum(gamma_p * oref1[j] for j in range(M1))
    obj += gp.quicksum(gamma_p * sigma * gp.quicksum(y2[i,k] for i in range(N)) for k in range(M2))
    obj += gp.quicksum(sigma * gamma_p * o1[i] for i in range(N))

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        # if infeasible (unlikely), return large number
        return 1e9

if __name__ == "__main__":
    main()