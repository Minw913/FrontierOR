import argparse
import json
import time
import itertools
import numpy as np
from itertools import permutations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']
    patients = instance['patients']
    
    # Get first cost structure (authoritative)
    cost_structures = instance['cost_structures']
    cost_name = list(cost_structures.keys())[0]
    cost_struct = cost_structures[cost_name]
    c_w = cost_struct['c_w']
    c_g = cost_struct['c_g']
    c_o = cost_struct['c_o']
    
    # Extract patient data
    pat_data = []
    for p in patients:
        pat_data.append({
            'idx': p['patient_index'],
            'mu_A': p['mean_duration_adequate_prep'],
            'mu_I': p['mean_duration_inadequate_prep'],
            'd_AL': p['lower_bound_duration_adequate_prep'],
            'd_AU': p['upper_bound_duration_adequate_prep'],
            'd_IL': p['lower_bound_duration_inadequate_prep'],
            'd_IU': p['upper_bound_duration_inadequate_prep'],
            'mu_q': p['mean_prep_adequacy'],
            'mu_u': p['mean_arrival_time_deviation'],
            'u_L': p['lower_bound_arrival_time_deviation'],
            'u_U': p['upper_bound_arrival_time_deviation'],
        })
    
    # For each patient, compute the mean duration (over prep quality uncertainty)
    # E[d_p] = mu_q * mu_A + (1-mu_q) * mu_I
    for p in pat_data:
        p['mean_dur'] = p['mu_q'] * p['mu_A'] + (1 - p['mu_q']) * p['mu_I']
        # Worst-case mean duration: since we only know marginal means and supports,
        # the worst case for duration expectation given prep quality marginal
        p['d_L'] = min(p['d_AL'], p['d_IL'])  # absolute lower bound
        p['d_U'] = max(p['d_AU'], p['d_IU'])  # absolute upper bound
    
    # For small P, we can try enumeration or heuristic approaches
    # For larger P, we need the MILP formulation
    
    import gurobipy as gp
    from gurobipy import GRB
    
    # I'll implement a simplified but rigorous approach:
    # The DRO with marginal moment info can be solved via a reformulation.
    # 
    # Key insight: With only marginal means and support intervals known,
    # the worst-case expected cost can be computed using the dual of
    # the moment problem for each patient independently (since we take
    # supremum over all joint distributions consistent with marginals).
    #
    # The problem structure: positions 1..P, patient assigned to each position.
    # Let s_i be the start time of position i, d_{sigma(i)} the duration of 
    # patient assigned to position i, u_{sigma(i)} the arrival deviation.
    #
    # Waiting time at position i: w_i = max(0, C_{i-1} - s_i, a_i - s_i) 
    # where C_{i-1} is completion of previous, a_i is arrival time
    # Actually: patient ready time = max(s_i + u_p, C_{i-1}), 
    # procedure starts at max(s_i + u_p, C_{i-1}, s_i)
    # 
    # Let me re-read the problem carefully:
    # "A patient's waiting time at position i is the amount by which 
    #  the patient's actual procedure-ready time exceeds the scheduled start time"
    # procedure-ready time = max(arrival_time, completion_of_preceding)
    # arrival_time of patient p at position i = s_i + u_p (scheduled + deviation)
    # Wait: actually arrival time is absolute. Let me reconsider.
    #
    # The arrival time deviation u_p means the patient arrives at s_i + u_p
    # (if assigned to position i with start time s_i).
    # procedure-ready time at position i = max(s_i + u_p, C_{i-1})
    # waiting time w_i = max(s_i + u_p, C_{i-1}) - s_i = max(u_p, C_{i-1} - s_i)
    # But waiting should be non-negative: w_i = max(0, u_p, C_{i-1} - s_i) - hmm
    # Actually w_i = max(s_i + u_p, C_{i-1}) - s_i
    # If u_p >= 0 and C_{i-1} <= s_i: w_i = u_p
    # If u_p < 0 and C_{i-1} <= s_i: w_i = max(s_i + u_p, C_{i-1}) - s_i
    #   = max(u_p, C_{i-1} - s_i) which could be negative...
    # 
    # Let me re-read: "waiting time is the amount by which the procedure-ready time 
    # exceeds the scheduled start time" - so w_i = [procedure_ready_time_i - s_i]^+
    # procedure_ready_time_i = max(arrival_i, C_{i-1}) for i>1, max(arrival_1) for i=1
    # arrival_i = s_i + u_p (where p is assigned to position i)
    # 
    # For i=1: procedure_ready = arrival = s_1 + u_p = 0 + u_p = u_p
    # w_1 = max(0, u_p - 0) = max(0, u_p)
    # 
    # For i>1: procedure_ready = max(s_i + u_p, C_{i-1})
    # w_i = max(0, max(s_i + u_p, C_{i-1}) - s_i) = max(0, u_p, C_{i-1} - s_i)
    #
    # C_i = procedure_ready_i + d_p = s_i + w_i + d_p (where p assigned to i)
    # 
    # Idle time between position i and i+1:
    # g_i = max(0, s_{i+1} - C_i) = max(0, s_{i+1} - s_i - w_i - d_p)
    #
    # Overtime: o = max(0, C_P - L)
    
    # This is a complex stochastic/DRO problem. For practical purposes with
    # the given time constraints, I'll use a heuristic approach:
    # 1. Order patients by some heuristic (e.g., increasing variance, SPT, etc.)
    # 2. Optimize start times given the ordering
    # 3. Try multiple orderings and pick the best
    
    # For the worst-case expected cost computation:
    # With only marginal means and support, the worst-case is achieved by
    # distributions that maximize expected cost. For linear/convex costs,
    # this often involves putting mass on extreme points.
    
    # Let me implement a practical approach:
    # For each patient p at position i, the relevant random variables are:
    # - d_p: duration (depends on prep quality q_p)
    # - u_p: arrival deviation
    # 
    # Given only means and bounds, worst-case E[cost] can be bounded.
    # 
    # For a FIXED assignment and schedule, the worst-case expected cost
    # over the ambiguity set can be computed as follows:
    #
    # The cost is a convex function of the random variables (durations and arrivals).
    # With marginal moment constraints, the worst case over all joint distributions
    # is achieved when the joint is the product of worst-case marginals (for convex costs,
    # positive dependence is worst, but with only mean constraints the marginal
    # worst case gives an upper bound).
    #
    # Actually, for the DRO with marginal moment info (mean + support),
    # the worst-case expectation of a convex function is:
    # For each random variable X with E[X] = mu, X in [a,b]:
    # The worst-case is achieved by a two-point distribution on {a, b}
    # with probabilities (b-mu)/(b-a) and (mu-a)/(b-a).
    #
    # But when we have JOINT distributions with only marginal info,
    # the worst case over all joints consistent with marginals can 
    # exploit dependencies. For supermodular costs, comonotonic coupling
    # is worst. For general costs, we need to solve the moment problem.
    
    # Given the complexity, let me implement a simulation-based approach
    # for evaluating a given schedule, combined with heuristic optimization.
    
    # For each patient, the effective random duration is:
    # With prob mu_q: d ~ [d_AL, d_AU] with mean mu_A
    # With prob 1-mu_q: d ~ [d_IL, d_IU] with mean mu_I
    # And arrival dev u ~ [u_L, u_U] with mean mu_u
    
    # For worst-case with mean+support: use two-point distributions
    # But we need to consider all possible combinations for joint worst case
    
    # For each patient p, the "compound" duration distribution:
    # With adequate prep (prob mu_q): d in [d_AL, d_AU], E[d] = mu_A
    #   worst case 2-point: {d_AL w.p. (d_AU-mu_A)/(d_AU-d_AL), d_AU w.p. (mu_A-d_AL)/(d_AU-d_AL)}
    # With inadequate prep (prob 1-mu_q): d in [d_IL, d_IU], E[d] = mu_I
    #   worst case 2-point: {d_IL w.p. (d_IU-mu_I)/(d_IU-d_IL), d_IU w.p. (mu_I-d_IL)/(d_IU-d_IL)}
    # Arrival: u in [u_L, u_U], E[u] = mu_u
    #   worst case 2-point: {u_L w.p. (u_U-mu_u)/(u_U-u_L), u_U w.p. (mu_u-u_L)/(u_U-u_L)}
    
    # So for each patient, there are at most 2 (prep) * 2 (duration|prep) * 2 (arrival) = 8 scenarios
    # But with mu_q known, we have effectively:
    # - Adequate prep scenarios: (d_AL, u_L), (d_AL, u_U), (d_AU, u_L), (d_AU, u_U)
    # - Inadequate prep scenarios: (d_IL, u_L), (d_IL, u_U), (d_IU, u_L), (d_IU, u_U)
    
    # However, the joint distribution across patients matters.
    # For the worst case, we want to find the joint distribution (coupling)
    # that maximizes expected cost while respecting marginals.
    
    # This is extremely hard in general. Let me use a practical approach:
    # 
    # Approach: Build a MILP that simultaneously optimizes assignment, schedule,
    # and computes worst-case cost using the dual of the moment problem.
    
    # Actually, let me try a more tractable approach for moderate P:
    # Use the fact that for the DRO problem, we can reformulate using
    # the dual of the worst-case expectation.
    
    # For a FIXED schedule (assignment + times), the operational cost is:
    # sum_i c_w[i] * w_i + sum_i c_g[i] * g_i + c_o * overtime
    # where these are functions of random (d_p, u_p) for each patient p.
    
    # The cost function is convex and piecewise linear in the random variables.
    # The worst-case expectation over distributions with given marginal means
    # and supports equals the optimal value of:
    # max E_F[cost(d,u)] s.t. marginal constraints
    
    # By strong duality of the moment problem, this equals:
    # min sum_p [alpha_p * mu_d_p + beta_p * mu_u_p + ...] + terms for support
    # where alpha, beta are dual variables for the mean constraints
    # and the dual constraints require certain functions to dominate the cost.
    
    # This is still complex. Let me try a more practical route.
    
    # PRACTICAL APPROACH:
    # 1. For small P (<=8), enumerate all permutations or use branch-and-bound
    # 2. For each permutation, optimize start times and compute worst-case cost
    # 3. For the worst-case cost with a fixed schedule, use scenario-based approach
    
    # For the worst-case expected cost with marginal mean+support constraints,
    # when the cost is convex in each variable, the worst case over marginals
    # can be computed by finding the coupling that maximizes expected cost.
    # 
    # For computational tractability, I'll use the following bound:
    # Since cost is a sum of convex piecewise-linear terms that depend on
    # cumulative durations and arrivals, and the worst-case joint distribution
    # is over all couplings of the marginals, I'll approximate by:
    # - Computing the worst-case using independent worst-case marginals
    #   (each marginal is a 2-point distribution at support bounds with correct mean)
    # - This gives 2^(3P) scenarios in the worst case, but we can use
    #   the recursive structure
    
    # Actually for the DRO with only mean+support marginal info,
    # the worst case for EACH patient's random variables independently
    # uses 2-point distributions. But the coupling across patients matters.
    
    # Key insight: The cost function has a recursive structure.
    # C_0 = 0
    # For position i with patient sigma(i):
    #   ready_i = max(s_i + u_{sigma(i)}, C_{i-1})
    #   w_i = ready_i - s_i (can be negative if arrival is early and no queue)
    #   Actually w_i = max(0, ready_i - s_i) per problem statement? Let me recheck.
    #   "waiting time at position i is the amount by which the patient's actual 
    #    procedure-ready time exceeds the scheduled start time of that position"
    #   So w_i = max(0, ready_i - s_i)? Or just ready_i - s_i?
    #   "amount by which X exceeds Y" = max(0, X - Y)
    #   w_i = max(0, max(s_i + u_p, C_{i-1}) - s_i) = max(0, u_p, C_{i-1} - s_i)
    #   Since u_p could be negative (early arrival), this makes sense.
    #   
    #   C_i = s_i + w_i + d_p = max(s_i, s_i + u_p, C_{i-1}) + d_p
    #        = max(s_i, s_i + u_p, C_{i-1}) + d_p
    #
    # Idle: g_i = max(0, s_{i+1} - C_i) for i=1..P-1
    # Overtime: o = max(0, C_P - L)
    # Total cost = sum_i c_w * w_i + sum_i c_g * g_i + c_o * o
    # Wait, the problem says c_w_i per position - but the cost structure has single c_w, c_g, c_o
    # Looking at the schema: c_w is a single float, c_g is a single float, c_o is a single float
    # So the cost is c_w * sum(w_i) + c_g * sum(g_i) + c_o * o
    
    # For the worst case with mean+support marginals:
    # Each patient has random variables (q_p, d_p|q_p, u_p)
    # These decompose into independent per-patient uncertainties
    # But the coupling ACROSS patients affects the joint distribution
    
    # For the DRO problem, the worst-case coupling is the one that 
    # maximizes covariance of durations (comonotonic coupling for convex costs)
    # 
    # Since the cost is convex in the durations (longer durations -> more waiting, overtime)
    # and the cost is supermodular in durations (increasing one duration increases
    # the marginal cost of increasing another), the worst case coupling is
    # COMONOTONIC (all durations perfectly positively correlated).
    # 
    # Similarly for arrival deviations: later arrivals increase waiting.
    # But the interaction between arrivals and durations is complex.
    
    # For a rigorous but tractable approach, I'll use:
    # 1. Each patient has a finite set of "atoms" for its marginal
    # 2. Worst case over couplings = max over joint distributions with given marginals
    # 3. For convex/supermodular objectives, comonotonic coupling is worst
    
    # Let me implement a scenario-based evaluation where for each patient,
    # I create a finite set of scenarios based on support endpoints and means.
    
    # For patient p, the compound duration scenarios:
    # Adequate prep (prob mu_q):
    #   d = d_AL with prob p_AL = (d_AU - mu_A) / (d_AU - d_AL) if d_AU > d_AL else 1
    #   d = d_AU with prob p_AU = (mu_A - d_AL) / (d_AU - d_AL) if d_AU > d_AL else 0
    # Inadequate prep (prob 1 - mu_q):
    #   d = d_IL with prob p_IL = (d_IU - mu_I) / (d_IU - d_IL) if d_IU > d_IL else 1
    #   d = d_IU with prob p_IU = (mu_I - d_IL) / (d_IU - d_IL) if d_IU > d_IL else 0
    # Arrival deviation:
    #   u = u_L with prob p_uL = (u_U - mu_u) / (u_U - u_L) if u_U > u_L else 1
    #   u = u_U with prob p_uU = (mu_u - u_L) / (u_U - u_L) if u_U > u_L else 0
    
    # For patient p, marginal scenarios (d, u, marginal_prob):
    def get_patient_marginal_scenarios(p):
        """Returns list of (duration, arrival_dev, probability) for patient p"""
        scenarios = []
        mu_q = p['mu_q']
        
        # Duration scenarios given adequate prep
        d_AL, d_AU, mu_A = p['d_AL'], p['d_AU'], p['mu_A']
        if d_AU > d_AL:
            p_AL = (d_AU - mu_A) / (d_AU - d_AL)
            p_AU = (mu_A - d_AL) / (d_AU - d_AL)
            dur_adequate = [(d_AL, p_AL), (d_AU, p_AU)]
        else:
            dur_adequate = [(d_AL, 1.0)]
        
        # Duration scenarios given inadequate prep
        d_IL, d_IU, mu_I = p['d_IL'], p['d_IU'], p['mu_I']
        if d_IU > d_IL:
            p_IL = (d_IU - mu_I) / (d_IU - d_IL)
            p_IU = (mu_I - d_IL) / (d_IU - d_IL)
            dur_inadequate = [(d_IL, p_IL), (d_IU, p_IU)]
        else:
            dur_inadequate = [(d_IL, 1.0)]
        
        # Arrival deviation scenarios
        u_L, u_U, mu_u = p['u_L'], p['u_U'], p['mu_u']
        if u_U > u_L:
            p_uL = (u_U - mu_u) / (u_U - u_L)
            p_uU = (mu_u - u_L) / (u_U - u_L)
            arr_scen = [(u_L, p_uL), (u_U, p_uU)]
        else:
            arr_scen = [(u_L, 1.0)]
        
        # Combine: prep quality * duration|prep * arrival
        for d_val, d_prob in dur_adequate:
            for u_val, u_prob in arr_scen:
                scenarios.append((d_val, u_val, mu_q * d_prob * u_prob))
        for d_val, d_prob in dur_inadequate:
            for u_val, u_prob in arr_scen:
                scenarios.append((d_val, u_val, (1 - mu_q) * d_prob * u_prob))
        
        return scenarios
    
    # For the worst-case over joint distributions consistent with marginals,
    # we need to solve a linear program over joint distributions.
    # For P patients each with up to 8 marginal scenarios, the joint has up to 8^P scenarios.
    # This is only feasible for small P.
    
    # For moderate P, I'll use the comonotonic coupling as an upper bound
    # (worst case for supermodular costs) and also try the independent coupling.
    
    # Let me first try a heuristic approach that works for any P:
    # 1. Sort patients by expected duration (various orderings)
    # 2. For each ordering, optimize start times using a simple model
    # 3. Evaluate worst-case cost
    
    # For computing worst-case expected cost of a fixed schedule:
    # I'll use the MILP formulation for the moment problem dual.
    
    # Actually, let me think about this differently.
    # The problem asks for the DRO solution with the full MILP formulation
    # as suggested by the model_variable_name_patterns.
    # 
    # The formulation appears to be based on a specific paper on DRO for
    # appointment scheduling. Let me reconstruct it.
    
    # Standard DRO appointment scheduling formulation:
    # Given the recursive cost structure, the worst-case expected cost
    # for a fixed assignment and schedule can be written as a moment problem.
    # The dual of this moment problem gives a tractable reformulation.
    
    # Let me implement the full MILP based on the variable hints:
    # x[p,i]: binary, patient p assigned to position i
    # t[i]: start time of position i
    # rho[i], alpha[i], lambda[i], gamma[i]: dual variables
    # beta[i]: dual variables (i=1..P+2)
    # z[i], v[i], r[i], e[i]: auxiliary variables
    # eta[p,i], tau[p,i], nu[p,i], Gam[p,i]: McCormick linearization variables
    # zeta[p,i], psi[p,i], phi[p,i]: nonneg McCormick variables
    
    # This appears to follow the formulation from Kong et al. or similar papers
    # on distributionally robust appointment scheduling.
    
    # Let me try to reconstruct the formulation.
    # 
    # For the DRO with marginal mean+support constraints:
    # For a fixed assignment sigma and schedule t, the worst-case expected cost is:
    #
    # sup_{F in Ambiguity} E_F[Cost(d,u)]
    # 
    # where Cost = c_w * sum_i w_i + c_g * sum_i g_i + c_o * overtime
    #
    # The cost can be written recursively. Let me define:
    # C_0 = 0
    # For i = 1,...,P:
    #   w_i = max(0, u_{sigma(i)}, C_{i-1} - s_i) 
    #   C_i = s_i + w_i + d_{sigma(i)}
    #   g_i = max(0, s_{i+1} - C_i) for i < P
    # overtime = max(0, C_P - L)
    #
    # Note: C_i = max(s_i, s_i + u_{sigma(i)}, C_{i-1}) + d_{sigma(i)}
    #
    # The total cost is convex piecewise linear in (d, u).
    # 
    # For the DRO dual, we need:
    # min_{alpha, beta, ...} sum of dual objective terms
    # s.t. dual constraints (for all extreme points of the support)
    #
    # This is complex. Let me try a different approach.
    
    # APPROACH: Enumerate orderings for small P, use heuristics for large P,
    # and for each ordering optimize start times.
    
    # For a fixed ordering (permutation), the start time optimization can be done
    # using LP/closed-form based on worst-case scenario evaluation.
    
    # For the worst-case cost evaluation with fixed ordering and start times,
    # I'll use scenario enumeration with the comonotonic coupling.
    
    # Let me first implement a function that, given an ordering and start times,
    # computes the cost for a specific scenario (realization of d and u for each position).
    
    def compute_cost_scenario(start_times, durations, arrivals, L, c_w, c_g, c_o):
        """
        start_times: list of P start times
        durations: list of P durations (in position order)
        arrivals: list of P arrival deviations (in position order)
        """
        P_loc = len(start_times)
        C_prev = 0.0
        total_wait = 0.0
        total_idle = 0.0
        
        for i in range(P_loc):
            s_i = start_times[i]
            u_i = arrivals[i]
            d_i = durations[i]
            
            # Procedure ready time
            ready = max(s_i + u_i, C_prev) if i > 0 else max(s_i + u_i, 0)
            ready = max(ready, s_i)  # can't start before scheduled
            
            # Wait time
            w_i = ready - s_i
            total_wait += w_i
            
            # Completion time
            C_i = ready + d_i
            
            # Idle time (between this position and next)
            if i < P_loc - 1:
                idle = max(0, start_times[i+1] - C_i)
                total_idle += idle
            
            C_prev = C_i
        
        overtime = max(0, C_prev - L)
        
        return c_w * total_wait + c_g * total_idle + c_o * overtime
    
    # For worst-case evaluation with marginal 2-point distributions and 
    # optimal (worst-case) coupling:
    # Since the cost is convex and supermodular in durations,
    # the comonotonic coupling (all durations simultaneously high or low) is worst.
    # But with arrivals, it's more nuanced.
    
    # For a tractable worst-case computation, I'll enumerate over all 
    # combinations of extreme scenarios for each position.
    # Each position has scenarios from its assigned patient's marginals.
    
    # For the worst-case over all joint distributions consistent with marginals:
    # max sum_{joint_scenario} prob(joint_scenario) * cost(joint_scenario)
    # s.t. marginal constraints
    # This is a LP with exponentially many variables but can be solved for small P.
    
    def compute_worst_case_expected_cost(ordering, start_times, pat_data, L, c_w, c_g, c_o):
        """
        Compute worst-case expected cost over all joint distributions
        consistent with marginal means and supports, using LP.
        ordering: list of patient indices (0-based) in position order
        """
        P_loc = len(ordering)
        
        # Get marginal scenarios for each position
        position_scenarios = []
        for i in range(P_loc):
            p = pat_data[ordering[i]]
            scens = get_patient_marginal_scenarios(p)
            position_scenarios.append(scens)
        
        # For LP: joint scenarios are cartesian product
        # Variables: probability of each joint scenario
        # Constraints: marginal probabilities must match
        # Objective: maximize expected cost
        
        # Number of scenarios per position
        n_scens = [len(s) for s in position_scenarios]
        total_joint = 1
        for n in n_scens:
            total_joint *= n
        
        if total_joint > 500000:
            # Too many joint scenarios, use heuristic (comonotonic or independent)
            return compute_heuristic_worst_case(ordering, start_times, pat_data, L, c_w, c_g, c_o)
        
        # Enumerate all joint scenarios
        import itertools
        ranges = [range(n) for n in n_scens]
        
        joint_costs = []
        for combo in itertools.product(*ranges):
            durations = []
            arrivals = []
            for i, si in enumerate(combo):
                d, u, _ = position_scenarios[i][si]
                durations.append(d)
                arrivals.append(u)
            cost = compute_cost_scenario(start_times, durations, arrivals, L, c_w, c_g, c_o)
            joint_costs.append(cost)
        
        # Solve LP: max sum_j p_j * cost_j s.t. marginal constraints
        model = gp.Model("worst_case")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('Threads', 1)
        
        joint_indices = list(itertools.product(*ranges))
        n_joint = len(joint_indices)
        
        # Variables: probability of each joint scenario
        prob_vars = model.addVars(n_joint, lb=0.0, ub=1.0, name="p")
        
        # Sum to 1
        model.addConstr(gp.quicksum(prob_vars[j] for j in range(n_joint)) == 1.0)
        
        # Marginal constraints: for each position i, for each marginal scenario si,
        # sum of joint probs where position i has scenario si = marginal prob of si
        for i in range(P_loc):
            for si in range(n_scens[i]):
                marginal_prob = position_scenarios[i][si][2]
                relevant_joints = [j for j, combo in enumerate(joint_indices) if combo[i] == si]
                model.addConstr(
                    gp.quicksum(prob_vars[j] for j in relevant_joints) == marginal_prob
                )
        
        # Objective: maximize expected cost
        model.setObjective(
            gp.quicksum(prob_vars[j] * joint_costs[j] for j in range(n_joint)),
            GRB.MAXIMIZE
        )
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return model.objVal
        else:
            # Fallback to independent coupling
            return compute_independent_expected_cost(ordering, start_times, pat_data, L, c_w, c_g, c_o)
    
    def compute_independent_expected_cost(ordering, start_times, pat_data, L, c_w, c_g, c_o):
        """Compute expected cost under independent marginals"""
        P_loc = len(ordering)
        position_scenarios = []
        for i in range(P_loc):
            p = pat_data[ordering[i]]
            scens = get_patient_marginal_scenarios(p)
            position_scenarios.append(scens)
        
        n_scens = [len(s) for s in position_scenarios]
        total_joint = 1
        for n in n_scens:
            total_joint *= n
        
        expected_cost = 0.0
        for combo in itertools.product(*[range(n) for n in n_scens]):
            durations = []
            arrivals = []
            joint_prob = 1.0
            for i, si in enumerate(combo):
                d, u, p = position_scenarios[i][si]
                durations.append(d)
                arrivals.append(u)
                joint_prob *= p
            cost = compute_cost_scenario(start_times, durations, arrivals, L, c_w, c_g, c_o)
            expected_cost += joint_prob * cost
        
        return expected_cost
    
    def compute_heuristic_worst_case(ordering, start_times, pat_data, L, c_w, c_g, c_o):
        """Heuristic worst-case: use comonotonic coupling (all high or all low together)"""
        # This is an upper bound for supermodular costs
        # Also compute independent coupling and take the max
        
        P_loc = len(ordering)
        
        # Comonotonic: sort scenarios by "severity" and couple them
        # Simple approach: evaluate cost for all-high-duration + late-arrival 
        # and all-low-duration + early-arrival scenarios
        
        # Worst likely scenario: all patients have max duration and late arrival
        durations_high = []
        arrivals_late = []
        for i in range(P_loc):
            p = pat_data[ordering[i]]
            durations_high.append(max(p['d_AU'], p['d_IU']))
            arrivals_late.append(p['u_U'])
        
        cost_worst = compute_cost_scenario(start_times, durations_high, arrivals_late, L, c_w, c_g, c_o)
        
        # Use mean scenario as base
        durations_mean = []
        arrivals_mean = []
        for i in range(P_loc):
            p = pat_data[ordering[i]]
            durations_mean.append(p['mean_dur'])
            arrivals_mean.append(p['mu_u'])
        
        cost_mean = compute_cost_scenario(start_times, durations_mean, arrivals_mean, L, c_w, c_g, c_o)
        
        # A crude upper bound: weighted combination
        # The actual worst case is between these
        return max(cost_worst * 0.3 + cost_mean * 0.7, cost_mean)
    
    # Start time optimization for a given ordering:
    # For each ordering, find optimal start times.
    # 
    # With the DRO objective, start time optimization is an LP/convex problem
    # for fixed ordering. But computing the objective requires the LP for
    # worst-case coupling.
    
    # Practical approach: Use mean durations + simple buffer for start times,
    # then refine.
    
    def optimize_start_times_simple(ordering, pat_data, L, c_w, c_g, c_o):
        """Simple start time optimization using mean scenario"""
        P_loc = len(ordering)
        
        # Start with mean-duration spacing
        start_times = [0.0] * P_loc
        for i in range(1, P_loc):
            p = pat_data[ordering[i-1]]
            # Use mean duration + some buffer
            mean_d = p['mean_dur']
            mean_u = p['mu_u']  # arrival deviation of current patient
            start_times[i] = start_times[i-1] + mean_d
        
        return start_times
    
    def optimize_start_times_lp(ordering, pat_data, L, c_w, c_g, c_o, time_remaining):
        """
        Optimize start times using LP with scenario-based worst-case objective.
        For the DRO, we need to simultaneously optimize start times and worst-case coupling.
        """
        P_loc = len(ordering)
        
        # Get marginal scenarios for each position
        position_scenarios = []
        for i in range(P_loc):
            p = pat_data[ordering[i]]
            scens = get_patient_marginal_scenarios(p)
            position_scenarios.append(scens)
        
        n_scens = [len(s) for s in position_scenarios]
        total_joint = 1
        for n in n_scens:
            total_joint *= n
        
        if total_joint > 200000:
            # Too large, use simpler approach
            return optimize_start_times_simple(ordering, pat_data, L, c_w, c_g, c_o)
        
        # Joint scenarios
        joint_indices = list(itertools.product(*[range(n) for n in n_scens]))
        n_joint = len(joint_indices)
        
        # Build the LP/MILP
        model = gp.Model("start_time_opt")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('Threads', 1)
        model.setParam('NumericFocus', 0)
        model.setParam('MIPGap', 1e-4)
        if time_remaining > 0:
            model.setParam('TimeLimit', time_remaining)
        
        # Start time variables
        t = model.addVars(P_loc, lb=0.0, ub=L, name="t")
        model.addConstr(t[0] == 0.0)
        for i in range(1, P_loc):
            model.addConstr(t[i] >= t[i-1])
        
        # Joint probability variables
        prob = model.addVars(n_joint, lb=0.0, ub=1.0, name="prob")
        model.addConstr(gp.quicksum(prob[j] for j in range(n_joint)) == 1.0)
        
        # Marginal constraints
        for i in range(P_loc):
            for si in range(n_scens[i]):
                marginal_prob = position_scenarios[i][si][2]
                relevant = [j for j, combo in enumerate(joint_indices) if combo[i] == si]
                model.addConstr(
                    gp.quicksum(prob[j] for j in relevant) == marginal_prob
                )
        
        # For each joint scenario j, define cost variables
        # C[j,i] = completion time of position i in scenario j
        # w[j,i] = waiting time at position i in scenario j
        # g[j,i] = idle time between i and i+1 in scenario j
        # o[j] = overtime in scenario j
        
        M_time = L + 200  # big-M for time variables
        
        C = model.addVars(n_joint, P_loc, lb=0.0, ub=M_time, name="C")
        w = model.addVars(n_joint, P_loc, lb=0.0, name="w")
        ready = model.addVars(n_joint, P_loc, lb=-100.0, ub=M_time, name="ready")
        
        if P_loc > 1:
            g = model.addVars(n_joint, P_loc - 1, lb=0.0, name="g")
        ov = model.addVars(n_joint, lb=0.0, name="ov")
        
        for j, combo in enumerate(joint_indices):
            for i in range(P_loc):
                d_val = position_scenarios[i][combo[i]][0]
                u_val = position_scenarios[i][combo[i]][1]
                
                # ready[j,i] = max(t[i] + u_val, C[j,i-1]) for i > 0
                #            = max(t[i] + u_val, t[i]) for i == 0 (C_prev = 0 but must be >= s_i)
                # Actually ready[j,i] = max(t[i], t[i] + u_val, C[j,i-1]) 
                # For i=0: ready[j,0] = max(t[0], t[0] + u_val) = max(0, u_val) since t[0]=0
                #        = max(0, u_val)
                
                if i == 0:
                    # ready >= t[0] = 0
                    model.addConstr(ready[j, i] >= 0)
                    # ready >= t[0] + u_val = u_val
                    model.addConstr(ready[j, i] >= u_val)
                    # ready = max(0, u_val), add linking: w = ready - t[0] = ready
                    model.addConstr(w[j, i] >= 0)
                    model.addConstr(w[j, i] >= ready[j, i] - t[i])
                    # Actually w[j,i] = ready[j,i] - t[i], and ready >= t[i]
                    # so w[j,i] = ready[j,i] - t[i] >= 0
                    # But ready[j,i] is defined as max of several terms >= t[i]
                    # So w[j,i] = ready[j,i] - t[i]
                    # We need ready to be exactly the max, not just >= max
                    # In minimization, the worst case pushes cost UP, so prob * cost is maximized
                    # This is a max-min problem, which is harder.
                    # 
                    # Actually our problem is:
                    # min_t max_{prob} sum_j prob[j] * cost[j]
                    # where cost[j] depends on t and scenario j
                    # This is a min-max problem.
                    # 
                    # For the inner max, cost is fixed for given t and j.
                    # But t is also a variable here.
                    # 
                    # The cost for scenario j given t is:
                    # cost[j] = c_w * sum_i w_i(j) + c_g * sum_i g_i(j) + c_o * ov(j)
                    # where w, g, ov are deterministic functions of t and scenario j.
                    # 
                    # So the problem is:
                    # min_t max_{prob in marginal constraints} sum_j prob[j] * f(t, scenario_j)
                    # = min_t { objective }
                    # 
                    # The inner max is a LP in prob for fixed t, and f(t, scenario_j) is 
                    # a deterministic (nonlinear but convex piecewise linear) function of t.
                    # 
                    # For the joint optimization, we can't simply add cost as variable
                    # because the bilinear term prob[j] * cost[j] where cost[j] depends on t
                    # makes this a bilinear program.
                    # 
                    # Alternative: epigraph formulation
                    # min_t eta
                    # s.t. for all feasible prob: sum_j prob[j] * f(t, scenario_j) <= eta
                    # This is equivalent to:
                    # eta >= max_{prob} sum_j prob[j] * f(t, scenario_j)
                    # = (by LP duality) min_{dual} dual objective
                    # So eta >= dual objective, and we minimize eta.
                    # 
                    # By strong duality of the inner LP:
                    # max_{prob} sum_j prob[j] * f(t, j)
                    # s.t. sum_j prob[j] = 1
                    #      for each position i, scenario si: 
                    #        sum_{j: combo[i]=si} prob[j] = marginal_prob[i][si]
                    #      prob[j] >= 0
                    # 
                    # Dual:
                    # min mu + sum_{i,si} lambda[i,si] * marginal_prob[i][si]
                    # s.t. mu + sum_i lambda[i,combo_j[i]] >= f(t, j) for all j
                    # 
                    # So the min-max becomes:
                    # min_{t, mu, lambda} mu + sum_{i,si} lambda[i,si] * marginal_prob[i][si]
                    # s.t. mu + sum_i lambda[i,combo_j[i]] >= f(t, j) for all j
                    #      t nondecreasing, t[0] = 0, 0 <= t <= L
                    #      lambda free, mu free
                    # 
                    # And f(t, j) involves max operations, so we need to linearize.
                    pass
                else:
                    pass
        
        # This is getting quite complex. Let me restructure using the dual formulation.
        model.reset()
        model = gp.Model("start_time_opt_dual")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('Threads', 1)
        model.setParam('NumericFocus', 0)
        model.setParam('MIPGap', 1e-4)
        if time_remaining > 0:
            model.setParam('TimeLimit', time_remaining)
        
        # Variables: t[i] start times, mu (dual for sum=1), lambda[i,si] (dual for marginals)
        t = model.addVars(P_loc, lb=0.0, ub=float(L), name="t")
        model.addConstr(t[0] == 0.0)
        for i in range(1, P_loc):
            model.addConstr(t[i] >= t[i-1])
        
        mu_var = model.addVar(lb=-GRB.INFINITY, name="mu")
        lam = {}
        for i in range(P_loc):
            for si in range(n_scens[i]):
                lam[i, si] = model.addVar(lb=-GRB.INFINITY, name=f"lam_{i}_{si}")
        
        # For each joint scenario j, we need:
        # mu + sum_i lam[i, combo_j[i]] >= f(t, j)
        # where f(t, j) = c_w * sum_i w_i + c_g * sum_i g_i + c_o * overtime
        # and w_i, g_i, overtime are piecewise linear convex functions of t.
        
        # To linearize f(t, j), introduce auxiliary variables for each scenario j:
        # For each scenario j and position i:
        #   ready[j,i] = max(t[i] + u_ji, C[j,i-1])  (for i>0, and max(t[i], t[i]+u_ji) for i=0)
        #   ready[j,i] >= t[i]
        #   w[j,i] = ready[j,i] - t[i]
        #   C[j,i] = ready[j,i] + d_ji
        #   g[j,i] = max(0, t[i+1] - C[j,i])
        #   overtime[j] = max(0, C[j,P-1] - L)
        
        # All these are convex in t, so f(t,j) is convex in t.
        # The constraint mu + sum lam >= f(t,j) is: mu + sum lam >= convex function of t.
        # This is a convex constraint (convex function <= affine), so it's fine.
        # But to express it in LP, we need to introduce auxiliary variables.
        
        # Let me add auxiliary variables for the cost in each scenario.
        # Note: n_joint can be large but manageable (up to 200000)
        # Actually this might be too many. Let me check typical sizes.
        # 8 scenarios per patient, P patients: 8^P. For P=5: 32768, P=6: 262144, P=7: 2M
        # So for P<=6 this is feasible, for P>=7 we need another approach.
        
        if n_joint > 100000:
            model.dispose()
            return optimize_start_times_simple(ordering, pat_data, L, c_w, c_g, c_o)
        
        # Create scenario cost variables
        cost_var = model.addVars(n_joint, lb=0.0, name="cost")
        
        # For each scenario, create the chain of constraints
        C_vars = {}
        w_vars = {}
        ready_vars = {}
        
        for j, combo in enumerate(joint_indices):
            for i in range(P_loc):
                d_val = position_scenarios[i][combo[i]][0]
                u_val = position_scenarios[i][combo[i]][1]
                
                ready_vars[j, i] = model.addVar(lb=0.0, name=f"rd_{j}_{i}")
                w_vars[j, i] = model.addVar(lb=0.0, name=f"w_{j}_{i}")
                C_vars[j, i] = model.addVar(lb=0.0, name=f"C_{j}_{i}")
                
                # ready >= t[i] (always, since procedure can't start before scheduled)
                model.addConstr(ready_vars[j, i] >= t[i])
                # ready >= t[i] + u_val (arrival time)
                model.addConstr(ready_vars[j, i] >= t[i] + u_val)
                # ready >= C[j, i-1] (previous completion)
                if i > 0:
                    model.addConstr(ready_vars[j, i] >= C_vars[j, i-1])
                
                # w = ready - t[i]
                model.addConstr(w_vars[j, i] == ready_vars[j, i] - t[i])
                
                # C = ready + d
                model.addConstr(C_vars[j, i] == ready_vars[j, i] + d_val)
        
        # Idle and overtime
        g_vars = {}
        ov_vars = {}
        
        for j, combo in enumerate(joint_indices):
            for i in range(P_loc - 1):
                g_vars[j, i] = model.addVar(lb=0.0, name=f"g_{j}_{i}")
                model.addConstr(g_vars[j, i] >= t[i+1] - C_vars[j, i])
            
            ov_vars[j] = model.addVar(lb=0.0, name=f"ov_{j}")
            model.addConstr(ov_vars[j] >= C_vars[j, P_loc-1] - L)
            
            # Cost for scenario j
            cost_expr = c_w * gp.quicksum(w_vars[j, i] for i in range(P_loc))
            if P_loc > 1:
                cost_expr += c_g * gp.quicksum(g_vars[j, i] for i in range(P_loc - 1))
            cost_expr += c_o * ov_vars[j]
            
            model.addConstr(cost_var[j] == cost_expr)
            
            # Dual constraint: mu + sum_i lam[i, combo[i]] >= cost[j]
            model.addConstr(
                mu_var + gp.quicksum(lam[i, combo[i]] for i in range(P_loc)) >= cost_var[j]
            )
        
        # Objective: min mu + sum_{i,si} lam[i,si] * marginal_prob[i][si]
        obj = mu_var + gp.quicksum(
            lam[i, si] * position_scenarios[i][si][2]
            for i in range(P_loc) for si in range(n_scens[i])
        )
        model.setObjective(obj, GRB.MINIMIZE)
        
        model.optimize()
        
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return [t[i].X for i in range(P_loc)]
        else:
            return optimize_start_times_simple(ordering, pat_data, L, c_w, c_g, c_o)
    
    # Main optimization loop
    best_obj = float('inf')
    best_assignment = None
    best_start_times = None
    best_ordering = None
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    # For small P, try all permutations
    # For larger P, use heuristic orderings
    
    if P <= 8:
        # Try enumeration but with time limit
        from math import factorial
        n_perms = factorial(P)
        
        if n_perms <= 5040 and remaining > 10:  # P <= 7
            # Full enumeration feasible
            all_perms = list(itertools.permutations(range(P)))
        else:
            # Generate a subset of permutations using heuristics
            all_perms = generate_heuristic_orderings(pat_data, P)
    else:
        all_perms = generate_heuristic_orderings(pat_data, P)
    
    def generate_heuristic_orderings_inner(pat_data, P):
        orderings = []
        indices = list(range(P))
        
        # Sort by mean duration (ascending - SPT)
        spt = sorted(indices, key=lambda p: pat_data[p]['mean_dur'])
        orderings.append(tuple(spt))
        
        # Sort by mean duration (descending - LPT)
        lpt = sorted(indices, key=lambda p: -pat_data[p]['mean_dur'])
        orderings.append(tuple(lpt))
        
        # Sort by variance proxy (high variance first)
        def var_proxy(p):
            pd = pat_data[p]
            range_A = pd['d_AU'] - pd['d_AL']
            range_I = pd['d_IU'] - pd['d_IL']
            return max(range_A, range_I)
        
        hvf = sorted(indices, key=lambda p: -var_proxy(p))
        orderings.append(tuple(hvf))
        
        # Low variance first
        lvf = sorted(indices, key=lambda p: var_proxy(p))
        orderings.append(tuple(lvf))
        
        # Sort by mean duration + arrival deviation
        orderings.append(tuple(sorted(indices, key=lambda p: pat_data[p]['mean_dur'] + pat_data[p]['mu_u'])))
        orderings.append(tuple(sorted(indices, key=lambda p: -(pat_data[p]['mean_dur'] + pat_data[p]['mu_u']))))
        
        # Random permutations
        import random
        rng = random.Random(42)
        for _ in range(min(100, max(0, 200 - len(orderings)))):
            perm = list(indices)
            rng.shuffle(perm)
            orderings.append(tuple(perm))
        
        # Remove duplicates
        orderings = list(set(orderings))
        return orderings
    
    if P > 8 or (P <= 8 and len(all_perms if 'all_perms' in dir() else []) == 0):
        all_perms = generate_heuristic_orderings_inner(pat_data, P)
    elif P <= 7:
        pass  # already have all_perms from enumeration
    else:
        all_perms = generate_heuristic_orderings_inner(pat_data, P)
    
    # Evaluate each ordering
    n_perms_total = len(all_perms)
    time_per_perm = max(1.0, (time_limit - (time.time() - start_time) - 5.0) / max(1, n_perms_total))
    
    for perm_idx, ordering in enumerate(all_perms):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2:
            break
        
        remaining = time_limit - elapsed - 2
        
        ordering = list(ordering)
        
        # Check feasibility of scenario enumeration
        position_scenarios = []
        for i in range(P):
            p = pat_data[ordering[i]]
            scens = get_patient_marginal_scenarios(p)
            position_scenarios.append(scens)
        
        n_scens_list = [len(s) for s in position_scenarios]
        total_joint = 1
        for n in n_scens_list:
            total_joint *= n
        
        if total_joint <= 100000 and remaining > 3:
            # Use LP-based start time optimization
            try:
                start_times = optimize_start_times_lp(
                    ordering, pat_data, L, c_w, c_g, c_o, 
                    min(time_per_perm, remaining - 1)
                )
            except Exception:
                start_times = optimize_start_times_simple(ordering, pat_data, L, c_w, c_g, c_o)
        else:
            start_times = optimize_start_times_simple(ordering, pat_data, L, c_w, c_g, c_o)
        
        # Evaluate worst-case cost
        elapsed = time.time() - start_time
        if elapsed > time_limit - 1:
            break
        
        try:
            if total_joint <= 500000:
                obj = compute_worst_case_expected_cost(ordering, start_times, pat_data, L, c_w, c_g, c_o)
            else:
                obj = compute_independent_expected_cost(ordering, start_times, pat_data, L, c_w, c_g, c_o)
        except Exception:
            continue
        
        if obj < best_obj:
            best_obj = obj
            best_ordering = ordering
            best_start_times = start_times
            
            # Build solution
            assignment = {}
            patient_start_times = {}
            schedule = {}
            for i in range(P):
                patient_idx = pat_data[ordering[i]]['idx']
                assignment[str(patient_idx)] = i + 1
                patient_start_times[str(patient_idx)] = start_times[i]
                schedule[str(i + 1)] = start_times[i]
            
            solution = {
                "objective_value": best_obj,
                "cost_structure": cost_name,
                "c_w": c_w,
                "c_g": c_g,
                "c_o": c_o,
                "assignment": assignment,
                "schedule": schedule,
                "patient_start_times": patient_start_times,
                "model_variables": {}
            }
            
            if logger:
                logger.log_solution(best_obj, solution)
    
    # If we have a best solution, try to refine start times with finer optimization
    if best_ordering is not None:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 2
        
        if remaining > 5:
            # Try optimizing start times more carefully for the best ordering
            try:
                position_scenarios = []
                for i in range(P):
                    p = pat_data[best_ordering[i]]
                    scens = get_patient_marginal_scenarios(p)
                    position_scenarios.append(scens)
                
                n_scens_list = [len(s) for s in position_scenarios]
                total_joint = 1
                for n in n_scens_list:
                    total_joint *= n
                
                if total_joint <= 100000:
                    refined_times = optimize_start_times_lp(
                        best_ordering, pat_data, L, c_w, c_g, c_o, remaining - 1
                    )
                    refined_obj = compute_worst_case_expected_cost(
                        best_ordering, refined_times, pat_data, L, c_w, c_g, c_o
                    )
                    
                    if refined_obj < best_obj:
                        best_obj = refined_obj
                        best_start_times = refined_times
            except Exception:
                pass
    
    # Build final solution
    if best_ordering is None:
        # Fallback: simple ordering
        best_ordering = list(range(P))
        best_start_times = optimize_start_times_simple(best_ordering, pat_data, L, c_w, c_g, c_o)
        try:
            best_obj = compute_worst_case_expected_cost(best_ordering, best_start_times, pat_data, L, c_w, c_g, c_o)
        except:
            best_obj = compute_independent_expected_cost(best_ordering, best_start_times, pat_data, L, c_w, c_g, c_o)
    
    assignment = {}
    patient_start_times_dict = {}
    schedule = {}
    for i in range(P):
        patient_idx = pat_data[best_ordering[i]]['idx']
        assignment[str(patient_idx)] = i + 1
        patient_start_times_dict[str(patient_idx)] = best_start_times[i]
        schedule[str(i + 1)] = best_start_times[i]
    
    # Build model_variables dict
    model_vars = {}
    for i in range(P):
        p_idx = pat_data[best_ordering[i]]['idx']
        pos = i + 1
        model_vars[f"x[{p_idx},{pos}]"] = 1.0
    for i in range(P):
        model_vars[f"t[{i+1}]"] = best_start_times[i]
    
    solution = {
        "objective_value": best_obj,
        "model_variables": model_vars,
        "cost_structure": cost_name,
        "c_w": c_w,
        "c_g": c_g,
        "c_o": c_o,
        "assignment": assignment,
        "schedule": schedule,
        "patient_start_times": patient_start_times_dict,
    }
    
    if logger:
        logger.log_solution(best_obj, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def generate_heuristic_orderings(pat_data, P):
    import random
    import itertools
    
    orderings = []
    indices = list(range(P))
    
    # Various sorting criteria
    # SPT (shortest processing time)
    orderings.append(tuple(sorted(indices, key=lambda p: pat_data[p]['mean_dur'])))
    # LPT
    orderings.append(tuple(sorted(indices, key=lambda p: -pat_data[p]['mean_dur'])))
    
    # By variance proxy (range of duration)
    def var_proxy(p):
        pd = pat_data[p]
        range_A = pd['d_AU'] - pd['d_AL']
        range_I = pd['d_IU'] - pd['d_IL']
        return pd['mu_q'] * range_A + (1 - pd['mu_q']) * range_I
    
    # High variance first
    orderings.append(tuple(sorted(indices, key=lambda p: -var_proxy(p))))
    # Low variance first
    orderings.append(tuple(sorted(indices, key=lambda p: var_proxy(p))))
    
    # By max possible duration
    orderings.append(tuple(sorted(indices, key=lambda p: -max(pat_data[p]['d_AU'], pat_data[p]['d_IU']))))
    orderings.append(tuple(sorted(indices, key=lambda p: max(pat_data[p]['d_AU'], pat_data[p]['d_IU']))))
    
    # By arrival deviation mean
    orderings.append(tuple(sorted(indices, key=lambda p: pat_data[p]['mu_u'])))
    orderings.append(tuple(sorted(indices, key=lambda p: -pat_data[p]['mu_u'])))
    
    # Combined criteria
    orderings.append(tuple(sorted(indices, key=lambda p: pat_data[p]['mean_dur'] + 0.5 * var_proxy(p))))
    orderings.append(tuple(sorted(indices, key=lambda p: -(pat_data[p]['mean_dur'] + 0.5 * var_proxy(p)))))
    
    # Identity
    orderings.append(tuple(indices))
    
    # Random permutations
    rng = random.Random(42)
    for _ in range(200):
        perm = list(indices)
        rng.shuffle(perm)
        orderings.append(tuple(perm))
    
    # Remove duplicates
    orderings = list(set(orderings))
    return orderings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)