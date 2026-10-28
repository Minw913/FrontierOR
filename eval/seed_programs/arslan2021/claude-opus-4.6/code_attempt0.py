import argparse
import json
import time
import numpy as np

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    W = data['problem_parameters']['knapsack_capacity']
    Gamma = data['problem_parameters']['uncertainty_budget']
    items = data['items']
    n = len(items)
    
    w = [item['weight'] for item in items]
    c = [item['nominal_profit'] for item in items]
    d = [item['max_degradation'] for item in items]
    p = [item['outsource_penalty'] for item in items]
    r = [item['repair_capacity'] for item in items]
    
    import gurobipy as gp
    from gurobipy import GRB
    
    start_time = time.time()
    
    # Master problem: min theta
    # x_i binary, theta >= first_stage_cost + second_stage_cost(delta) for all delta
    # We use cutting planes: for each delta scenario, we solve the recourse LP
    # and add a cut.
    
    master = gp.Model("master")
    master.Params.Seed = 0
    master.Params.MIPGap = 1e-4
    master.Params.NumericFocus = 0
    master.Params.Threads = 1
    master.Params.OutputFlag = 0
    
    x = master.addVars(n, vtype=GRB.BINARY, name="x")
    theta = master.addVar(lb=-1e15, name="theta")
    
    # First stage cost: sum_i x_i * (p_i - c_i)
    first_stage = gp.quicksum(x[i] * (p[i] - c[i]) for i in range(n))
    
    master.setObjective(first_stage + theta, GRB.MINIMIZE)
    
    # theta >= 0 is not necessarily true, but we need a lower bound
    # Actually theta can be negative (good recourse)
    
    best_obj = float('inf')
    best_x = None
    
    def solve_recourse(x_val, delta):
        """Given x and delta, solve the recourse LP to get optimal y, z and cost."""
        rec = gp.Model("recourse")
        rec.Params.OutputFlag = 0
        rec.Params.Threads = 1
        
        y = rec.addVars(n, lb=0, ub=1, name="y")
        z = rec.addVars(n, lb=0, ub=1, name="z")
        
        # Constraints:
        # z_i <= y_i
        # y_i <= x_i
        # sum_i w_i*y_i + r_i*z_i <= W
        for i in range(n):
            rec.addConstr(z[i] <= y[i])
            rec.addConstr(y[i] <= x_val[i])
        
        rec.addConstr(gp.quicksum(w[i]*y[i] + r[i]*z[i] for i in range(n)) <= W)
        
        # Second stage cost (to minimize):
        # sum_i (d_i*delta_i - p_i)*y_i - sum_i d_i*delta_i*z_i
        obj = gp.quicksum((d[i]*delta[i] - p[i])*y[i] for i in range(n)) - \
              gp.quicksum(d[i]*delta[i]*z[i] for i in range(n))
        
        rec.setObjective(obj, GRB.MINIMIZE)
        rec.optimize()
        
        if rec.Status == GRB.OPTIMAL:
            y_val = [y[i].X for i in range(n)]
            z_val = [z[i].X for i in range(n)]
            return rec.ObjVal, y_val, z_val
        else:
            return 0, [0]*n, [0]*n
    
    def solve_adversary(x_val):
        """Given x, find worst-case delta by solving max over delta of min recourse.
        
        This is a max-min problem. We dualize the inner min (recourse LP) to get
        a maximization problem.
        
        Inner LP (for fixed x, delta):
        min sum_i (d_i*delta_i - p_i)*y_i - sum_i d_i*delta_i*z_i
        s.t. z_i <= y_i  (dual: alpha_i >= 0)
             y_i <= x_i  (dual: beta_i >= 0)
             sum w_i*y_i + r_i*z_i <= W  (dual: gamma >= 0)
             y_i, z_i >= 0
        
        Dual of inner:
        max -sum_i beta_i*x_i - gamma*W
        s.t. d_i*delta_i - p_i >= -alpha_i + beta_i + w_i*gamma  (for y_i)
             -d_i*delta_i >= alpha_i + r_i*gamma                   (for z_i)
             alpha_i, beta_i, gamma >= 0
        
        Wait, let me be more careful. The primal is:
        min c'x where the "x" here is (y,z).
        
        Primal constraints in standard form Ax <= b:
        z_i - y_i <= 0     (alpha_i >= 0)
        y_i <= x_i          (beta_i >= 0)
        sum w_i*y_i + r_i*z_i <= W  (gamma >= 0)
        -y_i <= 0            (mu_i >= 0)
        -z_i <= 0            (nu_i >= 0)
        
        Dual: max 0*alpha + x*beta_neg... wait let me use standard form properly.
        
        Actually, let me reformulate. The dual of min{c^T u : Au >= b, u >= 0} is max{b^T v : A^T v <= c, v >= 0}.
        
        Primal: min over y,z of:
        sum_i (d_i*delta_i - p_i)*y_i + sum_i (-d_i*delta_i)*z_i
        s.t.
        y_i - z_i >= 0   for all i  [lambda_i >= 0]
        -y_i >= -x_i     for all i  [mu_i >= 0]  => y_i <= x_i
        -sum w_i*y_i - sum r_i*z_i >= -W  [eta >= 0]
        y_i >= 0, z_i >= 0
        
        Dual: max sum_i 0*lambda_i + sum_i (-x_i)*mu_i + (-W)*eta
        s.t.
        lambda_i - mu_i - w_i*eta <= d_i*delta_i - p_i   for all i  (y_i constraint)
        -lambda_i - r_i*eta <= -d_i*delta_i               for all i  (z_i constraint)
        lambda_i, mu_i, eta >= 0
        
        So dual obj = -sum_i x_i*mu_i - W*eta
        
        Now the adversary problem is:
        max_{delta, lambda, mu, eta} -sum_i x_i*mu_i - W*eta
        s.t.
        lambda_i - mu_i - w_i*eta <= d_i*delta_i - p_i
        -lambda_i - r_i*eta <= -d_i*delta_i
        sum_i delta_i <= Gamma
        0 <= delta_i <= 1
        lambda_i, mu_i, eta >= 0
        
        This has bilinear terms d_i*delta_i in the constraints with dual variables.
        But notice that delta only appears on the RHS of the dual constraints.
        
        From constraint 2: lambda_i >= d_i*delta_i - r_i*eta
        From constraint 1: mu_i >= lambda_i - w_i*eta - d_i*delta_i + p_i
        
        To maximize -sum x_i*mu_i - W*eta, we want mu_i as small as possible.
        So mu_i = max(0, lambda_i - w_i*eta - d_i*delta_i + p_i)
        And lambda_i = max(0, d_i*delta_i - r_i*eta)
        
        So this becomes:
        max -sum_i x_i * max(0, max(0, d_i*delta_i - r_i*eta) - w_i*eta - d_i*delta_i + p_i) - W*eta
        
        Let me simplify the inner max(0, ...):
        Let A_i = d_i*delta_i - r_i*eta
        lambda_i = max(0, A_i)
        
        Case 1: A_i >= 0, i.e., d_i*delta_i >= r_i*eta
        lambda_i = d_i*delta_i - r_i*eta
        mu_i = max(0, d_i*delta_i - r_i*eta - w_i*eta - d_i*delta_i + p_i)
             = max(0, -(r_i + w_i)*eta + p_i)
             = max(0, p_i - (r_i + w_i)*eta)
        
        Case 2: A_i < 0
        lambda_i = 0
        mu_i = max(0, -w_i*eta - d_i*delta_i + p_i)
             = max(0, p_i - w_i*eta - d_i*delta_i)
        
        This is complex but we can formulate it as an MILP or solve it with enumeration.
        Actually, since this is a single-level max problem with continuous variables 
        and some piecewise-linear structure, let me just formulate it as an LP 
        by introducing the right variables.
        """
        
        # The adversary problem with bilinear terms. Let me try a different approach.
        # For fixed x, I'll solve the max-min by discretizing delta or using 
        # column-and-constraint generation.
        
        # Actually, the key insight: for fixed x AND fixed eta, the problem 
        # separates by item. For each item, given eta, we can compute the 
        # optimal delta_i contribution.
        
        # Let me try: parametric in eta >= 0, solve for optimal delta.
        # For fixed eta, the inner dual becomes:
        # max -sum_i x_i * mu_i - W*eta
        # where mu_i, lambda_i are determined by delta_i and eta.
        
        # For fixed eta, the problem in delta is:
        # max_{delta} -sum_i x_i * f_i(delta_i, eta) - W*eta
        # s.t. sum delta_i <= Gamma, 0 <= delta_i <= 1
        # where f_i is the mu_i expression above.
        
        # For each item i with x_i = 1:
        # If d_i*delta_i >= r_i*eta:
        #   mu_i = max(0, p_i - (r_i+w_i)*eta) -- independent of delta_i!
        # If d_i*delta_i < r_i*eta:
        #   mu_i = max(0, p_i - w_i*eta - d_i*delta_i)
        
        # The adversary wants to MAXIMIZE -sum x_i*mu_i, i.e., make mu_i small 
        # (more negative contribution). Wait, mu_i >= 0, so -x_i*mu_i <= 0.
        # The adversary wants to maximize (make least negative), so wants mu_i 
        # as small as possible... but mu_i >= 0.
        
        # Hmm, actually the recourse cost can be negative (good for the company).
        # The adversary wants to maximize the recourse cost = dual objective.
        # dual obj = -sum x_i*mu_i - W*eta which is <= 0.
        # So adversary wants this to be as close to 0 as possible (least negative).
        
        # Actually I think I should just solve this numerically. Let me use
        # a different approach: solve it as a bilinear optimization using 
        # McCormick relaxations or binary expansion of delta.
        
        # Alternative cleaner approach: column-and-constraint generation.
        # For the master, we iterate:
        # 1. Fix x, solve adversary subproblem for worst delta
        # 2. Add cut to master, re-solve master
        # For step 1, fix x and solve the max-min:
        # We can solve this by noting that for fixed x, the problem is
        # max_delta min_{y,z} recourse_cost(y,z,delta)
        # 
        # Since the inner problem is an LP, by strong duality:
        # max_delta [dual of inner LP evaluated at optimal]
        #
        # This gives a bilinear problem. But we can solve it by 
        # discretizing or by another level of column generation.
        
        # Let me try a simpler approach: for the adversary, fix delta and solve 
        # recourse, then update delta. Use alternating optimization or 
        # enumeration of extreme delta vectors.
        
        # Key insight about extreme points of the uncertainty set:
        # The uncertainty set is {delta : sum delta_i <= Gamma, 0 <= delta_i <= 1}
        # Extreme points have at most one fractional delta_i.
        # So we can enumerate: choose a subset S of items with |S| <= floor(Gamma),
        # set delta_i = 1 for i in S, delta_j = Gamma - |S| for one j, rest = 0.
        
        # But this is exponential. Let's use the LP-based adversary.
        
        # For the adversary, given x, we solve:
        # max_delta min_{y,z} cost
        # = max_delta dual_of_min
        
        # The dual (derived above) is:
        # max -sum x_i*mu_i - W*eta
        # s.t. lambda_i - mu_i - w_i*eta <= d_i*delta_i - p_i  (i)
        #      -lambda_i - r_i*eta <= -d_i*delta_i              (ii)
        #      sum delta_i <= Gamma, 0<=delta_i<=1
        #      lambda_i, mu_i, eta >= 0
        
        # Bilinear terms: d_i*delta_i appears only on RHS.
        # 
        # For fixed dual variables, delta wants to make RHS as large as possible
        # in constraint (i) and as large as possible (less negative) in (ii).
        # But constraint (i) has d_i*delta_i on RHS (good for feasibility, 
        # relaxes constraint).
        # Constraint (ii) has -d_i*delta_i on RHS (making delta larger tightens it).
        # 
        # So delta_i has opposing effects on the two constraints.
        # This means we can't trivially decompose.
        
        # Let me try yet another approach. For items with x_i=0, they don't 
        # matter (y_i=z_i=0, contribution to recourse is 0).
        # For items with x_i=1, the recourse decides y_i, z_i.
        
        # I'll solve the bilinear adversary problem using Gurobi's bilinear 
        # capability (nonconvex QP) or reformulate.
        
        # Actually, let me linearize. The bilinear terms are delta_i * lambda_i 
        # type terms? No, delta_i appears only on RHS, not multiplied by a variable.
        # Wait, re-examining: in the dual constraints, delta_i appears linearly 
        # on the RHS. The objective is -sum x_i*mu_i - W*eta, which doesn't 
        # involve delta. So the full problem is:
        #
        # max -sum x_i*mu_i - W*eta
        # s.t. lambda_i - mu_i - w_i*eta - d_i*delta_i <= -p_i   (i)
        #      -lambda_i - r_i*eta + d_i*delta_i <= 0              (ii)
        #      sum delta_i <= Gamma
        #      0 <= delta_i <= 1
        #      lambda_i, mu_i, eta >= 0
        #
        # This is a LINEAR program! Because delta_i are just additional 
        # continuous variables. The whole thing is linear.
        # 
        # Wait, but this has x_i as parameters (fixed). So yes, this is an LP.
        # 
        # But in the master problem, x_i is binary. So the full problem is:
        # min_x { first_stage(x) + max_{delta, dual} LP(x, delta, dual) }
        # where the inner max is an LP parameterized by x.
        #
        # Since x is binary, we can use Benders decomposition:
        # Master: min first_stage(x) + theta
        #         theta >= cut_k for each scenario k
        # Subproblem: for fixed x*, solve the LP to get theta* and generate cut.
        
        # The subproblem LP for the adversary (with x fixed):
        adv = gp.Model("adversary")
        adv.Params.OutputFlag = 0
        adv.Params.Threads = 1
        
        delta = adv.addVars(n, lb=0, ub=1, name="delta")
        lam = adv.addVars(n, lb=0, name="lambda")
        mu = adv.addVars(n, lb=0, name="mu")
        eta = adv.addVar(lb=0, name="eta")
        
        # Constraints
        for i in range(n):
            # lambda_i - mu_i - w_i*eta - d_i*delta_i <= -p_i
            adv.addConstr(lam[i] - mu[i] - w[i]*eta - d[i]*delta[i] <= -p[i])
            # -lambda_i - r_i*eta + d_i*delta_i <= 0
            adv.addConstr(-lam[i] - r[i]*eta + d[i]*delta[i] <= 0)
        
        adv.addConstr(gp.quicksum(delta[i] for i in range(n)) <= Gamma)
        
        # Objective: max -sum x_i*mu_i - W*eta
        adv.setObjective(-gp.quicksum(x_val[i]*mu[i] for i in range(n)) - W*eta, GRB.MAXIMIZE)
        
        adv.optimize()
        
        if adv.Status == GRB.OPTIMAL:
            delta_val = [delta[i].X for i in range(n)]
            obj_val = adv.ObjVal
            # Also get dual information for Benders cuts
            return obj_val, delta_val
        else:
            return 0, [0]*n
    
    # Benders decomposition
    # Master: min sum_i x_i*(p_i - c_i) + theta
    # Cuts: theta >= h(x) where h(x) is the adversary's optimal value
    
    # For Benders, we need optimality cuts. The adversary subproblem is:
    # max -sum x_i*mu_i - W*eta
    # The dual of THIS LP (with respect to x as parameter) gives us the cut.
    
    # Actually, let me think about this more carefully.
    # The subproblem is parameterized by x (binary). 
    # At optimal (mu*, eta*, delta*, lambda*), the value is -sum x_i*mu_i* - W*eta*.
    # A valid cut is: theta >= -sum_i mu_i* * x_i - W*eta*
    # This is a Benders optimality cut.
    # But this is only valid if the dual variables are optimal for the particular x*.
    # However, since x appears only in the objective of the adversary LP, the 
    # feasible region doesn't depend on x. So any feasible (mu, eta, delta, lambda) 
    # gives a valid lower bound: theta >= -sum mu_i*x_i - W*eta.
    # This is exactly the Benders cut structure!
    
    # So the algorithm is:
    # 1. Solve master (with current cuts) -> get x*
    # 2. Solve adversary LP with x* -> get (mu*, eta*, delta*, lambda*)
    # 3. Add cut: theta >= -sum mu_i* * x_i - W*eta*
    # 4. Repeat until convergence
    
    max_iterations = 1000
    cuts = []  # list of (mu_vals, eta_val, rhs_constant)
    
    # Initial lower bound on theta: it can be very negative
    # Let's add a trivial cut or just let it be unbounded initially
    # We need at least one cut to bound theta from below
    
    # Actually, let's think about bounds. theta is the worst-case recourse cost.
    # If all items are selected and produced, the adversary can cause degradation.
    # If no items are selected, theta = 0 (no recourse actions possible).
    # The recourse cost can be negative (company benefits).
    
    # Lower bound on theta: at worst, -sum_i p_i (if all selected items are produced 
    # and there's no degradation, the outsourcing penalty is saved).
    # Actually theta can be as low as -sum_i p_i.
    
    # Let me just start with theta >= -sum_i p_i as a warm-start
    # Or just solve iteratively.
    
    # To handle the initial unboundedness, let's add a simple bound
    # theta >= -sum_i |p_i| - sum_i |c_i| 
    big_M_bound = sum(abs(p[i]) + abs(c[i]) + abs(d[i]) for i in range(n))
    master.addConstr(theta >= -big_M_bound)
    
    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2:
            break
        
        master.optimize()
        
        if master.Status != GRB.OPTIMAL:
            break
        
        x_val = [x[i].X for i in range(n)]
        master_obj = master.ObjVal
        
        # Solve adversary subproblem
        adv_obj, delta_val = solve_adversary(x_val)
        
        # Upper bound: first_stage(x_val) + adv_obj
        first_stage_val = sum(x_val[i] * (p[i] - c[i]) for i in range(n))
        upper = first_stage_val + adv_obj
        
        if upper < best_obj - 1e-6:
            best_obj = upper
            best_x = [round(x_val[i]) for i in range(n)]
            
            sol = {
                "objective_value": best_obj,
                "x": {str(i): int(best_x[i]) for i in range(n)}
            }
            if logger:
                logger.log_solution(best_obj, sol)
        
        # Check convergence
        theta_val = theta.X
        if abs(adv_obj - theta_val) < 1e-5:
            break
        
        # Add Benders cut
        # We need the optimal mu and eta from the adversary LP
        # Let me re-solve and extract them
        
        # Actually, I need to extract the solution values from the adversary.
        # Let me restructure to return them.
        
        # Re-solve adversary to get dual values
        adv = gp.Model("adversary_cut")
        adv.Params.OutputFlag = 0
        adv.Params.Threads = 1
        
        delta_v = adv.addVars(n, lb=0, ub=1, name="delta")
        lam_v = adv.addVars(n, lb=0, name="lambda")
        mu_v = adv.addVars(n, lb=0, name="mu")
        eta_v = adv.addVar(lb=0, name="eta")
        
        for i in range(n):
            adv.addConstr(lam_v[i] - mu_v[i] - w[i]*eta_v - d[i]*delta_v[i] <= -p[i])
            adv.addConstr(-lam_v[i] - r[i]*eta_v + d[i]*delta_v[i] <= 0)
        
        adv.addConstr(gp.quicksum(delta_v[i] for i in range(n)) <= Gamma)
        
        adv.setObjective(-gp.quicksum(x_val[i]*mu_v[i] for i in range(n)) - W*eta_v, GRB.MAXIMIZE)
        adv.optimize()
        
        if adv.Status == GRB.OPTIMAL:
            mu_vals = [mu_v[i].X for i in range(n)]
            eta_val_opt = eta_v.X
            
            # Benders cut: theta >= -sum_i mu_i* * x_i - W * eta*
            master.addConstr(theta >= -gp.quicksum(mu_vals[i] * x[i] for i in range(n)) - W * eta_val_opt)
    
    # If no solution found, try all zeros
    if best_x is None:
        best_x = [0] * n
        best_obj = 0.0
    
    sol = {
        "objective_value": best_obj,
        "x": {str(i): int(best_x[i]) for i in range(n)}
    }
    
    if logger:
        logger.log_solution(best_obj, sol)
    
    with open(solution_path, 'w') as f:
        json.dump(sol, f)


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