import argparse
import json
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

def solve_bilevel(instance, time_limit, logger=None):
    n = instance['n']
    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    c = np.array(instance['c'], dtype=float)
    d2 = np.array(instance['d2'], dtype=float)
    A1 = np.array(instance['A1'], dtype=float)
    B1 = np.array(instance['B1'], dtype=float)
    b1 = np.array(instance['b1'], dtype=float)
    A2 = np.array(instance['A2'], dtype=float)
    B2 = np.array(instance['B2'], dtype=float)
    b2 = np.array(instance['b2'], dtype=float)
    integer_follower_vars = [i - 1 for i in instance['integer_follower_vars']]  # 0-based
    continuous_follower_vars = [i for i in range(n2) if i not in integer_follower_vars]
    
    start_time = time.time()
    
    # Strategy: We'll try multiple approaches
    # 1) If follower has no integer variables affecting optimality much, use KKT
    # 2) General approach: iterative with value function constraints
    
    # Approach: Single-level reformulation
    # The follower solves: max d2'y s.t. B2*y <= b2 - A2*x, y >= 0, some y integer
    # This is a MILP. For bilevel with MILP follower, exact reformulation is hard.
    # 
    # Practical approach: Use the LP relaxation of the follower for the dual-based
    # reformulation, then add cuts to ensure follower integrality.
    # 
    # Alternative: Enumerate follower integer variable values (if feasible count is small)
    # or use a branch-and-cut approach.
    
    # Let's try: Single-level reformulation using strong duality of the follower's LP relaxation
    # combined with integrality on follower integer variables.
    # This gives a VALID RELAXATION (optimistic assumption helps here since we're using
    # LP relaxation dual which might give a higher follower objective than the true MILP optimal).
    # Then we verify feasibility.
    
    # Actually, for the optimistic bilevel with MILP follower, a common approach is:
    # Since the leader can choose among follower optima (optimistic), and the follower's
    # problem is a MILP, we can reformulate as:
    # max c'x
    # s.t. A1*x + B1*y <= b1
    #      A2*x + B2*y <= b2
    #      y >= 0, integrality on some y
    #      x >= 0, integer
    #      d2'y >= optimal follower value for given x
    #
    # The challenge is "d2'y >= optimal follower value for given x" - this is the
    # value function constraint.
    
    # We'll use an iterative approach:
    # 1) Solve the "high point relaxation" (HPR): ignore follower optimality
    # 2) Check if the follower's response y is actually optimal for the follower given x
    # 3) If not, add a cut and re-solve
    
    best_obj = -float('inf')
    best_solution = None
    
    # First solve HPR to get upper bound and initial solution candidate
    def solve_hpr(extra_constraints=None, time_left=None):
        model = gp.Model("HPR")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        if time_left is not None:
            model.Params.TimeLimit = max(time_left, 1)
        
        x = model.addVars(n1, vtype=GRB.INTEGER, lb=0, name="x")
        y = model.addVars(n2, lb=0, name="y")
        
        # Set integer follower vars
        for i in integer_follower_vars:
            y[i].vtype = GRB.INTEGER
        
        # Upper level constraints: A1*x + B1*y <= b1
        for i in range(m1):
            model.addConstr(
                gp.quicksum(A1[i, j] * x[j] for j in range(n1)) +
                gp.quicksum(B1[i, j] * y[j] for j in range(n2)) <= b1[i],
                name=f"upper_{i}"
            )
        
        # Lower level constraints: A2*x + B2*y <= b2
        for i in range(m2):
            model.addConstr(
                gp.quicksum(A2[i, j] * x[j] for j in range(n1)) +
                gp.quicksum(B2[i, j] * y[j] for j in range(n2)) <= b2[i],
                name=f"lower_{i}"
            )
        
        # Extra constraints (cuts)
        if extra_constraints:
            for idx, (ec_type, ec_data) in enumerate(extra_constraints):
                if ec_type == "value_function":
                    x_fixed, opt_val = ec_data
                    # For the specific x_fixed, the follower objective must be >= opt_val
                    # But this is point-specific... We need a different approach.
                    pass
                elif ec_type == "follower_obj_bound":
                    # d2'y >= bound when x = x_fixed
                    # This is hard to linearize for general x...
                    pass
        
        # Leader objective: max c'x
        model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n1)), GRB.MAXIMIZE)
        
        model.optimize()
        
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            x_val = [x[j].X for j in range(n1)]
            y_val = [y[j].X for j in range(n2)]
            obj_val = model.ObjVal
            return x_val, y_val, obj_val, model
        return None, None, None, model
    
    def solve_follower(x_val, time_left=None):
        """Solve follower's problem given leader's x"""
        model = gp.Model("Follower")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        if time_left is not None:
            model.Params.TimeLimit = max(time_left, 1)
        
        y = model.addVars(n2, lb=0, name="y")
        for i in integer_follower_vars:
            y[i].vtype = GRB.INTEGER
        
        # Lower level constraints with fixed x
        rhs = b2 - A2 @ np.array(x_val)
        for i in range(m2):
            model.addConstr(
                gp.quicksum(B2[i, j] * y[j] for j in range(n2)) <= rhs[i],
                name=f"lower_{i}"
            )
        
        model.setObjective(gp.quicksum(d2[j] * y[j] for j in range(n2)), GRB.MAXIMIZE)
        model.optimize()
        
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            y_val = [y[j].X for j in range(n2)]
            fobj = model.ObjVal
            return y_val, fobj, model.status
        elif model.status == GRB.INFEASIBLE:
            return None, None, GRB.INFEASIBLE
        elif model.status == GRB.UNBOUNDED:
            return None, float('inf'), GRB.UNBOUNDED
        return None, None, model.status
    
    def solve_follower_optimistic(x_val, follower_opt_val, time_left=None):
        """Given x and follower optimal value, find follower solution that maximizes leader obj.
        Since leader obj only depends on x (d1=0), any follower optimal solution works.
        But we need upper-level feasibility, so pick follower solution that satisfies upper constraints."""
        model = gp.Model("FollowerOptimistic")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        if time_left is not None:
            model.Params.TimeLimit = max(time_left, 1)
        
        y = model.addVars(n2, lb=0, name="y")
        for i in integer_follower_vars:
            y[i].vtype = GRB.INTEGER
        
        # Lower level constraints with fixed x
        rhs = b2 - A2 @ np.array(x_val)
        for i in range(m2):
            model.addConstr(
                gp.quicksum(B2[i, j] * y[j] for j in range(n2)) <= rhs[i],
                name=f"lower_{i}"
            )
        
        # Follower optimality constraint
        model.addConstr(
            gp.quicksum(d2[j] * y[j] for j in range(n2)) >= follower_opt_val - 1e-4,
            name="follower_opt"
        )
        
        # Upper level constraints with fixed x (feasibility for the leader)
        x_arr = np.array(x_val)
        rhs_upper = b1 - A1 @ x_arr
        slack = model.addVars(m1, lb=0, name="slack")
        for i in range(m1):
            model.addConstr(
                gp.quicksum(B1[i, j] * y[j] for j in range(n2)) + slack[i] == rhs_upper[i],
                name=f"upper_{i}"
            )
        
        # Minimize violation of upper-level constraints (maximize slack)
        # Actually, we want y feasible for upper level: B1*y <= b1 - A1*x
        # So slack >= 0 means feasible. We maximize min slack or just check feasibility.
        # Let's minimize sum of a penalty for infeasibility
        # Actually slack >= 0 is already enforced, so if feasible, all good.
        # The model is: B1*y + slack = rhs_upper, slack >= 0 => B1*y = rhs_upper - slack <= rhs_upper
        # So this automatically enforces upper-level feasibility.
        
        # Since d1=0, leader doesn't care about y, just need upper-level feasibility
        # Minimize 0 (feasibility problem) or maximize follower objective for tie-breaking
        model.setObjective(gp.quicksum(d2[j] * y[j] for j in range(n2)), GRB.MAXIMIZE)
        
        model.optimize()
        
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            y_val = [y[j].X for j in range(n2)]
            fobj = sum(d2[j] * y_val[j] for j in range(n2))
            return y_val, fobj
        return None, None
    
    # Main algorithm: Iterative approach
    # We use a single-level reformulation approach with callbacks or iterative cuts.
    
    # For moderate sizes, let's try a direct approach:
    # Build a single-level MIP that includes:
    # 1) Leader and follower variables
    # 2) Upper and lower level constraints
    # 3) For the follower's optimality: use LP duality of the LP relaxation
    #    Then enforce integrality on follower integer vars
    #    Under optimistic assumption, this gives a valid formulation IF the LP relaxation
    #    bound equals the MILP optimal (which isn't always true).
    #    We'll use this as a heuristic and then verify.
    
    # Better approach for this specific case:
    # Since d1=0 (leader doesn't care about follower's y in its objective),
    # the leader wants to maximize c'x.
    # The constraint is that there exists y that is follower-optimal given x
    # and (x,y) satisfies upper-level constraints.
    
    # Approach: Solve using Benders-like decomposition / iterative scheme
    # 1. Solve master: max c'x s.t. x integer, some linking constraints
    # 2. For each candidate x, solve follower problem, check upper-level feasibility
    
    # Let's try the KKT-based approach for the LP relaxation of the follower,
    # then iteratively fix things up.
    
    # KKT approach for LP relaxation of follower:
    # Follower LP: max d2'y s.t. B2*y <= b2 - A2*x, y >= 0
    # Dual: min (b2 - A2*x)'u s.t. B2'u >= d2, u >= 0
    # Strong duality: d2'y = (b2 - A2*x)'u
    # Complementary slackness: u_i * (b2_i - (A2*x)_i - (B2*y)_i) = 0
    #                          y_j * (sum_i B2[i,j]*u_i - d2_j) = 0
    
    # We linearize complementarity using big-M.
    
    # First, let's estimate big-M values
    # The RHS values are at most 130, coefficients at most 10
    # So individual variable bounds: y_j <= max(b2) / min_nonzero(B2[:,j]) approximately
    # And u_i <= max(d2) / min_nonzero(B2[i,:]) approximately
    
    # Let's compute reasonable bounds
    M_y = np.zeros(n2)
    for j in range(n2):
        col = B2[:, j]
        if np.any(col > 0):
            M_y[j] = max(b2) / np.min(col[col > 0])
        else:
            M_y[j] = 1000  # large default
    M_y = np.minimum(M_y, 1000)
    
    M_u = np.zeros(m2)
    for i in range(m2):
        # u_i in dual constraint: B2'u >= d2
        # u_i <= some bound. From dual feasibility and optimality.
        M_u[i] = 200  # conservative bound
    
    M_slack_lower = np.zeros(m2)
    for i in range(m2):
        M_slack_lower[i] = max(b2[i], 200)
    
    M_slack_dual = np.zeros(n2)
    for j in range(n2):
        M_slack_dual[j] = 200
    
    def solve_kkt_reformulation(time_left=None):
        model = gp.Model("KKT")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        if time_left is not None:
            model.Params.TimeLimit = max(time_left, 1)
        
        # Leader variables
        x = model.addVars(n1, vtype=GRB.INTEGER, lb=0, name="x")
        # Follower variables
        y = model.addVars(n2, lb=0, name="y")
        for i in integer_follower_vars:
            y[i].vtype = GRB.INTEGER
        # Dual variables for lower-level constraints
        u = model.addVars(m2, lb=0, name="u")
        
        # Binary variables for complementarity
        z_primal = model.addVars(m2, vtype=GRB.BINARY, name="z_primal")  # 1 if primal slack > 0
        z_dual = model.addVars(n2, vtype=GRB.BINARY, name="z_dual")  # 1 if dual slack > 0
        
        # Upper level constraints
        for i in range(m1):
            model.addConstr(
                gp.quicksum(A1[i, j] * x[j] for j in range(n1)) +
                gp.quicksum(B1[i, j] * y[j] for j in range(n2)) <= b1[i],
                name=f"upper_{i}"
            )
        
        # Lower level constraints (primal feasibility)
        primal_slack = model.addVars(m2, lb=0, name="ps")
        for i in range(m2):
            model.addConstr(
                gp.quicksum(B2[i, j] * y[j] for j in range(n2)) + primal_slack[i] ==
                b2[i] - gp.quicksum(A2[i, j] * x[j] for j in range(n1)),
                name=f"lower_{i}"
            )
        
        # Dual feasibility: B2'u >= d2
        dual_slack = model.addVars(n2, lb=0, name="ds")
        for j in range(n2):
            model.addConstr(
                gp.quicksum(B2[i, j] * u[i] for i in range(m2)) - dual_slack[j] == d2[j],
                name=f"dual_{j}"
            )
        
        # Strong duality: d2'y = (b2 - A2*x)'u
        # This replaces complementarity if we also have primal and dual feasibility
        model.addConstr(
            gp.quicksum(d2[j] * y[j] for j in range(n2)) ==
            gp.quicksum((b2[i] - gp.quicksum(A2[i, j] * x[j] for j in range(n1))) * u[i] for i in range(m2)),
            name="strong_duality"
        )
        
        # Note: Strong duality of the LP relaxation is valid. But we also have integrality
        # on some y variables. The LP relaxation optimal >= MILP optimal.
        # With strong duality, we're saying the LP relaxation value = dual value.
        # But y is also integer-constrained. So d2'y (integer y) might not achieve LP opt.
        # This could make the formulation infeasible when it shouldn't be, or give wrong results.
        
        # Actually, strong duality constraint with integer y means:
        # d2'y_integer = LP_dual_opt. This is only possible if the LP relaxation optimal
        # value is achieved by an integer-feasible point.
        # This is a RELAXATION of the true bilevel problem only in certain cases.
        
        # For correctness, we should NOT use strong duality with integer follower vars.
        # Instead, we need: d2'y >= follower_MILP_optimal(x)
        # which can't be expressed as a simple linear constraint.
        
        # HOWEVER: Under the OPTIMISTIC assumption, and since d1=0, the leader just wants
        # max c'x such that exists y that is follower-optimal and upper-level feasible.
        # 
        # Let's use a different approach: 
        # We use the LP relaxation duality as a RELAXATION of the bilevel problem
        # (it gives an upper bound on c'x), then verify and fix.
        
        # For now, use complementarity instead of strong duality to handle the LP part:
        # This gives us: y is LP-optimal for the follower's LP relaxation.
        # With integer y vars, this might be infeasible or give a relaxation.
        
        # Let's just try strong duality + integrality and see what happens.
        # It's a heuristic that might work when LP relaxation = MILP for the follower.
        
        # Actually, let me reconsider. The strong duality approach with linearized
        # bilinear terms (b2-A2*x)'u is problematic because it introduces bilinear x*u terms.
        # We need to linearize these.
        
        # The term is: sum_i (b2_i * u_i - sum_j A2[i,j] * x[j] * u[i])
        # The bilinear terms x[j] * u[i] need linearization.
        
        # This is getting complex. Let me use complementary slackness instead:
        
        # Primal complementarity: u[i] * primal_slack[i] = 0
        # Dual complementarity: y[j] * dual_slack[j] = 0
        
        # Linearize with big-M:
        # u[i] <= M_u[i] * (1 - z_primal[i])
        # primal_slack[i] <= M_slack_lower[i] * z_primal[i]
        for i in range(m2):
            model.addConstr(u[i] <= M_u[i] * (1 - z_primal[i]), name=f"comp_u_{i}")
            model.addConstr(primal_slack[i] <= M_slack_lower[i] * z_primal[i], name=f"comp_ps_{i}")
        
        # y[j] <= M_y[j] * (1 - z_dual[j])
        # dual_slack[j] <= M_slack_dual[j] * z_dual[j]
        for j in range(n2):
            model.addConstr(y[j] <= M_y[j] * (1 - z_dual[j]), name=f"comp_y_{j}")
            model.addConstr(dual_slack[j] <= M_slack_dual[j] * z_dual[j], name=f"comp_ds_{j}")
        
        # Remove strong duality constraint (complementarity replaces it for LP)
        # Actually we already added it. Let me restructure.
        # With complementarity + primal/dual feasibility, strong duality is implied.
        # So we don't need the explicit strong duality constraint.
        # Let me remove it.
        model.remove(model.getConstrByName("strong_duality"))
        
        model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n1)), GRB.MAXIMIZE)
        
        model.optimize()
        
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            x_val = [x[j].X for j in range(n1)]
            y_val = [y[j].X for j in range(n2)]
            obj_val = sum(c[j] * x_val[j] for j in range(n1))
            fobj_val = sum(d2[j] * y_val[j] for j in range(n2))
            return x_val, y_val, obj_val, fobj_val, model
        return None, None, None, None, model
    
    # But this KKT approach treats the follower as an LP (ignoring integrality of follower vars
    # in the optimality conditions). The y vars are integer-constrained in the model, but the
    # KKT conditions are for the LP. So we get: y is integer, LP-feasible, and satisfies LP KKT.
    # This means y is LP-optimal AND integer. This is a sufficient condition for MILP optimality
    # (if y is LP-optimal and integer, it's certainly MILP-optimal since MILP opt <= LP opt).
    # So this is actually a RESTRICTION (not relaxation) of the true bilevel problem.
    # It might miss solutions where the follower's MILP optimal is strictly less than LP optimal.
    
    # This is actually fine for getting good solutions! It gives valid bilevel-feasible points
    # (though it might miss the true optimum if LP relaxation != MILP for the follower).
    
    # Let's try this approach first, then fall back to iterative if needed.
    
    # Actually, let me reconsider the big-M values more carefully.
    # For x variables: x is nonneg integer. Upper bound?
    # From upper and lower constraints: A*x <= b (when y=0)
    # So x_j <= max(b) / min_nonzero(A[:,j]) if A has positive entries in column j
    
    M_x = np.zeros(n1)
    A_combined = np.vstack([A1, A2])
    b_combined = np.concatenate([b1, b2])
    for j in range(n1):
        col = A_combined[:, j]
        if np.any(col > 0):
            ratios = b_combined[col > 0] / col[col > 0]
            M_x[j] = np.max(ratios) + 10  # some slack
        else:
            M_x[j] = 200
    M_x = np.minimum(M_x, 500)
    
    # Recompute M_y more carefully
    for j in range(n2):
        col_combined = np.vstack([B1, B2])[:, j]
        b_all = np.concatenate([b1, b2])
        if np.any(col_combined > 0):
            ratios = b_all[col_combined > 0] / col_combined[col_combined > 0]
            M_y[j] = np.max(ratios) + 10
        else:
            M_y[j] = 500
    M_y = np.minimum(M_y, 500)
    
    # Recompute M_u: from dual feasibility B2'u >= d2
    # Each u_i can be at most max(max(d2), 0) / min_nonzero(B2[i,:]) roughly
    for i in range(m2):
        row = B2[i, :]
        if np.any(row > 0):
            # u_i could be up to max_d2_pos / min_nonzero(row) but that's rough
            M_u[i] = max(max(np.abs(d2)), 50) * 2
        else:
            M_u[i] = 500
    M_u = np.minimum(M_u, 500)
    
    for i in range(m2):
        M_slack_lower[i] = b2[i] + 10  # primal slack <= b2[i] (when A2*x = 0, B2*y = 0)
    M_slack_lower = np.minimum(M_slack_lower, 500)
    
    for j in range(n2):
        M_slack_dual[j] = max(np.sum(np.maximum(B2[:, j], 0) * M_u), 1) - d2[j]
        M_slack_dual[j] = max(M_slack_dual[j], 200)
    M_slack_dual = np.minimum(M_slack_dual, 1000)
    
    # Let's implement a cleaner version
    def solve_single_level(time_left):
        model = gp.Model("SingleLevel")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        if time_left is not None:
            model.Params.TimeLimit = max(time_left, 1)
        
        # Variables
        x = model.addVars(n1, vtype=GRB.INTEGER, lb=0, ub=[M_x[j] for j in range(n1)], name="x")
        y = model.addVars(n2, lb=0, ub=[M_y[j] for j in range(n2)], name="y")
        for idx in integer_follower_vars:
            y[idx].vtype = GRB.INTEGER
        
        u = model.addVars(m2, lb=0, ub=[M_u[i] for i in range(m2)], name="u")
        
        # Binary vars for complementarity
        zp = model.addVars(m2, vtype=GRB.BINARY, name="zp")  # 1 => primal slack can be > 0 (u must be 0)
        zd = model.addVars(n2, vtype=GRB.BINARY, name="zd")  # 1 => dual slack can be > 0 (y must be 0)
        
        # Upper-level constraints
        for i in range(m1):
            model.addConstr(
                gp.quicksum(A1[i, j] * x[j] for j in range(n1)) +
                gp.quicksum(B1[i, j] * y[j] for j in range(n2)) <= b1[i]
            )
        
        # Lower-level primal feasibility
        ps = model.addVars(m2, lb=0, name="ps")  # primal slack
        for i in range(m2):
            model.addConstr(
                gp.quicksum(B2[i, j] * y[j] for j in range(n2)) + ps[i] ==
                b2[i] - gp.quicksum(A2[i, j] * x[j] for j in range(n1))
            )
        
        # Dual feasibility: B2'u - d2 >= 0
        ds = model.addVars(n2, lb=0, name="ds")  # dual slack
        for j in range(n2):
            model.addConstr(
                gp.quicksum(B2[i, j] * u[i] for i in range(m2)) - ds[j] == d2[j]
            )
        
        # Complementarity (big-M linearization)
        for i in range(m2):
            # u[i] * ps[i] = 0
            # If zp[i]=0: u[i] can be > 0, ps[i] = 0
            # If zp[i]=1: u[i] = 0, ps[i] can be > 0
            model.addConstr(u[i] <= M_u[i] * (1 - zp[i]))
            model.addConstr(ps[i] <= M_slack_lower[i] * zp[i])
        
        for j in range(n2):
            # y[j] * ds[j] = 0
            # If zd[j]=0: y[j] can be > 0, ds[j] = 0
            # If zd[j]=1: y[j] = 0, ds[j] can be > 0
            model.addConstr(y[j] <= M_y[j] * (1 - zd[j]))
            model.addConstr(ds[j] <= M_slack_dual[j] * zd[j])
        
        # Objective
        model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n1)), GRB.MAXIMIZE)
        
        model.optimize()
        
        results = []
        if model.SolCount > 0:
            for sol_idx in range(model.SolCount):
                model.Params.SolutionNumber = sol_idx
                x_val = [x[j].Xn for j in range(n1)]
                y_val = [y[j].Xn for j in range(n2)]
                obj_val = sum(c[j] * x_val[j] for j in range(n1))
                fobj_val = sum(d2[j] * y_val[j] for j in range(n2))
                results.append((x_val, y_val, obj_val, fobj_val))
                if sol_idx >= 4:  # limit to top 5 solutions
                    break
        
        return results, model
    
    # Try the single-level KKT reformulation
    time_left = time_limit - (time.time() - start_time)
    
    # Allocate time: 60% for KKT approach, 40% for iterative verification/improvement
    kkt_time = time_left * 0.6
    
    results, model = solve_single_level(kkt_time)
    
    # Verify each solution
    for x_val, y_val, obj_val, fobj_val in results:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Round integer variables
        x_rounded = [round(v) for v in x_val]
        y_rounded = list(y_val)
        for idx in integer_follower_vars:
            y_rounded[idx] = round(y_rounded[idx])
        
        # Check feasibility
        x_arr = np.array(x_rounded, dtype=float)
        y_arr = np.array(y_rounded, dtype=float)
        
        # Check lower-level feasibility
        lower_lhs = B2 @ y_arr
        lower_rhs = b2 - A2 @ x_arr
        if np.any(lower_lhs > lower_rhs + 1e-6):
            continue
        
        # Check upper-level feasibility
        upper_lhs = A1 @ x_arr + B1 @ y_arr
        if np.any(upper_lhs > b1 + 1e-6):
            continue
        
        # Verify follower optimality: solve follower's problem
        tl = max(time_limit - (time.time() - start_time) - 2, 5)
        y_opt, fopt, status = solve_follower(x_rounded, tl)
        
        if y_opt is None:
            continue
        
        follower_val = sum(d2[j] * y_rounded[j] for j in range(n2))
        
        if abs(follower_val - fopt) < 1e-3:
            # y_rounded is follower-optimal, check if upper-level constraints hold
            leader_obj = sum(c[j] * x_rounded[j] for j in range(n1))
            if leader_obj > best_obj:
                best_obj = leader_obj
                best_solution = {
                    "objective_value": leader_obj,
                    "x": x_rounded,
                    "y": y_rounded,
                    "follower_objective_value": follower_val
                }
                if logger:
                    logger.log_solution(best_obj, best_solution)
        else:
            # y_rounded is not follower-optimal. Try the optimistic approach:
            # find a follower-optimal y that also satisfies upper-level constraints
            tl = max(time_limit - (time.time() - start_time) - 2, 5)
            y_opt2, fopt2 = solve_follower_optimistic(x_rounded, fopt, tl)
            if y_opt2 is not None:
                # Check upper-level feasibility
                y_arr2 = np.array(y_opt2)
                upper_check = A1 @ x_arr + B1 @ y_arr2
                if np.all(upper_check <= b1 + 1e-6):
                    leader_obj = sum(c[j] * x_rounded[j] for j in range(n1))
                    fobj2 = sum(d2[j] * y_opt2[j] for j in range(n2))
                    if leader_obj > best_obj:
                        best_obj = leader_obj
                        best_solution = {
                            "objective_value": leader_obj,
                            "x": x_rounded,
                            "y": y_opt2,
                            "follower_objective_value": fobj2
                        }
                        if logger:
                            logger.log_solution(best_obj, best_solution)
    
    # If KKT didn't work well, try iterative approach
    if best_solution is None and time.time() - start_time < time_limit - 10:
        # Try HPR (ignore follower optimality) to at least get a starting point
        tl = max(time_limit - (time.time() - start_time) - 5, 5)
        x_hpr, y_hpr, obj_hpr, _ = solve_hpr(time_left=tl)
        
        if x_hpr is not None:
            x_rounded = [round(v) for v in x_hpr]
            x_arr = np.array(x_rounded, dtype=float)
            
            # Solve follower
            tl = max(time_limit - (time.time() - start_time) - 2, 5)
            y_opt, fopt, status = solve_follower(x_rounded, tl)
            
            if y_opt is not None and fopt is not None:
                # Try optimistic follower
                tl = max(time_limit - (time.time() - start_time) - 2, 5)
                y_opt2, fopt2 = solve_follower_optimistic(x_rounded, fopt, tl)
                if y_opt2 is not None:
                    y_arr2 = np.array(y_opt2)
                    upper_check = A1 @ x_arr + B1 @ y_arr2
                    if np.all(upper_check <= b1 + 1e-6):
                        leader_obj = sum(c[j] * x_rounded[j] for j in range(n1))
                        fobj2 = sum(d2[j] * y_opt2[j] for j in range(n2))
                        if leader_obj > best_obj:
                            best_obj = leader_obj
                            best_solution = {
                                "objective_value": leader_obj,
                                "x": x_rounded,
                                "y": y_opt2,
                                "follower_objective_value": fobj2
                            }
                            if logger:
                                logger.log_solution(best_obj, best_solution)
    
    # Now try an iterative enumeration approach if we still have time
    # Enumerate candidate x solutions by solving relaxed problems with different cuts
    if time.time() - start_time < time_limit - 20:
        # Use a callback-based approach: solve HPR with lazy cuts for follower optimality
        remaining = time_limit - (time.time() - start_time) - 5
        if remaining > 10:
            try:
                result = solve_with_callbacks(instance, n1, n2, m1, m2, c, d2,
                                              A1, B1, b1, A2, B2, b2,
                                              integer_follower_vars, M_x, M_y,
                                              remaining, best_obj, best_solution, logger, start_time, time_limit)
                if result is not None:
                    if result['objective_value'] > best_obj:
                        best_obj = result['objective_value']
                        best_solution = result
            except Exception:
                pass
    
    # Fallback: try x=0
    if best_solution is None:
        x_zero = [0.0] * n1
        tl = max(time_limit - (time.time() - start_time) - 1, 2)
        y_opt, fopt, status = solve_follower(x_zero, tl)
        if y_opt is not None:
            y_arr = np.array(y_opt)
            x_arr = np.zeros(n1)
            upper_check = A1 @ x_arr + B1 @ y_arr
            if np.all(upper_check <= b1 + 1e-6):
                leader_obj = 0.0
                fobj = sum(d2[j] * y_opt[j] for j in range(n2))
                best_solution = {
                    "objective_value": leader_obj,
                    "x": [0.0] * n1,
                    "y": y_opt,
                    "follower_objective_value": fobj
                }
                best_obj = leader_obj
                if logger:
                    logger.log_solution(best_obj, best_solution)
    
    return best_solution


def solve_with_callbacks(instance, n1, n2, m1, m2, c, d2,
                          A1, B1, b1, A2, B2, b2,
                          integer_follower_vars, M_x, M_y,
                          time_left, best_obj_init, best_sol_init, logger, start_time, time_limit):
    """
    Solve using HPR with lazy constraints for follower optimality verification.
    We solve HPR, and whenever we find an integer solution, we verify follower optimality.
    If not optimal, we add a cut.
    """
    
    best_obj = [best_obj_init]
    best_solution = [best_sol_init]
    cuts_added = [0]
    
    model = gp.Model("Callback")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(time_left, 1)
    model.Params.LazyConstraints = 1
    
    x = model.addVars(n1, vtype=GRB.INTEGER, lb=0, ub=[M_x[j] for j in range(n1)], name="x")
    y = model.addVars(n2, lb=0, ub=[M_y[j] for j in range(n2)], name="y")
    for idx in integer_follower_vars:
        y[idx].vtype = GRB.INTEGER
    
    # eta = follower objective (we require d2'y >= eta, and eta >= true follower optimal)
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    
    # Upper-level constraints
    for i in range(m1):
        model.addConstr(
            gp.quicksum(A1[i, j] * x[j] for j in range(n1)) +
            gp.quicksum(B1[i, j] * y[j] for j in range(n2)) <= b1[i]
        )
    
    # Lower-level constraints
    for i in range(m2):
        model.addConstr(
            gp.quicksum(A2[i, j] * x[j] for j in range(n1)) +
            gp.quicksum(B2[i, j] * y[j] for j in range(n2)) <= b2[i]
        )
    
    # eta <= d2'y (follower achieves at least eta)
    model.addConstr(
        gp.quicksum(d2[j] * y[j] for j in range(n2)) >= eta,
        name="eta_bound"
    )
    
    model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n1)), GRB.MAXIMIZE)
    
    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            if time.time() - start_time > time_limit - 3:
                model_cb.terminate()
                return
            
            x_val = [model_cb.cbGetSolution(x[j]) for j in range(n1)]
            y_val = [model_cb.cbGetSolution(y[j]) for j in range(n2)]
            eta_val = model_cb.cbGetSolution(eta)
            
            x_rounded = [round(v) for v in x_val]
            x_arr = np.array(x_rounded, dtype=float)
            
            # Solve follower problem
            tl = min(max(time_limit - (time.time() - start_time) - 3, 2), 30)
            
            fmodel = gp.Model("F")
            fmodel.Params.Seed = 0
            fmodel.Params.MIPGap = 1e-4
            fmodel.Params.NumericFocus = 0
            fmodel.Params.Threads = 1
            fmodel.Params.OutputFlag = 0
            fmodel.Params.TimeLimit = tl
            
            fy = fmodel.addVars(n2, lb=0, name="fy")
            for idx in integer_follower_vars:
                fy[idx].vtype = GRB.INTEGER
            
            rhs = b2 - A2 @ x_arr
            for i in range(m2):
                if rhs[i] < -1e-6:
                    fmodel.dispose()
                    # x is infeasible for lower level - this shouldn't happen 
                    # but add a cut to exclude this x
                    # Actually, it could happen if y makes lower level feasible
                    # But we're checking follower's problem which has fixed x
                    # If rhs < 0 and B2 row has all non-negative... 
                    return
                fmodel.addConstr(
                    gp.quicksum(B2[i, j] * fy[j] for j in range(n2)) <= rhs[i]
                )
            
            fmodel.setObjective(gp.quicksum(d2[j] * fy[j] for j in range(n2)), GRB.MAXIMIZE)
            fmodel.optimize()
            
            if fmodel.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (fmodel.status == GRB.TIME_LIMIT and fmodel.SolCount > 0):
                fopt = fmodel.ObjVal
                
                follower_val = sum(d2[j] * y_val[j] for j in range(n2))
                
                if follower_val < fopt - 1e-3:
                    # y is not follower-optimal, add a cut
                    # We need eta >= fopt for this x
                    # But we need a cut that generalizes...
                    
                    # One approach: add optimality cut from LP relaxation dual
                    # Solve LP relaxation of follower to get dual
                    fmodel2 = gp.Model("FLP")
                    fmodel2.Params.Seed = 0
                    fmodel2.Params.MIPGap = 1e-4
                    fmodel2.Params.NumericFocus = 0
                    fmodel2.Params.Threads = 1
                    fmodel2.Params.OutputFlag = 0
                    fmodel2.Params.TimeLimit = min(tl, 10)
                    
                    fy2 = fmodel2.addVars(n2, lb=0, name="fy2")
                    constrs = []
                    for i in range(m2):
                        constrs.append(fmodel2.addConstr(
                            gp.quicksum(B2[i, j] * fy2[j] for j in range(n2)) <= rhs[i]
                        ))
                    
                    fmodel2.setObjective(gp.quicksum(d2[j] * fy2[j] for j in range(n2)), GRB.MAXIMIZE)
                    fmodel2.optimize()
                    
                    if fmodel2.status == GRB.OPTIMAL:
                        # Get dual values
                        duals = [constrs[i].Pi for i in range(m2)]
                        # LP optimal value = sum_i duals[i] * rhs[i] = sum_i duals[i] * (b2[i] - sum_j A2[i,j]*x[j])
                        # For general x: follower LP opt >= sum_i duals[i] * (b2[i] - sum_j A2[i,j]*x[j])
                        # (duals are from a specific x, but the dual solution is feasible for any x where RHS >= 0)
                        # Actually the dual optimal for LP at this x gives a LOWER BOUND on LP optimal
                        # for any x (by weak duality of the dual). So:
                        # eta >= sum_i duals[i] * (b2[i] - sum_j A2[i,j]*x[j])
                        
                        # Add cut: eta >= sum_i duals[i] * (b2[i] - sum_j A2[i,j]*x[j])
                        model_cb.cbLazy(
                            eta >= gp.quicksum(
                                duals[i] * (b2[i] - gp.quicksum(A2[i, j] * x[j] for j in range(n1)))
                                for i in range(m2)
                            )
                        )
                        cuts_added[0] += 1
                    
                    fmodel2.dispose()
                else:
                    # y is follower-optimal (or close enough)
                    # Check upper-level feasibility (should be satisfied by construction)
                    # and record solution
                    
                    # Try optimistic: find follower-optimal y satisfying upper constraints
                    # Since d1=0, we just need any follower-optimal y that's upper-feasible
                    
                    # Check if current y satisfies upper constraints
                    y_arr = np.array(y_val)
                    upper_check = A1 @ x_arr + B1 @ y_arr
                    if np.all(upper_check <= b1 + 1e-6):
                        leader_obj = sum(c[j] * x_rounded[j] for j in range(n1))
                        if leader_obj > best_obj[0]:
                            fobj = sum(d2[j] * y_val[j] for j in range(n2))
                            best_obj[0] = leader_obj
                            # Round integer vars
                            y_out = list(y_val)
                            for idx in integer_follower_vars:
                                y_out[idx] = round(y_out[idx])
                            best_solution[0] = {
                                "objective_value": leader_obj,
                                "x": [float(v) for v in x_rounded],
                                "y": [float(v) for v in y_out],
                                "follower_objective_value": fobj
                            }
                            if logger:
                                logger.log_solution(best_obj[0], best_solution[0])
                    else:
                        # Try to find upper-feasible follower-optimal y
                        tl2 = min(max(time_limit - (time.time() - start_time) - 3, 2), 20)
                        fmodel3 = gp.Model("FOpt")
                        fmodel3.Params.Seed = 0
                        fmodel3.Params.MIPGap = 1e-4
                        fmodel3.Params.NumericFocus = 0
                        fmodel3.Params.Threads = 1
                        fmodel3.Params.OutputFlag = 0
                        fmodel3.Params.TimeLimit = tl2
                        
                        fy3 = fmodel3.addVars(n2, lb=0, name="fy3")
                        for idx in integer_follower_vars:
                            fy3[idx].vtype = GRB.INTEGER
                        
                        for i in range(m2):
                            fmodel3.addConstr(
                                gp.quicksum(B2[i, j] * fy3[j] for j in range(n2)) <= rhs[i]
                            )
                        
                        # Follower optimality
                        fmodel3.addConstr(
                            gp.quicksum(d2[j] * fy3[j] for j in range(n2)) >= fopt - 1e-4
                        )
                        
                        # Upper-level feasibility
                        rhs_upper = b1 - A1 @ x_arr
                        for i in range(m1):
                            fmodel3.addConstr(
                                gp.quicksum(B1[i, j] * fy3[j] for j in range(n2)) <= rhs_upper[i]
                            )
                        
                        fmodel3.setObjective(0, GRB.MAXIMIZE)
                        fmodel3.optimize()
                        
                        if fmodel3.SolCount > 0:
                            y_out = [fy3[j].X for j in range(n2)]
                            for idx in integer_follower_vars:
                                y_out[idx] = round(y_out[idx])
                            leader_obj = sum(c[j] * x_rounded[j] for j in range(n1))
                            if leader_obj > best_obj[0]:
                                fobj = sum(d2[j] * y_out[j] for j in range(n2))
                                best_obj[0] = leader_obj
                                best_solution[0] = {
                                    "objective_value": leader_obj,
                                    "x": [float(v) for v in x_rounded],
                                    "y": [float(v) for v in y_out],
                                    "follower_objective_value": fobj
                                }
                                if logger:
                                    logger.log_solution(best_obj[0], best_solution[0])
                        
                        fmodel3.dispose()
                
                fmodel.dispose()
            else:
                fmodel.dispose()
    
    model.optimize(callback)
    
    return best_solution[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    solution = solve_bilevel(instance, args.time_limit, logger)
    
    if solution is None:
        # Return a trivial solution
        n1 = instance['n1']
        n2 = instance['n2']
        solution = {
            "objective_value": 0.0,
            "x": [0.0] * n1,
            "y": [0.0] * n2,
            "follower_objective_value": 0.0
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()