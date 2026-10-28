import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    instance_type = instance['instance_type']
    n = instance['n']
    m = instance['m']
    k = instance['k']
    gamma = instance['gamma']
    data_matrix = instance['data_matrix']
    node_weights = instance['node_weights']
    element_weights = instance['element_weights']

    # Precompute f(S) for small n, but for general, we'll use MIP
    # Since n can be up to say 100, we need MIP approach for cardinality constrained maximization

    model = gp.Model("NonSubmodularMax")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)  # Allow some gap for speed
    model.setParam('LogToConsole', 0)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Cardinality
    model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k, "card")

    if instance_type == "LOC":
        # Facility Location: sum over clients max over selected
        y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")
        for i in range(m):
            model.addConstr(y[i] <= gp.quicksum(data_matrix[i][j] * x[j] for j in range(n)), f"client_{i}")
            # Since benefits nonnegative and non-decreasing, max is captured by this with obj maximization
        objective = gp.quicksum(y[i] for i in range(m))
        model.setObjective(objective, GRB.MAXIMIZE)

    elif instance_type == "COV":
        # Weighted Coverage: sum weight_i * (1 if covered)
        z = model.addVars(m, vtype=GRB.BINARY, name="z")
        for i in range(m):
            if node_weights[i] > 1e-9:
                covered = gp.quicksum(data_matrix[i][j] * x[j] for j in range(n))
                model.addConstr(z[i] <= covered, f"cover_{i}")
                # Since binary and max sense, z[i] will be 1 if possible
        objective = gp.quicksum(node_weights[i] * z[i] for i in range(m))
        model.setObjective(objective, GRB.MAXIMIZE)

    elif instance_type == "INF":
        # Bipartite Influence: 1 - product (1-p) for selected
        # This is harder: use piecewise linear or additional constraints
        # Since it's monotone submodular-like but with gamma, we use MIP with log transform or auxiliary
        # For exact, introduce t_i for each target, maximize sum t_i where t_i <= 1 - prod (1-p_ij * xj)
        # Equivalent to t_i <= 1 - exp( sum log(1-p) * xj )
        # But to linearize, we can use McCormick or just use continuous approx but better use MIP with binary expansion or accept solver handles bilinear? No.
        # Actually for maximization of sum (1 - prod (1 - p_j x_j)), it's a standard problem but nonlinear.
        # Since we have gamma, but to get exact objective, we need correct f.
        # For exact, we can introduce auxiliary variables for the product.
        # Let u_i = prod_{j} (1 - p_{ij} x_j), then obj = sum (1 - u_i)
        # But product is nonlinear. Since x binary, we can use logarithmic formulation or dynamic inequalities but complicated.
        # Given time, for INF we will use a greedy approach separately if needed.
        # But since Gurobi can handle it with piecewise linear approximation or we can use conic but let's use exact MIP with additional binaries for each possible.
        # Since m and n not specified but likely small, but to be safe let's implement different for each.

        # For INF, use formulation with additional variables for marginal
        # But to keep it simple and efficient, let's first try to use a unified greedy approach that respects gamma.
        # The problem mentions gamma bar as upper bound on submodular ratio.
        # Actually, there is a standard algorithm for gamma-weakly submodular maximization under cardinality.
        # But since we have Gurobi, for LOC and COV it's linearizable exactly, for INF it's concave like.
        # Let's check typical sizes. Since single core and time limit, likely n<=100.
        # For INF, the function is monotone but not necessarily submodular, but approximately.
        # To get exact objective, we need to compute f exactly in the end anyway.
        # A good approach is to use MIP for LOC and COV which are easy, for INF we use greedy with CELF like considering gamma.

    # To make a robust solver that works for all, we will implement a MIP where possible, and for INF use a formulation with linearized product using additional variables.
    # But it can be expensive. Since the problem gives gamma, perhaps they expect a gamma-respecting greedy algorithm.

    # Let's think again: the problem is to maximize f(S) where |S|<=k and f is non-decreasing approximately submodular with given gamma_bar.
    # A known approximation is (1 - e^{-gamma}) / gamma approx but here we need to solve as well as possible.
    # Since Gurobi is allowed, and for exact computation of objective, we need to be able to compute f(S) anyway for any solution.
    # So let's implement the MIP for LOC and COV exactly, for INF we will implement an iterative greedy that logs improvements and is fast.
    # But to unify, let's first solve with a greedy algorithm that is fast and effective for all types, and use MIP when it makes sense.

    # Better idea: since we can compute f(S) given any S for all types (we can write a function to evaluate), we can use a MIP with lazy constraints or just use heuristic.
    # But for efficiency, let's use the following approach:
    # Use Gurobi to solve a linearized MIP where possible, and for INF use a good MIP formulation.

    def build_mip():
        if instance_type == "LOC":
            y = model.addVars(m, lb=0.0, name="y")
            for i in range(m):
                for j in range(n):
                    if data_matrix[i][j] > 0:
                        model.addConstr(y[i] <= data_matrix[i][j] * x[j] + (1 - x[j]) * max([data_matrix[i][jj] for jj in range(n)]), f"bigM_{i}_{j}") # but actually simpler:
            # Correct simple way for max:
            for i in range(m):
                model.addConstr(y[i] <= gp.quicksum(data_matrix[i][j] * x[j] for j in range(n)))
                # But to make it tight, since obj max, y_i will be = min( max_benefit, sum ... but since one at a time it works? No, because sum of benefits is not max.
            # Wrong! For facility location, the benefit for client i is the MAX benefit from any open facility, not the sum.
            # So correct is:
            # y[i] <= max possible but better use:
            for i in range(m):
                model.addConstr(y[i] <= gp.quicksum(data_matrix[i][j] * x[j] for j in range(n)))  # this is upper bound by sum but since we max y and if multiple, it overestimates!!
            # This is incorrect. If two facilities give benefit 5 and 6 to same client, sum would allow y=11 but should be 6!!
            # So correct formulation is:
            y = model.addVars(m, lb=0.0, name="y")
            for i in range(m):
                # y[i] <= max_j benefit_ij * xj  i.e. y[i] is <= every opened? No.
                # Standard is:
                # y[i] <= max_j (benefit[i][j] * x[j]) but that's not linear.
                # To linearize max, introduce y_i <= benefit[i][j] + M*(1-x[j]) for all? No.
                # Since it's max_j b_ij * x_j , we can set y_i <= sum_j b_ij * x_j is loose as above.
                # The correct way is y_i <= b_ij * x_j + M_i * (1 - x_j) but to force it to the max, it's:
                # For each client i, y[i] <= max_{j} b[i][j]   (upper bound)
                # and for each j, y[i] <= b[i][j] + (max_benefit_i - b[i][j]) * (1 - x[j]) but it's still not forcing.
                # Standard MILP for uncapacitated facility location for this objective:
                # Actually since the objective is sum_i max_{j in S} b_ij , it can be modeled as:
                # max sum_i y_i
                # y_i <= sum_j b_ij x_j     <-- this is incorrect as I said.
                # No:
                # To model y = max_j in S b_j , we do:
                # y <= b_j + M (1-x_j) for each j, and y <= max_b , but that forces y <= all selected b's +M but to get the lower bound in maximization we also need to have the option.
                # The standard way is:
                # For maximization of sum max, we use:
                # y_i <= sum_j b[i][j] * x[j]   wait no as before it overcounts if b additive but here it's not additive, it's MAX.
                # If b are the benefits, the value for a client is the best one.
                # So to model y_i = max_{j opened} b[i j ]
                # We can introduce:
                # y_i <= max_benefit_i for all i
                # For every j, y_i >= b[i][j] * x[j]     # forces y_i to be at least b if that j is selected. Since max sense, it will take the largest such lower bound.
                # Yes! That's it. Because obj maximizes y, it will be pushed up to the highest b of selected x's.
                # And since it's <= the global max implicitly or we can add y_i <= max_j b[i j ] but since the lower bounds are only from selected, and no upper other than that, but if no x selected y=0 is ok because lower bounds are 0.
                # If multiple x selected, y_i >= b1, y_i >= b2, so y >= max of those. And since maximizing, and if we don't have conflicting, it's perfect.
                # We need no upper bound constraint because the max possible is handled by the data.
                # Yes, this works beautifully because there is no penalty for larger y, so it will equal the max of the b's where x_j=1.
                # Perfect.

            # So let's correct:
            if instance_type == "LOC":
                y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
                max_b = [max(row) for row in data_matrix]
                for i in range(m):
                    model.addConstr(y[i] <= max_b[i], f"ub_y_{i}")
                    for j in range(n):
                        if data_matrix[i][j] > 0:
                            model.addConstr(y[i] >= data_matrix[i][j] * x[j], f"lb_y_{i}_{j}")
                objective = gp.quicksum(y[i] for i in range(m))
                model.setObjective(objective, GRB.MAXIMIZE)

        elif instance_type == "COV":
            z = model.addVars(m, vtype=GRB.BINARY, name="z")
            for i in range(m):
                # z_i = 1 if at least one sensor covering it is selected
                if node_weights[i] > 1e-9:
                    model.addConstr(z[i] <= gp.quicksum(data_matrix[i][j] * x[j] for j in range(n)), f"cover_constr_{i}")
                    # Since z binary and obj max, it will be forced to 1 if sum>=1
            objective = gp.quicksum(node_weights[i] * z[i] for i in range(m))
            model.setObjective(objective, GRB.MAXIMIZE)

        elif instance_type == "INF":
            # For INF, value = sum over targets ( 1 - prod_j (1 - p_ij * x_j) )
            # p_ij = element_weights[j] if data_matrix[i][j]==1 else 0
            # This is 1 - prod (1-p) over selected neighbors.
            # To model exactly in MIP is possible but needs either many variables or approximation.
            # One exact way is to use additional continuous var u_i for the survival prob (product), then u_i <= 1, u_i >=0
            # But product of (1 - p_j x_j).
            # Since x binary, we can linearize by taking log but it's not linear.
            # A common way is to use piecewise linear approximation for the function 1 - product, but since n<=100 perhaps but per target it's 2^n impossible.
            # So for practicality, for INF we will solve using a greedy algorithm that is known to work well for influence maximization like problems.
            # But since we want to use MIP as much as possible, let's implement exact MIP for LOC and COV, and for INF we'll use a MIP with log transformed but using Gurobi's general constraints or piecewise.
            # Gurobi has general constraints for product but for many it's not great.
            # Since time limit is given and single core, likely instances are small enough or expect heuristic.
            # To make it simple and efficient across all, let's implement a greedy algorithm with marginal gains, adjusted by gamma, that runs fast and provides good solutions, and use logger to record improvements.
            # This is likely the intended as the problem emphasizes gamma bar.
            # The gamma is an upper bound on the submodular ratio, so the function is gamma-weakly submodular.
            # There are algorithms with guarantees (1 - e^{-gamma k} or something).
            # But for practical solution, greedy by marginal gain / cost but here uniform.
            # So we will implement the standard greedy for all types, but to get better quality, we can run it and also try to improve with local search.
            # But since we have Gurobi, let's do MIP for LOC and COV, and for INF implement a MIP using additional variables for each possible "incremental" but that's hard.
            # For INF, we can use the fact that 1 - prod (1-p_j for j in S) = 1 - exp( sum log(1-p) for selected), and use piecewise linear on the sum of logs.
            # This is doable.
            # Let's implement it.

            if instance_type == "INF":
                # For each target i, compute p[j] = element_weights[j] if data_matrix[i][j] else 0
                # Then for each i, let s_i = sum_j -log(1 - p_ij) * x_j   (note -log(1-p) >0 )
                # Then u_i = exp(-s_i), then prob = 1 - u_i = 1 - exp(-s_i)
                # To maximize sum (1 - exp(-s_i)), we can approximate the concave function 1-exp(-s) with piecewise linear.
                # This gives a good approximation.
                # Let's define segments for piecewise.
                # Since p <=1, but practically p small? The type_specific_param is edge prob.
                # To be accurate let's make 10 segments or so.
                num_segments = 20
                # But to make exact within time, perhaps better to use dynamic programming per target if degree small but unknown.
                # Alternative: since the problem allows any solution approach, and aims for close to optimal, let's use a greedy algorithm for all instance types.
                # It is very fast, works for all, and with local search can improve.
                # This is practical given the single CPU constraint.

    # Let's switch to a robust approach that works for all: Greedy + local search (swap) within time limit.
    # This will be efficient and give good quality.

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    best_obj = 0.0
    best_S = []

    # Function to compute f(S)
    def compute_f(S):
        if not S:
            return 0.0
        S_set = set(S)
        if instance_type == "LOC":
            val = 0.0
            for i in range(m):
                max_ben = 0.0
                for j in S:
                    max_ben = max(max_ben, data_matrix[i][j])
                val += max_ben
            return val
        elif instance_type == "COV":
            val = 0.0
            for i in range(m):
                covered = False
                for j in S:
                    if data_matrix[i][j] > 0.5:
                        covered = True
                        break
                if covered:
                    val += node_weights[i]
            return val
        elif instance_type == "INF":
            val = 0.0
            for i in range(m):
                prod = 1.0
                for j in S:
                    if data_matrix[i][j] > 0.5:
                        p = element_weights[j]
                        prod *= (1.0 - p)
                val += (1.0 - prod)
            return val
        return 0.0

    # Greedy algorithm
    current_S = []
    remaining = set(range(n))
    marginal_cache = {}  # for CELF like

    while len(current_S) < k and remaining and (time.time() - start_time < args.time_limit):
        best_gain = -1.0
        best_elem = None
        for j in list(remaining):
            gain = compute_f(current_S + [j]) - compute_f(current_S)
            if gain > best_gain:
                best_gain = gain
                best_elem = j
        if best_gain <= 0:
            break
        current_S.append(best_elem)
        remaining.remove(best_elem)
        current_obj = compute_f(current_S)
        if current_obj > best_obj + 1e-6:
            best_obj = current_obj
            best_S = list(current_S)
            if logger:
                logger.log(best_obj)
            # print(f"Improved to {best_obj} at size {len(best_S)}", file=sys.stderr)

    # Now do local search: try to swap elements to improve
    improved = True
    while improved and (time.time() - start_time < args.time_limit * 0.95):
        improved = False
        current_obj = compute_f(best_S)
        for i in range(len(best_S)):
            for new_e in range(n):
                if new_e not in best_S:
                    new_set = best_S[:i] + best_S[i+1:] + [new_e]
                    new_obj = compute_f(new_set)
                    if new_obj > best_obj + 1e-6:
                        best_S = new_set
                        best_obj = new_obj
                        improved = True
                        if logger:
                            logger.log(best_obj)
                        break
            if improved:
                break
        # Also try removing one if beneficial but since non-decreasing unlikely
        # To speed up we can limit the number of swaps

    # Final improvement using MIP on a restricted set if time allows, but for simplicity we stop here.
    # For LOC and COV we can solve exactly with MIP since they are easy to formulate and n probably small.

    if instance_type in ["LOC", "COV"] and (time.time() - start_time < args.time_limit * 0.8):
        # Solve exactly with MIP if time remains
        try:
            model = gp.Model("Exact")
            model.setParam('TimeLimit', max(5, int(args.time_limit - (time.time() - start_time))))
            model.setParam('Threads', 1)
            model.setParam('LogToConsole', 0)
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k)

            if instance_type == "LOC":
                y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
                max_b_all = [max(data_matrix[i]) for i in range(m)]
                for i in range(m):
                    model.addConstr(y[i] <= max_b_all[i])
                    for j in range(n):
                        if data_matrix[i][j] > 1e-9:
                            model.addConstr(y[i] >= data_matrix[i][j] * x[j])
                obj_expr = gp.quicksum(y[i] for i in range(m))
            else:  # COV
                z = model.addVars(m, vtype=GRB.BINARY, name="z")
                for i in range(m):
                    if node_weights[i] > 1e-9:
                        model.addConstr(z[i] <= gp.quicksum(data_matrix[i][j] * x[j] for j in range(n)))
                obj_expr = gp.quicksum(node_weights[i] * z[i] for i in range(m))

            model.setObjective(obj_expr, GRB.MAXIMIZE)
            model.optimize()

            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
                if model.objVal > best_obj + 1e-5:
                    best_obj = model.objVal
                    best_S = [j for j in range(n) if x[j].X > 0.5]
                    if logger:
                        logger.log(best_obj)
        except Exception:
            # If Gurobi fails for any reason, stick with heuristic solution
            pass

    # For INF, the greedy + swap is quite good for influence like objectives.

    # Output
    solution = {
        "objective_value": float(best_obj),
        "selected_elements": [int(j) for j in sorted(best_S)]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final log
    if logger:
        logger.log(best_obj)  # make sure best is logged

if __name__ == "__main__":
    main()