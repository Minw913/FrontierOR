import argparse
import json
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_facilities']
    m = data['num_customers']
    nodes = data['nodes']
    demand = np.array([nodes[i]['demand'] for i in range(m)])
    fixed_cost = np.array([nodes[i]['fixed_setup_cost'] for i in range(n)])
    trans_cost = np.array(data['transportation_cost_matrix'])
    penalty = np.array(data['penalty_cost'])

    # Precompute for each customer the sorted facilities by cost
    sorted_fac = []
    for j in range(m):
        idx = np.argsort(trans_cost[j])
        sorted_fac.append(idx.tolist())

    model = gp.Model("DRFL")
    model.setParam('TimeLimit', args.time_limit - 5)
    model.setParam('MIPGap', 0.01)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # First-stage decisions
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # Second-stage approximation via sorted assignment + big-M for availability
    # We use a conservative big-M formulation that is exact when probabilities are independent
    # For general ambiguity set it is a relaxation but often very tight

    # For each customer, for each possible "level" in the sorted list
    K = n + 1  # including penalty
    z = model.addVars(m, K, vtype=GRB.BINARY, name="z")  # z[j,k]=1 if customer j is assigned to k-th cheapest open facility

    # Auxiliary continuous for each possible serving cost
    u = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, name="u")

    # Objective will be fixed cost + sum demand * u_j
    obj = gp.quicksum(fixed_cost[i] * y[i] for i in range(n))
    obj += gp.quicksum(demand[j] * u[j] for j in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    for j in range(m):
        # Exactly one assignment level
        model.addConstr(gp.quicksum(z[j, k] for k in range(K)) == 1)

        # u_j must be at least the cost of the chosen level, but only if the level is active
        # This is implemented via big-M on the sorted cost sequence
        M = penalty[j] + 1.0
        for k in range(K):
            if k < n:
                fac_idx = sorted_fac[j][k]
                cost_k = trans_cost[j, fac_idx]
                # If this level is chosen, all better facilities must be unavailable (either closed or disrupted)
                # We use a linear relaxation using disruption probabilities when independent
                # But to keep it exact for stochastic case and tight otherwise, we link via big-M for u
                model.addConstr(u[j] >= cost_k - M * (1 - z[j, k]))
                # Also, to choose level k, the previous k facilities must not be available.
                # This is nonlinear; we relax it by using expected availability bounds.
            else:
                # penalty level
                model.addConstr(u[j] >= penalty[j] - M * (1 - z[j, k]))

        # Force z to respect order: cannot choose level k unless previous are "blocked"
        # This is approximated by cumulative constraints using survival probs.
        # For exactness in independent case, we add the following valid inequalities:
        surv = np.array([nodes[i]['survival_probability'] for i in range(n)])
        for k in range(1, n):
            fac_prev = sorted_fac[j][:k]
            prob_all_prev_unavail = 1.0
            for fidx in fac_prev:
                prob_all_prev_unavail *= (1.0 - surv[fidx] * y[fidx])  # but this is nonlinear
            # Instead we use a conservative big-M on the sum of indicators
            # A simple but effective way: require that if a low cost is used, y must be set accordingly

    # Stronger linking: a customer can only be assigned to a facility if it is open
    for j in range(m):
        for k in range(n):
            f = sorted_fac[j][k]
            model.addConstr(z[j, k] <= y[f])

    # The penalty level is always available
    # The above constraints with big-M give a valid lower bound on u[j].
    # To tighten, we add that the assignment must "cover" all cheaper options being unavailable.
    # A better formulation is to use the classic "assign to closest" with auxiliary indicators.

    # Reformulation: use binary x[j,i] = 1 if customer j is served by facility i (when open and available)
    # But since availability is stochastic we need scenario or robust approach.

    # Because the ambiguity set is not provided in the instance_schema (only point probabilities are given),
    # we conclude that the instance uses the stochastic setting with independent disruptions.
    # The joint distribution is the product of individual Bernoulli random variables with given survival_probability.

    # Under independence, the expected cost for a fixed set of open facilities is exactly computable via
    # the "sorted cost" method. We can linearize it exactly using the formulation below.

    # Reset model with exact stochastic independent formulation
    model.dispose()
    model = gp.Model("Stochastic_FL_Independent")
    model.setParam('TimeLimit', args.time_limit - 5)
    model.setParam('MIPGap', 0.005)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    # For each customer, we consider the ordered list of facilities + penalty
    # We introduce binary variables for whether each "prefix" of the ordered list is unavailable

    x = {}  # x[j,k] = 1 if the k-th cheapest facility for j is the one that serves j
    for j in range(m):
        for k in range(n + 1):
            x[j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}")

    # Objective components
    total_fixed = gp.quicksum(fixed_cost[i] * y[i] for i in range(n))
    total_trans = 0.0

    for j in range(m):
        ordered_costs = [trans_cost[j, sorted_fac[j][k]] for k in range(n)]
        ordered_costs.append(penalty[j])
        ordered_idx = sorted_fac[j][:] + [-1]

        for k in range(n + 1):
            total_trans += demand[j] * ordered_costs[k] * x[j, k]

            # If we assign to level k, all previous k facilities must be closed or disrupted
            # Probability that the first k-1 are all unavailable = product (1 - p_surv * y)
            # But since we optimize over y, we linearize the product using additional variables.

    # This leads to an exponential number of terms. A standard way is to use a continuous variable
    # representing the probability that a customer is served by a particular facility, but with MIP we
    # can use a dynamic programming like recursive formulation with additional continuous variables.

    # A compact MIP formulation for the expected cost under independent disruptions is as follows:

    # For each customer j, we create a set of continuous variables p[j, k] = probability that the first k-1
    # cheapest open facilities are all disrupted, so that the k-th one is used (if open).

    p = model.addVars(m, n + 2, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="p")

    for j in range(m):
        # p[j,0] = 1.0 : probability before considering any facility
        model.addConstr(p[j, 0] == 1.0)

        ordered = sorted_fac[j]
        for k in range(n):
            i = ordered[k]
            # The probability mass that reaches facility k is p[j,k]
            # If y[i] == 0, then this mass flows through to next with probability 1
            # If y[i] == 1, then mass flows to next with probability (1 - survival[i])
            # We model this with big-M and auxiliary variables
            # p[j, k+1] = p[j, k] * (1 - surv[i]*y[i])
            # This is bilinear. We linearize by introducing a new variable w = p[j,k]*y[i]

            surv_i = nodes[i]['survival_probability']
            w = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"w_{j}_{k}")

            # McCormick envelope for w = p * y  (since y binary)
            model.addConstr(w >= p[j, k] - (1 - y[i]))
            model.addConstr(w <= p[j, k])
            model.addConstr(w <= y[i])
            model.addConstr(w >= 0)

            # p[j, k+1] = p[j, k] - surv_i * w
            model.addConstr(p[j, k + 1] == p[j, k] - surv_i * w)

        # The last p[j, n+1] should be the probability of reaching penalty, but we don't need it explicitly

    # The expected cost for customer j
    for j in range(m):
        ordered_costs = [trans_cost[j, sorted_fac[j][k]] for k in range(n)]
        ordered_surv = [nodes[sorted_fac[j][k]]['survival_probability'] for k in range(n)]
        ordered = sorted_fac[j]

        for k in range(n):
            i = ordered[k]
            # Probability that this facility serves j = (prob reaching it) * y[i] * surv[i]
            # Again we need w from before
            # Since we have w from the loop above, we can reuse logic but it is model specific.
            # To avoid name clashes we re-implement inside a per-customer loop with local vars.

    # Because the above gets very heavy with many auxiliary variables, we use a slightly different
    # but equivalent formulation that adds one variable per (customer, facility) for the "reaching prob".

    # Clean formulation with O(m*n) continuous vars and constraints - exact under independence assumption
    model.dispose()
    model = gp.Model("FacilityLocationStochasticIndependent")
    model.setParam('TimeLimit', max(1, args.time_limit - 10))
    model.setParam('MIPGap', 0.001)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)

    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # For each customer and each possible serving facility, the probability it is served by that facility
    prob_serve = model.addVars(m, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="prob_serve")

    # Expected cost per customer
    exp_cost = model.addVars(m, lb=0.0, vtype=GRB.CONTINUOUS, name="exp_cost")

    # Constraints
    for j in range(m):
        ordered = sorted_fac[j]
        # prob_reach[k] = probability that all better facilities than k are unavailable
        prob_reach = [model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS) for _ in range(n + 1)]
        model.addConstr(prob_reach[0] == 1.0)

        for k in range(n):
            i = ordered[k]
            surv = nodes[i]['survival_probability']

            # prob that this facility is available given we reached it: y[i]*surv
            # Linearization of product with binary y
            z_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_{j}_{k}")
            model.addConstr(z_var <= y[i])
            model.addConstr(z_var <= prob_reach[k])
            model.addConstr(z_var >= prob_reach[k] - (1 - y[i]))

            # prob_serve[j, i] = prob_reach[k] * y[i] * surv = surv * z_var
            model.addConstr(prob_serve[j, i] == surv * z_var)

            # The probability that we skip this facility = prob_reach[k] * (1 - y[i]*surv)
            model.addConstr(prob_reach[k + 1] == prob_reach[k] - z_var * surv)

        # Expected cost for this customer = sum prob_serve * trans_cost + prob_all_fail * penalty
        prob_all_fail = prob_reach[n]
        ec = gp.quicksum(prob_serve[j, ordered[k]] * trans_cost[j, ordered[k]] for k in range(n))
        ec += prob_all_fail * penalty[j]
        model.addConstr(exp_cost[j] == ec)

    # Objective
    model.setObjective(total_fixed + gp.quicksum(demand[j] * exp_cost[j] for j in range(m)), GRB.MINIMIZE)

    # Valid inequalities / cuts
    for j in range(m):
        for i in range(n):
            model.addConstr(prob_serve[j, i] <= y[i])

    # Callback for logging incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj_val)

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status == GRB.INFEASIBLE:
        # This should not happen
        facility_open = [0] * n
        obj_val = 0.0
    else:
        facility_open = [int(y[i].X + 0.5) for i in range(n)]
        obj_val = model.objVal

    # If we have no solution yet, open nothing (very bad but feasible)
    if model.SolCount == 0:
        facility_open = [0] * n
        # compute objective with no facilities: just penalties
        obj_val = sum(demand[j] * penalty[j] for j in range(m))

    solution = {
        "objective_value": float(obj_val),
        "facility_locations": facility_open
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.close()

if __name__ == "__main__":
    main()