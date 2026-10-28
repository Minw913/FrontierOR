import json
import argparse
import time
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

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

    N = data['N']
    holding_costs = np.array(data['second_stage']['holding_costs'])
    penalty_costs = np.array(data['second_stage']['penalty_costs'])
    trans_mean = np.array(data['second_stage']['transshipment_costs_mean'])
    demand_scenarios = np.array(data['second_stage']['demand_distribution']['scenarios'])
    probabilities = np.array(data['second_stage']['demand_distribution']['probabilities'])
    num_scenarios = len(probabilities)
    mean_demands = np.array(data['second_stage']['mean_demands'])
    lower_bounds = np.array(data['first_stage']['variable_lower_bounds'])

    # Sample uncertain costs using provided random seed for reproducibility
    np.random.seed(42)
    trans_costs = np.zeros((num_scenarios, N, N))
    for s in range(num_scenarios):
        trans_costs[s] = trans_mean.copy()
        for link in data['second_stage']['random_cost_links']:
            i, j = link
            if trans_mean[i][j] > 0:
                std = trans_mean[i][j] * data['second_stage']['random_cost_std_fraction']
                trans_costs[s, i, j] = np.random.normal(trans_mean[i][j], std)

    # Precompute for each scenario the optimal second-stage cost given s vector
    # This is a transportation problem: supply s, demand d, with local holding/penalty and transshipment
    def second_stage_cost(s, scenario_idx):
        d = demand_scenarios[scenario_idx]
        c = trans_costs[scenario_idx]
        h = holding_costs
        p = penalty_costs

        m = Model()
        m.setParam('OutputFlag', 0)
        m.setParam('Threads', 1)

        # Variables
        e = m.addVars(N, lb=0, name='e')      # ending inventory
        r = m.addVars(N, lb=0, name='r')      # backlog
        t = m.addVars(N, N, lb=0, name='t')   # transshipments

        # Supply balance: local use + out + e = s => local_use = s - sum_out - e
        # Demand balance: local_use + in + r = d => s - sum_out - e + sum_in + r = d
        # => r = d - s + sum_out - sum_in + e

        # Objective: sum h_i * e_i + sum p_i * r_i + sum c_ij * t_ij
        obj = quicksum(h[i] * e[i] for i in range(N))
        obj += quicksum(p[i] * r[i] for i in range(N))
        obj += quicksum(c[i,j] * t[i,j] for i in range(N) for j in range(N) if i != j)
        m.setObjective(obj, GRB.MINIMIZE)

        # r >= 0 implies: e_i - sum_j t_ji + sum_j t_ij >= s_i - d_i   for each i
        for i in range(N):
            inflow = quicksum(t[j,i] for j in range(N) if j != i)
            outflow = quicksum(t[i,j] for j in range(N) if j != i)
            m.addConstr(e[i] + outflow - inflow >= s[i] - d[i], name=f'balance_{i}')

        # Non-negativity of local use is implicit if we allow r and e freely (but costs will drive it)
        # However to be precise we also need local_use >=0 i.e. s_i >= e_i + outflow
        for i in range(N):
            outflow = quicksum(t[i,j] for j in range(N) if j != i)
            m.addConstr(e[i] + outflow <= s[i], name=f'supply_{i}')

        m.setParam('TimeLimit', 5.0)  # limit per subproblem
        m.optimize()

        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return m.objVal
        else:
            # If failure, use a myopic heuristic
            return np.sum(h * np.maximum(s - d, 0)) + np.sum(p * np.maximum(d - s, 0))

    # Now solve the first-stage problem: min sum_s prob_s * Q(s, xi_s) s.t. s >= lower_bounds
    # This is a convex optimization problem (Q is convex in s). We use Gurobi with callback for incumbent logging.

    master = Model("Stock_Positioning")
    master.setParam('OutputFlag', 0)
    master.setParam('Threads', 1)
    master.setParam('TimeLimit', args.time_limit - 5)  # reserve some time

    s_vars = master.addVars(N, lb=lower_bounds, name='s')
    # We will use epigraph variables for each scenario
    theta = master.addVars(num_scenarios, lb=-GRB.INFINITY, name='theta')

    master.setObjective(quicksum(probabilities[s] * theta[s] for s in range(num_scenarios)), GRB.MINIMIZE)

    # Initial cuts: very loose
    for k in range(num_scenarios):
        master.addConstr(theta[k] >= 0)

    # We will solve by adding subgradient cuts (outer approximation) since Q is convex
    incumbent_s = np.maximum(mean_demands, lower_bounds)
    best_obj = float('inf')
    start_time = time.time()
    iteration = 0
    max_iterations = 30

    while time.time() - start_time < args.time_limit - 10 and iteration < max_iterations:
        iteration += 1
        current_s = incumbent_s.copy()

        # Evaluate current solution
        total_expected = 0.0
        subgrad = np.zeros(N)
        scenario_objs = []

        for k in range(num_scenarios):
            Qval = second_stage_cost(current_s, k)
            scenario_objs.append(Qval)
            total_expected += probabilities[k] * Qval

            # Approximate subgradient from duals or finite difference
            # Here we resolve with slight perturbation to get subgradient
            eps = 1e-4
            for i in range(N):
                s_pert = current_s.copy()
                s_pert[i] += eps
                Qpert = second_stage_cost(s_pert, k)
                subgrad[i] += probabilities[k] * (Qpert - Qval) / eps

        if total_expected < best_obj:
            best_obj = total_expected
            best_s = current_s.copy()
            if logger:
                logger.log(best_obj)
            # Update incumbent
            incumbent_s = current_s.copy()

            # Add optimality cuts
            for k in range(num_scenarios):
                master.addConstr(
                    theta[k] >= scenario_objs[k] + quicksum(
                        subgrad[i] * (s_vars[i] - current_s[i]) for i in range(N)
                    )
                )

        # Also add a cut at the mean demand point if not yet evaluated
        if iteration == 1:
            mean_s = np.maximum(mean_demands * 1.2, lower_bounds)
            for k in range(num_scenarios):
                Qmean = second_stage_cost(mean_s, k)
                master.addConstr(theta[k] >= Qmean)

        # Solve the master to get new candidate
        master.optimize()

        if master.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            new_s = np.array([s_vars[i].X for i in range(N)])
            incumbent_s = np.maximum(new_s, lower_bounds)
        else:
            # If master fails, do local search
            incumbent_s = np.maximum(incumbent_s * 0.95, lower_bounds)
            incumbent_s += np.random.uniform(0, 5, N)

    # Final evaluation with best_s
    final_s = best_s if 'best_s' in locals() else incumbent_s
    final_expected = 0.0
    scenario_solutions = []

    for k in range(num_scenarios):
        d = demand_scenarios[k]
        c = trans_costs[k]
        h = holding_costs
        p = penalty_costs

        m = Model()
        m.setParam('OutputFlag', 0)
        m.setParam('Threads', 1)

        e_var = m.addVars(N, lb=0, name='e')
        r_var = m.addVars(N, lb=0, name='r')
        t_var = m.addVars(N, N, lb=0, name='t')
        f_var = m.addVars(N, lb=0, name='f')  # local usage
        q_var = m.addVars(N, lb=0, name='q')  # replenishment

        # Balance constraints as defined
        for i in range(N):
            outflow = quicksum(t_var[i,j] for j in range(N) if i != j)
            m.addConstr(f_var[i] + outflow + e_var[i] == final_s[i], name=f'supply_{i}_{k}')
            inflow = quicksum(t_var[j,i] for j in range(N) if i != j)
            m.addConstr(f_var[i] + inflow + r_var[i] == d[i], name=f'demand_{i}_{k}')
            m.addConstr(e_var[i] + q_var[i] == final_s[i], name=f'replenish_{i}_{k}')

        # Global balance (redundant but included)
        m.addConstr(quicksum(r_var[i] for i in range(N)) + quicksum(q_var[i] for i in range(N)) == quicksum(d), name='global')

        obj = quicksum(h[i] * e_var[i] for i in range(N)) + \
              quicksum(p[i] * r_var[i] for i in range(N)) + \
              quicksum(c[i,j] * t_var[i,j] for i in range(N) for j in range(N))
        m.setObjective(obj, GRB.MINIMIZE)

        m.setParam('TimeLimit', 10)
        m.optimize()

        if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            # Fallback heuristic
            e_sol = np.maximum(final_s - d, 0)
            r_sol = np.maximum(d - final_s, 0)
            f_sol = np.minimum(final_s, d)
            q_sol = final_s - e_sol
            t_sol = np.zeros((N, N))
            final_expected += probabilities[k] * (np.dot(h, e_sol) + np.dot(p, r_sol))
        else:
            e_sol = np.array([e_var[i].X for i in range(N)])
            r_sol = np.array([r_var[i].X for i in range(N)])
            f_sol = np.array([f_var[i].X for i in range(N)])
            q_sol = np.array([q_var[i].X for i in range(N)])
            t_sol = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if i != j:
                        t_sol[i,j] = t_var[i,j].X
            final_expected += probabilities[k] * m.objVal

        scenario_solutions.append({
            'e': e_sol.tolist(),
            'f': f_sol.tolist(),
            'q': q_sol.tolist(),
            'r': r_sol.tolist(),
            't': t_sol.tolist()
        })

    objective_value = final_expected

    solution = {
        "objective_value": float(objective_value),
        "s": final_s.tolist(),
        "scenarios": scenario_solutions
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()