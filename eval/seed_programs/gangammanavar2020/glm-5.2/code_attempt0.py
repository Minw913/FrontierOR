import json
import argparse
from gurobipy import Model, GRB, quicksum


def compute_newsvendor_solution(N, lb, h, p, num_scen, scenarios, probs):
    """Compute newsvendor solution ignoring transshipments."""
    s = [float(lb[i]) for i in range(N)]
    for i in range(N):
        if h[i] + p[i] == 0:
            s[i] = float(lb[i])
            continue
        fractile = p[i] / (h[i] + p[i])
        demand_probs = sorted([(scenarios[k][i], probs[k]) for k in range(num_scen)])
        cum_prob = 0.0
        s_i = float(lb[i])
        for d_val, pr in demand_probs:
            cum_prob += pr
            if cum_prob >= fractile:
                s_i = float(d_val)
                break
        s[i] = max(s_i, float(lb[i]))

    total_cost = 0.0
    scen_solutions = []
    for k in range(num_scen):
        d = scenarios[k]
        e_vals = [max(0.0, s[i] - d[i]) for i in range(N)]
        r_vals = [max(0.0, d[i] - s[i]) for i in range(N)]
        f_vals = [min(float(d[i]), float(s[i])) for i in range(N)]
        q_vals = [min(float(d[i]), float(s[i])) for i in range(N)]
        t_vals = [[0.0] * N for _ in range(N)]
        cost = sum(h[i] * e_vals[i] for i in range(N)) + sum(p[i] * r_vals[i] for i in range(N))
        total_cost += probs[k] * cost
        scen_solutions.append({'e': e_vals, 'f': f_vals, 'q': q_vals, 'r': r_vals, 't': t_vals})

    return total_cost, s, scen_solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    N = instance['N']
    lb = instance['first_stage']['variable_lower_bounds']
    h = instance['second_stage']['holding_costs']
    p = instance['second_stage']['penalty_costs']
    c_mean = instance['second_stage']['transshipment_costs_mean']
    num_scen = instance['second_stage']['demand_distribution']['num_scenarios']
    scenarios = instance['second_stage']['demand_distribution']['scenarios']
    probs = instance['second_stage']['demand_distribution']['probabilities']

    # Newsvendor heuristic as initial feasible solution
    init_cost, init_s, init_scen = compute_newsvendor_solution(N, lb, h, p, num_scen, scenarios, probs)
    init_solution = {'objective_value': init_cost, 's': init_s, 'scenarios': init_scen}

    if logger:
        logger.log_solution(init_cost, init_solution)

    best_solution = init_solution
    best_cost = init_cost

    # Try full LP
    try:
        total_prob = sum(probs)
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

        # Precompute cost coefficients for t: h[j] - h[i] + c_mean[i][j]
        ct = {}
        for (i, j) in pairs:
            ct[i, j] = h[j] - h[i] + c_mean[i][j]

        model = Model()
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, args.time_limit - 3))
        model.setParam('OutputFlag', 0)

        # First-stage variables
        s = model.addVars(N, lb=lb, name='s')

        # Second-stage variables
        r_indices = [(k, i) for k in range(num_scen) for i in range(N)]
        t_indices = [(k, i, j) for k in range(num_scen) for (i, j) in pairs]

        r = model.addVars(r_indices, lb=0, name='r')
        t = model.addVars(t_indices, lb=0, name='t')

        # Objective: sum_k prob_k * [sum_i h_i*(s_i - d_ki) + sum_i (h_i+p_i)*r_ki + sum_ij ct_ij * t_kij]
        obj = quicksum(h[i] * total_prob * s[i] for i in range(N))
        obj += quicksum(probs[k] * (h[i] + p[i]) * r[k, i] for (k, i) in r_indices)
        obj += quicksum(probs[k] * ct[i, j] * t[k, i, j] for (k, i, j) in t_indices)
        const = sum(probs[k] * sum(h[i] * scenarios[k][i] for i in range(N)) for k in range(num_scen))
        model.setObjective(obj - const, GRB.MINIMIZE)

        # Constraint 1 (f >= 0): r_{k,i} + sum_j t_{k,j,i} <= d_{k,i}
        model.addConstrs(
            (r[k, i] + t.sum(k, '*', i) <= scenarios[k][i]
             for k in range(num_scen) for i in range(N)),
            name='f'
        )

        # Constraint 2 (e >= 0): s_i - d_{k,i} + sum_j t_{k,j,i} + r_{k,i} - sum_j t_{k,i,j} >= 0
        model.addConstrs(
            (s[i] - scenarios[k][i] + r[k, i] + t.sum(k, '*', i) - t.sum(k, i, '*') >= 0
             for k in range(num_scen) for i in range(N)),
            name='e'
        )

        model.optimize()

        if model.SolCount > 0:
            s_vals = [s[i].X for i in range(N)]
            scen_solutions = []
            for k in range(num_scen):
                d = scenarios[k]
                r_vals = [r[k, i].X for i in range(N)]
                t_vals = [[0.0] * N for _ in range(N)]
                for (i, j) in pairs:
                    t_vals[i][j] = t[k, i, j].X

                e_vals = []
                f_vals = []
                q_vals = []
                for i in range(N):
                    sum_t_in = sum(t_vals[j][i] for j in range(N) if j != i)
                    sum_t_out = sum(t_vals[i][j] for j in range(N) if j != i)
                    f_i = max(0.0, d[i] - sum_t_in - r_vals[i])
                    e_i = max(0.0, s_vals[i] - f_i - sum_t_out)
                    q_i = max(0.0, s_vals[i] - e_i)
                    f_vals.append(f_i)
                    e_vals.append(e_i)
                    q_vals.append(q_i)

                scen_solutions.append({
                    'e': e_vals, 'f': f_vals, 'q': q_vals,
                    'r': r_vals, 't': t_vals
                })

            lp_cost = model.ObjVal
            lp_solution = {'objective_value': lp_cost, 's': s_vals, 'scenarios': scen_solutions}

            if lp_cost < best_cost:
                best_cost = lp_cost
                best_solution = lp_solution
                if logger:
                    logger.log_solution(lp_cost, lp_solution)
    except Exception:
        import traceback
        traceback.print_exc()

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


if __name__ == '__main__':
    main()