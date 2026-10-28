import argparse
import json
import sys
import time

def compute_cost(num_facilities, num_customers, fixed_costs, demands, survival_probs,
                 trans_cost, penalty_costs, sigma, open_mask):
    """Compute total cost (fixed + expected transport+penalty) for a given facility opening mask."""
    fixed = sum(fixed_costs[i] for i in range(num_facilities) if open_mask[i])
    transport = 0.0
    for j in range(num_customers):
        d_j = demands[j]
        pen_j = penalty_costs[j]
        order = sigma[j]
        s = 1.0  # probability that no facility so far is available
        exp_cost = 0.0
        for m in range(num_facilities):
            i = order[m]
            if open_mask[i]:
                p_i = survival_probs[i]
                exp_cost += trans_cost[j][i] * s * p_i
                s *= (1.0 - p_i)
        exp_cost += pen_j * s
        transport += d_j * exp_cost
    return fixed + transport

def solve(instance, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    global_start = time.time()

    num_nodes = instance["num_nodes"]
    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    nodes = instance["nodes"]

    fixed_costs = [nodes[i]["fixed_setup_cost"] for i in range(num_facilities)]
    demands = [nodes[j]["demand"] for j in range(num_customers)]
    survival_probs = [nodes[i]["survival_probability"] for i in range(num_facilities)]
    trans_cost = instance["transportation_cost_matrix"]
    penalty_costs = instance["penalty_cost"]

    # Sort facilities for each customer by transportation cost (ascending)
    sigma = []
    for j in range(num_customers):
        order = sorted(range(num_facilities), key=lambda i: trans_cost[j][i])
        sigma.append(order)

    # Heuristic: try all-open and none-open
    all_open = [1] * num_facilities
    none_open = [0] * num_facilities
    cost_all = compute_cost(num_facilities, num_customers, fixed_costs, demands, survival_probs,
                            trans_cost, penalty_costs, sigma, all_open)
    cost_none = compute_cost(num_facilities, num_customers, fixed_costs, demands, survival_probs,
                             trans_cost, penalty_costs, sigma, none_open)

    if cost_all <= cost_none:
        best_cost = cost_all
        best_mask = list(all_open)
    else:
        best_cost = cost_none
        best_mask = list(none_open)

    if logger:
        logger.log_solution(best_cost, {"objective_value": best_cost, "facility_locations": best_mask})

    # Quick local search to improve initial solution
    ls_time_limit = min(10, time_limit * 0.15)
    ls_start = time.time()
    improved = True
    while improved and time.time() - ls_start < ls_time_limit:
        improved = False
        for i in range(num_facilities):
            if time.time() - ls_start > ls_time_limit:
                break
            new_mask = list(best_mask)
            new_mask[i] = 1 - new_mask[i]
            new_cost = compute_cost(num_facilities, num_customers, fixed_costs, demands, survival_probs,
                                    trans_cost, penalty_costs, sigma, new_mask)
            if new_cost < best_cost - 1e-9:
                best_cost = new_cost
                best_mask = new_mask
                improved = True

    if logger:
        logger.log_solution(best_cost, {"objective_value": best_cost, "facility_locations": best_mask})

    # Determine if MIP is feasible given problem size
    total_vars = num_customers * (2 * num_facilities + 1)
    use_mip = total_vars <= 200000

    if use_mip:
        try:
            from gurobipy import Model, GRB, quicksum

            remaining_time = time_limit - (time.time() - global_start)
            if remaining_time < 3:
                raise Exception("Not enough time for MIP")

            model = Model("facility_location")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1, remaining_time - 2)
            model.Params.OutputFlag = 0

            # Binary: whether to open facility i
            x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")

            # s[j, m] = probability that none of the first m facilities (in sorted order for j) are available
            # s[j, 0] = 1, s[j, m+1] = s[j, m] * (1 - p_i * x_i) for facility at position m
            s = {}
            for j in range(num_customers):
                for m in range(num_facilities + 1):
                    s[j, m] = model.addVar(lb=0, ub=1, name=f"s_{j}_{m}")

            # t[j, m] = s[j, m] * x[sigma[j][m]]  (linearized product)
            t = {}
            for j in range(num_customers):
                for m in range(num_facilities):
                    t[j, m] = model.addVar(lb=0, ub=1, name=f"t_{j}_{m}")

            model.update()

            # Constraints
            for j in range(num_customers):
                model.addConstr(s[j, 0] == 1)
                for m in range(num_facilities):
                    i = sigma[j][m]
                    p_i = survival_probs[i]

                    # Linearize t[j,m] = s[j,m] * x[i]  (exact for binary x)
                    model.addConstr(t[j, m] <= s[j, m])
                    model.addConstr(t[j, m] <= x[i])
                    model.addConstr(t[j, m] >= s[j, m] + x[i] - 1)

                    # s[j, m+1] = s[j, m] - p_i * t[j, m]
                    model.addConstr(s[j, m + 1] == s[j, m] - p_i * t[j, m])

            # Objective: fixed costs + expected transport+penalty
            # For customer j with sorted order sigma[j]:
            # E_j = c_{order[0]} + sum_{m=1}^{N-1} (c_{order[m]} - c_{order[m-1]}) * s[j,m]
            #       + (pen_j - c_{order[N-1]}) * s[j,N]
            obj = quicksum(fixed_costs[i] * x[i] for i in range(num_facilities))

            for j in range(num_customers):
                d_j = demands[j]
                pen_j = penalty_costs[j]
                order = sigma[j]

                exp_cost = trans_cost[j][order[0]]
                for m in range(1, num_facilities):
                    delta = trans_cost[j][order[m]] - trans_cost[j][order[m - 1]]
                    exp_cost += delta * s[j, m]
                delta_pen = pen_j - trans_cost[j][order[num_facilities - 1]]
                exp_cost += delta_pen * s[j, num_facilities]

                obj += d_j * exp_cost

            model.setObjective(obj, GRB.MINIMIZE)

            # Warm start
            for i in range(num_facilities):
                x[i].start = best_mask[i]

            # Callback for logging intermediate solutions
            state = {'best_cost': best_cost, 'best_mask': list(best_mask), 'logger': logger,
                     'num_facilities': num_facilities}

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < state['best_cost'] - 1e-9:
                        state['best_cost'] = obj_val
                        x_vals = model.cbGetSolution([x[i] for i in range(state['num_facilities'])])
                        facility_locations = [int(round(v)) for v in x_vals]
                        state['best_mask'] = facility_locations
                        if state['logger']:
                            state['logger'].log_solution(
                                obj_val,
                                {"objective_value": obj_val, "facility_locations": facility_locations}
                            )

            model.optimize(callback)

            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val < state['best_cost'] - 1e-9:
                    state['best_cost'] = obj_val
                    facility_locations = [int(round(x[i].X)) for i in range(num_facilities)]
                    state['best_mask'] = facility_locations
                    if logger:
                        logger.log_solution(obj_val, {"objective_value": obj_val, "facility_locations": facility_locations})

            best_cost = state['best_cost']
            best_mask = state['best_mask']

        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)

    return best_cost, list(best_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    obj_val, facility_locations = solve(instance, args.time_limit, args.log_path)

    solution = {
        "objective_value": obj_val,
        "facility_locations": facility_locations
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()