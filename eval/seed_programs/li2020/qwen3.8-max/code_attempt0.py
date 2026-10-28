import json
import argparse
import time
import math
import random
from gurobipy import *
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    gp = instance["global_parameters"]
    ip = instance["instance_parameters"]
    interd_params = instance["interdiction_parameters"]
    customers = instance["customers"]
    l1_facs = instance["level1_facilities"]
    l2_facs = instance["level2_facilities"]
    dist_data = instance["distances"]

    theta = float(gp["theta"])
    sigma = float(gp["sigma"])
    alpha_c = float(gp["alpha"])
    beta_c = float(gp["beta"])
    gamma_c = float(gp["gamma"])
    alpha_p = float(gp["alpha_prime"])
    beta_p = float(gp["beta_prime"])
    gamma_p = float(gp["gamma_prime"])

    nc = int(ip["num_customers"])
    nf1 = int(ip["num_level1_facilities"])
    nf2 = int(ip["num_level2_facilities"])
    k_max = int(ip["k_max"])
    budget = float(ip["total_interdiction_budget"])

    d_c_f1 = dist_data["customer_to_level1_facility"]
    d_c_f2 = dist_data["customer_to_level2_facility"]
    d_f1_f2 = dist_data["level1_to_level2_facility"]

    # Compute level II to level II distances
    d_f2_f2 = [[0.0] * nf2 for _ in range(nf2)]
    for i in range(nf2):
        for j in range(nf2):
            dx = l2_facs["x_coordinates"][i] - l2_facs["x_coordinates"][j]
            dy = l2_facs["y_coordinates"][i] - l2_facs["y_coordinates"][j]
            d_f2_f2[i][j] = math.sqrt(dx * dx + dy * dy)

    # Precompute interdiction costs and reductions
    cost_l1 = [0.0] * (k_max + 1)
    cost_l2 = [0.0] * (k_max + 1)
    red_l1 = [0.0] * (k_max + 1)
    red_l2 = [0.0] * (k_max + 1)

    for k in range(1, k_max + 1):
        cost_l1[k] = float(interd_params["interdiction_cost_level1_h1"][str(k)])
        cost_l2[k] = float(interd_params["interdiction_cost_level2_h2"][str(k)])
        red_l1[k] = float(interd_params["capacity_reduction_level1_d1"][str(k)])
        red_l2[k] = float(interd_params["capacity_reduction_level2_d2"][str(k)])

    def total_interdiction_cost(p_l1, p_l2):
        c = 0.0
        for i in range(nf1):
            c += cost_l1[p_l1[i]]
        for j in range(nf2):
            c += cost_l2[p_l2[j]]
        return c

    def pattern_to_dict(p_l1, p_l2):
        d = {}
        for i in range(nf1):
            d[f'S1_{i}'] = int(p_l1[i])
        for j in range(nf2):
            d[f'S2_{j}'] = int(p_l2[j])
        return d

    # Build LP model once
    model = Model("defender")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", 0)
    model.setParam("Threads", 1)
    model.setParam("NumericFocus", 0)

    # Create variables
    x1 = [[model.addVar(lb=0) for _ in range(nf1)] for _ in range(nc)]
    x2 = [[model.addVar(lb=0) for _ in range(nf2)] for _ in range(nc)]
    y = [[model.addVar(lb=0) for _ in range(nf2)] for _ in range(nc)]
    ref1 = [[model.addVar(lb=0) for _ in range(nf2)] for _ in range(nf1)]
    ref2 = [[model.addVar(lb=0) for _ in range(nf2)] for _ in range(nf2)]
    o1 = [model.addVar(lb=0) for _ in range(nc)]
    o2 = [model.addVar(lb=0) for _ in range(nc)]
    oref1 = [model.addVar(lb=0) for _ in range(nf1)]
    oref2 = [model.addVar(lb=0) for _ in range(nf2)]

    model.update()

    # Demand balance constraints
    for c in range(nc):
        model.addConstr(
            quicksum(x1[c][i] for i in range(nf1)) +
            quicksum(x2[c][j] for j in range(nf2)) + o1[c] == theta * customers["demands"][c]
        )

    for c in range(nc):
        model.addConstr(
            quicksum(y[c][j] for j in range(nf2)) + o2[c] == (1 - theta) * customers["demands"][c]
        )

    # Referral balance constraints
    for i in range(nf1):
        model.addConstr(
            sigma * quicksum(x1[c][i] for c in range(nc)) ==
            quicksum(ref1[i][j] for j in range(nf2)) + oref1[i]
        )

    for j in range(nf2):
        model.addConstr(
            sigma * quicksum(x2[c][j] for c in range(nc)) ==
            quicksum(ref2[j][j2] for j2 in range(nf2)) + oref2[j]
        )

    # Capacity constraints (RHS depends on interdiction)
    cap_l1_constrs = []
    for i in range(nf1):
        cap_l1_constrs.append(model.addConstr(
            quicksum(x1[c][i] for c in range(nc)) <= l1_facs["capacity_type1"][i]
        ))

    cap_l2_t1_constrs = []
    for j in range(nf2):
        cap_l2_t1_constrs.append(model.addConstr(
            quicksum(x2[c][j] for c in range(nc)) <= l2_facs["capacity_type1"][j]
        ))

    cap_l2_t2_constrs = []
    for j in range(nf2):
        cap_l2_t2_constrs.append(model.addConstr(
            quicksum(y[c][j] for c in range(nc)) +
            quicksum(ref1[i][j] for i in range(nf1)) +
            quicksum(ref2[j2][j] for j2 in range(nf2)) <= l2_facs["capacity_type2"][j]
        ))

    # Objective
    obj = LinExpr()
    for c in range(nc):
        for i in range(nf1):
            coeff = alpha_c * d_c_f1[c][i]
            if coeff > 1e-12:
                obj += coeff * x1[c][i]
        for j in range(nf2):
            coeff = beta_c * d_c_f2[c][j]
            if coeff > 1e-12:
                obj += coeff * x2[c][j]
                obj += coeff * y[c][j]
        obj += (alpha_p + sigma * gamma_p) * o1[c]
        obj += beta_p * o2[c]

    for i in range(nf1):
        for j in range(nf2):
            coeff = gamma_c * d_f1_f2[i][j]
            if coeff > 1e-12:
                obj += coeff * ref1[i][j]
        obj += gamma_p * oref1[i]

    for j in range(nf2):
        for j2 in range(nf2):
            coeff = gamma_c * d_f2_f2[j][j2]
            if coeff > 1e-12:
                obj += coeff * ref2[j][j2]
        obj += gamma_p * oref2[j]

    model.setObjective(obj, GRB.MINIMIZE)

    def solve_defender(p_l1, p_l2):
        """Update capacities and solve defender's LP. Returns optimal cost."""
        for i in range(nf1):
            cap_l1_constrs[i].setAttr("RHS", l1_facs["capacity_type1"][i] * (1.0 - red_l1[p_l1[i]]))
        for j in range(nf2):
            cap_l2_t1_constrs[j].setAttr("RHS", l2_facs["capacity_type1"][j] * (1.0 - red_l2[p_l2[j]]))
            cap_l2_t2_constrs[j].setAttr("RHS", l2_facs["capacity_type2"][j] * (1.0 - red_l2[p_l2[j]]))

        model.update()
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return model.objVal
        else:
            return float('inf')

    def time_remaining():
        return time_limit - (time.time() - start_time)

    def log_if_better(cost, p_l1, p_l2):
        nonlocal best_defender_cost, best_l1, best_l2
        if cost > best_defender_cost + 1e-8:
            best_defender_cost = cost
            best_l1 = list(p_l1)
            best_l2 = list(p_l2)
            if logger:
                sol = {
                    "objective_value": best_defender_cost,
                    "interdiction_pattern": pattern_to_dict(best_l1, best_l2)
                }
                logger.log_solution(best_defender_cost, sol)

    # Initialize
    cur_l1 = [0] * nf1
    cur_l2 = [0] * nf2
    defender_cost = solve_defender(cur_l1, cur_l2)

    best_defender_cost = defender_cost
    best_l1 = list(cur_l1)
    best_l2 = list(cur_l2)

    if logger:
        sol = {
            "objective_value": best_defender_cost,
            "interdiction_pattern": pattern_to_dict(best_l1, best_l2)
        }
        logger.log_solution(best_defender_cost, sol)

    # Phase 1: Greedy construction
    while time_remaining() > 1.0:
        remaining_budget = budget - total_interdiction_cost(cur_l1, cur_l2)
        if remaining_budget <= 1e-10:
            break

        best_ratio = -float('inf')
        best_action = None
        best_new_cost = defender_cost

        # Try increasing level of each level I facility
        for i in range(nf1):
            for new_level in range(cur_l1[i] + 1, k_max + 1):
                additional_cost = cost_l1[new_level] - cost_l1[cur_l1[i]]
                if additional_cost > remaining_budget + 1e-10:
                    continue

                old = cur_l1[i]
                cur_l1[i] = new_level
                new_def_cost = solve_defender(cur_l1, cur_l2)
                cur_l1[i] = old

                increase = new_def_cost - defender_cost
                if additional_cost > 1e-12:
                    ratio = increase / additional_cost
                else:
                    ratio = float('inf') if increase > 0 else 0

                if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) < 1e-12 and new_def_cost > best_new_cost):
                    best_ratio = ratio
                    best_action = ('l1', i, new_level)
                    best_new_cost = new_def_cost

        # Try increasing level of each level II facility
        for j in range(nf2):
            for new_level in range(cur_l2[j] + 1, k_max + 1):
                additional_cost = cost_l2[new_level] - cost_l2[cur_l2[j]]
                if additional_cost > remaining_budget + 1e-10:
                    continue

                old = cur_l2[j]
                cur_l2[j] = new_level
                new_def_cost = solve_defender(cur_l1, cur_l2)
                cur_l2[j] = old

                increase = new_def_cost - defender_cost
                if additional_cost > 1e-12:
                    ratio = increase / additional_cost
                else:
                    ratio = float('inf') if increase > 0 else 0

                if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) < 1e-12 and new_def_cost > best_new_cost):
                    best_ratio = ratio
                    best_action = ('l2', j, new_level)
                    best_new_cost = new_def_cost

        if best_action is None or best_ratio < -1e-10:
            break

        if best_action[0] == 'l1':
            cur_l1[best_action[1]] = best_action[2]
        else:
            cur_l2[best_action[1]] = best_action[2]

        defender_cost = best_new_cost
        log_if_better(defender_cost, cur_l1, cur_l2)

    # Phase 2: Local search (best improvement)
    while time_remaining() > 1.0:
        improved = False
        best_neighbor_cost = defender_cost
        best_neighbor_action = None

        for i in range(nf1):
            for new_level in range(k_max + 1):
                if new_level == cur_l1[i]:
                    continue
                new_total_cost = total_interdiction_cost(cur_l1, cur_l2) - cost_l1[cur_l1[i]] + cost_l1[new_level]
                if new_total_cost > budget + 1e-10:
                    continue

                old = cur_l1[i]
                cur_l1[i] = new_level
                new_def_cost = solve_defender(cur_l1, cur_l2)
                cur_l1[i] = old

                if new_def_cost > best_neighbor_cost + 1e-8:
                    best_neighbor_cost = new_def_cost
                    best_neighbor_action = ('l1', i, new_level)

        for j in range(nf2):
            for new_level in range(k_max + 1):
                if new_level == cur_l2[j]:
                    continue
                new_total_cost = total_interdiction_cost(cur_l1, cur_l2) - cost_l2[cur_l2[j]] + cost_l2[new_level]
                if new_total_cost > budget + 1e-10:
                    continue

                old = cur_l2[j]
                cur_l2[j] = new_level
                new_def_cost = solve_defender(cur_l1, cur_l2)
                cur_l2[j] = old

                if new_def_cost > best_neighbor_cost + 1e-8:
                    best_neighbor_cost = new_def_cost
                    best_neighbor_action = ('l2', j, new_level)

        if best_neighbor_action is not None:
            if best_neighbor_action[0] == 'l1':
                cur_l1[best_neighbor_action[1]] = best_neighbor_action[2]
            else:
                cur_l2[best_neighbor_action[1]] = best_neighbor_action[2]
            defender_cost = best_neighbor_cost
            improved = True
            log_if_better(defender_cost, cur_l1, cur_l2)
        else:
            break

    # Phase 3: Random restart with perturbation + local search
    random.seed(42)
    while time_remaining() > 2.0:
        # Start from best known solution
        cur_l1 = list(best_l1)
        cur_l2 = list(best_l2)

        # Random perturbation
        num_perturb = random.randint(1, max(1, (nf1 + nf2) // 4))
        for _ in range(num_perturb):
            if nf1 + nf2 == 0:
                break
            if random.random() < nf1 / (nf1 + nf2) and nf1 > 0:
                i = random.randint(0, nf1 - 1)
                cur_l1[i] = random.randint(0, k_max)
            elif nf2 > 0:
                j = random.randint(0, nf2 - 1)
                cur_l2[j] = random.randint(0, k_max)

        # Fix budget violation
        while total_interdiction_cost(cur_l1, cur_l2) > budget + 1e-10:
            candidates = []
            for i in range(nf1):
                if cur_l1[i] > 0:
                    candidates.append(('l1', i))
            for j in range(nf2):
                if cur_l2[j] > 0:
                    candidates.append(('l2', j))
            if not candidates:
                break
            choice = random.choice(candidates)
            if choice[0] == 'l1':
                cur_l1[choice[1]] -= 1
            else:
                cur_l2[choice[1]] -= 1

        defender_cost = solve_defender(cur_l1, cur_l2)
        log_if_better(defender_cost, cur_l1, cur_l2)

        # Local search from perturbed solution
        for _ in range(30):
            if time_remaining() < 1.5:
                break

            best_neighbor_cost = defender_cost
            best_neighbor_action = None

            for i in range(nf1):
                for new_level in range(k_max + 1):
                    if new_level == cur_l1[i]:
                        continue
                    new_total_cost = total_interdiction_cost(cur_l1, cur_l2) - cost_l1[cur_l1[i]] + cost_l1[new_level]
                    if new_total_cost > budget + 1e-10:
                        continue

                    old = cur_l1[i]
                    cur_l1[i] = new_level
                    new_def_cost = solve_defender(cur_l1, cur_l2)
                    cur_l1[i] = old

                    if new_def_cost > best_neighbor_cost + 1e-8:
                        best_neighbor_cost = new_def_cost
                        best_neighbor_action = ('l1', i, new_level)

            for j in range(nf2):
                for new_level in range(k_max + 1):
                    if new_level == cur_l2[j]:
                        continue
                    new_total_cost = total_interdiction_cost(cur_l1, cur_l2) - cost_l2[cur_l2[j]] + cost_l2[new_level]
                    if new_total_cost > budget + 1e-10:
                        continue

                    old = cur_l2[j]
                    cur_l2[j] = new_level
                    new_def_cost = solve_defender(cur_l1, cur_l2)
                    cur_l2[j] = old

                    if new_def_cost > best_neighbor_cost + 1e-8:
                        best_neighbor_cost = new_def_cost
                        best_neighbor_action = ('l2', j, new_level)

            if best_neighbor_action is not None:
                if best_neighbor_action[0] == 'l1':
                    cur_l1[best_neighbor_action[1]] = best_neighbor_action[2]
                else:
                    cur_l2[best_neighbor_action[1]] = best_neighbor_action[2]
                defender_cost = best_neighbor_cost
                log_if_better(defender_cost, cur_l1, cur_l2)
            else:
                break

    # Final output
    solution_dict = {
        "objective_value": best_defender_cost,
        "interdiction_pattern": pattern_to_dict(best_l1, best_l2)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == "__main__":
    main()