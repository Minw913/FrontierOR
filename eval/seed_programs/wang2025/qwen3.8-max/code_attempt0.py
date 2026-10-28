import json
import argparse
import time
import numpy as np
from solution_logger import SolutionLogger


def compute_objective(selected, instance_type, data_matrix, node_weights, element_weights):
    """Compute objective value for a given subset."""
    if len(selected) == 0:
        return 0.0
    selected = list(selected)
    sub_matrix = data_matrix[:, selected]

    if instance_type == "LOC":
        return float(np.sum(np.max(sub_matrix, axis=1)))
    elif instance_type == "COV":
        covered = np.any(sub_matrix > 0.5, axis=1)
        return float(np.sum(node_weights[covered]))
    elif instance_type == "INF":
        ew = element_weights[selected]
        p_sub = sub_matrix * ew[np.newaxis, :]
        prob_not_reached = np.prod(1.0 - p_sub, axis=1)
        return float(np.sum(1.0 - prob_not_reached))


def greedy(instance_type, data_matrix, node_weights, element_weights, n, k, time_limit, logger, start_time):
    """Efficient greedy algorithm."""
    m = data_matrix.shape[0]
    selected = []
    selected_set = set()
    best_value = 0.0

    if instance_type == "LOC":
        current_max = np.zeros(m)
        for iteration in range(k):
            if time.time() - start_time > time_limit * 0.75:
                break
            gains = np.maximum(data_matrix - current_max[:, np.newaxis], 0)
            col_sums = np.sum(gains, axis=0)
            for j in selected_set:
                col_sums[j] = -1
            best_elem = int(np.argmax(col_sums))
            if col_sums[best_elem] <= 1e-10:
                break
            selected.append(best_elem)
            selected_set.add(best_elem)
            current_max = np.maximum(current_max, data_matrix[:, best_elem])
            best_value = float(np.sum(current_max))
            if logger:
                logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": list(selected)})

    elif instance_type == "COV":
        covered = np.zeros(m, dtype=bool)
        for iteration in range(k):
            if time.time() - start_time > time_limit * 0.75:
                break
            uncovered_weights = node_weights * (~covered).astype(float)
            gains = data_matrix.T @ uncovered_weights
            for j in selected_set:
                gains[j] = -1
            best_elem = int(np.argmax(gains))
            if gains[best_elem] <= 1e-10:
                break
            selected.append(best_elem)
            selected_set.add(best_elem)
            covered |= (data_matrix[:, best_elem] > 0.5)
            best_value = float(np.sum(node_weights[covered]))
            if logger:
                logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": list(selected)})

    elif instance_type == "INF":
        prob_not_reached = np.ones(m)
        for iteration in range(k):
            if time.time() - start_time > time_limit * 0.75:
                break
            # Marginal gain of adding j: sum_i prob_not_reached[i] * p_ij
            p_matrix = data_matrix * element_weights[np.newaxis, :]
            gains = p_matrix.T @ prob_not_reached
            for j in selected_set:
                gains[j] = -1
            best_elem = int(np.argmax(gains))
            if gains[best_elem] <= 1e-10:
                break
            selected.append(best_elem)
            selected_set.add(best_elem)
            p_best = data_matrix[:, best_elem] * element_weights[best_elem]
            prob_not_reached *= (1.0 - p_best)
            best_value = float(np.sum(1.0 - prob_not_reached))
            if logger:
                logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": list(selected)})

    return selected, best_value


def local_search(selected, instance_type, data_matrix, node_weights, element_weights, n, k, time_limit, start_time, logger, best_value):
    """1-swap local search improvement."""
    selected_set = set(selected)
    improved = True
    time_budget = time_limit * 0.9

    while improved and time.time() - start_time < time_budget:
        improved = False
        # Try swaps
        for i in range(len(selected)):
            if time.time() - start_time > time_budget:
                break
            elem_in = selected[i]
            for elem_out in range(n):
                if elem_out in selected_set:
                    continue
                # Swap
                selected[i] = elem_out
                new_val = compute_objective(selected, instance_type, data_matrix, node_weights, element_weights)
                if new_val > best_value + 1e-10:
                    selected_set.discard(elem_in)
                    selected_set.add(elem_out)
                    best_value = new_val
                    improved = True
                    if logger:
                        logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": list(selected)})
                    break
                else:
                    selected[i] = elem_in

        # Try additions if |selected| < k
        if len(selected) < k and time.time() - start_time < time_budget:
            for elem_out in range(n):
                if elem_out in selected_set:
                    continue
                new_val = compute_objective(selected + [elem_out], instance_type, data_matrix, node_weights, element_weights)
                if new_val > best_value + 1e-10:
                    selected.append(elem_out)
                    selected_set.add(elem_out)
                    best_value = new_val
                    improved = True
                    if logger:
                        logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": list(selected)})
                    break

    return selected, best_value


def solve_mip_LOC(instance, time_limit, logger, start_time):
    """Solve LOC using Gurobi MIP."""
    import gurobipy as gp
    from gurobipy import GRB

    n = instance["n"]
    m = instance["m"]
    k = instance["k"]
    data_matrix = np.array(instance["data_matrix"])

    model = gp.Model("facility_location")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - (time.time() - start_time))
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, name="y")

    model.setObjective(gp.quicksum(y[i] for i in range(m)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k)

    for i in range(m):
        max_val = float(np.max(data_matrix[i, :])) if n > 0 else 0
        for j in range(n):
            model.addConstr(y[i] <= float(data_matrix[i][j]) + max_val * (1 - x[j]))

    model.optimize()

    if model.SolCount > 0:
        selected = [j for j in range(n) if x[j].X > 0.5]
        obj_val = model.ObjVal
        return selected, obj_val
    return None, None


def solve_mip_COV(instance, time_limit, logger, start_time):
    """Solve COV using Gurobi MIP."""
    import gurobipy as gp
    from gurobipy import GRB

    n = instance["n"]
    m = instance["m"]
    k = instance["k"]
    data_matrix = np.array(instance["data_matrix"])
    node_weights = np.array(instance["node_weights"])

    model = gp.Model("weighted_coverage")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - (time.time() - start_time))
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

    model.setObjective(gp.quicksum(node_weights[i] * y[i] for i in range(m)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k)

    for i in range(m):
        model.addConstr(y[i] <= gp.quicksum(float(data_matrix[i][j]) * x[j] for j in range(n)))

    model.optimize()

    if model.SolCount > 0:
        selected = [j for j in range(n) if x[j].X > 0.5]
        obj_val = model.ObjVal
        return selected, obj_val
    return None, None


def solve_mip_INF(instance, time_limit, logger, start_time):
    """Solve INF using Gurobi with piecewise linear approximation."""
    import gurobipy as gp
    from gurobipy import GRB

    n = instance["n"]
    m = instance["m"]
    k = instance["k"]
    data_matrix = np.array(instance["data_matrix"])
    element_weights = np.array(instance["element_weights"])

    # Only attempt for small instances
    if n > 100 or m > 200:
        return None, None

    model = gp.Model("influence")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - (time.time() - start_time))
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

    model.setObjective(gp.quicksum(y[i] for i in range(m)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k)

    # For each target i, model y_i = 1 - prod_{j: edge} (1 - p_j * x_j)
    # Use: y_i <= sum_j p_ij * x_j (linear relaxation upper bound)
    # And piecewise linear for tighter bound
    for i in range(m):
        neighbors = [j for j in range(n) if data_matrix[i][j] > 0.5]
        if len(neighbors) == 0:
            model.addConstr(y[i] == 0)
            continue

        # Upper bound: y_i <= sum_j p_ij * x_j
        model.addConstr(y[i] <= gp.quicksum(
            float(data_matrix[i][j]) * element_weights[j] * x[j] for j in neighbors
        ))

        # For better approximation, use inclusion-exclusion for pairs
        if len(neighbors) <= 20:
            # y_i >= sum_j p_j*x_j - sum_{j<j'} p_j*p_j'*x_j*x_j' (second order)
            # This requires bilinear terms, skip for now
            pass

    model.optimize()

    if model.SolCount > 0:
        selected = [j for j in range(n) if x[j].X > 0.5]
        # Recompute true objective
        obj_val = compute_objective(selected, "INF", data_matrix,
                                     np.zeros(instance["m"]), element_weights)
        return selected, obj_val
    return None, None


def randomized_greedy(instance_type, data_matrix, node_weights, element_weights, n, k, time_limit, logger, start_time, num_restarts=5):
    """Randomized greedy with multiple restarts."""
    best_selected = []
    best_value = 0.0
    m = data_matrix.shape[0]

    for restart in range(num_restarts):
        if time.time() - start_time > time_limit * 0.5:
            break

        selected = []
        selected_set = set()
        current_value = 0.0

        if instance_type == "LOC":
            current_max = np.zeros(m)
            for iteration in range(k):
                if time.time() - start_time > time_limit * 0.5:
                    break
                gains = np.maximum(data_matrix - current_max[:, np.newaxis], 0)
                col_sums = np.sum(gains, axis=0)
                for j in selected_set:
                    col_sums[j] = 0
                total = np.sum(col_sums)
                if total <= 1e-10:
                    break
                # Sample proportional to gains
                probs = col_sums / total
                best_elem = np.random.choice(n, p=probs)
                selected.append(best_elem)
                selected_set.add(best_elem)
                current_max = np.maximum(current_max, data_matrix[:, best_elem])
                current_value = float(np.sum(current_max))

        elif instance_type == "COV":
            covered = np.zeros(m, dtype=bool)
            for iteration in range(k):
                if time.time() - start_time > time_limit * 0.5:
                    break
                uncovered_weights = node_weights * (~covered).astype(float)
                gains = data_matrix.T @ uncovered_weights
                for j in selected_set:
                    gains[j] = 0
                total = np.sum(gains)
                if total <= 1e-10:
                    break
                probs = gains / total
                best_elem = np.random.choice(n, p=probs)
                selected.append(best_elem)
                selected_set.add(best_elem)
                covered |= (data_matrix[:, best_elem] > 0.5)
                current_value = float(np.sum(node_weights[covered]))

        elif instance_type == "INF":
            prob_not_reached = np.ones(m)
            for iteration in range(k):
                if time.time() - start_time > time_limit * 0.5:
                    break
                p_matrix = data_matrix * element_weights[np.newaxis, :]
                gains = p_matrix.T @ prob_not_reached
                for j in selected_set:
                    gains[j] = 0
                total = np.sum(gains)
                if total <= 1e-10:
                    break
                probs = gains / total
                best_elem = np.random.choice(n, p=probs)
                selected.append(best_elem)
                selected_set.add(best_elem)
                p_best = data_matrix[:, best_elem] * element_weights[best_elem]
                prob_not_reached *= (1.0 - p_best)
                current_value = float(np.sum(1.0 - prob_not_reached))

        if current_value > best_value:
            best_value = current_value
            best_selected = list(selected)

    return best_selected, best_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    start_time = time.time()
    np.random.seed(42)

    instance_type = instance["instance_type"]
    n = instance["n"]
    m = instance["m"]
    k = instance["k"]
    data_matrix = np.array(instance["data_matrix"], dtype=np.float64)
    node_weights = np.array(instance["node_weights"], dtype=np.float64)
    element_weights = np.array(instance["element_weights"], dtype=np.float64)

    # Phase 1: Greedy
    selected_greedy, value_greedy = greedy(
        instance_type, data_matrix, node_weights, element_weights,
        n, k, args.time_limit, logger, start_time
    )

    best_selected = list(selected_greedy)
    best_value = value_greedy

    # Phase 2: MIP for LOC and COV
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 2:
        try:
            if instance_type == "LOC":
                mip_selected, mip_value = solve_mip_LOC(instance, args.time_limit, logger, start_time)
                if mip_selected is not None and mip_value is not None:
                    true_val = compute_objective(mip_selected, instance_type, data_matrix, node_weights, element_weights)
                    if true_val > best_value:
                        best_selected = mip_selected
                        best_value = true_val
                        if logger:
                            logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": best_selected})
            elif instance_type == "COV":
                mip_selected, mip_value = solve_mip_COV(instance, args.time_limit, logger, start_time)
                if mip_selected is not None and mip_value is not None:
                    true_val = compute_objective(mip_selected, instance_type, data_matrix, node_weights, element_weights)
                    if true_val > best_value:
                        best_selected = mip_selected
                        best_value = true_val
                        if logger:
                            logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": best_selected})
        except Exception:
            pass

    # Phase 3: Local search
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 1 and len(best_selected) > 0:
        ls_selected, ls_value = local_search(
            list(best_selected), instance_type, data_matrix, node_weights,
            element_weights, n, k, args.time_limit, start_time, logger, best_value
        )
        if ls_value > best_value:
            best_selected = ls_selected
            best_value = ls_value

    # Phase 4: Randomized greedy (if time permits and greedy wasn't great)
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 2:
        rand_selected, rand_value = randomized_greedy(
            instance_type, data_matrix, node_weights, element_weights,
            n, k, args.time_limit, logger, start_time, num_restarts=3
        )
        if rand_value > best_value:
            best_selected = rand_selected
            best_value = rand_value
            if logger:
                logger.log_solution(best_value, {"objective_value": best_value, "selected_elements": best_selected})

            # Local search on randomized greedy result
            remaining_time = args.time_limit - (time.time() - start_time)
            if remaining_time > 1:
                ls2_selected, ls2_value = local_search(
                    list(best_selected), instance_type, data_matrix, node_weights,
                    element_weights, n, k, args.time_limit, start_time, logger, best_value
                )
                if ls2_value > best_value:
                    best_selected = ls2_selected
                    best_value = ls2_value

    # Write final solution
    solution = {
        "objective_value": best_value,
        "selected_elements": [int(x) for x in best_selected]
    }

    if logger:
        logger.log_solution(best_value, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()