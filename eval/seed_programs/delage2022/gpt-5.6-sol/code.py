import argparse
import json
import math
import os

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def candidate_objective(candidate, setup_cost, repetition_cost, knife_cost):
    used_count = sum(1 for value in candidate["used"] if value)
    repetitions = sum(candidate["repetitions"])
    knives = sum(sum(row) for row in candidate["pieces"])
    return float(
        setup_cost * used_count
        + repetition_cost * repetitions
        + knife_cost * knives
    )


def make_output_solution(
    product_ids,
    num_patterns,
    problem_type,
    setup_cost,
    repetition_cost,
    knife_cost,
    used,
    repetitions,
    pieces,
    extra_pieces,
):
    patterns = {}
    production_quantities = {}

    for p in range(num_patterns):
        piece_dict = {}
        knife_count = 0

        for i, product_id in enumerate(product_ids):
            value = int(pieces[p][i])
            piece_dict[str(product_id)] = value
            knife_count += value
            production_quantities[
                f"product_{product_id}_pattern_{p}"
            ] = int(value * repetitions[p])

        pattern_data = {
            "used": bool(used[p]),
            "repetitions": int(repetitions[p]),
            "pieces": piece_dict,
        }
        if problem_type == "cscp_with_knife_dependent_setups":
            pattern_data["knife_count"] = int(knife_count)

        patterns[str(p)] = pattern_data

    extra_dict = {
        str(product_id): int(extra_pieces[i])
        for i, product_id in enumerate(product_ids)
    }

    objective = float(
        setup_cost * sum(1 for value in used if value)
        + repetition_cost * sum(repetitions)
        + knife_cost * sum(sum(row) for row in pieces)
    )

    return {
        "objective_value": objective,
        "patterns": patterns,
        "extra_pieces": extra_dict,
        "production_quantities": production_quantities,
    }


def build_packed_candidate(products, num_patterns, machine_length, M):
    """Construct a feasible candidate by putting each demanded product in one bin.

    Each product initially contributes one piece per pass. Once products have
    been assigned to a pattern, the smallest feasible common repetition count
    for that pattern is found by binary search.
    """
    n = len(products)
    demanded = [i for i, product in enumerate(products) if int(product["demand"]) > 0]

    if not demanded:
        return {
            "used": [False] * num_patterns,
            "repetitions": [0] * num_patterns,
            "pieces": [[0] * n for _ in range(num_patterns)],
            "extra": [0] * n,
        }

    for i in demanded:
        if int(products[i]["length"]) > machine_length or int(products[i]["N_i"]) < 1:
            return None

    # Best-fit decreasing bin packing of one representative piece per product.
    order = sorted(
        demanded,
        key=lambda i: (int(products[i]["length"]), int(products[i]["demand"])),
        reverse=True,
    )
    bins = []
    loads = []

    for i in order:
        length = int(products[i]["length"])
        best_bin = None
        best_remaining = None

        for b in range(len(bins)):
            new_load = loads[b] + length
            if new_load <= machine_length:
                remaining = machine_length - new_load
                if best_remaining is None or remaining < best_remaining:
                    best_bin = b
                    best_remaining = remaining

        if best_bin is None:
            if len(bins) >= num_patterns:
                return None
            bins.append([i])
            loads.append(length)
        else:
            bins[best_bin].append(i)
            loads[best_bin] += length

    pattern_rows = []
    for assigned in bins:
        lower = 1
        for i in assigned:
            demand = int(products[i]["demand"])
            capacity = int(products[i]["N_i"])
            lower = max(lower, (demand + capacity - 1) // capacity)

        if lower > M:
            return None

        def configuration_for(repetitions):
            row = [0] * n
            total_length = 0
            for i in assigned:
                demand = int(products[i]["demand"])
                count = (demand + repetitions - 1) // repetitions
                if count > int(products[i]["N_i"]):
                    return None
                row[i] = count
                total_length += count * int(products[i]["length"])
            if total_length > machine_length:
                return None
            return row

        lo, hi = lower, M
        feasible_row = configuration_for(hi)
        if feasible_row is None:
            return None

        while lo < hi:
            mid = (lo + hi) // 2
            row = configuration_for(mid)
            if row is not None:
                hi = mid
                feasible_row = row
            else:
                lo = mid + 1

        repetitions = lo
        feasible_row = configuration_for(repetitions)
        pattern_rows.append((repetitions, feasible_row))

    pattern_rows.sort(key=lambda item: item[0], reverse=True)

    used = [False] * num_patterns
    reps = [0] * num_patterns
    pieces = [[0] * n for _ in range(num_patterns)]

    for p, (rep, row) in enumerate(pattern_rows):
        used[p] = True
        reps[p] = rep
        pieces[p] = row

    return {
        "used": used,
        "repetitions": reps,
        "pieces": pieces,
        "extra": [0] * n,
    }


def build_all_extra_candidate(products, num_patterns, M):
    """Construct an open-ended solution using only extra pieces."""
    n = len(products)
    total_demand = sum(int(product["demand"]) for product in products)

    if total_demand == 0:
        return {
            "used": [False] * num_patterns,
            "repetitions": [0] * num_patterns,
            "pieces": [[0] * n for _ in range(num_patterns)],
            "extra": [0] * n,
        }

    if M <= 0 or total_demand > num_patterns * M:
        return None

    reps = []
    remaining = total_demand
    while remaining > 0:
        amount = min(M, remaining)
        reps.append(amount)
        remaining -= amount

    reps.sort(reverse=True)
    reps += [0] * (num_patterns - len(reps))

    return {
        "used": [rep > 0 for rep in reps],
        "repetitions": reps,
        "pieces": [[0] * n for _ in range(num_patterns)],
        "extra": [int(product["demand"]) for product in products],
    }


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    problem_type = instance["problem_type"]
    machine_length = int(instance["machine_length"])
    num_patterns = int(instance["num_patterns"])
    products = instance["products"]
    num_products = len(products)
    M = int(instance["M"])

    setup_cost = float(instance["cost_setup_pattern"])
    repetition_cost = float(instance["cost_repetition"])
    knife_cost = (
        float(instance.get("cost_knife_setup", 0.0))
        if problem_type == "cscp_with_knife_dependent_setups"
        else 0.0
    )

    closed_ended = (
        problem_type == "closed_ended_cscp"
        or bool(instance.get("x_i_forced_zero", False))
    )

    product_ids = [product["id"] for product in products]
    lengths = [int(product["length"]) for product in products]
    demands = [int(product["demand"]) for product in products]
    capacities = [int(product["N_i"]) for product in products]

    best_solution = None
    best_objective = math.inf

    # Build inexpensive feasible incumbents before invoking the MIP solver.
    candidates = []

    packed_candidate = build_packed_candidate(
        products, num_patterns, machine_length, M
    )
    if packed_candidate is not None:
        candidates.append(packed_candidate)

    if not closed_ended:
        extra_candidate = build_all_extra_candidate(products, num_patterns, M)
        if extra_candidate is not None:
            candidates.append(extra_candidate)

    best_candidate = None
    for candidate in candidates:
        value = candidate_objective(
            candidate, setup_cost, repetition_cost, knife_cost
        )
        if value < best_objective - 1e-8:
            solution = make_output_solution(
                product_ids,
                num_patterns,
                problem_type,
                setup_cost,
                repetition_cost,
                knife_cost,
                candidate["used"],
                candidate["repetitions"],
                candidate["pieces"],
                candidate["extra"],
            )
            best_candidate = candidate
            best_solution = solution
            best_objective = solution["objective_value"]
            if logger:
                logger.log_solution(best_objective, best_solution)

    # The all-zero solution is optimal when all demands are zero.
    if M == 0:
        if best_solution is None:
            zero_candidate = {
                "used": [False] * num_patterns,
                "repetitions": [0] * num_patterns,
                "pieces": [[0] * num_products for _ in range(num_patterns)],
                "extra": [0] * num_products,
            }
            best_solution = make_output_solution(
                product_ids,
                num_patterns,
                problem_type,
                setup_cost,
                repetition_cost,
                knife_cost,
                zero_candidate["used"],
                zero_candidate["repetitions"],
                zero_candidate["pieces"],
                zero_candidate["extra"],
            )
            if logger:
                logger.log_solution(best_solution["objective_value"], best_solution)

        output_directory = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(output_directory, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(best_solution, file, indent=2)
        return

    model = gp.Model("cutting_stock_configuration")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, args.time_limit)

    patterns = range(num_patterns)
    product_range = range(num_products)

    used_var = model.addVars(patterns, vtype=GRB.BINARY, name="used")
    repetition_var = model.addVars(
        patterns, lb=0, ub=M, vtype=GRB.INTEGER, name="repetitions"
    )
    piece_var = model.addVars(
        product_range,
        patterns,
        lb=0,
        vtype=GRB.INTEGER,
        name="pieces",
    )

    if closed_ended:
        extra_var = model.addVars(
            product_range, lb=0, ub=0, vtype=GRB.INTEGER, name="extra"
        )
    else:
        extra_var = model.addVars(
            product_range,
            lb=0,
            ub={i: demands[i] for i in product_range},
            vtype=GRB.INTEGER,
            name="extra",
        )

    # Binary expansion of each repetition count.
    weights = [1 << k for k in range(M.bit_length())]
    bit_range = range(len(weights))
    repetition_bit = model.addVars(
        patterns, bit_range, vtype=GRB.BINARY, name="repetition_bit"
    )

    # product_bit[i,p,k] = pieces[i,p] * repetition_bit[p,k].
    product_bit = model.addVars(
        product_range,
        patterns,
        bit_range,
        lb=0,
        vtype=GRB.INTEGER,
        name="product_bit",
    )

    for p in patterns:
        model.addConstr(repetition_var[p] <= M * used_var[p])
        model.addConstr(repetition_var[p] >= used_var[p])
        model.addConstr(
            repetition_var[p]
            == gp.quicksum(weights[k] * repetition_bit[p, k] for k in bit_range)
        )

        model.addConstr(
            gp.quicksum(lengths[i] * piece_var[i, p] for i in product_range)
            <= machine_length
        )

        for i in product_range:
            cap = capacities[i]
            model.addConstr(piece_var[i, p] <= cap * used_var[p])

            for k in bit_range:
                w = product_bit[i, p, k]
                b = repetition_bit[p, k]
                a = piece_var[i, p]

                model.addConstr(w <= a)
                model.addConstr(w <= cap * b)
                model.addConstr(w >= a - cap * (1 - b))

    # Interchangeable pattern slots are sorted by repetition count.
    for p in range(num_patterns - 1):
        model.addConstr(repetition_var[p] >= repetition_var[p + 1])

    for i in product_range:
        production = gp.quicksum(
            weights[k] * product_bit[i, p, k]
            for p in patterns
            for k in bit_range
        )
        model.addConstr(production + extra_var[i] >= demands[i])

    if not closed_ended:
        model.addConstr(
            gp.quicksum(extra_var[i] for i in product_range)
            <= gp.quicksum(repetition_var[p] for p in patterns)
        )

    objective = (
        setup_cost * gp.quicksum(used_var[p] for p in patterns)
        + repetition_cost * gp.quicksum(repetition_var[p] for p in patterns)
    )

    if problem_type == "cscp_with_knife_dependent_setups":
        objective += knife_cost * gp.quicksum(
            piece_var[i, p] for i in product_range for p in patterns
        )

    model.setObjective(objective, GRB.MINIMIZE)

    # Supply the best heuristic candidate as a complete MIP start.
    if best_candidate is not None:
        for p in patterns:
            used_value = int(best_candidate["used"][p])
            rep_value = int(best_candidate["repetitions"][p])
            used_var[p].Start = used_value
            repetition_var[p].Start = rep_value

            for k, weight in enumerate(weights):
                bit_value = 1 if (rep_value & weight) else 0
                repetition_bit[p, k].Start = bit_value

            for i in product_range:
                piece_value = int(best_candidate["pieces"][p][i])
                piece_var[i, p].Start = piece_value
                for k, weight in enumerate(weights):
                    bit_value = 1 if (rep_value & weight) else 0
                    product_bit[i, p, k].Start = piece_value * bit_value

        for i in product_range:
            extra_var[i].Start = int(best_candidate["extra"][i])

    def extract_solution(value_getter):
        used_values = []
        repetition_values = []
        piece_values = [[0] * num_products for _ in patterns]
        extra_values = []

        for p in patterns:
            rep = max(0, int(round(value_getter(repetition_var[p]))))
            selected = int(round(value_getter(used_var[p]))) > 0
            selected = selected or rep > 0

            repetition_values.append(rep)
            used_values.append(selected)

            for i in product_range:
                piece_values[p][i] = max(
                    0, int(round(value_getter(piece_var[i, p])))
                )

        for i in product_range:
            extra_values.append(max(0, int(round(value_getter(extra_var[i])))))

        return make_output_solution(
            product_ids,
            num_patterns,
            problem_type,
            setup_cost,
            repetition_cost,
            knife_cost,
            used_values,
            repetition_values,
            piece_values,
            extra_values,
        )

    incumbent_state = {
        "objective": best_objective,
        "solution": best_solution,
    }

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            solution = extract_solution(cb_model.cbGetSolution)
            objective_value = solution["objective_value"]

            if objective_value < incumbent_state["objective"] - 1e-7:
                incumbent_state["objective"] = objective_value
                incumbent_state["solution"] = solution
                if logger:
                    logger.log_solution(objective_value, solution)
        except Exception:
            # Logging or extraction must not terminate the optimization.
            return

    try:
        model.optimize(callback)
    except gp.GurobiError:
        # If optimization is interrupted, preserve any heuristic or callback
        # incumbent already obtained.
        pass

    best_solution = incumbent_state["solution"]
    best_objective = incumbent_state["objective"]

    if model.SolCount > 0:
        final_solution = extract_solution(lambda variable: variable.X)
        final_objective = final_solution["objective_value"]

        if final_objective < best_objective - 1e-7:
            best_solution = final_solution
            best_objective = final_objective
            if logger:
                logger.log_solution(best_objective, best_solution)

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, indent=2)


if __name__ == "__main__":
    main()