import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(product_ids, num_patterns, setup_cost, repetition_cost,
                  used_values, repetition_values, piece_values, extra_values):
    patterns = {}
    production = {}

    total_used = 0
    total_repetitions = 0

    for j in range(num_patterns):
        used = bool(used_values.get(j, False))
        repetitions = max(0, int(round(repetition_values.get(j, 0))))

        if not used or repetitions == 0:
            used = False
            repetitions = 0

        pieces = {}
        for product_id in product_ids:
            quantity = max(0, int(round(piece_values.get((product_id, j), 0))))
            if not used:
                quantity = 0

            pieces[str(product_id)] = quantity
            production[
                f"product_{product_id}_pattern_{j}"
            ] = quantity * repetitions

        patterns[str(j)] = {
            "used": used,
            "repetitions": repetitions,
            "pieces": pieces,
        }

        total_used += int(used)
        total_repetitions += repetitions

    extras = {
        str(product_id): max(0, int(round(extra_values.get(product_id, 0))))
        for product_id in product_ids
    }

    objective = (
        setup_cost * total_used
        + repetition_cost * total_repetitions
    )

    return {
        "objective_value": float(objective),
        "patterns": patterns,
        "extra_pieces": extras,
        "production_quantities": production,
    }


def build_singleton_heuristic(instance):
    """Construct a guaranteed feasible solution when one slot per demanded
    product is available and every demanded product fits inside the machine.
    """
    products = instance["products"]
    product_ids = [int(p["id"]) for p in products]
    num_patterns = int(instance["num_patterns"])
    setup_cost = int(instance["cost_setup_pattern"])
    repetition_cost = int(instance["cost_repetition"])
    max_repetitions = int(instance["M"])

    demanded = [
        p for p in products
        if int(p["demand"]) > 0
    ]

    if len(demanded) > num_patterns:
        return None

    entries = []
    for product in demanded:
        demand = int(product["demand"])
        capacity = int(product["N_i"])
        if capacity <= 0:
            return None

        repetitions = int(math.ceil(demand / capacity))
        if repetitions > max_repetitions:
            return None

        entries.append((
            repetitions,
            int(product["id"]),
            capacity,
        ))

    # Required pattern symmetry: repetitions are nonincreasing.
    entries.sort(key=lambda item: (-item[0], item[1]))

    used = {j: False for j in range(num_patterns)}
    repetitions = {j: 0 for j in range(num_patterns)}
    pieces = {
        (product_id, j): 0
        for product_id in product_ids
        for j in range(num_patterns)
    }
    extras = {product_id: 0 for product_id in product_ids}

    for j, (rep, product_id, capacity) in enumerate(entries):
        used[j] = True
        repetitions[j] = rep
        pieces[(product_id, j)] = capacity

    return make_solution(
        product_ids,
        num_patterns,
        setup_cost,
        repetition_cost,
        used,
        repetitions,
        pieces,
        extras,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(
        args.log_path, sense="minimize"
    ) if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    machine_length = int(instance["machine_length"])
    num_patterns = int(instance["num_patterns"])
    setup_cost = int(instance["cost_setup_pattern"])
    repetition_cost = int(instance["cost_repetition"])
    max_repetitions = int(instance["M"])

    products = instance["products"]
    product_ids = [int(p["id"]) for p in products]
    product_by_id = {int(p["id"]): p for p in products}

    lengths = {
        int(p["id"]): int(p["length"])
        for p in products
    }
    demands = {
        int(p["id"]): int(p["demand"])
        for p in products
    }
    capacities = {
        int(p["id"]): max(0, int(p["N_i"]))
        for p in products
    }

    heuristic_solution = build_singleton_heuristic(instance)
    best_logged_objective = float("inf")

    if heuristic_solution is not None:
        best_logged_objective = heuristic_solution["objective_value"]
        if logger:
            logger.log_solution(
                heuristic_solution["objective_value"],
                heuristic_solution,
            )

    model = gp.Model("open_ended_cutting")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    pattern_indices = range(num_patterns)

    # Binary expansion of each pattern's repetition count.
    bit_count = max(1, max_repetitions.bit_length())
    bits = range(bit_count)
    bit_weights = {b: 1 << b for b in bits}

    used = {
        j: model.addVar(vtype=GRB.BINARY, name=f"used_{j}")
        for j in pattern_indices
    }

    repetition_bit = {
        (j, b): model.addVar(
            vtype=GRB.BINARY,
            name=f"repetition_bit_{j}_{b}",
        )
        for j in pattern_indices
        for b in bits
    }

    pieces = {
        (product_id, j): model.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=capacities[product_id],
            name=f"pieces_{product_id}_{j}",
        )
        for product_id in product_ids
        for j in pattern_indices
    }

    # product_linearization[i,j,b] = pieces[i,j] when bit[j,b] is one,
    # and zero otherwise. It may be continuous because the surrounding
    # constraints make its value exact.
    product_linearization = {
        (product_id, j, b): model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=capacities[product_id],
            name=f"product_bit_{product_id}_{j}_{b}",
        )
        for product_id in product_ids
        for j in pattern_indices
        for b in bits
    }

    extra = {
        product_id: model.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=max(0, demands[product_id]),
            name=f"extra_{product_id}",
        )
        for product_id in product_ids
    }

    repetition_expr = {
        j: gp.quicksum(
            bit_weights[b] * repetition_bit[j, b]
            for b in bits
        )
        for j in pattern_indices
    }

    for j in pattern_indices:
        # A selected pattern is repeated at least once and at most M times.
        model.addConstr(
            repetition_expr[j] <= max_repetitions * used[j],
            name=f"repetition_upper_{j}",
        )
        model.addConstr(
            repetition_expr[j] >= used[j],
            name=f"repetition_lower_{j}",
        )

        model.addConstr(
            gp.quicksum(
                lengths[product_id] * pieces[product_id, j]
                for product_id in product_ids
            ) <= machine_length * used[j],
            name=f"pattern_length_{j}",
        )

        for product_id in product_ids:
            capacity = capacities[product_id]
            model.addConstr(
                pieces[product_id, j] <= capacity * used[j],
                name=f"pieces_activation_{product_id}_{j}",
            )

            for b in bits:
                linearized = product_linearization[product_id, j, b]
                bit = repetition_bit[j, b]
                piece_count = pieces[product_id, j]

                model.addConstr(
                    linearized <= piece_count,
                    name=f"product_bit_piece_upper_{product_id}_{j}_{b}",
                )
                model.addConstr(
                    linearized <= capacity * bit,
                    name=f"product_bit_binary_upper_{product_id}_{j}_{b}",
                )
                model.addConstr(
                    linearized >= piece_count - capacity * (1 - bit),
                    name=f"product_bit_lower_{product_id}_{j}_{b}",
                )

    for j in range(num_patterns - 1):
        model.addConstr(
            repetition_expr[j] >= repetition_expr[j + 1],
            name=f"pattern_order_{j}",
        )

    for product_id in product_ids:
        regular_production = gp.quicksum(
            bit_weights[b] * product_linearization[product_id, j, b]
            for j in pattern_indices
            for b in bits
        )
        model.addConstr(
            regular_production + extra[product_id] >= demands[product_id],
            name=f"demand_{product_id}",
        )

    model.addConstr(
        gp.quicksum(extra[product_id] for product_id in product_ids)
        <= gp.quicksum(repetition_expr[j] for j in pattern_indices),
        name="open_end_extra_limit",
    )

    model.setObjective(
        setup_cost * gp.quicksum(used[j] for j in pattern_indices)
        + repetition_cost
        * gp.quicksum(repetition_expr[j] for j in pattern_indices),
        GRB.MINIMIZE,
    )

    # Supply the singleton construction as a MIP start when available.
    if heuristic_solution is not None:
        heuristic_patterns = heuristic_solution["patterns"]
        heuristic_extras = heuristic_solution["extra_pieces"]

        for j in pattern_indices:
            pattern = heuristic_patterns[str(j)]
            rep = int(pattern["repetitions"])
            used[j].Start = 1.0 if pattern["used"] else 0.0

            for b in bits:
                bit_value = (rep >> b) & 1
                repetition_bit[j, b].Start = bit_value

            for product_id in product_ids:
                piece_count = int(pattern["pieces"][str(product_id)])
                pieces[product_id, j].Start = piece_count

                for b in bits:
                    bit_value = (rep >> b) & 1
                    product_linearization[
                        product_id, j, b
                    ].Start = piece_count * bit_value

        for product_id in product_ids:
            extra[product_id].Start = int(
                heuristic_extras[str(product_id)]
            )

    model.update()

    callback_state = {
        "best": best_logged_objective,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return

        try:
            objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective >= callback_state["best"] - 1e-7:
                return

            used_values = {
                j: round(cb_model.cbGetSolution(used[j])) > 0
                for j in pattern_indices
            }
            repetition_values = {
                j: sum(
                    bit_weights[b]
                    * int(round(cb_model.cbGetSolution(
                        repetition_bit[j, b]
                    )))
                    for b in bits
                )
                for j in pattern_indices
            }
            piece_values = {
                (product_id, j): int(round(cb_model.cbGetSolution(
                    pieces[product_id, j]
                )))
                for product_id in product_ids
                for j in pattern_indices
            }
            extra_values = {
                product_id: int(round(cb_model.cbGetSolution(
                    extra[product_id]
                )))
                for product_id in product_ids
            }

            solution = make_solution(
                product_ids,
                num_patterns,
                setup_cost,
                repetition_cost,
                used_values,
                repetition_values,
                piece_values,
                extra_values,
            )
            callback_state["best"] = solution["objective_value"]
            logger.log_solution(solution["objective_value"], solution)
        except Exception:
            # Logging must not interrupt the optimization.
            pass

    elapsed = time.monotonic() - start_time
    remaining = max(0.0, float(args.time_limit) - elapsed)

    if remaining > 0.0:
        model.Params.TimeLimit = remaining
        model.optimize(incumbent_callback)
    elif heuristic_solution is None:
        # Give Gurobi a minimal opportunity to obtain a feasible incumbent
        # when no independent construction was available.
        model.Params.TimeLimit = 0.01
        model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_used = {
            j: int(round(used[j].X)) > 0
            for j in pattern_indices
        }
        final_repetitions = {
            j: sum(
                bit_weights[b] * int(round(repetition_bit[j, b].X))
                for b in bits
            )
            for j in pattern_indices
        }
        final_pieces = {
            (product_id, j): int(round(pieces[product_id, j].X))
            for product_id in product_ids
            for j in pattern_indices
        }
        final_extras = {
            product_id: int(round(extra[product_id].X))
            for product_id in product_ids
        }

        final_solution = make_solution(
            product_ids,
            num_patterns,
            setup_cost,
            repetition_cost,
            final_used,
            final_repetitions,
            final_pieces,
            final_extras,
        )
    elif heuristic_solution is not None:
        final_solution = heuristic_solution
    else:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    if (
        logger
        and final_solution["objective_value"]
        < callback_state["best"] - 1e-7
    ):
        logger.log_solution(
            final_solution["objective_value"],
            final_solution,
        )

    solution_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as output_file:
        json.dump(final_solution, output_file, indent=2)


if __name__ == "__main__":
    main()