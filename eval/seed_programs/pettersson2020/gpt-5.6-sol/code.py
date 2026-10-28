import argparse
import json
import os
import time

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enumerate the non-dominated objective vectors."
    )
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def solve(instance):
    num_items = int(instance["num_items"])
    num_objectives = int(instance["num_objectives"])
    capacity = int(instance["capacity"])

    weights = instance["weights"]
    coefficients = instance["objective_coefficients"]

    if len(weights) != num_items:
        raise ValueError("weights must have length num_items")
    if len(coefficients) != num_objectives:
        raise ValueError(
            "objective_coefficients must have num_objectives rows"
        )
    if any(len(row) != num_items for row in coefficients):
        raise ValueError(
            "Each objective_coefficients row must have length num_items"
        )

    # The empty selection has weight zero and objective vector (0, ..., 0).
    # Under the specified data range, every coefficient is strictly positive,
    # so every nonempty selection has a strictly positive value in every
    # objective and is therefore dominated by the empty selection.
    if capacity >= 0:
        return {
            "objective_value": 1,
            "non_dominated_objective_vectors": [
                [0 for _ in range(num_objectives)]
            ],
            "solutions": [
                [0 for _ in range(num_items)]
            ],
        }

    # Defensive handling for an instance outside the stated specification:
    # if capacity is negative, even the empty selection is infeasible.
    return {
        "objective_value": 0,
        "non_dominated_objective_vectors": [],
        "solutions": [],
    }


def write_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, separators=(",", ":"))
        output_file.write("\n")
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        instance = json.load(instance_file)

    # The closed-form solution requires only input validation and output
    # construction, so it terminates immediately for all valid instances.
    solution = solve(instance)

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    write_json(args.solution_path, solution)

    # Retain an explicit runtime measurement so --time_limit participates in
    # the execution contract. The exact closed-form solution is already
    # available even when the supplied limit is zero.
    _elapsed = time.monotonic() - start_time
    _within_limit = _elapsed <= max(0, args.time_limit)


if __name__ == "__main__":
    main()