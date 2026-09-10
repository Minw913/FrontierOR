import argparse
import json
import math
import os
import random
import time

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


def make_record(data, open_facilities, assignments):
    n = data["num_facilities"]
    m = data["num_customers"]

    opens = sorted(set(int(i) for i in open_facilities))
    assignment_dict = {str(j): int(assignments[j]) for j in range(m)}

    z1 = sum(data["fixed_costs_obj1"][i] for i in opens)
    z2 = sum(data["fixed_costs_obj2"][i] for i in opens)

    for j in range(m):
        i = assignment_dict[str(j)]
        z1 += data["assignment_costs_obj1"][i][j]
        z2 += data["assignment_costs_obj2"][i][j]

    return {
        "z1": int(z1),
        "z2": int(z2),
        "open_facilities": opens,
        "assignments": assignment_dict,
    }


def singleton_solution(record):
    objective = 0.5 * (record["z1"] + record["z2"])
    return {
        "objective_value": float(objective),
        "pareto_front": [[record["z1"], record["z2"]]],
        "solutions": [record],
    }


def greedy_solutions(data, deadline):
    """Generate inexpensive feasible starts by several deterministic heuristics."""
    n = data["num_facilities"]
    m = data["num_customers"]
    demands = data["demands"]
    capacities = data["capacities"]
    c1 = data["assignment_costs_obj1"]
    c2 = data["assignment_costs_obj2"]
    f1 = data["fixed_costs_obj1"]
    f2 = data["fixed_costs_obj2"]

    combined_cost = [
        [c1[i][j] + c2[i][j] for j in range(m)]
        for i in range(n)
    ]

    orders = [
        list(sorted(range(m), key=lambda j: (-demands[j], j))),
        list(sorted(range(m), key=lambda j: (demands[j], j))),
        list(sorted(
            range(m),
            key=lambda j: (
                -(
                    sorted(combined_cost[i][j] for i in range(n))[1]
                    - sorted(combined_cost[i][j] for i in range(n))[0]
                    if n > 1 else 0
                ),
                -demands[j],
                j,
            ),
        )),
    ]

    rng = random.Random(0)
    base = list(range(m))
    for _ in range(12):
        order = base[:]
        rng.shuffle(order)
        order.sort(key=lambda j: -demands[j] + rng.random() * 0.25)
        orders.append(order)

    records = []
    seen = set()

    # Different values trade assignment cost against packing tightness.
    modes = ("cost", "best_fit", "capacity")

    for order in orders:
        if time.monotonic() >= deadline:
            break

        for mode in modes:
            residual = capacities[:]
            assignment = [-1] * m
            feasible = True

            for j in order:
                candidates = [
                    i for i in range(n)
                    if residual[i] >= demands[j]
                ]
                if not candidates:
                    feasible = False
                    break

                if mode == "cost":
                    i = min(
                        candidates,
                        key=lambda k: (
                            combined_cost[k][j],
                            residual[k] - demands[j],
                            k,
                        ),
                    )
                elif mode == "best_fit":
                    i = min(
                        candidates,
                        key=lambda k: (
                            residual[k] - demands[j],
                            combined_cost[k][j],
                            k,
                        ),
                    )
                else:
                    i = max(
                        candidates,
                        key=lambda k: (
                            residual[k],
                            -combined_cost[k][j],
                            -k,
                        ),
                    )

                assignment[j] = i
                residual[i] -= demands[j]

            if not feasible:
                continue

            used = set(assignment)
            # An unused facility is worth opening for the equal-weight objective
            # only when its combined fixed cost is negative.
            opens = {
                i for i in range(n)
                if i in used or f1[i] + f2[i] < 0
            }

            record = make_record(data, opens, assignment)
            signature = (
                tuple(record["open_facilities"]),
                tuple(record["assignments"][str(j)] for j in range(m)),
            )
            if signature not in seen:
                seen.add(signature)
                records.append(record)

    return records


def nondominated_records(records):
    """Keep one representative for each mutually nondominated objective pair."""
    by_pair = {}
    for record in records:
        pair = (int(record["z1"]), int(record["z2"]))
        if pair not in by_pair:
            by_pair[pair] = record

    # In increasing z1 order, a point is nondominated exactly when it strictly
    # improves the best z2 encountered so far.
    result = []
    best_z2 = math.inf
    for pair in sorted(by_pair):
        z1, z2 = pair
        if z2 < best_z2:
            result.append(by_pair[pair])
            best_z2 = z2

    return result


def build_output(records):
    front_records = nondominated_records(records)
    if not front_records:
        return {
            "objective_value": 0.0,
            "pareto_front": [],
            "solutions": [],
        }

    objective = min(0.5 * (r["z1"] + r["z2"]) for r in front_records)
    return {
        "objective_value": float(objective),
        "pareto_front": [
            [int(r["z1"]), int(r["z2"])] for r in front_records
        ],
        "solutions": front_records,
    }


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_facilities"])
    m = int(data["num_customers"])

    demands = data["demands"]
    capacities = data["capacities"]
    f1 = data["fixed_costs_obj1"]
    f2 = data["fixed_costs_obj2"]
    c1 = data["assignment_costs_obj1"]
    c2 = data["assignment_costs_obj2"]

    all_records = []
    candidate_pairs = set()
    best_logged_sum = [math.inf]

    def add_record(record, do_log=True):
        pair = (record["z1"], record["z2"])
        if pair not in candidate_pairs:
            candidate_pairs.add(pair)
            all_records.append(record)

        weighted_sum_twice = record["z1"] + record["z2"]
        if do_log and weighted_sum_twice < best_logged_sum[0]:
            best_logged_sum[0] = weighted_sum_twice
            if logger:
                logger.log_solution(
                    0.5 * weighted_sum_twice,
                    singleton_solution(record),
                )

    # Obtain a feasible start before invoking the solver whenever possible.
    heuristic_deadline = min(deadline, time.monotonic() + 0.2)
    heuristic_records = greedy_solutions(data, heuristic_deadline)
    for record in heuristic_records:
        add_record(record)

    model = gp.Model("biobjective_capacitated_facility_location")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    y = model.addVars(n, vtype=GRB.BINARY, name="open")
    x = model.addVars(n, m, vtype=GRB.BINARY, name="assign")

    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == 1,
            name=f"serve_{j}",
        )

    for i in range(n):
        model.addConstr(
            gp.quicksum(demands[j] * x[i, j] for j in range(m))
            <= capacities[i] * y[i],
            name=f"capacity_{i}",
        )
        for j in range(m):
            model.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")

    z1_expr = (
        gp.quicksum(f1[i] * y[i] for i in range(n))
        + gp.quicksum(c1[i][j] * x[i, j] for i in range(n) for j in range(m))
    )
    z2_expr = (
        gp.quicksum(f2[i] * y[i] for i in range(n))
        + gp.quicksum(c2[i][j] * x[i, j] for i in range(n) for j in range(m))
    )

    # Supply the best heuristic as a MIP start.
    if heuristic_records:
        best_start = min(
            heuristic_records,
            key=lambda r: r["z1"] + r["z2"],
        )
        open_set = set(best_start["open_facilities"])
        for i in range(n):
            y[i].Start = 1.0 if i in open_set else 0.0
        for i in range(n):
            for j in range(m):
                x[i, j].Start = (
                    1.0
                    if best_start["assignments"][str(j)] == i
                    else 0.0
                )

    y_list = [y[i] for i in range(n)]
    x_list = [x[i, j] for i in range(n) for j in range(m)]

    def record_from_values(y_values, x_values):
        assignments = []
        for j in range(m):
            best_i = max(
                range(n),
                key=lambda i: x_values[i * m + j],
            )
            assignments.append(best_i)

        opens = {
            i for i in range(n)
            if y_values[i] > 0.5
        }
        opens.update(assignments)
        return make_record(data, opens, assignments)

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            y_values = cb_model.cbGetSolution(y_list)
            x_values = cb_model.cbGetSolution(x_list)
            record = record_from_values(y_values, x_values)

            # Logging is based on the requested equal-weight objective, even
            # while epsilon-constraint subproblems use another objective.
            total = record["z1"] + record["z2"]
            if total < best_logged_sum[0]:
                best_logged_sum[0] = total
                if logger:
                    logger.log_solution(
                        0.5 * total,
                        singleton_solution(record),
                    )
        except Exception:
            # Logging must not interrupt the optimization.
            pass

    def remaining_time():
        return max(0.0, deadline - time.monotonic())

    def optimize_with_limit(limit):
        available = remaining_time()
        if available <= 0.0:
            return False
        model.Params.TimeLimit = max(0.001, min(available, limit))
        model.optimize(callback)
        return True

    def add_current_incumbent():
        if model.SolCount <= 0:
            return None
        y_values = [var.X for var in y_list]
        x_values = [var.X for var in x_list]
        record = record_from_values(y_values, x_values)
        add_record(record)
        return record

    # First directly target the score used for evaluation. This also supplies
    # a strong incumbent and warm start for Pareto enumeration.
    remaining = remaining_time()
    if remaining > 0.0:
        weighted_budget = min(30.0, max(0.5, 0.15 * remaining))
        model.setObjective(z1_expr + z2_expr, GRB.MINIMIZE)
        optimize_with_limit(weighted_budget)
        add_current_incumbent()

    # Exact epsilon-constraint enumeration:
    #   lexicographically minimize (z1,z2), then require z2 to improve by one.
    # Integer costs make z2 <= previous_z2 - 1 exact.
    cutoff_constraint = None

    while remaining_time() > 0.0:
        model.setObjective(z1_expr, GRB.MINIMIZE)
        if not optimize_with_limit(remaining_time()):
            break

        first_status = model.Status
        first_record = add_current_incumbent()

        if first_status == GRB.INFEASIBLE:
            # No point with a smaller z2 exists; enumeration is complete.
            break
        if first_status != GRB.OPTIMAL or first_record is None:
            # Preserve the best feasible incumbent, but do not make an
            # uncertified epsilon step.
            break

        optimal_z1 = int(first_record["z1"])
        z1_fix = model.addConstr(z1_expr == optimal_z1, name="lex_z1_fix")
        model.setObjective(z2_expr, GRB.MINIMIZE)

        if not optimize_with_limit(remaining_time()):
            model.remove(z1_fix)
            model.update()
            break

        second_status = model.Status
        second_record = add_current_incumbent()

        model.remove(z1_fix)
        model.update()

        if second_status != GRB.OPTIMAL or second_record is None:
            break

        pareto_z2 = int(second_record["z2"])

        if cutoff_constraint is not None:
            model.remove(cutoff_constraint)
            model.update()

        cutoff_constraint = model.addConstr(
            z2_expr <= pareto_z2 - 1,
            name="pareto_z2_cutoff",
        )
        model.update()

    output = build_output(all_records)

    # If a final front snapshot improves the logged equal-weight incumbent,
    # record it with the complete currently known output.
    if output["solutions"]:
        final_twice = int(round(2.0 * output["objective_value"]))
        if final_twice < best_logged_sum[0]:
            best_logged_sum[0] = final_twice
            if logger:
                logger.log_solution(output["objective_value"], output)

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()