import argparse
import json
import math
from typing import Dict, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    problem_type = instance["problem_type"]
    machine_length = int(instance["machine_length"])
    num_patterns = int(instance["num_patterns"])
    M = int(instance["M"])

    cost_setup_pattern = float(instance["cost_setup_pattern"])
    cost_repetition = float(instance["cost_repetition"])
    cost_knife_setup = float(instance.get("cost_knife_setup", 0.0))

    products = instance["products"]
    product_ids = [int(p["id"]) for p in products]
    lengths = [int(p["length"]) for p in products]
    demands = [int(p["demand"]) for p in products]
    N_i = [int(p["N_i"]) for p in products]
    num_products = len(products)

    closed_forced = bool(instance.get("x_i_forced_zero", False)) or (problem_type == "closed_ended_cscp")
    knife_variant = (problem_type == "cscp_with_knife_dependent_setups")
    open_variant = not closed_forced  # open_ended + knife-dependent variants

    model = gp.Model("cscp")

    # Required solver parameters
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)

    # Optional: quieter logs
    model.Params.OutputFlag = 0

    P = range(num_patterns)
    I = range(num_products)

    # Bit-length for repetition decomposition
    K = max(1, math.ceil(math.log2(M + 1))) if M >= 0 else 1
    B = range(K)
    bit_val = [1 << k for k in B]

    # Variables
    y = model.addVars(P, vtype=GRB.BINARY, name="used")
    r = model.addVars(P, vtype=GRB.INTEGER, lb=0, ub=M, name="repetitions")

    a = {}
    for i in I:
        ub_ai = N_i[i]
        for p in P:
            a[i, p] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_ai, name=f"a_{i}_{p}")

    b = model.addVars(P, B, vtype=GRB.BINARY, name="rep_bits")

    # w[i,p,k] = a[i,p] * b[p,k]
    w = {}
    for i in I:
        ub_ai = N_i[i]
        for p in P:
            for k in B:
                w[i, p, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=ub_ai, name=f"w_{i}_{p}_{k}")

    # Extra pieces
    x = {}
    for i in I:
        if closed_forced:
            x[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"x_{i}")
        else:
            # No need to exceed demand in optimal solutions with positive costs
            x[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=demands[i], name=f"x_{i}")

    model.update()

    # Constraints
    for p in P:
        # Repetitions only if used
        model.addConstr(r[p] <= M * y[p], name=f"rep_if_used_{p}")

        # Binary expansion of repetitions
        model.addConstr(
            r[p] == gp.quicksum(bit_val[k] * b[p, k] for k in B),
            name=f"rep_bits_def_{p}"
        )

        # Pieces only if used
        for i in I:
            model.addConstr(a[i, p] <= N_i[i] * y[p], name=f"a_if_used_{i}_{p}")

        # Pattern length capacity
        model.addConstr(
            gp.quicksum(lengths[i] * a[i, p] for i in I) <= machine_length * y[p],
            name=f"length_cap_{p}"
        )

        # If used, pattern has at least one cut
        model.addConstr(
            gp.quicksum(a[i, p] for i in I) >= y[p],
            name=f"nonempty_used_{p}"
        )

    # Linearization: w = a * b
    for i in I:
        ub_ai = N_i[i]
        for p in P:
            for k in B:
                model.addConstr(w[i, p, k] <= ub_ai * b[p, k], name=f"w_ub1_{i}_{p}_{k}")
                model.addConstr(w[i, p, k] <= a[i, p], name=f"w_ub2_{i}_{p}_{k}")
                model.addConstr(
                    w[i, p, k] >= a[i, p] - ub_ai * (1 - b[p, k]),
                    name=f"w_lb_{i}_{p}_{k}"
                )

    # Demand satisfaction
    for i in I:
        produced_i = gp.quicksum(bit_val[k] * w[i, p, k] for p in P for k in B)
        model.addConstr(produced_i + x[i] >= demands[i], name=f"demand_{i}")

    # Extra-piece rule for open-ended variants
    if open_variant:
        model.addConstr(
            gp.quicksum(x[i] for i in I) <= gp.quicksum(r[p] for p in P),
            name="extra_limit"
        )

    # Symmetry breaking
    for p in range(num_patterns - 1):
        model.addConstr(r[p] >= r[p + 1], name=f"sym_rep_{p}")
        model.addConstr(y[p] >= y[p + 1], name=f"sym_use_{p}")

    # Objective
    obj = (
        cost_setup_pattern * gp.quicksum(y[p] for p in P) +
        cost_repetition * gp.quicksum(r[p] for p in P)
    )
    if knife_variant:
        obj += cost_knife_setup * gp.quicksum(a[i, p] for i in I for p in P)

    model.setObjective(obj, GRB.MINIMIZE)

    # Solution extraction helper
    def build_solution(getv):
        patterns_out = {}
        extra_out = {}
        production_out = {}

        used_sum = 0
        rep_sum = 0
        knife_sum = 0

        # Cache values
        a_val = {}
        r_val = {}
        y_val = {}
        x_val = {}

        for p in P:
            yv = 1 if getv(y[p]) >= 0.5 else 0
            rv = int(round(getv(r[p])))
            rv = max(0, min(M, rv))
            y_val[p] = yv
            r_val[p] = rv
            used_sum += yv
            rep_sum += rv

        for i in I:
            xv = int(round(getv(x[i])))
            xv = max(0, xv)
            x_val[i] = xv
            extra_out[str(product_ids[i])] = xv

        for p in P:
            pieces_dict = {}
            knife_count = 0
            for i in I:
                av = int(round(getv(a[i, p])))
                av = max(0, min(N_i[i], av))
                a_val[i, p] = av
                pieces_dict[str(product_ids[i])] = av
                knife_count += av

                production_out[f"product_{product_ids[i]}_pattern_{p}"] = av * r_val[p]

            knife_sum += knife_count

            patt = {
                "used": bool(y_val[p]),
                "repetitions": int(r_val[p]),
                "pieces": pieces_dict
            }
            if knife_variant:
                patt["knife_count"] = int(knife_count)

            patterns_out[str(p)] = patt

        objective_value = cost_setup_pattern * used_sum + cost_repetition * rep_sum
        if knife_variant:
            objective_value += cost_knife_setup * knife_sum

        return {
            "objective_value": float(objective_value),
            "patterns": patterns_out,
            "extra_pieces": extra_out,
            "production_quantities": production_out
        }

    # Callback for incumbent logging
    model._logger = logger
    model._best_logged = float("inf")
    model._build_solution = build_solution

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            sol = m._build_solution(lambda var: m.cbGetSolution(var))
            objv = sol["objective_value"]
            if objv + 1e-9 < m._best_logged:
                m._best_logged = objv
                m._logger.log_solution(objv, sol)

    model.optimize(cb)

    # Final solution
    if model.SolCount > 0:
        final_solution = build_solution(lambda var: var.X)
    else:
        # Fallback (should rarely happen if instance is feasible)
        patterns_out = {}
        production_out = {}
        for p in P:
            pieces = {str(pid): 0 for pid in product_ids}
            patt = {"used": False, "repetitions": 0, "pieces": pieces}
            if knife_variant:
                patt["knife_count"] = 0
            patterns_out[str(p)] = patt
            for i in I:
                production_out[f"product_{product_ids[i]}_pattern_{p}"] = 0

        extra_out = {str(pid): 0 for pid in product_ids}
        final_solution = {
            "objective_value": float(1e30),
            "patterns": patterns_out,
            "extra_pieces": extra_out,
            "production_quantities": production_out
        }

    return final_solution


def main():
    parser = argparse.ArgumentParser(description="Solve CSCP variant instances.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()