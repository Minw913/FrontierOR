import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def configure_gurobi(model, time_limit=None):
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = max(0.01, float(time_limit))


def dict_value(mapping, key, default=0.0):
    if mapping is None:
        return default
    if str(key) in mapping:
        return float(mapping[str(key)])
    if key in mapping:
        return float(mapping[key])
    return default


def euclidean(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def make_solution(objective_value, pattern):
    return {
        "objective_value": float(objective_value),
        "interdiction_pattern": {
            key: int(value) for key, value in pattern.items()
        },
    }


def no_attack_pattern(n1, n2):
    pattern = {}
    for s in range(n1):
        pattern[f"S1_{s}"] = 0
    for b in range(n2):
        pattern[f"S2_{b}"] = 0
    return pattern


def prepare_data(instance):
    g = instance["global_parameters"]
    ip = instance["instance_parameters"]
    interdiction = instance["interdiction_parameters"]
    customers = instance["customers"]
    level1 = instance["level1_facilities"]
    level2 = instance["level2_facilities"]
    distances = instance.get("distances", {})

    n1 = int(ip["num_level1_facilities"])
    n2 = int(ip["num_level2_facilities"])
    nc = int(ip["num_customers"])

    theta = float(g["theta"])
    sigma = float(g["sigma"])
    alpha = float(g["alpha"])
    beta = float(g["beta"])
    gamma = float(g["gamma"])
    alpha_prime = float(g["alpha_prime"])
    beta_prime = float(g["beta_prime"])
    gamma_prime = float(g["gamma_prime"])

    demands = [float(v) for v in customers["demands"]]
    demand1 = [theta * d for d in demands]
    demand2 = [(1.0 - theta) * d for d in demands]

    cap1_l1 = [float(v) for v in level1["capacity_type1"]]
    cap1_l2 = [float(v) for v in level2["capacity_type1"]]
    cap2_l2 = [float(v) for v in level2["capacity_type2"]]

    d_c_l1 = distances.get("customer_to_level1_facility")
    if d_c_l1 is None:
        d_c_l1 = [
            [
                euclidean(
                    float(customers["x_coordinates"][i]),
                    float(customers["y_coordinates"][i]),
                    float(level1["x_coordinates"][s]),
                    float(level1["y_coordinates"][s]),
                )
                for s in range(n1)
            ]
            for i in range(nc)
        ]
    else:
        d_c_l1 = [[float(v) for v in row] for row in d_c_l1]

    d_c_l2 = distances.get("customer_to_level2_facility")
    if d_c_l2 is None:
        d_c_l2 = [
            [
                euclidean(
                    float(customers["x_coordinates"][i]),
                    float(customers["y_coordinates"][i]),
                    float(level2["x_coordinates"][b]),
                    float(level2["y_coordinates"][b]),
                )
                for b in range(n2)
            ]
            for i in range(nc)
        ]
    else:
        d_c_l2 = [[float(v) for v in row] for row in d_c_l2]

    d_l1_l2 = distances.get("level1_to_level2_facility")
    if d_l1_l2 is None:
        d_l1_l2 = [
            [
                euclidean(
                    float(level1["x_coordinates"][s]),
                    float(level1["y_coordinates"][s]),
                    float(level2["x_coordinates"][b]),
                    float(level2["y_coordinates"][b]),
                )
                for b in range(n2)
            ]
            for s in range(n1)
        ]
    else:
        d_l1_l2 = [[float(v) for v in row] for row in d_l1_l2]

    # Referral distances between level-II facilities are not supplied directly
    # in the schema, so compute them from coordinates.
    d_l2_l2 = [
        [
            euclidean(
                float(level2["x_coordinates"][s]),
                float(level2["y_coordinates"][s]),
                float(level2["x_coordinates"][b]),
                float(level2["y_coordinates"][b]),
            )
            for b in range(n2)
        ]
        for s in range(n2)
    ]

    k_max = int(ip["k_max"])
    levels = list(range(k_max + 1))

    h1_map = interdiction.get("interdiction_cost_level1_h1", {})
    h2_map = interdiction.get("interdiction_cost_level2_h2", {})
    red1_map = interdiction.get("capacity_reduction_level1_d1", {})
    red2_map = interdiction.get("capacity_reduction_level2_d2", {})

    h1 = {k: (0.0 if k == 0 else dict_value(h1_map, k)) for k in levels}
    h2 = {k: (0.0 if k == 0 else dict_value(h2_map, k)) for k in levels}
    red1 = {k: (0.0 if k == 0 else dict_value(red1_map, k)) for k in levels}
    red2 = {k: (0.0 if k == 0 else dict_value(red2_map, k)) for k in levels}

    # Guard against tiny input roundoff outside the intended [0,1] interval.
    red1 = {k: min(1.0, max(0.0, v)) for k, v in red1.items()}
    red2 = {k: min(1.0, max(0.0, v)) for k, v in red2.items()}

    return {
        "n1": n1,
        "n2": n2,
        "nc": nc,
        "levels": levels,
        "sigma": sigma,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "alpha_prime": alpha_prime,
        "beta_prime": beta_prime,
        "gamma_prime": gamma_prime,
        "demand1": demand1,
        "demand2": demand2,
        "cap1_l1": cap1_l1,
        "cap1_l2": cap1_l2,
        "cap2_l2": cap2_l2,
        "d_c_l1": d_c_l1,
        "d_c_l2": d_c_l2,
        "d_l1_l2": d_l1_l2,
        "d_l2_l2": d_l2_l2,
        "h1": h1,
        "h2": h2,
        "red1": red1,
        "red2": red2,
        "budget": float(ip["total_interdiction_budget"]),
    }


def referral_distance(data, source, destination):
    n1 = data["n1"]
    if source < n1:
        return data["d_l1_l2"][source][destination]
    return data["d_l2_l2"][source - n1][destination]


def evaluate_pattern_dual(data, pattern, time_limit):
    """
    Solve the defender LP through its dual for a fixed interdiction pattern.
    The dual is substantially smaller than the full allocation formulation.
    """
    n1 = data["n1"]
    n2 = data["n2"]
    nc = data["nc"]
    sigma = data["sigma"]

    outsource1_total = max(
        0.0, data["alpha_prime"] + sigma * data["gamma_prime"]
    )
    max_type2_value = max(
        0.0, data["beta_prime"], data["gamma_prime"]
    )

    cap1 = []
    for s in range(n1):
        k = int(pattern[f"S1_{s}"])
        cap1.append(data["cap1_l1"][s] * (1.0 - data["red1"][k]))
    for b in range(n2):
        k = int(pattern[f"S2_{b}"])
        cap1.append(data["cap1_l2"][b] * (1.0 - data["red2"][k]))

    cap2 = []
    for b in range(n2):
        k = int(pattern[f"S2_{b}"])
        cap2.append(data["cap2_l2"][b] * (1.0 - data["red2"][k]))

    model = gp.Model("fixed_pattern_defender_dual")
    configure_gurobi(model, time_limit)

    u = model.addVars(
        nc, lb=0.0, ub=outsource1_total, vtype=GRB.CONTINUOUS, name="u"
    )
    v = model.addVars(
        nc, lb=0.0, ub=max(0.0, data["beta_prime"]),
        vtype=GRB.CONTINUOUS, name="v"
    )
    w = model.addVars(
        n1 + n2, lb=0.0, ub=max(0.0, data["gamma_prime"]),
        vtype=GRB.CONTINUOUS, name="w"
    )
    a = model.addVars(
        n1 + n2, lb=0.0, ub=outsource1_total,
        vtype=GRB.CONTINUOUS, name="a"
    )
    bdual = model.addVars(
        n2, lb=0.0, ub=max_type2_value,
        vtype=GRB.CONTINUOUS, name="b"
    )

    for i in range(nc):
        for s in range(n1):
            model.addConstr(
                u[i] - sigma * w[s] - a[s]
                <= data["alpha"] * data["d_c_l1"][i][s]
            )
        for j in range(n2):
            source = n1 + j
            model.addConstr(
                u[i] - sigma * w[source] - a[source]
                <= data["beta"] * data["d_c_l2"][i][j]
            )
            model.addConstr(
                v[i] - bdual[j]
                <= data["beta"] * data["d_c_l2"][i][j]
            )

    for source in range(n1 + n2):
        for j in range(n2):
            model.addConstr(
                w[source] - bdual[j]
                <= data["gamma"] * referral_distance(data, source, j)
            )

    objective = gp.quicksum(data["demand1"][i] * u[i] for i in range(nc))
    objective += gp.quicksum(data["demand2"][i] * v[i] for i in range(nc))
    objective -= gp.quicksum(cap1[s] * a[s] for s in range(n1 + n2))
    objective -= gp.quicksum(cap2[j] * bdual[j] for j in range(n2))

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount > 0:
        return float(model.ObjVal)
    return None


def build_interdiction_model(data, time_limit):
    n1 = data["n1"]
    n2 = data["n2"]
    nc = data["nc"]
    levels = data["levels"]
    sigma = data["sigma"]

    outsource1_total = max(
        0.0, data["alpha_prime"] + sigma * data["gamma_prime"]
    )
    max_type2_value = max(
        0.0, data["beta_prime"], data["gamma_prime"]
    )

    model = gp.Model("nested_hierarchical_interdiction")
    configure_gurobi(model, time_limit)

    z1 = model.addVars(n1, levels, vtype=GRB.BINARY, name="z1")
    z2 = model.addVars(n2, levels, vtype=GRB.BINARY, name="z2")

    for s in range(n1):
        model.addConstr(gp.quicksum(z1[s, k] for k in levels) == 1)
    for j in range(n2):
        model.addConstr(gp.quicksum(z2[j, k] for k in levels) == 1)

    model.addConstr(
        gp.quicksum(data["h1"][k] * z1[s, k]
                    for s in range(n1) for k in levels)
        + gp.quicksum(data["h2"][k] * z2[j, k]
                      for j in range(n2) for k in levels)
        <= data["budget"],
        name="interdiction_budget",
    )

    u = model.addVars(
        nc, lb=0.0, ub=outsource1_total,
        vtype=GRB.CONTINUOUS, name="u"
    )
    v = model.addVars(
        nc, lb=0.0, ub=max(0.0, data["beta_prime"]),
        vtype=GRB.CONTINUOUS, name="v"
    )
    w = model.addVars(
        n1 + n2, lb=0.0, ub=max(0.0, data["gamma_prime"]),
        vtype=GRB.CONTINUOUS, name="w"
    )
    a = model.addVars(
        n1 + n2, lb=0.0, ub=outsource1_total,
        vtype=GRB.CONTINUOUS, name="a"
    )
    bdual = model.addVars(
        n2, lb=0.0, ub=max_type2_value,
        vtype=GRB.CONTINUOUS, name="b"
    )

    # Products a*z and b*z are linearized exactly using valid dual bounds.
    ya1 = model.addVars(
        n1, levels, lb=0.0, ub=outsource1_total,
        vtype=GRB.CONTINUOUS, name="ya1"
    )
    ya2 = model.addVars(
        n2, levels, lb=0.0, ub=outsource1_total,
        vtype=GRB.CONTINUOUS, name="ya2"
    )
    yb2 = model.addVars(
        n2, levels, lb=0.0, ub=max_type2_value,
        vtype=GRB.CONTINUOUS, name="yb2"
    )

    for s in range(n1):
        for k in levels:
            y = ya1[s, k]
            z = z1[s, k]
            model.addConstr(y <= a[s])
            model.addConstr(y <= outsource1_total * z)
            model.addConstr(y >= a[s] - outsource1_total * (1.0 - z))

    for j in range(n2):
        source = n1 + j
        for k in levels:
            y = ya2[j, k]
            z = z2[j, k]
            model.addConstr(y <= a[source])
            model.addConstr(y <= outsource1_total * z)
            model.addConstr(y >= a[source] - outsource1_total * (1.0 - z))

            yb = yb2[j, k]
            model.addConstr(yb <= bdual[j])
            model.addConstr(yb <= max_type2_value * z)
            model.addConstr(
                yb >= bdual[j] - max_type2_value * (1.0 - z)
            )

    # Dual feasibility constraints for type-I customer allocations.
    for i in range(nc):
        for s in range(n1):
            model.addConstr(
                u[i] - sigma * w[s] - a[s]
                <= data["alpha"] * data["d_c_l1"][i][s]
            )
        for j in range(n2):
            source = n1 + j
            model.addConstr(
                u[i] - sigma * w[source] - a[source]
                <= data["beta"] * data["d_c_l2"][i][j]
            )

    # Direct type-II allocations.
    for i in range(nc):
        for j in range(n2):
            model.addConstr(
                v[i] - bdual[j]
                <= data["beta"] * data["d_c_l2"][i][j]
            )

    # Referential type-II forwarding.
    for source in range(n1 + n2):
        for j in range(n2):
            model.addConstr(
                w[source] - bdual[j]
                <= data["gamma"] * referral_distance(data, source, j)
            )

    objective = gp.quicksum(
        data["demand1"][i] * u[i] + data["demand2"][i] * v[i]
        for i in range(nc)
    )

    for s in range(n1):
        objective -= data["cap1_l1"][s] * a[s]
        objective += data["cap1_l1"][s] * gp.quicksum(
            data["red1"][k] * ya1[s, k] for k in levels
        )

    for j in range(n2):
        source = n1 + j

        objective -= data["cap1_l2"][j] * a[source]
        objective += data["cap1_l2"][j] * gp.quicksum(
            data["red2"][k] * ya2[j, k] for k in levels
        )

        objective -= data["cap2_l2"][j] * bdual[j]
        objective += data["cap2_l2"][j] * gp.quicksum(
            data["red2"][k] * yb2[j, k] for k in levels
        )

    model.setObjective(objective, GRB.MAXIMIZE)
    return model, z1, z2


def extract_pattern(model, z1, z2, data):
    pattern = {}
    levels = data["levels"]

    for s in range(data["n1"]):
        best_k = max(levels, key=lambda k: z1[s, k].X)
        pattern[f"S1_{s}"] = int(best_k)

    for j in range(data["n2"]):
        best_k = max(levels, key=lambda k: z2[j, k].X)
        pattern[f"S2_{j}"] = int(best_k)

    return pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    data = prepare_data(instance)
    baseline_pattern = no_attack_pattern(data["n1"], data["n2"])

    # Establish a guaranteed feasible outer solution before the MIP search.
    remaining = max(0.01, deadline - time.monotonic())
    baseline_value = evaluate_pattern_dual(
        data, baseline_pattern, min(remaining, max(0.1, 0.15 * args.time_limit))
    )

    if baseline_value is None:
        # This is only an emergency fallback. The fixed-pattern LP is bounded
        # and normally solves essentially instantly.
        baseline_value = 0.0

    best_pattern = baseline_pattern
    best_value = baseline_value
    if logger:
        logger.log_solution(
            best_value, make_solution(best_value, best_pattern)
        )

    remaining = deadline - time.monotonic()
    if remaining > 0.05:
        # Reserve a small amount of time to re-evaluate the final pattern.
        reserve = min(2.0, max(0.05, 0.05 * args.time_limit))
        mip_time = max(0.01, remaining - reserve)

        model, z1, z2 = build_interdiction_model(data, mip_time)
        callback_best = [best_value]

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj <= callback_best[0] + 1e-7:
                    return

                pattern = {}
                for s in range(data["n1"]):
                    values = [
                        cb_model.cbGetSolution(z1[s, k])
                        for k in data["levels"]
                    ]
                    chosen = data["levels"][
                        max(range(len(values)), key=lambda idx: values[idx])
                    ]
                    pattern[f"S1_{s}"] = int(chosen)

                for j in range(data["n2"]):
                    values = [
                        cb_model.cbGetSolution(z2[j, k])
                        for k in data["levels"]
                    ]
                    chosen = data["levels"][
                        max(range(len(values)), key=lambda idx: values[idx])
                    ]
                    pattern[f"S2_{j}"] = int(chosen)

                callback_best[0] = obj
                if logger:
                    logger.log_solution(obj, make_solution(obj, pattern))
            except (gp.GurobiError, AttributeError, ValueError):
                pass

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            candidate_pattern = extract_pattern(model, z1, z2, data)
            remaining = max(0.01, deadline - time.monotonic())
            candidate_value = evaluate_pattern_dual(
                data, candidate_pattern, remaining
            )
            if candidate_value is None:
                candidate_value = float(model.ObjVal)

            if candidate_value > best_value + 1e-7:
                best_value = candidate_value
                best_pattern = candidate_pattern
                if logger:
                    logger.log_solution(
                        best_value,
                        make_solution(best_value, best_pattern),
                    )

    final_solution = make_solution(best_value, best_pattern)

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()