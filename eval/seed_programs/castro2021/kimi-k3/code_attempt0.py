import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    m = int(data["m"])
    supplies = [float(v) for v in data["supplies"]]
    demands = [float(v) for v in data["demands"]]
    cap = float(data["arc_capacity"])
    linear = data.get("linear_costs", [[0.0] * m for _ in range(n)])
    quadratic = data.get("quadratic_costs", [[0.0] * m for _ in range(n)])
    cost_type = data.get("cost_type", "linear_integer")
    is_quadratic = cost_type == "quadratic"
    return n, m, supplies, demands, cap, linear, quadratic, is_quadratic


def arcs_iterator(n, m):
    for i in range(n):
        for j in range(m):
            yield i, j


def compute_objective(flows, n, m, linear, quadratic, is_quadratic):
    obj = 0.0
    for i in range(n):
        fi = flows[i]
        li = linear[i]
        qi = quadratic[i]
        for j in range(m):
            x = fi[j]
            if x:
                obj += li[j] * x
                if is_quadratic:
                    q = qi[j]
                    if q:
                        obj += q * x * x
    return obj


def greedy_heuristic(n, m, supplies, demands, cap, linear, quadratic, is_quadratic):
    """Fast greedy feasible allocation; used as a fallback / starting point."""
    rem = list(supplies)
    flows = [[0.0] * m for _ in range(n)]
    for j in range(m):
        need = demands[j]
        if need <= 0.0:
            continue
        if is_quadratic:
            order = sorted(range(n), key=lambda i: (linear[i][j], quadratic[i][j]))
        else:
            order = sorted(range(n), key=lambda i: linear[i][j])
        for i in order:
            if need <= 1e-12:
                break
            avail = rem[i]
            if avail <= 0.0:
                continue
            take = avail if avail < cap else cap
            if take > need:
                take = need
            if take > 0.0:
                flows[i][j] = take
                rem[i] -= take
                need -= take
        if need > 1e-7:
            return None
    return flows


def build_solution_dict(flows, n, m, linear, quadratic, is_quadratic):
    flows_dict = {}
    for i in range(n):
        fi = flows[i]
        for j in range(m):
            v = fi[j]
            if abs(v) < 1e-12:
                v = 0.0
            flows_dict["x_{}_{}".format(i, j)] = v
    obj = compute_objective(flows, n, m, linear, quadratic, is_quadratic)
    return {"objective_value": float(obj), "flows": flows_dict}


def solve(instance_path, solution_path, time_limit, logger):
    start = time.time()
    n, m, supplies, demands, cap, linear, quadratic, is_quadratic = load_instance(instance_path)

    # Fallback heuristic guarantees (usually) a feasible incumbent quickly.
    heuristic_flows = greedy_heuristic(n, m, supplies, demands, cap, linear, quadratic, is_quadratic)
    best_solution = None
    best_obj = float("inf")
    if heuristic_flows is not None:
        best_obj = compute_objective(heuristic_flows, n, m, linear, quadratic, is_quadratic) - 1e-9
        best_solution = build_solution_dict(heuristic_flows, n, m, linear, quadratic, is_quadratic)
        if logger:
            logger.log_solution(best_solution["objective_value"], best_solution)

    elapsed = time.time() - start
    remaining = time_limit - elapsed
    flows = None

    # Leave a small reserve for extraction/output.
    if remaining > 0.5:
        model = gp.Model("bipartite_transportation")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam("LogToConsole", 0)
        model.setParam("TimeLimit", max(remaining - 0.25, 0.01))

        x = model.addVars(n, m, lb=0.0, ub=cap, name="x")

        # Customer demand: exactly met by incoming flows.
        for j in range(m):
            model.addConstr(
                gp.quicksum(x[i, j] for i in range(n)) == demands[j],
                name="demand_{}".format(j),
            )
        # Supplier supply: cannot exceed capacity.
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(m)) <= supplies[i],
                name="supply_{}".format(i),
            )

        obj = gp.QuadExpr()
        for i in range(n):
            li = linear[i]
            qi = quadratic[i]
            for j in range(m):
                c = li[j]
                v = x[i, j]
                if c != 0.0:
                    obj.add(c * v)
                if is_quadratic:
                    q = qi[j]
                    if q != 0.0:
                        obj.add(q * v * v)
        model.setObjective(obj, GRB.MINIMIZE)

        # Warm-start hint (helps simplex-based QP/LP paths; ignored harmlessly otherwise).
        if heuristic_flows is not None:
            for i in range(n):
                hi = heuristic_flows[i]
                for j in range(m):
                    if hi[j] > 0.0:
                        x[i, j].Start = hi[j]

        model.optimize()

        if model.SolCount > 0:
            extracted = []
            for i in range(n):
                row = []
                for j in range(m):
                    v = x[i, j].X
                    if v < 0.0:
                        v = 0.0 if v > -1e-9 else v
                    elif v > cap and v < cap + 1e-9:
                        v = cap
                    row.append(v)
                extracted.append(row)
            flows = extracted

    if flows is None:
        flows = heuristic_flows
    if flows is None:
        # Last-resort placeholder to keep the output schema valid.
        flows = [[0.0] * m for _ in range(n)]

    solution = build_solution_dict(flows, n, m, linear, quadratic, is_quadratic)

    if logger and solution["objective_value"] < best_obj - 1e-9:
        logger.log_solution(solution["objective_value"], solution)

    with open(solution_path, "w") as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description="Solve bipartite transportation problem.")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solve(args.instance_path, args.solution_path, args.time_limit, logger)


if __name__ == "__main__":
    main()