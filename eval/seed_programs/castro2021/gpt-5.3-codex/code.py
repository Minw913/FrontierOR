import argparse
import json
import math
from typing import Any, List

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_arc_capacity_getter(arc_capacity_field: Any, n: int, m: int):
    """
    Returns a function cap(i,j) robust to:
    - scalar capacity
    - matrix/list capacities
    """
    if isinstance(arc_capacity_field, (int, float)):
        cap_val = float(arc_capacity_field)

        def cap(i, j):
            return cap_val

        return cap

    # If a matrix/list is provided, try to index it.
    if isinstance(arc_capacity_field, list):
        # matrix n x m
        if len(arc_capacity_field) == n and isinstance(arc_capacity_field[0], list):
            def cap(i, j):
                return float(arc_capacity_field[i][j])
            return cap
        # flattened n*m
        if len(arc_capacity_field) == n * m:
            def cap(i, j):
                return float(arc_capacity_field[i * m + j])
            return cap

    # Fallback: very large capacity
    def cap(i, j):
        return GRB.INFINITY

    return cap


def compute_objective(flows, linear_costs, quadratic_costs, cost_type, n, m):
    obj = 0.0
    for i in range(n):
        for j in range(m):
            x = flows[i][j]
            c = float(linear_costs[i][j]) if linear_costs is not None else 0.0
            q = float(quadratic_costs[i][j]) if quadratic_costs is not None else 0.0
            if cost_type == "quadratic":
                obj += c * x + q * x * x
            else:
                obj += c * x
    return float(obj)


def greedy_fallback(instance: dict):
    """
    Emergency feasible constructor (best-effort).
    """
    n = int(instance["n"])
    m = int(instance["m"])
    supplies = [float(v) for v in instance["supplies"]]
    demands = [float(v) for v in instance["demands"]]
    linear_costs = instance.get("linear_costs", [[0.0] * m for _ in range(n)])
    quadratic_costs = instance.get("quadratic_costs", [[0.0] * m for _ in range(n)])
    cost_type = instance.get("cost_type", "linear_fractional")
    cap = get_arc_capacity_getter(instance.get("arc_capacity", GRB.INFINITY), n, m)

    flows = [[0.0 for _ in range(m)] for _ in range(n)]
    rem_supply = supplies[:]
    rem_demand = demands[:]

    # Process customers in descending demand, suppliers in ascending linear cost for each customer.
    customer_order = sorted(range(m), key=lambda j: rem_demand[j], reverse=True)
    for j in customer_order:
        supplier_order = sorted(range(n), key=lambda i: float(linear_costs[i][j]))
        for i in supplier_order:
            if rem_demand[j] <= 1e-12:
                break
            avail = min(rem_supply[i], cap(i, j) - flows[i][j])
            if avail <= 1e-12:
                continue
            take = min(avail, rem_demand[j])
            flows[i][j] += take
            rem_supply[i] -= take
            rem_demand[j] -= take

    feasible = all(d <= 1e-7 for d in rem_demand)
    obj = compute_objective(flows, linear_costs, quadratic_costs, cost_type, n, m)

    return feasible, obj, flows


def solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    n = int(instance["n"])
    m = int(instance["m"])
    supplies = [float(v) for v in instance["supplies"]]
    demands = [float(v) for v in instance["demands"]]
    linear_costs = instance.get("linear_costs", [[0.0] * m for _ in range(n)])
    quadratic_costs = instance.get("quadratic_costs", [[0.0] * m for _ in range(n)])
    cost_type = instance.get("cost_type", "linear_fractional")
    cap = get_arc_capacity_getter(instance.get("arc_capacity", GRB.INFINITY), n, m)

    model = gp.Model("bipartite_transport")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))

    # Decision variables x[i,j] >= 0 with upper bound capacity(i,j)
    x = {}
    for i in range(n):
        for j in range(m):
            ub = cap(i, j)
            x[i, j] = model.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
    model.update()

    # Demand satisfaction: sum_i x[i,j] == demand[j]
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == demands[j], name=f"demand_{j}")

    # Supply capacities: sum_j x[i,j] <= supply[i]
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) <= supplies[i], name=f"supply_{i}")

    # Objective
    if cost_type == "quadratic":
        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(m):
                c = float(linear_costs[i][j])
                q = float(quadratic_costs[i][j]) if quadratic_costs is not None else 0.0
                if c != 0.0:
                    obj.add(c * x[i, j])
                if q != 0.0:
                    obj.add(q * x[i, j] * x[i, j])
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        obj = gp.LinExpr()
        for i in range(n):
            for j in range(m):
                c = float(linear_costs[i][j])
                if c != 0.0:
                    obj.add(c * x[i, j])
        model.setObjective(obj, GRB.MINIMIZE)

    best_logged = [math.inf]

    def cb(mdl, where):
        if logger is None:
            return
        try:
            cur = None
            if where == GRB.Callback.SIMPLEX:
                cur = mdl.cbGet(GRB.Callback.SPX_OBJVAL)
            elif where == GRB.Callback.BARRIER:
                cur = mdl.cbGet(GRB.Callback.BARRIER_PRIMOBJ)

            if cur is not None and cur < best_logged[0] - 1e-9 and cur < GRB.INFINITY:
                best_logged[0] = float(cur)
                logger.log(float(cur))
        except gp.GurobiError:
            pass

    model.optimize(cb)

    if model.SolCount > 0:
        obj_val = float(model.ObjVal)
        if logger and obj_val < best_logged[0] - 1e-9:
            logger.log(obj_val)

        flows = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                val = x[i, j].X
                # avoid tiny negatives due numerical tolerance
                if val < 0 and val > -1e-9:
                    val = 0.0
                flows[i][j] = float(val)
        return obj_val, flows

    # If no solution found by solver, try a quick fallback heuristic.
    feasible, obj_val, flows = greedy_fallback(instance)
    if logger and feasible:
        logger.log(float(obj_val))
    return float(obj_val), flows


def build_output(instance: dict, objective_value: float, flows_matrix: List[List[float]]) -> dict:
    n = int(instance["n"])
    m = int(instance["m"])
    flows_dict = {}
    for i in range(n):
        for j in range(m):
            flows_dict[f"x_{i}_{j}"] = float(flows_matrix[i][j])
    return {
        "objective_value": float(objective_value),
        "flows": flows_dict
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    objective_value, flows = solve(instance, args.time_limit, logger)
    sol = build_output(instance, objective_value, flows)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()