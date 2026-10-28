import argparse
import json
import math
import sys
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_total = instance["num_activities_total"]
    M = instance["max_project_duration"]

    P = {i: 0 for i in range(n_total)}
    if "activities" in instance:
        for act in instance["activities"]:
            P[act["id"]] = act["processing_time"]

    explicit_edges = []
    if "temporal_constraints" in instance:
        for edge in instance["temporal_constraints"]:
            explicit_edges.append((edge["from_activity"], edge["to_activity"], edge["time_lag"]))

    edges = explicit_edges[:]
    # Implicit bounds guaranteeing S_i >= 0 and S_i <= M
    for i in range(n_total):
        edges.append((0, i, 0))
        edges.append((i, 0, -M))

    # Compute Earliest Start times (Longest paths from 0)
    dist_from_0 = [-math.inf] * n_total
    dist_from_0[0] = 0

    for step in range(n_total):
        changed = False
        for u, v, w in edges:
            if dist_from_0[u] != -math.inf and dist_from_0[u] + w > dist_from_0[v]:
                dist_from_0[v] = dist_from_0[u] + w
                changed = True
        if not changed:
            break

    if changed:
        raise ValueError("Infeasible temporal constraints (positive cycle from 0).")

    # Compute Latest Start times (Longest paths to 0)
    dist_to_0 = [-math.inf] * n_total
    dist_to_0[0] = 0
    reversed_edges = [(v, u, w) for u, v, w in edges]

    for step in range(n_total):
        changed = False
        for u, v, w in reversed_edges:
            if dist_to_0[u] != -math.inf and dist_to_0[u] + w > dist_to_0[v]:
                dist_to_0[v] = dist_to_0[u] + w
                changed = True
        if not changed:
            break

    if changed:
        raise ValueError("Infeasible temporal constraints (positive cycle to 0).")

    ES = dist_from_0[:]
    LS = [-x for x in dist_to_0]

    # Initialize Gurobi model
    model = gp.Model("ProjectScheduling")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    # Decision variables x_it = 1 if activity i starts at time t
    x = {}
    for i in range(n_total):
        for t in range(ES[i], LS[i] + 1):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # Start time exactly once constraints
    for i in range(n_total):
        model.addConstr(gp.quicksum(x[i, t] for t in range(ES[i], LS[i] + 1)) == 1, name=f"one_start_{i}")

    # Derived Start variable equations
    S = {}
    for i in range(n_total):
        S[i] = gp.quicksum(t * x[i, t] for t in range(ES[i], LS[i] + 1))

    # Basic continuous relations
    for u, v, w in explicit_edges:
        model.addConstr(S[v] - S[u] >= w)

    # Strong convex-hull CDF relations for explicit bounds
    CDF = {}
    for i in range(n_total):
        expr = gp.LinExpr()
        for t in range(ES[i], LS[i] + 1):
            expr.add(x[i, t], 1.0)
            CDF[i, t] = gp.LinExpr(expr)

    for u, v, w in explicit_edges:
        for t in range(ES[v], LS[v] + 1):
            t_u = t - w
            if t_u < LS[u]:
                if t_u < ES[u]:
                    model.addConstr(0 >= CDF[v, t])
                else:
                    model.addConstr(CDF[u, t_u] >= CDF[v, t])

    # Partially renewable resource constraints
    if "resources" in instance:
        for res in instance["resources"]:
            C_k = res["capacity"]
            Pi_k_set = set(res["Pi_k"])
            demands = res["demands"]

            expr_terms = []
            for u_str, demand_u in demands.items():
                u = int(u_str)
                if demand_u == 0 or P.get(u, 0) == 0:
                    continue

                P_u = P[u]
                for t in range(ES[u], LS[u] + 1):
                    # Find occupied resource subsets mapping onto exact period indexation logic
                    overlap = 0
                    for period in range(t + 1, t + P_u + 1):
                        if period in Pi_k_set:
                            overlap += 1

                    if overlap > 0:
                        expr_terms.append(demand_u * overlap * x[u, t])

            if expr_terms:
                model.addConstr(gp.quicksum(expr_terms) <= C_k)

    # Setting Objective (minimize the start time of the project-end dummy activity)
    model.setObjective(S[n_total - 1], GRB.MINIMIZE)

    # Integrating convergence logging
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    def optimization_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Execution
    model.optimize(optimization_callback)

    # Solution Extraction and Schema alignment
    if model.SolCount > 0:
        schedule = {}
        for i in range(n_total):
            start_time = int(sum(t * round(x[i, t].X) for t in range(ES[i], LS[i] + 1)))
            schedule[str(i)] = start_time

        obj_val = float(model.ObjVal)

        out = {
            "objective_value": obj_val,
            "schedule": schedule
        }

        with open(args.solution_path, 'w') as f:
            json.dump(out, f, indent=2)
    else:
        raise RuntimeError("No feasible solution found up to the defined limits.")

if __name__ == "__main__":
    solve()