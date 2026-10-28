import argparse
import json
import os
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


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_nodes = int(data["num_nodes"])
    num_commodities = int(data["num_commodities"])
    num_scenarios = int(data["num_scenarios"])
    num_edges = int(data["num_edges"])
    gamma = float(data["gamma"])
    edge_limit = int(data["c_0"])

    edges = [(int(a), int(b)) for a, b in data["edges"]]
    construction_costs = [float(v) for v in data["construction_costs"]]
    flow_costs = [float(v) for v in data["flow_costs"]]
    capacities = [float(v) for v in data["capacities"]]
    demands = data["demands"]

    outgoing = [[] for _ in range(num_nodes)]
    incoming = [[] for _ in range(num_nodes)]
    for e, (tail, head) in enumerate(edges):
        outgoing[tail].append(e)
        incoming[head].append(e)

    model = gp.Model("stochastic_capacitated_network_design")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    z = model.addVars(num_edges, vtype=GRB.BINARY, name="z")
    tf = model.addVars(
        num_scenarios, num_edges, lb=0.0, vtype=GRB.CONTINUOUS, name="tf"
    )
    x = model.addVars(
        num_commodities,
        num_scenarios,
        num_edges,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    model.addConstr(
        gp.quicksum(z[e] for e in range(num_edges)) <= edge_limit,
        name="edge_limit",
    )

    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(
                tf[r, e]
                == gp.quicksum(x[k, r, e] for k in range(num_commodities)),
                name=f"total_flow[{r},{e}]",
            )
            model.addConstr(
                tf[r, e] <= capacities[e] * z[e],
                name=f"capacity[{r},{e}]",
            )

    for k in range(num_commodities):
        for r in range(num_scenarios):
            for node in range(num_nodes):
                balance = gp.LinExpr()
                for e in outgoing[node]:
                    balance.add(x[k, r, e], 1.0)
                for e in incoming[node]:
                    balance.add(x[k, r, e], -1.0)
                model.addConstr(
                    balance == float(demands[r][k][node]),
                    name=f"balance[{k},{r},{node}]",
                )

    objective = gp.QuadExpr()
    for e in range(num_edges):
        objective.add(construction_costs[e] * z[e])

    scenario_weight = 1.0 / num_scenarios
    quadratic_coefficient = scenario_weight / (2.0 * gamma)
    for r in range(num_scenarios):
        for e in range(num_edges):
            objective.add(scenario_weight * flow_costs[e] * tf[r, e])
            objective.add(quadratic_coefficient * tf[r, e] * tf[r, e])

    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    z_vars = [z[e] for e in range(num_edges)]
    tf_vars = [
        tf[r, e]
        for r in range(num_scenarios)
        for e in range(num_edges)
    ]
    x_vars = [
        x[k, r, e]
        for k in range(num_commodities)
        for r in range(num_scenarios)
        for e in range(num_edges)
    ]

    def compute_objective(z_values, tf_values):
        value = 0.0
        for e in range(num_edges):
            value += construction_costs[e] * int(z_values[e] >= 0.5)

        index = 0
        for r in range(num_scenarios):
            for e in range(num_edges):
                flow = float(tf_values[index])
                value += scenario_weight * flow_costs[e] * flow
                value += quadratic_coefficient * flow * flow
                index += 1
        return float(value)

    def make_solution(z_values, tf_values, x_values, objective_value=None):
        model_variables = {}
        z_output = {}

        for e, value in enumerate(z_values):
            active = 1 if value >= 0.5 else 0
            z_output[str(e)] = active
            if active:
                model_variables[f"z[{e}]"] = 1

        index = 0
        for r in range(num_scenarios):
            for e in range(num_edges):
                value = float(tf_values[index])
                index += 1
                if abs(value) > 1e-10:
                    model_variables[f"tf[{r},{e}]"] = value

        index = 0
        for k in range(num_commodities):
            for r in range(num_scenarios):
                for e in range(num_edges):
                    value = float(x_values[index])
                    index += 1
                    if abs(value) > 1e-10:
                        model_variables[f"x[{k},{r},{e}]"] = value

        if objective_value is None:
            objective_value = compute_objective(z_values, tf_values)

        return {
            "objective_value": float(objective_value),
            "model_variables": model_variables,
            "z": z_output,
        }

    callback_state = {
        "best_solver_objective": float("inf"),
        "best_logged_objective": float("inf"),
        "best_solution": None,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            solver_objective = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            if solver_objective >= callback_state["best_solver_objective"] - 1e-9:
                return

            callback_state["best_solver_objective"] = solver_objective

            z_values = cb_model.cbGetSolution(z_vars)
            tf_values = cb_model.cbGetSolution(tf_vars)
            x_values = cb_model.cbGetSolution(x_vars)

            represented_objective = compute_objective(z_values, tf_values)
            solution = make_solution(
                z_values, tf_values, x_values, represented_objective
            )
            callback_state["best_solution"] = solution

            if logger:
                logger.log_solution(represented_objective, solution)
                callback_state["best_logged_objective"] = represented_objective
        except Exception:
            # Logging or snapshot creation must not interrupt optimization.
            pass

    elapsed = time.monotonic() - start_time
    remaining_time = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    model.optimize(incumbent_callback)

    final_solution = None

    if model.SolCount > 0:
        z_values = [var.X for var in z_vars]
        tf_values = [var.X for var in tf_vars]
        x_values = [var.X for var in x_vars]
        represented_objective = compute_objective(z_values, tf_values)
        final_solution = make_solution(
            z_values, tf_values, x_values, represented_objective
        )

        if (
            logger
            and represented_objective
            < callback_state["best_logged_objective"] - 1e-9
        ):
            logger.log_solution(represented_objective, final_solution)
    elif callback_state["best_solution"] is not None:
        final_solution = callback_state["best_solution"]
    else:
        # This branch is only expected for an infeasible instance or a time
        # limit reached before any feasible incumbent was found.
        final_solution = {
            "objective_value": 0.0,
            "model_variables": {},
            "z": {str(e): 0 for e in range(num_edges)},
        }

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    temporary_path = args.solution_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, args.solution_path)


if __name__ == "__main__":
    main()