import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def build_solution_dict(objective_value, var_names, var_values, z_values):
    model_variables = {}
    tol = 1e-9

    for n, v in zip(var_names, var_values):
        if abs(v) > tol:
            if n.startswith("z["):
                model_variables[n] = int(round(v))
            else:
                model_variables[n] = float(v)

    z_dict = {str(e): int(round(z_values[e])) for e in range(len(z_values))}

    return {
        "objective_value": float(objective_value),
        "model_variables": model_variables,
        "z": z_dict,
        "model_variable_name_patterns": {
            "z[e]": "binary design variable for edge e",
            "x[k,r,e]": "commodity k, scenario r, edge e flow",
            "tf[r,e]": "total scenario flow on edge e",
        },
    }


def make_callback(logger):
    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < model._best_logged_obj:
                all_vals = model.cbGetSolution(model._all_vars)
                z_vals = model.cbGetSolution(model._z_vars)
                sol = build_solution_dict(obj, model._all_var_names, all_vals, z_vals)
                logger.log_solution(obj, sol)
                model._best_logged_obj = obj
    return callback


def solve_instance(instance, time_limit, logger=None):
    N = instance["num_nodes"]
    K = instance["num_commodities"]
    R = instance["num_scenarios"]
    E = instance["num_edges"]

    gamma = float(instance["gamma"])
    c0 = int(instance["c_0"])

    edges = instance["edges"]
    construction_costs = instance["construction_costs"]
    flow_costs = instance["flow_costs"]
    capacities = instance["capacities"]
    demands = instance["demands"]

    # Build adjacency for flow conservation
    out_edges = [[] for _ in range(N)]
    in_edges = [[] for _ in range(N)]
    for e, (u, v) in enumerate(edges):
        out_edges[u].append(e)
        in_edges[v].append(e)

    m = Model("stochastic_multicommodity_network_design")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = int(time_limit)

    # Variables
    z = m.addVars(E, vtype=GRB.BINARY, name="z")
    x = m.addVars(K, R, E, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    tf = m.addVars(R, E, lb=0.0, vtype=GRB.CONTINUOUS, name="tf")

    # Total flow definition and capacity/linking
    for r in range(R):
        for e in range(E):
            m.addConstr(tf[r, e] == quicksum(x[k, r, e] for k in range(K)), name=f"tf_def[{r},{e}]")
            m.addConstr(tf[r, e] <= capacities[e] * z[e], name=f"cap[{r},{e}]")

    # Edge budget
    m.addConstr(quicksum(z[e] for e in range(E)) <= c0, name="edge_budget")

    # Flow conservation
    for r in range(R):
        for k in range(K):
            for n in range(N):
                outflow = quicksum(x[k, r, e] for e in out_edges[n]) if out_edges[n] else 0.0
                inflow = quicksum(x[k, r, e] for e in in_edges[n]) if in_edges[n] else 0.0
                m.addConstr(outflow - inflow == demands[r][k][n], name=f"flow[{k},{r},{n}]")

    # Objective
    invR = 1.0 / float(R)
    quad_coeff = 1.0 / (2.0 * gamma * float(R))

    construction_term = quicksum(construction_costs[e] * z[e] for e in range(E))
    transport_term = invR * quicksum(flow_costs[e] * tf[r, e] for r in range(R) for e in range(E))
    quadratic_term = quad_coeff * quicksum(tf[r, e] * tf[r, e] for r in range(R) for e in range(E))

    m.setObjective(construction_term + transport_term + quadratic_term, GRB.MINIMIZE)

    m.update()
    all_vars = m.getVars()
    all_var_names = [v.VarName for v in all_vars]
    z_vars = [z[e] for e in range(E)]

    m._all_vars = all_vars
    m._all_var_names = all_var_names
    m._z_vars = z_vars
    m._best_logged_obj = float("inf")

    cb = make_callback(logger)
    m.optimize(cb)

    # Build final solution
    if m.SolCount > 0:
        all_vals = m.getAttr("X", all_vars)
        z_vals = m.getAttr("X", z_vars)
        obj = m.ObjVal
        solution = build_solution_dict(obj, all_var_names, all_vals, z_vals)

        if logger is not None and obj + 1e-9 < m._best_logged_obj:
            logger.log_solution(obj, solution)
            m._best_logged_obj = obj
    else:
        # No feasible solution found
        solution = {
            "objective_value": 1e100,
            "model_variables": {},
            "z": {str(e): 0 for e in range(E)},
            "model_variable_name_patterns": {
                "z[e]": "binary design variable for edge e",
                "x[k,r,e]": "commodity k, scenario r, edge e flow",
                "tf[r,e]": "total scenario flow on edge e",
            },
        }

    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve stochastic capacitated multicommodity network design problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()