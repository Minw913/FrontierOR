import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    nodes = data["nodes"]
    arcs = data["arcs"]
    commodities = data["commodities"]
    fleet_types = data["fleet_types"]

    model = gp.Model("network_design")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Decision variables
    # x[k_idx, a_idx]: flow of commodity k on arc a
    # y[t_idx, a_idx]: number of vehicles of type t on arc a
    x = {}
    for ki, k in enumerate(commodities):
        for ai, a in enumerate(arcs):
            x[ki, ai] = model.addVar(
                lb=0.0,
                obj=float(a["flow_cost"]),
                vtype=GRB.CONTINUOUS,
                name=f"x_{ki}_{ai}",
            )
    y = {}
    for ti, t in enumerate(fleet_types):
        for ai, a in enumerate(arcs):
            y[ti, ai] = model.addVar(
                lb=0.0,
                obj=float(t["fixed_cost"]),
                vtype=GRB.INTEGER,
                name=f"y_{ti}_{ai}",
            )

    model.ModelSense = GRB.MINIMIZE

    # Precompute incidence
    out_arcs = {n: [] for n in nodes}
    in_arcs = {n: [] for n in nodes}
    for ai, a in enumerate(arcs):
        out_arcs[a["from"]].append(ai)
        in_arcs[a["to"]].append(ai)

    # Flow balance constraints
    for ki, k in enumerate(commodities):
        d = float(k["demand"])
        for n in nodes:
            if n == k["origin"]:
                rhs = d
            elif n == k["destination"]:
                rhs = -d
            else:
                rhs = 0.0
            model.addConstr(
                gp.quicksum(x[ki, ai] for ai in out_arcs[n])
                - gp.quicksum(x[ki, ai] for ai in in_arcs[n])
                == rhs
            )

    # Capacity constraints
    for ai, a in enumerate(arcs):
        model.addConstr(
            gp.quicksum(x[ki, ai] for ki in range(len(commodities)))
            <= gp.quicksum(
                float(t["capacity"]) * y[ti, ai] for ti, t in enumerate(fleet_types)
            )
        )

    # Vehicle balance constraints (design balance) per fleet type per node
    for ti in range(len(fleet_types)):
        for n in nodes:
            model.addConstr(
                gp.quicksum(y[ti, ai] for ai in in_arcs[n])
                == gp.quicksum(y[ti, ai] for ai in out_arcs[n])
            )

    # Time limit
    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    def build_solution(get_x, get_y, obj):
        flows = {}
        for ki, k in enumerate(commodities):
            kid = k["id"]
            for ai, a in enumerate(arcs):
                v = get_x(ki, ai)
                if v < 0:
                    v = 0.0
                flows[f"x_{kid}_{a['from']}_{a['to']}"] = v
        vehicles = {}
        for ti, t in enumerate(fleet_types):
            tid = t["id"]
            for ai, a in enumerate(arcs):
                v = get_y(ti, ai)
                vehicles[f"y_{tid}_{a['from']}_{a['to']}"] = int(round(v))
        return {
            "objective_value": float(obj),
            "flows": flows,
            "vehicles": vehicles,
        }

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    xvals = m.cbGetSolution([x[ki, ai]
                                             for ki in range(len(commodities))
                                             for ai in range(len(arcs))])
                    yvals = m.cbGetSolution([y[ti, ai]
                                             for ti in range(len(fleet_types))
                                             for ai in range(len(arcs))])
                    na = len(arcs)
                    sol = build_solution(
                        lambda ki, ai: float(xvals[ki * na + ai]),
                        lambda ti, ai: float(yvals[ti * na + ai]),
                        obj,
                    )
                    logger.log_solution(obj, sol)
                except Exception:
                    try:
                        logger.log(obj)
                    except Exception:
                        pass

    model.optimize(callback)

    solution = None
    if model.SolCount > 0:
        solution = build_solution(
            lambda ki, ai: float(x[ki, ai].X),
            lambda ti, ai: float(y[ti, ai].X),
            model.ObjVal,
        )
        if logger:
            try:
                logger.log_solution(model.ObjVal, solution)
            except Exception:
                pass
    else:
        # No feasible solution found within the limit: emit zeros as placeholder
        flows = {}
        for k in commodities:
            for a in arcs:
                flows[f"x_{k['id']}_{a['from']}_{a['to']}"] = 0.0
        vehicles = {}
        for t in fleet_types:
            for a in arcs:
                vehicles[f"y_{t['id']}_{a['from']}_{a['to']}"] = 0
        solution = {
            "objective_value": float("inf"),
            "flows": flows,
            "vehicles": vehicles,
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()