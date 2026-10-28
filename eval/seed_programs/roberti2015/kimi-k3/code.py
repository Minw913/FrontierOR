import argparse
import json
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

try:
    from solution_logger import SolutionLogger
except Exception:  # pragma: no cover
    SolutionLogger = None

FLOW_TOL = 1e-7


def build_solution(flow, variable_cost, fixed_cost, capacity):
    m = len(flow)
    n = len(flow[0]) if m else 0
    out_flow = []
    arcs = []
    obj = 0.0
    for i in range(m):
        frow = []
        arow = []
        for j in range(n):
            v = flow[i][j]
            if v <= FLOW_TOL:
                v = 0.0
                use = 0
            else:
                if v > capacity[i][j]:
                    v = capacity[i][j]
                use = 1
            obj += variable_cost[i][j] * v
            if use:
                obj += fixed_cost[i][j]
            frow.append(float(v))
            arow.append(int(use))
        out_flow.append(frow)
        arcs.append(arow)
    return {"objective_value": float(obj), "flow": out_flow, "arcs_used": arcs}, float(obj)


def greedy_flow(supply, demand, variable_cost, fixed_cost, capacity):
    """Feasible transportation plan: fill routes in order of approximate
    effective unit cost (variable cost + fixed cost amortized over capacity)."""
    m = len(supply)
    n = len(demand)
    rem_s = list(supply)
    rem_d = list(demand)
    flow = [[0.0] * n for _ in range(m)]
    order = []
    for i in range(m):
        for j in range(n):
            c = capacity[i][j]
            eff = variable_cost[i][j] + (fixed_cost[i][j] / c if c > 0 else float("inf"))
            order.append((eff, i, j))
    order.sort()
    for _, i, j in order:
        q = min(rem_s[i], rem_d[j], capacity[i][j])
        if q > 0.0:
            flow[i][j] += q
            rem_s[i] -= q
            rem_d[j] -= q
    # Safety pass (the instance structure guarantees this is not needed).
    for i in range(m):
        if rem_s[i] <= FLOW_TOL:
            continue
        for j in range(n):
            if rem_d[j] <= FLOW_TOL:
                continue
            q = min(rem_s[i], rem_d[j], max(capacity[i][j] - flow[i][j], 0.0))
            if q > 0.0:
                flow[i][j] += q
                rem_s[i] -= q
                rem_d[j] -= q
            if rem_s[i] <= FLOW_TOL:
                break
    return flow


def main():
    parser = argparse.ArgumentParser(description="Fixed-charge transportation problem solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.monotonic()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    m = int(inst["num_sources"])
    n = int(inst["num_sinks"])
    supply = [float(v) for v in inst["supply"]]
    demand = [float(v) for v in inst["demand"]]
    variable_cost = [[float(v) for v in row] for row in inst["variable_cost"]]
    fixed_cost = [[float(v) for v in row] for row in inst["fixed_cost"]]
    raw_cap = inst.get("capacity")
    capacity = []
    for i in range(m):
        row = []
        for j in range(n):
            c = float(raw_cap[i][j]) if raw_cap is not None else min(supply[i], demand[j])
            row.append(max(0.0, min(c, supply[i], demand[j])))
        capacity.append(row)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    # Initial feasible solution (also serves as fallback if the MIP is
    # interrupted before finding any incumbent).
    gflow = greedy_flow(supply, demand, variable_cost, fixed_cost, capacity)
    best_sol, best_obj = build_solution(gflow, variable_cost, fixed_cost, capacity)
    if logger:
        logger.log_solution(best_obj, best_sol)

    if gp is not None:
        remaining = float(args.time_limit) - (time.monotonic() - t0) - 0.2
        if remaining > 0.05:
            try:
                model = gp.Model("fixed_charge_transportation")
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1
                model.Params.TimeLimit = max(remaining, 0.01)

                x = [[model.addVar(lb=0.0, ub=capacity[i][j], name=f"x_{i}_{j}")
                      for j in range(n)] for i in range(m)]
                y = [[model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
                      for j in range(n)] for i in range(m)]

                for i in range(m):
                    model.addConstr(gp.quicksum(x[i][j] for j in range(n)) == supply[i],
                                    name=f"supply_{i}")
                for j in range(n):
                    model.addConstr(gp.quicksum(x[i][j] for i in range(m)) == demand[j],
                                    name=f"demand_{j}")
                for i in range(m):
                    for j in range(n):
                        model.addConstr(x[i][j] <= capacity[i][j] * y[i][j],
                                        name=f"link_{i}_{j}")

                model.setObjective(
                    gp.quicksum(variable_cost[i][j] * x[i][j] + fixed_cost[i][j] * y[i][j]
                                for i in range(m) for j in range(n)),
                    GRB.MINIMIZE,
                )

                # MIP start from the greedy feasible solution.
                for i in range(m):
                    for j in range(n):
                        x[i][j].Start = gflow[i][j]
                        y[i][j].Start = 1.0 if gflow[i][j] > FLOW_TOL else 0.0

                flat_x = [x[i][j] for i in range(m) for j in range(n)]

                def _callback(md, where):
                    nonlocal best_sol, best_obj
                    if where == GRB.Callback.MIPSOL:
                        try:
                            vals = md.cbGetSolution(flat_x)
                            fl = [[0.0] * n for _ in range(m)]
                            k = 0
                            for i in range(m):
                                for j in range(n):
                                    fl[i][j] = vals[k]
                                    k += 1
                            sol, obj = build_solution(fl, variable_cost, fixed_cost, capacity)
                            if obj < best_obj - 1e-9:
                                best_sol, best_obj = sol, obj
                                if logger:
                                    logger.log_solution(obj, sol)
                        except Exception:
                            pass

                model.optimize(_callback)

                if model.SolCount > 0:
                    vals = [v.X for v in flat_x]
                    fl = [[0.0] * n for _ in range(m)]
                    k = 0
                    for i in range(m):
                        for j in range(n):
                            fl[i][j] = vals[k]
                            k += 1
                    sol, obj = build_solution(fl, variable_cost, fixed_cost, capacity)
                    if obj < best_obj - 1e-9:
                        best_sol, best_obj = sol, obj
                        if logger:
                            logger.log_solution(obj, sol)
                model.dispose()
            except Exception:
                # Keep the best solution found so far (at minimum the greedy one).
                pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()