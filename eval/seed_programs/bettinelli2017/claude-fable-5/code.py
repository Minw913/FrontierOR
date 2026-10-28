import argparse
import json
import time

from solution_logger import SolutionLogger


def greedy_solution(n, profits, weights, capacity, adj):
    # Sort by profit/weight ratio
    order = sorted(range(n), key=lambda i: profits[i] / max(weights[i], 1e-9), reverse=True)
    selected = []
    sel_set = set()
    total_w = 0
    for i in order:
        if total_w + weights[i] > capacity:
            continue
        if any(j in sel_set for j in adj[i]):
            continue
        selected.append(i)
        sel_set.add(i)
        total_w += weights[i]
    obj = sum(profits[i] for i in selected)
    return selected, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["num_items"]
    capacity = data["capacity"]
    profits = data["items"]["profits"]
    weights = data["items"]["weights"]
    edges = data["conflict_graph"]["edges"]

    adj = [set() for _ in range(n)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    # Greedy initial solution
    best_sel, best_obj = greedy_solution(n, profits, weights, capacity, adj)
    if logger:
        logger.log_solution(float(best_obj), {
            "objective_value": float(best_obj),
            "selected_items": sorted(best_sel),
        })

    # Build MIP with Gurobi
    try:
        import gurobipy as gp
        from gurobipy import GRB

        remaining = max(1.0, args.time_limit - (time.time() - start_time) - 2.0)

        model = gp.Model("kpc")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

        seen = set()
        for e in edges:
            u, v = int(e[0]), int(e[1])
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            model.addConstr(x[u] + x[v] <= 1)

        model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

        # Warm start
        for i in range(n):
            x[i].Start = 0
        for i in best_sel:
            x[i].Start = 1

        state = {"best_obj": best_obj, "best_sel": best_sel}

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > state["best_obj"] + 1e-6:
                    vals = m.cbGetSolution([x[i] for i in range(n)])
                    sel = [i for i in range(n) if vals[i] > 0.5]
                    state["best_obj"] = obj
                    state["best_sel"] = sel
                    if logger:
                        logger.log_solution(float(obj), {
                            "objective_value": float(obj),
                            "selected_items": sorted(sel),
                        })

        model.optimize(callback)

        if model.SolCount > 0:
            obj = model.ObjVal
            if obj > state["best_obj"] - 1e-6:
                sel = [i for i in range(n) if x[i].X > 0.5]
                if obj > state["best_obj"] + 1e-6 and logger:
                    logger.log_solution(float(obj), {
                        "objective_value": float(obj),
                        "selected_items": sorted(sel),
                    })
                state["best_obj"] = obj
                state["best_sel"] = sel

        best_sel = state["best_sel"]
        best_obj = state["best_obj"]

    except Exception:
        # Fall back to greedy solution if solver fails
        pass

    solution = {
        "objective_value": float(sum(profits[i] for i in best_sel)),
        "selected_items": sorted(best_sel),
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()