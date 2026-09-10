import argparse
import json
import time

from solution_logger import SolutionLogger


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

    n = data["n"]
    capacity = data["capacity"]
    items = data["items"]

    ids = [it["id"] for it in items]
    profits = [it["profit"] for it in items]
    weights = [it["weight"] for it in items]

    # ---------- Greedy initial solution ----------
    order = sorted(range(n), key=lambda i: profits[i] / weights[i], reverse=True)
    greedy_sel = []
    rem = capacity
    greedy_obj = 0
    for i in order:
        if weights[i] <= rem:
            greedy_sel.append(i)
            rem -= weights[i]
            greedy_obj += profits[i]

    best_obj = greedy_obj
    best_sel = list(greedy_sel)

    def make_solution(obj, sel_idx):
        return {
            "objective_value": int(obj),
            "selected_items": [ids[i] for i in sel_idx],
        }

    if logger:
        logger.log_solution(best_obj, make_solution(best_obj, best_sel))

    def write_solution():
        with open(args.solution_path, "w") as f:
            json.dump(make_solution(best_obj, best_sel), f)

    write_solution()

    # ---------- Exact solve ----------
    # Try DP if state space is small enough; otherwise use Gurobi.
    dp_states = n * (capacity + 1)
    solved = False
    if dp_states <= 20_000_000:
        try:
            dp = [0] * (capacity + 1)
            choice = [bytearray(capacity + 1) for _ in range(n)]
            for i in range(n):
                w = weights[i]
                p = profits[i]
                ci = choice[i]
                for c in range(capacity, w - 1, -1):
                    cand = dp[c - w] + p
                    if cand > dp[c]:
                        dp[c] = cand
                        ci[c] = 1
                if time.time() - start_time > args.time_limit * 0.8:
                    break
            else:
                # completed DP
                obj = dp[capacity]
                sel = []
                c = capacity
                for i in range(n - 1, -1, -1):
                    if choice[i][c]:
                        sel.append(i)
                        c -= weights[i]
                if obj > best_obj:
                    best_obj = obj
                    best_sel = sel
                    if logger:
                        logger.log_solution(best_obj, make_solution(best_obj, best_sel))
                    write_solution()
                solved = True
        except MemoryError:
            solved = False

    if not solved:
        remaining = args.time_limit - (time.time() - start_time) - 1.0
        if remaining > 1.0:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                m = gp.Model("knapsack")
                m.Params.OutputFlag = 0
                m.Params.Seed = 0
                m.Params.MIPGap = 1e-4
                m.Params.NumericFocus = 0
                m.Params.Threads = 1
                m.Params.TimeLimit = max(1.0, remaining)

                x = m.addVars(n, vtype=GRB.BINARY)
                m.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
                m.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

                # warm start
                for i in range(n):
                    x[i].Start = 0
                for i in best_sel:
                    x[i].Start = 1

                state = {"best": best_obj, "sel": best_sel}

                def cb(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        obj_int = int(round(obj))
                        if obj_int > state["best"]:
                            vals = model.cbGetSolution([x[i] for i in range(n)])
                            sel = [i for i in range(n) if vals[i] > 0.5]
                            state["best"] = obj_int
                            state["sel"] = sel
                            if logger:
                                logger.log_solution(obj_int, make_solution(obj_int, sel))

                m.optimize(cb)

                if m.SolCount > 0:
                    obj_int = int(round(m.ObjVal))
                    if obj_int > best_obj:
                        best_obj = obj_int
                        best_sel = [i for i in range(n) if x[i].X > 0.5]
                        if logger:
                            logger.log_solution(best_obj, make_solution(best_obj, best_sel))
                elif state["best"] > best_obj:
                    best_obj = state["best"]
                    best_sel = state["sel"]
            except Exception:
                pass

    write_solution()


if __name__ == "__main__":
    main()