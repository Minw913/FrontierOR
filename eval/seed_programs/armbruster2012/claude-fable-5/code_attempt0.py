import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def cut_cost(assign, edges, ew):
    total = 0.0
    for (u, v), w in zip(edges, ew):
        if assign[u] != assign[v]:
            total += w
    return total


def build_solution(assign, edges, ew, n):
    obj = cut_cost(assign, edges, ew)
    ref = assign[0]
    S = [i for i in range(n) if assign[i] == ref]
    C = [i for i in range(n) if assign[i] != ref]
    return {
        "objective_value": float(obj),
        "partition_S": S,
        "partition_complement": C,
    }


def heuristic_solution(n, weights, edges, ew, F, total_w, time_budget, rng):
    """Return a feasible assignment (list of 0/1) or None. Uses greedy + FM-like local search."""
    # adjacency
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, ew[idx]))
        adj[v].append((u, ew[idx]))

    lower = total_w - F  # min weight in group 1 (and group 0)

    def make_initial():
        # LPT: assign heavy nodes to lighter side
        order = sorted(range(n), key=lambda i: -weights[i])
        assign = [0] * n
        w0, w1 = 0, 0
        for i in order:
            if w0 <= w1:
                if w0 + weights[i] <= F:
                    assign[i] = 0
                    w0 += weights[i]
                else:
                    assign[i] = 1
                    w1 += weights[i]
            else:
                if w1 + weights[i] <= F:
                    assign[i] = 1
                    w1 += weights[i]
                else:
                    assign[i] = 0
                    w0 += weights[i]
        if w0 <= F and w1 <= F:
            return assign, w0, w1
        return None, w0, w1

    assign, w0, w1 = make_initial()
    if assign is None:
        return None, None

    def local_search(assign, w0, w1, deadline):
        # gain of moving node i to other side = (internal cut change)
        improved = True
        best_obj = cut_cost(assign, edges, ew)
        while improved and time.time() < deadline:
            improved = False
            # compute gains
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                si = assign[i]
                # capacity check
                if si == 0:
                    nw0, nw1 = w0 - weights[i], w1 + weights[i]
                else:
                    nw0, nw1 = w0 + weights[i], w1 - weights[i]
                if nw0 > F or nw1 > F:
                    continue
                gain = 0.0
                for j, w in adj[i]:
                    if assign[j] == si:
                        gain -= w
                    else:
                        gain += w
                if gain > 1e-12:
                    assign[i] = 1 - si
                    w0, w1 = nw0, nw1
                    best_obj -= gain
                    improved = True
            # pairwise swap pass (limited)
            if time.time() >= deadline:
                break
        return assign, best_obj, w0, w1

    deadline = time.time() + time_budget
    best_assign = None
    best_obj = float("inf")
    attempt = 0
    while time.time() < deadline:
        if attempt == 0:
            cur, cw0, cw1 = make_initial()
        else:
            # random restart
            cur = None
            for _ in range(50):
                trial = [rng.randint(0, 1) for _ in range(n)]
                tw1 = sum(weights[i] for i in range(n) if trial[i] == 1)
                tw0 = total_w - tw1
                # repair
                items = sorted(range(n), key=lambda i: -weights[i])
                for i in items:
                    if tw0 > F and trial[i] == 0 and tw1 + weights[i] <= F:
                        trial[i] = 1
                        tw0 -= weights[i]
                        tw1 += weights[i]
                    elif tw1 > F and trial[i] == 1 and tw0 + weights[i] <= F:
                        trial[i] = 0
                        tw1 -= weights[i]
                        tw0 += weights[i]
                if tw0 <= F and tw1 <= F:
                    cur, cw0, cw1 = trial, tw0, tw1
                    break
            if cur is None:
                attempt += 1
                continue
        if cur is None:
            break
        cur, obj, cw0, cw1 = local_search(cur, cw0, cw1, deadline)
        if obj < best_obj:
            best_obj = obj
            best_assign = list(cur)
        attempt += 1
        if attempt > 30:
            break
    return best_assign, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = read_instance(args.instance_path)
    n = data["num_nodes"]
    m = data["num_edges"]
    F = data["bisection_capacity_F"]
    total_w = data["total_node_weight"]
    weights = data["node_weights"]
    edges = [list(e) for e in data["edges"]]
    ew = data["edge_weights"]

    rng = random.Random(0)

    if n == 0:
        sol = {"objective_value": 0.0, "partition_S": [], "partition_complement": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # ---- Heuristic warm start ----
    heur_budget = min(5.0, max(0.5, 0.05 * args.time_limit))
    warm_assign, warm_obj = heuristic_solution(
        n, weights, edges, ew, F, total_w, heur_budget, rng
    )

    best_sol = None
    if warm_assign is not None:
        best_sol = build_solution(warm_assign, edges, ew, n)
        if logger:
            logger.log_solution(best_sol["objective_value"], best_sol)

    # ---- Gurobi MIP ----
    remaining = args.time_limit - (time.time() - start) - 1.0
    if remaining > 1.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("bisection")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, remaining)

            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            y = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

            # symmetry break: node 0 in group 0
            model.addConstr(x[0] == 0)

            for idx in range(m):
                u, v = edges[idx]
                model.addConstr(y[idx] >= x[u] - x[v])
                model.addConstr(y[idx] >= x[v] - x[u])

            wsum = gp.quicksum(weights[i] * x[i] for i in range(n))
            model.addConstr(wsum <= F)
            model.addConstr(wsum >= total_w - F)

            model.setObjective(
                gp.quicksum(ew[idx] * y[idx] for idx in range(m)), GRB.MINIMIZE
            )

            if warm_assign is not None:
                ref = warm_assign[0]
                for i in range(n):
                    x[i].Start = 0 if warm_assign[i] == ref else 1

            state = {"best": best_sol["objective_value"] if best_sol else float("inf")}

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < state["best"] - 1e-9:
                        vals = model.cbGetSolution([x[i] for i in range(n)])
                        assign = [1 if v > 0.5 else 0 for v in vals]
                        sol = build_solution(assign, edges, ew, n)
                        state["best"] = sol["objective_value"]
                        state["sol"] = sol
                        if logger:
                            logger.log_solution(sol["objective_value"], sol)

            state["sol"] = best_sol
            model.optimize(callback)

            if model.SolCount > 0:
                vals = [x[i].X for i in range(n)]
                assign = [1 if v > 0.5 else 0 for v in vals]
                sol = build_solution(assign, edges, ew, n)
                if best_sol is None or sol["objective_value"] < best_sol["objective_value"] - 1e-9:
                    best_sol = sol
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)
                elif state.get("sol") is not None and state["sol"]["objective_value"] < best_sol["objective_value"]:
                    best_sol = state["sol"]
        except Exception:
            pass

    if best_sol is None:
        # last resort: try trivial split by sorting
        order = sorted(range(n), key=lambda i: -weights[i])
        assign = [0] * n
        w0, w1 = 0, 0
        for i in order:
            if w0 <= w1:
                assign[i] = 0
                w0 += weights[i]
            else:
                assign[i] = 1
                w1 += weights[i]
        best_sol = build_solution(assign, edges, ew, n)
        if logger:
            logger.log_solution(best_sol["objective_value"], best_sol)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()