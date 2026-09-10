import argparse
import json
import time
import numpy as np


def compute_objective(x, P):
    # x: 0/1 numpy vector, P: full symmetric matrix
    return float(x @ P @ x)


def greedy_solution(n, w, P, cap):
    # Greedy by (row profit potential)/weight
    rowsum = P.sum(axis=1)
    diag = np.diag(P)
    score = (diag + rowsum) / np.maximum(w, 1e-9)
    order = np.argsort(-score)
    x = np.zeros(n, dtype=np.int64)
    W = 0
    for i in order:
        if W + w[i] <= cap:
            x[i] = 1
            W += w[i]
    return x, W


def local_search(x, W, n, w, P, cap, end_time, logger=None):
    diag = np.diag(P).astype(np.float64)
    Pf = P.astype(np.float64)
    best_obj = compute_objective(x.astype(np.float64), Pf)
    if logger:
        logger.log_solution(best_obj, {"objective_value": best_obj,
                                       "selected_items": [int(v) for v in x]})
    improved = True
    while improved and time.time() < end_time:
        improved = False
        xf = x.astype(np.float64)
        s = Pf @ xf  # s_i = sum_j P[i][j] x_j

        # Try adds first
        add_gain = diag + 2.0 * s  # gain of adding item j (currently out)
        out_idx = np.where(x == 0)[0]
        if len(out_idx) > 0:
            feas = out_idx[w[out_idx] + W <= cap]
            if len(feas) > 0:
                gains = add_gain[feas]
                j = feas[int(np.argmax(gains))]
                if gains.max() > 1e-9:
                    x[j] = 1
                    W += w[j]
                    best_obj += add_gain[j]
                    improved = True
                    if logger:
                        logger.log_solution(best_obj, {"objective_value": best_obj,
                                                       "selected_items": [int(v) for v in x]})
                    continue

        # Try swaps: remove i (in), add j (out)
        in_idx = np.where(x == 1)[0]
        out_idx = np.where(x == 0)[0]
        if len(in_idx) == 0 or len(out_idx) == 0:
            break
        rem_loss = 2.0 * s[in_idx] - diag[in_idx]        # loss of removing i
        add_g = diag[out_idx] + 2.0 * s[out_idx]          # gain of adding j (before removal)
        # gain matrix: add_g[j] - rem_loss[i] - 2*P[i][j]
        G = add_g[None, :] - rem_loss[:, None] - 2.0 * Pf[np.ix_(in_idx, out_idx)]
        # feasibility: W - w[i] + w[j] <= cap
        feas_mat = (W - w[in_idx][:, None] + w[out_idx][None, :]) <= cap
        G = np.where(feas_mat, G, -np.inf)
        bi, bj = np.unravel_index(int(np.argmax(G)), G.shape)
        if G[bi, bj] > 1e-9:
            i = in_idx[bi]
            j = out_idx[bj]
            x[i] = 0
            x[j] = 1
            W = W - w[i] + w[j]
            best_obj += G[bi, bj]
            improved = True
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj,
                                               "selected_items": [int(v) for v in x]})
    return x, W, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.time_limit

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    cap = int(inst["capacity"])
    w = np.array(inst["weights"], dtype=np.int64)
    P = np.array(inst["profit_matrix"], dtype=np.int64)

    # ---------- Heuristic phase ----------
    heur_budget = min(5.0, 0.1 * args.time_limit)
    heur_end = min(deadline - 1.0, start_time + heur_budget)
    x, W = greedy_solution(n, w, P, cap)
    x, W, best_obj = local_search(x, W, n, w, P, cap, heur_end, logger)
    best_x = x.copy()

    # ---------- Gurobi phase ----------
    remaining = deadline - time.time() - 1.0
    if remaining > 2.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("qkp")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, remaining)

            xv = model.addVars(n, vtype=GRB.BINARY, name="x")
            model.addConstr(gp.quicksum(w[i] * xv[i] for i in range(n)) <= cap)

            obj = gp.QuadExpr()
            for i in range(n):
                if P[i, i] != 0:
                    obj.add(xv[i], float(P[i, i]))
                for j in range(i + 1, n):
                    if P[i, j] != 0:
                        obj.add(xv[i] * xv[j], 2.0 * float(P[i, j]))
            model.setObjective(obj, GRB.MAXIMIZE)

            # Warm start
            for i in range(n):
                xv[i].Start = int(best_x[i])

            state = {"best": best_obj, "x": best_x.copy()}

            def cb(m, where):
                if where == GRB.Callback.MIPSOL:
                    val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if val > state["best"] + 1e-6:
                        sol = m.cbGetSolution([xv[i] for i in range(n)])
                        xs = np.array([1 if s > 0.5 else 0 for s in sol], dtype=np.int64)
                        state["best"] = val
                        state["x"] = xs
                        if logger:
                            logger.log_solution(float(val), {
                                "objective_value": float(val),
                                "selected_items": [int(v) for v in xs]})

            model.optimize(cb)

            if model.SolCount > 0:
                sol = np.array([1 if xv[i].X > 0.5 else 0 for i in range(n)], dtype=np.int64)
                val = compute_objective(sol.astype(np.float64), P.astype(np.float64))
                if val > best_obj:
                    best_obj = val
                    best_x = sol
            if state["best"] > best_obj:
                best_obj = state["best"]
                best_x = state["x"]
        except Exception:
            pass

    # Verify feasibility
    if int(w @ best_x) > cap:
        best_x = np.zeros(n, dtype=np.int64)
        best_obj = 0.0
    best_obj = compute_objective(best_x.astype(np.float64), P.astype(np.float64))

    solution = {
        "objective_value": float(best_obj),
        "selected_items": [int(v) for v in best_x],
    }
    if logger:
        logger.log_solution(float(best_obj), solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()