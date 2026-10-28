#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    N = int(inst["N"])
    S = int(inst["S"])
    k = int(inst["k"])
    beta = float(inst["beta"])
    gamma = float(inst["gamma"])
    mu_bar = float(inst["mu_bar"])
    mu = np.asarray(inst["mu"], dtype=float).reshape(N)
    p = float(inst.get("p_s", 1.0 / S if S > 0 else 0.0))
    R = np.asarray(inst["scenarios"], dtype=float).reshape(S, N)

    c = 1.0 / (1.0 - beta)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # ---------- helper functions ----------
    def compute_cvar(losses):
        """Optimal (a, v) for a fixed portfolio x, where losses = -R @ x."""
        if S == 0:
            return 0.0, 0.0
        cum = np.arange(1, S + 1, dtype=float) * p
        idx = int(np.searchsorted(cum, beta, side="left"))
        if idx >= S:
            idx = S - 1
        if idx < 0:
            idx = 0
        a = float(np.partition(losses, idx)[idx])
        excess = np.maximum(losses - a, 0.0)
        v = float(c * p * np.sum(excess))
        return a, v

    def eval_obj(x, a, v):
        return float((1.0 / (2.0 * gamma)) * np.dot(x, x) + a + v)

    def make_solution(x_vals, z_vals=None):
        """
        Clean a raw solution, enforce structural constraints, and recompute the
        optimal CVaR pair (a, v) for the resulting x.
        """
        x = np.array(x_vals, dtype=float)
        x = np.maximum(x, 0.0)

        if z_vals is not None:
            z = np.array([1 if zv > 0.5 else 0 for zv in z_vals], dtype=int)
            x = np.where(z == 1, x, 0.0)
        else:
            z = np.zeros(N, dtype=int)

        # z must be 1 whenever x > 0
        z = np.where(x > 1e-9, 1, z)

        # cardinality enforcement
        if int(z.sum()) > k:
            order = np.argsort(x)[::-1][:k]
            z = np.zeros(N, dtype=int)
            z[order] = 1
            x = np.where(z == 1, x, 0.0)

        # drop zero-weight selections
        z = np.where(x > 1e-9, 1, 0)

        s = float(x.sum())
        if s > 0.0:
            diff = 1.0 - s
            if abs(diff) > 1e-9:
                pos = np.where(x > 0.0)[0]
                if len(pos) > 0:
                    if diff > 0.0:
                        j = pos[np.argmax(mu[pos])]
                    else:
                        j = pos[np.argmin(mu[pos])]
                    x[j] += diff
                    if x[j] < 0.0:
                        x[j] = 0.0
            s2 = float(x.sum())
            if s2 > 0.0 and abs(s2 - 1.0) > 1e-6:
                x = x / s2
        else:
            best_i = int(np.argmax(mu))
            x = np.zeros(N)
            x[best_i] = 1.0
            z = np.zeros(N, dtype=int)
            z[best_i] = 1

        z = np.where(x > 1e-9, 1, 0)
        if int(z.sum()) > k:
            order = np.argsort(x)[::-1][:k]
            z = np.zeros(N, dtype=int)
            z[order] = 1
            x = np.where(z == 1, x, 0.0)
            s = float(x.sum())
            if s > 0.0:
                x = x / s

        losses = -R.dot(x)
        a, v = compute_cvar(losses)
        obj = eval_obj(x, a, v)
        return obj, x.tolist(), [int(zz) for zz in z], float(a), float(v)

    def fallback_solution():
        best_i = int(np.argmax(mu))
        x = np.zeros(N)
        x[best_i] = 1.0
        z = np.zeros(N, dtype=int)
        z[best_i] = 1
        losses = -R[:, best_i]
        a, v = compute_cvar(losses)
        obj = eval_obj(x, a, v)
        return obj, x.tolist(), [int(zz) for zz in z], float(a), float(v)

    def is_feasible(x, z, a, v):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=int)
        if abs(float(x.sum()) - 1.0) > 1e-4:
            return False
        if np.any(x < -1e-6):
            return False
        if int(round(float(z.sum()))) > k:
            return False
        if np.any((x > 1e-6) & (z < 0.5)):
            return False
        if float(mu.dot(x)) < mu_bar - 1e-4:
            return False
        losses = -R.dot(x)
        req = c * p * np.sum(np.maximum(losses - a, 0.0))
        if float(v) < float(req) - 1e-4:
            return False
        return True

    def solve_continuous(selected, time_limit):
        """Polish/heuristic: solve the convex QP for a fixed support."""
        selected = [int(i) for i in selected if 0 <= i < N]
        selected = list(dict.fromkeys(selected))
        if not selected:
            return None
        if len(selected) > k:
            selected = selected[:k]

        n = len(selected)
        m = gp.Model("cont_polish")
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.Seed = 0
        m.Params.NumericFocus = 0
        m.Params.TimeLimit = max(0.01, float(time_limit))

        xs = m.addMVar(n, lb=0.0, ub=1.0)
        a = m.addVar(lb=-GRB.INFINITY, name="a")
        v = m.addVar(lb=0.0, name="v")
        u = m.addMVar(S, lb=0.0)

        m.addConstr(xs.sum() == 1.0, "budget")
        m.addConstr(mu[selected] @ xs >= mu_bar, "return")
        Rsel = R[:, selected]
        m.addConstr(u + Rsel @ xs + a >= 0.0, "cvar_loss")
        m.addConstr(v >= c * p * u.sum(), "cvar_v")

        xs_vars = xs.tolist()
        m.setObjective(
            (1.0 / (2.0 * gamma)) * gp.quicksum(xs_vars[j] * xs_vars[j] for j in range(n))
            + a
            + v,
            GRB.MINIMIZE,
        )
        m.optimize()

        if m.SolCount == 0:
            return None

        x_full = np.zeros(N)
        for j, i in enumerate(selected):
            x_full[i] = xs_vars[j].X
        return make_solution(x_full, None)

    # ---------- build perspective-strengthened MISOCP ----------
    model = gp.Model("card_cvar")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.NumericFocus = 0
    model.Params.MIPGap = 1e-4

    x = model.addMVar(N, lb=0.0, ub=1.0, name="x")
    z = model.addMVar(N, vtype=GRB.BINARY, name="z")
    y = model.addMVar(N, lb=0.0, ub=1.0, name="y")
    a_var = model.addVar(lb=-GRB.INFINITY, name="a")
    v_var = model.addVar(lb=0.0, name="v")
    u = model.addMVar(S, lb=0.0, name="u")

    model.addConstr(x.sum() == 1.0, "budget")
    model.addConstr(mu @ x >= mu_bar, "return")
    model.addConstr(x <= z, "link")
    model.addConstr(z.sum() <= k, "card")
    model.addConstr(u + R @ x + a_var >= 0.0, "cvar_loss")
    model.addConstr(v_var >= c * p * u.sum(), "cvar_v")

    x_list = x.tolist()
    z_list = z.tolist()
    y_list = y.tolist()
    u_list = u.tolist()

    # Perspective constraints: y_i >= x_i^2 / z_i for z_i > 0
    for i in range(N):
        model.addConstr(x_list[i] * x_list[i] <= z_list[i] * y_list[i], name=f"persp_{i}")

    model.setObjective((1.0 / (2.0 * gamma)) * y.sum() + a_var + v_var, GRB.MINIMIZE)
    model.update()

    # ---------- initial feasible solution ----------
    best_obj = float("inf")
    best_sol = None

    fb_obj, fb_x, fb_z, fb_a, fb_v = fallback_solution()
    fb_xnp = np.asarray(fb_x)
    fb_znp = np.asarray(fb_z)
    if is_feasible(fb_xnp, fb_znp, fb_a, fb_v):
        best_obj = fb_obj
        best_sol = {
            "objective_value": fb_obj,
            "x": fb_x,
            "z": fb_z,
            "a": fb_a,
            "v": fb_v,
        }
        if logger:
            logger.log_solution(fb_obj, best_sol)

        # Provide MIP start
        losses0 = -R.dot(fb_xnp)
        u0 = np.maximum(losses0 - fb_a, 0.0)
        for i in range(N):
            x_list[i].Start = float(fb_xnp[i])
            z_list[i].Start = float(fb_znp[i])
            y_list[i].Start = float(fb_xnp[i] * fb_xnp[i])
        a_var.Start = float(fb_a)
        v_var.Start = float(fb_v)
        if S <= 50000:
            for s in range(S):
                u_list[s].Start = float(u0[s])

    # ---------- callback for incumbent logging ----------
    def callback(model, where):
        nonlocal best_obj, best_sol
        if where == GRB.Callback.MIPSOL:
            try:
                sol = model.cbGet(GRB.Callback.MIPSOL_X)
                x_vals = np.array([sol[var.index] for var in x_list], dtype=float)
                z_vals = [int(round(sol[var.index])) for var in z_list]
                a_val = float(sol[a_var.index])
                v_val = float(sol[v_var.index])

                x_clean = np.maximum(x_vals, 0.0)
                z_arr = np.array(z_vals, dtype=int)
                x_clean = np.where(z_arr == 1, x_clean, 0.0)

                obj = float((1.0 / (2.0 * gamma)) * np.dot(x_clean, x_clean) + a_val + v_val)
                if obj < best_obj - 1e-12:
                    best_obj = obj
                    best_sol = {
                        "objective_value": obj,
                        "x": x_clean.tolist(),
                        "z": z_vals,
                        "a": a_val,
                        "v": v_val,
                    }
                    if logger:
                        logger.log_solution(obj, best_sol)
            except Exception:
                pass

    # ---------- optimize main model ----------
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.0:
        reserve = max(0.1, min(2.0, 0.05 * args.time_limit))
        main_time = max(0.01, remaining - reserve)
        model.Params.TimeLimit = main_time
        try:
            model.optimize(callback)
        except gp.GurobiError:
            pass

    # Extract Gurobi's best solution if available
    if model.SolCount > 0:
        try:
            x_vals = np.array([var.X for var in x_list], dtype=float)
            z_vals = [int(round(var.X)) for var in z_list]
            obj, x_out, z_out, a_out, v_out = make_solution(x_vals, z_vals)
            if obj < best_obj:
                best_obj = obj
                best_sol = {
                    "objective_value": obj,
                    "x": x_out,
                    "z": z_out,
                    "a": a_out,
                    "v": v_out,
                }
                if logger:
                    logger.log_solution(obj, best_sol)
        except Exception:
            pass

    # ---------- polishing if time remains and not proven optimal ----------
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.5 and best_sol is not None:
        if model.status != GRB.OPTIMAL:
            selected = [i for i, zz in enumerate(best_sol["z"]) if zz > 0.5]
            if not selected:
                selected = [int(np.argmax(mu))]
            polish_time = min(remaining - 0.2, remaining * 0.5)
            if polish_time > 0.1:
                try:
                    polish = solve_continuous(selected, polish_time)
                except Exception:
                    polish = None
                if polish is not None:
                    obj_p, x_p, z_p, a_p, v_p = polish
                    if obj_p < best_obj - 1e-12:
                        best_obj = obj_p
                        best_sol = {
                            "objective_value": obj_p,
                            "x": x_p,
                            "z": z_p,
                            "a": a_p,
                            "v": v_p,
                        }
                        if logger:
                            logger.log_solution(obj_p, best_sol)

    # ---------- final fallback ----------
    if best_sol is None:
        obj_fb, x_fb, z_fb, a_fb, v_fb = fallback_solution()
        best_obj = obj_fb
        best_sol = {
            "objective_value": obj_fb,
            "x": x_fb,
            "z": z_fb,
            "a": a_fb,
            "v": v_fb,
        }
        if logger:
            logger.log_solution(obj_fb, best_sol)

    solution = {
        "objective_value": float(best_sol["objective_value"]),
        "x": [float(xx) for xx in best_sol["x"]],
        "z": [int(zz) for zz in best_sol["z"]],
        "a": float(best_sol["a"]),
        "v": float(best_sol["v"]),
    }

    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()