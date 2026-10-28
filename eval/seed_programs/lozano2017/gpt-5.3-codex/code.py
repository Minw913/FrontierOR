import argparse
import json
import time
from typing import Dict, Any, Optional


def safe_float_list(vals, n):
    out = [0.0] * n
    if vals is None:
        return out
    for i in range(min(n, len(vals))):
        try:
            out[i] = float(vals[i])
        except Exception:
            out[i] = 0.0
    return out


def trivial_solution(data: Dict[str, Any]) -> Dict[str, Any]:
    n1 = int(data.get("n1", 0))
    n2 = int(data.get("n2", 0))
    return {
        "objective_value": float(-1e18),
        "x": [0.0] * n1,
        "y": [0.0] * n2,
        "follower_objective_value": 0.0,
    }


def compute_x_ubs(n1, A1, b1, A2, b2):
    default_ub = max([0] + list(b1) + list(b2))
    ubs = [default_ub] * n1
    for i in range(n1):
        best = None
        for r in range(len(A1)):
            a = A1[r][i]
            if a > 0:
                v = b1[r] // a
                best = v if best is None else min(best, v)
        for r in range(len(A2)):
            a = A2[r][i]
            if a > 0:
                v = b2[r] // a
                best = v if best is None else min(best, v)
        if best is not None:
            ubs[i] = min(ubs[i], int(best))
        ubs[i] = max(0, int(ubs[i]))
    return ubs


def solve_instance(data: Dict[str, Any], time_limit: int, logger: Optional[object]) -> Dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except BaseException:
        return trivial_solution(data)

    start = time.time()

    def remaining():
        return float(time_limit) - (time.time() - start)

    def set_params(model, tl=None):
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        if tl is not None:
            model.Params.TimeLimit = max(0.01, float(tl))

    def update_rhs(cons, A, b, x):
        for r, cns in enumerate(cons):
            cns.RHS = b[r] - sum(A[r][i] * x[i] for i in range(len(x)))

    def build_follower_model(B, b):
        m = gp.Model()
        y = []
        for j in range(n2):
            vtype = GRB.INTEGER if j in int_followers else GRB.CONTINUOUS
            y.append(m.addVar(lb=0.0, vtype=vtype, name=f"y_{j}"))
        cons = []
        for r in range(len(B)):
            cons.append(m.addConstr(gp.quicksum(B[r][j] * y[j] for j in range(n2)) <= b[r], name=f"c_{r}"))
        m.setObjective(gp.quicksum(d2[j] * y[j] for j in range(n2)), GRB.MAXIMIZE)
        m.update()
        set_params(m)
        return m, y, cons

    def add_no_good_cut(master, xvars, xstar, xubs, cid):
        z = []
        for i in range(len(xvars)):
            zi = master.addVar(vtype=GRB.BINARY, name=f"ng_{cid}_{i}")
            M = float(max(0, xubs[i]))
            master.addConstr(xvars[i] - xstar[i] <= M * zi, name=f"ng1_{cid}_{i}")
            master.addConstr(xstar[i] - xvars[i] <= M * zi, name=f"ng2_{cid}_{i}")
            z.append(zi)
        master.addConstr(gp.quicksum(z) >= 1, name=f"ngsum_{cid}")
        master.update()

    n1 = int(data["n1"])
    n2 = int(data["n2"])
    c = list(data["c"])
    d2 = list(data["d2"])
    A1 = data["A1"]
    B1 = data["B1"]
    b1 = data["b1"]
    A2 = data["A2"]
    B2 = data["B2"]
    b2 = data["b2"]
    int_followers = set(int(v) - 1 for v in data["integer_follower_vars"])
    x_ubs = compute_x_ubs(n1, A1, b1, A2, b2)
    all_leader_int = bool(data.get("all_leader_vars_integer", True))

    # Master
    master = gp.Model("master")
    xvars = []
    for i in range(n1):
        vtype = GRB.INTEGER if all_leader_int else GRB.CONTINUOUS
        xvars.append(master.addVar(lb=0.0, ub=float(x_ubs[i]), vtype=vtype, name=f"x_{i}"))
    yvars = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in int_followers else GRB.CONTINUOUS
        yvars.append(master.addVar(lb=0.0, vtype=vtype, name=f"y_{j}"))

    for r in range(int(data["m1"])):
        master.addConstr(
            gp.quicksum(A1[r][i] * xvars[i] for i in range(n1)) +
            gp.quicksum(B1[r][j] * yvars[j] for j in range(n2)) <= b1[r]
        )
    for r in range(int(data["m2"])):
        master.addConstr(
            gp.quicksum(A2[r][i] * xvars[i] for i in range(n1)) +
            gp.quicksum(B2[r][j] * yvars[j] for j in range(n2)) <= b2[r]
        )
    master.setObjective(gp.quicksum(c[i] * xvars[i] for i in range(n1)), GRB.MAXIMIZE)
    master.update()
    set_params(master)

    # Follower models
    low_model, low_y, low_cons = build_follower_model(B2, b2)

    both_model = gp.Model("follower_both")
    both_y = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in int_followers else GRB.CONTINUOUS
        both_y.append(both_model.addVar(lb=0.0, vtype=vtype, name=f"y_{j}"))
    both_low_cons = []
    for r in range(len(B2)):
        both_low_cons.append(
            both_model.addConstr(gp.quicksum(B2[r][j] * both_y[j] for j in range(n2)) <= b2[r])
        )
    both_up_cons = []
    for r in range(len(B1)):
        both_up_cons.append(
            both_model.addConstr(gp.quicksum(B1[r][j] * both_y[j] for j in range(n2)) <= b1[r])
        )
    both_model.setObjective(gp.quicksum(d2[j] * both_y[j] for j in range(n2)), GRB.MAXIMIZE)
    both_model.update()
    set_params(both_model)

    best = None
    best_obj = -1e100
    tol = 1e-6
    cut_id = 0

    while remaining() > 0.05:
        set_params(master, remaining())
        master.optimize()
        if master.SolCount == 0:
            break

        xstar = [int(round(v.X)) for v in xvars]
        leader_obj = float(sum(c[i] * xstar[i] for i in range(n1)))

        # Lower-level only
        update_rhs(low_cons, A2, b2, xstar)
        if remaining() <= 0.01:
            break
        set_params(low_model, remaining())
        low_model.optimize()
        if low_model.SolCount == 0 or low_model.Status != GRB.OPTIMAL:
            add_no_good_cut(master, xvars, xstar, x_ubs, cut_id)
            cut_id += 1
            continue
        z_low = float(low_model.ObjVal)

        # Lower + upper
        update_rhs(both_low_cons, A2, b2, xstar)
        update_rhs(both_up_cons, A1, b1, xstar)
        if remaining() <= 0.01:
            break
        set_params(both_model, remaining())
        both_model.optimize()
        if both_model.SolCount == 0 or both_model.Status != GRB.OPTIMAL:
            add_no_good_cut(master, xvars, xstar, x_ubs, cut_id)
            cut_id += 1
            continue
        z_both = float(both_model.ObjVal)

        # optimistic check
        if z_both + tol < z_low:
            add_no_good_cut(master, xvars, xstar, x_ubs, cut_id)
            cut_id += 1
            continue

        ystar = [float(both_y[j].X) for j in range(n2)]
        fobj = float(sum(d2[j] * ystar[j] for j in range(n2)))

        if leader_obj > best_obj + 1e-9:
            best_obj = leader_obj
            best = {
                "objective_value": float(best_obj),
                "x": [float(v) for v in xstar],
                "y": ystar,
                "follower_objective_value": fobj,
            }
            if logger:
                try:
                    logger.log_solution(best_obj, best)
                except Exception:
                    pass

        if master.Status == GRB.OPTIMAL:
            break

        add_no_good_cut(master, xvars, xstar, x_ubs, cut_id)
        cut_id += 1

        try:
            if best is not None and master.ObjBound <= best_obj + 1e-6:
                break
        except Exception:
            pass

    # Fallback x=0
    if best is None and remaining() > 0.05:
        x0 = [0] * n1
        update_rhs(low_cons, A2, b2, x0)
        set_params(low_model, remaining())
        low_model.optimize()
        if low_model.SolCount > 0 and low_model.Status == GRB.OPTIMAL:
            z_low = float(low_model.ObjVal)
            update_rhs(both_low_cons, A2, b2, x0)
            update_rhs(both_up_cons, A1, b1, x0)
            set_params(both_model, remaining())
            both_model.optimize()
            if both_model.SolCount > 0 and both_model.Status == GRB.OPTIMAL:
                z_both = float(both_model.ObjVal)
                if z_both + tol >= z_low:
                    y0 = [float(both_y[j].X) for j in range(n2)]
                    fobj0 = float(sum(d2[j] * y0[j] for j in range(n2)))
                    obj0 = float(sum(c[i] * x0[i] for i in range(n1)))
                    best = {
                        "objective_value": obj0,
                        "x": [0.0] * n1,
                        "y": y0,
                        "follower_objective_value": fobj0,
                    }
                    if logger:
                        try:
                            logger.log_solution(obj0, best)
                        except Exception:
                            pass

    if best is None:
        best = trivial_solution(data)

    best["x"] = safe_float_list(best.get("x", []), n1)
    best["y"] = safe_float_list(best.get("y", []), n2)
    best["objective_value"] = float(best.get("objective_value", -1e18))
    best["follower_objective_value"] = float(best.get("follower_objective_value", 0.0))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Import logger defensively (some environments may fail due system bus/keyring issues).
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    data = None
    solution = None
    try:
        with open(args.instance_path, "r") as f:
            data = json.load(f)
        solution = solve_instance(data, args.time_limit, logger)
    except BaseException:
        if data is None:
            solution = {
                "objective_value": float(-1e18),
                "x": [],
                "y": [],
                "follower_objective_value": 0.0,
            }
        else:
            solution = trivial_solution(data)

    # Ensure output write never crashes silently.
    try:
        with open(args.solution_path, "w") as f:
            json.dump(solution, f)
    except Exception:
        # Last resort: attempt minimal write
        fallback = solution if isinstance(solution, dict) else {
            "objective_value": -1e18, "x": [], "y": [], "follower_objective_value": 0.0
        }
        try:
            with open(args.solution_path, "w") as f:
                f.write(json.dumps(fallback))
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Never propagate exceptions to runner.
        pass