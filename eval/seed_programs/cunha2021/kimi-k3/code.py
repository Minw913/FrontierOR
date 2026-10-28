import argparse
import json
import os
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


ZERO_EPS = 1e-10
POS_EPS = 1e-8
NEG_CLIP = 1e-6
IMPROVE_EPS = 1e-9


def suffix_sums(rows):
    n_rows = len(rows)
    t = len(rows[0]) if n_rows else 0
    out = [[0.0] * t for _ in range(n_rows)]
    for i in range(n_rows):
        run = 0.0
        for k in range(t - 1, -1, -1):
            run += rows[i][k]
            out[i][k] = run
    return out


def prepare_instance(inst):
    r = int(inst["num_retailers"])
    w = int(inst["num_warehouses"])
    t = int(inst["num_periods"])

    assignment = [int(v) for v in inst["retailer_warehouse_assignment"]]
    warehouse_sets = [[] for _ in range(w)]
    for i, wi in enumerate(assignment):
        warehouse_sets[wi].append(i)

    demands = [[float(v) for v in row] for row in inst["demands"]]

    warehouse_demand = [
        [sum(demands[i][k] for i in warehouse_sets[wi]) for k in range(t)]
        for wi in range(w)
    ]
    plant_demand = [sum(warehouse_demand[wi][k] for wi in range(w)) for k in range(t)]

    rem_retailers = suffix_sums(demands)
    rem_warehouses = suffix_sums(warehouse_demand)
    rem_plant = [0.0] * t
    run = 0.0
    for k in range(t - 1, -1, -1):
        run += plant_demand[k]
        rem_plant[k] = run

    fixed = {
        "plant": [float(v) for v in inst["fixed_costs"]["plant"]],
        "warehouses": [[float(v) for v in row] for row in inst["fixed_costs"]["warehouses"]],
        "retailers": [[float(v) for v in row] for row in inst["fixed_costs"]["retailers"]],
    }
    holding = {
        "plant": float(inst["holding_costs"]["plant"]),
        "warehouses": float(inst["holding_costs"]["warehouses"]),
        "retailers": [float(v) for v in inst["holding_costs"]["retailers"]],
    }

    nonnegative_costs = (
        holding["plant"] >= -1e-12
        and holding["warehouses"] >= -1e-12
        and all(v >= -1e-12 for v in holding["retailers"])
        and all(v >= -1e-12 for v in fixed["plant"])
        and all(v >= -1e-12 for row in fixed["warehouses"] for v in row)
        and all(v >= -1e-12 for row in fixed["retailers"] for v in row)
    )

    return {
        "R": r,
        "W": w,
        "T": t,
        "assignment": assignment,
        "warehouse_sets": warehouse_sets,
        "demands": demands,
        "warehouse_demand": warehouse_demand,
        "plant_demand": plant_demand,
        "rem_retailers": rem_retailers,
        "rem_warehouses": rem_warehouses,
        "rem_plant": rem_plant,
        "fixed": fixed,
        "holding": holding,
        "nonnegative_costs": nonnegative_costs,
    }


def _clean(v):
    v = float(v)
    if abs(v) < ZERO_EPS:
        return 0.0
    return v


def _nonneg(v):
    if v < 0.0 and v > -NEG_CLIP:
        return 0.0
    return v


def build_output(data, xp_in, xw_in, xr_in):
    R = data["R"]
    W = data["W"]
    T = data["T"]
    sets = data["warehouse_sets"]
    d = data["demands"]
    fixed = data["fixed"]
    holding = data["holding"]

    xp = [_clean(v) for v in xp_in]
    xw = [[_clean(v) for v in row] for row in xw_in]
    xr = [[_clean(v) for v in row] for row in xr_in]

    sp = [0.0] * T
    for k in range(T):
        prev = sp[k - 1] if k > 0 else 0.0
        out = 0.0
        for wi in range(W):
            out += xw[wi][k]
        sp[k] = _nonneg(prev + xp[k] - out)

    sw = [[0.0] * T for _ in range(W)]
    for wi in range(W):
        rows = sets[wi]
        for k in range(T):
            prev = sw[wi][k - 1] if k > 0 else 0.0
            out = 0.0
            for i in rows:
                out += xr[i][k]
            sw[wi][k] = _nonneg(prev + xw[wi][k] - out)

    sr = [[0.0] * T for _ in range(R)]
    for i in range(R):
        di = d[i]
        xi = xr[i]
        si = sr[i]
        for k in range(T):
            prev = si[k - 1] if k > 0 else 0.0
            si[k] = _nonneg(prev + xi[k] - di[k])

    yp = [1 if xp[k] > POS_EPS else 0 for k in range(T)]
    yw = [[1 if xw[wi][k] > POS_EPS else 0 for k in range(T)] for wi in range(W)]
    yr = [[1 if xr[i][k] > POS_EPS else 0 for k in range(T)] for i in range(R)]

    obj = 0.0
    fp = fixed["plant"]
    fw = fixed["warehouses"]
    fr = fixed["retailers"]
    hp = holding["plant"]
    hw = holding["warehouses"]
    hr = holding["retailers"]

    for k in range(T):
        if yp[k]:
            obj += fp[k]
        obj += hp * sp[k]
    for wi in range(W):
        fwi = fw[wi]
        swi = sw[wi]
        ywi = yw[wi]
        for k in range(T):
            if ywi[k]:
                obj += fwi[k]
            obj += hw * swi[k]
    for i in range(R):
        fri = fr[i]
        sri = sr[i]
        yri = yr[i]
        hri = hr[i]
        for k in range(T):
            if yri[k]:
                obj += fri[k]
            obj += hri * sri[k]

    sol = {
        "objective_value": float(obj),
        "x_plant": [float(v) for v in xp],
        "s_plant": [float(v) for v in sp],
        "y_plant": [int(v) for v in yp],
        "x_warehouses": [[float(v) for v in row] for row in xw],
        "s_warehouses": [[float(v) for v in row] for row in sw],
        "y_warehouses": [[int(v) for v in row] for row in yw],
        "x_retailers": [[float(v) for v in row] for row in xr],
        "s_retailers": [[float(v) for v in row] for row in sr],
        "y_retailers": [[int(v) for v in row] for row in yr],
    }
    return float(obj), sol


def uls_quantities(demand, fixed, hold):
    n = len(demand)
    if n == 0:
        return []
    pref = [0.0] * (n + 1)
    pref_moment = [0.0] * (n + 1)
    for k, v in enumerate(demand):
        pref[k + 1] = pref[k] + v
        pref_moment[k + 1] = pref_moment[k] + k * v

    inf = float("inf")
    cost = [0.0] + [inf] * n
    prev = [-1] * (n + 1)

    for j in range(1, n + 1):
        pj = pref[j]
        qj = pref_moment[j]
        best = inf
        best_i = -1
        for i in range(j):
            ci = cost[i]
            if ci == inf:
                continue
            qty = pj - pref[i]
            if qty > ZERO_EPS:
                hold_cost = hold * ((qj - pref_moment[i]) - i * qty)
                val = ci + fixed[i] + hold_cost
            else:
                val = ci
            if val < best - 1e-12:
                best = val
                best_i = i
        cost[j] = best
        prev[j] = best_i

    x = [0.0] * n
    j = n
    while j > 0:
        i = prev[j]
        if i < 0:
            break
        qty = pref[j] - pref[i]
        if qty > ZERO_EPS:
            x[i] += qty
        j = i
    return x


def initial_best_solution(data, logger):
    xp = list(data["plant_demand"])
    xw = [list(row) for row in data["warehouse_demand"]]
    xr = [list(row) for row in data["demands"]]
    best_obj, best_sol = build_output(data, xp, xw, xr)
    best = {"obj": best_obj, "sol": best_sol}
    if logger:
        logger.log_solution(best_obj, best_sol)

    # A cheap bottom-up lot-sizing heuristic: batch retailers, then warehouses, then plant.
    R, W, T = data["R"], data["W"], data["T"]
    if data["nonnegative_costs"] and T > 0 and (R + W + 1) * T * T <= 5_000_000:
        try:
            retailer_ship = []
            for i in range(R):
                retailer_ship.append(
                    uls_quantities(data["demands"][i], data["fixed"]["retailers"][i], data["holding"]["retailers"][i])
                )
            warehouse_recv = []
            plant_out_demand = [0.0] * T
            for wi in range(W):
                need = [0.0] * T
                for i in data["warehouse_sets"][wi]:
                    ri = retailer_ship[i]
                    for k in range(T):
                        need[k] += ri[k]
                q = uls_quantities(need, data["fixed"]["warehouses"][wi], data["holding"]["warehouses"])
                warehouse_recv.append(q)
                for k in range(T):
                    plant_out_demand[k] += q[k]
            plant_prod = uls_quantities(plant_out_demand, data["fixed"]["plant"], data["holding"]["plant"])
            obj2, sol2 = build_output(data, plant_prod, warehouse_recv, retailer_ship)
            if obj2 < best["obj"] - IMPROVE_EPS:
                best["obj"] = obj2
                best["sol"] = sol2
                if logger:
                    logger.log_solution(obj2, sol2)
        except Exception:
            pass
    return best


def solution_from_flat_values(data, vals):
    T = data["T"]
    W = data["W"]
    R = data["R"]
    xp = vals[:T]
    off = T
    xw = [vals[off + wi * T: off + (wi + 1) * T] for wi in range(W)]
    off += W * T
    xr = [vals[off + i * T: off + (i + 1) * T] for i in range(R)]
    return build_output(data, xp, xw, xr)


def set_mip_start(model, data, sol, xp, sp, yp, xw, sw, yw, xr, sr, yr):
    T = data["T"]
    W = data["W"]
    R = data["R"]
    for k in range(T):
        xp[k].Start = sol["x_plant"][k]
        sp[k].Start = sol["s_plant"][k]
        yp[k].Start = sol["y_plant"][k]
    for wi in range(W):
        for k in range(T):
            xw[wi][k].Start = sol["x_warehouses"][wi][k]
            sw[wi][k].Start = sol["s_warehouses"][wi][k]
            yw[wi][k].Start = sol["y_warehouses"][wi][k]
    for i in range(R):
        for k in range(T):
            xr[i][k].Start = sol["x_retailers"][i][k]
            sr[i][k].Start = sol["s_retailers"][i][k]
            yr[i][k].Start = sol["y_retailers"][i][k]


def improve_with_gurobi(data, best, logger, start_time, time_limit):
    if gp is None or GRB is None:
        return
    if data["T"] == 0:
        return

    deadline = start_time + float(time_limit)
    if time.monotonic() >= deadline - 0.02:
        return

    R = data["R"]
    W = data["W"]
    T = data["T"]
    sets = data["warehouse_sets"]
    d = data["demands"]
    fixed = data["fixed"]
    holding = data["holding"]
    rem_r = data["rem_retailers"]
    rem_w = data["rem_warehouses"]
    rem_p = data["rem_plant"]

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError

    model = gp.Model("three_echelon_lot_sizing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.ModelSense = GRB.MINIMIZE

    xp = model.addVars(T, lb=0.0)
    sp = model.addVars(T, lb=0.0, obj=holding["plant"])
    yp = model.addVars(T, vtype=GRB.BINARY, obj=fixed["plant"])

    xw, sw, yw = [], [], []
    for wi in range(W):
        xw.append(model.addVars(T, lb=0.0))
        sw.append(model.addVars(T, lb=0.0, obj=holding["warehouses"]))
        yw.append(model.addVars(T, vtype=GRB.BINARY, obj=fixed["warehouses"][wi]))

    xr, sr, yr = [], [], []
    for i in range(R):
        xr.append(model.addVars(T, lb=0.0))
        sr.append(model.addVars(T, lb=0.0, obj=holding["retailers"][i]))
        yr.append(model.addVars(T, vtype=GRB.BINARY, obj=fixed["retailers"][i]))

    for k in range(T):
        if (k & 1023) == 0:
            check_deadline()
        prev = sp[k - 1] if k > 0 else 0.0
        model.addConstr(prev + xp[k] - sp[k] == gp.quicksum(xw[wi][k] for wi in range(W)))
        model.addConstr(xp[k] <= rem_p[k] * yp[k])

    for wi in range(W):
        check_deadline()
        rows = sets[wi]
        for k in range(T):
            prev = sw[wi][k - 1] if k > 0 else 0.0
            model.addConstr(prev + xw[wi][k] - sw[wi][k] == gp.quicksum(xr[i][k] for i in rows))
            model.addConstr(xw[wi][k] <= rem_w[wi][k] * yw[wi][k])

    for i in range(R):
        check_deadline()
        di = d[i]
        for k in range(T):
            prev = sr[i][k - 1] if k > 0 else 0.0
            model.addConstr(prev + xr[i][k] - sr[i][k] == di[k])
            model.addConstr(xr[i][k] <= rem_r[i][k] * yr[i][k])

    if data["nonnegative_costs"]:
        model.addConstr(sp[T - 1] == 0.0)
        for wi in range(W):
            model.addConstr(sw[wi][T - 1] == 0.0)
        for i in range(R):
            model.addConstr(sr[i][T - 1] == 0.0)

    set_mip_start(model, data, best["sol"], xp, sp, yp, xw, sw, yw, xr, sr, yr)

    xp_list = [xp[k] for k in range(T)]
    xw_list = [xw[wi][k] for wi in range(W) for k in range(T)]
    xr_list = [xr[i][k] for i in range(R) for k in range(T)]
    all_x_vars = xp_list + xw_list + xr_list

    remaining = deadline - time.monotonic() - 0.05
    if remaining <= 0.0:
        return
    model.Params.TimeLimit = max(0.0, remaining)

    def callback(m, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        except Exception:
            return
        if incumbent_obj >= best["obj"] - IMPROVE_EPS:
            return
        try:
            vals = m.cbGetSolution(all_x_vars)
            obj, sol = solution_from_flat_values(data, vals)
        except Exception:
            return
        if obj < best["obj"] - IMPROVE_EPS:
            best["obj"] = obj
            best["sol"] = sol
            if logger:
                logger.log_solution(obj, sol)

    model.optimize(callback)

    try:
        if model.SolCount > 0 and model.ObjVal < best["obj"] - IMPROVE_EPS:
            vals = [v.X for v in all_x_vars]
            obj, sol = solution_from_flat_values(data, vals)
            if obj < best["obj"] - IMPROVE_EPS:
                best["obj"] = obj
                best["sol"] = sol
                if logger:
                    logger.log_solution(obj, sol)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    data = prepare_instance(inst)

    best = initial_best_solution(data, logger)

    if args.time_limit > 0:
        try:
            improve_with_gurobi(data, best, logger, start_time, args.time_limit)
        except TimeoutError:
            pass
        except Exception:
            pass

    out_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(best["sol"], f)


if __name__ == "__main__":
    main()