import argparse
import json
import math
import time

from solution_logger import SolutionLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def _float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def prepare(raw):
    gp = raw.get("global_parameters", {}) or {}
    kappa2 = _float(gp.get("kappa_2", 0.0))
    kappa3 = _float(gp.get("kappa_3", 0.0))
    kappa4 = _float(gp.get("kappa_4", 0.0))

    patients = raw.get("patients", []) or []
    P = len(patients)
    patient_ids = [int(p.get("patient_id", i)) for i, p in enumerate(patients)]
    mandatory = []
    nonmandatory = []
    is_mand = []
    sched = {}
    c_unsched = [0.0] * P
    c_cancel = [0.0] * P
    rho = [0.0] * P

    for i, p in enumerate(patients):
        mand = bool(p.get("is_mandatory", False))
        is_mand.append(mand)
        if mand:
            mandatory.append(i)
        else:
            nonmandatory.append(i)
        rho[i] = _float(p.get("rho_p", 0.0))
        c_unsched[i] = _float(p.get("c_unsched", kappa2))
        c_cancel[i] = _float(
            p.get("c_cancel", kappa4 if mand else kappa3)
        )

    hospitals = raw.get("hospitals", []) or []
    global_k = int(raw.get("num_ors_per_hospital", 0) or 0)
    h_ids = []
    K = []
    B = {}
    F = {}
    G = {}
    hd_list = []
    day_label = {}

    for h, hosp in enumerate(hospitals):
        h_id = int(hosp.get("hospital_id", h))
        h_ids.append(h_id)
        kh = int(hosp.get("num_ors", global_k) or 0)
        K.append(kh)
        days = hosp.get("days", []) or []
        for idx, day in enumerate(days):
            d = int(day.get("day_id", idx))
            dl = int(day.get("day_label", d + 1))
            day_label[d] = dl
            hd_list.append((h, d))
            B[(h, d)] = _float(day.get("B_hd", 0.0))
            F[(h, d)] = _float(day.get("F_hd", 0.0))
            G[(h, d)] = _float(day.get("G_hd", 0.0))

    day_ids = sorted({d for _, d in hd_list})
    if not day_ids and int(raw.get("num_days", 0) or 0) > 0:
        day_ids = list(range(int(raw.get("num_days"))))
        for d in day_ids:
            day_label.setdefault(d, d + 1)

    for i, p in enumerate(patients):
        day_costs = p.get("c_sched_per_day", {}) or {}
        for d in day_ids:
            dl = day_label.get(d, d + 1)
            val = 0.0
            if str(dl) in day_costs:
                val = day_costs[str(dl)]
            elif str(d) in day_costs:
                val = day_costs[str(d)]
            elif dl in day_costs:
                val = day_costs[dl]
            elif d in day_costs:
                val = day_costs[d]
            # Per the prompt, c_sched_per_day is already a signed cost coefficient:
            # negative scheduling benefits reduce the minimization objective.
            sched[(i, d)] = _float(val, 0.0)

    scenarios = raw.get("scenarios", []) or []
    durations = []
    for s, sc in enumerate(scenarios):
        arr = sc.get("surgery_durations_minutes", []) or []
        row = [_float(arr[i], 0.0) for i in range(P)] if len(arr) >= P else [
            _float(arr[i], 0.0) if i < len(arr) else 0.0 for i in range(P)
        ]
        durations.append(row)

    return {
        "P": P,
        "patient_ids": patient_ids,
        "is_mand": is_mand,
        "mandatory": mandatory,
        "nonmandatory": nonmandatory,
        "rho": rho,
        "sched": sched,
        "c_unsched": c_unsched,
        "c_cancel": c_cancel,
        "hospitals": hospitals,
        "h_ids": h_ids,
        "K": K,
        "B": B,
        "F": F,
        "G": G,
        "hd_list": hd_list,
        "day_ids": day_ids,
        "day_label": day_label,
        "durations": durations,
        "S": len(durations),
    }


def room_min_cancel_cost(data, idxs, s, Bcap):
    if not idxs:
        return 0.0
    cancel = data["c_cancel"]
    dur = data["durations"][s]
    total = sum(cancel[i] for i in idxs)
    n = len(idxs)
    tol = 1e-9

    if n == 1:
        i = idxs[0]
        return cancel[i] if dur[i] > Bcap + tol else 0.0

    # Exact only for tiny rooms; fallback use only. The MIP below solves recourse exactly.
    if n <= 16:
        best_saved = 0.0
        for mask in range(1, 1 << n):
            sd = 0.0
            sv = 0.0
            for j, i in enumerate(idxs):
                if (mask >> j) & 1:
                    sd += dur[i]
                    sv += cancel[i]
                    if sd > Bcap + tol and sv <= best_saved:
                        # Cannot become best by adding more positive duration/value in this branch.
                        pass
            if sd <= Bcap + tol and sv > best_saved:
                best_saved = sv
        return total - best_saved

    def greedy(order):
        left = Bcap
        saved = 0.0
        for i in order:
            if dur[i] <= left + tol:
                left -= dur[i]
                saved += cancel[i]
        return saved

    by_ratio = sorted(idxs, key=lambda i: (cancel[i] / max(dur[i], 1e-9), cancel[i]), reverse=True)
    by_value = sorted(idxs, key=lambda i: cancel[i], reverse=True)
    by_short = sorted(idxs, key=lambda i: (dur[i], -cancel[i]))
    best = max(greedy(by_ratio), greedy(by_value), greedy(by_short))
    return total - best


def empty_solution(data, obj=math.inf):
    u = {}
    for h, d in data["hd_list"]:
        u[f"{data['h_ids'][h]},{d}"] = 0
    y = {}
    for h, d in data["hd_list"]:
        hid = data["h_ids"][h]
        for k in range(data["K"][h]):
            y[f"{hid},{d},{k}"] = 0
    w = {str(data["patient_ids"][i]): 0 for i in data["nonmandatory"]}
    return {"objective_value": float(obj), "u": u, "y": y, "x": {}, "w": w}


def fallback_solution(data):
    sol = empty_solution(data, math.inf)
    if data["P"] == 0:
        sol["objective_value"] = 0.0
        return sol

    day_ids = data["day_ids"] or [0]
    hd_order = sorted(
        data["hd_list"],
        key=lambda hd: (
            data["G"][hd] + data["F"][hd],
            data["day_label"].get(hd[1], hd[1] + 1),
            data["h_ids"][hd[0]],
            hd[1],
        ),
    )
    if not hd_order:
        return sol

    mandatory = sorted(data["mandatory"], key=lambda i: (data["rho"][i], -i), reverse=True)
    need = len(mandatory)
    slots = []
    for h, d in hd_order:
        if len(slots) >= need:
            break
        take = min(data["K"][h], max(0, need - len(slots)))
        for k in range(take):
            slots.append((h, d, k))

    # If there were no available slots but mandatory patients exist, still create one syntactic slot.
    if need > 0 and not slots:
        h, d = hd_order[0]
        if data["K"][h] > 0:
            slots.append((h, d, 0))

    u_set = {(h, d) for h, d, _ in slots}
    y_set = set(slots)
    remaining = list(slots)
    assigned = set()
    x_assign = []

    def best_slot_for(i, pool):
        return min(
            pool,
            key=lambda sl: (
                data["sched"].get((i, sl[1]), 0.0),
                data["day_label"].get(sl[1], sl[1] + 1),
                data["h_ids"][sl[0]],
                sl[2],
            ),
        )

    for i in mandatory:
        if remaining:
            sl = best_slot_for(i, remaining)
            remaining.remove(sl)
            h, d, k = sl
            x_assign.append((h, d, i, k))
            assigned.add(i)

    # Share rooms only if the horizon has fewer OR slots than mandatory patients.
    for i in mandatory:
        if i not in assigned:
            if slots:
                sl = best_slot_for(i, slots)
            else:
                sl = slots[0]
            h, d, k = sl
            x_assign.append((h, d, i, k))
            assigned.add(i)

    # Use leftover opened slots for non-mandatory patients when that beats postponement.
    for i in data["nonmandatory"]:
        if remaining:
            sl = best_slot_for(i, remaining)
            sched_cost = data["sched"].get((i, sl[1]), 0.0)
            if sched_cost < data["c_unsched"][i] - 1e-12:
                remaining.remove(sl)
                h, d, k = sl
                x_assign.append((h, d, i, k))
                assigned.add(i)

    u_set = {(h, d) for h, d, _ in slots} | {(h, d) for h, d, _, _ in x_assign}
    y_set = set(slots) | {(h, d, k) for h, d, _, k in x_assign}

    for h, d in data["hd_list"]:
        sol["u"][f"{data['h_ids'][h]},{d}"] = 1 if (h, d) in u_set else 0
    for h, d in data["hd_list"]:
        for k in range(data["K"][h]):
            sol["y"][f"{data['h_ids'][h]},{d},{k}"] = 1 if (h, d, k) in y_set else 0
    for h, d, i, k in x_assign:
        sol["x"][f"{data['h_ids'][h]},{d},{data['patient_ids'][i]},{k}"] = 1
    for i in data["nonmandatory"]:
        sol["w"][str(data["patient_ids"][i])] = 0 if i in assigned else 1

    first_stage = sum(data["G"][(h, d)] for h, d in u_set)
    first_stage += sum(data["F"][(h, d)] for h, d, _ in y_set)
    first_stage += sum(data["sched"].get((i, d), 0.0) for h, d, i, _ in x_assign)
    first_stage += sum(data["c_unsched"][i] for i in data["nonmandatory"] if i not in assigned)

    by_room = {}
    for h, d, i, k in x_assign:
        by_room.setdefault((h, d, k), []).append(i)

    S = max(1, data["S"])
    recourse = 0.0
    for s in range(data["S"]):
        for room, idxs in by_room.items():
            h, d, k = room
            recourse += room_min_cancel_cost(data, idxs, s, data["B"][(h, d)]) / S

    sol["objective_value"] = float(first_stage + recourse)
    return sol


def solve_with_gurobi(data, args, start_time, logger, fallback_obj, fallback_sol, last_logged):
    elapsed = time.time() - start_time
    remaining = float(args.time_limit) - elapsed
    if remaining <= 0.75:
        return fallback_obj, fallback_sol, False

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return fallback_obj, fallback_sol, False

    hd_list = data["hd_list"]
    P = data["P"]
    S = data["S"]
    room_keys = [(h, d, k) for h, d in hd_list for k in range(data["K"][h])]
    x_keys = [(h, d, i, k) for h, d, k in room_keys for i in range(P)]

    model = gp.Model("collaborative_hospital_or_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.0, remaining - 0.25)

    u_obj = {(h, d): data["G"][(h, d)] for h, d in hd_list}
    y_obj = {(h, d, k): data["F"][(h, d)] for h, d, k in room_keys}
    x_obj = {(h, d, i, k): data["sched"].get((i, d), 0.0) for h, d, i, k in x_keys}
    w_obj = {i: data["c_unsched"][i] for i in data["nonmandatory"]}

    u = model.addVars(hd_list, vtype=GRB.BINARY, obj=u_obj, name="u")
    y = model.addVars(room_keys, vtype=GRB.BINARY, obj=y_obj, name="y")
    x = model.addVars(x_keys, vtype=GRB.BINARY, obj=x_obj, name="x")
    w = model.addVars(data["nonmandatory"], vtype=GRB.BINARY, obj=w_obj, name="w")

    z = None
    if S > 0:
        denom = float(S)
        z_obj = {
            (h, d, i, k, s): data["c_cancel"][i] / denom
            for (h, d, i, k) in x_keys
            for s in range(S)
        }
        z = model.addVars(z_obj.keys(), vtype=GRB.BINARY, obj=z_obj, name="cancel")

    x_by_patient = [[] for _ in range(P)]
    for key in x_keys:
        x_by_patient[key[2]].append(key)

    for i in data["mandatory"]:
        model.addConstr(gp.quicksum(x[key] for key in x_by_patient[i]) == 1)
    for i in data["nonmandatory"]:
        model.addConstr(gp.quicksum(x[key] for key in x_by_patient[i]) + w[i] == 1)

    model.addConstrs((y[h, d, k] <= u[h, d] for h, d, k in room_keys), name="or_needs_suite")
    model.addConstrs((x[key] <= y[(key[0], key[1], key[3])] for key in x_keys), name="patient_needs_open_or")

    if S > 0:
        model.addConstrs(
            (
                z[(key[0], key[1], key[2], key[3], s)] <= x[key]
                for key in x_keys
                for s in range(S)
            ),
            name="cancel_only_if_assigned",
        )
        durations = data["durations"]
        for h, d, k in room_keys:
            Bcap = data["B"][(h, d)]
            yv = y[(h, d, k)]
            for s in range(S):
                ds = durations[s]
                expr = gp.quicksum(ds[i] * x[(h, d, i, k)] for i in range(P))
                expr -= gp.quicksum(ds[i] * z[(h, d, i, k, s)] for i in range(P))
                model.addConstr(expr <= Bcap * yv, name=f"capacity_{h}_{d}_{k}_{s}")

    def get_val(v, cb):
        if cb:
            return model.cbGetSolution(v)
        return v.X

    def build_solution(obj, cb=False):
        out = {"objective_value": float(obj), "u": {}, "y": {}, "x": {}, "w": {}}
        for h, d in hd_list:
            hid = data["h_ids"][h]
            out["u"][f"{hid},{d}"] = int(round(get_val(u[(h, d)], cb)))
        for h, d, k in room_keys:
            hid = data["h_ids"][h]
            out["y"][f"{hid},{d},{k}"] = int(round(get_val(y[(h, d, k)], cb)))
        for h, d, i, k in x_keys:
            if get_val(x[(h, d, i, k)], cb) > 0.5:
                hid = data["h_ids"][h]
                pid = data["patient_ids"][i]
                out["x"][f"{hid},{d},{pid},{k}"] = 1
        for i in data["nonmandatory"]:
            out["w"][str(data["patient_ids"][i])] = int(round(get_val(w[i], cb)))
        return out

    def callback(m, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
            SOLCNT = getattr(GRB.Callback, "MIPSOL_SOLCNT", None)
            if SOLCNT is not None and m.cbGet(SOLCNT) <= 0:
                return
            if obj < last_logged[0] - 1e-7:
                sol = build_solution(obj, cb=True)
                logger.log_solution(obj, sol)
                last_logged[0] = obj
        except Exception:
            try:
                obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj < last_logged[0] - 1e-7:
                    logger.log(obj)
            except Exception:
                pass

    try:
        model.optimize(callback if logger is not None else None)
    except Exception:
        return fallback_obj, fallback_sol, False

    try:
        if model.SolCount > 0:
            obj = float(model.ObjVal)
            sol = build_solution(obj, cb=False)
            return obj, sol, True
    except Exception:
        pass
    return fallback_obj, fallback_sol, False


def main():
    args = parse_args()
    start_time = time.time()
    raw = load_instance(args.instance_path)
    data = prepare(raw)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    fallback_sol = fallback_solution(data)
    fallback_obj = float(fallback_sol["objective_value"])
    last_logged = [math.inf]
    if logger is not None:
        logger.log_solution(fallback_obj, fallback_sol)
        last_logged[0] = fallback_obj

    obj, sol, used_gurobi = solve_with_gurobi(
        data, args, start_time, logger, fallback_obj, fallback_sol, last_logged
    )

    final_obj = obj if used_gurobi and obj < fallback_obj - 1e-9 else fallback_obj
    final_sol = sol if used_gurobi and obj < fallback_obj - 1e-9 else fallback_sol

    if logger is not None and final_obj < last_logged[0] - 1e-7:
        logger.log_solution(final_obj, final_sol)
        last_logged[0] = final_obj

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()