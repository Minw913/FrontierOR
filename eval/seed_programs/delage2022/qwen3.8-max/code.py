import argparse
import json
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    class SolutionLogger:
        def __init__(self, path=None, sense="minimize"):
            self.path = path
            self.sense = sense

        def log_solution(self, objective_value, solution):
            pass

        def log(self, objective_value):
            pass

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def ceildiv(a, b):
    if b <= 0:
        return 0
    return -(-int(a) // int(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    products = instance.get("products", [])
    all_ids = [int(p["id"]) for p in products]

    demand = {}
    length = {}
    N_orig = {}

    for p in products:
        pid = int(p["id"])
        demand[pid] = int(p.get("demand", 0))
        length[pid] = float(p.get("length", 0))
        N_orig[pid] = int(p.get("N_i", 0))

    L = float(instance.get("machine_length", 0))
    K = int(instance.get("num_patterns", 0))

    max_demand = max(demand.values(), default=0)
    if "M" in instance:
        try:
            M = int(instance["M"])
        except Exception:
            M = max_demand
        if M < 0:
            M = 0
        if M == 0 and max_demand > 0:
            M = max_demand
    else:
        M = max_demand

    problem_type = instance.get("problem_type", "closed_ended_cscp")
    forced_zero = bool(instance.get("x_i_forced_zero", False))
    closed = (problem_type == "closed_ended_cscp") or forced_zero
    open_allowed = not closed
    knife_needed = (problem_type == "cscp_with_knife_dependent_setups")

    setup_cost = float(instance.get("cost_setup_pattern", 0.0))
    rep_cost = float(instance.get("cost_repetition", 0.0))
    knife_cost = float(instance.get("cost_knife_setup", 0.0)) if knife_needed else 0.0

    active_ids = [pid for pid in all_ids if demand[pid] > 0]

    def max_pieces(pid):
        if demand[pid] <= 0:
            return 0
        ni = N_orig.get(pid, 0)
        if ni <= 0:
            return 0
        l = length.get(pid, 0.0)
        if l <= 0:
            fit = ni
        elif L <= 0 or l > L:
            fit = 0
        else:
            fit = int(L // l)
        return max(0, min(ni, demand[pid], fit))

    def normalize_patterns(patterns):
        norm = []

        for p in (patterns or [])[:K]:
            try:
                r = int(p.get("repetitions", 0))
            except Exception:
                r = 0

            if r < 0:
                r = 0
            if r > M:
                r = M

            pieces = {pid: 0 for pid in all_ids}

            if r > 0:
                raw_pieces = p.get("pieces", {}) or {}
                for pid, val in raw_pieces.items():
                    try:
                        pid_int = int(pid)
                    except Exception:
                        continue

                    if pid_int not in pieces:
                        continue

                    try:
                        iv = int(val)
                    except Exception:
                        iv = 0

                    if iv < 0:
                        iv = 0

                    mp = max_pieces(pid_int)
                    if iv > mp:
                        iv = mp

                    pieces[pid_int] = iv

            norm.append({"repetitions": r, "pieces": pieces})

        while len(norm) < K:
            norm.append({"repetitions": 0, "pieces": {pid: 0 for pid in all_ids}})

        norm = norm[:K]
        norm.sort(key=lambda x: (x["repetitions"], sum(x["pieces"].values())), reverse=True)

        for p in norm:
            if p["repetitions"] == 0:
                for pid in all_ids:
                    p["pieces"][pid] = 0

        return norm

    def objective_from_patterns(patterns):
        used = 0
        reps = 0
        knives = 0

        for p in patterns:
            r = int(p.get("repetitions", 0))
            if r > 0:
                used += 1
                reps += r
                if knife_needed:
                    knives += sum(int(v) for v in p.get("pieces", {}).values())

        obj = setup_cost * used + rep_cost * reps
        if knife_needed:
            obj += knife_cost * knives

        return float(obj)

    def build_output(patterns, extra):
        patterns_out = {}
        prod_out = {}

        obj = objective_from_patterns(patterns)

        for j, p in enumerate(patterns):
            r = int(p.get("repetitions", 0))
            pieces_out = {}
            knife_count = 0

            for pid in all_ids:
                val = int(p["pieces"].get(pid, 0))
                pieces_out[str(pid)] = val
                knife_count += val
                prod_out[f"product_{pid}_pattern_{j}"] = val * r

            pat = {
                "used": bool(r > 0),
                "repetitions": int(r),
                "pieces": pieces_out,
            }

            if knife_needed:
                pat["knife_count"] = int(knife_count)

            patterns_out[str(j)] = pat

        extra_out = {str(pid): int(extra.get(pid, 0)) for pid in all_ids}

        return {
            "objective_value": float(obj),
            "patterns": patterns_out,
            "extra_pieces": extra_out,
            "production_quantities": prod_out,
        }

    def finalize_patterns(patterns):
        norm = normalize_patterns(patterns)
        feasible = True

        prod = {pid: 0 for pid in all_ids}
        total_r = 0
        prev = M + 1

        for p in norm:
            r = int(p["repetitions"])
            total_r += r

            if r < 0 or r > M:
                feasible = False

            if r > prev:
                feasible = False
            prev = r

            lsum = 0.0

            for pid in all_ids:
                val = int(p["pieces"].get(pid, 0))

                if val < 0 or val > N_orig.get(pid, 0):
                    feasible = False

                if val > 0:
                    if r == 0:
                        feasible = False

                    lsum += float(length.get(pid, 0.0)) * val
                    prod[pid] += val * r

            if lsum > L + 1e-6:
                feasible = False

        extra = {pid: 0 for pid in all_ids}

        if open_allowed:
            total_e = 0

            for pid in all_ids:
                need = demand[pid] - prod[pid]
                if need < 0:
                    need = 0
                if need > demand[pid]:
                    need = demand[pid]

                extra[pid] = int(need)
                total_e += need

            if total_e > total_r:
                feasible = False
        else:
            for pid in all_ids:
                if prod[pid] < demand[pid]:
                    feasible = False

        sol = build_output(norm, extra)
        return sol, feasible, norm, extra

    def heuristic_dedicated():
        fittable = []
        unfitted = []

        for pid in active_ids:
            if max_pieces(pid) > 0:
                fittable.append(pid)
            else:
                unfitted.append(pid)

        if len(fittable) > K:
            return None

        patterns = []
        total_r = 0
        total_e = 0

        for pid in fittable:
            a = max_pieces(pid)
            d = demand[pid]

            if a <= 0:
                return None

            if open_allowed:
                r = ceildiv(d, a + 1)
            else:
                r = ceildiv(d, a)

            if r > M:
                r = M

            if r <= 0:
                return None

            prod = a * r
            e = max(0, d - prod) if open_allowed else 0

            if open_allowed:
                while e > r and r < M:
                    r += 1
                    prod = a * r
                    e = max(0, d - prod)

                if e > r:
                    return None

                total_e += e
            else:
                if prod < d:
                    return None

            patterns.append({"repetitions": r, "pieces": {pid: a}})
            total_r += r

        if unfitted:
            if not open_allowed:
                return None

            need = sum(demand[pid] for pid in unfitted)
            capacity = total_r - total_e

            idx = 0
            while capacity < need and idx < len(patterns):
                p = patterns[idx]
                add = min(M - p["repetitions"], need - capacity)

                if add > 0:
                    p["repetitions"] += add
                    capacity += add
                    total_r += add

                idx += 1

            while capacity < need and len(patterns) < K:
                add = min(M, need - capacity)
                if add <= 0:
                    break

                patterns.append({"repetitions": add, "pieces": {}})
                capacity += add
                total_r += add

            if capacity < need:
                return None

        return patterns

    def heuristic_partial_dedicated():
        if not open_allowed:
            return None

        fittable = [pid for pid in active_ids if max_pieces(pid) > 0]
        fittable.sort(key=lambda x: demand[x], reverse=True)

        max_num = min(K, len(fittable))
        tries = {max_num, max(0, max_num - 1), max(0, max_num - 2), 0}

        best_patterns = None
        best_obj = float("inf")

        for num in tries:
            if num < 0 or num > max_num:
                continue

            chosen = fittable[:num]
            patterns = []
            total_r = 0
            total_e = 0
            ok = True

            for pid in chosen:
                a = max_pieces(pid)
                d = demand[pid]

                if a <= 0:
                    ok = False
                    break

                r = ceildiv(d, a + 1)
                if r > M:
                    r = M

                prod = a * r
                e = max(0, d - prod)

                while e > r and r < M:
                    r += 1
                    prod = a * r
                    e = max(0, d - prod)

                if e > r:
                    ok = False
                    break

                patterns.append({"repetitions": r, "pieces": {pid: a}})
                total_r += r
                total_e += e

            if not ok:
                continue

            chosen_set = set(chosen)
            remaining = [pid for pid in active_ids if pid not in chosen_set]
            need = sum(demand[pid] for pid in remaining)
            capacity = total_r - total_e

            for p in patterns:
                if capacity >= need:
                    break

                add = min(M - p["repetitions"], need - capacity)
                if add > 0:
                    p["repetitions"] += add
                    capacity += add

            while capacity < need and len(patterns) < K:
                add = min(M, need - capacity)
                if add <= 0:
                    break

                patterns.append({"repetitions": add, "pieces": {}})
                capacity += add

            if capacity < need:
                continue

            sol, feasible, pnorm, _ = finalize_patterns(patterns)

            if feasible:
                obj = sol["objective_value"]
                if obj < best_obj:
                    best_obj = obj
                    best_patterns = pnorm

        return best_patterns

    def heuristic_binpack(sort_type):
        items = []
        unfitted = []

        for pid in active_ids:
            if max_pieces(pid) >= 1:
                items.append(pid)
            else:
                unfitted.append(pid)

        if sort_type == "length":
            items.sort(key=lambda x: length[x], reverse=True)
        elif sort_type == "demand":
            items.sort(key=lambda x: demand[x], reverse=True)
        else:
            items.sort(
                key=lambda x: demand[x] / (length[x] if length[x] > 0 else 1.0),
                reverse=True,
            )

        bins = []
        unassigned = []

        for pid in items:
            l = length[pid]

            if l > L + 1e-9:
                unassigned.append(pid)
                continue

            best_bin = None
            best_rem = None

            for b in bins:
                rem = L - b["length"]

                if rem + 1e-9 >= l:
                    new_rem = rem - l
                    if best_rem is None or new_rem < best_rem:
                        best_bin = b
                        best_rem = new_rem

            if best_bin is not None:
                best_bin["length"] += l
                best_bin["pieces"][pid] = best_bin["pieces"].get(pid, 0) + 1

                if demand[pid] > best_bin["max_demand"]:
                    best_bin["max_demand"] = demand[pid]

            elif len(bins) < K:
                bins.append({"length": l, "pieces": {pid: 1}, "max_demand": demand[pid]})

            else:
                unassigned.append(pid)

        need_ids = unassigned + unfitted

        if (not open_allowed) and need_ids:
            return None

        patterns = []
        total_r = 0

        for b in bins:
            if b["pieces"]:
                r = min(M, b["max_demand"])
                if r > 0:
                    patterns.append({"repetitions": r, "pieces": b["pieces"]})
                    total_r += r

        if open_allowed:
            need = sum(demand[pid] for pid in need_ids)
            capacity = total_r

            for p in patterns:
                if capacity >= need:
                    break

                add = min(M - p["repetitions"], need - capacity)
                if add > 0:
                    p["repetitions"] += add
                    capacity += add

            while capacity < need and len(patterns) < K:
                add = min(M, need - capacity)
                if add <= 0:
                    break

                patterns.append({"repetitions": add, "pieces": {}})
                capacity += add

            if capacity < need:
                return None

        return patterns

    def fallback_patterns():
        if open_allowed and K > 0 and M > 0:
            need = sum(demand[pid] for pid in all_ids)
            patterns = []
            rem = need

            for _ in range(K):
                rv = min(M, rem)
                if rv < 0:
                    rv = 0

                patterns.append({"repetitions": rv, "pieces": {}})
                rem -= rv

                if rem <= 0:
                    break

            sol, feasible, pnorm, extra_norm = finalize_patterns(patterns)

            if feasible:
                return pnorm, extra_norm, sol

        patterns = [{"repetitions": 0, "pieces": {}} for _ in range(K)]
        sol, feasible, pnorm, extra_norm = finalize_patterns(patterns)

        return pnorm, extra_norm, sol

    if not active_ids:
        patterns = [{"repetitions": 0, "pieces": {}} for _ in range(K)]
        sol, _, _, _ = finalize_patterns(patterns)

        if logger:
            logger.log_solution(sol["objective_value"], sol)

        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)

        return

    candidates = []

    def add_candidate(raw):
        if raw is None:
            return

        sol, feasible, pnorm, extra_norm = finalize_patterns(raw)

        if feasible:
            candidates.append((sol["objective_value"], pnorm, extra_norm, sol))

    add_candidate(heuristic_dedicated())
    add_candidate(heuristic_partial_dedicated())

    for st in ("length", "demand", "ratio"):
        add_candidate(heuristic_binpack(st))

    best_state = {
        "obj": float("inf"),
        "sol": None,
        "patterns": None,
        "extra": None,
    }

    for obj, pnorm, extra_norm, sol in candidates:
        if obj < best_state["obj"] - 1e-9:
            best_state["obj"] = obj
            best_state["sol"] = sol
            best_state["patterns"] = pnorm
            best_state["extra"] = extra_norm

            if logger:
                logger.log_solution(obj, sol)

    if best_state["sol"] is None:
        pnorm, extra_norm, sol = fallback_patterns()

        best_state["obj"] = sol["objective_value"]
        best_state["sol"] = sol
        best_state["patterns"] = pnorm
        best_state["extra"] = extra_norm

        if logger:
            logger.log_solution(best_state["obj"], sol)

    elapsed = time.time() - start_time
    remaining = float(args.time_limit) - elapsed

    I = len(active_ids)
    B = max(1, M.bit_length()) if M > 0 else 0
    estimated_w = I * K * B if (I and K and B) else 0
    MAX_W_VARS = 500000

    if (
        gp is not None
        and GRB is not None
        and K > 0
        and M > 0
        and I > 0
        and remaining > 2.0
        and estimated_w <= MAX_W_VARS
    ):
        try:
            model = gp.Model("cscp")

            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.1, remaining - 1.0)
            model.Params.LogToConsole = 0

            y = model.addVars(K, vtype=GRB.BINARY, name="y")
            r = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, ub=float(M), name="r")
            z = model.addVars(K, B, vtype=GRB.BINARY, name="z")

            a = model.addVars(I, K, vtype=GRB.INTEGER, lb=0, name="a")
            Ni_eff = [max_pieces(active_ids[i]) for i in range(I)]

            for i in range(I):
                for j in range(K):
                    a[i, j].ub = Ni_eff[i]

            w = model.addVars(I, K, B, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
            q = model.addVars(I, K, vtype=GRB.CONTINUOUS, lb=0.0, name="q")

            if open_allowed:
                e = model.addVars(I, vtype=GRB.INTEGER, lb=0, name="e")
                for i, pid in enumerate(active_ids):
                    e[i].ub = demand[pid]
            else:
                e = None

            model.update()

            model.addConstrs(
                (
                    r[j] == gp.quicksum((1 << b) * z[j, b] for b in range(B))
                    for j in range(K)
                ),
                name="r_def",
            )

            model.addConstrs(
                (r[j] <= M * y[j] for j in range(K)),
                name="r_link",
            )

            model.addConstrs(
                (y[j] <= r[j] for j in range(K)),
                name="y_link",
            )

            model.addConstrs(
                (r[j] >= r[j + 1] for j in range(K - 1)),
                name="order",
            )

            model.addConstrs(
                (a[i, j] <= Ni_eff[i] * y[j] for i in range(I) for j in range(K)),
                name="a_link",
            )

            model.addConstrs(
                (
                    gp.quicksum(
                        length[active_ids[i]] * a[i, j]
                        for i in range(I)
                        if length[active_ids[i]] != 0
                    )
                    <= L * y[j]
                    for j in range(K)
                ),
                name="pattern_length",
            )

            model.addConstrs(
                (
                    w[i, j, b] <= a[i, j]
                    for i in range(I)
                    for j in range(K)
                    for b in range(B)
                ),
                name="w_le_a",
            )

            model.addConstrs(
                (
                    w[i, j, b] <= Ni_eff[i] * z[j, b]
                    for i in range(I)
                    for j in range(K)
                    for b in range(B)
                ),
                name="w_le_z",
            )

            model.addConstrs(
                (
                    w[i, j, b] >= a[i, j] - Ni_eff[i] * (1 - z[j, b])
                    for i in range(I)
                    for j in range(K)
                    for b in range(B)
                ),
                name="w_ge",
            )

            model.addConstrs(
                (
                    q[i, j] == gp.quicksum((1 << b) * w[i, j, b] for b in range(B))
                    for i in range(I)
                    for j in range(K)
                ),
                name="q_def",
            )

            model.addConstrs(
                (
                    q[i, j] <= Ni_eff[i] * r[j]
                    for i in range(I)
                    for j in range(K)
                ),
                name="q_cap",
            )

            model.addConstrs(
                (
                    q[i, j] >= a[i, j]
                    for i in range(I)
                    for j in range(K)
                ),
                name="q_min",
            )

            model.addConstrs(
                (
                    gp.quicksum(
                        length[active_ids[i]] * q[i, j]
                        for i in range(I)
                        if length[active_ids[i]] != 0
                    )
                    <= L * r[j]
                    for j in range(K)
                ),
                name="prod_length",
            )

            if open_allowed:
                model.addConstrs(
                    (
                        gp.quicksum(q[i, j] for j in range(K)) + e[i]
                        >= demand[active_ids[i]]
                        for i in range(I)
                    ),
                    name="demand",
                )

                model.addConstr(
                    gp.quicksum(e[i] for i in range(I))
                    <= gp.quicksum(r[j] for j in range(K)),
                    name="extra_total",
                )
            else:
                model.addConstrs(
                    (
                        gp.quicksum(q[i, j] for j in range(K))
                        >= demand[active_ids[i]]
                        for i in range(I)
                    ),
                    name="demand",
                )

            knife = None
            max_knife = sum(Ni_eff)

            if (knife_needed or K > 1) and max_knife > 0:
                knife = model.addVars(
                    K,
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=float(max_knife),
                    name="knife",
                )

                model.addConstrs(
                    (
                        knife[j] == gp.quicksum(a[i, j] for i in range(I))
                        for j in range(K)
                    ),
                    name="knife_def",
                )

                if K > 1:
                    sort_s = model.addVars(K - 1, vtype=GRB.BINARY, name="sort")

                    model.addConstrs(
                        (
                            r[j] - r[j + 1] <= M * sort_s[j]
                            for j in range(K - 1)
                        ),
                        name="diff_le",
                    )

                    model.addConstrs(
                        (
                            r[j] - r[j + 1] >= sort_s[j]
                            for j in range(K - 1)
                        ),
                        name="diff_ge",
                    )

                    model.addConstrs(
                        (
                            knife[j] >= knife[j + 1] - max_knife * sort_s[j]
                            for j in range(K - 1)
                        ),
                        name="sort_knife",
                    )

            obj_expr = (
                setup_cost * gp.quicksum(y[j] for j in range(K))
                + rep_cost * gp.quicksum(r[j] for j in range(K))
            )

            if knife_needed and knife is not None:
                obj_expr += knife_cost * gp.quicksum(knife[j] for j in range(K))

            model.setObjective(obj_expr, GRB.MINIMIZE)
            model.update()

            if best_state["patterns"] is not None:
                try:
                    for j, p in enumerate(best_state["patterns"]):
                        rv = int(p["repetitions"])

                        y[j].Start = 1 if rv > 0 else 0
                        r[j].Start = float(rv)

                        for b in range(B):
                            z[j, b].Start = (rv >> b) & 1

                        for i, pid in enumerate(active_ids):
                            val = int(p["pieces"].get(pid, 0))
                            if val > Ni_eff[i]:
                                val = Ni_eff[i]
                            a[i, j].Start = val

                        if knife is not None:
                            knife[j].Start = float(
                                sum(int(p["pieces"].get(pid, 0)) for pid in active_ids)
                            )

                    if open_allowed and e is not None and best_state["extra"] is not None:
                        for i, pid in enumerate(active_ids):
                            e[i].Start = int(best_state["extra"].get(pid, 0))

                    model.update()
                except Exception:
                    pass

            r_list = [r[j] for j in range(K)]
            a_flat = [a[i, j] for i in range(I) for j in range(K)]
            e_list = [e[i] for i in range(I)] if (open_allowed and e is not None) else []

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)

                    if incumbent_obj >= best_state["obj"] - 1e-6:
                        return

                    r_raw = cb_model.cbGetSolution(r_list)
                    a_raw = cb_model.cbGetSolution(a_flat)

                    raw_patterns = []

                    for j in range(K):
                        try:
                            rv = int(round(float(r_raw[j])))
                        except Exception:
                            rv = 0

                        if rv < 0:
                            rv = 0
                        if rv > M:
                            rv = M

                        pieces = {pid: 0 for pid in all_ids}

                        if rv > 0:
                            for i, pid in enumerate(active_ids):
                                idx = i * K + j

                                try:
                                    val = int(round(float(a_raw[idx])))
                                except Exception:
                                    val = 0

                                if val < 0:
                                    val = 0
                                if val > Ni_eff[i]:
                                    val = Ni_eff[i]

                                pieces[pid] = val

                        raw_patterns.append({"repetitions": rv, "pieces": pieces})

                    sol, feasible, pnorm, extra_norm = finalize_patterns(raw_patterns)

                    if not feasible:
                        return

                    obj2 = sol["objective_value"]

                    if obj2 < best_state["obj"] - 1e-6:
                        best_state["obj"] = obj2
                        best_state["sol"] = sol
                        best_state["patterns"] = pnorm
                        best_state["extra"] = extra_norm

                        if logger:
                            logger.log_solution(obj2, sol)

                except Exception:
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                try:
                    final_obj = model.ObjVal

                    if final_obj < best_state["obj"] - 1e-6 or best_state["sol"] is None:
                        r_raw = model.getAttr("X", r_list)
                        a_raw = model.getAttr("X", a_flat)

                        raw_patterns = []

                        for j in range(K):
                            rv = int(round(float(r_raw[j])))

                            if rv < 0:
                                rv = 0
                            if rv > M:
                                rv = M

                            pieces = {pid: 0 for pid in all_ids}

                            if rv > 0:
                                for i, pid in enumerate(active_ids):
                                    idx = i * K + j
                                    val = int(round(float(a_raw[idx])))

                                    if val < 0:
                                        val = 0
                                    if val > Ni_eff[i]:
                                        val = Ni_eff[i]

                                    pieces[pid] = val

                            raw_patterns.append({"repetitions": rv, "pieces": pieces})

                        sol, feasible, pnorm, extra_norm = finalize_patterns(raw_patterns)

                        if feasible:
                            obj2 = sol["objective_value"]

                            if obj2 < best_state["obj"] - 1e-6 or best_state["sol"] is None:
                                best_state["obj"] = obj2
                                best_state["sol"] = sol
                                best_state["patterns"] = pnorm
                                best_state["extra"] = extra_norm

                                if logger:
                                    logger.log_solution(obj2, sol)

                except Exception:
                    pass

        except Exception:
            pass

    if best_state["sol"] is None:
        _, _, sol = fallback_patterns()
        best_state["sol"] = sol

        if logger:
            logger.log_solution(sol["objective_value"], sol)

    with open(args.solution_path, "w") as f:
        json.dump(best_state["sol"], f, indent=2)


if __name__ == "__main__":
    main()