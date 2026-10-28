import argparse
import json
import math
import os
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI = True
except Exception:
    GUROBI = False

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def safe_float(x, default=0.0):
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_num(d, names, default=0.0):
    if not isinstance(d, dict):
        return default
    for name in names:
        if name in d:
            return safe_float(d[name], default)
    return default


def get_cost_structure(instance):
    cs = instance.get("cost_structures", {}) or {}
    if not isinstance(cs, dict) or not cs:
        return "Cost1", {"c_w": 0.0, "c_g": 0.0, "c_o": 0.0}
    if "Cost1" in cs:
        name = "Cost1"
    else:
        name = list(cs.keys())[0]
    c = cs.get(name, {}) or {}
    if not isinstance(c, dict):
        c = {}
    return name, c


def get_procedure_types(instance, P):
    types = instance.get("procedure_types", None)
    if isinstance(types, list) and len(types) >= P:
        return [str(types[i]) for i in range(P)]

    patients = instance.get("patients", []) or []
    out = []
    for i in range(P):
        pt = patients[i] if i < len(patients) and isinstance(patients[i], dict) else {}
        out.append(str(pt.get("procedure_type", "C")))
    return out


def extract_features(instance, P):
    patients = instance.get("patients", []) or []
    feats = []

    for idx in range(P):
        pt = patients[idx] if idx < len(patients) and isinstance(patients[idx], dict) else {}

        q = clamp(
            get_num(pt, ["mean_prep_adequacy", "mean_prep_adequacy_p", "prep_adequacy_mean", "mu_q"], 0.5),
            0.0,
            1.0,
        )

        muA = get_num(pt, ["mean_duration_adequate_prep", "mean_duration_adequate", "mu_A"], 0.0)
        muI = get_num(pt, ["mean_duration_inadequate_prep", "mean_duration_inadequate", "mu_I"], 0.0)

        dAL = get_num(pt, ["lower_bound_duration_adequate_prep", "lower_duration_adequate_prep",
                           "lower_bound_duration_adequate", "duration_adequate_lower"], 0.0)
        dAU = get_num(pt, ["upper_bound_duration_adequate_prep", "upper_duration_adequate_prep",
                           "upper_bound_duration_adequate", "duration_adequate_upper"], dAL)

        dIL = get_num(pt, ["lower_bound_duration_inadequate_prep", "lower_duration_inadequate_prep",
                           "lower_bound_duration_inadequate", "duration_inadequate_lower"], 0.0)
        dIU = get_num(pt, ["upper_bound_duration_inadequate_prep", "upper_duration_inadequate_prep",
                           "upper_bound_duration_inadequate", "duration_inadequate_upper"], dIL)

        dAL = max(0.0, dAL)
        dAU = max(dAL, dAU)
        dIL = max(0.0, dIL)
        dIU = max(dIL, dIU)

        muA = clamp(muA, dAL, dAU)
        muI = clamp(muI, dIL, dIU)

        muU = get_num(pt, ["mean_arrival_deviation", "mean_arrival_deviation_u", "mu_u"], 0.0)
        uL = get_num(pt, ["lower_bound_arrival_deviation", "lower_arrival_deviation",
                          "arrival_deviation_lower"], 0.0)
        uU = get_num(pt, ["upper_bound_arrival_deviation", "upper_arrival_deviation",
                          "arrival_deviation_upper"], uL)

        if uL > uU:
            uL, uU = uU, uL
        muU = clamp(muU, uL, uU)

        meanD = q * muA + (1.0 - q) * muI

        dur_atoms = []
        if q > 1e-12:
            if dAU > dAL + 1e-12:
                pA = clamp((muA - dAL) / (dAU - dAL), 0.0, 1.0)
                dur_atoms.append((dAL, q * (1.0 - pA)))
                dur_atoms.append((dAU, q * pA))
            else:
                dur_atoms.append((muA, q))

        if 1.0 - q > 1e-12:
            if dIU > dIL + 1e-12:
                pI = clamp((muI - dIL) / (dIU - dIL), 0.0, 1.0)
                dur_atoms.append((dIL, (1.0 - q) * (1.0 - pI)))
                dur_atoms.append((dIU, (1.0 - q) * pI))
            else:
                dur_atoms.append((muI, 1.0 - q))

        dur_atoms = [(float(v), float(pr)) for v, pr in dur_atoms if pr > 1e-12]
        if not dur_atoms:
            dur_atoms = [(float(meanD), 1.0)]

        if uU > uL + 1e-12:
            pU = clamp((muU - uL) / (uU - uL), 0.0, 1.0)
            arr_atoms = [(uL, 1.0 - pU), (uU, pU)]
        else:
            arr_atoms = [(muU, 1.0)]

        feats.append({
            "q": q,
            "meanD": float(meanD),
            "meanU": float(muU),
            "dur_range": float(max(dAU, dIU) - min(dAL, dIL)),
            "arr_range": float(uU - uL),
            "max_dur": float(max(dAU, dIU, 0.0)),
            "u_U": float(uU),
            "score1": float(meanD + max(0.0, muU)),
            "score2": float((max(dAU, dIU) - min(dAL, dIL)) + (uU - uL)),
            "dur_atoms": dur_atoms,
            "arr_atoms": arr_atoms,
        })

    return feats


def expand_atoms(atoms, N):
    if N <= 0:
        return []
    if not atoms:
        return [0.0] * N

    total = sum(max(0.0, float(pr)) for _, pr in atoms)
    if total <= 1e-12:
        return [float(atoms[0][0])] * N

    probs = [max(0.0, float(pr)) / total for _, pr in atoms]
    counts = [int(p * N) for p in probs]
    rem = N - sum(counts)

    if rem > 0:
        order = sorted(range(len(probs)), key=lambda k: (probs[k] * N - counts[k], probs[k]), reverse=True)
        for j in range(rem):
            counts[order[j % len(order)]] += 1

    vals = []
    for (v, _), c in zip(atoms, counts):
        if c > 0:
            vals.extend([float(v)] * c)

    if len(vals) < N:
        vals.extend([vals[-1] if vals else 0.0] * (N - len(vals)))
    elif len(vals) > N:
        vals = vals[:N]

    return vals


def quantile_atoms(atoms, frac):
    if not atoms:
        return 0.0

    atoms = sorted(atoms, key=lambda x: float(x[0]))
    total = sum(max(0.0, float(pr)) for _, pr in atoms)
    if total <= 1e-12:
        return float(atoms[0][0])

    target = clamp(frac, 0.0, 1.0) * total
    cum = 0.0
    for v, pr in atoms:
        cum += max(0.0, float(pr))
        if cum >= target - 1e-12:
            return float(v)
    return float(atoms[-1][0])


def generate_scenarios(feats, N, seed=0):
    P = len(feats)
    N = max(1, int(N))

    scenarios = []
    groups = [[], [], []]

    mean_dur = [feats[p]["meanD"] for p in range(P)]
    mean_arr = [feats[p]["meanU"] for p in range(P)]
    scenarios.append({"dur": mean_dur, "arr": mean_arr})
    groups[0].append(0)

    rng = random.Random(seed)

    ind_dur = []
    ind_arr = []
    com_dur = []
    com_arr = []

    for p in range(P):
        dur_vals = expand_atoms(feats[p]["dur_atoms"], N)
        arr_vals = expand_atoms(feats[p]["arr_atoms"], N)

        idur = dur_vals[:]
        iarr = arr_vals[:]
        rng.shuffle(idur)
        rng.shuffle(iarr)

        ind_dur.append(idur)
        ind_arr.append(iarr)
        com_dur.append(sorted(dur_vals))
        com_arr.append(sorted(arr_vals))

    for k in range(N):
        scenarios.append({
            "dur": [ind_dur[p][k] for p in range(P)],
            "arr": [ind_arr[p][k] for p in range(P)],
        })
        groups[1].append(len(scenarios) - 1)

    for k in range(N):
        scenarios.append({
            "dur": [com_dur[p][k] for p in range(P)],
            "arr": [com_arr[p][k] for p in range(P)],
        })
        groups[2].append(len(scenarios) - 1)

    return scenarios, groups


def normalize_t(t, L, P=None):
    if P is None:
        P = len(t) if t is not None else 0
    if P <= 0:
        return []

    out = []
    for i in range(P):
        val = safe_float(t[i] if t is not None and i < len(t) else 0.0, 0.0)
        val = clamp(val, 0.0, max(0.0, L))
        if i == 0:
            val = 0.0
        elif val < out[i - 1]:
            val = out[i - 1]
        out.append(float(val))
    return out


def evaluate_schedule(seq, t, scenarios, groups, cw, cg, co, L):
    P = len(seq)
    t = normalize_t(t, L, P)

    group_costs = []
    for g in groups:
        if not g:
            continue
        total = 0.0
        for sidx in g:
            scen = scenarios[sidx]
            C = 0.0
            cost = 0.0
            for i, p in enumerate(seq):
                arr = scen["arr"][p]
                dur = scen["dur"][p]
                w = max(0.0, float(arr), C - t[i])
                C = t[i] + w + float(dur)
                cost += cw * w

                if i < P - 1:
                    gap = t[i + 1] - C
                    if gap > 0.0:
                        cost += cg * gap

            over = C - L
            if over > 0.0:
                cost += co * over

            total += cost

        group_costs.append(total / float(len(g)))

    if not group_costs:
        return 0.0, []
    return max(group_costs), group_costs


def fallback_schedule(seq, feats, cw, cg, co, L):
    P = len(seq)
    if P == 0:
        return []
    if P == 1:
        return [0.0]

    if cw + cg > 1e-12:
        frac = cw / (cw + cg)
    else:
        frac = 0.5

    if co > 1e-12 and cw + co > 1e-12:
        frac *= cw / (cw + co)

    frac = clamp(frac, 0.05, 0.95)

    t = [0.0] * P
    for i in range(P - 1):
        p = seq[i]
        qdur = quantile_atoms(feats[p]["dur_atoms"], frac)
        buffer = qdur + 0.5 * max(0.0, feats[p]["meanU"])
        buffer = max(0.0, buffer)
        t[i + 1] = t[i] + buffer

    return normalize_t(t, L, P)


def ensure_sequence(seq, P):
    if P <= 0:
        return []
    if seq is None or len(seq) != P:
        return list(range(P))
    try:
        seq = [int(x) for x in seq]
    except Exception:
        return list(range(P))
    if len(set(seq)) != P or any(x < 0 or x >= P for x in seq):
        return list(range(P))
    return seq


def generate_candidates(feats, proc_types, max_candidates=80, max_rand=20, seed=0):
    P = len(feats)
    if P == 0:
        return []

    candidates = []
    seen = set()

    def add(seq):
        seq = ensure_sequence(seq, P)
        tup = tuple(seq)
        if tup not in seen:
            seen.add(tup)
            candidates.append(list(tup))

    base = list(range(P))

    key_defs = [
        (lambda p: feats[p]["meanD"], False),
        (lambda p: feats[p]["meanD"], True),
        (lambda p: feats[p]["dur_range"], False),
        (lambda p: feats[p]["dur_range"], True),
        (lambda p: feats[p]["meanU"], False),
        (lambda p: feats[p]["meanU"], True),
        (lambda p: feats[p]["q"], False),
        (lambda p: feats[p]["q"], True),
        (lambda p: feats[p]["score1"], False),
        (lambda p: feats[p]["score2"], False),
        (lambda p: feats[p]["score2"], True),
    ]

    for fn, rev in key_defs:
        add(sorted(base, key=fn, reverse=rev))

    if proc_types is not None and len(proc_types) == P:
        add(sorted(base, key=lambda p: (0 if proc_types[p] == "UC" else 1, feats[p]["meanD"])))
        add(sorted(base, key=lambda p: (0 if proc_types[p] == "C" else 1, feats[p]["meanD"])))
        add(sorted(base, key=lambda p: (0 if proc_types[p] == "UC" else 1, -feats[p]["meanD"])))
        add(sorted(base, key=lambda p: (0 if proc_types[p] == "C" else 1, -feats[p]["meanD"])))

    rng = random.Random(seed + 1)
    for _ in range(max(0, max_rand)):
        seq = base[:]
        rng.shuffle(seq)
        add(seq)

    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates


def choose_N(P, time_limit):
    if time_limit <= 10:
        base = 3
    elif P <= 8:
        base = 20
    elif P <= 12:
        base = 15
    elif P <= 20:
        base = 10
    else:
        base = 6

    if time_limit < 60:
        base = min(base, 8)
    elif time_limit < 180:
        base = min(base, 12)

    return max(3, base)


def set_model_params(model, time_limit=None, mip=False):
    try:
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", 0)
        model.setParam("Threads", 1)
        model.setParam("NumericFocus", 0)
        if mip:
            model.setParam("MIPGap", 1e-4)
        if time_limit is not None:
            model.setParam("TimeLimit", max(0.01, float(time_limit)))
    except Exception:
        pass


def optimize_fixed_sequence(seq, scenarios, groups, cw, cg, co, L, UB, env, time_limit):
    if not GUROBI:
        return None

    P = len(seq)
    seq = ensure_sequence(seq, P)

    if P == 0:
        return (0.0, [])

    if abs(cw) + abs(cg) + abs(co) <= 1e-12:
        return (0.0, [0.0] * P)

    if time_limit is not None and time_limit <= 0.0:
        return None

    model = None
    try:
        if env is not None:
            model = gp.Model(env=env)
        else:
            model = gp.Model()
        set_model_params(model, time_limit, mip=False)

        tvars = [model.addVar(lb=0.0, ub=L, name=f"t[{i + 1}]") for i in range(P)]
        if P > 0:
            tvars[0].lb = 0.0
            tvars[0].ub = 0.0

        model.update()

        for i in range(1, P):
            model.addConstr(tvars[i] >= tvars[i - 1])

        cost_exprs = []
        reg = gp.LinExpr(0.0)

        for scen in scenarios:
            cost = gp.LinExpr(0.0)
            C_prev = None

            for i, p in enumerate(seq):
                W = model.addVar(lb=0.0, ub=UB)
                model.addConstr(W >= float(scen["arr"][p]))
                if i > 0:
                    model.addConstr(W >= C_prev - tvars[i])

                C = model.addVar(lb=0.0, ub=UB)
                model.addConstr(C == tvars[i] + W + float(scen["dur"][p]))

                if cw != 0.0:
                    cost.add(W, cw)
                reg.add(W, 1.0)

                if i < P - 1:
                    G = model.addVar(lb=0.0, ub=UB)
                    model.addConstr(G >= tvars[i + 1] - C)
                    if cg != 0.0:
                        cost.add(G, cg)
                    reg.add(G, 1.0)

                C_prev = C

            O = model.addVar(lb=0.0, ub=UB)
            model.addConstr(O >= C_prev - L)
            if co != 0.0:
                cost.add(O, co)
            reg.add(O, 1.0)

            cost_exprs.append(cost)

        eta = model.addVar(lb=0.0, name="eta")
        for g in groups:
            if not g:
                continue
            expr = gp.quicksum(cost_exprs[s] for s in g) * (1.0 / float(len(g)))
            model.addConstr(eta >= expr)

        model.setObjective(eta + 1e-7 * reg, GRB.MINIMIZE)
        model.optimize()

        if model.SolCount == 0:
            return None

        try:
            t = [v.X for v in tvars]
        except Exception:
            return None

        t = normalize_t(t, L, P)
        obj, _ = evaluate_schedule(seq, t, scenarios, groups, cw, cg, co, L)
        return obj, t

    except Exception:
        return None
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


def extract_permutation(mat):
    P = len(mat)
    unassigned = set(range(P))
    seq = [None] * P

    for i in range(P):
        best_p = None
        best_val = -float("inf")
        for p in list(unassigned):
            val = safe_float(mat[p][i], -1.0)
            if val > best_val:
                best_val = val
                best_p = p
        if best_p is None:
            best_p = list(unassigned)[0]
        seq[i] = best_p
        unassigned.remove(best_p)

    return seq


def solve_joint_mip(scenarios, groups, P, cw, cg, co, L, UB, env, time_limit, warm_seq=None, warm_t=None):
    if not GUROBI or P == 0:
        return None

    if abs(cw) + abs(cg) + abs(co) <= 1e-12:
        return None

    if time_limit is not None and time_limit <= 0.0:
        return None

    model = None
    try:
        if env is not None:
            model = gp.Model(env=env)
        else:
            model = gp.Model()
        set_model_params(model, time_limit, mip=True)

        x = [[model.addVar(vtype=GRB.BINARY, name=f"x[{p + 1},{i + 1}]") for i in range(P)] for p in range(P)]
        tvars = [model.addVar(lb=0.0, ub=L, name=f"t[{i + 1}]") for i in range(P)]
        if P > 0:
            tvars[0].lb = 0.0
            tvars[0].ub = 0.0

        model.update()

        for p in range(P):
            model.addConstr(gp.quicksum(x[p][i] for i in range(P)) == 1)
        for i in range(P):
            model.addConstr(gp.quicksum(x[p][i] for p in range(P)) == 1)
        for i in range(1, P):
            model.addConstr(tvars[i] >= tvars[i - 1])

        cost_exprs = []
        reg = gp.LinExpr(0.0)

        for scen in scenarios:
            cost = gp.LinExpr(0.0)
            C_prev = None

            for i in range(P):
                W = model.addVar(lb=0.0, ub=UB)
                arr_expr = gp.quicksum(x[p][i] * float(scen["arr"][p]) for p in range(P))
                model.addConstr(W >= arr_expr)
                if i > 0:
                    model.addConstr(W >= C_prev - tvars[i])

                C = model.addVar(lb=0.0, ub=UB)
                dur_expr = gp.quicksum(x[p][i] * float(scen["dur"][p]) for p in range(P))
                model.addConstr(C == tvars[i] + W + dur_expr)

                if cw != 0.0:
                    cost.add(W, cw)
                reg.add(W, 1.0)

                if i < P - 1:
                    G = model.addVar(lb=0.0, ub=UB)
                    model.addConstr(G >= tvars[i + 1] - C)
                    if cg != 0.0:
                        cost.add(G, cg)
                    reg.add(G, 1.0)

                C_prev = C

            O = model.addVar(lb=0.0, ub=UB)
            model.addConstr(O >= C_prev - L)
            if co != 0.0:
                cost.add(O, co)
            reg.add(O, 1.0)

            cost_exprs.append(cost)

        eta = model.addVar(lb=0.0, name="eta")
        for g in groups:
            if not g:
                continue
            expr = gp.quicksum(cost_exprs[s] for s in g) * (1.0 / float(len(g)))
            model.addConstr(eta >= expr)

        model.setObjective(eta + 1e-7 * reg, GRB.MINIMIZE)

        model.update()

        if warm_seq is not None:
            warm_seq = ensure_sequence(warm_seq, P)
            if len(warm_seq) == P and len(set(warm_seq)) == P:
                for p in range(P):
                    for i in range(P):
                        x[p][i].Start = 1.0 if warm_seq[i] == p else 0.0
                if warm_t is not None and len(warm_t) == P:
                    warm_t = normalize_t(warm_t, L, P)
                    for i in range(P):
                        tvars[i].Start = float(warm_t[i])

        model.optimize()

        if model.SolCount == 0:
            return None

        try:
            t = [v.X for v in tvars]
        except Exception:
            return None

        mat = [[safe_float(x[p][i].X, 0.0) for i in range(P)] for p in range(P)]
        seq = extract_permutation(mat)
        t = normalize_t(t, L, P)
        return seq, t

    except Exception:
        return None
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


def build_solution_dict(objective, seq, t, cost_name, cw, cg, co, P, L):
    seq = ensure_sequence(seq, P)
    t = normalize_t(t, L, P)

    assignment = {str(p + 1): i + 1 for i, p in enumerate(seq)}
    schedule = {str(i + 1): float(round(t[i], 6)) for i in range(P)}
    patient_start_times = {str(p + 1): float(round(t[i], 6)) for i, p in enumerate(seq)}

    model_variables = {}
    for i, p in enumerate(seq):
        model_variables[f"x[{p + 1},{i + 1}]"] = 1.0
    for i, val in enumerate(t):
        if abs(float(val)) > 1e-9:
            model_variables[f"t[{i + 1}]"] = float(round(val, 6))

    obj = safe_float(objective, 0.0)
    if math.isinf(obj) or math.isnan(obj):
        obj = 0.0

    return {
        "objective_value": float(round(obj, 6)),
        "cost_structure": str(cost_name),
        "c_w": float(cw),
        "c_g": float(cg),
        "c_o": float(co),
        "assignment": assignment,
        "schedule": schedule,
        "patient_start_times": patient_start_times,
        "model_variables": model_variables,
    }


def write_solution(solution, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-path", "--instance_path", dest="instance_path", required=True)
    parser.add_argument("--solution-path", "--solution_path", dest="solution_path", required=True)
    parser.add_argument("--time-limit", "--time_limit", dest="time_limit", type=float, default=60.0)
    parser.add_argument("--log-path", "--log_path", dest="log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = safe_float(args.time_limit, 60.0)
    if time_limit <= 0.0:
        time_limit = 60.0

    reserve = max(1.0, min(10.0, 0.1 * time_limit))

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    cost_name, cost = get_cost_structure(instance)
    cw = safe_float(cost.get("c_w", 0.0), 0.0)
    cg = safe_float(cost.get("c_g", 0.0), 0.0)
    co = safe_float(cost.get("c_o", 0.0), 0.0)

    P = int(safe_float(instance.get("num_patients_P", len(instance.get("patients", []) or [])), 0))
    P = max(0, P)

    L = max(0.0, safe_float(instance.get("session_length_L", 0.0), 0.0))
    proc_types = get_procedure_types(instance, P)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    if P == 0:
        sol = build_solution_dict(0.0, [], [], cost_name, cw, cg, co, P, L)
        write_solution(sol, args.solution_path)
        if logger is not None:
            try:
                logger.log_solution(0.0, sol)
            except Exception:
                pass
        return

    feats = extract_features(instance, P)
    N = choose_N(P, time_limit)
    scenarios, groups = generate_scenarios(feats, N, seed=0)

    UB = max(1.0, L + sum(feats[p]["max_dur"] + max(0.0, feats[p]["u_U"]) for p in range(P)) + 1.0)

    env = None
    if GUROBI:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Seed", 0)
            env.setParam("Threads", 1)
            env.setParam("NumericFocus", 0)
            env.start()
        except Exception:
            env = None

    best_obj = float("inf")
    best_seq = None
    best_t = None
    logged = False

    def consider(obj, seq, t):
        nonlocal best_obj, best_seq, best_t, logged
        obj = safe_float(obj, float("inf"))
        if math.isinf(obj) or math.isnan(obj):
            return
        seq = ensure_sequence(seq, P)
        t = normalize_t(t, L, P)
        if obj < best_obj - 1e-6:
            best_obj = obj
            best_seq = list(seq)
            best_t = list(t)
            if logger is not None:
                try:
                    sol = build_solution_dict(best_obj, best_seq, best_t, cost_name, cw, cg, co, P, L)
                    logger.log_solution(best_obj, sol)
                    logged = True
                except Exception:
                    pass

    initial_seq = sorted(range(P), key=lambda p: feats[p]["meanD"])
    initial_t = fallback_schedule(initial_seq, feats, cw, cg, co, L)
    initial_obj, _ = evaluate_schedule(initial_seq, initial_t, scenarios, groups, cw, cg, co, L)
    consider(initial_obj, initial_seq, initial_t)

    max_candidates = 30 if time_limit < 60 else 80
    max_rand = min(20, max(5, max_candidates // 3))
    candidates = generate_candidates(feats, proc_types, max_candidates=max_candidates, max_rand=max_rand, seed=0)

    quick_budget = min(max(5.0, 0.15 * time_limit), 30.0)
    quick_list = []

    for cand in candidates:
        if time.time() - start_time > quick_budget:
            break
        tq = fallback_schedule(cand, feats, cw, cg, co, L)
        objq, _ = evaluate_schedule(cand, tq, scenarios, groups, cw, cg, co, L)
        quick_list.append((objq, cand, tq))
        consider(objq, cand, tq)

    quick_list.sort(key=lambda z: z[0])

    if time_limit < 30:
        K_top = 3
    elif time_limit < 90:
        K_top = 6
    elif time_limit < 300:
        K_top = 10
    else:
        K_top = 15

    if P > 25:
        K_top = max(3, K_top // 2)

    joint_planned = GUROBI and P <= 18 and time_limit > 90.0
    joint_reserve = max(20.0, min(120.0, 0.35 * time_limit)) if joint_planned else 0.0
    lp_end = max(0.0, time_limit - reserve - joint_reserve)

    top = quick_list[:K_top]
    for idx, (_, cand, _) in enumerate(top):
        elapsed = time.time() - start_time
        if elapsed > lp_end:
            break
        rem = lp_end - elapsed
        if rem < 2.0:
            break
        tl = min(20.0, max(2.0, rem / float(max(1, len(top) - idx))))
        res = optimize_fixed_sequence(cand, scenarios, groups, cw, cg, co, L, UB, env, tl)
        if res is not None:
            consider(res[0], cand, res[1])

    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - reserve

    if joint_planned and remaining > 25.0:
        N_joint = min(N, 8 if P <= 12 else 5)
        N_joint = max(3, N_joint)

        if N_joint >= N:
            joint_scenarios, joint_groups = scenarios, groups
        else:
            joint_scenarios, joint_groups = generate_scenarios(feats, N_joint, seed=2)

        joint_time = min(max(10.0, remaining - 5.0), 180.0)
        joint_res = solve_joint_mip(
            joint_scenarios,
            joint_groups,
            P,
            cw,
            cg,
            co,
            L,
            UB,
            env,
            joint_time,
            best_seq,
            best_t,
        )

        if joint_res is not None:
            seqj, tj = joint_res
            objj, _ = evaluate_schedule(seqj, tj, scenarios, groups, cw, cg, co, L)
            consider(objj, seqj, tj)

            elapsed = time.time() - start_time
            rem = time_limit - elapsed - reserve
            if rem > 8.0:
                ref = optimize_fixed_sequence(
                    seqj,
                    scenarios,
                    groups,
                    cw,
                    cg,
                    co,
                    L,
                    UB,
                    env,
                    min(20.0, rem - 3.0),
                )
                if ref is not None:
                    consider(ref[0], seqj, ref[1])

    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - reserve

    if remaining > 10.0 and best_seq is not None:
        end_time = time.time() + remaining - 2.0
        seq = list(best_seq)
        t = list(best_t)
        obj = best_obj
        passes = 0

        while passes < 2 and time.time() < end_time:
            passes += 1
            improved = False

            moves = [(i, i + 1) for i in range(P - 1)]
            lrng = random.Random(99)
            for _ in range(min(10, P)):
                a = lrng.randrange(P)
                b = lrng.randrange(P)
                if a != b:
                    moves.append((a, b))
            lrng.shuffle(moves)

            for a, b in moves:
                if time.time() > end_time:
                    break

                cand = seq[:]
                cand[a], cand[b] = cand[b], cand[a]

                qobj, _ = evaluate_schedule(cand, t, scenarios, groups, cw, cg, co, L)
                if qobj < obj - 1e-6:
                    rem = end_time - time.time()
                    if rem > 5.0:
                        res = optimize_fixed_sequence(
                            cand,
                            scenarios,
                            groups,
                            cw,
                            cg,
                            co,
                            L,
                            UB,
                            env,
                            min(5.0, rem - 2.0),
                        )
                        if res is not None and res[0] < obj - 1e-6:
                            obj = res[0]
                            t = res[1]
                            seq = cand
                            consider(obj, seq, t)
                            improved = True
                            break
                    else:
                        obj = qobj
                        seq = cand
                        consider(obj, seq, t)
                        improved = True
                        break

            if not improved:
                break

    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - reserve

    if remaining > 3.0 and best_seq is not None:
        res = optimize_fixed_sequence(
            best_seq,
            scenarios,
            groups,
            cw,
            cg,
            co,
            L,
            UB,
            env,
            max(2.0, remaining - 2.0),
        )
        if res is not None:
            consider(res[0], best_seq, res[1])

    if best_seq is None:
        best_seq = list(range(P))
        best_t = fallback_schedule(best_seq, feats, cw, cg, co, L)
        best_obj, _ = evaluate_schedule(best_seq, best_t, scenarios, groups, cw, cg, co, L)

    if math.isinf(best_obj) or math.isnan(best_obj):
        best_obj = 0.0

    solution = build_solution_dict(best_obj, best_seq, best_t, cost_name, cw, cg, co, P, L)
    write_solution(solution, args.solution_path)

    if logger is not None and not logged:
        try:
            logger.log_solution(best_obj, solution)
        except Exception:
            pass


if __name__ == "__main__":
    main()