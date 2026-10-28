import argparse
import json
import math
import os
import random
import time
from bisect import bisect_left, bisect_right

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Instance loading / deterministic QC-profile reconstruction
# ---------------------------------------------------------------------------

def find_seed(inst):
    for key in ("seed", "random_seed", "rng_seed", "profile_seed", "profiles_seed"):
        v = inst.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    qp = inst.get("qc_profiles", {})
    if isinstance(qp, dict):
        for key in ("seed", "random_seed"):
            v = qp.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        for cls in ("big", "small"):
            sub = qp.get(cls)
            if isinstance(sub, dict):
                for key in ("seed", "random_seed"):
                    v = sub.get(key)
                    if isinstance(v, int) and not isinstance(v, bool):
                        return v
    return 42


def sample_shift_sequences(min_qc, max_qc, min_len, max_len, count, rng):
    """Reproduce the published rule: enumerate all admissible shift-level QC
    sequences (length in [min_len,max_len], entries in [min_qc,max_qc]) in
    lexicographic product order, then sample `count` of them with the seed."""
    min_qc = int(min_qc); max_qc = int(max_qc)
    min_len = int(min_len); max_len = int(max_len)
    count = int(count)
    if count <= 0 or max_qc < min_qc or max_len < min_len:
        return []
    m = max_qc - min_qc + 1
    lengths = list(range(min_len, max_len + 1))
    counts = [m ** L for L in lengths]
    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)
    total = cum[-1]
    if count >= total:
        idxs = list(range(total))
    else:
        # Sampling indices over range(total) uses exactly the same RNG draws as
        # random.sample over the materialized list of tuples.
        idxs = rng.sample(range(total), count)
    seqs = []
    for idx in idxs:
        li = bisect_right(cum, idx) - 1
        L = lengths[li]
        rem = idx - cum[li]
        seq = [0] * L
        for pos in range(L - 1, -1, -1):  # rightmost digit cycles fastest
            seq[pos] = min_qc + (rem % m)
            rem //= m
        seqs.append(tuple(seq))
    return seqs


def class_sequences(spec, count, rng):
    for key in ("profiles", "sequences", "profile_sequences"):
        if isinstance(spec, dict) and isinstance(spec.get(key), list):
            return [tuple(int(x) for x in s) for s in spec[key]]
    return sample_shift_sequences(spec["min_qc"], spec["max_qc"],
                                  spec["min_handling_shifts"],
                                  spec["max_handling_shifts"], count, rng)


class VesselData:
    __slots__ = ("id", "a", "b", "k_bar", "t_bar", "D", "phase",
                 "profiles", "starts", "berths_of_start")


def prepare(inst):
    S = int(inst["time_steps_per_shift"])
    T = int(inst["total_time_steps"])
    H = float(inst["time_horizon_hours"])
    steps_per_hour = (T / H) if H else 1.0
    c1 = float(inst["c1"])
    c2 = float(inst["c2"])
    K = int(inst["num_berths"])

    # Detect whether time steps in the file are 0-based or 1-based and work
    # internally with 0-based indices (offsets cancel in the objective).
    time_vals = []
    for v in inst["vessels"]:
        time_vals += [v["a"], v["b"], v["t_bar"]]
    for b in inst["berths"]:
        time_vals += [b["a_k"], b["b_k"]]
    one_based = (min(time_vals) >= 1) and (max(time_vals) >= T)
    off = 1 if one_based else 0

    seed = find_seed(inst)
    rng = random.Random(seed)
    qp = inst["qc_profiles"]
    big = class_sequences(qp["big"], int(inst["num_big_profiles"]), rng)
    small = class_sequences(qp["small"], int(inst["num_small_profiles"]), rng)
    combined = big + small

    qc_avail = None
    for key in ("qc_available", "qc_per_time_step", "qcs_available", "qc_availability"):
        if isinstance(inst.get(key), list):
            qc_avail = [int(x) for x in inst[key]]
            break
    if qc_avail is None or len(qc_avail) < T:
        qc_avail = [int(inst["total_qc"])] * T
    else:
        qc_avail = qc_avail[:T]

    berths_a = [max(0, int(b["a_k"]) - off) for b in inst["berths"]]
    berths_b = [min(T - 1, int(b["b_k"]) - off) for b in inst["berths"]]

    vessels = []
    for v in sorted(inst["vessels"], key=lambda x: x["id"]):
        vd = VesselData()
        vd.id = int(v["id"])
        vd.a = max(0, int(v["a"]) - off)
        vd.b = min(T - 1, int(v["b"]) - off)
        vd.k_bar = int(v["k_bar"])
        vd.t_bar = int(v["t_bar"]) - off
        D = int(round(float(v["duration_hours"]) * steps_per_hour))
        if D < 1:
            D = 1
        vd.D = D
        # A profile spanning L shifts started at within-shift index r lasts
        # L*S - r steps.  The vessel's feasible duration window pins down
        # L = ceil(D/S) and the matching phase r = L*S - D.
        L = (D + S - 1) // S
        r = L * S - D
        if r < 0 or r >= S:
            r = 0
            L = (D + S - 1) // S
            D = L * S
            vd.D = D
        vd.phase = r
        profs = [seq for seq in combined if len(seq) == L]
        if not profs:
            # Defensive fallback for inconsistent data: enumerate every
            # sequence of the required length over the union QC range.
            lo = min(qp["big"]["min_qc"], qp["small"]["min_qc"])
            hi = max(qp["big"]["max_qc"], qp["small"]["max_qc"])
            profs = sample_shift_sequences(lo, hi, L, L,
                                           (hi - lo + 1) ** L, random.Random(seed))
        vd.profiles = profs

        starts = []
        bos = {}
        end_max = T - 1
        tt = vd.a
        while tt <= vd.b:
            if tt % S == r and tt + D - 1 <= end_max:
                ks = [k for k in range(K)
                      if berths_a[k] <= tt and tt + D - 1 <= berths_b[k]]
                if ks:
                    starts.append(tt)
                    bos[tt] = ks
            tt += 1
        vd.starts = starts
        vd.berths_of_start = bos
        vessels.append(vd)

    return vessels, K, T, S, c1, c2, qc_avail, off


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assignment_objective(vessels, assn, c1, c2):
    obj = 0.0
    for vd in vessels:
        k, t, p = assn[vd.id]
        obj += c1 * abs(k - vd.k_bar) + c2 * abs(t - vd.t_bar)
    return obj


def build_solution(vessels, assn, c1, c2, off):
    out = []
    obj = 0.0
    for vd in sorted(vessels, key=lambda x: x.id):
        k, t, p = assn[vd.id]
        end = t + vd.D - 1
        obj += c1 * abs(k - vd.k_bar) + c2 * abs(t - vd.t_bar)
        out.append({"id": int(vd.id),
                    "berth": int(k),
                    "start_time": int(t + off),
                    "end_time": int(end + off),
                    "profile": int(p)})
    return {"objective_value": float(obj), "vessels": out}


def build_candidates(vessels, c1, c2):
    cand = {}
    prof_order = {}
    for vd in vessels:
        lst = []
        for t in vd.starts:
            base = c2 * abs(t - vd.t_bar)
            for k in vd.berths_of_start[t]:
                lst.append((c1 * abs(k - vd.k_bar) + base, k, t))
        lst.sort()
        cand[vd.id] = lst
        profs = vd.profiles
        prof_order[vd.id] = sorted(range(len(profs)), key=lambda p: sum(profs[p]))
    return cand, prof_order


def greedy(vessels, order, cand, prof_order, K, T, qc_avail):
    berth_used = [bytearray(T) for _ in range(K)]
    qc_used = [0] * T
    assn = {}
    for vid in order:
        vd = vessels[vid]
        D = vd.D
        placed = False
        for cost, k, t in cand[vid]:
            end = t + D
            if any(berth_used[k][t:end]):
                continue
            s0 = t // S_GLOBAL
            chosen = None
            for p in prof_order[vid]:
                seq = vd.profiles[p]
                ok = True
                for tau in range(t, end):
                    if qc_used[tau] + seq[(tau // S_GLOBAL) - s0] > qc_avail[tau]:
                        ok = False
                        break
                if ok:
                    chosen = p
                    break
            if chosen is None:
                continue
            # place
            berth_used[k][t:end] = b"\x01" * D
            seq = vd.profiles[chosen]
            for tau in range(t, end):
                qc_used[tau] += seq[(tau // S_GLOBAL) - s0]
            assn[vid] = (k, t, chosen)
            placed = True
            break
        if not placed:
            return None
    return assn


def local_search(vessels, assn, cand, prof_order, K, T, qc_avail, c1, c2, deadline):
    berth_used = [bytearray(T) for _ in range(K)]
    qc_used = [0] * T

    def place(vd, k, t, p, delta):
        D = vd.D
        end = t + D
        s0 = t // S_GLOBAL
        seq = vd.profiles[p]
        if delta > 0:
            berth_used[k][t:end] = b"\x01" * D
            for tau in range(t, end):
                qc_used[tau] += seq[(tau // S_GLOBAL) - s0]
        else:
            berth_used[k][t:end] = b"\x00" * D
            for tau in range(t, end):
                qc_used[tau] -= seq[(tau // S_GLOBAL) - s0]

    def fits(vd, k, t, p):
        D = vd.D
        end = t + D
        if any(berth_used[k][t:end]):
            return False
        s0 = t // S_GLOBAL
        seq = vd.profiles[p]
        for tau in range(t, end):
            if qc_used[tau] + seq[(tau // S_GLOBAL) - s0] > qc_avail[tau]:
                return False
        return True

    for vd in vessels:
        k, t, p = assn[vd.id]
        place(vd, k, t, p, +1)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for vd in vessels:
            if time.time() > deadline:
                break
            vid = vd.id
            k0, t0, p0 = assn[vid]
            cur_cost = c1 * abs(k0 - vd.k_bar) + c2 * abs(t0 - vd.t_bar)
            place(vd, k0, t0, p0, -1)
            best = None
            for cost, k, t in cand[vid]:
                if cost >= cur_cost:
                    break
                chosen = None
                for p in prof_order[vid]:
                    if fits(vd, k, t, p):
                        chosen = p
                        break
                if chosen is not None:
                    best = (k, t, chosen)
                    break
            if best is not None:
                assn[vid] = best
                place(vd, best[0], best[1], best[2], +1)
                improved = True
            else:
                place(vd, k0, t0, p0, +1)
    return assn


# ---------------------------------------------------------------------------
# Gurobi model
# ---------------------------------------------------------------------------

def solve_mip(vessels, K, T, S, c1, c2, qc_avail, init_assn, logger,
              time_remaining, off, incumbent_obj):
    model = gp.Model("bap_qc_profiles")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.1, float(time_remaining))

    y_vars = {}
    z_vars = {}
    for vd in vessels:
        vid = vd.id
        for t in vd.starts:
            base = c2 * abs(t - vd.t_bar)
            for k in vd.berths_of_start[t]:
                cost = c1 * abs(k - vd.k_bar) + base
                y_vars[(vid, k, t)] = model.addVar(vtype=GRB.BINARY,
                                                   obj=float(cost),
                                                   name="y_%d_%d_%d" % (vid, k, t))
            for p in range(len(vd.profiles)):
                z_vars[(vid, t, p)] = model.addVar(vtype=GRB.BINARY, obj=0.0,
                                                   name="z_%d_%d_%d" % (vid, t, p))
    model.update()

    # Exactly one profile per chosen start; profile/start linkage.
    for vd in vessels:
        vid = vd.id
        for t in vd.starts:
            ys = [y_vars[(vid, k, t)] for k in vd.berths_of_start[t]]
            zs = [z_vars[(vid, t, p)] for p in range(len(vd.profiles))]
            model.addConstr(gp.quicksum(zs) == gp.quicksum(ys))

    # Berth capacity: at most one vessel per berth per time step.
    for k in range(K):
        for tau in range(T):
            terms = []
            for vd in vessels:
                D = vd.D
                starts = vd.starts
                i = bisect_left(starts, tau - D + 1)
                n = len(starts)
                while i < n and starts[i] <= tau:
                    var = y_vars.get((vd.id, k, starts[i]))
                    if var is not None:
                        terms.append(var)
                    i += 1
            if terms:
                model.addConstr(gp.quicksum(terms) <= 1)

    # QC capacity per time step.
    for tau in range(T):
        coeffs = []
        vars_ = []
        shift_tau = tau // S
        for vd in vessels:
            D = vd.D
            starts = vd.starts
            i = bisect_left(starts, tau - D + 1)
            n = len(starts)
            while i < n and starts[i] <= tau:
                t = starts[i]
                e = shift_tau - (t // S)
                if 0 <= e:
                    for p, seq in enumerate(vd.profiles):
                        if e < len(seq):
                            q = seq[e]
                            if q:
                                coeffs.append(q)
                                vars_.append(z_vars[(vd.id, t, p)])
                i += 1
        if vars_:
            model.addConstr(gp.LinExpr(coeffs, vars_) <= qc_avail[tau])

    model.update()

    y_keys = list(y_vars.keys())
    y_list = [y_vars[k] for k in y_keys]
    z_keys = list(z_vars.keys())
    z_list = [z_vars[k] for k in z_keys]
    nV = len(vessels)
    vd_by_id = {vd.id: vd for vd in vessels}

    # MIP start from the heuristic incumbent.
    start_vals = [0.0] * (len(y_list) + len(z_list))
    if init_assn:
        y_index = {key: i for i, key in enumerate(y_keys)}
        z_index = {key: i for i, key in enumerate(z_keys)}
        for vid, (k, t, p) in init_assn.items():
            i = y_index.get((vid, k, t))
            if i is not None:
                start_vals[i] = 1.0
            j = z_index.get((vid, t, p))
            if j is not None:
                start_vals[len(y_list) + j] = 1.0
    all_list = y_list + z_list
    model.setAttr("Start", all_list, start_vals)
    model.update()

    class Callback:
        def __init__(self):
            self.best = incumbent_obj
            self.sol = None

        def __call__(self, m, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                yv = m.cbGetSolution(y_list)
                zv = m.cbGetSolution(z_list)
            except gp.GurobiError:
                return
            sel_y = {}
            for key, val in zip(y_keys, yv):
                if val > 0.5:
                    sel_y[key[0]] = (key[1], key[2])
            if len(sel_y) < nV:
                return
            sel_z = {}
            for key, val in zip(z_keys, zv):
                if val > 0.5:
                    sel_z[(key[0], key[1])] = key[2]
            assn = {}
            for vid, (k, t) in sel_y.items():
                p = sel_z.get((vid, t), 0)
                assn[vid] = (k, t, p)
            sol = build_solution(vessels, assn, c1, c2, off)
            obj = sol["objective_value"]
            if obj < self.best - 1e-9:
                self.best = obj
                self.sol = sol
                if logger:
                    logger.log_solution(float(obj), sol)

    cb = Callback()
    try:
        model.optimize(cb)
    except gp.GurobiError:
        pass

    # Final safety net: read the best available solution from Gurobi.
    if model.SolCount > 0:
        try:
            yv = model.getAttr("X", y_list)
            zv = model.getAttr("X", z_list)
            sel_y = {}
            for key, val in zip(y_keys, yv):
                if val > 0.5:
                    sel_y[key[0]] = (key[1], key[2])
            if len(sel_y) == nV:
                sel_z = {}
                for key, val in zip(z_keys, zv):
                    if val > 0.5:
                        sel_z[(key[0], key[1])] = key[2]
                assn = {}
                for vid, (k, t) in sel_y.items():
                    assn[vid] = (k, t, sel_z.get((vid, t), 0))
                sol = build_solution(vessels, assn, c1, c2, off)
                if sol["objective_value"] < cb.best - 1e-9:
                    cb.best = sol["objective_value"]
                    cb.sol = sol
                    if logger:
                        logger.log_solution(float(cb.best), sol)
        except gp.GurobiError:
            pass

    model.dispose()
    return cb.sol, cb.best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

S_GLOBAL = 1  # module-level shortcut set in main (used by greedy/local search)


def main():
    global S_GLOBAL
    args = parse_args()
    t_start = time.time()
    deadline = t_start + float(args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    vessels, K, T, S, c1, c2, qc_avail, off = prepare(inst)
    S_GLOBAL = S
    cand, prof_order = build_candidates(vessels, c1, c2)

    # ---------------- heuristic incumbent ----------------
    best_assn = None
    best_obj = math.inf
    ids = [vd.id for vd in vessels]
    orders = [
        sorted(ids, key=lambda i: (vessels[i].a, vessels[i].b, i)),
        sorted(ids, key=lambda i: (vessels[i].b - vessels[i].a, vessels[i].a, i)),
        sorted(ids, key=lambda i: (vessels[i].t_bar, i)),
        sorted(ids, key=lambda i: (-vessels[i].D, vessels[i].a, i)),
    ]
    for order in orders:
        if time.time() > deadline - 0.5:
            break
        assn = greedy(vessels, order, cand, prof_order, K, T, qc_avail)
        if assn is not None and len(assn) == len(vessels):
            obj = assignment_objective(vessels, assn, c1, c2)
            if obj < best_obj - 1e-9:
                best_obj = obj
                best_assn = assn
        if best_assn is not None:
            break  # first complete ordering is usually enough; others rarely differ

    if best_assn is not None and time.time() < deadline - 0.5:
        ls_deadline = min(deadline - 1.0,
                          time.time() + min(5.0, 0.10 * float(args.time_limit)))
        improved = local_search(vessels, best_assn, cand, prof_order,
                                K, T, qc_avail, c1, c2, ls_deadline)
        obj = assignment_objective(vessels, improved, c1, c2)
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_assn = improved

    best_sol = None
    if best_assn is not None:
        best_sol = build_solution(vessels, best_assn, c1, c2, off)
        if logger:
            logger.log_solution(float(best_obj), best_sol)

    # ---------------- MIP improvement ----------------
    # Estimate model size; skip MIP if it is clearly too large to be useful.
    est_terms = 0
    for vd in vessels:
        n_starts = len(vd.starts)
        n_y = sum(len(vd.berths_of_start[t]) for t in vd.starts)
        n_z = n_starts * len(vd.profiles)
        est_terms += vd.D * (n_y + n_z) + n_y + n_z
    n_vars = sum(sum(len(vd.berths_of_start[t]) for t in vd.starts)
                 + len(vd.starts) * len(vd.profiles) for vd in vessels)

    remaining = deadline - time.time() - 0.7
    if (best_assn is not None or True) and remaining > 1.0 \
            and est_terms <= 8_000_000 and n_vars <= 1_500_000 \
            and K * T * max(1, len(vessels)) <= 6_000_000:
        try:
            mip_sol, mip_obj = solve_mip(vessels, K, T, S, c1, c2, qc_avail,
                                         best_assn, logger, remaining, off,
                                         best_obj)
            if mip_sol is not None and mip_obj < best_obj - 1e-9:
                best_obj = mip_obj
                best_sol = mip_sol
        except Exception:
            pass
    elif best_assn is not None and remaining > 1.0:
        # Model too large: spend the remaining time on local search instead.
        improved = local_search(vessels, best_assn, cand, prof_order,
                                K, T, qc_avail, c1, c2, deadline - 0.5)
        obj = assignment_objective(vessels, improved, c1, c2)
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_sol = build_solution(vessels, improved, c1, c2, off)
            if logger:
                logger.log_solution(float(best_obj), best_sol)

    # ---------------- write output ----------------
    if best_sol is None:
        best_sol = {"objective_value": None, "vessels": []}
    out_path = args.solution_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_sol, f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()