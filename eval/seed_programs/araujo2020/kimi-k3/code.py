#!/usr/bin/env python3
"""
Multi-project, multi-mode RCPSP with total project delay objective
(lexicographic tie-break on makespan).

Approach:
  1. Parse instance, build precedence graph, compute per-project CPM
     critical-path lengths, earliest starts (forward pass) and backward
     longest-path distances exactly as described.
  2. Build a feasible schedule with a fast multi-start serial SGS
     heuristic -> gives per-project upper bounds and the maximum
     allowable total delay (TD).
  3. Use the time windows [ES_j, LS_j] implied by TD to formulate a
     time-indexed MILP, warm-started with the heuristic solution, and
     solve it with Gurobi (lexicographic: delay first, makespan second).
  4. Return the best solution found within --time_limit.
"""

import argparse
import heapq
import json
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Instance handling
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    projects = data["projects"]
    jobs = data["jobs"]
    resources = data.get("resources", {}) or {}
    renewable = resources.get("renewable", []) or []
    nonrenewable = resources.get("nonrenewable", []) or []

    ren_cap = [r["capacity"] for r in renewable]
    nr_cap = [r["capacity"] for r in nonrenewable]

    job_ids = [j["job_id"] for j in jobs]
    project_of = {j["job_id"]: j["project_id"] for j in jobs}
    release_of = {p["project_id"]: p["release_date"] for p in projects}
    sink_of = {p["project_id"]: p["artificial_sink_job_id"] for p in projects}
    source_of = {p["project_id"]: p["artificial_source_job_id"] for p in projects}

    modes = {}          # jid -> list of mode dicts (augmented)
    for j in jobs:
        ml = []
        for m in j.get("modes", []):
            ren = list(m.get("renewable_consumption", []))
            nr = list(m.get("nonrenewable_consumption", []))
            if len(ren) < len(ren_cap):
                ren += [0] * (len(ren_cap) - len(ren))
            if len(nr) < len(nr_cap):
                nr += [0] * (len(nr_cap) - len(nr))
            feas = all(ren[r] <= ren_cap[r] for r in range(len(ren_cap)))
            ml.append({
                "mode_id": m["mode_id"],
                "d": int(m["duration"]),
                "ren": ren,
                "nr": nr,
                "feas": feas,
            })
        if not ml:  # defensive: every job should have at least one mode
            ml.append({"mode_id": 0, "d": 0,
                       "ren": [0] * len(ren_cap),
                       "nr": [0] * len(nr_cap), "feas": True})
        ml.sort(key=lambda m: (m["d"], sum(m["nr"])))
        modes[j["job_id"]] = ml

    succ = {j["job_id"]: set(j.get("successors", []) or []) for j in jobs}
    pred = {jid: set() for jid in job_ids}
    for j in jobs:
        for s in succ[j["job_id"]]:
            if s in pred:
                pred[s].add(j["job_id"])

    dmin = {jid: min(m["d"] for m in modes[jid]) for jid in job_ids}

    # topological order (Kahn)
    indeg = {jid: len(pred[jid]) for jid in job_ids}
    heap = [jid for jid in job_ids if indeg[jid] == 0]
    heapq.heapify(heap)
    topo = []
    while heap:
        jid = heapq.heappop(heap)
        topo.append(jid)
        for s in succ[jid]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(heap, s)
    if len(topo) != len(job_ids):  # cyclic (should not happen)
        topo = sorted(job_ids)

    # Earliest starts (forward pass, min durations, release dates)
    ES = {jid: release_of[project_of[jid]] for jid in job_ids}
    for jid in topo:
        base = ES[jid] + dmin[jid]
        for s in succ[jid]:
            if base > ES[s]:
                ES[s] = base
    for jid in job_ids:
        if ES[jid] < release_of[project_of[jid]]:
            ES[jid] = release_of[project_of[jid]]

    # Backward distances (longest path job -> sink, min durations)
    back = {jid: dmin[jid] for jid in job_ids}
    for jid in reversed(topo):
        best = dmin[jid]
        for s in succ[jid]:
            cand = dmin[jid] + back[s]
            if cand > best:
                best = cand
        back[jid] = best

    CP = {}
    for p in projects:
        pid = p["project_id"]
        CP[pid] = ES[sink_of[pid]] - p["release_date"]

    return {
        "projects": projects,
        "job_ids": job_ids,
        "project_of": project_of,
        "release_of": release_of,
        "sink_of": sink_of,
        "source_of": source_of,
        "modes": modes,
        "succ": succ,
        "pred": pred,
        "dmin": dmin,
        "topo": topo,
        "ES": ES,
        "back": back,
        "CP": CP,
        "ren_cap": ren_cap,
        "nr_cap": nr_cap,
    }


# ---------------------------------------------------------------------------
# Serial SGS heuristic
# ---------------------------------------------------------------------------

def topo_with_key(inst, keyfunc):
    """Topological ordering; among currently-eligible jobs pick the one
    with the smallest key."""
    job_ids = inst["job_ids"]
    succ = inst["succ"]
    pred = inst["pred"]
    indeg = {jid: len(pred[jid]) for jid in job_ids}
    heap = [(keyfunc(jid), jid) for jid in job_ids if indeg[jid] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, jid = heapq.heappop(heap)
        order.append(jid)
        for s in succ[jid]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(heap, (keyfunc(s), s))
    if len(order) != len(job_ids):
        order = sorted(job_ids)
    return order


def run_sgs(inst, order):
    modes = inst["modes"]
    pred = inst["pred"]
    project_of = inst["project_of"]
    release_of = inst["release_of"]
    ren_cap = inst["ren_cap"]
    nr_cap = inst["nr_cap"]
    nrr = len(nr_cap)
    nren = len(ren_cap)

    # suffix per-resource minima of non-renewable consumption
    # (amount that must be reserved for jobs scheduled later)
    suff = [[0] * nrr for _ in range(len(order) + 1)]
    if nrr:
        for i in range(len(order) - 1, -1, -1):
            row = suff[i + 1][:]
            ml = modes[order[i]]
            for r in range(nrr):
                row[r] += min(m["nr"][r] for m in ml)
            suff[i] = row

    remaining = list(nr_cap)
    usage = [dict() for _ in ren_cap]
    finish = {}
    assign = {}

    def find_start(e, d, ren):
        t = e
        while True:
            jump = -1
            for off in range(d):
                tau = t + off
                for r in range(nren):
                    a = ren[r]
                    if a and usage[r].get(tau, 0) + a > ren_cap[r]:
                        jump = tau + 1
                        break
                if jump != -1:
                    break
            if jump == -1:
                return t
            t = jump

    for i, jid in enumerate(order):
        e = release_of[project_of[jid]]
        for p in pred[jid]:
            f = finish.get(p)
            if f is not None and f > e:
                e = f

        best = None
        for m in modes[jid]:
            if not m["feas"]:
                continue
            ok = True
            for r in range(nrr):
                if m["nr"][r] + suff[i + 1][r] > remaining[r]:
                    ok = False
                    break
            if not ok:
                continue
            d = m["d"]
            t = find_start(e, d, m["ren"]) if d > 0 else e
            key = (t + d, d, sum(m["nr"]))
            if best is None or key < best[0]:
                best = (key, t, m)

        if best is None:  # relax the reservation rule (defensive)
            cands = [m for m in modes[jid]
                     if all(m["nr"][r] <= remaining[r] for r in range(nrr))]
            if not cands:
                cands = [m for m in modes[jid] if m["feas"]] or modes[jid]
            m = min(cands, key=lambda mm: sum(mm["nr"]))
            t = find_start(e, m["d"], m["ren"]) if m["d"] > 0 else e
            best = ((t + m["d"], m["d"], sum(m["nr"])), t, m)

        _, t, m = best
        d = m["d"]
        assign[jid] = (m["mode_id"], t, d)
        finish[jid] = t + d
        if d > 0:
            for r in range(nren):
                a = m["ren"][r]
                if a:
                    u = usage[r]
                    for tau in range(t, t + d):
                        u[tau] = u.get(tau, 0) + a
        for r in range(nrr):
            remaining[r] -= m["nr"][r]

    return assign


def evaluate(inst, assign):
    comp = {}
    for p in inst["projects"]:
        pid = p["project_id"]
        comp[pid] = assign[inst["sink_of"][pid]][1]
    delay = 0
    for p in inst["projects"]:
        pid = p["project_id"]
        delay += comp[pid] - p["release_date"] - inst["CP"][pid]
    ms = max(comp.values()) if comp else 0
    return delay, ms


def heuristic(inst, t_end):
    ES = inst["ES"]
    back = inst["back"]
    keys = [
        lambda j: (ES[j], j),
        lambda j: (ES[j] - back[j], j),
        lambda j: (j, j),
        lambda j: (-back[j], j),
    ]
    best_assign = None
    best_key = None
    rnd = random.Random(0)
    it = 0
    while time.time() < t_end and (best_assign is None or it < 30):
        if it < len(keys):
            kf = keys[it]
        else:
            rank = {j: rnd.random() for j in inst["job_ids"]}
            kf = lambda j, rank=rank: (rank[j], j)
        try:
            order = topo_with_key(inst, kf)
            assign = run_sgs(inst, order)
            d, ms = evaluate(inst, assign)
        except Exception:
            it += 1
            continue
        if best_key is None or (d, ms) < best_key:
            best_key = (d, ms)
            best_assign = assign
        it += 1
    return best_assign, best_key


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def make_solution_dict(inst, assign):
    delay, ms = evaluate(inst, assign)
    schedule = [
        {"job_id": jid, "mode_id": assign[jid][0],
         "start_time": int(assign[jid][1])}
        for jid in sorted(assign.keys())
    ]
    out = {
        "objective_value": float(delay),
        "schedule": schedule,
        "makespan": int(ms),
    }
    return float(delay), int(ms), out


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------

def solve_milp(inst, heur_assign, logger, deadline):
    remaining = max(deadline - time.time(), 0)
    if remaining < 0.5:
        return None

    job_ids = inst["job_ids"]
    modes = inst["modes"]
    succ = inst["succ"]
    project_of = inst["project_of"]
    release_of = inst["release_of"]
    sink_of = inst["sink_of"]
    ES = inst["ES"]
    back = inst["back"]
    CP = inst["CP"]
    ren_cap = inst["ren_cap"]
    nr_cap = inst["nr_cap"]

    # upper bounds from heuristic
    TD = 0
    for p in inst["projects"]:
        pid = p["project_id"]
        ub = heur_assign[sink_of[pid]][1]
        TD += max(0, ub - p["release_date"] - CP[pid])

    LS = {}
    for jid in job_ids:
        pid = project_of[jid]
        ls = release_of[pid] + CP[pid] + TD - back[jid]
        if ls < ES[jid]:
            ls = ES[jid]
        LS[jid] = ls

    # mode variants usable in the model (statically feasible if possible)
    usable = {}
    for jid in job_ids:
        fl = [m for m in modes[jid] if m["feas"]]
        usable[jid] = fl if fl else list(modes[jid])

    nvars = sum(len(usable[j]) * (LS[j] - ES[j] + 1) for j in job_ids)
    if nvars > 600000:     # too large for a one-core MILP; keep heuristic
        return None

    try:
        model = gp.Model("mmrcpsp")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(remaining - 0.2, 1.0)

        x = {}
        var_list = []
        meta = []
        for jid in job_ids:
            lo, hi = ES[jid], LS[jid]
            for mi in range(len(usable[jid])):
                for t in range(lo, hi + 1):
                    v = model.addVar(vtype=GRB.BINARY,
                                     name="x[%d,%d,%d]" % (jid, mi, t))
                    x[(jid, mi, t)] = v
                    var_list.append(v)
                    meta.append((jid, mi, t))

        # exactly one mode/start per job
        for jid in job_ids:
            lo, hi = ES[jid], LS[jid]
            model.addConstr(
                gp.quicksum(x[(jid, mi, t)]
                            for mi in range(len(usable[jid]))
                            for t in range(lo, hi + 1)) == 1,
                name="assign[%d]" % jid)

        # precedence
        for jid in job_ids:
            fin_expr = gp.quicksum(
                (t + usable[jid][mi]["d"]) * x[(jid, mi, t)]
                for mi in range(len(usable[jid]))
                for t in range(ES[jid], LS[jid] + 1))
            for s in succ[jid]:
                st_expr = gp.quicksum(
                    t * x[(s, mi, t)]
                    for mi in range(len(usable[s]))
                    for t in range(ES[s], LS[s] + 1))
                model.addConstr(fin_expr <= st_expr,
                                name="prec[%d,%d]" % (jid, s))

        # non-renewable resources
        for r in range(len(nr_cap)):
            lhs = gp.quicksum(
                usable[jid][mi]["nr"][r] * x[(jid, mi, t)]
                for jid in job_ids
                for mi in range(len(usable[jid]))
                if usable[jid][mi]["nr"][r]
                for t in range(ES[jid], LS[jid] + 1))
            model.addConstr(lhs <= nr_cap[r], name="nr[%d]" % r)

        # renewable resources (per period)
        nren = len(ren_cap)
        if nren:
            terms = {}
            for jid in job_ids:
                for mi, m in enumerate(usable[jid]):
                    d = m["d"]
                    if d == 0:
                        continue
                    ren = m["ren"]
                    for t in range(ES[jid], LS[jid] + 1):
                        v = x[(jid, mi, t)]
                        for r in range(nren):
                            a = ren[r]
                            if a:
                                for tau in range(t, t + d):
                                    terms.setdefault((r, tau), []).append((a, v))
            for (r, tau), lst in terms.items():
                model.addConstr(
                    gp.quicksum(a * v for a, v in lst) <= ren_cap[r],
                    name="ren[%d,%d]" % (r, tau))

        # completion / makespan
        msvar = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="makespan")
        delay_parts = []
        const = 0.0
        for p in inst["projects"]:
            pid = p["project_id"]
            sink = sink_of[pid]
            comp = gp.quicksum(
                t * x[(sink, mi, t)]
                for mi in range(len(usable[sink]))
                for t in range(ES[sink], LS[sink] + 1))
            model.addConstr(msvar >= comp, name="ms[%s]" % str(pid))
            delay_parts.append(comp)
            const += p["release_date"] + CP[pid]

        delay_expr = gp.quicksum(delay_parts) - const
        model.setObjectiveN(delay_expr, index=0, priority=2, weight=1,
                            abstol=0, reltol=0, name="delay")
        model.setObjectiveN(msvar, index=1, priority=1, weight=1,
                            abstol=0, reltol=0, name="makespan")

        # warm start with heuristic schedule
        for (jid, mi, t), v in x.items():
            ha = heur_assign.get(jid)
            if (ha is not None and ha[0] == usable[jid][mi]["mode_id"]
                    and ha[1] == t):
                v.Start = 1.0
            else:
                v.Start = 0.0

        # incumbent logger callback
        def callback(m, where):
            if where != GRB.Callback.MIPSOL or logger is None:
                return
            try:
                vals = m.cbGetSolution(var_list)
                assign = {}
                for idx, val in enumerate(vals):
                    if val > 0.5:
                        jid, mi, t = meta[idx]
                        assign[jid] = (usable[jid][mi]["mode_id"], t,
                                       usable[jid][mi]["d"])
                if len(assign) < len(job_ids):
                    return
                d, ms, out = make_solution_dict(inst, assign)
                logger.log_solution(d, out)
            except Exception:
                pass

        model.optimize(callback)

        if model.SolCount > 0:
            vals = model.getAttr("X", var_list)
            assign = {}
            for idx, val in enumerate(vals):
                if val > 0.5:
                    jid, mi, t = meta[idx]
                    assign[jid] = (usable[jid][mi]["mode_id"], t,
                                   usable[jid][mi]["d"])
            if len(assign) == len(job_ids):
                return assign
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = (SolutionLogger(args.log_path, sense="minimize")
              if args.log_path else None)

    t0 = time.time()
    deadline = t0 + max(args.time_limit, 1)

    inst = load_instance(args.instance_path)

    # ---- heuristic phase -------------------------------------------------
    hbudget = min(max(args.time_limit * 0.15, 0.2), 8.0)
    heur_assign, _ = heuristic(inst, time.time() + hbudget)
    if heur_assign is None:      # paranoid fallback
        heur_assign = run_sgs(inst, inst["topo"])

    best_assign = heur_assign
    best_delay, best_ms, best_out = make_solution_dict(inst, best_assign)
    if logger is not None:
        try:
            logger.log_solution(best_delay, best_out)
        except Exception:
            pass

    # ---- MILP improvement ------------------------------------------------
    milp_assign = solve_milp(inst, heur_assign, logger, deadline)
    if milp_assign is not None:
        d, ms, out = make_solution_dict(inst, milp_assign)
        if (d, ms) < (best_delay, best_ms):
            best_delay, best_ms, best_out = d, ms, out

    # ---- write final solution --------------------------------------------
    with open(args.solution_path, "w") as f:
        json.dump(best_out, f)


if __name__ == "__main__":
    main()