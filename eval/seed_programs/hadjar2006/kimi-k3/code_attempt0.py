import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------------- Load instance ----------------
    tasks = inst.get("tasks", [])
    depots = inst.get("depots", [])
    n = len(tasks)
    D = len(depots)

    task_ids = [t["id"] for t in tasks]
    task_index = {tid: i for i, tid in enumerate(task_ids)}
    depot_ids = [d["id"] for d in depots]
    depot_index = {did: d for d, did in enumerate(depot_ids)}
    capacities = [d.get("capacity", n) for d in depots]
    start_times = [t.get("start_time", 0.0) for t in tasks]

    # Compatible trip-trip arcs
    base_pairs = []          # [(i, j)]
    pair_cost = []           # aligned cost
    succ_b = [[] for _ in range(n)]
    pred_b = [[] for _ in range(n)]
    comp_cost = {}
    for p in inst.get("compatible_pairs", []):
        try:
            i = task_index[p["task_i"]]
            j = task_index[p["task_j"]]
        except KeyError:
            continue
        if (i, j) in comp_cost:
            continue
        comp_cost[(i, j)] = p["cost"]
        b = len(base_pairs)
        base_pairs.append((i, j))
        pair_cost.append(p["cost"])
        succ_b[i].append(b)
        pred_b[j].append(b)
    B = len(base_pairs)
    pair_b = {base_pairs[b]: b for b in range(B)}

    po_cost = {}
    for e in inst.get("depot_to_task_costs", []):
        try:
            d = depot_index[e["depot_id"]]
            i = task_index[e["task_id"]]
        except KeyError:
            continue
        po_cost[(d, i)] = e["cost"]

    pi_cost = {}
    for e in inst.get("task_to_depot_costs", []):
        try:
            d = depot_index[e["depot_id"]]
            i = task_index[e["task_id"]]
        except KeyError:
            continue
        pi_cost[(i, d)] = e["cost"]

    active_depots = [d for d in range(D) if capacities[d] > 0]
    dpos = {d: p for p, d in enumerate(active_depots)}

    # ---------------- Greedy feasible solution (warm start + fallback) ----------------
    def greedy():
        order = sorted(range(n), key=lambda i: (start_times[i], i))
        chains = []          # each: [depot, task list, last task]
        endpoint = {}        # task -> chain id (chains' current endpoint)
        used = [0] * D
        arcs = set()
        covered = 0
        for t in order:
            best_ch = -1
            best_c = None
            best_prev = -1
            for b in pred_b[t]:
                h = base_pairs[b][0]
                ch = endpoint.get(h)
                if ch is not None:
                    c = pair_cost[b]
                    if best_c is None or c < best_c:
                        best_c = c
                        best_ch = ch
                        best_prev = h
            if best_ch >= 0:
                chain = chains[best_ch]
                d = chain[0]
                arcs.add(("c", d, best_prev, t))
                chain[1].append(t)
                del endpoint[best_prev]
                endpoint[t] = best_ch
                chain[2] = t
                covered += 1
            else:
                bd = -1
                bc = None
                for d in active_depots:
                    if used[d] < capacities[d]:
                        c = po_cost.get((d, t))
                        if c is not None and (bc is None or c < bc):
                            bc = c
                            bd = d
                if bd < 0 and active_depots:
                    for d in active_depots:
                        c = po_cost.get((d, t))
                        if c is not None and (bc is None or c < bc):
                            bc = c
                            bd = d
                if bd < 0:
                    continue
                used[bd] += 1
                cid = len(chains)
                chains.append([bd, [t], t])
                endpoint[t] = cid
                arcs.add(("p", bd, t))
                covered += 1
        routes = []
        obj = 0.0
        feasible = (covered == n) and all(used[d] <= capacities[d] for d in range(D))
        for d, seq, last in chains:
            arcs.add(("r", last, d))
            cost = po_cost.get((d, seq[0]), 0.0)
            for k in range(len(seq) - 1):
                cost += comp_cost.get((seq[k], seq[k + 1]), 0.0)
            cost += pi_cost.get((last, d), 0.0)
            routes.append({"depot": depot_ids[d],
                           "tasks": [task_ids[t] for t in seq],
                           "cost": float(cost)})
            obj += cost
        return {"objective_value": float(obj), "routes": routes}, arcs, feasible

    def build_solution_from(pos_arcs, obj_val):
        next_map = {}
        pullouts = []
        for a in pos_arcs:
            k = a[0]
            if k == "c":
                next_map[(a[1], a[2])] = a[3]
            elif k == "p":
                pullouts.append((a[1], a[2]))
        routes = []
        total = 0.0
        for d, i0 in pullouts:
            seq = [i0]
            seen = {i0}
            cur = i0
            while (d, cur) in next_map:
                cur = next_map[(d, cur)]
                if cur in seen:
                    break
                seen.add(cur)
                seq.append(cur)
            last = seq[-1]
            cost = po_cost.get((d, seq[0]), 0.0)
            for k in range(len(seq) - 1):
                cost += comp_cost.get((seq[k], seq[k + 1]), 0.0)
            cost += pi_cost.get((last, d), 0.0)
            routes.append({"depot": depot_ids[d],
                           "tasks": [task_ids[t] for t in seq],
                           "cost": float(cost)})
            total += cost
        val = obj_val if obj_val is not None else total
        return {"objective_value": float(val), "routes": routes}, float(total)

    greedy_sol, greedy_arcs, greedy_feasible = greedy()
    final_sol = greedy_sol
    state = {"last": greedy_sol["objective_value"] if greedy_feasible else float("inf")}
    if logger and greedy_feasible:
        try:
            logger.log_solution(greedy_sol["objective_value"], greedy_sol)
        except Exception:
            pass

    # ---------------- Exact MIP: multi-commodity (per-depot) min-cost flow ----------------
    time_left = args.time_limit - (time.time() - t_start)
    if time_left > 1 and active_depots:
        try:
            m = gp.Model("mdvsp")
            m.setParam("Seed", 0)
            m.setParam("MIPGap", 1e-4)
            m.setParam("NumericFocus", 0)
            m.setParam("Threads", 1)
            m.ModelSense = GRB.MINIMIZE

            P = len(active_depots)

            # Arc variables (binary; flows decompose into routes)
            comp_var_list = []   # per depot position: list over base pair index
            comp_flat = []
            if B > 0:
                for _p in range(P):
                    vs = m.addVars(B, vtype=GRB.BINARY, obj=pair_cost)
                    row = [vs[b] for b in range(B)]
                    comp_var_list.append(row)
                    comp_flat.extend(row)

            po_var_list = []     # per depot position: tupledict task -> var
            po_set_list = []
            po_flat = []
            po_meta = []
            po_pos_index = {}
            for p, d in enumerate(active_depots):
                keys = [i for i in range(n) if (d, i) in po_cost]
                objd = {i: po_cost[(d, i)] for i in keys}
                vs = m.addVars(keys, vtype=GRB.BINARY, obj=objd) if keys else m.addVars([])
                po_var_list.append(vs)
                po_set_list.append(set(keys))
                for i in keys:
                    po_pos_index[(d, i)] = len(po_flat)
                    po_flat.append(vs[i])
                    po_meta.append((d, i))

            pi_var_list = []
            pi_set_list = []
            pi_flat = []
            pi_meta = []
            pi_pos_index = {}
            for p, d in enumerate(active_depots):
                keys = [i for i in range(n) if (i, d) in pi_cost]
                objd = {i: pi_cost[(i, d)] for i in keys}
                vs = m.addVars(keys, vtype=GRB.BINARY, obj=objd) if keys else m.addVars([])
                pi_var_list.append(vs)
                pi_set_list.append(set(keys))
                for i in keys:
                    pi_pos_index[(i, d)] = len(pi_flat)
                    pi_flat.append(vs[i])
                    pi_meta.append((i, d))

            # Coverage: every trip served exactly once
            for i in range(n):
                terms = []
                for p in range(P):
                    if i in po_set_list[p]:
                        terms.append(po_var_list[p][i])
                preds = pred_b[i]
                if preds:
                    for p in range(P):
                        cv = comp_var_list[p]
                        for b in preds:
                            terms.append(cv[b])
                m.addConstr(gp.quicksum(terms) == 1.0)

            # Flow conservation per depot per trip
            for p in range(P):
                cv = comp_var_list[p]
                pov = po_var_list[p]
                piv = pi_var_list[p]
                pos_ = po_set_list[p]
                pis_ = pi_set_list[p]
                for i in range(n):
                    inflow = []
                    if i in pos_:
                        inflow.append(pov[i])
                    for b in pred_b[i]:
                        inflow.append(cv[b])
                    outflow = []
                    if i in pis_:
                        outflow.append(piv[i])
                    for b in succ_b[i]:
                        outflow.append(cv[b])
                    if inflow or outflow:
                        m.addConstr(gp.quicksum(inflow) - gp.quicksum(outflow) == 0.0)

            # Depot balance and capacity
            for p, d in enumerate(active_depots):
                pouts = gp.quicksum(po_var_list[p])
                pins = gp.quicksum(pi_var_list[p])
                m.addConstr(pouts - pins == 0.0)
                if capacities[d] < n:
                    m.addConstr(pouts <= capacities[d])

            # Warm start with the greedy solution
            all_flat = comp_flat + po_flat + pi_flat
            if greedy_feasible:
                try:
                    vals = [0.0] * len(all_flat)
                    for a in greedy_arcs:
                        if a[0] == "c":
                            _, d, i, j = a
                            bb = pair_b.get((i, j))
                            p = dpos.get(d)
                            if bb is not None and p is not None:
                                vals[p * B + bb] = 1.0
                        elif a[0] == "p":
                            _, d, i = a
                            idx = po_pos_index.get((d, i))
                            if idx is not None:
                                vals[len(comp_flat) + idx] = 1.0
                        else:
                            _, i, d = a
                            idx = pi_pos_index.get((i, d))
                            if idx is not None:
                                vals[len(comp_flat) + len(po_flat) + idx] = 1.0
                    m.setAttr("Start", all_flat, vals)
                except Exception:
                    pass

            remaining = args.time_limit - (time.time() - t_start)
            m.setParam("TimeLimit", max(1.0, remaining))

            def callback(model, where):
                if where == GRB.Callback.MIPSOL and logger is not None:
                    try:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if obj < state["last"] - 1e-9:
                            pos = set()
                            if B > 0:
                                cvals = model.cbGetSolution(comp_flat)
                                for idx, v in enumerate(cvals):
                                    if v > 0.5:
                                        p = idx // B
                                        b = idx % B
                                        d = active_depots[p]
                                        i, j = base_pairs[b]
                                        pos.add(("c", d, i, j))
                            for vals_list, meta_list, tag in (
                                    (model.cbGetSolution(po_flat), po_meta, "p"),
                                    (model.cbGetSolution(pi_flat), pi_meta, "r")):
                                for idx, v in enumerate(vals_list):
                                    if v > 0.5:
                                        a, b = meta_list[idx]
                                        if tag == "p":
                                            pos.add(("p", a, b))
                                        else:
                                            pos.add(("r", a, b))
                            sol, _tot = build_solution_from(pos, obj)
                            state["last"] = obj
                            logger.log_solution(obj, sol)
                    except Exception:
                        pass

            if logger:
                m.optimize(callback)
            else:
                m.optimize()

            if m.SolCount > 0:
                obj = m.ObjVal
                pos = set()
                if B > 0:
                    cvals = m.getAttr("X", comp_flat)
                    for idx, v in enumerate(cvals):
                        if v > 0.5:
                            p = idx // B
                            b = idx % B
                            d = active_depots[p]
                            i, j = base_pairs[b]
                            pos.add(("c", d, i, j))
                for vals_list, meta_list, tag in (
                        (m.getAttr("X", po_flat), po_meta, "p"),
                        (m.getAttr("X", pi_flat), pi_meta, "r")):
                    for idx, v in enumerate(vals_list):
                        if v > 0.5:
                            a, b = meta_list[idx]
                            if tag == "p":
                                pos.add(("p", a, b))
                            else:
                                pos.add(("r", a, b))
                sol, total = build_solution_from(pos, obj)
                sol["objective_value"] = total
                if total < final_sol["objective_value"] - 1e-9:
                    final_sol = sol
                if logger and total < state["last"] - 1e-9:
                    try:
                        logger.log_solution(total, sol)
                    except Exception:
                        pass
        except Exception:
            pass

    # ---------------- Write final solution ----------------
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()