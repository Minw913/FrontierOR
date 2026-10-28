import argparse
import bisect
import json
import time
from math import hypot


def main():
    parser = argparse.ArgumentParser(description="Refueling station location solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    time_limit = float(args.time_limit)

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    params = inst["parameters"]
    R = float(params["R"])
    p = int(params["p"])
    od_pairs = inst.get("od_pairs", [])
    nodes = inst.get("nodes", [])
    node_ids = [nd["id"] for nd in nodes]
    coord = {nd["id"]: (float(nd["x"]), float(nd["y"])) for nd in nodes}

    arc_dist = {}
    for a in inst.get("arcs", []):
        u, v = a["from"], a["to"]
        key = (u, v) if u <= v else (v, u)
        arc_dist[key] = float(a["distance"])

    TOL = 1e-9 * max(1.0, R)

    def seg_dist(u, v):
        key = (u, v) if u <= v else (v, u)
        d = arc_dist.get(key)
        if d is None:
            x1, y1 = coord[u]
            x2, y2 = coord[v]
            d = hypot(x1 - x2, y1 - y2)
        return d

    # ------------------------------------------------------------------
    # Preprocessing: build the round-trip path of every OD pair.
    # A directed arc spanning cumulative positions [a, b] needs a station
    # iff b > R.  A node occurrence at cumulative position s can service
    # that arc iff b - R <= s <= a  (window [b - R, a]).
    # ------------------------------------------------------------------
    K = len(od_pairs)
    flows = [float(od.get("flow", 0.0)) for od in od_pairs]
    pair_occ = []         # per pair: sorted list of (cumulative position, node id)
    pair_windows = []     # per pair: list of (lo, hi) windows needing a station
    pair_cover_sets = []  # per pair: list of node lists covering each window
    node_incidences = {}  # node -> list of (pair, window_index) it covers

    for k in range(K):
        path = od_pairs[k]["shortest_path"]
        L = len(path) - 1
        d = [seg_dist(path[i], path[i + 1]) for i in range(L)]
        pos = [0.0] * (L + 1)
        for i in range(L):
            pos[i + 1] = pos[i] + d[i]
        D = pos[L]

        # node occurrences along the round trip (outbound then inbound)
        occ = [(pos[j], path[j]) for j in range(L + 1)]
        for j in range(L - 1, -1, -1):
            occ.append((2.0 * D - pos[j], path[j]))
        occ.sort(key=lambda t: t[0])
        positions = [o[0] for o in occ]

        # directed arcs of the round trip as cumulative spans [a, b]
        spans = [(pos[j], pos[j + 1]) for j in range(L)]
        cum = D
        for j in range(L - 1, -1, -1):
            spans.append((cum, cum + d[j]))
            cum += d[j]

        windows = []
        cover_sets = []
        for a, b in spans:
            if b > R + TOL:  # not covered by the initial full tank
                lo, hi = b - R, a
                windows.append((lo, hi))
                lidx = bisect.bisect_left(positions, lo - TOL)
                ridx = bisect.bisect_right(positions, hi + TOL)
                seen = set()
                ns = []
                for t in range(lidx, ridx):
                    v = occ[t][1]
                    if v not in seen:
                        seen.add(v)
                        ns.append(v)
                cover_sets.append(ns)

        pair_occ.append(occ)
        pair_windows.append(windows)
        pair_cover_sets.append(cover_sets)
        for j, ns in enumerate(cover_sets):
            for v in ns:
                node_incidences.setdefault(v, []).append((k, j))

    # ------------------------------------------------------------------
    # Evaluation: OD pair refueled iff every window contains a station
    # occurrence.
    # ------------------------------------------------------------------
    def evaluate(stations_set):
        obj = 0.0
        covered = []
        for k in range(K):
            wins = pair_windows[k]
            if not wins:
                covered.append(k)
                obj += flows[k]
                continue
            occ = pair_occ[k]
            sp = sorted(o[0] for o in occ if o[1] in stations_set)
            ok = True
            for lo, hi in wins:
                i = bisect.bisect_left(sp, lo - TOL)
                if i >= len(sp) or sp[i] > hi + TOL:
                    ok = False
                    break
            if ok:
                covered.append(k)
                obj += flows[k]
        return obj, covered

    # ------------------------------------------------------------------
    # Greedy warm start (time-budgeted).
    # ------------------------------------------------------------------
    selected = []
    selected_set = set()
    greedy_budget = min(0.20 * time_limit, 30.0)
    if p > 0 and K > 0 and node_incidences:
        uncovered = [len(w) for w in pair_windows]
        window_covered = [[False] * len(w) for w in pair_windows]
        cand_nodes = list(node_incidences.keys())
        for _ in range(min(p, len(node_ids))):
            if time.time() - t0 > greedy_budget:
                break
            best_v = -1
            best_gain = 1e-12
            cnt = {}
            timed_out = False
            for idx, v in enumerate(cand_nodes):
                if (idx & 1023) == 0 and time.time() - t0 > greedy_budget:
                    timed_out = True
                    break
                if v in selected_set:
                    continue
                cnt.clear()
                for (k, j) in node_incidences[v]:
                    if uncovered[k] > 0 and not window_covered[k][j]:
                        cnt[k] = cnt.get(k, 0) + 1
                gain = 0.0
                for k, c in cnt.items():
                    if c == uncovered[k]:
                        gain += flows[k]
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if timed_out or best_v < 0:
                break
            selected.append(best_v)
            selected_set.add(best_v)
            for (k, j) in node_incidences[best_v]:
                if not window_covered[k][j]:
                    window_covered[k][j] = True
                    uncovered[k] -= 1

    # Pad to exactly p stations (coverage is monotone in the station set).
    if len(selected) < p:
        for v in node_ids:
            if len(selected) >= p:
                break
            if v not in selected_set:
                selected.append(v)
                selected_set.add(v)
    elif len(selected) > p:
        selected = selected[:p]
        selected_set = set(selected)

    best_obj, best_pairs = evaluate(selected_set)
    best = {"obj": best_obj, "stations": sorted(selected_set), "pairs": list(best_pairs)}

    def make_solution(obj, stations, pairs):
        return {
            "objective_value": float(obj),
            "stations": [int(v) for v in stations],
            "refueled_od_pairs": [int(k) for k in pairs],
        }

    if logger:
        logger.log_solution(best["obj"], make_solution(best["obj"], best["stations"], best["pairs"]))

    # ------------------------------------------------------------------
    # Exact MIP (maximum covering) via Gurobi, warm-started by the greedy.
    # ------------------------------------------------------------------
    remaining = time_limit - (time.time() - t0) - 1.0
    if remaining > 0.5 and p <= len(node_ids):
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("frlm")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = remaining

            x = model.addVars(node_ids, vtype=GRB.BINARY, name="x")
            y = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")
            model.addConstr(x.sum() == float(p), name="num_stations")
            for k in range(K):
                cs = pair_cover_sets[k]
                if not cs:  # whole round trip within the initial range
                    y[k].LB = 1.0
                    continue
                for ns in cs:
                    if ns:
                        model.addConstr(y[k] <= gp.quicksum(x[v] for v in ns))
                    else:  # window can never be serviced
                        y[k].UB = 0.0
            model.setObjective(
                gp.quicksum(flows[k] * y[k] for k in range(K)), GRB.MAXIMIZE
            )

            # MIP start from the greedy solution
            for v in selected_set:
                x[v].Start = 1.0
            covered_set = set(best["pairs"])
            for k in range(K):
                y[k].Start = 1.0 if k in covered_set else 0.0

            def callback(m, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        xv = m.cbGetSolution(x)
                        st = set(v for v in node_ids if xv[v] > 0.5)
                        obj, cov = evaluate(st)
                        if obj > best["obj"] + 1e-9 * max(1.0, abs(best["obj"])):
                            best["obj"] = obj
                            best["stations"] = sorted(st)
                            best["pairs"] = list(cov)
                            if logger:
                                logger.log_solution(
                                    obj,
                                    make_solution(obj, best["stations"], best["pairs"]),
                                )
                    except Exception:
                        pass

            model.optimize(callback)

            if model.SolCount > 0:
                try:
                    xv = model.getAttr("X", x)
                    st = set(v for v in node_ids if xv[v] > 0.5)
                    obj, cov = evaluate(st)
                    if obj > best["obj"] + 1e-9 * max(1.0, abs(best["obj"])):
                        best["obj"] = obj
                        best["stations"] = sorted(st)
                        best["pairs"] = list(cov)
                        if logger:
                            logger.log_solution(
                                obj,
                                make_solution(obj, best["stations"], best["pairs"]),
                            )
                except Exception:
                    pass
            try:
                model.dispose()
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Write the best feasible solution found.
    # ------------------------------------------------------------------
    solution = make_solution(best["obj"], best["stations"], best["pairs"])
    with open(args.solution_path, "w") as fh:
        json.dump(solution, fh)


if __name__ == "__main__":
    main()