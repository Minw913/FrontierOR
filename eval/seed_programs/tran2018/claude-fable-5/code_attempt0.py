import argparse
import json
import time

from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    params = inst["parameters"]
    R = float(params["R"])
    p = int(params["p"])
    nodes = inst["nodes"]
    all_node_ids = [nd["id"] for nd in nodes]
    coord = {nd["id"]: (float(nd["x"]), float(nd["y"])) for nd in nodes}

    # arc distance lookup (both orientations)
    arc_dist = {}
    for a in inst["arcs"]:
        u, v, d = a["from"], a["to"], float(a["distance"])
        arc_dist[(u, v)] = d
        arc_dist[(v, u)] = d

    def edge_len(u, v):
        if (u, v) in arc_dist:
            return arc_dist[(u, v)]
        xu, yu = coord[u]
        xv, yv = coord[v]
        return ((xu - xv) ** 2 + (yu - yv) ** 2) ** 0.5

    od_pairs = inst["od_pairs"]
    EPS = 1e-7

    # Build coverage requirement sets for each OD pair (AC-PC style formulation).
    # For each directed arc on the round trip not covered by the initial full tank,
    # collect the set of nodes that can host a station servicing that arc.
    pair_sets = []      # list of list of frozensets (minimal covering-node sets)
    pair_feasible = []  # False if some arc cannot be covered by any station
    pair_flow = []
    for od in od_pairs:
        path = list(od["shortest_path"])
        flow = float(od["flow"])
        pair_flow.append(flow)
        if len(path) < 2:
            pair_sets.append([])
            pair_feasible.append(True)
            continue
        rt = path + path[-2::-1]  # round trip node sequence
        cum = [0.0]
        for j in range(1, len(rt)):
            cum.append(cum[-1] + edge_len(rt[j - 1], rt[j]))
        sets = []
        feasible = True
        for j in range(1, len(rt)):
            if cum[j] <= R + EPS:
                continue  # covered by initial full tank
            cand = frozenset(rt[i] for i in range(j) if cum[j] - cum[i] <= R + EPS)
            if not cand:
                feasible = False
                break
            sets.append(cand)
        if not feasible:
            pair_sets.append(None)
            pair_feasible.append(False)
            continue
        # keep only minimal sets (dominance pruning)
        uniq = sorted(set(sets), key=len)
        minimal = []
        for s in uniq:
            if not any(m <= s for m in minimal):
                minimal.append(s)
        pair_sets.append(minimal)
        pair_feasible.append(True)

    Q = len(od_pairs)
    useful_nodes = sorted(set().union(*[s for q in range(Q) if pair_feasible[q]
                                        for s in pair_sets[q]])
                          ) if any(pair_feasible[q] and pair_sets[q] for q in range(Q)) else []

    def pad_stations(sts):
        sts = list(sts)
        seen = set(sts)
        for nid in all_node_ids:
            if len(sts) >= p:
                break
            if nid not in seen:
                sts.append(nid)
                seen.add(nid)
        return sts[:p]

    def evaluate(station_set):
        refueled = []
        total = 0.0
        for q in range(Q):
            if not pair_feasible[q]:
                continue
            ok = all(s & station_set for s in pair_sets[q])
            if ok:
                refueled.append(q)
                total += pair_flow[q]
        return total, refueled

    # ---------------- Greedy warm start ----------------
    greedy_sts = set()
    budget = min(p, len(useful_nodes))
    remaining = set(useful_nodes)
    for _ in range(budget):
        best_node, best_gain, best_sets = None, (-1.0, -1), None
        for nid in remaining:
            trial = greedy_sts | {nid}
            gain = 0.0
            sets_cov = 0
            for q in range(Q):
                if not pair_feasible[q]:
                    continue
                cov_all = True
                for s in pair_sets[q]:
                    if s & trial:
                        if nid in s and not (s & greedy_sts):
                            sets_cov += 1
                    else:
                        cov_all = False
                if cov_all:
                    covered_before = all(s & greedy_sts for s in pair_sets[q])
                    if not covered_before:
                        gain += pair_flow[q]
            key = (gain, sets_cov)
            if key > best_gain:
                best_gain = key
                best_node = nid
        if best_node is None:
            break
        greedy_sts.add(best_node)
        remaining.discard(best_node)

    best_obj, best_ref = evaluate(greedy_sts)
    best_stations = pad_stations(sorted(greedy_sts))
    if logger:
        logger.log_solution(best_obj, {
            "objective_value": best_obj,
            "stations": best_stations,
            "refueled_od_pairs": best_ref,
        })

    # ---------------- MIP with Gurobi ----------------
    solved = False
    if useful_nodes:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model("frlm", env=env)
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            tl = max(1.0, args.time_limit - (time.time() - t0) - 2.0)
            model.Params.TimeLimit = tl

            x = {nid: model.addVar(vtype=GRB.BINARY, name=f"x_{nid}") for nid in useful_nodes}
            y = {}
            for q in range(Q):
                if not pair_feasible[q]:
                    continue
                y[q] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                    obj=pair_flow[q], name=f"y_{q}")
            model.ModelSense = GRB.MAXIMIZE

            model.addConstr(gp.quicksum(x.values()) <= min(p, len(useful_nodes)))
            for q in y:
                for s in pair_sets[q]:
                    model.addConstr(y[q] <= gp.quicksum(x[k] for k in s))

            # warm start
            for nid in useful_nodes:
                x[nid].Start = 1.0 if nid in greedy_sts else 0.0

            xvars = [x[nid] for nid in useful_nodes]
            state = {"best": best_obj}

            def cb(m, where):
                if where == GRB.Callback.MIPSOL:
                    vals = m.cbGetSolution(xvars)
                    sts = {useful_nodes[i] for i, v in enumerate(vals) if v > 0.5}
                    obj, ref = evaluate(sts)
                    if obj > state["best"] + 1e-9:
                        state["best"] = obj
                        sol = {
                            "objective_value": obj,
                            "stations": pad_stations(sorted(sts)),
                            "refueled_od_pairs": ref,
                        }
                        if logger:
                            logger.log_solution(obj, sol)

            model.optimize(cb)

            if model.SolCount > 0:
                sts = {nid for nid in useful_nodes if x[nid].X > 0.5}
                obj, ref = evaluate(sts)
                if obj >= best_obj - 1e-9:
                    best_obj = obj
                    best_ref = ref
                    best_stations = pad_stations(sorted(sts))
                solved = True
        except Exception:
            solved = False

    # Final log & output
    if logger:
        logger.log_solution(best_obj, {
            "objective_value": best_obj,
            "stations": best_stations,
            "refueled_od_pairs": best_ref,
        })

    solution = {
        "objective_value": float(best_obj),
        "stations": [int(s) for s in best_stations],
        "refueled_od_pairs": [int(q) for q in best_ref],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()