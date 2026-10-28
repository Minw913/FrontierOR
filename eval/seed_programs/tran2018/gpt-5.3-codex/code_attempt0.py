import argparse
import bisect
import json
import math
import time
from typing import Dict, List, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(description="Refueling station location optimizer")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Max runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path")
    return parser.parse_args()


def build_distance_lookup(instance) -> Tuple[Dict[Tuple[int, int], float], Dict[int, Tuple[float, float]]]:
    edge_dist = {}
    for a in instance["arcs"]:
        u, v, d = int(a["from"]), int(a["to"]), float(a["distance"])
        if u > v:
            u, v = v, u
        edge_dist[(u, v)] = d

    coords = {}
    for nd in instance["nodes"]:
        coords[int(nd["id"])] = (float(nd["x"]), float(nd["y"]))
    return edge_dist, coords


def get_edge_distance(u: int, v: int, edge_dist: Dict[Tuple[int, int], float], coords: Dict[int, Tuple[float, float]]) -> float:
    a, b = (u, v) if u < v else (v, u)
    if (a, b) in edge_dist:
        return edge_dist[(a, b)]
    # Fallback (should rarely be needed if input is consistent)
    x1, y1 = coords[u]
    x2, y2 = coords[v]
    return math.hypot(x1 - x2, y1 - y2)


def preprocess_od(
    od: dict,
    R: float,
    edge_dist: Dict[Tuple[int, int], float],
    coords: Dict[int, Tuple[float, float]],
):
    """
    Returns:
      status: "always", "impossible", or "general"
      arc_cover_sets: list of frozenset(node_ids) for each required arc (deduplicated)
    """
    sp = [int(x) for x in od["shortest_path"]]
    if len(sp) < 2:
        return "always", []

    # One-way segment distances
    seg = [get_edge_distance(sp[i], sp[i + 1], edge_dist, coords) for i in range(len(sp) - 1)]

    # Round-trip route
    rt_nodes = sp + sp[-2::-1]          # e.g., [o,...,d,...,o]
    rt_dists = seg + seg[::-1]          # directed arcs on out + back

    # Cumulative distance at nodes along round-trip
    c = [0.0]
    for d in rt_dists:
        c.append(c[-1] + d)

    num_arcs = len(rt_dists)

    # Arcs requiring station coverage (not covered by initial full tank)
    required_arcs = [j for j in range(num_arcs) if c[j + 1] > R + EPS]
    if not required_arcs:
        return "always", []

    # Positions where each node appears in round-trip node sequence
    pos_by_node: Dict[int, List[int]] = {}
    for idx, node in enumerate(rt_nodes):
        pos_by_node.setdefault(node, []).append(idx)

    arc_cover_sets = []
    for j in required_arcs:
        arc_end_dist = c[j + 1]
        need_from = arc_end_dist - R
        coverers = set()

        # Node can cover arc j if it appears at some position t <= j and c[t] >= need_from
        for node, positions in pos_by_node.items():
            k = bisect.bisect_right(positions, j) - 1
            if k >= 0:
                t = positions[k]
                if c[t] + R + EPS >= arc_end_dist and c[t] + EPS >= need_from:
                    coverers.add(node)

        if not coverers:
            return "impossible", []

        arc_cover_sets.append(frozenset(coverers))

    # Deduplicate identical coverage constraints
    arc_cover_sets = list(set(arc_cover_sets))
    return "general", arc_cover_sets


def solve(instance, time_limit: int, logger: SolutionLogger = None):
    t0 = time.time()

    n = int(instance["parameters"]["n"])
    R = float(instance["parameters"]["R"])
    p = int(instance["parameters"]["p"])
    od_pairs = instance["od_pairs"]

    edge_dist, coords = build_distance_lookup(instance)

    all_node_ids = [int(nd["id"]) for nd in instance["nodes"]]
    all_node_ids_set = set(all_node_ids)

    # Preprocess OD constraints
    od_status = []
    od_cover_sets = []
    flows = []
    for od in od_pairs:
        flows.append(float(od["flow"]))
        st, cov = preprocess_od(od, R, edge_dist, coords)
        od_status.append(st)
        od_cover_sets.append(cov)

    prep_time = time.time() - t0
    remaining = max(1.0, time_limit - prep_time)

    model = gp.Model("alt_fuel_station_location")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = remaining
    model.Params.MIPFocus = 1

    # Variables
    x = {i: model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in all_node_ids}
    y = {k: model.addVar(vtype=GRB.BINARY, name=f"y_{k}") for k in range(len(od_pairs))}

    # Exactly p stations
    model.addConstr(gp.quicksum(x[i] for i in all_node_ids) == p, name="station_count")

    # OD feasibility constraints
    for k in range(len(od_pairs)):
        st = od_status[k]
        if st == "always":
            model.addConstr(y[k] == 1, name=f"od_always_{k}")
        elif st == "impossible":
            model.addConstr(y[k] == 0, name=f"od_impossible_{k}")
        else:
            for r, coverers in enumerate(od_cover_sets[k]):
                valid_coverers = [i for i in coverers if i in all_node_ids_set]
                if not valid_coverers:
                    model.addConstr(y[k] == 0, name=f"od_empty_{k}_{r}")
                else:
                    model.addConstr(
                        y[k] <= gp.quicksum(x[i] for i in valid_coverers),
                        name=f"od_cov_{k}_{r}"
                    )

    model.setObjective(gp.quicksum(flows[k] * y[k] for k in range(len(od_pairs))), GRB.MAXIMIZE)

    # Incumbent logging callback
    model._logger = logger
    model._best_obj_logged = -float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > m._best_obj_logged + 1e-6:
                m._best_obj_logged = obj
                m._logger.log(float(obj))

    model.optimize(cb)

    # Extract best found solution
    stations = []
    refueled_od_pairs = []
    objective_value = 0.0

    if model.SolCount > 0:
        stations = [i for i in all_node_ids if x[i].X > 0.5]
        refueled_od_pairs = [k for k in range(len(od_pairs)) if y[k].X > 0.5]
        objective_value = float(sum(flows[k] for k in refueled_od_pairs))
    else:
        # Fallback (should be rare)
        stations = all_node_ids[:min(p, len(all_node_ids))]
        refueled_od_pairs = []
        objective_value = 0.0

    return {
        "objective_value": objective_value,
        "stations": stations,
        "refueled_od_pairs": refueled_od_pairs,
    }


def main():
    args = parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()