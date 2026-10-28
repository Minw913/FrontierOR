#!/usr/bin/env python3
import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['parameters']['n']
    R = float(inst['parameters']['R'])
    p = int(inst['parameters']['p'])
    od_pairs = inst['od_pairs']
    nodes = inst['nodes']
    node_ids = [nd['id'] for nd in nodes]

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Undirected arc distance lookup
    arc_dist = {}
    for a in inst['arcs']:
        d = float(a['distance'])
        arc_dist[(a['from'], a['to'])] = d
        arc_dist[(a['to'], a['from'])] = d

    eps = 1e-6
    pair_arcs = []          # per q: list of frozensets (candidate node IDs) or None
    feasible_pairs = []

    for q, od in enumerate(od_pairs):
        path = od['shortest_path']
        L = len(path) - 1
        if L <= 0:
            feasible_pairs.append(q)
            pair_arcs.append([])
            continue

        arc_lengths = []
        ok = True
        for i in range(L):
            u, v = path[i], path[i+1]
            if (u, v) in arc_dist:
                arc_lengths.append(arc_dist[(u, v)])
            else:
                ok = False
                break
        if not ok:
            pair_arcs.append(None)
            continue

        # Round-trip arc lengths: forward then reversed
        rt_lengths = arc_lengths + list(reversed(arc_lengths))
        D = [0.0]
        for d in rt_lengths:
            D.append(D[-1] + d)

        def node_at(pos, L=L, path=path):
            if pos <= L:
                return path[pos]
            else:
                return path[2 * L - pos]

        arcs_constraints = []
        feasible = True
        for j in range(1, 2 * L + 1):
            if D[j] <= R + eps:
                continue  # auto-covered by initial tank
            cand_nodes = set()
            for k in range(j):
                if j > L and k == L:
                    continue  # no refuel at destination before return leg
                if D[j] - D[k] <= R + eps:
                    cand_nodes.add(node_at(k))
            if not cand_nodes:
                feasible = False
                break
            arcs_constraints.append(frozenset(cand_nodes))

        if feasible:
            feasible_pairs.append(q)
            pair_arcs.append(arcs_constraints)
        else:
            pair_arcs.append(None)

    # Handle infeasibility of placement
    if p > n or p < 0:
        solution = {"objective_value": 0.0, "stations": [], "refueled_od_pairs": []}
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Build MIP
    model = gp.Model('refuel')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    x = model.addVars(node_ids, vtype=GRB.BINARY, name='x')
    if feasible_pairs:
        y = model.addVars(feasible_pairs, vtype=GRB.BINARY, name='y')
        model.setObjective(
            gp.quicksum(float(od_pairs[q]['flow']) * y[q] for q in feasible_pairs),
            GRB.MAXIMIZE)
    else:
        y = {}
        model.setObjective(0, GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(x[i] for i in node_ids) == p)

    for q in feasible_pairs:
        for cand_set in pair_arcs[q]:
            model.addConstr(y[q] <= gp.quicksum(x[i] for i in cand_set))

    # Greedy warm start: rank nodes by total flow of OD pairs whose candidate
    # sets include the node.
    node_score = {i: 0.0 for i in node_ids}
    for q in feasible_pairs:
        flow = float(od_pairs[q]['flow'])
        all_cands = set()
        for cand_set in pair_arcs[q]:
            all_cands |= set(cand_set)
        for nd in all_cands:
            node_score[nd] += flow
    top_nodes = sorted(node_ids, key=lambda i: -node_score[i])[:p]
    open_set_warm = set(top_nodes)
    for i in node_ids:
        x[i].Start = 1.0 if i in open_set_warm else 0.0
    for q in feasible_pairs:
        covered = all(any(nd in open_set_warm for nd in cs) for cs in pair_arcs[q])
        y[q].Start = 1.0 if covered else 0.0

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol_x = cb_model.cbGetSolution(cb_model._x)
            sol_y = cb_model.cbGetSolution(cb_model._y) if cb_model._feasible_pairs else {}
            stations = [i for i in cb_model._node_ids if sol_x[i] > 0.5]
            if len(stations) < cb_model._p:
                sset = set(stations)
                for i in cb_model._node_ids:
                    if i not in sset and len(stations) < cb_model._p:
                        stations.append(i)
            refueled = [q for q in cb_model._feasible_pairs if sol_y[q] > 0.5]
            sol_dict = {
                "objective_value": float(obj),
                "stations": stations,
                "refueled_od_pairs": refueled,
            }
            if cb_model._logger:
                cb_model._logger.log_solution(float(obj), sol_dict)

    model._x = x
    model._y = y
    model._node_ids = node_ids
    model._feasible_pairs = feasible_pairs
    model._logger = logger
    model._p = p

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 2.0
    model.Params.TimeLimit = max(remaining, 1.0)
    model.optimize(callback)

    # Extract final solution
    if model.SolCount == 0:
        open_set = set(top_nodes)
        refueled = []
        obj_val = 0.0
        for q in feasible_pairs:
            if all(any(nd in open_set for nd in cs) for cs in pair_arcs[q]):
                refueled.append(q)
                obj_val += float(od_pairs[q]['flow'])
        stations = list(top_nodes)
        if len(stations) < p:
            sset = set(stations)
            for i in node_ids:
                if i not in sset and len(stations) < p:
                    stations.append(i)
        solution = {
            "objective_value": obj_val,
            "stations": stations,
            "refueled_od_pairs": refueled,
        }
    else:
        stations = [i for i in node_ids if x[i].X > 0.5]
        if len(stations) > p:
            stations = stations[:p]
        elif len(stations) < p:
            sset = set(stations)
            for i in node_ids:
                if i not in sset and len(stations) < p:
                    stations.append(i)
        refueled = []
        obj_val = 0.0
        for q in feasible_pairs:
            if y[q].X > 0.5:
                refueled.append(q)
                obj_val += float(od_pairs[q]['flow'])
        solution = {
            "objective_value": obj_val,
            "stations": stations,
            "refueled_od_pairs": refueled,
        }

    if logger:
        logger.log_solution(solution['objective_value'], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()