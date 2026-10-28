import argparse
import json
import math
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_graph(activity_ids, precedence_relations):
    preds = {i: [] for i in activity_ids}
    succs = {i: [] for i in activity_ids}
    indeg = {i: 0 for i in activity_ids}
    outdeg = {i: 0 for i in activity_ids}

    for i, j in precedence_relations:
        if i in succs and j in preds:
            succs[i].append(j)
            preds[j].append(i)
            indeg[j] += 1
            outdeg[i] += 1
    return preds, succs, indeg, outdeg


def topo_sort(activity_ids, succs, indeg):
    q = deque([i for i in activity_ids if indeg[i] == 0])
    indeg_work = indeg.copy()
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)
    if len(order) != len(activity_ids):
        # Cycle or malformed graph; fallback to input order
        return list(activity_ids)
    return order


def identify_fictitious_nodes(activity_ids, durations, reqs, indeg, outdeg):
    # Prefer zero-duration, zero-resource nodes among source/sink candidates
    def is_zero_node(a):
        return durations[a] == 0 and all(r == 0 for r in reqs[a])

    sources = [a for a in activity_ids if indeg[a] == 0]
    sinks = [a for a in activity_ids if outdeg[a] == 0]

    source = None
    sink = None

    zero_sources = [a for a in sources if is_zero_node(a)]
    zero_sinks = [a for a in sinks if is_zero_node(a)]

    if zero_sources:
        source = min(zero_sources)
    elif sources:
        source = min(sources)
    else:
        source = min(activity_ids)

    if zero_sinks:
        sink = max(zero_sinks)
    elif sinks:
        sink = max(sinks)
    else:
        sink = max(activity_ids)

    return source, sink


def compute_time_windows(activity_ids, durations, preds, succs, topo, deadline, source):
    es = {a: 0 for a in activity_ids}
    for a in topo:
        if preds[a]:
            es[a] = max(es[p] + durations[p] for p in preds[a])
        else:
            es[a] = 0
    es[source] = 0

    ls = {a: deadline for a in activity_ids}
    for a in reversed(topo):
        if succs[a]:
            ls[a] = min(ls[s] - durations[a] for s in succs[a])
        else:
            ls[a] = min(ls[a], deadline)

    ls[source] = 0

    # Ensure consistency lower bound
    for a in activity_ids:
        if ls[a] < es[a]:
            # keep as is; caller will detect infeasible domain
            pass

    return es, ls


def earliest_schedule(activity_ids, topo, preds, durations):
    s = {a: 0 for a in activity_ids}
    for a in topo:
        if preds[a]:
            s[a] = max(s[p] + durations[p] for p in preds[a])
        else:
            s[a] = 0
    return s


def evaluate_schedule_objective(start_times, activity_ids, durations, reqs, resource_costs, deadline):
    m = len(resource_costs)
    loads = [[0] * deadline for _ in range(m)]

    for a in activity_ids:
        d = durations[a]
        if d <= 0:
            continue
        sa = start_times[a]
        for tau in range(sa, min(sa + d, deadline)):
            ra = reqs[a]
            for k in range(m):
                if ra[k]:
                    loads[k][tau] += ra[k]

    resource_amounts = {}
    obj = 0.0
    for k in range(m):
        rk = max(loads[k]) if deadline > 0 else 0
        resource_amounts[str(k)] = int(rk)
        obj += resource_costs[k] * rk

    return obj, resource_amounts


def solve(instance, time_limit, logger=None):
    activities = instance["activities"]
    precedence_relations = instance["precedence_relations"]
    resource_costs = instance["resource_costs"]
    deadline = int(instance["max_project_duration"])
    m = int(instance["num_resource_types"])

    activity_ids = [a["id"] for a in activities]
    durations = {a["id"]: int(a["duration"]) for a in activities}
    reqs = {a["id"]: list(map(int, a["resource_requirements"])) for a in activities}

    preds, succs, indeg, outdeg = build_graph(activity_ids, precedence_relations)
    topo = topo_sort(activity_ids, succs, indeg)
    source, sink = identify_fictitious_nodes(activity_ids, durations, reqs, indeg, outdeg)

    es, ls = compute_time_windows(activity_ids, durations, preds, succs, topo, deadline, source)

    # Fast fallback schedule (precedence-feasible)
    heur_start = earliest_schedule(activity_ids, topo, preds, durations)
    heur_obj, heur_resource_amounts = evaluate_schedule_objective(
        heur_start, activity_ids, durations, reqs, resource_costs, deadline
    )

    # Build MIP
    model = gp.Model("rcpsp_resource_provision")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1

    # Variables
    S = {}
    x = {}
    for a in activity_ids:
        lb = es[a]
        ub = ls[a]
        if lb > ub:
            # Infeasible window; return fallback (best effort)
            return {
                "objective_value": float(heur_obj),
                "start_times": {str(i): int(heur_start[i]) for i in activity_ids},
                "resource_amounts": heur_resource_amounts,
            }

        S[a] = model.addVar(vtype=GRB.INTEGER, lb=lb, ub=ub, name=f"S_{a}")
        x[a] = {}
        for t in range(lb, ub + 1):
            x[a][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{t}")

    R = {k: model.addVar(vtype=GRB.INTEGER, lb=0, name=f"R_{k}") for k in range(m)}

    model.update()

    # Activity start exactly once + link with start time
    for a in activity_ids:
        model.addConstr(gp.quicksum(x[a].values()) == 1, name=f"start_once_{a}")
        model.addConstr(
            S[a] == gp.quicksum(t * x[a][t] for t in x[a]),
            name=f"link_S_{a}",
        )

    # Project start at 0
    model.addConstr(S[source] == 0, name="project_start_zero")

    # Project end deadline
    model.addConstr(S[sink] <= deadline, name="project_deadline")

    # Precedence constraints
    for i, j in precedence_relations:
        if i in S and j in S:
            model.addConstr(S[j] >= S[i] + durations[i], name=f"prec_{i}_{j}")

    # Resource constraints by period
    # period tau in [0, deadline-1]
    if deadline > 0:
        for k in range(m):
            for tau in range(deadline):
                expr = gp.LinExpr()
                for a in activity_ids:
                    r = reqs[a][k]
                    d = durations[a]
                    if r == 0 or d == 0:
                        continue
                    # active if t <= tau < t + d  => t in [tau-d+1, tau]
                    t_low = max(es[a], tau - d + 1)
                    t_high = min(ls[a], tau)
                    if t_low <= t_high:
                        expr += r * gp.quicksum(x[a][t] for t in range(t_low, t_high + 1))
                model.addConstr(expr <= R[k], name=f"res_{k}_{tau}")

    # Objective
    model.setObjective(gp.quicksum(resource_costs[k] * R[k] for k in range(m)), GRB.MINIMIZE)

    # MIP start from heuristic
    for a in activity_ids:
        hs = heur_start[a]
        if hs < es[a]:
            hs = es[a]
        if hs > ls[a]:
            hs = ls[a]
        S[a].Start = hs
        for t, var in x[a].items():
            var.Start = 1.0 if t == hs else 0.0

    # Compute heuristic resource capacities for MIP start
    hs_obj, hs_R = evaluate_schedule_objective(heur_start, activity_ids, durations, reqs, resource_costs, deadline)
    for k in range(m):
        R[k].Start = hs_R.get(str(k), 0)

    best_logged = [float("inf")]

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < best_logged[0]:
                best_logged[0] = obj_val
                logger.log(float(obj_val))

    model.optimize(cb)

    if model.SolCount > 0:
        start_times = {}
        for a in activity_ids:
            val = S[a].X
            start_times[str(a)] = int(round(val))

        resource_amounts = {}
        for k in range(m):
            resource_amounts[str(k)] = int(round(R[k].X))

        obj = float(model.ObjVal)
        return {
            "objective_value": obj,
            "start_times": start_times,
            "resource_amounts": resource_amounts,
        }

    # If no feasible MIP solution found within time, return fallback heuristic
    return {
        "objective_value": float(hs_obj),
        "start_times": {str(i): int(heur_start[i]) for i in activity_ids},
        "resource_amounts": hs_R,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()