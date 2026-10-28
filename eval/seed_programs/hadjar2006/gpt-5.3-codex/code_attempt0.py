import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_and_solve(instance, time_limit, logger=None):
    depots = instance["depots"]
    tasks = instance["tasks"]
    compatible_pairs = instance["compatible_pairs"]
    depot_to_task_costs = instance["depot_to_task_costs"]
    task_to_depot_costs = instance["task_to_depot_costs"]

    depot_ids = [d["id"] for d in depots]
    task_ids = [t["id"] for t in tasks]
    capacity = {d["id"]: d["capacity"] for d in depots}

    # Cost dictionaries for reconstruction/output
    po_cost = {}  # (depot, task) -> cost
    pi_cost = {}  # (task, depot) -> cost
    tt_cost = {}  # (i, j) -> cost
    compat_list = []  # list of (i, j, cost)

    for e in compatible_pairs:
        i = e["task_i"]
        j = e["task_j"]
        c = float(e["cost"])
        tt_cost[(i, j)] = c
        compat_list.append((i, j, c))

    for e in depot_to_task_costs:
        po_cost[(e["depot_id"], e["task_id"])] = float(e["cost"])

    for e in task_to_depot_costs:
        pi_cost[(e["task_id"], e["depot_id"])] = float(e["cost"])

    model = gp.Model("multi_depot_vehicle_scheduling")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Variables
    x_po = {}  # depot -> first task
    x_pi = {}  # last task -> depot
    x_tt = {}  # depot-specific task->task arc

    incoming_total = defaultdict(list)      # task -> list(vars)
    outgoing_total = defaultdict(list)      # task -> list(vars)
    incoming_by_depot = defaultdict(list)   # (depot, task) -> list(vars)
    outgoing_by_depot = defaultdict(list)   # (depot, task) -> list(vars)
    depot_starts = defaultdict(list)        # depot -> list(vars)
    depot_ends = defaultdict(list)          # depot -> list(vars)

    # Depot -> task arcs
    for (d, t), c in po_cost.items():
        v = model.addVar(vtype=GRB.BINARY, obj=c, name=f"po_{d}_{t}")
        x_po[(d, t)] = v
        incoming_total[t].append(v)
        incoming_by_depot[(d, t)].append(v)
        depot_starts[d].append(v)

    # Task -> depot arcs
    for (t, d), c in pi_cost.items():
        v = model.addVar(vtype=GRB.BINARY, obj=c, name=f"pi_{t}_{d}")
        x_pi[(t, d)] = v
        outgoing_total[t].append(v)
        outgoing_by_depot[(d, t)].append(v)
        depot_ends[d].append(v)

    # Task -> task arcs (replicated by depot/commodity)
    for d in depot_ids:
        for (i, j, c) in compat_list:
            v = model.addVar(vtype=GRB.BINARY, obj=c, name=f"tt_{d}_{i}_{j}")
            x_tt[(d, i, j)] = v
            outgoing_total[i].append(v)
            incoming_total[j].append(v)
            outgoing_by_depot[(d, i)].append(v)
            incoming_by_depot[(d, j)].append(v)

    model.update()

    # 1) Every task has exactly one predecessor arc (depot->task or task->task)
    for t in task_ids:
        vars_in = incoming_total[t]
        if not vars_in:
            # Impossible instance (no way to enter task)
            model.addConstr(0 == 1, name=f"infeas_in_{t}")
        else:
            model.addConstr(gp.quicksum(vars_in) == 1, name=f"cover_in_{t}")

    # 2) Every task has exactly one successor arc (task->task or task->depot)
    for t in task_ids:
        vars_out = outgoing_total[t]
        if not vars_out:
            # Impossible instance (no way to leave task)
            model.addConstr(0 == 1, name=f"infeas_out_{t}")
        else:
            model.addConstr(gp.quicksum(vars_out) == 1, name=f"cover_out_{t}")

    # 3) Per-depot flow conservation at each task
    for d in depot_ids:
        for t in task_ids:
            model.addConstr(
                gp.quicksum(incoming_by_depot[(d, t)]) - gp.quicksum(outgoing_by_depot[(d, t)]) == 0,
                name=f"flow_{d}_{t}",
            )

    # 4) Depot capacity on number of dispatched vehicles
    for d in depot_ids:
        model.addConstr(gp.quicksum(depot_starts[d]) <= capacity[d], name=f"cap_{d}")
        # Optional explicit balance starts == ends (usually implied, but helpful numerically)
        model.addConstr(gp.quicksum(depot_starts[d]) == gp.quicksum(depot_ends[d]), name=f"balance_{d}")

    # Incumbent logging callback
    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._best_logged_obj:
                m._best_logged_obj = obj
                logger.log(float(obj))

    model._best_logged_obj = math.inf
    model.optimize(cb if logger is not None else None)

    if model.SolCount == 0:
        return {"objective_value": 1e30, "routes": []}

    # Reconstruct routes
    selected_po = defaultdict(list)      # depot -> list of start tasks
    selected_succ = {}                   # (depot, task_i) -> task_j
    selected_pi = set()                  # (task, depot)

    for (d, t), v in x_po.items():
        if v.X > 0.5:
            selected_po[d].append(t)

    for (d, i, j), v in x_tt.items():
        if v.X > 0.5:
            selected_succ[(d, i)] = j

    for (t, d), v in x_pi.items():
        if v.X > 0.5:
            selected_pi.add((t, d))

    routes = []
    for d in depot_ids:
        for start in selected_po[d]:
            seq = []
            cur = start
            guard = 0
            while True:
                seq.append(cur)
                guard += 1
                if guard > len(task_ids) + 5:
                    break
                nxt = selected_succ.get((d, cur))
                if nxt is None:
                    break
                cur = nxt

            # Compute route cost from selected arcs
            if not seq:
                continue

            rcost = 0.0
            rcost += po_cost.get((d, seq[0]), 0.0)
            for a, b in zip(seq[:-1], seq[1:]):
                rcost += tt_cost.get((a, b), 0.0)
            rcost += pi_cost.get((seq[-1], d), 0.0)

            routes.append(
                {
                    "depot": d,
                    "tasks": seq,
                    "cost": float(rcost),
                }
            )

    solution = {
        "objective_value": float(model.ObjVal),
        "routes": routes,
    }
    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve multi-depot vehicle scheduling problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()