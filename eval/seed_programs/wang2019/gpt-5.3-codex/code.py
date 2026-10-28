import argparse
import json
import math
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


class IncumbentCallback:
    def __init__(self, logger=None):
        self.logger = logger
        self.best_obj = float("inf")

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < self.best_obj:
                self.best_obj = obj
                if self.logger:
                    self.logger.log(float(obj))


def bfs_reachable(start, adj):
    seen = set()
    dq = deque([start])
    seen.add(start)
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen


def solve(instance, time_limit, logger=None):
    nodes = instance["nodes"]
    arcs = instance["arcs"]
    commodities = instance["commodities"]
    fleet_types = instance["fleet_types"]

    num_arcs = len(arcs)
    num_comms = len(commodities)
    num_fleets = len(fleet_types)

    # Arc data indexed by arc position
    arc_from = [a["from"] for a in arcs]
    arc_to = [a["to"] for a in arcs]
    arc_cost = [a["flow_cost"] for a in arcs]

    # Build adjacency for reachability pruning
    out_neighbors = defaultdict(list)
    rev_neighbors = defaultdict(list)
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for a_idx, a in enumerate(arcs):
        u, v = a["from"], a["to"]
        out_neighbors[u].append(v)
        rev_neighbors[v].append(u)
        out_arcs[u].append(a_idx)
        in_arcs[v].append(a_idx)

    # Commodity data
    comm_origin = [c["origin"] for c in commodities]
    comm_dest = [c["destination"] for c in commodities]
    comm_demand = [c["demand"] for c in commodities]

    # Fleet data
    fleet_cap = [f["capacity"] for f in fleet_types]
    fleet_fixed = [f["fixed_cost"] for f in fleet_types]

    total_demand = sum(comm_demand)

    # Reachability-based pruning of x variables
    # allowed_arcs_by_k[k] = list of arc indices usable by commodity k
    allowed_arcs_by_k = []
    comms_on_arc = [[] for _ in range(num_arcs)]
    out_allowed = [defaultdict(list) for _ in range(num_comms)]
    in_allowed = [defaultdict(list) for _ in range(num_comms)]

    for k in range(num_comms):
        o = comm_origin[k]
        d = comm_dest[k]
        reach_from_o = bfs_reachable(o, out_neighbors)
        reach_to_d = bfs_reachable(d, rev_neighbors)  # nodes that can reach d in original graph

        allowed = []
        for a_idx in range(num_arcs):
            u = arc_from[a_idx]
            v = arc_to[a_idx]
            if u in reach_from_o and v in reach_to_d:
                allowed.append(a_idx)
                comms_on_arc[a_idx].append(k)
                out_allowed[k][u].append(a_idx)
                in_allowed[k][v].append(a_idx)
        allowed_arcs_by_k.append(allowed)

    model = gp.Model("FreightServiceNetwork")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))

    # Decision variables
    x = {}  # (k, a_idx) -> flow
    for k in range(num_comms):
        for a_idx in allowed_arcs_by_k[k]:
            x[(k, a_idx)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{k}_{a_idx}")

    y = {}  # (f, a_idx) -> #vehicles
    for f in range(num_fleets):
        cap = fleet_cap[f]
        ub = math.ceil(total_demand / cap) if cap > 0 else 0
        for a_idx in range(num_arcs):
            y[(f, a_idx)] = model.addVar(lb=0.0, ub=ub, vtype=GRB.INTEGER, name=f"y_{f}_{a_idx}")

    # Objective
    flow_obj = gp.quicksum(arc_cost[a_idx] * x[(k, a_idx)] for (k, a_idx) in x.keys())
    veh_obj = gp.quicksum(fleet_fixed[f] * y[(f, a_idx)] for f in range(num_fleets) for a_idx in range(num_arcs))
    model.setObjective(flow_obj + veh_obj, GRB.MINIMIZE)

    # Flow balance constraints per commodity/node
    for k in range(num_comms):
        o = comm_origin[k]
        d = comm_dest[k]
        dem = comm_demand[k]
        for n in nodes:
            rhs = dem if n == o else (-dem if n == d else 0)
            out_expr = gp.quicksum(x[(k, a_idx)] for a_idx in out_allowed[k].get(n, []))
            in_expr = gp.quicksum(x[(k, a_idx)] for a_idx in in_allowed[k].get(n, []))
            model.addConstr(out_expr - in_expr == rhs, name=f"flow_bal_k{k}_n{n}")

    # Capacity constraints per arc
    for a_idx in range(num_arcs):
        lhs = gp.quicksum(x[(k, a_idx)] for k in comms_on_arc[a_idx] if (k, a_idx) in x)
        rhs = gp.quicksum(fleet_cap[f] * y[(f, a_idx)] for f in range(num_fleets))
        model.addConstr(lhs <= rhs, name=f"cap_a{a_idx}")

    # Vehicle balance constraints per fleet/node
    for f in range(num_fleets):
        for n in nodes:
            in_expr = gp.quicksum(y[(f, a_idx)] for a_idx in in_arcs.get(n, []))
            out_expr = gp.quicksum(y[(f, a_idx)] for a_idx in out_arcs.get(n, []))
            model.addConstr(in_expr == out_expr, name=f"veh_bal_f{f}_n{n}")

    model.update()

    cb = IncumbentCallback(logger=logger)
    model.optimize(cb)

    # Build output dictionaries (include all required key patterns over commodities/arcs and fleets/arcs)
    flows_out = {}
    vehicles_out = {}

    feasible = model.SolCount > 0
    if feasible:
        obj_val = float(model.ObjVal)
        # Ensure final incumbent logged (in case callback missed for any reason)
        if logger and (obj_val + 1e-9 < cb.best_obj):
            logger.log(obj_val)
    else:
        obj_val = 1e30  # fallback for infeasible/no-solution within time

    # Output keys use from/to as required: x_{commodity}_{from}_{to}, y_{fleet}_{from}_{to}
    # If parallel arcs exist, values are aggregated under same key.
    for k, c in enumerate(commodities):
        cid = c["id"]
        for a_idx, a in enumerate(arcs):
            key = f"x_{cid}_{a['from']}_{a['to']}"
            val = 0.0
            if feasible and (k, a_idx) in x:
                val = float(x[(k, a_idx)].X)
            flows_out[key] = flows_out.get(key, 0.0) + val

    for f, ft in enumerate(fleet_types):
        fid = ft["id"]
        for a_idx, a in enumerate(arcs):
            key = f"y_{fid}_{a['from']}_{a['to']}"
            val = 0.0
            if feasible:
                val = float(y[(f, a_idx)].X)
                # remove tiny numerical noise
                if abs(val) < 1e-9:
                    val = 0.0
            vehicles_out[key] = vehicles_out.get(key, 0.0) + val

    return {
        "objective_value": obj_val,
        "flows": flows_out,
        "vehicles": vehicles_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve freight service network design and routing problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()