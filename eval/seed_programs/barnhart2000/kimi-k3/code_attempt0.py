import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    nodes = inst["network"]["nodes"]
    arcs = inst["network"]["arcs"]
    commodities = inst["commodities"]["commodity_list"]

    A = len(arcs)
    K = len(commodities)

    out_arcs = {n: [] for n in nodes}
    in_arcs = {n: [] for n in nodes}
    for a, arc in enumerate(arcs):
        out_arcs[arc["from_node"]].append(a)
        in_arcs[arc["to_node"]].append(a)

    demand = [c["demand"] for c in commodities]
    art_cost = [
        c.get("artificial_arc_cost", c.get("revenue", 0)) for c in commodities
    ]
    arc_cost = [arc.get("cost", 0) for arc in arcs]
    arc_cap = [arc["capacity"] for arc in arcs]

    def all_rejected_solution():
        return {
            "objective_value": float(sum(art_cost)),
            "commodities": [
                {
                    "commodity_id": c["commodity_id"],
                    "rejected": True,
                    "path_arcs": [],
                }
                for c in commodities
            ],
        }

    def extract_path(k, xv):
        """Extract a simple origin->destination path from the selected arcs."""
        c = commodities[k]
        o, d = c["origin"], c["destination"]
        if o == d:
            return []
        sel = [a for a in range(A) if xv.get((k, a), 0.0) > 0.5]
        selset = set(sel)
        sel_pair = {}
        for a in sel:
            sel_pair[(arcs[a]["from_node"], arcs[a]["to_node"])] = a

        used = set()
        cur = o
        walk_nodes = [o]
        while cur != d:
            nxt = None
            for a in out_arcs[cur]:
                if a in selset and a not in used:
                    nxt = a
                    break
            if nxt is None:
                return None
            used.add(nxt)
            cur = arcs[nxt]["to_node"]
            walk_nodes.append(cur)
            if len(used) > A + 1:
                return None

        # Loop-erased walk to obtain a simple path.
        le = []
        pos = {}
        for n in walk_nodes:
            if n in pos:
                p = pos[n]
                le = le[: p + 1]
                pos = {node: i for i, node in enumerate(le)}
            else:
                pos[n] = len(le)
                le.append(n)

        path = []
        for i in range(len(le) - 1):
            a = sel_pair.get((le[i], le[i + 1]))
            if a is None:
                return None
            path.append(a)
        return path

    def build_solution(yv, xv):
        total = 0.0
        entries = []
        for k, c in enumerate(commodities):
            routed = yv.get(k, 0.0) > 0.5
            path = extract_path(k, xv) if routed else None
            if routed and path is not None:
                path_entries = []
                for a in path:
                    total += demand[k] * arc_cost[a]
                    path_entries.append(
                        {
                            "from": arcs[a]["from_node"],
                            "to": arcs[a]["to_node"],
                            "arc_id": arcs[a]["arc_id"],
                        }
                    )
                entries.append(
                    {
                        "commodity_id": c["commodity_id"],
                        "rejected": False,
                        "path_arcs": path_entries,
                    }
                )
            else:
                total += art_cost[k]
                entries.append(
                    {
                        "commodity_id": c["commodity_id"],
                        "rejected": True,
                        "path_arcs": [],
                    }
                )
        return {"objective_value": float(total), "commodities": entries}

    # Log the trivially feasible all-rejected solution as the first incumbent.
    if logger:
        init_sol = all_rejected_solution()
        logger.log_solution(init_sol["objective_value"], init_sol)

    model = gp.Model("unsplittable_multicommodity_flow")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = max(1, int(args.time_limit - (time.time() - start_time)))
    model.Params.TimeLimit = remaining
    model.Params.OutputFlag = 0

    # Binary arc-selection variables per commodity and routed/rejected flags.
    x = model.addVars(K, A, vtype=GRB.BINARY, name="x")
    y = model.addVars(K, vtype=GRB.BINARY, name="y")
    r = model.addVars(K, vtype=GRB.BINARY, name="r")

    # Flow conservation: one unsplittable unit per routed commodity.
    for k, c in enumerate(commodities):
        o, d = c["origin"], c["destination"]
        for v in nodes:
            lhs = gp.quicksum(x[k, a] for a in out_arcs[v]) - gp.quicksum(
                x[k, a] for a in in_arcs[v]
            )
            rhs = 0
            if v == o:
                rhs = rhs + y[k]
            if v == d:
                rhs = rhs - y[k]
            model.addConstr(lhs == rhs, name=f"flow_{k}_{v}")

    # Shared arc capacities.
    for a in range(A):
        model.addConstr(
            gp.quicksum(demand[k] * x[k, a] for k in range(K)) <= arc_cap[a],
            name=f"cap_{a}",
        )

    # Each commodity is either routed or rejected.
    model.addConstrs((y[k] + r[k] == 1 for k in range(K)), name="route_or_reject")

    # Minimize rejection cost plus (zero-cost) physical routing costs.
    model.setObjective(
        gp.quicksum(art_cost[k] * r[k] for k in range(K))
        + gp.quicksum(
            demand[k] * arc_cost[a] * x[k, a] for k in range(K) for a in range(A)
        ),
        GRB.MINIMIZE,
    )

    model.update()

    # Complete MIP start: everything rejected (always feasible).
    for k in range(K):
        y[k].Start = 0.0
        r[k].Start = 1.0
        for a in range(A):
            x[k, a].Start = 0.0

    def incumbent_callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                yv = m.cbGetSolution(y)
                xv = m.cbGetSolution(x)
                sol = build_solution(yv, xv)
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        yv = model.getAttr("X", y)
        xv = model.getAttr("X", x)
        final_sol = build_solution(yv, xv)
    else:
        final_sol = all_rejected_solution()

    if logger:
        logger.log_solution(final_sol["objective_value"], final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()