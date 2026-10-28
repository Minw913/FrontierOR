import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    start_time = time.time()

    params = instance["problem_parameters"]
    num_periods = params["num_periods"]

    containers = instance["containers"]
    trades = instance["trades"]

    container_ids = [c["id"] for c in containers]
    cont_by_id = {c["id"]: c for c in containers}

    # Availability by period
    avail_by_period = defaultdict(list)
    for c in containers:
        for p in range(c["availability_start"], c["availability_end"] + 1):
            avail_by_period[p].append(c["id"])

    # Adjacency
    adj = {c["id"]: list(c["adjacent_containers"]) for c in containers}

    # Model
    model = gp.Model("container_assignment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))
    model.Params.MIPFocus = 1

    # Decision vars
    y = {}  # y[t,p,c] = 1 if container c assigned to trade t in period p
    tp_cands = {}
    cp_vars = defaultdict(list)

    trade_info = {}
    for t in trades:
        tid = t["id"]
        s = t["start_period"]
        e = t["end_period"]
        d = t["container_demand"]
        k = t["max_dispersion"]
        trade_info[tid] = (s, e, d, k)

        for p in range(s, e + 1):
            cands = avail_by_period.get(p, [])
            tp_cands[(tid, p)] = cands
            for c in cands:
                var = model.addVar(vtype=GRB.BINARY, name=f"y_{tid}_{p}_{c}")
                y[(tid, p, c)] = var
                cp_vars[(p, c)].append(var)

    # Demand constraints
    for t in trades:
        tid = t["id"]
        s, e, d, _ = trade_info[tid]
        for p in range(s, e + 1):
            cands = tp_cands[(tid, p)]
            model.addConstr(gp.quicksum(y[(tid, p, c)] for c in cands) == d,
                            name=f"demand_{tid}_{p}")

    # Container exclusivity per period
    for p in range(1, num_periods + 1):
        for c in avail_by_period.get(p, []):
            vars_pc = cp_vars.get((p, c), [])
            if vars_pc:
                model.addConstr(gp.quicksum(vars_pc) <= 1, name=f"exclusive_{p}_{c}")

    # Dispersion/connectivity constraints via rooted flow
    # For each trade-period where max_dispersion < demand, enforce <=k connected components.
    for t in trades:
        tid = t["id"]
        s, e, d, k = trade_info[tid]
        if k >= d:
            continue  # always satisfiable wrt components
        for p in range(s, e + 1):
            cands = tp_cands[(tid, p)]
            if not cands:
                continue

            cand_set = set(cands)

            z = {}  # root selection
            g = {}  # source flow into roots
            for c in cands:
                z[c] = model.addVar(vtype=GRB.BINARY, name=f"z_{tid}_{p}_{c}")
                g[c] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"g_{tid}_{p}_{c}")
                model.addConstr(z[c] <= y[(tid, p, c)], name=f"z_le_y_{tid}_{p}_{c}")
                model.addConstr(g[c] <= d * z[c], name=f"g_cap_{tid}_{p}_{c}")

            model.addConstr(gp.quicksum(z[c] for c in cands) <= k, name=f"roots_{tid}_{p}")

            # Directed flow on adjacency arcs
            f = {}
            in_arcs = defaultdict(list)
            out_arcs = defaultdict(list)

            for i in cands:
                for j in adj[i]:
                    if j in cand_set:
                        fv = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{tid}_{p}_{i}_{j}")
                        f[(i, j)] = fv
                        out_arcs[i].append(fv)
                        in_arcs[j].append(fv)
                        model.addConstr(fv <= d * y[(tid, p, i)], name=f"f_tail_{tid}_{p}_{i}_{j}")
                        model.addConstr(fv <= d * y[(tid, p, j)], name=f"f_head_{tid}_{p}_{i}_{j}")

            # Flow balance: each selected node consumes 1 unit
            for c in cands:
                model.addConstr(
                    g[c] + gp.quicksum(in_arcs[c]) - gp.quicksum(out_arcs[c]) == y[(tid, p, c)],
                    name=f"bal_{tid}_{p}_{c}"
                )

    # Re-assignment indicator per trade-period transition
    m = {}
    for t in trades:
        tid = t["id"]
        s, e, _, _ = trade_info[tid]
        for p in range(s + 1, e + 1):
            mv = model.addVar(vtype=GRB.BINARY, name=f"m_{tid}_{p}")
            m[(tid, p)] = mv

            prev_cands = set(tp_cands[(tid, p - 1)])
            curr_cands = set(tp_cands[(tid, p)])
            all_c = prev_cands | curr_cands

            for c in all_c:
                yc = y.get((tid, p, c), None)
                yp = y.get((tid, p - 1, c), None)

                if yc is not None and yp is not None:
                    model.addConstr(mv >= yc - yp, name=f"chg1_{tid}_{p}_{c}")
                    model.addConstr(mv >= yp - yc, name=f"chg2_{tid}_{p}_{c}")
                elif yc is not None:
                    model.addConstr(mv >= yc, name=f"chg3_{tid}_{p}_{c}")
                elif yp is not None:
                    model.addConstr(mv >= yp, name=f"chg4_{tid}_{p}_{c}")

    model.setObjective(gp.quicksum(m.values()), GRB.MINIMIZE)

    # Logging callback
    cb_state = {"best": float("inf")}

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < cb_state["best"]:
                cb_state["best"] = obj
                logger.log(float(obj))

    remaining = max(0, time_limit - int(time.time() - start_time))
    model.Params.TimeLimit = max(0, remaining)

    model.optimize(cb)

    # Build output assignments
    assignments = {}
    for t in trades:
        tid = t["id"]
        s, e, _, _ = trade_info[tid]
        assignments[str(tid)] = {}
        for p in range(s, e + 1):
            assignments[str(tid)][str(p)] = []

    if model.SolCount > 0:
        for t in trades:
            tid = t["id"]
            s, e, _, _ = trade_info[tid]
            for p in range(s, e + 1):
                cands = tp_cands[(tid, p)]
                selected = [c for c in cands if y[(tid, p, c)].X > 0.5]
                selected.sort()
                assignments[str(tid)][str(p)] = selected

        # Recompute objective as number of changed (trade,period) pairs
        objective_value = 0.0
        for t in trades:
            tid = t["id"]
            s, e, _, _ = trade_info[tid]
            prev = None
            for p in range(s, e + 1):
                cur = tuple(assignments[str(tid)][str(p)])
                if prev is not None and cur != prev:
                    objective_value += 1.0
                prev = cur
    else:
        # Fallback (no feasible found): keep schema-valid output
        objective_value = 1e18

    return {
        "objective_value": float(objective_value),
        "assignments": assignments
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()