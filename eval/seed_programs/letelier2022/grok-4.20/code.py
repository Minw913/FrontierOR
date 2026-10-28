import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    T = instance['parameters']['T']
    W = instance['parameters']['W']
    L_str = instance['parameters']['L']
    if L_str.lower() == 'infinity':
        L = float('inf')
    else:
        L = int(L_str)

    num_real_items = instance['num_real_items']
    source_id = instance['source_item_id']
    sink_id = instance['sink_item_id']

    items = {item['item_id']: item for item in instance['items']}
    real_items = [item for item in instance['items'] if item['type'] == 'treatment']
    item_ids = [item['item_id'] for item in real_items]
    weights = {item['item_id']: item['weight'] for item in real_items}

    arcs = instance['arcs']
    precedence = []
    for arc in arcs:
        precedence.append((arc['from'], arc['to'], arc['lag']))

    # Compute earliest and latest possible periods for each real item
    earliest = {iid: 1 for iid in item_ids}
    latest = {iid: T for iid in item_ids}
    for tail, head, lag in precedence:
        if tail == source_id and head in item_ids:
            earliest[head] = max(earliest.get(head, 1), lag)
        if head == sink_id and tail in item_ids:
            latest[tail] = min(latest.get(tail, T), T + 1 - lag)

    start_time = time.time()
    model = gp.Model("BinPeriodScheduling")
    model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - 5))
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.LogToConsole, 0)
    model.setParam(GRB.Param.Heuristics, 0.3)
    model.setParam(GRB.Param.Method, 2)

    # Variables
    t = {}
    for iid in item_ids:
        t[iid] = model.addVar(vtype=GRB.INTEGER, lb=earliest[iid], ub=latest[iid], name=f"t_{iid}")

    # Estimate number of bins needed per period
    total_w = sum(weights.values())
    B = max(5, int(total_w * 1.2 / W / (T * 0.6)) + 3)
    if L != float('inf'):
        B = min(B, int(L) + 2)
    B = min(B, 30)  # cap for memory/time

    x = {}
    for iid in item_ids:
        for p in range(earliest[iid], latest[iid] + 1):
            for b in range(B):
                x[iid, p, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{iid}_{p}_{b}")

    y = {}
    for b in range(B):
        for p in range(1, T + 1):
            y[b, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{b}_{p}")

    # Objective
    obj = gp.quicksum(y[b, p] for b in range(B) for p in range(1, T + 1))
    model.setObjective(obj, GRB.MINIMIZE)

    # Each item assigned exactly once
    for iid in item_ids:
        expr = gp.quicksum(x[iid, p, b] for p in range(earliest[iid], latest[iid] + 1) for b in range(B))
        model.addConstr(expr == 1, name=f"assign_{iid}")

    # Link x to t
    for iid in item_ids:
        model.addConstr(t[iid] == gp.quicksum(p * x.get((iid, p, b), 0) for p in range(earliest[iid], latest[iid] + 1) for b in range(B)),
                        name=f"link_t_{iid}")

    # Capacity
    for b in range(B):
        for p in range(1, T + 1):
            expr = gp.quicksum(weights.get(iid, 0) * x.get((iid, p, b), 0) for iid in item_ids)
            model.addConstr(expr <= W * y[b, p], name=f"cap_{b}_{p}")

    # Limit bins per period if applicable
    if L != float('inf'):
        for p in range(1, T + 1):
            model.addConstr(gp.quicksum(y[b, p] for b in range(B)) <= L, name=f"limit_{p}")

    # Precedence constraints
    for tail, head, lag in precedence:
        if tail == source_id and head in t:
            model.addConstr(t[head] >= lag, name=f"src_{head}")
        elif head == sink_id and tail in t:
            model.addConstr(t[tail] <= T + 1 - lag, name=f"snk_{tail}")
        elif tail in t and head in t:
            model.addConstr(t[head] >= t[tail] + lag, name=f"prec_{tail}_{head}")
        # ignore other cases (source to sink, etc.)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = None

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            cur_obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or cur_obj < best_obj - 1e-6:
                best_obj = cur_obj
                if logger:
                    logger.log(float(best_obj))

    model.optimize(callback)

    if model.SolCount == 0:
        # Fallback: assign each item to its own bin-period sequentially
        fallback_assignments = []
        fallback_active = []
        obj_val = 0
        period = 1
        bin_id = 0
        for idx, iid in enumerate(item_ids):
            fallback_assignments.append({
                "item_id": int(iid),
                "bin": int(bin_id),
                "period": int(period),
                "weight": int(weights[iid])
            })
            key = (bin_id, period)
            fallback_active.append({
                "bin": int(bin_id),
                "period": int(period),
                "items": [int(iid)],
                "total_weight": int(weights[iid])
            })
            obj_val += 1
            bin_id += 1
            if bin_id >= 5:
                bin_id = 0
                period += 1
                if period > T:
                    period = T
        solution = {
            "objective_value": float(obj_val),
            "assignments": fallback_assignments,
            "active_bin_periods": fallback_active,
            "num_active_bin_periods": int(obj_val)
        }
    else:
        objective_value = model.objVal
        assignments = []
        active = {}
        for iid in item_ids:
            assigned = False
            for p in range(earliest.get(iid, 1), latest.get(iid, T) + 1):
                for b in range(B):
                    var_key = (iid, p, b)
                    if var_key in x and x[var_key].X > 0.4:
                        assignments.append({
                            "item_id": int(iid),
                            "bin": int(b),
                            "period": int(p),
                            "weight": int(weights[iid])
                        })
                        key = (b, p)
                        if key not in active:
                            active[key] = {
                                "bin": int(b),
                                "period": int(p),
                                "items": [],
                                "total_weight": 0
                            }
                        active[key]["items"].append(int(iid))
                        active[key]["total_weight"] += int(weights[iid])
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                # fallback for this item
                p = max(1, min(T, int(t[iid].X) if t[iid].X is not None else 1))
                b = 0
                assignments.append({
                    "item_id": int(iid),
                    "bin": int(b),
                    "period": int(p),
                    "weight": int(weights[iid])
                })
                key = (b, p)
                if key not in active:
                    active[key] = {"bin": int(b), "period": int(p), "items": [], "total_weight": 0}
                active[key]["items"].append(int(iid))
                active[key]["total_weight"] += int(weights[iid])

        active_bin_periods = list(active.values())
        for abp in active_bin_periods:
            abp["items"] = sorted(abp["items"])
            abp["total_weight"] = int(abp["total_weight"])

        solution = {
            "objective_value": float(objective_value),
            "assignments": assignments,
            "active_bin_periods": active_bin_periods,
            "num_active_bin_periods": int(len(active_bin_periods))
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()