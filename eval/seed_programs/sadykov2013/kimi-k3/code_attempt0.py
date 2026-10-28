import argparse
import json
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


MAX_MIP_VARS = 500000
MAX_MIP_CONS = 2000000


def preprocess(data):
    raw_items = data.get("items", []) or []
    item_ids = []
    weights = []
    id_to_idx = {}

    for pos, rec in enumerate(raw_items):
        if isinstance(rec, dict):
            iid = rec.get("id", pos)
            w = rec.get("weight", 0)
        else:
            iid = pos
            w = rec if isinstance(rec, (int, float)) else 0
        item_ids.append(iid)
        weights.append(int(w))
        id_to_idx[iid] = pos

    n = len(weights)
    adj = [set() for _ in range(n)]
    edge_idx = []
    raw_edges = data.get("conflict_edges", []) or []

    for e in raw_edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        a, b = e[0], e[1]
        ia = id_to_idx.get(a)
        ib = id_to_idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        if ia > ib:
            ia, ib = ib, ia
        if ib not in adj[ia]:
            adj[ia].add(ib)
            adj[ib].add(ia)
            edge_idx.append((ia, ib))

    return item_ids, weights, id_to_idx, adj, edge_idx


def build_solution(item_ids, weights, assignment):
    bins = {}
    count = 0
    for items_idx in assignment:
        if not items_idx:
            continue
        count += 1
        total_weight = 0
        out_items = []
        for idx in items_idx:
            total_weight += weights[idx]
            out_items.append(item_ids[idx])
        bins[str(count)] = {
            "items": out_items,
            "total_weight": int(total_weight),
        }
    return {"objective_value": count, "bins": bins}


def solution_is_valid(sol, item_ids, weights, capacity, adj, id_to_idx):
    if not isinstance(sol, dict):
        return False
    bins = sol.get("bins", {})
    if sol.get("objective_value") != len(bins):
        return False

    seen = set()
    n = len(weights)
    for _, info in bins.items():
        if not isinstance(info, dict):
            return False
        items = info.get("items", [])
        if not isinstance(items, list):
            return False

        idxs = []
        total = 0
        for iid in items:
            idx = id_to_idx.get(iid)
            if idx is None or idx in seen or idx >= n:
                return False
            seen.add(idx)
            idxs.append(idx)
            total += weights[idx]

        if total != info.get("total_weight") or total > capacity:
            return False

        for p in range(len(idxs)):
            i = idxs[p]
            for q in range(p + 1, len(idxs)):
                j = idxs[q]
                if j in adj[i]:
                    return False

    return len(seen) == n


def greedy_pack(weights, capacity, adj, order, deadline):
    bins_items = []
    bins_weight = []
    blocked = []

    for count, i in enumerate(order):
        if deadline is not None and (count & 1023) == 0 and time.monotonic() > deadline:
            return None

        w = weights[i]
        chosen = -1
        best_rem = capacity + 1

        for b in range(len(bins_items)):
            if bins_weight[b] + w <= capacity and i not in blocked[b]:
                rem = capacity - (bins_weight[b] + w)
                if rem < best_rem:
                    best_rem = rem
                    chosen = b
                    if rem == 0:
                        break

        if chosen < 0:
            bins_items.append([i])
            bins_weight.append(w)
            blocked.append(set(adj[i]))
        else:
            bins_items[chosen].append(i)
            bins_weight[chosen] += w
            blocked[chosen].update(adj[i])

    return bins_items


def clique_lower_bound(adj):
    n = len(adj)
    if n == 0:
        return 0
    order = sorted(range(n), key=lambda i: (-len(adj[i]), i))
    clique = []
    for v in order:
        if all(u in adj[v] for u in clique):
            clique.append(v)
    return len(clique)


def write_solution(path, sol):
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)


def run_gurobi(weights, capacity, edge_idx, item_ids, adj, id_to_idx,
               start_assignment, lower_bound, deadline, logger, state):
    n = len(weights)
    B = len(start_assignment)
    if n == 0 or B == 0:
        return state

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return state

    model = gp.Model("bin_packing_with_conflicts")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.0, remaining)

    x = model.addVars(n, B, vtype=GRB.BINARY, name="x")
    y = model.addVars(B, vtype=GRB.BINARY, name="y")

    model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)

    if lower_bound > 0:
        model.addConstr(gp.quicksum(y[b] for b in range(B)) >= lower_bound,
                        name="objective_lower_bound")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, b] for b in range(B)) == 1)

    for b in range(B):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, b] for i in range(n)) <= capacity * y[b]
        )

    for i, j in edge_idx:
        for b in range(B):
            model.addConstr(x[i, b] + x[j, b] <= 1)

    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1])

    start_bin_of_item = {}
    for b, items in enumerate(start_assignment):
        for i in items:
            start_bin_of_item[i] = b

    for b in range(B):
        y[b].Start = 1.0
    for i in range(n):
        b0 = start_bin_of_item.get(i, -1)
        for b in range(B):
            x[i, b].Start = 1.0 if b == b0 else 0.0

    x_flat = [x[i, b] for i in range(n) for b in range(B)]
    y_flat = [y[b] for b in range(B)]

    def callback(m, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            yv = m.cbGetSolution(y_flat)
            xv = m.cbGetSolution(x_flat)
        except Exception:
            return

        assignment = []
        for b in range(B):
            if yv[b] <= 0.5:
                continue
            items_idx = []
            total_weight = 0
            for i in range(n):
                if xv[i * B + b] > 0.5:
                    items_idx.append(i)
                    total_weight += weights[i]
            if items_idx:
                assignment.append(items_idx)

        sol = build_solution(item_ids, weights, assignment)
        obj = sol["objective_value"]
        if obj < state["best_obj"]:
            state["best_obj"] = obj
            state["best_sol"] = sol
            state["best_assignment"] = assignment
            if logger:
                logger.log_solution(obj, sol)

    model.optimize(callback)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    capacity = int(data.get("bin_capacity", 0))
    item_ids, weights, id_to_idx, adj, edge_idx = preprocess(data)
    n = len(weights)

    def fallback_assignment():
        return [[i] for i in range(n)]

    if n == 0:
        sol = {"objective_value": 0, "bins": {}}
        if logger:
            logger.log_solution(0, sol)
        write_solution(args.solution_path, sol)
        return

    # If every pair conflicts, one item per bin is optimal.
    if len(edge_idx) == n * (n - 1) // 2:
        assignment = fallback_assignment()
        sol = build_solution(item_ids, weights, assignment)
        if logger:
            logger.log_solution(sol["objective_value"], sol)
        write_solution(args.solution_path, sol)
        return

    degrees = [len(s) for s in adj]
    orders = [
        sorted(range(n), key=lambda i: (-weights[i], -degrees[i], str(item_ids[i]))),
        sorted(range(n), key=lambda i: (-degrees[i], -weights[i], str(item_ids[i]))),
    ]

    state = {"best_obj": 10**9, "best_sol": None, "best_assignment": None}

    for order in orders:
        if time.monotonic() >= deadline:
            break
        assignment = greedy_pack(weights, capacity, adj, order, deadline)
        if assignment is None:
            break
        sol = build_solution(item_ids, weights, assignment)
        obj = sol["objective_value"]
        if obj < state["best_obj"]:
            state["best_obj"] = obj
            state["best_sol"] = sol
            state["best_assignment"] = assignment
            if logger:
                logger.log_solution(obj, sol)

    if state["best_sol"] is None or not solution_is_valid(
        state["best_sol"], item_ids, weights, capacity, adj, id_to_idx
    ):
        state["best_assignment"] = fallback_assignment()
        state["best_sol"] = build_solution(item_ids, weights, state["best_assignment"])
        state["best_obj"] = state["best_sol"]["objective_value"]
        if logger:
            logger.log_solution(state["best_obj"], state["best_sol"])

    total_weight = sum(weights)
    if capacity > 0:
        weight_lb = (total_weight + capacity - 1) // capacity if total_weight > 0 else 1
    else:
        weight_lb = 1
    lb = max(weight_lb, clique_lower_bound(adj), 1)

    if gp is not None and state["best_obj"] > lb:
        B = len(state["best_assignment"])
        est_vars = n * B + B
        est_cons = n + B + len(edge_idx) * B + max(0, B - 1) + 1
        remaining = deadline - time.monotonic()
        if (
            B > 1
            and remaining > 0.02
            and est_vars <= MAX_MIP_VARS
            and est_cons <= MAX_MIP_CONS
        ):
            try:
                run_gurobi(
                    weights=weights,
                    capacity=capacity,
                    edge_idx=edge_idx,
                    item_ids=item_ids,
                    adj=adj,
                    id_to_idx=id_to_idx,
                    start_assignment=state["best_assignment"],
                    lower_bound=lb,
                    deadline=deadline,
                    logger=logger,
                    state=state,
                )
            except Exception:
                pass

    if not solution_is_valid(state["best_sol"], item_ids, weights, capacity, adj, id_to_idx):
        state["best_assignment"] = fallback_assignment()
        state["best_sol"] = build_solution(item_ids, weights, state["best_assignment"])
        state["best_obj"] = state["best_sol"]["objective_value"]

    write_solution(args.solution_path, state["best_sol"])


if __name__ == "__main__":
    main()