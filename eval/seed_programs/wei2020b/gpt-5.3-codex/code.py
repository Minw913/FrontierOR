import argparse
import json
import math
import random
import time
from typing import List, Tuple, Dict, Optional

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def read_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    W = safe_int(data.get("W", 0), 0)
    V = safe_int(data.get("V", 0), 0)

    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raw_items = []

    items = []
    for it in raw_items:
        if isinstance(it, dict):
            w = safe_int(it.get("w", 0), 0)
            v = safe_int(it.get("v", 0), 0)
        else:
            w, v = 0, 0
        items.append((w, v))

    # Be robust to mismatched 'n' in input; trust actual items list.
    n = len(items)
    return n, W, V, items


def lower_bound(items: List[Tuple[int, int]], W: int, V: int) -> int:
    if W <= 0 or V <= 0:
        return len(items) if items else 0
    sw = sum(max(0, w) for w, _ in items)
    sv = sum(max(0, v) for _, v in items)
    return max(1 if items else 0, math.ceil(sw / W), math.ceil(sv / V))


def make_bin():
    return {"items": [], "tw": 0, "tv": 0}


def add_item_to_bin(b: Dict, i: int, items: List[Tuple[int, int]]):
    w, v = items[i]
    b["items"].append(i)
    b["tw"] += w
    b["tv"] += v


def can_fit(b: Dict, i: int, items: List[Tuple[int, int]], W: int, V: int) -> bool:
    w, v = items[i]
    return b["tw"] + w <= W and b["tv"] + v <= V


def greedy_pack(order: List[int], items: List[Tuple[int, int]], W: int, V: int):
    bins = []
    dW = max(1, W)
    dV = max(1, V)

    for i in order:
        w, v = items[i]
        best_idx = -1
        best_score = float("inf")
        for bi, b in enumerate(bins):
            nw, nv = b["tw"] + w, b["tv"] + v
            if nw <= W and nv <= V:
                score = (W - nw) / dW + (V - nv) / dV
                if score < best_score:
                    best_score = score
                    best_idx = bi

        if best_idx == -1:
            nb = make_bin()
            add_item_to_bin(nb, i, items)
            bins.append(nb)
        else:
            add_item_to_bin(bins[best_idx], i, items)
    return bins


def clone_bins(bins: List[Dict]) -> List[Dict]:
    return [{"items": b["items"][:], "tw": b["tw"], "tv": b["tv"]} for b in bins]


def try_reinsert_greedy(
    moving_items: List[int],
    bins: List[Dict],
    items: List[Tuple[int, int]],
    W: int,
    V: int,
) -> bool:
    dW = max(1, W)
    dV = max(1, V)

    order = sorted(
        moving_items,
        key=lambda i: (
            max(items[i][0] / dW, items[i][1] / dV),
            items[i][0] + items[i][1],
        ),
        reverse=True,
    )

    for i in order:
        w, v = items[i]
        best_idx = -1
        best_score = float("inf")
        for bi, b in enumerate(bins):
            nw, nv = b["tw"] + w, b["tv"] + v
            if nw <= W and nv <= V:
                score = (W - nw) / dW + (V - nv) / dV
                if score < best_score:
                    best_score = score
                    best_idx = bi
        if best_idx == -1:
            return False
        add_item_to_bin(bins[best_idx], i, items)
    return True


def try_reinsert_backtracking(
    moving_items: List[int],
    bins: List[Dict],
    items: List[Tuple[int, int]],
    W: int,
    V: int,
    node_limit: int = 40000,
) -> bool:
    dW = max(1, W)
    dV = max(1, V)

    order = sorted(
        moving_items,
        key=lambda i: (
            max(items[i][0] / dW, items[i][1] / dV),
            items[i][0] + items[i][1],
        ),
        reverse=True,
    )

    rem_w = [W - b["tw"] for b in bins]
    rem_v = [V - b["tv"] for b in bins]
    placed = [-1] * len(order)
    nodes = 0

    def dfs(pos: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return False
        if pos == len(order):
            return True

        i = order[pos]
        w, v = items[i]
        feas = []
        for bi in range(len(bins)):
            if rem_w[bi] >= w and rem_v[bi] >= v:
                score = (rem_w[bi] - w) / dW + (rem_v[bi] - v) / dV
                feas.append((score, bi))
        feas.sort(key=lambda x: x[0])

        for _, bi in feas:
            rem_w[bi] -= w
            rem_v[bi] -= v
            placed[pos] = bi
            if dfs(pos + 1):
                return True
            rem_w[bi] += w
            rem_v[bi] += v
            placed[pos] = -1
        return False

    ok = dfs(0)
    if not ok:
        return False

    for pos, i in enumerate(order):
        add_item_to_bin(bins[placed[pos]], i, items)
    return True


class IncumbentTracker:
    def __init__(self, logger: Optional[SolutionLogger]):
        self.logger = logger
        self.best = None

    def log_if_improved(self, obj: int):
        if self.best is None or obj < self.best:
            self.best = obj
            if self.logger:
                self.logger.log(int(obj))


def improve_bin_elimination(
    bins: List[Dict],
    items: List[Tuple[int, int]],
    W: int,
    V: int,
    deadline: float,
    tracker: IncumbentTracker,
):
    best = clone_bins(bins)
    improved = True

    while improved and time.time() < deadline:
        improved = False
        cand = list(range(len(best)))
        dW = max(1, W)
        dV = max(1, V)
        cand.sort(key=lambda bi: (best[bi]["tw"] / dW + best[bi]["tv"] / dV, len(best[bi]["items"])))

        for bi in cand:
            if time.time() >= deadline:
                break

            moving = best[bi]["items"]
            if not moving:
                continue

            remaining = [b for idx, b in enumerate(clone_bins(best)) if idx != bi]

            if try_reinsert_greedy(moving, remaining, items, W, V):
                best = remaining
                improved = True
                tracker.log_if_improved(len(best))
                break

            if len(moving) <= 18 and time.time() < deadline:
                remaining2 = [b for idx, b in enumerate(clone_bins(best)) if idx != bi]
                if try_reinsert_backtracking(moving, remaining2, items, W, V, node_limit=40000):
                    best = remaining2
                    improved = True
                    tracker.log_if_improved(len(best))
                    break

    return best


def heuristic_multistart(
    items: List[Tuple[int, int]],
    W: int,
    V: int,
    deadline: float,
    tracker: IncumbentTracker,
):
    n = len(items)
    if n == 0:
        tracker.log_if_improved(0)
        return []

    dW = max(1, W)
    dV = max(1, V)
    idx = list(range(n))
    rng = random.Random(0)

    orders = [
        sorted(idx, key=lambda i: max(items[i][0] / dW, items[i][1] / dV), reverse=True),
        sorted(idx, key=lambda i: (items[i][0] / dW + items[i][1] / dV), reverse=True),
        sorted(idx, key=lambda i: (items[i][0], items[i][1]), reverse=True),
        sorted(idx, key=lambda i: (items[i][1], items[i][0]), reverse=True),
        sorted(idx, key=lambda i: (items[i][0] * items[i][1], items[i][0] + items[i][1]), reverse=True),
    ]

    best = None
    best_obj = 10**18

    for order in orders:
        if time.time() >= deadline:
            break
        b = greedy_pack(order, items, W, V)
        if len(b) < best_obj:
            best, best_obj = b, len(b)
            tracker.log_if_improved(best_obj)

    if best is None:
        best = greedy_pack(idx, items, W, V)
        best_obj = len(best)
        tracker.log_if_improved(best_obj)

    base = sorted(idx, key=lambda i: (items[i][0] / dW + items[i][1] / dV), reverse=True)

    while time.time() < deadline:
        cand = base[:]
        swaps = max(1, n // 25)
        for _ in range(swaps):
            a = rng.randrange(n)
            b = rng.randrange(n)
            cand[a], cand[b] = cand[b], cand[a]
        bpack = greedy_pack(cand, items, W, V)
        if len(bpack) < best_obj:
            best, best_obj = bpack, len(bpack)
            tracker.log_if_improved(best_obj)

    return best


def solve_with_gurobi(
    items: List[Tuple[int, int]],
    W: int,
    V: int,
    incumbent_bins: List[Dict],
    time_limit: float,
    tracker: IncumbentTracker,
):
    if not GUROBI_AVAILABLE or time_limit <= 0:
        return incumbent_bins

    n = len(items)
    ub = len(incumbent_bins)
    if n == 0 or ub <= 1:
        return incumbent_bins

    try:
        m = gp.Model("bp2d")
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = float(max(0.0, time_limit))
        m.Params.MIPFocus = 1

        I = range(n)
        B = range(ub)

        x = m.addVars(I, B, vtype=GRB.BINARY, name="x")
        y = m.addVars(B, vtype=GRB.BINARY, name="y")

        for i in I:
            m.addConstr(gp.quicksum(x[i, b] for b in B) == 1)

        for b in B:
            m.addConstr(gp.quicksum(items[i][0] * x[i, b] for i in I) <= W * y[b])
            m.addConstr(gp.quicksum(items[i][1] * x[i, b] for i in I) <= V * y[b])

        for i in I:
            for b in B:
                m.addConstr(x[i, b] <= y[b])

        for b in range(ub - 1):
            m.addConstr(y[b] >= y[b + 1])

        m.setObjective(gp.quicksum(y[b] for b in B), GRB.MINIMIZE)

        # Warm start from incumbent
        for b in B:
            y[b].Start = 1.0 if b < ub else 0.0
        for b, bd in enumerate(incumbent_bins):
            for i in bd["items"]:
                if 0 <= i < n:
                    x[i, b].Start = 1.0

        best_cb = [len(incumbent_bins)]

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                iobj = int(round(obj))
                if iobj < best_cb[0]:
                    best_cb[0] = iobj
                    tracker.log_if_improved(iobj)

        m.optimize(cb)

        if m.SolCount == 0:
            return incumbent_bins

        assign = [-1] * n
        for i in I:
            for b in B:
                if x[i, b].X > 0.5:
                    assign[i] = b
                    break

        if any(a < 0 for a in assign):
            return incumbent_bins

        used = {}
        for i, b in enumerate(assign):
            if b not in used:
                used[b] = make_bin()
            add_item_to_bin(used[b], i, items)

        out = [used[k] for k in sorted(used.keys())]
        tracker.log_if_improved(len(out))
        return out

    except Exception:
        return incumbent_bins


def repair_if_needed(bins: List[Dict], items: List[Tuple[int, int]], W: int, V: int) -> List[Dict]:
    """If any bin violates capacity (e.g., due to malformed input), rebuild safely."""
    for b in bins:
        if b["tw"] > W or b["tv"] > V:
            n = len(items)
            order = sorted(range(n), key=lambda i: items[i][0] + items[i][1], reverse=True)
            return greedy_pack(order, items, W, V)
    return bins


def to_output(bins: List[Dict]):
    return {
        "objective_value": int(len(bins)),
        "bins": [
            {
                "items": [int(i) for i in b["items"]],
                "total_weight": int(b["tw"]),
                "total_volume": int(b["tv"]),
            }
            for b in bins
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    tracker = IncumbentTracker(logger)

    start = time.time()
    total_limit = max(0, int(args.time_limit))
    deadline = start + total_limit

    n, W, V, items = read_instance(args.instance_path)

    # Trivial fallback if capacities are nonpositive
    if n == 0:
        solution = {"objective_value": 0, "bins": []}
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
        return

    if W <= 0 or V <= 0:
        # Degenerate case: produce one item per bin (best effort robust behavior)
        bins = []
        for i in range(n):
            b = make_bin()
            add_item_to_bin(b, i, items)
            bins.append(b)
        tracker.log_if_improved(len(bins))
        with open(args.solution_path, "w") as f:
            json.dump(to_output(bins), f, indent=2)
        return

    # Phase 1: constructive + randomized
    phase1_deadline = start + int(0.45 * total_limit)
    if total_limit <= 1:
        phase1_deadline = deadline

    best_bins = heuristic_multistart(items, W, V, phase1_deadline, tracker)

    # Phase 2: local improvement
    if time.time() < deadline:
        best_bins = improve_bin_elimination(best_bins, items, W, V, deadline, tracker)

    # Phase 3: optional MIP polishing
    lb = lower_bound(items, W, V)
    if GUROBI_AVAILABLE and time.time() < deadline and len(best_bins) > lb:
        rem = deadline - time.time()
        best_bins = solve_with_gurobi(items, W, V, best_bins, rem, tracker)

    best_bins = repair_if_needed(best_bins, items, W, V)

    with open(args.solution_path, "w") as f:
        json.dump(to_output(best_bins), f, indent=2)


if __name__ == "__main__":
    main()