import argparse
import json
import time
import random
import math
import sys

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Instance handling
# ---------------------------------------------------------------------------

class Instance:
    def __init__(self, data):
        self.num_customers = data["num_customers"]
        self.num_nodes = data["num_nodes"]
        self.params = data.get("parameters", {})
        self.depot = data["depot_index"]
        self.customers = list(data["customer_indices"])
        self.sink = data["num_nodes"]  # virtual sink index used in output
        self.tt = data["truck_travel_times"]
        params = self.params
        self.K = params["num_drones"]
        self.cap = params["drone_battery_capacities"]
        self.bc = data["drone_battery_consumption"]      # [k][v][j]
        self.dt = data["drone_delivery_time"]            # [k][v][j]
        self.n = self.num_nodes

    def travel(self, i, j):
        # j == sink resolves to depot (same physical data)
        if j == self.sink:
            j = self.depot
        return self.tt[i][j]


# ---------------------------------------------------------------------------
# Solution representation
# ---------------------------------------------------------------------------

class Solution:
    def __init__(self, inst):
        self.inst = inst
        self.route = []                     # truck-visited customers in order
        self.assign = {}                    # customer -> (drone k, node v)
        self.used = [0] * inst.K            # battery used per drone
        self.times = {}                     # node v -> list of per-drone times
        self.await_ = {}                    # node v -> current waiting (max)
        self.count = {}                     # node v -> number of assignments
        self.travel = 0.0

    # ---------- basic helpers ----------
    def nodes(self):
        return [self.inst.depot] + self.route

    def compute_travel(self):
        inst = self.inst
        r = self.route
        if not r:
            return inst.tt[inst.depot][inst.depot]
        total = inst.tt[inst.depot][r[0]]
        for i in range(len(r) - 1):
            total += inst.tt[r[i]][r[i + 1]]
        total += inst.tt[r[-1]][inst.depot]
        return total

    def objective(self):
        return self.travel + sum(self.await_.values())

    def copy(self):
        s = Solution(self.inst)
        s.route = list(self.route)
        s.assign = dict(self.assign)
        s.used = list(self.used)
        s.times = {v: list(t) for v, t in self.times.items()}
        s.await_ = dict(self.await_)
        s.count = dict(self.count)
        s.travel = self.travel
        return s

    # ---------- assignment add/remove ----------
    def add_assign_delta(self, j, k, v):
        inst = self.inst
        cost = inst.bc[k][v][j]
        if self.used[k] + cost > inst.cap[k]:
            return None
        t = self.times.get(v)
        cur = t[k] if t is not None else 0.0
        newtot = cur + inst.dt[k][v][j]
        w = self.await_.get(v, 0.0)
        return max(w, newtot) - w

    def _apply_add_assign(self, j, k, v):
        inst = self.inst
        self.assign[j] = (k, v)
        self.used[k] += inst.bc[k][v][j]
        if v not in self.times:
            self.times[v] = [0.0] * inst.K
            self.count[v] = 0
            self.await_[v] = 0.0
        self.times[v][k] += inst.dt[k][v][j]
        self.count[v] += 1
        self.await_[v] = max(self.await_[v], self.times[v][k])

    def remove_assign_delta(self, j):
        inst = self.inst
        k, v = self.assign[j]
        t = self.times[v]
        w = self.await_[v]
        newt_k = t[k] - inst.dt[k][v][j]
        # recompute max after removal
        neww = 0.0
        for kk in range(inst.K):
            val = newt_k if kk == k else t[kk]
            if val > neww:
                neww = val
        return neww - w  # <= 0

    def _apply_remove_assign(self, j):
        inst = self.inst
        k, v = self.assign[j]
        self.used[k] -= inst.bc[k][v][j]
        self.times[v][k] -= inst.dt[k][v][j]
        self.count[v] -= 1
        neww = 0.0
        for kk in range(inst.K):
            if self.times[v][kk] > neww:
                neww = self.times[v][kk]
        self.await_[v] = neww
        del self.assign[j]

    # ---------- route add/remove delta ----------
    def remove_from_route_delta(self, pos):
        inst = self.inst
        r = self.route
        n = len(r)
        j = r[pos]
        p = r[pos - 1] if pos > 0 else inst.depot
        s = r[pos + 1] if pos < n - 1 else inst.depot
        return inst.tt[p][s] - inst.tt[p][j] - inst.tt[j][s]

    def insert_to_route_delta(self, j, pos):
        inst = self.inst
        r = self.route
        if not r:
            return inst.tt[inst.depot][j] + inst.tt[j][inst.depot] \
                   - inst.tt[inst.depot][inst.depot]
        if pos == 0:
            f = r[0]
            return inst.tt[inst.depot][j] + inst.tt[j][f] - inst.tt[inst.depot][f]
        if pos == len(r):
            l = r[-1]
            return inst.tt[l][j] + inst.tt[j][inst.depot] - inst.tt[l][inst.depot]
        a = r[pos - 1]
        b = r[pos]
        return inst.tt[a][j] + inst.tt[j][b] - inst.tt[a][b]

    def recompute_travel(self):
        self.travel = self.compute_travel()

    # ---------- export ----------
    def to_dict(self):
        inst = self.inst
        arcs = []
        prev = inst.depot
        for node in self.route:
            arcs.append([prev, node])
            prev = node
        arcs.append([prev, inst.sink])
        assigns = []
        for j, (k, v) in sorted(self.assign.items()):
            assigns.append({"drone": k, "dispatch_node": v, "delivery_node": j})
        waits = {}
        for v in self.nodes():
            waits[str(v)] = self.await_.get(v, 0.0)
        obj = self.objective()
        return {
            "objective_value": float(obj),
            "truck_route": arcs,
            "drone_assignments": assigns,
            "waiting_times": waits,
        }


# ---------------------------------------------------------------------------
# Local search moves
# ---------------------------------------------------------------------------

def find_best_move(sol, deadline):
    inst = sol.inst
    K = inst.K
    best = (0.0, None)
    eps = -1e-9

    # M1: route customer -> drone
    r = sol.route
    dispatch_used = sol.count
    nodes = sol.nodes()
    for pos in range(len(r)):
        if not (time.time() < deadline):
            break
        j = r[pos]
        if dispatch_used.get(j, 0) > 0:
            continue  # node currently dispatches drones; removing would break
        travel_d = sol.remove_from_route_delta(pos)
        for k in range(K):
            for v in nodes:
                dw = sol.add_assign_delta(j, k, v)
                if dw is None:
                    continue
                d = travel_d + dw
                if d < best[0]:
                    best = (d, ("m1", pos, j, k, v))

    # M2: drone customer -> route
    for j, (k, v) in list(sol.assign.items()):
        dw = sol.remove_assign_delta(j)  # <= 0
        for pos in range(len(r) + 1):
            tr = sol.insert_to_route_delta(j, pos)
            d = dw + tr
            if d < best[0]:
                best = (d, ("m2", j, pos))

    # M3: relocate route customer
    for pos in range(len(r)):
        j = r[pos]
        rem = sol.remove_from_route_delta(pos)
        # remove temporarily conceptually: positions shift
        for p2 in range(len(r)):
            if p2 == pos or p2 == pos + 1:
                continue
            # after removal, valid insertion positions are 0..len-1
            ins = sol._insert_delta_after_removal(j, p2, pos)
            d = rem + ins
            if d < best[0]:
                best = (d, ("m3", pos, p2))

    # M4: change assignment (k,v) -> (k',v')
    for j, (k, v) in list(sol.assign.items()):
        rem = sol.remove_assign_delta(j)
        # temporarily remove to evaluate additions properly
        sol._apply_remove_assign(j)
        for k2 in range(K):
            for v2 in nodes:
                dw = sol.add_assign_delta(j, k2, v2)
                if dw is None:
                    continue
                d = rem + dw
                if d < best[0]:
                    best = (d, ("m4", j, k2, v2))
        # restore
        sol._apply_add_assign(j, k, v)

    # M5: 2-opt
    n = len(r)
    if n >= 2:
        tt = inst.tt
        depot = inst.depot
        for i in range(n - 1):
            a = r[i - 1] if i > 0 else depot
            b = r[i]
            for jj in range(i + 1, n):
                c = r[jj]
                d_ = r[jj + 1] if jj + 1 < n else depot
                old = tt[a][b] + tt[c][d_]
                new = tt[a][c] + tt[b][d_]
                delta = new - old
                if delta < best[0]:
                    best = (delta, ("m5", i, jj))

    return best


def _insert_delta_after_removal(self, j, pos, removed_pos):
    """
    Delta of inserting j at position `pos` into route after `removed_pos`
    has been removed. pos ranges 0..len-1 where len is current route length.
    """
    inst = self.inst
    r = self.route
    # Build neighbors in the route without the removed element
    a = None
    b = None
    if pos > 0:
        idx = pos - 1
        if idx < removed_pos:
            a = r[idx]
        else:
            a = r[idx + 1]
        idx2 = pos - 1
    if pos < len(r) - 1:
        idx = pos
        if idx < removed_pos:
            b = r[idx]
        else:
            b = r[idx + 1]
    if a is None and b is None:
        return inst.tt[inst.depot][j] + inst.tt[j][inst.depot] - \
            inst.tt[inst.depot][inst.depot]
    if a is None:
        return inst.tt[inst.depot][j] + inst.tt[j][b] - inst.tt[inst.depot][b]
    if b is None:
        return inst.tt[a][j] + inst.tt[j][inst.depot] - \
            inst.tt[a][inst.depot]
    return inst.tt[a][j] + inst.tt[j][b] - inst.tt[a][b]


# Bind helper to Solution as private method
Solution._insert_delta_after_removal = _insert_delta_after_removal


def apply_move(sol, move):
    kind = move[0]
    if kind == "m1":
        _, pos, j, k, v = move
        d_travel = sol.remove_from_route_delta(pos)
        del sol.route[pos]
        sol._apply_add_assign(j, k, v)
        sol.travel += d_travel
    elif kind == "m2":
        _, j, pos = move
        sol._apply_remove_assign(j)
        d_travel = sol.insert_to_route_delta(j, pos)
        sol.route.insert(pos, j)
        sol.travel += d_travel
    elif kind == "m3":
        _, pos, p2 = move
        j = sol.route[pos]
        rem = sol.remove_from_route_delta(pos)
        ins = sol._insert_delta_after_removal(j, p2, pos)
        del sol.route[pos]
        ins_pos = p2 if p2 < pos else p2
        # recompute insertion position robustly:
        # after deletion, indices shift; safest is recompute delta position
        # Here we insert at p2 adjusted:
        if p2 > pos:
            ins_pos = p2 - 1 if p2 - 1 >= 0 else 0
            ins_pos = p2 - 1 if p2 > pos else p2
        sol.route.insert(ins_pos, j)
        sol.travel += rem + ins
    elif kind == "m4":
        _, j, k2, v2 = move
        sol._apply_remove_assign(j)
        sol._apply_add_assign(j, k2, v2)
    elif kind == "m5":
        _, i, jj = move
        sol.route[i:jj + 1] = reversed(sol.route[i:jj + 1])
        sol.recompute_travel()


def local_search(sol, deadline):
    while time.time() < deadline:
        delta, move = find_best_move(sol, deadline)
        if move is None or delta >= -1e-9:
            break
        apply_move(sol, move)
    sol.recompute_travel()


# ---------------------------------------------------------------------------
# Initial construction
# ---------------------------------------------------------------------------

def build_initial(inst, deadline):
    sol = Solution(inst)
    unvisited = list(inst.customers)
    route = []
    cur = inst.depot
    while unvisited:
        if not (time.time() < deadline):
            # fill remaining arbitrarily
            route.extend(unvisited)
            unvisited = []
            break
        bestj = min(unvisited, key=lambda j: inst.tt[cur][j])
        route.append(bestj)
        unvisited.remove(bestj)
        cur = bestj
    sol.route = route
    sol.recompute_travel()
    return sol


# ---------------------------------------------------------------------------
# LNS destroy & repair
# ---------------------------------------------------------------------------

def destroy(sol, q, rng):
    removed = []
    allc = sol.inst.customers
    tries = 0
    selected = set()
    while len(removed) < q and tries < 4 * len(allc):
        tries += 1
        j = rng.choice(allc)
        if j in selected:
            continue
        selected.add(j)
        if j in sol.route:
            pos = sol.route.index(j)
            # cascade: unassign drone deliveries dispatched from j
            cascade = [c for c, (k, v) in sol.assign.items() if v == j]
            d = sol.remove_from_route_delta(pos)
            del sol.route[pos]
            sol.travel += d
            removed.append(j)
            for c in cascade:
                sol._apply_remove_assign(c)
                removed.append(c)
        elif j in sol.assign:
            sol._apply_remove_assign(j)
            removed.append(j)
    return removed


def repair(sol, removed, rng, deadline):
    inst = sol.inst
    K = inst.K
    rng.shuffle(removed)
    for j in removed:
        if not (time.time() < deadline):
            # fallback: append to route (always feasible)
            pos = len(sol.route)
            sol.travel += sol.insert_to_route_delta(j, pos)
            sol.route.insert(pos, j)
            continue
        best_delta = float("inf")
        best_choice = None
        # options: drone assignment
        nodes = sol.nodes()
        for k in range(K):
            for v in nodes:
                dw = sol.add_assign_delta(j, k, v)
                if dw is None:
                    continue
                if dw < best_delta - 1e-12:
                    best_delta = dw
                    best_choice = ("drone", k, v)
                elif abs(dw - best_delta) <= 1e-12 and best_choice and rng.random() < 0.3:
                    best_choice = ("drone", k, v)
        # options: truck insertion
        for pos in range(len(sol.route) + 1):
            tr = sol.insert_to_route_delta(j, pos)
            if tr < best_delta - 1e-12:
                best_delta = tr
                best_choice = ("truck", pos)
        if best_choice is None:
            best_choice = ("truck", len(sol.route))
        if best_choice[0] == "drone":
            _, k, v = best_choice
            sol._apply_add_assign(j, k, v)
        else:
            _, pos = best_choice
            sol.travel += sol.insert_to_route_delta(j, pos)
            sol.route.insert(pos, j)
    sol.recompute_travel()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def solve(inst, time_limit, logger):
    start = time.time()
    deadline = start + max(0.0, time_limit - 0.5)
    rng = random.Random(0)

    best = build_initial(inst, deadline)
    local_search(best, deadline)
    best.recompute_travel()
    if logger:
        logger.log_solution(best.objective(), best.to_dict())

    n = len(inst.customers)
    iter_no = 0
    while time.time() < deadline:
        iter_no += 1
        cur = best.copy()
        qmax = max(2, min(n, max(4, n // 4)))
        q = rng.randint(2, qmax) if n >= 2 else 1
        removed = destroy(cur, q, rng)
        repair(cur, removed, rng, deadline)
        # light polish (time-limited)
        polish_deadline = min(deadline, time.time() + 0.25 * max(0.0, deadline - time.time()) / 4.0 + 0.05)
        local_search(cur, min(deadline, polish_deadline))
        cur.recompute_travel()
        if cur.objective() < best.objective() - 1e-9:
            best = cur
            if logger:
                logger.log_solution(best.objective(), best.to_dict())

    best.recompute_travel()
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)
    inst = Instance(data)

    start = time.time()
    best = solve(inst, args.time_limit, logger)
    out = best.to_dict()

    # sanity: ensure feasibility of produced solution; otherwise fall back to
    # all-truck route (always feasible)
    ok = True
    try:
        served = set(best.route) | set(best.assign.keys())
        if served != set(inst.customers):
            ok = False
        for k in range(inst.K):
            if best.used[k] > inst.cap[k] + 1e-6:
                ok = False
        nodes_set = set(best.nodes())
        for j, (k, v) in best.assign.items():
            if v not in nodes_set:
                ok = False
    except Exception:
        ok = False

    if not ok:
        fallback = Solution(inst)
        fallback.route = list(inst.customers)
        fallback.recompute_travel()
        out = fallback.to_dict()

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()