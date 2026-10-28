import sys, json, math, time, random, argparse
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GP = True
except Exception:
    HAVE_GP = False

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

RNG = random.Random(123)


class Problem:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.data = data
        self.params = data.get("parameters", {})
        self.H0, self.H1 = self.params.get("planning_horizon", [0.0, 1e9])
        self.max_dur = self.params.get("max_route_duration")
        if self.max_dur is None:
            self.max_dur = self.H1 - self.H0
        self.dec_time = self.params.get("decoupling_time", 0.0)
        self.cpl_time = self.params.get("coupling_time", 0.0)
        self.trf_time = self.params.get("transfer_time", 0.0)
        self.dec_cost = self.params.get("decoupling_cost", 0.0)
        self.cpl_cost = self.params.get("coupling_cost", 0.0)
        self.trf_cost = self.params.get("transfer_cost", 0.0)
        self.def_svc = self.params.get("service_time", 0.0)

        self.depot = data["depot"]["id"]
        self.D = data["distance_matrix"]
        self.TT = data["travel_time_matrix"]
        self.n_orig = len(self.D)

        self.customers = {}
        for c in data.get("truck_customers", []):
            self.customers[c["id"]] = {
                "id": c["id"], "supply": c["supply"],
                "svc": c.get("service_time", self.def_svc), "type": "T"}
        for c in data.get("trailer_customers", []):
            self.customers[c["id"]] = {
                "id": c["id"], "supply": c["supply"],
                "svc": c.get("service_time", self.def_svc), "type": "R"}
        self.truck_custs = [c["id"] for c in data.get("truck_customers", [])]
        self.trailer_custs = [c["id"] for c in data.get("trailer_customers", [])]
        self.trans = [t["id"] for t in data.get("transshipment_locations", [])]

        # parking list in published order: trailer customers then transshipment
        self.P = list(self.trailer_custs) + list(self.trans)
        self.p_index = {p: i for i, p in enumerate(self.P)}
        self.np = len(self.P)

        trucks = {t["id"]: t for t in data.get("fleet", {}).get("truck_types", [])}
        trailers = {t["id"]: t for t in data.get("fleet", {}).get("trailer_types", [])}
        self.trucks = trucks
        self.trailers = trailers
        self.classes = {}
        for vc in data.get("fleet", {}).get("vehicle_classes", []):
            cid = vc["class_id"]
            tt = vc["truck_type_id"]
            tr = vc.get("trailer_type_id", 0)
            info = {
                "cls": cid, "truck": tt, "trailer": tr,
                "capT": vc.get("truck_capacity", trucks.get(tt, {}).get("capacity", 0)),
                "capR": vc.get("trailer_capacity", 0),
                "fixed": vc.get("truck_fixed_cost", trucks.get(tt, {}).get("fixed_cost", 0.0)),
                "f": vc.get("truck_cost_factor", trucks.get(tt, {}).get("cost_factor", 1.0)),
                "tow": vc.get("trailer_towing_cost_multiplier", 0.0),
                "avail": vc.get("num_available", 10 ** 9),
            }
            self.classes[cid] = info
        self.cls_ids = list(self.classes.keys())

    def dec_id(self, p):
        return self.n_orig + self.p_index[p]

    def cpl_id(self, p):
        return self.n_orig + self.np + self.p_index[p]

    def is_customer(self, x):
        return x in self.customers


# ---------------- TSP helpers ----------------

def path_cost(D, nodes, start, end):
    if not nodes:
        return 0.0
    tot = D[start][nodes[0]] if start is not None else 0.0
    for i in range(len(nodes) - 1):
        tot += D[nodes[i]][nodes[i + 1]]
    if end is not None:
        tot += D[nodes[-1]][end]
    return tot


def nn_path(D, nodes, start, end=None):
    rem = list(nodes)
    path = []
    cur = start
    while rem:
        nxt = min(rem, key=lambda x: D[cur][x])
        rem.remove(nxt)
        path.append(nxt)
        cur = nxt
    return path


def two_opt_path(D, nodes, start, end=None):
    n = len(nodes)
    if n < 3:
        return nodes
    best = list(nodes)
    closed = (end is None and start is not None)

    def full_cost(p):
        tot = D[start][p[0]]
        for i in range(len(p) - 1):
            tot += D[p[i]][p[i + 1]]
        if closed:
            tot += D[p[-1]][start]
        elif end is not None:
            tot += D[p[-1]][end]
        return tot

    bc = full_cost(best)
    improved = True
    it = 0
    while improved and it < 40:
        improved = False
        it += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cc = full_cost(cand)
                if cc < bc - 1e-9:
                    best, bc = cand, cc
                    improved = True
    return best


def tsp_order(D, nodes, start, end=None):
    if not nodes:
        return []
    p = nn_path(D, nodes, start, end if end is not None else start)
    return two_opt_path(D, p, start, end)


# ---------------- Route construction / evaluation ----------------

class Builder:
    def __init__(self, prob):
        self.pr = prob
        self.cache = {}

    def partition_groups(self, members, cap):
        if not members:
            return []
        ordered = sorted(members, key=lambda c: -self.pr.customers[c]["supply"])
        groups, rem = [], []
        for c in ordered:
            s = self.pr.customers[c]["supply"]
            if s > cap:
                return None
            placed = False
            for g in range(len(groups)):
                if rem[g] >= s:
                    groups[g].append(c)
                    rem[g] -= s
                    placed = True
                    break
            if not placed:
                groups.append([c])
                rem.append(cap - s)
        return groups

    def build_events(self, struct):
        ev = []
        for c in struct.get("before", []):
            ev.append(("C", c))
        for park, mem in struct.get("blocks", []):
            ev.append(("D", park))
            for m in mem:
                ev.append(("S", m))
            ev.append(("R", park))
        for c in struct.get("after", []):
            ev.append(("C", c))
        return ev

    def eval_struct(self, struct):
        pr = self.pr
        cl = pr.classes[struct["cls"]]
        capT, capR, tow, f = cl["capT"], cl["capR"], cl["tow"], cl["f"]
        ev = self.build_events(struct)
        nodes = [pr.depot]
        phys = [pr.depot]
        dec_loc, cpl_loc = [], []
        t = pr.H0
        cost = cl["fixed"]
        qT = qR = 0.0
        attach = True
        arr = {pr.depot: t}
        custs = []
        prev = pr.depot

        def move(a, b):
            nonlocal t, cost
            w = f + (tow if attach else 0.0)
            t += pr.TT[a][b]
            cost += pr.D[a][b] * w

        def load_attached(cid):
            nonlocal qT, qR
            s = pr.customers[cid]["supply"]
            add = min(s, capR - qR)
            qR += add
            qT += s - add

        for kind, x in ev:
            if kind == "C":
                move(prev, x)
                c = pr.customers[x]
                t += c["svc"]
                load_attached(x)
                if qT > capT + 1e-9 or qR > capR + 1e-9:
                    return None
                nodes.append(x); phys.append(x); custs.append(x)
                arr[x] = t
                prev = x
            elif kind == "S":  # solo segment customer
                move(prev, x)
                c = pr.customers[x]
                t += c["svc"]
                qT += c["supply"]
                if qT > capT + 1e-9:
                    return None
                nodes.append(x); phys.append(x); custs.append(x)
                arr[x] = t
                prev = x
            elif kind == "D":
                move(prev, x)
                # parking is a trailer customer -> serve it now (attached)
                if self.pr.is_customer(x):
                    c = pr.customers[x]
                    t += c["svc"]
                    load_attached(x)
                    if qT > capT + 1e-9 or qR > capR + 1e-9:
                        return None
                    custs.append(x)
                t += pr.dec_time
                cost += pr.dec_cost
                attach = False
                nodes.append(pr.dec_id(x)); phys.append(x)
                dec_loc.append(x)
                arr[nodes[-1]] = t
                prev = x
            elif kind == "R":
                move(prev, x)
                t += pr.cpl_time
                cost += pr.cpl_cost
                tr = min(qT, capR - qR)
                if tr > 1e-9:
                    qR += tr
                    qT -= tr
                    t += pr.trf_time
                    cost += pr.trf_cost
                attach = True
                nodes.append(pr.cpl_id(x)); phys.append(x)
                cpl_loc.append(x)
                arr[nodes[-1]] = t
                prev = x

        move(prev, pr.depot)
        nodes.append(pr.depot)
        phys.append(pr.depot)
        if t > pr.H1 + 1e-6:
            return None
        if t - pr.H0 > pr.max_dur + 1e-6:
            return None
        arr[pr.depot] = t  # final return time (depot repeats)
        return {
            "cls": struct["cls"], "struct": struct,
            "nodes": nodes, "phys": phys, "dec": dec_loc, "cpl": cpl_loc,
            "arr": arr, "cost": cost, "customers": list(custs),
            "end": t,
        }

    def parking_pool(self, custs):
        pr = self.pr
        trl = [c for c in custs if pr.customers[c]["type"] == "R"]
        pool = list(trl)
        scored = []
        for x in pr.trans:
            d = min([pr.D[x][c] for c in custs] + [pr.D[x][pr.depot]])
            scored.append((d, x))
        scored.sort()
        for _, x in scored[:6]:
            pool.append(x)
        return pool

    def construct(self, custs, cls_id):
        pr = self.pr
        key = (cls_id, frozenset(custs))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        info = pr.classes[cls_id]
        has_trailer = info["trailer"] != 0 and info["capR"] > 0
        best = None

        if not has_trailer:
            if sum(pr.customers[c]["supply"] for c in custs) > info["capT"] + 1e-9:
                self.cache[key] = None
                return None
            seq = tsp_order(pr.D, custs, pr.depot, None)
            struct = {"cls": cls_id, "before": seq, "after": [], "blocks": []}
            best = self.eval_struct(struct)
            self.cache[key] = best
            return best

        truck_cs = [c for c in custs if pr.customers[c]["type"] == "T"]
        trl_cs = [c for c in custs if pr.customers[c]["type"] == "R"]

        if not truck_cs:
            seq = tsp_order(pr.D, custs, pr.depot, None)
            struct = {"cls": cls_id, "before": seq, "after": [], "blocks": []}
            best = self.eval_struct(struct)
            self.cache[key] = best
            return best

        pool = self.parking_pool(custs)
        if not pool:
            self.cache[key] = None
            return None

        # variant S sets: truck only, or truck + near trailer customers
        s_variants = [truck_cs]
        extra = []
        for c in trl_cs:
            d_dep = pr.D[pr.depot][c]
            d_pool = min(pr.D[p][c] for p in pool)
            if d_pool < 0.75 * d_dep:
                extra.append(c)
        s_variants.append(truck_cs + extra)

        for sset in s_variants:
            groups = self.partition_groups(list(sset), info["capT"])
            if groups is None or len(groups) > len(pool):
                continue
            # assign distinct parkings greedily (by group centroid)
            used = set()
            parks = []
            for g in groups:
                cx = sum(pr.data_x[c] for c in g) / len(g)
                cy = sum(pr.data_y[c] for c in g) / len(g)
                cand = None
                for p in pool:
                    if p in used:
                        continue
                    px, py = pr.data_x[p], pr.data_y[p]
                    dd = (px - cx) ** 2 + (py - cy) ** 2
                    if cand is None or dd < cand[0]:
                        cand = (dd, p)
                if cand is None:
                    parks = None
                    break
                used.add(cand[1])
                parks.append(cand[1])
            if parks is None:
                continue
            # order blocks from depot by NN
            blocks_nodes = list(parks)
            first = min(blocks_nodes, key=lambda p: pr.D[pr.depot][p])
            ordered_parks = [first]
            rest = [p for p in blocks_nodes if p != first]
            cur = first
            while rest:
                nxt = min(rest, key=lambda p: pr.D[cur][p])
                rest.remove(nxt)
                ordered_parks.append(nxt)
                cur = nxt
            park_to_group = dict(zip(parks, groups))
            blocks = []
            for p in ordered_parks:
                mem = tsp_order(pr.D, park_to_group[p], p, None)
                blocks.append((p, mem))
            A = [c for c in trl_cs if c not in set(parks) and c not in set(sset)]
            first_p, last_p = ordered_parks[0], ordered_parks[-1]

            for mode in (0, 1):
                if A:
                    if mode == 0:
                        before = tsp_order(pr.D, A, pr.depot, first_p)
                        after = []
                    else:
                        before = []
                        after = tsp_order(pr.D, A, last_p, pr.depot)
                else:
                    before, after = [], []
                struct = {"cls": cls_id, "before": before,
                          "after": after, "blocks": blocks}
                res = self.eval_struct(struct)
                if res and (best is None or res["cost"] < best["cost"]):
                    best = res
        # also try the pure-attached variant if no truck customers (covered above)
        self.cache[key] = best
        return best


# ---------------- Solver ----------------

class Solver:
    def __init__(self, prob, time_limit):
        self.pr = prob
        self.tl = time_limit
        self.start = time.time()
        self.builder = Builder(prob)
        prob.data_x = {}
        prob.data_y = {}
        locs = [prob.data["depot"]] + prob.data.get("truck_customers", []) + \
               prob.data.get("trailer_customers", []) + prob.data.get("transshipment_locations", [])
        for l in locs:
            prob.data_x[l["id"]] = l["x"]
            prob.data_y[l["id"]] = l["y"]
        self.pool = {}  # (cls, frozenset) -> eval
        self.best = None
        self.best_cost = float("inf")
        self.logger = None

    def time_left(self):
        return self.tl - (time.time() - self.start)

    def construct(self, custs, cls):
        r = self.builder.construct(custs, cls)
        if r:
            self.pool[(cls, frozenset(custs))] = r
        return r

    def usage_ok(self, routes):
        cls_cnt = defaultdict(int)
        tk_cnt = defaultdict(int)
        tr_cnt = defaultdict(int)
        for r in routes:
            cls = r["cls"]
            info = self.pr.classes[cls]
            cls_cnt[cls] += 1
            tk_cnt[info["truck"]] += 1
            if info["trailer"]:
                tr_cnt[info["trailer"]] += 1
        for c, v in cls_cnt.items():
            if v > self.pr.classes[c]["avail"]:
                return False
        for t, v in tk_cnt.items():
            if t in self.pr.trucks and v > self.pr.trucks[t].get("num_available", 10 ** 9):
                return False
        for t, v in tr_cnt.items():
            if t in self.pr.trailers and v > self.pr.trailers[t].get("num_available", 10 ** 9):
                return False
        return True

    def total(self, routes):
        return sum(r["cost"] for r in routes)

    def greedy_build(self, custs_subset, base_routes):
        routes = list(base_routes)
        for cust in custs_subset:
            best_delta = float("inf")
            best_choice = None
            for i, r in enumerate(routes):
                nr = self.construct(r["customers"] + [cust], r["cls"])
                if nr:
                    d = nr["cost"] - r["cost"]
                    if d < best_delta:
                        best_delta = d
                        best_choice = (i, nr)
            for cls in self.pr.cls_ids:
                if not self.pr.classes[cls]["avail"]:
                    continue
                nr = self.construct([cust], cls)
                if nr and nr["cost"] < best_delta:
                    best_delta = nr["cost"]
                    best_choice = (-1, nr)
            if best_choice is None:
                return None
            i, nr = best_choice
            if i == -1:
                routes.append(nr)
            else:
                routes[i] = nr
            routes = [r for r in routes if r["customers"]]
        return routes if self.usage_ok(routes) else None

    def initial_solution(self):
        pr = self.pr
        custs = list(pr.customers.keys())
        custs.sort(key=lambda c: -pr.D[pr.depot][c])
        sol = self.greedy_build(custs, [])
        if sol is None:
            raise RuntimeError("no feasible initial solution")
        return sol

    def log_best(self):
        if self.logger:
            try:
                d = self.solution_dict(self.best, self.best_cost)
                self.logger.log_solution(self.best_cost, d)
            except Exception:
                self.logger.log(self.best_cost)

    def solution_dict(self, routes, obj):
        pr = self.pr
        out = {"objective_value": float(obj), "routes": []}
        per_cls = defaultdict(int)
        for r in sorted(routes, key=lambda r: (r["cls"], -r["cost"])):
            idx = per_cls[r["cls"]]
            per_cls[r["cls"]] += 1
            out["routes"].append({
                "vehicle_class": int(r["cls"]),
                "vehicle_index": int(idx),
                "customers_served": [int(c) for c in r["customers"]],
                "route_sequence": [int(x) for x in r["nodes"]],
                "route_sequence_physical": [int(x) for x in r["phys"]],
                "decouple_locations": [int(x) for x in r["dec"]],
                "couple_locations": [int(x) for x in r["cpl"]],
                "arrival_times": {int(k): float(v) for k, v in r["arr"].items()},
            })
        out["expanded_node_convention"] = (
            "Let n_orig be the number of physical locations and P be the instance "
            "parking-location IDs in their published list order. Physical nodes "
            "retain IDs 0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, "
            "and the couple copy of P[i] has ID n_orig+len(P)+i. Both copies map "
            "back to physical location P[i].")
        return out

    def local_search(self, budget_end):
        pr = self.pr
        routes = self.initial_solution()
        cur = self.total(routes)
        self.best = list(routes)
        self.best_cost = cur
        self.log_best()
        allcusts = list(pr.customers.keys())
        T0 = max(cur / max(1, len(routes)), 1.0)
        n_iter = 0
        while time.time() < budget_end:
            n_iter += 1
            frac = max(0.0, (budget_end - time.time()) / max(1e-6, budget_end - self.start))
            T = T0 * frac + 1e-3
            op = RNG.random()
            new_routes = None
            if not routes:
                break
            if op < 0.45 and routes:  # relocate
                ri = RNG.randrange(len(routes))
                r = routes[ri]
                if r["customers"]:
                    c = RNG.choice(r["customers"])
                    rest = [x for x in r["customers"] if x != c]
                    targets = []
                    if rest:
                        nr = self.construct(rest, r["cls"])
                        if nr:
                            targets.append((ri, nr))
                    else:
                        targets.append((ri, None))
                    opts = [(ri, n) for ri, n in targets if n]
                    # try insertion into other routes / new route
                    choice = None
                    best_d = float("inf")
                    for j, rr in enumerate(routes):
                        if j == ri:
                            continue
                        nr = self.construct(rr["customers"] + [c], rr["cls"])
                        if nr:
                            d = nr["cost"] - rr["cost"] + (opts[0][1]["cost"] if opts else 0) - r["cost"] if False else None
                    # simple greedy insert
                    best_move = float("inf")
                    best_j = -1
                    best_nr = None
                    base_delta_removed = (opts[0][1]["cost"] if opts else 0.0) - r["cost"] if opts else -r["cost"]
                    for j, rr in enumerate(routes):
                        if j == ri:
                            continue
                        nr = self.construct(rr["customers"] + [c], rr["cls"])
                        if nr:
                            d = base_delta_removed + nr["cost"] - rr["cost"]
                            if d < best_move:
                                best_move = d
                                best_j = j
                                best_nr = nr
                    for cls in pr.cls_ids:
                        nr = self.construct([c], cls)
                        if nr:
                            d = base_delta_removed + nr["cost"]
                            if d < best_move:
                                best_move = d
                                best_j = -2
                                best_nr = nr
                    if best_nr is not None:
                        cand = list(routes)
                        if opts:
                            cand[ri] = opts[0][1]
                        else:
                            cand = [r0 for i0, r0 in enumerate(cand) if i0 != ri]
                        adj = best_j if best_j >= 0 else best_j
                        if best_j == -2:
                            cand.append(best_nr)
                        else:
                            if best_j > ri and not opts:
                                best_j -= 1
                            cand[best_j] = best_nr
                        cand = [x for x in cand if x["customers"]]
                        new_routes = cand
            elif op < 0.75 and len(routes) >= 2:  # swap
                i, j = RNG.sample(range(len(routes)), 2)
                a, b = routes[i], routes[j]
                if a["customers"] and b["customers"]:
                    ca = RNG.choice(a["customers"])
                    cb = RNG.choice(b["customers"])
                    na_set = [x for x in a["customers"] if x != ca] + [cb]
                    nb_set = [x for x in b["customers"] if x != cb] + [ca]
                    na = self.construct(na_set, a["cls"]) if na_set else None
                    nb = self.construct(nb_set, b["cls"]) if nb_set else None
                    if (na or not na_set) and (nb or not nb_set):
                        cand = list(routes)
                        cand[i] = na
                        cand[j] = nb
                        cand = [x for x in cand if x and x["customers"]]
                        new_routes = cand
            elif op < 0.9 and routes:  # change class
                ri = RNG.randrange(len(routes))
                r = routes[ri]
                opts = []
                for cls in pr.cls_ids:
                    if cls == r["cls"] or not pr.classes[cls]["avail"]:
                        continue
                    nr = self.construct(r["customers"], cls)
                    if nr:
                        opts.append(nr)
                if opts:
                    nr = min(opts, key=lambda x: x["cost"])
                    cand = list(routes)
                    cand[ri] = nr
                    new_routes = cand
            else:  # destroy & repair
                k = min(len(allcusts), RNG.randint(2, 6))
                removed = RNG.sample(allcusts, k)
                stripped = []
                for r in routes:
                    rest = [x for x in r["customers"] if x not in removed]
                    if not rest:
                        continue
                    nr = self.construct(rest, r["cls"])
                    if nr:
                        stripped.append(nr)
                    else:
                        stripped = None
                        break
                if stripped is not None:
                    cand = self.greedy_build(removed, stripped)
                    if cand:
                        new_routes = cand
            if new_routes is None:
                continue
            if not self.usage_ok(new_routes):
                continue
            new_cost = self.total(new_routes)
            delta = new_cost - cur
            if delta < 0 or RNG.random() < math.exp(-delta / max(T, 1e-6)):
                routes = new_routes
                cur = new_cost
                if cur < self.best_cost - 1e-9:
                    self.best = list(routes)
                    self.best_cost = cur
                    self.log_best()
            if n_iter % 64 == 0 and self.time_left() < 2.0:
                break
        return self.best, self.best_cost

    def set_partition(self, remain):
        if not HAVE_GP or not self.pool or remain < 2:
            return self.best, self.best_cost
        # add best routes to pool
        for r in self.best:
            self.pool[(r["cls"], frozenset(r["customers"]))] = r
        items = list(self.pool.items())
        n = len(items)
        custs = list(self.pr.customers.keys())
        try:
            m = gp.Model("sp")
            m.setParam("Seed", 0)
            m.setParam("MIPGap", 1e-4)
            m.setParam("NumericFocus", 0)
            m.setParam("Threads", 1)
            m.setParam("TimeLimit", max(1, remain - 1))
            m.setParam("OutputFlag", 0)
            x = m.addVars(n, vtype=GRB.BINARY)
            m.setObjective(gp.quicksum(items[j][1]["cost"] * x[j] for j in range(n)))
            for c in custs:
                m.addConstr(gp.quicksum(x[j] for j in range(n)
                                        if c in items[j][0][1]) == 1)
            for cls in self.pr.cls_ids:
                m.addConstr(gp.quicksum(x[j] for j in range(n)
                                        if items[j][0][0] == cls)
                            <= self.pr.classes[cls]["avail"])
            t_use = defaultdict(list)
            for j, (key, _) in enumerate(items):
                info = self.pr.classes[key[0]]
                t_use[info["truck"]].append(j)
            for t, js in t_use.items():
                if t in self.pr.trucks:
                    m.addConstr(gp.quicksum(x[j] for j in js)
                                <= self.pr.trucks[t].get("num_available", 10 ** 9))
            tr_use = defaultdict(list)
            for j, (key, _) in enumerate(items):
                info = self.pr.classes[key[0]]
                if info["trailer"]:
                    tr_use[info["trailer"]].append(j)
            for t, js in tr_use.items():
                if t in self.pr.trailers:
                    m.addConstr(gp.quicksum(x[j] for j in js)
                                <= self.pr.trailers[t].get("num_available", 10 ** 9))
            m.optimize()
            if m.SolCount > 0:
                obj = m.ObjVal
                if obj < self.best_cost - 1e-6:
                    sol = []
                    for j, (key, ev) in enumerate(items):
                        if x[j].X > 0.5:
                            sol.append(ev)
                    if self.usage_ok(sol):
                        self.best = sol
                        self.best_cost = obj
                        self.log_best()
        except Exception:
            pass
        return self.best, self.best_cost

    def run(self, logger):
        self.logger = logger
        t0 = self.start
        reserve_mip = max(2.0, 0.2 * self.tl) if HAVE_GP else 0.5
        ls_end = t0 + max(0.5, self.tl - reserve_mip)
        try:
            self.local_search(ls_end)
        except Exception:
            pass
        if HAVE_GP:
            remain = self.time_left()
            try:
                self.set_partition(remain)
            except Exception:
                pass
        if self.best is None:
            raise RuntimeError("no feasible solution produced")
        return self.solution_dict(self.best, self.best_cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")

    prob = Problem(args.instance_path)
    solver = Solver(prob, args.time_limit)
    result = solver.run(logger)
    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()