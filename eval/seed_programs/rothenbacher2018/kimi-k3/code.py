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
BIG = 1 << 30


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
        self.dec_time = float(self.params.get("decoupling_time", 0.0))
        self.cpl_time = float(self.params.get("coupling_time", 0.0))
        self.trf_time = float(self.params.get("transfer_time", 0.0))
        self.dec_cost = float(self.params.get("decoupling_cost", 0.0))
        self.cpl_cost = float(self.params.get("coupling_cost", 0.0))
        self.trf_cost = float(self.params.get("transfer_cost", 0.0))
        self.def_svc = float(self.params.get("service_time", 0.0))

        self.depot = data["depot"]["id"]
        self.D = data["distance_matrix"]
        self.TT = data["travel_time_matrix"]
        self.n_orig = len(self.D)

        self.customers = {}
        for c in data.get("truck_customers", []):
            self.customers[c["id"]] = {
                "id": c["id"], "supply": c["supply"],
                "svc": float(c.get("service_time", self.def_svc)), "type": "T"}
        for c in data.get("trailer_customers", []):
            self.customers[c["id"]] = {
                "id": c["id"], "supply": c["supply"],
                "svc": float(c.get("service_time", self.def_svc)), "type": "R"}
        self.truck_custs = [c["id"] for c in data.get("truck_customers", [])]
        self.trailer_custs = [c["id"] for c in data.get("trailer_customers", [])]
        self.trans = [t["id"] for t in data.get("transshipment_locations", [])]

        # parking locations in published order: trailer customers then transshipment
        self.P = list(self.trailer_custs) + list(self.trans)
        self.p_index = {p: i for i, p in enumerate(self.P)}
        self.np = len(self.P)

        self.trucks = {t["id"]: t for t in data.get("fleet", {}).get("truck_types", [])}
        self.trailers = {t["id"]: t for t in data.get("fleet", {}).get("trailer_types", [])}
        self.classes = {}
        for vc in data.get("fleet", {}).get("vehicle_classes", []):
            cid = vc["class_id"]
            tt = vc["truck_type_id"]
            tr = vc.get("trailer_type_id", 0) or 0
            tk = self.trucks.get(tt, {})
            self.classes[cid] = {
                "cls": cid, "truck": tt, "trailer": tr,
                "capT": vc.get("truck_capacity", tk.get("capacity", 0)),
                "capR": vc.get("trailer_capacity", 0) if tr else 0,
                "fixed": vc.get("truck_fixed_cost", tk.get("fixed_cost", 0.0)),
                "f": vc.get("truck_cost_factor", tk.get("cost_factor", 1.0)),
                "tow": vc.get("trailer_towing_cost_multiplier", 0.0) if tr else 0.0,
                "avail": vc.get("num_available", BIG),
            }
        self.cls_ids = list(self.classes.keys())

        # coordinates
        self.data_x = {data["depot"]["id"]: data["depot"]["x"]}
        self.data_y = {data["depot"]["id"]: data["depot"]["y"]}
        for lst in (data.get("truck_customers", []), data.get("trailer_customers", []),
                    data.get("transshipment_locations", [])):
            for l in lst:
                self.data_x[l["id"]] = l["x"]
                self.data_y[l["id"]] = l["y"]

    def dec_id(self, p):
        return self.n_orig + self.p_index[p]

    def cpl_id(self, p):
        return self.n_orig + self.np + self.p_index[p]

    def is_customer(self, x):
        return x in self.customers


# ---------------- TSP helpers ----------------

def nn_path(D, nodes, start):
    rem = list(nodes)
    path = []
    cur = start
    while rem:
        nxt = min(rem, key=lambda x: D[cur][x])
        rem.remove(nxt)
        path.append(nxt)
        cur = nxt
    return path


def two_opt_path(D, nodes, start, end):
    n = len(nodes)
    if n < 3:
        return list(nodes)
    p = list(nodes)
    improved = True
    it = 0
    while improved and it < 25:
        improved = False
        it += 1
        for i in range(n - 1):
            a = start if i == 0 else p[i - 1]
            b = p[i]
            j = i + 1
            while j < n:
                c = p[j]
                d = end if j == n - 1 else p[j + 1]
                if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-9:
                    p[i:j + 1] = p[i:j + 1][::-1]
                    improved = True
                    b = p[i]
                else:
                    j += 1
    return p


def tsp_order(D, nodes, start, end=None):
    if not nodes:
        return []
    if end is None:
        end = start  # closed tour
    p = nn_path(D, nodes, start)
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

        def load_attached(cid):
            nonlocal qT, qR
            s = pr.customers[cid]["supply"]
            add = min(s, capR - qR)
            qR += add
            qT += s - add

        for kind, x in ev:
            if kind in ("C", "S"):
                w = f + (tow if attach else 0.0)
                t += pr.TT[prev][x]
                cost += pr.D[prev][x] * w
                c = pr.customers[x]
                t += c["svc"]
                if kind == "C":
                    load_attached(x)
                else:
                    qT += c["supply"]
                if qT > capT + 1e-9 or qR > capR + 1e-9:
                    return None
                nodes.append(x); phys.append(x); custs.append(x)
                arr[x] = t
                prev = x
            elif kind == "D":
                w = f + tow
                t += pr.TT[prev][x]
                cost += pr.D[prev][x] * w
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
                w = f  # still detached while traveling back
                t += pr.TT[prev][x]
                cost += pr.D[prev][x] * w
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

        w = f + (tow if attach else 0.0)
        t += pr.TT[prev][pr.depot]
        cost += pr.D[prev][pr.depot] * w
        nodes.append(pr.depot)
        phys.append(pr.depot)
        if t > pr.H1 + 1e-6:
            return None
        if t - pr.H0 > pr.max_dur + 1e-6:
            return None
        arr[pr.depot] = t  # final return time
        return {
            "cls": struct["cls"], "struct": struct,
            "nodes": nodes, "phys": phys, "dec": dec_loc, "cpl": cpl_loc,
            "arr": arr, "cost": cost, "customers": list(custs),
            "end": t,
        }

    def parking_pool(self, custs):
        pr = self.pr
        pool = [c for c in custs if pr.customers[c]["type"] == "R"]
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
        custs = list(dict.fromkeys(custs))
        key = (cls_id, frozenset(custs))
        if key in self.cache:
            return self.cache[key]
        info = pr.classes[cls_id]
        has_trailer = info["trailer"] != 0 and info["capR"] > 0
        best = None

        if not has_trailer:
            if sum(pr.customers[c]["supply"] for c in custs) > info["capT"] + 1e-9:
                self.cache[key] = None
                return None
            seq = tsp_order(pr.D, custs, pr.depot, None)
            best = self.eval_struct({"cls": cls_id, "before": seq, "after": [], "blocks": []})
            self.cache[key] = best
            return best

        truck_cs = [c for c in custs if pr.customers[c]["type"] == "T"]
        trl_cs = [c for c in custs if pr.customers[c]["type"] == "R"]

        if not truck_cs:
            seq = tsp_order(pr.D, custs, pr.depot, None)
            best = self.eval_struct({"cls": cls_id, "before": seq, "after": [], "blocks": []})
            self.cache[key] = best
            return best

        pool = self.parking_pool(custs)
        if not pool:
            self.cache[key] = None
            return None

        s_variants = [truck_cs]
        extra = []
        for c in trl_cs:
            d_dep = pr.D[pr.depot][c]
            d_pool = min(pr.D[p][c] for p in pool)
            if d_pool < 0.75 * d_dep:
                extra.append(c)
        if extra:
            s_variants.append(truck_cs + extra)

        for sset in s_variants:
            groups = self.partition_groups(list(sset), info["capT"])
            if groups is None or len(groups) > len(pool):
                continue
            used = set()
            parks = []
            ok = True
            for g in groups:
                cx = sum(pr.data_x[c] for c in g) / len(g)
                cy = sum(pr.data_y[c] for c in g) / len(g)
                cand = None
                for p in pool:
                    if p in used:
                        continue
                    dd = (pr.data_x[p] - cx) ** 2 + (pr.data_y[p] - cy) ** 2
                    if cand is None or dd < cand[0]:
                        cand = (dd, p)
                if cand is None:
                    ok = False
                    break
                used.add(cand[1])
                parks.append(cand[1])
            if not ok:
                continue
            park_to_group = dict(zip(parks, groups))
            first = min(parks, key=lambda p: pr.D[pr.depot][p])
            ordered_parks = [first]
            rest = [p for p in parks if p != first]
            cur = first
            while rest:
                nxt = min(rest, key=lambda p: pr.D[cur][p])
                rest.remove(nxt)
                ordered_parks.append(nxt)
                cur = nxt
            blocks = []
            for p in ordered_parks:
                mem = tsp_order(pr.D, park_to_group[p], p, None)
                blocks.append((p, mem))
            sset_set = set(sset)
            park_set = set(parks)
            A = [c for c in trl_cs if c not in park_set and c not in sset_set]
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
                res = self.eval_struct({"cls": cls_id, "before": before,
                                        "after": after, "blocks": blocks})
                if res and (best is None or res["cost"] < best["cost"]):
                    best = res
        self.cache[key] = best
        return best


# ---------------- Solver ----------------

class Solver:
    def __init__(self, prob, time_limit):
        self.pr = prob
        self.tl = max(1, time_limit)
        self.start = time.time()
        self.builder = Builder(prob)
        self.pool = {}
        self.best = None
        self.best_cost = float("inf")
        self.logger = None

    def time_left(self):
        return self.tl - (time.time() - self.start)

    def construct(self, custs, cls):
        try:
            r = self.builder.construct(custs, cls)
        except Exception:
            r = None
        if r:
            self.pool[(cls, frozenset(r["customers"]))] = r
        return r

    def usage_ok(self, routes):
        cls_cnt = defaultdict(int)
        tk_cnt = defaultdict(int)
        tr_cnt = defaultdict(int)
        for r in routes:
            info = self.pr.classes[r["cls"]]
            cls_cnt[r["cls"]] += 1
            tk_cnt[info["truck"]] += 1
            if info["trailer"]:
                tr_cnt[info["trailer"]] += 1
        for c, v in cls_cnt.items():
            if v > self.pr.classes[c]["avail"]:
                return False
        for t, v in tk_cnt.items():
            if t in self.pr.trucks and v > self.pr.trucks[t].get("num_available", BIG):
                return False
        for t, v in tr_cnt.items():
            if t in self.pr.trailers and v > self.pr.trailers[t].get("num_available", BIG):
                return False
        return True

    def total(self, routes):
        return sum(r["cost"] for r in routes)

    # ---------- initial solution: sequential seed-and-grow ----------
    def seed_build(self):
        pr = self.pr
        unassigned = set(pr.customers.keys())
        routes = []
        used_cls = defaultdict(int)
        used_tk = defaultdict(int)
        used_tr = defaultdict(int)

        def avail(cls):
            info = pr.classes[cls]
            if used_cls[cls] >= info["avail"]:
                return False
            t = info["truck"]
            if t in pr.trucks and used_tk[t] >= pr.trucks[t].get("num_available", BIG):
                return False
            tr = info["trailer"]
            if tr and tr in pr.trailers and used_tr[tr] >= pr.trailers[tr].get("num_available", BIG):
                return False
            return True

        while unassigned:
            seed = max(unassigned, key=lambda c: pr.D[pr.depot][c])
            best_route = None
            best_key = None
            for cls in pr.cls_ids:
                if not avail(cls):
                    continue
                members = [seed]
                r = self.construct(members, cls)
                if r is None:
                    # seed may need a companion (e.g., truck customer needing a parking)
                    base = None
                    others = sorted(unassigned - {seed}, key=lambda c: pr.D[seed][c])[:10]
                    for other in others:
                        r = self.construct([seed, other], cls)
                        if r is not None:
                            base = [seed, other]
                            break
                    if base is None:
                        continue
                    members = base
                # greedily grow the route
                while True:
                    mem_set = set(members)
                    cands = sorted(unassigned - mem_set,
                                   key=lambda c: min(pr.D[c][m] for m in members))[:10]
                    cand_best = None
                    for c in cands:
                        nr = self.construct(members + [c], cls)
                        if nr is None:
                            continue
                        delta = nr["cost"] - r["cost"]
                        if cand_best is None or delta < cand_best[0]:
                            cand_best = (delta, c, nr)
                    if cand_best is None:
                        break
                    members = members + [cand_best[1]]
                    r = cand_best[2]
                key = (-len(members), r["cost"] / max(1, len(members)))
                if best_key is None or key < best_key:
                    best_key = key
                    best_route = r
            if best_route is None:
                return None
            routes.append(best_route)
            info = pr.classes[best_route["cls"]]
            used_cls[best_route["cls"]] += 1
            used_tk[info["truck"]] += 1
            if info["trailer"]:
                used_tr[info["trailer"]] += 1
            for c in best_route["customers"]:
                unassigned.discard(c)
        return routes

    # ---------- greedy insertion (used for destroy & repair) ----------
    def greedy_build(self, custs_subset, base_routes):
        pr = self.pr
        routes = list(base_routes)
        cls_cnt = defaultdict(int)
        tk = defaultdict(int)
        tr = defaultdict(int)
        for r in routes:
            info = pr.classes[r["cls"]]
            cls_cnt[r["cls"]] += 1
            tk[info["truck"]] += 1
            if info["trailer"]:
                tr[info["trailer"]] += 1

        def can_new(cls):
            info = pr.classes[cls]
            if cls_cnt[cls] + 1 > info["avail"]:
                return False
            t = info["truck"]
            if t in pr.trucks and tk[t] + 1 > pr.trucks[t].get("num_available", BIG):
                return False
            tt = info["trailer"]
            if tt and tt in pr.trailers and tr[tt] + 1 > pr.trailers[tt].get("num_available", BIG):
                return False
            return True

        def register(nr):
            info = pr.classes[nr["cls"]]
            cls_cnt[nr["cls"]] += 1
            tk[info["truck"]] += 1
            if info["trailer"]:
                tr[info["trailer"]] += 1

        for cust in custs_subset:
            best_delta = float("inf")
            best_choice = None
            for i, r in enumerate(routes):
                if cust in r["customers"]:
                    continue
                nr = self.construct(r["customers"] + [cust], r["cls"])
                if nr:
                    d = nr["cost"] - r["cost"]
                    if d < best_delta:
                        best_delta = d
                        best_choice = (i, nr)
            for cls in pr.cls_ids:
                if not can_new(cls):
                    continue
                nr = self.construct([cust], cls)
                if nr and nr["cost"] < best_delta:
                    best_delta = nr["cost"]
                    best_choice = (-1, nr)
            if best_choice is not None:
                i, nr = best_choice
                if i == -1:
                    routes.append(nr)
                    register(nr)
                else:
                    routes[i] = nr
                routes = [r for r in routes if r["customers"]]
                continue
            # last resort: pull a partner out of an existing route
            best_pair = None
            for i, r in enumerate(routes):
                for other in r["customers"]:
                    rest = [x for x in r["customers"] if x != other]
                    rem = self.construct(rest, r["cls"]) if rest else None
                    if rest and rem is None:
                        continue
                    for cls in pr.cls_ids:
                        if not can_new(cls):
                            continue
                        nr = self.construct([cust, other], cls)
                        if nr is None:
                            continue
                        d = (rem["cost"] if rem else 0.0) - r["cost"] + nr["cost"]
                        if best_pair is None or d < best_pair[0]:
                            best_pair = (d, i, rem, nr)
            if best_pair is None:
                return None
            _, i, rem, nr = best_pair
            if rem is None:
                routes.pop(i)
            else:
                routes[i] = rem
            routes.append(nr)
            register(nr)
        return routes if self.usage_ok(routes) else None

    def initial_solution(self):
        sol = None
        try:
            sol = self.seed_build()
        except Exception:
            sol = None
        if sol is None:
            pr = self.pr
            custs = sorted(pr.customers.keys(), key=lambda c: -pr.D[pr.depot][c])
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
                try:
                    self.logger.log(self.best_cost)
                except Exception:
                    pass

    def solution_dict(self, routes, obj):
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

    # ---------- local search ----------
    def local_search(self, budget_end, start_routes=None):
        pr = self.pr
        routes = start_routes if start_routes is not None else self.initial_solution()
        if routes is None:
            return
        cur = self.total(routes)
        if self.best is None or cur < self.best_cost:
            self.best = list(routes)
            self.best_cost = cur
            self.log_best()
        allcusts = list(pr.customers.keys())
        T0 = max(cur / max(1, len(routes)), 1.0)
        n_iter = 0
        while time.time() < budget_end:
            n_iter += 1
            try:
                frac = max(0.0, (budget_end - time.time()) /
                           max(1e-6, budget_end - self.start))
                T = T0 * frac + 1e-3
                op = RNG.random()
                new_routes = None
                if not routes:
                    break
                if op < 0.45:  # relocate
                    ri = RNG.randrange(len(routes))
                    r = routes[ri]
                    if not r["customers"]:
                        continue
                    c = RNG.choice(r["customers"])
                    rest = [x for x in r["customers"] if x != c]
                    if rest:
                        rem_r = self.construct(rest, r["cls"])
                        if rem_r is None:
                            continue
                        base_delta = rem_r["cost"] - r["cost"]
                    else:
                        rem_r = None
                        base_delta = -r["cost"]
                    best_mv = None
                    idxs = list(range(len(routes)))
                    RNG.shuffle(idxs)
                    for j in idxs[:12]:
                        if j == ri:
                            continue
                        rr = routes[j]
                        nr = self.construct(rr["customers"] + [c], rr["cls"])
                        if nr:
                            d = base_delta + nr["cost"] - rr["cost"]
                            if best_mv is None or d < best_mv[0]:
                                best_mv = (d, j, nr)
                    for cls in pr.cls_ids:
                        if not pr.classes[cls]["avail"]:
                            continue
                        nr = self.construct([c], cls)
                        if nr:
                            d = base_delta + nr["cost"]
                            if best_mv is None or d < best_mv[0]:
                                best_mv = (d, -1, nr)
                    if best_mv is None:
                        continue
                    cand = list(routes)
                    if rem_r is not None:
                        cand[ri] = rem_r
                    else:
                        cand.pop(ri)
                    _, j, nr = best_mv
                    if j == -1:
                        cand.append(nr)
                    else:
                        jj = j if (rem_r is not None or j < ri) else j - 1
                        cand[jj] = nr
                    cand = [x for x in cand if x and x["customers"]]
                    new_routes = cand
                elif op < 0.75 and len(routes) >= 2:  # swap
                    i, j = RNG.sample(range(len(routes)), 2)
                    a, b = routes[i], routes[j]
                    if not a["customers"] or not b["customers"]:
                        continue
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
                elif op < 0.9:  # change vehicle class of a route
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
                    ok = True
                    for r in routes:
                        restl = [x for x in r["customers"] if x not in removed]
                        if not restl:
                            continue
                        nr = self.construct(restl, r["cls"])
                        if nr is None:
                            ok = False
                            break
                        stripped.append(nr)
                    if ok:
                        new_routes = self.greedy_build(removed, stripped)
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
            except Exception:
                continue
            if n_iter % 64 == 0 and self.time_left() < 2.0:
                break

    # ---------- set partitioning over the route pool ----------
    def set_partition(self, remain):
        if not HAVE_GP or not self.pool or remain < 2:
            return
        for r in (self.best or []):
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
            m.setParam("TimeLimit", max(1, int(remain - 1)))
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
            tr_use = defaultdict(list)
            for j, (key, _) in enumerate(items):
                info = self.pr.classes[key[0]]
                t_use[info["truck"]].append(j)
                if info["trailer"]:
                    tr_use[info["trailer"]].append(j)
            for t, js in t_use.items():
                if t in self.pr.trucks:
                    m.addConstr(gp.quicksum(x[j] for j in js)
                                <= self.pr.trucks[t].get("num_available", BIG))
            for t, js in tr_use.items():
                if t in self.pr.trailers:
                    m.addConstr(gp.quicksum(x[j] for j in js)
                                <= self.pr.trailers[t].get("num_available", BIG))
            m.optimize()
            if m.SolCount > 0:
                obj = m.ObjVal
                if obj < self.best_cost - 1e-6:
                    sol = [items[j][1] for j in range(n) if x[j].X > 0.5]
                    if sol and self.usage_ok(sol):
                        self.best = sol
                        self.best_cost = obj
                        self.log_best()
        except Exception:
            pass

    def run(self, logger):
        self.logger = logger
        reserve = max(2.0, 0.2 * self.tl) if HAVE_GP else 0.3
        ls_end = self.start + max(0.5, self.tl - reserve)
        try:
            self.local_search(ls_end)
        except Exception:
            pass
        if self.best is None:
            try:
                sol = self.seed_build() or self.greedy_build(
                    sorted(self.pr.customers.keys(),
                           key=lambda c: -self.pr.D[self.pr.depot][c]), [])
                if sol:
                    self.best = sol
                    self.best_cost = self.total(sol)
                    self.log_best()
            except Exception:
                pass
        if HAVE_GP and self.best is not None:
            try:
                self.set_partition(self.time_left())
            except Exception:
                pass
        # polish with remaining time
        if self.best is not None and self.time_left() > 1.0:
            try:
                self.local_search(time.time() + self.time_left() - 0.5,
                                  start_routes=list(self.best))
            except Exception:
                pass
        if self.best is None:
            # last-resort: output an empty structure rather than crashing
            return {"objective_value": 0.0, "routes": [],
                    "expanded_node_convention": ""}
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
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    prob = Problem(args.instance_path)
    solver = Solver(prob, args.time_limit)
    result = solver.run(logger)
    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()