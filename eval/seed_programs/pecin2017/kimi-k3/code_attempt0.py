import argparse
import heapq
import json
import math
import random
import time
from array import array

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GP = True
except Exception:
    gp = None
    GRB = None
    HAVE_GP = False

EPS = 1e-7
TOL = 1e-9
BIGR = 1e18
POOL_CAP = 25000


class Solver:
    def __init__(self, inst):
        pp = inst.get("problem_parameters", {})
        self.Q = float(pp["vehicle_capacity"])
        self.H = float(pp["planning_horizon"])
        nodes = inst["nodes"]
        src = None
        snk = None
        cust = []
        for nd in nodes:
            t = nd.get("type", "")
            if t == "depot_source":
                src = nd
            elif t == "depot_sink":
                snk = nd
            else:
                cust.append(nd)
        if src is None or snk is None:
            src = nodes[0]
            snk = nodes[-1]
            cust = [nd for nd in nodes if nd is not src and nd is not snk]
        self.n = len(cust)
        N = self.n + 2
        self.N = N
        self.SRC = 0
        self.SNK = N - 1
        ordered = [src] + cust + [snk]
        self.orig_id = [int(nd["id"]) for nd in ordered]
        self.X = [float(nd["x"]) for nd in ordered]
        self.Y = [float(nd["y"]) for nd in ordered]
        self.dem = [float(nd["demand"]) for nd in ordered]
        self.ser = [float(nd["service_time"]) for nd in ordered]
        self.ea = [float(nd["time_window_open"]) for nd in ordered]
        self.la = [float(nd["time_window_close"]) for nd in ordered]
        self.dem[self.SRC] = 0.0
        self.dem[self.SNK] = 0.0
        self.ser[self.SRC] = 0.0
        self.ser[self.SNK] = 0.0
        self.ea[self.SRC] = 0.0
        self.la[self.SRC] = self.H
        self.ea[self.SNK] = 0.0
        self.la[self.SNK] = self.H
        X = self.X
        Y = self.Y
        D = []
        for i in range(N):
            row = array('d', [0.0] * N)
            xi = X[i]
            yi = Y[i]
            for j in range(N):
                dx = xi - X[j]
                dy = yi - Y[j]
                row[j] = math.sqrt(dx * dx + dy * dy)
            D.append(row)
        self.D = D
        self.cust = list(range(1, self.n + 1))
        G = min(self.n, 50)
        nb = {}
        for u in self.cust:
            Du = D[u]
            nb[u] = [v for v in heapq.nsmallest(G + 1, self.cust, key=lambda v, Du=Du: Du[v]) if v != u][:G]
        self.nb = nb
        self.next_uid = 0
        self.reset_routes()
        self.pool = {}
        self.best_cost = float('inf')
        self.best_routes = None
        self.last_log = 0.0
        self.logged_cost = float('inf')

    # ---------- state management ----------
    def reset_routes(self):
        self.routes = {}
        self.route_load = {}
        self.route_cost = {}
        self.route_E = {}
        self.route_L = {}
        self.route_of = {}
        self.pos_of = {}

    def new_route(self, members):
        uid = self.next_uid
        self.next_uid += 1
        self.routes[uid] = list(members)
        self.recompute(uid)
        return uid

    def delete_route(self, uid):
        self.routes.pop(uid, None)
        self.route_load.pop(uid, None)
        self.route_cost.pop(uid, None)
        self.route_E.pop(uid, None)
        self.route_L.pop(uid, None)

    def recompute(self, uid):
        r = self.routes[uid]
        D = self.D
        ser = self.ser
        ea = self.ea
        la = self.la
        m = len(r)
        nodes = [self.SRC] + r + [self.SNK]
        E = [0.0] * (m + 2)
        cost = 0.0
        for i in range(1, m + 2):
            a = nodes[i - 1]
            b = nodes[i]
            arr = E[i - 1] + ser[a] + D[a][b]
            eb = ea[b]
            E[i] = arr if arr > eb else eb
            cost += D[a][b]
        L = [0.0] * (m + 2)
        L[m + 1] = la[self.SNK]
        for i in range(m, -1, -1):
            a = nodes[i]
            b = nodes[i + 1]
            v = L[i + 1] - ser[a] - D[a][b]
            lb = la[a]
            L[i] = lb if lb < v else v
        self.route_E[uid] = E
        self.route_L[uid] = L
        self.route_cost[uid] = cost
        load = 0.0
        for u in r:
            load += self.dem[u]
        self.route_load[uid] = load
        ro = self.route_of
        po = self.pos_of
        for i, u in enumerate(r):
            ro[u] = uid
            po[u] = i

    def copy_routes(self):
        return {uid: list(r) for uid, r in self.routes.items()}

    def restore(self, cp):
        self.reset_routes()
        self.routes = {uid: list(r) for uid, r in cp.items()}
        for uid in self.routes:
            self.recompute(uid)

    def adopt_routes(self, sel):
        self.reset_routes()
        for rt in sel:
            if rt:
                self.new_route(rt)

    def total_cost(self):
        return sum(self.route_cost.values())

    def feasible(self, rt):
        load = 0.0
        for u in rt:
            load += self.dem[u]
        if load > self.Q + EPS:
            return False
        t = 0.0
        prev = self.SRC
        D = self.D
        ser = self.ser
        ea = self.ea
        la = self.la
        for u in rt:
            arr = t + ser[prev] + D[prev][u]
            eu = ea[u]
            st = arr if arr > eu else eu
            if st > la[u] + EPS:
                return False
            t = st
            prev = u
        return t + ser[prev] + D[prev][self.SNK] <= self.H + EPS

    # ---------- insertion helpers ----------
    def ins_ok(self, u, uid, pos):
        r = self.routes[uid]
        m = len(r)
        a = r[pos - 1] if pos > 0 else self.SRC
        b = r[pos] if pos < m else self.SNK
        E = self.route_E[uid]
        L = self.route_L[uid]
        D = self.D
        arr = E[pos] + self.ser[a] + D[a][u]
        eu = self.ea[u]
        st = arr if arr > eu else eu
        if st > self.la[u] + EPS:
            return False
        return st + self.ser[u] + D[u][b] <= L[pos + 1] + EPS

    def seg_insert_ok(self, seg, uid, pos, seg_dem):
        if self.route_load[uid] + seg_dem > self.Q + EPS:
            return False
        r = self.routes[uid]
        m = len(r)
        D = self.D
        ser = self.ser
        ea = self.ea
        la = self.la
        E = self.route_E[uid]
        L = self.route_L[uid]
        a = r[pos - 1] if pos > 0 else self.SRC
        b = r[pos] if pos < m else self.SNK
        u0 = seg[0]
        arr = E[pos] + ser[a] + D[a][u0]
        e0 = ea[u0]
        st = arr if arr > e0 else e0
        if st > la[u0] + EPS:
            return False
        for k in range(1, len(seg)):
            p = seg[k - 1]
            u = seg[k]
            arr = st + ser[p] + D[p][u]
            eu = ea[u]
            st = arr if arr > eu else eu
            if st > la[u] + EPS:
                return False
        last = seg[-1]
        return st + ser[last] + D[last][b] <= L[pos + 1] + EPS

    def insert_options(self, u, uid):
        r = self.routes[uid]
        if self.route_load[uid] + self.dem[u] > self.Q + EPS:
            return None, None
        D = self.D
        ser = self.ser
        ea = self.ea
        la = self.la
        E = self.route_E[uid]
        L = self.route_L[uid]
        eu = ea[u]
        lu = la[u]
        su = ser[u]
        best = None
        best2 = None
        m = len(r)
        for pos in range(m + 1):
            a = r[pos - 1] if pos > 0 else self.SRC
            b = r[pos] if pos < m else self.SNK
            arr = E[pos] + ser[a] + D[a][u]
            st = arr if arr > eu else eu
            if st <= lu + EPS and st + su + D[u][b] <= L[pos + 1] + EPS:
                delta = D[a][u] + D[u][b] - D[a][b]
                if best is None or delta < best[0]:
                    best2 = best
                    best = (delta, pos)
                elif best2 is None or delta < best2[0]:
                    best2 = (delta, pos)
        return best, best2

    # ---------- local search sweeps ----------
    def sweep_relocate(self, dl):
        D = self.D
        ap = 0
        for u in self.cust:
            uid1 = self.route_of.get(u)
            if uid1 is None:
                continue
            r1 = self.routes[uid1]
            i = self.pos_of[u]
            m1 = len(r1)
            a = r1[i - 1] if i > 0 else self.SRC
            b = r1[i + 1] if i + 1 < m1 else self.SNK
            rd = D[a][b] - D[a][u] - D[u][b]
            du = self.dem[u]
            moved = False
            for v in self.nb[u]:
                uid2 = self.route_of.get(v)
                if uid2 is None or uid2 == uid1:
                    continue
                if self.route_load[uid2] + du > self.Q + EPS:
                    continue
                r2 = self.routes[uid2]
                j = self.pos_of[v]
                m2 = len(r2)
                for pos in (j, j + 1):
                    c = r2[pos - 1] if pos > 0 else self.SRC
                    d = r2[pos] if pos < m2 else self.SNK
                    if rd + D[c][u] + D[u][d] - D[c][d] < -TOL and self.ins_ok(u, uid2, pos):
                        r1.pop(i)
                        if r1:
                            self.recompute(uid1)
                        else:
                            self.delete_route(uid1)
                        self.routes[uid2].insert(pos, u)
                        self.recompute(uid2)
                        ap += 1
                        moved = True
                        break
                if moved:
                    break
            if (u & 15) == 0 and time.time() > dl:
                return ap
        return ap

    def sweep_swap(self, dl):
        D = self.D
        ap = 0
        for u in self.cust:
            uid1 = self.route_of.get(u)
            if uid1 is None:
                continue
            r1 = self.routes[uid1]
            i = self.pos_of[u]
            m1 = len(r1)
            du = self.dem[u]
            a = r1[i - 1] if i > 0 else self.SRC
            b = r1[i + 1] if i + 1 < m1 else self.SNK
            base1 = D[a][u] + D[u][b]
            moved = False
            for v in self.nb[u]:
                if v <= u:
                    continue
                uid2 = self.route_of.get(v)
                if uid2 is None or uid2 == uid1:
                    continue
                r2 = self.routes[uid2]
                j = self.pos_of[v]
                m2 = len(r2)
                dv = self.dem[v]
                if self.route_load[uid1] - du + dv > self.Q + EPS:
                    continue
                if self.route_load[uid2] - dv + du > self.Q + EPS:
                    continue
                c = r2[j - 1] if j > 0 else self.SRC
                d = r2[j + 1] if j + 1 < m2 else self.SNK
                delta = D[a][v] + D[v][b] - base1 + D[c][u] + D[u][d] - D[c][v] - D[v][d]
                if delta < -TOL:
                    r1n = list(r1)
                    r1n[i] = v
                    r2n = list(r2)
                    r2n[j] = u
                    if self.feasible(r1n) and self.feasible(r2n):
                        self.routes[uid1] = r1n
                        self.recompute(uid1)
                        self.routes[uid2] = r2n
                        self.recompute(uid2)
                        ap += 1
                        moved = True
                        break
            if (u & 15) == 0 and time.time() > dl:
                return ap
        return ap

    def sweep_2opt(self, dl):
        D = self.D
        ap = 0
        for uid in list(self.routes.keys()):
            if uid not in self.routes:
                continue
            r = self.routes[uid]
            improved = True
            while improved:
                improved = False
                m = len(r)
                for p in range(m):
                    prev = r[p - 1] if p > 0 else self.SRC
                    Dpp = D[prev]
                    rp = r[p]
                    brk = False
                    for q in range(p + 1, m):
                        nxt = r[q + 1] if q + 1 < m else self.SNK
                        rq = r[q]
                        if Dpp[rq] + D[rp][nxt] - Dpp[rp] - D[rq][nxt] < -TOL:
                            rn = r[:p] + r[p:q + 1][::-1] + r[q + 1:]
                            if self.feasible(rn):
                                self.routes[uid] = rn
                                self.recompute(uid)
                                r = rn
                                ap += 1
                                improved = True
                                brk = True
                                break
                    if brk:
                        break
            if time.time() > dl:
                return ap
        return ap

    def sweep_2optstar(self, dl):
        D = self.D
        ap = 0
        for u in self.cust:
            uid1 = self.route_of.get(u)
            if uid1 is None:
                continue
            r1 = self.routes[uid1]
            i = self.pos_of[u]
            m1 = len(r1)
            nxt1 = r1[i + 1] if i + 1 < m1 else self.SNK
            Dun = D[u]
            moved = False
            for v in self.nb[u]:
                uid2 = self.route_of.get(v)
                if uid2 is None or uid2 == uid1:
                    continue
                r2 = self.routes[uid2]
                j = self.pos_of[v]
                m2 = len(r2)
                nxt2 = r2[j + 1] if j + 1 < m2 else self.SNK
                if Dun[nxt2] + D[v][nxt1] - Dun[nxt1] - D[v][nxt2] < -TOL:
                    r1n = r1[:i + 1] + r2[j + 1:]
                    r2n = r2[:j + 1] + r1[i + 1:]
                    if self.feasible(r1n) and self.feasible(r2n):
                        self.routes[uid1] = r1n
                        self.recompute(uid1)
                        if r2n:
                            self.routes[uid2] = r2n
                            self.recompute(uid2)
                        else:
                            self.delete_route(uid2)
                        ap += 1
                        moved = True
                        break
            if (u & 15) == 0 and time.time() > dl:
                return ap
        return ap

    def sweep_oropt(self, dl):
        D = self.D
        ap = 0
        for uid in list(self.routes.keys()):
            if uid not in self.routes:
                continue
            improved = True
            while improved:
                improved = False
                r = self.routes[uid]
                m = len(r)
                if m < 2:
                    break
                done = False
                for Ls in (1, 2, 3):
                    if done:
                        break
                    if m < Ls + 1:
                        continue
                    for i in range(m - Ls + 1):
                        if done:
                            break
                        seg = r[i:i + Ls]
                        s0 = seg[0]
                        sl = seg[-1]
                        a = r[i - 1] if i > 0 else self.SRC
                        b = r[i + Ls] if i + Ls < m else self.SNK
                        rem = D[a][b] - D[a][s0] - D[sl][b]
                        rest = r[:i] + r[i + Ls:]
                        mr = len(rest)
                        for pos in range(mr + 1):
                            c = rest[pos - 1] if pos > 0 else self.SRC
                            d = rest[pos] if pos < mr else self.SNK
                            if rem + D[c][s0] + D[sl][d] - D[c][d] < -TOL:
                                rn = rest[:pos] + seg + rest[pos:]
                                if self.feasible(rn):
                                    self.routes[uid] = rn
                                    self.recompute(uid)
                                    ap += 1
                                    improved = True
                                    done = True
                                    break
            if time.time() > dl:
                return ap
        return ap

    def sweep_seg_relocate(self, dl):
        D = self.D
        ap = 0
        for uid1 in list(self.routes.keys()):
            if uid1 not in self.routes:
                continue
            improved = True
            while improved and uid1 in self.routes:
                improved = False
                r1 = self.routes[uid1]
                m = len(r1)
                if m < 2:
                    break
                done = False
                for Ls in (2, 3):
                    if done:
                        break
                    if m < Ls:
                        continue
                    for i in range(m - Ls + 1):
                        if done:
                            break
                        seg = r1[i:i + Ls]
                        s0 = seg[0]
                        sl = seg[-1]
                        segdem = self.dem[s0] + self.dem[seg[1]] + (self.dem[seg[2]] if Ls == 3 else 0.0)
                        a = r1[i - 1] if i > 0 else self.SRC
                        b = r1[i + Ls] if i + Ls < m else self.SNK
                        rem = D[a][b] - D[a][s0] - D[sl][b]
                        for v in self.nb[s0]:
                            if done:
                                break
                            uid2 = self.route_of.get(v)
                            if uid2 is None or uid2 == uid1:
                                continue
                            if self.route_load[uid2] + segdem > self.Q + EPS:
                                continue
                            r2 = self.routes[uid2]
                            j = self.pos_of[v]
                            m2 = len(r2)
                            for pos in (j, j + 1):
                                c = r2[pos - 1] if pos > 0 else self.SRC
                                d = r2[pos] if pos < m2 else self.SNK
                                if rem + D[c][s0] + D[sl][d] - D[c][d] < -TOL and \
                                        self.seg_insert_ok(seg, uid2, pos, segdem):
                                    r1[i:i + Ls] = []
                                    if r1:
                                        self.recompute(uid1)
                                    else:
                                        self.delete_route(uid1)
                                    self.routes[uid2][pos:pos] = seg
                                    self.recompute(uid2)
                                    ap += 1
                                    improved = True
                                    done = True
                                    break
            if time.time() > dl:
                return ap
        return ap

    def local_search(self, dl, logger=None):
        while time.time() < dl:
            ap = 0
            ap += self.sweep_relocate(dl)
            if time.time() < dl:
                ap += self.sweep_2optstar(dl)
            if time.time() < dl:
                ap += self.sweep_swap(dl)
            if time.time() < dl:
                ap += self.sweep_2opt(dl)
            if time.time() < dl:
                ap += self.sweep_seg_relocate(dl)
            if time.time() < dl:
                ap += self.sweep_oropt(dl)
            if ap > 0:
                self.update_best(logger)
            if ap == 0:
                break

    # ---------- construction ----------
    def build_initial(self, dl):
        unrouted = set(self.cust)
        la = self.la
        D = self.D
        SRC = self.SRC
        while unrouted:
            if time.time() >= dl:
                for u in sorted(unrouted, key=lambda u: (la[u], -D[SRC][u])):
                    self.new_route([u])
                break
            seed = min(unrouted, key=lambda u: (la[u], -D[SRC][u]))
            unrouted.discard(seed)
            uid = self.new_route([seed])
            last = seed
            while time.time() < dl:
                cand = []
                seen = set()
                for w in self.nb[seed]:
                    if w in unrouted and w not in seen:
                        cand.append(w)
                        seen.add(w)
                for w in self.nb[last]:
                    if w in unrouted and w not in seen:
                        cand.append(w)
                        seen.add(w)
                if not cand:
                    cand = sorted(unrouted, key=lambda u: la[u])[:40]
                if len(cand) > 150:
                    cand = cand[:150]
                best_choice = None
                for u in cand:
                    b1, b2 = self.insert_options(u, uid)
                    if b1 is None:
                        continue
                    c1 = b1[0]
                    c2 = b2[0] if b2 is not None else BIGR
                    key = (c2 - c1, -c1)
                    if best_choice is None or key > best_choice[0]:
                        best_choice = (key, u, b1[1])
                if best_choice is None:
                    break
                _, u, pos = best_choice
                self.routes[uid].insert(pos, u)
                self.recompute(uid)
                unrouted.discard(u)
                last = u

    # ---------- LNS ----------
    def destroy_route(self, rng):
        uid = rng.choice(list(self.routes.keys()))
        sel = list(self.routes[uid])
        for u in sel:
            self.route_of.pop(u, None)
            self.pos_of.pop(u, None)
        self.delete_route(uid)
        return sel

    def destroy(self, op, q, rng):
        allc = self.cust
        D = self.D
        if op == 'worst':
            sav = []
            for u in allc:
                uid = self.route_of[u]
                r = self.routes[uid]
                i = self.pos_of[u]
                a = r[i - 1] if i > 0 else self.SRC
                b = r[i + 1] if i + 1 < len(r) else self.SNK
                sav.append((D[a][u] + D[u][b] - D[a][b], u))
            sav.sort(reverse=True)
            sel = []
            for idx, (s, u) in enumerate(sav):
                if len(sel) >= q:
                    break
                if idx == 0 or rng.random() < 0.4:
                    sel.append(u)
            idx = 0
            while len(sel) < q and idx < len(sav):
                u = sav[idx][1]
                if u not in sel:
                    sel.append(u)
                idx += 1
        elif op == 'related':
            seed = rng.choice(allc)
            H = self.H if self.H > 0 else 1.0
            Q = self.Q if self.Q > 0 else 1.0
            Ds = D[seed]
            maxD = max((Ds[v] for v in allc), default=1.0) or 1.0
            es = self.ea[seed]
            ds = self.dem[seed]
            rel = sorted(allc, key=lambda v: Ds[v] / maxD + abs(es - self.ea[v]) / H + abs(ds - self.dem[v]) / Q)
            sel = []
            for v in rel:
                if len(sel) >= q:
                    break
                if v == seed or rng.random() < 0.6:
                    sel.append(v)
            for v in rel:
                if len(sel) >= q:
                    break
                if v not in sel:
                    sel.append(v)
        else:
            sel = rng.sample(allc, min(q, len(allc)))
        affected = set()
        for u in sel:
            uid = self.route_of.pop(u, None)
            pos = self.pos_of.pop(u, None)
            if uid is None:
                continue
            self.routes[uid].pop(pos)
            affected.add(uid)
        for uid in affected:
            if self.routes[uid]:
                self.recompute(uid)
            else:
                self.delete_route(uid)
        return sel

    def repair(self, removed, rng, dl):
        cand = {}
        for u in removed:
            s = set()
            for v in self.nb[u]:
                uid = self.route_of.get(v)
                if uid is not None:
                    s.add(uid)
            cand[u] = s
        ins = {}
        for u in removed:
            d = {}
            for uid in cand[u]:
                b1, _ = self.insert_options(u, uid)
                if b1 is not None:
                    d[uid] = b1
            ins[u] = d
        remaining = list(removed)
        rem_set = set(removed)
        D = self.D
        while remaining:
            if time.time() > dl:
                for u in remaining:
                    self.new_route([u])
                return
            bestkey = None
            bestu = None
            bestopt = None
            for u in remaining:
                o = ins[u]
                c1 = None
                o1 = None
                c2 = None
                for uid, (dlta, pos) in o.items():
                    if c1 is None or dlta < c1:
                        c2 = c1
                        c1 = dlta
                        o1 = (uid, pos)
                    elif c2 is None or dlta < c2:
                        c2 = dlta
                nr = D[self.SRC][u] + D[u][self.SNK]
                if c1 is None or nr < c1:
                    c2 = c1
                    c1 = nr
                    o1 = None
                elif c2 is None or nr < c2:
                    c2 = nr
                if c2 is None:
                    c2 = BIGR
                key = (c2 - c1, -c1)
                if bestkey is None or key > bestkey:
                    bestkey = key
                    bestu = u
                    bestopt = o1
            u = bestu
            remaining.remove(u)
            rem_set.discard(u)
            ins.pop(u)
            if bestopt is None:
                uid = self.new_route([u])
                for v in self.nb[u]:
                    if v in rem_set:
                        cand[v].add(uid)
                        b1, _ = self.insert_options(v, uid)
                        if b1 is not None:
                            ins[v][uid] = b1
            else:
                uid, pos = bestopt
                self.routes[uid].insert(pos, u)
                self.recompute(uid)
                for w in remaining:
                    if uid in cand[w]:
                        b1, _ = self.insert_options(w, uid)
                        if b1 is not None:
                            ins[w][uid] = b1
                        else:
                            ins[w].pop(uid, None)

    def lns(self, dl, rng, logger, ls_cap):
        cur_cost = self.total_cost()
        since = 0
        while time.time() < dl:
            save = self.copy_routes()
            pre_best = self.best_cost
            r = rng.random()
            if r < 0.30:
                op = 'random'
            elif r < 0.60:
                op = 'worst'
            elif r < 0.85:
                op = 'related'
            else:
                op = 'route'
            if op == 'route' and len(self.routes) > 1:
                removed = self.destroy_route(rng)
            else:
                lo = max(2, int(0.05 * self.n))
                hi = max(lo + 1, int(0.30 * self.n))
                q = rng.randint(lo, hi)
                removed = self.destroy(op, q, rng)
            self.repair(removed, rng, dl)
            ls_dl = min(dl, time.time() + ls_cap)
            self.local_search(ls_dl, logger)
            self.pool_add()
            c = self.total_cost()
            if self.best_cost < pre_best - TOL:
                since = 0
            else:
                since += 1
            if c <= cur_cost + max(1e-9, 0.002 * abs(cur_cost)):
                cur_cost = c
            else:
                self.restore(save)
            if since >= 60:
                self.restore(self.best_routes)
                cur_cost = self.best_cost
                since = 0
        self.restore(self.best_routes)

    # ---------- pool ----------
    def pool_add(self):
        if len(self.pool) >= POOL_CAP:
            return
        for uid, r in self.routes.items():
            if len(self.pool) >= POOL_CAP:
                break
            t = tuple(r)
            if t not in self.pool:
                self.pool[t] = self.route_cost[uid]

    def add_singletons(self):
        for u in self.cust:
            t = (u,)
            if t not in self.pool:
                self.pool[t] = self.D[self.SRC][u] + self.D[u][self.SNK]

    # ---------- solution output / logging ----------
    def make_solution(self, cost, routes_dict):
        out = []
        oi = self.orig_id
        for uid, r in routes_dict.items():
            if not r:
                continue
            out.append([oi[self.SRC]] + [oi[u] for u in r] + [oi[self.SNK]])
        return {"objective_value": float(cost), "routes": out}

    def _log(self, logger, cost, routes):
        if logger is None:
            return
        try:
            logger.log_solution(float(cost), self.make_solution(cost, routes))
        except Exception:
            pass

    def update_best(self, logger, force=False):
        c = self.total_cost()
        if c < self.best_cost - TOL:
            self.best_cost = c
            self.best_routes = self.copy_routes()
            if logger and (force or time.time() - self.last_log >= 0.25):
                self.last_log = time.time()
                self.logged_cost = c
                self._log(logger, c, self.best_routes)
            return True
        return False

    def log_best(self, logger):
        if logger and self.best_routes is not None and self.best_cost < self.logged_cost - 1e-12:
            self.logged_cost = self.best_cost
            self.last_log = time.time()
            self._log(logger, self.best_cost, self.best_routes)

    # ---------- Gurobi models ----------
    def solve_mip(self, tlim):
        N = self.N
        D = self.D
        SRC = self.SRC
        SNK = self.SNK
        Q = self.Q
        ea = self.ea
        la = self.la
        ser = self.ser
        dem = self.dem
        arcs = []
        for i in range(N):
            if i == SNK:
                continue
            Di = D[i]
            ei = ea[i]
            si = ser[i]
            di = dem[i]
            for j in range(1, N):
                if i == j:
                    continue
                if i == SRC and j == SNK:
                    continue
                if di + dem[j] <= Q + EPS and ei + si + Di[j] <= la[j] + EPS:
                    arcs.append((i, j))
        m = gp.Model("vrptw")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, tlim)
        costs = [D[i][j] for (i, j) in arcs]
        x = m.addVars(arcs, vtype=GRB.BINARY, obj=costs, name="x")
        a = [m.addVar(lb=ea[i], ub=la[i], name="a%d" % i) for i in range(N)]
        ld = [m.addVar(lb=(dem[i] if 1 <= i <= self.n else 0.0), ub=Q, name="u%d" % i) for i in range(N)]
        a[SRC].LB = 0.0
        a[SRC].UB = 0.0
        ld[SRC].UB = 0.0
        m.ModelSense = GRB.MINIMIZE
        cust = self.cust
        m.addConstrs((x.sum('*', j) == 1 for j in cust), name="in")
        m.addConstrs((x.sum(i, '*') == 1 for i in cust), name="out")
        m.addConstr(x.sum(SRC, '*') == x.sum('*', SNK), name="veh")
        for (i, j) in arcs:
            M = la[i] + ser[i] + D[i][j] - ea[j]
            if M > 1e-9:
                m.addConstr(a[j] >= a[i] + ser[i] + D[i][j] - M * (1 - x[i, j]))
            if j != SNK:
                m.addConstr(ld[j] >= ld[i] + dem[j] - Q * (1 - x[i, j]))
        # MIP start from current routes
        arcset = set(arcs)
        a[SRC].Start = 0.0
        ld[SRC].Start = 0.0
        for uid, r in self.routes.items():
            seq = [SRC] + r + [SNK]
            t = 0.0
            load = 0.0
            for k in range(1, len(seq)):
                i = seq[k - 1]
                j = seq[k]
                if (i, j) in arcset:
                    x[i, j].Start = 1.0
                arr = t + ser[i] + D[i][j]
                st = arr if arr > ea[j] else ea[j]
                a[j].Start = st
                load += dem[j]
                ld[j].Start = min(load, Q)
                t = st
        m.optimize()
        if m.SolCount > 0:
            xsol = {}
            for (i, j) in arcs:
                if x[i, j].X > 0.5:
                    xsol[i] = j
            routes = []
            seen = set()
            for (i, j) in arcs:
                if i == SRC and x[i, j].X > 0.5:
                    rt = []
                    cur = j
                    guard = 0
                    while cur != SNK and guard <= N + 2:
                        rt.append(cur)
                        seen.add(cur)
                        cur = xsol.get(cur, SNK)
                        guard += 1
                    routes.append(rt)
            if len(seen) == self.n:
                return routes, m.ObjVal
        return None, None

    def solve_spp(self, tlim):
        keys = list(self.pool.keys())
        costs = [self.pool[k] for k in keys]
        R = len(keys)
        m = gp.Model("spp")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, tlim)
        y = m.addVars(R, vtype=GRB.BINARY, obj=costs, name="y")
        m.ModelSense = GRB.MINIMIZE
        inc = [[] for _ in range(self.n + 1)]
        for ri, rt in enumerate(keys):
            for u in rt:
                inc[u].append(ri)
        for u in self.cust:
            if inc[u]:
                m.addConstr(gp.quicksum(y[i] for i in inc[u]) == 1.0)
            else:
                m.addConstr(1.0 == 1.0)
        m.optimize()
        if m.SolCount > 0:
            sel = [list(keys[i]) for i in range(R) if y[i].X > 0.5]
            cnt = sum(len(rt) for rt in sel)
            if cnt == self.n:
                return sel, m.ObjVal
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    t0 = time.time()
    limit = float(args.time_limit)
    deadline = t0 + limit

    with open(args.instance_path) as f:
        inst = json.load(f)

    solver = Solver(inst)
    rng = random.Random(0)

    # Guaranteed-feasible fallback: one route per customer.
    for u in solver.cust:
        solver.new_route([u])
    solver.best_cost = solver.total_cost()
    solver.best_routes = solver.copy_routes()
    solver.reset_routes()

    reserve = max(1.5, 0.01 * limit)
    try:
        if solver.n > 0:
            solver.build_initial(deadline - reserve)
            solver.update_best(logger, force=True)

            spp_budget = min(max(5.0, 0.10 * limit), 120.0)
            ls_dl = min(deadline - reserve - spp_budget,
                        time.time() + max(2.0, 0.20 * limit))
            if ls_dl > time.time():
                solver.local_search(ls_dl, logger)
            solver.update_best(logger, force=True)
            solver.pool_add()

            if HAVE_GP and solver.n <= 40:
                lns_dl = min(deadline - reserve - spp_budget,
                             time.time() + max(1.0, 0.10 * limit))
                if lns_dl > time.time():
                    solver.lns(lns_dl, rng, logger, max(1.0, 0.05 * limit))
                    solver.log_best(logger)
                    solver.pool_add()
                solver.restore(solver.best_routes)
                remaining = deadline - reserve - time.time()
                mip_t = max(1.0, remaining - min(spp_budget, max(3.0, 0.15 * remaining)))
                if mip_t > 2.0:
                    try:
                        sel, obj = solver.solve_mip(mip_t)
                        if sel is not None:
                            solver.adopt_routes(sel)
                            solver.update_best(logger, force=True)
                            solver.pool_add()
                    except Exception:
                        pass
                remaining = deadline - reserve - time.time()
                if remaining > 2.0:
                    solver.add_singletons()
                    solver.pool_add()
                    try:
                        sel, obj = solver.solve_spp(remaining - 0.5)
                        if sel is not None:
                            solver.adopt_routes(sel)
                            solver.update_best(logger, force=True)
                    except Exception:
                        pass
            else:
                lns_dl = deadline - reserve - spp_budget
                if lns_dl > time.time():
                    solver.lns(lns_dl, rng, logger, max(2.0, 0.03 * limit))
                    solver.log_best(logger)
                solver.restore(solver.best_routes)
                remaining = deadline - reserve - time.time()
                if HAVE_GP and remaining > 2.0:
                    solver.add_singletons()
                    solver.pool_add()
                    try:
                        sel, obj = solver.solve_spp(remaining - 0.5)
                        if sel is not None:
                            solver.adopt_routes(sel)
                            solver.update_best(logger, force=True)
                            rem2 = deadline - reserve - time.time()
                            if rem2 > 1.0:
                                solver.local_search(min(rem2, time.time() + 5.0), logger)
                                solver.update_best(logger, force=True)
                    except Exception:
                        pass
            solver.log_best(logger)
    except Exception:
        pass

    if solver.best_routes is None:
        solver.best_routes = {}
        solver.best_cost = 0.0
    sol = solver.make_solution(solver.best_cost, solver.best_routes)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()