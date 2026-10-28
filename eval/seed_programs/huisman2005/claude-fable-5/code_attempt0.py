import argparse
import json
import sys
import time
import bisect
from collections import deque


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(5, args.time_limit)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    trips = sorted(data["trips"], key=lambda t: (t["start_time"], t["trip_id"]))
    n = len(trips)
    TT = data.get("travel_times", {})

    def tt(a, b):
        if a == b:
            return 0
        try:
            return TT[a][b]
        except Exception:
            return 0

    depots = data["depots"]
    depot_names = [d["name"] for d in depots]
    relief = set(data.get("relief_locations", []))

    st = [t["start_time"] for t in trips]
    en = [t["end_time"] for t in trips]
    sl = [t["start_location"] for t in trips]
    el = [t["end_location"] for t in trips]
    ids = [t["trip_id"] for t in trips]

    # ---------------- trivial empty case ----------------
    if n == 0:
        sol = {"objective_value": 0,
               "vehicle_schedule": {name: [] for name in depot_names},
               "crew_schedule": []}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=1)
        return

    # ================= 1. VEHICLE SCHEDULING (min path cover) =================
    # Greedy best-fit chaining (small idle preferred)
    veh = []  # (last_end_time, last_end_loc, [trip positions])
    for j in range(n):
        best = -1
        bestr = None
        for vi in range(len(veh)):
            v = veh[vi]
            r = v[0] + tt(v[1], sl[j])
            if r <= st[j] and (bestr is None or r > bestr):
                bestr = r
                best = vi
        if best >= 0:
            v = veh[best]
            v[2].append(j)
            veh[best] = (en[j], el[j], v[2])
        else:
            veh.append((en[j], el[j], [j]))
    greedy_chains = [v[2] for v in veh]

    # Hopcroft-Karp maximum matching -> minimum number of vehicles
    hk_chains = None
    if time.time() < deadline - 5:
        try:
            adj = [[] for _ in range(n)]
            cap = 10 ** 9 if n <= 1500 else 80
            for i in range(n):
                lo = bisect.bisect_left(st, en[i])
                cnt = 0
                ei, eli = en[i], el[i]
                for j in range(lo, n):
                    if ei + tt(eli, sl[j]) <= st[j]:
                        adj[i].append(j)
                        cnt += 1
                        if cnt >= cap:
                            break
            sys.setrecursionlimit(400000)
            INF = float("inf")
            matchL = [-1] * n
            matchR = [-1] * n
            while True:
                if time.time() > deadline - 3:
                    break
                dist = [INF] * n
                q = deque()
                for i in range(n):
                    if matchL[i] == -1:
                        dist[i] = 0
                        q.append(i)
                found = False
                while q:
                    i = q.popleft()
                    di = dist[i]
                    for j in adj[i]:
                        k = matchR[j]
                        if k == -1:
                            found = True
                        elif dist[k] == INF:
                            dist[k] = di + 1
                            q.append(k)
                if not found:
                    break

                def dfs(i):
                    for j in adj[i]:
                        k = matchR[j]
                        if k == -1 or (dist[k] == dist[i] + 1 and dfs(k)):
                            matchL[i] = j
                            matchR[j] = i
                            return True
                    dist[i] = INF
                    return False

                for i in range(n):
                    if matchL[i] == -1 and dist[i] == 0:
                        dfs(i)
            # build chains
            chains = []
            for j in range(n):
                if matchR[j] == -1:  # j is a chain start
                    ch = [j]
                    cur = j
                    while matchL[cur] != -1:
                        cur = matchL[cur]
                        ch.append(cur)
                    chains.append(ch)
            covered = sum(len(c) for c in chains)
            if covered == n:
                hk_chains = chains
        except Exception:
            hk_chains = None

    if hk_chains is not None and len(hk_chains) < len(greedy_chains):
        blocks = hk_chains
    else:
        blocks = greedy_chains

    # depot assignment per block: nearest for pull-out + pull-in
    block_depot = []
    for ch in blocks:
        d = min(depot_names, key=lambda dn: tt(dn, sl[ch[0]]) + tt(el[ch[-1]], dn))
        block_depot.append(d)

    # ================= 2. CREW: cut blocks into pieces =================
    dts = data["duty_types"]
    pl_min_all = min(t["piece_length_min"] for t in dts.values())
    pl_max_all = min(t["piece_length_max"] for t in dts.values())

    cp = data["crew_parameters"]
    sign_on = cp["sign_on_time_depot_minutes"]
    sign_off = cp["sign_off_time_depot_minutes"]
    extra = cp["extra_time_non_depot_relief_minutes"]
    inc_dh = cp["extra_time_includes_deadhead_to_depot"]

    def start_oh(loc, depot):
        if loc == depot:
            return sign_on
        return extra if inc_dh else extra + tt(loc, depot)

    def end_oh(loc, depot):
        if loc == depot:
            return sign_off
        return extra if inc_dh else extra + tt(loc, depot)

    def decompose(block, depot):
        nb = len(block)
        bopts = []
        f0 = block[0]
        bopts.append([(None, None, st[f0] - tt(depot, sl[f0]), depot)])
        for k in range(1, nb):
            a = block[k - 1]
            b = block[k]
            opts = []
            if en[a] + tt(el[a], depot) + tt(depot, sl[b]) <= st[b]:
                # long arc: vehicle returns to depot
                opts.append((en[a] + tt(el[a], depot), depot,
                             st[b] - tt(depot, sl[b]), depot))
            else:
                if el[a] in relief:
                    opts.append((en[a], el[a], en[a], el[a]))
                if sl[b] in relief:
                    opts.append((st[b], sl[b], st[b], sl[b]))
            bopts.append(opts)
        lastt = block[-1]
        bopts.append([(en[lastt] + tt(el[lastt], depot), depot, None, None)])

        INF = float("inf")
        dp = [[INF] * len(o) for o in bopts]
        par = [[None] * len(o) for o in bopts]
        dp[0][0] = 0.0
        for b in range(1, nb + 1):
            for ob, optb in enumerate(bopts[b]):
                tpe = optb[0]
                best = INF
                bestpar = None
                for a in range(b):
                    for oa, opta in enumerate(bopts[a]):
                        base = dp[a][oa]
                        if base >= INF:
                            continue
                        tns = opta[2]
                        dur = tpe - tns
                        if dur <= 0:
                            pen = 1e7
                        elif dur > pl_max_all:
                            pen = 5000.0 + (dur - pl_max_all)
                        elif dur < pl_min_all:
                            pen = 300.0
                        else:
                            pen = 0.0
                        c = base + 1.0 + pen + dur * 1e-5
                        if c < best:
                            best = c
                            bestpar = (a, oa)
                dp[b][ob] = best
                par[b][ob] = bestpar
        cuts = []
        b, ob = nb, 0
        while b > 0:
            a, oa = par[b][ob]
            cuts.append((a, oa, b, ob))
            b, ob = a, oa
        cuts.reverse()
        pieces = []
        for (a, oa, b, ob) in cuts:
            sa = bopts[a][oa]
            eb = bopts[b][ob]
            pieces.append({"trips": block[a:b],
                           "start": sa[2], "sloc": sa[3],
                           "end": eb[0], "eloc": eb[1],
                           "depot": depot})
        return pieces

    pieces = []
    for bi, ch in enumerate(blocks):
        ps = decompose(ch, block_depot[bi])
        for p in ps:
            p["dur"] = p["end"] - p["start"]
            pieces.append(p)
    P = len(pieces)

    singles = [(name, t) for name, t in dts.items() if t.get("num_pieces", 2) == 1]
    doubles = [(name, t) for name, t in dts.items() if t.get("num_pieces", 2) == 2]
    fallback_single = singles[0][0] if singles else list(dts.keys())[0]

    def check_single(p):
        d = p["dur"]
        so = start_oh(p["sloc"], p["depot"])
        se = end_oh(p["eloc"], p["depot"])
        ds = p["start"] - so
        de = p["end"] + se
        for name, t in singles:
            if d < t["piece_length_min"] or d > t["piece_length_max"]:
                continue
            if t["duty_length_max"] is not None and de - ds > t["duty_length_max"]:
                continue
            if t["work_time_max"] is not None and d + so + se > t["work_time_max"]:
                continue
            if t["start_time_min"] is not None and ds < t["start_time_min"]:
                continue
            if t["start_time_max"] is not None and ds > t["start_time_max"]:
                continue
            if t["end_time_max"] is not None and de > t["end_time_max"]:
                continue
            return name
        return None

    def check_pair(p, q):
        gap = q["start"] - p["end"]
        if gap < 0:
            return None
        if gap < tt(p["eloc"], q["sloc"]):
            return None
        so = start_oh(p["sloc"], p["depot"])
        se = end_oh(q["eloc"], q["depot"])
        ds = p["start"] - so
        de = q["end"] + se
        dp_, dq_ = p["dur"], q["dur"]
        for name, t in doubles:
            if dp_ < t["piece_length_min"] or dp_ > t["piece_length_max"]:
                continue
            if dq_ < t["piece_length_min"] or dq_ > t["piece_length_max"]:
                continue
            if t["break_length_min"] is not None and gap < t["break_length_min"]:
                continue
            if t["duty_length_max"] is not None and de - ds > t["duty_length_max"]:
                continue
            if t["work_time_max"] is not None and dp_ + dq_ + so + se > t["work_time_max"]:
                continue
            if t["start_time_min"] is not None and ds < t["start_time_min"]:
                continue
            if t["start_time_max"] is not None and ds > t["start_time_max"]:
                continue
            if t["end_time_max"] is not None and de > t["end_time_max"]:
                continue
            return name
        return None

    single_type = [check_single(p) for p in pieces]
    single_ok = [s is not None for s in single_type]

    def build_solution(pairs):
        used = [False] * P
        crew = []
        for (pi, qi, tpn) in pairs:
            used[pi] = used[qi] = True
            p, q = pieces[pi], pieces[qi]
            crew.append({"depot": p["depot"], "type": tpn,
                         "trips": [ids[i] for i in p["trips"]] +
                                  [ids[i] for i in q["trips"]]})
        for pi in range(P):
            if not used[pi]:
                p = pieces[pi]
                tpn = single_type[pi] if single_type[pi] else fallback_single
                crew.append({"depot": p["depot"], "type": tpn,
                             "trips": [ids[i] for i in p["trips"]]})
        vs = {name: [] for name in depot_names}
        for bi, ch in enumerate(blocks):
            vs[block_depot[bi]].append([ids[i] for i in ch])
        obj = len(blocks) + len(crew)
        return obj, {"objective_value": obj,
                     "vehicle_schedule": vs,
                     "crew_schedule": crew}

    # initial incumbent: every piece is its own duty
    best_obj, best_sol = build_solution([])
    if logger:
        logger.log_solution(best_obj, best_sol)

    # ================= 3. pair pieces into two-piece duties =================
    edges = []
    if doubles and time.time() < deadline - 2:
        maxspan = max((t["duty_length_max"] if t["duty_length_max"] is not None
                       else 10 ** 6) for _, t in doubles)
        for pi in range(P):
            p = pieces[pi]
            for qi in range(P):
                if pi == qi:
                    continue
                q = pieces[qi]
                if q["start"] < p["end"]:
                    continue
                if q["depot"] != p["depot"]:
                    continue
                if q["end"] - p["start"] > maxspan + 200:
                    continue
                tpn = check_pair(p, q)
                if tpn:
                    edges.append((pi, qi, tpn))
            if time.time() > deadline - 2:
                break

    pairs = []
    if edges:
        solved = False
        try:
            import gurobipy as gp
            from gurobipy import GRB
            m = gp.Model("pairing")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time() - 3)
            E = len(edges)
            xs = m.addVars(E, vtype=GRB.BINARY)
            w = []
            for (pi, qi, tpn) in edges:
                ww = 1.0
                if not single_ok[pi]:
                    ww += 100.0
                if not single_ok[qi]:
                    ww += 100.0
                w.append(ww)
            m.setObjective(gp.quicksum(w[e] * xs[e] for e in range(E)), GRB.MAXIMIZE)
            inc = [[] for _ in range(P)]
            for e, (pi, qi, tpn) in enumerate(edges):
                inc[pi].append(e)
                inc[qi].append(e)
            for pi in range(P):
                if inc[pi]:
                    m.addConstr(gp.quicksum(xs[e] for e in inc[pi]) <= 1)
            m.optimize()
            if m.SolCount > 0:
                pairs = [edges[e] for e in range(E) if xs[e].X > 0.5]
                solved = True
        except Exception:
            solved = False
        if not solved:
            # greedy fallback matching
            in_edges = [[] for _ in range(P)]
            for (pi, qi, tpn) in edges:
                in_edges[qi].append((pi, tpn))
            matched = [False] * P
            pairs = []
            for qi in sorted(range(P), key=lambda i: pieces[i]["start"]):
                if matched[qi]:
                    continue
                bestp = None
                bestt = None
                for (pi, tpn) in in_edges[qi]:
                    if matched[pi] or pi == qi:
                        continue
                    if bestp is None or pieces[pi]["end"] > pieces[bestp]["end"]:
                        bestp = pi
                        bestt = tpn
                if bestp is not None:
                    matched[bestp] = matched[qi] = True
                    pairs.append((bestp, qi, bestt))

    obj, sol = build_solution(pairs)
    if obj < best_obj:
        best_obj, best_sol = obj, sol
        if logger:
            logger.log_solution(best_obj, best_sol)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=1)


if __name__ == "__main__":
    main()