import argparse
import json
import math
import time
import sys

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

EPS = 1e-6

START_TIME = 0.0
TIME_LIMIT = 0

DEPOT_ID = 0
N_ORIG = 0

CUSTOMERS = {}
ALL_CUSTOMER_IDS = set()
UNASSIGNED = set()

DIST_MATRIX = []
TIME_MATRIX = []
COORDS = {}

PARKING_IDS = []
PARK_INDEX = {}
TRANS_IDS = []

CLASSES = {}
ALL_CLASS_IDS = []

TRUCK_REMAIN = {}
TRAILER_REMAIN = {}
CLASS_REMAIN = {}

ROUTES = []

H_START = 0.0
H_END = 1e9
MAX_DURATION = 1e9

DECOUPLE_TIME = 0.0
COUPLING_TIME = 0.0
TRANSFER_TIME = 0.0

DECOUPLE_COST = 0.0
COUPLING_COST = 0.0
TRANSFER_COST = 0.0

HAS_CUSTOM_TW = False

LOGGER = None
BEST_OBJ = float("inf")
BEST_SOLUTION = None


def time_left():
    if TIME_LIMIT is None:
        return 1e9
    return TIME_LIMIT - (time.time() - START_TIME)


def parse_time_window(obj, h_start, h_end):
    tw = None
    if isinstance(obj, dict):
        if "tw" in obj:
            tw = obj["tw"]
        elif "time_window" in obj:
            tw = obj["time_window"]
        elif "earliest" in obj and "latest" in obj:
            tw = [obj["earliest"], obj["latest"]]
    if tw is None:
        return [float(h_start), float(h_end)]
    try:
        return [float(tw[0]), float(tw[1])]
    except Exception:
        return [float(h_start), float(h_end)]


def dist(i, j):
    if i is None or j is None:
        return 0.0
    try:
        i = int(i)
        j = int(j)
    except Exception:
        return 0.0
    try:
        return float(DIST_MATRIX[i][j])
    except Exception:
        ci = COORDS.get(i, (0.0, 0.0))
        cj = COORDS.get(j, (0.0, 0.0))
        return math.hypot(ci[0] - cj[0], ci[1] - cj[1])


def ttime(i, j):
    if i is None or j is None:
        return 0.0
    try:
        i = int(i)
        j = int(j)
    except Exception:
        return 0.0
    try:
        return float(TIME_MATRIX[i][j])
    except Exception:
        return dist(i, j)


def greedy_order(custs, start, end, seed_mode="far"):
    custs = list(custs)
    if len(custs) <= 1:
        return custs[:]

    rem = set(custs)
    if seed_mode == "near":
        seed = min(rem, key=lambda c: ttime(start, c))
    elif seed_mode == "early":
        seed = min(rem, key=lambda c: CUSTOMERS[c]["tw"][1])
    else:
        seed = max(rem, key=lambda c: ttime(start, c))

    seq = [seed]
    rem.remove(seed)

    while rem:
        best = None
        for c in rem:
            for pos in range(len(seq) + 1):
                prev = start if pos == 0 else seq[pos - 1]
                nxt = end if pos == len(seq) else seq[pos]
                inc = ttime(prev, c) + ttime(c, nxt) - ttime(prev, nxt)
                if best is None or inc < best[0] - EPS:
                    best = (inc, c, pos)
        if best is None:
            break
        _, c, pos = best
        seq.insert(pos, c)
        rem.remove(c)

    return seq


def generate_sequences(custs, start, end, fast=True):
    custs = list(set(custs))
    if not custs:
        return [[]]
    if len(custs) == 1:
        return [custs]

    seqs = []
    modes = ["far"] if fast else ["far", "near", "early"]
    for m in modes:
        s = greedy_order(custs, start, end, m)
        if s not in seqs:
            seqs.append(s)
    return seqs


def evaluate_route(class_id, mode, park=None, main_seq=None, prefix=None, truckseg=None, suffix=None, transfer=False):
    vc = CLASSES.get(class_id)
    if vc is None:
        return None

    stops = []
    stops.append({
        "type": "depot",
        "loc": DEPOT_ID,
        "cust": None,
        "arc_attached": None,
        "duration": 0.0
    })

    def add_customer(c, attached_service, arc_attached):
        cu = CUSTOMERS[c]
        stops.append({
            "type": "customer",
            "loc": c,
            "cust": c,
            "arc_attached": arc_attached,
            "load_attached": attached_service,
            "duration": float(cu["service_time"]),
            "tw": cu["tw"]
        })

    def add_op(op_type, loc, arc_attached):
        dur = DECOUPLE_TIME if op_type == "decouple" else COUPLING_TIME
        stops.append({
            "type": op_type,
            "loc": loc,
            "cust": None,
            "arc_attached": arc_attached,
            "duration": float(dur)
        })

    if mode in ("solo", "attached"):
        attached = (mode == "attached")
        for c in (main_seq or []):
            add_customer(c, attached, attached)
        stops.append({
            "type": "depot",
            "loc": DEPOT_ID,
            "cust": None,
            "arc_attached": attached,
            "duration": 0.0
        })
    elif mode == "combined":
        if park is None or park not in PARK_INDEX:
            return None
        for c in (prefix or []):
            add_customer(c, True, True)
        add_op("decouple", park, True)
        for c in (truckseg or []):
            add_customer(c, False, False)
        add_op("couple", park, False)
        for c in (suffix or []):
            add_customer(c, True, True)
        stops.append({
            "type": "depot",
            "loc": DEPOT_ID,
            "cust": None,
            "arc_attached": True,
            "duration": 0.0
        })
    else:
        return None

    # Capacity simulation.
    truck_load = 0.0
    trailer_load = 0.0
    transfer_used = False

    for st in stops:
        if st["type"] == "customer":
            q = float(CUSTOMERS[st["cust"]]["supply"])
            if st.get("load_attached", False) and vc["has_trailer"]:
                to_trailer = min(q, vc["trailer_capacity"] - trailer_load)
                to_truck = q - to_trailer
                if truck_load + to_truck > vc["truck_capacity"] + EPS:
                    return None
                truck_load += to_truck
                trailer_load += to_trailer
            else:
                if truck_load + q > vc["truck_capacity"] + EPS:
                    return None
                truck_load += q
        elif st["type"] == "couple" and transfer:
            move = min(truck_load, vc["trailer_capacity"] - trailer_load)
            if move > EPS:
                truck_load -= move
                trailer_load += move
                transfer_used = True

    if transfer_used:
        for st in stops:
            if st["type"] == "couple":
                st["duration"] += TRANSFER_TIME
                break

    # Travel cost.
    travel_cost = 0.0
    prev = stops[0]["loc"]
    for st in stops[1:]:
        dv = dist(prev, st["loc"])
        travel_cost += vc["truck_cost_factor"] * dv
        if vc["has_trailer"] and st.get("arc_attached", False):
            travel_cost += vc["trailer_towing_cost_multiplier"] * dv
        prev = st["loc"]

    handling_cost = 0.0
    if mode == "combined":
        handling_cost += DECOUPLE_COST + COUPLING_COST
        if transfer_used:
            handling_cost += TRANSFER_COST

    cost = vc["truck_fixed_cost"] + travel_cost + handling_cost

    # Scheduling.
    def stop_id(st):
        if st["type"] == "depot":
            return DEPOT_ID
        if st["type"] == "customer":
            return st["loc"]
        if st["type"] == "decouple":
            return N_ORIG + PARK_INDEX[st["loc"]]
        if st["type"] == "couple":
            return N_ORIG + len(PARKING_IDS) + PARK_INDEX[st["loc"]]
        return st["loc"]

    if not HAS_CUSTOM_TW:
        cum = 0.0
        out_rel = [0.0]
        prev = stops[0]["loc"]
        for st in stops[1:]:
            cum += ttime(prev, st["loc"])
            out_rel.append(cum)
            cum += st["duration"]
            prev = st["loc"]

        total = cum
        if total > MAX_DURATION + EPS:
            return None

        if H_END < 1e8:
            latest = H_END - total
            if latest < H_START - EPS:
                return None
            start = latest
        else:
            start = H_START

        out_abs = [start + x for x in out_rel]
        duration = total
    else:
        def simulate(start):
            t = float(start)
            prev_loc = stops[0]["loc"]
            out = [t]

            for idx in range(1, len(stops)):
                st = stops[idx]
                t += ttime(prev_loc, st["loc"])
                arr = t

                if st["type"] == "customer":
                    e, l = st["tw"]
                    ss = arr if arr >= e else e
                    if ss > l + EPS:
                        return None
                    t = ss + st["duration"]
                    out.append(ss)
                else:
                    t += st["duration"]
                    out.append(arr)

                prev_loc = st["loc"]

            if t > H_END + EPS:
                return None
            return (t, t - start, out)

        if simulate(H_START) is None:
            return None

        lo = H_START
        hi = H_END
        if hi > 1e8:
            max_l = max([H_START] + [cu["tw"][1] for cu in CUSTOMERS.values()])
            hi = max_l + 1.0

        for _ in range(25):
            mid = (lo + hi) / 2.0
            if simulate(mid) is not None:
                lo = mid
            else:
                hi = mid

        sim = simulate(lo)
        if sim is None:
            return None

        finish, duration, out_abs = sim
        if duration > MAX_DURATION + EPS:
            return None
        start = lo

    route_sequence = [stop_id(st) for st in stops]
    route_sequence_physical = [st["loc"] for st in stops]

    arrival_times = {}
    for idx, st in enumerate(stops):
        sid = stop_id(st)
        key = str(sid)
        if st["type"] == "depot" and idx > 0 and key in arrival_times:
            continue
        arrival_times[key] = float(out_abs[idx])

    if mode in ("solo", "attached"):
        cust_list = list(main_seq or [])
    else:
        cust_list = list(prefix or []) + list(truckseg or []) + list(suffix or [])

    return {
        "class_id": class_id,
        "mode": mode,
        "park": park if mode == "combined" else None,
        "customers": cust_list,
        "cost": float(cost),
        "route_sequence": route_sequence,
        "route_sequence_physical": route_sequence_physical,
        "arrival_times": arrival_times,
        "decouple_locations": [park] if mode == "combined" else [],
        "couple_locations": [park] if mode == "combined" else [],
        "transfer_used": transfer_used
    }


def valid_parking_for_set(p, trailer_set):
    if p not in CUSTOMERS:
        return True
    if CUSTOMERS[p]["kind"] != "trailer":
        return False
    return p in trailer_set


def parking_candidates(truck_list, trailer_set, hint=None, limit=3):
    cands = []
    seen = set()

    def add(p):
        if p in seen:
            return
        if not valid_parking_for_set(p, trailer_set):
            return
        score = dist(DEPOT_ID, p) + sum(dist(p, c) for c in truck_list)
        cands.append((score, p))
        seen.add(p)

    if hint is not None:
        add(hint)

    for p in TRANS_IDS:
        add(p)

    for p in trailer_set:
        add(p)

    if hint is not None and hint in seen:
        cands.sort(key=lambda x: (0 if x[1] == hint else 1, x[0]))
    else:
        cands.sort(key=lambda x: x[0])

    return [p for _, p in cands[:limit]]


def build_combined(cust_set, class_id, truck_list, trailer_list, hint, fast):
    vc = CLASSES[class_id]
    trailer_set = set(trailer_list)
    limit = 3 if fast else 6
    cands = parking_candidates(truck_list, trailer_set, hint, limit)

    best = None

    for p in cands:
        variants = []
        if p in trailer_set:
            variants.append(({p}, set(truck_list), trailer_set - {p}))
            variants.append((set(), set(truck_list) | {p}, trailer_set - {p}))
        else:
            variants.append((set(), set(truck_list), set(trailer_set)))

        for prefix_set, truck_set, suffix_set in variants:
            if sum(CUSTOMERS[c]["supply"] for c in truck_set) > vc["truck_capacity"] + EPS:
                continue

            if prefix_set:
                others = [c for c in prefix_set if c != p]
                pref = []
                if others:
                    pref = generate_sequences(others, DEPOT_ID, p, fast)[0]
                if p in prefix_set:
                    pref.append(p)
            else:
                pref = []

            if p in truck_set:
                others = [c for c in truck_set if c != p]
                if others:
                    base = generate_sequences(others, p, p, fast)
                    truck_seqs = [[p] + s for s in base[:1 if fast else 2]]
                else:
                    truck_seqs = [[p]]
            else:
                truck_seqs = generate_sequences(list(truck_set), p, p, fast)[:1 if fast else 2]

            if not truck_seqs:
                truck_seqs = [[]]

            suffix_custs = [c for c in suffix_set if c != p]
            suffix_seqs = generate_sequences(suffix_custs, p, DEPOT_ID, fast)[:1 if fast else 2]
            if not suffix_seqs:
                suffix_seqs = [[]]

            for tseq in truck_seqs:
                for sseq in suffix_seqs:
                    det = evaluate_route(
                        class_id,
                        "combined",
                        park=p,
                        prefix=pref,
                        truckseg=tseq,
                        suffix=sseq,
                        transfer=False
                    )
                    if det is not None and (best is None or det["cost"] < best["cost"] - EPS):
                        best = det

                    if det is None:
                        det = evaluate_route(
                            class_id,
                            "combined",
                            park=p,
                            prefix=pref,
                            truckseg=tseq,
                            suffix=sseq,
                            transfer=True
                        )
                        if det is not None and (best is None or det["cost"] < best["cost"] - EPS):
                            best = det

    return best


def build_route(cust_set, class_id, park_hint=None, fast=True):
    cust_set = set(cust_set)
    if not cust_set:
        return None

    vc = CLASSES.get(class_id)
    if vc is None:
        return None

    truck_list = [c for c in cust_set if CUSTOMERS[c]["kind"] == "truck"]
    trailer_list = [c for c in cust_set if CUSTOMERS[c]["kind"] == "trailer"]

    total_demand = sum(CUSTOMERS[c]["supply"] for c in cust_set)
    truck_demand = sum(CUSTOMERS[c]["supply"] for c in truck_list)

    best = None

    if not vc["has_trailer"]:
        if total_demand > vc["truck_capacity"] + EPS:
            return None
        for seq in generate_sequences(cust_set, DEPOT_ID, DEPOT_ID, fast):
            det = evaluate_route(class_id, "solo", main_seq=seq)
            if det is not None and (best is None or det["cost"] < best["cost"] - EPS):
                best = det
        return best

    if not truck_list:
        if total_demand > vc["total_capacity"] + EPS:
            return None
        for seq in generate_sequences(cust_set, DEPOT_ID, DEPOT_ID, fast):
            det = evaluate_route(class_id, "attached", main_seq=seq)
            if det is not None and (best is None or det["cost"] < best["cost"] - EPS):
                best = det
        return best

    if truck_demand > vc["truck_capacity"] + EPS:
        return None
    if total_demand > vc["total_capacity"] + EPS:
        return None

    return build_combined(cust_set, class_id, truck_list, trailer_list, park_hint, fast)


def release(cid):
    vc = CLASSES[cid]
    t = vc["truck_type_id"]
    TRUCK_REMAIN[t] = TRUCK_REMAIN.get(t, 0) + 1
    if vc["trailer_type_id"]:
        r = vc["trailer_type_id"]
        TRAILER_REMAIN[r] = TRAILER_REMAIN.get(r, 0) + 1
    CLASS_REMAIN[cid] = CLASS_REMAIN.get(cid, 0) + 1


def reserve(cid):
    vc = CLASSES[cid]
    t = vc["truck_type_id"]
    if TRUCK_REMAIN.get(t, 0) <= 0:
        return False
    if vc["trailer_type_id"]:
        r = vc["trailer_type_id"]
        if TRAILER_REMAIN.get(r, 0) <= 0:
            return False
    if CLASS_REMAIN.get(cid, 0) <= 0:
        return False

    TRUCK_REMAIN[t] -= 1
    if vc["trailer_type_id"]:
        TRAILER_REMAIN[vc["trailer_type_id"]] -= 1
    CLASS_REMAIN[cid] -= 1
    return True


def can_reserve_with_releases(release_cids, reserve_cids):
    t = dict(TRUCK_REMAIN)
    tr = dict(TRAILER_REMAIN)
    cl = dict(CLASS_REMAIN)

    for cid in release_cids:
        vc = CLASSES[cid]
        tt = vc["truck_type_id"]
        t[tt] = t.get(tt, 0) + 1
        if vc["trailer_type_id"]:
            rr = vc["trailer_type_id"]
            tr[rr] = tr.get(rr, 0) + 1
        cl[cid] = cl.get(cid, 0) + 1

    for cid in reserve_cids:
        vc = CLASSES[cid]
        tt = vc["truck_type_id"]
        if t.get(tt, 0) <= 0:
            return False
        if vc["trailer_type_id"]:
            rr = vc["trailer_type_id"]
            if tr.get(rr, 0) <= 0:
                return False
        if cl.get(cid, 0) <= 0:
            return False

        t[tt] -= 1
        if vc["trailer_type_id"]:
            tr[vc["trailer_type_id"]] -= 1
        cl[cid] -= 1

    return True


def rough_class_can_serve_set(vc, cust_set):
    total = sum(CUSTOMERS[c]["supply"] for c in cust_set)
    truck_dem = sum(CUSTOMERS[c]["supply"] for c in cust_set if CUSTOMERS[c]["kind"] == "truck")

    if not vc["has_trailer"]:
        return total <= vc["truck_capacity"] + EPS

    if truck_dem > vc["truck_capacity"] + EPS:
        return False

    return total <= vc["total_capacity"] + EPS


def alternative_classes_for_set(cust_set, current_cid=None, limit=6):
    res = []
    for cid in ALL_CLASS_IDS:
        if cid == current_cid:
            continue
        vc = CLASSES[cid]
        if rough_class_can_serve_set(vc, cust_set):
            res.append(cid)
        if len(res) >= limit:
            break
    return res


def candidate_classes_for_customer(c):
    res = []
    for cid in ALL_CLASS_IDS:
        vc = CLASSES[cid]
        if rough_class_can_serve_set(vc, {c}):
            res.append(cid)
        if len(res) >= 8:
            break
    return res


def make_route_dict(cid, cust_set, detail):
    return {
        "class_id": cid,
        "customers": set(cust_set),
        "park": detail.get("park"),
        "detail": detail
    }


def apply_update_route(route, new_cid, new_det):
    old_cid = route["class_id"]
    if new_cid != old_cid:
        release(old_cid)
        reserve(new_cid)

    route["class_id"] = new_cid
    route["customers"] = set(new_det["customers"])
    route["park"] = new_det.get("park")
    route["detail"] = new_det


def construct_initial():
    order = sorted(
        ALL_CUSTOMER_IDS,
        key=lambda c: (CUSTOMERS[c]["kind"] != "truck", -CUSTOMERS[c]["supply"], -dist(DEPOT_ID, c))
    )

    for c in order:
        if time_left() < 1.0:
            break
        if c not in UNASSIGNED:
            continue

        # Insert into existing routes with same class.
        best = None
        for idx, r in enumerate(ROUTES):
            new_set = r["customers"] | {c}
            det = build_route(new_set, r["class_id"], r.get("park"), fast=True)
            if det is not None:
                delta = det["cost"] - r["detail"]["cost"]
                if best is None or delta < best[0] - EPS:
                    best = (delta, idx, r["class_id"], det)

        if best is not None:
            apply_update_route(ROUTES[best[1]], best[2], best[3])
            UNASSIGNED.remove(c)
            continue

        # Create a new route.
        new_route = None
        for cid in candidate_classes_for_customer(c):
            if can_reserve_with_releases([], [cid]):
                det = build_route({c}, cid, None, fast=True)
                if det is not None:
                    if new_route is None or det["cost"] < new_route[1]["cost"] - EPS:
                        new_route = (cid, det)

        if new_route is not None:
            cid, det = new_route
            if reserve(cid):
                ROUTES.append(make_route_dict(cid, {c}, det))
                UNASSIGNED.remove(c)
                continue

        # Insert with class change.
        best = None
        for idx, r in enumerate(ROUTES):
            old_cid = r["class_id"]
            new_set = r["customers"] | {c}
            for alt in alternative_classes_for_set(new_set, old_cid, limit=4):
                if can_reserve_with_releases([old_cid], [alt]):
                    det = build_route(new_set, alt, r.get("park"), fast=True)
                    if det is not None:
                        delta = det["cost"] - r["detail"]["cost"]
                        if best is None or delta < best[0] - EPS:
                            best = (delta, idx, alt, det)

        if best is not None:
            apply_update_route(ROUTES[best[1]], best[2], best[3])
            UNASSIGNED.remove(c)
            continue


def emergency_merge():
    if len(ROUTES) < 2:
        return False

    for i in range(len(ROUTES)):
        for j in range(i + 1, len(ROUTES)):
            if i >= len(ROUTES) or j >= len(ROUTES):
                continue

            A = ROUTES[i]
            B = ROUTES[j]
            union = A["customers"] | B["customers"]
            release_cids = [A["class_id"], B["class_id"]]

            cands = [A["class_id"], B["class_id"]] + alternative_classes_for_set(union, None, limit=4)
            seen = set()
            for alt in cands:
                if alt in seen:
                    continue
                seen.add(alt)

                if not can_reserve_with_releases(release_cids, [alt]):
                    continue

                det = build_route(union, alt, None, fast=True)
                if det is None:
                    continue

                release(A["class_id"])
                release(B["class_id"])
                reserve(alt)

                ROUTES[i] = make_route_dict(alt, union, det)
                ROUTES.remove(B)
                return True

    return False


def repair_unassigned(max_merges=5):
    merges = 0

    while UNASSIGNED and time_left() > 2.0:
        progress = False

        for c in list(UNASSIGNED):
            # Insert into existing routes, allowing class changes.
            best = None
            for idx, r in enumerate(ROUTES):
                new_set = r["customers"] | {c}
                old_cid = r["class_id"]

                det = build_route(new_set, old_cid, r.get("park"), fast=True)
                if det is not None:
                    delta = det["cost"] - r["detail"]["cost"]
                    best = (delta, idx, old_cid, det)

                for alt in alternative_classes_for_set(new_set, old_cid, limit=3):
                    if can_reserve_with_releases([old_cid], [alt]):
                        det = build_route(new_set, alt, r.get("park"), fast=True)
                        if det is not None:
                            delta = det["cost"] - r["detail"]["cost"]
                            if best is None or delta < best[0] - EPS:
                                best = (delta, idx, alt, det)

            if best is not None:
                apply_update_route(ROUTES[best[1]], best[2], best[3])
                UNASSIGNED.remove(c)
                progress = True
                break

            # Create a new route.
            new_route = None
            for cid in candidate_classes_for_customer(c):
                if can_reserve_with_releases([], [cid]):
                    det = build_route({c}, cid, None, fast=True)
                    if det is not None:
                        if new_route is None or det["cost"] < new_route[1]["cost"] - EPS:
                            new_route = (cid, det)

            if new_route is not None:
                cid, det = new_route
                if reserve(cid):
                    ROUTES.append(make_route_dict(cid, {c}, det))
                    UNASSIGNED.remove(c)
                    progress = True
                    break

        if progress:
            continue

        if merges < max_merges and emergency_merge():
            merges += 1
            continue

        break


def find_class_change_move():
    best = None

    for i, r in enumerate(ROUTES):
        old_cid = r["class_id"]
        cust = r["customers"]

        for alt in alternative_classes_for_set(cust, old_cid, limit=4):
            if not can_reserve_with_releases([old_cid], [alt]):
                continue

            det = build_route(cust, alt, r.get("park"), fast=True)
            if det is not None and det["cost"] < r["detail"]["cost"] - EPS:
                delta = det["cost"] - r["detail"]["cost"]
                if best is None or delta < best[0] - EPS:
                    best = (delta, "class", i, alt, det)

    return best


def find_merge_move(allow_worse=False):
    if len(ROUTES) < 2:
        return None

    pairs = []
    for i in range(len(ROUTES)):
        for j in range(i + 1, len(ROUTES)):
            pairs.append((i, j))

    if len(pairs) > 60:
        pairs = pairs[:60]

    best = None

    for i, j in pairs:
        if i >= len(ROUTES) or j >= len(ROUTES):
            continue

        A = ROUTES[i]
        B = ROUTES[j]
        union = A["customers"] | B["customers"]
        old_sum = A["detail"]["cost"] + B["detail"]["cost"]
        release_cids = [A["class_id"], B["class_id"]]

        cands = [A["class_id"], B["class_id"]] + alternative_classes_for_set(union, None, limit=4)
        seen = set()

        for alt in cands:
            if alt in seen:
                continue
            seen.add(alt)

            if not can_reserve_with_releases(release_cids, [alt]):
                continue

            det = build_route(union, alt, None, fast=True)
            if det is None:
                continue

            delta = det["cost"] - old_sum
            if allow_worse or delta < -EPS:
                if best is None or delta < best[0] - EPS:
                    best = (delta, "merge", A, B, alt, det)

    return best


def find_relocate_move():
    best = None

    pairs = []
    for i, r in enumerate(ROUTES):
        for c in list(r["customers"]):
            pairs.append((i, c))

    if len(pairs) > 100:
        step = max(1, len(pairs) // 100)
        pairs = pairs[::step][:100]

    for i, c in pairs:
        if i >= len(ROUTES):
            continue

        A = ROUTES[i]
        if c not in A["customers"]:
            continue

        setA = A["customers"] - {c}
        if setA:
            detA = build_route(setA, A["class_id"], A.get("park"), fast=True)
            if detA is None:
                continue
            costA = detA["cost"]
        else:
            detA = None
            costA = 0.0

        for j, B in enumerate(ROUTES):
            if B is A:
                continue
            if c in B["customers"]:
                continue

            setB = B["customers"] | {c}
            detB = build_route(setB, B["class_id"], B.get("park"), fast=True)
            if detB is None:
                continue

            delta = costA + detB["cost"] - A["detail"]["cost"] - B["detail"]["cost"]
            if delta < -EPS:
                if best is None or delta < best[0] - EPS:
                    best = (delta, "relocate", A, B, c, detA, detB)

    return best


def apply_move(move):
    if move[1] == "class":
        _, _, idx, alt, det = move
        apply_update_route(ROUTES[idx], alt, det)

    elif move[1] == "merge":
        _, _, A, B, alt, det = move
        union = A["customers"] | B["customers"]

        release(A["class_id"])
        release(B["class_id"])
        reserve(alt)

        ROUTES.append(make_route_dict(alt, union, det))
        ROUTES.remove(A)
        ROUTES.remove(B)

    elif move[1] == "relocate":
        _, _, A, B, c, detA, detB = move

        B["customers"] = set(detB["customers"])
        B["park"] = detB.get("park")
        B["detail"] = detB

        if detA is None:
            release(A["class_id"])
            ROUTES.remove(A)
        else:
            A["customers"] = set(detA["customers"])
            A["park"] = detA.get("park")
            A["detail"] = detA


def get_total_cost():
    return sum(r["detail"]["cost"] for r in ROUTES)


def build_solution(routes):
    out_routes = []
    counters = {}
    total = 0.0

    for r in routes:
        det = r["detail"]
        cid = r["class_id"]

        counters[cid] = counters.get(cid, 0) + 1

        out_routes.append({
            "vehicle_class": cid,
            "vehicle_index": counters[cid],
            "customers_served": list(det["customers"]),
            "route_sequence": list(det["route_sequence"]),
            "route_sequence_physical": list(det["route_sequence_physical"]),
            "decouple_locations": list(det["decouple_locations"]),
            "couple_locations": list(det["couple_locations"]),
            "arrival_times": {str(k): float(v) for k, v in det["arrival_times"].items()}
        })

        total += det["cost"]

    return {
        "objective_value": float(total),
        "routes": out_routes,
        "expanded_node_convention": (
            "Let n_orig be the number of physical locations and P be the instance parking-location IDs "
            "in their published list order. Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] "
            "has ID n_orig+i, and the couple copy of P[i] has ID n_orig+len(P)+i. Both copies map back to "
            "physical location P[i]."
        )
    }


def maybe_log():
    global BEST_OBJ, BEST_SOLUTION

    if UNASSIGNED:
        return

    sol = build_solution(ROUTES)
    obj = float(sol["objective_value"])

    if obj < BEST_OBJ - EPS:
        BEST_OBJ = obj
        BEST_SOLUTION = sol

        if LOGGER is not None:
            try:
                LOGGER.log_solution(obj, sol)
            except Exception:
                pass


def improve_solution():
    repair_unassigned()
    maybe_log()

    while time_left() > 2.0:
        move = None

        if not UNASSIGNED:
            move = find_class_change_move()
            if move is None:
                move = find_merge_move(False)
            if move is None:
                move = find_relocate_move()
        else:
            before = len(UNASSIGNED)
            repair_unassigned(max_merges=2)
            if len(UNASSIGNED) < before:
                maybe_log()
                continue
            move = find_merge_move(True)

        if move is None:
            break

        apply_move(move)
        maybe_log()


def main():
    global START_TIME, TIME_LIMIT
    global DEPOT_ID, N_ORIG
    global CUSTOMERS, ALL_CUSTOMER_IDS, UNASSIGNED
    global DIST_MATRIX, TIME_MATRIX, COORDS
    global PARKING_IDS, PARK_INDEX, TRANS_IDS
    global CLASSES, ALL_CLASS_IDS
    global TRUCK_REMAIN, TRAILER_REMAIN, CLASS_REMAIN
    global ROUTES
    global H_START, H_END, MAX_DURATION
    global DECOUPLE_TIME, COUPLING_TIME, TRANSFER_TIME
    global DECOUPLE_COST, COUPLING_COST, TRANSFER_COST
    global HAS_CUSTOM_TW
    global LOGGER
    global BEST_OBJ, BEST_SOLUTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--output", default=None)
    args, _ = parser.parse_known_args()

    START_TIME = time.time()
    TIME_LIMIT = float(args.time_limit)

    if SolutionLogger is not None:
        try:
            LOGGER = SolutionLogger(args.log_path) if args.log_path else SolutionLogger()
        except Exception:
            LOGGER = None

    with open(args.instance, "r") as f:
        data = json.load(f)

    params = data.get("parameters", {})

    horizon = params.get("planning_horizon", [0.0, 1e9])
    try:
        H_START = float(horizon[0])
    except Exception:
        H_START = 0.0
    try:
        H_END = float(horizon[1])
    except Exception:
        H_END = 1e9

    default_max = H_END - H_START if H_END < 1e8 else 1e9
    try:
        MAX_DURATION = float(params.get("max_route_duration", default_max))
    except Exception:
        MAX_DURATION = float(default_max)

    service_default = float(params.get("service_time", 0.0))

    DECOUPLE_TIME = float(params.get("decoupling_time", 0.0))
    COUPLING_TIME = float(params.get("coupling_time", 0.0))
    TRANSFER_TIME = float(params.get("transfer_time", 0.0))

    DECOUPLE_COST = float(params.get("decoupling_cost", 0.0))
    COUPLING_COST = float(params.get("coupling_cost", 0.0))
    TRANSFER_COST = float(params.get("transfer_cost", 0.0))

    depot_raw = data.get("depot", {})
    DEPOT_ID = depot_raw.get("id", 0)
    COORDS[DEPOT_ID] = (float(depot_raw.get("x", 0.0)), float(depot_raw.get("y", 0.0)))

    truck_customers_raw = data.get("truck_customers", [])
    trailer_customers_raw = data.get("trailer_customers", [])
    trailer_customer_ids = []

    for c in truck_customers_raw:
        cid = c.get("id")
        if cid is None:
            continue
        CUSTOMERS[cid] = {
            "id": cid,
            "kind": "truck",
            "supply": float(c.get("supply", c.get("demand", 0.0))),
            "service_time": float(c.get("service_time", service_default)),
            "tw": parse_time_window(c, H_START, H_END)
        }
        COORDS[cid] = (float(c.get("x", 0.0)), float(c.get("y", 0.0)))

    for c in trailer_customers_raw:
        cid = c.get("id")
        if cid is None:
            continue
        CUSTOMERS[cid] = {
            "id": cid,
            "kind": "trailer",
            "supply": float(c.get("supply", c.get("demand", 0.0))),
            "service_time": float(c.get("service_time", service_default)),
            "tw": parse_time_window(c, H_START, H_END)
        }
        COORDS[cid] = (float(c.get("x", 0.0)), float(c.get("y", 0.0)))
        trailer_customer_ids.append(cid)

    trans_raw = data.get("transshipment_locations", [])
    for t in trans_raw:
        if isinstance(t, dict):
            tid = t.get("id")
            if tid is None:
                continue
            TRANS_IDS.append(tid)
            COORDS[tid] = (float(t.get("x", 0.0)), float(t.get("y", 0.0)))
        else:
            TRANS_IDS.append(t)

    ALL_CUSTOMER_IDS = set(CUSTOMERS.keys())
    UNASSIGNED = set(ALL_CUSTOMER_IDS)

    DIST_MATRIX = data.get("distance_matrix", [])
    TIME_MATRIX = data.get("travel_time_matrix", data.get("time_matrix", DIST_MATRIX))

    all_phys_ids = [DEPOT_ID] + list(CUSTOMERS.keys()) + list(TRANS_IDS)
    max_id = max(all_phys_ids) if all_phys_ids else 0

    if DIST_MATRIX:
        N_ORIG = len(DIST_MATRIX)
        if max_id >= N_ORIG:
            N_ORIG = int(max_id) + 1
    else:
        N_ORIG = int(max_id) + 1

    PARKING_IDS = []
    seen_park = set()
    for cid in trailer_customer_ids:
        if cid not in seen_park:
            PARKING_IDS.append(cid)
            seen_park.add(cid)
    for tid in TRANS_IDS:
        if tid not in seen_park:
            PARKING_IDS.append(tid)
            seen_park.add(tid)

    PARK_INDEX = {p: i for i, p in enumerate(PARKING_IDS)}

    fleet = data.get("fleet", {})
    truck_types_raw = fleet.get("truck_types", data.get("truck_types", []))
    trailer_types_raw = fleet.get("trailer_types", data.get("trailer_types", []))
    vehicle_classes_raw = fleet.get("vehicle_classes", data.get("vehicle_classes", []))

    truck_types = {}
    for t in truck_types_raw:
        tid = t.get("id")
        if tid is None:
            continue
        truck_types[tid] = t

    trailer_types = {}
    for r in trailer_types_raw:
        rid = r.get("id")
        if rid is None:
            continue
        trailer_types[rid] = r

    next_cid = 1
    CLASSES = {}

    for vc in vehicle_classes_raw:
        cid = vc.get("class_id", vc.get("id", next_cid))
        try:
            cid = int(cid)
        except Exception:
            pass

        if cid in CLASSES:
            cid = next_cid

        truck_type_id = vc.get("truck_type_id", 0)
        if truck_type_id is None:
            truck_type_id = 0

        trailer_type_id = vc.get("trailer_type_id", 0)
        if trailer_type_id is None:
            trailer_type_id = 0

        tinfo = truck_types.get(truck_type_id, {})
        rinfo = trailer_types.get(trailer_type_id, {}) if trailer_type_id else {}

        truck_capacity = float(vc.get("truck_capacity", tinfo.get("capacity", 0.0)))
        trailer_capacity = float(vc.get("trailer_capacity", rinfo.get("capacity", 0.0)))

        fixed = float(vc.get("truck_fixed_cost", tinfo.get("fixed_cost", 0.0)))
        factor = float(vc.get("truck_cost_factor", tinfo.get("cost_factor", 0.0)))
        tow = float(vc.get("trailer_towing_cost_multiplier", rinfo.get("towing_cost_multiplier", 0.0)))

        avail_raw = vc.get("num_available", None)
        avail = int(avail_raw) if avail_raw is not None else 10**9

        CLASSES[cid] = {
            "class_id": cid,
            "truck_type_id": truck_type_id,
            "trailer_type_id": trailer_type_id,
            "truck_capacity": truck_capacity,
            "trailer_capacity": trailer_capacity,
            "total_capacity": truck_capacity + trailer_capacity,
            "has_trailer": trailer_capacity > EPS,
            "truck_fixed_cost": fixed,
            "truck_cost_factor": factor,
            "trailer_towing_cost_multiplier": tow,
            "num_available": avail
        }

        try:
            next_cid = max(next_cid, int(cid) + 1)
        except Exception:
            next_cid += 1

    if not CLASSES:
        cid = 1
        for t in truck_types.values():
            CLASSES[cid] = {
                "class_id": cid,
                "truck_type_id": t.get("id"),
                "trailer_type_id": 0,
                "truck_capacity": float(t.get("capacity", 0.0)),
                "trailer_capacity": 0.0,
                "total_capacity": float(t.get("capacity", 0.0)),
                "has_trailer": False,
                "truck_fixed_cost": float(t.get("fixed_cost", 0.0)),
                "truck_cost_factor": float(t.get("cost_factor", 0.0)),
                "trailer_towing_cost_multiplier": 0.0,
                "num_available": int(t.get("num_available", 1))
            }
            cid += 1

        for t in truck_types.values():
            for r in trailer_types.values():
                truck_cap = float(t.get("capacity", 0.0))
                trailer_cap = float(r.get("capacity", 0.0))
                CLASSES[cid] = {
                    "class_id": cid,
                    "truck_type_id": t.get("id"),
                    "trailer_type_id": r.get("id"),
                    "truck_capacity": truck_cap,
                    "trailer_capacity": trailer_cap,
                    "total_capacity": truck_cap + trailer_cap,
                    "has_trailer": trailer_cap > EPS,
                    "truck_fixed_cost": float(t.get("fixed_cost", 0.0)),
                    "truck_cost_factor": float(t.get("cost_factor", 0.0)),
                    "trailer_towing_cost_multiplier": float(r.get("towing_cost_multiplier", 0.0)),
                    "num_available": min(int(t.get("num_available", 1)), int(r.get("num_available", 1)))
                }
                cid += 1

    if not CLASSES:
        max_demand = max([CUSTOMERS[c]["supply"] for c in CUSTOMERS], default=1.0)
        CLASSES[1] = {
            "class_id": 1,
            "truck_type_id": 1,
            "trailer_type_id": 0,
            "truck_capacity": max(1000.0, float(max_demand)),
            "trailer_capacity": 0.0,
            "total_capacity": max(1000.0, float(max_demand)),
            "has_trailer": False,
            "truck_fixed_cost": 0.0,
            "truck_cost_factor": 1.0,
            "trailer_towing_cost_multiplier": 0.0,
            "num_available": max(1, len(CUSTOMERS))
        }

    TRUCK_REMAIN = {}
    for tid, t in truck_types.items():
        TRUCK_REMAIN[tid] = int(t.get("num_available", 1))

    TRAILER_REMAIN = {}
    for rid, r in trailer_types.items():
        TRAILER_REMAIN[rid] = int(r.get("num_available", 1))

    CLASS_REMAIN = {}
    for cid, vc in CLASSES.items():
        CLASS_REMAIN[cid] = int(vc["num_available"])

        tt = vc["truck_type_id"]
        if tt not in TRUCK_REMAIN:
            TRUCK_REMAIN[tt] = int(vc["num_available"])

        if vc["trailer_type_id"]:
            rr = vc["trailer_type_id"]
            if rr not in TRAILER_REMAIN:
                TRAILER_REMAIN[rr] = int(vc["num_available"])

    ALL_CLASS_IDS = sorted(
        CLASSES.keys(),
        key=lambda x: (
            CLASSES[x]["truck_fixed_cost"],
            CLASSES[x]["truck_cost_factor"] + CLASSES[x]["trailer_towing_cost_multiplier"],
            -CLASSES[x]["total_capacity"]
        )
    )

    HAS_CUSTOM_TW = False
    for cu in CUSTOMERS.values():
        if abs(cu["tw"][0] - H_START) > EPS or abs(cu["tw"][1] - H_END) > EPS:
            HAS_CUSTOM_TW = True
            break

    ROUTES = []

    if not CUSTOMERS:
        sol = {
            "objective_value": 0.0,
            "routes": [],
            "expanded_node_convention": (
                "Let n_orig be the number of physical locations and P be the instance parking-location IDs "
                "in their published list order. Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] "
                "has ID n_orig+i, and the couple copy of P[i] has ID n_orig+len(P)+i. Both copies map back to "
                "physical location P[i]."
            )
        }
        if LOGGER is not None:
            try:
                LOGGER.log_solution(0.0, sol)
            except Exception:
                pass
        else:
            print(json.dumps(sol))
        return

    construct_initial()
    repair_unassigned()
    maybe_log()
    improve_solution()

    if BEST_SOLUTION is not None:
        final_solution = BEST_SOLUTION
    else:
        final_solution = build_solution(ROUTES)

    if LOGGER is None:
        print(json.dumps(final_solution))
    else:
        if BEST_SOLUTION is None and not UNASSIGNED:
            try:
                LOGGER.log_solution(float(final_solution["objective_value"]), final_solution)
            except Exception:
                pass

    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(final_solution, f)
        except Exception:
            pass


if __name__ == "__main__":
    main()