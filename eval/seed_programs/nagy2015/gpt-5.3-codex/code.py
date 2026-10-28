import argparse
import json
import math
import random
import time
from typing import List, Dict, Tuple, Optional

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False


def route_cost_customer_seq(seq: List[int], dist: List[List[int]]) -> float:
    if not seq:
        return 0.0
    c = dist[0][seq[0]]
    for i in range(len(seq) - 1):
        c += dist[seq[i]][seq[i + 1]]
    c += dist[seq[-1]][0]
    return float(c)


def internal_cost(seq: List[int], dist: List[List[int]]) -> float:
    c = 0.0
    for i in range(len(seq) - 1):
        c += dist[seq[i]][seq[i + 1]]
    return float(c)


def two_opt(seq: List[int], dist: List[List[int]], deadline: float) -> List[int]:
    if len(seq) < 4:
        return seq
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(seq)
        for i in range(n - 2):
            if time.time() >= deadline:
                break
            a = 0 if i == 0 else seq[i - 1]
            b = seq[i]
            for j in range(i + 1, n - 1):
                c = seq[j]
                d = seq[j + 1]
                old = dist[a][b] + dist[c][d]
                new = dist[a][c] + dist[b][d]
                if new + 1e-9 < old:
                    seq[i:j + 1] = reversed(seq[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return seq


def clarke_wright(
    customers: List[int],
    demand: Dict[int, int],
    capacity: int,
    dist: List[List[int]],
    deadline: float,
    randomization: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    if rng is None:
        rng = random.Random(0)

    for c in customers:
        if demand[c] > capacity:
            raise ValueError(f"Infeasible instance: demand {demand[c]} > capacity {capacity} for customer {c}")

    if not customers:
        return []

    routes = {}
    route_load = {}
    cust_route = {}
    rid = 0
    for c in customers:
        routes[rid] = [c]
        route_load[rid] = demand[c]
        cust_route[c] = rid
        rid += 1

    savings = []
    m = len(customers)
    max_base = 0.0
    for i_idx in range(m):
        i = customers[i_idx]
        for j_idx in range(i_idx + 1, m):
            j = customers[j_idx]
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((i, j, float(s)))
            if s > max_base:
                max_base = s

    noise_scale = randomization * max_base if max_base > 0 else 0.0
    if noise_scale > 0:
        scored = []
        for i, j, s in savings:
            scored.append((s + rng.uniform(-noise_scale, noise_scale), i, j, s))
        scored.sort(reverse=True, key=lambda x: x[0])
        ordered_pairs = [(i, j, s) for _, i, j, s in scored]
    else:
        savings.sort(reverse=True, key=lambda x: x[2])
        ordered_pairs = savings

    for i, j, _ in ordered_pairs:
        if time.time() >= deadline:
            break
        r1 = cust_route.get(i)
        r2 = cust_route.get(j)
        if r1 is None or r2 is None or r1 == r2:
            continue
        if route_load[r1] + route_load[r2] > capacity:
            continue

        seq1 = routes[r1]
        seq2 = routes[r2]
        if i != seq1[0] and i != seq1[-1]:
            continue
        if j != seq2[0] and j != seq2[-1]:
            continue

        if seq1[-1] != i:
            seq1 = list(reversed(seq1))
        if seq2[0] != j:
            seq2 = list(reversed(seq2))

        new_seq = seq1 + seq2
        new_rid = r1
        old_rid = r2

        routes[new_rid] = new_seq
        route_load[new_rid] = route_load[r1] + route_load[r2]

        for c in routes[old_rid]:
            cust_route[c] = new_rid
        for c in routes[new_rid]:
            cust_route[c] = new_rid

        del routes[old_rid]
        del route_load[old_rid]

    final_routes = list(routes.values())

    # Intra-route 2-opt polishing
    for idx in range(len(final_routes)):
        if time.time() >= deadline:
            break
        final_routes[idx] = two_opt(final_routes[idx], dist, deadline)

    return final_routes


def build_route_info(routes: List[List[int]], demand: Dict[int, int], dist: List[List[int]]) -> List[Dict]:
    infos = []
    for seq in routes:
        ld = sum(demand[c] for c in seq)
        infos.append({
            "seq": seq,
            "load": ld,
            "cost": route_cost_customer_seq(seq, dist),
            "internal": internal_cost(seq, dist),
        })
    return infos


def pair_routes(
    del_infos: List[Dict],
    pick_infos: List[Dict],
    dist: List[List[int]],
    deadline: float
) -> Tuple[Dict[int, Tuple[int, List[int], List[int]]], float]:
    # returns matched delivery idx -> (pickup idx, oriented_del_seq, oriented_pick_seq), total saving
    if not del_infos or not pick_infos:
        return {}, 0.0

    pair_data = []  # (a, b, saving, best_del_seq, best_pick_seq)
    for a, di in enumerate(del_infos):
        dseq = di["seq"]
        dseq_r = list(reversed(dseq))
        dint = di["internal"]
        dcost = di["cost"]
        for b, pi in enumerate(pick_infos):
            pseq = pi["seq"]
            pseq_r = list(reversed(pseq))
            pint = pi["internal"]
            pcost = pi["cost"]

            best = None
            best_pair = None
            for ds in (dseq, dseq_r):
                for ps in (pseq, pseq_r):
                    c = dist[0][ds[0]] + dint + dist[ds[-1]][ps[0]] + pint + dist[ps[-1]][0]
                    if best is None or c < best:
                        best = c
                        best_pair = (ds, ps)
            saving = (dcost + pcost - best) if best is not None else 0.0
            if saving > 1e-9:
                pair_data.append((a, b, float(saving), best_pair[0], best_pair[1]))

    if not pair_data:
        return {}, 0.0

    matched = {}
    total_saving = 0.0

    # Try exact max-weight bipartite matching via Gurobi
    if GUROBI_AVAILABLE and time.time() < deadline:
        try:
            rem = max(0.1, deadline - time.time())
            model = gp.Model("pairing")
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = rem

            x = {}
            for k, (a, b, s, _, _) in enumerate(pair_data):
                x[k] = model.addVar(vtype=GRB.BINARY, obj=s, name=f"x_{k}")

            # delivery-side constraints
            by_a = {}
            by_b = {}
            for k, (a, b, s, _, _) in enumerate(pair_data):
                by_a.setdefault(a, []).append(k)
                by_b.setdefault(b, []).append(k)

            for a, ks in by_a.items():
                model.addConstr(gp.quicksum(x[k] for k in ks) <= 1)
            for b, ks in by_b.items():
                model.addConstr(gp.quicksum(x[k] for k in ks) <= 1)

            model.ModelSense = GRB.MAXIMIZE
            model.optimize()

            if model.SolCount > 0:
                for k, var in x.items():
                    if var.X > 0.5:
                        a, b, s, ds, ps = pair_data[k]
                        matched[a] = (b, list(ds), list(ps))
                        total_saving += s
                return matched, total_saving
        except Exception:
            pass

    # Greedy fallback
    pair_data.sort(key=lambda t: t[2], reverse=True)
    used_a = set()
    used_b = set()
    for a, b, s, ds, ps in pair_data:
        if time.time() >= deadline:
            break
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        matched[a] = (b, list(ds), list(ps))
        total_saving += s

    return matched, total_saving


def evaluate_and_build_solution(
    del_routes: List[List[int]],
    pick_routes: List[List[int]],
    n: int,
    dist: List[List[int]],
) -> Tuple[float, List[List[int]]]:
    del_infos = build_route_info(del_routes, {c: 1 for r in del_routes for c in r}, dist)  # dummy demand not used
    pick_infos = build_route_info(pick_routes, {c: 1 for r in pick_routes for c in r}, dist)  # dummy demand not used
    # overwrite correct costs only (load irrelevant in this function)
    for idx, r in enumerate(del_routes):
        del_infos[idx]["seq"] = r
        del_infos[idx]["cost"] = route_cost_customer_seq(r, dist)
        del_infos[idx]["internal"] = internal_cost(r, dist)
    for idx, r in enumerate(pick_routes):
        pick_infos[idx]["seq"] = r
        pick_infos[idx]["cost"] = route_cost_customer_seq(r, dist)
        pick_infos[idx]["internal"] = internal_cost(r, dist)

    matched, _ = pair_routes(del_infos, pick_infos, dist, deadline=time.time() + 1e6)

    used_del = set(matched.keys())
    used_pick = set(v[0] for v in matched.values())

    expanded_routes = []

    for a, (b, dseq, pseq) in matched.items():
        r = [0] + dseq + [x + n for x in pseq] + [0]
        expanded_routes.append(r)

    for a, r in enumerate(del_routes):
        if a not in used_del:
            expanded_routes.append([0] + r + [0])

    for b, r in enumerate(pick_routes):
        if b not in used_pick:
            expanded_routes.append([0] + [x + n for x in r] + [0])

    # objective
    obj = 0.0
    for r in expanded_routes:
        for i in range(len(r) - 1):
            u = r[i]
            v = r[i + 1]
            ou = 0 if u == 0 else (u if u <= n else u - n)
            ov = 0 if v == 0 else (v if v <= n else v - n)
            obj += dist[ou][ov]

    return float(obj), expanded_routes


def build_detailed_routes(routes: List[List[int]], n: int, customers: List[Dict]) -> List[List[Dict]]:
    d_dem = {c["id"]: c["delivery_demand"] for c in customers}
    p_dem = {c["id"]: c["pickup_demand"] for c in customers}
    detailed = []

    for r in routes:
        rr = []
        for node in r:
            if node == 0:
                rr.append({
                    "node_id": 0,
                    "role": "depot",
                    "customer_id": 0,
                    "delivery_quantity": 0,
                    "pickup_quantity": 0
                })
            elif 1 <= node <= n:
                cid = node
                rr.append({
                    "node_id": node,
                    "role": "linehaul",
                    "customer_id": cid,
                    "delivery_quantity": int(d_dem.get(cid, 0)),
                    "pickup_quantity": 0
                })
            else:
                cid = node - n
                rr.append({
                    "node_id": node,
                    "role": "backhaul",
                    "customer_id": cid,
                    "delivery_quantity": 0,
                    "pickup_quantity": int(p_dem.get(cid, 0))
                })
        detailed.append(rr)

    return detailed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    deadline = start + max(1, args.time_limit)

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["num_customers"])
    C = int(instance["vehicle_capacity"])
    customers = instance["customers"]
    dist = instance["distance_matrix"]

    d_dem = {c["id"]: int(c["delivery_demand"]) for c in customers}
    p_dem = {c["id"]: int(c["pickup_demand"]) for c in customers}

    d_customers = [c["id"] for c in customers if d_dem[c["id"]] > 0]
    p_customers = [c["id"] for c in customers if p_dem[c["id"]] > 0]

    # Feasibility quick-check
    infeas = False
    for cid in d_customers:
        if d_dem[cid] > C:
            infeas = True
    for cid in p_customers:
        if p_dem[cid] > C:
            infeas = True

    best_obj = float("inf")
    best_routes = []

    if infeas:
        # Return empty as fail-safe (schema-valid output); instance should generally be feasible.
        output = {
            "objective_value": float("inf"),
            "routes": [],
            "routes_detailed": []
        }
        with open(args.solution_path, "w") as f:
            json.dump(output, f, indent=2)
        return

    rng = random.Random(42)
    it = 0

    while time.time() < deadline - 0.05:
        rem = deadline - time.time()
        if rem <= 0.05:
            break

        # deterministic first, then randomized multi-start
        rand_strength = 0.0 if it == 0 else min(0.35, 0.08 * it)

        # split remaining time budget across two CVRPs
        sub_deadline_1 = time.time() + max(0.01, (deadline - time.time()) * 0.45)
        del_routes = clarke_wright(
            d_customers, d_dem, C, dist, sub_deadline_1,
            randomization=rand_strength, rng=rng
        )

        sub_deadline_2 = time.time() + max(0.01, (deadline - time.time()) * 0.45)
        pick_routes = clarke_wright(
            p_customers, p_dem, C, dist, sub_deadline_2,
            randomization=rand_strength, rng=rng
        )

        # Pairing with exact/greedy matching under remaining time
        del_infos = build_route_info(del_routes, d_dem, dist)
        pick_infos = build_route_info(pick_routes, p_dem, dist)
        match_deadline = time.time() + max(0.01, (deadline - time.time()) * 0.8)
        matched, _ = pair_routes(del_infos, pick_infos, dist, match_deadline)

        used_del = set(matched.keys())
        used_pick = set(v[0] for v in matched.values())

        routes = []
        for a, (b, dseq, pseq) in matched.items():
            routes.append([0] + dseq + [x + n for x in pseq] + [0])
        for a, r in enumerate(del_routes):
            if a not in used_del:
                routes.append([0] + r + [0])
        for b, r in enumerate(pick_routes):
            if b not in used_pick:
                routes.append([0] + [x + n for x in r] + [0])

        obj = 0.0
        for r in routes:
            for i in range(len(r) - 1):
                u = r[i]
                v = r[i + 1]
                ou = 0 if u == 0 else (u if u <= n else u - n)
                ov = 0 if v == 0 else (v if v <= n else v - n)
                obj += dist[ou][ov]

        if obj + 1e-9 < best_obj:
            best_obj = float(obj)
            best_routes = routes
            if logger:
                logger.log(best_obj)

        it += 1
        if it >= 200:  # sanity guard
            break

    # fallback in rare edge case
    if not best_routes and (d_customers or p_customers):
        best_routes = []
        for c in d_customers:
            best_routes.append([0, c, 0])
        for c in p_customers:
            best_routes.append([0, c + n, 0])
        best_obj = 0.0
        for r in best_routes:
            for i in range(len(r) - 1):
                u, v = r[i], r[i + 1]
                ou = 0 if u == 0 else (u if u <= n else u - n)
                ov = 0 if v == 0 else (v if v <= n else v - n)
                best_obj += dist[ou][ov]
        if logger:
            logger.log(best_obj)

    routes_detailed = build_detailed_routes(best_routes, n, customers)

    output = {
        "objective_value": float(best_obj if math.isfinite(best_obj) else 0.0),
        "routes": best_routes,
        "routes_detailed": routes_detailed
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()