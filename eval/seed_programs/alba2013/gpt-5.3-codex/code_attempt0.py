import argparse
import json
import math
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_id_index(coords_list):
    # Distance matrix rows/cols follow coordinates list order
    return {entry["vertex_id"]: idx for idx, entry in enumerate(coords_list)}


def tour_cost(tour: List[int], dist: List[List[int]], id_to_idx: Dict[int, int]) -> float:
    m = len(tour)
    c = 0.0
    for i in range(m):
        a = tour[i]
        b = tour[(i + 1) % m]
        c += dist[id_to_idx[a]][id_to_idx[b]]
    return c


def nearest_neighbor_tour(vertices: List[int], depot: int, dist: List[List[int]], id_to_idx: Dict[int, int]) -> List[int]:
    unvisited = set(vertices)
    unvisited.remove(depot)
    tour = [depot]
    cur = depot
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[id_to_idx[cur]][id_to_idx[j]])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def two_opt(tour: List[int], dist: List[List[int]], id_to_idx: Dict[int, int], time_budget: float = 0.2) -> List[int]:
    # tour is cyclic representation with depot included once at index 0
    start = time.time()
    n = len(tour)
    if n <= 4:
        return tour[:]
    improved = True
    best = tour[:]
    while improved and (time.time() - start) < time_budget:
        improved = False
        for i in range(1, n - 1):
            if (time.time() - start) >= time_budget:
                break
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a = best[i - 1]
                b = best[i]
                c = best[j]
                d = best[(j + 1) % n]
                old = dist[id_to_idx[a]][id_to_idx[b]] + dist[id_to_idx[c]][id_to_idx[d]]
                new = dist[id_to_idx[a]][id_to_idx[c]] + dist[id_to_idx[b]][id_to_idx[d]]
                if new + 1e-9 < old:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
        # loop again if improvement
    return best


def greedy_stack_assignment(
    pickup_order_reqs: List[int],
    delivery_order_reqs: List[int],
    s: int,
    l: int
) -> Optional[Dict[int, int]]:
    # Assign each request to one of s stacks (0..s-1), each stack length <= l,
    # and rd must decrease along pickup order within each stack.
    rd_rank = {r: i for i, r in enumerate(delivery_order_reqs)}
    last_rd = [10**9] * s
    size = [0] * s
    assign = {}

    for r in pickup_order_reqs:
        rd = rd_rank[r]
        candidates = [k for k in range(s) if size[k] < l and last_rd[k] > rd]
        if not candidates:
            return None
        # Best-fit: smallest last_rd that is still > rd
        k = min(candidates, key=lambda kk: last_rd[kk])
        assign[r] = k
        size[k] += 1
        last_rd[k] = rd
    return assign


def arcs_to_tour(vertices: List[int], depot: int, arc_values: Dict[Tuple[int, int], float]) -> List[int]:
    succ = {}
    for (i, j), v in arc_values.items():
        if v > 0.5:
            succ[i] = j
    tour = [depot]
    cur = depot
    used = {depot}
    for _ in range(len(vertices) - 1):
        if cur not in succ:
            break
        cur = succ[cur]
        if cur in used:
            break
        tour.append(cur)
        used.add(cur)
    # If reconstruction incomplete, append missing vertices (safety fallback)
    if len(tour) < len(vertices):
        missing = [v for v in vertices if v not in used]
        tour.extend(missing)
    return tour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    s = int(inst["parameters"]["num_stacks_s"])
    l = int(inst["parameters"]["stack_height_l"])
    n = int(inst["parameters"]["num_requests_n"])

    pickup_depot = inst["pickup_region"]["depot_index"]
    delivery_depot = inst["delivery_region"]["depot_index"]

    p_coords = inst["pickup_region"]["coordinates"]
    d_coords = inst["delivery_region"]["coordinates"]
    p_dist = inst["pickup_region"]["distance_matrix"]
    d_dist = inst["delivery_region"]["distance_matrix"]

    p_id_to_idx = build_id_index(p_coords)
    d_id_to_idx = build_id_index(d_coords)

    pickup_vertices = [c["vertex_id"] for c in p_coords]
    delivery_vertices = [c["vertex_id"] for c in d_coords]

    reqs = inst["requests"]
    req_ids = [r["request_id"] for r in reqs]
    pv_of = {r["request_id"]: r["pickup_vertex"] for r in reqs}
    dv_of = {r["request_id"]: r["delivery_vertex"] for r in reqs}
    pv_to_req = {r["pickup_vertex"]: r["request_id"] for r in reqs}
    dv_to_req = {r["delivery_vertex"]: r["request_id"] for r in reqs}

    # ---------------- Heuristic initial solution ----------------
    p_tour = nearest_neighbor_tour(pickup_vertices, pickup_depot, p_dist, p_id_to_idx)
    p_tour = two_opt(p_tour, p_dist, p_id_to_idx, time_budget=min(0.5, args.time_limit * 0.05))

    d_tour = nearest_neighbor_tour(delivery_vertices, delivery_depot, d_dist, d_id_to_idx)
    d_tour = two_opt(d_tour, d_dist, d_id_to_idx, time_budget=min(0.5, args.time_limit * 0.05))

    pickup_order_reqs = [pv_to_req[v] for v in p_tour if v != pickup_depot]
    delivery_order_reqs = [dv_to_req[v] for v in d_tour if v != delivery_depot]

    assign = greedy_stack_assignment(pickup_order_reqs, delivery_order_reqs, s, l)

    if assign is None:
        # Guaranteed-feasible fallback construction (if capacity allows)
        # Partition pickup order into stack chunks, then delivery is concatenation of reversed chunks.
        if n <= s * l:
            stacks = [[] for _ in range(s)]
            for i, r in enumerate(pickup_order_reqs):
                k = i // l
                if k >= s:
                    k = s - 1
                stacks[k].append(r)
            assign = {}
            for k in range(s):
                for r in stacks[k]:
                    assign[r] = k
            delivery_order_reqs = []
            for k in range(s):
                delivery_order_reqs.extend(reversed(stacks[k]))
            d_tour = [delivery_depot] + [dv_of[r] for r in delivery_order_reqs]
            d_tour = two_opt(d_tour, d_dist, d_id_to_idx, time_budget=min(0.3, args.time_limit * 0.03))
            # re-check; if 2-opt broke feasibility, revert
            d_check_reqs = [dv_to_req[v] for v in d_tour if v != delivery_depot]
            if greedy_stack_assignment(pickup_order_reqs, d_check_reqs, s, l) is None:
                d_tour = [delivery_depot] + [dv_of[r] for r in delivery_order_reqs]
        else:
            # Infeasible by capacity; keep independent tours (best effort)
            assign = {r: (i % max(1, s)) for i, r in enumerate(pickup_order_reqs)}

    best_pickup = p_tour[:]
    best_delivery = d_tour[:]
    best_obj = tour_cost(best_pickup, p_dist, p_id_to_idx) + tour_cost(best_delivery, d_dist, d_id_to_idx)
    if logger:
        logger.log(best_obj)

    # ---------------- Optional MIP improvement ----------------
    elapsed = time.time() - t0
    remaining = args.time_limit - elapsed

    # Full model can become large; skip for very large n or very low remaining time
    if remaining > 2 and n <= 45:
        model = gp.Model("2TSPMS")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(1, int(remaining))
        model.Params.Threads = 1
        model.Params.MIPFocus = 1

        # Arc vars
        x = {}
        for i in pickup_vertices:
            for j in pickup_vertices:
                if i != j:
                    x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        y = {}
        for i in delivery_vertices:
            for j in delivery_vertices:
                if i != j:
                    y[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

        # Order vars (requests only)
        u = {r: model.addVar(vtype=GRB.INTEGER, lb=1, ub=n, name=f"u_{r}") for r in req_ids}
        v = {r: model.addVar(vtype=GRB.INTEGER, lb=1, ub=n, name=f"v_{r}") for r in req_ids}

        # Stack assignment
        z = {(r, k): model.addVar(vtype=GRB.BINARY, name=f"z_{r}_{k}") for r in req_ids for k in range(s)}

        # Pair order binaries
        pairs = []
        for i in range(len(req_ids)):
            for j in range(i + 1, len(req_ids)):
                pairs.append((req_ids[i], req_ids[j]))

        a = {(r, q): model.addVar(vtype=GRB.BINARY, name=f"a_{r}_{q}") for (r, q) in pairs}  # r before q pickup
        b = {(r, q): model.addVar(vtype=GRB.BINARY, name=f"b_{r}_{q}") for (r, q) in pairs}  # r before q delivery

        w = {(r, q, k): model.addVar(vtype=GRB.BINARY, name=f"w_{r}_{q}_{k}") for (r, q) in pairs for k in range(s)}

        model.update()

        # Degree constraints pickup
        for i in pickup_vertices:
            model.addConstr(gp.quicksum(x[(i, j)] for j in pickup_vertices if j != i) == 1)
            model.addConstr(gp.quicksum(x[(j, i)] for j in pickup_vertices if j != i) == 1)

        # Degree constraints delivery
        for i in delivery_vertices:
            model.addConstr(gp.quicksum(y[(i, j)] for j in delivery_vertices if j != i) == 1)
            model.addConstr(gp.quicksum(y[(j, i)] for j in delivery_vertices if j != i) == 1)

        # MTZ pickup on requests
        for r in req_ids:
            for q in req_ids:
                if r == q:
                    continue
                model.addConstr(u[r] - u[q] + n * x[(pv_of[r], pv_of[q])] <= n - 1)

        # MTZ delivery on requests
        for r in req_ids:
            for q in req_ids:
                if r == q:
                    continue
                model.addConstr(v[r] - v[q] + n * y[(dv_of[r], dv_of[q])] <= n - 1)

        # Stack assignment
        for r in req_ids:
            model.addConstr(gp.quicksum(z[(r, k)] for k in range(s)) == 1)
        for k in range(s):
            model.addConstr(gp.quicksum(z[(r, k)] for r in req_ids) <= l)

        M = n
        # Link a,b with order variables
        for (r, q) in pairs:
            # a=1 => u[r] < u[q], a=0 => u[q] < u[r]
            model.addConstr(u[r] - u[q] <= -1 + M * (1 - a[(r, q)]))
            model.addConstr(u[q] - u[r] <= -1 + M * a[(r, q)])

            # b=1 => v[r] < v[q], b=0 => v[q] < v[r]
            model.addConstr(v[r] - v[q] <= -1 + M * (1 - b[(r, q)]))
            model.addConstr(v[q] - v[r] <= -1 + M * b[(r, q)])

        # same-stack + opposite-order constraints
        for (r, q) in pairs:
            for k in range(s):
                model.addConstr(w[(r, q, k)] <= z[(r, k)])
                model.addConstr(w[(r, q, k)] <= z[(q, k)])
                model.addConstr(w[(r, q, k)] >= z[(r, k)] + z[(q, k)] - 1)

                # if w=1 => a+b=1
                model.addConstr(a[(r, q)] + b[(r, q)] <= 1 + (1 - w[(r, q, k)]))
                model.addConstr(a[(r, q)] + b[(r, q)] >= 1 - (1 - w[(r, q, k)]))

        # Objective
        obj_p = gp.quicksum(
            p_dist[p_id_to_idx[i]][p_id_to_idx[j]] * x[(i, j)]
            for i in pickup_vertices for j in pickup_vertices if i != j
        )
        obj_d = gp.quicksum(
            d_dist[d_id_to_idx[i]][d_id_to_idx[j]] * y[(i, j)]
            for i in delivery_vertices for j in delivery_vertices if i != j
        )
        model.setObjective(obj_p + obj_d, GRB.MINIMIZE)

        # Warm start from heuristic
        def set_tour_start(var_dict, tour):
            arcs = set((tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour)))
            for key, var in var_dict.items():
                var.Start = 1.0 if key in arcs else 0.0

        set_tour_start(x, best_pickup)
        set_tour_start(y, best_delivery)

        p_req_order = [pv_to_req[v] for v in best_pickup if v != pickup_depot]
        d_req_order = [dv_to_req[v] for v in best_delivery if v != delivery_depot]
        pos_u = {r: i + 1 for i, r in enumerate(p_req_order)}
        pos_v = {r: i + 1 for i, r in enumerate(d_req_order)}
        for r in req_ids:
            if r in pos_u:
                u[r].Start = pos_u[r]
            if r in pos_v:
                v[r].Start = pos_v[r]

        start_assign = greedy_stack_assignment(p_req_order, d_req_order, s, l)
        if start_assign is None:
            start_assign = assign
        for r in req_ids:
            for k in range(s):
                z[(r, k)].Start = 1.0 if start_assign.get(r, 0) == k else 0.0

        for (r, q) in pairs:
            if r in pos_u and q in pos_u:
                a[(r, q)].Start = 1.0 if pos_u[r] < pos_u[q] else 0.0
            if r in pos_v and q in pos_v:
                b[(r, q)].Start = 1.0 if pos_v[r] < pos_v[q] else 0.0
            for k in range(s):
                wk = 1.0 if (start_assign.get(r, -1) == k and start_assign.get(q, -1) == k) else 0.0
                w[(r, q, k)].Start = wk

        # Incumbent logging callback
        model._best_logged = best_obj
        model._logger = logger

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < m._best_logged:
                    m._best_logged = obj
                    if m._logger:
                        m._logger.log(float(obj))

        model.optimize(cb)

        if model.SolCount > 0:
            x_sol = {(i, j): x[(i, j)].X for (i, j) in x}
            y_sol = {(i, j): y[(i, j)].X for (i, j) in y}
            cand_pickup = arcs_to_tour(pickup_vertices, pickup_depot, x_sol)
            cand_delivery = arcs_to_tour(delivery_vertices, delivery_depot, y_sol)

            cand_obj = tour_cost(cand_pickup, p_dist, p_id_to_idx) + tour_cost(cand_delivery, d_dist, d_id_to_idx)
            if cand_obj + 1e-9 < best_obj:
                best_obj = cand_obj
                best_pickup = cand_pickup
                best_delivery = cand_delivery

    # Final output
    solution = {
        "objective_value": float(best_obj),
        "pickup_tour": best_pickup,
        "delivery_tour": best_delivery
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()