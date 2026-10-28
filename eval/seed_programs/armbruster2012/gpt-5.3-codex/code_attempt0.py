import argparse
import json
import math
import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_cut_objective(x: List[int], edges: List[Tuple[int, int]], costs: List[float]) -> float:
    obj = 0.0
    for (u, v), c in zip(edges, costs):
        if x[u] != x[v]:
            obj += c
    return float(obj)


def build_adjacency(num_nodes: int, edges: List[Tuple[int, int]], costs: List[float]):
    adj = [[] for _ in range(num_nodes)]
    for (u, v), c in zip(edges, costs):
        adj[u].append((v, c))
        adj[v].append((u, c))
    return adj


def flip_delta(i: int, x: List[int], adj) -> float:
    # Change in cut objective if node i is flipped
    xi = x[i]
    d = 0.0
    for j, c in adj[i]:
        if x[j] == xi:
            d += c
        else:
            d -= c
    return d


def greedy_initial_solution(
    num_nodes: int,
    node_weights: List[int],
    edges: List[Tuple[int, int]],
    costs: List[float],
    F: int,
    total_w: int,
    time_deadline: float,
):
    """
    Build a feasible solution with x[0]=1 if possible.
    Feasibility requires sum_w_S in [L, U], where L=total_w-F, U=F.
    """
    L = total_w - F
    U = F

    x = [0] * num_nodes
    x[0] = 1
    wS = node_weights[0]

    adj = build_adjacency(num_nodes, edges, costs)

    # If already impossible with node 0 fixed in S
    if wS > U:
        return x, compute_cut_objective(x, edges, costs), False

    # Start from all others in complement, then add nodes until reaching L.
    inS = {0}
    candidates = set(range(1, num_nodes))

    while wS < L and candidates and time.time() < time_deadline:
        best_i = None
        best_key = None  # (delta_per_weight, delta, weight)
        for i in candidates:
            wi = node_weights[i]
            if wS + wi > U:
                continue
            # delta for moving 0->1 from current assignment
            # (current x[i]=0)
            delta = flip_delta(i, x, adj)
            key = (delta / max(1, wi), delta, wi)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i

        if best_i is None:
            break

        x[best_i] = 1
        inS.add(best_i)
        candidates.remove(best_i)
        wS += node_weights[best_i]

    feasible = (L <= wS <= U)

    # If still infeasible, try simple fill by smallest weights
    if not feasible and wS < L:
        rem = sorted(
            [i for i in range(1, num_nodes) if x[i] == 0],
            key=lambda i: node_weights[i]
        )
        for i in rem:
            wi = node_weights[i]
            if wS + wi <= U:
                x[i] = 1
                wS += wi
                if wS >= L:
                    break
        feasible = (L <= wS <= U)

    # Local improvement (1-flip)
    if feasible:
        improved = True
        while improved and time.time() < time_deadline:
            improved = False
            best_i = None
            best_delta = -1e-12
            for i in range(1, num_nodes):  # keep node 0 fixed
                wi = node_weights[i]
                if x[i] == 1:
                    new_w = wS - wi
                else:
                    new_w = wS + wi
                if not (L <= new_w <= U):
                    continue
                d = flip_delta(i, x, adj)
                if d < best_delta:
                    best_delta = d
                    best_i = i
            if best_i is not None:
                wi = node_weights[best_i]
                x[best_i] = 1 - x[best_i]
                wS = wS + wi if x[best_i] == 1 else wS - wi
                improved = True

    obj = compute_cut_objective(x, edges, costs)
    return x, obj, feasible


class IncumbentCallback:
    def __init__(self, logger: SolutionLogger = None, initial_best: float = float("inf")):
        self.logger = logger
        self.best = initial_best

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < self.best:
                self.best = obj
                if self.logger:
                    self.logger.log(float(obj))


def solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    n = int(instance["num_nodes"])
    m = int(instance["num_edges"])
    F = int(instance["bisection_capacity_F"])
    total_w = int(instance["total_node_weight"])
    node_weights = list(map(int, instance["node_weights"]))
    edges = [tuple(map(int, e)) for e in instance["edges"]]
    costs = [float(c) for c in instance["edge_weights"]]

    start_time = time.time()
    deadline = start_time + max(0, time_limit)

    # Heuristic start (small budget)
    heur_budget = min(2.0, max(0.2, 0.15 * time_limit))
    hx, hobj, hfeas = greedy_initial_solution(
        n, node_weights, edges, costs, F, total_w, time.time() + heur_budget
    )

    logged_best = float("inf")
    if hfeas and logger:
        logger.log(float(hobj))
        logged_best = float(hobj)

    # Build MIP
    model = gp.Model("capacitated_graph_bisection")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    remaining = max(0.0, deadline - time.time())
    model.Params.TimeLimit = remaining

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

    # Symmetry breaking: node 0 in S
    model.addConstr(x[0] == 1, name="fix_node0")

    # Capacity on both partitions:
    # total-F <= sum(w_i x_i) <= F
    w_expr = gp.quicksum(node_weights[i] * x[i] for i in range(n))
    model.addConstr(w_expr <= F, name="cap_S")
    model.addConstr(w_expr >= total_w - F, name="cap_C")

    # y_e = |x_u - x_v| linearization
    for e, (u, v) in enumerate(edges):
        model.addConstr(y[e] >= x[u] - x[v], name=f"abs1_{e}")
        model.addConstr(y[e] >= x[v] - x[u], name=f"abs2_{e}")
        model.addConstr(y[e] <= x[u] + x[v], name=f"abs3_{e}")
        model.addConstr(y[e] <= 2 - x[u] - x[v], name=f"abs4_{e}")

    model.setObjective(gp.quicksum(costs[e] * y[e] for e in range(m)), GRB.MINIMIZE)

    # Warm start from heuristic if feasible
    if hfeas:
        for i in range(n):
            x[i].Start = hx[i]
        for e, (u, v) in enumerate(edges):
            y[e].Start = 1.0 if hx[u] != hx[v] else 0.0

    cb = IncumbentCallback(logger=logger, initial_best=logged_best)

    try:
        model.optimize(cb)
    except gp.GurobiError:
        pass

    # Extract best available solution
    best_x = None
    best_obj = None

    if model.SolCount > 0:
        sol_x = [1 if x[i].X >= 0.5 else 0 for i in range(n)]
        best_x = sol_x
        best_obj = compute_cut_objective(sol_x, edges, costs)
    elif hfeas:
        best_x = hx
        best_obj = hobj
    else:
        # Last-resort fallback (may be infeasible if instance itself infeasible with x0=1)
        best_x = [0] * n
        best_x[0] = 1
        best_obj = compute_cut_objective(best_x, edges, costs)

    # Ensure node 0 is in partition_S per schema
    if best_x[0] == 0:
        best_x = [1 - v for v in best_x]
        best_obj = compute_cut_objective(best_x, edges, costs)

    partition_S = [i for i in range(n) if best_x[i] == 1]
    partition_complement = [i for i in range(n) if best_x[i] == 0]

    # Final log safeguard
    if logger and best_obj + 1e-9 < cb.best:
        logger.log(float(best_obj))

    return {
        "objective_value": float(best_obj),
        "partition_S": partition_S,
        "partition_complement": partition_complement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()