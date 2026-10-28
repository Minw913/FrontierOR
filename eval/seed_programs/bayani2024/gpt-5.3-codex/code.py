import argparse
import json
import time
from typing import List, Tuple, Dict, Optional

from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve client-server quadratic assignment problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path for incumbents.")
    return parser.parse_args()


def load_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    n_clients = int(data["n_clients"])
    n_servers = int(data["n_servers"])
    linear_costs = data["linear_costs"]

    quad_raw = data.get("quadratic_costs", [])
    quadratic_costs: List[Tuple[int, int, int, int, float]] = []
    for e in quad_raw:
        i, s, j, t, c = e
        quadratic_costs.append((int(i), int(s), int(j), int(t), float(c)))

    return n_clients, n_servers, linear_costs, quadratic_costs


def compute_objective(
    assignment: List[int],
    linear_costs: List[List[float]],
    quadratic_costs: List[Tuple[int, int, int, int, float]],
) -> float:
    total = 0.0
    for i, s in enumerate(assignment):
        total += float(linear_costs[i][s])
    for i, s, j, t, c in quadratic_costs:
        if assignment[i] == s and assignment[j] == t:
            total += c
    return float(total)


def build_client_incidence(
    n_clients: int,
    quadratic_costs: List[Tuple[int, int, int, int, float]],
):
    # For local move delta calculation:
    # entries: (my_server_required, other_client, other_server_required, cost)
    inc = [[] for _ in range(n_clients)]
    for i, s, j, t, c in quadratic_costs:
        inc[i].append((s, j, t, c))
        if j != i:
            inc[j].append((t, i, s, c))
        else:
            # same client appears both sides; the same term depends only on this client's server.
            # We add second-side equivalent only once to avoid double-counting in deltas for self-self terms.
            pass
    return inc


def greedy_local_search(
    n_clients: int,
    n_servers: int,
    linear_costs: List[List[float]],
    quadratic_costs: List[Tuple[int, int, int, int, float]],
    time_budget: float,
) -> Tuple[List[int], float]:
    start = time.time()

    # Initial greedy by linear costs
    assignment = [min(range(n_servers), key=lambda s: linear_costs[i][s]) for i in range(n_clients)]
    current_obj = compute_objective(assignment, linear_costs, quadratic_costs)

    if time.time() - start >= time_budget:
        return assignment, current_obj

    inc = build_client_incidence(n_clients, quadratic_costs)

    # Coordinate descent passes
    improved = True
    while improved and (time.time() - start < time_budget):
        improved = False
        for i in range(n_clients):
            if time.time() - start >= time_budget:
                break

            old_s = assignment[i]
            best_s = old_s
            best_delta = 0.0

            for new_s in range(n_servers):
                if new_s == old_s:
                    continue

                delta = float(linear_costs[i][new_s] - linear_costs[i][old_s])

                # Quadratic terms touching client i
                for req_my_s, other_c, req_other_s, c in inc[i]:
                    other_active = (assignment[other_c] == req_other_s)
                    if not other_active:
                        continue
                    before = 1 if old_s == req_my_s else 0
                    after = 1 if new_s == req_my_s else 0
                    if before != after:
                        delta += c * (after - before)

                if delta < best_delta:
                    best_delta = delta
                    best_s = new_s

            if best_s != old_s:
                assignment[i] = best_s
                current_obj += best_delta
                improved = True

    # Recompute once for numerical consistency
    current_obj = compute_objective(assignment, linear_costs, quadratic_costs)
    return assignment, current_obj


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    t0 = time.time()

    n_clients, n_servers, linear_costs, quadratic_costs = load_instance(args.instance_path)

    # --- Heuristic phase (small fraction of total time) ---
    heuristic_budget = min(max(0.5, 0.15 * args.time_limit), 10.0)
    heuristic_budget = min(heuristic_budget, max(0.0, args.time_limit - 0.1))
    heuristic_assignment, heuristic_obj = greedy_local_search(
        n_clients, n_servers, linear_costs, quadratic_costs, heuristic_budget
    )

    best_known_obj = heuristic_obj
    best_known_assignment = heuristic_assignment[:]
    if logger:
        logger.log(float(best_known_obj))

    elapsed = time.time() - t0
    remaining = args.time_limit - elapsed

    # If no time remains, return heuristic
    if remaining <= 0.01:
        out = {
            "objective_value": float(best_known_obj),
            "assignment": {str(i): int(best_known_assignment[i]) for i in range(n_clients)},
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # --- MIQP model ---
    model = Model("client_server_quadratic_assignment")
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, remaining)
    model.Params.NonConvex = 2
    model.Params.OutputFlag = 0

    x = {}
    for i in range(n_clients):
        for s in range(n_servers):
            x[i, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")

    model.update()

    for i in range(n_clients):
        model.addConstr(quicksum(x[i, s] for s in range(n_servers)) == 1, name=f"assign_{i}")

    # Warm start from heuristic
    for i in range(n_clients):
        hs = heuristic_assignment[i]
        for s in range(n_servers):
            x[i, s].Start = 1.0 if s == hs else 0.0

    lin_obj = quicksum(float(linear_costs[i][s]) * x[i, s] for i in range(n_clients) for s in range(n_servers))
    quad_obj = quicksum(c * x[i, s] * x[j, t] for (i, s, j, t, c) in quadratic_costs)
    model.setObjective(lin_obj + quad_obj, GRB.MINIMIZE)

    best_logged = [float(best_known_obj)]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj + 1e-9 < best_logged[0]:
                    best_logged[0] = obj
                    if logger:
                        logger.log(obj)
            except Exception:
                pass

    model.optimize(cb)

    if model.SolCount > 0:
        sol_assignment = []
        for i in range(n_clients):
            best_s = 0
            best_v = -1.0
            for s in range(n_servers):
                v = x[i, s].X
                if v > best_v:
                    best_v = v
                    best_s = s
            sol_assignment.append(best_s)

        sol_obj = compute_objective(sol_assignment, linear_costs, quadratic_costs)
        if sol_obj < best_known_obj:
            best_known_obj = sol_obj
            best_known_assignment = sol_assignment
    # else keep heuristic

    output = {
        "objective_value": float(best_known_obj),
        "assignment": {str(i): int(best_known_assignment[i]) for i in range(n_clients)},
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()