import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0.0, float(args.time_limit))

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_data = {int(n["node_id"]): n for n in data["nodes"]}
    origin_nodes = [
        i for i, n in node_data.items() if n["node_type"] == "depot_origin"
    ]
    destination_nodes = [
        i for i, n in node_data.items() if n["node_type"] == "depot_destination"
    ]
    if len(origin_nodes) != 1 or len(destination_nodes) != 1:
        raise ValueError("Instance must have exactly one origin and one destination depot")

    origin = origin_nodes[0]
    destination = destination_nodes[0]
    capacity = float(data["vehicle_capacity"])

    request_records = {}
    for nid, n in node_data.items():
        if n["node_type"] in ("pickup", "delivery"):
            rid = n["request_id"]
            if rid not in request_records:
                request_records[rid] = {}
            request_records[rid][n["node_type"]] = nid

    external_request_ids = sorted(request_records, key=lambda x: (str(type(x)), str(x)))
    n_requests = len(external_request_ids)
    request_index = {rid: i for i, rid in enumerate(external_request_ids)}

    pickup = [None] * n_requests
    delivery = [None] * n_requests
    node_request = {}
    node_kind = {}

    for rid, rec in request_records.items():
        if "pickup" not in rec or "delivery" not in rec:
            raise ValueError(f"Request {rid} does not have both pickup and delivery nodes")
        r = request_index[rid]
        pickup[r] = int(rec["pickup"])
        delivery[r] = int(rec["delivery"])
        node_request[pickup[r]] = r
        node_request[delivery[r]] = r
        node_kind[pickup[r]] = "pickup"
        node_kind[delivery[r]] = "delivery"

    arc_cost = {}
    arc_time = {}
    for a in data["arcs"]:
        key = (int(a["from"]), int(a["to"]))
        arc_cost[key] = float(a["cost"])
        arc_time[key] = float(a["travel_time"])

    planning_horizon = data.get("planning_horizon", [0, 0])
    origin_early = float(node_data[origin].get("tw_early", planning_horizon[0]))

    def evaluate_events(events):
        """Return a complete feasible route record, or None."""
        sequence = [origin] + list(events) + [destination]
        stack = []
        picked = set()
        delivered = set()
        load = 0.0
        current_time = max(origin_early, float(node_data[origin]["tw_early"]))
        if current_time > float(node_data[origin]["tw_late"]) + EPS:
            return None

        starts = [current_time]
        loads = [0.0]
        total_cost = 0.0

        for pos in range(1, len(sequence)):
            u = sequence[pos - 1]
            v = sequence[pos]
            key = (u, v)
            if key not in arc_cost:
                return None

            total_cost += arc_cost[key]
            departure = (
                starts[-1]
                + float(node_data[u]["service_time"])
                + arc_time[key]
            )
            service_start = max(departure, float(node_data[v]["tw_early"]))
            if service_start > float(node_data[v]["tw_late"]) + EPS:
                return None

            if v != destination:
                if v not in node_request:
                    return None
                r = node_request[v]
                kind = node_kind[v]

                if kind == "pickup":
                    if r in picked:
                        return None
                    picked.add(r)
                    stack.append(r)
                else:
                    if r not in picked or r in delivered:
                        return None
                    if not stack or stack[-1] != r:
                        return None
                    stack.pop()
                    delivered.add(r)

                load += float(node_data[v]["load"])
                if load < -EPS or load > capacity + EPS:
                    return None
            else:
                if stack:
                    return None
                if abs(load) > 1e-6:
                    return None
                load = 0.0

            starts.append(service_start)
            loads.append(max(0.0, load) if abs(load) < 1e-8 else load)

        if picked != delivered:
            return None

        # Explicit pickup-to-delivery precedence constraint.
        start_by_node = dict(zip(sequence, starts))
        for r in picked:
            p = pickup[r]
            d = delivery[r]
            direct_key = (p, d)
            if direct_key not in arc_time:
                return None
            minimum_delivery_start = (
                start_by_node[p]
                + float(node_data[p]["service_time"])
                + arc_time[direct_key]
            )
            if start_by_node[d] + EPS < minimum_delivery_start:
                return None

        mask = 0
        for r in picked:
            mask |= 1 << r

        return {
            "events": tuple(events),
            "sequence": sequence,
            "starts": starts,
            "loads": loads,
            "cost": float(total_cost),
            "mask": mask,
            "requests": frozenset(picked),
        }

    def solution_from_routes(routes):
        output_routes = []
        objective = 0.0
        for vehicle_id, route in enumerate(routes):
            objective += route["cost"]
            details = []
            for nid, service_start, load_after in zip(
                route["sequence"], route["starts"], route["loads"]
            ):
                details.append(
                    {
                        "node_id": int(nid),
                        "arrival_time": float(service_start),
                        "load_after": float(load_after),
                    }
                )
            output_routes.append(
                {
                    "vehicle_id": int(vehicle_id),
                    "route_sequence": [int(x) for x in route["sequence"]],
                    "route_details": details,
                }
            )
        return {
            "objective_value": float(objective),
            "routes": output_routes,
        }

    # Only the cheapest route for a given covered request set is useful in the
    # set-partitioning master problem.
    candidate_by_mask = {}

    def add_candidate(route):
        if route is None or route["mask"] == 0:
            return False
        old = candidate_by_mask.get(route["mask"])
        if old is None or route["cost"] < old["cost"] - EPS:
            candidate_by_mask[route["mask"]] = route
            return True
        return False

    singleton = [None] * n_requests
    for r in range(n_requests):
        route = evaluate_events((pickup[r], delivery[r]))
        if route is None:
            raise RuntimeError(f"Request {external_request_ids[r]} has no feasible route")
        singleton[r] = route
        add_candidate(route)

    best_routes = list(singleton)
    best_solution = solution_from_routes(best_routes)
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    rng = random.Random(0)

    def all_insertions(base_events, r, keep=2):
        """Find the best feasible ways to insert one pickup-delivery pair."""
        m = len(base_events)
        found = []
        pnode = pickup[r]
        dnode = delivery[r]

        for ppos in range(m + 1):
            with_pickup = (
                tuple(base_events[:ppos])
                + (pnode,)
                + tuple(base_events[ppos:])
            )
            # Delivery must occur after its pickup.
            for dpos in range(ppos + 1, m + 2):
                events = (
                    with_pickup[:dpos]
                    + (dnode,)
                    + with_pickup[dpos:]
                )
                route = evaluate_events(events)
                if route is not None:
                    found.append(route)

        found.sort(key=lambda x: x["cost"])
        result = []
        seen_events = set()
        for route in found:
            if route["events"] not in seen_events:
                result.append(route)
                seen_events.add(route["events"])
                if len(result) >= keep:
                    break
        return result

    total_budget = max(0.0, float(args.time_limit))
    generation_budget = min(total_budget * 0.45, max(0.0, total_budget - 0.5))
    generation_deadline = min(deadline, start_time + generation_budget)

    # Generate all best two-request routes when practical.
    if n_requests <= 80:
        for i in range(n_requests):
            if time.monotonic() >= generation_deadline:
                break
            for j in range(i + 1, n_requests):
                if time.monotonic() >= generation_deadline:
                    break
                alternatives = all_insertions(singleton[i]["events"], j, keep=2)
                for route in alternatives:
                    add_candidate(route)

    def normalized_partition(routes):
        normalized = []
        for route in routes:
            normalized.append(candidate_by_mask.get(route["mask"], route))
        return normalized

    def consider_solution(routes):
        nonlocal best_routes, best_solution, best_objective
        routes = normalized_partition(routes)
        objective = sum(r["cost"] for r in routes)
        if objective < best_objective - 1e-7:
            best_routes = list(routes)
            best_solution = solution_from_routes(best_routes)
            best_objective = best_solution["objective_value"]
            if logger:
                logger.log_solution(best_objective, best_solution)

    # Repeated randomized insertion packing. Every route remains LIFO-feasible
    # because feasibility is checked after each insertion.
    iteration = 0
    max_iterations = max(20, min(3000, 12000 // max(1, n_requests)))
    while (
        time.monotonic() < generation_deadline
        and iteration < max_iterations
        and n_requests > 1
    ):
        iteration += 1
        unassigned = list(range(n_requests))
        rng.shuffle(unassigned)
        constructed = []

        while unassigned and time.monotonic() < generation_deadline:
            seed = unassigned.pop()
            current = singleton[seed]
            add_candidate(current)

            while unassigned and time.monotonic() < generation_deadline:
                if len(unassigned) <= 30:
                    sampled = list(unassigned)
                else:
                    sampled = rng.sample(unassigned, 25)

                choices = []
                for r in sampled:
                    if time.monotonic() >= generation_deadline:
                        break
                    alternatives = all_insertions(current["events"], r, keep=2)
                    if alternatives:
                        best_alt = alternatives[0]
                        saving_delta = (
                            best_alt["cost"] - current["cost"] - singleton[r]["cost"]
                        )
                        choices.append((saving_delta, rng.random(), r, alternatives))

                if not choices:
                    break

                choices.sort(key=lambda x: (x[0], x[1]))
                rcl_size = min(3, len(choices))
                if iteration == 1:
                    chosen = choices[0]
                else:
                    chosen = choices[rng.randrange(rcl_size)]

                _, _, chosen_r, alternatives = chosen
                # Occasionally use the second-best sequence to diversify later
                # insertion possibilities.
                if len(alternatives) > 1 and rng.random() < 0.18:
                    current = alternatives[1]
                else:
                    current = alternatives[0]

                unassigned.remove(chosen_r)
                add_candidate(current)
                for alt in alternatives:
                    add_candidate(alt)

            constructed.append(current)

        # If time expired in the middle of construction, retain feasibility by
        # assigning all remaining requests to singleton routes.
        for r in unassigned:
            constructed.append(singleton[r])
        consider_solution(constructed)

    # Solve a set-partitioning master over all generated feasible routes.
    remaining = deadline - time.monotonic()
    if remaining > 0.05 and n_requests > 0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            candidates = list(candidate_by_mask.values())
            model = gp.Model("LIFO_PDPTW_route_master")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining)

            x = model.addVars(
                len(candidates),
                vtype=GRB.BINARY,
                obj=[r["cost"] for r in candidates],
                name="route",
            )

            containing = [[] for _ in range(n_requests)]
            for k, route in enumerate(candidates):
                mask = route["mask"]
                for r in range(n_requests):
                    if (mask >> r) & 1:
                        containing[r].append(k)

            for r in range(n_requests):
                model.addConstr(
                    gp.quicksum(x[k] for k in containing[r]) == 1,
                    name=f"cover_{r}",
                )

            model.ModelSense = GRB.MINIMIZE

            # Warm-start with the best heuristic partition.
            best_masks = {route["mask"] for route in best_routes}
            for k, route in enumerate(candidates):
                x[k].Start = 1.0 if route["mask"] in best_masks else 0.0

            callback_best = [best_objective]

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                    if objective >= callback_best[0] - 1e-7:
                        return
                    values = cb_model.cbGetSolution([x[k] for k in range(len(candidates))])
                    selected = [
                        candidates[k]
                        for k, value in enumerate(values)
                        if value > 0.5
                    ]
                    if not selected:
                        return
                    covered = 0
                    valid = True
                    for route in selected:
                        if covered & route["mask"]:
                            valid = False
                            break
                        covered |= route["mask"]
                    if not valid or covered != (1 << n_requests) - 1:
                        return
                    sol = solution_from_routes(selected)
                    callback_best[0] = sol["objective_value"]
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)
                except Exception:
                    # Logging must never interrupt optimization.
                    pass

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                selected = [
                    candidates[k]
                    for k in range(len(candidates))
                    if x[k].X > 0.5
                ]
                if selected:
                    master_solution = solution_from_routes(selected)
                    if master_solution["objective_value"] < best_objective - 1e-7:
                        best_routes = selected
                        best_solution = master_solution
                        best_objective = master_solution["objective_value"]
                        if logger:
                            logger.log_solution(best_objective, best_solution)

        except Exception:
            # The heuristic incumbent is always a complete feasible fallback.
            pass

    # Rebuild once to guarantee consistency between the objective and routes.
    best_solution = solution_from_routes(best_routes)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()