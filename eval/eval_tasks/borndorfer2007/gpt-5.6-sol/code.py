import argparse
import json
import math
import os
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enumerate_candidate_lines(instance, start_time, time_limit):
    network = instance["network"]
    edges = network["edges"]
    modes = instance["modes"]
    max_length = int(instance["global_parameters"]["max_line_length_edges"])

    edge_by_id = {int(e["id"]): e for e in edges}
    edge_position = {int(e["id"]): i for i, e in enumerate(edges)}

    # Leave most of the time for model construction and optimization.
    enumeration_budget = min(12.0, max(0.25, 0.15 * max(1, time_limit)))
    enumeration_deadline = start_time + enumeration_budget

    # Bound model size while still permitting a broad line set.
    candidate_limit = 25000
    if time_limit <= 10:
        candidate_limit = 5000
    elif time_limit <= 30:
        candidate_limit = 12000

    candidates = []
    seen = set()

    def add_candidate(mode_index, mode, nodes, edge_ids):
        if not edge_ids or len(edge_ids) > max_length:
            return
        forward = tuple(edge_ids)
        reverse = tuple(reversed(edge_ids))
        canonical_edges = min(forward, reverse)
        key = (mode_index, canonical_edges)
        if key in seen:
            return
        seen.add(key)

        if forward != canonical_edges:
            nodes = list(reversed(nodes))
            edge_ids = list(reversed(edge_ids))

        fixed_cost = float(
            mode.get(
                "fixed_cost_per_line",
                instance["global_parameters"].get("fixed_cost_per_line", 0.0),
            )
        )
        operating_cost = sum(
            float(edge_by_id[eid].get(
                "operating_cost",
                mode.get(
                    "operating_cost_per_edge",
                    instance["global_parameters"].get("operating_cost_per_edge", 0.0),
                ),
            ))
            for eid in edge_ids
        )

        candidates.append(
            {
                "line_index": len(candidates),
                "mode_index": mode_index,
                "mode": mode["name"],
                "nodes": list(nodes),
                "edges": list(edge_ids),
                "edge_set": frozenset(edge_ids),
                "vehicle_capacity": float(mode["vehicle_capacity"]),
                "fixed_cost": fixed_cost,
                "operating_cost": operating_cost,
            }
        )

    for mode_index, mode in enumerate(modes):
        if len(candidates) >= candidate_limit or time.monotonic() >= enumeration_deadline:
            break

        terminals = sorted(set(int(v) for v in mode.get("terminals", [])))
        terminal_set = set(terminals)
        if len(terminals) < 2:
            continue

        mode_edge_ids = []
        for raw in mode.get("edge_indices", []):
            value = int(raw)
            if value in edge_by_id:
                mode_edge_ids.append(value)
            elif 0 <= value < len(edges):
                mode_edge_ids.append(int(edges[value]["id"]))

        adjacency = defaultdict(list)
        for eid in mode_edge_ids:
            e = edge_by_id[eid]
            u, v = map(int, e["endpoints"])
            adjacency[u].append((v, eid))
            adjacency[v].append((u, eid))

        for node in adjacency:
            adjacency[node].sort(key=lambda item: (item[0], item[1]))

        terminal_pairs = [
            (terminals[i], terminals[j])
            for i in range(len(terminals))
            for j in range(i + 1, len(terminals))
        ]
        if not terminal_pairs:
            continue

        # First add one minimum-hop path for every terminal pair. This gives
        # useful coverage even when exhaustive enumeration must be curtailed.
        for source, target in terminal_pairs:
            if len(candidates) >= candidate_limit:
                break
            queue = deque([source])
            parent = {source: None}
            depth = {source: 0}

            while queue and target not in parent:
                u = queue.popleft()
                if depth[u] >= max_length:
                    continue
                for v, eid in adjacency.get(u, []):
                    if v in parent:
                        continue
                    parent[v] = (u, eid)
                    depth[v] = depth[u] + 1
                    queue.append(v)
                    if v == target:
                        break

            if target in parent:
                rev_nodes = [target]
                rev_edges = []
                cur = target
                while cur != source:
                    prev, eid = parent[cur]
                    rev_edges.append(eid)
                    rev_nodes.append(prev)
                    cur = prev
                add_candidate(
                    mode_index,
                    mode,
                    list(reversed(rev_nodes)),
                    list(reversed(rev_edges)),
                )

        remaining_slots = max(0, candidate_limit - len(candidates))
        per_pair_limit = max(
            20,
            min(1500, remaining_slots // max(1, len(terminal_pairs))),
        )

        # Enumerate node-simple terminal-to-terminal paths in a balanced way.
        for source, target in terminal_pairs:
            if len(candidates) >= candidate_limit:
                break
            if time.monotonic() >= enumeration_deadline:
                break

            generated_for_pair = 0
            visited = {source}
            path_nodes = [source]
            path_edges = []

            def dfs(u):
                nonlocal generated_for_pair
                if generated_for_pair >= per_pair_limit:
                    return
                if len(candidates) >= candidate_limit:
                    return
                if time.monotonic() >= enumeration_deadline:
                    return
                if len(path_edges) >= max_length:
                    return

                for v, eid in adjacency.get(u, []):
                    if v in visited:
                        continue

                    visited.add(v)
                    path_nodes.append(v)
                    path_edges.append(eid)

                    if v == target:
                        before = len(candidates)
                        add_candidate(
                            mode_index, mode, path_nodes[:], path_edges[:]
                        )
                        if len(candidates) > before:
                            generated_for_pair += 1
                    elif len(path_edges) < max_length:
                        dfs(v)

                    path_edges.pop()
                    path_nodes.pop()
                    visited.remove(v)

                    if generated_for_pair >= per_pair_limit:
                        return
                    if len(candidates) >= candidate_limit:
                        return
                    if time.monotonic() >= enumeration_deadline:
                        return

            dfs(source)

    return candidates, edge_by_id, edge_position


def build_and_solve(instance, candidates, edge_by_id, start_time, time_limit, logger):
    network = instance["network"]
    edges = network["edges"]
    od_rows = [
        row for row in instance.get("od_matrix", [])
        if float(row.get("demand", 0.0)) > 0.0
    ]

    nodes = [int(n["id"]) for n in network["nodes"]]
    destinations = sorted(set(int(row["destination"]) for row in od_rows))

    demand_by_destination = {
        d: defaultdict(float) for d in destinations
    }
    for row in od_rows:
        o = int(row["origin"])
        d = int(row["destination"])
        demand_by_destination[d][o] += float(row["demand"])

    arcs = []
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    arc_by_key = {}

    for edge_pos, e in enumerate(edges):
        eid = int(e["id"])
        u, v = map(int, e["endpoints"])
        travel_time = float(e["traveling_time_seconds"])

        # Directed arc convention: 2*edge_id is endpoints[0] -> endpoints[1],
        # and 2*edge_id+1 is the reverse arc.
        a_forward = {
            "key": (eid, 0),
            "arc_id": 2 * eid,
            "edge_id": eid,
            "tail": u,
            "head": v,
            "time": travel_time,
        }
        a_reverse = {
            "key": (eid, 1),
            "arc_id": 2 * eid + 1,
            "edge_id": eid,
            "tail": v,
            "head": u,
            "time": travel_time,
        }

        for arc in (a_forward, a_reverse):
            arcs.append(arc)
            outgoing[arc["tail"]].append(arc["key"])
            incoming[arc["head"]].append(arc["key"])
            arc_by_key[arc["key"]] = arc

    lines_by_edge = defaultdict(list)
    for i, line in enumerate(candidates):
        for eid in line["edges"]:
            lines_by_edge[eid].append(i)

    model = gp.Model("multimodal_line_planning")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.Heuristics = 0.15

    y = {}
    frequency = {}
    F = float(instance["global_parameters"]["frequency_upper_bound_F"])

    for i, line in enumerate(candidates):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        frequency[i] = model.addVar(
            lb=0.0, ub=F, vtype=GRB.CONTINUOUS, name=f"f_{i}"
        )

    flow = {}
    for d in destinations:
        for arc in arcs:
            key = arc["key"]
            flow[d, key] = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"x_{d}_{key[0]}_{key[1]}",
            )

    model.update()

    for i in range(len(candidates)):
        model.addConstr(frequency[i] <= F * y[i], name=f"activate_{i}")

    for e in edges:
        eid = int(e["id"])
        capacity = float(e["edge_capacity"])
        model.addConstr(
            gp.quicksum(frequency[i] for i in lines_by_edge.get(eid, []))
            <= capacity,
            name=f"edge_frequency_{eid}",
        )

    for d in destinations:
        total_to_destination = sum(demand_by_destination[d].values())
        for node in nodes:
            rhs = demand_by_destination[d].get(node, 0.0)
            if node == d:
                rhs -= total_to_destination

            model.addConstr(
                gp.quicksum(flow[d, a] for a in outgoing.get(node, []))
                - gp.quicksum(flow[d, a] for a in incoming.get(node, []))
                == rhs,
                name=f"balance_{d}_{node}",
            )

    for arc in arcs:
        eid = arc["edge_id"]
        provided_capacity = gp.quicksum(
            candidates[i]["vehicle_capacity"] * frequency[i]
            for i in lines_by_edge.get(eid, [])
        )
        model.addConstr(
            gp.quicksum(flow[d, arc["key"]] for d in destinations)
            <= provided_capacity,
            name=f"passenger_capacity_{eid}_{arc['key'][1]}",
        )

    line_cost = gp.quicksum(
        candidates[i]["fixed_cost"] * y[i]
        + candidates[i]["operating_cost"] * frequency[i]
        for i in range(len(candidates))
    )
    passenger_time = gp.quicksum(
        arc["time"] * flow[d, arc["key"]]
        for d in destinations
        for arc in arcs
    )

    weight = float(instance["global_parameters"]["lambda"])
    model.setObjective(
        weight * line_cost + (1.0 - weight) * passenger_time,
        GRB.MINIMIZE,
    )

    remaining = max(0.1, time_limit - (time.monotonic() - start_time))
    model.Params.TimeLimit = remaining

    best_callback_value = [math.inf]

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                value = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if value + 1e-7 < best_callback_value[0]:
                    best_callback_value[0] = value
                    if logger:
                        logger.log(float(value))
            except Exception:
                pass

    model.optimize(callback)

    relaxed = False
    solution_model = model

    if model.SolCount == 0:
        remaining = time_limit - (time.monotonic() - start_time)
        if remaining > 0.15:
            relaxed_model = model.relax()
            relaxed_model.Params.OutputFlag = 0
            relaxed_model.Params.Seed = 0
            relaxed_model.Params.MIPGap = 1e-4
            relaxed_model.Params.NumericFocus = 0
            relaxed_model.Params.Threads = 1
            relaxed_model.Params.TimeLimit = max(0.1, remaining)
            relaxed_model.optimize()
            if relaxed_model.SolCount > 0:
                solution_model = relaxed_model
                relaxed = True

    return {
        "model": solution_model,
        "original_model": model,
        "relaxed": relaxed,
        "candidates": candidates,
        "destinations": destinations,
        "demand_by_destination": demand_by_destination,
        "arcs": arcs,
        "arc_by_key": arc_by_key,
        "outgoing": outgoing,
        "y": y,
        "frequency": frequency,
        "flow": flow,
        "weight": weight,
    }


def extract_solution(instance, solved):
    model = solved["model"]
    if model.SolCount == 0:
        return {
            "objective_value": 0.0,
            "active_lines": [],
            "active_passenger_paths": [],
        }

    relaxed = solved["relaxed"]
    candidates = solved["candidates"]
    destinations = solved["destinations"]
    demand_by_destination = solved["demand_by_destination"]
    arcs = solved["arcs"]
    arc_by_key = solved["arc_by_key"]
    outgoing = solved["outgoing"]
    weight = solved["weight"]

    def value_by_name(name):
        var = model.getVarByName(name)
        return 0.0 if var is None else float(var.X)

    frequencies = []
    activations = []
    for i in range(len(candidates)):
        frequencies.append(max(0.0, value_by_name(f"f_{i}")))
        activations.append(max(0.0, value_by_name(f"y_{i}")))

    active_lines = []
    active_indices = []

    for i, line in enumerate(candidates):
        f = frequencies[i]
        if relaxed:
            active = f > 1e-9
        else:
            active = activations[i] > 0.5 or f > 1e-9

        if active:
            active_indices.append(i)
            active_lines.append(
                {
                    "line_index": int(line["line_index"]),
                    "mode": line["mode"],
                    "nodes": [int(v) for v in line["nodes"]],
                    "edges": [int(e) for e in line["edges"]],
                    "frequency": float(f),
                }
            )

    active_passenger_paths = []

    for d in destinations:
        residual = {}
        for arc in arcs:
            eid, direction = arc["key"]
            val = value_by_name(f"x_{d}_{eid}_{direction}")
            residual[arc["key"]] = max(0.0, val)

        for origin in sorted(demand_by_destination[d]):
            required = float(demand_by_destination[d][origin])
            if required <= 0.0:
                continue

            if origin == d:
                # A zero-edge path has zero travel time. Such rows are unusual,
                # but retaining an empty arc sequence is the natural witness.
                active_passenger_paths.append(
                    {
                        "origin": int(origin),
                        "destination": int(d),
                        "arcs": [],
                        "flow": float(required),
                    }
                )
                continue

            remaining = required
            produced_indices = []

            while remaining > 1e-8:
                queue = deque([origin])
                parent_node = {origin: None}
                parent_arc = {}

                while queue and d not in parent_node:
                    u = queue.popleft()
                    for key in outgoing.get(u, []):
                        if residual.get(key, 0.0) <= 1e-11:
                            continue
                        v = arc_by_key[key]["head"]
                        if v in parent_node:
                            continue
                        parent_node[v] = u
                        parent_arc[v] = key
                        queue.append(v)
                        if v == d:
                            break

                if d not in parent_node:
                    break

                path_keys = []
                cur = d
                while cur != origin:
                    key = parent_arc[cur]
                    path_keys.append(key)
                    cur = parent_node[cur]
                path_keys.reverse()

                amount = min(
                    remaining,
                    min(residual[key] for key in path_keys),
                )
                if amount <= 1e-12:
                    break

                for key in path_keys:
                    residual[key] = max(0.0, residual[key] - amount)

                entry = {
                    "origin": int(origin),
                    "destination": int(d),
                    "arcs": [
                        int(arc_by_key[key]["arc_id"]) for key in path_keys
                    ],
                    "flow": float(amount),
                }
                active_passenger_paths.append(entry)
                produced_indices.append(len(active_passenger_paths) - 1)
                remaining -= amount

            # Correct only tiny numerical flow-balance residuals.
            if produced_indices and abs(remaining) <= 1e-5:
                idx = produced_indices[-1]
                active_passenger_paths[idx]["flow"] = float(
                    active_passenger_paths[idx]["flow"] + remaining
                )
                remaining = 0.0

    edge_time = {
        int(e["id"]): float(e["traveling_time_seconds"])
        for e in instance["network"]["edges"]
    }

    total_line_cost = 0.0
    for i in active_indices:
        line = candidates[i]
        total_line_cost += line["fixed_cost"]
        total_line_cost += line["operating_cost"] * frequencies[i]

    total_passenger_time = 0.0
    for path in active_passenger_paths:
        path_time = 0.0
        for arc_id in path["arcs"]:
            eid = int(arc_id) // 2
            path_time += edge_time[eid]
        total_passenger_time += path_time * float(path["flow"])

    objective = (
        weight * total_line_cost
        + (1.0 - weight) * total_passenger_time
    )

    return {
        "objective_value": float(objective),
        "active_lines": active_lines,
        "active_passenger_paths": active_passenger_paths,
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    instance = load_instance(args.instance_path)

    candidates, edge_by_id, _ = enumerate_candidate_lines(
        instance, start_time, args.time_limit
    )

    solved = build_and_solve(
        instance,
        candidates,
        edge_by_id,
        start_time,
        args.time_limit,
        logger,
    )

    solution = extract_solution(instance, solved)
    write_solution(args.solution_path, solution)

    if logger:
        logger.log_solution(solution["objective_value"], solution)


if __name__ == "__main__":
    main()