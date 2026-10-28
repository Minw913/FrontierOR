import argparse
import ast
import bz2
import gzip
import json
import lzma
import random
import re
import sys
import time
from typing import List, Set

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Maximum Weighted Clique Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path for JSONL incumbent logs")
    return parser.parse_args()


def robust_load_json(path: str):
    # Read raw bytes
    with open(path, "rb") as f:
        raw = f.read()

    # Decompress if needed (by magic bytes)
    try:
        if raw.startswith(b"\x1f\x8b"):  # gzip
            raw = gzip.decompress(raw)
        elif raw.startswith(b"BZh"):  # bzip2
            raw = bz2.decompress(raw)
        elif raw.startswith(b"\xfd7zXZ\x00"):  # xz/lzma
            raw = lzma.decompress(raw)
    except Exception:
        # If decompression fails, keep original bytes
        pass

    # Decode text
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        raise ValueError("Could not decode instance file as text.")

    text = text.strip()
    if not text:
        raise ValueError("Instance file is empty.")

    # 1) Strict JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Remove trailing commas (common non-strict JSON issue)
    try:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(cleaned)
    except Exception:
        pass

    # 3) Try parsing the largest {...} region
    try:
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and i < j:
            candidate = text[i:j + 1]
            return json.loads(candidate)
    except Exception:
        pass

    # 4) Python literal fallback
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 5) JSONL fallback: first valid object line
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    raise ValueError("Unable to parse instance file as JSON-like data.")


def is_clique(vertices: List[int], adj: List[Set[int]]) -> bool:
    for i, u in enumerate(vertices):
        nu = adj[u]
        for v in vertices[i + 1:]:
            if v not in nu:
                return False
    return True


def greedy_repair(vertices: List[int], weights: List[float], adj: List[Set[int]]) -> List[int]:
    order = sorted(vertices, key=lambda v: weights[v], reverse=True)
    clique = []
    for v in order:
        ok = True
        for u in clique:
            if v not in adj[u]:
                ok = False
                break
        if ok:
            clique.append(v)
    return clique


def greedy_construct(order: List[int], adj: List[Set[int]]) -> List[int]:
    clique = []
    for v in order:
        ok = True
        for u in clique:
            if v not in adj[u]:
                ok = False
                break
        if ok:
            clique.append(v)
    return clique


def solve(args):
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    data = robust_load_json(args.instance_path)
    graph = data["graph"]

    vertices_ids = graph["vertices"]
    n = len(vertices_ids)

    edges = graph.get("edges", [])
    w_raw = data["vertex_weights"]

    # ID <-> internal index
    id_to_idx = {vid: i for i, vid in enumerate(vertices_ids)}

    # Robust weight handling
    weights = [1.0] * n
    if isinstance(w_raw, list):
        if len(w_raw) == n:
            # Prefer order aligned with vertices list
            for i in range(n):
                try:
                    weights[i] = float(w_raw[i])
                except Exception:
                    weights[i] = 1.0
        elif all(isinstance(vid, int) and 0 <= vid < len(w_raw) for vid in vertices_ids):
            # If weights indexed by vertex id
            for i, vid in enumerate(vertices_ids):
                try:
                    weights[i] = float(w_raw[vid])
                except Exception:
                    weights[i] = 1.0

    # Build adjacency
    adj = [set() for _ in range(n)]
    for e in edges:
        if not isinstance(e, list) or len(e) != 2:
            continue
        u_id, v_id = e
        if u_id not in id_to_idx or v_id not in id_to_idx:
            continue
        u, v = id_to_idx[u_id], id_to_idx[v_id]
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    best_obj = 0.0
    best_set: List[int] = []

    def try_update(sol: List[int]):
        nonlocal best_obj, best_set
        if not sol:
            return
        if not is_clique(sol, adj):
            sol = greedy_repair(sol, weights, adj)
        obj = float(sum(weights[v] for v in sol))
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_set = list(sol)
            if logger:
                logger.log(best_obj)

    # Heuristic phase
    elapsed = time.time() - start_time
    remaining = max(0.0, args.time_limit - elapsed)
    heur_budget = min(2.0, 0.2 * args.time_limit, 0.5 * remaining)

    deg = [len(adj[i]) for i in range(n)]
    h_start = time.time()

    order1 = sorted(range(n), key=lambda i: (weights[i], deg[i]), reverse=True)
    try_update(greedy_construct(order1, adj))

    order2 = sorted(range(n), key=lambda i: (weights[i] * (deg[i] + 1.0)), reverse=True)
    try_update(greedy_construct(order2, adj))

    rng = random.Random(0)
    while time.time() - h_start < heur_budget:
        scored = []
        for i in range(n):
            scored.append((weights[i] + 0.25 * deg[i] + rng.random(), i))
        scored.sort(reverse=True)
        order = [i for _, i in scored]
        try_update(greedy_construct(order, adj))

    # Exact MIP with lazy non-edge cuts
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.01:
        model = gp.Model("max_weight_clique")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.0, remaining)
        model.Params.LazyConstraints = 1
        model.Params.MIPFocus = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

        if best_set:
            s = set(best_set)
            for i in range(n):
                x[i].Start = 1.0 if i in s else 0.0

        state = {"best_obj": best_obj, "best_set": list(best_set)}

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                vals = m.cbGetSolution(x)
                selected = [i for i in range(n) if vals[i] > 0.5]

                # Separate violated non-edge constraints
                added = 0
                max_add = 2000
                k = len(selected)
                for a in range(k):
                    u = selected[a]
                    nu = adj[u]
                    for b in range(a + 1, k):
                        v = selected[b]
                        if v not in nu:
                            m.cbLazy(x[u] + x[v] <= 1)
                            added += 1
                            if added >= max_add:
                                break
                    if added >= max_add:
                        break

                if added == 0:
                    obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                    if obj > state["best_obj"] + 1e-9:
                        state["best_obj"] = obj
                        state["best_set"] = selected
                        if logger:
                            logger.log(obj)

        model.optimize(cb)

        if state["best_obj"] > best_obj + 1e-9:
            best_obj = state["best_obj"]
            best_set = state["best_set"]

        if model.SolCount > 0:
            sol = [i for i in range(n) if x[i].X > 0.5]
            if not is_clique(sol, adj):
                sol = greedy_repair(sol, weights, adj)
            obj = float(sum(weights[i] for i in sol))
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_set = sol
                if logger:
                    logger.log(best_obj)

    # Final safety
    if best_set and not is_clique(best_set, adj):
        best_set = greedy_repair(best_set, weights, adj)
        best_obj = float(sum(weights[i] for i in best_set))

    clique_vertices = sorted(vertices_ids[i] for i in best_set)
    out = {
        "objective_value": float(best_obj),
        "clique_vertices": clique_vertices,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


def main():
    args = parse_args()
    try:
        solve(args)
    except Exception as e:
        # Last-resort fail-safe: never crash evaluator
        try:
            with open(args.solution_path, "w") as f:
                json.dump({"objective_value": 0.0, "clique_vertices": []}, f)
        except Exception:
            pass
        print(f"[ERROR] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()