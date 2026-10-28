import argparse
import ast
import gzip
import json
import random
import re
import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def robust_load_json(path: str):
    # Read raw bytes first (allows gzip detection / BOM handling).
    with open(path, "rb") as f:
        raw = f.read()

    # If gzipped content by magic bytes, decompress.
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        raw = gzip.decompress(raw)

    text = raw.decode("utf-8-sig", errors="replace").strip()

    # Try strict JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # If file has extra text around JSON, try extracting first {...} block.
    try:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l : r + 1])
    except Exception:
        pass

    # Relaxed parse fallback (single quotes, python dict syntax, etc.).
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    # Last-resort cleanup: remove trailing commas before } or ]
    try:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Failed to parse instance file '{path}': {e}")


def normalize_instance(inst):
    if not isinstance(inst, dict):
        raise ValueError("Instance root must be a JSON object/dict.")

    # Accept minor key variations.
    n = inst.get("n", inst.get("N"))
    A = inst.get("A", inst.get("a"))
    B = inst.get("B", inst.get("b"))
    C = inst.get("C", inst.get("c"))

    if n is None:
        # Infer n if possible
        if A is not None:
            n = len(A)
        elif B is not None:
            n = len(B)
        elif C is not None:
            n = len(C)
        else:
            raise ValueError("Cannot determine n from instance.")

    n = int(n)

    if A is None or B is None or C is None:
        raise ValueError("Instance must contain matrices A, B, C.")

    if len(A) != n or len(B) != n or len(C) != n:
        raise ValueError("Matrix row dimensions do not match n.")

    # Convert matrix entries to float (safe for int too)
    A2 = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    B2 = [[float(B[i][j]) for j in range(n)] for i in range(n)]
    C2 = [[float(C[i][j]) for j in range(n)] for i in range(n)]
    return n, A2, B2, C2


def evaluate_perm(perm: List[int], A, B, C) -> Tuple[float, float, float, float]:
    sa = sb = sc = 0.0
    for i, j in enumerate(perm):
        sa += A[i][j]
        sb += B[i][j]
        sc += C[i][j]
    return sa, sb, sc, sa * sb + sc


def solve_lap(cost, time_limit: float) -> List[int]:
    n = len(cost)
    m = gp.Model("lap")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = time_limit

    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    m.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)))
    m.addConstrs((gp.quicksum(x[i, j] for i in range(n)) == 1 for j in range(n)))
    m.setObjective(gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount <= 0:
        perm = list(range(n))
        random.shuffle(perm)
        return perm

    perm = [-1] * n
    used = set()
    for i in range(n):
        j = max(range(n), key=lambda jj: x[i, jj].X)
        if j not in used:
            perm[i] = j
            used.add(j)

    missing = [j for j in range(n) if j not in used]
    for i in range(n):
        if perm[i] == -1:
            perm[i] = missing.pop()
    return perm


def local_search(perm, A, B, C, end_time):
    n = len(perm)
    sa, sb, sc, best_obj = evaluate_perm(perm, A, B, C)

    improved = True
    while improved and time.time() < end_time:
        improved = False
        for i in range(n - 1):
            if time.time() >= end_time:
                break
            j1 = perm[i]
            for k in range(i + 1, n):
                j2 = perm[k]
                dA = A[i][j2] + A[k][j1] - A[i][j1] - A[k][j2]
                dB = B[i][j2] + B[k][j1] - B[i][j1] - B[k][j2]
                dC = C[i][j2] + C[k][j1] - C[i][j1] - C[k][j2]
                new_obj = (sa + dA) * (sb + dB) + (sc + dC)
                if new_obj + 1e-12 < best_obj:
                    perm[i], perm[k] = perm[k], perm[i]
                    sa += dA
                    sb += dB
                    sc += dC
                    best_obj = new_obj
                    improved = True
                    break
            if improved:
                break

    return perm, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    t0 = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        inst = robust_load_json(args.instance_path)
        n, A, B, C = normalize_instance(inst)
    except Exception:
        # Always produce valid output, even if instance parsing fails.
        out = {"objective_value": float("inf"), "assignment": []}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    def rem():
        return args.time_limit - (time.time() - t0)

    # Initial random feasible assignment
    best_perm = list(range(n))
    random.shuffle(best_perm)
    _, _, _, best_obj = evaluate_perm(best_perm, A, B, C)
    if logger:
        logger.log(float(best_obj))

    if rem() > 0.05:
        # Heuristic seeds
        try:
            tl = max(0.01, min(1.0, 0.2 * rem()))
            p_c = solve_lap(C, tl)
            _, _, _, obj = evaluate_perm(p_c, A, B, C)
            if obj < best_obj:
                best_perm, best_obj = p_c, obj
                if logger:
                    logger.log(float(best_obj))

            # One linearization round
            sa, sb, _, _ = evaluate_perm(best_perm, A, B, C)
            lin = [[sb * A[i][j] + sa * B[i][j] + C[i][j] for j in range(n)] for i in range(n)]
            tl = max(0.01, min(1.0, 0.15 * rem()))
            p_lin = solve_lap(lin, tl)
            _, _, _, obj2 = evaluate_perm(p_lin, A, B, C)
            if obj2 < best_obj:
                best_perm, best_obj = p_lin, obj2
                if logger:
                    logger.log(float(best_obj))
        except Exception:
            pass

    # Local search for remaining time (keep small buffer)
    end_ls = t0 + max(0.0, args.time_limit - 0.05)
    if time.time() < end_ls:
        best_perm, ls_obj = local_search(best_perm, A, B, C, end_ls)
        if ls_obj < best_obj:
            best_obj = ls_obj
            if logger:
                logger.log(float(best_obj))

    # Final exact objective recomputation
    _, _, _, best_obj = evaluate_perm(best_perm, A, B, C)

    out = {
        "objective_value": float(best_obj),
        "assignment": [int(x) for x in best_perm],
    }
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()