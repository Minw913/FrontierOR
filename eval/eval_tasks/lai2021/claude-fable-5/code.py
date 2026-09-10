import argparse
import json
import time
import random
import numpy as np

from solution_logger import SolutionLogger


def build_distance_matrix(N, upper):
    D = np.zeros((N, N), dtype=np.float64)
    idx = 0
    for i in range(N - 1):
        cnt = N - 1 - i
        row = np.asarray(upper[idx:idx + cnt], dtype=np.float64)
        D[i, i + 1:] = row
        D[i + 1:, i] = row
        idx += cnt
    return D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.6

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    N = int(inst["N"])
    m = int(inst["m"])
    caps = inst["group_capacities"]
    L = np.array([int(c["L_g"]) for c in caps], dtype=np.int64)
    U = np.array([int(c["U_g"]) for c in caps], dtype=np.int64)
    upper = inst.get("distances_upper_triangular", [])

    D = build_distance_matrix(N, upper)

    rng = np.random.default_rng(0)
    random.seed(0)

    # ---------- Initial feasible solution ----------
    grp = np.full(N, -1, dtype=np.int64)
    sizes = np.zeros(m, dtype=np.int64)
    order = list(rng.permutation(N))
    pos = 0
    # satisfy lower bounds
    for g in range(m):
        need = L[g]
        while need > 0 and pos < N:
            grp[order[pos]] = g
            sizes[g] += 1
            pos += 1
            need -= 1
    # assign remaining
    g = 0
    while pos < N:
        # find group with slack
        placed = False
        for _ in range(m):
            if sizes[g] < U[g]:
                grp[order[pos]] = g
                sizes[g] += 1
                pos += 1
                placed = True
                break
            g = (g + 1) % m
        if not placed:
            # no slack anywhere (shouldn't happen if feasible); force into group 0
            grp[order[pos]] = 0
            sizes[0] += 1
            pos += 1
        g = (g + 1) % m

    # ---------- contribution matrix ----------
    def compute_contrib(grp_arr):
        C = np.zeros((N, m), dtype=np.float64)
        for gg in range(m):
            members = np.where(grp_arr == gg)[0]
            if members.size > 0:
                C[:, gg] = D[:, members].sum(axis=1)
        return C

    contrib = compute_contrib(grp)
    ar = np.arange(N)

    def current_obj(grp_arr, C):
        return 0.5 * float(C[ar, grp_arr].sum())

    def exact_obj(grp_arr):
        s = 0.0
        for gg in range(m):
            idx = np.where(grp_arr == gg)[0]
            if idx.size > 1:
                s += D[np.ix_(idx, idx)].sum() / 2.0
        return float(s)

    obj = current_obj(grp, contrib)

    def build_solution(grp_arr, obj_val):
        assignment = {str(i): int(grp_arr[i]) for i in range(N)}
        groups = {}
        for gg in range(m):
            groups[str(gg)] = sorted(int(x) for x in np.where(grp_arr == gg)[0])
        return {
            "objective_value": obj_val,
            "assignment": assignment,
            "groups": groups,
        }

    best_grp = grp.copy()
    best_obj = obj
    if logger:
        logger.log_solution(exact_obj(best_grp), build_solution(best_grp, exact_obj(best_grp)))

    # ---------- Local search ----------
    NEG_INF = -np.inf

    def apply_move(i, g1, g2, grp_arr, sz, C):
        C[:, g1] -= D[i]
        C[:, g2] += D[i]
        grp_arr[i] = g2
        sz[g1] -= 1
        sz[g2] += 1

    def local_search(grp_arr, sz, C, obj_val):
        improved = True
        check_counter = 0
        while improved:
            improved = False
            perm = rng.permutation(N)
            for i in perm:
                check_counter += 1
                if (check_counter & 63) == 0 and time.time() >= deadline:
                    return obj_val, True
                gi = grp_arr[i]
                ci = C[i]
                # swap gains with every j in another group
                a = ci[grp_arr] - ci[gi]
                b = C[ar, gi] - C[ar, grp_arr]
                gain = a + b - 2.0 * D[i]
                same = grp_arr == gi
                gain[same] = NEG_INF
                if N > 1:
                    jbest = int(np.argmax(gain))
                    gswap = gain[jbest]
                else:
                    jbest = -1
                    gswap = NEG_INF
                # move gains
                gmove = NEG_INF
                hbest = -1
                if sz[gi] > L[gi]:
                    mv = ci - ci[gi]
                    feas = sz < U
                    feas[gi] = False
                    if feas.any():
                        mv2 = np.where(feas, mv, NEG_INF)
                        hbest = int(np.argmax(mv2))
                        gmove = mv2[hbest]
                bestg = gswap if gswap >= gmove else gmove
                if bestg > 1e-9:
                    if gswap >= gmove:
                        gj = grp_arr[jbest]
                        apply_move(i, gi, gj, grp_arr, sz, C)
                        apply_move(jbest, gj, gi, grp_arr, sz, C)
                        obj_val += gswap
                    else:
                        apply_move(i, gi, hbest, grp_arr, sz, C)
                        obj_val += gmove
                    improved = True
        return obj_val, False

    def perturb(grp_arr, sz, C, obj_val, strength):
        for _ in range(strength):
            i = int(rng.integers(N))
            gi = grp_arr[i]
            # try a random swap partner in a different group
            others = np.where(grp_arr != gi)[0]
            if others.size == 0:
                break
            j = int(others[rng.integers(others.size)])
            gj = grp_arr[j]
            gain = (C[i, gj] - C[i, gi]) + (C[j, gi] - C[j, gj]) - 2.0 * D[i, j]
            apply_move(i, gi, gj, grp_arr, sz, C)
            apply_move(j, gj, gi, grp_arr, sz, C)
            obj_val += gain
        return obj_val

    # Initial local search
    obj, timed_out = local_search(grp, sizes, contrib, obj)
    if obj > best_obj + 1e-9:
        best_obj = obj
        best_grp = grp.copy()
        eo = exact_obj(best_grp)
        if logger:
            logger.log_solution(eo, build_solution(best_grp, eo))

    # ---------- Iterated local search ----------
    if m > 1 and N > 1:
        strength = max(2, N // 12)
        no_improve = 0
        while time.time() < deadline:
            # restart from best
            grp = best_grp.copy()
            sizes = np.zeros(m, dtype=np.int64)
            for gg in range(m):
                sizes[gg] = int((grp == gg).sum())
            contrib = compute_contrib(grp)
            obj = current_obj(grp, contrib)

            k = max(2, min(N // 2, strength + int(rng.integers(0, max(1, strength)))))
            obj = perturb(grp, sizes, contrib, obj, k)
            obj, timed_out = local_search(grp, sizes, contrib, obj)

            if obj > best_obj + 1e-9:
                best_obj = obj
                best_grp = grp.copy()
                no_improve = 0
                strength = max(2, N // 12)
                eo = exact_obj(best_grp)
                if logger:
                    logger.log_solution(eo, build_solution(best_grp, eo))
            else:
                no_improve += 1
                if no_improve % 10 == 0:
                    strength = min(max(2, N // 3), strength + max(1, N // 20))
            if timed_out:
                break

    # ---------- Output ----------
    final_obj = exact_obj(best_grp)
    solution = build_solution(best_grp, final_obj)
    if logger:
        logger.log_solution(final_obj, solution)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()