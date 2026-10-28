import argparse
import json
import time
import random
import numpy as np
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.7

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    m = int(data["m"])
    C = int(data["capacity"])
    w = np.array(data["weights"], dtype=np.int64)
    v = np.array(data["values"], dtype=np.float64)
    P = np.array(data["pairwise_values"], dtype=np.float64)
    if P.shape != (n, n):
        P = P.reshape(n, n)
    np.fill_diagonal(P, 0.0)

    rng = random.Random(0)
    np.random.seed(0)

    def exact_obj(assign_vec):
        tot = 0.0
        for k in range(m):
            mem = np.where(assign_vec == k)[0]
            if len(mem):
                tot += v[mem].sum() + 0.5 * P[np.ix_(mem, mem)].sum()
        return tot

    def sol_dict(objv, assign_vec):
        return {
            "objective_value": float(objv),
            "assignment": {str(i): int(assign_vec[i]) for i in range(n) if assign_vec[i] >= 0},
        }

    class State:
        def __init__(self):
            self.assign = np.full(n, -1, dtype=np.int64)
            self.load = np.zeros(m, dtype=np.int64)
            # gain[i,k] = v[i] + sum_{j assigned to k} P[i,j]
            self.gain = np.tile(v[:, None], (1, m))
            self.obj = 0.0

        def add(self, i, k):
            self.obj += self.gain[i, k]
            self.gain[:, k] += P[:, i]
            self.assign[i] = k
            self.load[k] += w[i]

        def remove(self, i):
            k = self.assign[i]
            self.gain[:, k] -= P[:, i]
            self.obj -= self.gain[i, k]
            self.assign[i] = -1
            self.load[k] -= w[i]

    def build_state(assign_vec):
        st = State()
        for i in range(n):
            if assign_vec[i] >= 0:
                st.add(i, int(assign_vec[i]))
        return st

    def greedy_fill(st):
        while time.time() < deadline:
            U = np.where(st.assign < 0)[0]
            if len(U) == 0:
                break
            feas = st.load[None, :] + w[U, None] <= C
            d = np.where(feas, st.gain[U], -np.inf)
            idx = int(np.argmax(d))
            best = d.flat[idx]
            if best <= 1e-9:
                break
            i = int(U[idx // m])
            k = idx % m
            st.add(i, k)

    def apply_move(st, move):
        typ = move[0]
        if typ == "add":
            st.add(move[1], move[2])
        elif typ == "rem":
            st.remove(move[1])
        elif typ == "rel":
            st.remove(move[1])
            st.add(move[1], move[2])
        elif typ == "swap":
            i, j = move[1], move[2]
            k1, k2 = int(st.assign[i]), int(st.assign[j])
            st.remove(i)
            st.remove(j)
            st.add(i, k2)
            st.add(j, k1)
        elif typ == "sau":
            i, j = move[1], move[2]  # i assigned, j unassigned
            k = int(st.assign[i])
            st.remove(i)
            st.add(j, k)

    def local_search(st):
        while time.time() < deadline:
            best_delta = 1e-9
            best_move = None
            A = np.where(st.assign >= 0)[0]
            U = np.where(st.assign < 0)[0]

            # insertion of unassigned items
            if len(U):
                feas = st.load[None, :] + w[U, None] <= C
                d = np.where(feas, st.gain[U], -np.inf)
                idx = int(np.argmax(d))
                if d.flat[idx] > best_delta:
                    best_delta = float(d.flat[idx])
                    best_move = ("add", int(U[idx // m]), idx % m)

            if len(A):
                ga = st.assign[A]
                cur = st.gain[A, ga]  # current contribution of each assigned item

                # removal
                jmin = int(np.argmin(cur))
                if -cur[jmin] > best_delta:
                    best_delta = float(-cur[jmin])
                    best_move = ("rem", int(A[jmin]))

                # relocation
                feas = st.load[None, :] + w[A, None] <= C
                d = np.where(feas, st.gain[A], -np.inf) - cur[:, None]
                d[np.arange(len(A)), ga] = -np.inf
                idx = int(np.argmax(d))
                if d.flat[idx] > best_delta:
                    best_delta = float(d.flat[idx])
                    best_move = ("rel", int(A[idx // m]), idx % m)

                slack = C - st.load[ga] + w[A]  # room in item's knapsack if it leaves

                # swap between two assigned items
                if len(A) > 1:
                    cross = st.gain[A][:, ga]  # cross[i,j] = gain[A[i], knap(A[j])]
                    D = cross + cross.T - cur[:, None] - cur[None, :] - 2.0 * P[np.ix_(A, A)]
                    feas2 = (w[A][None, :] <= slack[:, None]) & (w[A][:, None] <= slack[None, :])
                    feas2 &= ga[:, None] != ga[None, :]
                    D = np.where(feas2, D, -np.inf)
                    idx = int(np.argmax(D))
                    if D.flat[idx] > best_delta:
                        ii, jj = divmod(idx, len(A))
                        best_delta = float(D.flat[idx])
                        best_move = ("swap", int(A[ii]), int(A[jj]))

                # swap assigned item with unassigned item
                if len(U):
                    gu = st.gain[U][:, ga]  # gu[j,i] = gain[U[j], knap(A[i])]
                    D2 = gu.T - P[np.ix_(A, U)] - cur[:, None]
                    feasu = w[U][None, :] <= slack[:, None]
                    D2 = np.where(feasu, D2, -np.inf)
                    idx = int(np.argmax(D2))
                    if D2.flat[idx] > best_delta:
                        ii, jj = divmod(idx, len(U))
                        best_delta = float(D2.flat[idx])
                        best_move = ("sau", int(A[ii]), int(U[jj]))

            if best_move is None:
                break
            apply_move(st, best_move)

    # ---------- initial solution ----------
    st = State()
    greedy_fill(st)
    local_search(st)

    best_assign = st.assign.copy()
    best_obj = exact_obj(best_assign)
    if logger:
        logger.log_solution(best_obj, sol_dict(best_obj, best_assign))

    # ---------- iterated local search ----------
    strength = 0.12
    no_improve = 0
    while time.time() < deadline:
        # perturb current state
        A = list(np.where(st.assign >= 0)[0])
        rng.shuffle(A)
        r = max(1, int(strength * max(1, len(A)))) if A else 1
        for i in A[:r]:
            st.remove(int(i))
        # random insertions to diversify
        U = list(np.where(st.assign < 0)[0])
        rng.shuffle(U)
        cnt = 0
        for i in U:
            if cnt >= max(1, r // 2):
                break
            ks = [k for k in range(m) if st.load[k] + w[i] <= C]
            if ks:
                st.add(int(i), rng.choice(ks))
                cnt += 1

        greedy_fill(st)
        local_search(st)

        cur_obj = exact_obj(st.assign)
        if cur_obj > best_obj + 1e-9:
            best_obj = cur_obj
            best_assign = st.assign.copy()
            no_improve = 0
            strength = 0.12
            if logger:
                logger.log_solution(best_obj, sol_dict(best_obj, best_assign))
        else:
            no_improve += 1
            if no_improve % 5 == 0:
                strength = min(0.5, strength + 0.05)
            # restart from best solution
            if time.time() < deadline:
                st = build_state(best_assign)

    # ---------- output ----------
    final_obj = exact_obj(best_assign)
    solution = sol_dict(final_obj, best_assign)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()