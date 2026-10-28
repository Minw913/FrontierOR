import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional


class SafeLogger:
    """Robust logger wrapper that never raises."""
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._inner = None

        if not log_path:
            return

        # Try to use the required solution_logger, but never fail if environment
        # has DBus/system-bus restrictions.
        try:
            from solution_logger import SolutionLogger  # type: ignore
            try:
                self._inner = SolutionLogger(log_path, sense="minimize")
            except Exception:
                self._inner = None
        except Exception:
            self._inner = None

    def log(self, objective_value: float):
        if self._inner is not None:
            try:
                self._inner.log(float(objective_value))
            except Exception:
                pass

    def log_solution(self, objective_value: float, solution_dict: dict):
        if self._inner is not None:
            try:
                self._inner.log_solution(float(objective_value), solution_dict)
            except Exception:
                pass


class Solver:
    def __init__(self, inst: dict, time_limit: int, logger: SafeLogger):
        self.inst = inst
        self.time_limit = max(1, int(time_limit))
        self.logger = logger

        self.start_time = time.time()
        self.n = int(inst["num_customers"])
        self.K = int(inst["num_vehicles"])
        self.Q = int(inst["vehicle_capacity"])
        self.end = self.n + 1

        self.dist = inst["distance_matrix"]
        self.rho = float(inst.get("inconvenience_cost_function", {}).get("rho", 1.0))

        depot = inst["depot"]
        depot_end = inst["depot_end"]
        customers = inst["customers"]
        global_service = float(inst.get("service_time", 0))

        self.idx_to_id = {i: int(customers[i - 1]["id"]) for i in range(1, self.n + 1)}
        self.dem = {0: 0, self.end: 0}
        self.e = {0: float(depot["time_window"][0]), self.end: float(depot_end["time_window"][0])}
        self.l = {0: float(depot["time_window"][1]), self.end: float(depot_end["time_window"][1])}
        self.s = {0: 0.0, self.end: 0.0}
        self.p = {}

        for i in range(1, self.n + 1):
            c = customers[i - 1]
            self.dem[i] = int(c["demand"])
            self.e[i] = float(c["time_window"][0])
            self.l[i] = float(c["time_window"][1])
            self.s[i] = float(c.get("service_time", global_service))
            self.p[i] = float(c["preferred_time"])

        self.cache: Dict[Tuple[int, ...], Tuple[bool, float, Dict[int, float], int]] = {}

    def timeout(self) -> bool:
        return (time.time() - self.start_time) >= self.time_limit

    def route_load(self, seq: List[int]) -> int:
        return sum(self.dem[i] for i in seq)

    def eval_route(self, seq: List[int]) -> Tuple[bool, float, Dict[int, float], int]:
        key = tuple(seq)
        if key in self.cache:
            return self.cache[key]

        load = self.route_load(seq)
        if load > self.Q:
            ans = (False, float("inf"), {}, load)
            self.cache[key] = ans
            return ans

        m = len(seq)
        if m == 0:
            ans = (True, 0.0, {}, 0)
            self.cache[key] = ans
            return ans

        E = [0.0] * m
        t = self.e[0]
        prev = 0
        for k, c in enumerate(seq):
            t = max(self.e[c], t + self.s[prev] + float(self.dist[prev][c]))
            if t > self.l[c] + 1e-9:
                ans = (False, float("inf"), {}, load)
                self.cache[key] = ans
                return ans
            E[k] = t
            prev = c

        if E[-1] + self.s[seq[-1]] + float(self.dist[seq[-1]][self.end]) > self.l[self.end] + 1e-9:
            ans = (False, float("inf"), {}, load)
            self.cache[key] = ans
            return ans

        L = [0.0] * m
        L[-1] = min(self.l[seq[-1]], self.l[self.end] - self.s[seq[-1]] - float(self.dist[seq[-1]][self.end]))
        if E[-1] > L[-1] + 1e-9:
            ans = (False, float("inf"), {}, load)
            self.cache[key] = ans
            return ans

        for k in range(m - 2, -1, -1):
            c = seq[k]
            nxt = seq[k + 1]
            L[k] = min(self.l[c], L[k + 1] - self.s[c] - float(self.dist[c][nxt]))
            if E[k] > L[k] + 1e-9:
                ans = (False, float("inf"), {}, load)
                self.cache[key] = ans
                return ans

        T = [0.0] * m
        for k, c in enumerate(seq):
            T[k] = min(max(self.p[c], E[k]), L[k])

        for k in range(1, m):
            prev = seq[k - 1]
            T[k] = max(T[k], T[k - 1] + self.s[prev] + float(self.dist[prev][seq[k]]), E[k])
            if T[k] > L[k] + 1e-9:
                ans = (False, float("inf"), {}, load)
                self.cache[key] = ans
                return ans

        for k in range(m - 2, -1, -1):
            c = seq[k]
            nxt = seq[k + 1]
            T[k] = min(T[k], T[k + 1] - self.s[c] - float(self.dist[c][nxt]), L[k])
            if T[k] < E[k] - 1e-9:
                ans = (False, float("inf"), {}, load)
                self.cache[key] = ans
                return ans

        for k in range(1, m):
            prev = seq[k - 1]
            need = T[k - 1] + self.s[prev] + float(self.dist[prev][seq[k]])
            if T[k] < need:
                T[k] = need
            if T[k] > L[k] + 1e-9:
                ans = (False, float("inf"), {}, load)
                self.cache[key] = ans
                return ans

        route_cost = float(self.dist[0][seq[0]])
        for k in range(m - 1):
            route_cost += float(self.dist[seq[k]][seq[k + 1]])
        route_cost += float(self.dist[seq[-1]][self.end])

        inc = 0.0
        sched = {}
        for k, c in enumerate(seq):
            sched[c] = T[k]
            d = T[k] - self.p[c]
            inc += self.rho * d * d

        total = route_cost + inc
        ans = (True, total, sched, load)
        self.cache[key] = ans
        return ans

    def construct(self, seed: int) -> Optional[Tuple[List[List[int]], float, Dict[int, float]]]:
        rnd = random.Random(seed)
        unrouted = set(range(1, self.n + 1))
        routes: List[List[int]] = []
        robj: List[float] = []
        rload: List[int] = []
        rsched: List[Dict[int, float]] = []

        while unrouted and not self.timeout():
            best = None
            best_delta = float("inf")
            cand = list(unrouted)
            rnd.shuffle(cand)

            for c in cand:
                for r in range(len(routes)):
                    if rload[r] + self.dem[c] > self.Q:
                        continue
                    seq = routes[r]
                    old_obj = robj[r]
                    for pos in range(len(seq) + 1):
                        new_seq = seq[:pos] + [c] + seq[pos:]
                        feas, new_obj, new_sc, new_ld = self.eval_route(new_seq)
                        if feas:
                            delta = new_obj - old_obj
                            if delta < best_delta:
                                best_delta = delta
                                best = ("ins", c, r, new_seq, new_obj, new_sc, new_ld)

                if len(routes) < self.K:
                    feas, new_obj, new_sc, new_ld = self.eval_route([c])
                    if feas and new_obj < best_delta:
                        best_delta = new_obj
                        best = ("new", c, [c], new_obj, new_sc, new_ld)

            if best is None:
                return None

            if best[0] == "ins":
                _, c, r, seq, obj, sc, ld = best
                routes[r] = seq
                robj[r] = obj
                rsched[r] = sc
                rload[r] = ld
            else:
                _, c, seq, obj, sc, ld = best
                routes.append(seq)
                robj.append(obj)
                rsched.append(sc)
                rload.append(ld)

            unrouted.remove(c)

        if unrouted:
            return None

        total = sum(robj)
        sched = {}
        for sc in rsched:
            sched.update(sc)
        return routes, total, sched

    def improve(self, routes: List[List[int]], cur_obj: float, sched: Dict[int, float]):
        improved = True
        while improved and not self.timeout():
            improved = False
            R = len(routes)

            for r1 in range(R):
                if self.timeout():
                    break
                seq1 = routes[r1]
                if not seq1:
                    continue

                feas1_old, obj1_old, _, _ = self.eval_route(seq1)
                if not feas1_old:
                    continue

                for i in range(len(seq1)):
                    c = seq1[i]
                    rem1 = seq1[:i] + seq1[i + 1:]
                    feas_rem, obj_rem, _, _ = self.eval_route(rem1)
                    if not feas_rem:
                        continue

                    for r2 in range(R):
                        if self.timeout():
                            break
                        seq2_old = routes[r2]
                        feas2_old, obj2_old, _, _ = self.eval_route(seq2_old)
                        if not feas2_old:
                            continue

                        base2 = rem1 if r1 == r2 else seq2_old

                        for j in range(len(base2) + 1):
                            if r1 == r2 and j == i:
                                continue
                            new2 = base2[:j] + [c] + base2[j:]
                            if r1 != r2 and self.route_load(new2) > self.Q:
                                continue
                            feas2_new, obj2_new, _, _ = self.eval_route(new2)
                            if not feas2_new:
                                continue

                            if r1 == r2:
                                delta = obj2_new - obj1_old
                            else:
                                delta = (obj_rem + obj2_new) - (obj1_old + obj2_old)

                            if delta < -1e-8:
                                if r1 == r2:
                                    routes[r1] = new2
                                else:
                                    routes[r1] = rem1
                                    routes[r2] = new2
                                    if len(routes[r1]) == 0:
                                        routes.pop(r1)

                                cur_obj += delta
                                sched = {}
                                ok = True
                                for rr in routes:
                                    f, _, sc, _ = self.eval_route(rr)
                                    if not f:
                                        ok = False
                                        break
                                    sched.update(sc)
                                if not ok:
                                    continue

                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return routes, cur_obj, sched

    def pack_solution(self, routes: List[List[int]], obj: float, sched_idx: Dict[int, float]) -> dict:
        out_routes = [[0] + r + [self.end] for r in routes]
        schedule = {str(self.idx_to_id[i]): float(sched_idx.get(i, self.p[i])) for i in range(1, self.n + 1)}
        return {
            "objective_value": float(obj),
            "routes": out_routes,
            "schedule": schedule,
        }

    def fallback_solution(self) -> dict:
        schedule = {str(self.idx_to_id[i]): float(self.p[i]) for i in range(1, self.n + 1)}
        return {"objective_value": 1e30, "routes": [], "schedule": schedule}

    def solve(self) -> dict:
        best_sol = None
        best_obj = float("inf")
        seed = 0

        while not self.timeout():
            self.cache.clear()
            init = self.construct(seed)
            seed += 1

            if init is None:
                if seed >= 20 and (time.time() - self.start_time) > 0.7 * self.time_limit:
                    break
                continue

            routes, obj, sched = init
            routes, obj, sched = self.improve(routes, obj, sched)

            if obj < best_obj:
                best_obj = obj
                best_sol = self.pack_solution(routes, obj, sched)
                self.logger.log_solution(best_obj, best_sol)

            if seed >= 40:
                break

        if best_sol is None:
            best_sol = self.fallback_solution()
            self.logger.log_solution(best_sol["objective_value"], best_sol)

        return best_sol


def build_emergency_solution(inst: dict) -> dict:
    try:
        n = int(inst["num_customers"])
        customers = inst["customers"]
        schedule = {}
        for i in range(1, n + 1):
            cid = int(customers[i - 1]["id"])
            schedule[str(cid)] = float(customers[i - 1]["preferred_time"])
        return {"objective_value": 1e30, "routes": [], "schedule": schedule}
    except Exception:
        return {"objective_value": 1e30, "routes": [], "schedule": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SafeLogger(args.log_path)

    solution = None
    try:
        with open(args.instance_path, "r") as f:
            inst = json.load(f)

        solver = Solver(inst, args.time_limit, logger)
        solution = solver.solve()

    except Exception:
        try:
            with open(args.instance_path, "r") as f:
                inst = json.load(f)
            solution = build_emergency_solution(inst)
        except Exception:
            solution = {"objective_value": 1e30, "routes": [], "schedule": {}}

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()