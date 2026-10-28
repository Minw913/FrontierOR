import argparse
import json
import math
import random
import time
from bisect import bisect_right

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

from solution_logger import SolutionLogger


class TDTSPHeuristicSolver:
    def __init__(self, data, time_limit, logger=None):
        self.data = data
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()
        self.deadline = self.start_time + time_limit

        self.params = data["parameters"]
        self.n = int(self.params["total_vertices"])
        self.H = int(self.params["H"])
        self.boundaries = self.params["period_boundaries"]
        self.depot = int(data["depot"]["index"])

        self.dist = data["arcs"]["distances"]
        self.max_speeds = data["arcs"]["max_speeds_u_ij"]

        if "speeds_v_ijh" in data["arcs"] and data["arcs"]["speeds_v_ijh"] is not None:
            self.speeds = data["arcs"]["speeds_v_ijh"]
        else:
            # Fallback reconstruction
            b_h = self.params["b_h"]
            delta = data["arcs"]["delta_ijh"]
            self.speeds = [[[delta[i][j][h] * b_h[h] * self.max_speeds[i][j] for h in range(self.H)]
                            for j in range(self.n)] for i in range(self.n)]

        self.customers = [v for v in range(self.n) if v != self.depot]
        self.rng = random.Random(42)

        self.best_route = None
        self.best_obj = float("inf")

        self.static_cost = self._build_static_cost_matrix()

    def time_left(self):
        return self.deadline - time.time()

    def _build_static_cost_matrix(self):
        durations = [self.boundaries[h + 1] - self.boundaries[h] for h in range(self.H)]
        total_dur = sum(max(0.0, d) for d in durations)
        if total_dur <= 0:
            total_dur = 1.0

        c = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    c[i][j] = 1e9
                    continue
                avg_v = 0.0
                for h in range(self.H):
                    avg_v += max(0.0, self.speeds[i][j][h]) * max(0.0, durations[h])
                avg_v /= total_dur
                if avg_v <= 1e-9:
                    c[i][j] = 1e9
                else:
                    c[i][j] = self.dist[i][j] / avg_v
        return c

    def arc_travel_time(self, i, j, depart_t):
        d = self.dist[i][j]
        if d <= 1e-12:
            return 0.0

        t0 = depart_t
        t = depart_t
        rem = d
        max_iter = self.H + 5  # safeguard

        for _ in range(max_iter):
            h = bisect_right(self.boundaries, t) - 1
            if h < 0:
                h = 0
            if h >= self.H:
                h = self.H - 1

            v = self.speeds[i][j][h]
            if v <= 1e-12:
                return 1e12

            if h == self.H - 1:
                # Assume last period speed persists if needed
                return (t - t0) + rem / v

            period_end = self.boundaries[h + 1]
            dt_possible = period_end - t
            if dt_possible <= 1e-12:
                t = period_end
                continue

            can_cover = v * dt_possible
            if can_cover + 1e-12 >= rem:
                t += rem / v
                return t - t0
            else:
                rem -= can_cover
                t = period_end

        return 1e12

    def evaluate_route(self, route):
        t = 0.0
        for k in range(len(route) - 1):
            dt = self.arc_travel_time(route[k], route[k + 1], t)
            if dt >= 1e11:
                return 1e12
            t += dt
            if t >= 1e12:
                return 1e12
        return t

    def maybe_update_best(self, route, obj):
        if obj + 1e-9 < self.best_obj:
            self.best_obj = float(obj)
            self.best_route = route[:]
            if self.logger:
                self.logger.log(self.best_obj)

    def construct_greedy_randomized(self, alpha=0.0):
        unvisited = set(self.customers)
        route = [self.depot]
        cur = self.depot
        t = 0.0

        while unvisited:
            cand = []
            cmin = float("inf")
            cmax = -float("inf")
            for j in unvisited:
                dt = self.arc_travel_time(cur, j, t)
                cand.append((j, dt))
                if dt < cmin:
                    cmin = dt
                if dt > cmax:
                    cmax = dt

            if not cand:
                break

            if alpha <= 1e-12:
                nxt, dt = min(cand, key=lambda x: x[1])
            else:
                threshold = cmin + alpha * (cmax - cmin)
                rcl = [(j, dt) for (j, dt) in cand if dt <= threshold + 1e-12]
                nxt, dt = self.rng.choice(rcl if rcl else cand)

            route.append(nxt)
            unvisited.remove(nxt)
            t += dt
            cur = nxt

        route.append(self.depot)
        return route

    def solve_atsp_surrogate(self):
        if not GUROBI_AVAILABLE:
            return None, None
        if self.n > 90:
            return None, None

        rem = self.time_left()
        if rem < 3:
            return None, None

        try:
            m = gp.Model("atsp_surrogate")
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, min(rem * 0.4, rem - 0.5))

            x = {}
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

            u = {}
            for i in range(self.n):
                if i != self.depot:
                    u[i] = m.addVar(lb=1.0, ub=self.n - 1, vtype=GRB.CONTINUOUS, name=f"u_{i}")

            m.update()

            # Assignment constraints
            for i in range(self.n):
                m.addConstr(gp.quicksum(x[i, j] for j in range(self.n) if j != i) == 1)
            for j in range(self.n):
                m.addConstr(gp.quicksum(x[i, j] for i in range(self.n) if i != j) == 1)

            # MTZ subtour elimination (for non-depot nodes)
            for i in range(self.n):
                if i == self.depot:
                    continue
                for j in range(self.n):
                    if j == self.depot or i == j:
                        continue
                    m.addConstr(u[i] - u[j] + self.n * x[i, j] <= self.n - 1)

            m.setObjective(
                gp.quicksum(self.static_cost[i][j] * x[i, j] for i in range(self.n) for j in range(self.n) if i != j),
                GRB.MINIMIZE
            )

            m.optimize()

            if m.SolCount <= 0:
                return None, None

            succ = {}
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and x[i, j].X > 0.5:
                        succ[i] = j

            route = [self.depot]
            cur = self.depot
            seen = {self.depot}
            for _ in range(self.n - 1):
                nxt = succ.get(cur, None)
                if nxt is None or nxt in seen:
                    return None, None
                route.append(nxt)
                seen.add(nxt)
                cur = nxt

            if succ.get(cur, None) != self.depot:
                return None, None
            route.append(self.depot)

            obj = self.evaluate_route(route)
            return route, obj
        except Exception:
            return None, None

    def apply_random_move(self, route):
        m = len(route) - 2  # number of customers
        if m <= 1:
            return route[:]

        move_type = self.rng.choice(["swap", "relocate", "two_opt"])
        npos = len(route)

        if move_type == "swap":
            i, j = self.rng.sample(range(1, npos - 1), 2)
            if i > j:
                i, j = j, i
            cand = route[:]
            cand[i], cand[j] = cand[j], cand[i]
            return cand

        if move_type == "relocate":
            i, j = self.rng.sample(range(1, npos - 1), 2)
            cand = route[:]
            node = cand.pop(i)
            if j >= len(cand):
                j = len(cand) - 1
            cand.insert(j, node)
            return cand

        # two_opt
        i, j = self.rng.sample(range(1, npos - 1), 2)
        if i > j:
            i, j = j, i
        cand = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
        return cand

    def perturb(self, route, strength=2):
        cand = route[:]
        for _ in range(strength):
            if len(cand) <= 4:
                break
            i, j = self.rng.sample(range(1, len(cand) - 1), 2)
            cand[i], cand[j] = cand[j], cand[i]
        return cand

    def local_search(self, init_route, init_obj):
        current_route = init_route[:]
        current_obj = init_obj
        self.maybe_update_best(current_route, current_obj)

        no_improve_rounds = 0
        while self.time_left() > 0.05:
            improved = False
            # Number of sampled neighbors per round
            samples = max(20, min(200, 4 * (self.n - 1)))
            for _ in range(samples):
                if self.time_left() <= 0.05:
                    break
                cand = self.apply_random_move(current_route)
                obj = self.evaluate_route(cand)
                if obj + 1e-9 < current_obj:
                    current_route, current_obj = cand, obj
                    improved = True
                    self.maybe_update_best(current_route, current_obj)
                    break

            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= 3:
                    # restart near best
                    if self.best_route is not None:
                        current_route = self.perturb(self.best_route, strength=3)
                        current_obj = self.evaluate_route(current_route)
                        self.maybe_update_best(current_route, current_obj)
                    no_improve_rounds = 0
            else:
                no_improve_rounds = 0

    def solve(self):
        # 1) Greedy deterministic
        r0 = self.construct_greedy_randomized(alpha=0.0)
        o0 = self.evaluate_route(r0)
        self.maybe_update_best(r0, o0)

        # 2) Optional ATSP surrogate (good permutation seed)
        if self.time_left() > 2.0:
            r_atsp, o_atsp = self.solve_atsp_surrogate()
            if r_atsp is not None:
                self.maybe_update_best(r_atsp, o_atsp)

        # 3) Additional GRASP constructions
        attempts = 0
        while self.time_left() > max(1.0, 0.4 * self.time_limit) and attempts < 10:
            alpha = self.rng.uniform(0.05, 0.35)
            r = self.construct_greedy_randomized(alpha=alpha)
            o = self.evaluate_route(r)
            self.maybe_update_best(r, o)
            attempts += 1

        # 4) Intensification with local search
        if self.best_route is None:
            # fallback
            route = [self.depot] + self.customers[:] + [self.depot]
            self.best_route = route
            self.best_obj = self.evaluate_route(route)
            if self.logger:
                self.logger.log(self.best_obj)

        self.local_search(self.best_route, self.best_obj)

        # Final sanity
        if self.best_route is None:
            self.best_route = [self.depot] + self.customers[:] + [self.depot]
            self.best_obj = self.evaluate_route(self.best_route)

        return self.best_route, float(self.best_obj)


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

    solver = TDTSPHeuristicSolver(
        data=instance,
        time_limit=max(1, int(args.time_limit)),
        logger=logger
    )

    tour, obj = solver.solve()

    solution = {
        "objective_value": float(obj),
        "tour": [int(x) for x in tour]
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()