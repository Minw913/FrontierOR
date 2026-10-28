import argparse
import json
import math
import random
import time
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


class DefenderLP:
    def __init__(self, data, time_limit_getter):
        self.data = data
        self.time_left = time_limit_getter

        gp.setParam("OutputFlag", 0)
        self.model = gp.Model("defender_lp")
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1

        g = data["global_parameters"]
        ip = data["instance_parameters"]

        self.theta = float(g["theta"])
        self.sigma = float(g["sigma"])
        self.alpha = float(g["alpha"])
        self.beta = float(g["beta"])
        self.gamma = float(g["gamma"])
        self.alpha_p = float(g["alpha_prime"])
        self.beta_p = float(g["beta_prime"])
        self.gamma_p = float(g["gamma_prime"])

        self.nC = int(ip["num_customers"])
        self.n1 = int(ip["num_level1_facilities"])
        self.n2 = int(ip["num_level2_facilities"])

        self.dem = list(map(float, data["customers"]["demands"]))
        self.cap1 = list(map(float, data["level1_facilities"]["capacity_type1"]))
        self.cap21 = list(map(float, data["level2_facilities"]["capacity_type1"]))
        self.cap22 = list(map(float, data["level2_facilities"]["capacity_type2"]))

        self.d_c1 = data["distances"]["customer_to_level1_facility"]
        self.d_c2 = data["distances"]["customer_to_level2_facility"]
        self.d_12 = data["distances"]["level1_to_level2_facility"]

        x2 = data["level2_facilities"]["x_coordinates"]
        y2 = data["level2_facilities"]["y_coordinates"]
        self.d_22 = [[0.0] * self.n2 for _ in range(self.n2)]
        for a in range(self.n2):
            for b in range(self.n2):
                dx = float(x2[a]) - float(x2[b])
                dy = float(y2[a]) - float(y2[b])
                self.d_22[a][b] = math.hypot(dx, dy)

        # Variables
        self.x1 = self.model.addVars(self.nC, self.n1, lb=0.0, name="x1")   # cust->L1 type I
        self.x2 = self.model.addVars(self.nC, self.n2, lb=0.0, name="x2")   # cust->L2 type I
        self.z = self.model.addVars(self.nC, self.n2, lb=0.0, name="z")     # cust->L2 type II
        self.o1 = self.model.addVars(self.nC, lb=0.0, name="o1")            # outsource type I
        self.o2 = self.model.addVars(self.nC, lb=0.0, name="o2")            # outsource type II

        self.r1 = self.model.addVars(self.n1, self.n2, lb=0.0, name="r1")   # referral L1->L2
        self.r2 = self.model.addVars(self.n2, self.n2, lb=0.0, name="r2")   # referral L2->L2
        self.ro1 = self.model.addVars(self.n1, lb=0.0, name="ro1")          # referral outsource at L1
        self.ro2 = self.model.addVars(self.n2, lb=0.0, name="ro2")          # referral outsource at L2

        # Constraints
        self.dem1_con = []
        self.dem2_con = []
        self.ref1_con = []
        self.ref2_con = []
        self.cap1_con = []
        self.cap21_con = []
        self.cap22_con = []

        for c in range(self.nC):
            rhs1 = self.theta * self.dem[c]
            rhs2 = (1.0 - self.theta) * self.dem[c]
            self.dem1_con.append(
                self.model.addConstr(
                    gp.quicksum(self.x1[c, i] for i in range(self.n1)) +
                    gp.quicksum(self.x2[c, j] for j in range(self.n2)) +
                    self.o1[c] == rhs1,
                    name=f"dem1_{c}"
                )
            )
            self.dem2_con.append(
                self.model.addConstr(
                    gp.quicksum(self.z[c, j] for j in range(self.n2)) +
                    self.o2[c] == rhs2,
                    name=f"dem2_{c}"
                )
            )

        for i in range(self.n1):
            self.ref1_con.append(
                self.model.addConstr(
                    self.sigma * gp.quicksum(self.x1[c, i] for c in range(self.nC)) ==
                    gp.quicksum(self.r1[i, j] for j in range(self.n2)) + self.ro1[i],
                    name=f"ref1_{i}"
                )
            )

        for a in range(self.n2):
            self.ref2_con.append(
                self.model.addConstr(
                    self.sigma * gp.quicksum(self.x2[c, a] for c in range(self.nC)) ==
                    gp.quicksum(self.r2[a, b] for b in range(self.n2)) + self.ro2[a],
                    name=f"ref2_{a}"
                )
            )

        for i in range(self.n1):
            self.cap1_con.append(
                self.model.addConstr(
                    gp.quicksum(self.x1[c, i] for c in range(self.nC)) <= self.cap1[i],
                    name=f"cap1_{i}"
                )
            )

        for j in range(self.n2):
            self.cap21_con.append(
                self.model.addConstr(
                    gp.quicksum(self.x2[c, j] for c in range(self.nC)) <= self.cap21[j],
                    name=f"cap21_{j}"
                )
            )
            self.cap22_con.append(
                self.model.addConstr(
                    gp.quicksum(self.z[c, j] for c in range(self.nC)) +
                    gp.quicksum(self.r1[i, j] for i in range(self.n1)) +
                    gp.quicksum(self.r2[a, j] for a in range(self.n2)) <= self.cap22[j],
                    name=f"cap22_{j}"
                )
            )

        # Objective
        obj = gp.LinExpr()

        # customer->L1 type I
        for c in range(self.nC):
            for i in range(self.n1):
                obj += self.alpha * float(self.d_c1[c][i]) * self.x1[c, i]

        # customer->L2 type I and type II
        for c in range(self.nC):
            for j in range(self.n2):
                dij = float(self.d_c2[c][j])
                obj += self.beta * dij * self.x2[c, j]
                obj += self.beta * dij * self.z[c, j]

        # referral shipping
        for i in range(self.n1):
            for j in range(self.n2):
                obj += self.gamma * float(self.d_12[i][j]) * self.r1[i, j]

        for a in range(self.n2):
            for b in range(self.n2):
                obj += self.gamma * float(self.d_22[a][b]) * self.r2[a, b]

        # outsourcing
        for c in range(self.nC):
            obj += (self.alpha_p + self.sigma * self.gamma_p) * self.o1[c]
            obj += self.beta_p * self.o2[c]

        for i in range(self.n1):
            obj += self.gamma_p * self.ro1[i]

        for a in range(self.n2):
            obj += self.gamma_p * self.ro2[a]

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()

    def solve(self, rem_cap1: List[float], rem_cap21: List[float], rem_cap22: List[float]) -> Tuple[float, List[float], List[float], List[float]]:
        for i in range(self.n1):
            self.cap1_con[i].RHS = rem_cap1[i]
        for j in range(self.n2):
            self.cap21_con[j].RHS = rem_cap21[j]
            self.cap22_con[j].RHS = rem_cap22[j]

        tl = max(0.05, self.time_left())
        self.model.Params.TimeLimit = tl
        self.model.optimize()

        if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            # fallback very large
            return float("inf"), [0.0] * self.n1, [0.0] * self.n2, [0.0] * self.n2

        obj = float(self.model.ObjVal)

        # Shadow prices on capacity constraints
        pi1 = [float(self.cap1_con[i].Pi) for i in range(self.n1)]
        pi21 = [float(self.cap21_con[j].Pi) for j in range(self.n2)]
        pi22 = [float(self.cap22_con[j].Pi) for j in range(self.n2)]

        return obj, pi1, pi21, pi22


def build_level_arrays(k_max, cost_dict, red_dict):
    cost = [0.0] * (k_max + 1)
    red = [0.0] * (k_max + 1)
    for k, v in cost_dict.items():
        cost[int(k)] = float(v)
    for k, v in red_dict.items():
        red[int(k)] = float(v)
    return cost, red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()

    def time_left():
        return max(0.0, args.time_limit - (time.time() - t0))

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    ip = data["instance_parameters"]
    inter = data["interdiction_parameters"]

    n1 = int(ip["num_level1_facilities"])
    n2 = int(ip["num_level2_facilities"])
    k_max = int(ip["k_max"])
    B = float(ip["total_interdiction_budget"])

    h1, d1 = build_level_arrays(k_max, inter["interdiction_cost_level1_h1"], inter["capacity_reduction_level1_d1"])
    h2, d2 = build_level_arrays(k_max, inter["interdiction_cost_level2_h2"], inter["capacity_reduction_level2_d2"])

    cap1 = list(map(float, data["level1_facilities"]["capacity_type1"]))
    cap21 = list(map(float, data["level2_facilities"]["capacity_type1"]))
    cap22 = list(map(float, data["level2_facilities"]["capacity_type2"]))

    defender = DefenderLP(data, time_left)

    def pattern_cost(L1, L2):
        return sum(h1[l] for l in L1) + sum(h2[l] for l in L2)

    def capacities_from_pattern(L1, L2):
        rem1 = [cap1[i] * (1.0 - d1[L1[i]]) for i in range(n1)]
        rem21 = [cap21[j] * (1.0 - d2[L2[j]]) for j in range(n2)]
        rem22 = [cap22[j] * (1.0 - d2[L2[j]]) for j in range(n2)]
        return rem1, rem21, rem22

    def eval_pattern(L1, L2):
        rem1, rem21, rem22 = capacities_from_pattern(L1, L2)
        return defender.solve(rem1, rem21, rem22)

    # Initial pattern (no interdiction)
    current_L1 = [0] * n1
    current_L2 = [0] * n2
    cur_cost = 0.0
    cur_obj, pi1, pi21, pi22 = eval_pattern(current_L1, current_L2)

    best_obj = cur_obj
    best_L1 = current_L1[:]
    best_L2 = current_L2[:]
    if logger:
        logger.log(best_obj)

    # local search from a starting point
    def improve_from(start_L1, start_L2, start_obj):
        nonlocal best_obj, best_L1, best_L2

        L1 = start_L1[:]
        L2 = start_L2[:]
        obj = start_obj
        cost_used = pattern_cost(L1, L2)

        # get duals for surrogate guidance
        _, pi1_loc, pi21_loc, pi22_loc = eval_pattern(L1, L2)

        while time_left() > 0.2:
            remB = B - cost_used
            if remB < 1e-9:
                break

            candidates = []
            # L1 upgrades
            for i in range(n1):
                cur = L1[i]
                for nl in range(cur + 1, k_max + 1):
                    extra = h1[nl] - h1[cur]
                    if extra <= remB + 1e-9:
                        delta_red = d1[nl] - d1[cur]
                        est_gain = max(0.0, -pi1_loc[i]) * cap1[i] * max(0.0, delta_red)
                        score = est_gain / (extra + 1e-6)
                        candidates.append((score, est_gain, "S1", i, nl, extra))

            # L2 upgrades
            for j in range(n2):
                cur = L2[j]
                for nl in range(cur + 1, k_max + 1):
                    extra = h2[nl] - h2[cur]
                    if extra <= remB + 1e-9:
                        delta_red = d2[nl] - d2[cur]
                        est_gain = (
                            max(0.0, -pi21_loc[j]) * cap21[j] * max(0.0, delta_red) +
                            max(0.0, -pi22_loc[j]) * cap22[j] * max(0.0, delta_red)
                        )
                        score = est_gain / (extra + 1e-6)
                        candidates.append((score, est_gain, "S2", j, nl, extra))

            if not candidates:
                break

            candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
            topK = min(10, len(candidates))

            best_move = None
            best_move_obj = obj
            best_move_duals = None

            for idx in range(topK):
                if time_left() <= 0.1:
                    break
                _, _, tier, fidx, nl, extra = candidates[idx]

                old = L1[fidx] if tier == "S1" else L2[fidx]
                if tier == "S1":
                    L1[fidx] = nl
                else:
                    L2[fidx] = nl

                new_obj, p1n, p21n, p22n = eval_pattern(L1, L2)

                # restore
                if tier == "S1":
                    L1[fidx] = old
                else:
                    L2[fidx] = old

                if new_obj > best_move_obj + 1e-7:
                    best_move_obj = new_obj
                    best_move = (tier, fidx, nl, extra)
                    best_move_duals = (p1n, p21n, p22n)

            if best_move is None:
                break

            tier, fidx, nl, extra = best_move
            if tier == "S1":
                L1[fidx] = nl
            else:
                L2[fidx] = nl
            cost_used += extra
            obj = best_move_obj
            pi1_loc, pi21_loc, pi22_loc = best_move_duals

            if obj > best_obj + 1e-7:
                best_obj = obj
                best_L1 = L1[:]
                best_L2 = L2[:]
                if logger:
                    logger.log(best_obj)

        return L1, L2, obj

    # Improve from zero start
    current_L1, current_L2, cur_obj = improve_from(current_L1, current_L2, cur_obj)

    # Additional randomized starts until time
    random.seed(0)
    while time_left() > 0.5:
        L1 = [0] * n1
        L2 = [0] * n2
        used = 0.0

        # random greedy fill
        facilities = [("S1", i) for i in range(n1)] + [("S2", j) for j in range(n2)]
        random.shuffle(facilities)

        for tier, idx in facilities:
            if time_left() <= 0.2:
                break
            if tier == "S1":
                feasible_levels = [k for k in range(1, k_max + 1) if used + h1[k] <= B + 1e-9]
                if feasible_levels:
                    k = random.choice(feasible_levels + [0, 0])  # bias to sparse attacks
                    if k > 0:
                        L1[idx] = k
                        used += h1[k]
            else:
                feasible_levels = [k for k in range(1, k_max + 1) if used + h2[k] <= B + 1e-9]
                if feasible_levels:
                    k = random.choice(feasible_levels + [0, 0])
                    if k > 0:
                        L2[idx] = k
                        used += h2[k]

        obj, _, _, _ = eval_pattern(L1, L2)
        if obj > best_obj + 1e-7:
            best_obj = obj
            best_L1 = L1[:]
            best_L2 = L2[:]
            if logger:
                logger.log(best_obj)

        improve_from(L1, L2, obj)

    interdiction_pattern = {}
    for i in range(n1):
        interdiction_pattern[f"S1_{i}"] = int(best_L1[i])
    for j in range(n2):
        interdiction_pattern[f"S2_{j}"] = int(best_L2[j])

    sol = {
        "objective_value": float(best_obj),
        "interdiction_pattern": interdiction_pattern
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()