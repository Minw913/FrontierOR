import argparse
import json
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional

from solution_logger import SolutionLogger


def exp_dynamics_step(x, v, u, T):
    e = math.exp(-T)
    a = 1.0 - e
    b = T - 1.0 + e
    v_next = e * v + a * u
    x_next = x + a * v + b * u
    return x_next, v_next


def exp_dynamics_interp(xk, vk, uk, tau):
    if tau <= 0:
        return xk
    e = math.exp(-tau)
    a = 1.0 - e
    b = tau - 1.0 + e
    return xk + a * vk + b * uk


def regular_polygon_normals(M: int) -> List[Tuple[float, float]]:
    return [(math.cos(2.0 * math.pi * m / M), math.sin(2.0 * math.pi * m / M)) for m in range(M)]


def inside_polygon_halfspaces(x: float, y: float, normals: List[Tuple[float, float]], apothem: float, tol=1e-9) -> bool:
    for c, s in normals:
        if c * x + s * y > apothem + tol:
            return False
    return True


def inside_square_obstacle(x: float, y: float, R: float, tol=1e-9) -> bool:
    return (abs(x) <= R + tol) and (abs(y) <= R + tol)


def defender_pos_at_time(def_sol: Dict[str, List[float]], t: float, T_u: float, N_u: int) -> Tuple[float, float]:
    x = def_sol["x"]
    y = def_sol["y"]
    xdot = def_sol["xdot"]
    ydot = def_sol["ydot"]
    ux = def_sol["ux"]
    uy = def_sol["uy"]

    if t <= 0:
        return x[0], y[0]
    total_T = N_u * T_u
    if t >= total_T - 1e-12:
        return x[N_u], y[N_u]

    n = int(t // T_u)
    if n >= N_u:
        return x[N_u], y[N_u]
    tau = t - n * T_u
    if tau <= 1e-12:
        return x[n], y[n]

    xi = exp_dynamics_interp(x[n], xdot[n], ux[n], tau)
    yi = exp_dynamics_interp(y[n], ydot[n], uy[n], tau)
    return xi, yi


def build_defender_trajectory(instance: Dict[str, Any], mode: str, rng: random.Random) -> List[Dict[str, List[float]]]:
    N_D = instance["N_D"]
    N_u = instance["N_u"]
    T_u = instance["T_u"]
    R_dz = instance["R_dz"]
    M_u = instance["M_u"]

    attackers = instance["attackers"]
    N_A = instance["N_A"]
    epsilon = instance["epsilon"]

    # Conservative feasible control radius (inside all halfspaces)
    u_max = math.cos(math.pi / M_u)

    # Candidate controls
    K_dir = max(8, min(24, M_u * 2))
    dirs = [(math.cos(2.0 * math.pi * k / K_dir), math.sin(2.0 * math.pi * k / K_dir)) for k in range(K_dir)]
    cand_controls = [(0.0, 0.0)]
    for dx, dy in dirs:
        cand_controls.append((u_max * dx, u_max * dy))
        cand_controls.append((0.5 * u_max * dx, 0.5 * u_max * dy))

    defenders_out = []
    for i in range(N_D):
        d0 = instance["defenders"][i]
        x = [0.0] * (N_u + 1)
        y = [0.0] * (N_u + 1)
        xdot = [0.0] * (N_u + 1)
        ydot = [0.0] * (N_u + 1)
        ux = [0.0] * N_u
        uy = [0.0] * N_u

        x[0] = float(d0["x_s"])
        y[0] = float(d0["y_s"])
        xdot[0] = float(d0["xdot_s"])
        ydot[0] = float(d0["ydot_s"])

        for k in range(N_u):
            t_next = (k + 1) * T_u

            # Build target set from straight-line attacker prediction (always-active approximation)
            predicted = []
            for a in attackers:
                px = float(a["p_s"]) + float(a["v_p"]) * t_next
                py = float(a["q_s"]) + float(a["v_q"]) * t_next
                r = math.hypot(px, py)
                predicted.append((px, py, r))

            # For zero mode
            if mode == "zero":
                best_u = (0.0, 0.0)
            else:
                # Greedy scoring
                best_score = -1e100
                best_u = (0.0, 0.0)

                # Randomized weights for diversification
                w_threat = 0.2 + 0.8 * rng.random()
                w_effort = 0.1 + 0.5 * epsilon
                noise_amp = 0.02

                for (uix, uiy) in cand_controls:
                    xn, vxn = exp_dynamics_step(x[k], xdot[k], uix, T_u)
                    yn, vyn = exp_dynamics_step(y[k], ydot[k], uiy, T_u)

                    # Threat-based nearest distance
                    d_best = 1e18
                    for (px, py, rr) in predicted:
                        d = math.hypot(xn - px, yn - py) - w_threat / (rr + 1e-3)
                        if d < d_best:
                            d_best = d

                    score = -d_best
                    score -= w_effort * (abs(uix) + abs(uiy))

                    # Obstacle outside preference (must stay outside square if possible)
                    if inside_square_obstacle(xn, yn, R_dz):
                        pen = max(0.0, R_dz - abs(xn)) + max(0.0, R_dz - abs(yn))
                        score -= 500.0 + 50.0 * pen
                    else:
                        score += 5.0

                    # Mild outward preference near center
                    score += 0.02 * math.hypot(xn, yn)

                    score += noise_amp * (2.0 * rng.random() - 1.0)

                    if score > best_score:
                        best_score = score
                        best_u = (uix, uiy)

            ux[k], uy[k] = best_u
            x[k + 1], xdot[k + 1] = exp_dynamics_step(x[k], xdot[k], ux[k], T_u)
            y[k + 1], ydot[k + 1] = exp_dynamics_step(y[k], ydot[k], uy[k], T_u)

        defenders_out.append(
            {
                "x": x,
                "y": y,
                "xdot": xdot,
                "ydot": ydot,
                "ux": ux,
                "uy": uy,
            }
        )

    return defenders_out


def simulate_attackers_and_objective(instance: Dict[str, Any], defenders_sol: List[Dict[str, List[float]]]) -> Tuple[float, List[Dict[str, Any]]]:
    N_A = instance["N_A"]
    N_D = instance["N_D"]
    N_a = instance["N_a"]
    T_a = instance["T_a"]
    T_u = instance["T_u"]
    N_u = instance["N_u"]
    R_dz = instance["R_dz"]
    R_I = instance["R_I"]
    M_dz = instance["M_dz"]
    M_I = instance["M_I"]
    epsilon = instance["epsilon"]

    dz_normals = regular_polygon_normals(M_dz)
    int_normals = regular_polygon_normals(M_I)

    attackers_out = []
    total_gamma = 0.0

    for j in range(N_A):
        a_data = instance["attackers"][j]
        vp = float(a_data["v_p"])
        vq = float(a_data["v_q"])

        p = [0.0] * (N_a + 1)
        q = [0.0] * (N_a + 1)
        a = [0.0] * (N_a + 1)
        gamma = [0.0] * (N_a + 1)
        delta = [[0.0] * (N_a + 1) for _ in range(N_D)]

        p[0] = float(a_data["p_s"])
        q[0] = float(a_data["q_s"])
        a[0] = 1.0

        for k in range(N_a + 1):
            # Zone occupancy at current step
            gk = 1.0 if inside_polygon_halfspaces(p[k], q[k], dz_normals, R_dz) else 0.0
            gamma[k] = gk

            # Interception indicators at current step (only if active)
            any_int = 0.0
            if a[k] > 0.5:
                t = k * T_a
                for i in range(N_D):
                    dx, dy = defender_pos_at_time(defenders_sol[i], t, T_u, N_u)
                    relx = p[k] - dx
                    rely = q[k] - dy
                    hit = 1.0 if inside_polygon_halfspaces(relx, rely, int_normals, R_I) else 0.0
                    delta[i][k] = hit
                    if hit > 0.5:
                        any_int = 1.0
            else:
                for i in range(N_D):
                    delta[i][k] = 0.0

            # Immediate deactivation at this sample step
            if a[k] > 0.5 and (gk > 0.5 or any_int > 0.5):
                a[k] = 0.0

            # Move to next sample using post-check activity
            if k < N_a:
                a[k + 1] = a[k]
                p[k + 1] = p[k] + a[k] * T_a * vp
                q[k + 1] = q[k] + a[k] * T_a * vq

        total_gamma += sum(gamma[1:])  # steps 1..N_a

        attackers_out.append(
            {
                "p": p,
                "q": q,
                "a": a,
                "gamma": gamma,
                "delta": delta,
            }
        )

    effort = 0.0
    for d in defenders_sol:
        for ux, uy in zip(d["ux"], d["uy"]):
            effort += abs(ux) + abs(uy)

    objective = total_gamma + epsilon * effort
    return objective, attackers_out


def build_full_solution(instance: Dict[str, Any], defenders_sol: List[Dict[str, List[float]]]) -> Dict[str, Any]:
    obj, attackers_sol = simulate_attackers_and_objective(instance, defenders_sol)
    return {
        "objective_value": float(obj),
        "defenders": defenders_sol,
        "attackers": attackers_sol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    rng = random.Random(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    best_solution: Optional[Dict[str, Any]] = None
    best_obj = float("inf")

    # Baseline: zero controls
    defenders_zero = build_defender_trajectory(instance, mode="zero", rng=rng)
    sol_zero = build_full_solution(instance, defenders_zero)
    best_solution = sol_zero
    best_obj = sol_zero["objective_value"]
    if logger:
        logger.log_solution(best_obj, best_solution)

    # Randomized greedy multi-start until time limit
    # keep small safety margin for output writing
    safety = 0.05
    it = 0
    while time.time() - start_time < max(0.0, args.time_limit - safety):
        it += 1
        defenders_greedy = build_defender_trajectory(instance, mode="greedy", rng=rng)
        sol = build_full_solution(instance, defenders_greedy)
        obj = sol["objective_value"]

        if obj + 1e-12 < best_obj:
            best_obj = obj
            best_solution = sol
            if logger:
                logger.log_solution(best_obj, best_solution)

        # very large instances: avoid too many iterations
        if it >= 200:
            break

    # Fallback safeguard
    if best_solution is None:
        defenders_zero = build_defender_trajectory(instance, mode="zero", rng=rng)
        best_solution = build_full_solution(instance, defenders_zero)

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()