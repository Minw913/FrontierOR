import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def polygon_normals(m):
    return [
        (math.cos(2.0 * math.pi * j / m), math.sin(2.0 * math.pi * j / m))
        for j in range(m)
    ]


def ray_control(direction, normals, rhs):
    dx, dy = direction
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return 0.0, 0.0

    dx /= norm
    dy /= norm
    scale = float("inf")
    for nx, ny in normals:
        projection = nx * dx + ny * dy
        if projection > 1e-12:
            scale = min(scale, rhs / projection)

    if not math.isfinite(scale):
        scale = 0.0
    return scale * dx, scale * dy


def compute_defender_trajectories(instance, controls):
    nd = instance["N_D"]
    nu = instance["N_u"]
    tu = float(instance["T_u"])

    decay = math.exp(-tu)
    vel_u = 1.0 - decay
    pos_v = 1.0 - decay
    pos_u = tu - 1.0 + decay

    trajectories = []
    for d in range(nd):
        defender = instance["defenders"][d]
        x = [float(defender["x_s"])]
        y = [float(defender["y_s"])]
        vx = [float(defender["xdot_s"])]
        vy = [float(defender["ydot_s"])]

        ux = []
        uy = []
        for h in range(nu):
            ux_h = float(controls[d][h][0])
            uy_h = float(controls[d][h][1])
            ux.append(ux_h)
            uy.append(uy_h)

            x.append(x[h] + pos_v * vx[h] + pos_u * ux_h)
            y.append(y[h] + pos_v * vy[h] + pos_u * uy_h)
            vx.append(decay * vx[h] + vel_u * ux_h)
            vy.append(decay * vy[h] + vel_u * uy_h)

        trajectories.append({
            "x": x,
            "y": y,
            "xdot": vx,
            "ydot": vy,
            "ux": ux,
            "uy": uy,
        })

    return trajectories


def defender_position_at_time(trajectory, t, tu):
    nu = len(trajectory["ux"])
    horizon = nu * tu

    if t <= 0.0:
        return trajectory["x"][0], trajectory["y"][0]
    if t >= horizon - 1e-12:
        return trajectory["x"][nu], trajectory["y"][nu]

    h = min(int(math.floor(t / tu)), nu - 1)
    tau = t - h * tu
    if tau <= 1e-12:
        return trajectory["x"][h], trajectory["y"][h]

    decay = math.exp(-tau)
    pos_v = 1.0 - decay
    pos_u = tau - 1.0 + decay

    px = (
        trajectory["x"][h]
        + pos_v * trajectory["xdot"][h]
        + pos_u * trajectory["ux"][h]
    )
    py = (
        trajectory["y"][h]
        + pos_v * trajectory["ydot"][h]
        + pos_u * trajectory["uy"][h]
    )
    return px, py


def inside_polygon(px, py, normals, rhs, tolerance=1e-7):
    return all(nx * px + ny * py <= rhs + tolerance for nx, ny in normals)


def build_solution(instance, controls):
    nd = instance["N_D"]
    na = instance["N_A"]
    n_a = instance["N_a"]
    ta = float(instance["T_a"])
    tu = float(instance["T_u"])
    epsilon = float(instance["epsilon"])
    r_dz = float(instance["R_dz"])
    r_i = float(instance["R_I"])

    dz_normals = polygon_normals(instance["M_dz"])
    intercept_normals = polygon_normals(instance["M_I"])

    defender_trajectories = compute_defender_trajectories(instance, controls)

    defender_positions = [
        [
            defender_position_at_time(defender_trajectories[d], k * ta, tu)
            for k in range(n_a + 1)
        ]
        for d in range(nd)
    ]

    attacker_outputs = []
    penetration_cost = 0

    for j in range(na):
        attacker = instance["attackers"][j]
        p = [float(attacker["p_s"])]
        q = [float(attacker["q_s"])]
        vp = float(attacker["v_p"])
        vq = float(attacker["v_q"])

        active = []
        gamma = []
        delta = [[0.0] * (n_a + 1) for _ in range(nd)]

        for k in range(n_a + 1):
            if k > 0:
                if active[k - 1] >= 0.5:
                    p.append(p[k - 1] + ta * vp)
                    q.append(q[k - 1] + ta * vq)
                else:
                    p.append(p[k - 1])
                    q.append(q[k - 1])

            in_zone = inside_polygon(p[k], q[k], dz_normals, r_dz)
            gamma.append(1.0 if in_zone else 0.0)

            intercepted = False
            for d in range(nd):
                dx = p[k] - defender_positions[d][k][0]
                dy = q[k] - defender_positions[d][k][1]
                hit = inside_polygon(dx, dy, intercept_normals, r_i)
                delta[d][k] = 1.0 if hit else 0.0
                intercepted = intercepted or hit

            if k == 0:
                is_active = not (in_zone or intercepted)
            else:
                is_active = (
                    active[k - 1] >= 0.5 and not in_zone and not intercepted
                )
            active.append(1.0 if is_active else 0.0)

            if k >= 1 and in_zone:
                penetration_cost += 1

        attacker_outputs.append({
            "p": p,
            "q": q,
            "a": active,
            "gamma": gamma,
            "delta": delta,
        })

    effort = 0.0
    for d_controls in controls:
        for ux, uy in d_controls:
            effort += abs(float(ux)) + abs(float(uy))

    objective = float(penetration_cost + epsilon * effort)
    return {
        "objective_value": objective,
        "defenders": defender_trajectories,
        "attackers": attacker_outputs,
    }


def controls_are_feasible(instance, controls, tolerance=2e-5):
    control_normals = polygon_normals(instance["M_u"])
    control_rhs = math.cos(math.pi / instance["M_u"])
    obstacle_normals = polygon_normals(4)
    r_dz = float(instance["R_dz"])

    for defender_controls in controls:
        for ux, uy in defender_controls:
            for nx, ny in control_normals:
                if nx * ux + ny * uy > control_rhs + tolerance:
                    return False

    trajectories = compute_defender_trajectories(instance, controls)
    for trajectory in trajectories:
        for k in range(1, instance["N_u"] + 1):
            x = trajectory["x"][k]
            y = trajectory["y"][k]
            if not any(
                nx * x + ny * y >= r_dz - tolerance
                for nx, ny in obstacle_normals
            ):
                return False

    return True


def heuristic_controls(instance, outward):
    nd = instance["N_D"]
    nu = instance["N_u"]
    tu = float(instance["T_u"])
    normals = polygon_normals(instance["M_u"])
    rhs = math.cos(math.pi / instance["M_u"])

    decay = math.exp(-tu)
    vel_u = 1.0 - decay
    pos_v = 1.0 - decay
    pos_u = tu - 1.0 + decay

    all_controls = []
    for d in range(nd):
        defender = instance["defenders"][d]
        x = float(defender["x_s"])
        y = float(defender["y_s"])
        vx = float(defender["xdot_s"])
        vy = float(defender["ydot_s"])
        d_controls = []

        for _ in range(nu):
            if outward:
                direction = (x, y)
                if math.hypot(*direction) <= 1e-9:
                    direction = (-vx, -vy)
                if math.hypot(*direction) <= 1e-9:
                    direction = (1.0, 0.0)
                ux, uy = ray_control(direction, normals, rhs)
            else:
                ux, uy = 0.0, 0.0

            d_controls.append([ux, uy])
            new_x = x + pos_v * vx + pos_u * ux
            new_y = y + pos_v * vy + pos_u * uy
            new_vx = decay * vx + vel_u * ux
            new_vy = decay * vy + vel_u * uy
            x, y, vx, vy = new_x, new_y, new_vx, new_vy

        all_controls.append(d_controls)

    return all_controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    nd = int(instance["N_D"])
    na = int(instance["N_A"])
    nu = int(instance["N_u"])
    n_a = int(instance["N_a"])
    mu = int(instance["M_u"])
    mi = int(instance["M_I"])
    mdz = int(instance["M_dz"])

    tu = float(instance["T_u"])
    ta = float(instance["T_a"])
    r_dz = float(instance["R_dz"])
    r_i = float(instance["R_I"])
    epsilon = float(instance["epsilon"])

    control_normals = polygon_normals(mu)
    intercept_normals = polygon_normals(mi)
    dz_normals = polygon_normals(mdz)
    obstacle_normals = polygon_normals(4)
    control_rhs = math.cos(math.pi / mu)

    best_solution = None
    best_objective = float("inf")
    best_controls = None

    for outward in (False, True):
        controls = heuristic_controls(instance, outward)
        solution = build_solution(instance, controls)
        if controls_are_feasible(instance, controls):
            if solution["objective_value"] < best_objective - 1e-9:
                best_objective = solution["objective_value"]
                best_solution = solution
                best_controls = controls
                if logger:
                    logger.log_solution(best_objective, best_solution)
        elif best_solution is None:
            best_solution = solution
            best_objective = solution["objective_value"]
            best_controls = controls

    model = gp.Model("defense_zone")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining = max(0.01, args.time_limit - (time.monotonic() - start_time))
    model.Params.TimeLimit = remaining

    x = [[model.addVar(lb=-GRB.INFINITY, name=f"x_{d}_{k}")
          for k in range(nu + 1)] for d in range(nd)]
    y = [[model.addVar(lb=-GRB.INFINITY, name=f"y_{d}_{k}")
          for k in range(nu + 1)] for d in range(nd)]
    vx = [[model.addVar(lb=-GRB.INFINITY, name=f"vx_{d}_{k}")
           for k in range(nu + 1)] for d in range(nd)]
    vy = [[model.addVar(lb=-GRB.INFINITY, name=f"vy_{d}_{k}")
           for k in range(nu + 1)] for d in range(nd)]

    ux = [[model.addVar(lb=-GRB.INFINITY, name=f"ux_{d}_{h}")
           for h in range(nu)] for d in range(nd)]
    uy = [[model.addVar(lb=-GRB.INFINITY, name=f"uy_{d}_{h}")
           for h in range(nu)] for d in range(nd)]
    ex = [[model.addVar(lb=0.0, name=f"ex_{d}_{h}")
           for h in range(nu)] for d in range(nd)]
    ey = [[model.addVar(lb=0.0, name=f"ey_{d}_{h}")
           for h in range(nu)] for d in range(nd)]

    decay = math.exp(-tu)
    vel_u = 1.0 - decay
    pos_v = 1.0 - decay
    pos_u = tu - 1.0 + decay

    obstacle_binary = [
        [
            [model.addVar(vtype=GRB.BINARY, name=f"obs_{d}_{k}_{s}")
             for s in range(4)]
            for k in range(1, nu + 1)
        ]
        for d in range(nd)
    ]

    for d in range(nd):
        defender = instance["defenders"][d]
        model.addConstr(x[d][0] == float(defender["x_s"]))
        model.addConstr(y[d][0] == float(defender["y_s"]))
        model.addConstr(vx[d][0] == float(defender["xdot_s"]))
        model.addConstr(vy[d][0] == float(defender["ydot_s"]))

        for h in range(nu):
            model.addConstr(vx[d][h + 1] == decay * vx[d][h] + vel_u * ux[d][h])
            model.addConstr(vy[d][h + 1] == decay * vy[d][h] + vel_u * uy[d][h])
            model.addConstr(
                x[d][h + 1] == x[d][h] + pos_v * vx[d][h] + pos_u * ux[d][h]
            )
            model.addConstr(
                y[d][h + 1] == y[d][h] + pos_v * vy[d][h] + pos_u * uy[d][h]
            )

            for nx, ny in control_normals:
                model.addConstr(nx * ux[d][h] + ny * uy[d][h] <= control_rhs)

            model.addConstr(ex[d][h] >= ux[d][h])
            model.addConstr(ex[d][h] >= -ux[d][h])
            model.addConstr(ey[d][h] >= uy[d][h])
            model.addConstr(ey[d][h] >= -uy[d][h])

        for k in range(1, nu + 1):
            for s, (nx, ny) in enumerate(obstacle_normals):
                model.addGenConstrIndicator(
                    obstacle_binary[d][k - 1][s],
                    True,
                    nx * x[d][k] + ny * y[d][k] >= r_dz,
                )
            model.addConstr(gp.quicksum(obstacle_binary[d][k - 1]) >= 1)

    defender_px = [[None] * (n_a + 1) for _ in range(nd)]
    defender_py = [[None] * (n_a + 1) for _ in range(nd)]

    for d in range(nd):
        for k in range(n_a + 1):
            t = k * ta
            if t <= 1e-12:
                defender_px[d][k] = x[d][0]
                defender_py[d][k] = y[d][0]
            elif t >= nu * tu - 1e-12:
                defender_px[d][k] = x[d][nu]
                defender_py[d][k] = y[d][nu]
            else:
                h = min(int(math.floor(t / tu)), nu - 1)
                tau = t - h * tu
                local_decay = math.exp(-tau)
                local_pos_v = 1.0 - local_decay
                local_pos_u = tau - 1.0 + local_decay
                defender_px[d][k] = (
                    x[d][h] + local_pos_v * vx[d][h] + local_pos_u * ux[d][h]
                )
                defender_py[d][k] = (
                    y[d][h] + local_pos_v * vy[d][h] + local_pos_u * uy[d][h]
                )

    attacker_p = []
    attacker_q = []
    active = []
    gamma = []
    delta = []

    geometry_tolerance = 1e-7

    for j in range(na):
        p_j = [
            model.addVar(lb=-GRB.INFINITY, name=f"p_{j}_{k}")
            for k in range(n_a + 1)
        ]
        q_j = [
            model.addVar(lb=-GRB.INFINITY, name=f"q_{j}_{k}")
            for k in range(n_a + 1)
        ]
        a_j = [
            model.addVar(vtype=GRB.BINARY, name=f"a_{j}_{k}")
            for k in range(n_a + 1)
        ]
        g_j = [
            model.addVar(vtype=GRB.BINARY, name=f"gamma_{j}_{k}")
            for k in range(n_a + 1)
        ]
        d_j = [
            [
                model.addVar(vtype=GRB.BINARY, name=f"delta_{j}_{d}_{k}")
                for k in range(n_a + 1)
            ]
            for d in range(nd)
        ]

        attacker_p.append(p_j)
        attacker_q.append(q_j)
        active.append(a_j)
        gamma.append(g_j)
        delta.append(d_j)

        attacker = instance["attackers"][j]
        vp = float(attacker["v_p"])
        vq = float(attacker["v_q"])

        model.addConstr(p_j[0] == float(attacker["p_s"]))
        model.addConstr(q_j[0] == float(attacker["q_s"]))

        for k in range(n_a):
            model.addConstr(p_j[k + 1] == p_j[k] + ta * vp * a_j[k])
            model.addConstr(q_j[k + 1] == q_j[k] + ta * vq * a_j[k])

        for k in range(n_a + 1):
            zone_violations = []
            for s, (nx, ny) in enumerate(dz_normals):
                z = model.addVar(vtype=GRB.BINARY, name=f"gz_{j}_{k}_{s}")
                zone_violations.append(z)
                expression = nx * p_j[k] + ny * q_j[k]
                model.addGenConstrIndicator(
                    g_j[k], True, expression <= r_dz + geometry_tolerance
                )
                model.addGenConstrIndicator(
                    z, True, expression >= r_dz + geometry_tolerance
                )
                model.addConstr(z <= 1 - g_j[k])
            model.addConstr(gp.quicksum(zone_violations) >= 1 - g_j[k])

            for d in range(nd):
                intercept_violations = []
                rel_x = p_j[k] - defender_px[d][k]
                rel_y = q_j[k] - defender_py[d][k]

                for s, (nx, ny) in enumerate(intercept_normals):
                    z = model.addVar(
                        vtype=GRB.BINARY, name=f"dz_{j}_{d}_{k}_{s}"
                    )
                    intercept_violations.append(z)
                    expression = nx * rel_x + ny * rel_y
                    model.addGenConstrIndicator(
                        d_j[d][k],
                        True,
                        expression <= r_i + geometry_tolerance,
                    )
                    model.addGenConstrIndicator(
                        z,
                        True,
                        expression >= r_i + geometry_tolerance,
                    )
                    model.addConstr(z <= 1 - d_j[d][k])

                model.addConstr(
                    gp.quicksum(intercept_violations) >= 1 - d_j[d][k]
                )

            events = g_j[k] + gp.quicksum(d_j[d][k] for d in range(nd))
            if k == 0:
                model.addConstr(a_j[k] <= 1 - g_j[k])
                for d in range(nd):
                    model.addConstr(a_j[k] <= 1 - d_j[d][k])
                model.addConstr(a_j[k] >= 1 - events)
            else:
                model.addConstr(a_j[k] <= a_j[k - 1])
                model.addConstr(a_j[k] <= 1 - g_j[k])
                for d in range(nd):
                    model.addConstr(a_j[k] <= 1 - d_j[d][k])
                model.addConstr(a_j[k] >= a_j[k - 1] - events)

    objective = gp.quicksum(
        gamma[j][k] for j in range(na) for k in range(1, n_a + 1)
    )
    if epsilon > 0.0:
        objective += epsilon * gp.quicksum(
            ex[d][h] + ey[d][h] for d in range(nd) for h in range(nu)
        )
    model.setObjective(objective, GRB.MINIMIZE)

    if best_controls is not None:
        for d in range(nd):
            for h in range(nu):
                ux[d][h].Start = best_controls[d][h][0]
                uy[d][h].Start = best_controls[d][h][1]

    flat_control_vars = []
    for d in range(nd):
        for h in range(nu):
            flat_control_vars.append(ux[d][h])
            flat_control_vars.append(uy[d][h])

    callback_state = {
        "best_objective": best_objective if controls_are_feasible(
            instance, best_controls
        ) else float("inf"),
        "best_solution": best_solution if controls_are_feasible(
            instance, best_controls
        ) else None,
        "best_controls": best_controls if controls_are_feasible(
            instance, best_controls
        ) else None,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = cb_model.cbGetSolution(flat_control_vars)
            candidate_controls = [
                [[0.0, 0.0] for _ in range(nu)] for _ in range(nd)
            ]
            index = 0
            for d in range(nd):
                for h in range(nu):
                    candidate_controls[d][h][0] = float(values[index])
                    candidate_controls[d][h][1] = float(values[index + 1])
                    index += 2

            if not controls_are_feasible(instance, candidate_controls):
                return

            candidate_solution = build_solution(instance, candidate_controls)
            candidate_objective = candidate_solution["objective_value"]

            if candidate_objective < callback_state["best_objective"] - 1e-9:
                callback_state["best_objective"] = candidate_objective
                callback_state["best_solution"] = candidate_solution
                callback_state["best_controls"] = candidate_controls
                if logger:
                    logger.log_solution(candidate_objective, candidate_solution)
        except Exception:
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_controls = [
            [[0.0, 0.0] for _ in range(nu)] for _ in range(nd)
        ]
        for d in range(nd):
            for h in range(nu):
                final_controls[d][h][0] = float(ux[d][h].X)
                final_controls[d][h][1] = float(uy[d][h].X)

        if controls_are_feasible(instance, final_controls):
            final_solution = build_solution(instance, final_controls)
            if final_solution["objective_value"] < callback_state["best_objective"] - 1e-9:
                callback_state["best_objective"] = final_solution["objective_value"]
                callback_state["best_solution"] = final_solution
                callback_state["best_controls"] = final_controls
                if logger:
                    logger.log_solution(
                        final_solution["objective_value"], final_solution
                    )

    if callback_state["best_solution"] is not None:
        output_solution = callback_state["best_solution"]
    else:
        output_solution = best_solution

    if output_solution is None:
        fallback_controls = heuristic_controls(instance, True)
        output_solution = build_solution(instance, fallback_controls)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(output_solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()