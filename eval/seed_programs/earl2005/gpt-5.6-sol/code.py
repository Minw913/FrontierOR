import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


TOL = 1e-7


def polygon_normals(side_count):
    return [
        (
            math.cos(2.0 * math.pi * side / side_count),
            math.sin(2.0 * math.pi * side / side_count),
        )
        for side in range(side_count)
    ]


def inside_polygon(x, y, normals, apothem, tolerance=TOL):
    return all(
        nx * x + ny * y <= apothem + tolerance
        for nx, ny in normals
    )


def control_on_ray(dx, dy, normals, rhs, strength=1.0):
    length = math.hypot(dx, dy)
    if length <= 1e-12 or strength <= 0.0:
        return 0.0, 0.0

    dx /= length
    dy /= length

    maximum_scale = float("inf")
    for nx, ny in normals:
        projection = nx * dx + ny * dy
        if projection > 1e-12:
            maximum_scale = min(maximum_scale, rhs / projection)

    if not math.isfinite(maximum_scale):
        return 0.0, 0.0

    scale = max(0.0, min(1.0, strength)) * maximum_scale
    return scale * dx, scale * dy


def project_control(ux, uy, normals, rhs):
    maximum = max(nx * ux + ny * uy for nx, ny in normals)
    if maximum <= rhs + 1e-12:
        return ux, uy
    if maximum <= 1e-12:
        return 0.0, 0.0
    scale = rhs / maximum
    return scale * ux, scale * uy


def obstacle_margin(x, y, r_dz):
    # The four-side regular polygon used by the released formulation has
    # normals (1,0), (0,1), (-1,0), and (0,-1).
    return max(abs(x), abs(y)) - r_dz


def control_vertices(side_count):
    # Intersection vertices of the half-space representation whose normals
    # are at angles 2*pi*j/M and whose RHS is cos(pi/M).
    return [
        (
            math.cos((2.0 * j + 1.0) * math.pi / side_count),
            math.sin((2.0 * j + 1.0) * math.pi / side_count),
        )
        for j in range(side_count)
    ]


def next_state(x, y, vx, vy, ux, uy, coefficients):
    decay, vel_u, pos_v, pos_u = coefficients
    next_x = x + pos_v * vx + pos_u * ux
    next_y = y + pos_v * vy + pos_u * uy
    next_vx = decay * vx + vel_u * ux
    next_vy = decay * vy + vel_u * uy
    return next_x, next_y, next_vx, next_vy


def candidate_controls(
    desired_dx,
    desired_dy,
    strength,
    x,
    y,
    vx,
    vy,
    instance,
    normals,
    rhs,
):
    candidates = [(0.0, 0.0)]
    vertices = control_vertices(instance["M_u"])

    desired_angle = math.atan2(desired_dy, desired_dx)
    if math.hypot(desired_dx, desired_dy) > 1e-10:
        for angular_offset in (
            0.0,
            math.pi / 12.0,
            -math.pi / 12.0,
            math.pi / 6.0,
            -math.pi / 6.0,
            math.pi / 3.0,
            -math.pi / 3.0,
            math.pi / 2.0,
            -math.pi / 2.0,
        ):
            dx = math.cos(desired_angle + angular_offset)
            dy = math.sin(desired_angle + angular_offset)
            for multiplier in (strength, 0.65 * strength, 0.35 * strength):
                candidates.append(
                    control_on_ray(dx, dy, normals, rhs, multiplier)
                )

    # Braking and velocity cancellation directions.
    if math.hypot(vx, vy) > 1e-10:
        candidates.append(control_on_ray(-vx, -vy, normals, rhs, 1.0))
        candidates.append(control_on_ray(-vx, -vy, normals, rhs, 0.5))

    # Controls that move directly away from the obstacle.
    if abs(x) >= abs(y):
        outward = (1.0 if x >= 0.0 else -1.0, 0.0)
    else:
        outward = (0.0, 1.0 if y >= 0.0 else -1.0)

    candidates.append(control_on_ray(outward[0], outward[1], normals, rhs, 1.0))
    candidates.append(control_on_ray(outward[0], outward[1], normals, rhs, 0.5))

    for ux, uy in vertices:
        candidates.append((ux, uy))
        candidates.append((0.5 * ux, 0.5 * uy))

    result = []
    seen = set()
    for ux, uy in candidates:
        ux, uy = project_control(ux, uy, normals, rhs)
        key = (round(ux, 10), round(uy, 10))
        if key not in seen:
            seen.add(key)
            result.append((ux, uy))
    return result


def choose_control(
    x,
    y,
    vx,
    vy,
    target,
    strength,
    instance,
    coefficients,
    control_normals,
    control_rhs,
):
    r_dz = float(instance["R_dz"])

    if target is None:
        desired_dx = 0.0
        desired_dy = 0.0
    else:
        desired_dx = target[0] - x
        desired_dy = target[1] - y

    candidates = candidate_controls(
        desired_dx,
        desired_dy,
        strength,
        x,
        y,
        vx,
        vy,
        instance,
        control_normals,
        control_rhs,
    )

    best_control = None
    best_score = float("inf")

    for ux, uy in candidates:
        nx, ny, nvx, nvy = next_state(
            x, y, vx, vy, ux, uy, coefficients
        )
        margin = obstacle_margin(nx, ny, r_dz)
        if margin < -2e-7:
            continue

        # Check whether coasting for one more step would immediately enter
        # the obstacle. This encourages early braking or outward motion.
        cx, cy, _, _ = next_state(
            nx, ny, nvx, nvy, 0.0, 0.0, coefficients
        )
        coast_margin = obstacle_margin(cx, cy, r_dz)

        effort = abs(ux) + abs(uy)
        safety_penalty = 0.0
        if coast_margin < 0.0:
            safety_penalty += 40.0 * (-coast_margin)
        if margin < 0.03:
            safety_penalty += 5.0 * (0.03 - margin)

        if target is None:
            score = effort + safety_penalty
        else:
            distance = math.hypot(target[0] - nx, target[1] - ny)
            target_length = math.hypot(target[0] - nx, target[1] - ny)
            velocity_reward = 0.0
            if target_length > 1e-10:
                velocity_reward = (
                    nvx * (target[0] - nx) + nvy * (target[1] - ny)
                ) / target_length

            score = (
                distance
                - 0.18 * velocity_reward
                + 0.002 * effort
                + safety_penalty
            )

        if score < best_score:
            best_score = score
            best_control = (ux, uy)

    if best_control is not None:
        return best_control

    # If none of the ordinary controls is feasible, use the control which
    # maximizes the next boundary's obstacle margin.
    best_margin = -float("inf")
    fallback = (0.0, 0.0)
    for ux, uy in candidates:
        nx, ny, _, _ = next_state(
            x, y, vx, vy, ux, uy, coefficients
        )
        margin = obstacle_margin(nx, ny, r_dz)
        if margin > best_margin:
            best_margin = margin
            fallback = (ux, uy)
    return fallback


def predicted_attacker_position(instance, attacker_index, sample_index):
    attacker = instance["attackers"][attacker_index]
    t = float(instance["T_a"]) * sample_index
    return (
        float(attacker["p_s"]) + t * float(attacker["v_p"]),
        float(attacker["q_s"]) + t * float(attacker["v_q"]),
    )


def make_control_plan(instance, assignments, hit_samples, strengths):
    nd = int(instance["N_D"])
    nu = int(instance["N_u"])
    ta = float(instance["T_a"])
    tu = float(instance["T_u"])

    decay = math.exp(-tu)
    coefficients = (
        decay,
        1.0 - decay,
        1.0 - decay,
        tu - 1.0 + decay,
    )

    control_normals = polygon_normals(int(instance["M_u"]))
    control_rhs = math.cos(math.pi / int(instance["M_u"]))

    controls = []

    for d in range(nd):
        defender = instance["defenders"][d]
        x = float(defender["x_s"])
        y = float(defender["y_s"])
        vx = float(defender["xdot_s"])
        vy = float(defender["ydot_s"])

        assignment = assignments[d]
        hit_sample = hit_samples[d]
        strength = strengths[d]

        if assignment is None:
            target = None
            target_time = -1.0
        else:
            target = predicted_attacker_position(
                instance, assignment, hit_sample
            )
            target_time = hit_sample * ta

        defender_controls = []

        for h in range(nu):
            current_time = h * tu

            if target is not None and current_time > target_time + tu:
                current_target = None
            else:
                current_target = target

            ux, uy = choose_control(
                x,
                y,
                vx,
                vy,
                current_target,
                strength,
                instance,
                coefficients,
                control_normals,
                control_rhs,
            )
            defender_controls.append([ux, uy])
            x, y, vx, vy = next_state(
                x, y, vx, vy, ux, uy, coefficients
            )

        controls.append(defender_controls)

    return controls


def compute_defender_trajectories(instance, controls):
    nd = int(instance["N_D"])
    nu = int(instance["N_u"])
    tu = float(instance["T_u"])

    decay = math.exp(-tu)
    coefficients = (
        decay,
        1.0 - decay,
        1.0 - decay,
        tu - 1.0 + decay,
    )

    trajectories = []

    for d in range(nd):
        defender = instance["defenders"][d]
        x = [float(defender["x_s"])]
        y = [float(defender["y_s"])]
        xdot = [float(defender["xdot_s"])]
        ydot = [float(defender["ydot_s"])]
        ux_values = []
        uy_values = []

        for h in range(nu):
            ux = float(controls[d][h][0])
            uy = float(controls[d][h][1])
            ux_values.append(ux)
            uy_values.append(uy)

            nx, ny, nvx, nvy = next_state(
                x[h],
                y[h],
                xdot[h],
                ydot[h],
                ux,
                uy,
                coefficients,
            )
            x.append(nx)
            y.append(ny)
            xdot.append(nvx)
            ydot.append(nvy)

        trajectories.append(
            {
                "x": x,
                "y": y,
                "xdot": xdot,
                "ydot": ydot,
                "ux": ux_values,
                "uy": uy_values,
            }
        )

    return trajectories


def defender_position_at_time(trajectory, time_value, tu):
    nu = len(trajectory["ux"])
    horizon = nu * tu

    if time_value <= 1e-12:
        return trajectory["x"][0], trajectory["y"][0]
    if time_value >= horizon - 1e-12:
        return trajectory["x"][nu], trajectory["y"][nu]

    h = min(int(math.floor(time_value / tu)), nu - 1)
    tau = time_value - h * tu

    if tau <= 1e-12:
        return trajectory["x"][h], trajectory["y"][h]

    decay = math.exp(-tau)
    pos_v = 1.0 - decay
    pos_u = tau - 1.0 + decay

    x = (
        trajectory["x"][h]
        + pos_v * trajectory["xdot"][h]
        + pos_u * trajectory["ux"][h]
    )
    y = (
        trajectory["y"][h]
        + pos_v * trajectory["ydot"][h]
        + pos_u * trajectory["uy"][h]
    )
    return x, y


def controls_are_feasible(instance, controls):
    nd = int(instance["N_D"])
    nu = int(instance["N_u"])
    r_dz = float(instance["R_dz"])
    normals = polygon_normals(int(instance["M_u"]))
    rhs = math.cos(math.pi / int(instance["M_u"]))

    for d in range(nd):
        for h in range(nu):
            ux = float(controls[d][h][0])
            uy = float(controls[d][h][1])
            if any(
                nx * ux + ny * uy > rhs + 2e-6
                for nx, ny in normals
            ):
                return False

    trajectories = compute_defender_trajectories(instance, controls)
    for trajectory in trajectories:
        for k in range(1, nu + 1):
            if obstacle_margin(
                trajectory["x"][k], trajectory["y"][k], r_dz
            ) < -2e-6:
                return False

    return True


def build_solution(instance, controls):
    nd = int(instance["N_D"])
    na = int(instance["N_A"])
    n_a = int(instance["N_a"])
    ta = float(instance["T_a"])
    tu = float(instance["T_u"])
    epsilon = float(instance["epsilon"])

    dz_normals = polygon_normals(int(instance["M_dz"]))
    intercept_normals = polygon_normals(int(instance["M_I"]))
    r_dz = float(instance["R_dz"])
    r_i = float(instance["R_I"])

    trajectories = compute_defender_trajectories(instance, controls)

    sampled_defender_positions = [
        [
            defender_position_at_time(
                trajectories[d], k * ta, tu
            )
            for k in range(n_a + 1)
        ]
        for d in range(nd)
    ]

    attackers_output = []
    penetration_count = 0

    for attacker_index in range(na):
        attacker = instance["attackers"][attacker_index]
        vp = float(attacker["v_p"])
        vq = float(attacker["v_q"])

        p = [float(attacker["p_s"])]
        q = [float(attacker["q_s"])]
        active = []
        gamma = []
        delta = [[0.0] * (n_a + 1) for _ in range(nd)]

        was_active = True

        for k in range(n_a + 1):
            if k > 0:
                if was_active:
                    p.append(p[k - 1] + ta * vp)
                    q.append(q[k - 1] + ta * vq)
                else:
                    p.append(p[k - 1])
                    q.append(q[k - 1])

            in_zone = inside_polygon(
                p[k], q[k], dz_normals, r_dz
            )
            gamma.append(1.0 if in_zone else 0.0)

            intercepted = False
            for d in range(nd):
                defender_x, defender_y = sampled_defender_positions[d][k]
                hit = inside_polygon(
                    p[k] - defender_x,
                    q[k] - defender_y,
                    intercept_normals,
                    r_i,
                )
                delta[d][k] = 1.0 if hit else 0.0
                intercepted = intercepted or hit

            current_active = was_active and not in_zone and not intercepted
            active.append(1.0 if current_active else 0.0)
            was_active = current_active

            if k >= 1 and in_zone:
                penetration_count += 1

        attackers_output.append(
            {
                "p": p,
                "q": q,
                "a": active,
                "gamma": gamma,
                "delta": delta,
            }
        )

    effort = sum(
        abs(float(controls[d][h][0]))
        + abs(float(controls[d][h][1]))
        for d in range(nd)
        for h in range(int(instance["N_u"]))
    )

    objective = float(penetration_count + epsilon * effort)

    return {
        "objective_value": objective,
        "defenders": trajectories,
        "attackers": attackers_output,
    }


def attacker_candidate_samples(instance, attacker_index):
    n_a = int(instance["N_a"])
    dz_normals = polygon_normals(int(instance["M_dz"]))
    r_dz = float(instance["R_dz"])
    r_i = float(instance["R_I"])

    result = []
    for k in range(1, n_a + 1):
        x, y = predicted_attacker_position(instance, attacker_index, k)

        # Prefer interception samples before the attacker is deeply within
        # the defense zone. Samples close to its boundary are retained.
        maximum_violation = max(
            nx * x + ny * y - r_dz
            for nx, ny in dz_normals
        )
        if maximum_violation >= -r_i:
            result.append(k)

    if not result:
        result = list(range(1, n_a + 1))

    if len(result) <= 12:
        return result

    selected = {result[0], result[-1]}
    for index in range(1, 11):
        selected.add(result[round(index * (len(result) - 1) / 11)])
    return sorted(selected)


def safe_emergency_plan(instance):
    nd = int(instance["N_D"])
    assignments = [None] * nd
    hit_samples = [1] * nd
    strengths = [1.0] * nd
    return make_control_plan(
        instance, assignments, hit_samples, strengths
    )


def register_candidate(
    instance,
    controls,
    state,
    logger,
):
    if not controls_are_feasible(instance, controls):
        return

    solution = build_solution(instance, controls)
    objective = float(solution["objective_value"])

    if objective < state["best_objective"] - 1e-9:
        state["best_objective"] = objective
        state["best_solution"] = solution
        state["best_controls"] = controls
        if logger:
            logger.log_solution(objective, solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    nd = int(instance["N_D"])
    na = int(instance["N_A"])
    n_a = int(instance["N_a"])

    state = {
        "best_objective": float("inf"),
        "best_solution": None,
        "best_controls": None,
    }

    # Always construct a deterministic safety-oriented incumbent first.
    emergency_controls = safe_emergency_plan(instance)
    register_candidate(
        instance, emergency_controls, state, logger
    )

    # Also try the zero-control plan, which is important when fuel is
    # penalized and the defenders' initial trajectories are already safe.
    zero_controls = [
        [[0.0, 0.0] for _ in range(int(instance["N_u"]))]
        for _ in range(nd)
    ]
    register_candidate(instance, zero_controls, state, logger)

    sample_options = [
        attacker_candidate_samples(instance, j)
        for j in range(na)
    ]

    # Deterministic single-target plans.
    if na > 0:
        for attacker_index in range(na):
            for hit_sample in sample_options[attacker_index]:
                if time.monotonic() >= deadline:
                    break

                for strength in (1.0, 0.75, 0.5):
                    assignments = [attacker_index] * nd
                    hit_samples = [hit_sample] * nd
                    strengths = [strength] * nd
                    controls = make_control_plan(
                        instance,
                        assignments,
                        hit_samples,
                        strengths,
                    )
                    register_candidate(
                        instance, controls, state, logger
                    )

            if time.monotonic() >= deadline:
                break

    # Spread defenders among attackers in round-robin arrangements.
    if na > 0 and time.monotonic() < deadline:
        for offset in range(na):
            assignments = [
                (d + offset) % na for d in range(nd)
            ]
            for quantile in (0.25, 0.5, 0.75):
                hit_samples = []
                for attacker_index in assignments:
                    options = sample_options[attacker_index]
                    index = min(
                        len(options) - 1,
                        int(quantile * len(options)),
                    )
                    hit_samples.append(options[index])

                controls = make_control_plan(
                    instance,
                    assignments,
                    hit_samples,
                    [1.0] * nd,
                )
                register_candidate(
                    instance, controls, state, logger
                )

                if time.monotonic() >= deadline:
                    break
            if time.monotonic() >= deadline:
                break

    # Randomized assignment and interception-time search. The fixed seed
    # keeps results reproducible.
    rng = random.Random(0)
    iteration = 0

    while na > 0 and time.monotonic() < deadline:
        assignments = []
        hit_samples = []
        strengths = []

        for d in range(nd):
            if rng.random() < 0.08:
                assignments.append(None)
                hit_samples.append(1)
                strengths.append(1.0)
                continue

            attacker_index = rng.randrange(na)
            assignments.append(attacker_index)

            options = sample_options[attacker_index]
            if state["best_solution"] is not None and rng.random() < 0.4:
                # Bias toward samples near zone arrival.
                option_index = min(
                    len(options) - 1,
                    int((0.55 + 0.45 * rng.random()) * len(options)),
                )
                hit_samples.append(options[option_index])
            else:
                hit_samples.append(rng.choice(options))

            strengths.append(
                rng.choice((0.45, 0.6, 0.75, 0.9, 1.0))
            )

        controls = make_control_plan(
            instance, assignments, hit_samples, strengths
        )
        register_candidate(instance, controls, state, logger)
        iteration += 1

        # Avoid spending excessive time after finding the theoretical
        # zero-penetration and zero-effort lower bound.
        if state["best_objective"] <= 1e-10:
            break

    if state["best_solution"] is None:
        # This branch is only expected for an intrinsically infeasible
        # instance. It still provides a schema-complete result.
        state["best_solution"] = build_solution(
            instance, emergency_controls
        )

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as output_file:
        json.dump(
            state["best_solution"],
            output_file,
            separators=(",", ":"),
            allow_nan=False,
        )


if __name__ == "__main__":
    main()