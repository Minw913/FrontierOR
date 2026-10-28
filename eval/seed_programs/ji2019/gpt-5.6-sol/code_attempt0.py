import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def forced_modes(ship):
    """Return (force_lock, force_transshipment), supporting common optional fields."""
    force_lock = bool(ship.get("lock_only", False))
    force_trans = bool(
        ship.get("transshipment_only", False)
        or ship.get("quay_only", False)
    )

    value = ship.get("restricted_mode", ship.get("mode_restriction"))
    if isinstance(value, str):
        value = value.lower().replace("-", "_").replace(" ", "_")
        if value in {"lock", "lock_only", "passage"}:
            force_lock = True
        elif value in {"transshipment", "transshipment_only", "quay", "berth"}:
            force_trans = True

    allowed = ship.get("allowed_modes")
    if isinstance(allowed, list):
        normalized = {
            str(v).lower().replace("-", "_").replace(" ", "_")
            for v in allowed
        }
        lock_allowed = bool(normalized & {"0", "lock", "passage", "lock_mode"})
        trans_allowed = bool(
            normalized & {"1", "transshipment", "quay", "berth", "transshipment_mode"}
        )
        if lock_allowed and not trans_allowed:
            force_lock = True
        elif trans_allowed and not lock_allowed:
            force_trans = True

    return force_lock, force_trans


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)
    os.replace(temporary, path)


def main():
    args = parse_args()
    started_at = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ships = data["ships"]
    berths = data["berths"]
    handling = data["handling_times"]

    n = int(data["num_ships"])
    nb = int(data["num_berths"])
    W = float(data["chamber_width_W"])
    L = float(data["chamber_length_L"])
    duration = float(data["lockage_duration_Du"])
    min_interval = float(data["min_time_between_lockages_MT"])
    lock_start = float(data["lock_start_availability_sc"])
    berth_end = float(data["berth_end_availability_ek"])
    cmax = float(data["C_max"])

    ship_ids = [ship["id"] for ship in ships]
    berth_ids = [berth["id"] for berth in berths]
    ship_index = {sid: i for i, sid in enumerate(ship_ids)}
    berth_index = {bid: k for k, bid in enumerate(berth_ids)}

    widths = [float(s["width"]) for s in ships]
    lengths = [float(s["length"]) for s in ships]
    arrivals = [float(s["arrival_time"]) for s in ships]
    penalties = [float(s["transshipment_penalty"]) for s in ships]
    berth_starts = [float(b["start_availability_sk"]) for b in berths]

    eligible = []
    for ship in ships:
        values = []
        for bid in ship.get("eligible_berths_B", []):
            if bid in berth_index:
                values.append(berth_index[bid])
            elif isinstance(bid, int) and 0 <= bid < nb:
                values.append(bid)
        eligible.append(sorted(set(values)))

    mooring_allowed = []
    for ship in ships:
        values = set()
        for sid in ship.get("mooring_set_MR", []):
            if sid in ship_index:
                values.add(ship_index[sid])
            elif isinstance(sid, int) and 0 <= sid < n:
                values.add(sid)
        mooring_allowed.append(values)

    force_lock = [False] * n
    force_trans = [False] * n
    for i, ship in enumerate(ships):
        fl, ft = forced_modes(ship)
        force_lock[i] = fl
        force_trans[i] = ft
        if not eligible[i]:
            force_lock[i] = True
        if widths[i] > W + EPS or lengths[i] > L + EPS:
            force_trans[i] = True

    if any(force_lock[i] and force_trans[i] for i in range(n)):
        raise RuntimeError("Instance contains a ship with conflicting or infeasible mode restrictions.")

    maximum_handling_sum = sum(
        max([float(handling[i][k]) for k in eligible[i]] or [0.0])
        for i in range(n)
    )
    max_time_value = max(
        [0.0, cmax, berth_end, lock_start]
        + arrivals
        + berth_starts
    )
    big_m_time = max(
        1.0,
        max_time_value + maximum_handling_sum + duration * (n + 1)
        + min_interval * (n + 1) + 100.0
    )
    big_m_space = max(
        1.0,
        W + max(widths or [0.0]) + 1.0,
        L + max(lengths or [0.0]) + 1.0
    )

    model = gp.Model("integrated_lock_and_quay")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # z[i] = 1 means transshipment; 0 means lock mode.
    z = model.addVars(n, vtype=GRB.BINARY, name="trans")
    x = model.addVars(n, n, vtype=GRB.BINARY, name="lock_assign")
    used = model.addVars(n, vtype=GRB.BINARY, name="lock_used")
    lock_completion = model.addVars(
        n, lb=0.0, ub=max(cmax, big_m_time), name="lock_completion"
    )
    departure = model.addVars(n, lb=0.0, ub=big_m_time, name="departure")
    lock_delay = model.addVars(n, lb=0.0, ub=big_m_time, name="lock_delay")

    xpos = model.addVars(
        n, lb=0.0, ub=max(0.0, W), vtype=GRB.INTEGER, name="xpos"
    )
    ypos = model.addVars(
        n, lb=0.0, ub=max(0.0, L), vtype=GRB.INTEGER, name="ypos"
    )

    berth_assign = {}
    for i in range(n):
        for k in eligible[i]:
            berth_assign[i, k] = model.addVar(vtype=GRB.BINARY, name=f"berth_{i}_{k}")

    service_start = model.addVars(
        n, lb=0.0, ub=big_m_time, name="service_start"
    )
    service_completion = model.addVars(
        n, lb=0.0, ub=big_m_time, name="service_completion"
    )
    berth_delay = model.addVars(
        n, lb=0.0, ub=big_m_time, name="berth_delay"
    )

    # Mode and assignment constraints.
    for i in range(n):
        if force_lock[i]:
            model.addConstr(z[i] == 0)
        if force_trans[i]:
            model.addConstr(z[i] == 1)

        model.addConstr(gp.quicksum(x[i, t] for t in range(n)) == 1 - z[i])
        model.addConstr(
            gp.quicksum(berth_assign[i, k] for k in eligible[i]) == z[i]
        )

        for t in range(n):
            model.addConstr(x[i, t] <= used[t])

    for t in range(n):
        model.addConstr(gp.quicksum(x[i, t] for i in range(n)) >= used[t])
        if t + 1 < n:
            model.addConstr(used[t] >= used[t + 1])

    # Lock timing and departure times.
    for t in range(n):
        model.addConstr(
            lock_completion[t] >= lock_start + duration
            - big_m_time * (1 - used[t])
        )
        if t > 0:
            model.addConstr(lock_completion[t] >= lock_completion[t - 1])
            model.addConstr(
                lock_completion[t] >= lock_completion[t - 1] + min_interval
                - big_m_time * (1 - used[t])
            )
        if cmax > 0:
            model.addConstr(lock_completion[t] <= cmax)

        for i in range(n):
            model.addConstr(
                lock_completion[t] >= arrivals[i] + duration
                - big_m_time * (1 - x[i, t])
            )
            model.addConstr(
                departure[i] >= lock_completion[t]
                - big_m_time * (1 - x[i, t])
            )
            model.addConstr(
                departure[i] <= lock_completion[t]
                + big_m_time * (1 - x[i, t])
            )

    for i in range(n):
        # A transshipment-mode ship's physical departure time is its arrival.
        model.addConstr(
            departure[i] >= arrivals[i] - big_m_time * (1 - z[i])
        )
        model.addConstr(
            departure[i] <= arrivals[i] + big_m_time * (1 - z[i])
        )
        model.addConstr(
            lock_delay[i] >= departure[i] - arrivals[i] - big_m_time * z[i]
        )
        model.addConstr(lock_delay[i] <= big_m_time * (1 - z[i]))

    lock_number = [
        gp.quicksum((t + 1) * x[i, t] for t in range(n))
        for i in range(n)
    ]

    # FCFS for all strictly earlier arrivals.
    for i in range(n):
        for j in range(n):
            if arrivals[i] + EPS < arrivals[j]:
                model.addConstr(
                    lock_number[i] <= lock_number[j] + n * (z[i] + z[j])
                )
                model.addConstr(
                    departure[i] <= departure[j]
                    + big_m_time * (z[i] + z[j])
                )

    # Chamber bounds.
    for i in range(n):
        model.addConstr(
            xpos[i] + widths[i] <= W + big_m_space * z[i]
        )
        model.addConstr(
            ypos[i] + lengths[i] <= L + big_m_space * z[i]
        )
        model.addConstr(xpos[i] <= W * (1 - z[i]))
        model.addConstr(ypos[i] <= L * (1 - z[i]))

    # Pairwise non-overlap relations. Relations are global because two ships
    # can share at most one lockage.
    packing_relations = {}
    for i in range(n):
        for j in range(i + 1, n):
            relations = []

            if widths[i] + widths[j] <= W + EPS:
                rij = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
                rji = model.addVar(vtype=GRB.BINARY, name=f"left_{j}_{i}")
                model.addConstr(
                    xpos[i] + widths[i] <= xpos[j] + big_m_space * (1 - rij)
                )
                model.addConstr(
                    xpos[j] + widths[j] <= xpos[i] + big_m_space * (1 - rji)
                )
                relations.extend([rij, rji])

            if lengths[i] + lengths[j] <= L + EPS:
                rij = model.addVar(vtype=GRB.BINARY, name=f"behind_{i}_{j}")
                rji = model.addVar(vtype=GRB.BINARY, name=f"behind_{j}_{i}")
                model.addConstr(
                    ypos[i] + lengths[i] <= ypos[j] + big_m_space * (1 - rij)
                )
                model.addConstr(
                    ypos[j] + lengths[j] <= ypos[i] + big_m_space * (1 - rji)
                )
                relations.extend([rij, rji])

            packing_relations[i, j] = relations
            for t in range(n):
                if relations:
                    model.addConstr(
                        gp.quicksum(relations) >= x[i, t] + x[j, t] - 1
                    )
                else:
                    model.addConstr(x[i, t] + x[j, t] <= 1)

    # Mooring variables and constraints.
    wall_left = model.addVars(n, vtype=GRB.BINARY, name="wall_left")
    wall_right = model.addVars(n, vtype=GRB.BINARY, name="wall_right")
    moor_left = {}
    moor_right = {}

    for i in range(n):
        model.addConstr(wall_left[i] <= 1 - z[i])
        model.addConstr(wall_right[i] <= 1 - z[i])
        model.addConstr(xpos[i] <= big_m_space * (1 - wall_left[i]))
        model.addConstr(
            xpos[i] + widths[i] >= W - big_m_space * (1 - wall_right[i])
        )
        model.addConstr(
            xpos[i] + widths[i] <= W + big_m_space * (1 - wall_right[i])
        )

        attachments = [wall_left[i], wall_right[i]]

        for j in sorted(mooring_allowed[i]):
            if j == i:
                continue

            ml = model.addVar(vtype=GRB.BINARY, name=f"moor_left_{i}_{j}")
            mr = model.addVar(vtype=GRB.BINARY, name=f"moor_right_{i}_{j}")
            moor_left[i, j] = ml
            moor_right[i, j] = mr
            attachments.extend([ml, mr])

            for mvar in (ml, mr):
                model.addConstr(mvar <= 1 - z[i])
                model.addConstr(mvar <= 1 - z[j])
                model.addConstr(
                    lock_number[i] - lock_number[j] <= n * (1 - mvar)
                )
                model.addConstr(
                    lock_number[j] - lock_number[i] <= n * (1 - mvar)
                )
                model.addConstr(
                    ypos[i] >= ypos[j] - big_m_space * (1 - mvar)
                )
                model.addConstr(
                    ypos[i] + lengths[i] <= ypos[j] + lengths[j]
                    + big_m_space * (1 - mvar)
                )

            # i is directly to the left of j.
            model.addConstr(
                xpos[i] + widths[i] - xpos[j] <= big_m_space * (1 - ml)
            )
            model.addConstr(
                xpos[j] - xpos[i] - widths[i] <= big_m_space * (1 - ml)
            )

            # i is directly to the right of j.
            model.addConstr(
                xpos[j] + widths[j] - xpos[i] <= big_m_space * (1 - mr)
            )
            model.addConstr(
                xpos[i] - xpos[j] - widths[j] <= big_m_space * (1 - mr)
            )

        model.addConstr(gp.quicksum(attachments) >= 1 - z[i])

    for (i, j), ml in moor_left.items():
        reverse = moor_right.get((j, i))
        if reverse is not None:
            model.addConstr(ml + reverse <= 1)

    # Berth service equations and availability windows.
    for i in range(n):
        handling_expr = gp.quicksum(
            float(handling[i][k]) * berth_assign[i, k]
            for k in eligible[i]
        )
        model.addConstr(service_completion[i] == service_start[i] + handling_expr)

        model.addConstr(
            service_start[i] >= arrivals[i] - big_m_time * (1 - z[i])
        )
        for k in eligible[i]:
            model.addConstr(
                service_start[i] >= berth_starts[k]
                - big_m_time * (1 - berth_assign[i, k])
            )

        model.addConstr(
            service_completion[i] <= berth_end + big_m_time * (1 - z[i])
        )
        model.addConstr(
            berth_delay[i] >= service_completion[i] - arrivals[i]
            - big_m_time * (1 - z[i])
        )
        model.addConstr(berth_delay[i] <= big_m_time * z[i])

    # Pairwise berth sequencing.
    berth_orders = {}
    for k in range(nb):
        candidates = [i for i in range(n) if (i, k) in berth_assign]
        for pos_i in range(len(candidates)):
            i = candidates[pos_i]
            for pos_j in range(pos_i + 1, len(candidates)):
                j = candidates[pos_j]
                order = model.addVar(
                    vtype=GRB.BINARY, name=f"berth_order_{i}_{j}_{k}"
                )
                berth_orders[i, j, k] = order
                both_relax = 2 - berth_assign[i, k] - berth_assign[j, k]

                model.addConstr(
                    service_start[j] >= service_completion[i]
                    - big_m_time * (1 - order)
                    - big_m_time * both_relax
                )
                model.addConstr(
                    service_start[i] >= service_completion[j]
                    - big_m_time * order
                    - big_m_time * both_relax
                )

    objective = gp.quicksum(
        lock_delay[i] + berth_delay[i] + penalties[i] * z[i]
        for i in range(n)
    )
    model.setObjective(objective, GRB.MINIMIZE)

    model.update()

    def construct_solution(value_getter):
        modes = {}
        completion_times = {}
        lockage_assignments = {}
        lockage_completion_times = {}
        berth_assignments_out = {}

        selected_berth = {}
        selected_lock = {}

        for i, sid in enumerate(ship_ids):
            is_trans = value_getter(z[i]) > 0.5
            modes[str(sid)] = 1 if is_trans else 0

            if is_trans:
                completion_times[str(sid)] = float(arrivals[i])
                candidates = eligible[i]
                if not candidates:
                    continue
                k = max(
                    candidates,
                    key=lambda kk: value_getter(berth_assign[i, kk])
                )
                selected_berth[i] = k
            else:
                t = max(range(n), key=lambda tt: value_getter(x[i, tt]))
                selected_lock[i] = t
                completion_times[str(sid)] = float(value_getter(lock_completion[t]))
                lockage_assignments[str(sid)] = t + 1

        used_lockages = sorted(set(selected_lock.values()))
        for t in used_lockages:
            lockage_completion_times[str(t + 1)] = float(
                value_getter(lock_completion[t])
            )

        for k in range(nb):
            assigned = [i for i, kk in selected_berth.items() if kk == k]
            assigned.sort(
                key=lambda i: (
                    value_getter(service_start[i]),
                    value_getter(service_completion[i]),
                    ship_ids[i],
                )
            )
            for order_position, i in enumerate(assigned, start=1):
                berth_assignments_out[str(ship_ids[i])] = {
                    "berth": berth_ids[k],
                    "order": order_position,
                }

        objective_value = 0.0
        for i in range(n):
            if modes[str(ship_ids[i])] == 0:
                objective_value += (
                    completion_times[str(ship_ids[i])] - arrivals[i]
                )
            else:
                objective_value += (
                    value_getter(service_completion[i])
                    - arrivals[i]
                    + penalties[i]
                )

        if abs(objective_value) < 1e-9:
            objective_value = 0.0

        return {
            "objective_value": float(objective_value),
            "modes": modes,
            "completion_times": completion_times,
            "lockage_assignments": lockage_assignments,
            "lockage_completion_times": lockage_completion_times,
            "berth_assignments": berth_assignments_out,
        }

    # Construct a partial MIP start: use transshipment only when required,
    # and give every remaining ship an individual lockage.
    start_trans = {i for i in range(n) if force_trans[i]}
    start_lock = [i for i in range(n) if i not in start_trans]
    start_lock.sort(key=lambda i: (arrivals[i], ship_ids[i]))

    berth_available = berth_starts[:]
    start_berth = {}
    start_service = {}
    heuristic_valid = True

    for i in sorted(start_trans, key=lambda q: (arrivals[q], ship_ids[q])):
        options = []
        for k in eligible[i]:
            begin = max(arrivals[i], berth_starts[k], berth_available[k])
            finish = begin + float(handling[i][k])
            if finish <= berth_end + EPS:
                options.append((finish, begin, k))
        if not options:
            heuristic_valid = False
            break
        finish, begin, k = min(options)
        start_berth[i] = k
        start_service[i] = begin
        berth_available[k] = finish

    start_lock_completion = {}
    previous_completion = None
    if heuristic_valid:
        for t, i in enumerate(start_lock):
            completion = max(lock_start + duration, arrivals[i] + duration)
            if previous_completion is not None:
                completion = max(completion, previous_completion + min_interval)
            if cmax > 0 and completion > cmax + EPS:
                heuristic_valid = False
                break
            start_lock_completion[t] = completion
            previous_completion = completion

    if heuristic_valid:
        for i in range(n):
            z[i].Start = 1.0 if i in start_trans else 0.0
            xpos[i].Start = 0.0
            ypos[i].Start = 0.0
            wall_left[i].Start = 0.0 if i in start_trans else 1.0
            wall_right[i].Start = 0.0
            for t in range(n):
                x[i, t].Start = 0.0
            for k in eligible[i]:
                berth_assign[i, k].Start = (
                    1.0 if start_berth.get(i) == k else 0.0
                )

        for t in range(n):
            used[t].Start = 1.0 if t < len(start_lock) else 0.0
            if t < len(start_lock):
                i = start_lock[t]
                x[i, t].Start = 1.0
                lock_completion[t].Start = start_lock_completion[t]
                departure[i].Start = start_lock_completion[t]
                lock_delay[i].Start = start_lock_completion[t] - arrivals[i]
            elif previous_completion is not None:
                lock_completion[t].Start = previous_completion

        for i in start_trans:
            service_start[i].Start = start_service[i]
            service_completion[i].Start = (
                start_service[i] + float(handling[i][start_berth[i]])
            )
            departure[i].Start = arrivals[i]
            lock_delay[i].Start = 0.0
            berth_delay[i].Start = (
                start_service[i]
                + float(handling[i][start_berth[i]])
                - arrivals[i]
            )

        for i in start_lock:
            service_start[i].Start = 0.0
            service_completion[i].Start = 0.0
            berth_delay[i].Start = 0.0

        for (i, j, k), order_var in berth_orders.items():
            if start_berth.get(i) == k and start_berth.get(j) == k:
                order_var.Start = 1.0 if start_service[i] <= start_service[j] else 0.0

        for variable in moor_left.values():
            variable.Start = 0.0
        for variable in moor_right.values():
            variable.Start = 0.0
        for relations in packing_relations.values():
            for variable in relations:
                variable.Start = 0.0

    best_logged = [math.inf]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj + 1e-7 >= best_logged[0]:
                return

            def getter(var):
                return cb_model.cbGetSolution(var)

            solution = construct_solution(getter)
            best_logged[0] = solution["objective_value"]
            if logger:
                logger.log_solution(solution["objective_value"], solution)
        except Exception:
            # Logging must never interrupt optimization.
            pass

    elapsed = time.monotonic() - started_at
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining
    model.optimize(callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    final_solution = construct_solution(lambda var: var.X)

    if logger and final_solution["objective_value"] + 1e-7 < best_logged[0]:
        logger.log_solution(final_solution["objective_value"], final_solution)

    write_solution(args.solution_path, final_solution)


if __name__ == "__main__":
    main()