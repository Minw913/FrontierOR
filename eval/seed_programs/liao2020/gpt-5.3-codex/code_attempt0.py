import argparse
import json
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_authoritative_cost_structure(cost_structures: Dict) -> Tuple[str, float, float, float]:
    # First declared structure is authoritative
    name, vals = next(iter(cost_structures.items()))
    return name, float(vals["c_w"]), float(vals["c_g"]), float(vals["c_o"])


def compute_effective_parameters(patients: List[Dict], c_w: float, c_g: float, c_o: float):
    # Risk factor: higher waiting/overtime pressure => more conservative durations/lateness
    denom = c_w + c_g + c_o + 1e-9
    rf = min(1.0, max(0.0, (c_w + 0.5 * c_o) / denom))

    s_eff = {}
    u_eff = {}
    for pat in patients:
        p = int(pat["patient_index"])
        q = float(pat["mean_prep_adequacy"])

        muA = float(pat["mean_duration_adequate_prep"])
        muI = float(pat["mean_duration_inadequate_prep"])
        dAL = float(pat["lower_bound_duration_adequate_prep"])
        dAU = float(pat["upper_bound_duration_adequate_prep"])
        dIL = float(pat["lower_bound_duration_inadequate_prep"])
        dIU = float(pat["upper_bound_duration_inadequate_prep"])

        mu_u = float(pat["mean_arrival_time_deviation"])
        uL = float(pat["lower_bound_arrival_time_deviation"])
        uU = float(pat["upper_bound_arrival_time_deviation"])

        mean_d = q * muA + (1.0 - q) * muI
        range_d = q * (dAU - dAL) + (1.0 - q) * (dIU - dIL)
        s_eff[p] = mean_d + 0.5 * rf * max(0.0, range_d)

        range_u = max(0.0, uU - uL)
        u_eff[p] = mu_u + 0.5 * rf * range_u

    return s_eff, u_eff


def heuristic_assignment_schedule(
    patient_ids: List[int], s_eff: Dict[int, float], u_eff: Dict[int, float], L: float
):
    # Simple sequencing heuristic: shorter effective duration first
    order = sorted(patient_ids, key=lambda p: (s_eff[p], u_eff[p], p))
    P = len(order)

    assignment = {p: i + 1 for i, p in enumerate(order)}
    inv = {i + 1: p for i, p in enumerate(order)}

    schedule = {1: 0.0}
    for i in range(2, P + 1):
        prev_p = inv[i - 1]
        inc = max(1.0, s_eff[prev_p] + 0.5 * max(0.0, u_eff[prev_p]))
        schedule[i] = min(float(L), max(schedule[i - 1], schedule[i - 1] + inc))

    return assignment, schedule


def evaluate_cost(
    assignment: Dict[int, int],
    schedule: Dict[int, float],
    s_eff: Dict[int, float],
    u_eff: Dict[int, float],
    c_w: float,
    c_g: float,
    c_o: float,
    L: float,
):
    P = len(assignment)
    pos_to_pat = {pos: p for p, pos in assignment.items()}

    completion_prev = 0.0
    total_wait = 0.0
    total_idle = 0.0

    for i in range(1, P + 1):
        p = pos_to_pat[i]
        t_i = schedule[i]
        arrival = t_i + u_eff[p]
        start = max(t_i, arrival, completion_prev)
        wait = max(0.0, start - t_i)
        if i > 1:
            idle = max(0.0, t_i - completion_prev)
            total_idle += idle
        completion = start + s_eff[p]
        total_wait += wait
        completion_prev = completion

    overtime = max(0.0, completion_prev - L)
    return c_w * total_wait + c_g * total_idle + c_o * overtime


def build_solution_dict(
    objective_value: float,
    cost_name: str,
    c_w: float,
    c_g: float,
    c_o: float,
    assignment: Dict[int, int],
    schedule: Dict[int, float],
):
    P = len(assignment)
    pos_to_pat = {pos: p for p, pos in assignment.items()}
    patient_start_times = {str(p): float(schedule[assignment[p]]) for p in assignment}

    model_variables = {}
    for p in sorted(assignment.keys()):
        i = assignment[p]
        model_variables[f"x[{p},{i}]"] = 1.0
    for i in range(1, P + 1):
        model_variables[f"t[{i}]"] = float(schedule[i])

    sol = {
        "objective_value": float(objective_value),
        "model_variables": model_variables,
        "cost_structure": cost_name,
        "c_w": float(c_w),
        "c_g": float(c_g),
        "c_o": float(c_o),
        "assignment": {str(p): int(assignment[p]) for p in sorted(assignment.keys())},
        "schedule": {str(i): float(schedule[i]) for i in range(1, P + 1)},
        "patient_start_times": patient_start_times,
        "model_variable_name_patterns": {
            "x[p,i]": "patient-position assignment, p and i are 1..P",
            "t[i]": "position start time",
            "rho[i], alpha[i], lambda[i], gamma[i]": "dual variables",
            "beta[i]": "i is 1..P+2",
            "z[i], v[i], r[i], e[i]": "one-dimensional auxiliary variables",
            "eta[p,i], tau[p,i], nu[p,i], Gam[p,i]": "McCormick variables",
            "zeta[p,i], psi[p,i], phi[p,i]": "nonnegative McCormick variables",
        },
    }
    return sol


def solve_instance(instance: Dict, time_limit: int, logger: SolutionLogger = None):
    start_time = time.time()

    P = int(instance["num_patients_P"])
    L = float(instance["provider_service_hours_L_minutes"])
    patients = instance["patients"]

    cost_name, c_w, c_g, c_o = get_authoritative_cost_structure(instance["cost_structures"])

    patient_ids = sorted(int(p["patient_index"]) for p in patients)
    s_eff, u_eff = compute_effective_parameters(patients, c_w, c_g, c_o)

    # Heuristic fallback / warm start
    heur_assign, heur_sched = heuristic_assignment_schedule(patient_ids, s_eff, u_eff, L)
    heur_obj = evaluate_cost(heur_assign, heur_sched, s_eff, u_eff, c_w, c_g, c_o, L)
    best_assign, best_sched, best_obj = heur_assign, heur_sched, heur_obj

    # Log initial heuristic incumbent
    if logger:
        heur_sol = build_solution_dict(heur_obj, cost_name, c_w, c_g, c_o, heur_assign, heur_sched)
        logger.log_solution(heur_obj, heur_sol)

    remaining = max(1, int(time_limit - (time.time() - start_time)))
    if remaining <= 0:
        return build_solution_dict(best_obj, cost_name, c_w, c_g, c_o, best_assign, best_sched)

    # Build MILP surrogate
    m = gp.Model("colonoscopy_surrogate")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = remaining
    m.Params.OutputFlag = 0

    I = range(1, P + 1)

    # Variables
    x = m.addVars(patient_ids, I, vtype=GRB.BINARY, name="x")
    t = m.addVars(I, lb=0.0, ub=L, vtype=GRB.CONTINUOUS, name="t")
    B = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="B")   # actual start
    C = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="C")   # completion
    w = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    g = m.addVars(range(2, P + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="g")
    ot = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="ot")

    # Assignment constraints
    for p in patient_ids:
        m.addConstr(gp.quicksum(x[p, i] for i in I) == 1, name=f"assign_p_{p}")
    for i in I:
        m.addConstr(gp.quicksum(x[p, i] for p in patient_ids) == 1, name=f"assign_i_{i}")

    # Schedule monotonicity and first slot at 0
    m.addConstr(t[1] == 0.0, name="first_time_zero")
    for i in range(2, P + 1):
        m.addConstr(t[i] >= t[i - 1], name=f"mono_{i}")

    # Flow constraints
    for i in I:
        arr_expr = t[i] + gp.quicksum(u_eff[p] * x[p, i] for p in patient_ids)
        dur_expr = gp.quicksum(s_eff[p] * x[p, i] for p in patient_ids)

        m.addConstr(B[i] >= t[i], name=f"B_ge_t_{i}")
        m.addConstr(B[i] >= arr_expr, name=f"B_ge_arr_{i}")
        if i >= 2:
            m.addConstr(B[i] >= C[i - 1], name=f"B_ge_prevC_{i}")

        m.addConstr(C[i] == B[i] + dur_expr, name=f"C_def_{i}")
        m.addConstr(w[i] >= B[i] - t[i], name=f"w_def_{i}")

        if i >= 2:
            m.addConstr(g[i] >= t[i] - C[i - 1], name=f"g_def_{i}")

    m.addConstr(ot >= C[P] - L, name="overtime_def")

    obj = c_w * gp.quicksum(w[i] for i in I)
    if P >= 2:
        obj += c_g * gp.quicksum(g[i] for i in range(2, P + 1))
    obj += c_o * ot
    m.setObjective(obj, GRB.MINIMIZE)

    # Warm start
    for p in patient_ids:
        for i in I:
            x[p, i].Start = 1.0 if heur_assign[p] == i else 0.0
    for i in I:
        t[i].Start = heur_sched[i]

    cb_best = [heur_obj]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if val + 1e-9 < cb_best[0]:
                    cb_best[0] = val
                    logger.log(float(val))
            except Exception:
                pass

    m.optimize(cb)

    if m.SolCount > 0:
        # Extract assignment
        assign = {}
        for p in patient_ids:
            chosen = None
            for i in I:
                if x[p, i].X > 0.5:
                    chosen = i
                    break
            if chosen is None:
                chosen = max(I, key=lambda i: x[p, i].X)
            assign[p] = int(chosen)

        # Repair potential duplicates (rare, tolerance issues)
        used = {}
        for p, i in assign.items():
            used.setdefault(i, []).append(p)

        free_pos = [i for i in I if i not in used]
        if free_pos:
            for i, plist in used.items():
                while len(plist) > 1 and free_pos:
                    p_move = plist.pop()
                    assign[p_move] = free_pos.pop(0)

        sched = {i: float(t[i].X) for i in I}

        # Enforce exact monotonic / bounds cleanup
        sched[1] = 0.0
        for i in range(2, P + 1):
            sched[i] = max(sched[i], sched[i - 1])
            sched[i] = min(L, max(0.0, sched[i]))

        obj_val = evaluate_cost(assign, sched, s_eff, u_eff, c_w, c_g, c_o, L)

        if obj_val < best_obj:
            best_obj, best_assign, best_sched = obj_val, assign, sched
        else:
            # Usually MILP solution is better; still keep best deterministic-evaluated
            best_obj, best_assign, best_sched = obj_val, assign, sched

    solution = build_solution_dict(best_obj, cost_name, c_w, c_g, c_o, best_assign, best_sched)

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve robust outpatient scheduling instance.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, default=None, help="Optional JSONL log path")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()