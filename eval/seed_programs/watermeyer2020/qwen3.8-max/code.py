#!/usr/bin/env python3
"""
Solver for the Resource-Constrained Project Scheduling Problem
with Partially Renewable Resources and General Temporal Constraints.
Uses a time-indexed MIP formulation solved with Gurobi, with SSGS heuristic
as warm start and fallback.
"""

import json
import argparse
import time
import sys
from bisect import bisect_left, bisect_right


def compute_bounds(num_activities, d_bar, temporal_constraints):
    """Compute earliest start (ES) and latest start (LS) times via forward/backward passes."""
    ES = [0] * num_activities
    for _ in range(num_activities + 1):
        changed = False
        for tc in temporal_constraints:
            i, j, delta = tc['i'], tc['j'], tc['delta']
            val = ES[i] + delta
            if val > ES[j]:
                ES[j] = val
                changed = True
        if not changed:
            break

    LS = [d_bar] * num_activities
    for _ in range(num_activities + 1):
        changed = False
        for tc in temporal_constraints:
            i, j, delta = tc['i'], tc['j'], tc['delta']
            val = LS[j] - delta
            if val < LS[i]:
                LS[i] = val
                changed = True
        if not changed:
            break

    for i in range(num_activities):
        ES[i] = max(0, ES[i])
        LS[i] = min(d_bar, LS[i])

    return ES, LS


def compute_overlap(periods_sorted, s, p_i):
    """Count periods in sorted list that fall in [s+1, s+p_i]."""
    left = bisect_left(periods_sorted, s + 1)
    right = bisect_right(periods_sorted, s + p_i)
    return right - left


def serial_ssgs(num_activities, d_bar, activities, temporal_constraints,
                resources, ES, LS, order):
    """Serial Schedule Generation Scheme with given activity order."""
    predecessors = {i: [] for i in range(num_activities)}
    for tc in temporal_constraints:
        predecessors[tc['j']].append((tc['i'], tc['delta']))

    sorted_periods = {}
    for r_id, r_data in resources.items():
        sorted_periods[r_id] = sorted(r_data['periods'])

    start_times = {}
    start_times[0] = 0
    resource_consumption = {r_id: 0 for r_id in resources}

    for i in order:
        if i == 0:
            continue
        act_id = str(i)
        p_i = activities.get(act_id, {}).get('processing_time', 0)

        earliest = max(0, ES[i])
        for pred, delta in predecessors[i]:
            if pred in start_times:
                earliest = max(earliest, start_times[pred] + delta)

        best_s = None
        for s in range(earliest, min(LS[i], d_bar) + 1):
            feasible = True
            for r_id, r_data in resources.items():
                demand = activities.get(act_id, {}).get('resource_demands', {}).get(r_id, 0)
                if demand == 0:
                    continue
                overlap = compute_overlap(sorted_periods[r_id], s, p_i)
                if resource_consumption[r_id] + demand * overlap > r_data['capacity']:
                    feasible = False
                    break
            if feasible:
                best_s = s
                break

        if best_s is None:
            best_s = earliest

        start_times[i] = best_s

        for r_id, r_data in resources.items():
            demand = activities.get(act_id, {}).get('resource_demands', {}).get(r_id, 0)
            if demand > 0:
                overlap = compute_overlap(sorted_periods[r_id], best_s, p_i)
                resource_consumption[r_id] += demand * overlap

    return start_times


def check_resource_feasibility(start_times, num_activities, activities, resources):
    """Check if resource constraints are satisfied."""
    for r_id, r_data in resources.items():
        capacity = r_data['capacity']
        periods_set = set(r_data['periods'])
        total = 0
        for i in range(1, num_activities - 1):
            act_id = str(i)
            if act_id not in activities:
                continue
            p_i = activities[act_id]['processing_time']
            demand = activities[act_id]['resource_demands'].get(r_id, 0)
            if demand == 0:
                continue
            si = start_times.get(i, start_times.get(str(i), 0))
            overlap = len([t for t in periods_set if si < t <= si + p_i])
            total += demand * overlap
        if total > capacity:
            return False
    return True


def run_heuristics(instance, ES, LS):
    """Run multiple SSGS heuristics with different priority rules."""
    n = instance['n']
    num_activities = instance['num_activities']
    d_bar = instance['d_bar']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    real_acts = list(range(1, num_activities - 1))

    # Build successor count
    succ_count = {i: 0 for i in range(num_activities)}
    for tc in temporal_constraints:
        succ_count[tc['i']] += 1

    # Priority orders
    orders = []

    # 1. Earliest start
    order1 = sorted(real_acts, key=lambda i: (ES[i], i))
    orders.append([0] + order1 + [num_activities - 1])

    # 2. Latest start (ascending)
    order2 = sorted(real_acts, key=lambda i: (LS[i], i))
    orders.append([0] + order2 + [num_activities - 1])

    # 3. Total float (ascending)
    order3 = sorted(real_acts, key=lambda i: (LS[i] - ES[i], i))
    orders.append([0] + order3 + [num_activities - 1])

    # 4. Most successors first
    order4 = sorted(real_acts, key=lambda i: (-succ_count[i], ES[i], i))
    orders.append([0] + order4 + [num_activities - 1])

    # 5. Largest processing time first
    order5 = sorted(real_acts, key=lambda i: (-activities.get(str(i), {}).get('processing_time', 0), ES[i]))
    orders.append([0] + order5 + [num_activities - 1])

    best_solution = None
    best_obj = float('inf')

    for order in orders:
        st = serial_ssgs(num_activities, d_bar, activities,
                         temporal_constraints, resources, ES, LS, order)
        obj = st.get(num_activities - 1, st.get(str(num_activities - 1), d_bar))

        # Convert keys to int for internal use
        st_int = {int(k) if isinstance(k, str) else k: v for k, v in st.items()}

        if check_resource_feasibility(st_int, num_activities, activities, resources):
            if obj < best_obj:
                best_obj = obj
                best_solution = st_int

    # If no resource-feasible solution found, use the one with smallest objective
    if best_solution is None:
        for order in orders:
            st = serial_ssgs(num_activities, d_bar, activities,
                             temporal_constraints, resources, ES, LS, order)
            obj = st.get(num_activities - 1, st.get(str(num_activities - 1), d_bar))
            st_int = {int(k) if isinstance(k, str) else k: v for k, v in st.items()}
            if obj < best_obj:
                best_obj = obj
                best_solution = st_int

    return best_solution, best_obj


def solve_mip(instance, ES, LS, time_limit, logger=None, warm_start=None):
    """Solve using Gurobi time-indexed MIP formulation."""
    import gurobipy as gp
    from gurobipy import GRB

    num_activities = instance['num_activities']
    d_bar = instance['d_bar']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    model = gp.Model("RCPSP_PR")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_limit))
    model.setParam('OutputFlag', 0)

    # Decision variables: y[i,s] = 1 if activity i starts at time s
    y = {}
    for i in range(num_activities):
        lo, hi = ES[i], LS[i]
        if lo > hi:
            return None
        for s in range(lo, hi + 1):
            y[i, s] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{s}")

    model.update()

    # Each activity assigned exactly one start time
    for i in range(num_activities):
        lo, hi = ES[i], LS[i]
        model.addConstr(
            gp.quicksum(y[i, s] for s in range(lo, hi + 1)) == 1,
            name=f"assign_{i}"
        )

    # Project start at time 0
    if (0, 0) in y:
        model.addConstr(y[0, 0] == 1, name="start_zero")

    # Temporal constraints: S_j - S_i >= delta
    for idx, tc in enumerate(temporal_constraints):
        i, j, delta = tc['i'], tc['j'], tc['delta']
        expr = gp.LinExpr()
        for s in range(ES[j], LS[j] + 1):
            expr += s * y[j, s]
        for s in range(ES[i], LS[i] + 1):
            expr -= s * y[i, s]
        model.addConstr(expr >= delta, name=f"tc_{idx}")

    # Resource constraints
    sorted_periods = {}
    for r_id, r_data in resources.items():
        sorted_periods[r_id] = sorted(r_data['periods'])

    for r_id, r_data in resources.items():
        capacity = r_data['capacity']
        expr = gp.LinExpr()

        for i in range(1, num_activities - 1):
            act_id = str(i)
            if act_id not in activities:
                continue
            p_i = activities[act_id]['processing_time']
            demand = activities[act_id]['resource_demands'].get(r_id, 0)
            if demand == 0:
                continue

            for s in range(ES[i], LS[i] + 1):
                overlap = compute_overlap(sorted_periods[r_id], s, p_i)
                if overlap > 0:
                    expr += demand * overlap * y[i, s]

        model.addConstr(expr <= capacity, name=f"res_{r_id}")

    # Objective: minimize start time of project-end activity
    end_act = num_activities - 1
    obj_expr = gp.LinExpr()
    for s in range(ES[end_act], LS[end_act] + 1):
        obj_expr += s * y[end_act, s]
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Warm start from heuristic
    if warm_start is not None:
        for i in range(num_activities):
            s_val = warm_start.get(i, ES[i])
            s_val = max(ES[i], min(s_val, LS[i]))
            if (i, s_val) in y:
                y[i, s_val].Start = 1.0

    # Callback for logging incumbent solutions
    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 0.5:
                best_obj[0] = obj_val
                start_times = {}
                for i in range(num_activities):
                    for s in range(ES[i], LS[i] + 1):
                        if model.cbGetSolution(y[i, s]) > 0.5:
                            start_times[str(i)] = int(s)
                            break
                    else:
                        start_times[str(i)] = int(ES[i])
                solution = {
                    "objective_value": int(round(obj_val)),
                    "start_times": start_times
                }
                if logger:
                    logger.log_solution(int(round(obj_val)), solution)

    try:
        model.optimize(callback)
    except Exception as e:
        return None

    if model.SolCount > 0:
        start_times = {}
        for i in range(num_activities):
            found = False
            for s in range(ES[i], LS[i] + 1):
                if y[i, s].X > 0.5:
                    start_times[str(i)] = int(s)
                    found = True
                    break
            if not found:
                start_times[str(i)] = int(ES[i])

        objective_value = start_times[str(end_act)]
        return objective_value, start_times

    return None


def main():
    parser = argparse.ArgumentParser(description="RCPSP with Partially Renewable Resources")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    overall_start = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n = instance['n']
    num_activities = instance['num_activities']
    d_bar = instance['d_bar']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    # Compute bounds
    ES, LS = compute_bounds(num_activities, d_bar, temporal_constraints)

    # Run heuristics for initial solution
    heuristic_solution, heuristic_obj = run_heuristics(instance, ES, LS)

    best_solution = None
    if heuristic_solution is not None:
        best_solution = {
            "objective_value": int(heuristic_obj),
            "start_times": {str(k): int(v) for k, v in heuristic_solution.items()}
        }
        if logger:
            logger.log_solution(int(heuristic_obj), best_solution)

    # Determine if MIP is tractable
    total_vars = sum(max(0, LS[i] - ES[i] + 1) for i in range(num_activities))
    elapsed = time.time() - overall_start
    remaining = args.time_limit - elapsed

    # Solve MIP if model size is reasonable and time permits
    if total_vars <= 500000 and remaining > 10:
        try:
            mip_time = remaining - 3  # Reserve time for output
            result = solve_mip(instance, ES, LS, mip_time, logger, heuristic_solution)
            if result is not None:
                obj, start_times = result
                if best_solution is None or obj < best_solution["objective_value"]:
                    best_solution = {
                        "objective_value": int(obj),
                        "start_times": start_times
                    }
                    if logger:
                        logger.log_solution(int(obj), best_solution)
        except Exception:
            pass

    # Fallback: if no solution found yet, use ES-based solution
    if best_solution is None:
        start_times = {str(i): int(ES[i]) for i in range(num_activities)}
        best_solution = {
            "objective_value": int(ES[num_activities - 1]),
            "start_times": start_times
        }
        if logger:
            logger.log_solution(int(ES[num_activities - 1]), best_solution)

    # Write output
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()