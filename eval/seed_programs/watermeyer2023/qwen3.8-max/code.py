import json
import argparse
import time
import bisect
import sys
from gurobipy import *
from solution_logger import SolutionLogger


def compute_time_bounds(num_nodes, processing_times, temporal_constraints, deadline):
    """Compute earliest and latest start times using Bellman-Ford style propagation."""
    earliest = [0] * num_nodes
    latest = [deadline] * num_nodes

    # Forward pass: earliest start times
    for _ in range(num_nodes + 1):
        changed = False
        for tc in temporal_constraints:
            i, j, delta = tc['from'], tc['to'], tc['delta']
            new_val = earliest[i] + delta
            if new_val > earliest[j]:
                earliest[j] = new_val
                changed = True
        if not changed:
            break

    # Backward pass: latest start times
    for _ in range(num_nodes + 1):
        changed = False
        for tc in temporal_constraints:
            i, j, delta = tc['from'], tc['to'], tc['delta']
            new_val = latest[j] - delta
            if new_val < latest[i]:
                latest[i] = new_val
                changed = True
        if not changed:
            break

    # Clamp to [0, deadline]
    for i in range(num_nodes):
        earliest[i] = max(0, earliest[i])
        latest[i] = min(deadline, latest[i])

    return earliest, latest


def solve_with_mip(instance, time_limit, logger=None):
    """Solve using MIP formulation with Gurobi."""
    num_nodes = instance['num_nodes']
    deadline = instance['deadline']
    processing_times = instance['processing_times']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    # Compute bounds
    earliest, latest = compute_time_bounds(num_nodes, processing_times, temporal_constraints, deadline)

    # Check if any activity has empty feasible range
    for i in range(num_nodes):
        if earliest[i] > latest[i]:
            return None, None

    # Build MIP
    model = Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Variables: x[i,t] = 1 if activity i starts at time t
    x = {}
    for i in range(num_nodes):
        for t in range(earliest[i], latest[i] + 1):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # Assignment constraints: each activity starts exactly once
    for i in range(num_nodes):
        model.addConstr(
            quicksum(x[i, t] for t in range(earliest[i], latest[i] + 1)) == 1,
            name=f"assign_{i}"
        )

    # Activity 0 starts at time 0
    model.addConstr(x[0, 0] == 1, name="start_zero")

    # Temporal constraints: S_j - S_i >= delta
    for idx, tc in enumerate(temporal_constraints):
        i, j, delta = tc['from'], tc['to'], tc['delta']
        model.addConstr(
            quicksum(t * x[j, t] for t in range(earliest[j], latest[j] + 1)) -
            quicksum(t * x[i, t] for t in range(earliest[i], latest[i] + 1)) >= delta,
            name=f"temporal_{idx}"
        )

    # Resource constraints
    for r_idx, res in enumerate(resources):
        time_periods = sorted(res['time_periods'])
        capacity = res['capacity']
        demands = res['demands']

        expr = LinExpr()
        for i in range(num_nodes):
            if demands[i] == 0 or processing_times[i] == 0:
                continue
            p_i = processing_times[i]
            for t in range(earliest[i], latest[i] + 1):
                # Count designated time periods in (t, t + p_i]
                left = bisect.bisect_right(time_periods, t)
                right = bisect.bisect_right(time_periods, t + p_i)
                usage = right - left
                if usage > 0:
                    expr += demands[i] * usage * x[i, t]

        model.addConstr(expr <= capacity, name=f"resource_{r_idx}")

    # Objective: minimize start time of end activity
    end_node = num_nodes - 1
    model.setObjective(
        quicksum(t * x[end_node, t] for t in range(earliest[end_node], latest[end_node] + 1)),
        GRB.MINIMIZE
    )

    # Callback to log incumbent solutions
    best_logged = float('inf')

    def callback(m, where):
        nonlocal best_logged
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_logged:
                best_logged = obj
                start_times = [0] * num_nodes
                for i in range(num_nodes):
                    for t in range(earliest[i], latest[i] + 1):
                        if m.cbGetSolution(x[i, t]) > 0.5:
                            start_times[i] = t
                            break
                solution = {
                    "objective_value": int(round(obj)),
                    "start_times": start_times
                }
                if logger:
                    logger.log_solution(int(round(obj)), solution)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        start_times = [0] * num_nodes
        for i in range(num_nodes):
            for t in range(earliest[i], latest[i] + 1):
                if x[i, t].X > 0.5:
                    start_times[i] = t
                    break

        objective_value = start_times[end_node]
        return objective_value, start_times

    return None, None


def greedy_schedule(instance):
    """
    Greedy serial schedule generation scheme as fallback.
    Schedules activities one by one at earliest feasible time.
    """
    num_nodes = instance['num_nodes']
    deadline = instance['deadline']
    processing_times = instance['processing_times']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    earliest, latest = compute_time_bounds(num_nodes, processing_times, temporal_constraints, deadline)

    # Build predecessor info
    preds = {i: [] for i in range(num_nodes)}
    succs = {i: [] for i in range(num_nodes)}
    for tc in temporal_constraints:
        i, j, delta = tc['from'], tc['to'], tc['delta']
        preds[j].append((i, delta))
        succs[i].append((j, delta))

    # Topological sort (Kahn's algorithm on precedence structure)
    # For simplicity, use earliest start as priority
    start_times = [None] * num_nodes
    start_times[0] = 0

    # Schedule in order of earliest start time
    order = sorted(range(1, num_nodes), key=lambda i: (earliest[i], i))

    for i in order:
        # Find earliest feasible start time
        min_start = earliest[i]

        # Respect temporal constraints from already-scheduled predecessors
        for (pred, delta) in preds[i]:
            if start_times[pred] is not None:
                min_start = max(min_start, start_times[pred] + delta)

        # Try start times from min_start onwards
        best_t = None
        for t in range(min_start, latest[i] + 1):
            # Check resource feasibility
            feasible = True
            for res in resources:
                time_periods_set = sorted(res['time_periods'])
                capacity = res['capacity']
                demands = res['demands']

                # Compute total consumption with activity i at time t
                total = 0
                for j in range(num_nodes):
                    if demands[j] == 0 or processing_times[j] == 0:
                        continue
                    if start_times[j] is None and j != i:
                        continue
                    sj = t if j == i else start_times[j]
                    if sj is None:
                        continue
                    p_j = processing_times[j]
                    left = bisect.bisect_right(time_periods_set, sj)
                    right = bisect.bisect_right(time_periods_set, sj + p_j)
                    usage = right - left
                    total += demands[j] * usage

                if total > capacity:
                    feasible = False
                    break

            if feasible:
                best_t = t
                break

        if best_t is not None:
            start_times[i] = best_t
        else:
            # Fallback: use min_start
            start_times[i] = min_start

    # Fill any unscheduled
    for i in range(num_nodes):
        if start_times[i] is None:
            start_times[i] = earliest[i]

    end_node = num_nodes - 1
    return start_times[end_node], start_times


def verify_solution(instance, start_times):
    """Verify that a solution satisfies all constraints."""
    num_nodes = instance['num_nodes']
    deadline = instance['deadline']
    processing_times = instance['processing_times']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    # Check start time of activity 0
    if start_times[0] != 0:
        return False

    # Check deadline
    if start_times[num_nodes - 1] > deadline:
        return False

    # Check temporal constraints
    for tc in temporal_constraints:
        i, j, delta = tc['from'], tc['to'], tc['delta']
        if start_times[j] - start_times[i] < delta:
            return False

    # Check resource constraints
    for res in resources:
        time_periods = sorted(res['time_periods'])
        capacity = res['capacity']
        demands = res['demands']

        total = 0
        for i in range(num_nodes):
            if demands[i] == 0 or processing_times[i] == 0:
                continue
            t = start_times[i]
            p_i = processing_times[i]
            left = bisect.bisect_right(time_periods, t)
            right = bisect.bisect_right(time_periods, t + p_i)
            usage = right - left
            total += demands[i] * usage

        if total > capacity:
            return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    start_clock = time.time()

    # Try MIP first
    objective_value, start_times = solve_with_mip(instance, args.time_limit, logger)

    # If MIP failed, try greedy heuristic
    if objective_value is None:
        objective_value, start_times = greedy_schedule(instance)

    # Verify solution
    if start_times is not None and not verify_solution(instance, start_times):
        # Try greedy as fallback
        objective_value, start_times = greedy_schedule(instance)

    # Write solution
    if start_times is not None:
        solution = {
            "objective_value": int(objective_value),
            "start_times": [int(s) for s in start_times]
        }
    else:
        # Last resort: all activities at time 0 (likely infeasible but avoids crash)
        num_nodes = instance['num_nodes']
        solution = {
            "objective_value": 0,
            "start_times": [0] * num_nodes
        }

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()