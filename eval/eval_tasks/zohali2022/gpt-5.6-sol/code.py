import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def normalize_precedences(relations, n):
    """Use zero-based internally. Explicit occurrence of n indicates one-based input."""
    if not relations:
        return []

    one_based = any(int(a) == n or int(b) == n for a, b in relations)
    result = []
    seen = set()

    for a, b in relations:
        a, b = int(a), int(b)
        if one_based:
            a -= 1
            b -= 1
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Invalid precedence relation: {[a, b]}")
        if a == b:
            continue
        if (a, b) not in seen:
            seen.add((a, b))
            result.append((a, b))
    return result


def station_cost(sequence, processing, forward, backward):
    if not sequence:
        return math.inf
    total = sum(processing[i] for i in sequence)
    for i in range(len(sequence) - 1):
        total += forward[sequence[i]][sequence[i + 1]]
    total += backward[sequence[-1]][sequence[0]]
    return float(total)


def make_solution(sequences, processing, forward, backward):
    assignment = {}
    station_sequences = {}
    loads = []

    for s, seq in enumerate(sequences, start=1):
        station_sequences[str(s)] = [i + 1 for i in seq]
        for i in seq:
            assignment[str(i + 1)] = s
        loads.append(station_cost(seq, processing, forward, backward))

    objective = float(max(loads)) if loads else 0.0
    return {
        "objective_value": objective,
        "assignment": assignment,
        "station_sequences": station_sequences,
        "cycle_time": objective,
    }


def topological_order(n, successors, indegree):
    indeg = indegree[:]
    ready = [i for i in range(n) if indeg[i] == 0]
    ready.sort()
    order = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for nxt in successors[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    if len(order) != n:
        raise ValueError("The precedence graph contains a directed cycle")
    return order


def randomized_topological_order(
    n,
    successors,
    indegree,
    processing,
    forward,
    downstream_weight,
    rng,
    mode,
):
    indeg = indegree[:]
    ready = [i for i in range(n) if indeg[i] == 0]
    order = []
    previous = None

    while ready:
        if mode == 0:
            chosen = min(ready)
        elif mode == 1:
            chosen = max(
                ready,
                key=lambda x: (
                    downstream_weight[x],
                    processing[x],
                    len(successors[x]),
                    -x,
                ),
            )
        elif mode == 2:
            chosen = max(
                ready,
                key=lambda x: (
                    processing[x],
                    downstream_weight[x],
                    rng.random(),
                ),
            )
        elif mode == 3 and previous is not None:
            chosen = min(
                ready,
                key=lambda x: (
                    forward[previous][x]
                    - 0.15 * downstream_weight[x]
                    - 0.05 * processing[x],
                    rng.random(),
                ),
            )
        else:
            chosen = ready[rng.randrange(len(ready))]

        ready.remove(chosen)
        order.append(chosen)
        previous = chosen

        for nxt in successors[chosen]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)

    if len(order) != n:
        raise ValueError("The precedence graph contains a directed cycle")
    return order


def optimal_partition(order, m, processing, forward, backward, deadline=None):
    """Minimax partition of a topological order into m nonempty consecutive blocks."""
    n = len(order)
    if m > n or m <= 0:
        return None

    proc_prefix = [0.0] * (n + 1)
    forward_prefix = [0.0] * (n + 1)

    for k in range(n):
        proc_prefix[k + 1] = proc_prefix[k] + processing[order[k]]
        if k > 0:
            forward_prefix[k + 1] = (
                forward_prefix[k] + forward[order[k - 1]][order[k]]
            )

    def segment_cost(left, right):
        # Segment is order[left:right], right > left.
        process_sum = proc_prefix[right] - proc_prefix[left]
        internal_forward = forward_prefix[right] - forward_prefix[left + 1]
        return (
            process_sum
            + internal_forward
            + backward[order[right - 1]][order[left]]
        )

    inf = float("inf")
    previous = [inf] * (n + 1)
    previous[0] = 0.0
    parents = [[-1] * (n + 1) for _ in range(m + 1)]

    for station_count in range(1, m + 1):
        current = [inf] * (n + 1)
        min_j = station_count
        max_j = n - (m - station_count)

        for j in range(min_j, max_j + 1):
            best = inf
            best_left = -1

            for left in range(station_count - 1, j):
                if previous[left] == inf:
                    continue
                value = max(previous[left], segment_cost(left, j))
                if value < best:
                    best = value
                    best_left = left

            current[j] = best
            parents[station_count][j] = best_left

            if (
                deadline is not None
                and (j & 31) == 0
                and time.monotonic() >= deadline
            ):
                return None

        previous = current

    if previous[n] == inf:
        return None

    sequences = []
    right = n
    for station_count in range(m, 0, -1):
        left = parents[station_count][right]
        if left < 0:
            return None
        sequences.append(order[left:right])
        right = left
    sequences.reverse()
    return sequences


def improve_station_orders(
    sequences,
    edge_set,
    processing,
    forward,
    backward,
    rng,
    deadline,
):
    """Fast adjacent-swap descent. Adjacent incomparable tasks may be exchanged."""
    result = [seq[:] for seq in sequences]

    for s in range(len(result)):
        seq = result[s]
        if len(seq) <= 1:
            continue

        best_cost = station_cost(seq, processing, forward, backward)
        improved = True
        passes = 0

        while improved and passes < 4 and time.monotonic() < deadline:
            improved = False
            passes += 1
            positions = list(range(len(seq) - 1))
            rng.shuffle(positions)

            for p in positions:
                a, b = seq[p], seq[p + 1]
                # Since they are adjacent in a topological order, a direct a->b
                # edge is the only possible precedence preventing this swap.
                if (a, b) in edge_set:
                    continue

                seq[p], seq[p + 1] = b, a
                new_cost = station_cost(seq, processing, forward, backward)
                if new_cost + 1e-9 < best_cost:
                    best_cost = new_cost
                    improved = True
                else:
                    seq[p], seq[p + 1] = a, b

                if time.monotonic() >= deadline:
                    break

    return result


def solve_with_gurobi(
    n,
    m,
    processing,
    precedence,
    forward,
    backward,
    upper_bound,
    warm_sequences,
    deadline,
    logger,
    incumbent_holder,
):
    if gp is None:
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0.5:
        return

    model = gp.Model("assembly_line_balancing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.1, remaining)

    stations = range(m)
    tasks = range(n)

    x = model.addVars(tasks, stations, vtype=GRB.BINARY, name="x")
    first = model.addVars(tasks, stations, vtype=GRB.BINARY, name="first")
    last = model.addVars(tasks, stations, vtype=GRB.BINARY, name="last")
    pos = model.addVars(
        tasks, stations, lb=0.0, ub=max(0, n - 1), vtype=GRB.CONTINUOUS, name="pos"
    )

    arc_keys = [(i, j, s) for s in stations for i in tasks for j in tasks if i != j]
    arcs = model.addVars(arc_keys, vtype=GRB.BINARY, name="arc")

    return_arc = model.addVars(
        [(i, j, s) for s in stations for i in tasks for j in tasks],
        vtype=GRB.BINARY,
        name="return",
    )

    cycle_ub = GRB.INFINITY
    if upper_bound is not None and float(upper_bound) >= 0:
        cycle_ub = float(upper_bound)
    cycle = model.addVar(lb=0.0, ub=cycle_ub, vtype=GRB.CONTINUOUS, name="cycle")

    for i in tasks:
        model.addConstr(gp.quicksum(x[i, s] for s in stations) == 1)

    for s in stations:
        model.addConstr(gp.quicksum(x[i, s] for i in tasks) >= 1)
        model.addConstr(gp.quicksum(first[i, s] for i in tasks) == 1)
        model.addConstr(gp.quicksum(last[i, s] for i in tasks) == 1)

        for i in tasks:
            outgoing = gp.quicksum(arcs[i, j, s] for j in tasks if j != i)
            incoming = gp.quicksum(arcs[j, i, s] for j in tasks if j != i)
            model.addConstr(outgoing + last[i, s] == x[i, s])
            model.addConstr(incoming + first[i, s] == x[i, s])

            model.addConstr(
                gp.quicksum(return_arc[i, j, s] for j in tasks) == last[i, s]
            )
            model.addConstr(
                gp.quicksum(return_arc[j, i, s] for j in tasks) == first[i, s]
            )

        for i in tasks:
            for j in tasks:
                if i != j:
                    model.addConstr(
                        pos[j, s] >= pos[i, s] + 1 - n * (1 - arcs[i, j, s])
                    )

        load = gp.quicksum(processing[i] * x[i, s] for i in tasks)
        load += gp.quicksum(
            forward[i][j] * arcs[i, j, s]
            for i in tasks
            for j in tasks
            if i != j
        )
        load += gp.quicksum(
            backward[i][j] * return_arc[i, j, s] for i in tasks for j in tasks
        )
        model.addConstr(cycle >= load)

    for i, j in precedence:
        model.addConstr(
            gp.quicksum((s + 1) * x[i, s] for s in stations)
            <= gp.quicksum((s + 1) * x[j, s] for s in stations)
        )
        for s in stations:
            model.addConstr(
                pos[j, s]
                >= pos[i, s] + 1 - n * (2 - x[i, s] - x[j, s])
            )

    model.setObjective(cycle, GRB.MINIMIZE)

    # MIP start.
    warm_objective = 0.0
    for s, seq in enumerate(warm_sequences):
        load = station_cost(seq, processing, forward, backward)
        warm_objective = max(warm_objective, load)

        for p, i in enumerate(seq):
            x[i, s].Start = 1.0
            pos[i, s].Start = float(p)

        first[seq[0], s].Start = 1.0
        last[seq[-1], s].Start = 1.0
        return_arc[seq[-1], seq[0], s].Start = 1.0

        for p in range(len(seq) - 1):
            arcs[seq[p], seq[p + 1], s].Start = 1.0

    cycle.Start = warm_objective

    callback_best = [incumbent_holder["objective"]]

    def extract_callback_solution(cb_model):
        sequences = [[] for _ in stations]
        for s in stations:
            selected = []
            for i in tasks:
                xv = cb_model.cbGetSolution(x[i, s])
                if xv > 0.5:
                    pv = cb_model.cbGetSolution(pos[i, s])
                    selected.append((pv, i))
            selected.sort()
            sequences[s] = [i for _, i in selected]

        if any(not seq for seq in sequences):
            return None
        if sum(len(seq) for seq in sequences) != n:
            return None
        return make_solution(sequences, processing, forward, backward)

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            solution = extract_callback_solution(cb_model)
            if solution is None:
                return
            objective = solution["objective_value"]
            if objective + 1e-7 < callback_best[0]:
                callback_best[0] = objective
                incumbent_holder["objective"] = objective
                incumbent_holder["solution"] = solution
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            pass

    try:
        model.optimize(callback)
    except gp.GurobiError:
        return

    if model.SolCount <= 0:
        return

    sequences = [[] for _ in stations]
    for s in stations:
        selected = [(pos[i, s].X, i) for i in tasks if x[i, s].X > 0.5]
        selected.sort()
        sequences[s] = [i for _, i in selected]

    if any(not seq for seq in sequences):
        return
    if sum(len(seq) for seq in sequences) != n:
        return

    solution = make_solution(sequences, processing, forward, backward)
    objective = solution["objective_value"]
    if objective + 1e-7 < incumbent_holder["objective"]:
        incumbent_holder["objective"] = objective
        incumbent_holder["solution"] = solution
        if logger:
            logger.log_solution(objective, solution)


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_tasks"])
    m = int(data["num_stations"])
    processing = [float(v) for v in data["processing_times"]]
    forward = [[float(v) for v in row] for row in data["forward_setup_times"]]
    backward = [[float(v) for v in row] for row in data["backward_setup_times"]]
    upper_bound = data.get("cycle_time")

    if n <= 0 or m <= 0 or m > n:
        raise ValueError("A feasible instance requires 1 <= num_stations <= num_tasks")
    if len(processing) != n or len(forward) != n or len(backward) != n:
        raise ValueError("Invalid task or setup matrix dimensions")
    if any(len(row) != n for row in forward + backward):
        raise ValueError("Setup matrices must be square num_tasks by num_tasks")

    precedence = normalize_precedences(data.get("precedence_relations", []), n)
    successors = [[] for _ in range(n)]
    indegree = [0] * n
    edge_set = set(precedence)

    for i, j in precedence:
        successors[i].append(j)
        indegree[j] += 1

    base_order = topological_order(n, successors, indegree)

    downstream_weight = [0.0] * n
    for i in reversed(base_order):
        if successors[i]:
            downstream_weight[i] = processing[i] + max(
                downstream_weight[j] for j in successors[i]
            )
        else:
            downstream_weight[i] = processing[i]

    initial_sequences = optimal_partition(
        base_order, m, processing, forward, backward, deadline=None
    )
    if initial_sequences is None:
        raise ValueError("Could not construct an initial feasible solution")

    best_solution = make_solution(initial_sequences, processing, forward, backward)
    best_sequences = [seq[:] for seq in initial_sequences]
    incumbent = {
        "objective": best_solution["objective_value"],
        "solution": best_solution,
    }

    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    rng = random.Random(0)

    # Reserve most of the budget for exact MIP on moderate-size instances.
    model_size = 2 * m * n * n + 5 * m * n
    use_mip = gp is not None and model_size <= 35000 and args.time_limit >= 2
    heuristic_budget = (
        min(max(0.25, args.time_limit * 0.18), 4.0)
        if use_mip
        else max(0.0, args.time_limit - 0.15)
    )
    heuristic_deadline = min(deadline - 0.1, start_time + heuristic_budget)

    iteration = 0
    while time.monotonic() < heuristic_deadline:
        mode = iteration % 5
        order = randomized_topological_order(
            n,
            successors,
            indegree,
            processing,
            forward,
            downstream_weight,
            rng,
            mode,
        )
        sequences = optimal_partition(
            order, m, processing, forward, backward, deadline=heuristic_deadline
        )
        if sequences is None:
            break

        sequences = improve_station_orders(
            sequences,
            edge_set,
            processing,
            forward,
            backward,
            rng,
            heuristic_deadline,
        )
        solution = make_solution(sequences, processing, forward, backward)

        if solution["objective_value"] + 1e-9 < incumbent["objective"]:
            incumbent["objective"] = solution["objective_value"]
            incumbent["solution"] = solution
            best_sequences = [seq[:] for seq in sequences]
            if logger:
                logger.log_solution(solution["objective_value"], solution)

        iteration += 1

    if use_mip and time.monotonic() < deadline - 0.25:
        solve_with_gurobi(
            n=n,
            m=m,
            processing=processing,
            precedence=precedence,
            forward=forward,
            backward=backward,
            upper_bound=upper_bound,
            warm_sequences=best_sequences,
            deadline=deadline - 0.05,
            logger=logger,
            incumbent_holder=incumbent,
        )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(incumbent["solution"], f, indent=2)


if __name__ == "__main__":
    main()