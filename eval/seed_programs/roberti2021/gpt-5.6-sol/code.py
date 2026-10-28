import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def travel_time(matrix, u, v, destination):
    """Map the duplicated destination depot to matrix node 0."""
    matrix_u = 0 if u == destination else u
    matrix_v = 0 if v == destination else v
    return float(matrix[matrix_u][matrix_v])


def build_initial_route(n, truck_times, drone_times, destination):
    """Nearest-neighbor route followed by symmetric 2-opt improvements."""
    customers = list(range(1, n + 1))

    def synchronized_cost(u, v):
        return max(
            travel_time(truck_times, u, v, destination),
            travel_time(drone_times, u, v, destination),
        )

    route = [0]
    remaining = set(customers)
    current = 0

    while remaining:
        nxt = min(remaining, key=lambda j: (synchronized_cost(current, j), j))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt

    route.append(destination)

    # The stated metrics are symmetric. Verify this before using the
    # constant-time symmetric 2-opt delta.
    symmetric = True
    physical_nodes = list(range(0, n + 1))
    for i in physical_nodes:
        for j in physical_nodes:
            if abs(truck_times[i][j] - truck_times[j][i]) > 1e-9:
                symmetric = False
                break
            if abs(drone_times[i][j] - drone_times[j][i]) > 1e-9:
                symmetric = False
                break
        if not symmetric:
            break

    if symmetric:
        max_passes = min(30, max(1, n))
        for _ in range(max_passes):
            improved = False
            for i in range(1, len(route) - 2):
                a, b = route[i - 1], route[i]
                for k in range(i + 1, len(route) - 1):
                    c, d = route[k], route[k + 1]
                    old_cost = synchronized_cost(a, b) + synchronized_cost(c, d)
                    new_cost = synchronized_cost(a, c) + synchronized_cost(b, d)
                    if new_cost + 1e-9 < old_cost:
                        route[i:k + 1] = reversed(route[i:k + 1])
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    return route


def solution_from_routes(
    truck_route_internal,
    drone_route_internal,
    customer_types,
    n,
    destination,
    truck_times,
    drone_times,
):
    """Compute the earliest common arrival times for two fixed routes."""
    vertex_count = destination + 1
    arrivals = [0.0] * vertex_count
    precedence_edges = []

    for route, matrix in (
        (truck_route_internal, truck_times),
        (drone_route_internal, drone_times),
    ):
        for u, v in zip(route, route[1:]):
            precedence_edges.append(
                (u, v, travel_time(matrix, u, v, destination))
            )

    # Longest paths in the union of the two route precedence graphs.
    # A positive cycle cannot occur in a feasible MIP incumbent.
    changed_on_last = False
    for iteration in range(vertex_count):
        changed = False
        for u, v, cost in precedence_edges:
            candidate = arrivals[u] + cost
            if candidate > arrivals[v] + 1e-9:
                arrivals[v] = candidate
                changed = True
        changed_on_last = changed
        if not changed:
            break

    if changed_on_last:
        # Perform an additional check for a positive precedence cycle.
        for u, v, cost in precedence_edges:
            if arrivals[u] + cost > arrivals[v] + 1e-7:
                raise ValueError("Fixed routes contain an infeasible precedence cycle")

    output_truck_route = [
        0 if node == destination else int(node)
        for node in truck_route_internal
    ]
    output_drone_route = [
        0 if node == destination else int(node)
        for node in drone_route_internal
    ]

    arrival_dict = {
        str(node): float(arrivals[node])
        for node in range(destination + 1)
    }

    return {
        "objective_value": float(arrivals[destination]),
        "truck_route": output_truck_route,
        "drone_route": output_drone_route,
        "customer_types": {
            str(i): customer_types[i] for i in range(1, n + 1)
        },
        "arrival_times": arrival_dict,
    }


def extract_route(arc_values, arcs, destination):
    successors = {}
    for arc, value in zip(arcs, arc_values):
        if value > 0.5:
            u, v = arc
            successors[u] = v

    route = [0]
    current = 0
    seen = {0}

    while current != destination:
        if current not in successors:
            raise ValueError("Incomplete incumbent route")
        current = successors[current]
        route.append(current)

        if current != destination:
            if current in seen:
                raise ValueError("Cycle detected while extracting incumbent route")
            seen.add(current)

        if len(route) > destination + 2:
            raise ValueError("Incumbent route is too long")

    return route


def make_solution_from_values(
    x_values,
    y_values,
    truck_type_values,
    drone_type_values,
    combined_type_values,
    arcs,
    n,
    destination,
    truck_times,
    drone_times,
):
    truck_route = extract_route(x_values, arcs, destination)
    drone_route = extract_route(y_values, arcs, destination)

    customer_types = {}
    for index, customer in enumerate(range(1, n + 1)):
        values = (
            truck_type_values[index],
            drone_type_values[index],
            combined_type_values[index],
        )
        best = max(range(3), key=lambda k: values[k])
        customer_types[customer] = ("truck", "drone", "combined")[best]

    return solution_from_routes(
        truck_route,
        drone_route,
        customer_types,
        n,
        destination,
        truck_times,
        drone_times,
    )


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2, sort_keys=False)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    n = int(instance["num_customers"])
    destination = n + 1
    customers = list(range(1, n + 1))
    truck_times = instance["truck_travel_times"]
    drone_times = instance["drone_travel_times"]

    if n < 2:
        raise ValueError("The problem requires at least two customers")
    if len(truck_times) < n + 1 or len(drone_times) < n + 1:
        raise ValueError("Travel-time matrices do not match num_customers")

    # A guaranteed feasible incumbent: both vehicles follow the same route,
    # and every customer is classified as combined.
    initial_route = build_initial_route(
        n, truck_times, drone_times, destination
    )
    initial_types = {i: "combined" for i in customers}
    best_solution = solution_from_routes(
        initial_route,
        initial_route,
        initial_types,
        n,
        destination,
        truck_times,
        drone_times,
    )
    best_objective = [best_solution["objective_value"]]

    if logger:
        logger.log_solution(best_objective[0], best_solution)

    # Preserve a usable result even if model construction or optimization
    # cannot be completed within the available environment/time.
    if time.monotonic() >= deadline:
        write_solution(args.solution_path, best_solution)
        return

    try:
        model = gp.Model("truck_drone_routing")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        arcs = []
        arcs.extend((0, j) for j in customers)
        arcs.extend(
            (i, j)
            for i in customers
            for j in customers
            if i != j
        )
        arcs.extend((i, destination) for i in customers)

        x = model.addVars(arcs, vtype=GRB.BINARY, name="truck_arc")
        y = model.addVars(arcs, vtype=GRB.BINARY, name="drone_arc")

        truck_only = model.addVars(customers, vtype=GRB.BINARY, name="truck_only")
        drone_only = model.addVars(customers, vtype=GRB.BINARY, name="drone_only")
        combined = model.addVars(customers, vtype=GRB.BINARY, name="combined")

        time_ub = max(1.0, best_objective[0] + 1e-6)
        arrival = model.addVars(
            range(destination + 1),
            lb=0.0,
            ub=time_ub,
            vtype=GRB.CONTINUOUS,
            name="arrival",
        )

        # Separate route-order variables rule out zero-travel-time subtours.
        truck_order = model.addVars(
            customers, lb=0.0, ub=float(n), name="truck_order"
        )
        drone_order = model.addVars(
            customers, lb=0.0, ub=float(n), name="drone_order"
        )

        model.addConstr(arrival[0] == 0.0)

        model.addConstr(gp.quicksum(x[0, j] for j in customers) == 1)
        model.addConstr(
            gp.quicksum(x[i, destination] for i in customers) == 1
        )
        model.addConstr(gp.quicksum(y[0, j] for j in customers) == 1)
        model.addConstr(
            gp.quicksum(y[i, destination] for i in customers) == 1
        )

        for i in customers:
            truck_incoming = x[0, i] + gp.quicksum(
                x[j, i] for j in customers if j != i
            )
            truck_outgoing = x[i, destination] + gp.quicksum(
                x[i, j] for j in customers if j != i
            )
            drone_incoming = y[0, i] + gp.quicksum(
                y[j, i] for j in customers if j != i
            )
            drone_outgoing = y[i, destination] + gp.quicksum(
                y[i, j] for j in customers if j != i
            )

            model.addConstr(
                truck_only[i] + drone_only[i] + combined[i] == 1
            )
            model.addConstr(truck_incoming == truck_outgoing)
            model.addConstr(
                truck_outgoing == truck_only[i] + combined[i]
            )
            model.addConstr(drone_incoming == drone_outgoing)
            model.addConstr(
                drone_outgoing == drone_only[i] + combined[i]
            )

            # The drone may not make a depot-customer-depot singleton trip.
            model.addConstr(y[0, i] + y[i, destination] <= 1)

        for i in customers:
            for j in customers:
                if i >= j:
                    continue
                model.addConstr(
                    y[i, j] + y[j, i] <= combined[i] + combined[j]
                )

        for u, v in arcs:
            truck_cost = travel_time(truck_times, u, v, destination)
            drone_cost = travel_time(drone_times, u, v, destination)

            model.addGenConstrIndicator(
                x[u, v],
                True,
                arrival[v] >= arrival[u] + truck_cost,
            )
            model.addGenConstrIndicator(
                y[u, v],
                True,
                arrival[v] >= arrival[u] + drone_cost,
            )

            if u in customers and v in customers:
                model.addGenConstrIndicator(
                    x[u, v],
                    True,
                    truck_order[v] >= truck_order[u] + 1.0,
                )
                model.addGenConstrIndicator(
                    y[u, v],
                    True,
                    drone_order[v] >= drone_order[u] + 1.0,
                )

        model.setObjective(arrival[destination], GRB.MINIMIZE)

        # Install the guaranteed feasible solution as a MIP start.
        selected_initial_arcs = set(zip(initial_route, initial_route[1:]))
        initial_positions = {
            node: position
            for position, node in enumerate(initial_route)
            if node in customers
        }

        for arc in arcs:
            selected = 1.0 if arc in selected_initial_arcs else 0.0
            x[arc].Start = selected
            y[arc].Start = selected

        for i in customers:
            truck_only[i].Start = 0.0
            drone_only[i].Start = 0.0
            combined[i].Start = 1.0
            truck_order[i].Start = float(initial_positions[i])
            drone_order[i].Start = float(initial_positions[i])

        for node in range(destination + 1):
            arrival[node].Start = float(
                best_solution["arrival_times"][str(node)]
            )

        arc_list = list(arcs)
        x_list = [x[a] for a in arc_list]
        y_list = [y[a] for a in arc_list]
        truck_type_list = [truck_only[i] for i in customers]
        drone_type_list = [drone_only[i] for i in customers]
        combined_type_list = [combined[i] for i in customers]
        callback_variables = (
            x_list
            + y_list
            + truck_type_list
            + drone_type_list
            + combined_type_list
        )

        arc_count = len(arc_list)

        def incumbent_callback(callback_model, where):
            nonlocal best_solution
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = callback_model.cbGetSolution(callback_variables)
                offset = 0

                x_values = values[offset:offset + arc_count]
                offset += arc_count
                y_values = values[offset:offset + arc_count]
                offset += arc_count
                truck_values = values[offset:offset + n]
                offset += n
                drone_values = values[offset:offset + n]
                offset += n
                combined_values = values[offset:offset + n]

                candidate = make_solution_from_values(
                    x_values,
                    y_values,
                    truck_values,
                    drone_values,
                    combined_values,
                    arc_list,
                    n,
                    destination,
                    truck_times,
                    drone_times,
                )
                candidate_objective = candidate["objective_value"]

                if candidate_objective + 1e-7 < best_objective[0]:
                    best_objective[0] = candidate_objective
                    best_solution = candidate
                    if logger:
                        logger.log_solution(candidate_objective, candidate)
            except Exception:
                # A malformed numerical callback incumbent should not prevent
                # Gurobi from continuing or destroy the guaranteed fallback.
                return

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            model.Params.TimeLimit = max(0.001, remaining)
            model.optimize(incumbent_callback)

        if model.SolCount > 0:
            x_values = [var.X for var in x_list]
            y_values = [var.X for var in y_list]
            truck_values = [var.X for var in truck_type_list]
            drone_values = [var.X for var in drone_type_list]
            combined_values = [var.X for var in combined_type_list]

            candidate = make_solution_from_values(
                x_values,
                y_values,
                truck_values,
                drone_values,
                combined_values,
                arc_list,
                n,
                destination,
                truck_times,
                drone_times,
            )
            candidate_objective = candidate["objective_value"]

            if candidate_objective + 1e-7 < best_objective[0]:
                best_objective[0] = candidate_objective
                best_solution = candidate
                if logger:
                    logger.log_solution(candidate_objective, candidate)

    except Exception:
        # The initial all-combined solution remains valid.
        pass

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()