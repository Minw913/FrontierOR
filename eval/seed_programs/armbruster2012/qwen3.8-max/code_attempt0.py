import json
import argparse
import time
import random
import math
from solution_logger import SolutionLogger


def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def compute_objective(assignment, edges, edge_weights):
    """Compute total cost of cut edges given binary assignment array."""
    total = 0.0
    for idx, (u, v) in enumerate(edges):
        if assignment[u] != assignment[v]:
            total += edge_weights[idx]
    return total


def solve_heuristic(data, time_limit, logger=None):
    """Solve using multi-start local search."""
    num_nodes = data['num_nodes']
    node_weights = data['node_weights']
    edges = data['edges']
    edge_weights = data['edge_weights']
    F = data['bisection_capacity_F']
    total_weight = data['total_node_weight']

    start_time = time.time()

    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, edge_weights[idx]))
        adj[v].append((u, edge_weights[idx]))

    lower_bound = total_weight - F  # minimum weight for S
    upper_bound = F                 # maximum weight for S

    best_assignment = None
    best_obj = float('inf')

    def get_feasible_assignment():
        """Generate a random feasible assignment."""
        assignment = [0] * num_nodes
        assignment[0] = 1
        current_weight = node_weights[0]

        nodes = list(range(1, num_nodes))
        random.shuffle(nodes)

        for node in nodes:
            if current_weight + node_weights[node] <= upper_bound:
                if random.random() < 0.5:
                    assignment[node] = 1
                    current_weight += node_weights[node]

        # Ensure lower bound is met
        if current_weight < lower_bound:
            for node in nodes:
                if assignment[node] == 0 and current_weight + node_weights[node] <= upper_bound:
                    assignment[node] = 1
                    current_weight += node_weights[node]
                    if current_weight >= lower_bound:
                        break

        if current_weight < lower_bound:
            return None, 0

        return assignment, current_weight

    def local_search_pass(assignment, current_weight):
        """One pass of improving moves."""
        obj = compute_objective(assignment, edges, edge_weights)
        improved = False

        order = list(range(1, num_nodes))
        random.shuffle(order)

        for i in order:
            if assignment[i] == 1:
                # Try moving to complement
                new_weight = current_weight - node_weights[i]
                if new_weight < lower_bound:
                    continue
                delta = 0
                for (neighbor, cost) in adj[i]:
                    if assignment[neighbor] == 1:
                        delta += cost
                    else:
                        delta -= cost
                if delta < -1e-10:
                    assignment[i] = 0
                    current_weight = new_weight
                    obj += delta
                    improved = True
            else:
                # Try moving to S
                new_weight = current_weight + node_weights[i]
                if new_weight > upper_bound:
                    continue
                delta = 0
                for (neighbor, cost) in adj[i]:
                    if assignment[neighbor] == 0:
                        delta += cost
                    else:
                        delta -= cost
                if delta < -1e-10:
                    assignment[i] = 1
                    current_weight = new_weight
                    obj += delta
                    improved = True

        return assignment, current_weight, obj, improved

    def swap_search_pass(assignment, current_weight):
        """One pass of swap moves (exchange one node from S with one from complement)."""
        obj = compute_objective(assignment, edges, edge_weights)
        improved = False

        s_nodes = [i for i in range(1, num_nodes) if assignment[i] == 1]
        c_nodes = [i for i in range(1, num_nodes) if assignment[i] == 0]

        if not s_nodes or not c_nodes:
            return assignment, current_weight, obj, improved

        # Sample swaps
        num_attempts = min(len(s_nodes) * len(c_nodes), 500)
        attempts = 0

        random.shuffle(s_nodes)
        random.shuffle(c_nodes)

        for i in s_nodes:
            for j in c_nodes:
                if attempts >= num_attempts:
                    break
                attempts += 1

                weight_change = node_weights[j] - node_weights[i]
                new_weight = current_weight + weight_change
                if new_weight < lower_bound or new_weight > upper_bound:
                    continue

                # Compute delta for swapping i (S->comp) and j (comp->S)
                delta = 0
                for (neighbor, cost) in adj[i]:
                    if neighbor == j:
                        continue
                    if assignment[neighbor] == 1:
                        delta += cost
                    else:
                        delta -= cost
                for (neighbor, cost) in adj[j]:
                    if neighbor == i:
                        continue
                    if assignment[neighbor] == 0:
                        delta += cost
                    else:
                        delta -= cost

                if delta < -1e-10:
                    assignment[i] = 0
                    assignment[j] = 1
                    current_weight = new_weight
                    obj += delta
                    improved = True
            if attempts >= num_attempts:
                break

        return assignment, current_weight, obj, improved

    # Multiple random restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit * 0.85:
            break

        assignment, current_weight = get_feasible_assignment()
        if assignment is None:
            continue

        # Iterative local search
        for _ in range(100):
            assignment, current_weight, obj, improved = local_search_pass(assignment, current_weight)
            if not improved:
                break

        # Try swap search
        for _ in range(20):
            assignment, current_weight, obj, improved = swap_search_pass(assignment, current_weight)
            if not improved:
                break
            # Also do single moves after swaps
            assignment, current_weight, obj, _ = local_search_pass(assignment, current_weight)

        obj = compute_objective(assignment, edges, edge_weights)
        if obj < best_obj:
            best_obj = obj
            best_assignment = assignment[:]
            if logger:
                partition_S = sorted([i for i in range(num_nodes) if best_assignment[i] == 1])
                partition_comp = sorted([i for i in range(num_nodes) if best_assignment[i] == 0])
                solution_dict = {
                    "objective_value": best_obj,
                    "partition_S": partition_S,
                    "partition_complement": partition_comp
                }
                logger.log_solution(best_obj, solution_dict)

    if best_assignment is None:
        # Fallback
        best_assignment = [1] * num_nodes
        best_obj = compute_objective(best_assignment, edges, edge_weights)
        if logger:
            solution_dict = {
                "objective_value": best_obj,
                "partition_S": list(range(num_nodes)),
                "partition_complement": []
            }
            logger.log_solution(best_obj, solution_dict)

    return best_assignment, best_obj


def solve_with_gurobi(data, time_limit, logger=None, warm_start=None):
    """Solve using Gurobi MIP formulation."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, float('inf')

    num_nodes = data['num_nodes']
    node_weights = data['node_weights']
    edges = data['edges']
    edge_weights = data['edge_weights']
    F = data['bisection_capacity_F']
    total_weight = data['total_node_weight']

    model = gp.Model("bisection")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(len(edges), vtype=GRB.BINARY, name="y")

    # Node 0 must be in S
    model.addConstr(x[0] == 1)

    # Bisection weight constraints
    model.addConstr(gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes)) <= F,
                    name="upper_weight")
    model.addConstr(gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes)) >= total_weight - F,
                    name="lower_weight")

    # Edge cut linearization
    for idx, (u, v) in enumerate(edges):
        model.addConstr(y[idx] >= x[u] - x[v], name=f"cut_upper_{idx}")
        model.addConstr(y[idx] >= x[v] - x[u], name=f"cut_lower_{idx}")

    # Objective: minimize total cut cost
    model.setObjective(
        gp.quicksum(edge_weights[idx] * y[idx] for idx in range(len(edges))),
        GRB.MINIMIZE
    )

    # Warm start from heuristic
    if warm_start is not None:
        for i in range(num_nodes):
            x[i].Start = warm_start[i]
        for idx, (u, v) in enumerate(edges):
            y[idx].Start = 1 if warm_start[u] != warm_start[v] else 0

    # Callback to log incumbent solutions
    best_logged_obj = [float('inf')]

    def incumbent_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_logged_obj[0] - 1e-10:
                best_logged_obj[0] = obj
                if logger:
                    try:
                        x_vals = model.cbGetSolution(x)
                        partition_S = sorted([i for i in range(num_nodes) if x_vals[i] > 0.5])
                        partition_comp = sorted([i for i in range(num_nodes) if x_vals[i] <= 0.5])
                        solution_dict = {
                            "objective_value": obj,
                            "partition_S": partition_S,
                            "partition_complement": partition_comp
                        }
                        logger.log_solution(obj, solution_dict)
                    except Exception:
                        logger.log(obj)

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        assignment = [0] * num_nodes
        for i in range(num_nodes):
            assignment[i] = 1 if x[i].X > 0.5 else 0
        obj = compute_objective(assignment, edges, edge_weights)
        return assignment, obj

    return None, float('inf')


def main():
    parser = argparse.ArgumentParser(description="Bisection Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = read_instance(args.instance_path)
    start_time = time.time()

    num_nodes = data['num_nodes']

    # Time allocation: heuristic gets a portion, Gurobi gets the rest
    if args.time_limit <= 10:
        heuristic_time = 1
    elif args.time_limit <= 60:
        heuristic_time = min(5, args.time_limit * 0.2)
    else:
        heuristic_time = min(20, args.time_limit * 0.15)

    # Phase 1: Heuristic
    heuristic_assignment, heuristic_obj = solve_heuristic(data, heuristic_time, logger)

    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    final_assignment = heuristic_assignment
    final_obj = heuristic_obj

    # Phase 2: Gurobi MIP
    if remaining_time > 1:
        gurobi_assignment, gurobi_obj = solve_with_gurobi(
            data, remaining_time, logger, heuristic_assignment
        )

        if gurobi_assignment is not None and gurobi_obj < final_obj:
            final_assignment = gurobi_assignment
            final_obj = gurobi_obj

    # Write final solution
    partition_S = sorted([i for i in range(num_nodes) if final_assignment[i] == 1])
    partition_comp = sorted([i for i in range(num_nodes) if final_assignment[i] == 0])

    solution = {
        "objective_value": final_obj,
        "partition_S": partition_S,
        "partition_complement": partition_comp
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()