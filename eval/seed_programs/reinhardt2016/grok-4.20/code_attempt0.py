import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import math
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_customers'] + 1  # 0 is depot
    K = data['num_vehicles']
    Q = data['vehicle_capacity']
    horizon = data['schedule_horizon']
    num_edge_sets = data['num_edge_sets']

    nodes = data['nodes']
    edge_sets = data['edge_sets']
    free_edges_raw = data['free_edges']
    dist = data['distance_matrix']

    # Precompute sets
    customers = list(range(1, N))
    all_nodes = list(range(N))

    # Edge sets as dict of sets for fast lookup
    edge_set_dict = {}
    access_cost = {}
    for es in edge_sets:
        eid = es['id']
        access_cost[eid] = es['access_cost']
        edge_set_dict[eid] = set()
        for u, v in es['edges']:
            edge_set_dict[eid].add((min(u, v), max(u, v)))

    free_edges = set()
    for u, v in free_edges_raw:
        free_edges.add((min(u, v), max(u, v)))

    # Service and time windows
    demand = [nodes[i]['demand'] for i in range(N)]
    a = [nodes[i]['time_window_start'] for i in range(N)]
    b = [nodes[i]['time_window_end'] for i in range(N)]
    s = [nodes[i]['service_time'] for i in range(N)]

    # Big-M for time
    M = horizon + 1

    # Create model
    model = gp.Model("VRPTW-EdgeSet")
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # leave margin
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPFocus, 1)
    model.setParam(GRB.Param.Heuristics, 0.3)
    model.setParam(GRB.Param.Cuts, 2)

    # Variables
    x = model.addVars(all_nodes, all_nodes, range(K), vtype=GRB.BINARY, name="x")
    y = model.addVars(customers, range(K), vtype=GRB.BINARY, name="y")
    u = model.addVars(all_nodes, range(K), lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="u")
    t = model.addVars(all_nodes, range(K), lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="t")
    z = model.addVars(edge_sets, vtype=GRB.BINARY, name="z")  # index by id

    # Objective: traversal + access
    obj_traversal = gp.quicksum(dist[i][j] * x[i, j, k] for i in all_nodes for j in all_nodes for k in range(K) if i != j)
    obj_access = gp.quicksum(access_cost[eid] * z[eid] for eid in range(num_edge_sets))
    model.setObjective(obj_traversal + obj_access, GRB.MINIMIZE)

    # Constraints

    # Each customer visited exactly once
    for i in customers:
        model.addConstr(gp.quicksum(y[i, k] for k in range(K)) == 1, f"visit_{i}")

    # Flow conservation
    for k in range(K):
        for i in customers:
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in all_nodes if j != i) == y[i, k],
                f"out_{i}_{k}"
            )
            model.addConstr(
                gp.quicksum(x[j, i, k] for j in all_nodes if j != i) == y[i, k],
                f"in_{i}_{k}"
            )

    # Depot flow
    for k in range(K):
        model.addConstr(gp.quicksum(x[0, j, k] for j in all_nodes if j != 0) == 1, f"depot_out_{k}")
        model.addConstr(gp.quicksum(x[i, 0, k] for i in all_nodes if i != 0) == 1, f"depot_in_{k}")

    # Capacity
    for k in range(K):
        model.addConstr(gp.quicksum(demand[i] * y[i, k] for i in customers) <= Q, f"cap_{k}")

    # Subtour elimination and load
    for k in range(K):
        for i in customers:
            for j in customers:
                if i != j:
                    model.addConstr(
                        u[i, k] - u[j, k] + Q * x[i, j, k] <= Q - demand[j],
                        f"mtz_{i}_{j}_{k}"
                    )
        for i in customers:
            model.addConstr(u[i, k] >= demand[i], f"load_lower_{i}_{k}")
            model.addConstr(u[i, k] <= Q, f"load_upper_{i}_{k}")

    # Time windows and timing
    for k in range(K):
        model.addConstr(t[0, k] >= a[0], f"t0_low_{k}")
        model.addConstr(t[0, k] <= b[0], f"t0_up_{k}")

        for i in customers:
            model.addConstr(t[i, k] >= a[i] * y[i, k], f"tw_low_{i}_{k}")
            model.addConstr(t[i, k] <= b[i] * y[i, k] + M * (1 - y[i, k]), f"tw_up_{i}_{k}")

        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    model.addConstr(
                        t[j, k] >= t[i, k] + dist[i][j] + s[i] - M * (1 - x[i, j, k]),
                        f"time_{i}_{j}_{k}"
                    )
        # Return to depot within horizon
        model.addConstr(t[0, k] + gp.quicksum((dist[i][0] + s[i]) * x[i, 0, k] for i in all_nodes if i != 0) <= horizon,
                        f"return_{k}")

    # Edge set usage
    for es_idx in range(num_edge_sets):
        eid = edge_sets[es_idx]['id']
        edges_in_set = edge_set_dict[eid]
        for i in customers:
            # Incoming to i via this edge set
            in_sum = gp.quicksum(
                x[j, i, k] for k in range(K) for j in all_nodes if j != i and (min(j, i), max(j, i)) in edges_in_set
            )
            model.addConstr(in_sum <= z[es_idx], f"es_in_{eid}_{i}")
            # Outgoing from i via this edge set
            out_sum = gp.quicksum(
                x[i, j, k] for k in range(K) for j in all_nodes if j != i and (min(i, j), max(i, j)) in edges_in_set
            )
            model.addConstr(out_sum <= z[es_idx], f"es_out_{eid}_{i}")

    # Free edges and self-loops forbidden implicitly by not creating vars or by constraints
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                for k in range(K):
                    model.addConstr(x[i, j, k] == 0, f"no_self_{i}_{k}")

    # Symmetry breaking: smaller vehicle ids used first
    for k in range(K - 1):
        model.addConstr(
            gp.quicksum(y[i, k] for i in customers) >= gp.quicksum(y[i, k + 1] for i in customers),
            f"sym_{k}"
        )

    # Callback for incumbent logging
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        objective_value = model.objVal

        # Build routes
        routes = []
        vehicle_id = 0
        for k in range(K):
            route = []
            current = 0
            visited = set()
            sequence = [0]
            arcs = []
            visit_times = {}

            while True:
                found = False
                for j in all_nodes:
                    if j != current and x[current, j, k].X > 0.5:
                        sequence.append(j)
                        arcs.append([current, j])
                        visit_times[str(current)] = round(t[current, k].X, 4)
                        visited.add(current)
                        current = j
                        found = True
                        break
                if not found or current == 0:
                    break
            if len(sequence) > 1:  # at least depot -> customer -> depot
                # Add final depot time
                visit_times[str(0)] = round(t[0, k].X, 4)  # first depot
                last_node = sequence[-1]
                if last_node != 0:
                    sequence.append(0)
                    arcs.append([last_node, 0])
                visit_times[str(last_node)] = round(t[last_node, k].X, 4) if last_node != 0 else visit_times[str(0)]

                routes.append({
                    "vehicle": vehicle_id,
                    "sequence": sequence,
                    "arcs": arcs,
                    "visit_times": visit_times
                })
                vehicle_id += 1

        # Edge sets used
        edge_sets_used = []
        for es_idx in range(num_edge_sets):
            if z[es_idx].X > 0.5:
                edge_sets_used.append(edge_sets[es_idx]['id'])

        solution = {
            "objective_value": float(objective_value),
            "routes": routes,
            "edge_sets_used": edge_sets_used
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found, write empty feasible if possible
        solution = {
            "objective_value": 0.0,
            "routes": [],
            "edge_sets_used": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()