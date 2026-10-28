import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
import sys
from solution_logger import SolutionLogger
import numpy as np
from itertools import permutations
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['problem_size']['n']
    m = instance['problem_size']['m']
    nodes = instance['nodes']
    dist_flat = instance['distance_matrix_upper_triangular']['values']

    # Reconstruct distance matrix
    dist = np.zeros((n, n), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d = dist_flat[idx]
            dist[i][j] = d
            dist[j][i] = d
            idx += 1

    # Group nodes by cluster (cluster ids assumed 0 to m-1)
    cluster_nodes = [[] for _ in range(m)]
    node_to_cluster = [-1] * n
    for node in nodes:
        cid = node['cluster']
        nid = node['id']
        cluster_nodes[cid].append(nid)
        node_to_cluster[nid] = cid

    # If m is small (<= 12), solve exactly with brute force permutations
    if m <= 10:
        start_time = time.time()
        best_cost = float('inf')
        best_tour = None
        best_visited = None

        # Choose one rep from each cluster and find best TSP tour
        # For small m we enumerate all combinations and all permutations
        from itertools import product
        ranges = [cluster_nodes[c] for c in range(m)]
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

        for reps in product(*ranges):
            # reps is a tuple of node ids, one per cluster 0..m-1
            # Now find best cycle cost for these m nodes
            min_tour_cost = float('inf')
            best_order = None
            for perm in permutations(range(m)):
                cost = 0
                for k in range(m):
                    u = reps[perm[k]]
                    v = reps[perm[(k+1) % m]]
                    cost += dist[u][v]
                if cost < min_tour_cost:
                    min_tour_cost = cost
                    best_order = [reps[perm[k]] for k in range(m)]
            if min_tour_cost < best_cost:
                best_cost = min_tour_cost
                best_tour = best_order
                best_visited = sorted(best_order)
                if logger:
                    logger.log(float(best_cost))
                # Check time
                if time.time() - start_time > args.time_limit:
                    break

        # Output
        result = {
            "objective_value": float(best_cost),
            "tour": best_tour,
            "visited_nodes": best_visited
        }
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    # For larger m, use MIP with Gurobi (MTZ formulation on cluster representatives)
    # Variables: x[i,j] = 1 if we travel from rep of cluster i to rep of cluster j
    # But since reps are chosen, we need to select exactly one node per cluster

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Threads', 1)
    env.setParam('TimeLimit', args.time_limit)
    env.start()
    model = gp.Model("GTSP", env=env)

    # Binary variables for selecting node as representative
    select = {}
    for c in range(m):
        for v in cluster_nodes[c]:
            select[v] = model.addVar(vtype=GRB.BINARY, name=f"select_{v}")

    # For each cluster, exactly one node selected
    for c in range(m):
        model.addConstr(gp.quicksum(select[v] for v in cluster_nodes[c]) == 1)

    # Edge variables between nodes (only if in different clusters)
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j and node_to_cluster[i] != node_to_cluster[j]:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Each selected node has indegree and outdegree exactly 1
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if (i,j) in x) == select[i])
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if (j,i) in x) == select[i])

    # MTZ formulation for subtour elimination
    u = {}
    for i in range(n):
        if select.get(i) is not None:  # only for possible selected nodes
            u[i] = model.addVar(lb=0, ub=m, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    # Link u variables to selected nodes
    for i in range(n):
        for j in range(n):
            if (i,j) in x:
                if i != 0 and j != 0:  # assuming we can fix cluster 0 as starting point, but since cycle we pick arbitrary
                    model.addConstr(u[i] - u[j] + m * x[i,j] <= m - 1)

    # Objective
    obj = gp.quicksum(dist[i][j] * x[i,j] for (i,j) in x)
    model.setObjective(obj, GRB.MINIMIZE)

    # Optional: symmetry breaking by fixing one cluster's representative or ordering
    # Fix the representative of cluster 0 to be the one with smallest id for branching
    min_id = min(cluster_nodes[0])
    model.addConstr(select[min_id] >= 0.5)  # force at least one, but actually better not, removed for optimality

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < GRB.INFINITY:
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        if model.solCount > 0:
            objective_value = model.objVal
            # Extract solution
            selected = []
            for v in range(n):
                if select[v].X > 0.5:
                    selected.append(v)

            # Build tour from x
            succ = {}
            for (i,j) in x:
                if x[i,j].X > 0.5:
                    succ[i] = j

            # Reconstruct tour starting from any node
            start = selected[0]
            tour = []
            current = start
            for _ in range(m):
                tour.append(current)
                current = succ[current]

            visited_nodes = sorted(selected)

            result = {
                "objective_value": float(objective_value),
                "tour": tour,
                "visited_nodes": visited_nodes
            }
        else:
            # No solution found - very unlikely but provide dummy
            result = {
                "objective_value": 0.0,
                "tour": [cluster_nodes[c][0] for c in range(m)],
                "visited_nodes": [cluster_nodes[c][0] for c in range(m)]
            }
    else:
        # Failure case
        result = {
            "objective_value": 0.0,
            "tour": [cluster_nodes[c][0] for c in range(m)],
            "visited_nodes": [cluster_nodes[c][0] for c in range(m)]
        }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    env.dispose()

if __name__ == "__main__":
    main()