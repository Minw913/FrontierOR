import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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

    num_nodes = instance['num_nodes']
    num_edges = instance['num_edges']
    bisection_capacity = instance['bisection_capacity_F']
    node_weights = instance['node_weights']
    edges = instance['edges']
    edge_costs = instance['edge_weights']

    # Build adjacency for greedy heuristic
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        u, v = edges[i]
        c = edge_costs[i]
        adj[u].append((v, c, i))
        adj[v].append((u, c, i))

    # Greedy heuristic: start from node 0, grow cluster by minimum cut-cost increase
    start_time = time.time()
    in_s = [False] * num_nodes
    in_s[0] = True
    current_weight = node_weights[0]

    for u in range(1, num_nodes):
        if node_weights[u] == 0:
            in_s[u] = True

    best_obj = 0.0
    best_partition = None

    # Compute initial cut cost
    cut_cost = 0.0
    for u in range(num_nodes):
        if in_s[u]:
            for v, c, _ in adj[u]:
                if not in_s[v]:
                    cut_cost += c
    cut_cost /= 2.0
    best_obj = cut_cost
    best_partition = [i for i in range(num_nodes) if in_s[i]]

    if logger:
        logger.log(best_obj)

    # Grow until capacity
    while current_weight < bisection_capacity and time.time() - start_time < args.time_limit - 2:
        best_delta = float('inf')
        best_node = -1
        for u in range(num_nodes):
            if in_s[u]:
                continue
            if current_weight + node_weights[u] > bisection_capacity:
                continue
            delta = 0.0
            for v, c, _ in adj[u]:
                if in_s[v]:
                    delta += c
                else:
                    delta -= c
            if delta < best_delta:
                best_delta = delta
                best_node = u
        if best_node == -1:
            break
        in_s[best_node] = True
        current_weight += node_weights[best_node]
        cut_cost += best_delta
        best_obj = cut_cost
        best_partition = [i for i in range(num_nodes) if in_s[i]]
        if logger:
            logger.log(best_obj)

    # Now try MIP with Gurobi using remaining time
    remaining_time = max(1, int(args.time_limit - (time.time() - start_time)))
    
    try:
        m = gp.Model("GraphBisection")
        m.setParam(GRB.Param.TimeLimit, remaining_time)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.LogToConsole, 0)

        x = [m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(num_nodes)]
        y = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{i}") for i in range(num_edges)]

        # Fix reference node
        m.addConstr(x[0] == 1, "ref")

        # Capacity constraint
        m.addConstr(gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes)) <= bisection_capacity,
                    "capacity_S")

        # Edge linearization
        for i in range(num_edges):
            u, v = edges[i]
            m.addConstr(y[i] >= x[u] - x[v])
            m.addConstr(y[i] >= x[v] - x[u])

        # Objective
        m.setObjective(gp.quicksum(edge_costs[i] * y[i] for i in range(num_edges)), GRB.MINIMIZE)

        # Callback to log improving solutions
        best_obj_mip = best_obj
        best_partition_mip = best_partition[:]

        def callback(model, where):
            nonlocal best_obj_mip, best_partition_mip
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj < best_obj_mip - 1e-6:
                    best_obj_mip = obj
                    sol_x = model.cbGetSolution(x)
                    best_partition_mip = [i for i in range(num_nodes) if abs(sol_x[i] - 1.0) < 1e-6]
                    if logger:
                        logger.log(best_obj_mip)

        m.optimize(callback)

        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            if m.objVal < best_obj - 1e-6:
                best_obj = m.objVal
                sol_x = m.getAttr(GRB.Attr.X, x)
                best_partition = [i for i in range(num_nodes) if sol_x[i] > 0.5]
            elif best_obj_mip < best_obj - 1e-6:
                best_obj = best_obj_mip
                best_partition = best_partition_mip

    except Exception:
        # Fall back to greedy solution
        pass

    # Final output
    partition_S = sorted(best_partition)
    partition_complement = sorted(i for i in range(num_nodes) if i not in set(partition_S))

    solution = {
        "objective_value": float(best_obj),
        "partition_S": partition_S,
        "partition_complement": partition_complement
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()