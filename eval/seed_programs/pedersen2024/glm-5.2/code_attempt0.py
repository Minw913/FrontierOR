import argparse
import json
import heapq
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    params = data['parameters']
    nodes = data['nodes']
    edges = data['edges']
    quota = params['quota']
    alpha = params['alpha']

    # Identify node types
    substation_ids = set()
    turbine_ids = set()
    steiner_ids = set()
    id_to_node = {}
    for n in nodes:
        id_to_node[n['id']] = n
        if n['type'] == 'substation':
            substation_ids.add(n['id'])
        elif n['type'] == 'potential_turbine':
            turbine_ids.add(n['id'])
        elif n['type'] == 'steiner':
            steiner_ids.add(n['id'])

    # Contract all substations into a single root node (index 0)
    non_sub_nodes = [n for n in nodes if n['type'] != 'substation']
    M = len(non_sub_nodes)
    node_index = {}  # original id -> contracted index (1..M)
    contracted_nodes = [None]  # index 0 is root
    for n in non_sub_nodes:
        idx = len(contracted_nodes)
        node_index[n['id']] = idx
        contracted_nodes.append(n)

    # Build contracted edges
    contracted_edges = []  # (u, v, weight, orig_from, orig_to)
    for e in edges:
        u = e['from']
        v = e['to']
        u_sub = u in substation_ids
        v_sub = v in substation_ids
        if u_sub and v_sub:
            continue  # self-loop at root, ignore
        if u_sub:
            new_u = 0
            new_v = node_index[v]
            orig_from = u
            orig_to = v
        elif v_sub:
            new_u = 0
            new_v = node_index[u]
            orig_from = u
            orig_to = v
        else:
            new_u = node_index[u]
            new_v = node_index[v]
            orig_from = u
            orig_to = v
        w = alpha * e['cost'] + (1.0 - alpha) * e['scenic_impact']
        contracted_edges.append((new_u, new_v, w, orig_from, orig_to))

    # Precompute turbine data
    turbine_weight = {}
    turbine_profit = {}
    turbine_contracted_idx = {}
    for n in non_sub_nodes:
        if n['type'] == 'potential_turbine':
            idx = node_index[n['id']]
            w = alpha * n['cost'] + (1.0 - alpha) * n['scenic_impact']
            turbine_weight[idx] = w
            turbine_profit[idx] = n['profit']
            turbine_contracted_idx[n['id']] = idx

    # Build adjacency for Dijkstra (used for initial heuristic)
    adj = [[] for _ in range(M + 1)]
    for i, (u, v, w, of, ot) in enumerate(contracted_edges):
        adj[u].append((v, w, i))
        adj[v].append((u, w, i))

    # ---------------- Heuristic initial solution ----------------
    y_start = [0] * (M + 1)  # indexed by contracted node (1..M)
    x_start = [0] * len(contracted_edges)

    if quota <= 0:
        # Empty tree (just root) is feasible
        pass
    else:
        # Greedily pick turbines by maximum profit until quota is met
        sorted_turbines = sorted(
            turbine_contracted_idx.items(),
            key=lambda x: id_to_node[x[0]]['profit'],
            reverse=True
        )
        selected_turbine_ids = []
        total_profit = 0.0
        for tid, idx in sorted_turbines:
            selected_turbine_ids.append(tid)
            total_profit += id_to_node[tid]['profit']
            if total_profit >= quota:
                break

        # Run Dijkstra from root
        dist = [float('inf')] * (M + 1)
        parent_edge = [-1] * (M + 1)
        parent_node = [-1] * (M + 1)
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w, e_idx in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent_edge[v] = e_idx
                    parent_node[v] = u
                    heapq.heappush(pq, (nd, v))

        # Union of shortest paths to selected turbines forms a tree
        tree_edges = set()
        tree_nodes = {0}
        for tid in selected_turbine_ids:
            cur = turbine_contracted_idx[tid]
            while cur != 0:
                tree_nodes.add(cur)
                e_idx = parent_edge[cur]
                tree_edges.add(e_idx)
                cur = parent_node[cur]

        for n_idx in tree_nodes:
            if n_idx != 0:
                y_start[n_idx] = 1
        for e_idx in tree_edges:
            x_start[e_idx] = 1

    # ---------------- Gurobi MIP model ----------------
    model = gp.Model("wind_farm")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Variables
    y = model.addVars(M, vtype=GRB.BINARY, name="y")
    x = model.addVars(len(contracted_edges), vtype=GRB.BINARY, name="x")
    a = model.addVars(len(contracted_edges), 2, vtype=GRB.BINARY, name="a")
    f = model.addVars(len(contracted_edges), 2, lb=0.0, name="f")

    # Set MIP start
    for i in range(M):
        y[i].Start = y_start[i + 1]
    for e in range(len(contracted_edges)):
        x[e].Start = x_start[e]

    # Arc-edge linking
    for e in range(len(contracted_edges)):
        model.addConstr(a[e, 0] + a[e, 1] == x[e])

    # Incoming/outgoing arc lists for each contracted node
    in_arcs = [[] for _ in range(M + 1)]
    out_arcs = [[] for _ in range(M + 1)]
    for e, (u, v, w, of, ot) in enumerate(contracted_edges):
        out_arcs[u].append((e, 0))
        in_arcs[v].append((e, 0))
        out_arcs[v].append((e, 1))
        in_arcs[u].append((e, 1))

    total_y = gp.quicksum(y[i] for i in range(M))

    # Flow balance
    model.addConstr(
        gp.quicksum(f[e, d] for e, d in out_arcs[0]) -
        gp.quicksum(f[e, d] for e, d in in_arcs[0]) == total_y
    )
    for i in range(1, M + 1):
        model.addConstr(
            gp.quicksum(f[e, d] for e, d in out_arcs[i]) -
            gp.quicksum(f[e, d] for e, d in in_arcs[i]) == -y[i - 1]
        )

    # Flow capacity (big-M = M)
    for e in range(len(contracted_edges)):
        model.addConstr(f[e, 0] <= M * a[e, 0])
        model.addConstr(f[e, 1] <= M * a[e, 1])

    # Edge count = number of selected non-root nodes
    model.addConstr(gp.quicksum(x[e] for e in range(len(contracted_edges))) == total_y)

    # Degree lower bound: selected node must have at least one incident edge
    for i in range(1, M + 1):
        incident_edges = [e for e, (u, v, w, of, ot) in enumerate(contracted_edges) if u == i or v == i]
        if incident_edges:
            model.addConstr(gp.quicksum(x[e] for e in incident_edges) >= y[i - 1])

    # Quota
    if turbine_profit:
        model.addConstr(
            gp.quicksum(turbine_profit[i + 1] * y[i] for i in range(M) if (i + 1) in turbine_profit) >= quota
        )
    else:
        if quota > 0:
            # No turbines available, infeasible
            model.addConstr(0 >= quota)

    # Objective
    obj = gp.quicksum(contracted_edges[e][2] * x[e] for e in range(len(contracted_edges)))
    obj += gp.quicksum(turbine_weight[i + 1] * y[i] for i in range(M) if (i + 1) in turbine_weight)
    model.setObjective(obj, GRB.MINIMIZE)

    # ---------------- Logging callback ----------------
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_vals = model.cbGetSolution(y)
            x_vals = model.cbGetSolution(x)
            selected_turbines = []
            for i in range(M):
                if y_vals[i] > 0.5:
                    node = contracted_nodes[i + 1]
                    if node['type'] == 'potential_turbine':
                        selected_turbines.append(node['id'])
            selected_edges = []
            for e in range(len(contracted_edges)):
                if x_vals[e] > 0.5:
                    u, v, w, of, ot = contracted_edges[e]
                    selected_edges.append({"from": of, "to": ot})
            sol = {
                "objective_value": obj_val,
                "selected_turbines": selected_turbines,
                "selected_edges": selected_edges
            }
            if logger:
                logger.log_solution(obj_val, sol)

    model.optimize(callback)

    # ---------------- Extract final solution ----------------
    if model.SolCount > 0:
        obj_val = model.ObjVal
        selected_turbines = []
        for i in range(M):
            if y[i].X > 0.5:
                node = contracted_nodes[i + 1]
                if node['type'] == 'potential_turbine':
                    selected_turbines.append(node['id'])
        selected_edges = []
        for e in range(len(contracted_edges)):
            if x[e].X > 0.5:
                u, v, w, of, ot = contracted_edges[e]
                selected_edges.append({"from": of, "to": ot})
    else:
        # Fallback to heuristic start
        obj_val = (
            sum(contracted_edges[e][2] * x_start[e] for e in range(len(contracted_edges))) +
            sum(turbine_weight[i + 1] * y_start[i + 1] for i in range(M) if (i + 1) in turbine_weight)
        )
        selected_turbines = []
        for i in range(M):
            if y_start[i + 1] > 0.5:
                node = contracted_nodes[i + 1]
                if node['type'] == 'potential_turbine':
                    selected_turbines.append(node['id'])
        selected_edges = []
        for e in range(len(contracted_edges)):
            if x_start[e] > 0.5:
                u, v, w, of, ot = contracted_edges[e]
                selected_edges.append({"from": of, "to": ot})

    solution = {
        "objective_value": obj_val,
        "selected_turbines": selected_turbines,
        "selected_edges": selected_edges
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(obj_val, solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()