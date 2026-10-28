import argparse
import json
import math
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


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True


def make_solution(selected_nodes, selected_edge_indices, nodes, edges,
                  turbine_indices, node_weight, edge_weight):
    selected_turbines = sorted(
        int(nodes[i]["id"]) for i in turbine_indices if i in selected_nodes
    )

    selected_edges = []
    for ei in sorted(selected_edge_indices, key=lambda k: int(edges[k]["id"])):
        selected_edges.append({
            "from": int(edges[ei]["from"]),
            "to": int(edges[ei]["to"])
        })

    objective = sum(node_weight[i] for i in selected_nodes)
    objective += sum(edge_weight[ei] for ei in selected_edge_indices)

    if abs(objective) < 1e-12:
        objective = 0.0

    return {
        "objective_value": float(objective),
        "selected_turbines": selected_turbines,
        "selected_edges": selected_edges
    }


def construct_initial_solution(nodes, edges, endpoint_indices, substations,
                               turbine_indices, quota, node_weight, edge_weight,
                               fixed_edges, disabled_edges):
    """
    Builds a feasible spanning tree with Kruskal's algorithm and then repeatedly
    removes unnecessary optional leaves. This supplies both a fallback solution
    and a MIP start.
    """
    n = len(nodes)
    if n == 0 or not substations:
        return None

    dsu = DSU(n)
    tree_edges = set()

    for ei in fixed_edges:
        u, v = endpoint_indices[ei]
        if dsu.union(u, v):
            tree_edges.add(ei)

    candidates = []
    for ei, (u, v) in enumerate(endpoint_indices):
        if ei in disabled_edges or ei in fixed_edges:
            continue

        # Charging endpoint node weights encourages expensive turbines to be
        # leaves, where they can subsequently be pruned.
        score = edge_weight[ei] + node_weight[u] + node_weight[v]
        candidates.append((score, edge_weight[ei], int(edges[ei]["id"]), ei))

    candidates.sort()

    for _, _, _, ei in candidates:
        u, v = endpoint_indices[ei]
        if dsu.union(u, v):
            tree_edges.add(ei)
            if len(tree_edges) == n - 1:
                break

    if n > 1 and len(tree_edges) != n - 1:
        return None

    selected_nodes = set(range(n))
    total_profit = sum(float(nodes[i]["profit"]) for i in turbine_indices)
    if total_profit + 1e-8 < quota:
        return None

    adjacency = [set() for _ in range(n)]
    for ei in tree_edges:
        u, v = endpoint_indices[ei]
        adjacency[u].add(ei)
        adjacency[v].add(ei)

    mandatory = set(substations)
    turbine_set = set(turbine_indices)

    # Optional leaves can be removed without harming connectivity among the
    # remaining selected nodes.
    stack = [
        i for i in range(n)
        if i not in mandatory and len(adjacency[i]) <= 1
    ]

    while stack:
        i = stack.pop()
        if i not in selected_nodes or i in mandatory or len(adjacency[i]) > 1:
            continue

        removable = False
        if i in turbine_set:
            p = float(nodes[i]["profit"])
            if total_profit - p + 1e-8 >= quota:
                removable = True
        else:
            # Steiner nodes have no energy yield.
            removable = True

        if not removable:
            continue

        if len(adjacency[i]) == 1:
            ei = next(iter(adjacency[i]))
            u, v = endpoint_indices[ei]
            other = v if u == i else u
            adjacency[i].remove(ei)
            adjacency[other].remove(ei)
            tree_edges.remove(ei)
            if other not in mandatory and len(adjacency[other]) <= 1:
                stack.append(other)

        selected_nodes.remove(i)
        if i in turbine_set:
            total_profit -= float(nodes[i]["profit"])

    return selected_nodes, tree_edges


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    nodes = instance["nodes"]
    edges = instance["edges"]
    params = instance["parameters"]

    alpha = float(params["alpha"])
    quota = float(params["quota"])

    n = len(nodes)
    m = len(edges)

    id_to_index = {int(node["id"]): i for i, node in enumerate(nodes)}

    endpoint_indices = []
    for edge in edges:
        u_id = int(edge["from"])
        v_id = int(edge["to"])
        if u_id not in id_to_index or v_id not in id_to_index:
            raise ValueError("Edge references an unknown node identifier")
        endpoint_indices.append((id_to_index[u_id], id_to_index[v_id]))

    substations = [
        i for i, node in enumerate(nodes) if node["type"] == "substation"
    ]
    turbine_indices = [
        i for i, node in enumerate(nodes) if node["type"] == "potential_turbine"
    ]

    if not substations:
        raise ValueError("The instance must contain at least one substation")

    node_weight = [
        alpha * float(node["cost"])
        + (1.0 - alpha) * float(node["scenic_impact"])
        for node in nodes
    ]
    edge_weight = [
        alpha * float(edge["cost"])
        + (1.0 - alpha) * float(edge["scenic_impact"])
        for edge in edges
    ]

    # Canonicalize the zero-cost interconnection of substations. When all
    # root-to-substation edges exist, an optimal tree can always be transformed
    # into one containing this fixed substation star.
    fixed_edges = set()
    disabled_edges = set()
    root = substations[0]

    if len(substations) > 1:
        best_root_edge = {}
        substation_set = set(substations)

        for ei, (u, v) in enumerate(endpoint_indices):
            if u in substation_set and v in substation_set:
                other = None
                if u == root and v != root:
                    other = v
                elif v == root and u != root:
                    other = u

                if other is not None:
                    key = (
                        edge_weight[ei],
                        float(edges[ei]["cost"]),
                        float(edges[ei]["scenic_impact"]),
                        int(edges[ei]["id"])
                    )
                    if other not in best_root_edge or key < best_root_edge[other][0]:
                        best_root_edge[other] = (key, ei)

        canonical_available = all(
            s == root or s in best_root_edge for s in substations
        )

        if canonical_available:
            fixed_edges = {
                best_root_edge[s][1] for s in substations if s != root
            }
            for ei, (u, v) in enumerate(endpoint_indices):
                if u in substation_set and v in substation_set and ei not in fixed_edges:
                    disabled_edges.add(ei)

    initial = construct_initial_solution(
        nodes=nodes,
        edges=edges,
        endpoint_indices=endpoint_indices,
        substations=substations,
        turbine_indices=turbine_indices,
        quota=quota,
        node_weight=node_weight,
        edge_weight=edge_weight,
        fixed_edges=fixed_edges,
        disabled_edges=disabled_edges
    )

    best_solution = None
    best_objective = math.inf
    best_selected_nodes = None
    best_selected_edges = None

    def record_solution(selected_nodes, selected_edge_indices):
        nonlocal best_solution, best_objective
        nonlocal best_selected_nodes, best_selected_edges

        solution = make_solution(
            selected_nodes,
            selected_edge_indices,
            nodes,
            edges,
            turbine_indices,
            node_weight,
            edge_weight
        )
        objective = solution["objective_value"]

        tolerance = 1e-8 * max(1.0, abs(best_objective)) if math.isfinite(best_objective) else 0.0
        if objective < best_objective - tolerance:
            best_objective = objective
            best_solution = solution
            best_selected_nodes = set(selected_nodes)
            best_selected_edges = set(selected_edge_indices)
            if logger:
                try:
                    logger.log_solution(objective, solution)
                except Exception:
                    pass

    if initial is not None:
        record_solution(initial[0], initial[1])

    model = gp.Model("quota_wind_farm_tree")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.monotonic() - start_time
    remaining = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    y = model.addVars(n, vtype=GRB.BINARY, name="node")
    x = model.addVars(m, vtype=GRB.BINARY, name="edge")

    # Two directed nonnegative flow variables per undirected edge.
    flow = model.addVars(m, 2, lb=0.0, vtype=GRB.CONTINUOUS, name="flow")

    for s in substations:
        y[s].LB = 1.0
        y[s].UB = 1.0

    for ei in fixed_edges:
        x[ei].LB = 1.0
        x[ei].UB = 1.0

    for ei in disabled_edges:
        x[ei].LB = 0.0
        x[ei].UB = 0.0

    incident = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(endpoint_indices):
        incident[u].append((ei, 0, 1))  # outgoing orientation 0, incoming 1
        incident[v].append((ei, 1, 0))  # outgoing orientation 1, incoming 0

        model.addConstr(x[ei] <= y[u], name=f"edge_node_u_{ei}")
        model.addConstr(x[ei] <= y[v], name=f"edge_node_v_{ei}")

    # A connected selected graph with |E| = |V|-1 is a tree.
    model.addConstr(
        gp.quicksum(x[ei] for ei in range(m))
        == gp.quicksum(y[i] for i in range(n)) - 1,
        name="tree_cardinality"
    )

    flow_capacity = max(0, n - 1)
    for ei in range(m):
        model.addConstr(
            flow[ei, 0] <= flow_capacity * x[ei],
            name=f"flow_cap_forward_{ei}"
        )
        model.addConstr(
            flow[ei, 1] <= flow_capacity * x[ei],
            name=f"flow_cap_reverse_{ei}"
        )

    for i in range(n):
        incoming = gp.quicksum(flow[ei, incoming_dir]
                               for ei, _, incoming_dir in incident[i])
        outgoing = gp.quicksum(flow[ei, outgoing_dir]
                               for ei, outgoing_dir, _ in incident[i])

        if i == root:
            model.addConstr(
                outgoing - incoming
                == gp.quicksum(y[j] for j in range(n) if j != root),
                name="root_flow_balance"
            )
        else:
            model.addConstr(
                incoming - outgoing == y[i],
                name=f"flow_balance_{i}"
            )

    model.addConstr(
        gp.quicksum(float(nodes[i]["profit"]) * y[i] for i in turbine_indices)
        >= quota,
        name="energy_quota"
    )

    objective_expr = (
        gp.quicksum(node_weight[i] * y[i] for i in range(n))
        + gp.quicksum(edge_weight[ei] * x[ei] for ei in range(m))
    )
    model.setObjective(objective_expr, GRB.MINIMIZE)

    # Supply the constructive solution as a partial MIP start.
    if initial is not None:
        initial_nodes, initial_edges = initial
        for i in range(n):
            y[i].Start = 1.0 if i in initial_nodes else 0.0
        for ei in range(m):
            x[ei].Start = 1.0 if ei in initial_edges else 0.0

    # Turbine choices are the principal combinatorial decisions.
    for i in turbine_indices:
        y[i].BranchPriority = 10

    model._x_vars = [x[ei] for ei in range(m)]
    model._y_vars = [y[i] for i in range(n)]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            y_values = cb_model.cbGetSolution(cb_model._y_vars)
            x_values = cb_model.cbGetSolution(cb_model._x_vars)

            selected_nodes = {
                i for i, value in enumerate(y_values) if value > 0.5
            }
            selected_edge_indices = {
                ei for ei, value in enumerate(x_values) if value > 0.5
            }
            record_solution(selected_nodes, selected_edge_indices)
        except Exception:
            # Logging or extraction failures must not interrupt optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        selected_nodes = {i for i in range(n) if y[i].X > 0.5}
        selected_edge_indices = {ei for ei in range(m) if x[ei].X > 0.5}
        record_solution(selected_nodes, selected_edge_indices)

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found. The instance may be infeasible "
            "or the time limit may be too small to construct an incumbent."
        )

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()