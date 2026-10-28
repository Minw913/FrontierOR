import json
import argparse
import sys
import os
import time
from collections import deque, defaultdict
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def build_solution_dict(accepted, rejected, idx_to_comp_id, arcs_by_id, reject_cost):
    """Build the solution dictionary in the required output schema."""
    commodities_out = []
    for k in range(len(idx_to_comp_id)):
        comp_id = idx_to_comp_id[k]
        if k in accepted:
            path_arcs = []
            for arc_id in accepted[k]:
                u, v, cap, cost = arcs_by_id[arc_id]
                path_arcs.append({"from": u, "to": v, "arc_id": arc_id})
            commodities_out.append({
                "commodity_id": comp_id,
                "rejected": False,
                "path_arcs": path_arcs
            })
        else:
            commodities_out.append({
                "commodity_id": comp_id,
                "rejected": True,
                "path_arcs": []
            })
    obj = sum(reject_cost[k] for k in rejected)
    return {"objective_value": float(obj), "commodities": commodities_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    network = instance['network']
    commodities_data = instance['commodities']

    num_nodes = network['num_nodes']
    nodes = network['nodes']
    arcs = network['arcs']

    # Build arc structures
    arcs_by_id = {}
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    arc_capacity = {}
    for arc in arcs:
        arc_id = arc['arc_id']
        u = arc['from_node']
        v = arc['to_node']
        cap = arc['capacity']
        cost = arc['cost']
        arcs_by_id[arc_id] = (u, v, cap, cost)
        out_arcs[u].append(arc_id)
        in_arcs[v].append(arc_id)
        arc_capacity[arc_id] = cap

    commodities = commodities_data['commodity_list']
    num_commodities = len(commodities)

    idx_to_comp_id = []
    comp_id_to_idx = {}
    demand = [0] * num_commodities
    reject_cost = [0] * num_commodities
    revenue = [0] * num_commodities
    origin = [0] * num_commodities
    dest = [0] * num_commodities

    for k, comm in enumerate(commodities):
        comp_id = comm['commodity_id']
        idx_to_comp_id.append(comp_id)
        comp_id_to_idx[comp_id] = k
        demand[k] = comm['demand']
        reject_cost[k] = comm['artificial_arc_cost']
        revenue[k] = comm['revenue']
        origin[k] = comm['origin']
        dest[k] = comm['destination']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Precompute usable arcs per commodity (arcs that lie on some feasible path)
    usable_arcs = [[] for _ in range(num_commodities)]
    usable_out = [defaultdict(list) for _ in range(num_commodities)]
    usable_in = [defaultdict(list) for _ in range(num_commodities)]
    usable_arcs_set = [set() for _ in range(num_commodities)]

    for k in range(num_commodities):
        d = demand[k]
        o = origin[k]
        t = dest[k]
        if o == t:
            usable_arcs[k] = []
            continue
        # Forward reachability from origin
        visited_from = set([o])
        q = deque([o])
        while q:
            u = q.popleft()
            for arc_id in out_arcs[u]:
                if arc_capacity[arc_id] >= d:
                    v = arcs_by_id[arc_id][1]
                    if v not in visited_from:
                        visited_from.add(v)
                        q.append(v)
        if t not in visited_from:
            usable_arcs[k] = []
            continue
        # Backward reachability to destination
        visited_to = set([t])
        q = deque([t])
        while q:
            u = q.popleft()
            for arc_id in in_arcs[u]:
                if arc_capacity[arc_id] >= d:
                    v = arcs_by_id[arc_id][0]
                    if v not in visited_to:
                        visited_to.add(v)
                        q.append(v)
        usable_nodes = visited_from.intersection(visited_to)
        u_arcs = []
        for arc_id in arc_capacity:
            if arc_capacity[arc_id] >= d:
                u, v, cap, cost = arcs_by_id[arc_id]
                if u in usable_nodes and v in usable_nodes:
                    u_arcs.append(arc_id)
        usable_arcs[k] = u_arcs
        usable_arcs_set[k] = set(u_arcs)
        for arc_id in u_arcs:
            u, v, cap, cost = arcs_by_id[arc_id]
            usable_out[k][u].append(arc_id)
            usable_in[k][v].append(arc_id)

    # ------------------------------------------------------------------
    # Greedy heuristic to get a good initial feasible solution
    # ------------------------------------------------------------------
    start_time = time.time()
    residual_cap = dict(arc_capacity)
    accepted = {}
    rejected = set()
    # Sort by rejection cost (revenue) descending
    sorted_commodities = sorted(range(num_commodities), key=lambda k: reject_cost[k], reverse=True)
    for k in sorted_commodities:
        o = origin[k]
        t = dest[k]
        d = demand[k]
        if o == t:
            accepted[k] = []
            continue
        # BFS for a path with sufficient residual capacity
        parent = {}
        visited = set([o])
        q = deque([o])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for arc_id in out_arcs[u]:
                if residual_cap[arc_id] >= d and arc_id in usable_arcs_set[k]:
                    v = arcs_by_id[arc_id][1]
                    if v not in visited:
                        visited.add(v)
                        parent[v] = (arc_id, u)
                        q.append(v)
        if found:
            path = []
            curr = t
            while curr != o:
                arc_id, prev = parent[curr]
                path.append(arc_id)
                curr = prev
            path.reverse()
            for arc_id in path:
                residual_cap[arc_id] -= d
            accepted[k] = path
        else:
            rejected.add(k)

    greedy_obj = sum(reject_cost[k] for k in rejected)
    greedy_solution = build_solution_dict(accepted, rejected, idx_to_comp_id, arcs_by_id, reject_cost)
    if logger:
        logger.log_solution(greedy_obj, greedy_solution)

    # ------------------------------------------------------------------
    # Gurobi model: binary arc formulation with degree constraints
    # ------------------------------------------------------------------
    model = gp.Model("unsplittable_flow")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)

    # Decision variables
    z = model.addVars(num_commodities, vtype=GRB.BINARY, name="z")
    y_keys = [(k, arc_id) for k in range(num_commodities) for arc_id in usable_arcs[k]]
    if y_keys:
        y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    else:
        y = {}
    model.update()

    # Flow conservation and degree constraints
    for k in range(num_commodities):
        o = origin[k]
        t = dest[k]
        if o == t:
            continue
        for node in nodes:
            has_out = node in usable_out[k]
            has_in = node in usable_in[k]
            if not has_out and not has_in:
                continue
            out_sum = gp.quicksum(y[k, arc_id] for arc_id in usable_out[k].get(node, []))
            in_sum = gp.quicksum(y[k, arc_id] for arc_id in usable_in[k].get(node, []))
            if node == o:
                model.addConstr(out_sum - in_sum == z[k])
            elif node == t:
                model.addConstr(out_sum - in_sum == -z[k])
            else:
                model.addConstr(out_sum - in_sum == 0)
            # Degree constraints to enforce simple paths (no cycles, no splitting)
            if node != t:
                model.addConstr(out_sum <= 1)
            if node != o:
                model.addConstr(in_sum <= 1)

    # Capacity constraints
    arc_users = defaultdict(list)
    for k in range(num_commodities):
        for arc_id in usable_arcs[k]:
            arc_users[arc_id].append(k)
    for arc_id, users in arc_users.items():
        model.addConstr(
            gp.quicksum(demand[k] * y[k, arc_id] for k in users) <= arc_capacity[arc_id],
            name=f"cap_{arc_id}"
        )

    # Objective: minimize total rejection cost
    model.setObjective(
        gp.quicksum(reject_cost[k] * (1 - z[k]) for k in range(num_commodities)),
        GRB.MINIMIZE
    )

    # Provide MIP start from greedy solution
    z_start = [1 if k in accepted else 0 for k in range(num_commodities)]
    model.setAttr('Start', [z[k] for k in range(num_commodities)], z_start)
    if y_keys:
        y_start = []
        for k in range(num_commodities):
            for arc_id in usable_arcs[k]:
                y_start.append(1 if (k in accepted and arc_id in accepted[k]) else 0)
        model.setAttr('Start', [y[k, arc_id] for k, arc_id in y_keys], y_start)

    # Prepare data for incumbent callback
    z_vars = [z[k] for k in range(num_commodities)]
    all_y_vars = [y[k, arc_id] for k, arc_id in y_keys]
    all_y_info = [(k, arc_id) for k, arc_id in y_keys]
    y_info_by_commodity = [[] for _ in range(num_commodities)]
    for i, (k, arc_id) in enumerate(all_y_info):
        y_info_by_commodity[k].append((i, arc_id))

    callback_data = {'last_obj': float('inf')}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < callback_data['last_obj'] - 1e-6:
                callback_data['last_obj'] = obj
                if logger:
                    z_vals = model.cbGetSolution(z_vars)
                    y_vals = model.cbGetSolution(all_y_vars)
                    acc = {}
                    rej = set()
                    for k in range(num_commodities):
                        if z_vals[k] > 0.5:
                            path = []
                            for i, arc_id in y_info_by_commodity[k]:
                                if y_vals[i] > 0.5:
                                    path.append(arc_id)
                            acc[k] = path
                        else:
                            rej.add(k)
                    sol_dict = build_solution_dict(acc, rej, idx_to_comp_id, arcs_by_id, reject_cost)
                    logger.log_solution(obj, sol_dict)

    # Set Gurobi time limit accounting for time already spent
    elapsed = time.time() - start_time
    time_limit_gurobi = max(1, args.time_limit - elapsed)
    model.setParam('TimeLimit', time_limit_gurobi)

    try:
        model.optimize(callback)
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}", file=sys.stderr)

    # Extract best solution found
    if model.SolCount > 0:
        z_vals = model.getAttr('X', z_vars)
        y_vals = model.getAttr('X', all_y_vars)
        acc = {}
        rej = set()
        for k in range(num_commodities):
            if z_vals[k] > 0.5:
                path = []
                for i, arc_id in y_info_by_commodity[k]:
                    if y_vals[i] > 0.5:
                        path.append(arc_id)
                acc[k] = path
            else:
                rej.add(k)
        final_solution = build_solution_dict(acc, rej, idx_to_comp_id, arcs_by_id, reject_cost)
    else:
        final_solution = greedy_solution

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()