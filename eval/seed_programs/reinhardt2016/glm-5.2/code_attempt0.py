import json
import argparse
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_customers']
    V_max = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    horizon = instance['schedule_horizon']

    demand, tw_start, tw_end, svc_time = {}, {}, {}, {}
    for node in instance['nodes']:
        idx = node['id']
        demand[idx] = node['demand']
        tw_start[idx] = node['time_window_start']
        tw_end[idx] = node['time_window_end']
        svc_time[idx] = node['service_time']

    dist = instance['distance_matrix']

    # Arc to edge set mapping
    arc_to_es = {}
    for es in instance['edge_sets']:
        for edge in es['edges']:
            i, j = edge[0], edge[1]
            arc_to_es[(i, j)] = es['id']
            arc_to_es[(j, i)] = es['id']

    es_access_cost = {es['id']: es['access_cost'] for es in instance['edge_sets']}
    es_ids = [es['id'] for es in instance['edge_sets']]

    all_nodes = list(range(0, n + 1))
    customers = list(range(1, n + 1))

    # Feasible arcs with tight big-M
    feasible_arcs = set()
    M_time = {}
    for i in all_nodes:
        for j in customers:
            if i == j:
                continue
            earliest = tw_start[i] + svc_time[i] + dist[i][j]
            if earliest <= tw_end[j]:
                feasible_arcs.add((i, j))
                M_time[(i, j)] = max(1.0, tw_end[i] + svc_time[i] + dist[i][j] - tw_start[j])

    for i in customers:
        if tw_start[i] + svc_time[i] + dist[i][0] <= tw_end[0]:
            feasible_arcs.add((i, 0))

    # Greedy initial solution for warm start
    greedy_routes = []
    unvisited = set(customers)
    while unvisited:
        route = [0]
        load = 0
        time_now = tw_start[0]
        cur = 0
        while True:
            best, best_d = None, float('inf')
            for c in unvisited:
                if load + demand[c] > Q:
                    continue
                arr = time_now + dist[cur][c] + svc_time[cur]
                if arr > tw_end[c]:
                    continue
                d = dist[cur][c]
                if d < best_d:
                    best_d, best = d, c
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            load += demand[best]
            arr = time_now + dist[cur][best] + svc_time[cur]
            time_now = max(arr, tw_start[best])
            cur = best
        route.append(0)
        greedy_routes.append(route)

    greedy_feasible = (not unvisited) and (len(greedy_routes) <= V_max)

    # Gurobi model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model('VRPTW', env=env)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', float(args.time_limit))

    x = {}
    for (i, j) in feasible_arcs:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

    z = {eid: model.addVar(vtype=GRB.BINARY, name=f'z_{eid}') for eid in es_ids}

    t = {i: model.addVar(lb=tw_start[i], ub=tw_end[i], name=f't_{i}') for i in all_nodes}
    u = {i: model.addVar(lb=demand[i], ub=Q, name=f'u_{i}') for i in customers}

    model.update()

    # Each customer visited exactly once (in-degree and out-degree = 1)
    for j in customers:
        model.addConstr(gp.quicksum(x[i, j] for i in all_nodes if (i, j) in x) == 1)
        model.addConstr(gp.quicksum(x[j, k] for k in all_nodes if (j, k) in x) == 1)

    # Depot flow conservation and vehicle count limit
    model.addConstr(
        gp.quicksum(x[0, j] for j in customers if (0, j) in x) ==
        gp.quicksum(x[i, 0] for i in customers if (i, 0) in x)
    )
    model.addConstr(gp.quicksum(x[0, j] for j in customers if (0, j) in x) <= V_max)

    # Capacity MTZ constraints
    for j in customers:
        for i in customers:
            if i != j and (i, j) in x:
                model.addConstr(u[j] >= u[i] + demand[j] - Q * (1 - x[i, j]))

    # Fix depot time to earliest
    model.addConstr(t[0] == tw_start[0])

    # Travel time constraints (for arcs ending at customers)
    for (i, j) in feasible_arcs:
        if j in customers:
            model.addConstr(
                t[j] >= t[i] + dist[i][j] + svc_time[i] - M_time[(i, j)] * (1 - x[i, j])
            )

    # Return to depot within time window
    for i in customers:
        if (i, 0) in x:
            M_ret = max(1.0, tw_end[i] + dist[i][0] + svc_time[i] - tw_end[0])
            model.addConstr(
                t[i] + dist[i][0] + svc_time[i] <= tw_end[0] + M_ret * (1 - x[i, 0])
            )

    # Edge set access constraints
    for (i, j) in feasible_arcs:
        es_id = arc_to_es.get((i, j))
        if es_id is not None and es_id in z:
            model.addConstr(x[i, j] <= z[es_id])

    # Objective: traversal costs + access costs
    obj = gp.quicksum(dist[i][j] * x[i, j] for (i, j) in feasible_arcs)
    obj += gp.quicksum(es_access_cost[eid] * z[eid] for eid in es_ids)
    model.setObjective(obj, GRB.MINIMIZE)

    # Warm start from greedy solution
    if greedy_feasible:
        for (i, j) in feasible_arcs:
            x[i, j].Start = 0.0
        used_sets = set()
        for route in greedy_routes:
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                if (i, j) in x:
                    x[i, j].Start = 1.0
                es_id = arc_to_es.get((i, j))
                if es_id is not None:
                    used_sets.add(es_id)
        for eid in es_ids:
            z[eid].Start = 1.0 if eid in used_sets else 0.0

    # Callback for logging incumbents
    def log_callback(m, where):
        if where == GRB.Callback.MIPSOL and logger:
            logger.log(m.cbGet(GRB.Callback.MIPSOL_OBJ))

    model.optimize(log_callback)

    # Extract solution
    if model.SolCount == 0:
        solution = {"objective_value": 0.0, "routes": [], "edge_sets_used": []}
    else:
        obj_val = model.ObjVal

        arcs_used = [(i, j) for (i, j) in feasible_arcs if x[i, j].X > 0.5]
        out_map = {}
        for (i, j) in arcs_used:
            out_map.setdefault(i, []).append(j)

        routes = []
        visited = set()
        vid = 0
        for sc in out_map.get(0, []):
            if sc in visited:
                continue
            seq = [0]
            cur = sc
            while cur != 0 and cur not in visited:
                seq.append(cur)
                visited.add(cur)
                nxt = out_map.get(cur, [])
                if not nxt:
                    break
                cur = nxt[0]
            seq.append(0)

            visit_times = {}
            for nd in set(seq):
                if nd in t:
                    visit_times[str(nd)] = float(t[nd].X)
                else:
                    visit_times[str(nd)] = float(tw_start[nd])

            r_arcs = [[seq[k], seq[k + 1]] for k in range(len(seq) - 1)]
            routes.append({
                "vehicle": vid,
                "sequence": seq,
                "arcs": r_arcs,
                "visit_times": visit_times
            })
            vid += 1

        es_used = [eid for eid in es_ids if z[eid].X > 0.5]
        solution = {
            "objective_value": float(obj_val),
            "routes": routes,
            "edge_sets_used": es_used
        }

        if logger:
            logger.log_solution(obj_val, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()