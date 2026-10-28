import json
import argparse
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def construct_heuristic(n, m, K, Q, demands, routing_cost, assignment_cost, node_id, id_to_idx):
    """Simple construction heuristic for fallback / warm start."""
    site_load = [0] * (n + 1)
    assignment = {}

    cust_order = sorted(range(m), key=lambda i: -demands[i])
    for i in cust_order:
        site_order = sorted(range(1, n + 1), key=lambda j: assignment_cost[i][j - 1])
        placed = False
        for j in site_order:
            if site_load[j] + demands[i] <= Q:
                assignment[i] = j
                site_load[j] += demands[i]
                placed = True
                break
        if not placed:
            j = site_order[0]
            assignment[i] = j
            site_load[j] += demands[i]

    used_sites = [j for j in range(1, n + 1) if site_load[j] > 0]
    tours = []
    remaining = sorted(used_sites, key=lambda j: -site_load[j])

    while remaining and len(tours) < K:
        tour = []
        load = 0
        new_remaining = []
        for j in remaining:
            if load + site_load[j] <= Q:
                tour.append(j)
                load += site_load[j]
            else:
                new_remaining.append(j)
        if tour:
            tours.append(tour)
        remaining = new_remaining
        if not tour and remaining:
            tours.append([remaining.pop(0)])

    while len(tours) > K:
        last = tours.pop()
        tours[-1].extend(last)

    routes = {}
    assignments = {}
    customer_tours = {}

    for k, tour_sites in enumerate(tours):
        if not tour_sites:
            continue
        seq = [0]
        unvisited = list(tour_sites)
        curr = 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: routing_cost[curr][j])
            seq.append(nxt)
            curr = nxt
            unvisited.remove(nxt)
        seq.append(0)

        arcs = [[node_id[seq[i]], node_id[seq[i + 1]]] for i in range(len(seq) - 1)]
        seq_ids = [node_id[s] for s in seq]
        routes[str(k)] = {'arcs': arcs, 'sequence': seq_ids}

        for j in tour_sites:
            for i in range(m):
                if assignment.get(i) == j:
                    assignments[str(i)] = node_id[j]
                    customer_tours[str(i)] = {'vehicle': k, 'site': node_id[j]}

    obj = 0.0
    for k_str in routes:
        for a in routes[k_str]['arcs']:
            fi = id_to_idx[a[0]]
            ti = id_to_idx[a[1]]
            obj += routing_cost[fi][ti]
    for i in range(m):
        j = assignment[i]
        obj += assignment_cost[i][j - 1]

    return {
        'objective_value': obj,
        'routes': routes,
        'assignments': assignments,
        'customer_tours': customer_tours
    }


def extract_solution(model, from_callback, n, m, K, node_id, x, z, u):
    """Extract solution dict from model (callback or final)."""
    if from_callback:
        x_vals = model.cbGetSolution(x)
        z_vals = model.cbGetSolution(z)
        u_vals = model.cbGetSolution(u)
    else:
        x_vals = {k: v.X for k, v in x.items()}
        z_vals = {k: v.X for k, v in z.items()}
        u_vals = {k: v.X for k, v in u.items()}

    routes = {}
    assignments = {}
    customer_tours = {}

    for i in range(m):
        for j in range(1, n + 1):
            if z_vals.get((i, j), 0) > 0.5:
                assignments[str(i)] = node_id[j]
                break

    for i in range(m):
        for j in range(1, n + 1):
            for k in range(K):
                if u_vals.get((i, j, k), 0) > 0.5:
                    customer_tours[str(i)] = {'vehicle': k, 'site': node_id[j]}
                    break
            if str(i) in customer_tours:
                break

    for k in range(K):
        arc_pairs = []
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j and x_vals.get((i, j, k), 0) > 0.5:
                    arc_pairs.append((i, j))
        if not arc_pairs:
            continue
        adj = {}
        for a in arc_pairs:
            adj.setdefault(a[0], []).append(a[1])
        seq = [0]
        curr = 0
        while True:
            nexts = adj.get(curr, [])
            if not nexts:
                break
            nxt = nexts.pop(0)
            seq.append(nxt)
            curr = nxt
            if curr == 0:
                break
        seq_ids = [node_id[s] for s in seq]
        arcs_out = [[node_id[a[0]], node_id[a[1]]] for a in arc_pairs]
        routes[str(k)] = {'arcs': arcs_out, 'sequence': seq_ids}

    return {
        'objective_value': 0,
        'routes': routes,
        'assignments': assignments,
        'customer_tours': customer_tours
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['num_delivery_sites']
    m = inst['num_customers']
    Q = inst['vehicle_capacity']
    K = inst['num_vehicles']

    routing_cost = inst['routing_cost_matrix']
    assignment_cost = inst['assignment_cost_matrix']
    demands = [inst['customers'][i]['demand'] for i in range(m)]

    depot_id = inst['depot']['id']
    site_ids = [inst['delivery_sites'][j]['id'] for j in range(n)]
    node_id = [depot_id] + site_ids
    id_to_idx = {node_id[i]: i for i in range(n + 1)}

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Heuristic for fallback
    heur_sol = construct_heuristic(n, m, K, Q, demands, routing_cost, assignment_cost, node_id, id_to_idx)
    if logger:
        logger.log_solution(heur_sol['objective_value'], heur_sol)

    N0 = list(range(n + 1))
    S = list(range(1, n + 1))
    C = list(range(m))
    V = list(range(K))

    model = Model('LRP')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(args.time_limit)

    x = model.addVars(
        [(i, j, k) for k in V for i in N0 for j in N0 if i != j],
        vtype=GRB.BINARY, name='x'
    )
    y = model.addVars(S, vtype=GRB.BINARY, name='y')
    z = model.addVars(C, S, vtype=GRB.BINARY, name='z')
    u = model.addVars(C, S, V, lb=0, ub=1, name='u')
    q = model.addVars(S, V, lb=0, name='q')

    model.update()

    model.setObjective(
        quicksum(routing_cost[i][j] * x[i, j, k] for k in V for i in N0 for j in N0 if i != j) +
        quicksum(assignment_cost[i][j - 1] * z[i, j] for i in C for j in S),
        GRB.MINIMIZE
    )

    # Each customer assigned to exactly one site
    for i in C:
        model.addConstr(quicksum(z[i, j] for j in S) == 1)

    # Assignment requires open site
    for i in C:
        for j in S:
            model.addConstr(z[i, j] <= y[j])

    # Site visited at most once and visited implies open
    for j in S:
        visited = quicksum(x[i, j, k] for k in V for i in N0 if i != j)
        model.addConstr(visited <= y[j])
        model.addConstr(visited <= quicksum(z[i, j] for i in C))

    # Flow conservation
    for k in V:
        for i in N0:
            model.addConstr(
                quicksum(x[h, i, k] for h in N0 if h != i) ==
                quicksum(x[i, j, k] for j in N0 if j != i)
            )

    # At most one depot-leaving arc per tour
    for k in V:
        model.addConstr(quicksum(x[0, j, k] for j in S) <= 1)

    # Service pairing bounds
    for i in C:
        for j in S:
            for k in V:
                v_jk = quicksum(x[h, j, k] for h in N0 if h != j)
                model.addConstr(u[i, j, k] <= z[i, j])
                model.addConstr(u[i, j, k] <= v_jk)
                model.addConstr(u[i, j, k] >= z[i, j] + v_jk - 1)

    # Customer coverage
    for i in C:
        model.addConstr(quicksum(u[i, j, k] for j in S for k in V) == 1)

    # Capacity per tour
    for k in V:
        model.addConstr(quicksum(u[i, j, k] * demands[i] for i in C for j in S) <= Q)

    # q bounds (cumulative delivered)
    for j in S:
        for k in V:
            model.addConstr(q[j, k] >= quicksum(u[i, j, k] * demands[i] for i in C))
            model.addConstr(q[j, k] <= Q * quicksum(x[j, i, k] for i in N0 if i != j))

    # Subtour elimination via cumulative flow
    for k in V:
        for i in S:
            for j in S:
                if i != j:
                    demand_jk = quicksum(u[h, j, k] * demands[h] for h in C)
                    model.addConstr(q[j, k] >= q[i, k] + demand_jk - Q * (1 - x[i, j, k]))

    # Warm start from heuristic
    try:
        for i in C:
            j_heur = None
            for key, val in heur_sol['assignments'].items():
                if int(key) == i:
                    j_heur = id_to_idx[val]
                    break
            if j_heur is not None:
                z[i, j_heur].Start = 1.0
    except Exception:
        pass

    model._x = x
    model._z = z
    model._u = u
    model._n = n
    model._m = m
    model._K = K
    model._node_id = node_id
    model._logger = logger

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = extract_solution(model, True, model._n, model._m, model._K, model._node_id,
                                       model._x, model._z, model._u)
                sol['objective_value'] = obj
                if model._logger:
                    model._logger.log_solution(obj, sol)
            except Exception:
                pass

    model.optimize(callback)

    if model.SolCount > 0:
        solution = extract_solution(model, False, n, m, K, node_id, x, z, u)
        solution['objective_value'] = model.ObjVal
    else:
        solution = heur_sol

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()