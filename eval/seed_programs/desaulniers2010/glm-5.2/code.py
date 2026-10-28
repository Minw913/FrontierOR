import argparse
import json
import math
import time
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_arcs(instance):
    n = instance['num_customers']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']

    e_0 = depot['time_window'][0]
    l_0 = depot['time_window'][1]

    tw = [None] * (n + 2)
    st = [0] * (n + 2)
    tw[0] = (e_0, l_0)
    tw[n + 1] = (e_0, l_0)

    cust_by_id = {c['id']: c for c in customers}
    for i in range(1, n + 1):
        c = cust_by_id[i]
        tw[i] = (c['time_window'][0], c['time_window'][1])
        st[i] = c['service_time']

    def get_dist(i, j):
        ii = 0 if i == n + 1 else i
        jj = 0 if j == n + 1 else j
        return dist[ii][jj]

    arcs = []
    for i in range(n + 2):
        if i == n + 1:
            continue
        for j in range(n + 2):
            if j == 0 or i == j:
                continue
            d_ij = get_dist(i, j)
            tt = st[i] + d_ij
            if tw[i][0] + tt <= tw[j][1]:
                arcs.append((i, j, float(d_ij), float(tt)))

    return arcs, tw


def trivial_solution(instance, arcs):
    n = instance['num_customers']
    Q = instance['vehicle_capacity']
    K_max = instance['num_vehicles_upper_bound']
    cust_by_id = {c['id']: c for c in instance['customers']}

    arc_dict = {(a[0], a[1]): a for a in arcs}

    routes = []
    deliveries = {str(i): {'demand': cust_by_id[i]['demand'], 'total_delivered': 0.0}
                  for i in range(1, n + 1)}

    total_obj = 0.0
    vid = 0

    for i in range(1, n + 1):
        d_i = cust_by_id[i]['demand']
        if d_i == 0:
            continue
        if (0, i) not in arc_dict or (i, n + 1) not in arc_dict:
            return None
        remaining = d_i
        while remaining > 0:
            if vid >= K_max:
                return None
            qty = min(remaining, Q)
            route = [0, i, n + 1]
            d_out = arc_dict[(0, i)][2]
            d_in = arc_dict[(i, n + 1)][2]
            total_obj += d_out + d_in
            routes.append({
                'vehicle': vid,
                'route': route,
                'deliveries': {str(i): float(qty)}
            })
            deliveries[str(i)]['total_delivered'] += float(qty)
            remaining -= qty
            vid += 1

    if vid == 0:
        return {
            'objective_value': 0.0,
            'num_vehicles_used': 0,
            'routes': [],
            'deliveries': deliveries
        }

    return {
        'objective_value': float(total_obj),
        'num_vehicles_used': vid,
        'routes': routes,
        'deliveries': deliveries
    }


def build_solution_from_values(x_vals, q_vals, z_vals, K, n, arc_list, arcs_out, cust_by_id):
    routes = []
    deliveries = {str(i): {'demand': cust_by_id[i]['demand'], 'total_delivered': 0.0}
                  for i in range(1, n + 1)}

    vehicle_id = 0
    total_obj = 0.0

    for k in range(K):
        if z_vals.get(k, 0) < 0.5:
            continue
        route = [0]
        current = 0
        vehicle_deliveries = {}
        visited = set()
        guard = 0
        while current != n + 1:
            found = False
            for idx in arcs_out[current]:
                if x_vals.get((idx, k), 0) > 0.5:
                    j = arc_list[idx][1]
                    total_obj += arc_list[idx][2]
                    route.append(j)
                    if j != n + 1 and j not in visited:
                        visited.add(j)
                        q_val = q_vals.get((j, k), 0)
                        if q_val > 1e-6:
                            vehicle_deliveries[str(j)] = float(q_val)
                            deliveries[str(j)]['total_delivered'] += float(q_val)
                    current = j
                    found = True
                    break
            if not found:
                return None
            guard += 1
            if guard > n + 5:
                return None

        routes.append({
            'vehicle': vehicle_id,
            'route': route,
            'deliveries': vehicle_deliveries
        })
        vehicle_id += 1

    if vehicle_id == 0:
        return None

    # Verify all demands met
    for i in range(1, n + 1):
        if cust_by_id[i]['demand'] > 0:
            if deliveries[str(i)]['total_delivered'] < cust_by_id[i]['demand'] - 1e-4:
                return None

    return {
        'objective_value': float(total_obj),
        'num_vehicles_used': vehicle_id,
        'routes': routes,
        'deliveries': deliveries
    }


def solve_mip(instance, arcs, tw, time_limit, logger, start_time, K):
    n = instance['num_customers']
    Q = instance['vehicle_capacity']
    K_max = instance['num_vehicles_upper_bound']
    cust_by_id = {c['id']: c for c in instance['customers']}
    total_demand = sum(c['demand'] for c in instance['customers'])
    K_min = math.ceil(total_demand / Q) if total_demand > 0 else 0

    arc_list = list(arcs)
    num_arcs = len(arc_list)

    arcs_out = {i: [] for i in range(n + 2)}
    arcs_in = {j: [] for j in range(n + 2)}
    for idx, (i, j, d, t) in enumerate(arc_list):
        arcs_out[i].append(idx)
        arcs_in[j].append(idx)

    for i in range(1, n + 1):
        if cust_by_id[i]['demand'] > 0 and not arcs_in[i]:
            return None

    idle_idx = None
    for idx in arcs_out[0]:
        if arc_list[idx][1] == n + 1:
            idle_idx = idx
            break

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    elapsed = time.time() - start_time
    remaining = max(1, int(time_limit - elapsed) - 1)
    model.setParam('TimeLimit', remaining)

    x = {}
    for k in range(K):
        for idx in range(num_arcs):
            d = arc_list[idx][2]
            x[idx, k] = model.addVar(vtype=GRB.BINARY, obj=d, name=f'x_{idx}_{k}')

    q = {}
    for k in range(K):
        for i in range(1, n + 1):
            q[i, k] = model.addVar(lb=0, ub=min(cust_by_id[i]['demand'], Q),
                                   vtype=GRB.CONTINUOUS, name=f'q_{i}_{k}')

    tvar = {}
    for k in range(K):
        for i in range(n + 2):
            tvar[i, k] = model.addVar(lb=tw[i][0], ub=tw[i][1],
                                      vtype=GRB.CONTINUOUS, name=f't_{i}_{k}')

    z = {}
    for k in range(K):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f'z_{k}')

    model.update()

    # Demand coverage
    for i in range(1, n + 1):
        if cust_by_id[i]['demand'] > 0:
            model.addConstr(gp.quicksum(x[idx, k] for k in range(K) for idx in arcs_in[i]) >= 1)

    # Flow conservation
    for k in range(K):
        for i in range(1, n + 1):
            model.addConstr(
                gp.quicksum(x[idx, k] for idx in arcs_out[i]) ==
                gp.quicksum(x[idx, k] for idx in arcs_in[i]))

    # Single visit per customer per vehicle (simple path)
    for k in range(K):
        for i in range(1, n + 1):
            model.addConstr(gp.quicksum(x[idx, k] for idx in arcs_in[i]) <= 1)

    # Depot
    for k in range(K):
        model.addConstr(gp.quicksum(x[idx, k] for idx in arcs_out[0]) == 1)
        model.addConstr(gp.quicksum(x[idx, k] for idx in arcs_in[n + 1]) == 1)

    # z definition
    cust_arcs_out_0 = [idx for idx in arcs_out[0] if arc_list[idx][1] != n + 1]
    for k in range(K):
        model.addConstr(z[k] == gp.quicksum(x[idx, k] for idx in cust_arcs_out_0))

    # Idle arc
    if idle_idx is not None:
        for k in range(K):
            model.addConstr(x[idle_idx, k] == 1 - z[k])

    # Vehicle count
    if K_min > 0:
        model.addConstr(gp.quicksum(z[k] for k in range(K)) >= K_min)
    model.addConstr(gp.quicksum(z[k] for k in range(K)) <= K_max)

    # Symmetry breaking
    for k in range(K - 1):
        model.addConstr(z[k] >= z[k + 1])

    # Capacity
    for k in range(K):
        model.addConstr(gp.quicksum(q[i, k] for i in range(1, n + 1)) <= Q * z[k])

    # Link q to visit
    for k in range(K):
        for i in range(1, n + 1):
            model.addConstr(q[i, k] <= Q * gp.quicksum(x[idx, k] for idx in arcs_in[i]))

    # Demand fulfillment
    for i in range(1, n + 1):
        model.addConstr(gp.quicksum(q[i, k] for k in range(K)) >= cust_by_id[i]['demand'])

    # Time propagation with arc-specific Big-M
    for k in range(K):
        for idx in range(num_arcs):
            i, j, d, t = arc_list[idx]
            M_ij = tw[i][1] + t - tw[j][0] + 1
            model.addConstr(tvar[j, k] >= tvar[i, k] + t - M_ij * (1 - x[idx, k]))

    model.ModelSense = GRB.MINIMIZE

    best_obj = [float('inf')]
    best_sol = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-9:
                best_obj[0] = obj
                if logger:
                    try:
                        x_keys = [(idx, k) for k in range(K) for idx in range(num_arcs)]
                        x_vars_list = [x[idx, k] for (idx, k) in x_keys]
                        x_vals_list = model.cbGetSolution(x_vars_list)
                        x_vals = dict(zip(x_keys, x_vals_list))

                        q_keys = [(i, k) for k in range(K) for i in range(1, n + 1)]
                        q_vars_list = [q[i, k] for (i, k) in q_keys]
                        q_vals_list = model.cbGetSolution(q_vars_list)
                        q_vals = dict(zip(q_keys, q_vals_list))

                        z_vars_list = [z[k] for k in range(K)]
                        z_vals_list = model.cbGetSolution(z_vars_list)
                        z_vals = {k: z_vals_list[k] for k in range(K)}

                        sol = build_solution_from_values(x_vals, q_vals, z_vals, K, n,
                                                          arc_list, arcs_out, cust_by_id)
                        if sol is not None:
                            best_sol[0] = sol
                            logger.log_solution(obj, sol)
                        else:
                            logger.log(obj)
                    except Exception:
                        logger.log(obj)

    model.optimize(callback)

    if model.SolCount > 0:
        x_vals = {(idx, k): x[idx, k].X for idx in range(num_arcs) for k in range(K)}
        q_vals = {(i, k): q[i, k].X for i in range(1, n + 1) for k in range(K)}
        z_vals = {k: z[k].X for k in range(K)}
        final_sol = build_solution_from_values(x_vals, q_vals, z_vals, K, n,
                                               arc_list, arcs_out, cust_by_id)
        if final_sol is not None:
            return final_sol
    return best_sol[0]


def write_solution(sol, path):
    with open(path, 'w') as f:
        json.dump(sol, f, indent=2)


def empty_solution(instance):
    return {
        'objective_value': 0.0,
        'num_vehicles_used': 0,
        'routes': [],
        'deliveries': {str(i): {'demand': c['demand'], 'total_delivered': 0.0}
                       for i, c in enumerate(instance['customers'], 1)}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()
    instance = load_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    arcs, tw = build_arcs(instance)

    fallback = trivial_solution(instance, arcs)
    best_solution = fallback
    if best_solution and logger:
        logger.log_solution(best_solution['objective_value'], best_solution)

    if not HAS_GUROBI:
        if best_solution is None:
            best_solution = empty_solution(instance)
        write_solution(best_solution, args.solution_path)
        return

    n = instance['num_customers']
    Q = instance['vehicle_capacity']
    K_max = instance['num_vehicles_upper_bound']
    total_demand = sum(c['demand'] for c in instance['customers'])
    K_min = math.ceil(total_demand / Q) if total_demand > 0 else 0

    if K_min == 0 or K_max < K_min:
        if best_solution is None:
            best_solution = empty_solution(instance)
        write_solution(best_solution, args.solution_path)
        return

    K_target = min(K_max, K_min + 5)
    arc_count = len(arcs)
    while K_target > K_min and K_target * arc_count > 300000:
        K_target -= 1

    sol = solve_mip(instance, arcs, tw, args.time_limit, logger, start_time, K_target)
    if sol is not None:
        if best_solution is None or sol['objective_value'] < best_solution['objective_value'] - 1e-9:
            best_solution = sol
            if logger:
                logger.log_solution(sol['objective_value'], sol)

    # If MIP failed, try with more vehicles
    if sol is None:
        for extra in [5, 10, 20, 50]:
            K_alt = min(K_max, K_target + extra)
            if K_alt <= K_target:
                continue
            if K_alt * arc_count > 500000:
                continue
            remaining = args.time_limit - int(time.time() - start_time)
            if remaining < 3:
                break
            sol = solve_mip(instance, arcs, tw, args.time_limit, logger, start_time, K_alt)
            if sol is not None:
                if best_solution is None or sol['objective_value'] < best_solution['objective_value'] - 1e-9:
                    best_solution = sol
                    if logger:
                        logger.log_solution(sol['objective_value'], sol)
                break

    if best_solution is None:
        best_solution = empty_solution(instance)

    write_solution(best_solution, args.solution_path)


if __name__ == '__main__':
    main()