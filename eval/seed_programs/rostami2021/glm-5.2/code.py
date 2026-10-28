import argparse
import json
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from itertools import combinations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    n = instance['n']
    S = instance['num_scenarios']
    chi = instance['cost_parameters']['chi']
    alpha = instance['cost_parameters']['alpha']
    delta = instance['cost_parameters']['delta']
    p_hubs = instance.get('p_hubs')
    fixed_costs = instance['fixed_costs']
    distances = instance['distances']
    hub_capacities = instance.get('hub_capacities')
    scenarios = instance['scenarios']

    # Precompute outgoing, incoming flows and demand matrices per scenario
    O = []   # O[s][i] = total outgoing flow from node i in scenario s
    I_flow = []  # I_flow[s][i] = total incoming flow to node i in scenario s
    W = []   # W[s][i][k] = demand from i to k in scenario s
    probs = []
    for sc in scenarios:
        demands = sc['demands']
        probs.append(sc['probability'])
        O_s = [sum(demands[i]) for i in range(n)]
        I_s = [sum(demands[k][i] for k in range(n)) for i in range(n)]
        O.append(O_s)
        I_flow.append(I_s)
        W.append(demands)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ------------------------------------------------------------------
    # Heuristic solution
    # ------------------------------------------------------------------
    heur_obj, heur_hubs, heur_alloc = compute_heuristic(
        n, S, chi, alpha, delta, distances, fixed_costs, p_hubs,
        hub_capacities, O, I_flow, W, probs
    )

    heur_solution = build_solution_dict(heur_obj, heur_hubs, heur_alloc, n, S)
    if logger:
        logger.log_solution(heur_obj, heur_solution)

    # ------------------------------------------------------------------
    # Decide whether to attempt MILP
    # ------------------------------------------------------------------
    total_h = 0
    for s in range(S):
        for i in range(n):
            if O[s][i] > 0:
                total_h += n * (n - 1)

    use_milp = (total_h <= 400000 and n <= 50)

    if use_milp:
        try:
            result = solve_milp(
                n, S, chi, alpha, delta, distances, fixed_costs, p_hubs,
                hub_capacities, O, I_flow, W, probs, args.time_limit,
                logger, heur_hubs, heur_alloc
            )
            obj_val, hubs, allocations, z_vals, x_vals = result
        except Exception as e:
            print(f"MILP failed: {e}", file=sys.stderr)
            obj_val, hubs, allocations, z_vals, x_vals = (
                heur_obj, heur_hubs,
                heur_solution['allocations'],
                heur_solution['z_values'],
                heur_solution['x_values'],
            )
    else:
        obj_val, hubs, allocations, z_vals, x_vals = (
            heur_obj, heur_hubs,
            heur_solution['allocations'],
            heur_solution['z_values'],
            heur_solution['x_values'],
        )

    solution = {
        "objective_value": float(obj_val),
        "hubs": hubs,
        "allocations": allocations,
        "z_values": z_vals,
        "x_values": x_vals,
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


# ---------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------
def evaluate_hubs_greedy(hubs, n, S, chi, alpha, delta, distances,
                         fixed_costs, O, I_flow, W, probs, hub_capacities, p_hubs):
    """Evaluate a set of hubs using greedy nearest-hub assignment."""
    hub_set = set(hubs)
    if not hub_set:
        return float('inf'), {}

    allocations = {}

    for s in range(S):
        alloc = {}
        # Assign hub nodes to themselves
        for i in range(n):
            if i in hub_set:
                alloc[i] = i

        # Assign non-hub nodes greedily
        if hub_capacities is not None:
            # Capacity-aware greedy: sort by outgoing flow descending
            remaining_caps = {h: hub_capacities[h] for h in hubs}
            for h in hubs:
                remaining_caps[h] -= O[s][h]  # hub's own flow

            non_hub_nodes = [i for i in range(n) if i not in hub_set]
            non_hub_nodes.sort(key=lambda i: -O[s][i])
            for i in non_hub_nodes:
                best_hub = None
                best_cost = float('inf')
                for h in hubs:
                    if remaining_caps[h] >= O[s][i]:
                        cost = distances[i][h] * (chi * O[s][i] + delta * I_flow[s][i])
                        if cost < best_cost:
                            best_cost = cost
                            best_hub = h
                if best_hub is None:
                    # Infeasible – pick nearest ignoring capacity
                    for h in hubs:
                        cost = distances[i][h] * (chi * O[s][i] + delta * I_flow[s][i])
                        if cost < best_cost:
                            best_cost = cost
                            best_hub = h
                if best_hub is None:
                    best_hub = hubs[0]
                alloc[i] = best_hub
                if hub_capacities is not None and best_hub in remaining_caps:
                    remaining_caps[best_hub] -= O[s][i]
        else:
            for i in range(n):
                if i not in hub_set:
                    best_hub = hubs[0]
                    best_cost = float('inf')
                    for h in hubs:
                        cost = distances[i][h] * (chi * O[s][i] + delta * I_flow[s][i])
                        if cost < best_cost:
                            best_cost = cost
                            best_hub = h
                    alloc[i] = best_hub

        allocations[s] = alloc

    # Compute objective
    obj = 0.0
    if p_hubs is None:
        for h in hubs:
            obj += fixed_costs[h]

    for s in range(S):
        p_s = probs[s]
        alloc = allocations[s]
        for i in range(n):
            h = alloc[i]
            obj += p_s * distances[i][h] * (chi * O[s][i] + delta * I_flow[s][i])
        for i in range(n):
            for k in range(n):
                if W[s][i][k] > 0:
                    h_i = alloc[i]
                    h_k = alloc[k]
                    obj += p_s * alpha * W[s][i][k] * distances[h_i][h_k]

    return obj, allocations


def compute_heuristic(n, S, chi, alpha, delta, distances, fixed_costs,
                      p_hubs, hub_capacities, O, I_flow, W, probs):
    best_obj = float('inf')
    best_hubs = None
    best_alloc = None

    if p_hubs is not None:
        p = min(p_hubs, n)
        # Enumerate if feasible
        num_combos = 1
        for i in range(p):
            num_combos = num_combos * (n - i) // (i + 1)
        if num_combos <= 5000:
            for combo in combinations(range(n), p):
                hubs = list(combo)
                obj, alloc = evaluate_hubs_greedy(
                    hubs, n, S, chi, alpha, delta, distances, fixed_costs,
                    O, I_flow, W, probs, hub_capacities, p_hubs
                )
                if obj < best_obj:
                    best_obj = obj
                    best_hubs = hubs
                    best_alloc = alloc
        else:
            # Greedy add
            selected = []
            remaining = list(range(n))
            for _ in range(p):
                best_add = None
                best_add_obj = float('inf')
                best_add_alloc = None
                for node in remaining:
                    hubs = selected + [node]
                    obj, alloc = evaluate_hubs_greedy(
                        hubs, n, S, chi, alpha, delta, distances, fixed_costs,
                        O, I_flow, W, probs, hub_capacities, p_hubs
                    )
                    if obj < best_add_obj:
                        best_add_obj = obj
                        best_add = node
                        best_add_alloc = alloc
                if best_add is not None:
                    selected.append(best_add)
                    remaining.remove(best_add)
                    best_obj = best_add_obj
                    best_hubs = selected[:]
                    best_alloc = best_add_alloc

            # Swap local search
            improved = True
            while improved:
                improved = False
                for idx in range(len(best_hubs)):
                    for node in range(n):
                        if node not in best_hubs:
                            new_hubs = best_hubs[:]
                            new_hubs[idx] = node
                            obj, alloc = evaluate_hubs_greedy(
                                new_hubs, n, S, chi, alpha, delta, distances,
                                fixed_costs, O, I_flow, W, probs,
                                hub_capacities, p_hubs
                            )
                            if obj < best_obj:
                                best_obj = obj
                                best_hubs = new_hubs
                                best_alloc = alloc
                                improved = True
    else:
        # Fixed-cost variant: add/drop heuristic
        # Start with all nodes as hubs
        best_hubs = list(range(n))
        best_obj, best_alloc = evaluate_hubs_greedy(
            best_hubs, n, S, chi, alpha, delta, distances, fixed_costs,
            O, I_flow, W, probs, hub_capacities, p_hubs
        )

        # Drop phase
        improved = True
        while improved and len(best_hubs) > 1:
            improved = False
            for h in list(best_hubs):
                new_hubs = [x for x in best_hubs if x != h]
                obj, alloc = evaluate_hubs_greedy(
                    new_hubs, n, S, chi, alpha, delta, distances, fixed_costs,
                    O, I_flow, W, probs, hub_capacities, p_hubs
                )
                if obj < best_obj:
                    best_obj = obj
                    best_hubs = new_hubs
                    best_alloc = alloc
                    improved = True
                    break

        # Add phase
        improved = True
        while improved:
            improved = False
            for node in range(n):
                if node not in best_hubs:
                    new_hubs = best_hubs + [node]
                    obj, alloc = evaluate_hubs_greedy(
                        new_hubs, n, S, chi, alpha, delta, distances,
                        fixed_costs, O, I_flow, W, probs,
                        hub_capacities, p_hubs
                    )
                    if obj < best_obj:
                        best_obj = obj
                        best_hubs = new_hubs
                        best_alloc = alloc
                        improved = True
                        break

    if best_hubs is None:
        best_hubs = [0]
        best_obj, best_alloc = evaluate_hubs_greedy(
            best_hubs, n, S, chi, alpha, delta, distances, fixed_costs,
            O, I_flow, W, probs, hub_capacities, p_hubs
        )

    return best_obj, best_hubs, best_alloc


# ---------------------------------------------------------------------
# Solution dict builder
# ---------------------------------------------------------------------
def build_solution_dict(obj_val, hubs, alloc, n, S):
    hub_set = set(hubs)
    z_vals = [0.0] * n
    for h in hubs:
        z_vals[h] = 1.0

    allocations = {}
    x_vals = {}
    for s in range(S):
        a = {}
        xv = {}
        for i in range(n):
            assigned = alloc[s][i]
            a[str(i)] = assigned
            if i not in hub_set:
                xv[str(i)] = {str(assigned): 1.0}
        allocations[str(s)] = a
        x_vals[str(s)] = xv

    return {
        "objective_value": float(obj_val),
        "hubs": hubs,
        "allocations": allocations,
        "z_values": z_vals,
        "x_values": x_vals,
    }


# ---------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------
def solve_milp(n, S, chi, alpha, delta, distances, fixed_costs, p_hubs,
               hub_capacities, O, I_flow, W, probs, time_limit, logger,
               heur_hubs, heur_alloc):

    model = gp.Model('hub_location')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - 5)  # reserve time for output
    model.Params.OutputFlag = 0

    # ---- Variables ----
    z = model.addVars(n, vtype=GRB.BINARY, name='z')

    x = {}
    for s in range(S):
        for i in range(n):
            for j in range(n):
                x[s, i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{s}_{i}_{j}')

    # Active (s, i) pairs (nodes with outgoing flow)
    active_si = [(s, i) for s in range(S) for i in range(n) if O[s][i] > 0]

    # G[s, i, l] = sum_k W[s][i][k] * x[s, k, l]
    G_idx = [(s, i, l) for (s, i) in active_si for l in range(n)]
    G = model.addVars(G_idx, vtype=GRB.CONTINUOUS, lb=0, name='G')

    # H[s, i, j, l] = G[s, i, l] * x[s, i, j]  (for j != l)
    H_idx = [(s, i, j, l) for (s, i) in active_si
             for j in range(n) for l in range(n) if j != l]
    H = model.addVars(H_idx, vtype=GRB.CONTINUOUS, lb=0, name='H')

    # ---- Constraints ----
    # Assignment: each node assigned to exactly one hub
    model.addConstrs(
        (gp.quicksum(x[s, i, j] for j in range(n)) == 1
         for s in range(S) for i in range(n)),
        name='assign'
    )

    # Can only assign to open hubs
    model.addConstrs(
        (x[s, i, j] <= z[j]
         for s in range(S) for i in range(n) for j in range(n)),
        name='link'
    )

    # Hub self-assignment
    model.addConstrs(
        (x[s, j, j] == z[j] for s in range(S) for j in range(n)),
        name='self'
    )

    # At least one hub
    model.addConstr(gp.quicksum(z[j] for j in range(n)) >= 1, name='atleast1')

    # p-hub median
    if p_hubs is not None:
        model.addConstr(gp.quicksum(z[j] for j in range(n)) == p_hubs, name='pcount')

    # Capacity
    if hub_capacities is not None:
        model.addConstrs(
            (gp.quicksum(O[s][i] * x[s, i, j] for i in range(n)) <= hub_capacities[j] * z[j]
             for s in range(S) for j in range(n)),
            name='cap'
        )

    # G definition
    model.addConstrs(
        (G[s, i, l] == gp.quicksum(W[s][i][k] * x[s, k, l] for k in range(n))
         for (s, i) in active_si for l in range(n)),
        name='Gdef'
    )

    # H linearization: H[s,i,j,l] = G[s,i,l] * x[s,i,j]
    for (s, i) in active_si:
        M_si = O[s][i]
        for j in range(n):
            for l in range(n):
                if j != l:
                    model.addConstr(H[s, i, j, l] <= G[s, i, l], name=f'Hub1_{s}_{i}_{j}_{l}')
                    model.addConstr(H[s, i, j, l] <= M_si * x[s, i, j], name=f'Hub2_{s}_{i}_{j}_{l}')
                    model.addConstr(H[s, i, j, l] >= G[s, i, l] - M_si * (1 - x[s, i, j]), name=f'Hlb_{s}_{i}_{j}_{l}')

    # ---- Objective ----
    obj = gp.LinExpr()

    if p_hubs is None:
        obj += gp.quicksum(fixed_costs[j] * z[j] for j in range(n))

    for s in range(S):
        p_s = probs[s]
        # Collection + distribution (linear)
        for i in range(n):
            for j in range(n):
                cost_ij = distances[i][j] * (chi * O[s][i] + delta * I_flow[s][i])
                if cost_ij != 0:
                    obj += p_s * cost_ij * x[s, i, j]

        # Transfer cost (linear via H)
        for (s2, i) in active_si:
            if s2 != s:
                continue
            for j in range(n):
                for l in range(n):
                    if j != l and distances[j][l] > 0:
                        coef = p_s * alpha * distances[j][l]
                        if coef != 0:
                            obj += coef * H[s, i, j, l]

    model.setObjective(obj, GRB.MINIMIZE)

    # ---- Warm start ----
    try:
        for j in range(n):
            z[j].Start = 1 if j in heur_hubs else 0
        for s in range(S):
            for i in range(n):
                for j in range(n):
                    x[s, i, j].Start = 1 if heur_alloc[s][i] == j else 0
    except Exception:
        pass

    # ---- Callback for logging ----
    best_cb_obj = [float('inf')]
    best_cb_sol = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_cb_obj[0]:
                best_cb_obj[0] = obj_val
                z_cb = model.cbGetSolution(z)
                x_cb = {}
                for s in range(S):
                    for i in range(n):
                        for j in range(n):
                            x_cb[s, i, j] = model.cbGetSolution(x[s, i, j])

                z_vals_cb = [0] * n
                for j in range(n):
                    z_vals_cb[j] = 1 if z_cb[j] > 0.5 else 0
                hubs_cb = [j for j in range(n) if z_vals_cb[j] == 1]

                alloc = {}
                xv = {}
                for s in range(S):
                    a = {}
                    xv_s = {}
                    for i in range(n):
                        assigned = None
                        for j in range(n):
                            if x_cb[s, i, j] > 0.5:
                                assigned = j
                                break
                        if assigned is None:
                            assigned = hubs_cb[0] if hubs_cb else 0
                        a[str(i)] = assigned
                        if z_vals_cb[i] == 0:
                            xv_s[str(i)] = {str(assigned): 1.0}
                    alloc[str(s)] = a
                    xv[str(s)] = xv_s

                sol_dict = {
                    "objective_value": float(obj_val),
                    "hubs": hubs_cb,
                    "allocations": alloc,
                    "z_values": [float(v) for v in z_vals_cb],
                    "x_values": xv,
                }
                best_cb_sol[0] = sol_dict
                if logger:
                    logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    # ---- Extract solution ----
    if model.SolCount > 0:
        z_vals = [0] * n
        for j in range(n):
            z_vals[j] = 1 if z[j].X > 0.5 else 0
        hubs = [j for j in range(n) if z_vals[j] == 1]

        if not hubs:
            hubs = [0]
            z_vals[0] = 1

        allocations = {}
        x_vals = {}
        for s in range(S):
            a = {}
            xv = {}
            for i in range(n):
                assigned = None
                for j in range(n):
                    if x[s, i, j].X > 0.5:
                        assigned = j
                        break
                if assigned is None:
                    assigned = hubs[0] if hubs else 0
                a[str(i)] = assigned
                if z_vals[i] == 0:
                    xv[str(i)] = {str(assigned): 1.0}
            allocations[str(s)] = a
            x_vals[str(s)] = xv

        obj_val = model.ObjVal
        z_vals_float = [float(v) for v in z_vals]

        # Use best callback solution if better
        if best_cb_sol[0] is not None and best_cb_obj[0] < obj_val:
            return (best_cb_obj[0], best_cb_sol[0]['hubs'],
                    best_cb_sol[0]['allocations'],
                    best_cb_sol[0]['z_values'],
                    best_cb_sol[0]['x_values'])

        return obj_val, hubs, allocations, z_vals_float, x_vals
    else:
        # Fallback to heuristic
        hub_set = set(heur_hubs)
        z_vals = [0.0] * n
        for h in heur_hubs:
            z_vals[h] = 1.0

        allocations = {}
        x_vals = {}
        for s in range(S):
            a = {}
            xv = {}
            for i in range(n):
                assigned = heur_alloc[s][i]
                a[str(i)] = assigned
                if i not in hub_set:
                    xv[str(i)] = {str(assigned): 1.0}
            allocations[str(s)] = a
            x_vals[str(s)] = xv

        return heur_obj, heur_hubs, allocations, z_vals, x_vals


if __name__ == '__main__':
    main()