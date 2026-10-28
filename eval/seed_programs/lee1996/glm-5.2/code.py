import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

def compute_mst(active, interconnect_costs):
    if len(active) <= 1:
        return 0, []
    n_active = len(active)
    in_tree = [False] * n_active
    in_tree[0] = True
    edges = []
    total_cost = 0
    for _ in range(n_active - 1):
        best_cost = float('inf')
        best_edge = None
        best_idx = -1
        for u in range(n_active):
            if not in_tree[u]:
                continue
            for v in range(n_active):
                if in_tree[v]:
                    continue
                c = interconnect_costs[active[u]][active[v]]
                if c < best_cost:
                    best_cost = c
                    best_edge = [min(active[u], active[v]), max(active[u], active[v])]
                    best_idx = v
        if best_idx == -1:
            break
        edges.append(best_edge)
        total_cost += best_cost
        in_tree[best_idx] = True
    return total_cost, edges

def compute_cost(active_set, m, n, assign_costs, interconnect_costs, weights):
    if len(active_set) == 0:
        return float('inf'), None, None, None
    active = sorted(active_set)
    assignments = {}
    assign_cost = 0
    for i in range(m):
        best_j = active[0]
        best_c = assign_costs[i][active[0]]
        for j in active[1:]:
            c = assign_costs[i][j]
            if c < best_c:
                best_c = c
                best_j = j
        assignments[i] = best_j
        assign_cost += best_c
    tree_cost, tree_edges = compute_mst(active, interconnect_costs)
    act_cost = sum(weights[j] for j in active)
    total = assign_cost + tree_cost + act_cost
    return total, assignments, tree_edges, active

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    m = instance['parameters']['m']
    n = instance['parameters']['n']
    assign_costs = np.array(instance['cost_matrices']['assignment_costs']['matrix'], dtype=float)
    interconnect_costs = np.array(instance['cost_matrices']['interconnection_costs']['matrix'], dtype=float)
    weights = np.array(instance['steiner_nodes']['node_weights'], dtype=float)

    def make_solution_dict(cost, assign, tree, active):
        return {
            'objective_value': float(cost),
            'active_steiner_nodes': sorted(list(active)),
            'assignments': {str(i): int(assign[i]) for i in range(m)},
            'steiner_tree_edges': [list(e) for e in tree]
        }

    best_cost = float('inf')
    best_assign = None
    best_tree = None
    best_active = None

    def try_update(cost, assign, tree, active):
        nonlocal best_cost, best_assign, best_tree, best_active
        if cost < best_cost - 1e-9:
            best_cost = cost
            best_assign = dict(assign)
            best_tree = [list(e) for e in tree]
            best_active = list(active)
            if logger:
                logger.log_solution(best_cost, make_solution_dict(best_cost, best_assign, best_tree, best_active))
            return True
        return False

    # Heuristic 1: Greedy removal from all nodes
    if n > 0:
        active_set = set(range(n))
        cost, assign, tree, active = compute_cost(active_set, m, n, assign_costs, interconnect_costs, weights)
        try_update(cost, assign, tree, active)

        improved = True
        while improved:
            improved = False
            for j in sorted(active_set):
                new_set = active_set - {j}
                cost, assign, tree, active = compute_cost(new_set, m, n, assign_costs, interconnect_costs, weights)
                if try_update(cost, assign, tree, active):
                    active_set = new_set
                    improved = True
                    break

    # Heuristic 2: Greedy addition from best single node
    if n > 0:
        best_single = min(range(n), key=lambda j: sum(assign_costs[i][j] for i in range(m)) + weights[j])
        active_set2 = {best_single}
        cost, assign, tree, active = compute_cost(active_set2, m, n, assign_costs, interconnect_costs, weights)
        try_update(cost, assign, tree, active)

        improved = True
        while improved:
            improved = False
            for j in range(n):
                if j in active_set2:
                    continue
                new_set = active_set2 | {j}
                cost, assign, tree, active = compute_cost(new_set, m, n, assign_costs, interconnect_costs, weights)
                if try_update(cost, assign, tree, active):
                    active_set2 = new_set
                    improved = True
                    break

    # Heuristic 3: Local search swap
    if n > 0 and best_active is not None and time.time() - start_time < args.time_limit * 0.3:
        active_set3 = set(best_active)
        improved = True
        while improved and time.time() - start_time < args.time_limit * 0.3:
            improved = False
            for j_out in sorted(active_set3):
                for j_in in range(n):
                    if j_in in active_set3:
                        continue
                    new_set = (active_set3 - {j_out}) | {j_in}
                    cost, assign, tree, active = compute_cost(new_set, m, n, assign_costs, interconnect_costs, weights)
                    if try_update(cost, assign, tree, active):
                        active_set3 = new_set
                        improved = True
                        break
                if improved:
                    break

    # MIP with Gurobi
    remaining_time = args.time_limit - (time.time() - start_time) - 2
    if remaining_time > 5 and n > 0:
        model = gp.Model('steiner_network')
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining_time
        model.Params.OutputFlag = 0

        y = model.addVars(n, vtype=GRB.BINARY, name='y')
        x = model.addVars(m, n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='x')
        z = {}
        for j in range(n):
            for k in range(j+1, n):
                z[j, k] = model.addVar(vtype=GRB.BINARY, name=f'z_{j}_{k}')
        z_dummy = model.addVars(n, vtype=GRB.BINARY, name='z_dummy')

        f = {}
        for j in range(n):
            for k in range(n):
                if j != k:
                    f[j, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f_{j}_{k}')
        fd = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name='fd')

        model.update()

        obj = gp.LinExpr()
        for i in range(m):
            for j in range(n):
                obj += float(assign_costs[i][j]) * x[i, j]
        for j in range(n):
            for k in range(j+1, n):
                obj += float(interconnect_costs[j][k]) * z[j, k]
        for j in range(n):
            obj += float(weights[j]) * y[j]
        model.setObjective(obj, GRB.MINIMIZE)

        for i in range(m):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)

        for i in range(m):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j])

        if m > 0:
            model.addConstr(gp.quicksum(y[j] for j in range(n)) >= 1)

        for j in range(n):
            for k in range(j+1, n):
                model.addConstr(z[j, k] <= y[j])
                model.addConstr(z[j, k] <= y[k])

        for j in range(n):
            model.addConstr(z_dummy[j] <= y[j])

        model.addConstr(
            gp.quicksum(z[j, k] for j in range(n) for k in range(j+1, n)) +
            gp.quicksum(z_dummy[j] for j in range(n)) ==
            gp.quicksum(y[j] for j in range(n))
        )

        M_flow = n
        for j in range(n):
            inflow = fd[j] + gp.quicksum(f[k, j] for k in range(n) if k != j)
            outflow = gp.quicksum(f[j, k] for k in range(n) if k != j)
            model.addConstr(inflow - outflow == y[j])

        model.addConstr(gp.quicksum(fd[j] for j in range(n)) == gp.quicksum(y[j] for j in range(n)))

        for j in range(n):
            model.addConstr(fd[j] <= M_flow * z_dummy[j])
            for k in range(n):
                if j != k:
                    jk, kk = min(j, k), max(j, k)
                    model.addConstr(f[j, k] <= M_flow * z[jk, kk])

        # Warm start
        if best_assign is not None:
            for j in range(n):
                y[j].Start = 1 if j in best_active else 0
            for i in range(m):
                for j in range(n):
                    x[i, j].Start = 1.0 if best_assign.get(i) == j else 0.0
            for j in range(n):
                for k in range(j+1, n):
                    z[j, k].Start = 1 if [j, k] in best_tree else 0
            for j in range(n):
                z_dummy[j].Start = 0

        model.optimize()

        if model.SolCount > 0:
            mip_obj = model.ObjVal
            if mip_obj < best_cost - 1e-6:
                best_cost = mip_obj
                best_active = [j for j in range(n) if y[j].X > 0.5]
                best_assign = {}
                for i in range(m):
                    for j in range(n):
                        if x[i, j].X > 0.5:
                            best_assign[i] = j
                            break
                    if i not in best_assign:
                        # Fallback: assign to first active
                        if best_active:
                            best_assign[i] = best_active[0]
                best_tree = []
                for j in range(n):
                    for k in range(j+1, n):
                        if z[j, k].X > 0.5:
                            best_tree.append([j, k])
                if logger:
                    logger.log_solution(best_cost, make_solution_dict(best_cost, best_assign, best_tree, best_active))

    # Ensure we have a valid solution
    if best_assign is None:
        if n > 0:
            best_active = [0]
            best_assign = {i: 0 for i in range(m)}
            best_tree = []
            best_cost = float(sum(assign_costs[i][0] for i in range(m)) + weights[0])
        else:
            best_active = []
            best_assign = {}
            best_tree = []
            best_cost = 0.0

    solution = make_solution_dict(best_cost, best_assign, best_tree, best_active)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    solve()