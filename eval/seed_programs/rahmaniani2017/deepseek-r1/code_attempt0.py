import argparse
import json
import time
import math
import gurobipy as grb
from gurobipy import GRB
import random
from solution_logger import SolutionLogger

def build_trivial_solution(instance):
    nodes = instance['nodes']
    commodities = instance['commodities']
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    
    n = len(nodes)
    fixed_costs = [node['fixed_cost'] for node in nodes]
    min_fixed_cost = min(fixed_costs)
    min_index = fixed_costs.index(min_fixed_cost)
    open_hubs = [min_index]
    
    assignments = []
    total_cost = min_fixed_cost
    for idx_k, com in enumerate(commodities):
        o = com['origin']
        d = com['destination']
        flow = com['flow']
        i = min_index
        j = min_index
        cost_val = flow * (chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d])
        total_cost += cost_val
        hub_edge = [i] if i == j else sorted([i, j])
        assignments.append({
            "commodity": idx_k,
            "origin": o,
            "destination": d,
            "hub_edge": hub_edge,
            "cost": cost_val
        })
    
    return {
        "objective_value": total_cost,
        "open_hubs": open_hubs,
        "assignments": assignments
    }

def build_solution(S, instance, deadline=None):
    nodes = instance['nodes']
    commodities = instance['commodities']
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    
    n = len(nodes)
    fixed_costs = [node['fixed_cost'] for node in nodes]
    open_hubs = sorted(S)
    total_cost = sum(fixed_costs[i] for i in S)
    assignments = []
    for idx_k, com in enumerate(commodities):
        if deadline is not None and time.time() > deadline:
            return None
        o = com['origin']
        d = com['destination']
        flow = com['flow']
        best_cost_val = float('inf')
        best_i = None
        best_j = None
        for i in S:
            for j in S:
                cost_val = chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d]
                if cost_val < best_cost_val:
                    best_cost_val = cost_val
                    best_i = i
                    best_j = j
        transport_cost = flow * best_cost_val
        total_cost += transport_cost
        if best_i == best_j:
            hub_edge = [best_i]
        else:
            hub_edge = sorted([best_i, best_j])
        assignments.append({
            "commodity": idx_k,
            "origin": o,
            "destination": d,
            "hub_edge": hub_edge,
            "cost": transport_cost
        })
    
    return {
        "objective_value": total_cost,
        "open_hubs": open_hubs,
        "assignments": assignments
    }

def compute_total_cost(S, instance, deadline=None):
    nodes = instance['nodes']
    commodities = instance['commodities']
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    
    if deadline is not None and time.time() > deadline:
        return float('inf')
    total_fixed = sum(nodes[i]['fixed_cost'] for i in S)
    total_transport = 0
    for com in commodities:
        if deadline is not None and time.time() > deadline:
            return float('inf')
        o = com['origin']
        d = com['destination']
        flow = com['flow']
        best_cost_val = float('inf')
        for i in S:
            best_j_cost = float('inf')
            for j in S:
                cost_val = chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d]
                if cost_val < best_j_cost:
                    best_j_cost = cost_val
            if best_j_cost < best_cost_val:
                best_cost_val = best_j_cost
        total_transport += flow * best_cost_val
    return total_fixed + total_transport

def run_heuristic(instance, time_limit, logger=None):
    start_time = time.time()
    deadline = start_time + time_limit
    nodes_data = instance['nodes']
    n = len(nodes_data)
    fixed_costs = [node['fixed_cost'] for node in nodes_data]
    commodities = instance['commodities']
    distances = instance['distances']
    
    current_S = set()
    best_S = None
    best_cost = float('inf')
    
    for i in range(n):
        if time.time() > deadline:
            break
        S_candidate = {i}
        cost_candidate = compute_total_cost(S_candidate, instance, deadline)
        if cost_candidate < best_cost:
            best_cost = cost_candidate
            best_S = S_candidate
            if logger:
                logger.log(best_cost)
    if best_S is None:
        return build_trivial_solution(instance)
    current_S = best_S.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        moves = []
        for j in range(n):
            if j not in current_S:
                moves.append(('add', j))
        if len(current_S) > 1:
            for j in current_S:
                moves.append(('remove', j))
        for j in current_S:
            for k in range(n):
                if k not in current_S:
                    moves.append(('swap', j, k))
        random.shuffle(moves)
        for move in moves:
            if time.time() > deadline:
                break
            if move[0] == 'add':
                j = move[1]
                new_S = current_S | {j}
            elif move[0] == 'remove':
                j = move[1]
                new_S = current_S - {j}
            else:
                j, k = move[1], move[2]
                new_S = current_S - {j} | {k}
            cost_new = compute_total_cost(new_S, instance, deadline)
            if cost_new < best_cost:
                best_cost = cost_new
                best_S = new_S.copy()
                current_S = new_S.copy()
                improved = True
                if logger:
                    logger.log(best_cost)
                break
    solution = build_solution(best_S, instance, deadline)
    return solution

def run_MIP(instance, time_limit, logger=None):
    nodes_data = instance['nodes']
    n = len(nodes_data)
    commodities = instance['commodities']
    num_commodities = len(commodities)
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    fixed_costs = [node['fixed_cost'] for node in nodes_data]
    
    try:
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', time_limit)
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        z = model.addVars(num_commodities, n, n, vtype=GRB.BINARY, name="z")
        fixed_cost_expr = grb.quicksum(fixed_costs[i] * y[i] for i in range(n))
        transport_cost_expr = 0
        for idx_k, com in enumerate(commodities):
            o = com['origin']
            d = com['destination']
            flow = com['flow']
            for i in range(n):
                for j in range(n):
                    cost_val = flow * (chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d])
                    transport_cost_expr += z[idx_k, i, j] * cost_val
        model.setObjective(fixed_cost_expr + transport_cost_expr, GRB.MINIMIZE)
        for idx_k in range(num_commodities):
            model.addConstr(grb.quicksum(z[idx_k, i, j] for i in range(n) for j in range(n)) == 1)
        for idx_k in range(num_commodities):
            for i in range(n):
                for j in range(n):
                    model.addConstr(z[idx_k, i, j] <= y[i])
                    model.addConstr(z[idx_k, i, j] <= y[j])
        if logger:
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)
            model.optimize(callback)
        else:
            model.optimize()
        if model.SolCount > 0:
            open_hubs = [i for i in range(n) if y[i].x > 0.5]
            assignments = []
            for idx_k, com in enumerate(commodities):
                found = False
                for i in range(n):
                    for j in range(n):
                        if z[idx_k, i, j].x > 0.5:
                            if i == j:
                                hub_edge = [i]
                            else:
                                hub_edge = sorted([i, j])
                            o = com['origin']
                            d = com['destination']
                            flow = com['flow']
                            cost_val = flow * (chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d])
                            assignments.append({
                                "commodity": idx_k,
                                "origin": o,
                                "destination": d,
                                "hub_edge": hub_edge,
                                "cost": cost_val
                            })
                            found = True
                            break
                    if found:
                        break
            return {
                "objective_value": model.objVal,
                "open_hubs": open_hubs,
                "assignments": assignments
            }
        else:
            return None
    except grb.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None
    except Exception as e:
        print(f"Error in MIP: {e}")
        return None

def main(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    n = instance['num_nodes']
    k = instance['num_commodities']
    time_left = time_limit - (time.time() - start_time)
    if time_left <= 0:
        solution = build_trivial_solution(instance)
        if logger:
            logger.log(solution['objective_value'])
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    if k * n * n <= 1000000:
        solution = run_MIP(instance, time_left, logger)
        if solution is not None:
            if logger:
                logger.log(solution['objective_value'])
            with open(solution_path, 'w') as f:
                json.dump(solution, f)
            return
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0:
        solution = run_heuristic(instance, time_left, logger)
        if solution is not None:
            if logger:
                logger.log(solution['objective_value'])
            with open(solution_path, 'w') as f:
                json.dump(solution, f)
            return
    solution = build_trivial_solution(instance)
    if logger:
        logger.log(solution['objective_value'])
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)