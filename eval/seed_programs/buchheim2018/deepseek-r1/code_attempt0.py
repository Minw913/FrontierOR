import argparse
import json
import time
import heapq
from solution_logger import SolutionLogger

def shortest_path_by_linear_costs(nodes, arcs, linear_costs, source, target):
    all_node_ids = [node['id'] for node in nodes]
    graph = {node_id: [] for node_id in all_node_ids}
    for idx, arc in enumerate(arcs):
        u = arc['from_node']
        v = arc['to_node']
        graph[u].append((v, idx, linear_costs[idx]))
    
    if min(linear_costs) >= 0:
        dist = {node_id: float('inf') for node_id in all_node_ids}
        prev = {node_id: None for node_id in all_node_ids}
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == target:
                break
            if d != dist[u]:
                continue
            for v, arc_idx, cost in graph[u]:
                new_d = d + cost
                if new_d < dist[v]:
                    dist[v] = new_d
                    prev[v] = (u, arc_idx)
                    heapq.heappush(heap, (new_d, v))
        if dist[target] == float('inf'):
            return None
    else:
        dist = {node_id: float('inf') for node_id in all_node_ids}
        prev = {node_id: None for node_id in all_node_ids}
        dist[source] = 0
        for _ in range(len(all_node_ids) - 1):
            updated = False
            for arc_idx, arc in enumerate(arcs):
                u = arc['from_node']
                v = arc['to_node']
                cost = linear_costs[arc_idx]
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, arc_idx)
                    updated = True
            if not updated:
                break
        if dist[target] == float('inf'):
            return None
    
    path_arcs = []
    current = target
    while current != source:
        if prev[current] is None:
            return None
        u, arc_idx = prev[current]
        path_arcs.append(arc_idx)
        current = u
    path_arcs.reverse()
    return path_arcs

def compute_total_cost(arcs_indices, linear_costs, quadratic_costs):
    linear_part = sum(linear_costs[i] for i in arcs_indices)
    quadratic_part = 0
    for i in arcs_indices:
        for j in arcs_indices:
            quadratic_part += quadratic_costs[i][j]
    return linear_part + quadratic_part

def extract_arcs_in_order(arcs_list, selected_arc_indices, source, target):
    next_arc_dict = {}
    for arc_idx in selected_arc_indices:
        arc = arcs_list[arc_idx]
        u = arc['from_node']
        if u in next_arc_dict:
            return None
        next_arc_dict[u] = (arc['to_node'], arc_idx)
    
    ordered_arcs = []
    current = source
    while current != target:
        if current not in next_arc_dict:
            return None
        next_node, arc_idx = next_arc_dict[current]
        ordered_arcs.append(arc_idx)
        current = next_node
    return ordered_arcs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_objective = float('inf')
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    nodes = instance['nodes']
    arcs = instance['arcs']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']
    source_node = instance['source_node']
    target_node = instance['target_node']
    num_arcs = len(arcs)
    
    outgoing = {}
    incoming = {}
    for node in nodes:
        nid = node['id']
        outgoing[nid] = []
        incoming[nid] = []
    for idx, arc in enumerate(arcs):
        u = arc['from_node']
        v = arc['to_node']
        outgoing[u].append(idx)
        incoming[v].append(idx)
    
    solution_arcs_indices = None
    objective_value = float('inf')
    
    if num_arcs <= 500:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            time_used = time.time() - start_time
            time_remaining = max(1, args.time_limit - time_used)
            model.setParam('TimeLimit', time_remaining)
            
            x = model.addVars(num_arcs, vtype=GRB.BINARY, name="x")
            
            model.addConstr(gp.quicksum(x[i] for i in outgoing[source_node]) == 1)
            model.addConstr(gp.quicksum(x[i] for i in incoming[source_node]) == 0)
            model.addConstr(gp.quicksum(x[i] for i in incoming[target_node]) == 1)
            model.addConstr(gp.quicksum(x[i] for i in outgoing[target_node]) == 0)
            
            for node in nodes:
                nid = node['id']
                if nid == source_node or nid == target_node:
                    continue
                model.addConstr(gp.quicksum(x[i] for i in incoming[nid]) == gp.quicksum(x[j] for j in outgoing[nid]))
            
            linear_expr = gp.quicksum(linear_costs[i] * x[i] for i in range(num_arcs))
            quadratic_expr = gp.quicksum(quadratic_costs[i][j] * x[i] * x[j] for i in range(num_arcs) for j in range(num_arcs))
            model.setObjective(linear_expr + quadratic_expr)
            
            best_so_far = float('inf')
            def callback(model, where):
                nonlocal best_so_far, logger
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < best_so_far:
                        best_so_far = obj
                        if logger:
                            logger.log(obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                selected_arc_indices = [i for i in range(num_arcs) if x[i].X > 0.5]
                solution_arcs_indices = extract_arcs_in_order(arcs, selected_arc_indices, source_node, target_node)
                if solution_arcs_indices is None:
                    solution_arcs_indices = selected_arc_indices
                objective_value = model.ObjVal
                if logger and objective_value < best_objective:
                    logger.log(objective_value)
                    best_objective = objective_value
            else:
                solution_arcs_indices = shortest_path_by_linear_costs(nodes, arcs, linear_costs, source_node, target_node)
                if solution_arcs_indices is not None:
                    objective_value = compute_total_cost(solution_arcs_indices, linear_costs, quadratic_costs)
                    if logger and objective_value < best_objective:
                        logger.log(objective_value)
                        best_objective = objective_value
        except Exception as e:
            solution_arcs_indices = shortest_path_by_linear_costs(nodes, arcs, linear_costs, source_node, target_node)
            if solution_arcs_indices is not None:
                objective_value = compute_total_cost(solution_arcs_indices, linear_costs, quadratic_costs)
                if logger and objective_value < best_objective:
                    logger.log(objective_value)
                    best_objective = objective_value
    else:
        solution_arcs_indices = shortest_path_by_linear_costs(nodes, arcs, linear_costs, source_node, target_node)
        if solution_arcs_indices is not None:
            objective_value = compute_total_cost(solution_arcs_indices, linear_costs, quadratic_costs)
            if logger and objective_value < best_objective:
                logger.log(objective_value)
                best_objective = objective_value
    
    if solution_arcs_indices is None:
        solution_arcs_indices = []
        objective_value = 0
    
    solution_arcs = []
    for arc_idx in solution_arcs_indices:
        arc = arcs[arc_idx]
        solution_arcs.append({
            "id": arc['id'],
            "from_node": arc['from_node'],
            "to_node": arc['to_node']
        })
    
    solution = {
        "objective_value": objective_value,
        "solution_arcs": solution_arcs
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()