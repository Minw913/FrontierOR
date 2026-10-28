import argparse
import json
import time
from collections import defaultdict, deque
import heapq

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    network = instance['network']
    commodities_data = instance['commodities']
    
    nodes = network['nodes']
    arcs = network['arcs']
    num_arcs = network['num_arcs']
    commodity_list = commodities_data['commodity_list']
    num_commodities = commodities_data['num_commodities']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build arc data structures
    arc_by_id = {}
    for arc in arcs:
        arc_by_id[arc['arc_id']] = arc
    
    outgoing = defaultdict(list)
    for arc in arcs:
        outgoing[arc['from_node']].append((arc['to_node'], arc['arc_id']))
    
    # Current remaining capacity
    remaining_cap = {}
    for arc in arcs:
        remaining_cap[arc['arc_id']] = arc['capacity']
    
    def find_shortest_path(origin, destination, demand):
        """BFS to find shortest path that has enough capacity."""
        if origin == destination:
            return []
        
        visited = {origin}
        queue = deque([(origin, [])])
        
        while queue:
            node, path = queue.popleft()
            for to_node, a_id in outgoing[node]:
                if to_node not in visited and remaining_cap[a_id] >= demand:
                    new_path = path + [a_id]
                    if to_node == destination:
                        return new_path
                    visited.add(to_node)
                    queue.append((to_node, new_path))
        
        return None  # No path found
    
    def find_widest_path(origin, destination, demand):
        """Find path maximizing minimum residual capacity (that is >= demand)."""
        if origin == destination:
            return []
        
        # Dijkstra-like with negative bottleneck
        # Use max-heap on bottleneck capacity
        dist = {}
        prev = {}
        # (neg_bottleneck, node)
        heap = [(-float('inf'), origin)]
        dist[origin] = float('inf')
        
        while heap:
            neg_bw, node = heapq.heappop(heap)
            bw = -neg_bw
            
            if bw < dist.get(node, 0):
                continue
            
            if node == destination:
                # Reconstruct path
                path = []
                cur = destination
                while cur != origin:
                    a_id = prev[cur]
                    path.append(a_id)
                    cur = arc_by_id[a_id]['from_node']
                path.reverse()
                return path
            
            for to_node, a_id in outgoing[node]:
                cap = remaining_cap[a_id]
                if cap < demand:
                    continue
                new_bw = min(bw, cap)
                if new_bw > dist.get(to_node, 0):
                    dist[to_node] = new_bw
                    prev[to_node] = a_id
                    heapq.heappush(heap, (-new_bw, to_node))
        
        return None
    
    # Greedy heuristic: sort commodities by revenue (descending) / demand ratio
    # Try to route high-value commodities first
    sorted_commodities = sorted(commodity_list, key=lambda c: -c['revenue'])
    
    # Track solution
    commodity_solution = {}
    total_rejection_cost = sum(c['artificial_arc_cost'] for c in commodity_list)
    
    # Initially reject all
    for comm in commodity_list:
        commodity_solution[comm['commodity_id']] = {
            'rejected': True,
            'path_arcs': [],
            'path_arc_ids': []
        }
    
    if logger:
        logger.log(total_rejection_cost)
    
    # Phase 1: Greedy routing by revenue
    for comm in sorted_commodities:
        if time.time() - start_time > time_limit * 0.3:
            break
        
        k = comm['commodity_id']
        origin = comm['origin']
        destination = comm['destination']
        demand = comm['demand']
        
        if origin == destination:
            commodity_solution[k] = {
                'rejected': False,
                'path_arcs': [],
                'path_arc_ids': []
            }
            total_rejection_cost -= comm['artificial_arc_cost']
            if logger:
                logger.log(total_rejection_cost)
            continue
        
        path = find_shortest_path(origin, destination, demand)
        
        if path is not None:
            # Route this commodity
            for a_id in path:
                remaining_cap[a_id] -= demand
            
            commodity_solution[k] = {
                'rejected': False,
                'path_arcs': path,
                'path_arc_ids': path
            }
            total_rejection_cost -= comm['artificial_arc_cost']
            if logger:
                logger.log(total_rejection_cost)
    
    best_cost = total_rejection_cost
    best_solution = dict(commodity_solution)
    
    # Phase 2: Try improving by swapping - unroute low-value, reroute high-value
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.8:
        # Sort by revenue/demand ratio descending
        sorted_by_efficiency = sorted(commodity_list, key=lambda c: -c['revenue'] / max(c['demand'], 1))
        
        improved = True
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.8:
            improved = False
            iteration += 1
            
            # Try to route rejected commodities
            for comm in sorted_by_efficiency:
                if time.time() - start_time > time_limit * 0.8:
                    break
                
                k = comm['commodity_id']
                if not commodity_solution[k]['rejected']:
                    continue
                
                origin = comm['origin']
                destination = comm['destination']
                demand = comm['demand']
                
                if origin == destination:
                    continue
                
                path = find_shortest_path(origin, destination, demand)
                if path is not None:
                    for a_id in path:
                        remaining_cap[a_id] -= demand
                    commodity_solution[k] = {
                        'rejected': False,
                        'path_arcs': path,
                        'path_arc_ids': path
                    }
                    total_rejection_cost -= comm['artificial_arc_cost']
                    improved = True
                    if total_rejection_cost < best_cost:
                        best_cost = total_rejection_cost
                        best_solution = dict(commodity_solution)
                        if logger:
                            logger.log(best_cost)
            
            if not improved:
                break
    
    # Phase 3: Try MIP for small/medium instances if time remains
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed - 2
    
    if remaining_time > 10 and num_commodities <= 200 and num_arcs <= 2000:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Reset capacities
            rem_cap_mip = {}
            for arc in arcs:
                rem_cap_mip[arc['arc_id']] = arc['capacity']
            
            incoming_arcs_by_node = defaultdict(list)
            outgoing_arcs_by_node = defaultdict(list)
            for arc in arcs:
                incoming_arcs_by_node[arc['to_node']].append(arc['arc_id'])
                outgoing_arcs_by_node[arc['from_node']].append(arc['arc_id'])
            
            def get_relevant_arcs(origin, destination):
                forward_reachable = set()
                stack = [origin]
                while stack:
                    node = stack.pop()
                    if node in forward_reachable:
                        continue
                    forward_reachable.add(node)
                    for to_node, arc_id in outgoing[node]:
                        if to_node not in forward_reachable:
                            stack.append(to_node)
                
                incoming_map = defaultdict(list)
                for arc in arcs:
                    incoming_map[arc['to_node']].append(arc['from_node'])
                
                backward_reachable = set()
                stack = [destination]
                while stack:
                    node = stack.pop()
                    if node in backward_reachable:
                        continue
                    backward_reachable.add(node)
                    for from_node in incoming_map[node]:
                        if from_node not in backward_reachable:
                            stack.append(from_node)
                
                relevant_nodes = forward_reachable & backward_reachable
                relevant_arcs_list = []
                for arc in arcs:
                    if arc['from_node'] in relevant_nodes and arc['to_node'] in relevant_nodes:
                        relevant_arcs_list.append(arc['arc_id'])
                
                return relevant_arcs_list, relevant_nodes
            
            model = gp.Model("unsplittable_mcf")
            model.setParam('TimeLimit', max(1, remaining_time))
            model.setParam('Threads', 1)
            model.setParam('MIPFocus', 1)
            model.setParam('OutputFlag', 0)
            
            x = {}
            r = {}
            commodity_relevant_arcs = {}
            commodity_relevant_nodes = {}
            
            for comm in commodity_list:
                k = comm['commodity_id']
                origin = comm['origin']
                destination = comm['destination']
                
                rel_arcs, rel_nodes = get_relevant_arcs(origin, destination)
                commodity_relevant_arcs[k] = set(rel_arcs)
                commodity_relevant_nodes[k] = rel_nodes
                
                r[k] = model.addVar(vtype=GRB.BINARY, name=f"r_{k}", obj=comm['artificial_arc_cost'])
                
                for a_id in rel_arcs:
                    x[k, a_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{a_id}")
            
            model.update()
            
            # Set warm start from greedy
            for comm in commodity_list:
                k = comm['commodity_id']
                sol = best_solution[k]
                r[k].Start = 1.0 if sol['rejected'] else 0.0
                
                used_arcs_set = set(sol.get('path_arc_ids', []))
                for a_id in commodity_relevant_arcs[k]:
                    if (k, a_id) in x:
                        x[k, a_id].Start = 1.0 if a_id in used_arcs_set else 0.0
            
            for comm in commodity_list:
                k = comm['commodity_id']
                origin = comm['origin']
                destination = comm['destination']
                rel_arcs = commodity_relevant_arcs[k]
                rel_nodes = commodity_relevant_nodes[k]
                
                for node in rel_nodes:
                    out_flow = gp.LinExpr()
                    in_flow = gp.LinExpr()
                    
                    for a_id in outgoing_arcs_by_node[node]:
                        if a_id in rel_arcs:
                            out_flow += x[k, a_id]
                    
                    for a_id in incoming_arcs_by_node[node]:
                        if a_id in rel_arcs:
                            in_flow += x[k, a_id]
                    
                    if node == origin and node == destination:
                        model.addConstr(out_flow - in_flow == 0)
                    elif node == origin:
                        model.addConstr(out_flow - in_flow == 1 - r[k])
                    elif node == destination:
                        model.addConstr(in_flow - out_flow == 1 - r[k])
                    else:
                        model.addConstr(in_flow - out_flow == 0)
                
                if origin == destination:
                    model.addConstr(r[k] == 0)
            
            # Capacity constraints
            arc_commodity_vars = defaultdict(list)
            for comm in commodity_list:
                k = comm['commodity_id']
                demand = comm['demand']
                for a_id in commodity_relevant_arcs[k]:
                    arc_commodity_vars[a_id].append((k, demand))
            
            for arc in arcs:
                a_id = arc['arc_id']
                if a_id in arc_commodity_vars:
                    flow_expr = gp.LinExpr()
                    for k, demand in arc_commodity_vars[a_id]:
                        flow_expr += demand * x[k, a_id]
                    model.addConstr(flow_expr <= arc['capacity'])
            
            model.setAttr('ModelSense', GRB.MINIMIZE)
            
            mip_best = [best_cost]
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < mip_best[0] - 1e-6:
                        mip_best[0] = obj_val
                        if logger:
                            logger.log(obj_val)
            
            model.optimize(callback)
            
            if model.SolCount > 0 and model.ObjVal < best_cost - 1e-6:
                best_cost = model.ObjVal
                
                for comm in commodity_list:
                    k = comm['commodity_id']
                    rejected = r[k].X > 0.5
                    path_arc_ids = []
                    
                    if not rejected:
                        for a_id in commodity_relevant_arcs[k]:
                            if x[k, a_id].X > 0.5:
                                path_arc_ids.append(a_id)
                    
                    best_solution[k] = {
                        'rejected': rejected,
                        'path_arcs': path_arc_ids,
                        'path_arc_ids': path_arc_ids
                    }
        except Exception:
            pass
    
    # Build output
    solution_commodities = []
    for comm in commodity_list:
        k = comm['commodity_id']
        sol = best_solution[k]
        
        path_arcs_out = []
        if not sol['rejected']:
            arc_ids = sol.get('path_arc_ids', sol.get('path_arcs', []))
            if arc_ids:
                # Order the arcs into a path
                arc_from_node = {}
                for a_id in arc_ids:
                    a = arc_by_id[a_id]
                    arc_from_node[a['from_node']] = a_id
                
                current = comm['origin']
                visited = set()
                while current != comm['destination'] and current not in visited:
                    visited.add(current)
                    if current not in arc_from_node:
                        break
                    a_id = arc_from_node[current]
                    a = arc_by_id[a_id]
                    path_arcs_out.append({
                        "from": a['from_node'],
                        "to": a['to_node'],
                        "arc_id": a_id
                    })
                    current = a['to_node']
        
        solution_commodities.append({
            "commodity_id": k,
            "rejected": sol['rejected'],
            "path_arcs": path_arcs_out
        })
    
    solution = {
        "objective_value": best_cost,
        "commodities": solution_commodities
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == '__main__':
    main()