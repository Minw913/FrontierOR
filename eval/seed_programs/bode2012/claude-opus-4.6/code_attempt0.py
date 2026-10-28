import argparse
import json
import heapq
import time
import random
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['graph']['num_nodes']
    num_edges = instance['graph']['num_edges']
    depot = instance['depot']
    num_vehicles = instance['fleet']['num_vehicles']
    vehicle_capacity = instance['fleet']['vehicle_capacity']
    edges = instance['edges']

    # Build adjacency for shortest path computation
    adj = defaultdict(list)
    edge_map = {}
    required_edges = []

    for e in edges:
        eid = e['edge_id']
        u, v = e['endpoints']
        cost = e['cost']
        edge_map[eid] = e
        adj[u].append((v, cost, eid))
        adj[v].append((u, cost, eid))
        if e['is_required']:
            required_edges.append(e)

    # Compute shortest paths from all relevant nodes using Dijkstra
    # Relevant nodes: depot + all endpoints of required edges
    relevant_nodes = set([depot])
    for e in required_edges:
        relevant_nodes.add(e['endpoints'][0])
        relevant_nodes.add(e['endpoints'][1])
    relevant_nodes = list(relevant_nodes)

    def dijkstra(source):
        dist = {}
        dist[source] = 0
        pq = [(0, source)]
        prev = {}
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for v, w, eid in adj[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, eid)
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    # Precompute shortest path distances and predecessors from relevant nodes
    sp_dist = {}
    sp_prev = {}
    for node in relevant_nodes:
        dist, prev = dijkstra(node)
        sp_dist[node] = dist
        sp_prev[node] = prev

    def shortest_path_cost(u, v):
        if u == v:
            return 0
        return sp_dist[u].get(v, float('inf'))

    def reconstruct_path_edges(source, target):
        """Return list of edge_ids on shortest path from source to target."""
        if source == target:
            return []
        edges_on_path = []
        cur = target
        prev = sp_prev[source]
        while cur != source:
            p, eid = prev[cur]
            edges_on_path.append(eid)
            cur = p
        edges_on_path.reverse()
        return edges_on_path

    # For each required edge, compute the cost of a route that just services that edge:
    # depot -> u -> (service u-v) -> v -> depot (or depot -> v -> service -> u -> depot)
    # The "service" traversal uses service_cost. Deadheading is the shortest path cost.

    # We model this as a Capacitated Arc Routing Problem (CARP).
    # We'll use a route-first, cluster-second heuristic, followed by local search.

    # For CARP, we can think of required edges as "tasks". Each task (u,v) can be served
    # in direction u->v or v->u. The cost to go from end of one task to start of another
    # is the shortest path between those nodes.

    num_req = len(required_edges)

    if num_req == 0:
        # No required edges, all vehicles stay at depot
        solution = {
            "objective_value": 0.0,
            "routes": [{"vehicle": i, "serviced_edges": [], "deadheaded_edges": []} for i in range(num_vehicles)]
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log(0.0)
        return

    # Represent each required edge as a task with two possible orientations
    # task i: edge required_edges[i], endpoints (u_i, v_i)
    # If served u->v: start=u, end=v
    # If served v->u: start=v, end=u

    # Cost of traversal between tasks considers:
    # - Going from current position to start of next task (deadheading)
    # - Servicing the next task (service_cost)
    # We want to minimize total deadheading cost. Service costs are fixed.

    # Greedy nearest-neighbor construction for initial solution
    task_demands = [e['demand'] for e in required_edges]
    task_service_costs = [e['service_cost'] for e in required_edges]
    task_costs = [e['cost'] for e in required_edges]  # traversal cost of the edge itself
    task_endpoints = [(e['endpoints'][0], e['endpoints'][1]) for e in required_edges]
    task_eids = [e['edge_id'] for e in required_edges]

    total_service_cost = sum(task_service_costs)

    # For going from node `pos` to task `t`, the cost is:
    # min(sp(pos, u_t) + service_cost_t, sp(pos, v_t) + service_cost_t)
    # But service_cost is constant, so we minimize deadheading:
    # min(sp(pos, u_t), sp(pos, v_t)) and then the service cost is added separately.
    # After serving task t in direction u->v, position is v; in direction v->u, position is u.

    def task_insertion_cost(pos, t_idx):
        """Cost to go from pos, serve task t_idx, return. Returns (deadhead_to, direction, end_pos, service_cost)."""
        u, v = task_endpoints[t_idx]
        sc = task_service_costs[t_idx]
        d_u = shortest_path_cost(pos, u)
        d_v = shortest_path_cost(pos, v)
        if d_u <= d_v:
            return d_u, 0, v, sc  # go to u, serve u->v, end at v
        else:
            return d_v, 1, u, sc  # go to v, serve v->u, end at u

    def route_cost(route):
        """Compute total deadheading cost for a route (list of task indices)."""
        if not route:
            return 0
        cost = 0
        pos = depot
        for t_idx in route:
            u, v = task_endpoints[t_idx]
            d_u = shortest_path_cost(pos, u)
            d_v = shortest_path_cost(pos, v)
            if d_u <= d_v:
                cost += d_u
                pos = v
            else:
                cost += d_v
                pos = u
        # Return to depot
        cost += shortest_path_cost(pos, depot)
        return cost

    def route_cost_with_direction(route):
        """Compute total deadheading cost and directions for a route."""
        if not route:
            return 0, []
        cost = 0
        pos = depot
        directions = []
        for t_idx in route:
            u, v = task_endpoints[t_idx]
            d_u = shortest_path_cost(pos, u)
            d_v = shortest_path_cost(pos, v)
            if d_u <= d_v:
                cost += d_u
                pos = v
                directions.append(0)
            else:
                cost += d_v
                pos = u
                directions.append(1)
        cost += shortest_path_cost(pos, depot)
        return cost, directions

    def route_demand(route):
        return sum(task_demands[t] for t in route)

    # Path-scanning heuristic for initial solution
    def path_scanning():
        unassigned = set(range(num_req))
        routes = []
        
        while unassigned:
            if len(routes) >= num_vehicles:
                # Must fit remaining into existing routes or last route
                # Force all remaining into last route (may violate capacity, we'll fix later)
                routes[-1].extend(list(unassigned))
                unassigned.clear()
                break
            
            route = []
            capacity_left = vehicle_capacity
            pos = depot
            
            while unassigned:
                best_t = None
                best_cost = float('inf')
                best_end = None
                
                for t in unassigned:
                    if task_demands[t] > capacity_left:
                        continue
                    u, v = task_endpoints[t]
                    d_u = shortest_path_cost(pos, u)
                    d_v = shortest_path_cost(pos, v)
                    if d_u <= d_v:
                        c = d_u
                        end = v
                    else:
                        c = d_v
                        end = u
                    if c < best_cost or (c == best_cost and best_t is not None and 
                                          shortest_path_cost(end, depot) > shortest_path_cost(best_end, depot)):
                        best_cost = c
                        best_t = t
                        best_end = end
                
                if best_t is None:
                    break
                
                route.append(best_t)
                unassigned.remove(best_t)
                capacity_left -= task_demands[best_t]
                pos = best_end
            
            routes.append(route)
        
        return routes

    # Augment-merge heuristic
    def augment_merge():
        # Start with one route per required edge
        routes = [[i] for i in range(num_req)]
        demands = [task_demands[i] for i in range(num_req)]
        costs = [route_cost([i]) for i in range(num_req)]
        
        # Greedily merge routes
        merged = [False] * len(routes)
        
        # Compute savings for merging routes
        # savings(i,j) = cost(i) + cost(j) - cost(merged)
        # We try to merge until we have num_vehicles routes or no more beneficial merges
        
        while True:
            active = [i for i in range(len(routes)) if not merged[i]]
            if len(active) <= num_vehicles:
                break
            
            best_saving = -float('inf')
            best_pair = None
            best_merged_route = None
            
            # Only check a subset if too many
            if len(active) > 200:
                # Just use path scanning
                break
            
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    i, j = active[ii], active[jj]
                    if demands[i] + demands[j] > vehicle_capacity:
                        continue
                    # Try merging route i + route j
                    mr1 = routes[i] + routes[j]
                    mr2 = routes[i] + list(reversed(routes[j]))
                    mr3 = list(reversed(routes[i])) + routes[j]
                    mr4 = routes[j] + routes[i]
                    
                    best_mr = None
                    best_mr_cost = float('inf')
                    for mr in [mr1, mr2, mr3, mr4]:
                        c = route_cost(mr)
                        if c < best_mr_cost:
                            best_mr_cost = c
                            best_mr = mr
                    
                    saving = costs[i] + costs[j] - best_mr_cost
                    if saving > best_saving:
                        best_saving = saving
                        best_pair = (i, j)
                        best_merged_route = best_mr
                        best_merged_cost = best_mr_cost
            
            if best_pair is None or best_saving <= 0:
                break
            
            i, j = best_pair
            routes[i] = best_merged_route
            demands[i] = demands[i] + demands[j]
            costs[i] = best_merged_cost
            merged[j] = True
            
            if time.time() - start_time > args.time_limit * 0.3:
                break
        
        return [routes[i] for i in range(len(routes)) if not merged[i]]

    # Build initial solution
    if num_req <= 150:
        routes = augment_merge()
    else:
        routes = path_scanning()

    # If we have more routes than vehicles, we need to merge/reassign
    while len(routes) > num_vehicles:
        # Find the smallest route and merge it with the route that has the most remaining capacity
        routes.sort(key=lambda r: route_demand(r))
        smallest = routes.pop(0)
        
        # Try to insert each task from smallest into best route
        for t in smallest:
            best_route_idx = None
            best_increase = float('inf')
            
            for ri in range(len(routes)):
                if route_demand(routes[ri]) + task_demands[t] <= vehicle_capacity:
                    old_cost = route_cost(routes[ri])
                    new_route = routes[ri] + [t]
                    new_cost = route_cost(new_route)
                    increase = new_cost - old_cost
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = ri
            
            if best_route_idx is not None:
                routes[best_route_idx].append(t)
            else:
                # Force into route with most remaining capacity
                caps = [(vehicle_capacity - route_demand(routes[ri]), ri) for ri in range(len(routes))]
                caps.sort(reverse=True)
                routes[caps[0][1]].append(t)

    # Ensure we have exactly num_vehicles routes (add empty ones if needed)
    while len(routes) < num_vehicles:
        routes.append([])

    def total_cost(routes):
        tc = total_service_cost
        for r in routes:
            tc += route_cost(r)
        return tc

    best_obj = total_cost(routes)
    best_routes = [r[:] for r in routes]
    if logger:
        logger.log(best_obj)

    # Local search improvements
    def try_or_opt(routes):
        """Try moving a segment of 1-3 tasks from one route to another."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > args.time_limit * 0.85:
                return routes
            for ri in range(len(routes)):
                if not routes[ri]:
                    continue
                for seg_len in [1, 2, 3]:
                    if seg_len > len(routes[ri]):
                        continue
                    for pos in range(len(routes[ri]) - seg_len + 1):
                        segment = routes[ri][pos:pos + seg_len]
                        seg_demand = sum(task_demands[t] for t in segment)
                        remaining = routes[ri][:pos] + routes[ri][pos + seg_len:]
                        old_ri_cost = route_cost(routes[ri])
                        new_ri_cost = route_cost(remaining)
                        saving_remove = old_ri_cost - new_ri_cost
                        
                        for rj in range(len(routes)):
                            if ri == rj:
                                continue
                            if route_demand(routes[rj]) + seg_demand > vehicle_capacity:
                                continue
                            
                            old_rj_cost = route_cost(routes[rj])
                            best_insert_cost = float('inf')
                            best_insert_pos = None
                            best_seg = None
                            
                            for ip in range(len(routes[rj]) + 1):
                                for seg_order in [segment, list(reversed(segment))]:
                                    new_rj = routes[rj][:ip] + seg_order + routes[rj][ip:]
                                    c = route_cost(new_rj)
                                    if c < best_insert_cost:
                                        best_insert_cost = c
                                        best_insert_pos = ip
                                        best_seg = seg_order
                            
                            cost_increase = best_insert_cost - old_rj_cost
                            
                            if saving_remove - cost_increase > 0.001:
                                routes[ri] = remaining
                                routes[rj] = routes[rj][:best_insert_pos] + best_seg + routes[rj][best_insert_pos:]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def two_opt_within_route(route):
        """2-opt improvement within a single route."""
        if len(route) <= 2:
            return route
        improved = True
        while improved:
            improved = False
            best_cost = route_cost(route)
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                    c = route_cost(new_route)
                    if c < best_cost - 0.001:
                        route = new_route
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break
        return route

    def swap_between_routes(routes):
        """Try swapping single tasks between routes."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > args.time_limit * 0.85:
                return routes
            for ri in range(len(routes)):
                for rj in range(ri + 1, len(routes)):
                    if not routes[ri] or not routes[rj]:
                        continue
                    for pi in range(len(routes[ri])):
                        for pj in range(len(routes[rj])):
                            ti = routes[ri][pi]
                            tj = routes[rj][pj]
                            
                            new_demand_i = route_demand(routes[ri]) - task_demands[ti] + task_demands[tj]
                            new_demand_j = route_demand(routes[rj]) - task_demands[tj] + task_demands[ti]
                            
                            if new_demand_i > vehicle_capacity or new_demand_j > vehicle_capacity:
                                continue
                            
                            old_cost = route_cost(routes[ri]) + route_cost(routes[rj])
                            
                            new_ri = routes[ri][:pi] + [tj] + routes[ri][pi+1:]
                            new_rj = routes[rj][:pj] + [ti] + routes[rj][pj+1:]
                            
                            new_cost = route_cost(new_ri) + route_cost(new_rj)
                            
                            if new_cost < old_cost - 0.001:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Apply 2-opt within each route
    for i in range(len(routes)):
        if time.time() - start_time > args.time_limit * 0.5:
            break
        routes[i] = two_opt_within_route(routes[i])

    cur_obj = total_cost(routes)
    if cur_obj < best_obj:
        best_obj = cur_obj
        best_routes = [r[:] for r in routes]
        if logger:
            logger.log(best_obj)

    # Apply inter-route improvements
    if time.time() - start_time < args.time_limit * 0.7:
        routes = try_or_opt(routes)
        cur_obj = total_cost(routes)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_routes = [r[:] for r in routes]
            if logger:
                logger.log(best_obj)

    if time.time() - start_time < args.time_limit * 0.8:
        routes = swap_between_routes(routes)
        cur_obj = total_cost(routes)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_routes = [r[:] for r in routes]
            if logger:
                logger.log(best_obj)

    # Apply 2-opt again after inter-route moves
    for i in range(len(routes)):
        if time.time() - start_time > args.time_limit * 0.9:
            break
        routes[i] = two_opt_within_route(routes[i])

    cur_obj = total_cost(routes)
    if cur_obj < best_obj:
        best_obj = cur_obj
        best_routes = [r[:] for r in routes]
        if logger:
            logger.log(best_obj)

    # Perturbation + local search (simple ILS)
    routes = [r[:] for r in best_routes]
    iteration = 0
    while time.time() - start_time < args.time_limit * 0.95:
        iteration += 1
        # Perturb: randomly move a task from one route to another
        perturbed = [r[:] for r in routes]
        non_empty = [i for i in range(len(perturbed)) if len(perturbed[i]) > 0]
        if not non_empty:
            break
        
        ri = random.choice(non_empty)
        if not perturbed[ri]:
            continue
        pi = random.randint(0, len(perturbed[ri]) - 1)
        task = perturbed[ri].pop(pi)
        
        # Find a feasible route to insert
        candidates = list(range(len(perturbed)))
        random.shuffle(candidates)
        inserted = False
        for rj in candidates:
            if rj == ri and not perturbed[rj]:  # don't put back in same empty route always
                continue
            if route_demand(perturbed[rj]) + task_demands[task] <= vehicle_capacity:
                # Insert at random position
                ip = random.randint(0, len(perturbed[rj]))
                perturbed[rj].insert(ip, task)
                inserted = True
                break
        
        if not inserted:
            perturbed[ri].insert(pi, task)
            continue
        
        # Local search on perturbed
        for i in range(len(perturbed)):
            if time.time() - start_time > args.time_limit * 0.93:
                break
            perturbed[i] = two_opt_within_route(perturbed[i])
        
        p_obj = total_cost(perturbed)
        if p_obj < best_obj:
            best_obj = p_obj
            best_routes = [r[:] for r in perturbed]
            routes = [r[:] for r in perturbed]
            if logger:
                logger.log(best_obj)
        elif p_obj < total_cost(routes) + 50:  # Accept slightly worse
            routes = perturbed

    routes = best_routes

    # Build solution output
    solution_routes = []
    for vi in range(num_vehicles):
        route = routes[vi] if vi < len(routes) else []
        if not route:
            solution_routes.append({
                "vehicle": vi,
                "serviced_edges": [],
                "deadheaded_edges": []
            })
            continue
        
        serviced_edges = [task_eids[t] for t in route]
        
        # Compute the actual path and deadheaded edges
        deadhead_counts = defaultdict(int)
        pos = depot
        _, directions = route_cost_with_direction(route)
        
        for idx, t in enumerate(route):
            u, v = task_endpoints[t]
            if directions[idx] == 0:
                start_node = u
            else:
                start_node = v
            
            # Deadhead from pos to start_node
            if pos != start_node:
                path_edges = reconstruct_path_edges(pos, start_node)
                for eid in path_edges:
                    deadhead_counts[eid] += 1
            
            # Service the edge (not counted as deadheading)
            if directions[idx] == 0:
                pos = v
            else:
                pos = u
        
        # Return to depot
        if pos != depot:
            path_edges = reconstruct_path_edges(pos, depot)
            for eid in path_edges:
                deadhead_counts[eid] += 1
        
        deadheaded_edges = [{"edge_id": eid, "times": cnt} for eid, cnt in deadhead_counts.items() if cnt > 0]
        
        solution_routes.append({
            "vehicle": vi,
            "serviced_edges": serviced_edges,
            "deadheaded_edges": deadheaded_edges
        })

    # Recompute objective value precisely
    obj = total_service_cost
    for sr in solution_routes:
        for de in sr['deadheaded_edges']:
            obj += edge_map[de['edge_id']]['cost'] * de['times']

    solution = {
        "objective_value": obj,
        "routes": solution_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()