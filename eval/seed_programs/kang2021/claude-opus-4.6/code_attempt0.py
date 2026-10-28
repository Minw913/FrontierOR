import json
import argparse
import time
import math
import random
import copy
from solution_logger import SolutionLogger

def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_customers = instance["num_customers"]
    num_nodes = instance["num_nodes"]
    params = instance["parameters"]
    num_drones = params["num_drones"]
    depot = instance["depot_index"]
    customers = instance["customer_indices"]
    truck_tt = instance["truck_travel_times"]
    drone_bc = instance["drone_battery_consumption"]
    drone_dt = instance["drone_delivery_time"]
    drone_caps = params["drone_battery_capacities"]
    
    virtual_sink = num_nodes  # virtual sink node
    
    # Precompute drone delivery costs
    # For each customer, find best drone delivery options from each possible launch node
    
    def evaluate_solution(truck_route_customers, drone_assignments_list):
        """
        truck_route_customers: list of customer indices visited by truck (ordered)
        drone_assignments_list: list of (drone_idx, dispatch_node, customer)
        
        Returns (objective, waiting_times_dict, is_feasible)
        """
        # Build full route: depot -> customers -> depot
        route = [depot] + truck_route_customers + [depot]  # last depot is virtual sink conceptually
        
        # Check all customers are served
        truck_served = set(truck_route_customers)
        drone_served = set(a[2] for a in drone_assignments_list)
        
        if truck_served & drone_served:
            return float('inf'), {}, False
        if truck_served | drone_served != set(customers):
            return float('inf'), {}, False
        
        # Check dispatch nodes are on route
        route_nodes = set(route)
        for d, launch, cust in drone_assignments_list:
            if launch not in route_nodes:
                return float('inf'), {}, False
        
        # Check battery constraints
        drone_battery_used = [0.0] * num_drones
        for d, launch, cust in drone_assignments_list:
            drone_battery_used[d] += drone_bc[d][launch][cust]
        for d in range(num_drones):
            if drone_battery_used[d] > drone_caps[d]:
                return float('inf'), {}, False
        
        # Compute waiting times at each node
        # Group drone assignments by dispatch node
        from collections import defaultdict
        node_drone_assignments = defaultdict(list)
        for d, launch, cust in drone_assignments_list:
            node_drone_assignments[launch].append((d, cust))
        
        waiting_times = {}
        for node in route:
            if node in node_drone_assignments:
                # For each drone, sum up delivery times from this node
                drone_times = defaultdict(float)
                for d, cust in node_drone_assignments[node]:
                    drone_times[d] += drone_dt[d][node][cust]
                waiting_times[node] = max(drone_times.values()) if drone_times else 0
            else:
                waiting_times[node] = 0
        
        # Compute total time
        total = 0
        for i in range(len(route) - 1):
            total += truck_tt[route[i]][route[i+1]]
        
        # Add waiting times (at each unique visit)
        # Note: depot appears at start and end. We need waiting at each stop.
        # Actually the route visits nodes, and at each node the truck waits
        for i in range(len(route)):
            node = route[i]
            if node in waiting_times:
                total += waiting_times[node]
        
        # But wait - if depot appears twice (start and end), waiting times might be counted separately
        # Actually, we should compute waiting per route position, not per node
        # Let me reconsider: the waiting time at each visited node
        # If a node appears multiple times... depot appears at start and end
        # Let me recompute properly
        
        # Actually, let me reconsider. The drone assignments specify dispatch_node.
        # If depot is visited at start, drones can be dispatched from depot.
        # The truck also returns to depot at the end.
        # Drones dispatched from depot would be at the start (or end?).
        # For simplicity, let's assign depot dispatches to the first occurrence.
        
        # Let me redo: compute waiting at each position in route
        total = 0
        for i in range(len(route) - 1):
            total += truck_tt[route[i]][route[i+1]]
        
        # Waiting: for each position in route, compute waiting
        # We need to be more careful about which position the drones are dispatched from
        # For now, sum waiting at each unique node (depot counted once for dispatches)
        total_waiting = 0
        for node, wt in waiting_times.items():
            total_waiting += wt
        total += total_waiting
        
        return total, waiting_times, True
    
    def evaluate_solution_v2(truck_route_customers, drone_assignments_list):
        """
        More careful evaluation.
        truck_route_customers: ordered list of customer indices visited by truck
        drone_assignments_list: list of (drone_idx, dispatch_node, customer)
        """
        route = [depot] + truck_route_customers  # + return to depot
        
        truck_served = set(truck_route_customers)
        drone_served_set = set()
        for a in drone_assignments_list:
            drone_served_set.add(a[2])
        
        if truck_served & drone_served_set:
            return float('inf'), {}, False
        if truck_served | drone_served_set != set(customers):
            return float('inf'), {}, False
        
        route_nodes_set = set(route)
        for d, launch, cust in drone_assignments_list:
            if launch not in route_nodes_set:
                return float('inf'), {}, False
        
        drone_battery_used = [0.0] * num_drones
        for d, launch, cust in drone_assignments_list:
            drone_battery_used[d] += drone_bc[d][launch][cust]
        for d in range(num_drones):
            if drone_battery_used[d] > drone_caps[d]:
                return float('inf'), {}, False
        
        from collections import defaultdict
        node_drone_assignments = defaultdict(list)
        for d, launch, cust in drone_assignments_list:
            node_drone_assignments[launch].append((d, cust))
        
        # Compute travel time
        total_travel = 0
        for i in range(len(route) - 1):
            total_travel += truck_tt[route[i]][route[i+1]]
        # Return to depot
        total_travel += truck_tt[route[-1]][depot]
        
        # Compute waiting at each node in route
        total_waiting = 0
        waiting_times = {}
        for node in route:
            if node in node_drone_assignments:
                drone_times = defaultdict(float)
                for d, cust in node_drone_assignments[node]:
                    drone_times[d] += drone_dt[d][node][cust]
                wt = max(drone_times.values())
                waiting_times[node] = wt
                total_waiting += wt
            # else no waiting needed
        
        return total_travel + total_waiting, waiting_times, True
    
    # Better approach: represent solution as (truck_route, drone_assigns)
    # where truck_route is ordered list of customers visited by truck
    # and drone_assigns is list of (drone, launch_node, customer)
    
    def compute_objective(truck_route_cust, drone_assigns):
        """Returns (obj, waiting_dict, feasible)"""
        route = [depot] + list(truck_route_cust)
        
        # Travel time
        total_travel = 0
        for i in range(len(route) - 1):
            total_travel += truck_tt[route[i]][route[i+1]]
        total_travel += truck_tt[route[-1]][depot]
        
        # Waiting times
        from collections import defaultdict
        node_drone = defaultdict(lambda: defaultdict(float))
        for d, launch, cust in drone_assigns:
            node_drone[launch][d] += drone_dt[d][launch][cust]
        
        waiting = {}
        total_wait = 0
        for node, drone_times in node_drone.items():
            wt = max(drone_times.values())
            waiting[node] = wt
            total_wait += wt
        
        return total_travel + total_wait, waiting, True
    
    def check_feasibility(truck_route_cust, drone_assigns):
        truck_set = set(truck_route_cust)
        drone_set = set(a[2] for a in drone_assigns)
        if truck_set & drone_set:
            return False
        if truck_set | drone_set != set(customers):
            return False
        route_nodes = set([depot] + list(truck_route_cust))
        for d, launch, cust in drone_assigns:
            if launch not in route_nodes:
                return False
        drone_bat = [0] * num_drones
        for d, launch, cust in drone_assigns:
            drone_bat[d] += drone_bc[d][launch][cust]
        for d in range(num_drones):
            if drone_bat[d] > drone_caps[d]:
                return False
        return True
    
    # Greedy construction: start with TSP-like truck route, then try to offload customers to drones
    
    # Simple nearest-neighbor TSP
    def nn_tsp(customer_list):
        if not customer_list:
            return []
        unvisited = set(customer_list)
        route = []
        current = depot
        while unvisited:
            nearest = min(unvisited, key=lambda c: truck_tt[current][c])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return route
    
    # Build initial solution: all customers on truck route
    initial_route = nn_tsp(customers)
    
    # Try to improve TSP with 2-opt
    def two_opt_improve(route):
        improved = True
        while improved:
            improved = False
            for i in range(len(route)):
                for j in range(i + 2, len(route)):
                    # Compute current cost of segment
                    prev_i = depot if i == 0 else route[i-1]
                    next_j = depot if j == len(route) - 1 else route[j+1]
                    
                    old_cost = truck_tt[prev_i][route[i]] + truck_tt[route[j]][next_j]
                    new_cost = truck_tt[prev_i][route[j]] + truck_tt[route[i]][next_j]
                    
                    if new_cost < old_cost:
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
        return route
    
    initial_route = two_opt_improve(initial_route)
    
    # Now try to offload customers to drones
    def try_drone_offload(truck_route_cust, drone_assigns):
        """Try to move customers from truck route to drones."""
        route_nodes = set([depot] + list(truck_route_cust))
        
        # For each customer on truck route, see if a drone can serve it
        # from some node on the route
        best_route = list(truck_route_cust)
        best_assigns = list(drone_assigns)
        
        # Compute current drone battery usage
        drone_bat = [0] * num_drones
        for d, launch, cust in best_assigns:
            drone_bat[d] += drone_bc[d][launch][cust]
        
        improved = True
        while improved:
            improved = False
            for idx in range(len(best_route)):
                cust = best_route[idx]
                # Try to assign this customer to a drone from some launch node
                best_saving = 0
                best_move = None
                
                # Compute savings from removing this customer from truck route
                prev_node = depot if idx == 0 else best_route[idx-1]
                next_node = depot if idx == len(best_route) - 1 else best_route[idx+1]
                
                truck_saving = (truck_tt[prev_node][cust] + truck_tt[cust][next_node] 
                               - truck_tt[prev_node][next_node])
                
                # Try each launch node and each drone
                potential_launches = set([depot] + [c for i, c in enumerate(best_route) if i != idx])
                
                for launch in potential_launches:
                    for d in range(num_drones):
                        bc = drone_bc[d][launch][cust]
                        if drone_bat[d] + bc > drone_caps[d]:
                            continue
                        dt_val = drone_dt[d][launch][cust]
                        
                        # The waiting time increase at launch node
                        # Current max waiting at launch
                        from collections import defaultdict
                        node_drone_times = defaultdict(float)
                        for dd, ll, cc in best_assigns:
                            if ll == launch:
                                node_drone_times[dd] += drone_dt[dd][ll][cc]
                        
                        current_wait = max(node_drone_times.values()) if node_drone_times else 0
                        node_drone_times[d] += dt_val
                        new_wait = max(node_drone_times.values()) if node_drone_times else 0
                        wait_increase = new_wait - current_wait
                        
                        saving = truck_saving - wait_increase
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (idx, d, launch, cust)
                
                if best_move is not None:
                    idx_rm, d, launch, cust = best_move
                    best_route.pop(idx_rm)
                    best_assigns.append((d, launch, cust))
                    drone_bat[d] += drone_bc[d][launch][cust]
                    improved = True
                    break  # restart loop
        
        return best_route, best_assigns
    
    truck_route, drone_assigns = try_drone_offload(initial_route, [])
    
    # Make sure feasible
    if not check_feasibility(truck_route, drone_assigns):
        # Fallback: all truck
        truck_route = list(initial_route)
        drone_assigns = []
    
    obj, waiting, _ = compute_objective(truck_route, drone_assigns)
    
    best_obj = obj
    best_route = list(truck_route)
    best_assigns = list(drone_assigns)
    best_waiting = dict(waiting)
    
    def build_solution_dict(route_cust, assigns, waiting_dict, obj_val):
        # Build truck_route arcs
        full_route = [depot] + list(route_cust)
        arcs = []
        for i in range(len(full_route) - 1):
            arcs.append([full_route[i], full_route[i+1]])
        # Last arc to virtual sink
        arcs.append([full_route[-1], virtual_sink])
        
        da_list = []
        for d, launch, cust in assigns:
            da_list.append({
                "drone": d,
                "dispatch_node": launch,
                "delivery_node": cust
            })
        
        wt_dict = {}
        # Include all route nodes
        for node in [depot] + list(route_cust):
            wt_dict[str(node)] = waiting_dict.get(node, 0)
        
        return {
            "objective_value": obj_val,
            "truck_route": arcs,
            "drone_assignments": da_list,
            "waiting_times": wt_dict
        }
    
    sol_dict = build_solution_dict(best_route, best_assigns, best_waiting, best_obj)
    if logger:
        logger.log_solution(best_obj, sol_dict)
    
    # Local search improvements
    random.seed(0)
    
    def perturb_and_improve(truck_route_cust, drone_assigns, temperature=1.0):
        """Apply various local search moves."""
        route = list(truck_route_cust)
        assigns = list(drone_assigns)
        
        move_type = random.choice(['swap_truck', 'move_to_drone', 'move_to_truck', 
                                    'reassign_drone', 'or_opt', 'two_opt_step',
                                    'swap_truck_drone', 'change_launch'])
        
        if move_type == 'swap_truck' and len(route) >= 2:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        
        elif move_type == 'move_to_drone' and len(route) >= 1:
            # Move a random truck customer to drone
            idx = random.randint(0, len(route) - 1)
            cust = route[idx]
            
            # Find a feasible drone assignment
            remaining_route = route[:idx] + route[idx+1:]
            launch_nodes = [depot] + remaining_route
            
            options = []
            drone_bat = [0] * num_drones
            for d, l, c in assigns:
                drone_bat[d] += drone_bc[d][l][c]
            
            for launch in launch_nodes:
                for d in range(num_drones):
                    bc = drone_bc[d][launch][cust]
                    if drone_bat[d] + bc <= drone_caps[d]:
                        options.append((d, launch, cust))
            
            if options:
                choice = random.choice(options)
                route = remaining_route
                assigns = assigns + [choice]
            else:
                return None  # infeasible
        
        elif move_type == 'move_to_truck' and len(assigns) >= 1:
            # Move a drone customer to truck
            idx = random.randint(0, len(assigns) - 1)
            d, launch, cust = assigns[idx]
            assigns = assigns[:idx] + assigns[idx+1:]
            
            # Insert at best position
            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(route) + 1):
                prev = depot if pos == 0 else route[pos-1]
                nxt = depot if pos == len(route) else route[pos]
                cost = truck_tt[prev][cust] + truck_tt[cust][nxt] - truck_tt[prev][nxt]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            route.insert(best_pos, cust)
        
        elif move_type == 'reassign_drone' and len(assigns) >= 1:
            # Change which drone or launch node for a drone customer
            idx = random.randint(0, len(assigns) - 1)
            d_old, launch_old, cust = assigns[idx]
            
            route_nodes = [depot] + route
            drone_bat = [0] * num_drones
            for i, (d, l, c) in enumerate(assigns):
                if i != idx:
                    drone_bat[d] += drone_bc[d][l][c]
            
            options = []
            for launch in route_nodes:
                for d in range(num_drones):
                    bc = drone_bc[d][launch][cust]
                    if drone_bat[d] + bc <= drone_caps[d]:
                        options.append((d, launch, cust))
            
            if options:
                choice = random.choice(options)
                assigns[idx] = choice
            else:
                return None
        
        elif move_type == 'change_launch' and len(assigns) >= 1:
            idx = random.randint(0, len(assigns) - 1)
            d, launch_old, cust = assigns[idx]
            
            route_nodes = [depot] + route
            drone_bat_d = sum(drone_bc[d][l][c] for dd, l, c in assigns if dd == d) - drone_bc[d][launch_old][cust]
            
            options = []
            for launch in route_nodes:
                bc = drone_bc[d][launch][cust]
                if drone_bat_d + bc <= drone_caps[d]:
                    options.append(launch)
            
            if options:
                new_launch = random.choice(options)
                assigns[idx] = (d, new_launch, cust)
            else:
                return None
        
        elif move_type == 'or_opt' and len(route) >= 2:
            # Move a segment of 1-2 customers to another position
            seg_len = random.randint(1, min(2, len(route)))
            i = random.randint(0, len(route) - seg_len)
            segment = route[i:i+seg_len]
            remaining = route[:i] + route[i+seg_len:]
            
            if len(remaining) == 0:
                pos = 0
            else:
                pos = random.randint(0, len(remaining))
            route = remaining[:pos] + segment + remaining[pos:]
        
        elif move_type == 'two_opt_step' and len(route) >= 3:
            i = random.randint(0, len(route) - 2)
            j = random.randint(i + 1, len(route) - 1)
            route[i:j+1] = route[i:j+1][::-1]
        
        elif move_type == 'swap_truck_drone' and len(route) >= 1 and len(assigns) >= 1:
            # Swap a truck customer with a drone customer
            t_idx = random.randint(0, len(route) - 1)
            d_idx = random.randint(0, len(assigns) - 1)
            
            t_cust = route[t_idx]
            d_drone, d_launch, d_cust = assigns[d_idx]
            
            # Put d_cust on truck at position t_idx
            route[t_idx] = d_cust
            
            # Assign t_cust to some drone from some launch
            route_nodes = [depot] + route
            drone_bat = [0] * num_drones
            new_assigns = assigns[:d_idx] + assigns[d_idx+1:]
            for d, l, c in new_assigns:
                drone_bat[d] += drone_bc[d][l][c]
            
            options = []
            for launch in route_nodes:
                for d in range(num_drones):
                    bc = drone_bc[d][launch][t_cust]
                    if drone_bat[d] + bc <= drone_caps[d]:
                        options.append((d, launch, t_cust))
            
            if options:
                choice = random.choice(options)
                assigns = new_assigns + [choice]
            else:
                return None
        else:
            return None
        
        # Check feasibility
        if not check_feasibility(route, assigns):
            return None
        
        obj_new, wait_new, _ = compute_objective(route, assigns)
        return (obj_new, route, assigns, wait_new)
    
    # Run local search with simulated annealing
    current_route = list(best_route)
    current_assigns = list(best_assigns)
    current_obj = best_obj
    
    iterations = 0
    temp = max(1.0, best_obj * 0.02)
    cooling = 0.9999
    
    while time.time() - start_time < time_limit - 1:
        iterations += 1
        
        if iterations % 1000 == 0:
            temp *= 0.99
        
        result = perturb_and_improve(current_route, current_assigns, temp)
        
        if result is None:
            continue
        
        new_obj, new_route, new_assigns, new_waiting = result
        
        delta = new_obj - current_obj
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / max(temp, 0.01))):
            current_route = new_route
            current_assigns = new_assigns
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_route = list(current_route)
                best_assigns = list(current_assigns)
                best_waiting = dict(new_waiting)
                
                sol_dict = build_solution_dict(best_route, best_assigns, best_waiting, best_obj)
                if logger:
                    logger.log_solution(best_obj, sol_dict)
        
        temp *= cooling
    
    # After local search, do a final optimization of drone assignments
    # For each drone customer, find the best (drone, launch) combo
    def optimize_drone_assignments(route_cust, assigns):
        from collections import defaultdict
        
        # Get drone customers
        drone_custs = [(a[2], i) for i, a in enumerate(assigns)]
        
        improved = True
        while improved:
            improved = False
            for ci, (orig_idx_info) in enumerate(drone_custs):
                # Actually let me just iterate over assigns
                pass
            
            for idx in range(len(assigns)):
                d_old, launch_old, cust = assigns[idx]
                
                # Current objective contribution: need full recompute for accuracy
                # Just try all options and pick best
                route_nodes = [depot] + list(route_cust)
                
                drone_bat = [0] * num_drones
                for i, (d, l, c) in enumerate(assigns):
                    if i != idx:
                        drone_bat[d] += drone_bc[d][l][c]
                
                best_option = (d_old, launch_old)
                test_assigns = list(assigns)
                obj_current, _, _ = compute_objective(route_cust, test_assigns)
                best_local_obj = obj_current
                
                for launch in route_nodes:
                    for d in range(num_drones):
                        bc = drone_bc[d][launch][cust]
                        if drone_bat[d] + bc <= drone_caps[d]:
                            test_assigns[idx] = (d, launch, cust)
                            obj_test, _, _ = compute_objective(route_cust, test_assigns)
                            if obj_test < best_local_obj:
                                best_local_obj = obj_test
                                best_option = (d, launch)
                
                if best_option != (d_old, launch_old):
                    assigns[idx] = (best_option[0], best_option[1], cust)
                    improved = True
            
            if time.time() - start_time > time_limit - 0.5:
                break
        
        return assigns
    
    if time.time() - start_time < time_limit - 2:
        best_assigns = optimize_drone_assignments(best_route, list(best_assigns))
        obj_final, wait_final, _ = compute_objective(best_route, best_assigns)
        if obj_final < best_obj:
            best_obj = obj_final
            best_waiting = wait_final
            sol_dict = build_solution_dict(best_route, best_assigns, best_waiting, best_obj)
            if logger:
                logger.log_solution(best_obj, sol_dict)
    
    # Also try 2-opt on final truck route
    if time.time() - start_time < time_limit - 1:
        improved_2opt = True
        while improved_2opt and time.time() - start_time < time_limit - 0.5:
            improved_2opt = False
            for i in range(len(best_route)):
                if time.time() - start_time > time_limit - 0.5:
                    break
                for j in range(i + 2, len(best_route)):
                    test_route = list(best_route)
                    test_route[i:j+1] = test_route[i:j+1][::-1]
                    obj_test, wait_test, _ = compute_objective(test_route, best_assigns)
                    if obj_test < best_obj:
                        best_obj = obj_test
                        best_route = test_route
                        best_waiting = wait_test
                        improved_2opt = True
                        
                        sol_dict = build_solution_dict(best_route, best_assigns, best_waiting, best_obj)
                        if logger:
                            logger.log_solution(best_obj, sol_dict)
                        break
                if improved_2opt:
                    break
    
    sol_dict = build_solution_dict(best_route, best_assigns, best_waiting, best_obj)
    return sol_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()