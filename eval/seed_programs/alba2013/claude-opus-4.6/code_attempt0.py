import argparse
import json
import time
import random
import math
import sys
from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    params = instance["parameters"]
    num_stacks = params["num_stacks_s"]
    stack_height = params["stack_height_l"]
    n = params["num_requests_n"]
    
    pickup_region = instance["pickup_region"]
    delivery_region = instance["delivery_region"]
    requests = instance["requests"]
    
    pickup_depot = pickup_region["depot_index"]
    delivery_depot = delivery_region["depot_index"]
    pickup_dist = pickup_region["distance_matrix"]
    delivery_dist = delivery_region["distance_matrix"]
    
    # Build request mapping: pickup_vertex -> request_id, delivery_vertex -> request_id
    pickup_to_request = {}
    delivery_to_request = {}
    request_pickup = {}
    request_delivery = {}
    
    for req in requests:
        rid = req["request_id"]
        pv = req["pickup_vertex"]
        dv = req["delivery_vertex"]
        pickup_to_request[pv] = rid
        delivery_to_request[dv] = rid
        request_pickup[rid] = pv
        request_delivery[rid] = dv
    
    # Get list of pickup vertices (excluding depot) and delivery vertices (excluding depot)
    pickup_vertices = [req["pickup_vertex"] for req in requests]
    delivery_vertices = [req["delivery_vertex"] for req in requests]
    
    def tour_cost(tour, dist_matrix):
        cost = 0
        for i in range(len(tour) - 1):
            cost += dist_matrix[tour[i]][tour[i+1]]
        cost += dist_matrix[tour[-1]][tour[0]]
        return cost
    
    def check_lifo_feasibility(pickup_order, delivery_order):
        """Check if the given pickup and delivery orders are feasible with LIFO stacks."""
        # pickup_order: list of request_ids in pickup order
        # delivery_order: list of request_ids in delivery order
        # We need to assign each item to a stack such that LIFO is respected
        # LIFO constraint: for items in the same stack, if item i is picked up before item j,
        # then item j must be delivered before item i.
        
        # This is equivalent to: items in the same stack must have their delivery order
        # be the reverse of their pickup order.
        
        # We can model this as: given pickup positions and delivery positions,
        # partition items into s stacks of size <= l such that within each stack,
        # the delivery order is the reverse of the pickup order.
        
        # For each stack, items form an increasing sequence in pickup order
        # and a decreasing sequence in delivery order (i.e., last picked up = first delivered).
        
        # This means: if we consider items sorted by pickup order, and look at their
        # delivery positions, we need to partition the delivery position sequence into
        # s subsequences, each of length <= l, where each subsequence is decreasing.
        
        # By Dilworth's theorem, the minimum number of decreasing subsequences equals
        # the length of the longest increasing subsequence of delivery positions.
        # But we also have the stack height constraint.
        
        # Get delivery positions for each request
        delivery_pos = {}
        for i, rid in enumerate(delivery_order):
            delivery_pos[rid] = i
        
        # Sequence of delivery positions in pickup order
        seq = [delivery_pos[rid] for rid in pickup_order]
        
        # Greedy assignment to stacks
        # For each item (in pickup order), assign to a stack where the top item
        # has a delivery position > current item's delivery position (LIFO: current delivered first)
        # and the stack is not full.
        # Use greedy: assign to the stack whose top has the smallest delivery position
        # that is still > current delivery position.
        
        stacks = []  # Each stack: (top_delivery_pos, size)
        
        for dp in seq:
            # Find best stack: top delivery position > dp, and size < stack_height
            best_idx = -1
            best_top = float('inf')
            for i, (top_dp, sz) in enumerate(stacks):
                if top_dp > dp and sz < stack_height:
                    if top_dp < best_top:
                        best_top = top_dp
                        best_idx = i
            
            if best_idx >= 0:
                stacks[best_idx] = (dp, stacks[best_idx][1] + 1)
            else:
                if len(stacks) < num_stacks:
                    stacks.append((dp, 1))
                else:
                    return False
        
        return True
    
    def get_stack_assignment(pickup_order, delivery_order):
        """Get stack assignment if feasible, else None."""
        delivery_pos = {}
        for i, rid in enumerate(delivery_order):
            delivery_pos[rid] = i
        
        seq = [delivery_pos[rid] for rid in pickup_order]
        
        stacks = []  # Each stack: (top_delivery_pos, size, list of items)
        assignment = {}  # request_id -> stack_index
        
        for idx, dp in enumerate(seq):
            rid = pickup_order[idx]
            best_idx = -1
            best_top = float('inf')
            for i, (top_dp, sz, _) in enumerate(stacks):
                if top_dp > dp and sz < stack_height:
                    if top_dp < best_top:
                        best_top = top_dp
                        best_idx = i
            
            if best_idx >= 0:
                stacks[best_idx] = (dp, stacks[best_idx][1] + 1, stacks[best_idx][2] + [rid])
                assignment[rid] = best_idx
            else:
                if len(stacks) < num_stacks:
                    assignment[rid] = len(stacks)
                    stacks.append((dp, 1, [rid]))
                else:
                    return None
        
        return assignment
    
    def nearest_neighbor_tour(vertices, depot, dist_matrix):
        """Construct a tour using nearest neighbor heuristic."""
        if not vertices:
            return [depot]
        
        unvisited = set(vertices)
        tour = [depot]
        current = depot
        
        while unvisited:
            nearest = min(unvisited, key=lambda v: dist_matrix[current][v])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def tour_to_order(tour, vertex_to_request, depot):
        """Convert tour to request order (excluding depot)."""
        order = []
        for v in tour:
            if v != depot:
                order.append(vertex_to_request[v])
        return order
    
    # Build initial solution using nearest neighbor
    pickup_tour = nearest_neighbor_tour(pickup_vertices, pickup_depot, pickup_dist)
    delivery_tour = nearest_neighbor_tour(delivery_vertices, delivery_depot, delivery_dist)
    
    pickup_order = tour_to_order(pickup_tour, pickup_to_request, pickup_depot)
    delivery_order = tour_to_order(delivery_tour, delivery_to_request, delivery_depot)
    
    # If not feasible, try to fix delivery order
    def make_feasible_delivery(pickup_order):
        """Create a delivery order that is guaranteed feasible: reverse of pickup order."""
        # Reverse pickup order means each stack only needs 1 item or items are in perfect LIFO
        # Actually, the simplest feasible delivery order with unlimited stacks of height >= n
        # is any order. But with limited stacks, we need to be careful.
        # The trivially feasible delivery order is the exact reverse of pickup order
        # (all items can go in one stack if stack_height >= n, else distribute).
        
        # With reverse order, all items can go into 1 stack (if stack_height >= n)
        # or we can split into ceil(n/stack_height) stacks.
        
        # Actually, reverse of pickup order means: last picked up is first delivered.
        # All items can go in 1 stack. If n > stack_height, we need multiple stacks.
        # With reverse order, we can assign items to stacks in groups:
        # stack 0: first l items picked up (they'll be delivered in reverse = last l delivered)
        # stack 1: next l items picked up, etc.
        # But delivery order must be: items from last stack first (reversed), then previous stack, etc.
        
        # Let's just build the delivery tour from the reverse pickup order
        # partitioned into stacks.
        
        num_needed_stacks = math.ceil(n / stack_height)
        if num_needed_stacks > num_stacks:
            # This approach won't work directly, need a smarter approach
            return None
        
        # Partition pickup_order into stacks
        stack_contents = []
        for i in range(0, n, stack_height):
            stack_contents.append(pickup_order[i:i+stack_height])
        
        # Delivery order: empty stacks in reverse order of filling
        # Last stack filled should be emptied first
        # Within each stack, items are emptied in reverse order of pickup (LIFO)
        delivery_order = []
        for stack in reversed(stack_contents):
            delivery_order.extend(reversed(stack))
        
        return delivery_order
    
    if not check_lifo_feasibility(pickup_order, delivery_order):
        # Try reverse-based delivery
        new_delivery_order = make_feasible_delivery(pickup_order)
        if new_delivery_order is not None and check_lifo_feasibility(pickup_order, new_delivery_order):
            delivery_order = new_delivery_order
            delivery_tour = [delivery_depot] + [request_delivery[rid] for rid in delivery_order]
        else:
            # Just use reverse of pickup order as delivery order
            delivery_order = list(reversed(pickup_order))
            delivery_tour = [delivery_depot] + [request_delivery[rid] for rid in delivery_order]
    
    best_pickup_tour = list(pickup_tour)
    best_delivery_tour = list(delivery_tour)
    best_pickup_order = list(pickup_order)
    best_delivery_order = list(delivery_order)
    best_cost = tour_cost(pickup_tour, pickup_dist) + tour_cost(delivery_tour, delivery_dist)
    
    if logger:
        logger.log(best_cost)
    
    def two_opt(tour, dist_matrix):
        """Apply 2-opt improvement to a tour."""
        n_tour = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(1, n_tour - 1):
                for j in range(i + 1, n_tour):
                    # Reverse segment from i to j
                    old_cost = dist_matrix[tour[i-1]][tour[i]] + dist_matrix[tour[j]][tour[(j+1) % n_tour]]
                    new_cost = dist_matrix[tour[i-1]][tour[j]] + dist_matrix[tour[i]][tour[(j+1) % n_tour]]
                    if new_cost < old_cost:
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        return tour
    
    # Apply 2-opt independently to both tours, then check feasibility
    def optimize_tours(pickup_tour, delivery_tour, pickup_order, delivery_order):
        # Try 2-opt on pickup
        new_pt = two_opt(list(pickup_tour), pickup_dist)
        new_po = tour_to_order(new_pt, pickup_to_request, pickup_depot)
        if check_lifo_feasibility(new_po, delivery_order):
            pickup_tour = new_pt
            pickup_order = new_po
        
        # Try 2-opt on delivery
        new_dt = two_opt(list(delivery_tour), delivery_dist)
        new_do = tour_to_order(new_dt, delivery_to_request, delivery_depot)
        if check_lifo_feasibility(pickup_order, new_do):
            delivery_tour = new_dt
            delivery_order = new_do
        
        return pickup_tour, delivery_tour, pickup_order, delivery_order
    
    pickup_tour, delivery_tour, pickup_order, delivery_order = optimize_tours(
        best_pickup_tour, best_delivery_tour, best_pickup_order, best_delivery_order
    )
    
    cost = tour_cost(pickup_tour, pickup_dist) + tour_cost(delivery_tour, delivery_dist)
    if cost < best_cost:
        best_cost = cost
        best_pickup_tour = list(pickup_tour)
        best_delivery_tour = list(delivery_tour)
        best_pickup_order = list(pickup_order)
        best_delivery_order = list(delivery_order)
        if logger:
            logger.log(best_cost)
    
    # Now use a more sophisticated local search with simulated annealing
    # that considers both tours jointly
    
    current_pickup_tour = list(best_pickup_tour)
    current_delivery_tour = list(best_delivery_tour)
    current_pickup_order = list(best_pickup_order)
    current_delivery_order = list(best_delivery_order)
    current_cost = best_cost
    
    temperature = best_cost * 0.05 if best_cost > 0 else 100
    cooling_rate = 0.9995
    min_temp = 0.01
    
    iteration = 0
    max_no_improve = max(n * n * 20, 5000)
    no_improve = 0
    
    def or_opt_move(tour, dist_matrix, seg_len=1):
        """Try Or-opt move: relocate a segment of length seg_len."""
        n_tour = len(tour)
        best_delta = 0
        best_move = None
        
        for i in range(1, n_tour):
            if i + seg_len > n_tour:
                break
            # Cost of removing segment [i, i+seg_len-1]
            prev_i = i - 1
            next_seg = (i + seg_len) % n_tour
            
            removal_cost = (dist_matrix[tour[prev_i]][tour[i]] + 
                          dist_matrix[tour[i + seg_len - 1]][tour[next_seg]])
            reconnect_cost = dist_matrix[tour[prev_i]][tour[next_seg]]
            
            for j in range(n_tour):
                if j >= i - 1 and j <= i + seg_len - 1:
                    continue
                next_j = (j + 1) % n_tour
                if next_j >= i and next_j <= i + seg_len - 1:
                    continue
                
                insert_cost = (dist_matrix[tour[j]][tour[i]] + 
                             dist_matrix[tour[i + seg_len - 1]][tour[next_j]])
                current_arc = dist_matrix[tour[j]][tour[next_j]]
                
                delta = reconnect_cost + insert_cost - removal_cost - current_arc
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, seg_len, j)
        
        return best_delta, best_move
    
    def apply_or_opt(tour, move):
        i, seg_len, j = move
        segment = tour[i:i+seg_len]
        new_tour = tour[:i] + tour[i+seg_len:]
        # Find new position of j
        if j < i:
            insert_pos = j + 1
        else:
            insert_pos = j - seg_len + 1
        new_tour = new_tour[:insert_pos] + segment + new_tour[insert_pos:]
        return new_tour
    
    def perturb_tour(tour, dist_matrix):
        """Random perturbation: random 2-opt or or-opt or swap."""
        n_tour = len(tour)
        if n_tour <= 2:
            return list(tour)
        
        move_type = random.randint(0, 2)
        new_tour = list(tour)
        
        if move_type == 0:
            # Random 2-opt
            i = random.randint(1, n_tour - 2)
            j = random.randint(i + 1, n_tour - 1)
            new_tour[i:j+1] = new_tour[i:j+1][::-1]
        elif move_type == 1:
            # Random swap
            i = random.randint(1, n_tour - 1)
            j = random.randint(1, n_tour - 1)
            if i != j:
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        else:
            # Random or-opt (relocate single vertex)
            i = random.randint(1, n_tour - 1)
            vertex = new_tour.pop(i)
            j = random.randint(1, len(new_tour))
            new_tour.insert(j, vertex)
        
        return new_tour
    
    # Main SA loop
    random.seed(42)
    
    while time.time() - start_time < time_limit - 1.0:
        iteration += 1
        
        # Choose which tour to perturb
        which = random.random()
        
        if which < 0.45:
            # Perturb pickup tour
            new_pt = perturb_tour(current_pickup_tour, pickup_dist)
            new_po = tour_to_order(new_pt, pickup_to_request, pickup_depot)
            new_dt = current_delivery_tour
            new_do = current_delivery_order
        elif which < 0.9:
            # Perturb delivery tour
            new_pt = current_pickup_tour
            new_po = current_pickup_order
            new_dt = perturb_tour(current_delivery_tour, delivery_dist)
            new_do = tour_to_order(new_dt, delivery_to_request, delivery_depot)
        else:
            # Perturb both
            new_pt = perturb_tour(current_pickup_tour, pickup_dist)
            new_po = tour_to_order(new_pt, pickup_to_request, pickup_depot)
            new_dt = perturb_tour(current_delivery_tour, delivery_dist)
            new_do = tour_to_order(new_dt, delivery_to_request, delivery_depot)
        
        new_cost = tour_cost(new_pt, pickup_dist) + tour_cost(new_dt, delivery_dist)
        delta = new_cost - current_cost
        
        accept = False
        if delta < 0:
            # Check feasibility only for improvements or SA acceptance
            if check_lifo_feasibility(new_po, new_do):
                accept = True
        elif temperature > min_temp:
            if random.random() < math.exp(-delta / temperature):
                if check_lifo_feasibility(new_po, new_do):
                    accept = True
        
        if accept:
            current_pickup_tour = new_pt
            current_delivery_tour = new_dt
            current_pickup_order = new_po
            current_delivery_order = new_do
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_pickup_tour = list(current_pickup_tour)
                best_delivery_tour = list(current_delivery_tour)
                best_pickup_order = list(current_pickup_order)
                best_delivery_order = list(current_delivery_order)
                no_improve = 0
                if logger:
                    logger.log(best_cost)
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        temperature *= cooling_rate
        
        # Periodic restart from best
        if no_improve > max_no_improve:
            current_pickup_tour = list(best_pickup_tour)
            current_delivery_tour = list(best_delivery_tour)
            current_pickup_order = list(best_pickup_order)
            current_delivery_order = list(best_delivery_order)
            current_cost = best_cost
            temperature = best_cost * 0.02 if best_cost > 0 else 50
            no_improve = 0
    
    # Final 2-opt pass on best solution
    def two_opt_feasible_pickup(pickup_tour, delivery_order):
        """2-opt on pickup tour checking LIFO feasibility."""
        n_tour = len(pickup_tour)
        improved = True
        while improved and time.time() - start_time < time_limit - 0.2:
            improved = False
            for i in range(1, n_tour - 1):
                for j in range(i + 1, n_tour):
                    old_cost = (pickup_dist[pickup_tour[i-1]][pickup_tour[i]] + 
                              pickup_dist[pickup_tour[j]][pickup_tour[(j+1) % n_tour]])
                    new_cost = (pickup_dist[pickup_tour[i-1]][pickup_tour[j]] + 
                              pickup_dist[pickup_tour[i]][pickup_tour[(j+1) % n_tour]])
                    if new_cost < old_cost:
                        new_tour = list(pickup_tour)
                        new_tour[i:j+1] = new_tour[i:j+1][::-1]
                        new_order = tour_to_order(new_tour, pickup_to_request, pickup_depot)
                        if check_lifo_feasibility(new_order, delivery_order):
                            pickup_tour = new_tour
                            improved = True
                            break
                if improved:
                    break
        return pickup_tour
    
    def two_opt_feasible_delivery(pickup_order, delivery_tour):
        """2-opt on delivery tour checking LIFO feasibility."""
        n_tour = len(delivery_tour)
        improved = True
        while improved and time.time() - start_time < time_limit - 0.2:
            improved = False
            for i in range(1, n_tour - 1):
                for j in range(i + 1, n_tour):
                    old_cost = (delivery_dist[delivery_tour[i-1]][delivery_tour[i]] + 
                              delivery_dist[delivery_tour[j]][delivery_tour[(j+1) % n_tour]])
                    new_cost = (delivery_dist[delivery_tour[i-1]][delivery_tour[j]] + 
                              delivery_dist[delivery_tour[i]][delivery_tour[(j+1) % n_tour]])
                    if new_cost < old_cost:
                        new_tour = list(delivery_tour)
                        new_tour[i:j+1] = new_tour[i:j+1][::-1]
                        new_order = tour_to_order(new_tour, delivery_to_request, delivery_depot)
                        if check_lifo_feasibility(pickup_order, new_order):
                            delivery_tour = new_tour
                            improved = True
                            break
                if improved:
                    break
        return delivery_tour
    
    if time.time() - start_time < time_limit - 2.0:
        final_pt = two_opt_feasible_pickup(list(best_pickup_tour), best_delivery_order)
        final_po = tour_to_order(final_pt, pickup_to_request, pickup_depot)
        final_dt = two_opt_feasible_delivery(final_po, list(best_delivery_tour))
        final_do = tour_to_order(final_dt, delivery_to_request, delivery_depot)
        
        final_cost = tour_cost(final_pt, pickup_dist) + tour_cost(final_dt, delivery_dist)
        if final_cost < best_cost and check_lifo_feasibility(final_po, final_do):
            best_cost = final_cost
            best_pickup_tour = final_pt
            best_delivery_tour = final_dt
            if logger:
                logger.log(best_cost)
    
    return best_cost, best_pickup_tour, best_delivery_tour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    best_cost, best_pickup_tour, best_delivery_tour = solve(instance, args.time_limit, logger)
    
    solution = {
        "objective_value": best_cost,
        "pickup_tour": best_pickup_tour,
        "delivery_tour": best_delivery_tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()