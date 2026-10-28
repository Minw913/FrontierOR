import argparse
import json
import time
import random
import math
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    num_edges = instance['num_edges']
    F = instance['bisection_capacity_F']
    node_weights = instance['node_weights']
    edges = instance['edges']
    edge_weights = instance['edge_weights']
    total_weight = instance['total_node_weight']

    start_time = time.time()
    deadline = start_time + args.time_limit - 1.0  # leave 1 second buffer

    # Build adjacency list with edge costs
    # adj[node] = list of (neighbor, edge_cost, edge_index)
    adj = defaultdict(list)
    for idx, (e, w) in enumerate(zip(edges, edge_weights)):
        u, v = e[0], e[1]
        adj[u].append((v, w, idx))
        adj[v].append((u, w, idx))

    # partition[i] = 0 means node i is in group S (containing node 0)
    # partition[i] = 1 means node i is in group complement
    partition = [0] * num_nodes
    
    # Weight of each group
    weight = [0, 0]
    
    def compute_cut_cost():
        cost = 0.0
        for idx, (e, w) in enumerate(zip(edges, edge_weights)):
            if partition[e[0]] != partition[e[1]]:
                cost += w
        return cost

    # gain[i] = reduction in cut cost if node i is moved to the other group
    # gain[i] = sum of edge costs to neighbors in same group - sum of edge costs to neighbors in different group
    def compute_gains():
        gains = [0.0] * num_nodes
        for i in range(num_nodes):
            g = 0.0
            p = partition[i]
            for (nb, w, _) in adj[i]:
                if partition[nb] == p:
                    g -= w  # moving i away from same-group neighbor increases cut
                else:
                    g += w  # moving i toward different-group neighbor decreases cut
            gains[i] = g
        return gains

    # Initialize: try to find a good initial partition
    # Use greedy bisection: assign nodes to minimize initial cut while respecting capacity
    
    # Simple initial partition: put node 0 in group 0, greedily assign rest
    # Actually, let's try a BFS-based approach or just split by weight
    
    # Sort nodes by weight descending and do a balanced partition
    sorted_nodes = sorted(range(num_nodes), key=lambda x: -node_weights[x])
    
    # Greedy assignment: assign each node to the lighter group
    weight = [0, 0]
    partition = [0] * num_nodes
    
    for n in sorted_nodes:
        # Try to assign to lighter group, but respect capacity
        if weight[0] <= weight[1]:
            preferred = 0
        else:
            preferred = 1
        
        if weight[preferred] + node_weights[n] <= F:
            partition[n] = preferred
            weight[preferred] += node_weights[n]
        elif weight[1 - preferred] + node_weights[n] <= F:
            partition[n] = 1 - preferred
            weight[1 - preferred] += node_weights[n]
        else:
            # Shouldn't happen if F >= ceil(total/2)
            partition[n] = preferred
            weight[preferred] += node_weights[n]
    
    # Make sure node 0 is in group 0
    if partition[0] != 0:
        # Swap all assignments
        for i in range(num_nodes):
            partition[i] = 1 - partition[i]
        weight[0], weight[1] = weight[1], weight[0]
    
    current_cost = compute_cut_cost()
    best_cost = current_cost
    best_partition = partition[:]
    
    if logger:
        logger.log(best_cost)

    # Kernighan-Lin / Fiduccia-Mattheyses style local search
    def fm_pass():
        nonlocal partition, weight, current_cost, best_cost, best_partition
        
        gains = compute_gains()
        locked = [False] * num_nodes
        
        # Track the sequence of moves and costs
        moves = []
        costs = []
        running_cost = current_cost
        best_pass_cost = running_cost
        best_pass_idx = -1
        
        for step in range(num_nodes):
            if time.time() > deadline:
                break
            
            # Find the best unlocked node to move
            best_gain = -float('inf')
            best_node = -1
            
            for i in range(num_nodes):
                if locked[i]:
                    continue
                p = partition[i]
                other = 1 - p
                # Check if moving i to other group is feasible
                if weight[other] + node_weights[i] > F:
                    continue
                if gains[i] > best_gain:
                    best_gain = gains[i]
                    best_node = i
            
            if best_node == -1:
                break
            
            # Move best_node
            i = best_node
            old_p = partition[i]
            new_p = 1 - old_p
            
            partition[i] = new_p
            weight[old_p] -= node_weights[i]
            weight[new_p] += node_weights[i]
            locked[i] = True
            running_cost -= best_gain
            
            # Update gains of neighbors
            for (nb, w, _) in adj[i]:
                if locked[nb]:
                    continue
                if partition[nb] == new_p:
                    # nb and i are now in the same group
                    # Before move: if nb was same as old_p, edge was internal -> now cut
                    # nb is in new_p, i moved to new_p, so they're same now
                    # The gain of nb changes: this edge was cut (i was in old_p, nb in new_p) -> now internal
                    # For nb's gain calculation: moving nb away from new_p
                    # Before: this edge contributed +w to nb's gain (different groups)
                    # After: this edge contributes -w to nb's gain (same group)
                    gains[nb] -= 2 * w
                else:
                    # nb is in old_p, i moved to new_p
                    # Before: same group -> now different group
                    # Before: this edge contributed -w to nb's gain
                    # After: this edge contributes +w to nb's gain
                    gains[nb] += 2 * w
            
            moves.append(i)
            costs.append(running_cost)
            
            if running_cost < best_pass_cost - 1e-10:
                best_pass_cost = running_cost
                best_pass_idx = step
        
        if best_pass_idx == -1:
            # No improvement, undo all moves
            for m in reversed(moves):
                p = partition[m]
                partition[m] = 1 - p
                weight[p] -= node_weights[m]
                weight[1 - p] += node_weights[m]
            return False
        
        # Undo moves after best_pass_idx
        for j in range(len(moves) - 1, best_pass_idx, -1):
            m = moves[j]
            p = partition[m]
            partition[m] = 1 - p
            weight[p] -= node_weights[m]
            weight[1 - p] += node_weights[m]
        
        current_cost = best_pass_cost
        
        if current_cost < best_cost - 1e-10:
            best_cost = current_cost
            best_partition[:] = partition[:]
            if logger:
                logger.log(best_cost)
            return True
        
        return best_pass_cost < running_cost + sum(g for g in []) + 1e-10  # just check if any improvement happened
    
    # Run FM passes
    improved = True
    while improved and time.time() < deadline:
        improved = fm_pass()
    
    # Now try swap-based local search (pairwise swaps)
    def swap_local_search():
        nonlocal partition, weight, current_cost, best_cost, best_partition
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            gains = compute_gains()
            
            # Try single moves first
            for i in range(num_nodes):
                if time.time() > deadline:
                    return
                p = partition[i]
                other = 1 - p
                if weight[other] + node_weights[i] <= F and gains[i] > 1e-10:
                    # Move i
                    partition[i] = other
                    weight[p] -= node_weights[i]
                    weight[other] += node_weights[i]
                    current_cost -= gains[i]
                    
                    if current_cost < best_cost - 1e-10:
                        best_cost = current_cost
                        best_partition[:] = partition[:]
                        if logger:
                            logger.log(best_cost)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try pairwise swaps
            found = False
            group0 = [i for i in range(num_nodes) if partition[i] == 0]
            group1 = [i for i in range(num_nodes) if partition[i] == 1]
            
            for i in group0:
                if time.time() > deadline:
                    return
                for j in group1:
                    # Swap i and j: i goes to group 1, j goes to group 0
                    new_w0 = weight[0] - node_weights[i] + node_weights[j]
                    new_w1 = weight[1] - node_weights[j] + node_weights[i]
                    if new_w0 > F or new_w1 > F:
                        continue
                    
                    # Compute gain of swapping
                    # gain_i + gain_j - 2 * edge_cost(i,j) if connected
                    swap_gain = gains[i] + gains[j]
                    # Check if i and j are connected
                    for (nb, w, _) in adj[i]:
                        if nb == j:
                            swap_gain -= 2 * w
                            break
                    
                    if swap_gain > 1e-10:
                        partition[i] = 1
                        partition[j] = 0
                        weight[0] = new_w0
                        weight[1] = new_w1
                        current_cost -= swap_gain
                        
                        if current_cost < best_cost - 1e-10:
                            best_cost = current_cost
                            best_partition[:] = partition[:]
                            if logger:
                                logger.log(best_cost)
                        found = True
                        improved = True
                        break
                if found:
                    break
    
    swap_local_search()
    
    # Multi-start with perturbation
    def perturb_and_search(num_perturbations=5):
        nonlocal partition, weight, current_cost, best_cost, best_partition
        
        for _ in range(100):
            if time.time() > deadline:
                return
            
            # Start from best known solution
            partition[:] = best_partition[:]
            weight[0] = sum(node_weights[i] for i in range(num_nodes) if partition[i] == 0)
            weight[1] = sum(node_weights[i] for i in range(num_nodes) if partition[i] == 1)
            current_cost = compute_cut_cost()
            
            # Random perturbation: move some random nodes
            nodes_to_perturb = random.sample(range(num_nodes), min(num_perturbations, num_nodes))
            for n in nodes_to_perturb:
                p = partition[n]
                other = 1 - p
                if weight[other] + node_weights[n] <= F:
                    partition[n] = other
                    weight[p] -= node_weights[n]
                    weight[other] += node_weights[n]
            
            current_cost = compute_cut_cost()
            
            # Run local search
            swap_local_search()
            
            # Run FM
            imp = True
            while imp and time.time() < deadline:
                imp = fm_pass()
            
            swap_local_search()
    
    # Try for small instances: use Gurobi
    if num_nodes <= 200 and num_edges <= 2000:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = deadline - time.time()
            if remaining_time > 2:
                m = gp.Model()
                m.setParam('TimeLimit', remaining_time - 1)
                m.setParam('OutputFlag', 0)
                m.setParam('Threads', 1)
                
                # x[i] = 1 if node i in group 1 (complement), 0 if in group 0 (S)
                x = m.addVars(num_nodes, vtype=GRB.BINARY, name="x")
                
                # Node 0 must be in group 0
                m.addConstr(x[0] == 0)
                
                # Capacity constraints
                m.addConstr(gp.quicksum(node_weights[i] * (1 - x[i]) for i in range(num_nodes)) <= F)
                m.addConstr(gp.quicksum(node_weights[i] * x[i] for i in range(num_nodes)) <= F)
                
                # Edge cut variables
                y = m.addVars(num_edges, vtype=GRB.BINARY, name="y")
                for idx, e in enumerate(edges):
                    u, v = e[0], e[1]
                    m.addConstr(y[idx] >= x[u] - x[v])
                    m.addConstr(y[idx] >= x[v] - x[u])
                
                m.setObjective(gp.quicksum(edge_weights[idx] * y[idx] for idx in range(num_edges)), GRB.MINIMIZE)
                
                # Set initial solution
                for i in range(num_nodes):
                    x[i].Start = best_partition[i]
                
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if logger:
                            logger.log(obj)
                
                m.optimize(callback)
                
                if m.SolCount > 0:
                    sol_cost = m.ObjVal
                    if sol_cost < best_cost - 1e-10:
                        best_cost = sol_cost
                        for i in range(num_nodes):
                            best_partition[i] = int(round(x[i].X))
                        if logger:
                            logger.log(best_cost)
        except Exception:
            pass
    elif num_nodes <= 1000:
        # For medium instances, try Gurobi with relaxation-guided approach or just more local search
        perturb_and_search(max(3, num_nodes // 50))
    else:
        perturb_and_search(max(5, num_nodes // 100))
    
    # Additional perturbation rounds if time remains
    while time.time() < deadline - 0.5:
        perturb_size = random.randint(2, max(3, num_nodes // 10))
        
        partition[:] = best_partition[:]
        weight[0] = sum(node_weights[i] for i in range(num_nodes) if partition[i] == 0)
        weight[1] = sum(node_weights[i] for i in range(num_nodes) if partition[i] == 1)
        current_cost = compute_cut_cost()
        
        nodes_to_perturb = random.sample(range(1, num_nodes), min(perturb_size, num_nodes - 1))
        for n in nodes_to_perturb:
            p = partition[n]
            other = 1 - p
            if weight[other] + node_weights[n] <= F:
                partition[n] = other
                weight[p] -= node_weights[n]
                weight[other] += node_weights[n]
        
        current_cost = compute_cut_cost()
        swap_local_search()
        
        imp = True
        while imp and time.time() < deadline - 0.5:
            imp = fm_pass()
        
        swap_local_search()

    # Build output
    partition_S = [i for i in range(num_nodes) if best_partition[i] == 0]
    partition_complement = [i for i in range(num_nodes) if best_partition[i] == 1]
    
    # Recompute objective
    obj = 0.0
    for idx, e in enumerate(edges):
        if best_partition[e[0]] != best_partition[e[1]]:
            obj += edge_weights[idx]
    
    solution = {
        "objective_value": obj,
        "partition_S": partition_S,
        "partition_complement": partition_complement
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()