import sys
import time
import json
import random
import math
import argparse

# Global instance data
capacity = 0
num_vehicles = 0
demand = []
ready_time = []
due_date = []
service_time = []
dist = []

class Route:
    def __init__(self, r_id):
        self.r_id = r_id
        self.nodes = [0, 0]  # Depot implicitly at 0
        self.demand = 0
        self.A = []
        self.W = []
        self.D = []
        self.MaxDelay = []
        self.update()
        
    def update(self):
        n = len(self.nodes)
        self.A = [0.0] * n
        self.W = [0.0] * n
        self.D = [0.0] * n
        self.MaxDelay = [0.0] * n
        
        self.A[0] = ready_time[0]
        self.W[0] = 0.0
        self.D[0] = ready_time[0]
        
        for i in range(1, n):
            u = self.nodes[i]
            prev = self.nodes[i-1]
            self.A[i] = self.D[i-1] + dist[prev][u]
            self.W[i] = max(0.0, ready_time[u] - self.A[i])
            self.D[i] = self.A[i] + self.W[i] + service_time[u]
            
        self.MaxDelay[n-1] = max(0.0, due_date[0] - self.A[n-1])
        for i in range(n-2, -1, -1):
            valid_forward_delay = self.W[i+1] + self.MaxDelay[i+1]
            local_delay = due_date[self.nodes[i]] - self.A[i]
            self.MaxDelay[i] = max(0.0, min(local_delay, valid_forward_delay))
            
    def insert(self, idx, u):
        self.nodes.insert(idx, u)
        self.demand += demand[u]
        self.update()
        
    def remove(self, u):
        self.nodes.remove(u)
        self.demand -= demand[u]
        self.update()
        
    def route_distance(self):
        d = 0.0
        for i in range(len(self.nodes)-1):
            d += dist[self.nodes[i]][self.nodes[i+1]]
        return d

def find_valid_insertions(tgt_r, u):
    valid_insertions = []
    if tgt_r.demand + demand[u] > capacity:
        return valid_insertions
        
    n = len(tgt_r.nodes)
    for i in range(n - 1):
        prev = tgt_r.nodes[i]
        nxt = tgt_r.nodes[i+1]
        
        A_u = tgt_r.D[i] + dist[prev][u]
        if A_u > due_date[u] + 1e-5:
            continue
            
        W_u = max(0.0, ready_time[u] - A_u)
        D_u = A_u + W_u + service_time[u]
        
        delay_nxt = D_u + dist[u][nxt] - tgt_r.A[i+1]
        if delay_nxt <= tgt_r.MaxDelay[i+1] + 1e-5:
            cost = dist[prev][u] + dist[u][nxt] - dist[prev][nxt]
            valid_insertions.append((cost, i+1))
            
    return valid_insertions

def get_new_route_insertion_cost(u):
    if demand[u] > capacity:
        return float('inf')
    A_u = ready_time[0] + dist[0][u]
    if A_u > due_date[u] + 1e-5:
        return float('inf')
    W_u = max(0.0, ready_time[u] - A_u)
    D_u = A_u + W_u + service_time[u]
    A_end = D_u + dist[u][0]
    if A_end > due_date[0] + 1e-5:
        return float('inf')
    return dist[0][u] + dist[u][0]

def repair_regret(routes, unassigned, regret_k=2):
    while unassigned:
        insert_options = {u: [] for u in unassigned}
        for r_idx, r in enumerate(routes):
            for u in unassigned:
                insertions = find_valid_insertions(r, u)
                for cost, r_ins_idx in insertions:
                    insert_options[u].append((cost, r_idx, r_ins_idx))
                    
        best_u = None
        max_regret = -1.0
        best_ins = None
        min_cost0 = float('inf')
        
        unassigned_list = list(unassigned)
        random.shuffle(unassigned_list)
        
        for u in unassigned_list:
            opts = insert_options[u]
            new_route_cost = get_new_route_insertion_cost(u)
            if new_route_cost < float('inf'):
                opts.append((new_route_cost, -1, 1))
            opts.sort(key=lambda x: x[0])
            
            if not opts:
                continue
                
            cost0 = opts[0][0]
            cost_k = opts[regret_k-1][0] if regret_k <= len(opts) else 1e9
            
            regret = cost_k - cost0
            if regret > max_regret or (regret == max_regret and cost0 < min_cost0):
                max_regret = regret
                best_u = u
                best_ins = opts[0]
                min_cost0 = cost0
                
        if best_u is None:
            return False  # Failed to insert remaining
            
        u = best_u
        cost, r_idx, idx = best_ins
        if r_idx == -1:
            new_r = Route(len(routes))
            new_r.insert(1, u)
            routes.append(new_r)
        else:
            routes[r_idx].insert(idx, u)
        unassigned.remove(u)
    return True

def destroy_random(routes, q):
    assigned = []
    for r in routes:
        for node in r.nodes[1:-1]:
            assigned.append((node, r))
    
    q = min(q, len(assigned))
    removed = random.sample(assigned, q)
    
    unassigned = set()
    for u, r in removed:
        r.remove(u)
        unassigned.add(u)
        
    routes[:] = [r for r in routes if len(r.nodes) > 2]
    return unassigned

def destroy_worst(routes, q, power=3):
    removals = []
    for r in routes:
        n = len(r.nodes)
        for i in range(1, n-1):
            u = r.nodes[i]
            prev = r.nodes[i-1]
            nxt = r.nodes[i+1]
            cost_saving = dist[prev][u] + dist[u][nxt] - dist[prev][nxt]
            removals.append((cost_saving, u, r))
            
    removals.sort(key=lambda x: x[0], reverse=True)
    unassigned = set()
    q = min(q, len(removals))
    
    for _ in range(q):
        if not removals:
            break
        idx = int((random.random() ** power) * len(removals))
        saving, u, r = removals.pop(idx)
        if u not in unassigned:
            r.remove(u)
            unassigned.add(u)
            
    routes[:] = [r for r in routes if len(r.nodes) > 2]
    return unassigned

def destroy_related(routes, q, power=3):
    assigned = [node for r in routes for node in r.nodes[1:-1]]
    if not assigned:
        return set()
        
    seed = random.choice(assigned)
    distances = [(dist[seed][u], u) for u in assigned if u != seed]
    distances.sort(key=lambda x: x[0])
    
    unassigned = set([seed])
    q = min(q, len(assigned))
    
    for _ in range(q - 1):
        if not distances:
            break
        idx = int((random.random() ** power) * len(distances))
        d, u = distances.pop(idx)
        unassigned.add(u)
        
    for r in routes:
        for u in list(r.nodes[1:-1]):
            if u in unassigned:
                r.remove(u)
                
    routes[:] = [r for r in routes if len(r.nodes) > 2]
    return unassigned

def local_search_relocate(routes):
    global_improved = False
    improved = True
    while improved:
        improved = False
        all_nodes = []
        for r_idx, r in enumerate(routes):
            if len(r.nodes) <= 2: continue
            for i in range(1, len(r.nodes)-1):
                all_nodes.append((r.nodes[i], r_idx))
                
        random.shuffle(all_nodes)
        
        for u, r_idx in all_nodes:
            r = routes[r_idx]
            if u not in r.nodes: continue
            
            idx = r.nodes.index(u)
            prev = r.nodes[idx-1]
            nxt = r.nodes[idx+1]
            saving = dist[prev][u] + dist[u][nxt] - dist[prev][nxt]
            
            r.remove(u)
            
            best_cost = saving - 1e-4
            best_r_idx = -1
            best_ins_idx = -1
            
            for j, tgt_r in enumerate(routes):
                insertions = find_valid_insertions(tgt_r, u)
                for cost, r_ins_idx in insertions:
                    if cost < best_cost:
                        best_cost = cost
                        best_r_idx = j
                        best_ins_idx = r_ins_idx
                        
            new_cost = get_new_route_insertion_cost(u)
            if new_cost < best_cost:
                best_cost = new_cost
                best_r_idx = -2
                best_ins_idx = 1
                
            if best_r_idx != -1:
                if best_r_idx == -2:
                    new_r = Route(len(routes))
                    new_r.insert(1, u)
                    routes.append(new_r)
                else:
                    routes[best_r_idx].insert(best_ins_idx, u)
                improved = True
                global_improved = True
                break 
            else:
                r.insert(idx, u)
                
    routes[:] = [r for r in routes if len(r.nodes) > 2]
    return global_improved

def check_route_feasible(r):
    n = len(r.nodes)
    A = [0.0] * n
    W = [0.0] * n
    D = [0.0] * n
    
    A[0] = ready_time[0]
    W[0] = 0.0
    D[0] = ready_time[0]
    
    for i in range(1, n):
        u = r.nodes[i]
        prev = r.nodes[i-1]
        A[i] = D[i-1] + dist[prev][u]
        if A[i] > due_date[u] + 1e-5:
            return False
        W[i] = max(0.0, ready_time[u] - A[i])
        D[i] = A[i] + W[i] + service_time[u]
        
    r.A = A
    r.W = W
    r.D = D
    
    MaxDelay = [0.0] * n
    MaxDelay[n-1] = max(0.0, due_date[0] - A[n-1])
    for i in range(n-2, -1, -1):
        MaxDelay[i] = max(0.0, min(due_date[r.nodes[i]] - A[i], W[i+1] + MaxDelay[i+1]))
    r.MaxDelay = MaxDelay
    return True

def local_search_exchange(routes):
    global_improved = False
    improved = True
    while improved:
        improved = False
        all_nodes = []
        for r_idx, r in enumerate(routes):
            if len(r.nodes) <= 2: continue
            for i in range(1, len(r.nodes)-1):
                all_nodes.append((r.nodes[i], r_idx))
                
        random.shuffle(all_nodes)
        
        for idx1 in range(len(all_nodes)):
            u, r1_idx = all_nodes[idx1]
            r1 = routes[r1_idx]
            if u not in r1.nodes: continue
            
            for idx2 in range(idx1 + 1, len(all_nodes)):
                v, r2_idx = all_nodes[idx2]
                if r1_idx == r2_idx: continue
                r2 = routes[r2_idx]
                if v not in r2.nodes: continue
                
                pos_u = r1.nodes.index(u)
                pos_v = r2.nodes.index(v)
                
                prev_u = r1.nodes[pos_u-1]
                nxt_u = r1.nodes[pos_u+1]
                prev_v = r2.nodes[pos_v-1]
                nxt_v = r2.nodes[pos_v+1]
                
                orig_cost = dist[prev_u][u] + dist[u][nxt_u] + dist[prev_v][v] + dist[v][nxt_v]
                new_cost = dist[prev_u][v] + dist[v][nxt_u] + dist[prev_v][u] + dist[u][nxt_v]
                
                if new_cost >= orig_cost - 1e-4:
                    continue
                    
                if r1.demand - demand[u] + demand[v] > capacity: continue
                if r2.demand - demand[v] + demand[u] > capacity: continue
                
                r1.nodes[pos_u] = v
                r2.nodes[pos_v] = u
                r1.demand += demand[v] - demand[u]
                r2.demand += demand[u] - demand[v]
                
                if check_route_feasible(r1):
                    if check_route_feasible(r2):
                        improved = True
                        global_improved = True
                        break
                    else:
                        r1.nodes[pos_u] = u
                        r2.nodes[pos_v] = v
                        r1.demand += demand[u] - demand[v]
                        r2.demand += demand[v] - demand[u]
                        r1.update()
                        r2.update()
                else:
                    r1.nodes[pos_u] = u
                    r2.nodes[pos_v] = v
                    r1.demand += demand[u] - demand[v]
                    r2.demand += demand[v] - demand[u]
                    r1.update()
                    
            if improved:
                break
                
    routes[:] = [r for r in routes if len(r.nodes) > 2]
    return global_improved

def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
        
    global capacity, demand, ready_time, due_date, service_time, dist, num_vehicles
    
    capacity = data['vehicle_capacity']
    num_vehicles = data.get('num_vehicles', 1000)
    num_customers = data['num_customers']
    
    N = num_customers + 1
    demand = [0] * N
    ready_time = [0.0] * N
    due_date = [0.0] * N
    service_time = [0.0] * N
    
    coords = []
    
    depot = data['depot']
    coords.append((depot['x'], depot['y']))
    ready_time[0] = depot['ready_time']
    due_date[0] = depot['due_date']
    
    id_map = {depot['id']: 0}
    reverse_map = {0: depot['id']}
    
    customer_list = []
    for i, c in enumerate(data['customers']):
        idx = i + 1
        id_map[c['id']] = idx
        reverse_map[idx] = c['id']
        coords.append((c['x'], c['y']))
        demand[idx] = c['demand']
        ready_time[idx] = c['ready_time']
        due_date[idx] = c['due_date']
        service_time[idx] = c['service_time']
        customer_list.append(idx)
        
    dist = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist[i][j] = math.hypot(dx, dy)
            
    return customer_list, reverse_map

def solve(customers, time_limit, args, reverse_map):
    start_time = time.time()
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    routes = []
    success = repair_regret(routes, set(customers), regret_k=3)
    if not success:
        unassigned = [c for c in customers if not any(c in r.nodes for r in routes)]
        for u in unassigned:
            new_r = Route(len(routes))
            new_r.insert(1, u)
            routes.append(new_r)
            
    routes[:] = [r for r in routes if len(r.nodes) > 2]
    
    local_search_relocate(routes)
    local_search_exchange(routes)
    
    best_dist = sum(r.route_distance() for r in routes)
    best_routes_snapshot = [list(r.nodes) for r in routes]
    if logger: logger.log(best_dist)
        
    current_routes = [Route(i) for i in range(len(routes))]
    for i, r in enumerate(current_routes):
        r.nodes = list(routes[i].nodes)
        r.demand = routes[i].demand
        r.update()
        
    operators = [destroy_random, destroy_worst, destroy_related]
    weights = [1.0, 1.0, 1.0]
    
    margin = 1.0 if time_limit > 2.0 else 0.5
    
    while time.time() - start_time < time_limit - margin:
        backup_routes = [Route(i) for i in range(len(current_routes))]
        for i, r in enumerate(backup_routes):
            r.nodes = list(current_routes[i].nodes)
            r.demand = current_routes[i].demand
            r.update()
            
        total_w = sum(weights)
        probs = [w/total_w for w in weights]
        op_idx = random.choices(range(len(operators)), weights=probs)[0]
        destroy_op = operators[op_idx]
        
        max_len = len(customers)
        max_q = min(max_len, max(2, int(0.4 * max_len)))
        min_q = max(2, int(0.15 * max_len))
        q_to_remove = random.randint(min_q, max_q)
        
        unassigned = destroy_op(current_routes, q_to_remove)
        success = repair_regret(current_routes, unassigned, regret_k=random.choice([1, 2, 3, 4]))
        
        if success:
            local_search_relocate(current_routes)
            local_search_exchange(current_routes)
            
            new_dist = sum(r.route_distance() for r in current_routes)
            delta = new_dist - sum(r.route_distance() for r in backup_routes)
            
            elapsed = time.time() - start_time
            fraction = min(1.0, elapsed / max(0.1, time_limit - margin))
            temp = max(best_dist * 0.001, best_dist * 0.05 * (1.0 - fraction))
            
            accept = False
            if delta < -1e-5:
                accept = True
                weights[op_idx] += 0.5
            else:
                p = math.exp(-delta / max(1e-5, temp))
                if random.random() < p:
                    accept = True
                    weights[op_idx] += 0.1
                    
            if accept:
                if new_dist < best_dist - 1e-5:
                    best_dist = new_dist
                    best_routes_snapshot = [list(r.nodes) for r in current_routes]
                    if logger: logger.log(best_dist)
                    weights[op_idx] += 2.0
            else:
                current_routes = backup_routes
        else:
            current_routes = backup_routes
            weights[op_idx] -= 0.1
            
        weights = [max(0.1, w) for w in weights]

    return best_dist, best_routes_snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    
    customers, reverse_map = load_instance(args.instance_path)
    best_dist, best_routes = solve(customers, args.time_limit, args, reverse_map)
    
    output = {
        "objective_value": best_dist,
        "routes": []
    }
    for r_nodes in best_routes:
        customer_ids = [reverse_map[x] for x in r_nodes[1:-1]]
        path = [reverse_map[x] for x in r_nodes]
        output["routes"].append({
            "customer_ids": customer_ids,
            "path": path
        })
        
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)