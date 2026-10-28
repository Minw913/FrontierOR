import argparse
import json
import time
import math
import random

def main():
    parser = argparse.ArgumentParser(description="Solve VRPSDP with separate deliveries and pickups")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    capacity = data['vehicle_capacity']
    num_customers = data['num_customers']
    
    # 0 = depot
    # 1..n = deliveries (linehaul)
    # n+1..2n = pickups (backhaul)
    node_info = [(0, 0, 0)] * (2 * num_customers + 1)
    for cust in data['customers']:
        c_id = cust['id']
        d = cust['delivery_demand']
        p = cust['pickup_demand']
        node_info[c_id] = (d, 0, d)
        node_info[c_id + num_customers] = (0, p, p)
        
    num_active_nodes = sum(1 for i in range(1, 2 * num_customers + 1) if node_info[i] != (0, 0, 0))
    if num_active_nodes == 0:
        sol = {
            "objective_value": 0.0,
            "routes": [],
            "routes_detailed": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return
        
    dist_mat = [[0] * (2 * num_customers + 1) for _ in range(2 * num_customers + 1)]
    dist_in = data['distance_matrix']
    for u in range(2 * num_customers + 1):
        for v in range(2 * num_customers + 1):
            cu = u if u <= num_customers else u - num_customers
            cv = v if v <= num_customers else v - num_customers
            if u == 0: cu = 0
            if v == 0: cv = 0
            dist_mat[u][v] = dist_in[cu][cv]

    class Route:
        def __init__(self, nodes):
            self.nodes = list(nodes)
            self.update()

        def update(self):
            # Maintains prefix and suffix associative logic for O(1) capacity check
            self.pref = [(0, 0, 0)] * (len(self.nodes) + 1)
            D, P, M = 0, 0, 0
            for i, node in enumerate(self.nodes):
                d_n, p_n, m_n = node_info[node]
                M = M + d_n if M + d_n > m_n + P else m_n + P
                D += d_n
                P += p_n
                self.pref[i+1] = (D, P, M)
            
            self.suff = [(0, 0, 0)] * (len(self.nodes) + 1)
            D, P, M = 0, 0, 0
            for i in range(len(self.nodes)-1, -1, -1):
                node = self.nodes[i]
                d_n, p_n, m_n = node_info[node]
                M = m_n + D if m_n + D > M + p_n else M + p_n
                D += d_n
                P += p_n
                self.suff[i] = (D, P, M)
                
            dist = 0
            prev = 0
            for node in self.nodes:
                dist += dist_mat[prev][node]
                prev = node
            dist += dist_mat[prev][0]
            self.distance = dist

        def try_insert(self, i, D_B, P_B, M_B):
            D_A, P_A, M_A = self.pref[i]
            D_C, P_C, M_C = self.suff[i]
            return max(M_A + D_B + D_C, M_B + P_A + D_C, M_C + P_A + P_B)

        def try_remove(self, i):
            D_A, P_A, M_A = self.pref[i]
            D_C, P_C, M_C = self.suff[i+1]
            return max(M_A + D_C, M_C + P_A)

        def remove_dist_diff(self, i):
            prev_node = 0 if i == 0 else self.nodes[i-1]
            nxt_node = 0 if i == len(self.nodes)-1 else self.nodes[i+1]
            curr = self.nodes[i]
            return dist_mat[prev_node][nxt_node] - dist_mat[prev_node][curr] - dist_mat[curr][nxt_node]

        def insert_dist_diff(self, i, B_node):
            prev_node = 0 if i == 0 else self.nodes[i-1]
            nxt_node = 0 if i == len(self.nodes) else self.nodes[i]
            return dist_mat[prev_node][B_node] + dist_mat[B_node][nxt_node] - dist_mat[prev_node][nxt_node]
            
    def initial_solution():
        routes = []
        for i in range(1, 2 * num_customers + 1):
            if node_info[i] != (0, 0, 0):
                routes.append(Route([i]))
        
        savings = []
        for r1_idx, r1 in enumerate(routes):
            for r2_idx, r2 in enumerate(routes):
                if r1_idx == r2_idx: continue
                u = r1.nodes[-1]
                v = r2.nodes[0]
                s = dist_mat[u][0] + dist_mat[0][v] - dist_mat[u][v]
                if s > 0:
                    savings.append((s, u, v, r1_idx, r2_idx))
        savings.sort(key=lambda x: x[0], reverse=True)
        
        node_to_route = {r.nodes[0]: r for r in routes}
        active_routes = set(routes)
        
        for s, u, v, _, _ in savings:
            r_u = node_to_route[u]
            r_v = node_to_route[v]
            if r_u == r_v: continue
            if r_u.nodes[-1] != u or r_v.nodes[0] != v: continue
            
            D1, P1, M1 = r_u.pref[-1]
            D2, P2, M2 = r_v.pref[-1]
            if max(M1 + D2, M2 + P1) <= capacity:
                new_nodes = r_u.nodes + r_v.nodes
                new_r = Route(new_nodes)
                active_routes.remove(r_u)
                active_routes.remove(r_v)
                active_routes.add(new_r)
                for n in new_nodes:
                    node_to_route[n] = new_r
        return list(active_routes)

    def destroy_random(routes, q):
        all_nodes = [n for r in routes for n in r.nodes]
        q = min(q, len(all_nodes))
        removed = random.sample(all_nodes, q)
        removed_set = set(removed)
        return [Route([n for n in r.nodes if n not in removed_set]) for r in routes if [n for n in r.nodes if n not in removed_set]], removed

    def destroy_worst(routes, q):
        savings = [(r.remove_dist_diff(i), r.nodes[i]) for r in routes for i in range(len(r.nodes))]
        savings.sort(key=lambda x: x[0], reverse=True)
        removed, removed_set = [], set()
        while len(removed) < q and savings:
            idx = int(random.random() ** 3 * len(savings))
            _, node = savings.pop(idx)
            removed.append(node)
            removed_set.add(node)
        return [Route([n for n in r.nodes if n not in removed_set]) for r in routes if [n for n in r.nodes if n not in removed_set]], removed

    def repair_greedy(routes, unassigned):
        random.shuffle(unassigned)
        for node in unassigned:
            best_cost, best_r, best_i = float('inf'), -1, -1
            d_n, p_n, m_n = node_info[node]
            for r_idx, r in enumerate(routes):
                for i in range(len(r.nodes) + 1):
                    if r.try_insert(i, d_n, p_n, m_n) <= capacity:
                        c = r.insert_dist_diff(i, node)
                        if c < best_cost:
                            best_cost, best_r, best_i = c, r_idx, i
            nc = dist_mat[0][node] + dist_mat[node][0]
            if nc < best_cost and m_n <= capacity:
                best_cost, best_r = nc, -1
            if best_r != -1:
                routes[best_r].nodes.insert(best_i, node)
                routes[best_r].update()
            else:
                routes.append(Route([node]))

    def local_search(routes):
        improved = True
        while improved:
            routes[:] = [r for r in routes if r.nodes]
            improved = False
            
            # 1. Relocate moves (Cross route & Intra route)
            for r1_idx, r1 in enumerate(routes):
                for i in range(len(r1.nodes)):
                    node = r1.nodes[i]
                    d_n, p_n, m_n = node_info[node]
                    
                    best_r, best_j, best_diff = r1_idx, i, 0.0
                    rem_diff = r1.remove_dist_diff(i)
                    
                    for r2_idx, r2 in enumerate(routes):
                        if r1_idx == r2_idx: continue
                        for j in range(len(r2.nodes) + 1):
                            if r2.try_insert(j, d_n, p_n, m_n) <= capacity:
                                ins_diff = r2.insert_dist_diff(j, node)
                                diff = rem_diff + ins_diff
                                if diff < best_diff - 1e-5:
                                    best_diff, best_r, best_j = diff, r2_idx, j
                                    
                    if best_diff < -1e-5 and best_r != r1_idx:
                        r1.nodes.pop(i)
                        routes[best_r].nodes.insert(best_j, node)
                        r1.update()
                        routes[best_r].update()
                        improved = True
                        break
                if improved: break
            if improved: continue

            # 2. Swap moves (Cross route)
            for r1_idx in range(len(routes)):
                r1 = routes[r1_idx]
                break_all = False
                for i in range(len(r1.nodes)):
                    for r2_idx in range(r1_idx + 1, len(routes)):
                        r2 = routes[r2_idx]
                        for j in range(len(r2.nodes)):
                            ni, nj = r1.nodes[i], r2.nodes[j]
                            d_j, p_j, m_j = node_info[nj]
                            d_i, p_i, m_i = node_info[ni]
                            
                            D_A, P_A, M_A = r1.pref[i]
                            D_C, P_C, M_C = r1.suff[i+1]
                            if max(M_A + d_j + D_C, m_j + P_A + D_C, M_C + P_A + p_j) > capacity: continue
                            
                            D_A2, P_A2, M_A2 = r2.pref[j]
                            D_C2, P_C2, M_C2 = r2.suff[j+1]
                            if max(M_A2 + d_i + D_C2, m_i + P_A2 + D_C2, M_C2 + P_A2 + p_i) > capacity: continue
                            
                            prev_i, nxt_i = (0 if i == 0 else r1.nodes[i-1]), (0 if i == len(r1.nodes)-1 else r1.nodes[i+1])
                            prev_j, nxt_j = (0 if j == 0 else r2.nodes[j-1]), (0 if j == len(r2.nodes)-1 else r2.nodes[j+1])
                            
                            rem_diff = (dist_mat[prev_i][ni] + dist_mat[ni][nxt_i]) + (dist_mat[prev_j][nj] + dist_mat[nj][nxt_j])
                            ins_diff = (dist_mat[prev_i][nj] + dist_mat[nj][nxt_i]) + (dist_mat[prev_j][ni] + dist_mat[ni][nxt_j])
                            
                            if (ins_diff - rem_diff) < -1e-5:
                                r1.nodes[i], r2.nodes[j] = nj, ni
                                r1.update(); r2.update()
                                improved = break_all = True
                                break
                        if break_all: break
                    if break_all: break
                if break_all: break
            if improved: continue

            # 3. 2-Opt* (Tail swap)
            for r1_idx in range(len(routes)):
                r1 = routes[r1_idx]
                if len(r1.nodes) < 2: continue
                break_all = False
                for r2_idx in range(r1_idx + 1, len(routes)):
                    r2 = routes[r2_idx]
                    if len(r2.nodes) < 2: continue
                    for i in range(len(r1.nodes) - 1):
                        for j in range(len(r2.nodes) - 1):
                            D_A, P_A, M_A = r1.pref[i+1]
                            D_C2, P_C2, M_C2 = r2.suff[j+1]
                            if max(M_A + D_C2, M_C2 + P_A) > capacity: continue
                            
                            D_A2, P_A2, M_A2 = r2.pref[j+1]
                            D_C, P_C, M_C = r1.suff[i+1]
                            if max(M_A2 + D_C, M_C + P_A2) > capacity: continue
                            
                            n1_i, n1_nxt = r1.nodes[i], r1.nodes[i+1]
                            n2_j, n2_nxt = r2.nodes[j], r2.nodes[j+1]
                            
                            if (dist_mat[n1_i][n2_nxt] + dist_mat[n2_j][n1_nxt] - dist_mat[n1_i][n1_nxt] - dist_mat[n2_j][n2_nxt]) < -1e-5:
                                new_r1_nodes = r1.nodes[:i+1] + r2.nodes[j+1:]
                                new_r2_nodes = r2.nodes[:j+1] + r1.nodes[i+1:]
                                r1.nodes, r2.nodes = new_r1_nodes, new_r2_nodes
                                r1.update(); r2.update()
                                improved = break_all = True
                                break
                        if break_all: break
                if break_all: break
            
            # 4. Intra-route 2-Opt
            if not improved:
                for r in routes:
                    if len(r.nodes) < 3: continue
                    for i in range(len(r.nodes)-1):
                        for j in range(i+1, len(r.nodes)):
                            new_nodes = r.nodes[:i] + r.nodes[i:j+1][::-1] + r.nodes[j+1:]
                            D, P, M, valid = 0, 0, 0, True
                            for n in new_nodes:
                                d, p, m = node_info[n]
                                M = max(M + d, m + P)
                                if M > capacity:
                                    valid = False; break
                                D, P = D + d, P + p
                            if valid:
                                prv, nxt = (0 if i==0 else r.nodes[i-1]), (0 if j==len(r.nodes)-1 else r.nodes[j+1])
                                if (dist_mat[prv][r.nodes[j]] + dist_mat[r.nodes[i]][nxt] - dist_mat[prv][r.nodes[i]] - dist_mat[r.nodes[j]][nxt]) < -1e-5:
                                    r.nodes = new_nodes
                                    r.update()
                                    improved = True
                                    break
                        if improved: break
                    if improved: break

    # ALNS Start
    current_sol = initial_solution()
    local_search(current_sol)
    current_cost = sum(r.distance for r in current_sol)
    best_sol = [Route(r.nodes) for r in current_sol]
    best_cost = current_cost

    if logger:
        logger.log(float(best_cost))

    max_time = max(0.1, args.time_limit - 1.0)
    T_start = max(1.0, current_cost * 0.1)
    T_end = 0.01

    while time.time() - start_time < max_time:
        frac = (time.time() - start_time) / max_time
        T = T_start * (T_end / T_start) ** frac if frac < 1.0 else T_end
        
        q = max(1, int(random.uniform(0.15, 0.35) * num_active_nodes))
        destroy_op = random.choice([destroy_random, destroy_worst] * 2 + [destroy_random]) 
        new_routes, unassigned = destroy_op(current_sol, q)
        
        repair_greedy(new_routes, unassigned)
        local_search(new_routes)
        
        new_cost = sum(r.distance for r in new_routes)
        if new_cost < current_cost:
            current_sol = [Route(r.nodes) for r in new_routes]
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_sol = [Route(r.nodes) for r in current_sol]
                if logger:
                    logger.log(float(best_cost))
        else:
            if random.random() < math.exp((current_cost - new_cost) / T):
                current_sol = [Route(r.nodes) for r in new_routes]
                current_cost = new_cost

    out_routes = []
    out_details = []
    for r in best_sol:
        if not r.nodes: continue
        out_routes.append([0] + r.nodes + [0])
        
        rd = [{"role": "depot", "customer_id": 0, "delivery_quantity": 0, "pickup_quantity": 0}]
        for node in r.nodes:
            if node <= num_customers:
                rd.append({
                    "role": "linehaul",
                    "customer_id": node,
                    "delivery_quantity": node_info[node][0],
                    "pickup_quantity": 0
                })
            else:
                rd.append({
                    "role": "backhaul",
                    "customer_id": node - num_customers,
                    "delivery_quantity": 0,
                    "pickup_quantity": node_info[node][1]
                })
        rd.append({"role": "depot", "customer_id": 0, "delivery_quantity": 0, "pickup_quantity": 0})
        out_details.append(rd)
        
    final_solution = {
        "objective_value": float(best_cost),
        "routes": out_routes,
        "routes_detailed": out_details
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()