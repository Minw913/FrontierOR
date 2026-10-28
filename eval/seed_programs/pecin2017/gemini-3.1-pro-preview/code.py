import argparse
import json
import math
import sys
import time
import random

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class VRPTWInstance:
    def __init__(self, data):
        self.n_nodes = data['problem_parameters']['n_nodes']
        self.vehicle_capacity = data['problem_parameters']['vehicle_capacity']
        self.planning_horizon = data['problem_parameters']['planning_horizon']
        
        nodes_info = data['nodes']
        nodes_info.sort(key=lambda n: n['id'])
        
        self.id_to_idx = {n['id']: i for i, n in enumerate(nodes_info)}
        self.idx_to_id = {i: n['id'] for i, n in enumerate(nodes_info)}
        
        self.tw_open = [0.0] * self.n_nodes
        self.tw_close = [0.0] * self.n_nodes
        self.service_time = [0.0] * self.n_nodes
        self.demands = [0] * self.n_nodes
        self.xs = [0.0] * self.n_nodes
        self.ys = [0.0] * self.n_nodes
        
        self.source_id = -1
        self.sink_id = -1
        self.all_customers = []
        
        for n in nodes_info:
            idx = self.id_to_idx[n['id']]
            self.tw_open[idx] = float(n['time_window_open'])
            self.tw_close[idx] = float(n['time_window_close'])
            self.service_time[idx] = float(n['service_time'])
            self.demands[idx] = int(n['demand'])
            self.xs[idx] = float(n['x'])
            self.ys[idx] = float(n['y'])
            
            if n['type'] == 'depot_source':
                self.source_id = idx
            elif n['type'] == 'depot_sink':
                self.sink_id = idx
            elif n['type'] == 'customer':
                self.all_customers.append(idx)
                
        self.dist = [[0.0] * self.n_nodes for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.dist[i][j] = math.hypot(self.xs[i] - self.xs[j], self.ys[i] - self.ys[j])

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default="")
    args = parser.parse_args()

    start_time = time.time()
    
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    inst = VRPTWInstance(data)
    
    if not inst.all_customers:
        best_cost = 0.0
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": best_cost, "routes": []}, f, indent=2)
        sys.exit(0)

    class Route:
        def __init__(self, customers):
            self.customers = list(customers)
            self.E = []
            self.L = []
            self.cost = 0.0
            self.demand = 0
            self.update()

        def update(self):
            self.demand = sum(inst.demands[c] for c in self.customers)
            if not self.customers:
                self.E = [inst.tw_open[inst.source_id], inst.tw_open[inst.sink_id]]
                self.L = [inst.tw_close[inst.source_id], inst.tw_close[inst.sink_id]]
                self.cost = 0.0
                return

            route = [inst.source_id] + self.customers + [inst.sink_id]
            n = len(route)
            
            self.E = [0.0] * n
            self.E[0] = inst.tw_open[inst.source_id]
            cst = 0.0
            for i in range(1, n):
                u, v = route[i-1], route[i]
                self.E[i] = max(inst.tw_open[v], self.E[i-1] + inst.service_time[u] + inst.dist[u][v])
                cst += inst.dist[u][v]
            self.cost = cst
            
            self.L = [0.0] * n
            self.L[-1] = inst.tw_close[inst.sink_id]
            for i in range(n-2, -1, -1):
                u, v = route[i], route[i+1]
                self.L[i] = min(inst.tw_close[u], self.L[i+1] - inst.service_time[u] - inst.dist[u][v])

        def eval_insertion(self, c, pos):
            if not self.customers:
                if self.demand + inst.demands[c] > inst.vehicle_capacity:
                    return float('inf')
                start_c = max(inst.tw_open[c], inst.tw_open[inst.source_id] + inst.service_time[inst.source_id] + inst.dist[inst.source_id][c])
                if start_c > inst.tw_close[c]:
                    return float('inf')
                start_sink = max(inst.tw_open[inst.sink_id], start_c + inst.service_time[c] + inst.dist[c][inst.sink_id])
                if start_sink > inst.tw_close[inst.sink_id]:
                    return float('inf')
                return inst.dist[inst.source_id][c] + inst.dist[c][inst.sink_id]

            prev_node = inst.source_id if pos == 0 else self.customers[pos-1]
            next_node = inst.sink_id if pos == len(self.customers) else self.customers[pos]
            
            if self.demand + inst.demands[c] > inst.vehicle_capacity:
                return float('inf')
                
            E_prev = self.E[pos]
            E_c = max(inst.tw_open[c], E_prev + inst.service_time[prev_node] + inst.dist[prev_node][c])
            if E_c > inst.tw_close[c]:
                return float('inf')
                
            E_next = max(inst.tw_open[next_node], E_c + inst.service_time[c] + inst.dist[c][next_node])
            if E_next > self.L[pos+1]:
                return float('inf')
                
            return inst.dist[prev_node][c] + inst.dist[c][next_node] - inst.dist[prev_node][next_node]

        def clone(self):
            new_r = Route.__new__(Route)
            new_r.customers = list(self.customers)
            new_r.demand = self.demand
            new_r.cost = self.cost
            new_r.E = list(self.E)
            new_r.L = list(self.L)
            return new_r

    class Solution:
        def __init__(self, routes=None, unrouted=None):
            self.routes = [r.clone() for r in routes] if routes else []
            self.unrouted = list(unrouted) if unrouted else []
            self.cost = sum(r.cost for r in self.routes)
            
        def clone(self):
            return Solution(self.routes, self.unrouted)
        
        def cleanup(self):
            self.routes = [r for r in self.routes if r.customers]
            self.cost = sum(r.cost for r in self.routes)

    def repair(solution, unrouted, method='greedy'):
        if not any(len(r.customers) == 0 for r in solution.routes):
            solution.routes.append(Route([]))
            
        insertions = {c: {} for c in unrouted}
        for c in unrouted:
            for r_idx, r in enumerate(solution.routes):
                best_cost = float('inf')
                best_pos = -1
                for pos in range(len(r.customers) + 1):
                    cost = r.eval_insertion(c, pos)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                insertions[c][r_idx] = (best_cost, best_pos)
                
        while unrouted:
            best_c = None
            best_r_idx = -1
            best_pos = -1
            best_overall_cost = float('inf')
            best_score = -float('inf')
            
            for c in unrouted:
                opts = sorted(insertions[c].items(), key=lambda x: x[1][0])
                cst1, pos1 = opts[0][1]
                r1 = opts[0][0]
                
                if method == 'greedy':
                    score = -cst1
                elif method == 'regret2':
                    cst2 = opts[1][1][0] if len(opts) > 1 else float('inf')
                    score = cst2 - cst1
                elif method == 'regret3':
                    cst2 = opts[1][1][0] if len(opts) > 1 else float('inf')
                    cst3 = opts[2][1][0] if len(opts) > 2 else float('inf')
                    score = cst2 + cst3 - 2 * cst1
                elif method == 'regret4':
                    cst2 = opts[1][1][0] if len(opts) > 1 else float('inf')
                    cst3 = opts[2][1][0] if len(opts) > 2 else float('inf')
                    cst4 = opts[3][1][0] if len(opts) > 3 else float('inf')
                    score = cst2 + cst3 + cst4 - 3 * cst1
                    
                if score > best_score or (score == best_score and cst1 < best_overall_cost):
                    best_score = score
                    best_overall_cost = cst1
                    best_c = c
                    best_r_idx = r1
                    best_pos = pos1
                    
            if best_c is None or best_overall_cost == float('inf'):
                return False
                
            route_mod = solution.routes[best_r_idx]
            route_mod.customers.insert(best_pos, best_c)
            route_mod.update()
            unrouted.remove(best_c)
            del insertions[best_c]
            
            if len(route_mod.customers) == 1:
                new_r = Route([])
                solution.routes.append(new_r)
                new_r_idx = len(solution.routes) - 1
                for c in unrouted:
                    cst = new_r.eval_insertion(c, 0)
                    insertions[c][new_r_idx] = (cst, 0)
                    
            for c in unrouted:
                best_cost = float('inf')
                best_pos = -1
                for pos in range(len(route_mod.customers) + 1):
                    cost = route_mod.eval_insertion(c, pos)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                insertions[c][best_r_idx] = (best_cost, best_pos)

        solution.cleanup()
        return True

    def remove_customers(solution, customers_to_remove):
        remove_set = set(customers_to_remove)
        solution.unrouted.extend(customers_to_remove)
        for r in solution.routes:
            if any(c in remove_set for c in r.customers):
                r.customers = [c for c in r.customers if c not in remove_set]
                r.update()
        solution.cleanup()

    def random_destroy(solution, q):
        c_list = [c for r in solution.routes for c in r.customers]
        return random.sample(c_list, q) if q <= len(c_list) else list(c_list)

    def worst_destroy(solution, q):
        savings = []
        for r in solution.routes:
            if len(r.customers) == 1:
                savings.append((r.customers[0], r.cost))
            else:
                for i, c in enumerate(r.customers):
                    prev_node = inst.source_id if i == 0 else r.customers[i-1]
                    next_node = inst.sink_id if i == len(r.customers) - 1 else r.customers[i+1]
                    save = inst.dist[prev_node][c] + inst.dist[c][next_node] - inst.dist[prev_node][next_node]
                    savings.append((c, save))
        savings.sort(key=lambda x: x[1], reverse=True)
        removed = []
        while len(removed) < q and savings:
            idx = int(len(savings) * (random.random() ** 3))
            removed.append(savings.pop(idx)[0])
        return removed

    def shaw_destroy(solution, q):
        c_list = [c for r in solution.routes for c in r.customers]
        if not c_list: return []
        seed = random.choice(c_list)
        rel = []
        for c in c_list:
            if c == seed: continue
            score = inst.dist[seed][c] + 0.1 * abs(inst.tw_open[seed] - inst.tw_open[c]) + 0.1 * abs(inst.demands[seed] - inst.demands[c])
            rel.append((c, score))
        rel.sort(key=lambda x: x[1])
        removed = [seed]
        while len(removed) < q and rel:
            idx = int(len(rel) * (random.random() ** 3))
            removed.append(rel.pop(idx)[0])
        return removed

    # Initial solution
    sol = Solution([], inst.all_customers)
    success = repair(sol, sol.unrouted, method='regret2')
    if not success or sol.unrouted:
        sol = Solution([Route([c]) for c in inst.all_customers], [])

    best_solution = sol.clone()
    best_cost = sol.cost
    if logger:
        logger.log(best_cost)

    route_pool = {}
    for r in sol.routes:
        if r.customers:
            route_pool[tuple(r.customers)] = r.cost

    if best_cost <= 0:
        T_start = 1.0
    else:
        T_start = max(0.05 * best_cost / 0.693, 1.0)
    T_end = 0.01

    w_destroy = [1.0, 1.0, 1.0]
    w_repair = [1.0, 1.0, 1.0, 1.0]

    alns_end_time = start_time + (args.time_limit - 5.0 if args.time_limit > 10 else args.time_limit * 0.5)

    while time.time() < alns_end_time:
        max_duration = alns_end_time - start_time
        if max_duration <= 0: break
        elapsed_ratio = (time.time() - start_time) / max_duration
        T = T_start * ((T_end / T_start) ** elapsed_ratio)
        
        sum_wd = sum(w_destroy)
        pd = [w / sum_wd for w in w_destroy]
        destroy_idx = random.choices([0, 1, 2], weights=pd)[0]
        
        sum_wr = sum(w_repair)
        pr = [w / sum_wr for w in w_repair]
        repair_idx = random.choices([0, 1, 2, 3], weights=pr)[0]
        
        q_max = max(4, int(0.2 * len(inst.all_customers)))
        q_min = min(2, q_max)
        q = random.randint(q_min, q_max)
        
        curr_sol = sol.clone()
        
        if destroy_idx == 0:
            rem = random_destroy(curr_sol, q)
        elif destroy_idx == 1:
            rem = worst_destroy(curr_sol, q)
        else:
            rem = shaw_destroy(curr_sol, q)
            
        remove_customers(curr_sol, rem)
        
        rep_methods = ['greedy', 'regret2', 'regret3', 'regret4']
        success = repair(curr_sol, curr_sol.unrouted, method=rep_methods[repair_idx])
        
        score = 0.0
        if success:
            delta = curr_sol.cost - sol.cost
            accepted = False
            if delta < -1e-6:
                sol = curr_sol
                score = 1.2
                accepted = True
                if sol.cost < best_cost - 1e-6:
                    best_solution = sol.clone()
                    best_cost = sol.cost
                    score = 1.5
                    if logger: logger.log(best_cost)
            else:
                prob = math.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    sol = curr_sol
                    score = 0.8
                    accepted = True
                else:
                    score = 0.0

            if accepted:
                for r in curr_sol.routes:
                    if r.customers:
                        route_pool[tuple(r.customers)] = r.cost
        
        w_destroy[destroy_idx] = 0.8 * w_destroy[destroy_idx] + 0.2 * score
        w_repair[repair_idx] = 0.8 * w_repair[repair_idx] + 0.2 * score

    # Set Partitioning Step
    def run_set_partitioning(best_sol, best_cst, pool, time_left):
        if time_left <= 0:
            return best_sol, best_cst
        try:
            import gurobipy as gp
            from gurobipy import GRB
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model("SetPartitioning", env=env)
            model.setParam("Threads", 1)
            
            unique_routes = list(pool.keys())
            route_costs = [pool[r] for r in unique_routes]
            best_route_tuples = set(tuple(r.customers) for r in best_sol.routes if r.customers)
            
            x = model.addVars(len(unique_routes), vtype=GRB.BINARY, obj=route_costs)
            for i, r in enumerate(unique_routes):
                if r in best_route_tuples:
                    x[i].Start = 1
                else:
                    x[i].Start = 0
                    
            for c in inst.all_customers:
                model.addConstr(gp.quicksum(x[i] for i, r in enumerate(unique_routes) if c in r) == 1)
                
            model.setParam('TimeLimit', max(1.0, time_left))
            model.optimize()
            
            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and model.SolCount > 0:
                if model.ObjVal < best_cst - 1e-6:
                    best_cst = model.ObjVal
                    best_routes = []
                    for i, r in enumerate(unique_routes):
                        if x[i].X > 0.5:
                            best_routes.append(list(r))
                    if logger: logger.log(best_cst)
                    return Solution([Route(r) for r in best_routes]), best_cst
        except Exception:
            pass
        return best_sol, best_cst

    sp_time_left = args.time_limit - (time.time() - start_time)
    best_solution, best_cost = run_set_partitioning(best_solution, best_cost, route_pool, sp_time_left)

    output = {
        "objective_value": best_cost,
        "routes": []
    }
    for r in best_solution.routes:
        if not r.customers: continue
        full_route = [inst.source_id] + r.customers + [inst.sink_id]
        original_id_route = [inst.idx_to_id[idx] for idx in full_route]
        output["routes"].append(original_id_route)
        
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    solve()