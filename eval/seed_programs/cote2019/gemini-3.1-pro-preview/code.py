import argparse
import json
import itertools
import random
import time
import gurobipy as gp
from gurobipy import GRB

ROUTE_EVAL_CACHE = {}

def parse_instance(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    num_customers = data['parameters']['n_customers']
    K = data['vehicle']['K']
    Q = data['vehicle']['Q']
    H = data['vehicle']['H']
    W = data['vehicle']['W']
    recourse_cost = data['parameters']['recourse_cost_cf']
    
    cust_data = {}
    for c in data['customers']:
        cid = c['id']
        items = c['items']
        
        e_area = 0.0
        e_weight = 0.0
        
        for item in items:
            for r in item['realizations']:
                prob = r['probability']
                area = r['width'] * r['height']
                wt = r['weight']
                e_area += prob * area
                e_weight += prob * wt
                
        cust_data[cid] = {
            'expected_area': e_area,
            'expected_weight': e_weight,
            'items': items,
            'is_stochastic': c['is_stochastic']
        }
        
    dist_matrix = data['distance_matrix']
    
    return num_customers, K, Q, H, W, recourse_cost, cust_data, dist_matrix

def pack_items(items, skyline, W, H):
    current_skyline = list(skyline)
    for w, h in items:
        best_y = H + 1
        best_x = -1
        for i in range(len(current_skyline)):
            x = current_skyline[i][0]
            if x + w <= W:
                max_y = 0
                for seg in current_skyline:
                    seg_x1, seg_x2, seg_y = seg
                    if max(x, seg_x1) < min(x + w, seg_x2):
                        if seg_y > max_y: 
                            max_y = seg_y
                if max_y + h <= H and max_y < best_y:
                    best_y = max_y
                    best_x = x
        if best_y + h > H:
            return None
            
        new_skyline = []
        nx1, nx2, ny = best_x, best_x + w, best_y + h
        for sx1, sx2, sy in current_skyline:
            if sx2 <= nx1 or sx1 >= nx2:
                new_skyline.append((sx1, sx2, sy))
            else:
                if sx1 < nx1: 
                    new_skyline.append((sx1, nx1, sy))
                new_skyline.append((max(sx1, nx1), min(sx2, nx2), ny))
                if sx2 > nx2: 
                    new_skyline.append((nx2, sx2, sy))
        
        new_skyline.sort()
        merged = []
        for seg in new_skyline:
            if not merged: 
                merged.append(seg)
            else:
                if merged[-1][2] == seg[2] and merged[-1][1] == seg[0]:
                    merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
                elif seg[0] < seg[1]:
                    merged.append(seg)
        current_skyline = merged
    return current_skyline

def check_pack(sub, scenario_items, W, H):
    skyline1 = [(0, W, 0)]
    skyline2 = [(0, W, 0)]
    valid1 = True
    valid2 = True
    for c in reversed(sub):
        items = scenario_items[c]
        if valid1:
            items1 = sorted(items, key=lambda x: x[0]*x[1], reverse=True)
            skyline1 = pack_items(items1, skyline1, W, H)
            if skyline1 is None: valid1 = False
        if valid2:
            items2 = sorted(items, key=lambda x: max(x[0], x[1]), reverse=True)
            skyline2 = pack_items(items2, skyline2, W, H)
            if skyline2 is None: valid2 = False
        if not valid1 and not valid2:
            return False
    return True

def max_valid_subset_size(route_tuple, scenario_items, scenario_weight, Q, W, H):
    cust_area = {c: sum(w*h for w,h in scenario_items[c]) for c in route_tuple}
    
    tot_w = sum(scenario_weight.values())
    tot_a = sum(cust_area.values())
    if tot_w <= Q and tot_a <= W * H:
        if check_pack(route_tuple, scenario_items, W, H):
            return len(route_tuple)
            
    for size in range(len(route_tuple)-1, -1, -1):
        for sub in itertools.combinations(route_tuple, size):
            swt = sum(scenario_weight[c] for c in sub)
            if swt > Q: continue
            sarea = sum(cust_area[c] for c in sub)
            if sarea > W * H: continue
            
            if check_pack(sub, scenario_items, W, H):
                return size
    return 0

def evaluate_route(route_tuple, cust_data, param_Q, param_W, param_H, recourse_cost):
    stoch_items = []
    det_items = {c: [] for c in route_tuple}
    for c in route_tuple:
        for item in cust_data[c]['items']:
            if item['is_stochastic']:
                stoch_items.append((c, item))
            else:
                det_items[c].append(item['realizations'][0])
                
    scenarios = []
    if not stoch_items:
        scenarios.append(((), 1.0))
    elif len(stoch_items) > 8:
        for _ in range(500):
            combo = []
            for c, item in stoch_items:
                r = random.choices(item['realizations'], weights=[x['probability'] for x in item['realizations']])[0]
                combo.append(r)
            scenarios.append((combo, 1.0/500))
    else:
        pools = [item['realizations'] for c, item in stoch_items]
        for combo in itertools.product(*pools):
            prob = 1.0
            for r in combo: prob *= r['probability']
            scenarios.append((combo, prob))
        
    expected_unserved = 0.0
    any_scenario_full_pack = False
    
    for combo, prob in scenarios:
        scenario_items = {c: [(r['width'], r['height']) for r in det_items[c]] for c in route_tuple}
        scenario_weight = {c: sum(r['weight'] for r in det_items[c]) for c in route_tuple}
        
        for idx, (c, _) in enumerate(stoch_items):
            realization = combo[idx]
            scenario_items[c].append((realization['width'], realization['height']))
            scenario_weight[c] += realization['weight']
            
        m = max_valid_subset_size(route_tuple, scenario_items, scenario_weight, param_Q, param_W, param_H)
        unserved = len(route_tuple) - m
        if unserved == 0:
            any_scenario_full_pack = True
        expected_unserved += prob * unserved
        
    if not any_scenario_full_pack:
        return None
        
    return expected_unserved * recourse_cost

def get_route_cost(route_tuple, cust_data, param_Q, param_W, param_H, recourse_cost, dist_matrix):
    if route_tuple in ROUTE_EVAL_CACHE:
        return ROUTE_EVAL_CACHE[route_tuple]
        
    penalty = evaluate_route(route_tuple, cust_data, param_Q, param_W, param_H, recourse_cost)
    if penalty is None:
        ROUTE_EVAL_CACHE[route_tuple] = None
        return None
        
    rc = 0
    if len(route_tuple) > 0:
        rc += dist_matrix[0][route_tuple[0]]
        for i in range(len(route_tuple)-1):
            rc += dist_matrix[route_tuple[i]][route_tuple[i+1]]
        rc += dist_matrix[route_tuple[-1]][0]
        
    total_cost = rc + penalty
    ROUTE_EVAL_CACHE[route_tuple] = total_cost
    return total_cost

def clarke_wright(cust_list, cust_data, cost_matrix, Q, W, H, recourse):
    routes = {c: [c] for c in cust_list}
    savings = []
    for i in cust_list:
        for j in cust_list:
            if i < j:
                sav = cost_matrix[0][i] + cost_matrix[0][j] - cost_matrix[i][j]
                savings.append((sav, i, j))
    savings.sort(reverse=True)
    
    for sav, i, j in savings:
        r_i = None
        r_j = None
        for r in routes.values():
            if r and r[0] == j: r_j = r
            if r and r[-1] == i: r_i = r
        if r_i and r_j and r_i != r_j:
            cand = tuple(r_i + r_j)
            ea = sum(cust_data[x]['expected_area'] for x in cand)
            ew = sum(cust_data[x]['expected_weight'] for x in cand)
            if ea <= W*H and ew <= Q:
                if get_route_cost(cand, cust_data, Q, W, H, recourse, cost_matrix) is not None:
                    routes[r_i[0]] = list(cand)
                    for c in r_j:
                        routes[c] = routes[r_i[0]]
                    r_j.clear()
                    
    return [tuple(r) for r in routes.values() if r]

def find_initial_solution(cust_list, cust_data, cost_matrix, Q, W, H, recourse, K):
    best_sol = None
    best_cost = float('inf')
    
    sort_keys = [
        lambda c: cust_data[c]['expected_area'],
        lambda c: cust_data[c]['expected_weight'],
        lambda c: random.random()
    ]
    
    for _ in range(50):
        unrouted = list(cust_list)
        random.shuffle(unrouted) 
        if random.random() < 0.5:
            key = random.choice(sort_keys)
            unrouted.sort(key=key, reverse=True)
            
        routes = []
        while unrouted:
            curr_route = [unrouted.pop(0)]
            to_remove = []
            for nc in unrouted:
                cand = tuple(curr_route + [nc])
                ea = sum(cust_data[x]['expected_area'] for x in cand)
                ew = sum(cust_data[x]['expected_weight'] for x in cand)
                if ea <= W*H and ew <= Q:
                    if get_route_cost(cand, cust_data, Q, W, H, recourse, cost_matrix) is not None:
                        curr_route.append(nc)
                        to_remove.append(nc)
            for x in to_remove:
                unrouted.remove(x)
            routes.append(tuple(curr_route))
            
        if len(routes) <= K:
            cost = sum(get_route_cost(r, cust_data, Q, W, H, recourse, cost_matrix) for r in routes)
            if cost < best_cost:
                best_cost = cost
                best_sol = routes
                
    return best_sol, best_cost

def optimize_solution(current_sol):
    new_routes = set()
    for r in current_sol:
        for i in range(len(r)):
            for j in range(i+2, len(r)+1):
                new_r = r[:i] + tuple(reversed(r[i:j])) + r[j:]
                new_routes.add(new_r)
                
    for i, r1 in enumerate(current_sol):
        for j, r2 in enumerate(current_sol):
            if i == j: continue
            for c_idx in range(len(r1)):
                c = r1[c_idx]
                rem_r1 = r1[:c_idx] + r1[c_idx+1:]
                new_routes.add(rem_r1)
                for ins_idx in range(len(r2)+1):
                    new_r2 = r2[:ins_idx] + (c,) + r2[ins_idx:]
                    new_routes.add(new_r2)
                    
    for i, r1 in enumerate(current_sol):
        for j, r2 in enumerate(current_sol):
            if i >= j: continue
            for c1_idx in range(len(r1)):
                for c2_idx in range(len(r2)):
                    c1 = r1[c1_idx]
                    c2 = r2[c2_idx]
                    new_r1 = r1[:c1_idx] + (c2,) + r1[c1_idx+1:]
                    new_r2 = r2[:c2_idx] + (c1,) + r2[c2_idx+1:]
                    new_routes.add(new_r1)
                    new_routes.add(new_r2)
                    
    return list(new_routes)

def solve_ilp(valid_routes, num_customers, K, time_limit):
    if time_limit <= 0:
        return None, None
        
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)  # Adhere strictly to single-core environment requirement
    env.start()
    m = gp.Model("set_partitioning", env=env)
    m.setParam('TimeLimit', time_limit)
        
    x = []
    for r, cost in valid_routes:
        x.append(m.addVar(vtype=GRB.BINARY, obj=cost))
        
    x_empty = m.addVar(vtype=GRB.INTEGER, obj=0, name="empty_routes")
        
    for i in range(1, num_customers + 1):
        m.addConstr(gp.quicksum(x[j] for j, (r, c) in enumerate(valid_routes) if i in r) == 1)
        
    m.addConstr(gp.quicksum(x) + x_empty == K)
    m.optimize()
    
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        sol_routes = []
        for j, (r, c) in enumerate(valid_routes):
            if x[j].X > 0.5:
                sol_routes.append(tuple(r))
        while len(sol_routes) < K:
            sol_routes.append(())
        return m.ObjVal, sol_routes
    else:
        return None, None

def write_solution(sol_routes, best_cost, sol_path):
    out = {
        "objective_value": best_cost,
        "routes": sol_routes
    }
    with open(sol_path, 'w') as f:
        json.dump(out, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass
            
    num_customers, K, Q, H, W, recourse_cost, cust_data, dist_matrix = parse_instance(args.instance_path)
    cust_list = list(range(1, num_customers + 1))
    
    pool = set()
    for c in cust_list:
        pool.add((c,))
        get_route_cost((c,), cust_data, Q, W, H, recourse_cost, dist_matrix)
        
    cw_sol = clarke_wright(cust_list, cust_data, dist_matrix, Q, W, H, recourse_cost)
    for r in cw_sol: 
        pool.add(r)
        
    global_best_cost = float('inf')
    global_best_sol = None
    
    h_sol, h_cost = find_initial_solution(cust_list, cust_data, dist_matrix, Q, W, H, recourse_cost, K)
    if h_sol and h_cost < global_best_cost:
        global_best_cost = h_cost
        global_best_sol = list(h_sol)
        while len(global_best_sol) < K: global_best_sol.append(())
        for r in h_sol: pool.add(r)
        if logger: logger.log(global_best_cost)
        
    def run_gurobi_solver():
        nonlocal global_best_cost, global_best_sol
        valid_routes = [(r, ROUTE_EVAL_CACHE[r]) for r in pool if ROUTE_EVAL_CACHE.get(r) is not None]
        t_left = args.time_limit - (time.time() - start_time) - 1
        obj, sol = solve_ilp(valid_routes, num_customers, K, t_left)
        if sol and obj < global_best_cost - 1e-4:
            global_best_cost = obj
            global_best_sol = sol
            if logger: logger.log(global_best_cost)

    run_gurobi_solver()
    current_sol = [r for r in global_best_sol if r] if global_best_sol else cw_sol

    while time.time() - start_time < args.time_limit - 5:
        neighbors = optimize_solution(current_sol)
        random.shuffle(neighbors)
        
        added_to_pool = 0
        for cand in neighbors:
            if time.time() - start_time > args.time_limit - 5: break
            if cand not in ROUTE_EVAL_CACHE:
                ea = sum(cust_data[x]['expected_area'] for x in cand)
                ew = sum(cust_data[x]['expected_weight'] for x in cand)
                if ea <= W*H and ew <= Q:
                    cost = get_route_cost(cand, cust_data, Q, W, H, recourse_cost, dist_matrix)
                    if cost is not None:
                        pool.add(cand)
                        added_to_pool += 1
                        
            if added_to_pool >= 200:
                break
                
        run_gurobi_solver()
        
        if global_best_sol:
            current_sol = [r for r in global_best_sol if r]
            if len(current_sol) > 0:
                idx = random.randint(0, len(current_sol)-1)
                r_split = current_sol.pop(idx)
                if len(r_split) > 1:
                    split_pt = random.randint(1, len(r_split)-1)
                    r1 = r_split[:split_pt]
                    r2 = r_split[split_pt:]
                    current_sol.append(r1)
                    current_sol.append(r2)
                    pool.add(r1)
                    pool.add(r2)
        else:
            run_gurobi_solver()

    run_gurobi_solver()

    out_routes = []
    if global_best_sol:
        for r in global_best_sol:
            out_routes.append(list(r))
    else:
        out_routes = [[] for _ in range(K)]
        global_best_cost = 0.0

    write_solution(out_routes, global_best_cost, args.solution_path)

if __name__ == "__main__":
    main()