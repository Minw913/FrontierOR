import argparse
import json
import time
import math
import random
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_solution(path, sol):
    with open(path, 'w') as f:
        json.dump(sol, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    instance = load_instance(args.instance_path)
    params = instance["parameters"]
    
    N = instance["num_sites"]
    num_customers = instance["num_customers"]
    T = params["T"]
    delta = params["delta"]
    epsilon = params["epsilon"]
    L = params["L"]
    c = params["c"]
    q = params["q"]
    
    # Process sites and customers array structures
    sites_data = instance["sites"]
    customers_data = instance["customers"]
    
    # Dictionaries and mappings
    fixed_cost = np.zeros(N + 1)
    rev_per_cust = np.zeros(N + 1)
    max_events = np.zeros(N + 1, dtype=int)
    capacity = np.zeros(N + 1)
    
    for i in range(N):
        s_id = sites_data[i]["id"]
        fixed_cost[s_id] = sites_data[i]["fixed_cost"]
        rev_per_cust[s_id] = sites_data[i]["revenue_per_customer"]
        max_events[s_id] = sites_data[i]["max_events"]
        capacity[s_id] = sites_data[i]["capacity"]
        
    pop = np.zeros(num_customers)
    S_c = np.zeros(num_customers)
    
    A_base = np.zeros((num_customers, N + 1))
    
    for i, c_dat in enumerate(customers_data):
        c_id = c_dat["id"]
        pop[i] = c_dat["population"]
        S_c[i] = c_dat["num_businesses"] / params["avg_businesses_customers"]
        for s_dat in sites_data:
            s_id = s_dat["id"]
            d = instance["customer_to_site_distances"][f"{c_id}_{s_id}"]
            if d == 0: d = 1e-6
            site_attr = s_dat["num_businesses"] / params["avg_businesses_sites"]
            A_base[i, s_id] = site_attr / (d ** q)
            
    cycle = np.zeros(T + 1)
    for t in range(1, T + 1):
        cycle[t] = params["weekly_attraction_cycle"][(t - 1) % 7]
        
    d_site_site = np.zeros((N + 1, N + 1))
    for key, d in instance["site_to_site_distances"].items():
        s1, s2 = map(int, key.split('_'))
        if s1 != 0 and s2 != 0:
            d_site_site[s1, s2] = d

    # DP for shortest paths
    min_cost = np.full((T + 1, N + 1, N + 1), np.inf)
    parent = np.zeros((T + 1, N + 1, N + 1), dtype=int)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if d_site_site[i, j] <= L:
                min_cost[1, i, j] = d_site_site[i, j] * c
                
    for k in range(2, T + 1):
        for i in range(1, N + 1):
            temp = min_cost[k-1, i, 1:N+1].reshape(N, 1) + min_cost[1, 1:N+1, 1:N+1]
            idx = np.argmin(temp, axis=0)
            min_vals = np.min(temp, axis=0)
            min_cost[k, i, 1:N+1] = min_vals
            parent[k, i, 1:N+1] = idx + 1
            
    def get_path(k, i, j):
        path = [0] * (k + 1)
        path[0] = i
        path[k] = j
        curr_j = j
        for step in range(k, 1, -1):
            curr_m = parent[step, i, curr_j]
            path[step - 1] = curr_m
            curr_j = curr_m
        return path

    def compute_revenue(events_array, event_times):
        total_revenue = 0.0
        event_A = np.zeros((num_customers, T + 1))
        
        for t in event_times:
            s = events_array[t]
            event_A[:, t] = A_base[:, s] * cycle[t]
            
        attendance_dict = {}
        for t in event_times:
            s = events_array[t]
            t_start = max(1, t - epsilon)
            t_end = min(T, t + epsilon)
            
            denom = S_c + np.sum(event_A[:, t_start:t_end+1], axis=1)
            Pi = event_A[:, t] / denom
            att = np.sum(pop * Pi)
            att = min(capacity[s], att)
            attendance_dict[f"{s}_{t}"] = att
            total_revenue += att * rev_per_cust[s]
            
        return total_revenue, attendance_dict

    def build_and_evaluate(events_array):
        event_times = np.nonzero(events_array)[0]
        if len(event_times) == 0:
            return 0.0, np.ones(T+1, dtype=int), {}
            
        unique, counts = np.unique(events_array[event_times], return_counts=True)
        for u, c_val in zip(unique, counts):
            if c_val > max_events[u]:
                return -np.inf, None, None
                
        bool_events = (events_array > 0).astype(int)
        if len(bool_events[1:]) >= delta + 2:
            window_sums = np.convolve(bool_events[1:], np.ones(delta + 2, dtype=int), mode='valid')
            if np.any(window_sums > delta):
                return -np.inf, None, None

        travel_cost = 0.0
        t1 = event_times[0]
        s1 = events_array[t1]
        
        if t1 == 1:
            start_cost = 0.0
        else:
            start_cost = np.min(min_cost[t1 - 1, 1:N+1, s1])
        if np.isinf(start_cost): return -np.inf, None, None
        travel_cost += start_cost
        
        for i in range(len(event_times) - 1):
            tm = event_times[i]
            sm = events_array[tm]
            tn = event_times[i+1]
            sn = events_array[tn]
            cost_step = min_cost[tn - tm, sm, sn]
            if np.isinf(cost_step): return -np.inf, None, None
            travel_cost += cost_step
            
        fixed_c = np.sum(fixed_cost[events_array[event_times]])
        total_revenue, attendance_dict = compute_revenue(events_array, event_times)
        
        profit = total_revenue - fixed_c - travel_cost
        return profit, None, attendance_dict

    def get_marginal_matrix(events_array):
        M = np.zeros((N + 1, T + 1))
        e_times = np.nonzero(events_array)[0]
        for t in range(1, T + 1):
            old_s = events_array[t]
            events_array[t] = 0
            base_rev, _ = compute_revenue(events_array, np.nonzero(events_array)[0])
            for s in range(1, N + 1):
                events_array[t] = s
                new_rev, _ = compute_revenue(events_array, np.nonzero(events_array)[0])
                M[s, t] = new_rev - base_rev
            events_array[t] = old_s
        return M

    def solve_mip(M_matrix):
        if 'gp' not in globals():
            return None
        model = gp.Model("TourMIP")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', 2.0)
        
        Y = model.addVars(N+1, T+1, vtype=GRB.BINARY)
        X = model.addVars(N+1, T+1, vtype=GRB.BINARY)
        
        Z = {}
        for t in range(2, T+1):
            for u in range(1, N+1):
                for v in range(1, N+1):
                    if d_site_site[u, v] <= L:
                        Z[u, v, t] = model.addVar(vtype=GRB.BINARY)

        obj = gp.LinExpr()
        for t in range(1, T+1):
            for s in range(1, N+1):
                obj += (M_matrix[s, t] - fixed_cost[s]) * X[s, t]

        for t in range(2, T+1):
            for u in range(1, N+1):
                for v in range(1, N+1):
                    if (u, v, t) in Z:
                        obj -= Z[u, v, t] * c * d_site_site[u, v]
                        
        model.setObjective(obj, GRB.MAXIMIZE)

        for t in range(1, T+1):
            model.addConstr(gp.quicksum(Y[s, t] for s in range(1, N+1)) == 1)
            for s in range(1, N+1):
                model.addConstr(X[s, t] <= Y[s, t])
                
        for s in range(1, N+1):
            model.addConstr(gp.quicksum(X[s, t] for t in range(1, T+1)) <= max_events[s])
            
        for t in range(1, T - delta):
            model.addConstr(gp.quicksum(X[s, tau] for tau in range(t, t + delta + 2) for s in range(1, N+1)) <= delta)
            
        for t in range(2, T+1):
            for u in range(1, N+1):
                model.addConstr(gp.quicksum(Z[u, v, t] for v in range(1, N+1) if (u, v, t) in Z) == Y[u, t-1])
            for v in range(1, N+1):
                model.addConstr(gp.quicksum(Z[u, v, t] for u in range(1, N+1) if (u, v, t) in Z) == Y[v, t])

        model.optimize()
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
            if model.SolCount > 0:
                events_new = np.zeros(T + 1, dtype=int)
                for t in range(1, T+1):
                    for s in range(1, N+1):
                        if X[s, t].X > 0.5:
                            events_new[t] = s
                return events_new
        return None

    # Algorithm execution
    best_global_events = np.zeros(T + 1, dtype=int)
    best_global_profit = 0.0

    current_events = np.zeros(T + 1, dtype=int)
    no_improve = 0

    max_sla_iters = 5
    for iteration in range(max_sla_iters):
        if time.time() - start_time > args.time_limit * 0.2:
            break
            
        M = get_marginal_matrix(current_events)
        new_events = solve_mip(M)
        
        if new_events is None:
            break
            
        profit, _, _ = build_and_evaluate(new_events)
        if profit > best_global_profit:
            best_global_profit = profit
            best_global_events = new_events.copy()
            current_events = new_events.copy()
            no_improve = 0
            if logger:
                logger.log(best_global_profit)
        else:
            no_improve += 1
            if no_improve >= 2:
                break

    # Simulated Annealing
    T_start = max(10000.0, abs(best_global_profit) * 0.1)
    T_end = 1.0

    curr_events = best_global_events.copy()
    curr_profit, _, curr_att = build_and_evaluate(curr_events)
    if curr_profit == -np.inf:
        curr_profit = 0.0
        curr_events = np.zeros(T + 1, dtype=int)

    best_events = curr_events.copy()
    best_profit = curr_profit

    SA_START_TIME = time.time()
    SA_TIME_LIMIT = (args.time_limit - (time.time() - start_time)) * 0.95

    while SA_TIME_LIMIT > 0 and time.time() - SA_START_TIME < SA_TIME_LIMIT:
        frac = (time.time() - SA_START_TIME) / SA_TIME_LIMIT
        temp = T_start * ( (T_end / T_start) ** frac ) if T_start > 0 else 1.0
        
        new_events = curr_events.copy()
        
        op = random.random()
        if op < 0.3:
            t_rand = random.randint(1, T)
            s_rand = random.randint(1, N)
            new_events[t_rand] = s_rand
        elif op < 0.5:
            t_rand = random.randint(1, T)
            new_events[t_rand] = 0
        elif op < 0.7:
            non_zeros = np.nonzero(new_events)[0]
            if len(non_zeros) > 0:
                t_rand = random.choice(non_zeros)
                new_events[t_rand] = random.randint(1, N)
        elif op < 0.85:
            t1, t2 = random.sample(range(1, T + 1), 2)
            new_events[t1], new_events[t2] = new_events[t2], new_events[t1]
        else:
            non_zeros = np.nonzero(new_events)[0]
            zeros = np.where(new_events[1:] == 0)[0] + 1
            if len(non_zeros) > 0 and len(zeros) > 0:
                t_nz = random.choice(non_zeros)
                t_z = random.choice(zeros)
                new_events[t_z] = new_events[t_nz]
                new_events[t_nz] = 0

        new_profit, _, _ = build_and_evaluate(new_events)
        
        if new_profit > best_profit:
            best_profit = new_profit
            best_events = new_events.copy()
            if logger:
                logger.log(best_profit)
                
        if new_profit == -np.inf:
            accept = False
        else:
            diff = new_profit - curr_profit
            if diff >= 0:
                accept = True
            else:
                if diff / temp < -700:
                    accept = False
                else:
                    accept = random.random() < math.exp(diff / temp)
                    
        if accept:
            curr_profit = new_profit
            curr_events = new_events.copy()

    # Reconstruction
    Y = np.zeros(T + 1, dtype=int)
    event_times = np.nonzero(best_events)[0]
    
    if len(event_times) == 0:
        Y.fill(1)
    else:
        t1 = event_times[0]
        s1 = best_events[t1]
        if t1 == 1:
            Y[1] = s1
        else:
            s0 = np.argmin(min_cost[t1 - 1, 1:N+1, s1]) + 1
            path = get_path(t1 - 1, s0, s1)
            for idx, t in enumerate(range(1, t1 + 1)):
                Y[t] = path[idx]
                
        for i in range(len(event_times) - 1):
            tm = event_times[i]
            sm = best_events[tm]
            tn = event_times[i+1]
            sn = best_events[tn]
            path = get_path(tn - tm, sm, sn)
            for idx, t in enumerate(range(tm, tn + 1)):
                Y[t] = path[idx]
                
        tk = event_times[-1]
        sk = best_events[tk]
        for t in range(tk + 1, T + 1):
            Y[t] = sk

    final_profit, _, att_dict = build_and_evaluate(best_events)
    if final_profit == -np.inf: final_profit = 0.0

    output_tour = {}
    output_events = {}
    output_event_sites = {}
    
    for t in range(1, T + 1):
        output_tour[str(t)] = int(Y[t])
        if best_events[t] > 0:
            output_events[str(t)] = 1
            output_event_sites[str(t)] = int(best_events[t])
        else:
            output_events[str(t)] = 0
            
    solution = {
        "objective_value": float(final_profit),
        "tour": output_tour,
        "events": output_events,
        "event_sites": output_event_sites,
        "attendance": {k: float(v) for k, v in att_dict.items()}
    }

    write_solution(args.solution_path, solution)

if __name__ == '__main__':
    main()