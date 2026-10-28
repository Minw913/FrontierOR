import argparse
import json
import math
import random
import time
import copy
import sys

class State:
    def __init__(self, num_days):
        self.routes = {d: [] for d in range(num_days)}
        self.schedules = {}
        self.unassigned = {d: set() for d in range(num_days)}

def deepcopy_state(state, num_days):
    new_s = State(num_days)
    new_s.routes = {d: [list(r) for r in state.routes[d]] for d in range(num_days)}
    new_s.unassigned = {d: set(state.unassigned[d]) for d in range(num_days)}
    new_s.schedules = dict(state.schedules)
    return new_s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default="")
    args = parser.parse_args()

    start_time = time.time()
    end_time = start_time + args.time_limit - 0.5

    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_days = instance['num_days']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    max_route_duration = instance['max_route_duration']
    depot_id = instance['depot']['id']
    
    customers = instance['customers']
    N = len(customers)
    
    # Mappings
    cid_to_idx = {depot_id: 0}
    idx_to_cid = {0: depot_id}
    for i, c in enumerate(customers):
        cid_to_idx[c['id']] = i + 1
        idx_to_cid[i + 1] = c['id']

    dist = [[0.0] * (N + 1) for _ in range(N + 1)]
    coords = [None] * (N + 1)
    coords[0] = (instance['depot']['x'], instance['depot']['y'])
    for i, c in enumerate(customers):
        coords[i + 1] = (c['x'], c['y'])

    for i in range(N + 1):
        x1, y1 = coords[i]
        for j in range(N + 1):
            x2, y2 = coords[j]
            dist[i][j] = math.hypot(x1 - x2, y1 - y2)

    cust_st = [0] * (N + 1)
    cust_tw = [(0, 0)] * (N + 1)
    cust_st[0] = instance['depot']['service_time']
    cust_tw[0] = tuple(instance['depot']['time_window'])
    depot_tw = cust_tw[0]

    for i, c in enumerate(customers):
        cust_st[i + 1] = c['service_time']
        cust_tw[i + 1] = tuple(c['time_window'])

    customer_schedules = [None] * (N + 1)
    cust_day_to_part = {}
    for i, c in enumerate(customers):
        idx = i + 1
        customer_schedules[idx] = c['schedules']
        for s_idx, sched in enumerate(c['schedules']):
            for p in sched['parts']:
                cust_day_to_part[(idx, s_idx, p['visit_day'])] = p

    # Fast feasibility checking function
    def is_route_valid(r, day, schedules):
        if not r: return True
        tot_dem = 0
        for idx in r:
            s_idx = schedules[idx]
            p = cust_day_to_part[(idx, s_idx, day)]
            tot_dem += p['demand_per_visit']
        if tot_dem > vehicle_capacity: return False

        L = depot_tw[1]
        for i in range(len(r)-1, -1, -1):
            idx = r[i]
            nxt = r[i+1] if i < len(r)-1 else 0
            d_val = dist[idx][nxt]
            L = min(cust_tw[idx][1], L - d_val - cust_st[idx])
            if L < cust_tw[idx][0]:
                return False
            
        D_star = L - dist[0][r[0]]
        if D_star < depot_tw[0]:
            return False
            
        S = max(cust_tw[r[0]][0], D_star + dist[0][r[0]])
        for i in range(1, len(r)):
            idx = r[i]
            prev = r[i-1]
            S = max(cust_tw[idx][0], S + cust_st[prev] + dist[prev][idx])
            
        ret_time = S + cust_st[r[-1]] + dist[r[-1]][0]
        if ret_time > depot_tw[1]:
            return False
        if ret_time - D_star > max_route_duration:
            return False
        return True

    def route_cost(r):
        if not r: return 0.0
        c = dist[0][r[0]]
        for i in range(len(r)-1):
            c += dist[r[i]][r[i+1]]
        c += dist[r[-1]][0]
        return c

    def calc_fitness(state, P_VEH, P_UNASSIGNED):
        t_cost = 0.0
        t_un = 0
        t_veh = 0
        for d in range(num_days):
            rd = [r for r in state.routes[d] if r]
            for r in rd:
                t_cost += route_cost(r)
            t_un += len(state.unassigned[d])
            t_veh += max(0, len(rd) - num_vehicles)
        return t_cost + P_VEH*t_veh + P_UNASSIGNED*t_un, t_cost, t_un, t_veh

    # Construct Initial Solution
    def create_initial_state():
        s = State(num_days)
        for idx in range(1, N + 1):
            s.schedules[idx] = 0
            for p in customer_schedules[idx][0]['parts']:
                s.unassigned[p['visit_day']].add(idx)

        order = list(range(1, N + 1))
        random.shuffle(order)
        if random.random() < 0.5:
            order.sort(key=lambda idx: cust_tw[idx][1] - cust_tw[idx][0])

        for idx in order:
            best_s_idx = -1
            best_insertions = None
            best_cost = float('inf')

            for s_idx, sched in enumerate(customer_schedules[idx]):
                valid = True
                insertions = {}
                cost_increase = 0.0
                s.schedules[idx] = s_idx

                for p in sched['parts']:
                    d = p['visit_day']
                    b_r_idx, b_pos, b_d_cost = -1, -1, float('inf')

                    for r_idx, r in enumerate(s.routes[d]):
                        for pos in range(len(r) + 1):
                            new_r = r[:pos] + [idx] + r[pos:]
                            if is_route_valid(new_r, d, s.schedules):
                                prev = r[pos-1] if pos > 0 else 0
                                nxt = r[pos] if pos < len(r) else 0
                                delta = dist[prev][idx] + dist[idx][nxt] - dist[prev][nxt]
                                if delta < b_d_cost:
                                    b_d_cost, b_r_idx, b_pos = delta, r_idx, pos

                    if b_r_idx == -1:
                        new_r = [idx]
                        if is_route_valid(new_r, d, s.schedules):
                            proxy = 2 * dist[0][idx] + (1000.0 if len([rt for rt in s.routes[d] if rt]) >= num_vehicles else 0.0)
                            cost_increase += proxy
                            insertions[d] = ('new', 0)
                        else:
                            valid = False
                            break
                    else:
                        cost_increase += b_d_cost
                        insertions[d] = (b_r_idx, b_pos)

                if valid and cost_increase < best_cost:
                    best_cost = cost_increase
                    best_s_idx = s_idx
                    best_insertions = insertions

            if best_s_idx != -1:
                s.schedules[idx] = best_s_idx
                for p in customer_schedules[idx][best_s_idx]['parts']:
                    d = p['visit_day']
                    s.unassigned[d].discard(idx)
                    r_idx, pos = best_insertions[d]
                    if r_idx == 'new':
                        s.routes[d].append([idx])
                    else:
                        s.routes[d][r_idx].insert(pos, idx)
            else:
                s.schedules[idx] = 0
                for p in customer_schedules[idx][0]['parts']:
                    s.unassigned[p['visit_day']].add(idx)

        return s

    best_init_obj = float('inf')
    best_init_state = None
    for _ in range(3):
        s = create_initial_state()
        fit, _, _, _ = calc_fitness(s, 1000.0, 1000.0)
        if fit < best_init_obj:
            best_init_obj = fit
            best_init_state = s
            
    state = best_init_state

    # SA Operators
    def accept(delta, T):
        if delta < 0: return True
        if delta / T > 20: return False
        return random.random() < math.exp(-delta / T)

    def attempt_relocate(state, curr_T, P_VEH, P_UNASSIGNED):
        d = random.randrange(num_days)
        r_d = [r for r in state.routes[d] if r]
        if not r_d: return False
        
        r1_idx = random.randrange(len(r_d))
        r1 = r_d[r1_idx]
        pos1 = random.randrange(len(r1))
        idx = r1[pos1]
        
        r2_idx = random.randrange(len(r_d) + 1)
        r2 = [] if r2_idx == len(r_d) else r_d[r2_idx]
        pos2 = random.randrange(len(r2) + 1)
        if r1_idx == r2_idx and pos1 == pos2: return False
        
        if r1_idx == r2_idx:
            new_r = r1[:]
            new_r.pop(pos1)
            new_r.insert(pos2 if pos2 <= pos1 else pos2 - 1, idx)
            if not is_route_valid(new_r, d, state.schedules): return False
            delta = route_cost(new_r) - route_cost(r1)
            if accept(delta, curr_T):
                state.routes[d][r1_idx] = new_r
                return True
        else:
            new_r1, new_r2 = r1[:], r2[:]
            new_r1.pop(pos1)
            new_r2.insert(pos2, idx)
            if not is_route_valid(new_r1, d, state.schedules): return False
            if not is_route_valid(new_r2, d, state.schedules): return False
            
            d_cost = route_cost(new_r1) + route_cost(new_r2) - route_cost(r1) - route_cost(r2)
            v_bef = len(r_d)
            v_aft = v_bef - (1 if len(new_r1) == 0 else 0) + (1 if not r2 and new_r2 else 0)
            d_pen = P_VEH * (max(0, v_aft - num_vehicles) - max(0, v_bef - num_vehicles))
            
            if accept(d_cost + d_pen, curr_T):
                state.routes[d][r1_idx] = new_r1
                if r2_idx == len(r_d): state.routes[d].append(new_r2)
                else: state.routes[d][r2_idx] = new_r2
                return True
        return False

    def attempt_swap(state, curr_T, P_VEH, P_UNASSIGNED):
        d = random.randrange(num_days)
        r_d = [r for r in state.routes[d] if r]
        if len(r_d) < 2: return False
        r1_idx, r2_idx = random.sample(range(len(r_d)), 2)
        r1, r2 = r_d[r1_idx], r_d[r2_idx]
        p1, p2 = random.randrange(len(r1)), random.randrange(len(r2))
        
        new_r1, new_r2 = r1[:], r2[:]
        new_r1[p1], new_r2[p2] = r2[p2], r1[p1]
        if not is_route_valid(new_r1, d, state.schedules) or not is_route_valid(new_r2, d, state.schedules): return False
        
        delta = route_cost(new_r1) + route_cost(new_r2) - route_cost(r1) - route_cost(r2)
        if accept(delta, curr_T):
            state.routes[d][r1_idx], state.routes[d][r2_idx] = new_r1, new_r2
            return True
        return False

    def attempt_2opt(state, curr_T, P_VEH, P_UNASSIGNED):
        d = random.randrange(num_days)
        r_d = [r for r in state.routes[d] if len(r) >= 2]
        if not r_d: return False
        
        r_idx = random.randrange(len(r_d))
        r = r_d[r_idx]
        p1, p2 = sorted(random.sample(range(len(r)), 2))
        new_r = r[:p1] + r[p1:p2+1][::-1] + r[p2+1:]
        
        if not is_route_valid(new_r, d, state.schedules): return False
        delta = route_cost(new_r) - route_cost(r)
        if accept(delta, curr_T):
            state.routes[d][r_idx] = new_r
            return True
        return False

    def attempt_change_schedule(state, curr_T, P_VEH, P_UNASSIGNED):
        idx = random.randrange(1, N + 1)
        old_s = state.schedules[idx]
        avails = customer_schedules[idx]
        if len(avails) <= 1: return False
        new_s = random.choice([i for i in range(len(avails)) if i != old_s])
        
        old_days = [p['visit_day'] for p in avails[old_s]['parts']]
        new_days = [p['visit_day'] for p in avails[new_s]['parts']]
        
        mod_days = {}
        for d in set(old_days + new_days):
            rd = [list(r) for r in state.routes[d] if r]
            if d in old_days:
                for r in rd:
                    if idx in r:
                        r.remove(idx)
                        break
            mod_days[d] = rd

        state.schedules[idx] = new_s
        possible = True
        for d in new_days:
            rd = mod_days[d]
            b_r_idx, b_pos, b_d_cost = -1, -1, float('inf')
            
            for r_idx, r in enumerate(rd):
                for pos in range(len(r)+1):
                    new_r = r[:pos] + [idx] + r[pos:]
                    if is_route_valid(new_r, d, state.schedules):
                        delta = route_cost(new_r) - route_cost(r)
                        if delta < b_d_cost:
                            b_d_cost, b_r_idx, b_pos = delta, r_idx, pos
                            
            if b_r_idx != -1:
                r = rd[b_r_idx]
                rd[b_r_idx] = r[:b_pos] + [idx] + r[b_pos:]
            else:
                if is_route_valid([idx], d, state.schedules):
                    rd.append([idx])
                else:
                    possible = False
                    break
                    
        if not possible:
            state.schedules[idx] = old_s
            return False
            
        c_bef = sum(route_cost(r) for d in mod_days for r in state.routes[d] if r)
        c_aft = sum(route_cost(r) for rd in mod_days.values() for r in rd)
        
        v_bef = sum(max(0, len([r for r in state.routes[d] if r]) - num_vehicles) for d in mod_days)
        v_aft = sum(max(0, len([r for r in rd if r]) - num_vehicles) for rd in mod_days.values())
        
        u_bef = sum(idx in state.unassigned[d] for d in old_days)
        d_pen = P_VEH * (v_aft - v_bef) - (u_bef * P_UNASSIGNED)
        
        if accept((c_aft - c_bef) + d_pen, curr_T):
            for d, rd in mod_days.items():
                state.routes[d] = [r for r in rd if r]
            for d in old_days:
                state.unassigned[d].discard(idx)
            return True
            
        state.schedules[idx] = old_s
        return False

    def attempt_insert_unassigned(state, curr_T, P_VEH, P_UNASSIGNED):
        ulist = [(d, idx) for d in range(num_days) for idx in state.unassigned[d]]
        if not ulist: return False
        d, idx = random.choice(ulist)
        
        rd = [list(r) for r in state.routes[d] if r]
        b_r_idx, b_pos, b_d_cost = -1, -1, float('inf')
        
        for r_idx, r in enumerate(rd):
            for pos in range(len(r)+1):
                new_r = r[:pos] + [idx] + r[pos:]
                if is_route_valid(new_r, d, state.schedules):
                    delta = route_cost(new_r) - route_cost(r)
                    if delta < b_d_cost:
                        b_d_cost, b_r_idx, b_pos = delta, r_idx, pos
                        
        if b_r_idx != -1:
            r = rd[b_r_idx]
            rd[b_r_idx] = r[:b_pos] + [idx] + r[b_pos:]
            d_cost, d_pen = b_d_cost, -P_UNASSIGNED
        else:
            if is_route_valid([idx], d, state.schedules):
                rd.append([idx])
                d_cost, d_pen = 2 * dist[0][idx], -P_UNASSIGNED
                if len(rd) > num_vehicles: d_pen += P_VEH
            else:
                return False
                
        if accept(d_cost + d_pen, curr_T):
            state.routes[d] = rd
            state.unassigned[d].discard(idx)
            return True
        return False

    def attempt_ruin(state, curr_T, P_VEH, P_UNASSIGNED):
        idx = random.randrange(1, N + 1)
        s_idx = state.schedules[idx]
        days = [p['visit_day'] for p in customer_schedules[idx][s_idx]['parts']]
        if all(idx in state.unassigned[d] for d in days): return False
        
        mod_days = {}
        c_red = 0.0
        u_added = 0
        
        for d in days:
            rd = [list(r) for r in state.routes[d] if r]
            found = False
            for r in rd:
                if idx in r:
                    c_red += route_cost(r)
                    r.remove(idx)
                    c_red -= route_cost(r)
                    found = True
                    break
            if found: u_added += 1
            mod_days[d] = [r for r in rd if r]
            
        v_bef = sum(max(0, len([r for r in state.routes[d] if r]) - num_vehicles) for d in days)
        v_aft = sum(max(0, len(mod_days[d]) - num_vehicles) for d in days)
        
        d_pen = (v_aft - v_bef) * P_VEH + (u_added * P_UNASSIGNED)
        if accept(-c_red + d_pen, curr_T):
            for d, rd in mod_days.items():
                state.routes[d] = rd
                state.unassigned[d].add(idx)
            return True
        return False

    T_start = 200.0
    T_end = 0.5

    best_valid_obj = float('inf')
    best_valid_state = None
    best_fallback_key = (float('inf'), float('inf'), float('inf'))
    best_fallback_state = None
    
    # Init Check
    fit, t_cost, t_un, t_veh = calc_fitness(state, 1000.0, 1000.0)
    if t_un == 0 and t_veh == 0:
        best_valid_obj = t_cost
        best_valid_state = deepcopy_state(state, num_days)
        if logger: logger.log(best_valid_obj)
    best_fallback_key = (t_un, t_veh, t_cost)
    best_fallback_state = deepcopy_state(state, num_days)

    iters = 0
    curr_P_VEH, curr_P_UNASSIGNED = 1000.0, 2000.0
    progress = 0.0
    
    while True:
        iters += 1
        if iters % 500 == 0:
            now = time.time()
            if now > end_time: break
            progress = (now - start_time) / (args.time_limit - 0.5)
            curr_P_VEH = 1000.0 + 4000.0 * progress
            curr_P_UNASSIGNED = 2000.0 + 8000.0 * progress
            
            fit, t_cost, t_un, t_veh = calc_fitness(state, curr_P_VEH, curr_P_UNASSIGNED)
            if t_un == 0 and t_veh == 0:
                if t_cost < best_valid_obj:
                    best_valid_obj = t_cost
                    best_valid_state = deepcopy_state(state, num_days)
                    if logger: logger.log(best_valid_obj)
            else:
                cand_key = (t_un, t_veh, t_cost)
                if cand_key < best_fallback_key:
                    best_fallback_key = cand_key
                    best_fallback_state = deepcopy_state(state, num_days)

        curr_T = T_start * ((T_end / T_start) ** progress)
        m = random.choices(['rel', 'swp', '2opt', 'sch', 'ruin', 'ins'], weights=[30, 20, 20, 10, 5, 15])[0]
        
        if m == 'rel': attempt_relocate(state, curr_T, curr_P_VEH, curr_P_UNASSIGNED)
        elif m == 'swp': attempt_swap(state, curr_T, curr_P_VEH, curr_P_UNASSIGNED)
        elif m == '2opt': attempt_2opt(state, curr_T, curr_P_VEH, curr_P_UNASSIGNED)
        elif m == 'sch': attempt_change_schedule(state, curr_T, curr_P_VEH, curr_P_UNASSIGNED)
        elif m == 'ruin': attempt_ruin(state, curr_T, curr_P_VEH, curr_P_UNASSIGNED)
        elif m == 'ins': attempt_insert_unassigned(state, curr_T, curr_P_VEH, curr_P_UNASSIGNED)

    final_state = best_valid_state if best_valid_state else best_fallback_state

    # Format Solution
    routes_out = {}
    total_opt = 0.0
    for d in range(num_days):
        routes_out[str(d)] = []
        for r in final_state.routes[d]:
            if not r: continue
            rc = route_cost(r)
            total_opt += rc
            dem = 0
            pdict = {}
            for idx in r:
                s_idx = final_state.schedules[idx]
                p = cust_day_to_part[(idx, s_idx, d)]
                dem += p['demand_per_visit']
                pdict[str(idx_to_cid[idx])] = p
                
            routes_out[str(d)].append({
                "customers": [idx_to_cid[idx] for idx in r],
                "cost": rc,
                "demand": dem,
                "parts": pdict
            })

    schedules_out = {}
    for idx in range(1, N + 1):
        s_idx = final_state.schedules[idx]
        sched_obj = customer_schedules[idx][s_idx]
        schedules_out[str(idx_to_cid[idx])] = {
            "schedule_index": s_idx,
            "days": sched_obj['days'],
            "visit_frequency": sched_obj['visit_frequency']
        }

    sol = {
        "objective_value": total_opt,
        "routes": routes_out,
        "schedules": schedules_out
    }

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)

if __name__ == '__main__':
    main()