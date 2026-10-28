import argparse
import json
import time
import math
import random

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args, _ = parser.parse_known_args()

    start_time = time.time()
    
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    n = instance['n']
    m = instance['m']
    num_scenarios = instance['num_scenarios']
    facility_capacity_limits = instance['facility_capacity_limits']
    capacity_requirements = instance['capacity_requirements']
    release_times = instance['release_times']
    mean_processing_times = instance['mean_processing_times']
    scenarios = instance['scenarios']

    if n == 0:
        sol = {
            "objective_value": 0.0,
            "assignment": {},
            "schedule": {str(s): {} for s in range(num_scenarios)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    # Determine feasible facilities per job to avoid capacity overflow
    feasible_facilities = []
    for j in range(n):
        feas = [k for k in range(m) if capacity_requirements[k][j] <= facility_capacity_limits[k]]
        # Fallback in case of invalid data: allow all
        if not feas:
            feas = list(range(m))
        feasible_facilities.append(feas)

    # ------------------ Scheduling Logic ------------------

    def schedule_job(profile, r, p, cap):
        curr_t = r
        while True:
            valid = True
            start_idx = 0
            for i in range(len(profile)):
                if i == len(profile) - 1 or profile[i+1][0] > curr_t:
                    start_idx = i
                    break
                    
            for i in range(start_idx, len(profile)):
                if profile[i][0] >= curr_t + p:
                    break
                if profile[i][1] < cap:
                    valid = False
                    if i + 1 < len(profile):
                        curr_t = profile[i+1][0]
                    else:
                        curr_t += 1
                    break
                    
            if valid:
                break
                
        def insert_point(t):
            for i in range(len(profile)):
                if profile[i][0] == t:
                    return
                if profile[i][0] > t:
                    profile.insert(i, [t, profile[i-1][1]])
                    return
            profile.append([t, profile[-1][1]])

        insert_point(curr_t)
        insert_point(curr_t + p)
        
        for i in range(len(profile)):
            if curr_t <= profile[i][0] < curr_t + p:
                profile[i][1] -= cap
                
        new_profile = [profile[0]]
        for i in range(1, len(profile)):
            if profile[i][1] != new_profile[-1][1]:
                new_profile.append(profile[i])
        profile[:] = new_profile
        return curr_t

    schedule_cache = {}

    def get_schedule(k, s, jobs_tuple):
        if not jobs_tuple:
            return 0, {}
        key = (k, s, jobs_tuple)
        if key in schedule_cache:
            return schedule_cache[key]
        
        best_mksp = float('inf')
        best_sched = None
        
        jobs_list = list(jobs_tuple)
        s_data = scenarios[s]
        
        rules = [
            lambda j: (release_times[j], -s_data['processing_times'][k][j], -capacity_requirements[k][j]),
            lambda j: (release_times[j], s_data['processing_times'][k][j], capacity_requirements[k][j]),
            lambda j: (release_times[j], -capacity_requirements[k][j], -s_data['processing_times'][k][j])
        ]
        
        for rule in rules:
            sorted_jobs = sorted(jobs_list, key=rule)
            profile = [[0, facility_capacity_limits[k]]]
            scenario_schedule = {}
            mksp = 0
            for j in sorted_jobs:
                r = release_times[j]
                p = s_data['processing_times'][k][j]
                cap = capacity_requirements[k][j]
                if p == 0:
                    scenario_schedule[j] = {'facility': k, 'start': r, 'finish': r}
                    if r > mksp: mksp = r
                    continue
                st = schedule_job(profile, r, p, cap)
                fin = st + p
                scenario_schedule[j] = {'facility': k, 'start': st, 'finish': fin}
                if fin > mksp:
                    mksp = fin
            if mksp < best_mksp:
                best_mksp = mksp
                best_sched = scenario_schedule
                
        schedule_cache[key] = (best_mksp, best_sched)
        return best_mksp, best_sched

    # ------------------ State / Local Search ------------------

    class SolutionState:
        def __init__(self, y):
            self.y = list(y)
            self.k_jobs = [set() for _ in range(m)]
            for j, k in enumerate(y):
                self.k_jobs[k].add(j)
            
            self.k_s_makespan = [[0]*m for _ in range(num_scenarios)]
            self.k_s_schedule = [[{}]*m for _ in range(num_scenarios)]
            
            for k in range(m):
                jobs = tuple(sorted(self.k_jobs[k]))
                for s in range(num_scenarios):
                    mksp, sched = get_schedule(k, s, jobs)
                    self.k_s_makespan[s][k] = mksp
                    self.k_s_schedule[s][k] = sched
            
            self.expected_makespan = self._calc_obj()

        def _calc_obj(self):
            obj = 0
            for s in range(num_scenarios):
                obj += scenarios[s]['probability'] * max(self.k_s_makespan[s])
            return obj

        def copy(self):
            new_state = SolutionState.__new__(SolutionState)
            new_state.y = list(self.y)
            new_state.k_jobs = [set(s) for s in self.k_jobs]
            new_state.k_s_makespan = [list(lst) for lst in self.k_s_makespan]
            new_state.k_s_schedule = [[dict(d) for d in lst] for lst in self.k_s_schedule]
            new_state.expected_makespan = self.expected_makespan
            return new_state

    def eval_neighbor(state, j, k_old, k_new):
        jobs_old = tuple(sorted([x for x in state.k_jobs[k_old] if x != j]))
        jobs_new = tuple(sorted(list(state.k_jobs[k_new]) + [j]))
        
        obj = 0
        for s in range(num_scenarios):
            m_old, _ = get_schedule(k_old, s, jobs_old)
            m_new, _ = get_schedule(k_new, s, jobs_new)
            
            max_m = 0
            for k in range(m):
                if k == k_old: v = m_old
                elif k == k_new: v = m_new
                else: v = state.k_s_makespan[s][k]
                if v > max_m: max_m = v
            obj += scenarios[s]['probability'] * max_m
        return obj

    def apply_move(state, j, k_old, k_new):
        state.y[j] = k_new
        state.k_jobs[k_old].remove(j)
        state.k_jobs[k_new].add(j)
        jobs_old = tuple(sorted(state.k_jobs[k_old]))
        jobs_new = tuple(sorted(state.k_jobs[k_new]))
        
        for s in range(num_scenarios):
            state.k_s_makespan[s][k_old], state.k_s_schedule[s][k_old] = get_schedule(k_old, s, jobs_old)
            state.k_s_makespan[s][k_new], state.k_s_schedule[s][k_new] = get_schedule(k_new, s, jobs_new)
        state.expected_makespan = state._calc_obj()

    def swap_move(state, j1, j2):
        k1 = state.y[j1]
        k2 = state.y[j2]
        
        jobs_k1 = tuple(sorted([x for x in state.k_jobs[k1] if x != j1] + [j2]))
        jobs_k2 = tuple(sorted([x for x in state.k_jobs[k2] if x != j2] + [j1]))
        
        obj = 0
        for s in range(num_scenarios):
            m1, _ = get_schedule(k1, s, jobs_k1)
            m2, _ = get_schedule(k2, s, jobs_k2)
            max_m = 0
            for k in range(m):
                if k == k1: v = m1
                elif k == k2: v = m2
                else: v = state.k_s_makespan[s][k]
                if v > max_m: max_m = v
            obj += scenarios[s]['probability'] * max_m
        return obj

    def apply_swap(state, j1, j2):
        k1 = state.y[j1]
        k2 = state.y[j2]
        state.y[j1] = k2
        state.y[j2] = k1
        state.k_jobs[k1].remove(j1)
        state.k_jobs[k1].add(j2)
        state.k_jobs[k2].remove(j2)
        state.k_jobs[k2].add(j1)
        
        jobs_k1 = tuple(sorted(state.k_jobs[k1]))
        jobs_k2 = tuple(sorted(state.k_jobs[k2]))
        
        for s in range(num_scenarios):
            state.k_s_makespan[s][k1], state.k_s_schedule[s][k1] = get_schedule(k1, s, jobs_k1)
            state.k_s_makespan[s][k2], state.k_s_schedule[s][k2] = get_schedule(k2, s, jobs_k2)
        state.expected_makespan = state._calc_obj()

    def run_sa(initial_state, time_until):
        best_state = initial_state
        current_state = initial_state.copy()
        
        T = 100.0
        T_min = 0.01
        alpha = 0.99
        
        while time.time() < time_until:
            for _ in range(200):
                if len(schedule_cache) > 100000:
                    schedule_cache.clear()
                    
                if random.random() < 0.7:
                    j = random.randint(0, n-1)
                    k_old = current_state.y[j]
                    candidates = [k for k in feasible_facilities[j] if k != k_old]
                    if not candidates: continue
                    k_new = random.choice(candidates)
                    
                    new_obj = eval_neighbor(current_state, j, k_old, k_new)
                    delta = new_obj - current_state.expected_makespan
                    if delta < -1e-5 or random.random() < math.exp(-delta / T):
                        apply_move(current_state, j, k_old, k_new)
                        if current_state.expected_makespan < best_state.expected_makespan - 1e-5:
                            best_state = current_state.copy()
                            if logger: logger.log(best_state.expected_makespan)
                else:
                    j1 = random.randint(0, n-1)
                    j2 = random.randint(0, n-1)
                    k1, k2 = current_state.y[j1], current_state.y[j2]
                    if k1 == k2: continue
                    if k1 not in feasible_facilities[j2] or k2 not in feasible_facilities[j1]: continue
                    
                    new_obj = swap_move(current_state, j1, j2)
                    delta = new_obj - current_state.expected_makespan
                    if delta < -1e-5 or random.random() < math.exp(-delta / T):
                        apply_swap(current_state, j1, j2)
                        if current_state.expected_makespan < best_state.expected_makespan - 1e-5:
                            best_state = current_state.copy()
                            if logger: logger.log(best_state.expected_makespan)
            
            T = max(T_min, T * alpha)
            if T == T_min:
                T = 50.0
                if random.random() < 0.5:
                    current_state = best_state.copy()
                    
        return best_state

    # Initial Greedy Solution
    y_init = [0] * n
    loads = [0.0] * m
    for j in sorted(range(n), key=lambda x: -sum(scenarios[s]['probability'] * release_times[x] for s in range(num_scenarios))):
        best_k = min(feasible_facilities[j], key=lambda k: loads[k] + sum(scenarios[s]['probability'] * scenarios[s]['processing_times'][k][j] for s in range(num_scenarios)))
        y_init[j] = best_k
        loads[best_k] += sum(scenarios[s]['probability'] * scenarios[s]['processing_times'][best_k][j] for s in range(num_scenarios))
        
    best_state = SolutionState(y_init)
    if logger: logger.log(best_state.expected_makespan)

    # ------------------ Optimization Flow ------------------
    
    time_limit_sa = start_time + min(15.0, max(2.0, args.time_limit * 0.2))
    best_state = run_sa(best_state, time_limit_sa)
    
    def formulate_and_solve_mip(state, time_left):
        T_bounds = []
        total_x = 0
        for s in range(num_scenarios):
            s_data = scenarios[s]
            max_t = 0
            for k in range(m):
                t_k = max([release_times[j] for j in range(n)]) if n > 0 else 0
                t_k += sum(s_data['processing_times'][k][j] for j in range(n))
                max_t = max(max_t, t_k)
            T_bounds.append(max_t)
            for j in range(n):
                for k in feasible_facilities[j]:
                    total_x += max(0, max_t - release_times[j] - s_data['processing_times'][k][j] + 1)
                    
        if total_x > 150000:
            return state, False
            
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("Monolithic", env=env)
        model.setParam("TimeLimit", max(1.0, time_left - 1.0))
        model.setParam("Threads", 1)
        
        y_vars = {}
        for j in range(n):
            for k in feasible_facilities[j]:
                y_vars[j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")
                
        x_vars = {}
        C_max = {}
        
        for s in range(num_scenarios):
            C_max[s] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Cmax_{s}")
            s_data = scenarios[s]
            for j in range(n):
                for k in feasible_facilities[j]:
                    p = s_data['processing_times'][k][j]
                    max_start = T_bounds[s] - p
                    for t in range(release_times[j], max_start + 1):
                        x_vars[j, k, s, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}_{s}_{t}")
                        
        for j in range(n):
            model.addConstr(gp.quicksum(y_vars[j, k] for k in feasible_facilities[j]) == 1)
            
        for s in range(num_scenarios):
            s_data = scenarios[s]
            for j in range(n):
                for k in feasible_facilities[j]:
                    p = s_data['processing_times'][k][j]
                    max_start = T_bounds[s] - p
                    model.addConstr(gp.quicksum(x_vars[j, k, s, t] for t in range(release_times[j], max_start + 1)) == y_vars[j, k])
                    
            for k in range(m):
                limit = facility_capacity_limits[k]
                for t in range(T_bounds[s] + 1):
                    expr = []
                    for j in range(n):
                        if k not in feasible_facilities[j]: continue
                        cap = capacity_requirements[k][j]
                        if cap == 0: continue
                        p = s_data['processing_times'][k][j]
                        min_st = max(release_times[j], t - p + 1)
                        max_st = min(t, T_bounds[s] - p)
                        if min_st <= max_st:
                            for tau in range(min_st, max_st + 1):
                                expr.append(cap * x_vars[j, k, s, tau])
                    if expr:
                        model.addConstr(gp.quicksum(expr) <= limit)
                        
            for j in range(n):
                for k in feasible_facilities[j]:
                    p = s_data['processing_times'][k][j]
                    max_start = T_bounds[s] - p
                    model.addConstr(C_max[s] >= gp.quicksum((t + p) * x_vars[j, k, s, t] for t in range(release_times[j], max_start + 1)))

        model.setObjective(gp.quicksum(scenarios[s]['probability'] * C_max[s] for s in range(num_scenarios)), GRB.MINIMIZE)
        
        for j in range(n):
            k_val = state.y[j]
            if k_val in feasible_facilities[j]:
                y_vars[j, k_val].Start = 1
                for k_other in feasible_facilities[j]:
                    if k_other != k_val:
                        y_vars[j, k_other].Start = 0
                
        for s in range(num_scenarios):
            for k in range(m):
                sched_k = state.k_s_schedule[s].get(k, {}) if s < len(state.k_s_schedule) and k < m else {}
                for j, info in sched_k.items():
                    if k in feasible_facilities[j]:
                        st = info['start']
                        if (j, k, s, st) in x_vars:
                            x_vars[j, k, s, st].Start = 1
                            
        model.optimize()
        
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
            if model.SolCount > 0 and model.ObjVal < state.expected_makespan - 1e-4:
                new_y = [0]*n
                for j in range(n):
                    for k in feasible_facilities[j]:
                        if y_vars[j, k].X > 0.5:
                            new_y[j] = k
                            break
                            
                new_state = SolutionState(new_y)
                for s in range(num_scenarios):
                    s_data = scenarios[s]
                    for k in range(m):
                        new_state.k_s_schedule[s][k] = {}
                        max_fin = 0
                        for j in range(n):
                            if new_y[j] == k:
                                p = s_data['processing_times'][k][j]
                                max_start = T_bounds[s] - p
                                started_at = release_times[j]
                                for t in range(release_times[j], max_start + 1):
                                    if (j, k, s, t) in x_vars and x_vars[j, k, s, t].X > 0.5:
                                        started_at = t
                                        break
                                fin = started_at + p
                                new_state.k_s_schedule[s][k][j] = {'facility': k, 'start': int(started_at), 'finish': int(fin)}
                                max_fin = max(max_fin, fin)
                        new_state.k_s_makespan[s][k] = max_fin
                new_state.expected_makespan = new_state._calc_obj()
                if logger: logger.log(new_state.expected_makespan)
                return new_state, True
        return state, True

    if GUROBI_AVAILABLE:
        tl_mip = args.time_limit - (time.time() - start_time) - 1.0
        best_state, mip_run = formulate_and_solve_mip(best_state, tl_mip)
        if not mip_run:
            time_left_sa = start_time + args.time_limit - 1.5
            best_state = run_sa(best_state, time_left_sa)
    else:
        time_left_sa = start_time + args.time_limit - 1.5
        best_state = run_sa(best_state, time_left_sa)

    # Output Formulation
    solution = {
        "objective_value": float(best_state.expected_makespan),
        "assignment": {str(j): int(best_state.y[j]) for j in range(n)},
        "schedule": {}
    }
    
    for s_idx in range(num_scenarios):
        s_str = str(s_idx)
        solution["schedule"][s_str] = {}
        for k in range(m):
            sched_k = best_state.k_s_schedule[s_idx][k]
            for j, info in sched_k.items():
                solution["schedule"][s_str][str(j)] = {
                    "facility": int(info["facility"]),
                    "start": int(info["start"]),
                    "finish": int(info["finish"])
                }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()