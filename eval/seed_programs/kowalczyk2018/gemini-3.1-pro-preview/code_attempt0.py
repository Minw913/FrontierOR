import argparse
import json
import time
import bisect
import random
import sys

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class Optimizer:
    def __init__(self, m, p, w, logger, time_limit):
        self.m = m
        self.p = p
        self.w = w
        self.n = len(p)
        self.logger = logger
        self.time_limit = time_limit
        self.start_time = time.time()
        self.best_obj = float('inf')
        self.best_assignment = None

    def time_left(self):
        return self.time_limit - (time.time() - self.start_time)

    def update_best(self, assignment, obj):
        if obj < self.best_obj:
            self.best_obj = obj
            self.best_assignment = list(assignment)
            if self.logger:
                self.logger.log(float(obj))

    def greedy_initial(self):
        machine_loads = [0] * self.m
        assignment = [0] * self.n
        for j in range(self.n):
            best_m = 0
            min_load = machine_loads[0]
            for idx in range(1, self.m):
                if machine_loads[idx] < min_load:
                    min_load = machine_loads[idx]
                    best_m = idx
            assignment[j] = best_m
            machine_loads[best_m] += self.p[j]
        return assignment

    def eval_full(self, assignment):
        machine_jobs = [[] for _ in range(self.m)]
        for j in range(self.n):
            machine_jobs[assignment[j]].append(j)
        
        total_cost = 0
        for i in range(self.m):
            load = 0
            for j in machine_jobs[i]:
                load += self.p[j]
                total_cost += self.w[j] * load
        return total_cost

    def local_search(self, start_assignment):
        assignment = list(start_assignment)
        machine_jobs = [[] for _ in range(self.m)]
        for j in range(self.n):
            machine_jobs[assignment[j]].append(j)
            
        def cost_of_machine_list(jobs):
            c = 0
            l = 0
            for j in jobs:
                l += self.p[j]
                c += self.w[j] * l
            return c
            
        machine_costs = [cost_of_machine_list(mj) for mj in machine_jobs]
        current_obj = sum(machine_costs)
        
        jobs_order = list(range(self.n))
        
        improved = True
        while improved and self.time_left() > 2:
            improved = False
            random.shuffle(jobs_order)
            
            # Shifts
            for j in jobs_order:
                if self.time_left() < 2: break
                old_m = assignment[j]
                
                old_m_jobs_without_j = [x for x in machine_jobs[old_m] if x != j]
                cost_old_if_removed = cost_of_machine_list(old_m_jobs_without_j)
                delta_old = machine_costs[old_m] - cost_old_if_removed
                
                best_gain = 0
                best_m = -1
                
                mach_order = list(range(self.m))
                random.shuffle(mach_order)
                
                new_m_jobs_best = None
                cost_new_best = 0
                
                for new_m in mach_order:
                    if new_m == old_m: continue
                    idx = bisect.bisect_left(machine_jobs[new_m], j)
                    new_m_jobs_with_j = machine_jobs[new_m][:idx] + [j] + machine_jobs[new_m][idx:]
                    cost_new_if_added = cost_of_machine_list(new_m_jobs_with_j)
                    delta_new = cost_new_if_added - machine_costs[new_m]
                    
                    gain = delta_old - delta_new
                    if gain > 0:
                        best_gain = gain
                        best_m = new_m
                        new_m_jobs_best = new_m_jobs_with_j
                        cost_new_best = cost_new_if_added
                        break
                        
                if best_gain > 0:
                    assignment[j] = best_m
                    machine_jobs[old_m] = old_m_jobs_without_j
                    machine_jobs[best_m] = new_m_jobs_best  
                    machine_costs[old_m] = cost_old_if_removed
                    machine_costs[best_m] = cost_new_best
                    current_obj -= best_gain
                    self.update_best(assignment, current_obj)
                    improved = True

            if improved:
                continue
                
            # Swaps
            for j1 in jobs_order:
                if self.time_left() < 2: break
                m1 = assignment[j1]
                
                window = 50
                j2_candidates = list(range(max(0, j1 - window), min(self.n, j1 + window + 1)))
                random.shuffle(j2_candidates)
                
                swap_done = False
                m1_jobs_without_j1 = [x for x in machine_jobs[m1] if x != j1]
                
                for j2 in j2_candidates:
                    if j1 == j2: continue
                    m2 = assignment[j2]
                    if m1 == m2: continue
                    
                    idx1 = bisect.bisect_left(m1_jobs_without_j1, j2)
                    m1_new_jobs = m1_jobs_without_j1[:idx1] + [j2] + m1_jobs_without_j1[idx1:]
                    m1_new_cost = cost_of_machine_list(m1_new_jobs)
                    
                    m2_jobs_without_j2 = [x for x in machine_jobs[m2] if x != j2]
                    idx2 = bisect.bisect_left(m2_jobs_without_j2, j1)
                    m2_new_jobs = m2_jobs_without_j2[:idx2] + [j1] + m2_jobs_without_j2[idx2:]
                    m2_new_cost = cost_of_machine_list(m2_new_jobs)
                    
                    gain = (machine_costs[m1] + machine_costs[m2]) - (m1_new_cost + m2_new_cost)
                    
                    if gain > 0:
                        assignment[j1], assignment[j2] = m2, m1
                        machine_jobs[m1] = m1_new_jobs
                        machine_jobs[m2] = m2_new_jobs
                        machine_costs[m1] = m1_new_cost
                        machine_costs[m2] = m2_new_cost
                        current_obj -= gain
                        self.update_best(assignment, current_obj)
                        improved = True
                        swap_done = True
                        break
                        
                if swap_done:
                    break

        return assignment, current_obj

    def get_canonical_assignment(self, assignment):
        mapped = {}
        next_m = 0
        canonical = [0] * self.n
        for j in range(self.n):
            m = assignment[j]
            if m not in mapped:
                mapped[m] = next_m
                next_m += 1
            canonical[j] = mapped[m]
        return canonical

    def solve_mip(self, config_assignment):
        time_left = self.time_left() - 2
        if time_left <= 0: return
        
        canonical_assignment = self.get_canonical_assignment(config_assignment)
        
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(env=env)
        
        model.setParam('TimeLimit', time_left)
        model.setParam('MIPFocus', 1) 
        
        x = model.addVars(self.m, self.n, vtype=GRB.BINARY, name="x")
        for j in range(self.n):
            for i in range(self.m):
                x[i, j].start = 1 if canonical_assignment[j] == i else 0

        for j in range(self.n):
            model.addConstr(sum(x[i, j] for i in range(self.m)) == 1)
            
        for j in range(self.n):
            if j < self.m:
                for i in range(j + 1, self.m):
                    model.addConstr(x[i, j] == 0)

        obj = gp.QuadExpr()
        obj += sum(self.w[j] * self.p[j] for j in range(self.n))
        
        coeffs = []
        vars1 = []
        vars2 = []
        for i in range(self.m):
            for j in range(self.n):
                if self.w[j] == 0: continue
                for k in range(j):
                    if self.p[k] == 0: continue
                    coeff = self.w[j] * self.p[k]
                    if coeff > 0:
                        coeffs.append(coeff)
                        vars1.append(x[i, k])
                        vars2.append(x[i, j])
                        
        if coeffs:
            obj.addTerms(coeffs, vars1, vars2)

        model.setObjective(obj, GRB.MINIMIZE)
        
        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if self.logger:
                    self.logger.log(obj_val)
                    
        model.optimize(cb)
        
        if model.solCount > 0:
            ans = [0] * self.n
            for j in range(self.n):
                for i in range(self.m):
                    if x[i, j].X > 0.5:
                        ans[j] = i
                        break
            obj_eval = self.eval_full(ans)
            self.update_best(ans, obj_eval)

    def iterated_local_search(self, start_assignment):
        curr_assignment = list(start_assignment)
        while self.time_left() > 2:
            perturbed = list(curr_assignment)
            changes = max(1, int(self.n * 0.05))
            for _ in range(changes):
                perturbed[random.randint(0, self.n - 1)] = random.randint(0, self.m - 1)
                
            new_assign, new_obj = self.local_search(perturbed)
            
            if new_obj <= self.best_obj * 1.05:
                curr_assignment = new_assign
            else:
                curr_assignment = list(self.best_assignment)

    def optimize(self):
        if self.m == 1:
            assignment = [0] * self.n
            obj = self.eval_full(assignment)
            self.update_best(assignment, obj)
            return self.best_assignment

        assignment = self.greedy_initial()
        obj = self.eval_full(assignment)
        self.update_best(assignment, obj)
        
        assignment, obj = self.local_search(assignment)
        
        num_terms = self.m * self.n * (self.n - 1) // 2
        
        if GUROBI_AVAILABLE and num_terms <= 10_000_000:
            time_for_ils = self.time_left() * 0.2
            ils_end_time = time.time() + time_for_ils
            curr_assignment = list(self.best_assignment)
            
            while time.time() < ils_end_time and self.time_left() > 5:
                perturbed = list(curr_assignment)
                changes = max(1, int(self.n * 0.05))
                for _ in range(changes):
                    perturbed[random.randint(0, self.n - 1)] = random.randint(0, self.m - 1)
                new_assign, new_obj = self.local_search(perturbed)
                if new_obj <= self.best_obj * 1.05:
                    curr_assignment = new_assign
                else:
                    curr_assignment = list(self.best_assignment)
                    
            self.solve_mip(self.best_assignment)
        else:
            self.iterated_local_search(self.best_assignment)
            
        return self.best_assignment

def build_output(assignment, p, w, j_orig, m):
    machine_jobs = [[] for _ in range(m)]
    for list_idx, mach in enumerate(assignment):
        machine_jobs[mach].append(list_idx)
        
    obj_val = 0
    machines_out = []
    
    for jobs in machine_jobs:
        load = 0
        schedule_cost = 0
        jobs_out = []
        for j_idx in jobs:
            load += p[j_idx]
            cj = load
            cost = w[j_idx] * cj
            schedule_cost += cost
            
            jobs_out.append({
                "j_orig": j_orig[j_idx],
                "p": p[j_idx],
                "w": w[j_idx],
                "C_j": int(cj)
            })
            
        obj_val += schedule_cost
        machines_out.append({
            "schedule_cost": int(schedule_cost),
            "jobs": jobs_out
        })
        
    return {
        "objective_value": float(obj_val),
        "machines": machines_out
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Machine Scheduling")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    m = data['m']
    
    jobs = []
    for idx, j_data in enumerate(data['jobs']):
        jobs.append({
            'j_orig': j_data['j'],
            'p': j_data['p'],
            'w': j_data['w']
        })
        
    jobs.sort(key=lambda x: (
        x['w']/x['p'] if x['p'] != 0 else (float('inf') if x['w'] > 0 else 0),
        x['w'],
        x['p'],
        -x['j_orig']
    ), reverse=True)
    
    p = [job['p'] for job in jobs]
    w = [job['w'] for job in jobs]
    j_orig = [job['j_orig'] for job in jobs]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    
    optimizer = Optimizer(m, p, w, logger, args.time_limit)
    best_assignment = optimizer.optimize()
    
    final_output = build_output(best_assignment, p, w, j_orig, m)
    
    with open(args.solution_path, 'w') as f:
        json.dump(final_output, f, indent=2)