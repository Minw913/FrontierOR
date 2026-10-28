import sys
import json
import time
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    # Fallback if solution_logger is not found
    class SolutionLogger:
        def __init__(self, log_path, sense="minimize"):
            pass
        def log(self, obj):
            pass

class BiObjectiveCFLPSolver:
    def __init__(self, instance, time_limit, logger):
        self.instance = instance
        self.time_limit = time_limit
        self.start_time = time.time()
        self.logger = logger
        self.best_wsum = float('inf')
        self.found_solutions = []
        
        self.num_f = instance["num_facilities"]
        self.num_c = instance["num_customers"]
        self.demands = instance["demands"]
        self.capacities = instance["capacities"]
        self.F1 = instance["fixed_costs_obj1"]
        self.F2 = instance["fixed_costs_obj2"]
        self.A1 = instance["assignment_costs_obj1"]
        self.A2 = instance["assignment_costs_obj2"]

    def get_remaining_time(self):
        return max(0.0, self.time_limit - (time.time() - self.start_time))

    def add_sols(self, sols):
        # Cache solutions and update log if a new best weighted sum is discovered
        for sol in sols:
            wsum = 0.5 * sol["z1"] + 0.5 * sol["z2"]
            if wsum < self.best_wsum - 1e-4:
                self.best_wsum = wsum
                if self.logger:
                    self.logger.log(wsum)
            self.found_solutions.append(sol)

    def phase1_callback(self, model_cb, where):
        # Call logging mechanism whenever Gurobi finds a new incumbent solution in Phase 1
        if where == GRB.Callback.MIPSOL:
            try:
                # Phase 1 optimizes Z1 + Z2, so combined objective weight is 0.5 * obj
                obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
                wsum = 0.5 * obj
                if wsum < self.best_wsum - 1e-4:
                    self.best_wsum = wsum
                    if self.logger:
                        self.logger.log(wsum)
            except Exception:
                pass

    def extract_all_solutions(self, model, y, x):
        # Retrieve all mixed-integer solutions found in Gurobi's solution pool
        sols = []
        if model.SolCount > 0:
            for i in range(model.SolCount):
                model.setParam('SolutionNumber', i)
                y_val = [int(round(y[f].Xn)) for f in range(self.num_f)]
                x_val = [[int(round(x[f, c].Xn)) for c in range(self.num_c)] for f in range(self.num_f)]
                
                # Manually evaluate objective values against explicit parameters
                z1_val = sum(self.F1[f] * y_val[f] for f in range(self.num_f)) + \
                         sum(self.A1[f][c] * x_val[f][c] for f in range(self.num_f) for c in range(self.num_c))
                z2_val = sum(self.F2[f] * y_val[f] for f in range(self.num_f)) + \
                         sum(self.A2[f][c] * x_val[f][c] for f in range(self.num_f) for c in range(self.num_c))
                         
                assignments = {}
                for c in range(self.num_c):
                    for f in range(self.num_f):
                        if x_val[f][c] == 1:
                            assignments[str(c)] = f
                            break
                            
                open_facs = [f for f in range(self.num_f) if y_val[f] == 1]
                
                sols.append({
                    "z1": z1_val,
                    "z2": z2_val,
                    "open_facilities": open_facs,
                    "assignments": assignments
                })
        return sols

    def solve(self):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        
        # Binary assignment variables and binary facility open variables
        y = model.addVars(self.num_f, vtype=GRB.BINARY, name="y")
        x = model.addVars(self.num_f, self.num_c, vtype=GRB.BINARY, name="x")
        
        # Each customer must be strictly assigned to 1 facility
        for c in range(self.num_c):
            model.addConstr(gp.quicksum(x[f, c] for f in range(self.num_f)) == 1)
            
        # Total demand from all customers assigned to an open facility respect capacity 
        # and customers cannot be assigned to closed facilities 
        for f in range(self.num_f):
            model.addConstr(
                gp.quicksum(self.demands[c] * x[f, c] for c in range(self.num_c)) <= 
                self.capacities[f] * y[f]
            )
            # Tightening structure constraints (link variables)
            for c in range(self.num_c):
                model.addConstr(x[f, c] <= y[f])
                
        # Independent linear expressions for each isolated scenario map
        Z1 = gp.quicksum(self.F1[f] * y[f] for f in range(self.num_f)) + \
             gp.quicksum(self.A1[f][c] * x[f, c] for f in range(self.num_f) for c in range(self.num_c))
        Z2 = gp.quicksum(self.F2[f] * y[f] for f in range(self.num_f)) + \
             gp.quicksum(self.A2[f][c] * x[f, c] for f in range(self.num_f) for c in range(self.num_c))
             
        # Phase 1: Minimize the single weighted sum to obtain strong upper bounds natively
        # Optimizing Z1 + Z2 corresponds directly to identifying optimal 0.5 * Z1 + 0.5 * Z2 configurations
        model.setObjective(Z1 + Z2, GRB.MINIMIZE)
        rem_time = self.get_remaining_time()
        
        # Enforce conservative initial time bounds for Phase 1 to guarantee multi-objective time later
        p1_time = min(rem_time, max(10.0, rem_time * 0.4))
        
        model.setParam('TimeLimit', p1_time)
        model.setParam('Threads', 1) # Problem specification restricts execution to a single core
        
        # Due to integer cost metrics, evaluating single objectives with 0.99 tolerances closes 
        # optimizations safely for strict exact formulations
        model.setParam('MIPGapAbs', 0.99)
        model.setParam('MIPGap', 0.0)
        
        model.optimize(self.phase1_callback)
        self.add_sols(self.extract_all_solutions(model, y, x))
        
        # Phase 2: Complete the Pareto front utilizing multi-objective lexicographic epsilon-constraints
        model.ModelSense = GRB.MINIMIZE
        model.NumObj = 2
        # Prioritization builds exact extreme non-dominated segments unconditionally, descending strictly
        model.setObjectiveN(Z1, index=0, priority=2, name="obj1")
        model.setObjectiveN(Z2, index=1, priority=1, name="obj2")
        
        z2_constr = None
        z2_ub = None
        
        while True:
            rem_time = self.get_remaining_time()
            if rem_time <= 0:
                break
                
            model.setParam('TimeLimit', rem_time)
            model.optimize()
            
            sols = self.extract_all_solutions(model, y, x)
            self.add_sols(sols)
            
            if model.Status == GRB.INFEASIBLE:
                break
            elif model.Status == GRB.TIME_LIMIT:
                break
            elif model.Status == GRB.OPTIMAL or model.SolCount > 0:
                if len(sols) > 0:
                    best_sol = sols[0]
                    # Establish next strict epsilon interval bounds relative to Z2 discoveries
                    next_z2_ub = best_sol["z2"] - 1
                    
                    if z2_ub is not None and next_z2_ub >= z2_ub:
                        next_z2_ub = z2_ub - 1
                        
                    z2_ub = next_z2_ub
                    if z2_constr is not None:
                        model.remove(z2_constr)
                    z2_constr = model.addConstr(Z2 <= z2_ub)
                else:
                    break
            else:
                break
                
        return self.filter_nondominated(self.found_solutions)
        
    def filter_nondominated(self, sols):
        # Retain optimally dominant sets relative to all logged historical searches unconditionally
        unique_points = {}
        for sol in sols:
            pt = (sol["z1"], sol["z2"])
            if pt not in unique_points:
                unique_points[pt] = sol
                
        filtered_sols = []
        for pt, sol in unique_points.items():
            dominated = False
            for other_pt in unique_points.keys():
                if pt == other_pt:
                    continue
                if other_pt[0] <= pt[0] and other_pt[1] <= pt[1] and \
                   (other_pt[0] < pt[0] or other_pt[1] < pt[1]):
                    dominated = True
                    break
            if not dominated:
                filtered_sols.append(sol)
                
        # Sort sequentially mapping the final pareto configuration set linearly by Objective 1
        filtered_sols.sort(key=lambda x: x["z1"])
        
        if not filtered_sols:
            return {
                "objective_value": 0.0,
                "pareto_front": [],
                "solutions": []
            }
            
        pareto_front = [[s["z1"], s["z2"]] for s in filtered_sols]
        final_best_wsum = min(0.5 * s["z1"] + 0.5 * s["z2"] for s in filtered_sols)
        
        return {
            "objective_value": final_best_wsum,
            "pareto_front": pareto_front,
            "solutions": filtered_sols
        }

def main():
    parser = argparse.ArgumentParser(description='Bi-Objective Capacitated Facility Location')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log JSONL')
    args = parser.parse_args()
    
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    solver = BiObjectiveCFLPSolver(instance, args.time_limit, logger)
    output_data = solver.solve()
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()