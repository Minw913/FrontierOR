import sys
import json
import time
import math
import argparse
import gurobipy as gp
from gurobipy import GRB

# Attempt to import SolutionLogger as requested
try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, path, sense="maximize"):
            self.path = path
        def log(self, value):
            pass

class GurobiCallback:
    def __init__(self, logger, n, k, evaluate_func, start_time, time_limit):
        self.logger = logger
        self.n = n
        self.k = k
        self.evaluate_func = evaluate_func
        self.best_val = -1.0
        self.start_time = start_time
        self.time_limit = time_limit
        self.best_sol = []
        self._y = None
        
    def __call__(self, model, where):
        if time.time() - self.start_time >= self.time_limit:
            model.terminate()
            
        if where == GRB.Callback.MIPSOL:
            try:
                y_sol = model.cbGetSolution(self._y)
                S = [j for j in range(self.n) if y_sol[j] > 0.5]
                if len(S) <= self.k:
                    val = self.evaluate_func(S)
                    if val > self.best_val + 1e-9:
                        self.best_val = val
                        self.best_sol = S.copy()
                        if self.logger:
                            self.logger.log(val)
            except Exception:
                pass

def make_callback(cb_obj):
    def callback_func(model, where):
        cb_obj(model, where)
    return callback_func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    
    args = parser.parse_args()
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    instance_type = instance['instance_type']
    n = instance['n']
    m = instance['m']
    k = instance['k']
    data = instance['data_matrix']
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # --- Exact Evaluators ---
    def eval_loc(S):
        if not S: return 0.0
        val = 0.0
        for i in range(m):
            mx = 0.0
            for j in S:
                if data[i][j] > mx: mx = data[i][j]
            val += mx
        return val

    def eval_cov(S):
        if not S: return 0.0
        val = 0.0
        nw = instance.get('node_weights', [])
        for i in range(m):
            for j in S:
                if data[i][j] > 0:
                    val += nw[i]
                    break
        return val

    def eval_inf(S):
        if not S: return 0.0
        val = 0.0
        ew = instance.get('element_weights', [])
        for i in range(m):
            prod = 1.0
            for j in S:
                if data[i][j] > 0:
                    prod *= (1.0 - ew[j])
            val += (1.0 - prod)
        return val
        
    if instance_type == 'LOC':
        evaluator = eval_loc
    elif instance_type == 'COV':
        evaluator = eval_cov
    else:
        evaluator = eval_inf

    # --- Lightning Fast Exact Greedy Initialization ---
    S_greedy = []
    elements = set(range(n))
    
    if instance_type == 'LOC':
        loc_data = [[(i, data[i][j]) for i in range(m) if data[i][j] > 0] for j in range(n)]
        current_max = [0.0] * m
        for _ in range(k):
            best_gain = -1.0
            best_j = -1
            for j in elements:
                gain = 0.0
                for i, val in loc_data[j]:
                    if val > current_max[i]:
                        gain += val - current_max[i]
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_gain <= 1e-9: break
            S_greedy.append(best_j)
            elements.remove(best_j)
            for i, val in loc_data[best_j]:
                if val > current_max[i]:
                    current_max[i] = val
                    
            if logger: logger.log(eval_loc(S_greedy))
            
    elif instance_type == 'COV':
        nw = instance.get('node_weights', [])
        cov_data = [[i for i in range(m) if data[i][j] > 0] for j in range(n)]
        covered = [False] * m
        for _ in range(k):
            best_gain = -1.0
            best_j = -1
            for j in elements:
                gain = 0.0
                for i in cov_data[j]:
                    if not covered[i]:
                        gain += nw[i]
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_gain <= 1e-9: break
            S_greedy.append(best_j)
            elements.remove(best_j)
            for i in cov_data[best_j]:
                covered[i] = True
                
            if logger: logger.log(eval_cov(S_greedy))

    else: # INF
        ew = instance.get('element_weights', [])
        inf_data = [[i for i in range(m) if data[i][j] > 0] for j in range(n)]
        current_prod = [1.0] * m
        for _ in range(k):
            best_gain = -1.0
            best_j = -1
            for j in elements:
                pj = ew[j]
                if pj <= 0: continue
                gain = 0.0
                for i in inf_data[j]:
                    gain += current_prod[i] * pj
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_gain <= 1e-9: break
            S_greedy.append(best_j)
            elements.remove(best_j)
            pj = ew[best_j]
            for i in inf_data[best_j]:
                current_prod[i] *= (1.0 - pj)
                
            if logger: logger.log(eval_inf(S_greedy))

    best_global_sol = S_greedy.copy()
    best_global_val = evaluator(best_global_sol)
    
    # --- Exact Gurobi MIP Solving Process ---
    elapsed = time.time() - start_time
    rem_time = args.time_limit - elapsed
    
    if rem_time > 1.0:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        model = gp.Model("SetFunc", env=env)
        
        cb_obj = GurobiCallback(logger, n, k, evaluator, start_time, args.time_limit)
        cb_obj.best_val = best_global_val
        cb_obj.best_sol = best_global_sol.copy()
        
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        cb_obj._y = y
        
        for j in range(n):
            y[j].Start = 1 if j in S_greedy else 0
            
        model.addConstr(y.sum() <= k, name="k_limit")
        
        if instance_type == 'LOC':
            i_indices = [[] for _ in range(m)]
            x_indices = []
            for j in range(n):
                for i, val in loc_data[j]:
                    idx = (i, j)
                    x_indices.append(idx)
                    i_indices[i].append(idx)

            x = model.addVars(x_indices, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")

            for i in range(m):
                if i_indices[i]:
                    model.addConstr(gp.quicksum(x[idx] for idx in i_indices[i]) <= 1)

            for idx in x_indices:
                model.addConstr(x[idx] <= y[idx[1]])

            obj = gp.quicksum(data[idx[0]][idx[1]] * x[idx] for idx in x_indices)
            model.setObjective(obj, GRB.MAXIMIZE)

        elif instance_type == 'COV':
            z = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
            i_indices = [[] for _ in range(m)]
            for j in range(n):
                for i in cov_data[j]:
                    i_indices[i].append(j)

            for i in range(m):
                if i_indices[i]:
                    model.addConstr(z[i] <= gp.quicksum(y[j] for j in i_indices[i]))
                else:
                    model.addConstr(z[i] == 0)

            nw = instance.get('node_weights', [])
            obj = gp.quicksum(nw[i] * z[i] for i in range(m))
            model.setObjective(obj, GRB.MAXIMIZE)
            
        else: # INF exact configuration using Exp GenConstrs over binaries
            try: model.setParam("FuncNonlinear", 1)
            except Exception: pass
            
            try: 
                model.setParam("FuncPieces", -1)
                model.setParam("FuncPieceError", 1e-5)
            except Exception: pass
            
            v = model.addVars(m, lb=-GRB.INFINITY, ub=0.0, name="v")
            u = model.addVars(m, lb=0.0, ub=1.0, name="u")
            
            i_indices = [[] for _ in range(m)]
            for j in range(n):
                for i in inf_data[j]:
                    i_indices[i].append(j)
                    
            ew = instance.get('element_weights', [])
            for i in range(m):
                expr = gp.LinExpr()
                for j in i_indices[i]:
                    pj = ew[j]
                    if pj >= 0.999999: # Cap extremely high elements to avoid -infinity bounds issues
                        expr.addTerms(-25.0, y[j])
                    elif pj > 0:
                        expr.addTerms(math.log(1.0 - pj), y[j])
                
                model.addConstr(v[i] == expr)
                model.addGenConstrExp(v[i], u[i])
                
            model.setObjective(gp.quicksum(u[i] for i in range(m)), GRB.MINIMIZE)
            
        rem_time = args.time_limit - (time.time() - start_time)
        if rem_time > 1.0:
            model.setParam('TimeLimit', rem_time)
            model.setParam('MIPGap', 1e-5)
            model.optimize(make_callback(cb_obj))
            
            if cb_obj.best_val > best_global_val:
                best_global_val = cb_obj.best_val
                best_global_sol = cb_obj.best_sol
                
            try: 
                if model.SolCount > 0:
                    y_sol = [j for j in range(n) if y[j].X > 0.5]
                    if len(y_sol) <= k:
                        val = evaluator(y_sol)
                        if val > best_global_val + 1e-9:
                            best_global_val = val
                            best_global_sol = y_sol
            except Exception:
                pass

    # --- Write Results ---
    solution = {
        "objective_value": float(best_global_val),
        "selected_elements": [int(x) for x in best_global_sol]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()