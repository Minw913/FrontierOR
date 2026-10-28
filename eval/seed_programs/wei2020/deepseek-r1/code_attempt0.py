import json
import argparse
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    n = instance['n']
    c = instance['c']
    weights = instance['weights']
    incompatible_pairs = instance.get('incompatible_pairs', [])
    
    adj = [set() for _ in range(n)]
    for pair in incompatible_pairs:
        i, j = pair
        adj[i].add(j)
        adj[j].add(i)
    
    def heuristic_bin_packing(n, c, weights, adj):
        items = list(range(n))
        items.sort(key=lambda i: (weights[i], i), reverse=True)
        
        bins = [] 
        bin_contents = [] 
        bin_weights = [] 
        
        for item in items:
            found = False
            for idx in range(len(bins)):
                if bin_weights[idx] + weights[item] <= c:
                    if bin_contents[idx].isdisjoint(adj[item]):
                        bins[idx].append(item)
                        bin_contents[idx].add(item)
                        bin_weights[idx] += weights[item]
                        found = True
                        break
            if not found:
                bins.append([item])
                bin_contents.append({item})
                bin_weights.append(weights[item])
                
        return bins, len(bins)
    
    bins_heuristic, obj_heuristic = heuristic_bin_packing(n, c, weights, adj)
    
    if logger:
        logger.log(obj_heuristic)
    
    time_so_far = time.time() - start_time
    time_remaining = args.time_limit - time_so_far
    
    if time_remaining <= 0:
        output_solution = {
            "objective_value": float(obj_heuristic),
            "bins": bins_heuristic
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output_solution, f)
        return
    
    skip_MIP = False
    if n * obj_heuristic > 1000000:
        skip_MIP = True
    elif incompatible_pairs and len(incompatible_pairs) * obj_heuristic > 1000000:
        skip_MIP = True
    
    if skip_MIP:
        output_solution = {
            "objective_value": float(obj_heuristic),
            "bins": bins_heuristic
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output_solution, f)
        return
    
    try:
        import gurobipy
    except ImportError:
        output_solution = {
            "objective_value": float(obj_heuristic),
            "bins": bins_heuristic
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output_solution, f)
        return
    
    try:
        model = gurobipy.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(1, time_remaining))
        model.setParam('Threads', 1)
        
        x = {}
        for i in range(n):
            for k in range(obj_heuristic):
                x[i, k] = model.addVar(vtype=gurobipy.GRB.BINARY, name=f"x_{i}_{k}")
        
        y = {}
        for k in range(obj_heuristic):
            y[k] = model.addVar(vtype=gurobipy.GRB.BINARY, name=f"y_{k}")
        
        model.setObjective(gurobipy.quicksum(y[k] for k in range(obj_heuristic)), gurobipy.GRB.MINIMIZE)
        
        for i in range(n):
            model.addConstr(gurobipy.quicksum(x[i, k] for k in range(obj_heuristic)) == 1)
        
        for k in range(obj_heuristic):
            model.addConstr(gurobipy.quicksum(weights[i] * x[i, k] for i in range(n)) <= c * y[k])
        
        if incompatible_pairs:
            for pair in incompatible_pairs:
                i, j = pair
                for k in range(obj_heuristic):
                    model.addConstr(x[i, k] + x[j, k] <= 1)
        
        for k in range(obj_heuristic - 1):
            model.addConstr(y[k] >= y[k+1])
        
        for k in range(obj_heuristic):
            y[k].Start = 1.0
            for i in range(n):
                if i in bins_heuristic[k]:
                    x[i, k].Start = 1.0
                else:
                    x[i, k].Start = 0.0
        
        if logger:
            class CallbackState:
                def __init__(self, logger, best_obj):
                    self.logger = logger
                    self.best_obj = best_obj
            
            cb_state = CallbackState(logger, obj_heuristic)
            
            def mycallback(model, where):
                if where == gurobipy.GRB.Callback.MIPSOL:
                    obj = model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ)
                    if obj < cb_state.best_obj:
                        cb_state.best_obj = obj
                        cb_state.logger.log(obj)
            
            model.optimize(mycallback)
        else:
            model.optimize()
        
        if model.status == gurobipy.GRB.OPTIMAL or (model.status == gurobipy.GRB.TIME_LIMIT and model.SolCount > 0):
            objective_value = float(model.ObjVal)
            bins_solution = []
            for k in range(obj_heuristic):
                if y[k].X < 0.5:
                    break
                bin_items = []
                for i in range(n):
                    if x[i, k].X > 0.5:
                        bin_items.append(i)
                bins_solution.append(bin_items)
            output_solution = {
                "objective_value": objective_value,
                "bins": bins_solution
            }
        else:
            output_solution = {
                "objective_value": float(obj_heuristic),
                "bins": bins_heuristic
            }
    except Exception as e:
        output_solution = {
            "objective_value": float(obj_heuristic),
            "bins": bins_heuristic
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_solution, f)

if __name__ == "__main__":
    main()