import argparse
import json
import time
import itertools
import random
import math

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    pv = data['pairwise_values']
    
    start_time = time.time()
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # assignment: item -> knapsack (0..m-1), or -1 if unpacked
    best_assignment = {}
    best_obj = 0.0  # empty solution has value 0
    
    if logger:
        logger.log(best_obj)
    
    def compute_objective(assignment):
        """Compute objective from assignment dict (item -> knapsack)."""
        # Group items by knapsack
        knapsacks = {}
        for item, k in assignment.items():
            if k not in knapsacks:
                knapsacks[k] = []
            knapsacks[k].append(item)
        
        total = 0.0
        for k, items in knapsacks.items():
            for i in items:
                total += values[i]
            for idx1 in range(len(items)):
                for idx2 in range(idx1 + 1, len(items)):
                    total += pv[items[idx1]][items[idx2]]
        return total
    
    def compute_knapsack_weight(items_in_k):
        return sum(weights[i] for i in items_in_k)
    
    # For small instances, try Gurobi MIP
    use_mip = (n <= 60 and m <= 10) or (n * m <= 200)
    
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("qkp")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 2))
            model.setParam('Threads', 1)
            
            # x[i][k] = 1 if item i is in knapsack k
            x = {}
            for i in range(n):
                for k in range(m):
                    x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
            
            # Each item in at most one knapsack
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)
            
            # Capacity constraints
            for k in range(m):
                model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity)
            
            # For pairwise values, we need y[i,j,k] = x[i,k] * x[j,k]
            # Linearize: y[i,j,k] <= x[i,k], y[i,j,k] <= x[j,k], y[i,j,k] >= x[i,k] + x[j,k] - 1
            
            obj = gp.LinExpr()
            for i in range(n):
                for k in range(m):
                    obj += values[i] * x[i, k]
            
            # Only create y variables for pairs with non-zero pairwise value
            y = {}
            for i in range(n):
                for j in range(i + 1, n):
                    if pv[i][j] != 0:
                        for k in range(m):
                            y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")
                            model.addConstr(y[i, j, k] <= x[i, k])
                            model.addConstr(y[i, j, k] <= x[j, k])
                            model.addConstr(y[i, j, k] >= x[i, k] + x[j, k] - 1)
                            obj += pv[i][j] * y[i, j, k]
            
            model.setObjective(obj, GRB.MAXIMIZE)
            
            # Provide warm start from greedy if we have one
            model.optimize()
            
            if model.SolCount > 0:
                assignment = {}
                for i in range(n):
                    for k in range(m):
                        if x[i, k].X > 0.5:
                            assignment[i] = k
                            break
                
                obj_val = model.ObjVal
                if obj_val > best_obj:
                    best_obj = obj_val
                    best_assignment = assignment
                    if logger:
                        logger.log(best_obj)
            
            # Write solution
            output = {
                "objective_value": best_obj,
                "assignment": {str(item): knapsack for item, knapsack in best_assignment.items()}
            }
            with open(solution_path, 'w') as f:
                json.dump(output, f)
            return
            
        except Exception as e:
            pass  # Fall through to heuristic
    
    # Heuristic approach for larger instances
    # Greedy construction + local search
    
    # Data structures for efficient evaluation
    # knapsack_items[k] = set of items in knapsack k
    # item_knapsack[i] = k or -1
    # knapsack_weight[k] = total weight
    # knapsack_value[k] = total value (individual + pairwise)
    
    def greedy_construct():
        """Greedy construction heuristic."""
        knapsack_items = [set() for _ in range(m)]
        knapsack_weight = [0] * m
        item_knapsack = [-1] * n
        total_obj = 0.0
        
        # Sort items by value density (individual value / weight), descending
        item_order = sorted(range(n), key=lambda i: values[i] / max(weights[i], 1), reverse=True)
        
        for i in item_order:
            best_gain = 0  # gain of 0 means don't pack (better to leave unpacked)
            best_k = -1
            
            for k in range(m):
                if knapsack_weight[k] + weights[i] <= capacity:
                    gain = values[i]
                    for j in knapsack_items[k]:
                        gain += pv[i][j]
                    if gain > best_gain:
                        best_gain = gain
                        best_k = k
            
            if best_k >= 0:
                knapsack_items[best_k].add(i)
                knapsack_weight[best_k] += weights[i]
                item_knapsack[i] = best_k
                total_obj += best_gain
        
        return knapsack_items, knapsack_weight, item_knapsack, total_obj
    
    def compute_item_contribution(i, k, knapsack_items):
        """Compute the value contribution of item i in knapsack k."""
        contrib = values[i]
        for j in knapsack_items[k]:
            if j != i:
                contrib += pv[i][j]
        return contrib
    
    def local_search(knapsack_items, knapsack_weight, item_knapsack, total_obj, time_left):
        """Local search with move, swap, and transfer operations."""
        ls_start = time.time()
        improved = True
        
        while improved and (time.time() - ls_start) < time_left:
            improved = False
            
            # Move: move item from current position to another knapsack or unpack
            for i in range(n):
                if time.time() - ls_start > time_left:
                    break
                    
                current_k = item_knapsack[i]
                
                if current_k >= 0:
                    # Value lost by removing i from current_k
                    loss = compute_item_contribution(i, current_k, knapsack_items)
                else:
                    loss = 0
                
                best_gain_net = 0
                best_target = current_k  # stay where you are
                
                # Try unpacking
                if current_k >= 0 and (-loss) > best_gain_net:
                    best_gain_net = -loss
                    best_target = -1
                
                # Try each knapsack
                for k in range(m):
                    if k == current_k:
                        continue
                    w_available = capacity - knapsack_weight[k]
                    if weights[i] <= w_available:
                        gain = values[i]
                        for j in knapsack_items[k]:
                            gain += pv[i][j]
                        net = gain - loss
                        if net > best_gain_net:
                            best_gain_net = net
                            best_target = k
                
                if best_target != current_k and best_gain_net > 1e-9:
                    # Execute move
                    if current_k >= 0:
                        knapsack_items[current_k].remove(i)
                        knapsack_weight[current_k] -= weights[i]
                    
                    if best_target >= 0:
                        knapsack_items[best_target].add(i)
                        knapsack_weight[best_target] += weights[i]
                    
                    item_knapsack[i] = best_target
                    total_obj += best_gain_net
                    improved = True
            
            # Swap: swap two items between different knapsacks
            if not improved or True:
                for i in range(n):
                    if time.time() - ls_start > time_left:
                        break
                    k_i = item_knapsack[i]
                    
                    for j in range(i + 1, n):
                        if time.time() - ls_start > time_left:
                            break
                        k_j = item_knapsack[j]
                        
                        if k_i == k_j:
                            continue
                        
                        # Check feasibility of swap
                        if k_i >= 0 and k_j >= 0:
                            # Both packed in different knapsacks
                            new_w_ki = knapsack_weight[k_i] - weights[i] + weights[j]
                            new_w_kj = knapsack_weight[k_j] - weights[j] + weights[i]
                            if new_w_ki > capacity or new_w_kj > capacity:
                                continue
                            
                            # Compute net gain
                            # Remove i from k_i
                            loss_i = compute_item_contribution(i, k_i, knapsack_items)
                            # Remove j from k_j (note: i is still counted in k_i but we're computing independently)
                            loss_j = compute_item_contribution(j, k_j, knapsack_items)
                            
                            # After removing both, add j to k_i and i to k_j
                            # gain of j in k_i (without i)
                            gain_j_ki = values[j]
                            for item in knapsack_items[k_i]:
                                if item != i:
                                    gain_j_ki += pv[j][item]
                            
                            # gain of i in k_j (without j)
                            gain_i_kj = values[i]
                            for item in knapsack_items[k_j]:
                                if item != j:
                                    gain_i_kj += pv[i][item]
                            
                            net = gain_j_ki + gain_i_kj - loss_i - loss_j
                            
                            if net > 1e-9:
                                knapsack_items[k_i].remove(i)
                                knapsack_items[k_j].remove(j)
                                knapsack_items[k_i].add(j)
                                knapsack_items[k_j].add(i)
                                knapsack_weight[k_i] = knapsack_weight[k_i] - weights[i] + weights[j]
                                knapsack_weight[k_j] = knapsack_weight[k_j] - weights[j] + weights[i]
                                item_knapsack[i] = k_j
                                item_knapsack[j] = k_i
                                total_obj += net
                                improved = True
                        
                        elif k_i >= 0 and k_j < 0:
                            # i packed, j unpacked: try putting j in k_i and removing i
                            new_w = knapsack_weight[k_i] - weights[i] + weights[j]
                            if new_w > capacity:
                                continue
                            
                            loss_i = compute_item_contribution(i, k_i, knapsack_items)
                            gain_j = values[j]
                            for item in knapsack_items[k_i]:
                                if item != i:
                                    gain_j += pv[j][item]
                            
                            net = gain_j - loss_i
                            if net > 1e-9:
                                knapsack_items[k_i].remove(i)
                                knapsack_items[k_i].add(j)
                                knapsack_weight[k_i] = new_w
                                item_knapsack[i] = -1
                                item_knapsack[j] = k_i
                                total_obj += net
                                improved = True
                        
                        elif k_i < 0 and k_j >= 0:
                            new_w = knapsack_weight[k_j] - weights[j] + weights[i]
                            if new_w > capacity:
                                continue
                            
                            loss_j = compute_item_contribution(j, k_j, knapsack_items)
                            gain_i = values[i]
                            for item in knapsack_items[k_j]:
                                if item != j:
                                    gain_i += pv[i][item]
                            
                            net = gain_i - loss_j
                            if net > 1e-9:
                                knapsack_items[k_j].remove(j)
                                knapsack_items[k_j].add(i)
                                knapsack_weight[k_j] = new_w
                                item_knapsack[j] = -1
                                item_knapsack[i] = k_j
                                total_obj += net
                                improved = True
        
        return knapsack_items, knapsack_weight, item_knapsack, total_obj
    
    # Run greedy + local search
    knapsack_items, knapsack_weight, item_knapsack, total_obj = greedy_construct()
    
    if total_obj > best_obj:
        best_obj = total_obj
        best_assignment = {i: item_knapsack[i] for i in range(n) if item_knapsack[i] >= 0}
        if logger:
            logger.log(best_obj)
    
    remaining = time_limit - (time.time() - start_time) - 1
    if remaining > 0:
        knapsack_items, knapsack_weight, item_knapsack, total_obj = local_search(
            knapsack_items, knapsack_weight, item_knapsack, total_obj, remaining * 0.4
        )
        
        if total_obj > best_obj:
            best_obj = total_obj
            best_assignment = {i: item_knapsack[i] for i in range(n) if item_knapsack[i] >= 0}
            if logger:
                logger.log(best_obj)
    
    # Perturbation + re-optimization (iterated local search)
    def perturb(knapsack_items, knapsack_weight, item_knapsack, total_obj, strength=3):
        """Randomly move/swap some items to escape local optima."""
        ki = [set(s) for s in knapsack_items]
        kw = list(knapsack_weight)
        ik = list(item_knapsack)
        obj = total_obj
        
        packed_items = [i for i in range(n) if ik[i] >= 0]
        
        for _ in range(strength):
            if not packed_items:
                break
            action = random.random()
            
            if action < 0.4 and packed_items:
                # Remove a random item
                i = random.choice(packed_items)
                k = ik[i]
                loss = values[i]
                for j in ki[k]:
                    if j != i:
                        loss += pv[i][j]
                ki[k].remove(i)
                kw[k] -= weights[i]
                ik[i] = -1
                obj -= loss
                packed_items.remove(i)
            
            elif action < 0.7 and len(packed_items) >= 2:
                # Swap two random items from different knapsacks
                random.shuffle(packed_items)
                for idx1 in range(len(packed_items)):
                    found = False
                    for idx2 in range(idx1 + 1, len(packed_items)):
                        i = packed_items[idx1]
                        j = packed_items[idx2]
                        if ik[i] != ik[j]:
                            k_i, k_j = ik[i], ik[j]
                            new_w_ki = kw[k_i] - weights[i] + weights[j]
                            new_w_kj = kw[k_j] - weights[j] + weights[i]
                            if new_w_ki <= capacity and new_w_kj <= capacity:
                                loss_i = values[i]
                                for item in ki[k_i]:
                                    if item != i:
                                        loss_i += pv[i][item]
                                loss_j = values[j]
                                for item in ki[k_j]:
                                    if item != j:
                                        loss_j += pv[j][item]
                                
                                ki[k_i].remove(i)
                                ki[k_j].remove(j)
                                ki[k_i].add(j)
                                ki[k_j].add(i)
                                kw[k_i] = new_w_ki
                                kw[k_j] = new_w_kj
                                ik[i] = k_j
                                ik[j] = k_i
                                
                                gain_j = values[j]
                                for item in ki[k_i]:
                                    if item != j:
                                        gain_j += pv[j][item]
                                gain_i = values[i]
                                for item in ki[k_j]:
                                    if item != i:
                                        gain_i += pv[i][item]
                                
                                obj = obj - loss_i - loss_j + gain_j + gain_i
                                found = True
                                break
                    if found:
                        break
            else:
                # Move a random item to a random knapsack
                i = random.choice(list(range(n)))
                k_from = ik[i]
                k_to = random.randint(0, m - 1)
                if k_to == k_from:
                    continue
                
                if k_from >= 0:
                    loss = values[i]
                    for j in ki[k_from]:
                        if j != i:
                            loss += pv[i][j]
                    ki[k_from].remove(i)
                    kw[k_from] -= weights[i]
                    obj -= loss
                    if i in packed_items:
                        packed_items.remove(i)
                
                if kw[k_to] + weights[i] <= capacity:
                    gain = values[i]
                    for j in ki[k_to]:
                        gain += pv[i][j]
                    ki[k_to].add(i)
                    kw[k_to] += weights[i]
                    ik[i] = k_to
                    obj += gain
                    packed_items.append(i)
                else:
                    ik[i] = -1
        
        return ki, kw, ik, obj
    
    # ILS loop
    iteration = 0
    while True:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 1
        if remaining < 0.5:
            break
        
        iteration += 1
        strength = random.randint(2, max(2, min(n // 3, 8)))
        
        p_ki, p_kw, p_ik, p_obj = perturb(
            knapsack_items, knapsack_weight, item_knapsack, total_obj, strength
        )
        
        ls_time = min(remaining * 0.5, max(remaining * 0.1, 2.0))
        p_ki, p_kw, p_ik, p_obj = local_search(p_ki, p_kw, p_ik, p_obj, ls_time)
        
        # Accept if better (or sometimes accept worse with SA-like criterion)
        accept = False
        if p_obj > total_obj:
            accept = True
        else:
            # Accept worse with some probability
            diff = p_obj - total_obj
            temp = max(1.0, abs(best_obj) * 0.01) if best_obj != 0 else 10.0
            if random.random() < math.exp(diff / temp):
                accept = True
        
        if accept:
            knapsack_items = p_ki
            knapsack_weight = p_kw
            item_knapsack = p_ik
            total_obj = p_obj
        
        if total_obj > best_obj:
            best_obj = total_obj
            best_assignment = {i: item_knapsack[i] for i in range(n) if item_knapsack[i] >= 0}
            if logger:
                logger.log(best_obj)
    
    # Try Gurobi for medium instances if time remains
    remaining = time_limit - (time.time() - start_time) - 2
    if remaining > 5 and n <= 100 and m <= 15:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("qkp")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining))
            model.setParam('Threads', 1)
            
            x = {}
            for i in range(n):
                for k in range(m):
                    x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
            
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)
            
            for k in range(m):
                model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity)
            
            obj = gp.LinExpr()
            for i in range(n):
                for k in range(m):
                    obj += values[i] * x[i, k]
            
            y = {}
            for i in range(n):
                for j in range(i + 1, n):
                    if pv[i][j] != 0:
                        for k in range(m):
                            y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")
                            model.addConstr(y[i, j, k] <= x[i, k])
                            model.addConstr(y[i, j, k] <= x[j, k])
                            model.addConstr(y[i, j, k] >= x[i, k] + x[j, k] - 1)
                            obj += pv[i][j] * y[i, j, k]
            
            model.setObjective(obj, GRB.MAXIMIZE)
            
            # Warm start
            for i in range(n):
                for k in range(m):
                    if i in best_assignment and best_assignment[i] == k:
                        x[i, k].Start = 1.0
                    else:
                        x[i, k].Start = 0.0
            
            model.optimize()
            
            if model.SolCount > 0 and model.ObjVal > best_obj:
                assignment = {}
                for i in range(n):
                    for k in range(m):
                        if x[i, k].X > 0.5:
                            assignment[i] = k
                            break
                best_obj = model.ObjVal
                best_assignment = assignment
                if logger:
                    logger.log(best_obj)
        except Exception:
            pass
    
    # Write solution
    output = {
        "objective_value": best_obj,
        "assignment": {str(item): knapsack for item, knapsack in best_assignment.items()}
    }
    with open(solution_path, 'w') as f:
        json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)