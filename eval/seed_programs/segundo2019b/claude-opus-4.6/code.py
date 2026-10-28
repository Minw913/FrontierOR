import argparse
import json
import time
import sys
import random
import os
from solution_logger import SolutionLogger


def load_instance(instance_path):
    """Try multiple approaches to load the instance data."""
    # First try standard json.load
    try:
        with open(instance_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass
    except MemoryError:
        pass
    
    # Try reading with a larger buffer
    try:
        with open(instance_path, 'r', buffering=1024*1024*64) as f:
            content = f.read()
        return json.loads(content)
    except Exception:
        pass
    
    # Try ijson for streaming parse if available
    try:
        import ijson
        result = {}
        with open(instance_path, 'rb') as f:
            parser = ijson.items(f, '')
            for item in parser:
                result = item
                break
        return result
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try orjson if available
    try:
        import orjson
        with open(instance_path, 'rb') as f:
            return orjson.loads(f.read())
    except ImportError:
        pass
    except Exception:
        pass
    
    # Last resort: read in chunks
    with open(instance_path, 'r') as f:
        content = ""
        while True:
            chunk = f.read(1024 * 1024 * 16)
            if not chunk:
                break
            content += chunk
    return json.loads(content)


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    instance = load_instance(instance_path)
    
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    vertices = graph['vertices']
    edges = graph['edges']
    vertex_weights = instance['vertex_weights']
    
    n = num_vertices
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build adjacency sets using indices
    adj_idx = [set() for _ in range(n)]
    for u, v in edges:
        ui, vi = v_to_idx[u], v_to_idx[v]
        adj_idx[ui].add(vi)
        adj_idx[vi].add(ui)
    
    # Convert to frozensets for faster intersection
    adj_frozen = [frozenset(s) for s in adj_idx]
    
    w = [vertex_weights[i] for i in range(n)]
    
    best_clique_idx = []
    best_weight = 0
    
    def update_best(clique_idx):
        nonlocal best_clique_idx, best_weight
        wt = sum(w[i] for i in clique_idx)
        if wt > best_weight:
            best_weight = wt
            best_clique_idx = list(clique_idx)
            if logger:
                logger.log(best_weight)
    
    def remaining_time():
        return time_limit - (time.time() - start_time)
    
    def is_connected_to_all(v, clique_set):
        return clique_set.issubset(adj_idx[v])
    
    def greedy_clique(order):
        clique = []
        clique_set = set()
        for v in order:
            if time.time() - start_time > time_limit - 0.5:
                break
            if clique_set.issubset(adj_idx[v]):
                clique.append(v)
                clique_set.add(v)
        return clique
    
    # Initial greedy solutions
    random.seed(42)
    
    # By weight descending
    order = sorted(range(n), key=lambda i: w[i], reverse=True)
    update_best(greedy_clique(order))
    
    # By degree descending then weight
    if remaining_time() > 1:
        degs = [len(adj_idx[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: (degs[i], w[i]), reverse=True)
        update_best(greedy_clique(order))
    
    # By weighted degree
    if remaining_time() > 1:
        wdeg = [sum(w[j] for j in adj_idx[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: wdeg[i], reverse=True)
        update_best(greedy_clique(order))
    
    # By weight*degree
    if remaining_time() > 1:
        order = sorted(range(n), key=lambda i: w[i] * (len(adj_idx[i]) + 1), reverse=True)
        update_best(greedy_clique(order))
    
    # Branch and bound
    num_edges_count = graph['num_edges']
    
    if remaining_time() > 2 and n <= 3000:
        try:
            call_count = [0]
            
            def bnb(candidates, clique, clique_weight, clique_set):
                nonlocal best_weight
                call_count[0] += 1
                if call_count[0] % 5000 == 0:
                    if time.time() - start_time > time_limit - 1:
                        return
                
                if not candidates:
                    if clique_weight > best_weight:
                        update_best(list(clique))
                    return
                
                # Upper bound
                ub = clique_weight + sum(w[v] for v in candidates)
                if ub <= best_weight:
                    return
                
                cands = sorted(candidates, key=lambda v: w[v], reverse=True)
                
                running_sum = sum(w[v] for v in cands)
                
                for i, v in enumerate(cands):
                    if time.time() - start_time > time_limit - 1:
                        return
                    if clique_weight + running_sum <= best_weight:
                        return
                    
                    new_candidates = [u for u in cands[i + 1:] if u in adj_idx[v]]
                    clique.append(v)
                    clique_set.add(v)
                    bnb(new_candidates, clique, clique_weight + w[v], clique_set)
                    clique.pop()
                    clique_set.discard(v)
                    running_sum -= w[v]
            
            bnb(list(range(n)), [], 0, set())
        except Exception:
            pass
    
    # Gurobi for instances where complement is manageable
    num_complement_edges = n * (n - 1) // 2 - num_edges_count
    use_gurobi = False
    if remaining_time() > 3:
        if n <= 5000 and num_complement_edges <= 500000:
            use_gurobi = True
        elif n <= 2000:
            use_gurobi = True
    
    if use_gurobi:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("max_weight_clique")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining_time() - 1))
            model.setParam('Threads', 1)
            
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            model.setObjective(
                gp.quicksum(w[i] * x[i] for i in range(n)),
                GRB.MAXIMIZE
            )
            
            # Set warm start
            best_set = set(best_clique_idx)
            for i in range(n):
                x[i].Start = 1.0 if i in best_set else 0.0
            
            edge_set = set()
            for u, v in edges:
                ui, vi = v_to_idx[u], v_to_idx[v]
                edge_set.add((min(ui, vi), max(ui, vi)))
            
            for i in range(n):
                if time.time() - start_time > time_limit - 2:
                    break
                for j in range(i + 1, n):
                    if (i, j) not in edge_set:
                        model.addConstr(x[i] + x[j] <= 1)
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    vals = model.cbGetSolution([x[i] for i in range(n)])
                    clique = [i for i in range(n) if vals[i] > 0.5]
                    update_best(clique)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                clique = [i for i in range(n) if x[i].X > 0.5]
                update_best(clique)
        except Exception:
            pass
    
    # Local search with random restarts
    while time.time() - start_time < time_limit - 0.5:
        order = list(range(n))
        random.shuffle(order)
        clique = greedy_clique(order)
        current_clique = set(clique)
        current_weight = sum(w[v] for v in current_clique)
        if current_weight > best_weight:
            update_best(list(current_clique))
        
        improved = True
        while improved and time.time() - start_time < time_limit - 0.5:
            improved = False
            clique_list = sorted(current_clique, key=lambda v: w[v])
            for remove_v in clique_list[:min(len(clique_list), 3)]:
                if time.time() - start_time > time_limit - 0.5:
                    break
                new_clique = set(current_clique) - {remove_v}
                new_wt = current_weight - w[remove_v]
                add_cands = sorted([v for v in range(n) if v not in new_clique and new_clique.issubset(adj_idx[v])], key=lambda v: w[v], reverse=True)
                for v in add_cands:
                    if new_clique.issubset(adj_idx[v]):
                        new_clique.add(v)
                        new_wt += w[v]
                if new_wt > current_weight:
                    current_clique = new_clique
                    current_weight = new_wt
                    improved = True
                    if current_weight > best_weight:
                        update_best(list(current_clique))
                    break
    
    if not best_clique_idx and n > 0:
        best_v = max(range(n), key=lambda i: w[i])
        best_clique_idx = [best_v]
        best_weight = w[best_v]
        if logger:
            logger.log(best_weight)
    
    clique_vertices = [vertices[i] for i in best_clique_idx]
    
    solution = {
        "objective_value": float(best_weight),
        "clique_vertices": clique_vertices
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()