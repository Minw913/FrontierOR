import sys
import time
import json
import random
import argparse
import itertools
import collections
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    max_time = args.time_limit - 1.0

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    nodes = instance['graph']['vertices']
    edges = instance['graph']['edges']
    settings = instance['problem_settings']

    ctype = settings['relaxed_clique_type'].lower()
    req_conn = settings['connectivity_required']
    decomp_type = settings['decomposition_type'].lower()

    param = None
    for k, v in settings.items():
        if k.startswith('parameter_'):
            param = v
            break
    if param is None:
        param = 1

    nodes_array = list(nodes)
    adj_list = {u: [] for u in nodes}
    adj_set = {u: set() for u in nodes}
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    node_degrees = {u: len(adj_list[u]) for u in nodes}
    cands_by_degree = sorted(nodes_array, key=lambda n: node_degrees[n], reverse=True)

    best_obj = float(len(nodes))
    best_sol = [[u] for u in nodes]
    if logger:
        logger.log(best_obj)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    orig_dist = {}
    if ctype == 's-clique':
        for i in nodes:
            orig_dist[i] = {j: float('inf') for j in nodes}
            orig_dist[i][i] = 0
            q = collections.deque([i])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if orig_dist[i][v] == float('inf'):
                        orig_dist[i][v] = orig_dist[i][u] + 1
                        q.append(v)

    def is_valid_group(nodes_list, nodes_set):
        n_len = len(nodes_list)
        if n_len <= 1:
            return True

        if req_conn:
            if n_len > 1:
                visited = {nodes_list[0]}
                queue = [nodes_list[0]]
                head = 0
                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    for neighbor in adj_list[curr]:
                        if neighbor in nodes_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(visited) != n_len:
                    return False

        if ctype == 's-plex':
            min_deg = n_len - param
            for u in nodes_list:
                if len(adj_set[u].intersection(nodes_set)) < min_deg:
                    return False

        elif ctype == 'k-core':
            for u in nodes_list:
                if len(adj_set[u].intersection(nodes_set)) < param:
                    return False

        elif ctype == 's-clique':
            for i in range(n_len):
                u = nodes_list[i]
                for j in range(i+1, n_len):
                    v = nodes_list[j]
                    if orig_dist[u][v] > param:
                        return False

        elif ctype in ('s-club', 's_club'):
            for u in nodes_list:
                visited = {u: 0}
                queue = [u]
                head = 0
                max_d = 0
                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    d = visited[curr]
                    if d < param:
                        for neighbor in adj_list[curr]:
                            if neighbor in nodes_set and neighbor not in visited:
                                visited[neighbor] = d + 1
                                queue.append(neighbor)
                                max_d = max(max_d, d + 1)
                if len(visited) != n_len or max_d > param:
                    return False

        elif ctype in ('gamma-quasi-clique', 'gamma_quasi_clique'):
            edges_count = sum(len(adj_set[u].intersection(nodes_set)) for u in nodes_list) // 2
            max_e = n_len * (n_len - 1) / 2.0
            if edges_count < param * max_e:
                return False

        elif ctype in ('s-defective clique', 's-defective-clique', 's_defective_clique'):
            edges_count = sum(len(adj_set[u].intersection(nodes_set)) for u in nodes_list) // 2
            max_e = (n_len * (n_len - 1)) // 2
            if max_e - edges_count > param:
                return False

        elif ctype in ('k-block', 's-bundle'):
            H = G.subgraph(nodes_list)
            conn = nx.node_connectivity(H)
            if ctype == 'k-block' and conn < param:
                return False
            if ctype == 's-bundle' and conn < n_len - param:
                return False

        return True

    valid_groups = set()
    valid_groups_list = []

    def add_valid(grp):
        if grp not in valid_groups:
            valid_groups.add(grp)
            valid_groups_list.append(grp)

    # Initialize pool with singletons, pairs, and triplets to seed search
    for u in nodes:
        add_valid((u,))

    for u in nodes:
        for v in adj_list[u]:
            if u < v:
                if is_valid_group([u, v], {u, v}):
                    add_valid(tuple(sorted([u, v])))

    if len(nodes) <= 50:
        for combo in itertools.combinations(nodes, 3):
            if is_valid_group(list(combo), set(combo)):
                add_valid(tuple(sorted(combo)))

    node_covers = {u: 0 for u in nodes}
    node_weights = {u: 1.0 for u in nodes}

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("RelaxMaster", env=env)
    model.setParam("Threads", 1)

    cover_constrs = {}
    for u in nodes:
        if decomp_type == 'partitioning':
            cover_constrs[u] = model.addConstr(gp.LinExpr() == 1, name=f"cov_{u}")
        else:
            cover_constrs[u] = model.addConstr(gp.LinExpr() >= 1, name=f"cov_{u}")

    x = []

    def add_columns_to_model(new_groups_batch):
        for grp in new_groups_batch:
            col = gp.Column()
            for u in grp:
                col.addTerms(1.0, cover_constrs[u])
                node_covers[u] += 1
            var = model.addVar(obj=1.0, vtype=GRB.BINARY, column=col)
            if len(grp) == 1:
                var.Start = 1.0
            else:
                var.Start = 0.0
            x.append((var, grp))

    processed_groups = 0
    add_columns_to_model(valid_groups_list)
    processed_groups = len(valid_groups_list)

    iter_count = 0
    while time.time() - start_time < max_time:
        iter_count += 1
        
        gen_time_budget = 1.0 + min(3.0, iter_count * 0.5)
        rem_time = max_time - (time.time() - start_time)
        if rem_time < 0.5: break
        
        gen_time_budget = min(gen_time_budget, rem_time * 0.5)
        
        weights_array = []
        for u in nodes_array:
            node_weights[u] = 100.0 / (1.0 + node_covers[u])
            weights_array.append(node_weights[u])
            
        cands_by_weight = sorted(nodes_array, key=lambda n: node_weights[n], reverse=True)

        gen_start = time.time()
        while time.time() - gen_start < gen_time_budget:
            if len(valid_groups) > 250000:
                break
                
            if valid_groups_list and random.random() < 0.5:
                start_grp = random.choice(valid_groups_list)
                curr_list = list(start_grp)
                curr_set = set(start_grp)
            else:
                u = random.choices(nodes_array, weights=weights_array)[0]
                curr_list = [u]
                curr_set = {u}

            strat = random.random()
            if strat < 0.3:
                cands = nodes_array[:]
                random.shuffle(cands)
            elif strat < 0.6:
                cands = cands_by_weight[:]
            elif strat < 0.8:
                cands = cands_by_degree[:]
            else:
                cands = nodes_array[:]
                cands.sort(key=lambda n: len(adj_set[n].intersection(curr_set)), reverse=True)

            for v in cands:
                if v in curr_set: continue
                if req_conn:
                    if not any(n in curr_set for n in adj_list[v]):
                        continue
                
                curr_list.append(v)
                curr_set.add(v)
                if is_valid_group(curr_list, curr_set):
                    add_valid(tuple(sorted(curr_list)))
                else:
                    curr_list.pop()
                    curr_set.remove(v)
                    
            if decomp_type == 'partitioning' and len(curr_list) > 2:
                if random.random() < 0.3:
                    rem_node = random.choice(curr_list)
                    sub_list = [n for n in curr_list if n != rem_node]
                    sub_set = set(sub_list)
                    if is_valid_group(sub_list, sub_set):
                        add_valid(tuple(sorted(sub_list)))

        if len(valid_groups) > 250000:
            rem_time = max_time - (time.time() - start_time)
            if rem_time > 0.5:
                model.setParam("TimeLimit", rem_time)
                model.optimize()
                if model.SolCount > 0:
                    obj = model.ObjVal
                    if obj < best_obj - 0.5:
                        best_obj = obj
                        best_sol = [list(grp) for var, grp in x if var.X > 0.5]
                        if logger: logger.log(best_obj)
            break

        new_batch = valid_groups_list[processed_groups:]
        if new_batch:
            add_columns_to_model(new_batch)
            processed_groups = len(valid_groups_list)

        rem_time = max_time - (time.time() - start_time)
        if rem_time < 0.5: break
        
        model.setParam("TimeLimit", min(5.0, rem_time))
        model.optimize()
        
        if model.SolCount > 0:
            obj = model.ObjVal
            if obj < best_obj - 0.5:
                best_obj = obj
                best_sol = [list(grp) for var, grp in x if var.X > 0.5]
                if logger: logger.log(best_obj)
                
        if best_obj <= 1.5:
            if best_obj == 1.0 or len(nodes) == 1:
                break

    with open(args.solution_path, 'w') as f:
        json.dump({
            "objective_value": float(best_obj),
            "solution": best_sol
        }, f)

if __name__ == "__main__":
    main()