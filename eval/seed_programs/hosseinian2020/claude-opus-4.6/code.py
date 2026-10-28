import argparse
import json
import time
import sys
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance['num_vertices']
    edges = instance['edges']
    vertices = instance['vertices']

    start_time = time.time()
    deadline = start_time + args.time_limit

    # Build adjacency structures
    adj = defaultdict(set)
    weight_map = {}
    vertex_set = set(vertices)
    
    for e in edges:
        i, j, w = e['i'], e['j'], e['weight']
        adj[i].add(j)
        adj[j].add(i)
        key = (min(i, j), max(i, j))
        weight_map[key] = w

    def clique_weight(clique):
        total = 0
        clique_list = list(clique)
        for a in range(len(clique_list)):
            for b in range(a + 1, len(clique_list)):
                key = (min(clique_list[a], clique_list[b]), max(clique_list[a], clique_list[b]))
                total += weight_map.get(key, 0)
        return total

    def is_clique(clique_set):
        clique_list = list(clique_set)
        for a in range(len(clique_list)):
            for b in range(a + 1, len(clique_list)):
                if clique_list[b] not in adj[clique_list[a]]:
                    return False
        return True

    best_clique = []
    best_value = 0.0

    def update_best(clique):
        nonlocal best_clique, best_value
        val = clique_weight(clique)
        if val > best_value:
            best_value = val
            best_clique = list(clique)
            if logger:
                logger.log(best_value)

    # Start with each single vertex
    for v in vertices:
        update_best([v])

    # Greedy approach: for each vertex, try to build a clique greedily
    # Sort vertices by weighted degree (sum of edge weights)
    vertex_wdeg = {}
    for v in vertices:
        vertex_wdeg[v] = sum(weight_map.get((min(v, u), max(v, u)), 0) for u in adj[v])

    sorted_vertices = sorted(vertices, key=lambda v: -vertex_wdeg[v])

    for start_v in sorted_vertices:
        if time.time() > deadline:
            break
        clique = [start_v]
        clique_set = {start_v}
        # Candidates: neighbors of start_v
        candidates = list(adj[start_v])
        # Sort candidates by potential weight contribution to clique
        while candidates:
            # For each candidate, compute weight it would add
            best_cand = None
            best_cand_weight = -1
            new_candidates = []
            for c in candidates:
                # Check if c is connected to all in clique
                connected = True
                added_weight = 0
                for cv in clique:
                    if c not in adj[cv]:
                        connected = False
                        break
                    key = (min(c, cv), max(c, cv))
                    added_weight += weight_map.get(key, 0)
                if connected:
                    new_candidates.append((c, added_weight))
            
            if not new_candidates:
                break
            
            # Pick the candidate with highest added weight
            new_candidates.sort(key=lambda x: -x[1])
            chosen, chosen_weight = new_candidates[0]
            clique.append(chosen)
            clique_set.add(chosen)
            
            # Update candidates: only those connected to chosen as well
            candidates = [c for c, w in new_candidates[1:] if c in adj[chosen]]
        
        update_best(clique)

    # Now try to improve using local search: swap vertices in/out
    # Start from best clique found so far
    def local_search(clique, time_limit_ls):
        nonlocal best_clique, best_value
        clique_set = set(clique)
        current_value = clique_weight(clique)
        improved = True
        
        while improved and time.time() < time_limit_ls:
            improved = False
            
            # Try adding a vertex
            # Candidates for addition: vertices connected to all in clique
            candidates_add = []
            for v in vertices:
                if v in clique_set:
                    continue
                if all(v in adj[cv] for cv in clique_set):
                    added_w = sum(weight_map.get((min(v, cv), max(v, cv)), 0) for cv in clique_set)
                    candidates_add.append((v, added_w))
            
            if candidates_add:
                candidates_add.sort(key=lambda x: -x[1])
                best_add_v, best_add_w = candidates_add[0]
                if best_add_w > 0:
                    clique_set.add(best_add_v)
                    clique.append(best_add_v)
                    current_value += best_add_w
                    improved = True
                    update_best(clique)
                    continue
            
            # Try swap: remove one vertex, add another
            best_swap_gain = 0
            best_remove = None
            best_add = None
            
            for rv in list(clique_set):
                if time.time() > time_limit_ls:
                    break
                # Weight lost by removing rv
                lost_w = sum(weight_map.get((min(rv, cv), max(rv, cv)), 0) for cv in clique_set if cv != rv)
                
                # After removing rv, what vertices could be added?
                remaining = clique_set - {rv}
                for av in vertices:
                    if av in clique_set:
                        continue
                    if all(av in adj[cv] for cv in remaining):
                        gained_w = sum(weight_map.get((min(av, cv), max(av, cv)), 0) for cv in remaining)
                        net = gained_w - lost_w
                        if net > best_swap_gain:
                            best_swap_gain = net
                            best_remove = rv
                            best_add = av
            
            if best_swap_gain > 0 and best_remove is not None:
                clique_set.remove(best_remove)
                clique.remove(best_remove)
                clique_set.add(best_add)
                clique.append(best_add)
                current_value += best_swap_gain
                improved = True
                update_best(clique)
                continue
            
            # Try removing a vertex if it improves (shouldn't normally, but for plateau escape)
            break
        
        return clique

    remaining_time = deadline - time.time()
    if remaining_time > 1:
        local_search(list(best_clique), deadline - 0.5)

    # Try Gurobi for smaller instances
    remaining_time = deadline - time.time()
    if remaining_time > 2 and num_vertices <= 500:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            m = gp.Model()
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', max(1, remaining_time - 1))
            m.setParam('Threads', 1)
            
            # Warm start with best clique
            x = {}
            best_set = set(best_clique)
            for v in vertices:
                x[v] = m.addVar(vtype=GRB.BINARY, name=f"x_{v}")
                x[v].Start = 1.0 if v in best_set else 0.0
            
            # Clique constraint: if edge (i,j) not in graph, x[i] + x[j] <= 1
            edge_set = set()
            for e in edges:
                edge_set.add((min(e['i'], e['j']), max(e['i'], e['j'])))
            
            for a_idx in range(len(vertices)):
                for b_idx in range(a_idx + 1, len(vertices)):
                    va, vb = vertices[a_idx], vertices[b_idx]
                    key = (min(va, vb), max(va, vb))
                    if key not in edge_set:
                        m.addConstr(x[va] + x[vb] <= 1)
            
            # Objective: maximize sum of weights for edges where both endpoints selected
            obj = gp.LinExpr()
            # Use linearization: for each edge, w_e * y_e where y_e = x_i * x_j
            y = {}
            for e in edges:
                i, j, w = e['i'], e['j'], e['weight']
                key = (min(i, j), max(i, j))
                y[key] = m.addVar(vtype=GRB.BINARY, name=f"y_{key[0]}_{key[1]}")
                m.addConstr(y[key] <= x[i])
                m.addConstr(y[key] <= x[j])
                m.addConstr(y[key] >= x[i] + x[j] - 1)
                obj += w * y[key]
            
            m.setObjective(obj, GRB.MAXIMIZE)
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val > best_value:
                        pass  # We'll check after optimization
            
            m.optimize()
            
            if m.SolCount > 0:
                sol_val = m.ObjVal
                if sol_val > best_value:
                    sol_clique = [v for v in vertices if x[v].X > 0.5]
                    best_value = sol_val
                    best_clique = sol_clique
                    if logger:
                        logger.log(best_value)
        except Exception as e:
            pass

    # Write output
    result = {
        "objective_value": best_value,
        "clique": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    solve()