import json
import argparse
import time
import sys
from collections import defaultdict, deque

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    nodes = instance['network']['nodes']
    arcs = instance['network']['arcs']
    commodities = instance['commodities']['commodity_list']

    num_nodes = len(nodes)
    num_arcs = len(arcs)
    num_commodities = len(commodities)

    # Build adjacency structures
    adj_out = defaultdict(list)  # node -> list of arcs leaving node
    adj_in = defaultdict(list)   # node -> list of arcs entering node
    arc_by_id = {}
    for arc in arcs:
        adj_out[arc['from_node']].append(arc)
        adj_in[arc['to_node']].append(arc)
        arc_by_id[arc['arc_id']] = arc

    node_set = set(nodes)

    # BFS shortest path
    def find_path_bfs(origin, destination):
        if origin == destination:
            return []
        visited = {origin}
        queue = deque([(origin, [])])
        while queue:
            node, path = queue.popleft()
            for arc in adj_out[node]:
                to_node = arc['to_node']
                if to_node == destination:
                    return path + [arc]
                if to_node not in visited:
                    visited.add(to_node)
                    queue.append((to_node, path + [arc]))
        return None

    # Compute reachability for pruning
    def compute_reachable_from(start):
        reachable = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for arc in adj_out[node]:
                if arc['to_node'] not in reachable:
                    reachable.add(arc['to_node'])
                    queue.append(arc['to_node'])
        return reachable

    def compute_can_reach(target):
        can_reach = {target}
        queue = deque([target])
        while queue:
            node = queue.popleft()
            for arc in adj_in[node]:
                if arc['from_node'] not in can_reach:
                    can_reach.add(arc['from_node'])
                    queue.append(arc['from_node'])
        return can_reach

    # Precompute reachability for each commodity
    comm_reachable = {}
    for comm in commodities:
        k = comm['commodity_id']
        reach_from = compute_reachable_from(comm['origin'])
        reach_to = compute_can_reach(comm['destination'])
        comm_reachable[k] = reach_from & reach_to

    # Greedy heuristic
    def greedy_solve(commodity_order):
        remaining_cap = {arc['arc_id']: arc['capacity'] for arc in arcs}
        solution = {}
        total_cost = 0

        for comm in commodity_order:
            k = comm['commodity_id']
            if comm['origin'] == comm['destination']:
                solution[k] = {
                    'commodity_id': k,
                    'rejected': False,
                    'path_arcs': []
                }
                continue

            path = find_path_bfs(comm['origin'], comm['destination'])
            if path is not None:
                feasible = all(remaining_cap[a['arc_id']] >= comm['demand'] for a in path)
                if feasible:
                    for a in path:
                        remaining_cap[a['arc_id']] -= comm['demand']
                    solution[k] = {
                        'commodity_id': k,
                        'rejected': False,
                        'path_arcs': [{'from': a['from_node'], 'to': a['to_node'], 'arc_id': a['arc_id']} for a in path]
                    }
                else:
                    total_cost += comm['revenue']
                    solution[k] = {
                        'commodity_id': k,
                        'rejected': True,
                        'path_arcs': []
                    }
            else:
                total_cost += comm['revenue']
                solution[k] = {
                    'commodity_id': k,
                    'rejected': True,
                    'path_arcs': []
                }

        return total_cost, solution

    # Multiple greedy orderings
    order1 = sorted(commodities, key=lambda c: c['revenue'] / max(c['demand'], 1), reverse=True)
    order2 = sorted(commodities, key=lambda c: c['revenue'], reverse=True)
    order3 = sorted(commodities, key=lambda c: c['demand'])
    order4 = sorted(commodities, key=lambda c: (c['revenue'] / max(c['demand'], 1), -c['demand']), reverse=True)

    best_cost = float('inf')
    best_solution = None

    for order in [order1, order2, order3, order4]:
        cost, sol = greedy_solve(order)
        if cost < best_cost:
            best_cost = cost
            best_solution = sol

    # Log greedy solution
    sol_list = [best_solution[c['commodity_id']] for c in commodities]
    sol_dict = {'objective_value': float(best_cost), 'commodities': sol_list}
    if logger:
        logger.log_solution(best_cost, sol_dict)

    # MIP with Gurobi
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    # Check if MIP is feasible size-wise
    # Count relevant variables (prune unreachable arcs)
    total_vars = 0
    relevant_arcs = {}
    for comm in commodities:
        k = comm['commodity_id']
        relevant = [a for a in arcs if a['from_node'] in comm_reachable[k] and a['to_node'] in comm_reachable[k]]
        relevant_arcs[k] = relevant
        total_vars += len(relevant)

    if remaining_time > 5 and total_vars <= 2000000:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("unsplittable_flow")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(remaining_time - 3, 1))
            model.setParam('OutputFlag', 0)

            # Variables
            y = {}
            for comm in commodities:
                y[comm['commodity_id']] = model.addVar(vtype=GRB.BINARY, name=f"y_{comm['commodity_id']}")

            x = {}
            for comm in commodities:
                k = comm['commodity_id']
                for arc in relevant_arcs[k]:
                    x[(k, arc['arc_id'])] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{k}_{arc['arc_id']}")

            model.update()

            # Objective: minimize sum of revenue * (1 - y[k])
            obj = gp.quicksum(comm['revenue'] * (1 - y[comm['commodity_id']]) for comm in commodities)
            model.setObjective(obj, GRB.MINIMIZE)

            # Flow conservation
            for comm in commodities:
                k = comm['commodity_id']
                if comm['origin'] == comm['destination']:
                    model.addConstr(y[k] == 1)
                    continue

                for node in nodes:
                    if node not in comm_reachable[k]:
                        continue
                    out_arcs = [a for a in adj_out[node] if (k, a['arc_id']) in x]
                    in_arcs = [a for a in adj_in[node] if (k, a['arc_id']) in x]

                    out_expr = gp.quicksum(x[(k, a['arc_id'])] for a in out_arcs) if out_arcs else 0
                    in_expr = gp.quicksum(x[(k, a['arc_id'])] for a in in_arcs) if in_arcs else 0

                    if node == comm['origin']:
                        model.addConstr(out_expr - in_expr == y[k])
                    elif node == comm['destination']:
                        model.addConstr(out_expr - in_expr == -y[k])
                    else:
                        model.addConstr(out_expr - in_expr == 0)

            # Capacity constraints
            for arc in arcs:
                aid = arc['arc_id']
                terms = []
                for comm in commodities:
                    k = comm['commodity_id']
                    if (k, aid) in x:
                        terms.append(comm['demand'] * x[(k, aid)])
                if terms:
                    model.addConstr(gp.quicksum(terms) <= arc['capacity'])

            # Warm start from greedy
            for comm in commodities:
                k = comm['commodity_id']
                if best_solution[k]['rejected']:
                    y[k].Start = 0
                    for arc in relevant_arcs[k]:
                        if (k, arc['arc_id']) in x:
                            x[(k, arc['arc_id'])].Start = 0
                else:
                    y[k].Start = 1
                    path_arc_ids = set(a['arc_id'] for a in best_solution[k]['path_arcs'])
                    for arc in relevant_arcs[k]:
                        if (k, arc['arc_id']) in x:
                            x[(k, arc['arc_id'])].Start = 1 if arc['arc_id'] in path_arc_ids else 0

            model.update()
            model.optimize()

            if model.SolCount > 0:
                mip_cost = 0
                mip_solution = {}

                for comm in commodities:
                    k = comm['commodity_id']
                    if comm['origin'] == comm['destination']:
                        mip_solution[k] = {
                            'commodity_id': k,
                            'rejected': False,
                            'path_arcs': []
                        }
                        continue

                    if y[k].X > 0.5:
                        path_arcs = []
                        for arc in relevant_arcs[k]:
                            if (k, arc['arc_id']) in x and x[(k, arc['arc_id'])].X > 0.5:
                                path_arcs.append({
                                    'from': arc['from_node'],
                                    'to': arc['to_node'],
                                    'arc_id': arc['arc_id']
                                })
                        mip_solution[k] = {
                            'commodity_id': k,
                            'rejected': False,
                            'path_arcs': path_arcs
                        }
                    else:
                        mip_cost += comm['revenue']
                        mip_solution[k] = {
                            'commodity_id': k,
                            'rejected': True,
                            'path_arcs': []
                        }

                if mip_cost < best_cost:
                    best_cost = mip_cost
                    best_solution = mip_solution

                sol_list = [best_solution[c['commodity_id']] for c in commodities]
                sol_dict = {'objective_value': float(best_cost), 'commodities': sol_list}
                if logger:
                    logger.log_solution(best_cost, sol_dict)

        except Exception as e:
            print(f"Gurobi error: {e}", file=sys.stderr)

    # Write final solution
    sol_list = [best_solution[c['commodity_id']] for c in commodities]
    final_dict = {'objective_value': float(best_cost), 'commodities': sol_list}

    with open(args.solution_path, 'w') as f:
        json.dump(final_dict, f, indent=2)

if __name__ == '__main__':
    main()