import json
import argparse
import heapq
from collections import deque
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def is_dag(num_nodes, arcs):
    """Check if the directed graph is acyclic using Kahn's algorithm."""
    adj = {v: [] for v in range(num_nodes)}
    indeg = {v: 0 for v in range(num_nodes)}
    for arc in arcs:
        u = arc['from_node']
        v = arc['to_node']
        adj[u].append(v)
        indeg[v] += 1
    q = deque([v for v in range(num_nodes) if indeg[v] == 0])
    cnt = 0
    while q:
        u = q.popleft()
        cnt += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return cnt == num_nodes


def dijkstra(num_nodes, arcs, linear_costs, source, target):
    """Find a shortest path using linear costs only (for a warm start)."""
    adj = {v: [] for v in range(num_nodes)}
    for i, arc in enumerate(arcs):
        adj[arc['from_node']].append((arc['to_node'], linear_costs[i], i))
    dist = {v: float('inf') for v in range(num_nodes)}
    dist[source] = 0
    prev = {v: None for v in range(num_nodes)}
    prev_arc = {v: None for v in range(num_nodes)}
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == target:
            break
        for v, w, i in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                prev[v] = u
                prev_arc[v] = i
                heapq.heappush(pq, (dist[v], v))
    if dist[target] == float('inf'):
        return None
    path = []
    v = target
    while v != source:
        path.append(prev_arc[v])
        v = prev[v]
    return path


def bfs_path(num_nodes, arcs, source, target):
    """Fallback: find any path ignoring costs."""
    adj = {v: [] for v in range(num_nodes)}
    for i, arc in enumerate(arcs):
        adj[arc['from_node']].append((arc['to_node'], i))
    q = deque([source])
    prev = {v: None for v in range(num_nodes)}
    prev_arc = {v: None for v in range(num_nodes)}
    visited = {source}
    while q:
        u = q.popleft()
        if u == target:
            break
        for v, i in adj[u]:
            if v not in visited:
                visited.add(v)
                prev[v] = u
                prev_arc[v] = i
                q.append(v)
    if prev_arc[target] is None:
        return None
    path = []
    v = target
    while v != source:
        path.append(prev_arc[v])
        v = prev[v]
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    num_nodes = inst['num_nodes']
    num_arcs = inst['num_arcs']
    source = inst['source_node']
    target = inst['target_node']
    arcs = inst['arcs']
    linear_costs = inst['linear_costs']
    Q = inst['quadratic_costs']

    dag = is_dag(num_nodes, arcs)

    model = gp.Model('qspp')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    x = {}
    x_list = []
    for i in range(num_arcs):
        var = model.addVar(vtype=GRB.BINARY, name=f'x_{i}')
        x[i] = var
        x_list.append(var)

    # Flow conservation constraints
    out_arcs = {v: [] for v in range(num_nodes)}
    in_arcs = {v: [] for v in range(num_nodes)}
    for i, arc in enumerate(arcs):
        out_arcs[arc['from_node']].append(i)
        in_arcs[arc['to_node']].append(i)

    for v in range(num_nodes):
        if v == source:
            model.addConstr(gp.quicksum(x[i] for i in out_arcs[v]) == 1)
            model.addConstr(gp.quicksum(x[i] for i in in_arcs[v]) == 0)
        elif v == target:
            model.addConstr(gp.quicksum(x[i] for i in out_arcs[v]) == 0)
            model.addConstr(gp.quicksum(x[i] for i in in_arcs[v]) == 1)
        else:
            model.addConstr(gp.quicksum(x[i] for i in in_arcs[v]) == gp.quicksum(x[i] for i in out_arcs[v]))

    # A simple path has at most num_nodes - 1 arcs
    model.addConstr(gp.quicksum(x_list) <= num_nodes - 1)

    # MTZ subtour elimination only if the graph is not acyclic
    if not dag:
        u = {}
        for v in range(num_nodes):
            u[v] = model.addVar(lb=0, ub=num_nodes - 1, vtype=GRB.CONTINUOUS, name=f'u_{v}')
        model.addConstr(u[source] == 0)
        for i, arc in enumerate(arcs):
            a = arc['from_node']
            b = arc['to_node']
            model.addConstr(u[b] >= u[a] + 1 - num_nodes * (1 - x[i]))

    # Objective: linear + quadratic (using symmetry to halve terms)
    obj = gp.LinExpr()
    for i in range(num_arcs):
        obj += linear_costs[i] * x[i]
    quad = gp.QuadExpr()
    for i in range(num_arcs):
        xi = x[i]
        for j in range(i + 1, num_arcs):
            q = Q[i][j]
            if q != 0:
                quad += 2 * q * xi * x[j]
    obj += quad
    model.setObjective(obj, GRB.MINIMIZE)

    # Warm start: shortest path ignoring quadratic costs
    start_path = dijkstra(num_nodes, arcs, linear_costs, source, target)
    if start_path is None:
        start_path = bfs_path(num_nodes, arcs, source, target)
    if start_path:
        start = [0] * num_arcs
        for i in start_path:
            start[i] = 1
        for i in range(num_arcs):
            x[i].Start = start[i]
        # Log the initial solution
        obj_val = sum(linear_costs[i] for i in start_path)
        for idx1 in range(len(start_path)):
            i = start_path[idx1]
            for idx2 in range(idx1 + 1, len(start_path)):
                j = start_path[idx2]
                obj_val += 2 * Q[i][j]
        selected = [{"id": arcs[i]['id'], "from_node": arcs[i]['from_node'], "to_node": arcs[i]['to_node']} for i in start_path]
        sol_dict = {"objective_value": float(obj_val), "solution_arcs": selected}
        if logger:
            logger.log_solution(float(obj_val), sol_dict)

    # Callback to log every new incumbent
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                x_vals = model.cbGetSolution(x_list)
                selected = []
                for i in range(num_arcs):
                    if x_vals[i] > 0.5:
                        selected.append({
                            "id": arcs[i]['id'],
                            "from_node": arcs[i]['from_node'],
                            "to_node": arcs[i]['to_node']
                        })
                sol_dict = {"objective_value": float(obj_val), "solution_arcs": selected}
                logger.log_solution(float(obj_val), sol_dict)
        model.optimize(callback)
    else:
        model.optimize()

    # Extract the best solution found
    if model.SolCount > 0:
        x_vals = model.getAttr('X', x_list)
        selected = []
        sel_set = set()
        for i in range(num_arcs):
            if x_vals[i] > 0.5:
                selected.append({
                    "id": arcs[i]['id'],
                    "from_node": arcs[i]['from_node'],
                    "to_node": arcs[i]['to_node']
                })
                sel_set.add(i)
        # Recompute the exact objective value from the selected arcs
        obj_val = 0.0
        sel_list = list(sel_set)
        for i in sel_list:
            obj_val += linear_costs[i]
        for idx1 in range(len(sel_list)):
            i = sel_list[idx1]
            for idx2 in range(idx1 + 1, len(sel_list)):
                j = sel_list[idx2]
                obj_val += 2 * Q[i][j]
        solution = {
            "objective_value": float(obj_val),
            "solution_arcs": selected
        }
    else:
        solution = {"objective_value": 0.0, "solution_arcs": []}

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()