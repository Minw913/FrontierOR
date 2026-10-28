import sys
import math
from collections import defaultdict, deque
import itertools

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

INF = float('inf')


def read_input():
    data = sys.stdin.read().strip().split('\n')
    idx = 0
    n_tasks, n_depots = map(int, data[idx].split()); idx += 1
    tasks = []
    for i in range(n_tasks):
        parts = data[idx].split(); idx += 1
        s = int(parts[0]); e = int(parts[1]); dep = int(parts[2])
        tasks.append((s, e, dep))
    depots = []
    for i in range(n_depots):
        parts = data[idx].split(); idx += 1
        depots.append((int(parts[0]), int(parts[1])))
    params = list(map(int, data[idx].split())); idx += 1
    SP = params[0]
    SR = params[1]
    SD = params[2]
    CP = params[3]
    CR = params[4]
    CD = params[5]
    return tasks, depots, SP, SR, SD, CP, CR, CD


def vehicle_scheduling(tasks, depots):
    """
    Minimum path cover via bipartite matching.
    Each task is a node. Edge from task i to task j if end_i <= start_j.
    Returns list of vehicle routes (each route is list of task indices).
    """
    n = len(tasks)
    # Build adjacency: task i can be followed by task j
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and tasks[i][1] <= tasks[j][0]:
                adj[i].append(j)

    # Hopcroft-Karp for maximum bipartite matching
    # Left side: predecessors, Right side: successors
    pair_u = [-1] * n  # task i -> matched successor
    pair_v = [-1] * n  # task j -> matched predecessor
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1

    # Reconstruct paths
    # A task is a "start" if it has no predecessor in matching
    has_pred = [False] * n
    for j in range(n):
        if pair_v[j] != -1:
            has_pred[j] = True

    routes = []
    for i in range(n):
        if not has_pred[i]:
            route = [i]
            cur = i
            while pair_u[cur] != -1:
                cur = pair_u[cur]
                route.append(cur)
            routes.append(route)

    return routes


def compute_vehicle_cost(routes, tasks, depots, SP, SR, SD):
    """Compute total vehicle cost."""
    total = 0
    for route in routes:
        if not route:
            continue
        # Pull-out from depot to first task
        first = route[0]
        depot_idx = tasks[first][2]
        depot_loc = depots[depot_idx]
        total += SP + SR * abs(depot_loc[0] - tasks[first][0]) + SD * abs(depot_loc[1] - tasks[first][1])
        # Deadhead between tasks
        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            total += SR * abs(tasks[i][1] - tasks[j][0]) + SD * abs(tasks[i][1] - tasks[j][1])
        # Pull-in from last task to depot
        last = route[-1]
        total += SP + SR * abs(tasks[last][1] - depot_loc[0]) + SD * abs(tasks[last][1] - depot_loc[1])
    return total


def split_into_pieces(route, tasks, depots, CP, CR, CD):
    """
    Split a vehicle route into pieces using DP.
    A piece is a contiguous subsequence of tasks.
    Cost of a piece includes deadhead within and relief at boundaries.
    """
    if not route:
        return [], 0

    n = len(route)
    # Precompute deadhead cost between consecutive tasks in route
    # and pull-out/pull-in costs

    # piece_cost[i][j] = cost of piece covering tasks route[i..j]
    # This includes: pull-out to route[i], deadheads between, pull-in from route[j]
    # Actually for crew pieces, we consider the deadhead within the piece
    # and relief costs at boundaries

    # Let's define piece cost as the deadhead cost within the piece
    # (between consecutive tasks) plus pull-out and pull-in
    # Relief cost is added when joining pieces into duties

    # Precompute deadhead between consecutive tasks
    dh = [0] * (n - 1)
    for k in range(n - 1):
        i = route[k]
        j = route[k + 1]
        dh[k] = CR * abs(tasks[i][1] - tasks[j][0]) + CD * abs(tasks[i][1] - tasks[j][1])

    # piece_cost[i][j] = sum of dh[i..j-1] + pull-out + pull-in
    # For crew, pull-out and pull-in are part of the piece cost
    first_task = route[0]
    last_task = route[-1]
    depot_idx = tasks[first_task][2]
    depot_loc = depots[depot_idx]

    # DP to find optimal split points
    # dp[k] = min cost to split route[0..k-1] into pieces
    # We also track the pieces

    # piece internal cost for route[i..j]:
    # pull-out + deadheads i..j-1 + pull-in
    def piece_internal_cost(i, j):
        cost = CP  # pull-out fixed cost
        cost += CR * abs(depot_loc[0] - tasks[route[i]][0]) + CD * abs(depot_loc[1] - tasks[route[i]][1])
        for k in range(i, j):
            cost += dh[k]
        cost += CP  # pull-in fixed cost
        cost += CR * abs(tasks[route[j]][1] - depot_loc[0]) + CD * abs(tasks[route[j]][1] - depot_loc[1])
        return cost

    # DP: minimize total piece cost + relief cost between pieces
    # Relief cost between piece ending at route[j] and piece starting at route[j+1]
    # = CR * |end_j - start_{j+1}| + CD * |end_j - start_{j+1}| (same as deadhead)
    # Actually relief cost is the cost of the gap between pieces

    dp = [INF] * (n + 1)
    dp[0] = 0
    parent = [-1] * (n + 1)

    for j in range(n):
        for i in range(j + 1):
            # piece from route[i..j]
            pcost = piece_internal_cost(i, j)
            # relief cost before this piece (if i > 0)
            relief = 0
            if i > 0:
                prev = route[i - 1]
                cur = route[i]
                relief = CR * abs(tasks[prev][1] - tasks[cur][0]) + CD * abs(tasks[prev][1] - tasks[cur][1])
            total = dp[i] + pcost + relief
            if total < dp[j + 1]:
                dp[j + 1] = total
                parent[j + 1] = i

    # Reconstruct pieces
    pieces = []
    j = n
    while j > 0:
        i = parent[j]
        pieces.append((i, j))  # route[i..j-1]
        j = i
    pieces.reverse()

    return pieces, dp[n]


def pair_pieces_into_duties(pieces_info, tasks, depots, CP, CR, CD):
    """
    Pair pieces into duties. A duty consists of up to 2 pieces.
    Use greedy matching or Gurobi if available.
    pieces_info: list of (route_idx, piece_start, piece_end, piece_cost, start_time, end_time, start_loc, end_loc)
    """
    n = len(pieces_info)
    if n == 0:
        return 0, []

    # Cost of pairing piece i with piece j (i before j):
    # deadhead from end of piece i to start of piece j
    # If they can't be paired (time constraint), cost is INF
    def pairing_cost(i, j):
        pi = pieces_info[i]
        pj = pieces_info[j]
        # piece i must end before piece j starts
        if pi[5] > pj[4]:
            return INF
        # deadhead cost
        return CR * abs(pi[6] - pj[7]) + CD * abs(pi[6] - pj[7])

    # We want to minimize total cost = sum of single-piece duties + paired duties
    # Single piece duty cost = piece_cost
    # Paired duty cost = piece_i_cost + piece_j_cost + pairing_cost

    if HAS_GUROBI and n > 2:
        try:
            model = gp.Model("crew_duties")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 10)

            # x[i] = 1 if piece i is a single-piece duty
            # y[i][j] = 1 if pieces i and j are paired (i before j)
            x = {}
            for i in range(n):
                x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

            y = {}
            for i in range(n):
                for j in range(n):
                    if i < j:
                        c = pairing_cost(i, j)
                        if c < INF:
                            y[(i, j)] = model.addVar(vtype=GRB.BINARY, obj=c, name=f"y_{i}_{j}")
                        c2 = pairing_cost(j, i)
                        if c2 < INF and j != i:
                            if (j, i) not in y:
                                y[(j, i)] = model.addVar(vtype=GRB.BINARY, obj=c2, name=f"y_{j}_{i}")

            model.update()

            # Each piece is either single or paired exactly once
            for i in range(n):
                paired_vars = []
                for (a, b), var in y.items():
                    if a == i or b == i:
                        paired_vars.append(var)
                model.addConstr(x[i] + gp.quicksum(paired_vars) == 1, f"assign_{i}")

            model.setObjective(gp.quicksum(x[i] * pieces_info[i][3] for i in range(n)) +
                               gp.quicksum(var * (pieces_info[a][3] + pieces_info[b][3])
                                          for (a, b), var in y.items()),
                               GRB.MINIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                duties = []
                used = set()
                for (a, b), var in y.items():
                    if var.X > 0.5:
                        duties.append((a, b))
                        used.add(a)
                        used.add(b)
                for i in range(n):
                    if x[i].X > 0.5:
                        duties.append((i,))
                        used.add(i)
                total = model.ObjVal
                return total, duties
        except Exception:
            pass

    # Greedy fallback
    # Sort pieces by end time
    order = sorted(range(n), key=lambda i: pieces_info[i][5])

    used = [False] * n
    duties = []
    total = 0

    for idx in range(n):
        i = order[idx]
        if used[i]:
            continue
        # Try to find best piece to pair with
        best_j = -1
        best_cost = INF
        for jdx in range(n):
            j = order[jdx]
            if used[j] or i == j:
                continue
            c = pairing_cost(i, j)
            if c < INF:
                # Total cost of pairing vs two singles
                pair_total = pieces_info[i][3] + pieces_info[j][3] + c
                single_total = pieces_info[i][3] + pieces_info[j][3]
                if pair_total < best_cost:
                    best_cost = pair_total
                    best_j = j

        if best_j >= 0 and best_cost < pieces_info[i][3] + pieces_info[best_j][3]:
            duties.append((i, best_j))
            total += best_cost
            used[i] = True
            used[best_j] = True
        else:
            duties.append((i,))
            total += pieces_info[i][3]
            used[i] = True

    return total, duties


def solve(tasks, depots, SP, SR, SD, CP, CR, CD):
    # Step 1: Vehicle scheduling
    routes = vehicle_scheduling(tasks, depots)
    n_vehicles = len(routes)
    vehicle_cost = compute_vehicle_cost(routes, tasks, depots, SP, SR, SD)

    # Step 2: Split each route into pieces
    all_pieces_info = []
    for ridx, route in enumerate(routes):
        pieces, piece_cost = split_into_pieces(route, tasks, depots, CP, CR, CD)
        for (i, j) in pieces:
            # piece covers route[i..j-1]
            first_task_idx = route[i]
            last_task_idx = route[j - 1]
            start_time = tasks[first_task_idx][0]
            end_time = tasks[last_task_idx][1]
            start_loc = tasks[first_task_idx][0]
            end_loc = tasks[last_task_idx][1]
            depot_idx = tasks[first_task_idx][2]
            depot_loc = depots[depot_idx]

            # Compute piece cost
            pcost = CP  # pull-out
            pcost += CR * abs(depot_loc[0] - tasks[first_task_idx][0]) + CD * abs(depot_loc[1] - tasks[first_task_idx][1])
            for k in range(i, j - 1):
                ti = route[k]
                tj = route[k + 1]
                pcost += CR * abs(tasks[ti][1] - tasks[tj][0]) + CD * abs(tasks[ti][1] - tasks[tj][1])
            pcost += CP  # pull-in
            pcost += CR * abs(tasks[last_task_idx][1] - depot_loc[0]) + CD * abs(tasks[last_task_idx][1] - depot_loc[1])

            all_pieces_info.append((ridx, i, j, pcost, start_time, end_time, end_loc, start_loc))

    # Step 3: Pair pieces into duties
    crew_cost, duties = pair_pieces_into_duties(all_pieces_info, tasks, depots, CP, CR, CD)
    n_duties = len(duties)

    total_cost = vehicle_cost + crew_cost

    return {
        'n_vehicles': n_vehicles,
        'n_duties': n_duties,
        'vehicle_cost': vehicle_cost,
        'crew_cost': crew_cost,
        'total_cost': total_cost,
        'routes': routes,
        'duties': duties,
    }


def main():
    tasks, depots, SP, SR, SD, CP, CR, CD = read_input()
    result = solve(tasks, depots, SP, SR, SD, CP, CR, CD)
    print(f"Vehicles: {result['n_vehicles']}")
    print(f"Duties: {result['n_duties']}")
    print(f"Vehicle cost: {result['vehicle_cost']}")
    print(f"Crew cost: {result['crew_cost']}")
    print(f"Total cost: {result['total_cost']}")
    print()
    for i, route in enumerate(result['routes']):
        task_strs = [f"({tasks[t][0]},{tasks[t][1]})" for t in route]
        print(f"Vehicle {i+1}: {' -> '.join(task_strs)}")
    print()
    for i, duty in enumerate(result['duties']):
        pieces_str = []
        for p in duty:
            info = result['duties']  # placeholder
        print(f"Duty {i+1}: pieces {duty}")


if __name__ == '__main__':
    main()