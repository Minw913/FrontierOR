import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict
import itertools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    depot = instance['depot']
    K = instance['fleet']['num_vehicles']
    Q = instance['fleet']['vehicle_capacity']
    edges = instance['edges']
    num_nodes = instance['graph']['num_nodes']

    # Build graph data
    edge_list = []  # (u, v, cost, is_req, demand, serv_cost, edge_id)
    required = []   # list of (idx_in_edge_list, edge_id, demand, serv_cost)
    total_service_cost = 0
    edge_id_to_idx = {}
    adj = defaultdict(list)

    for e in edges:
        u, v = e['endpoints']
        eid = e['edge_id']
        cost = e['cost']
        is_req = e['is_required']
        dem = e['demand']
        sc = e['service_cost']
        idx = len(edge_list)
        edge_list.append((u, v, cost, is_req, dem, sc, eid))
        edge_id_to_idx[eid] = idx
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        if is_req:
            required.append((idx, eid, dem, sc))
            total_service_cost += sc

    R = len(required)
    if R == 0:
        # Trivial case
        obj = 0.0
        routes = []
        for k in range(K):
            routes.append({
                "vehicle": k,
                "serviced_edges": [],
                "deadheaded_edges": []
            })
        output = {
            "objective_value": float(obj),
            "routes": routes
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        if logger:
            logger.log(obj)
        return

    # Precompute shortest path distances (for lower bound and potential use)
    # Floyd-Warburg for all pairs (small n assumed)
    INF = 10**9
    dist = [[INF] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        dist[i][i] = 0
    for u, v, c, _, _, _, _ in edge_list:
        dist[u][v] = min(dist[u][v], c)
        dist[v][u] = min(dist[v][u], c)
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Lower bound on deadheading: each required edge needs to be connected to depot somehow
    # Very rough LB
    min_dead_lb = 0
    for _, _, d, _ in required:
        # at least go to nearest required and back, but very loose
        pass

    # We will use column generation like approach but since single core and time limited
    # we solve a MIP with arc formulation for multi-vehicle CARP with capacity.
    # But full MIP can be heavy. Since K is small and R is probably small in instances,
    # we generate all feasible single vehicle tours (as subset of required edges) and then solve set partitioning.

    # First, identify all possible feasible "routes" i.e. subsets of required edges whose total demand <= Q
    # For each subset, compute the minimal cost to service exactly those edges (Chinese Postman like for that subset + depot)

    # This is only feasible if R <= 15 or so. We need to see.
    # If R is larger, we fall back to a simple construction heuristic.

    # To make it general, let's implement a MIP formulation using MTZ like or flow for connectivity with capacity.
    # But for CARP, standard is to use edge traversal variables.

    # Let's define the MIP:
    # Variables:
    # For each vehicle k, for each edge e (undir so we consider both directions? but since undir we can use undirected model)

    # This is known to be hard. A standard formulation for CARP is complicated with many variables.
    # Given time and single core, let's try to solve it with a compact formulation.

    # Since it's a programming competition like, instances are small. Let's assume R <= 20, n <= 30.
    # We will enumerate all feasible subsets for each vehicle.

    # Generate all feasible combinations of required edges per vehicle.
    feasible_routes = []  # list of (subset_tuple, dead_cost, deadhead_counter)
    # where subset_tuple is sorted tuple of serviced edge_ids

    req_dem = [req[2] for req in required]
    req_eid = [req[1] for req in required]

    # To compute the min deadheading cost for a given set S of required edges.
    # This is equivalent to finding min cost closed walk at depot covering all edges in S (service once, deadhead any).
    # This is still the Rural Postman Problem which is NP-hard, but for small |S| we can solve it by enumerating pairings.

    # For small |S|, we can use MIP per subset but that would be too slow.
    # A better way: use dynamic programming for TSP like on required edges + depot.

    # Let's first see how large R can be. Since not specified, we need a method that scales.
    # Let's implement a greedy heuristic that constructs K routes.

    start_time = time.time()
    time_limit = args.time_limit

    # Simple construction: sort required edges by demand descending, assign to vehicles like bin packing but with routing cost estimate.

    # Better: use Clarke and Wright like savings for CARP.

    # For simplicity and to guarantee feasibility, we will assign each required edge to a vehicle in a round-robin way after sorting by demand.
    # Then for each vehicle, solve a TSP like on its required edges + depot to get a good tour, but since it's edge servicing, it's not node TSP.

    # The problem is the classic Capacitated Arc Routing Problem (CARP).

    # To solve it reasonably, since Gurobi is available, let's formulate a MIP.

    # We will use the following formulation:
    # - x_{e,k} binary: whether vehicle k services required edge e
    # - For every edge (required or not), y_{i,j,k} = number of times vehicle k traverses from i to j (directed version)

    # Then constraints:
    # sum_k x_e,k = 1 for each required e
    # sum demands * x <= Q for each k
    # Flow conservation for each vehicle at each node
    # Connectivity / subtour elimination is hard. We can use MTZ or cut separation but in single run it's tricky.

    # Since single CPU and time limit, let's implement a heuristic that iteratively improves.

    # But to get a good solution fast, let's use the following approach:
    # 1. Assign required edges to vehicles using greedy bin packing (largest demand first, best fit vehicle with current load)
    # 2. For each vehicle, solve the Rural Postman Problem approximately for its set of required edges.

    # For RPP on undirected graph with depot, a simple way is to solve TSP on the required edges' endpoints with shortest path distances, then add the service traversals.

    # More precisely:
    # - The required edges must be traversed at least once (serviced).
    # - We need to add minimum cost deadheading to make the graph Eulerian (even degrees) and connected to depot.

    # Implementing full RPP solver is complex but for small instances possible.

    # Let's check average size. Since it's a test, let's assume we can use Gurobi to solve a set partitioning type with precomputed columns if R<=15.

    R_max_for_enum = 12  # if larger, use heuristic

    if R > R_max_for_enum:
        # Use heuristic
        # Sort required by demand descending
        sorted_req = sorted(enumerate(required), key=lambda x: -x[1][2])
        vehicle_load = [0] * K
        assignment = [-1] * R   # which vehicle for each required idx
        for i, (ridx, eid, dem, sc) in sorted_req:
            # find vehicle with enough capacity and smallest current load
            best_v = -1
            best_load = 10**9
            for v in range(K):
                if vehicle_load[v] + dem <= Q and vehicle_load[v] < best_load:
                    best_load = vehicle_load[v]
                    best_v = v
            if best_v == -1:
                # Over capacity, assign to first with space or force
                for v in range(K):
                    if vehicle_load[v] + dem <= Q:
                        best_v = v
                        break
                else:
                    best_v = 0  # force, even if over (but problem requires feasible)
            assignment[ridx] = best_v
            vehicle_load[best_v] += dem

        # Now for each vehicle, we need to build a route that visits all its required edges.
        # Simple way: for each required edge assigned to vehicle, add two deadheads: depot to one end, other end to depot.
        # This gives an upper bound. Each required edge serviced in its own "trip".
        # This is feasible but not great.

        routes = []
        total_dead_cost = 0
        deadhead_usage = [defaultdict(int) for _ in range(K)]

        for k in range(K):
            serviced = []
            for r_idx, (eidx, eid, dem, sc) in enumerate(required):
                if assignment[r_idx] == k:
                    serviced.append(eid)
                    # compute shortest from depot to both ends, choose min round trip
                    u, v, cost, _, _, _, _ = edge_list[eidx]
                    d1 = dist[depot][u]
                    d2 = dist[depot][v]
                    if d1 <= d2:
                        # depot->u , service u->v, v->depot
                        if d1 > 0:
                            # We need to add the deadhead path, but since we only need total cost not path, for output we need the deadheaded edges.
                            # This is problem. The output requires explicit deadheaded_edges with multiplicity.
                            # So we need the actual multiset of deadhead traversals.
                            # Simple separate trip is easy but we need the actual path.
                    # This is getting complicated for heuristic.
                    # Let's change strategy.

        # Better heuristic: since output needs the counts, let's model it as each required edge is serviced separately with its own tour from depot.
        # That is, K vehicles but since they are identical, but each required edge gets its own "mini-trip" but we have only K vehicles.
        # The number of vehicles is fixed but they can do multiple services in one closed walk.

        # Let's implement a MIP with Gurobi that can run with time limit.

    # Let's build a directed model.
    # We make the graph directed by duplicating each undirected edge in both directions.
    # But for service, required edges can be serviced in either direction.

    # Standard CARP formulation is a bit involved.

    # Let's use the following compact formulation with node copies or something. It may be too slow.

    # Since time is limited, let's solve it by assigning subsets if R is small, and if R is large use the separate trip approach but group them.

    # To satisfy the output schema, we need to output deadheaded edges with times.

    # Let's do the following feasible solution:
    # Each vehicle will service a disjoint set of required edges.
    # For simplicity, give each required edge its own "trip" from depot: go to one endpoint (deadhead), service the edge, go back from the other endpoint (deadhead).
    # Then we assign these trips to the K vehicles, making sure capacity per vehicle is respected.
    # But to make it one route per vehicle, we need to combine the trips into one closed walk per vehicle.
    # Combining requires solving a TSP on the trips which is again complex.

    # A very simple but always feasible way if total_demand <= K*Q:
    # We will have each vehicle do a full Chinese Postman Tour on the whole graph but that violates the "exactly one vehicle services" rule.

    # The service must be done by exactly one.

    # To make it simple and correct, let's create one big tour that services all required edges, then split it into K feasible capacity parts.
    # But splitting is not trivial.

    # Let's use Gurobi to solve an assignment + flow model but without connectivity first, then fix.

    # Let's look at expected instance size. Since it's "fleet of vehicles" and "service company", probably n<=50, R<=30 or so.

    # For efficiency, let's implement the following:
    # Use MIP to assign required edges to vehicles with capacity constraint.
    # Then for each vehicle, compute a lower bound on deadheading cost using shortest paths from depot to required edges.
    # But for actual solution, we need a feasible routing.

    # For the actual deadheading, we can output that each vehicle goes from depot to each of its required edges separately and back, and count the deadhead costs and edges accordingly.
    # This is feasible because the route can be depot -> ... -> service edge1 -> ... -> depot -> ... -> service edge2 -> ... -> depot.
    # It is a single closed walk (though not efficient).

    # Yes, this satisfies all constraints.

    # So let's implement that.

    # First, solve assignment of required edges to vehicles to respect capacity (like bin packing but with K bins).

    # Use Gurobi for assignment to minimize estimated routing cost.

    m = gp.Model("CARP_Assignment")
    m.setParam('TimeLimit', args.time_limit - 5 if args.time_limit > 10 else args.time_limit)
    m.setParam('Threads', 1)

    # x[r, k] = 1 if required r assigned to vehicle k
    x = {}
    for r in range(R):
        for k in range(K):
            x[r, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{r}_{k}")

    # capacity
    for k in range(K):
        m.addConstr(gp.quicksum(required[r][2] * x[r, k] for r in range(R)) <= Q)

    # each required assigned exactly once
    for r in range(R):
        m.addConstr(gp.quicksum(x[r, k] for k in range(K)) == 1)

    # Objective: to minimize estimated deadheading.
    # For each required edge, the min cost to go to it from depot, service, return from other end.
    # We can precompute for each required the min roundtrip deadhead cost.
    round_trip_cost = []
    for eidx, eid, dem, sc in required:
        u, v, dead_cost, _, _, _, _ = edge_list[eidx]
        # deadhead cost is for traversal without service. So to reach u, service u-v, then from v back to depot.
        cost_uv = dist[depot][u] + dist[v][depot]
        cost_vu = dist[depot][v] + dist[u][depot]
        round_trip_cost.append(min(cost_uv, cost_vu))

    # To encourage balanced or good assignment, we can minimize sum round_trip * x but since all will be paid anyway, it is constant.
    # So the estimated total dead is always the same. So we can set arbitrary objective or use random.

    m.setObjective(0, GRB.MINIMIZE)  # just find any feasible

    m.optimize()

    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        # If infeasible, we have a problem but assume instances are feasible.
        pass

    assignment = [-1] * R
    for r in range(R):
        for k in range(K):
            if x[r, k].X > 0.5:
                assignment[r] = k
                break

    # Now build the routes: for each vehicle, list of serviced edge ids
    vehicle_serviced = [[] for _ in range(K)]
    for r in range(R):
        k = assignment[r]
        vehicle_serviced[k].append(required[r][1])  # eid

    # Now for deadheading: for each required edge of a vehicle, we will add a round trip from depot.
    # We need to choose direction and record which edges are deadheaded how many times.
    # To do that, we need the actual shortest path.

    # To get shortest path, we can use prev matrix from Floyd but it's all pairs, we need to reconstruct path.
    # For simplicity, since we need to output the counts, we can traverse each required edge separately and add 2*shortest path edges for to and from.

    # But to reconstruct the actual edges in shortest path, we need path reconstruction.
    # Let's compute predecessor matrix.

    pred = [[-1] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                for neighbor, e_idx in adj[i]:
                    if neighbor == j and dist[i][j] == edge_list[e_idx][2]:
                        pred[i][j] = neighbor
                        break
                else:
                    for k2 in range(num_nodes):
                        if dist[i][k2] + dist[k2][j] == dist[i][j] and dist[i][k2] < INF:
                            pred[i][j] = k2
                            break

    def get_path(start, end):
        # returns list of edge indices in the shortest path from start to end (not including service)
        if dist[start][end] >= INF or start == end:
            return []
        path = []
        current = start
        while current != end:
            next_node = pred[current][end]
            if next_node == -1:
                return []  # failure
            # find edge between current and next_node
            found = False
            for neigh, e_idx in adj[current]:
                if neigh == next_node:
                    path.append(e_idx)
                    found = True
                    break
            if not found:
                return []
            current = next_node
        return path

    # Now for each vehicle and each required it services, choose best direction, add the deadhead paths to and from.
    vehicle_dead = [defaultdict(int) for _ in range(K)]  # edge_idx -> times

    total_dead_cost = 0.0

    for r_idx in range(R):
        k = assignment[r_idx]
        e_idx, eid, dem, sc = required[r_idx]
        u, v, dead_c, isr, d, s_c, eid2 = edge_list[e_idx]
        # two options: service u->v or v->u. But since undir, doesn't matter for service but for connection does.
        # Option 1: arrive at u (dead), service to v, then dead from v to depot
        path_to = get_path(depot, u)
        path_from = get_path(v, depot)
        cost1 = sum(edge_list[eidxx][2] for eidxx in path_to) + sum(edge_list[eidxx][2] for eidxx in path_from)
        # Option 2
        path_to2 = get_path(depot, v)
        path_from2 = get_path(u, depot)
        cost2 = sum(edge_list[eidxx][2] for eidxx in path_to2) + sum(edge_list[eidxx][2] for eidxx in path_from2)

        if cost1 <= cost2:
            for ee in path_to:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]
            for ee in path_from:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]
        else:
            for ee in path_to2:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]
            for ee in path_from2:
                vehicle_dead[k][ee] += 1
                total_dead_cost += edge_list[ee][2]

    objective = total_service_cost + total_dead_cost

    # Build output
    routes = []
    for k in range(K):
        dead_list = []
        for ee, cnt in vehicle_dead[k].items():
            if cnt > 0:
                dead_list.append({
                    "edge_id": edge_list[ee][6],  # eid
                    "times": cnt
                })
        routes.append({
            "vehicle": k,
            "serviced_edges": sorted(vehicle_serviced[k]),  # good practice
            "deadheaded_edges": dead_list
        })

    # If logger, log the objective
    if logger:
        logger.log(objective)

    output = {
        "objective_value": float(objective),
        "routes": routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()