import argparse
import json
import time
import itertools
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    params = instance['parameters']
    n = params['n']
    R = params['R']
    p = params['p']
    od_pairs = instance['od_pairs']
    num_od_pairs = params['num_od_pairs']
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Build distance lookup for arcs
    arc_dist = {}
    for arc in instance['arcs']:
        arc_dist[(arc['from'], arc['to'])] = arc['distance']
        arc_dist[(arc['to'], arc['from'])] = arc['distance']
    
    # For each OD pair, precompute the round-trip path and determine which sets of stations can cover it
    # Round trip: origin -> destination -> origin (same path reversed)
    
    # For each OD pair, compute the round-trip sequence of nodes
    # and determine for each "segment" (directed arc) whether it needs a station
    
    def preprocess_od_pair(od_idx):
        """
        For an OD pair, compute the set of candidate station nodes that could help,
        and determine the minimal station requirements.
        
        Returns: (flow, candidate_nodes_set, check_function) or (flow, None) if infeasible
        """
        od = od_pairs[od_idx]
        path = od['shortest_path']
        flow = od['flow']
        
        if len(path) <= 1:
            # Origin == destination, always covered
            return flow, set(), True, None
        
        # Build round-trip path: forward path + reverse path
        round_trip = path + path[-2::-1]  # path reversed without repeating the destination
        
        # Compute cumulative distances along the round trip
        num_nodes_rt = len(round_trip)
        # distances between consecutive nodes in round trip
        seg_dists = []
        for i in range(num_nodes_rt - 1):
            u, v = round_trip[i], round_trip[i+1]
            d = arc_dist.get((u, v))
            if d is None:
                # This shouldn't happen if data is consistent
                return flow, None, False, None
            seg_dists.append(d)
        
        # Check if any single segment exceeds R (impossible even with station at both ends)
        for d in seg_dists:
            if d > R:
                return flow, None, False, None
        
        # Total round trip distance
        total_rt_dist = sum(seg_dists)
        if total_rt_dist <= R:
            # Can complete round trip on initial full tank alone
            return flow, set(), True, None
        
        # Now we need to figure out which stations along the path enable completion
        # The vehicle starts with a full tank at round_trip[0]
        # At each node on the round trip, if there's a station, the vehicle refuels to full
        # We need to check if the vehicle can traverse all segments
        
        # Nodes in round trip (with positions)
        # For efficiency, we precompute the "segments" between potential refueling points
        
        # Get unique candidate nodes (nodes on the path excluding maybe origin?)
        # Actually, any node on the path could be a station
        # The origin node: if it has a station, the vehicle can refuel there at the start (already full)
        # and also when returning (at the end of the round trip)
        
        # Candidate nodes are all nodes on the shortest path
        path_nodes = set(path)
        
        # For checking feasibility given a set of open stations:
        # Simulate the round trip, refueling at open stations
        # The vehicle starts full. At each node, if it's an open station, refuel to R.
        # Check that fuel never goes negative.
        
        # Precompute for efficient checking:
        # round_trip_nodes and seg_dists are what we need
        
        return flow, path_nodes, None, (round_trip, seg_dists)
    
    # Preprocess all OD pairs
    od_data = []
    always_covered = []  # OD pairs that are always covered (no stations needed)
    infeasible = []  # OD pairs that can never be covered
    needs_stations = []  # OD pairs that need stations - (od_idx, flow, candidate_nodes, rt_data)
    
    for idx in range(num_od_pairs):
        flow, candidates, auto_covered, rt_data = preprocess_od_pair(idx)
        if auto_covered is True:
            always_covered.append(idx)
        elif auto_covered is False and candidates is None:
            infeasible.append(idx)
        else:
            needs_stations.append((idx, flow, candidates, rt_data))
    
    # Function to check if an OD pair is covered given a set of open stations
    def is_covered(rt_data, open_stations):
        round_trip, seg_dists = rt_data
        fuel = R
        for i in range(len(seg_dists)):
            # Traverse segment i: from round_trip[i] to round_trip[i+1]
            # Before traversing, check if round_trip[i] is a station (refuel)
            if round_trip[i] in open_stations:
                fuel = R
            fuel -= seg_dists[i]
            if fuel < -1e-9:
                return False
        # Check last node too (for completeness, though we've traversed all segments)
        # Actually the last node is the origin again, if it has a station we'd refuel but trip is done
        return True
    
    # Collect all candidate nodes across all OD pairs that need stations
    all_candidates = set()
    for idx, flow, candidates, rt_data in needs_stations:
        all_candidates.update(candidates)
    
    all_candidates = sorted(all_candidates)
    
    # For each candidate node, compute which OD pairs it appears in
    node_to_od = defaultdict(set)
    for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
        for node in candidates:
            node_to_od[node].add(i)
    
    # Use Gurobi MIP formulation
    # Variables:
    # y_j = 1 if station at node j (binary)
    # z_q = 1 if OD pair q is covered (binary)
    # Constraint: sum(y_j) = p
    # Constraint: z_q = 1 only if the stations cover OD pair q
    
    # The tricky part is the covering constraint for z_q.
    # The standard FRLM formulation uses "pre-generation" of feasible combinations.
    
    # Alternative approach: For each OD pair, enumerate the minimal sets of stations needed.
    # This can be complex. Let me use the "arc cover" formulation.
    
    # For each OD pair q with round trip, define "refueling points" along the path.
    # A directed arc (segment) on the round trip is "covered" if there exists an open station
    # within range before the arc such that the vehicle can reach past the arc.
    
    # Actually, let me use a different formulation based on "combo" sets.
    # For each OD pair, find all minimal subsets of path nodes that make the trip feasible.
    # Then z_q <= sum of indicators for those subsets... but that's exponential.
    
    # Better approach: Use the "segment" formulation.
    # For each OD pair's round trip, identify "critical segments" - maximal sub-paths 
    # between consecutive refueling opportunities.
    
    # Let me use the formulation from the literature:
    # For each OD pair q, let the round trip pass through nodes v_0, v_1, ..., v_K.
    # Let d(i,j) be the distance from v_i to v_j along the path.
    # The trip is feasible iff for every consecutive pair of "refueling nodes" (including start),
    # the distance between them is <= R.
    
    # Formulation using "combination" variables or "path-based" constraints:
    # For each OD pair, we can use a flow-based or layered formulation.
    
    # Let me try a practical approach: 
    # Since the problem might be large, let me try a greedy + local search heuristic first,
    # then optionally use MIP.
    
    # Greedy approach: iteratively add the station that maximizes additional covered flow
    
    def compute_covered_flow(open_stations_set):
        total = sum(od_pairs[idx]['flow'] for idx in always_covered)
        covered_indices = list(always_covered)
        for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
            if is_covered(rt_data, open_stations_set):
                total += flow
                covered_indices.append(idx)
        return total, covered_indices
    
    # For each OD pair that needs stations, precompute which individual nodes 
    # are on the round trip path and their positions
    # This helps in quickly checking coverage
    
    # Precompute for each OD pair: for each subset of candidate nodes on its path,
    # can it be covered? This is too expensive for large instances.
    
    # Let's try greedy first
    best_objective = sum(od_pairs[idx]['flow'] for idx in always_covered)
    best_stations = []
    best_covered = list(always_covered)
    
    if logger:
        logger.log(best_objective)
    
    if p == 0:
        result = {
            "objective_value": best_objective,
            "stations": best_stations,
            "refueled_od_pairs": sorted(best_covered)
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f)
        return
    
    # Greedy construction
    selected_stations = set()
    remaining_candidates = set(all_candidates)
    # Also consider all nodes as candidates (stations can be at any node)
    all_nodes = set(node['id'] for node in instance['nodes'])
    # Actually, stations should only be placed at nodes on some OD pair's path to be useful
    # But we might want to include all nodes as candidates
    # For efficiency, only consider nodes that appear on at least one OD pair's path
    
    # For greedy, evaluate each candidate
    for step in range(p):
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.3:  # Reserve time for improvement
            break
        
        best_gain = -1
        best_node = None
        
        for node in remaining_candidates:
            if node in selected_stations:
                continue
            # Evaluate adding this node
            test_stations = selected_stations | {node}
            gain = 0
            for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
                if node not in candidates:
                    continue
                if not is_covered(rt_data, selected_stations) and is_covered(rt_data, test_stations):
                    gain += flow
            
            if gain > best_gain:
                best_gain = gain
                best_node = node
        
        if best_node is not None:
            selected_stations.add(best_node)
            remaining_candidates.discard(best_node)
        else:
            # Pick any remaining candidate
            if remaining_candidates - selected_stations:
                node = next(iter(remaining_candidates - selected_stations))
                selected_stations.add(node)
                remaining_candidates.discard(node)
            else:
                # Pick any node not yet selected
                for node in all_nodes:
                    if node not in selected_stations:
                        selected_stations.add(node)
                        break
    
    # Fill remaining stations if greedy stopped early
    while len(selected_stations) < p:
        for node in all_nodes:
            if node not in selected_stations:
                selected_stations.add(node)
                if len(selected_stations) >= p:
                    break
    
    obj, covered = compute_covered_flow(selected_stations)
    if obj > best_objective:
        best_objective = obj
        best_stations = sorted(selected_stations)
        best_covered = covered
        if logger:
            logger.log(best_objective)
    
    # Local search: swap stations
    selected_list = sorted(selected_stations)
    non_selected = sorted(all_candidates - selected_stations)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.8:
        improved = False
        for i in range(len(selected_list)):
            if (time.time() - start_time) > time_limit * 0.8:
                break
            for j in range(len(non_selected)):
                if (time.time() - start_time) > time_limit * 0.8:
                    break
                # Try swapping selected_list[i] with non_selected[j]
                test_stations = set(selected_list)
                test_stations.discard(selected_list[i])
                test_stations.add(non_selected[j])
                
                test_obj, test_covered = compute_covered_flow(test_stations)
                if test_obj > best_objective:
                    best_objective = test_obj
                    old_sel = selected_list[i]
                    new_sel = non_selected[j]
                    selected_list[i] = new_sel
                    non_selected[j] = old_sel
                    best_stations = sorted(test_stations)
                    best_covered = test_covered
                    improved = True
                    if logger:
                        logger.log(best_objective)
                    break
            if improved:
                break
    
    # Try MIP if time permits and problem is not too large
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    if remaining_time > 5 and len(all_candidates) <= 500 and len(needs_stations) <= 2000:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # For each OD pair that needs stations, we create a "reachability" formulation
            # along the round trip path.
            
            # Variables:
            # y[j] = 1 if station at node j
            # z[q] = 1 if OD pair q is covered
            # For each OD pair q, auxiliary variables to model feasibility
            
            # Use the "node-based" formulation:
            # For OD pair q with round trip nodes v0, v1, ..., vK:
            # Define cumulative distances cum[0]=0, cum[i] = cum[i-1] + dist(v_{i-1}, v_i)
            # The trip is feasible if there's a sequence of refueling points such that
            # the distance between consecutive refueling points (and from start) <= R
            
            # We can model this with a set-cover like approach per OD pair:
            # For each pair of consecutive nodes (v_i, v_{i+1}) on the round trip,
            # define the set of nodes that can "bridge" across this segment.
            
            # Actually, let me use a DP-based precomputation:
            # For each OD pair, enumerate which subsets of its path nodes make it feasible.
            # This is exponential, so instead use a flow-based formulation.
            
            # Flow formulation per OD pair:
            # For OD pair q with round trip v_0, ..., v_K:
            # f[q,i] = 1 if v_i is a "last refueling point" for the stretch ending at v_i
            # But this gets complex. Let me use a simpler formulation.
            
            # Alternative: For each OD pair q, define binary variables:
            # r[q,i] = 1 if the vehicle refuels at position i on the round trip (i=0,...,K)
            # r[q,0] = 1 (starts with full tank)
            # r[q,i] <= y[v_i] for i > 0 (can only refuel at open stations)
            # For feasibility: for each i, there exists j <= i with r[q,j]=1 and cum[i+1]-cum[j] <= R
            
            # This is hard to linearize directly. Instead, use:
            # For each pair (j, i) where j < i and cum[i]-cum[j] <= R:
            #   The segment from j to i can be covered if there's a refueling at j
            #   and no intermediate refueling is needed (all intermediate segments fit)
            
            # Actually, the standard formulation for FRLM uses "pre-generated combinations":
            # For each OD pair, generate all minimal sets of nodes that make it feasible.
            # Then z_q <= sum over feasible combinations.
            # But this is exponential.
            
            # Practical approach: Use a "big-M" or indicator-based formulation with 
            # position-based fuel tracking.
            
            # Let's use: fuel[q,i] = fuel level when arriving at node v_i on round trip of OD pair q
            # fuel[q,0] = R (starts full)
            # fuel[q,i] >= 0
            # If y[v_i] = 1 (station at v_i), then fuel after refueling = R
            # If y[v_i] = 0, fuel stays the same
            # fuel[q,i+1] = (refueled fuel at i) - dist(v_i, v_{i+1})
            
            # With refueling: let g[q,i] be fuel after potential refueling at node i
            # g[q,i] >= fuel[q,i]  (at least what we have)
            # g[q,i] <= R  (can't exceed tank)
            # If y[v_i] = 1: g[q,i] = R
            # If y[v_i] = 0: g[q,i] = fuel[q,i]
            # This means: g[q,i] >= R * y[v_i] + fuel[q,i] * (1 - y[v_i])... no, that's not linear easily
            
            # Better: g[q,i] = fuel[q,i] + (R - fuel[q,i]) * y[v_i]  -- not linear
            
            # Use: g[q,i] >= fuel[q,i]
            #       g[q,i] <= R
            #       g[q,i] >= R - M*(1 - y[v_i])  -- if station, fuel >= R, combined with <= R gives = R
            #       g[q,i] <= fuel[q,i] + M*y[v_i]  -- if no station, fuel <= fuel[q,i], combined with >= gives =
            
            # fuel[q,i+1] = g[q,i] - d_i  (where d_i is dist from v_i to v_{i+1})
            # fuel[q,i+1] >= 0  (but only if z_q = 1)
            # fuel[q,i+1] >= -M*(1 - z_q)  (if not covered, fuel can be negative)
            
            # This formulation has O(sum of round trip lengths * num_od_pairs) continuous variables
            # and constraints. Could be large but tractable for moderate instances.
            
            model = gp.Model("FRLM")
            model.setParam('TimeLimit', max(remaining_time - 2, 1))
            model.setParam('Threads', 1)
            model.setParam('OutputFlag', 0)
            
            # Station variables
            y = {}
            for node in all_candidates:
                y[node] = model.addVar(vtype=GRB.BINARY, name=f"y_{node}")
            
            # Constraint: exactly p stations
            model.addConstr(gp.quicksum(y[node] for node in all_candidates) <= p)
            
            # Also allow stations at nodes not in any OD path (but they're useless)
            # So we only use all_candidates
            
            # OD pair coverage variables
            z = {}
            M = R + max(arc['distance'] for arc in instance['arcs']) * 2  # Big-M
            
            total_rt_nodes = 0
            for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
                round_trip, seg_dists = rt_data
                total_rt_nodes += len(round_trip)
            
            # Check if formulation is too large
            if total_rt_nodes > 50000:
                raise Exception("Too large for MIP")
            
            obj_expr = gp.LinExpr()
            # Add always covered
            always_covered_flow = sum(od_pairs[idx]['flow'] for idx in always_covered)
            
            for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
                round_trip, seg_dists = rt_data
                K = len(round_trip)  # number of nodes in round trip
                
                z[i] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}")
                obj_expr += flow * z[i]
                
                # Fuel variables: fuel[j] = fuel arriving at node j, g[j] = fuel after refueling at j
                fuel = {}
                g = {}
                
                for j in range(K):
                    fuel[j] = model.addVar(lb=-M, ub=R, name=f"fuel_{i}_{j}")
                    g[j] = model.addVar(lb=-M, ub=R, name=f"g_{i}_{j}")
                
                # Initial fuel
                model.addConstr(fuel[0] == R)
                
                # Node 0 is origin, may or may not have a station
                # For node 0: refueling at start means g[0] = R regardless
                # Actually, the vehicle starts with full tank, so g[0] = R always
                # But if node 0 has a station, on the return trip it will refuel there
                # Node 0 appears at position 0 and at position K-1 (end of round trip)
                model.addConstr(g[0] == R)  # starts full
                
                for j in range(1, K):
                    node_j = round_trip[j]
                    
                    # fuel[j] = g[j-1] - seg_dists[j-1]
                    model.addConstr(fuel[j] == g[j-1] - seg_dists[j-1])
                    
                    # Refueling logic at node j
                    if node_j in y:
                        # g[j] >= fuel[j]
                        model.addConstr(g[j] >= fuel[j])
                        # g[j] <= R
                        model.addConstr(g[j] <= R)
                        # If y[node_j] = 1: g[j] >= R => g[j] = R
                        model.addConstr(g[j] >= R - M * (1 - y[node_j]))
                        # If y[node_j] = 0: g[j] <= fuel[j] => g[j] = fuel[j]
                        model.addConstr(g[j] <= fuel[j] + M * y[node_j])
                    else:
                        # No station possible here
                        model.addConstr(g[j] == fuel[j])
                    
                    # Fuel must be non-negative if this OD pair is covered
                    # fuel[j] >= 0 if z[i] = 1
                    # fuel[j] >= -M * (1 - z[i])
                    model.addConstr(fuel[j] >= -M * (1 - z[i]))
            
            model.setObjective(obj_expr, GRB.MAXIMIZE)
            
            # Warm start with greedy solution
            for node in all_candidates:
                if node in set(best_stations):
                    y[node].Start = 1
                else:
                    y[node].Start = 0
            
            for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
                if idx in set(best_covered):
                    z[i].Start = 1
                else:
                    z[i].Start = 0
            
            model.optimize()
            
            if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED] and model.SolCount > 0:
                mip_obj = model.ObjVal + always_covered_flow
                mip_stations = [node for node in all_candidates if y[node].X > 0.5]
                
                if mip_obj > best_objective:
                    best_objective = mip_obj
                    best_stations = mip_stations
                    # Recompute covered OD pairs
                    stations_set = set(best_stations)
                    _, best_covered = compute_covered_flow(stations_set)
                    if logger:
                        logger.log(best_objective)
        
        except Exception as e:
            pass  # Fall back to heuristic solution
    
    # Second round of local search with remaining time
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.95:
        selected_stations = set(best_stations)
        selected_list = sorted(selected_stations)
        non_selected = sorted(all_candidates - selected_stations)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit * 0.95:
            improved = False
            for i in range(len(selected_list)):
                if (time.time() - start_time) > time_limit * 0.95:
                    break
                for j in range(len(non_selected)):
                    if (time.time() - start_time) > time_limit * 0.95:
                        break
                    test_stations = set(selected_list)
                    test_stations.discard(selected_list[i])
                    test_stations.add(non_selected[j])
                    
                    test_obj, test_covered = compute_covered_flow(test_stations)
                    if test_obj > best_objective:
                        best_objective = test_obj
                        old_sel = selected_list[i]
                        new_sel = non_selected[j]
                        selected_list[i] = new_sel
                        non_selected[j] = old_sel
                        best_stations = sorted(test_stations)
                        best_covered = test_covered
                        improved = True
                        if logger:
                            logger.log(best_objective)
                        break
                if improved:
                    break
    
    result = {
        "objective_value": best_objective,
        "stations": sorted(best_stations),
        "refueled_od_pairs": sorted(best_covered)
    }
    
    with open(solution_path, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)